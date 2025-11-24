# flake8: noqa
import ast
import os.path as osp
import decord
import re
import math

from ..smp import *
from ..smp.file import LMUDataRoot, load
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

class SparBench(ImageBaseDataset):
    TYPE = 'VQA'

    SPAR_TASKS = [
        '',
        '50'
    ]

    LMUData_root = LMUDataRoot()
    DATASET_URL = {}

    DATASET_URL['SparBench'] = '/mnt/aigc/wangyubo/data/UG/data/benchmark/spar_zoe/tsv/SparBench.tsv'
    DATASET_URL['SparBench_50'] = '/mnt/aigc/wangyubo/data/UG/data/benchmark/spar_zoe/tsv/SparBench_50.tsv'

    DATASET_MD5 = {key: None for key in DATASET_URL}

    print(f"spar urls: {DATASET_URL}")

    @classmethod
    def get_task_type(self, task):
        MCA_QUESTION_TYPES = [
            "obj_spatial_relation_oo",
            "obj_spatial_relation_oc_mv",
            "obj_spatial_relation_oo_mv",
            "spatial_imagination_oc",
            "spatial_imagination_oo",
            "spatial_imagination_oc_mv",
            "spatial_imagination_oo_mv",
            "position_matching",
            "camera_motion_infer",
            "distance_infer_center_oo",
            "distance_infer_center_oo_mv"
        ]
        NA_QUESTION_TYPES = [
            "depth_prediction_oc",
            "depth_prediction_oo",
            "distance_prediction_oc",
            "distance_prediction_oo",

            "depth_prediction_oc_mv",
            "depth_prediction_oo_mv",
            "distance_prediction_oo_mv",
            "distance_prediction_oc_mv",
        ]

        SPECIAL_QUESTION_TYPES = [
            "view_change_infer",
        ]

        if task in MCA_QUESTION_TYPES:
            return 'MCQ'
        elif task in NA_QUESTION_TYPES:
            return 'NA'
        elif task in SPECIAL_QUESTION_TYPES:
            return 'SPECIAL'
        else:
            raise ValueError(f"Unsupported SparBench task type: {task}")

    def build_prompt(self, line):

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        task = line['task']
        task_type = self.get_task_type(task)

        pre_prompt = ''

        if task_type == 'NA':
            post_prompt = "Please answer the question using a single word or phrase."
            prompt = pre_prompt + "\n" + question + "\n" + post_prompt

        elif task_type == 'MCQ':
            post_prompt = ""
            if task in ['position_matching', "camera_motion_infer"]:
                post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
            post_prompt2 = "Answer with the option's letter from the given choices directly."
            prompt = pre_prompt + "\n" + question + "\n" + post_prompt + "\n" + post_prompt2

        elif task_type == 'SPECIAL':
            post_prompt1 = ""
            post_prompt2 = ""
            prompt = pre_prompt + "\n" + question + "\n" + post_prompt1 + "\n" + post_prompt2

        else:
            raise ValueError(f"Unknown question type: {task}")

        print(f"prompt: {prompt}")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_rel_bench.cal_scores import (
            compute_mcq_score, compute_na_score, mean_relative_accuracy
        )

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        data['task_type'] = data['task'].apply(self.get_task_type)

        mcq_data = data[data['task_type'] == 'MCQ'].copy()
        na_data = data[data['task_type'] == 'NA' ].copy()
        special_data = data[data['task_type'] == 'SPECIAL' ].copy()

        print(f"[split] MCQ={len(mcq_data)}, NA={len(na_data)}, SPECIAL={len(special_data)}")

        # 计算每类题型的逐样本分数列
        if len(mcq_data):
            mcq_scored = compute_mcq_score(mcq_data)
        else:
            mcq_scored = mcq_data

        if len(na_data):
            na_scored  = compute_na_score(na_data)
        else:
            na_scored = na_data

        if len(special_data):
            sp_scored  = self.compute_special_score(special_data)
        else:
            sp_scored = special_data

        # 聚合（任务均值 + overall + Low/Middle/High）
        summary = self._aggregate(mcq_scored, na_scored, sp_scored)

        # 打印或保存；与上游保持“返回 overall*100”
        print(f"[SparBench] summary: {summary}")

        # ---- save pkl dump ----
        try:
            to_dump = {
                'mcq_scored': mcq_scored,
                'na_scored': na_scored,
                'special_scored': sp_scored,
                'summary': summary
            }
            import pickle
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f"[save] result saved to {result_file}")
        except Exception as e:
            warnings.warn(f"[save] failed to save result to {result_file}: {e}")

        # ---- prepare paths for xlsx / tsv ----
        base_no_suffix = eval_file[:-(len(suffix) + 1)]
        xlsx_path = f"{base_no_suffix}_extract_matching.xlsx"
        acc_tsv_path = f"{base_no_suffix}_acc.tsv"

        # ---- save extract_matching.xlsx ----
        try:
            import pandas as pd

            frames = []

            if len(mcq_scored):
                df_mcq = mcq_scored.copy()
                df_mcq['task_type'] = 'MCQ'
                frames.append(df_mcq)

            if len(na_scored):
                df_na = na_scored.copy()
                df_na['task_type'] = 'NA'
                frames.append(df_na)

            if len(sp_scored):
                df_sp = sp_scored.copy()
                df_sp['task_type'] = 'SPECIAL'
                frames.append(df_sp)

            if frames:
                merged = pd.concat(frames, axis=0, ignore_index=True)
            else:
                # fallback：无数据
                merged = pd.DataFrame()

            prefer_front = [
                'index', 'task', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05', 'vci_metric'
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                    [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                merged.to_excel(writer, sheet_name="ALL", index=False)

            print(f"[save] extract & matching saved to {xlsx_path}")

        except Exception as e:
            warnings.warn(f"[save] failed to save merged extract xlsx: {e}")

        # ---- save acc.tsv ----
        try:
            # 1. 去掉内部 meta key
            summary_clean = {
                k: v for k, v in summary.items()
                if k not in ('tabulated_keys', 'tabulated_results')
            }

            # 2. 排序：先 overall/Low/Middle/High，再其它 task-level metrics
            front_keys = ['overall', 'Low', 'Middle', 'High']
            other_keys = [k for k in summary_clean.keys() if k not in front_keys]

            final_order = front_keys + other_keys

            # 3. 生成一行的 DataFrame
            acc_df = pd.DataFrame({k: [summary_clean.get(k, None)] for k in final_order})

            # 4. 保存
            acc_df.to_csv(acc_tsv_path, sep="\t", index=False)
            print(f"[save] accuracy table saved to {acc_tsv_path}")

        except Exception as e:
            warnings.warn(f"[save] failed to save acc tsv: {e}")

        print(f"[{self.dataset_name}] summary: {summary}")
        return summary

    @staticmethod
    def _parse_instruction(instruction: str) -> dict[str, float]:
        # "move_right:0.3,move_left:0.1" -> dict
        if instruction is None:
            return {}
        d = {}
        for item in str(instruction).split(','):
            item = item.strip()
            if not item or ':' not in item:
                continue
            k, v = item.split(':', 1)
            try:
                d[k.strip()] = float(v.strip())
            except Exception:
                pass
        return d

    @classmethod
    def _compute_vci_metric(cls, pred: str, answer: str) -> float:
        # 与你给的 compute_vci_metric 等价，但修正 MRA 参数顺序为 (pred, target)
        action_order = [
            ("move_right", "move_left"),
            ("move_up", "move_down"),
            ("move_forward", "move_backward"),
            ("rotate_right", "rotate_left"),
            ("rotate_up", "rotate_down"),
        ]
        p = cls._parse_instruction(pred)
        g = cls._parse_instruction(answer)

        vals = []
        for a_pos, a_neg in action_order:
            pred_v = p.get(a_pos, 0.0) - p.get(a_neg, 0.0)
            gt_v   = g.get(a_pos, 0.0) - g.get(a_neg, 0.0)
            vals.append(mean_relative_accuracy(pred_v, gt_v, .5, .95, .05))
        return float(np.mean(vals)) if len(vals) else 0.0

    def compute_special_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        当前只有 view_change_infer：输出新增列 'vci_metric'
        """
        vals = []
        for _, r in df.iterrows():
            try:
                vals.append(self._compute_vci_metric(r['prediction'], r['answer']))
            except Exception:
                vals.append(0.0)
        df = df.copy()
        df['vci_metric'] = vals
        return df

    # ==== 汇总（与 sparbench_aggregate_results 一致）====
    def _aggregate(self, mcq_df, na_df, sp_df) -> dict:
        out = {}

        # 每个 task 的均值
        if len(mcq_df):
            for task, sub in mcq_df.groupby('task'):
                metric_name = 'hit'  # compute_mcq_score 的正确率列名
                out[f'{task}_accuracy'] = float(sub[metric_name].mean())

        if len(na_df):
            for task, sub in na_df.groupby('task'):
                out[f'{task}_MRA:.5:.95:.05'] = float(sub['MRA:.5:.95:.05'].mean())

        if len(sp_df):
            # 目前仅 view_change_infer
            for task, sub in sp_df.groupby('task'):
                if 'vci_metric' in sub.columns:
                    out[f'{task}_vci_metric'] = float(sub['vci_metric'].mean())

        # overall = 这些 task-level 均值的简单等权平均
        if len(out):
            out['overall'] = float(np.mean(list(out.values())))
        else:
            out['overall'] = 0.0

        # 可选：Low/Middle/High（仅用于日志观察，不影响返回值）
        Low = set([
            "depth_prediction_oc","depth_prediction_oo",
            "distance_prediction_oc","distance_prediction_oo",
            "depth_prediction_oc_mv","depth_prediction_oo_mv",
            "distance_prediction_oo_mv","distance_prediction_oc_mv",
        ])
        Middle = set(["view_change_infer","position_matching","camera_motion_infer"])
        High = set([
            "obj_spatial_relation_oo","obj_spatial_relation_oc_mv","obj_spatial_relation_oo_mv",
            "spatial_imagination_oc","spatial_imagination_oo","spatial_imagination_oc_mv","spatial_imagination_oo_mv",
            "distance_infer_center_oo","distance_infer_center_oo_mv",
        ])

        lows, mids, highs = [], [], []
        for k, v in out.items():
            if k == 'overall':
                continue
            task_name = "_".join(k.split("_")[:-1])  # 去掉最后一个度量字段
            if task_name in Low:
                lows.append(v)
            elif task_name in Middle:
                mids.append(v)
            elif task_name in High:
                highs.append(v)

        out['Low'] = float(np.mean(lows)) if lows else 0.0
        out['Middle'] = float(np.mean(mids)) if mids else 0.0
        out['High'] = float(np.mean(highs)) if highs else 0.0
        return out
