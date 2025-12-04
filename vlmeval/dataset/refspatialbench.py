import os
import re
import ast
import pandas as pd
import numpy as np
import json

from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from .image_vqa import ImageVQADataset
from ..smp.file import LMUDataRoot, load, dump
from ..smp.misc import toliststr, get_cache_path, modelscope_flag_set

from huggingface_hub import snapshot_download


class RefSpatialBench(ImageVQADataset):

    DATASET_URL = {
        'RefSpatial': '/mnt/aigc/wangyubo/data/UG/data/benchmark/opensource_tsv/RefSpatial.tsv'
    }
    DATASET_MD5 = {
        'RefSpatial': None
    }

    def _task_category(self):
        return ['location', 'placement', 'unseen']

    def prepare_tsv(self, url, file_md5=None, repo_id='BAAI/RefSpatial-Bench'):
        data = super().prepare_tsv(url, file_md5)

        SENTINEL_NAME = ".refspatial_extracted"
        cache_path = get_cache_path(repo_id)

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text="ok"):
                tmp = sentinel_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            sentinel_path = os.path.join(dataset_path, SENTINEL_NAME)
            _write_sentinel(sentinel_path, text="done")

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)
            data['mask_path'] = data['mask_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        suffix = line['suffix']

        prompt = f"{question} {suffix}"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        eval_file: TSV/JSONL，已经包含模型预测后的表格，至少有：
        - category
        - mask_path
        - prediction    (字符串形式: '(x,y)' 或 '[(x0,y0,x1,y1), ...]' 等)

        返回:
        {
            "overall": overall_acc,
            "location": acc_location,
            "placement": acc_placement,
            "unseen": acc_unseen,
        }
        """
        # 用你自己的 load 封装读表
        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')

        # 统一成字符串，避免后面出类型问题
        data['prediction'] = data['prediction'].astype(str)

        required = ['category', 'mask_path', 'prediction']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in eval_file: {eval_file}")

        acc_all = []
        acc_by_cat = defaultdict(list)

        for _, row in data.iterrows():
            cat = str(row['category']).lower()
            pred_text = row['prediction']

            # 处理 mask 路径：可能是 ['xxx.png'] 这样的形式
            mask_raw = row['mask_path']
            mask_path = _normalize_path(mask_raw)

            mask_path = str(mask_path)

            print(f"mask path: {mask_path}")

            if not os.path.exists(mask_path):
                print(f"[WARN] mask not found: {mask_path}")
                continue

            mask = np.array(Image.open(mask_path)) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)

            print(f"mask loaded, pred_text: {pred_text}")

            try:
                points = _text2pts(pred_text, mask.shape[1], mask.shape[0])
            except Exception as e:
                print(f"[WARN] failed to parse prediction: {pred_text} ({e})")
                continue

            print('mask prepared')

            acc = 0.0
            if len(points) > 0:
                in_range = (
                    (points[:, 0] >= 0)
                    & (points[:, 0] < mask.shape[1])
                    & (points[:, 1] >= 0)
                    & (points[:, 1] < mask.shape[0])
                )
                if in_range.any():
                    vals = mask[points[in_range, 1], points[in_range, 0]]
                    # 图外的点补 0
                    vals = np.concatenate([vals, np.zeros(points.shape[0] - in_range.sum())])
                    acc = float(vals.mean())

            acc_all.append(acc)
            acc_by_cat[cat].append(acc)

        if not acc_all:
            raise ValueError("No valid accuracy computed; check eval_file format and mask_path.")

        overall = float(np.mean(acc_all))
        results = {"overall": overall}
        for cat in self._task_category():
            vals = acc_by_cat.get(cat, [])
            if vals:
                results[cat] = float(np.mean(vals))

        return results


def _text2pts(text: str, width=640, height=480) -> np.ndarray:
    """
    从自由文本中解析 (x, y) 或 (x0, y0, x1, y1) 坐标。
    对每个 vector 自适应判断是归一化(0~1)还是像素坐标：
      - 若所有值在 [0, 1.5] 之间，视为归一化 -> 乘以宽高
      - 否则视为像素坐标 -> 直接使用
    """
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        nums = [float(num) for num in match.split(',')]
        # 判定是否“看起来像归一化”
        max_abs = max(abs(v) for v in nums)
        is_norm = (0.0 <= max_abs <= 1.5)  # 留一点余量，防止 1.0 / 1.00 这种

        if len(nums) == 2:
            x, y = nums
            if is_norm:
                x = x * width
                y = y * height
            points.append((int(round(x)), int(round(y))))

        elif len(nums) == 4:
            x0, y0, x1, y1 = nums
            if is_norm:
                x0 = x0 * width
                y0 = y0 * height
                x1 = x1 * width
                y1 = y1 * height

            x0, y0, x1, y1 = map(int, map(round, (x0, y0, x1, y1)))
            # 避免 y1<y0 或 x1<x0 的反向情况
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            # 将框展开为所有像素点（如果太大，可以视情况改成采样）
            h = max(0, y1 - y0)
            w = max(0, x1 - x0)
            if h > 0 and w > 0:
                yy, xx = np.where(np.ones((h, w), dtype=np.uint8))
                pts = np.stack([xx + x0, yy + y0], axis=1)
                points.extend(pts.tolist())

    return np.array(points, dtype=int)


def _normalize_path(p):
    """把 ['xxx.png'] 或 ['xxx'] 字符串 / list 统一成单个 path 字符串"""
    # 已经是 list/tuple
    if isinstance(p, (list, tuple)):
        return p[0]
    # 是形如 "['xxx']" 的字符串
    if isinstance(p, str) and p.startswith('[') and p.endswith(']'):
        try:
            v = ast.literal_eval(p)
            if isinstance(v, (list, tuple)) and v:
                return v[0]
        except Exception:
            pass
    return p
