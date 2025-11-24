import os.path as osp
import string
import pandas as pd
import re

from collections import OrderedDict

from .image_mcq import ImageMCQDataset
from ..smp.file import LMUDataRoot, load, dump, get_intermediate_file_path
from ..smp.misc import toliststr


class SpatialVizBench(ImageMCQDataset):
    TYPE = 'MCQ'

    LMUData_root = LMUDataRoot()

    # TODO: upload spatialviz data to somewhere, backup place: zoe hf? or opencompass space
    DATASET_URL = {
        'SpatialVizBench': osp.join(LMUData_root, 'SpatialVizBench.tsv'),
        'SpatialVizBench_CoT': osp.join(LMUData_root, 'SpatialVizBench_CoT.tsv'),
    }

    DATASET_MD5 = {
        'SpatialVizBench': None,  # TODO: check this
        'SpatialVizBench_CoT': None,  # TODO: check this
    }

    CATEGORY_TASK_ORDER = OrderedDict([
        ("MentalRotation", ["2DRotation", "3DRotation", "3ViewProjection"]),
        ("MentalFolding", ["PaperFolding", "CubeUnfolding", "CubeReconstruction"]),
        ("VisualPenetration", ["CrossSection", "CubeCounting", "CubeAssembly"]),
        ("MentalAnimation", ["ArrowMoving", "BlockMoving", "MechanicalSystem"]),
    ])

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset='SpatialVizBench', skip_noimg=skip_noimg)

        self.use_cot = self.parse_dataset_name(dataset)

    @staticmethod
    def parse_dataset_name(name: str) -> bool:

        print(f"--------------- in use cot : {name}- -------------------")

        if not isinstance(name, str):
            return False

        lower = name.lower()
        return lower.endswith("_cot")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        options_text = ''
        for key, item in options.items():
            options_text += f'{key}. {item}\n'

        # prompt format follow spatialviz paper
        if not self.use_cot:
            pre_prompt = (
                "Answer with a single option letter (A, B, C, or D), enclosed within the <answer></answer> tag."
                "For example: <answer>A</answer>. Ensure that your output contains only the final answer, "
                "without any intermediate reasoning or additional content."
            )
        else:
            pre_prompt = (
                "You should first provide a reasoning process, "
                "then provide a single option (A, B, C or D) as the final answer. "
                "The reasoning process and the answer are enclosed within "
                "<think></think> and <answer></answer> tags, respectively, "
                "i.e., <think>reasoning process</think>, <answer>answer</answer>."
            )

        prompt = pre_prompt + "\n" + "Question:" + question + '\n' + options_text

        print(f"Final input prompt: {prompt}")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import compute_mcq_score, eval_mcq_core

        raw = eval_mcq_core(
            load_fn=load,
            eval_file=eval_file,
            score_fn=compute_mcq_score,
            group_col=['category', 'task'],
            order={
                'category': list(self.CATEGORY_TASK_ORDER.keys()),
                'task': sum(self.CATEGORY_TASK_ORDER.values(), []),
            },
            dataset_name=getattr(self, 'dataset_name', 'SpatialVizBench'),
        )

        # raw 里有：
        #   overall
        #   MentalRotation_accuracy / ...
        #   task.2DRotation_accuracy / ...

        pretty = OrderedDict()
        pretty['overall'] = raw['overall']

        # 按你想要的顺序拼：子类 -> 大类 avg
        for cat, tasks in self.CATEGORY_TASK_ORDER.items():
            for t in tasks:
                k = f"task.{t}_accuracy"
                if k in raw:
                    pretty[f"{t}_accuracy"] = raw[k]
            cat_key = f"{cat}_accuracy"
            if cat_key in raw:
                pretty[cat_key] = raw[cat_key]

        # 最后再补一行 tabulated，方便直接塞 LaTeX
        keys_str = ", ".join(pretty.keys())
        vals_str = ", ".join(f"{v:.3f}" for v in pretty.values())
        pretty['tabulated_keys'] = keys_str
        pretty['tabulated_results'] = vals_str

        # 你如果还想保留原始 raw，可以一起返回
        return pretty
