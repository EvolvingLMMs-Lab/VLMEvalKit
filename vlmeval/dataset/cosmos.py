import os
import ast
import json
import decord
import numpy as np

from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

from ..smp.misc import get_cache_path
from ..smp.file import LMUDataRoot, load
from .video_base import VideoBaseDataset


class CosmosBench(VideoBaseDataset):
    """
    Cosmos-Reason1.

    Reference:
      Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning
      https://arxiv.org/abs/2503.15558
    """

    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'CosmosReason1': '/mnt/aigc/wangyubo/data/UG/data/benchmark/cosmos/CosmosReason1.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'CosmosReason1': None,
    }

    def __init__(self, dataset, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['CosmosReason1']

    def _task_category(self):
        return [
            'bridgev2',
            'robovqa',
            'holoassist',
            'robofail',
        ]

    def prepare_dataset(self, dataset_name: str):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = '/mnt/aigc/wangyubo/data/UG/data/benchmark/cosmos'
        self.dataset_path = dataset_path = '/mnt/aigc/wangyubo/data/UG/data/benchmark/cosmos'

        variant_data_file = os.path.join(self.LMUData_root, f"{dataset_name}.tsv")

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        indices = []

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()

            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            if self.sample_strategy == 'uniform_tail' and (video_nframes - 1) != indices[-1]:
                indices.append(video_nframes - 1)

            frame_paths = self.frame_paths_fps(video_path, len(indices))

        flag = np.all([os.path.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not os.path.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        prompt = question

        message = []

        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.data_root, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(line['video'], video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'CosmosBench'),
        )
