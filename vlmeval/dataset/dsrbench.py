import os
import ast
import decord
import string
import json
import numpy as np

from PIL import Image
from tqdm import tqdm
from huggingface_hub import snapshot_download

from ..smp.misc import get_cache_path, modelscope_flag_set
from ..smp.file import LMUDataRoot, load
from .video_base import VideoBaseDataset


class DSRBench(VideoBaseDataset):

    MD5 = ''
    TYPE = 'MCQ'
    MODALITY = 'VIDEO'

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""
    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'DSRBench': '/mnt/aigc/wangyubo/data/UG/data/benchmark/opensource_tsv/DSRBench.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'DSRBench': None,
    }

    def __init__(self, dataset, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        # self.use_subtitle = use_subtitle
        # self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['DSRBench']

    def _task_category(self):
        return [
            'abs_dis',
            'abs_dir',
            'abs_ori',
            'abs_spd',
            'abs_spd_comp',
            'abs_dir_pred',
            'rel_dis',
            'rel_dir',
            'rel_ori',
            'rel_spd',
            'rel_spd_comp',
            'rel_dir_pred',
            'non_temp',
        ]

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = "/mnt/umm/users/wc_workspace/dsr-bench"
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):
        vid_path = os.path.join(self.data_root, video_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()

        indices = []

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            frame_paths = self.frame_paths(video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path, len(indices))

        flag = np.all([os.path.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not os.path.exists(pth) and not video_llm:
                    im.save(pth)

        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, _, _ = self.save_video_frames(line['video'], video_llm)

        pre_prompt = self.FRAMES_TMPL_NOSUB
        question = line['question']
        options = line['options']

        if isinstance(options, str):
            try:
                options = json.loads(options)
            except Exception:
                options = ast.literal_eval(options)

        question += '\n' + '\n'.join(options)
        prompt = f'Question: {question}\nAnswer: '

        message = []
        message.append(dict(type='text', value=pre_prompt))
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
            group_col='task_type',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'DSRBench')
        )
