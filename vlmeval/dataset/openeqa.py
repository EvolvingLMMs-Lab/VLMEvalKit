import os
import decord
import numpy as np
import pandas as pd

from PIL import Image

from ..smp.file import LMUDataRoot, load
from .video_base import VideoBaseDataset


class OpenEQA(VideoBaseDataset):
    TYPE = 'Video-VQA'

    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        'OpenEQA': '/mnt/aigc/wangyubo/data/UG/data/benchmark/opensource_tsv/openeqa.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'OpenEQA': None,
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        subsets = ['OpenEQA']
        return subsets

    def _task_category(self):
        return [
            'object recognition',
            'object localization',
            'attribute recognition',
            'spatial understanding',
            'object state recognition',
            'functional reasoning',
            'world knowledge',
        ]

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = '/mnt/aigc/wangyubo/data/UG/data/benchmark/openeqa'
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f"{dataset_name}.tsv")

        return dict(data_file=variant_data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False):
        vid_path = video
        rel_video_path = os.path.relpath(video, self.dataset_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            # Use os.path.relpath for robust relative path extraction
            frame_paths = self.frame_paths(rel_video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(rel_video_path, len(indices))

        missing = [
            (idx, pth) for idx, pth in zip(indices, frame_paths)
            if not os.path.exists(pth)
        ]

        if missing and not video_llm:
            for frame_idx, pth in missing:
                try:
                    frame_data = vid[frame_idx].asnumpy()
                    Image.fromarray(frame_data).save(pth)
                except Exception as e:
                    error_msg = f"Error saving frame {frame_idx} from {vid_path}: {str(e)}"
                    print(error_msg)

                    raise ValueError(error_msg) from e

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question = line['question']
        prompt = question

        message = []
        if video_llm:
            message.append(dict(type='video', value=os.path.join(self.dataset_path, line['video'])))
        else:
            frames, _, _ = self.save_video_frames(os.path.join(self.dataset_path, line['video']), video_llm)
            for im in frames:
                message.append(dict(type='image', value=im))

        message.append(dict(type='text', value=prompt))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_vqa_score_fn, eval_vqa_score

        # Select VQA scoring function (LLM-based) according to judge_kwargs['model'].
        score_fn = build_vqa_score_fn(judge_mode='likert5', **judge_kwargs)

        return eval_vqa_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'OpenEQA')
        )
