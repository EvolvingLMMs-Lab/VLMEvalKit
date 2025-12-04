import os
import re
import ast
import pandas as pd

from tqdm import tqdm

from .image_vqa import ImageVQADataset
from ..smp.file import LMUDataRoot, load, dump
from ..smp.misc import toliststr, get_cache_path, modelscope_flag_set

from huggingface_hub import snapshot_download


class RoboSpatialBench(ImageVQADataset):

    DATASET_URL = {
        'RoboSpatialHome': '/mnt/aigc/wangyubo/data/UG/data/benchmark/opensource_tsv/RoboSpatial.tsv'
    }
    DATASET_MD5 = {
        'RoboSpatialHome': None
    }

    def _task_category(self):
        return ['compatibility', 'configuration', 'context']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        prompt = question

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    @staticmethod
    def point_in_polygon(x, y, poly):
        """
        Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
        Uses the ray-casting algorithm.
        """
        num = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, num + 1):
            p2x, p2y = poly[i % num]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def evaluate_answer(ground_truth, generated_answer):
        """
        Evaluates if the generated answer is correct based on the ground truth.
        Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
        """
        gen_answer = generated_answer.strip().lower()
        gt_lower = ground_truth.strip().lower()

        # Check if this is a binary yes/no question
        if gt_lower in ["yes", "no"]:
            is_binary = True
            is_gt_yes = (gt_lower == "yes")
            is_parsable = len(gen_answer) > 0
            if is_gt_yes:
                correct = gen_answer.startswith("yes")
            else:
                correct = gen_answer.startswith("no")
            return correct, is_binary, generated_answer.strip(), is_parsable

        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False

        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable

            # (x, y)
            tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', generated_answer)
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = RoboSpatialBench.point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass

            # [x, y]
            list_match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', generated_answer)
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = RoboSpatialBench.point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass

            # fallback: 提取第一个 [...]，用 literal_eval 兜底
            try:
                match = re.search(r'\[(.*?)\]', generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable

                list_content = match.group(1)
                list_content = re.sub(r',(\S)', r', \1', list_content)
                list_content = list_content.strip()
                if list_content.endswith(','):
                    list_content = list_content[:-1]

                list_str = '[' + list_content + ']'

                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    tuple_match = re.search(
                        r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', list_content
                    )
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = RoboSpatialBench.point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable

                # 归一到 (x, y)
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable
                    if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                        gen_point = tuple(gen_val)
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable

                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = RoboSpatialBench.point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                return False, is_binary, parsed_answer, is_parsable

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_result.pkl')

        base_no_suffix = eval_file[:-(len(suffix) + 1)]
        xlsx_path = f"{base_no_suffix}_results.xlsx"
        acc_tsv_path = f"{base_no_suffix}_acc.tsv"

        # Load model predictions (DataFrame; must contain columns like
        # question / answer / prediction / category)
        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')

        num_total = 0
        num_correct = 0
        illformed_responses = 0
        category_stats = {}

        is_correct_list = []
        is_binary_list = []
        parsed_answer_list = []
        is_parsable_list = []

        # Iterate rows and score one by one
        for _, row in data.iterrows():
            gt = row['answer']
            pred = row['prediction']
            category = row['category'] or 'unknown'

            assert category in self._task_category(), (
                f"Except RoboSpatial category to be one of {self._task_category()}, "
                f"but got {category}."
            )

            if category not in category_stats:
                category_stats[category] = {'num_correct': 0, 'num_total': 0}
            category_stats[category]['num_total'] += 1
            num_total += 1

            # Call unified evaluation logic (decides yes/no or polygon
            # based on the ground-truth format)
            correct, is_binary, parsed_answer, is_parsable = RoboSpatialBench.evaluate_answer(
                gt, pred
            )

            if not is_parsable:
                illformed_responses += 1
            if correct:
                num_correct += 1
                category_stats[category]['num_correct'] += 1

            is_correct_list.append(bool(correct))
            is_binary_list.append(bool(is_binary))
            parsed_answer_list.append(None if parsed_answer is None else str(parsed_answer))
            is_parsable_list.append(bool(is_parsable))

        # Attach evaluation fields back to the DataFrame
        data['is_correct'] = is_correct_list
        data['is_binary'] = is_binary_list
        data['parsed_answer'] = parsed_answer_list
        data['is_parsable'] = is_parsable_list

        accuracy = 100.0 * num_correct / num_total if num_total > 0 else 0.0

        # Save detailed results
        dump(data, result_file)

        # Export xlsx (for manual inspection)
        try:
            data.to_excel(xlsx_path, index=False)
        except Exception as e:
            print(f"[WARN] failed to save xlsx to {xlsx_path}: {e}")

        # Aggregate accuracy results into a single table and save as TSV
        rows = []

        # Overall summary row
        rows.append(
            dict(
                dataset="RoboSpatial",
                category="ALL",
                accuracy=accuracy,
                num_correct=num_correct,
                num_total=num_total,
            )
        )

        # Per-category rows
        for cat, stat in category_stats.items():
            cat_total = stat["num_total"]
            cat_acc = 100.0 * stat["num_correct"] / cat_total if cat_total > 0 else 0.0
            rows.append(
                dict(
                    dataset="RoboSpatial",
                    category=cat,
                    accuracy=cat_acc,
                    num_correct=stat["num_correct"],
                    num_total=cat_total,
                )
            )

        acc_df = pd.DataFrame(rows)
        acc_df.to_csv(
            acc_tsv_path,
            sep="\t",
            index=False,
            float_format="%.2f",
        )

        print(
            f"[RoboSpatial] accuracy = {accuracy:.2f} "
            f"(num_correct={num_correct}, num_total={num_total}, "
            f"illformed_resp={illformed_responses})"
        )

        return {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "illformed_responses": illformed_responses,
            "category_stats": category_stats,
        }
