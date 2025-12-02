import re
import ast
import string
import pandas as pd

from .tools.utils import build_choices
from ....smp.log import get_logger


GENERIC_EXTRACT_JUDGE_PROMPT = (
    "You are an expert grading assistant.\n"
    "Your job has TWO tasks:\n"
    "1) From the candidate's full response, EXTRACT the final answer in a concise, normalized form.\n"
    "2) Compare this extracted answer with the STANDARD ANSWER and grade it as:\n"
    "   - A: CORRECT\n"
    "   - B: INCORRECT\n"
    "   - C: INVALID\n"
    "\n"
    "Here are the detailed evaluation criteria:\n"

    "1. ALWAYS refer to the given STANDARD ANSWER. You do NOT need to re-solve the question; the standard answer has "
    "already been provided and is always correct. Your only job is to judge whether the candidate's final answer is "
    "consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT "
    "AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.\n"

    "2. ONLY compare the FINAL ANSWER — COMPLETELY IGNORE any potential errors or issues in the REASONING PROCESS. "
    "Even if the reasoning is wrong, as long as the final answer matches the standard answer, grade it as CORRECT.\n"

    "3. Answers may be expressed in different ways (e.g., mathematical expressions, textual descriptions). As long as "
    "the meaning is the same as the standard answer, treat them as equivalent. If the standard answer does not specify "
    "a unit but the candidate's answer includes a correct unit for the given value, consider it CORRECT.\n"

    "4. Some answers may consist of multiple items, such as multiple-choice questions with multiple correct options, "
    "multi-select questions, or multi-blank fill-in-the-blank questions. Regardless of the question type, the final "
    "answer is considered CORRECT only if it matches the standard answer exactly at the level of all required items. "
    "For multi-select or multi-blank questions, ALL parts must be answered correctly and match the standard answer "
    "exactly to be deemed CORRECT.\n"

    "5. If the candidate's answer is wrapped in LaTeX-style markers like \\boxed{{...}}, IGNORE the \\boxed and only "
    "use the inner content as the candidate's final answer when comparing with the standard answer.\n"

    "6. If the candidate's answer is INVALID — for example, incomplete (cut off mid-response), containing a large "
    "amount of abnormal repetitive content, clearly irrelevant to the question, or explicitly refusing to answer due "
    "to ethical concerns, lack of information, or other external factors — then you MUST grade it as C: INVALID.\n"

    "7. This instruction applies to all problem types, including single-choice MCQ, multi-select MCQ, numeric "
    "problems, short-answer questions, and general VQA-style questions. In all cases, only the FINAL ANSWER and its "
    "consistency with the standard answer matter.\n"

    "8. The question or options may contain image placeholders such as '<image>', '<image id=1>', or similar tokens. "
    "You CANNOT see these images. Treat these placeholders as unknown content and DO NOT hallucinate or infer any "
    "specific visual details from them. If the standard answer or candidate's answer refers to an option associated "
    "with an image (e.g., 'choose A'), judge correctness only based on the stated answer, not by imagining the image.\n"

    "\n"
    "IMPORTANT – OUTPUT FORMAT:\n"
    "• You MUST return EXACTLY ONE line in the following format:\n"
    "  <GRADE>\\t<EXTRACTED_ANSWER>\n"
    "  where <GRADE> is one of A, B, or C.\n"
    "• <EXTRACTED_ANSWER> should be the final answer you extracted from the candidate's response, in a normalized, "
    "concise form (e.g., a number, a letter option, or a short phrase).\n"
    "• If you cannot extract any meaningful answer, or the response is INVALID, output:\n"
    "  C\\tN/A\n"
    "• Do NOT add any extra text, explanation, or additional lines.\n"
    "\n"
    "Now, judge the following question.\n"
    "<Original Question Begin>\n"
    "{question}\n"
    "{options_block}"
    "<Original Question End>\n"
    "<Standard Answer Begin>\n"
    "{gold_answer}\n"
    "<Standard Answer End>\n"
    "<Candidate's Answer Begin>\n"
    "{llm_response}\n"
    "<Candidate's Answer End>\n"
    "Your output:"
)


def build_option_str(option_dict):
    s = ''
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def call_llm_extract(
    model,
    max_retry: int,
    question: str,
    prediction: str,
    gold_answer: str,
    options_block: str = ""
):
    """
    通用的 LLM 调用 + 解析函数。

    返回: (grade, extracted_answer)
      - grade: 'A' / 'B' / 'C'
      - extracted_answer: LLM 认为的最终答案（字符串），若无则 'N/A'
    """
    logger = get_logger('LLM Extract')

    prompt = GENERIC_EXTRACT_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        llm_response=prediction,
        options_block=options_block,
    )

    for attempt in range(max_retry):
        ans = model.generate(prompt).strip()
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
            continue

        # 只看第一行，避免 LLM 啰嗦
        first_line = ans.splitlines()[0].strip()

        # 1) 最宽松：拿第一个非空 token，当作 grade
        #    兼容 "A\tB. Yes" / "A: B. Yes" / "A B. Yes" / "A,B. Yes"
        m = re.match(r'^\s*([ABC])\b(.*)$', first_line)
        if m:
            grade = m.group(1)
            # 去掉前导分隔符/空白
            rest = m.group(2).lstrip(" \t,:|")
            extracted = rest if rest else "N/A"
            return grade, extracted

        # 2) 如果只有一个字母 A/B/C 也接受
        m2 = re.match(r'^\s*([ABC])\s*$', first_line)
        if m2:
            grade = m2.group(1)
            return grade, "N/A"

        logger.warning(f"Unparsable LLM output: {ans}")

    # 多次失败兜底
    logger.warning("LLM extract failed after max_retry, fallback to INVALID.")
    return "C", "N/A"


def extract_ans_by_llm(
    model,
    row: pd.Series,
    mode: str = 'mcq',
    max_retry: int = 3
):
    """
    通用 LLM 抽取 + 打分入口。

    mode:
        - 'mcq': 带选项的选择题（使用 build_choices 构造 Options）
        - 'vqa': 开放式问答（无选项）
    返回: (grade, extracted_answer)
        grade ∈ {'A','B','C'}
    """
    valid_mode = ['mcq', 'vqa']
    assert mode in valid_mode, ValueError(f"Extract llm func mode must be in {valid_mode}, but got {mode}!")

    question = str(row.get('question', ''))
    prediction = str(row.get('prediction', ''))
    gold_raw = row.get('answer', '')

    # ---- 构造标准答案文本 gold_answer / answer_text ----
    if mode == 'mcq':
        # 选项
        choices = build_choices(row)
        option_str = build_option_str(choices) if choices else ""

        # 把 options 拼到 question 后面（方便 LLM 看）
        options_block = ""
        if option_str:
            options_block = "Options:\n" + option_str + "\n"
        else:
            options_block = ""

        # 标准答案：优先按字母 + 文本的形式展开
        answer_letter = str(gold_raw).strip().upper()
        if choices and answer_letter in choices:
            gold_answer = f"{answer_letter}. {choices[answer_letter]}"
        else:
            # 回退：直接用原始 answer 字段
            gold_answer = str(gold_raw)

    else:  # 'vqa'
        # 没有选项
        options_block = ""
        gold_answer = str(gold_raw)

    # ---- 调用 LLM ----
    grade, extracted = call_llm_extract(
        model=model,
        max_retry=max_retry,
        question=question,
        prediction=prediction,
        gold_answer=gold_answer,
        options_block=options_block,
    )

    return grade, extracted


def compute_mcq_score_llm(
    df: pd.DataFrame,
    model,
    *,
    mode: str = 'mcq',
    max_retry: int = 3,
) -> pd.DataFrame:
    """
    LLM-based MCQ scoring.

    model: 具有 .generate(prompt: str) -> str 的 judge 实例
    """
    grades, extracted_list, hits = [], [], []

    for _, row in df.iterrows():
        grade, extracted = extract_ans_by_llm(
            model=model,
            row=row,
            mode=mode,
            max_retry=max_retry,
        )
        grades.append(grade)
        extracted_list.append(extracted)
        hits.append(1 if grade == 'A' else 0)

    df = df.copy()
    df['judge_grade'] = grades          # 'A' / 'B' / 'C'
    df['pred_extracted'] = extracted_list
    df['hit'] = hits
    return df
