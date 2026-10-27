"""Long-context QA data and scoring for the migrated MemAgent workflow."""

import os
import re
import string

from .base import BaseTask


JUDGE_PROMPT = """Please judge whether the predicted answer is correct based on the standard answer.

Question: {question}
Standard Answer: {answer}
Predicted Answer: {prediction}

Is the predicted answer consistent with the standard answer? Please output only "True" or "False".
"""


def extract_boxed_content(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = idx + len("\\boxed")
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text) or text[start] != "{":
        return None
    depth = 0
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 1:end]
    return None


def normalize_answer(text):
    text = "".join(c for c in str(text).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


class LongContextMemoryTask(BaseTask):
    @staticmethod
    def normalize_instance(instance):
        context = instance.get("context")
        question = instance.get("question", instance.get("input"))
        if question is None:
            question = instance.get("extra_info", {}).get("question")
        answers = instance.get("answers", instance.get("ground_truth"))
        if answers is None:
            answers = instance.get("reward_model", {}).get("ground_truth")
        if not isinstance(context, str) or not isinstance(question, str):
            raise ValueError("Memory QA requires string context and question/input fields")
        if isinstance(answers, str):
            answers = [answers]
        elif answers is not None:
            answers = list(answers)
        if not answers or not all(isinstance(answer, str) for answer in answers):
            raise ValueError("Memory QA requires at least one string reference answer")
        return {"context": context, "question": question, "answers": answers}

    @classmethod
    async def initialize_runtime(cls, instance):
        return cls.normalize_instance(instance)

    @classmethod
    def get_instruction(cls, instance):
        return [{"role": "user", "content": cls.normalize_instance(instance)["question"]}]

    @classmethod
    def complete_runtime(cls, result):
        return result

    @classmethod
    async def evaluate_result(cls, result, instance, options=None):
        options = options or {}
        prediction = extract_boxed_content(result["answer"])
        if prediction is None:
            return 0.0
        mode = options.get("reward_mode", "llm_judge")
        if mode == "exact_match":
            return float(any(normalize_answer(prediction) == normalize_answer(a) for a in instance["answers"]))
        if mode != "llm_judge":
            raise ValueError(f"Unsupported reward_mode: {mode}")
        from openai import AsyncOpenAI

        # A judge failure must not silently become a valid negative training label.
        async with AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ) as client:
            response = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4.1-nano"), temperature=0,
                messages=[
                    {"role": "system", "content": "You are an impartial judge."},
                    {"role": "user", "content": JUDGE_PROMPT.format(
                        question=instance["question"], answer=instance["answers"][0], prediction=prediction,
                    )},
                ],
            )
        verdict = response.choices[0].message.content
        if not verdict:
            raise ValueError("Empty judge response")
        verdict = verdict.split("</think>")[-1].strip()
        if verdict not in ("True", "False"):
            raise ValueError(f"Invalid judge verdict: {verdict!r}")
        return float(verdict == "True")
