"""Long-context question answering for memory agents."""

from ..base import BaseTask
from ..verifiers.memagent import compute_score


class LongContextMemoryTask(BaseTask):
    @classmethod
    async def initialize_runtime(cls, instance):
        return cls.normalize_instance(instance)

    @staticmethod
    def normalize_instance(instance):
        context = instance.get("context")
        question = instance.get("question", instance.get("input"))
        if question is None:
            question = (instance.get("extra_info") or {}).get("question")
        answers = instance.get("answers", instance.get("ground_truth"))
        if answers is None:
            answers = (instance.get("reward_model") or {}).get("ground_truth")
        if not isinstance(context, str) or not isinstance(question, str) or not question.strip():
            raise ValueError("Long-context QA requires a text context and a nonempty question")
        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        if isinstance(answers, str):
            answers = [answers]
        if not isinstance(answers, (list, tuple)) or not answers or not all(
            isinstance(answer, str) and answer.strip() for answer in answers
        ):
            raise ValueError("Long-context QA requires at least one nonempty reference answer")
        return {"context": context, "question": question, "answers": list(answers)}

    @classmethod
    def get_instruction(cls, instance):
        return [{"role": "user", "content": instance["question"]}]

    @classmethod
    def complete_runtime(cls, result):
        return result

    @classmethod
    async def evaluate_result(cls, result, instance):
        return compute_score(result, instance["answers"])
