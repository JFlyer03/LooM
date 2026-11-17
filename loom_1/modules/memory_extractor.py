"""Extraction prompt and conversation preparation migrated from old LooM."""

from ..common.registry import MODULE_REGISTRY
from ..envs.memory_bank_utils import ConversationMessage, format_conversation
from .base import BaseModule

EXTRACTION_PROMPT_EN = """You are a memory extraction expert.
Your task is to extract memories from the user's perspective based on the conversation between the user and the assistant. This means identifying information the user might remember—including the user's own experiences, thoughts, plans, or statements and actions made by others (such as the assistant) that affect the user or are acknowledged by the user.

Please perform the following operations:
1. Identify information reflecting the user's experiences, beliefs, concerns, decisions, plans, or responses—including meaningful information from the assistant acknowledged or responded to by the user.

2. Clearly parse all references to time, people, and events:
   - If possible, convert relative time expressions (e.g., "yesterday", "next Friday") to absolute dates using message timestamps.
   - Clearly distinguish between event time and message time.
   - If specific locations are mentioned, include them.
   - Resolve all pronouns, aliases, and vague references to full names or clear identities.

3. Always write in the third person perspective, using "User" to refer to the user, rather than the first person.

4. Do not omit any information the user might remember.
   - Include all key experiences, thoughts, emotional reactions, and plans.
   - Prioritize completeness and fidelity over brevity.

Return a valid JSON object with the following structure:

{{
  "memory_list": [
    {{
      "key": "<string, unique and concise memory title>",
      "memory_type": "<string, 'LongTermMemory' or 'UserMemory'>",
      "value": "<detailed, independent, and unambiguous memory statement>",
      "tags": ["<list of relevant topic keywords>"]
    }}
  ],
  "summary": "<paragraph naturally summarizing the above memories from the user's perspective, 120-200 words>"
}}

Conversation:
{conversation}

Your output:"""


@MODULE_REGISTRY.register("naive_extractor")
class NaiveExtractor(BaseModule):
    def build_prompt(self, fact):
        messages = [ConversationMessage(
            role=msg.get("role", "user"), content=msg.get("content", ""),
            timestamp=msg.get("timestamp", ""),
        ) for msg in fact]
        return EXTRACTION_PROMPT_EN.format(conversation=format_conversation(messages))

    async def rollout(self, fact, generate, **kwargs):
        return await generate(self.build_prompt(fact), "extraction")
