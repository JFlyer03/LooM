"""Update planning, numbered IDs and memory operations migrated from old LooM."""

import json

from ..common.registry import MODULE_REGISTRY
from ..common.utils import parse_json_from_text
from ..envs.memory_bank_utils import MemoryItem, generate_id
from .base import BaseModule

UPDATE_MEMORY_PROMPT = """You are a smart memory manager.
You have two lists of memories:
1. **Existing Memories** (from the database).
2. **New Candidate Memories** (extracted from the latest conversation).

Your goal is to decide how to update the memory database.

**Operations Allowed:**

For **Existing Memories**:
- `NONE`: Keep as is.
- `DEL`: Delete this memory (e.g., if it is contradicted by new info, or merged into a new memory).

For **New Candidate Memories**:
- `ADD`: Add this memory to the database.
- `NONE`: Ignore this memory (e.g., if it's redundant or already covered by existing memories).
- `UPDATE`: Modify this memory before adding (e.g., to merge information from an old memory).

**Merging Strategy:**
If a New Candidate (ID: Y) contains updated information for an Existing Memory (ID: X):
1. Mark Existing Memory X as `DEL`.
2. Mark New Candidate Y as `UPDATE` and provide the merged content.

**Output Format:**
Return a JSON object with a list of operations.
You MUST include an operation for **EVERY** memory item (both Existing and Candidate) in the input lists. Do not skip any IDs.

Format:
```json
{{
  "operations": [
    {{ "id": <id>, "op": "NONE" }},
    {{ "id": <id>, "op": "DEL" }},
    {{ "id": <id>, "op": "ADD" }},
    {{ "id": <id>, "op": "UPDATE", "key": "...", "value": "..." }}
  ]
}}
```

**Task:**

Existing Memories:
{context_memory}

New Candidate Memories:
{candidate_memory}

Output:"""


@MODULE_REGISTRY.register("naive_updater")
class NaiveUpdater(BaseModule):
    def prepare_memory_lists(self, context_memory, extraction_output):
        """
        Prepares numbered lists for prompt and mapping for execution.
        Returns:
            context_list_fmt: List[Dict] with 'id', 'key', 'value'
            candidate_list_fmt: List[Dict] with 'id', 'key', 'value'
            id_map: Dict[int, Any] - maps temp ID to (type, original_obj)
                type 'context': original_obj is MemoryItem
                type 'candidate': original_obj is Dict (from extraction)
        """
        context_memory = [ MemoryItem.from_dict(mem) for mem in context_memory ]

        id_counter = 1
        id_map = {}

        # 1. Process Context Memory
        context_list_fmt = []
        for mem in context_memory:
            temp_id = id_counter
            id_counter += 1
            id_map[temp_id] = ("context", mem)
            context_list_fmt.append({
                "id": temp_id,
                "key": mem.key,
                "value": mem.value
            })

        # 2. Process Extraction Output
        candidate_list_fmt = []
        ext_json = parse_json_from_text(extraction_output)
        if isinstance(ext_json, dict) and "memory_list" in ext_json and isinstance(ext_json["memory_list"], list):
            for item in ext_json["memory_list"]:
                if not isinstance(item, dict): continue
                temp_id = id_counter
                id_counter += 1
                id_map[temp_id] = ("candidate", item)
                candidate_list_fmt.append({
                    "id": temp_id,
                    "key": item.get("key", "Unknown"),
                    "value": item.get("value", "")
                })

        return context_list_fmt, candidate_list_fmt, id_map

    def build_prompt(self, context_memory, extraction_output):
        existing, candidates, _ = self.prepare_memory_lists(context_memory, extraction_output)
        return UPDATE_MEMORY_PROMPT.format(
            context_memory=json.dumps(existing, ensure_ascii=False, indent=2),
            candidate_memory=json.dumps(candidates, ensure_ascii=False, indent=2),
        )

    async def rollout(self, context_memory, extraction_output, generate, **kwargs):
        return await generate(self.build_prompt(context_memory, extraction_output), "update")

    def apply_update_plan(self, store, context_memory, extraction_output, update_output):
        """Execute the old DEL/NONE and ADD/UPDATE/NONE operations on one store.

        Validate the complete plan before writing, so malformed model output
        cannot leave a half-applied update behind.
        """
        plan = parse_json_from_text(update_output)
        if not isinstance(plan, dict) or not isinstance(plan.get("operations"), list):
            raise ValueError("Update output must contain an operations list")
        _, _, id_map = self.prepare_memory_lists(context_memory, extraction_output)
        prepared = []
        seen_ids = set()
        for operation in plan["operations"]:
            if not isinstance(operation, dict):
                raise ValueError("Each memory operation must be an object")
            temp_id = operation.get("id")
            if type(temp_id) is not int or temp_id not in id_map:
                raise ValueError(f"Unknown memory operation ID: {temp_id!r}")
            if temp_id in seen_ids:
                raise ValueError(f"Duplicate memory operation ID: {temp_id}")
            seen_ids.add(temp_id)
            origin_type, origin = id_map[temp_id]
            action = operation.get("op", "NONE")
            if not isinstance(action, str):
                raise ValueError("Memory operation name must be a string")
            action = action.upper()
            allowed = {"NONE", "DEL"} if origin_type == "context" else {"NONE", "ADD", "UPDATE"}
            if action not in allowed:
                raise ValueError(f"Invalid {origin_type} memory operation: {action}")
            item = None
            if action in {"ADD", "UPDATE"}:
                item = MemoryItem(
                    id=generate_id(), key=operation.get("key", origin.get("key")),
                    value=operation.get("value", origin.get("value")),
                    memory_type=origin.get("memory_type", "UserMemory"), tags=origin.get("tags", []),
                )
            prepared.append((action, origin, item))
        for action, origin, item in prepared:
            if action == "DEL" and not store.delete(origin.id):
                raise RuntimeError(f"Failed to delete memory {origin.id}")
            if item is not None and not store.save(item):
                raise RuntimeError("Failed to save candidate memory")
        return id_map
