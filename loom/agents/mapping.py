AGENT_GENERATOR_REGISTRY = {
    "loom.agents.oh_codeact.OHCodeActAgent": "loom.agents.base.AgentRunner",
    "loom.agents.react.ReActAgent": "loom.agents.base.AgentRunner",
    "loom.agents.long_context_memory.MemAgent": "loom.agents.base.AgentRunner",
}

AGENT_TRAJECTORY_REGISTRY = {
    "loom.agents.oh_codeact.OHCodeActAgent": "loom.agents.oh_codeact.CodeActTrajectory",
    "loom.agents.react.ReActAgent": "loom.agents.react.ReActTrajectory",
    "loom.agents.long_context_memory.MemAgent": "loom.agents.long_context_memory.LongContextMemoryTrajectory",
}
