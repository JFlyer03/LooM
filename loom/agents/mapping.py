AGENT_GENERATOR_REGISTRY = {
    "loom.agents.memory.MemAgent": "loom.agents.memory.MemoryAgentRunner",
    "loom.agents.oh_codeact.OHCodeActAgent": "loom.agents.base.AgentRunner",
    "loom.agents.react.ReActAgent": "loom.agents.base.AgentRunner",
}

AGENT_TRAJECTORY_REGISTRY = {
    "loom.agents.memory.MemAgent": "loom.agents.memory.MemoryTrajectory",
    "loom.agents.oh_codeact.OHCodeActAgent": "loom.agents.oh_codeact.CodeActTrajectory",
    "loom.agents.react.ReActAgent": "loom.agents.react.ReActTrajectory",
}
