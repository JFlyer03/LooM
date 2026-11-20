# 项目开发约定

## 命名与目录

- 目录名表达应用场景，类名表达具体方法。
- `long_context_memory` 表示“长上下文任务中使用记忆机制”的应用场景；复现 MemAgent 论文的 Agent 类命名为 `MemAgent`。
- 按应用场景建立包，按职责拆分文件，参照已有的 `react`、`oh_codeact`、`general_react` 和 `swebench` 结构。

```text
loom/agents/long_context_memory/
├── __init__.py
├── long_context_memory_agent.py
└── long_context_memory_runner.py

loom/tasks/long_context_memory/
├── __init__.py
└── utils.py
```

- Agent 负责记忆更新与回答；Runner 负责轨迹执行与奖励赋值；Task 负责数据规范化、任务指令和结果评分。
- 通过各包的 `__init__.py` 统一导出公开类，外部调用和配置使用包级入口，避免依赖内部文件布局。

## 轨迹与类型

- 当前长上下文记忆轨迹的 runtime、Agent 和初始记忆统一在 `generate_trajectory()` 中初始化，`initialize_trajectory()` 留空。
- 其他任务若确实需要独立初始化阶段，应保留其生命周期设计，不机械套用上述约定。
- Agent 实例使用明确的类型注解，构造时显式传入关键字参数：

```python
self.agent: MemAgent = self.agent_cls(
    traj_config=self.cfg,
    infer_engine=self.infer_engine,
    tokenizer=self.tokenizer,
)
```

## 修改与验证

- 移动或重命名时，同步更新导入、注册表、示例配置和相关测试，保持行为一致。
- 优先复用现有实现，只修改完成当前任务所需的部分。

## 沟通方式

- 先说明具体改动和实际验证结果。
- 所有防御性说明，包括限制、未验证事项、风险和适用边界，统一集中放在输出末尾，不穿插在主要结果中。
