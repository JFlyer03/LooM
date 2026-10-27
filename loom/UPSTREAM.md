# Source and local modifications

This is loom's maintained runtime package: `loom`.

It is derived from `skyrl-agent/skyrl_agent` in
<https://github.com/NovaSky-AI/SkyRL>, local base commit
`e303d385f77c682ef166340025bcc4beee00bc64`.
The copied workspace included two Chinese explanatory comments in
`integrations/skyrl_train/skyrl_train_main.py`.

Upstream attribution and Apache-2.0 license are retained. The sibling
`skyrl-agent` copy and the root `SkyRL` checkout are reference sources;
Python imports in this package use `loom`, never the separate
installed `skyrl_agent` package.

Local additions include MemAgent memory modules, task/trajectory implementations,
lazy backend imports, and an adapter for SkyRL's native step-wise trainer.
Existing SWE/ReAct/VERL/Tinker integrations are retained and namespaced but have
not been validated with their full optional dependencies.

MemAgent's prompts and workflow are migrated from the legacy implementation now
stored in `loom_agent`. Its original trainers are not used by the new path.
