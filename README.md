# Operator Allocation Agent

自动处理算子开发需求的 Agent 系统，将需求表中的算子识别归属仓库并关联开发表。

## 项目结构

- `packages/agent_framework/` — 通用 agent 框架（LLM 调用、MCP 通信、ReAct/PlanSolve 循环、Tracing）
- `packages/operator_allocation/` — 算子分配业务逻辑
- `tests/` — 框架单元测试与业务集成测试

## 快速开始

```bash
# 开发环境
uv run python -m operator_allocation --config config.toml --input requirement.xlsx

# 生产环境（安装后）
./install.sh --venv /opt/agent-env
operator-allocation --config config.toml --input requirement.xlsx
```

## License

Apache License 2.0
