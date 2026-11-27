# Operator Allocation Agent

自动处理算子开发需求的 Agent 系统，将需求表中的算子识别归属仓库、过滤无需开发的算子，并关联或新增到开发表。

## 项目结构

```
operator-agent/
├── packages/
│   ├── agent_framework/        # 通用 Agent 框架（LLM 调用、MCP、ReAct/PlanSolve、Tracing）
│   └── operator_allocation/    # 算子分配业务逻辑
├── resources/
│   └── no_dev/                 # no_dev 过滤运行时资产（受版本控制）
│       ├── torch_no_dev_ops.md          # FlagGems 已确认 no_dev 算子列表（人工维护）
│       ├── torch_no_dev_ops_pending.md  # FlagGems LLM 判定待确认列表（自动追加）
│       ├── vllm_no_dev_ops.md           # FlagGems-vllm 已确认 no_dev 算子列表
│       ├── vllm_no_dev_ops_pending.md   # FlagGems-vllm LLM 判定待确认列表
│       └── prompts/
│           ├── torch_no_dev_prompt.md   # FlagGems Tier 2 LLM 分类 system prompt
│           └── vllm_no_dev_prompt.md    # FlagGems-vllm Tier 2 LLM 分类 system prompt
├── tests/                      # 框架单元测试与业务集成测试
├── data/                       # 本地业务数据（gitignore，不进入版本控制）
│   ├── requirement.xlsx        # 需求表输入
│   └── dev_table.xlsx          # 开发表输入
├── repo/                       # 外部库本地克隆（gitignore，不进入版本控制）
│   ├── pytorch/
│   ├── vllm/
│   └── sglang/
├── config.toml                 # 项目配置（模板，提交到版本控制）
├── config.local.toml           # 本地敏感配置覆盖（gitignore）
└── install.sh                  # 一键安装脚本
```

## 配置文件

项目使用两个 TOML 配置文件，均位于项目根目录：

| 文件 | 用途 | ���否提交 |
|------|------|---------|
| `config.toml` | 配置模板，包含所有配置项及默认值 | ✅ 提交 |
| `config.local.toml` | 本地覆盖，存放敏感值（API key 等） | ❌ gitignore |

运行时两个文件合并加载，`config.local.toml` 中的值优先。

### config.toml 各节说明

**`[llm]`** — LLM 服务配置

```toml
[llm]
base_url = ""          # LLM API endpoint，如 "https://api.openai.com/v1"
api_key = ""           # API 密钥，建议写在 config.local.toml 或设置环境变量 LLM_API_KEY
model = ""             # 模型名称，如 "gpt-4o"
timeout = 120          # 单次请求超时（秒）
max_retries = 3        # LLM 调用失败重试次数
structured_output_mode = "openai"  # 结构化输出模式："openai" | "claude" | "fallback"
```

**`[repos]`** — 源码仓库配置

```toml
[repos]
base_dir = ""          # 本地 repo 克隆的根目录，如 "/mnt/d/AI_Infra/operator-agent/repo"
                       # 该目录下应存在 pytorch/、vllm/、sglang/ 等子目录
```

**`[excel]`** — Excel 文件路径

```toml
[excel]
dev_table_path = ""    # 开发表 Excel 路径，也可通过 --dev-table 命令行参数指定
output_dir = "output"  # 输出目录（结果 Excel、日志、trace）
```

**`[lib_repo_mapping]`** — 开发表与源码仓库的映射

key 为开发表分组名（Excel sheet 名），value 为 `repos.base_dir` 下的目录名。新增仓库只需在此添加一行，无需改代码。

```toml
[lib_repo_mapping]
FlagGems = "pytorch"
FlagGems-vllm = "vllm"
FlagGems-sglang = "sglang"
```

**`[no_dev_filter]`** — no_dev 过滤配置（per-lib）

每个算子库单独配置。未配置的库跳过过滤，所有算子直接进入分发。

```toml
[no_dev_filter.FlagGems]
known_list_path = "resources/no_dev/torch_no_dev_ops.md"      # 已确认 no_dev 列表
pending_list_path = "resources/no_dev/torch_no_dev_ops_pending.md"  # 待确认列表（自动追加）
prompt_path = "resources/no_dev/prompts/torch_no_dev_prompt.md"     # LLM 分类 prompt
llm_batch_size = 8          # Tier 2 每批算子数
confidence_threshold = "high"  # 低于此置信度保守处理为 needs_dev
```

**`[agent]`** — Agent 运行参数

```toml
[agent]
identify_max_turns = 20        # 算子识别 ReAct 循环上限
process_max_turns = 30         # 任务分发 ReAct 循环上限
max_retries = 10               # PlanSolve reflexion 重试上限
max_tool_result_chars = 8000   # 工具返回结果截断长度（防止大文件淹没上下文）
context_compress_threshold = 30000  # 触发上下文压缩的估算 token 数
```

### 本地配置示例

在项目根目录创建 `config.local.toml`，填入敏感值：

```toml
[llm]
base_url = "https://your-llm-endpoint/v1"
api_key = "sk-xxxxxx"
model = "gpt-4o"

[repos]
base_dir = "/mnt/d/AI_Infra/operator-agent/repo"
```

## 快速开始

### 1. 安装依赖

```bash
# 开发环境（推荐，使用 uv）
uv sync

# 生产环境（安装为 CLI 工具）
./install.sh --venv /opt/agent-env
```

### 2. 配置

复制并填写本地配置：

```bash
# config.local.toml 已在 .gitignore 中，直接创建即可
cp config.toml config.local.toml
# 编辑 config.local.toml，填入 base_url、api_key、model、repos.base_dir
```

### 3. 运行

```bash
# 开发环境
uv run python -m operator_allocation \
  --config config.toml \
  --input data/requirement.xlsx \
  --dev-table data/dev_table.xlsx

# 生产环境（安装后）
operator-allocation \
  --config config.toml \
  --input /path/to/requirement.xlsx \
  --dev-table /path/to/dev_table.xlsx
```

输出文件写入 `output/`（由 `excel.output_dir` 配置）：
- `result.xlsx` — 需求结果 + 更新后的开发表
- `logs/{trace_id}.jsonl` — 结构化执行日志
- `traces/{trace_id}.json` — 完整 trace

## 运行测试

```bash
uv run pytest tests/
```

## License

Apache License 2.0
