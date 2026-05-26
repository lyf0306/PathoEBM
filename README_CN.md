# PathoEBM

AI 驱动的妇科肿瘤循证医学（EBM）管线。输入 MDT 诊疗方案，执行多轮深层文献检索（PubMed + ClinicalTrials.gov），输出经过证据校验的最终报告并附上弥合证据缺口的引用文献。

## 特性

- **自动混合模型路由** — 外部 API（DeepSeek V4、GPT-4.1 mini），自动回退到本地 vLLM
- **结构化合并症分类** — 基于关键词的分级（危急感染 / 主要合并症 / 伴随发现）+ LLM 安全网二次分类
- **多轮深层搜索** — 迭代检索 PubMed + ClinicalTrials.gov，辅以 FDA 药品标签查询
- **证据锚定报告生成** — 合并基线参考文献与新证据引用，自动去除前言、脱敏处理
- **REST API + 任务管理** — 异步任务队列、进度追踪、取消、30 天自动清理
- **并发基础设施** — 连接池、熔断器、速率限制器、优雅关闭

## 项目结构

```
PathoEBM-main/
├── pyproject.toml
├── local_deep_research/         # 主包
│   ├── main.py                  # CLI 入口 & 管线编排
│   ├── config.py                # 配置 & LLM 工厂函数
│   ├── search_system.py         # AdvancedSearchSystem 搜索引擎
│   ├── connect_mcp.py           # MCP 工具服务器客户端
│   ├── tool_selector.py         # LLM 工具选择器
│   ├── tool_executor.py         # MCP 工具执行器
│   ├── _settings/               # 配置文件 + 提示词模板
│   ├── agents/                  # 专用 LLM Agent
│   ├── api/                     # FastAPI REST 服务
│   ├── concurrency/             # 连接池、速率限制器等
│   ├── evaluation/              # 报告质量评估
│   ├── pipeline/                # 搜索规划器、知识处理器
│   ├── prompts/                 # PromptManager 版本化提示词加载器
│   ├── skills/                  # NCCN 随访 & 预后模块
│   ├── utilities/               # 通用工具函数
│   └── tools/                   # 工具信息 & 嵌入缓存
├── scripts/                     # 工具/调试脚本
│   ├── check_mcp.py             # MCP 端点健康探测
│   ├── debug_selector.py        # 工具选择器调试
│   ├── evaluate_local.py        # 批量评估框架
│   ├── score_evaluation_results.py  # LLM 评委评分
│   ├── init_tools.py            # 工具白名单初始化
│   └── fix_clinical_tools.py    # 临床工具配置修复
└── tests/                       # 测试文件
    ├── test_MCP.py              # MCP 连接测试
    ├── test_pubmed_direct.py    # PubMed 直接搜索测试
    ├── test_pubmed_query.py     # PubMed 查询格式测试
    ├── test_pubmed_raw.py       # PubMed 原始返回检查
    └── test_gog0258_diagnosis.py  # GOG-0258 检索缺口诊断
```

## 环境要求

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip
- 至少一个 LLM 的 API Key：DeepSeek、OpenAI，或本地 vLLM 实例
- 运行中的 MCP 工具服务器（PubMed、ClinicalTrials.gov、FDA 药品标签）

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd PathoEBM-main

# 使用 uv 安装（推荐）
uv sync

# 或使用 pip 安装
pip install -e .
```

## 配置

创建 `local_deep_research/_settings/deploy_config.toml`：

```toml
[openai]
api_key = "sk-..."
base_url = "https://api.openai.com/v1"

[deepseek]
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"

[embedding]
api_key = "sk-..."
model = "text-embedding-3-small"

[local]
base_url = "http://localhost:8000/v1"
model_name = "qwen3-32b"

[mcp]
url = "http://localhost:8788"

[model]
provider = "auto"          # auto | deepseek | gpt | local
fallback = "local"         # local | none

[api]
api_key = ""               # 可选：设置后启用 X-API-Key 认证

[storage]
jobs_dir = "api/jobs_output"
retention_days = 30
```

## 使用方式

### CLI 模式

```bash
python -m local_deep_research.main
```

按交互提示粘贴诊疗方案（Markdown 格式）或从文件加载，然后选择模型和搜索轮数。

### API 服务器

```bash
uvicorn local_deep_research.api.app:app --host 0.0.0.0 --port 8000
```

## API 接口

| 方法 | 路径 | 说明 |
|--------|------|------|
| `GET` | `/health` | 健康检查（状态、活跃任务数、运行时长） |
| `POST` | `/jobs` | 提交新的 EBM 管线任务 |
| `GET` | `/jobs` | 列出所有任务（最新优先） |
| `GET` | `/jobs/{job_id}` | 查询任务状态及进度快照 |
| `GET` | `/jobs/{job_id}/result` | 获取已完成任务的最终报告 |
| `DELETE` | `/jobs/{job_id}` | 取消正在运行/等待的任务 |

### 示例：提交任务

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "treatment_context": "# MDT 会诊报告\n...",
    "model_choice": "auto",
    "max_iterations": 2
  }'
```

### 示例：获取结果

```bash
curl http://localhost:8000/jobs/{job_id}/result \
  -H "X-API-Key: your-api-key"
```

## 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试
python tests/test_MCP.py
```

## 运行脚本

```bash
# 检查 MCP 服务器连通性
python scripts/check_mcp.py

# 调试工具选择逻辑
python scripts/debug_selector.py

# 批量评估
python scripts/evaluate_local.py
```

## 依赖

核心运行时依赖：

- **LLM**：`langchain`、`langchain-openai`、`langchain-deepseek`、`openai`
- **MCP**：`mcp`、`langchain-mcp-adapters`
- **API**：`fastapi`、`uvicorn`、`pydantic`
- **数据处理**：`datasets`、`scikit-learn`、`networkx`
- **文档工具**：`pymupdf`、`python-docx`、`python-pptx`、`openpyxl`、`markdown2`

完整列表见 `pyproject.toml`。

## 许可证

专有软件。保留所有权利。
