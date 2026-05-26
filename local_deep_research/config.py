# local_deep_research/config.py
import logging
import os
from types import SimpleNamespace
from langchain_openai import ChatOpenAI
from pathlib import Path
import tomllib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOY_CONFIG_PATH = PROJECT_ROOT / "_settings" / "deploy_config.toml"
CONFIG_PATH = PROJECT_ROOT / "_settings" / ".secrets.toml"

# 优先加载 deploy_config.toml，不存在则回退到 .secrets.toml
if DEPLOY_CONFIG_PATH.exists():
    with DEPLOY_CONFIG_PATH.open("rb") as f:
        secrets = tomllib.load(f)
    logger.info("Loaded config from deploy_config.toml")
elif CONFIG_PATH.exists():
    with CONFIG_PATH.open("rb") as f:
        secrets = tomllib.load(f)
    logger.info("Loaded config from .secrets.toml (deploy_config.toml not found)")
else:
    print(f"⚠️ Warning: No config file found. Create deploy_config.toml in _settings/")
    secrets = {}

def get_secret(section, key, default=""):
    return secrets.get(section, {}).get(key, default)

settings = SimpleNamespace(
    quick    = SimpleNamespace(iteration=2, questions_per_iteration=4),
    detailed = SimpleNamespace(iteration=2, questions_per_iteration=6),
    embedding_api_key = get_secret("embedding", "api_key", "EMPTY_KEY"),
    embedding_cache   = get_secret("embedding", "cache", "embedding_cache.pkl"),
)

endpoint_openai_api_base_url   = get_secret("openai", "api_base", "https://api.openai.com/v1")
endpoint_openai_api_key        = get_secret("openai", "api_key", "EMPTY_KEY")

deepseek__openai_api_base_url  = get_secret("deepseek", "api_base", "https://api.deepseek.com")
deepseek_openai_api_key        = get_secret("deepseek", "api_key", "EMPTY_KEY")

# 🔧 修正MCP服务器URL和端口
mcp_url = get_secret("mcp", "server_url", "http://localhost:8788")  # ✅ 改为8788

template_embedding_api_base_url = get_secret("template", "api_base", "")
template_embedding_api_key      = get_secret("template", "api_key", "EMPTY_KEY")

def get_gpt4_1() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_gpt4_1_mini() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_claude_openai() -> ChatOpenAI:
    return ChatOpenAI(
        model="claude-3-opus-20240229",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_deepseek_r1() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-reasoner",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_deepseek_v3() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_deepseek_v4() -> ChatOpenAI:
    """DeepSeek V4 Pro — 外部 API。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_local_model(temperature: float = 0.1, request_timeout: float = 600.0):
    """
    连接本地 vLLM 部署的模型（配置从 deploy_config.toml [local] 读取）。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Please install langchain_openai")
    local_host = get_secret("local", "host", "http://localhost")
    local_port = get_secret("local", "port", "8000")
    local_model_name = get_secret("local", "model_name", "OriClinical")
    local_max_tokens = int(get_secret("local", "max_tokens", "16384"))
    local_timeout = float(get_secret("local", "request_timeout", str(request_timeout)))
    return ChatOpenAI(
        model=local_model_name,
        base_url=f"{local_host}:{local_port}/v1",
        api_key="EMPTY",
        temperature=temperature,
        max_tokens=local_max_tokens,
        request_timeout=local_timeout,
    )

def get_model_provider() -> str:
    """读取配置文件中的模型提供方偏好。

    返回值:
        "local" | "deepseek" | "gpt" | "auto"
    默认 "auto" 表示自动检测（先尝试本地，不可用则回退云端）。
    """
    return get_secret("model", "provider", "auto")


def get_model_fallback() -> str:
    """读取配置中的回退策略。"""
    return get_secret("model", "fallback", "local")


def check_external_model_health(llm, timeout: float = 5.0) -> bool:
    """快速检测外部 LLM API 是否可达（发送 "Hi" 测试调用）。"""
    try:
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = timeout
        llm.invoke("Hi")
        return True
    except Exception:
        return False
