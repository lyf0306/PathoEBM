import ast
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from colorlog import ColoredFormatter
from fuzzywuzzy import fuzz

# ============= 核心修改：临床专用实体 =============
biological_entities = [
    "Clinical",
    "Drug/Drug class",
    "Disease",
    "Biomarker",
]

# ============= 以下代码保持不变 =============

def extract_and_convert_dict(text):
    """从文本中提取字典"""
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        dict_str = match.strip()
        parsers = [ast.literal_eval, eval]
        
        for parser in parsers:
            try:
                if parser == eval:
                    if any(dangerous in dict_str for dangerous in [
                        "import", "__", "exec", "eval", "open", "file"
                    ]):
                        continue
                
                result = parser(dict_str)
                if isinstance(result, dict):
                    return result
            except:
                continue
    
    return None


def exact_match_entity_type(entity_type, entity_list=biological_entities, threshold=60):
    """
    临床实体匹配（降低了阈值以适应临床术语的多样性）
    
    Examples:
        "endometrial cancer" -> "Disease"
        "POLE mutation" -> "Biomarker"
        "chemotherapy regimen" -> "Drug/Drug class"
    """
    if entity_type in entity_list:
        return entity_type
    
    if not entity_type or not isinstance(entity_type, str):
        print(f"Invalid entity type: {entity_type}, returning 'Clinical'")
        return "Clinical"  # 默认归类为临床决策
    
    entity_type = entity_type.lower().strip()
    
    max_similarity = 0
    best_match = "Clinical"  # 默认类别改为 Clinical
    
    for category in entity_list:
        category_lower = category.lower()
        similarity = fuzz.partial_ratio(entity_type, category_lower)
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = category
        
        if entity_type == category_lower:
            return category
    
    if max_similarity >= threshold:
        return best_match
    else:
        # 临床系统更宽容：低于阈值时尝试关键词匹配
        if any(kw in entity_type for kw in ["cancer", "tumor", "carcinoma", "adenocarcinoma"]):
            return "Disease"
        elif any(kw in entity_type for kw in ["drug", "therapy", "treatment", "chemotherapy"]):
            return "Drug/Drug class"
        elif any(kw in entity_type for kw in ["mutation", "marker", "expression", "mismatch"]):
            return "Biomarker"
        else:
            print(f"Entity '{entity_type}' classified as Clinical (similarity {max_similarity})")
            return "Clinical"


def generate_tools_descriptions(tool_list):
    """生成工具描述（保持不变）"""
    all_tool_desc = ""
    for tool in tool_list:
        tool_name = tool.name
        tool_desc = tool.description
        tool_args_schema = tool.args_schema
        description = (
            "#" * 20
            + f"\nTool Name: {tool_name}\nTool Purpose: {tool_desc}\nTool Input Schema: {tool_args_schema}\n\n"
        )
        all_tool_desc += description
    return all_tool_desc


# ============= Logger 类（保持不变）=============
class ResearchLogger:
    def __init__(self, name: str = "clinical_audit", debug_mode: bool = False):
        self.logger = logging.getLogger(name)
        self.debug_mode = debug_mode
        self._setup_logger()
    
    def _setup_logger(self):
        if self.logger.handlers:
            return
        
        level = logging.DEBUG if self.debug_mode else logging.INFO
        self.logger.setLevel(level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        colored_formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s%(reset)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(colored_formatter)
        
        if self.debug_mode:
            log_dir = Path("dist/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"clinical_audit_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.DEBUG)
            
            file_formatter = logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def progress(self, message: str, progress: Optional[int] = None, **kwargs):
        progress_text = f" ({progress}%)" if progress is not None else ""
        self.logger.info(f"🔄 {message}{progress_text}", extra=kwargs)
    
    def result(self, message: str, count: int = 0, **kwargs):
        self.logger.info(f"📊 {message} (count: {count})", extra=kwargs)
    
    def tool_call(self, tool_name: str, query: str, **kwargs):
        self.logger.info(
            f"🔧 {tool_name}: {query[:50]}{'...' if len(query) > 50 else ''}",
            extra=kwargs,
        )


# ============= 其他工具函数（保持不变）=============
def detect_content_type(url: str) -> str:
    url_lower = url.lower()
    
    if any(url_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]):
        return "image"
    elif any(url_lower.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mkv"]):
        return "video"
    elif url_lower.endswith(".pdf"):
        return "pdf"
    else:
        return "iframe"


def clean_text_format(text: str) -> str:
    if not text:
        return ""
    
    text = str(text).replace("\\n", "\n").replace("\\t", " ")
    
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    return "\n".join(cleaned_lines)


def extract_json_from_response(response_text: str) -> Optional[dict]:
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


# 全局 Logger 实例
research_logger = ResearchLogger("clinical_audit", debug_mode=False)

def log_debug(message: str, **kwargs):
    research_logger.debug(message, **kwargs)

def log_info(message: str, **kwargs):
    research_logger.info(message, **kwargs)

def log_warning(message: str, **kwargs):
    research_logger.warning(message, **kwargs)

def log_error(message: str, **kwargs):
    research_logger.error(message, **kwargs)

def log_progress(message: str, progress: Optional[int] = None, **kwargs):
    research_logger.progress(message, progress, **kwargs)

def log_tool_call(tool_name: str, query: str, **kwargs):
    research_logger.tool_call(tool_name, query, **kwargs)