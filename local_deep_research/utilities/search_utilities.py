import logging
import re
import asyncio
import fcntl
import json
import time
from ..prompts import prompt_manager


def write_log_process_safe(log_path, message):
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            fcntl.flock(log_file, fcntl.LOCK_EX)
            log_file.write(message)
            log_file.flush()
            fcntl.flock(log_file, fcntl.LOCK_UN)
    except Exception as log_exc:
        print(f"Failed to write to log file: {log_exc}")


def write_json_log_process_safe(log_path, new_dict):
    """
    Write a new dictionary to a JSON log file. The file content is a list of dictionaries.
    
    Args:
        log_path: JSON log file path
        new_dict: The new dictionary to be appended
    """
    try:
        data = []
        with open(log_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                if content:
                    data = json.loads(content)
                
                data.append(new_dict)
                
                # Clear file and write
                f.seek(0)
                f.truncate()
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Failed to write to JSON log file: {e}")
        


async def invoke_with_timeout_and_retry(llm, messages, timeout=90.0, max_retries=3,
                                        retry_delay=60.0, circuit_breaker=None):
    """带超时、重试、熔断保护的 LLM 调用。

    Args:
        circuit_breaker: 可选 CircuitBreaker 实例；未传时自动使用 "llm" 熔断器。
    """
    if circuit_breaker is None:
        from ..concurrency.circuit_breaker import get_circuit_breaker
        circuit_breaker = get_circuit_breaker("llm")

    for attempt in range(max_retries):
        try:
            # 熔断器快速失败检查
            if circuit_breaker.is_open():
                from ..concurrency.circuit_breaker import CircuitBreakerOpenError
                raise CircuitBreakerOpenError(circuit_breaker.name)

            _t0 = time.time()
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=timeout
            )
            _elapsed = time.time() - _t0
            circuit_breaker.record_success()
            logging.info(f"LLM invoke 成功 | attempt {attempt+1}/{max_retries} | 耗时 {_elapsed:.1f}s")
            return response

        except asyncio.CancelledError:
            logging.warning("LLM invoke 被取消 (CancelledError)")
            raise

        except asyncio.TimeoutError:
            circuit_breaker.record_failure()
            logging.warning(f"LLM invoke 超时 | attempt {attempt+1}/{max_retries} | timeout={timeout}s")
            if attempt < max_retries - 1:
                logging.info(f"将在 {retry_delay}s 后重试...")
                await asyncio.sleep(retry_delay)

        except Exception as e:
            circuit_breaker.record_failure()
            # 熔断器打开时直接抛出，不重试
            if "CircuitBreakerOpenError" in type(e).__name__:
                raise
            logging.warning(f"LLM invoke 失败 | attempt {attempt+1}/{max_retries} | {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                logging.info(f"将在 {retry_delay}s 后重试...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error("已达最大重试次数，放弃。")
                raise Exception(f"Failed after {max_retries} attempts: {e}")

    raise Exception("Failed to get a response after multiple attempts.")


def remove_think_tags(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned:
        cleaned = re.sub(r".*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_links_from_search_results(search_results: list) -> list:
    """
    Extracts links and titles from a list of search result dictionaries.

    Each dictionary is expected to have at least the keys "title" and "link".

    Returns a list of dictionaries with 'title' and 'url' keys.
    """
    links = []
    for result in search_results:
        try:
            title = result.get("title", "").strip()
            url = result.get("link", "").strip()
            index = result.get("index", "")
            if type(index) == str:
                index = index.strip()

            if title and url:
                links.append({"title": title, "url": url, "index": index})
        except Exception:
            continue
    return links


def format_links(links):
    formatted_links = ""
    formatted_links += "SOURCES:\n"
    for i, link in enumerate(links, 1):
        formatted_links += f"{link['index']}. {link['title']}\n   URL: {link['url']}\n"
    formatted_links += "\n"
    return formatted_links


def format_findings_to_text(findings_list, current_knowledge, questions_by_iteration):
    formatted_text = "COMPLETE RESEARCH OUTPUT \n\n"

    # Store the full current knowledge

    formatted_text += f"{current_knowledge}\n\n"
    formatted_text += "=" * 80 + "\n\n"

    # Store questions by iteration
    formatted_text += "SEARCH QUESTIONS BY ITERATION:\n"
    for iter_num, questions in questions_by_iteration.items():
        formatted_text += f"\nIteration {iter_num}:\n"
        for i, q in enumerate(questions, 1):
            formatted_text += f"{i}. {q}\n"
    formatted_text += "\n" + "=" * 80 + "\n\n"

    # Store detailed findings
    formatted_text += "DETAILED FINDINGS:\n\n"
    all_links = []  # To collect all sources

    for finding in findings_list:
        # Phase header
        formatted_text += f"{'=' * 80}\n"
        formatted_text += f"PHASE: {finding['phase']}\n"
        formatted_text += f"{'=' * 80}\n\n"

        # If this is a follow-up phase, show the corresponding question
        if finding["phase"].startswith("Follow-up"):
            iteration = int(finding["phase"].split(".")[0].split()[-1])
            question_index = int(finding["phase"].split(".")[-1]) - 1
            if iteration in questions_by_iteration and question_index < len(
                questions_by_iteration[iteration]
            ):
                formatted_text += f"SEARCH QUESTION:\n{questions_by_iteration[iteration][question_index]}\n\n"

        # Content
        formatted_text += f"CONTENT:\n{finding['content']}\n\n"

        # Search results if they exist
        if "search_results" in finding:
            # formatted_text += "SEARCH RESULTS:\n"
            # formatted_text += f"{finding['search_results']}\n\n"

            # Extract and format links for this finding
            links = extract_links_from_search_results(finding["search_results"])
            if links:
                formatted_text += "SOURCES USED IN THIS SECTION:\n"
                for i, link in enumerate(links, 1):
                    formatted_text += f"{i}. {link['title']}\n   URL: {link['url']}\n"
                formatted_text += "\n"
                all_links.extend(links)

        formatted_text += f"{'_' * 80}\n\n"

    # Add summary of all sources at the end
    if all_links:
        formatted_text += "\nALL SOURCES USED IN RESEARCH:\n"
        formatted_text += "=" * 80 + "\n\n"
        seen_urls = set()  # To prevent duplicates
        for i, link in enumerate(all_links, 1):
            if link["url"] not in seen_urls:
                formatted_text += f"{i}. {link['title']}\n   URL: {link['url']}\n"
                seen_urls.add(link["url"])
        formatted_text += "\n" + "=" * 80 + "\n"

    return formatted_text


def print_search_results(search_results):
    formatted_text = ""
    links = extract_links_from_search_results(search_results)
    if links:
        formatted_text = format_links(links=links)
    logging.info(formatted_text)


def english_alpha_ratio(text: str) -> float:
    """Ratio of Latin alphabet chars among all alphabetic chars (0.0–1.0)."""
    if not text:
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    latin = sum(1 for c in alpha if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    return latin / len(alpha)


async def ensure_chinese_output(
    text: str,
    llm,
    label: str = "",
    logger: logging.Logger = None,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> str:
    """If text is >90% English, translate via LLM to Chinese.

    Args:
        text: The text to check and potentially translate.
        llm: A ChatOpenAI (or compatible) instance for translation.
        label: Short descriptor for logging (e.g. "MDTReport", "MainPlan").
        logger: Logger instance; uses root logger if None.
        timeout: Per-call timeout in seconds.
        max_retries: Max retry count for the translation LLM call.

    Returns:
        Translated text if >90% English, otherwise the original text unchanged.
    """
    if not text:
        return text

    log = logger or logging.getLogger(__name__)

    ratio = english_alpha_ratio(text)
    if ratio < 0.90:
        return text

    label_prefix = f"[{label}] " if label else ""
    log.info(f"  🌐 [兜底翻译] {label_prefix}英文占比 {ratio:.1%}, 调用LLM翻译为中文...")

    prompt = prompt_manager.get("chinese_fallback").format(
        text=text[:15000],
    )

    for attempt in range(max_retries):
        try:
            _t0 = time.time()
            resp = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=timeout,
            )
            _elapsed = time.time() - _t0
            translated = remove_think_tags(resp.content).strip() if hasattr(resp, 'content') else str(resp).strip()
            if translated and english_alpha_ratio(translated) < ratio * 0.7:
                log.info(
                    f"  ✅ [兜底翻译] {label_prefix}完成, "
                    f"英文占比 {ratio:.1%} → {english_alpha_ratio(translated):.1%}"
                    f" (耗时 {_elapsed:.1f}s)"
                )
                return translated
            else:
                log.warning(
                    f"  ⚠️ [兜底翻译] {label_prefix}"
                    f"翻译后英文占比未显著降低 ({english_alpha_ratio(translated):.1%}), "
                    f"返回原文"
                )
                return text
        except asyncio.TimeoutError:
            log.warning(f"  ⚠️ [兜底翻译] {label_prefix}超时 attempt {attempt+1}/{max_retries}")
        except Exception as e:
            log.warning(f"  ⚠️ [兜底翻译] {label_prefix}失败 attempt {attempt+1}: {e}")

    log.warning(f"  ⚠️ [兜底翻译] {label_prefix}所有尝试均失败, 返回原文")
    return text


# ── LLM 输出清洗 ──

_LLM_PREAMBLE_RE = re.compile(
    r'^('
    # ── 中文单句开场 ──
    r'好的[，,.]?\s*'
    r'|明白[，,]?\s*我[^。.]*?[。.]\s*'
    r'|收到[，,]?\s*我[^。.]*?[。.]\s*'
    # ── 英文单句开场 ──
    r'|OK[，,]\s*I\'?ll\s[^.]*?\.\s*'
    r'|Understood[，,]\s*I\s+will\s[^.]*?\.\s*'
    r'|Certainly[，,]\s*(?:I\s+will|here\s+is)[^.]*?\.\s*'
    r'|Sure[，,]\s*(?:I\'?ll|I\s+will|here)[^.]*?\.\s*'
    r')',
    re.IGNORECASE,
)

_LLM_META_SEP_RE = re.compile(r'^.*?^---\s*$', re.MULTILINE | re.DOTALL)

# 中文编号标题：匹配行首 "1、" "1." "## 一、" 等章节开头
_LLM_NUMBERED_HEADER_RE = re.compile(
    r'^.*?^(?=\d+[、.]|##\s+[一二三四五六七八九十\d]+[、.])',
    re.MULTILINE | re.DOTALL,
)


def strip_llm_preamble(text: str) -> str:
    """移除 LLM 输出的冗余开场白和元 commentary。

    策略：
    1. 以第一个 --- 为界（LLM 常用的元/正文分隔符）。
    2. 如无 ---，以第一个中文编号标题为界（如 "1、**合并症管理**"）。
    3. 否则仅移除行首的简单礼貌语（好的、OK 等）。
    """
    if not text:
        return text
    # 策略 1：以第一个 --- 为界
    m = _LLM_META_SEP_RE.match(text)
    if m and m.end() > 3 and len(text) - m.end() > 60:
        prefix_len = len(text[:m.start()].strip())
        if prefix_len < 600:
            text = text[m.end():]
    else:
        # 策略 2：以第一个中文编号标题为界
        m2 = _LLM_NUMBERED_HEADER_RE.match(text)
        if m2 and m2.end() > 3 and len(text) - m2.end() > 60:
            prefix_len = len(text[:m2.end()].strip())
            if prefix_len < 600:
                text = text[m2.end():]
        else:
            # 策略 3：移除行首简单礼貌语
            text = _LLM_PREAMBLE_RE.sub('', text, count=1)
    # 清理残留的 --- 分隔线
    text = re.sub(r'^\s*---\s*\n+', '', text)
    return text.strip()


def depersonalize_report(text: str) -> str:
    """去除报告中的第二人称，转为临床第三人称。

    将 "您接受的..."、"您的合并症..." 等面向患者的语言
    转写为 "患者接受的..."、"患者的合并症..."，使报告
    符合医生间 MDT 会诊的专业语气。
    """
    if not text:
        return text
    # 您/你 → 患者
    text = text.replace('您', '患者')
    text = text.replace('你', '患者')
    # 修复叠词
    while '患者患者' in text:
        text = text.replace('患者患者', '患者')
    # 面向患者的 "请务必" → 面向医生的 "建议"
    text = text.replace('请务必', '建议')
    return text
