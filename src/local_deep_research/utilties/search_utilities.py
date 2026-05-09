import logging
import re
import asyncio
import fcntl
import json
import time


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
    import textwrap

    if not text:
        return text

    log = logger or logging.getLogger(__name__)

    ratio = english_alpha_ratio(text)
    if ratio < 0.90:
        return text

    label_prefix = f"[{label}] " if label else ""
    log.info(f"  🌐 [兜底翻译] {label_prefix}英文占比 {ratio:.1%}, 调用LLM翻译为中文...")

    prompt = textwrap.dedent(f"""
    请将以下循证医学分析内容翻译为中文。严格按以下规则执行：

    【必须保留，原样输出，不翻译】：
    - 所有数值：百分比（如 85.3%）、HR值（如 HR=0.62）、95%CI、P值
    - 所有引用标记：[^^n] 格式的引用编号
    - 所有文献标题行：#### 开头的原文英文标题（含 [^^n]）
    - 所有Markdown格式：标题、列表缩进、空行

    【翻译要求】：
    - 描述性文字用自然的中文叙事，将数据嵌入句中
    - 医学术语使用中国临床指南通用译名
    - 保持原文结构（每段对应原文段落，不要合并或拆分）

    【待翻译内容】：
    {text[:15000]}

    请直接输出翻译后的完整内容，不要加任何额外说明。
    """)

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
