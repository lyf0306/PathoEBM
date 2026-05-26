import asyncio
import logging
import sys
import os
import json
import re
import time

from .config import settings, get_local_model, get_gpt4_1_mini, get_deepseek_v4, get_model_provider
from .search_system import AdvancedSearchSystem
from .utilities.search_utilities import invoke_with_timeout_and_retry, strip_llm_preamble, depersonalize_report
from .concurrency.task_manager import gather_safe

# 配置基础日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MDT_DELIMITER = "# 妇科肿瘤 MDT 初始会诊报告"


async def ainput(prompt: str = "") -> str:
    """异步版 input()——在 daemon 线程中运行，不阻塞事件循环。

    使用 daemon 线程确保 Ctrl+C 取消主任务后，残留的 I/O 线程不会阻止进程退出。
    轮询间隔 100ms，信号延迟最多 100ms。
    """
    import sys as _sys
    from threading import Thread
    if prompt:
        _sys.stdout.write(prompt)
        _sys.stdout.flush()

    result: list = []
    exc: list = []

    def _read():
        try:
            result.append(_sys.stdin.readline().rstrip("\n"))
        except Exception as e:
            exc.append(e)

    t = Thread(target=_read, daemon=True)
    t.start()
    while t.is_alive():
        await asyncio.sleep(0.1)

    if exc:
        raise exc[0]
    return result[0] if result else ""

def check_local_model_health(timeout: float = 2.0) -> bool:
    """检测本地模型(vLLM)是否可用"""
    print("🔍 Checking local model connectivity...", end=" ", flush=True)
    try:
        llm = get_local_model(temperature=0.1)
        if hasattr(llm, "request_timeout"):
            llm.request_timeout = timeout
        llm.invoke("Hi") 
        print("✅ ONLINE")
        return True
    except Exception as e:
        print(f"❌ OFFLINE (Error: {str(e)[:50]}...)")
        return False

def read_context_from_file(file_path: str) -> str:
    """Helper: 从文件读取治疗方案内容"""
    try:
        file_path = file_path.strip('"').strip("'")
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading context file: {e}")
        return ""

def parse_graph_ec_report(raw_text: str):
    """
    分离 PathoRAG 生成的正文和参考文献，并用正则安全提取最大文献序号
    """
    separator = "==================== 参考文献 (References) ===================="
    max_index = 0
    ref_text = ""
    report_body = raw_text
    
    if separator in raw_text:
        parts = raw_text.split(separator)
        report_body = parts[0].strip()
        ref_text = parts[1].strip()
        
        # 匹配 "[数字] PMID:" 或 "[数字] DocID:" 或 "[数字] URL:"
        matches = re.findall(r'\[(\d+)\]\s*(?:PMID|DocID|URL)', ref_text)
        if matches:
            max_index = max([int(m) for m in matches])
            
    return report_body, max_index, ref_text, separator


def _split_sections(raw_text: str) -> tuple[str | None, str | None]:
    """Split bundled test input into patient section and MDT report section."""
    if MDT_DELIMITER in raw_text:
        parts = raw_text.split(MDT_DELIMITER, 1)
        patient_text = parts[0].strip()
        mdt_text = MDT_DELIMITER + "\n" + parts[1].lstrip()
        return patient_text, mdt_text
    return None, None


def _repair_json(text: str) -> str:
    """
    Repair common JSON issues from LLM output before parsing.
    Handles: control chars, trailing commas, truncated braces/brackets.
    """
    # Strip control characters (except newlines and tabs in strings)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Fix trailing commas before } and ]
    text = re.sub(r',\s*}', '\n}', text)
    text = re.sub(r',\s*\]', '\n]', text)
    # Close unclosed braces/brackets (truncation recovery)
    diff = text.count('{') - text.count('}')
    if diff > 0:
        text = text.rstrip() + '\n' + '}' * diff
    diff = text.count('[') - text.count(']')
    if diff > 0:
        text = text.rstrip() + '\n' + ']' * diff
    return text


def _safe_json_loads(text: str) -> dict | None:
    """
    Try parsing JSON with multiple strategies:
      1. Direct json.loads
      2. After repair (control chars, trailing commas, truncation)
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    # Strategy 2: after repair
    try:
        return json.loads(_repair_json(text), strict=False)
    except json.JSONDecodeError:
        pass

    return None


def _extract_fields_fallback(text: str) -> dict:
    """
    Regex-based fallback: extract individual JSON fields when full parse fails.
    """
    result = {
        "oncology_core": {},
        "comorbidities": {"critical_infections": [], "major_comorbidities": [], "incidental_findings": []},
        "proposed_plan": {},
        "clinical_questions_for_ebm": [],
    }

    # -- oncology_core --
    m = re.search(r'"basic_info"\s*:\s*"([^"]+)"', text)
    if m:
        result["oncology_core"]["basic_info"] = m.group(1)
    m = re.search(r'"diagnosis_and_stage"\s*:\s*"([^"]+)"', text)
    if m:
        result["oncology_core"]["diagnosis_and_stage"] = m.group(1)
    m = re.search(r'"pathology_and_molecular"\s*:\s*"([^"]+)"', text)
    if m:
        result["oncology_core"]["pathology_and_molecular"] = m.group(1)
    m = re.search(r'"surgery_type"\s*:\s*"([^"]+)"', text)
    if m:
        result["oncology_core"]["surgery_type"] = m.group(1)

    # -- comorbidities: try to parse each array --
    m = re.search(r'"critical_infections"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_text = "[" + m.group(1).strip() + "]"
        try:
            result["comorbidities"]["critical_infections"] = json.loads(arr_text)
        except json.JSONDecodeError:
            result["comorbidities"]["critical_infections"] = [arr_text[:2000]]
    m = re.search(r'"major_comorbidities"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_text = "[" + m.group(1).strip() + "]"
        try:
            result["comorbidities"]["major_comorbidities"] = json.loads(arr_text)
        except json.JSONDecodeError:
            result["comorbidities"]["major_comorbidities"] = [arr_text[:2000]]
    m = re.search(r'"incidental_findings"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_text = "[" + m.group(1).strip() + "]"
        try:
            result["comorbidities"]["incidental_findings"] = json.loads(arr_text)
        except json.JSONDecodeError:
            result["comorbidities"]["incidental_findings"] = [arr_text[:2000]]

    # -- proposed_plan --
    m = re.search(r'"main_oncology_treatment"\s*:\s*"([^"]+)"', text)
    if m:
        result["proposed_plan"]["main_oncology_treatment"] = m.group(1)
    m = re.search(r'"follow_up_schedule"\s*:\s*"([^"]+)"', text)
    if m:
        result["proposed_plan"]["follow_up_schedule"] = m.group(1)

    # -- clinical_questions_for_ebm: extract first question at minimum --
    m = re.search(r'"clinical_questions_for_ebm"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if m:
        arr_text = "[" + m.group(1).strip() + "]"
        try:
            result["clinical_questions_for_ebm"] = json.loads(arr_text)
        except json.JSONDecodeError:
            # Extract first question via regex if array parse fails
            q1 = re.search(r'"([^"]+)"', m.group(1))
            if q1:
                result["clinical_questions_for_ebm"] = [q1.group(1)]

    return result


def _classify_conditions_python(conditions: list) -> dict:
    """
    纯 Python 三级分流分类器——零 LLM 延迟。
    用关键词/正则将扁平合并症列表归入 critical_infections / major_comorbidities / incidental_findings。
    规则与下游 mdt_report_agent / followup_skill 的医学逻辑完全对齐。
    """

    # ---- 第一级：致命红线 keywords ----
    # 活动性感染、活动性炎症、活动性溃疡/出血风险
    _critical_patterns = [
        # 活动性修饰词 + 感染/炎症/溃疡
        r'(?<!非)活动性(?!\s*(?:携带|状态))',  # 活动性炎/感染/溃疡，但排除"非活动性""非活动期携带"
        r'现症',
        r'急性(?!愈|后)',          # 急性感染/炎症，但排除"急性...已愈"
        r'渗出性',
        r'伴发热',
        r'伴感染',
        r'(?<!非)活动期(?!前|间)',
        r'活动性出血',
        r'出血风险',
        # 具体病名
        r'HP\s*现症',             # HP 现症感染（幽门螺杆菌活动期）
        r'HP\s*阳性(?!史)',       # HP 阳性但不是"阳性史"
        r'幽门螺杆菌.*(?:阳性|感染|活动)',  # HP 活动期
        r'HP(?!V|v)\s*(?!.*(?:根除|已愈|已治|既往|阴性|正常|(-)))\S*',  # HP 相关（非已根治），匹配"HP现症感染""HP(+)""HP阳性"等；HP(?!V) 排除 HPV
        r'(?:胃|贲门|幽门|窦).*(?:活动性|急性|出血).*糜烂',  # 活动性/急性胃糜烂（化疗期血小板低谷出血风险）
        r'(?:活动性|急性|出血性)糜烂性胃炎',
        r'十二指肠溃疡',
        r'消化道出血',
        r'肺部炎性病灶',
        r'肺部感染',
        r'肺.*慢性炎(?!.*(?:已愈|陈旧))',  # 肺部慢性炎症（化疗前需排除隐匿性感染）
        r'两肺.*炎(?!.*(?:已愈|陈旧))',    # 两肺散在炎症
        r'支气管扩张伴感染',
        r'活动性结核',
        r'HBV-DNA\s*阳性',
        r'乙肝活动',
        r'活动性疱疹',
        r'牙源性感染',
        r'牙周脓肿',
        r'皮肤软组织感染',
        r'泌尿系感染',
        r'HPV(?!.*(?:阳性史|非活动|携带|已愈)).*(?:尖锐湿疣|VAIN|宫颈病变|阴道上皮内瘤变)',  # HPV 活动性器质性病变
        r'HPV.*感染(?!.*(?:阳性史|非活动|已愈))',  # HPV 活动性感染（排除非活动期携带）
    ]
    _critical_re = re.compile('|'.join(_critical_patterns))

    # ---- 第三级：静默噪音 keywords ----
    _incidental_patterns = [
        # 良性囊性/结节性
        r'(?:单纯性|良性|双肾|双肾多发|多发性)?\s*(?:肝|肾|卵巢|乳腺|甲状腺)?\s*(?:多发)?\s*囊肿',
        r'(?:肝|肾)(?:右叶|左叶)?\s*(?:多发)?\s*囊肿(?!.*(?:分隔|实性|复杂))',
        r'肝小血管瘤',
        r'血管瘤.*(?:<|≤|小)',
        r'良性.*结节',
        r'BI-RADS\s*[≤<=]\s*3',
        r'TI-RADS\s*[≤<=]\s*3',
        r'肺微?小?结节.*(?:<|≤)\s*6\s*mm',
        r'微?小?结节.*无.*高危',
        # 退行性/钙化性
        r'钙化(?!.*狭窄)',        # 钙化但无狭窄
        r'主动脉.*钙化',
        r'冠状动脉.*钙化',
        r'椎体.*退行性',
        r'骨赘',
        r'陈旧性.*腔隙性梗死',
        r'腔隙性梗死.*(?:陈旧|无症状)',
        # 消化道壁略厚/增厚（无梗阻/占位）——化疗前无需专科干预
        r'胆囊壁\s*(?:略|稍|轻度|增|毛糙|欠光滑)',
        r'十二指肠.*(?:增厚|略厚|稍厚|肠壁.*厚)',
        # 轻度生化异常
        r'轻度脂肪肝(?!.*(?:功能|显著|异常))',
        r'轻度贫血.*Hb\s*(?:[>＞]\s*)?9\d',
        r'血脂轻度升高',
        # 已愈/非活动期感染史
        r'HPV\s*阳性史(?!.*(?:活动|病变))',  # 非活动期 HPV 携带史
        r'HPV.*非活动',
        r'既往.*感染.*已愈',
        r'既往.*已治愈',
        r'陈旧性肺结核',
        r'HBV.*已愈',
        r'TB.*已愈',
        # 无需干预的小问题
        r'(?:双)?肾.*微?小?结石(?!.*(?:梗阻|积水|绞痛))',
        r'微?小?结石.*(?:下盏|肾盏)(?!.*(?:梗阻|积水|绞痛))',
        # 陈旧/创伤性骨骼发现（无急性干预指征）
        r'肋骨.*(?:皮质扭曲|皮质凹陷|陈旧|形态不规整)',
        r'陈旧性.*骨折',
    ]
    _incidental_re = re.compile('|'.join(_incidental_patterns))

    critical_infections = []
    major_comorbidities = []
    incidental_findings = []

    for cond in conditions:
        cond_str = str(cond).strip()
        if not cond_str:
            continue

        # HP/HPV 核心区分：
        # HP 现症感染（活动期）→ critical；HP 已根除 → 丢弃（不提取）
        # HPV 活动性病变 → critical；HPV 阳性史（非活动期）→ incidental
        hp_rooted = bool(re.search(r'HP.*(?:根除|已愈|已治|既往)', cond_str))
        if hp_rooted:
            continue  # HP 已根除，不提取

        # 检查 critical
        if _critical_re.search(cond_str):
            critical_infections.append(cond_str)
            continue

        # 检查 incidental
        if _incidental_re.search(cond_str):
            incidental_findings.append(cond_str)
            continue

        # 默认 → major_comorbidities
        major_comorbidities.append(cond_str)

    # 汇总日志中静默噪音的去重/合并提醒（不改变逻辑）
    if incidental_findings:
        logger.info(
            "Python 分类器：%d 条归入静默噪音（偶发发现合并声明）",
            len(incidental_findings),
        )

    return {
        "critical_infections": critical_infections,
        "major_comorbidities": major_comorbidities,
        "incidental_findings": incidental_findings,
    }


async def _llm_deduplicate_conditions(conditions: list, fast_llm) -> list:
    """
    LLM-based semantic deduplication: merge conditions the extraction LLM
    failed to merge.  Replaces regex-based _deduplicate_conditions which
    required manual rule maintenance and still missed semantically-equivalent
    descriptions (e.g. 胃镜 vs 病理对同一病灶的不同措辞).

    Only invoked when the list has ≥2 items — costs one cheap LLM call.
    """
    if len(conditions) <= 1:
        return conditions

    prompt = f"""\
你是一名医疗数据去重专家。以下是从患者病历中提取的合并症/异常列表。
同一病灶可能在不同检查报告中有不同措辞，请识别并合并。

【核心铁律】
🚨 禁止删除任何条目！每条输入必须出现在输出中——要么原样保留，要么合并后保留。
合并不是删除，是用更完整的表述替代多条。输出条目数 < 输入条目数是正常合并结果，
但所有输入条目的信息必须可追溯到输出中的某一条。

【去重规则】
- 同一器官 + 同一病灶类型 → 合并为一条，保留信息最完整的表述
- 同一病原体感染 + 其导致的器官病变 → 合并为一条（因果关联，不是两个独立疾病）
- 不同器官 或 不同病灶类型 → 保持独立，禁止合并
- 示例该合并的：
  · "主动脉管壁钙化" + "腹主动脉壁钙化" → 合并为更完整的一条
  · "室上性早搏" + "室性早搏" → 合并为更完整的一条
  · "慢性非萎缩性胃炎伴糜烂" + "幽门螺杆菌感染" → 合并为"慢性非萎缩性胃炎伴糜烂，HP(+)"
  · 胃镜"慢性非萎缩性胃炎伴糜烂" + 病理"(胃窦)浅表黏膜中度慢性炎" → 合并
- 示例不该合并的：
  · "肾囊肿" vs "肾结石" → 病灶类型不同，各自独立
  · "高血压" vs "糖尿病" → 不同疾病，各自独立
  · "二尖瓣少量反流" vs "主动脉瓣微量反流" vs "三尖瓣少量反流" → 不同瓣膜，各自独立

【待去重列表】
{json.dumps(conditions, ensure_ascii=False)}

仅输出合并后的 JSON 字符串数组，不要其他内容：
["条目1", "条目2", ...]
"""
    try:
        response = await invoke_with_timeout_and_retry(
            fast_llm, prompt, timeout=120.0, max_retries=2
        )
        raw = response.content
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            # raw_decode 只解析第一个完整 JSON 值，忽略后面的多余内容
            # (LLM 可能意外输出了两个数组或其他尾随文本)
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(cleaned)
        if isinstance(result, list) and len(result) > 0:
            _before = len(conditions)
            _after = len(result)
            if _after < _before:
                logger.info(
                    "[Dedup] LLM 合并: %d → %d 条 (移除 %d)",
                    _before, _after, _before - _after,
                )
            elif _after == _before:
                logger.info(f"[Dedup] LLM 复查确认无重复: {_before} 条")
            else:
                logger.info(f"[Dedup] LLM 返回 {_after} 条 (输入 {_before} 条)")
            return result
        else:
            logger.warning("[Dedup] LLM 返回空列表，回退原始列表")
    except Exception as e:
        logger.warning(f"[Dedup] LLM 去重失败，回退到原始列表: {e}")

    return conditions


async def _llm_critical_safety_net(unmatched: list, fast_llm) -> tuple:
    """
    双向 LLM 安全网：复查 major 桶中的条目。
    - 向上提升：漏网的致命红线（活动性感染/炎症/溃疡/出血）→ critical
    - 向下沉淀：伪重大合并症（良性偶发发现/已切除器官/退行性改变/解剖变异）→ incidental
    返回 (promoted, demoted) 两个列表。
    """
    if not unmatched:
        return [], []

    logger.info(f"[SafetyNet] {len(unmatched)} 条未被正则命中，LLM 双向复查中...")
    _st0 = time.time()
    prompt = f"""你是一名临床安全审核员。以下条目已被自动分类为"重大合并症"，请逐条审视并完成两项任务：

【任务1 — 向上提升（寻找漏网的致命红线）】
找出属于**活动性感染/活动性炎症/活动性溃疡/活动性出血/现症病原体阳性**的条目。
没有则返回空数组。

【任务2 — 向下沉淀（清理伪重大合并症）】
找出以下类型的条目，这些不属于需要专科长期随诊的"重大合并症"：
- 良性退行性改变：骨质增生、退行性变、椎间盘膨出等
- 影像学偶发发现：肠壁略厚/增厚、血管钙化/斑块（非狭窄性）、轻度脂肪肝、微小结石/囊肿
- 解剖学变异或术后正常状态：已切除器官的描述、术后改变、无临床意义的解剖变异
- 陈旧性/已愈改变：陈旧性骨折/梗死、已愈感染
- 无病变的器官描述：如"XX未见病变""XX正常""XX无明显异常"
- 轻度/微量生化异常无需干预者
没有则返回空数组。

【核心判断标准】：
- 真正保留在 major 里的，只能是高血压、糖尿病、冠心病、慢性肾病、慢性肝病、慢性呼吸系统疾病等需要长期服药/监测的实质性系统疾病
- 凡是不需要专科定期随诊、不影响肿瘤治疗方案选择的，都应归入"向下沉淀"

请严格输出 JSON 对象（不要其他内容）：
{{"promote_to_critical": ["条目A"], "demote_to_incidental": ["条目C", "条目D"]}}

【待审核条目】：
{json.dumps(unmatched, ensure_ascii=False)}
"""
    try:
        response = await invoke_with_timeout_and_retry(
            fast_llm, prompt, timeout=300.0, max_retries=3
        )
        raw = response.content
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start:end+1]
        result = json.loads(cleaned)
        if isinstance(result, dict):
            promoted = result.get("promote_to_critical", [])
            demoted = result.get("demote_to_incidental", [])
            if not isinstance(promoted, list):
                promoted = []
            if not isinstance(demoted, list):
                demoted = []
            _st1 = time.time()
            logger.info(f"[SafetyNet] 双向复查完成 | 耗时 {_st1-_st0:.1f}s | "
                        f"提升 {len(promoted)} 条 → critical, 沉淀 {len(demoted)} 条 → incidental")
            return promoted, demoted
    except Exception as e:
        logger.warning(f"[SafetyNet] LLM 调用失败，保留原分类: {e}")

    return [], []


def _truncate_for_comorbidity(text: str) -> str:
    """
    Truncate surgical-pathology noise from the tail of the text.

    Everything from the first surgical / post-operative pathology section in the
    *latter half* onward has near-zero comorbidity extraction value (it's all
    tumour detail, IHC, gross pathology).  Cutting it shortens the context
    window so the LLM can focus on the history, chronic diseases and imaging
    reports.

    We deliberately only search the latter half so a mention of prior surgery
    in the HPI doesn't trigger truncation.
    """
    _SURGICAL_MARKERS = [
        '术后病理',
        '手术记录',
        '手术方式',
        '术中所见',
    ]
    # Only consider markers in the latter half of the text
    search_from = len(text) // 2
    earliest = len(text)
    for marker in _SURGICAL_MARKERS:
        pos = text.find(marker, search_from)
        if pos != -1 and pos < earliest:
            earliest = pos

    if earliest < len(text):
        result = text[:earliest].strip()
        # ── 术后诊断 section sits after surgery/pathology and gets truncated,
        # but it often carries key comorbidities (HTN, DM, etc.).  Rescue it.
        tail = text[earliest:]
        diag_m = re.search(
            r'(?:^|\n)(#{1,3}\s*术后诊断[：:].*?)(?=\n#{1,3}\s|\n---|\Z)',
            tail, re.DOTALL,
        )
        if diag_m:
            result = result + "\n\n" + diag_m.group(1).strip()
        return result

    return text




def _regex_pre_screen(text: str) -> list[str]:
    """
    Extract numbered items from 检查结论/诊断意见 sections of imaging reports,
    plus 术前诊断/术后诊断 lists (which often carry key comorbidities like
    HTN, DM, obesity that the LLM might otherwise overlook).

    These are purely mechanical extractions — used as attention primers for the
    LLM, never as final output.  The LLM still verifies every item against the
    original text.
    """
    _NOISE_RE = re.compile(
        r'癌|肿瘤|占位|转移|浸润|LVSI|FIGO|Grade|IHC|免疫组化|p53|MMR|'
        r'Ki-67|ER|PR|未见异常|未见明显|大小正常|形态正常|无明显异常|'
        r'无异常|通畅|尚可'
    )
    # Strip trailing "请结合临床XXX" suffixes — they're radiologist recommendations,
    # not reasons to exclude a finding.
    _CLINICAL_SUFFIX_RE = re.compile(r'[，,。]\s*请结合临床[^。；;]*')
    candidates = []

    # ── Source 1: imaging report conclusions ──
    for match in re.finditer(
        r'(?:检查结论|诊断意见)\s+(.+?)(?=\n\s*(?:【|\n)|\Z)',
        text, re.DOTALL,
    ):
        body = match.group(1)
        for item_m in re.finditer(r'(?:^|[\s。；])\d+[\.、\)]\s*([^。；\n]+)', body):
            raw = item_m.group(1).strip()
            if len(raw) >= 3 and not _NOISE_RE.search(raw):
                raw = _CLINICAL_SUFFIX_RE.sub('', raw).strip()
                if raw:
                    candidates.append(raw)

    # ── Source 2: 术前诊断 / 术后诊断 lists ──
    # Format: "1、子宫内膜恶性肿瘤:..., 2、高血压, 3、糖尿病..."
    # Items are separated by Chinese commas (，) or semicolons.
    for match in re.finditer(
        r'(?:^|\n)#{1,3}\s*(?:术前诊断|术后诊断)[：:]\s*(.+?)(?=\n#{1,3}\s|\n---|\Z)',
        text, re.DOTALL,
    ):
        body = match.group(1)
        for item_m in re.finditer(
            r'(?:^|[，,;\s])\d+[、.\)）]\s*([^，,;\n]+(?:可能)?)',
            body,
        ):
            raw = item_m.group(1).strip()
            if len(raw) >= 2 and not _NOISE_RE.search(raw):
                candidates.append(raw)
    # Dedup preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


_COMORBIDITY_PROMPT = """\
你是一名医疗数据结构化专家。从以下患者病情中**完整、不遗漏**地提取所有非肿瘤性合并症/异常。

=========================================
{pre_screen_hints}【患者病情】
{chunk_text}
=========================================

═══════════════════════════════════════════
🔄 **【提取流程——必须逐步执行，跳过任一步骤视为不合格】**
═══════════════════════════════════════════

**第 1 步：速览摘要——抓取明确列出的慢性病**
先定位病历摘要/患者基本信息部分。摘要中常以"合并XX、XX""既往XX病史"
等形式列出主要合并症。将这些**明确列出的慢性病逐条记入清单**——这些是绝对不能遗漏的。
⚠️ 慢性病关键词提示（帮助识别，但必须有原文诊断依据）：高血压、糖尿病、冠心病/冠状动脉粥
样硬化性心脏病、脑梗/卒中史、肝炎/肝硬化、慢性肾病、COPD、骨质疏松等。

**第 2 步：逐报告扫描——每份检查报告的诊断意见都必须过目**
按顺序逐份扫读以下报告的**末端诊断意见/结论**部分，提取其中的阳性非肿瘤发现：
  a) **影像报告**（CT/MRI/超声/X线/CTA/DSA）：提取诊断意见中的非肿瘤异常——
     包括但不限于：炎症/浸润/实变、结节（注明大小）、囊肿、结石、钙化/斑块、
     脂肪肝、动脉硬化/狭窄、血管斑块、反流（瓣膜）、胸膜增厚、肠壁增厚等。
     🛑 影像诊断意见通常以"检查结论""诊断意见：""结论："开头——这些段落必须逐字阅读。
  b) **内镜报告**（胃镜/肠镜/支气管镜）：提取内镜诊断中的糜烂、溃疡、息肉、炎症等。
  c) **实验室/病原学检查**：仅提取有明确异常判断的指标。
  d) **既往史/个人史/手术史**：提取明确诊断的慢性病和既往非肿瘤手术史。

**第 3 步：去重合并** → 见下方去重规则。

═══════════════════════════════════════════
✅ **【应该提取】**（仅限原文明确写了的内容，但以下所有类型都必须覆盖）：
- 慢性病：高血压、糖尿病、冠心病、脑梗、肝炎、慢性肾病等（需原文明确诊断）
- 急/慢性感染、病原体携带：HP感染（幽门螺杆菌）、HPV感染、HBV等（需原文明确写出）
- 影像学非肿瘤异常：肺结节/炎症/浸润、囊肿、结石、钙化、脂肪肝、血管斑块/狭窄、
  动脉硬化、瓣膜反流、胸膜增厚、肠壁增厚等（需阅片诊断意见中写明的）
- 内镜非肿瘤异常：糜烂、溃疡、息肉等（需内镜诊断中写明的）
- 既往手术史（非本次肿瘤手术）、过敏史

【提取要求】：逐条列出，保留关键修饰词（"活动性""轻度""陈旧性""多发"等），
HP和HPV按原文如实记录，禁止混淆（HP=幽门螺杆菌，HPV=人乳头瘤病毒）。

═══════════════════════════════════════════
🚨 **【排除规则——以下禁止出现在输出中】**：
a) 肿瘤本身（癌症诊断/FIGO分期/组织学/Grade/浸润/LVSI等）
b) IHC/分子/免疫组化（p53/MMR/ER/PR/Ki-67/POLE/MSI等所有分子检测结果）
c) 正常/阴性检查所见（"未见异常""无腹水""大小正常""外观正常""未见转移"等）
d) 肿瘤原发灶影像描述（"宫腔占位""内膜不均""肌层浸润信号"）

🚨 **【禁止推断】**：
- 原文写"否认XX病史"→ 不得提取"XX功能异常"——否认意味着没有
- 原文只描述体征/症状未给诊断结论 → 不得自行归纳为诊断名
  例："分泌物色黄，无异味"≠ 阴道炎
- 实验室异常须有明确异常判断方可提取

仅输出 JSON，不要其他内容：
{{
  "all_conditions": ["高血压1级", "2型糖尿病", "HBV携带", "..."]
}}
"""


async def _extract_comorbidities(patient_text: str, fast_llm) -> list:
    """
    Stream A: Extract non-tumor comorbidities from patient section only.

    Pipeline:
      1. Truncate surgical / post-op pathology noise from the tail.
      2. Single-pass LLM extraction — no splitting, no merge problems.
    """
    _t0 = time.time()
    _input_chars = len(patient_text)
    logger.info(f"[Comorbidity] 开始提取 | 输入 {_input_chars} 字符")

    # ── Truncate surgical noise ──
    text = _truncate_for_comorbidity(patient_text)
    if len(text) < _input_chars:
        logger.info(f"[Comorbidity] 截断手术/病理尾部 | "
                    f"{_input_chars} → {len(text)} 字符 "
                    f"(移除 {_input_chars - len(text)} 字符)")

    # ── Pre-screen: mechanically extract imaging conclusion items as LLM hints ──
    pre_screened = _regex_pre_screen(text)
    if pre_screened:
        items = "\n".join(f"- {item}" for item in pre_screened)
        hints = (
            "【预筛提示——以下项目已从影像报告结论中机械提取，请逐条核实原文后纳入最终输出，"
            "同时补充预筛遗漏的阳性发现】\n"
            f"{items}\n\n"
        )
        logger.info(f"[Comorbidity] 预筛 {len(pre_screened)} 条影像发现作为 LLM 提示")
    else:
        hints = ""

    prompt = _COMORBIDITY_PROMPT.format(chunk_text=text, pre_screen_hints=hints)
    conditions = []
    try:
        response = await invoke_with_timeout_and_retry(
            fast_llm, prompt, timeout=300.0, max_retries=3
        )
        raw_resp = response.content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL | re.IGNORECASE).strip()
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned = cleaned[start_idx:end_idx + 1]
        result = _safe_json_loads(cleaned) or {}
        result_list = result.get("all_conditions", [])
        if isinstance(result_list, list):
            conditions = result_list
            logger.info(f"[Comorbidity] 提取完成 | {len(conditions)} 条")
    except Exception as e:
        logger.error(f"[Comorbidity] LLM 提取失败: {e}")

    _t1 = time.time()
    logger.info(f"[Comorbidity] 耗时 {_t1 - _t0:.1f}s | {len(conditions)} 条")
    return conditions


async def _extract_oncology(mdt_text: str, fast_llm) -> dict:
    """
    Stream B: Extract oncology core data from MDT report section only.
    No comorbidity extraction — that's Stream A's job.
    """
    _t0 = time.time()
    _input_chars = len(mdt_text)
    logger.info(f"[Oncology] 开始提取 | 输入 {_input_chars} 字符")

    prompt = f"""
你是一名医疗数据结构化专家。从以下 MDT 会诊报告中提取肿瘤专科数据。

=========================================
【MDT 会诊报告】
{mdt_text}
=========================================

提取以下字段：

1. **oncology_core** — 肿瘤核心数据：
   - basic_info：年龄、绝经状态、体能评分
   - diagnosis_and_stage：术后诊断及FIGO分期
   - pathology_and_molecular：病理类型、浸润深度、淋巴结、LVSI、MMR、p53、分子分型状态等。
     🚨 **分子分型强制格式**：必须区分"IHC 已见"与"NGS 是否回报"两个层面：
       * 若原文写"分子分型检测已送检，结果待回报"或类似措辞 → 必须写"IHC示p53突变型模式/pMMR，NGS分子分型待回报"，禁止只写"p53突变型"
       * 若原文写"结合免疫组化为伴p53突变"且未提NGS结果 → 必须写"IHC p53突变型模式，NGS分子分型待确认"
       * 仅当原文明确写了NGS已出结果（如"NGS提示p53突变"、"分子分型为p53abn"且无"待回报"修饰）→ 方可写"分子分型：p53abn（NGS已确认）"
     * 禁止将 IHC p53 异常等同于 p53abn 分子分型——这是 TCGA 多重分类赋予原则的基本要求
     * 禁止将IHC代理指标直接当作最终分子分型
   - surgery_type：已接受的具体手术方式名称

2. **proposed_plan**：
   - main_oncology_treatment：最终肿瘤治疗方案。若草稿中有因"严重合并症/心血管高危"导致的方案降级，只提取降级后的方案
   - follow_up_schedule：随访计划

3. **clinical_questions_for_ebm**：提取 PICO 问题，保留原文中的临床试验代号

仅输出 JSON，不要其他内容：
{{
  "oncology_core": {{
    "basic_info": "...",
    "diagnosis_and_stage": "...",
    "pathology_and_molecular": "...",
    "surgery_type": "..."
  }},
  "proposed_plan": {{
    "main_oncology_treatment": "...",
    "follow_up_schedule": "..."
  }},
  "clinical_questions_for_ebm": []
}}
"""
    try:
        response = await invoke_with_timeout_and_retry(
            fast_llm, prompt, timeout=300.0, max_retries=3
        )
        raw_resp = response.content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL | re.IGNORECASE).strip()
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        result = _safe_json_loads(cleaned) or {}
        _t1 = time.time()
        logger.info(f"[Oncology] 提取完成 | 耗时 {_t1-_t0:.1f}s | keys: {list(result.keys())}")
        return result
    except Exception as e:
        logger.error(f"[Oncology] LLM 提取失败: {e}")

    return {}


async def _extract_unified_fallback(raw_text: str, fast_llm) -> dict:
    """
    Fallback: single-LLM unified extraction when section split fails.
    Preserves the original extraction logic for backward compatibility.
    """
    _t0 = time.time()
    _input_chars = len(raw_text)
    logger.info(f"[Parser-Fallback] 回退到统一单轮提取 | 输入 {_input_chars} 字符")

    # Truncate input to prevent LLM timeout on very long records.
    _MAX_INPUT = 32000
    _text_for_llm = raw_text
    if _input_chars > _MAX_INPUT:
        _head = raw_text[:_MAX_INPUT * 2 // 3]
        _tail = raw_text[-_MAX_INPUT // 3:]
        _text_for_llm = _head + "\n...(中间段落省略)...\n" + _tail
        logger.info(f"[Parser-Fallback] 输入截断: {_input_chars} → {len(_text_for_llm)} 字符")

    prompt = f"""
你是一名医疗数据结构化专家。从以下病历中无损提取所有关键信息。

=========================================
【病历数据】
{_text_for_llm}
=========================================

提取以下字段：

1. **oncology_core** — 肿瘤核心数据：
   - basic_info：年龄、绝经状态、体能评分
   - diagnosis_and_stage：术后诊断及FIGO分期
   - pathology_and_molecular：病理类型、浸润深度、淋巴结、LVSI、MMR、p53、分子分型状态等。
     🚨 **分子分型强制格式**：必须区分"IHC 已见"与"NGS 是否回报"两个层面：
       * 若原文写"分子分型检测已送检，结果待回报"或类似措辞 → 必须写"IHC示p53突变型模式/pMMR，NGS分子分型待回报"，禁止只写"p53突变型"
       * 若原文写"结合免疫组化为伴p53突变"且未提NGS结果 → 必须写"IHC p53突变型模式，NGS分子分型待确认"，禁止将IHC代理指标直接当作最终分子分型
       * 仅当原文明确写了NGS已出结果（如"NGS提示p53突变"、"分子分型为p53abn"且无"待回报"修饰）→ 方可写"分子分型：p53abn（NGS已确认）"
     * 禁止将 IHC p53 异常等同于 p53abn 分子分型——这是 TCGA 多重分类赋予原则的基本要求
   - surgery_type：已接受的具体手术方式名称

2. **all_conditions** — 提取患者的**非肿瘤性合并症/合并异常**。🚨 以下内容已归入 oncology_core，绝对禁止重复出现在此数组中。

   【🚨 排除规则——以下四类绝对禁止出现在 all_conditions】：
   a) **肿瘤本身**：癌症诊断名、FIGO分期、组织学亚型、Grade/分级、浸润深度、淋巴结转移状态、LVSI——已归入 oncology_core
   b) **IHC/分子/免疫组化**：p53、MMR(MLH1/MSH2/MSH6/PMS2)、ER/PR、Ki-67、PTEN、β-catenin、L1CAM、POLE、MSI 等所有免疫组化和分子检测结果——已归入 oncology_core.pathology_and_molecular
   c) **正常/阴性检查所见**："未见异常""未见肿大淋巴结""无腹水""无积液""大小正常""外观正常""探查未见异常""未见恶性细胞""未见癌转移""未见脉管内癌栓"等——无临床管理需求，不提取
   d) **肿瘤原发灶影像描述**："宫腔占位""宫腔肿块""内膜不均""肌层浸润信号"——属于诊断范畴，已归入 oncology_core

   【🚨 最高原则——只提取原文明确写了的内容，严禁任何形式的推断或补全】：
   - 原文没有明确写出的疾病/异常，一律不得提取。禁止根据"常见合并症""通常会有"推测。
   - 原文只描述了体征/症状但未给诊断结论的，不得自行归纳为诊断名。
     例：原文写"分泌物色黄，无异味"，不得提取"阴道炎"——因为原文未下此诊断。
   - 原文"否认XX病史"的，不得提取"XX功能异常"——否认意味着没有。
   - 实验室异常必须有明确的异常值/诊断陈述方可提取，禁止仅因"可能有"就列出。

   【✅ 应该提取——非肿瘤性异常/疾病/合并症（仅限原文明确写了的内容）】：
   - 慢性病：高血压、糖尿病、冠心病等（需原文明确诊断或描述）
   - 急/慢性感染、病原体携带：HP感染、HPV感染、HBV等（需原文明确写出的）
   - 影像学非肿瘤异常：肺结节、炎症/浸润、囊肿、结石、钙化、脂肪肝等（需阅片结论中写明的）
   - 内镜非肿瘤异常：糜烂、溃疡、息肉等（需内镜诊断或病理诊断中写明的）
   - 既往手术史(非本次肿瘤手术)、过敏史

   【提取要求】：逐条列出，保留关键修饰词（"活动性""轻度""陈旧性""多发"等），HP和HPV按原文如实记录禁止混淆

   【去重规则——同一病灶不同描述合并为一条】：
   - 如果同一器官/部位的同一病灶在不同检查报告中有多种表述（如胃镜所见 vs 病理诊断、影像 vs 内镜），必须合并为一条，使用信息最完整的表述，禁止拆成多条
   - 示例：胃镜"慢性非萎缩性胃炎伴糜烂"和病理"(胃窦)浅表黏膜中度慢性炎伴糜烂"→合并为"慢性非萎缩性胃炎伴糜烂，HP(+)"
   - 去重原则：同一器官的炎症/糜烂/溃疡/感染不论多少种措辞，就是一条

3. **proposed_plan**：
   - main_oncology_treatment：最终肿瘤治疗方案
     若草稿中有因"严重合并症/心血管高危"导致的方案降级，只提取降级后的方案
   - follow_up_schedule：随访计划
   - comorbidity_management_list：逐条提取"患者[合并症]，建议[科室]随诊"格式的建议，无则[]

4. **clinical_questions_for_ebm**：提取 PICO 问题，保留原文中的临床试验代号

仅输出 JSON，不要其他内容：
{{
  "oncology_core": {{
    "basic_info": "...",
    "diagnosis_and_stage": "...",
    "pathology_and_molecular": "...",
    "surgery_type": "..."
  }},
  "all_conditions": [
    "高血压2级",
    "右肺上叶炎性浸润",
    "乙型肝炎表面抗原阳性",
    "胃窦黏膜慢性炎伴糜烂",
    "轻度贫血（Hb 105g/L）",
    "... 仅列出原文明确写了的非肿瘤性合并症/异常，禁止推测或补全，禁止包含 oncology_core 已有内容"
  ],
  "proposed_plan": {{
    "main_oncology_treatment": "...",
    "follow_up_schedule": "...",
    "comorbidity_management_list": []
  }},
  "clinical_questions_for_ebm": []
}}
"""
    flat = {}
    try:
        _prompt_len = len(prompt)
        logger.info(f"[Parser] Prompt 就绪 ({_prompt_len} 字符), 开始 LLM 调用 (timeout=300s, retries=3)...")
        _t1 = time.time()
        response = await invoke_with_timeout_and_retry(
            fast_llm, prompt, timeout=300.0, max_retries=3
        )
        _t2 = time.time()
        _resp_len = len(response.content) if response and hasattr(response, 'content') else 0
        logger.info(f"[Parser] LLM 响应完成 | 耗时 {_t2-_t1:.1f}s | 响应 {_resp_len} 字符")
        raw_resp = response.content
        cleaned = re.sub(r"<think>.*?</think>", "", raw_resp, flags=re.DOTALL | re.IGNORECASE).strip()
        start_idx = cleaned.find('{')
        end_idx = cleaned.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
        flat = _safe_json_loads(cleaned) or {}
        _t3 = time.time()
        logger.info(f"[Parser] JSON 解析完成 | 耗时 {_t3-_t2:.1f}s | 提取到 {len(flat)} 个顶层键")
        if flat.get("oncology_core"):
            logger.info(f"[Parser] oncology_core: {json.dumps(flat['oncology_core'], ensure_ascii=False)}")
        if flat.get("all_conditions"):
            _ac = flat["all_conditions"]
            logger.info(f"[Parser] all_conditions ({len(_ac)} 条): {json.dumps(_ac, ensure_ascii=False)}")
    except Exception as e:
        logger.error(f"[Parser] LLM 提取失败: {e}")

    if not flat or not flat.get("all_conditions"):
        logger.warning("[Parser] 提取结果为空，使用降级输出。")
        logger.warning(f"[Parser] flat keys: {list(flat.keys()) if flat else 'None'}")
        return {
            "oncology_core": flat.get("oncology_core", {"raw": "Extraction failed"}),
            "comorbidities": {"critical_infections": [], "major_comorbidities": [], "incidental_findings": []},
            "proposed_plan": flat.get("proposed_plan", {}),
            "clinical_questions_for_ebm": flat.get("clinical_questions_for_ebm", []),
        }

    all_conditions = flat.get("all_conditions", [])
    _t4 = time.time()
    logger.info(f"[Parser] LLM 提取到 {len(all_conditions)} 条 conditions | 总耗时至此 {_t4-_t0:.1f}s")

    _before_dedup = len(all_conditions)
    all_conditions = await _llm_deduplicate_conditions(all_conditions, fast_llm)
    if len(all_conditions) < _before_dedup:
        logger.info(f"[Parser] Dedup 合并: {_before_dedup} → {len(all_conditions)} 条")

    classified = _classify_conditions_python(all_conditions)
    total = sum(len(v) for v in classified.values())
    _t5 = time.time()
    logger.info(f"[Parser] Python 分类完成 | {_t5-_t4:.2f}s | {total} total "
                f"(critical={len(classified['critical_infections'])}, "
                f"major={len(classified['major_comorbidities'])}, "
                f"incidental={len(classified['incidental_findings'])})")
    if classified["critical_infections"]:
        logger.info(f"[Parser] critical_infections ({len(classified['critical_infections'])} 条): "
                    f"{json.dumps(classified['critical_infections'], ensure_ascii=False)}")
    if classified["major_comorbidities"]:
        logger.info(f"[Parser] major_comorbidities ({len(classified['major_comorbidities'])} 条): "
                    f"{json.dumps(classified['major_comorbidities'], ensure_ascii=False)}")
    if classified["incidental_findings"]:
        logger.info(f"[Parser] incidental_findings ({len(classified['incidental_findings'])} 条): "
                    f"{json.dumps(classified['incidental_findings'], ensure_ascii=False)}")

    if classified["major_comorbidities"]:
        logger.info(f"[Parser] 启动 SafetyNet 双向复查 {len(classified['major_comorbidities'])} 条 major 条目...")
        _t5a = time.time()
        promoted, demoted = await _llm_critical_safety_net(classified["major_comorbidities"], fast_llm)
        _t5b = time.time()
        logger.info(f"[Parser] SafetyNet 完成 | 耗时 {_t5b-_t5a:.1f}s | "
                    f"提升 {len(promoted)} 条 → critical, 沉淀 {len(demoted)} 条 → incidental")
        if promoted:
            for item in promoted:
                if item in classified["major_comorbidities"]:
                    classified["major_comorbidities"].remove(item)
                    classified["critical_infections"].append(item)
        if demoted:
            for item in demoted:
                if item in classified["major_comorbidities"]:
                    classified["major_comorbidities"].remove(item)
                    classified["incidental_findings"].append(item)
            logger.info(f"[Parser] 沉淀详情: {json.dumps(demoted, ensure_ascii=False)}")
        logger.info(f"[Parser] SafetyNet 后最终分类: "
                    f"critical={len(classified['critical_infections'])}, "
                    f"major={len(classified['major_comorbidities'])}, "
                    f"incidental={len(classified['incidental_findings'])}")

    result = {
        "oncology_core": flat.get("oncology_core", {}),
        "comorbidities": classified,
        "proposed_plan": flat.get("proposed_plan", {}),
        "clinical_questions_for_ebm": flat.get("clinical_questions_for_ebm", []),
    }
    _t6 = time.time()
    logger.info(f"[Parser] 结构化提取完成 | 总耗时 {_t6-_t0:.1f}s | "
                f"final: crit={len(classified['critical_infections'])}, "
                f"major={len(classified['major_comorbidities'])}, "
                f"incid={len(classified['incidental_findings'])}")
    return result


async def extract_structured_task(
    raw_text: str = "",
    fast_llm=None,
    patient_text: str = None,
    mdt_text: str = None,
) -> dict:
    """
    Dual-stream parallel extraction orchestrator.

    When possible, splits input into patient section and MDT report section,
    then runs comorbidity extraction (Stream A) and oncology extraction (Stream B)
    in parallel for reduced context length and improved task focus.

    In production: pass `patient_text` and `mdt_text` directly to skip splitting.
    In testing: pass `raw_text` and the splitter handles bundling.

    Falls back to unified single-LLM extraction when splitting is not possible.
    """
    _t0 = time.time()

    # Resolve input: production can pass split texts directly
    if patient_text is None and mdt_text is None and raw_text:
        patient_text, mdt_text = _split_sections(raw_text)

    if patient_text and mdt_text:
        logger.info("[Parser] 启动双流并行提取（合并症 | 肿瘤专科）...")
        _t_parallel_start = time.time()

        alt_results = await gather_safe(
            _extract_comorbidities(patient_text, fast_llm),
            _extract_oncology(mdt_text, fast_llm),
        )
        all_conditions = alt_results[0].value if alt_results[0].success else {}
        oncology_result = alt_results[1].value if alt_results[1].success else {}
        _t_parallel_end = time.time()
        logger.info(f"[Parser] 双流并行完成 | 总耗时 {_t_parallel_end-_t_parallel_start:.1f}s")

        oncology_core = oncology_result.get("oncology_core", {})
        proposed_plan = oncology_result.get("proposed_plan", {})
        clinical_questions = oncology_result.get("clinical_questions_for_ebm", [])
    elif patient_text:
        # Only patient text available (no MDT report)
        logger.info("[Parser] 仅有患者病情文本，仅提取合并症...")
        all_conditions = await _extract_comorbidities(patient_text, fast_llm)
        oncology_core = {}
        proposed_plan = {}
        clinical_questions = []
    else:
        # Cannot split — fall back to unified extraction
        logger.info("[Parser] 无法拆分章节，回退到统一单轮提取...")
        return await _extract_unified_fallback(raw_text, fast_llm)

    # === Post-processing (same pipeline regardless of extraction path) ===
    _t_post_start = time.time()

    all_conditions = await _llm_deduplicate_conditions(all_conditions, fast_llm)
    logger.info(f"[Parser] Dedup 后: {len(all_conditions)} 条 → "
                f"{json.dumps(all_conditions, ensure_ascii=False)}")

    classified = _classify_conditions_python(all_conditions)
    logger.info(f"[Parser] Python 分类完成 | "
                f"critical={len(classified['critical_infections'])}, "
                f"major={len(classified['major_comorbidities'])}, "
                f"incidental={len(classified['incidental_findings'])}")

    if classified["major_comorbidities"]:
        logger.info(f"[Parser] 启动 SafetyNet 双向复查 {len(classified['major_comorbidities'])} 条 major 条目...")
        promoted, demoted = await _llm_critical_safety_net(
            classified["major_comorbidities"], fast_llm
        )
        logger.info(f"[Parser] SafetyNet 完成 | 提升 {len(promoted)} → critical, 沉淀 {len(demoted)} → incidental")
        if promoted:
            for item in promoted:
                if item in classified["major_comorbidities"]:
                    classified["major_comorbidities"].remove(item)
                    classified["critical_infections"].append(item)
        if demoted:
            for item in demoted:
                if item in classified["major_comorbidities"]:
                    classified["major_comorbidities"].remove(item)
                    classified["incidental_findings"].append(item)

    _t_post_end = time.time()

    result = {
        "oncology_core": oncology_core,
        "comorbidities": classified,
        "proposed_plan": proposed_plan,
        "clinical_questions_for_ebm": clinical_questions,
    }
    _t6 = time.time()
    logger.info(f"[Parser] 结构化提取完成 | 总耗时 {_t6-_t0:.1f}s (后处理 {_t_post_end-_t_post_start:.1f}s) | "
                f"final: crit={len(classified['critical_infections'])}, "
                f"major={len(classified['major_comorbidities'])}, "
                f"incid={len(classified['incidental_findings'])}")
    return result


async def run_evidence_update(treatment_context: str, model_choice: str = "auto"):
    """
    执行核心循证更新。
    路由策略：重大合并症送入Deep Search检索毒性与预后，次要异常直接生成转诊话术。

    参数:
        model_choice: "local" | "deepseek" | "gpt" | "auto"
    """
    # ── 模型选择 ──
    if model_choice == "auto":
        model_choice = get_model_provider()

    if model_choice == "local":
        if check_local_model_health():
            print("🚀 使用本地 vLLM 模型 (Free & Private).")
            current_mode = "local"
            fast_llm = get_local_model(temperature=0.1)
        else:
            print("❌ 本地模型不可用。请检查 vLLM 是否已启动 (http://localhost:8000/v1)。")
            print("   提示：修改 _settings/.secrets.toml 中 [model] provider 为 \"deepseek\" 或 \"gpt\"。")
            return
    elif model_choice == "deepseek":
        print("🌐 使用 DeepSeek V4 Pro API.")
        current_mode = "deepseek"
        fast_llm = get_deepseek_v4()
    elif model_choice == "gpt":
        print("🌐 使用 OpenAI GPT-4.1 API.")
        current_mode = "gpt"
        fast_llm = get_gpt4_1_mini()
    else:
        print(f"❌ 未知模型提供方: {model_choice}")
        print("   请在 _settings/.secrets.toml 中设置 provider = \"local\" | \"deepseek\" | \"gpt\"")
        return

    # 1. 拆分图谱初步报告的主体与参考文献
    report_body, max_index, baseline_refs, separator = parse_graph_ec_report(treatment_context)
    print(f"✅ [Parser] Found {max_index} baseline references from graph-ec.")

    # 2. LLM 结构化拆解与合并症分级
    structured_task = await extract_structured_task(treatment_context, fast_llm)
    structured_task["baseline_references"] = {"max_index": max_index}
    
    # 🚨 核心深搜 Payload 组装：
    # 将 oncology_core 和 major_comorbidities 组装在一起交给 Deep Search
    # 丢弃 incidental_findings 以防干扰主检索链路
    search_payload = {
        "oncology_profile": structured_task.get("oncology_core", {}),
        "critical_infections": structured_task.get("comorbidities", {}).get("critical_infections", []),
        "major_comorbidities_affecting_treatment": structured_task.get("comorbidities", {}).get("major_comorbidities", []),
        "incidental_findings": structured_task.get("comorbidities", {}).get("incidental_findings", []),
        "preliminary_plan": structured_task.get("proposed_plan", {}),
        "specific_pico_questions": structured_task.get("clinical_questions_for_ebm", []),
        "surgery_type": structured_task.get("oncology_core", {}).get("surgery_type", ""),
        "baseline_references": {"max_index": max_index}
    }
    
    print(f"\n🔄 Clinical Evidence Update System Activated.")
    print(f"   Context Length: {len(report_body)} characters")
    print(f"   Major Comorbidities to Analyze: {len(search_payload['major_comorbidities_affecting_treatment'])}")
    print("   Targeting Sources: PubMed (2024+), ClinicalTrials.gov\n")

    # 严格限制加载的工具列表，防止 Token 爆炸导致超时
    my_target_tools = [
        "search_recent_pubmed",  
        "get_studies",           
        "get_adverse_reactions_by_drug_name", 
        "get_warnings_and_cautions_by_drug_name" 
    ]

    # 3. 初始化深搜系统
    system = AdvancedSearchSystem(
        max_iterations=settings.detailed.iteration, 
        questions_per_iteration=settings.detailed.questions_per_iteration,
        is_report=True,
        treatment_context=report_body, 
        structured_task=search_payload, # 👈 只喂给它肿瘤和重大合并症数据
        using_model=current_mode,
        chosen_tools=my_target_tools
    )

    try:
        from .concurrency.shutdown import get_shutdown_manager
        shutdown_mgr = get_shutdown_manager()

        if shutdown_mgr.shutdown_event.is_set():
            print("\n⚠️ 收到关闭信号，取消搜索。")
            return

        await system.initialize()

        if shutdown_mgr.shutdown_event.is_set():
            print("\n⚠️ 收到关闭信号，返回部分结果。")
            await system.cleanup()
            return

        query = "Please validate the preliminary treatment plan, carefully assess the impact of major comorbidities (if any) on drug toxicity and overall survival, and answer the specific clinical questions provided."
        results = await system.analyze_topic(query)
        print(f"\n✅ Evidence synthesis task completed.")
        
        if results.get("final_report"):
             print("\n" + "="*60)
             print("   FINAL COMBINED CLINICAL REPORT   ")
             print("="*60 + "\n")

             final_resp_text = results['final_report']

             # 移除 LLM 冗余礼貌性开场白
             final_resp_text = strip_llm_preamble(final_resp_text)
             # 第二人称 → 临床第三人称
             final_resp_text = depersonalize_report(final_resp_text)
             # 移除所有 Markdown 粗体标记 **
             final_resp_text = re.sub(r'\*\*', '', final_resp_text)

             # 4. 完美拼接与【次要异常的拦截转诊】
             new_evidence_text = final_resp_text
             new_refs_text = ""
             split_marker = "=================================================="
             
             if split_marker in final_resp_text:
                 parts = final_resp_text.split(split_marker)
                 new_evidence_text = parts[0].strip()
                 new_refs_text = parts[1].strip()
             
             # 将重写后的终极正文、次要并发症转诊、旧文献列表、新文献列表按序无缝缝合
             combined_report = f"### 🏥 循证校验与优化的最终治疗方案 (Deep EBM Synthesized Plan)\n\n" \
                               f"{new_evidence_text}\n" \
                               f"\n{separator}\n" \
                               f"{baseline_refs}\n"
             
             if new_refs_text:
                 combined_report += f"{new_refs_text}\n"
             
             print(combined_report)
             
             # 保存到文件
             report_path = "evidence_update_report.md"
             with open(report_path, "w", encoding="utf-8") as f:
                 f.write(combined_report)
             print(f"\n📄 Report saved to: {os.path.abspath(report_path)}")

    except asyncio.CancelledError:
        logger.warning("证据更新任务被取消")
        raise
    except Exception as e:
        logger.error(f"Run failed: {e}")
        print(f"\n❌ Error during execution: {e}")

async def main():
    """主程序入口"""
    # 安装优雅关闭信号处理器（shutdown manager 内置了 HTTP 连接池清理）
    try:
        from .concurrency.shutdown import install_signal_handlers, get_shutdown_manager
        shutdown_mgr = get_shutdown_manager()
        shutdown_mgr.set_main_task(asyncio.current_task())
        install_signal_handlers()
    except Exception as e:
        logger.debug(f"无法安装信号处理器: {e}")
        shutdown_mgr = None

    print("==================================================")
    print("   OriGene Clinical Evidence Validator (Auto-Hybrid)")
    print("==================================================")
    print("Strategy: Extract Context -> Deep Search -> Auto Gap-Closing Citations")
    print("Type 'quit' to exit at any time.")
    print("Press Ctrl+C to gracefully shutdown (partial results preserved).")

    # ── 模型选择 ──
    config_provider = get_model_provider()
    provider_labels = {"local": "本地 vLLM 模型", "deepseek": "DeepSeek V4 Pro API", "gpt": "OpenAI GPT-4.1 API", "auto": "自动检测"}
    print(f"\n📋 当前配置的模型提供方: {provider_labels.get(config_provider, config_provider)}")
    print("   可选模型:")
    print("   1) 本地 vLLM 模型 (Free & Private)")
    print("   2) DeepSeek V4 Pro API (外部 API)")
    print("   3) OpenAI GPT-4.1 API (外部 API)")
    print("   Enter = 使用配置文件默认值")

    model_choice = config_provider
    raw = (await ainput("\n选择模型 (1/2/3, Enter=默认): ")).strip()
    if raw == "1":
        model_choice = "local"
    elif raw == "2":
        model_choice = "deepseek"
    elif raw == "3":
        model_choice = "gpt"
    elif raw and raw not in ("1", "2", "3"):
        print(f"⚠️ 无效选择 '{raw}'，使用默认值。")

    try:
        while True:
            if shutdown_mgr and shutdown_mgr.shutdown_event.is_set():
                print("\n⚠️ 收到关闭信号，退出主循环。")
                break

            print("\n--------------------------------------------------")
            print("Select Input Method:")
            print("1) Paste Treatment Plan Text (Markdown with References)")
            print("2) Load Plan from File (.txt/.md)")

            choice = (await ainput("\nEnter number (1 or 2): ")).strip()

            if choice.lower() == 'quit':
                break

            treatment_context = ""

            if choice == "2":
                path = (await ainput("Enter file path: ")).strip()
                if path.lower() == 'quit': break
                treatment_context = read_context_from_file(path)
            elif choice == "1":
                print("\n👇 Please paste the Clinical Treatment Plan (Markdown) below.")
                print("Type 'END' on a new line when finished:\n")
                lines = []
                while True:
                    if shutdown_mgr and shutdown_mgr.shutdown_event.is_set():
                        break
                    line = await ainput()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                if shutdown_mgr and shutdown_mgr.shutdown_event.is_set():
                    break
                treatment_context = "\n".join(lines)
            else:
                print("Invalid selection. Please enter 1 or 2.")
                continue

            if not treatment_context.strip():
                print("❌ Empty context provided. Please try again.")
                continue

            # 执行核心任务
            await run_evidence_update(treatment_context, model_choice)
    except asyncio.CancelledError:
        logger.warning("主循环收到 CancelledError，正在退出...")
        raise
    finally:
        # 优雅关闭：通过 ShutdownManager 统一清理所有资源
        if shutdown_mgr:
            await shutdown_mgr.cleanup(grace_period=30.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except asyncio.CancelledError:
        print("\n\n程序已优雅关闭.")
