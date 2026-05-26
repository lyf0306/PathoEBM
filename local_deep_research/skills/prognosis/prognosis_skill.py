"""
Prognosis Skill — 提供权威预后生存率参考数据。

用户只需更新 references/ 目录下的生存率数据表（SEER/NCDB 等），
PrognosisAgent 会将其作为不可辩驳的基线数据优先采用。

目录结构：
  skills/prognosis/
    ├── __init__.py
    ├── prognosis_skill.py    ← 本文件
    └── references/           ← 用户在此放生存率 .md 文件
        ├── SEER_子宫内膜癌分期生存率.md
        ├── NCDB_浆液性癌生存率.md
        └── ...
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class PrognosisSkill:
    """
    Skill: 提供权威预后生存率参考数据。
    用户只需更新 references/ 目录下的 .md 文件即可同步最新生存率数据。
    """

    def __init__(self, references_dir: str = None):
        if references_dir is None:
            references_dir = os.path.join(os.path.dirname(__file__), "references")
        self.references_dir = Path(references_dir)
        self.data = self._load_references()

    def _load_references(self) -> str:
        if not self.references_dir.exists():
            logger.warning(f"预后参考文献目录不存在: {self.references_dir}")
            return ""
        md_files = sorted(self.references_dir.glob("*.md"))
        if not md_files:
            logger.warning(f"预后目录下没有 .md 文件: {self.references_dir}")
            return ""

        contents = []
        for f in md_files:
            try:
                text = f.read_text(encoding="utf-8")
                if len(text) > 80000:
                    logger.info(f"{f.name} 较长，截取前 80000 字符")
                    text = text[:80000] + "\n\n...[已截断]..."
                contents.append(f"===== {f.stem} =====\n{text}")
                logger.info(f"已加载预后参考: {f.name} ({len(text)} 字符)")
            except Exception as e:
                logger.warning(f"加载预后文件失败 {f}: {e}")

        return "\n\n".join(contents)

    def has_data(self) -> bool:
        return bool(self.data.strip())

    def get_data(self) -> str:
        return self.data if self.data.strip() else ""
