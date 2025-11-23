from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.postprocessing.substituent_text import abbr_parse


@dataclass
class RGroup(ABC):
    _smiles: str = field(default=None, init=True, repr=False)  # 隐藏字段的初始化和文本化

    @property
    @abstractmethod
    def smiles(self):
        pass


@dataclass
class RGroupAbbr(RGroup):
    abbr: str = ""  # 默认为 “” , 此时解析会返回 H

    @property
    def smiles(self):
        if not self._smiles:
            self._smiles = abbr_parse(self.abbr) or ""

        return self._smiles

    def __str__(self):  # 用于绘制
        return f"abbr: '{self.abbr}', smi: '{self.smiles}'"

    def to_dict(self):
        return {"abbr": self.abbr, "smiles": self.smiles}


@dataclass
class Compound:
    r_groups: dict[str: RGroup] = None
    origin_smi: str = field(default='', init=False, repr=False)  # 隐藏字段的初始化和文本化
    cmpd_id: str = '-'
    smiles: str = ''

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {
            "cmpd_id": self.cmpd_id,
            "smiles": self.smiles,
            "r_groups": {k: v.to_dict() for k, v in self.r_groups.items()}
        }

    @classmethod
    def from_dict(cls, data):
        data['r_groups'] = {k: {'_smiles' if k2 == 'smiles' else k2: v2 for k2, v2 in v.items()}
                            for k, v in data['r_groups'].items()}
        return cls(
            cmpd_id=data["cmpd_id"],
            smiles=data["smiles"],
            r_groups={k: RGroupAbbr(**v) for k, v in data["r_groups"].items()}
        )