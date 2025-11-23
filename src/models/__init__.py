from .mol_segment import MolSegmentor, RGroupSegmentor
from .mol_recognize import MolRecognizer
from .table_detect import TableDetector
from .table_recognize import TableRecognizer
from .text_recognize import TextDetector, TextRecognizer, MathTextRecognizer
from .common import ModelConfig, BaseModel
from src.settings import settings


class ModelLoader:
    """模型加载器，集中管理，懒加载"""

    def __init__(self, torch_device=settings.TORCH_DEVICE_MODEL, lazy_load=settings.LAZY_LOAD):
        self.torch_device=torch_device
        self._mol_seg = None
        self._mol_rec = None
        self._table_det = None
        self._table_rec = None
        self._text_det = None
        self._text_rec = None
        self._math_rec = None
        self._r_group_seg = None
        if not lazy_load:
            self.load_models()

    def load_models(self):
        _ = self.mol_rec
        _ = self.mol_seg
        _ = self.table_det
        _ = self.table_rec
        _ = self.text_rec
        _ = self.text_det
        _ = self.math_rec
        _ = self.r_group_seg

    @property
    def mol_seg(self) -> MolSegmentor:
        if self._mol_seg is None:
            config = ModelConfig(model_path=settings.MOL_SEG_CHECKPOINT)  #
            self._mol_seg = MolSegmentor.load_model(config)
        return self._mol_seg

    @property
    def r_group_seg(self) -> RGroupSegmentor:
        if self._r_group_seg is None:
            config = ModelConfig(model_path=settings.RGROUP_DET_CHECKPOINT, device=self.torch_device)
            self._r_group_seg = RGroupSegmentor.load_model(config)
        return self._r_group_seg

    @property
    def mol_rec(self) -> MolRecognizer:
        if self._mol_rec is None:
            config = ModelConfig(model_path=settings.MOL_REC_CHECKPOINT, device=self.torch_device)
            self._mol_rec = MolRecognizer.load_model(config)
        return self._mol_rec

    @property
    def table_det(self) -> TableDetector:
        if self._table_det is None:
            config = ModelConfig(model_path=settings.TABLE_DET_CHECKPOINT, device=self.torch_device)
            self._table_det = TableDetector.load_model(config)
        return self._table_det

    @property
    def table_rec(self) -> TableRecognizer:
        if self._table_rec is None:
            config = ModelConfig(model_path=settings.TABLE_REC_CHECKPOINT, device=self.torch_device)
            self._table_rec = TableRecognizer.load_model(config)
        return self._table_rec

    @property
    def text_det(self) -> TextDetector:
        if self._text_det is None:
            config = ModelConfig(model_path=settings.TEXT_DET_CHECKPOINT, device=self.torch_device)
            self._text_det = TextDetector.load_model(config)
        return self._text_det

    @property
    def text_rec(self) -> TextRecognizer:
        if self._text_rec is None:
            config = ModelConfig(model_path=settings.TEXT_REC_CHECKPOINT, device=self.torch_device)
            self._text_rec = TextRecognizer.load_model(config)
            self._text_rec.detect_model = self.text_det.model  # 给文本识别模型默认的文本检测模型
        return self._text_rec

    @property
    def math_rec(self) -> MathTextRecognizer:
        if self._math_rec is None:
            config = ModelConfig(model_path=settings.MATH_REC_CHECKPOINT)
            self._math_rec = MathTextRecognizer.load_model(config)
        return self._math_rec
