from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
from PIL import Image


@dataclass
class ModelConfig:
    """统一模型配置类"""
    model_path: str
    device: str= 'cpu'
    model_name: str = None


class BaseModel(ABC):
    """模型抽象基类"""

    @abstractmethod
    def predict(self, images: List[Image.Image | np.ndarray], **kwargs) -> List[Any]:
        """图像预测接口"""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, config: ModelConfig) -> 'BaseModel':
        """加载模型的类方法"""
        pass

    @staticmethod
    @abstractmethod
    def visualize(image: Image.Image | np.ndarray, prediction: Any) -> Image.Image | np.ndarray:
        """模型输出可视化接口"""
        pass

