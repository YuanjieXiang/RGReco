from typing import Dict

import torch

from .common.predictor import BasePredictor
from src.external.surya.detection import DetectionPredictor
from src.external.surya.layout import LayoutPredictor
from src.external.surya.table_rec import TableRecPredictor
from src.external.surya.recognition import RecognitionPredictor


def load_predictors(
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None
) -> Dict[str, BasePredictor]:
    return {
        "layout": LayoutPredictor(device=device, dtype=dtype),
        "detection": DetectionPredictor(device=device, dtype=dtype),
        "recognition": RecognitionPredictor(device=device, dtype=dtype),
        "table_rec": TableRecPredictor(device=device, dtype=dtype)
    }