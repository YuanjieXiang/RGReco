from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont

from src.external.surya.models import TableRecPredictor
from src.external.surya.table_rec.schema import TableResult, TableCell
from .common import BaseModel, ModelConfig


class TableRecognizer(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = TableRecPredictor(checkpoint=config.model_path, device=config.device, dtype=torch.float16)

    def predict(self, images: List[Image.Image], **kwargs) -> List[TableResult]:
        return self.model(images)

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'TableRecognizer':
        return cls(config)

    @staticmethod
    def visualize(image: Image.Image, prediction: TableResult) -> Image.Image:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        for cell in prediction.cells:
            draw.polygon([tuple(point) for point in cell.polygon], outline="red")  # 绘制cell线才能看出单元格合并
            show_str = f'{cell.row_id}|{cell.col_id}'
            draw.text((cell.polygon[0][0] + 5, cell.polygon[0][1] + 12), show_str, fill="green", font=font)
        # for row in prediction.rows:
        #     draw.polygon([tuple(point) for point in row.polygon], outline="red")
        # for col in prediction.cols:
        #     draw.polygon([tuple(point) for point in col.polygon], outline="blue")
        return canvas

