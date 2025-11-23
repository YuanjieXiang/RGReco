import torch
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from src.utils.image_processing import make_square, crop_bboxes
from src.utils.box_processing import move_polygon, expand_bbox, convert_polygon_to_bbox
from src.external.surya.models import LayoutPredictor
from .common import BaseModel, ModelConfig


class TableDetector(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = LayoutPredictor(checkpoint=config.model_path, device=config.device, dtype=torch.float16)

    def predict(self, images: List[Image.Image], **kwargs) -> list[dict[str, list[list] | float | None]]:
        # 填充图像防止变形，以提高检测率
        square_images = []
        offsets = []
        for image in images:
            quare_image, offset = make_square(image)
            square_images.append(quare_image)
            offsets.append(offset)
        predictions = self.model(square_images)

        # 检测框偏移回原位，每张图只保留最大的表格
        for i, layout_res in enumerate(predictions):
            source_image_size = images[i].size
            layout_res.image_bbox[2] = source_image_size[0]
            layout_res.image_bbox[3] = source_image_size[1]
            layout_res.bboxes = [layout_box for layout_box in layout_res.bboxes if layout_box.label == 'Table']
            if len(layout_res.bboxes) > 1:
                layout_res.bboxes = [max(layout_res.bboxes, key=lambda box: box.area)]  # 如果有多张表格，只保留最大的表格
            for layout_box in layout_res.bboxes:
                layout_box.polygon = move_polygon(layout_box.polygon, offsets[i], source_image_size)

        responses = []
        for pred in predictions:
            if pred.bboxes:
                bbox = pred.bboxes[0]
                # 每个输入图像只取一个 bbox，并处理它的 polygon 坐标为整数
                raw_polygon = bbox.polygon
                confidence = bbox.confidence
                rounded_polygon = [[round(x), round(y)] for x, y in raw_polygon]
                responses.append({"confidence": confidence, "polygon": rounded_polygon})
            else:
                responses.append({})
        return responses

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'TableDetector':
        return TableDetector(config)

    @staticmethod
    def visualize(image: Image.Image, prediction: dict[str, list[list] | float | None]) -> Image.Image:
        assert (prediction.get("confidence", None) and prediction.get("polygon", None) and len(prediction.get("polygon", None)) >= 4)
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        color = 'green'
        box = convert_polygon_to_bbox(prediction.get("polygon"))
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 5, box[1] + 5), f"Table", fill=color, font=font)
        return canvas

    @staticmethod
    def crop_bboxes(image: Image.Image, prediction: dict[str, list[list] | float | None]) -> Tuple[List[Image.Image], Image.Image]:
        assert prediction.get("polygon", None) and len(prediction.get("polygon", None)) >= 4
        box = expand_bbox(image.size, convert_polygon_to_bbox(prediction.bbox.get("polygon")))
        cropped_images, erased_image = crop_bboxes(image, [box])
        return cropped_images, erased_image

