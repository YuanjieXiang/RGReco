import logging
from functools import cmp_to_key
from typing import Dict, Any, List

import cv2
import numpy as np

from src.settings import settings
from src.external import decimer_segmentation
from src.external.r_group_detection import yolo_predict
from src.utils.box_processing import convert_mask_to_polygon, is_above, is_left, convert_mask_to_box
from src.utils.image_processing import crop_masks, remove_masks
from .common import BaseModel, ModelConfig

log = logging.getLogger(__name__)
PLETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
          [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128],
          [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128],
          [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64], [128, 0, 64], [0, 128, 64], [128, 128, 64],
          [0, 0, 192], [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64], [192, 0, 64], [64, 128, 64],
          [192, 128, 64], [64, 0, 192], [192, 0, 192], [64, 128, 192], [192, 128, 192], [0, 64, 64], [128, 64, 64],
          [0, 192, 64], [128, 192, 64], [0, 64, 192], [128, 64, 192], [0, 192, 192], [128, 192, 192], [64, 64, 64],
          [192, 64, 64], [64, 192, 64], [192, 192, 64], [64, 64, 192], [192, 64, 192], [64, 192, 192], [192, 192, 192],
          [32, 0, 0], [160, 0, 0], [32, 128, 0], [160, 128, 0], [32, 0, 128], [160, 0, 128], [32, 128, 128],
          [160, 128, 128], [96, 0, 0], [224, 0, 0], [96, 128, 0], [224, 128, 0], [96, 0, 128], [224, 0, 128],
          [96, 128, 128], [224, 128, 128], [32, 64, 0], [160, 64, 0], [32, 192, 0], [160, 192, 0], [32, 64, 128],
          [160, 64, 128], [32, 192, 128], [160, 192, 128], [96, 64, 0], [224, 64, 0], [96, 192, 0], [224, 192, 0],
          [96, 64, 128], [224, 64, 128], [96, 192, 128], [224, 192, 128], [32, 0, 64], [160, 0, 64], [32, 128, 64],
          [160, 128, 64], [32, 0, 192], [160, 0, 192], [32, 128, 192], [160, 128, 192], [96, 0, 64], [224, 0, 64],
          [96, 128, 64], [224, 128, 64], [96, 0, 192], [224, 0, 192], [96, 128, 192], [224, 128, 192], [32, 64, 64],
          [160, 64, 64], [32, 192, 64], [160, 192, 64], [32, 64, 192], [160, 64, 192], [32, 192, 192], [160, 192, 192],
          [96, 64, 64], [224, 64, 64], [96, 192, 64], [224, 192, 64], [96, 64, 192], [224, 64, 192], [96, 192, 192],
          [224, 192, 192], [0, 32, 0], [128, 32, 0], [0, 160, 0], [128, 160, 0], [0, 32, 128], [128, 32, 128],
          [0, 160, 128], [128, 160, 128], [64, 32, 0], [192, 32, 0], [64, 160, 0], [192, 160, 0], [64, 32, 128],
          [192, 32, 128], [64, 160, 128], [192, 160, 128], [0, 96, 0], [128, 96, 0], [0, 224, 0], [128, 224, 0],
          [0, 96, 128], [128, 96, 128], [0, 224, 128], [128, 224, 128], [64, 96, 0], [192, 96, 0], [64, 224, 0],
          [192, 224, 0], [64, 96, 128], [192, 96, 128], [64, 224, 128], [192, 224, 128], [0, 32, 64], [128, 32, 64],
          [0, 160, 64], [128, 160, 64], [0, 32, 192], [128, 32, 192], [0, 160, 192], [128, 160, 192], [64, 32, 64],
          [192, 32, 64], [64, 160, 64], [192, 160, 64], [64, 32, 192], [192, 32, 192], [64, 160, 192], [192, 160, 192],
          [0, 96, 64], [128, 96, 64], [0, 224, 64], [128, 224, 64], [0, 96, 192], [128, 96, 192], [0, 224, 192],
          [128, 224, 192], [64, 96, 64], [192, 96, 64], [64, 224, 64], [192, 224, 64], [64, 96, 192], [192, 96, 192],
          [64, 224, 192], [192, 224, 192], [32, 32, 0], [160, 32, 0], [32, 160, 0], [160, 160, 0], [32, 32, 128],
          [160, 32, 128], [32, 160, 128], [160, 160, 128], [96, 32, 0], [224, 32, 0], [96, 160, 0], [224, 160, 0],
          [96, 32, 128], [224, 32, 128], [96, 160, 128], [224, 160, 128], [32, 96, 0], [160, 96, 0], [32, 224, 0],
          [160, 224, 0], [32, 96, 128], [160, 96, 128], [32, 224, 128], [160, 224, 128], [96, 96, 0], [224, 96, 0],
          [96, 224, 0], [224, 224, 0], [96, 96, 128], [224, 96, 128], [96, 224, 128], [224, 224, 128], [32, 32, 64],
          [160, 32, 64], [32, 160, 64], [160, 160, 64], [32, 32, 192], [160, 32, 192], [32, 160, 192], [160, 160, 192],
          [96, 32, 64], [224, 32, 64], [96, 160, 64], [224, 160, 64], [96, 32, 192], [224, 32, 192], [96, 160, 192],
          [224, 160, 192], [32, 96, 64], [160, 96, 64], [32, 224, 64], [160, 224, 64], [32, 96, 192], [160, 96, 192],
          [32, 224, 192], [160, 224, 192], [96, 96, 64], [224, 96, 64], [96, 224, 64], [224, 224, 64], [96, 96, 192],
          [224, 96, 192], [96, 224, 192], [224, 224, 192]]


class RGroupSegmentor(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = yolo_predict.RGroupSegmentor(model_path=config.model_path, device=config.device)

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'RGroupSegmentor':
        return cls(config)

    def predict(self, images: List[np.ndarray], **kwargs) -> list[dict[str: Any]]:
        is_backbone = False
        if "is_backbone" in kwargs and kwargs["is_backbone"]:
            is_backbone = True

        return self.model.detect_images(images, is_backbone)

    @staticmethod
    def visualize(image: np.ndarray, prediction: List) -> np.ndarray:
        for det in prediction:
            pts, score, cls_id = det["polygon"], det["confidence"], det["class_id"]
            if pts:
                label = f"{cls_id} {score:.2f}"
                color = (0, 255, 0)
                pts_np = np.array(pts, dtype=np.int32)
                image = image.copy()
                # 绘制多边形
                cv2.polylines(image, [pts_np], isClosed=True, color=color, thickness=2)
                # 标注类别和置信度（在第一个点上）
                cv2.putText(image, label, tuple(pts_np[0]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1)
        return image

    @staticmethod
    def replace_polygon(image: np.ndarray, polygon: list[list[int]], cls_id: int) -> np.ndarray:
        image = yolo_predict.postprocess(image, polygon, cls_id)
        return image


class MolSegmentor(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = decimer_segmentation.DECIMERSegmentor(config.model_path, providers=settings.ONNX_PROVIDERS)

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'MolSegmentor':
        return cls(config)

    def predict(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """图像预测接口"""
        predictions = []
        for image in images:
            image_copy = image.copy()
            rois, scores, masks = self.model.predict(image_copy)

            min_height = 30
            valid_indices = []
            # bboxes = []
            for i in range(len(rois)):
                y1, x1, y2, x2 = rois[i]
                height = y2 - y1
                if height > min_height:
                    valid_indices.append(i)
                    # bboxes.append([x1, y1, x2, y2])

            scores = [scores[i] for i in valid_indices]
            masks = masks[:, :, valid_indices] if len(valid_indices) > 0 else np.zeros(
                (masks.shape[0], masks.shape[1], 0))

            polygons = []
            for i in range(len(scores)):
                mask_slice = masks[:, :, i]  # 取出第 i 个掩膜
                polygon = convert_mask_to_polygon(mask_slice)
                if polygon:
                    polygons.append(polygon)

            # postprocess
            processed_masks = decimer_segmentation.complete_structure_mask(image_copy, masks, image_copy.shape[:2])
            scores = np.array(scores)
            structures = [
                {
                    "mask": processed_masks[:, :, i],
                    "score": scores[i],
                    "bbox": convert_mask_to_box(processed_masks[:, :, i]),
                    "polygon": polygons[i]
                }
                for i in range(len(scores))
            ]

            # 通过位置排序
            def compare_objects(a, b):
                bbox_a = a['bbox']
                bbox_b = b['bbox']
                if is_above(bbox_a, bbox_b):  # a 在 b 下面 → b 排前面
                    return 1
                elif is_above(bbox_b, bbox_a):  # b 在 a 下面 → a 排前面
                    return -1
                if is_left(bbox_a, bbox_b):  # a 在 b 左边 → a 排前面
                    return -1
                elif is_left(bbox_b, bbox_a):  # b 在 a 左边 → b 排前面
                    return 1
                return 0

            structures = sorted(structures, key=cmp_to_key(compare_objects))

            predictions.append({"structures": structures})
        return predictions

    @staticmethod
    def visualize(image: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        """模型输出可视化接口"""
        structures = prediction.get('structures', [])
        if len(structures) == 0:
            return image

        num_instances = len(structures)
        overlay = image.copy()  # 创建叠加图像
        # 将每个实例的掩膜叠加到原图上
        for i in range(num_instances):
            instance_mask = structures[i].get('mask', None)  # 获取当前实例处理后的掩膜
            if instance_mask is None:
                continue
            color = PLETTE[i % 255]  # 当前实例的颜色
            for c in range(3):  # 对 RGB 通道分别处理
                overlay[:, :, c] = np.where(
                    instance_mask == True,  # 掩膜条件
                    overlay[:, :, c] * 0.5 + color[c] * 0.5,  # 半透明叠加
                    overlay[:, :, c]  # 保留原图
                )
            # cv2.imshow('test', overlay)
            # cv2.waitKey(0)
        return overlay

    @staticmethod
    def crop_masks(image: np.ndarray, masks: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        # masks = prediction.get('masks', np.array([]))
        return crop_masks(image, masks), remove_masks(image, masks)
