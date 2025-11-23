import torch
import logging
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from typing import Any, List
from pylatexenc.latex2text import LatexNodes2Text
import re

from .common import BaseModel, ModelConfig
from src.settings import settings
from src.utils.image_processing import make_square
from src.utils.box_processing import move_polygon
from src.external.surya.models import DetectionPredictor, RecognitionPredictor
from src.external.surya.detection.schema import TextDetectionResult
from src.external.surya.recognition.schema import OCRResult, TextLine
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import TrOCRProcessor

log = logging.getLogger(__name__)


class TextDetector(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = DetectionPredictor(checkpoint=config.model_path, device=config.device, dtype=torch.float32)

    def predict(self, images: List[Image.Image], **kwargs) -> List[TextDetectionResult]:
        # 填充图像为矩形防止变形，以提高检测率
        square_images = []
        offsets = []
        for image in images:
            quare_image, offset = make_square(image)
            square_images.append(quare_image)
            offsets.append(offset)
        predictions = self.model(square_images)
        # 恢复边界框
        for i, det_res in enumerate(predictions):
            source_image_size = images[i].size
            det_res.image_bbox[2] = source_image_size[0]
            det_res.image_bbox[3] = source_image_size[1]
            for poly_box in det_res.bboxes:
                poly_box.polygon = move_polygon(poly_box.polygon, offsets[i], source_image_size)
        return predictions

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'TextDetector':
        return cls(config)

    @staticmethod
    def visualize(image: Image.Image, prediction: TextDetectionResult) -> Image.Image:
        assert image.size[0] == prediction.image_bbox[2] and image.size[1] == prediction.image_bbox[
            3], f"The dimensions {image.size} and {prediction.image_bbox[2:]} do not match."
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        text_bboxes = prediction.bboxes
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for layout_box in text_bboxes:
            bbox = layout_box.bbox
            draw.rectangle(bbox, outline='green', width=2)
            draw.text((bbox[0] + 2, bbox[1] - 12), f"{layout_box.confidence:.2f}", fill='blue', font=font)
        return canvas


class TextRecognizer(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = RecognitionPredictor(checkpoint=config.model_path, device=config.device, dtype=torch.float32)
        self.detect_model: DetectionPredictor | None = None

    def predict(self, images: List[Image.Image], **kwargs) -> List[OCRResult]:
        if "text_detector" in kwargs:
            self.detect_model = kwargs['text_detector']
        assert self.detect_model is not None, "需要文本检测模型"
        # 填充图像为矩形防止变形，以提高检测率
        square_images = []
        offsets = []
        for image in images:
            square_image, offset = make_square(image)
            square_images.append(square_image)
            offsets.append(offset)
        predictions = self.model(square_images, len(images) * [['en']], det_predictor=self.detect_model)
        # 恢复边界框
        for i, ocr_res in enumerate(predictions):
            source_image_size = images[i].size
            ocr_res.image_bbox[2] = source_image_size[0]
            ocr_res.image_bbox[3] = source_image_size[1]
            for text_line in ocr_res.text_lines:
                text_line.text = text_line.text.strip().replace('CI', 'Cl').replace('RI', 'R1')  # 将I修复为l
                text_line.polygon = move_polygon(text_line.polygon, offsets[i], source_image_size)
        return predictions

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'TextRecognizer':
        return cls(config)

    @staticmethod
    def visualize(image: Image.Image | np.ndarray, prediction: OCRResult) -> Image.Image:
        assert image.size[0] == prediction.image_bbox[2] and image.size[1] == prediction.image_bbox[
            3], f"The dimensions {image.size} and {prediction.image_bbox[2:]} do not match."
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        text_lines = prediction.text_lines
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for text_box in text_lines:
            bbox = text_box.bbox
            draw.rectangle(bbox, outline='green', width=2)
            draw.text((bbox[0] + 2, bbox[1] - 12), f"{text_box.text} {text_box.confidence:.2f}", fill='red', font=font)
        return canvas

    @staticmethod
    def sort_and_merge_text_lines(text_lines: List[TextLine], structure_lines: List[TextLine]) -> List[str]:
        """
        先按y坐标聚合成行，再对每行内的文本按x坐标排序合并

        Args:
            text_lines: TextLine对象列表
            structure_lines :  TextLine对象列表

        Returns:
            排序后的文本行列表
        """
        if not text_lines:
            return []

        x_threshold = 50 if len(structure_lines) == 0 else 300  # 当存在结构式，大大增加同行的横坐标阈值（R基团结构式的宽度一般在50-250左右，用300保一点余量）

        # Step 1: 提取每个 TextLine 的坐标信息
        lines_info = []
        for tl in text_lines:
            x0, y0, x1, y1 = tl.bbox
            center_y = (y0 + y1) / 2
            center_x = (x0 + x1) / 2  # 使用左边界作为x坐标
            lines_info.append((center_y, center_x, tl))

        # Step 2: 先按 y 坐标排序（从上到下）
        lines_info.sort(key=lambda x: x[0])

        # Step 3: 聚合为"行" - 根据y坐标的相似性分组，阈值为10是因为大多数情况下文本行高一般大于10
        def is_same_line(y1, y2, x1, x2, y_threshold=10, x_threshold=20):
            """判断两个y坐标是否在同一行"""
            return abs(y1 - y2) <= y_threshold and abs(x1 - x2) < x_threshold

        text_rows = []  # 存储每一行的TextLine列表
        current_row = []

        for i, (center_y, left_x, tl) in enumerate(lines_info):
            if i == 0:
                # 第一个元素
                current_row.append((left_x, tl))
            else:
                prev_y, prev_x, _ = lines_info[i - 1]
                if is_same_line(center_y, prev_y, left_x, prev_x, x_threshold=x_threshold):
                    # 在同一行
                    current_row.append((left_x, tl))
                else:
                    # 新的一行
                    if current_row:
                        text_rows.append(current_row)
                    current_row = [(left_x, tl)]

        # 处理最后一行
        if current_row:
            text_rows.append(current_row)

        # 将结构式插入行中
        for stl in structure_lines:
            x1, y1, _, y2 = stl.bbox
            for row in text_rows:
                _, y3, _, y4 = row[0][1].bbox
                if y3 > y1 and y4 < y2: # 文本的纵坐标区间在结构式的纵坐标区间内
                    row.append((x1, stl))
                    break

        # Step 4: 对每一行内的文本按x坐标排序，然后合并
        result = []
        for row in text_rows:
            # 按x坐标排序（从左到右）
            row.sort(key=lambda x: x[0])

            # 合并该行的文本
            row_texts = [tl.text for _, tl in row]
            merged_text = " ".join(row_texts)

            # 处理可能的换行符
            if '\n' in merged_text:
                result.extend(line.strip() for line in merged_text.split('\n'))
            else:
                result.append(merged_text.strip())

        # Step 5: 清理空行
        result = [line for line in result if line.strip()]

        return result


class MathTextRecognizer(BaseModel):
    def __init__(self, config):
        self.config = config
        self.processor = TrOCRProcessor.from_pretrained(config.model_path)
        self.model = ORTModelForVision2Seq.from_pretrained(config.model_path, use_cache=False, use_io_binding=False,
                                                           providers=settings.ONNX_PROVIDERS)
        config.device = settings.ONNX_DEVICE
        self.model.to(config.device)

    def predict(self, images: List[Image.Image], **kwargs) -> List[str]:
        """由于数学模型容易将下标5识别错误为s，启用repair会自动修复识别结果"""
        batch_size = 8

        if "batch_size" in kwargs and isinstance(kwargs["batch_size"], int):
            batch_size = kwargs["batch_size"]

        results = []
        for i in range(0, len(images), batch_size):
            part_imgs = images[i: i + batch_size]
            pixel_values = self.processor(images=part_imgs, return_tensors="pt").pixel_values
            ocr_res = self.model.generate(pixel_values.to(self.config.device), return_dict_in_generate=True,
                                          output_scores=True)
            generated_texts = self.processor.batch_decode(ocr_res.sequences, skip_special_tokens=True)

            if "parse_latex" in kwargs and kwargs["parse_latex"] is True:
                generated_texts = [self.parse_latex(text.replace(' ', '')) for text in generated_texts]

                if "repair" in kwargs and kwargs["repair"] is True:
                    # 已知错误，将下标 5 识别为 s，将 1 识别为 l
                    for j, text in enumerate(generated_texts):
                        generated_texts[j] = (
                            text.replace('_s', '_5')
                            .replace('_S', '_5')
                            .replace('_l', '_1')
                            .replace('_i', '_1')
                            .replace('_I', '_1')
                            .replace('^I', '^1')
                            .replace('_T', '_1')
                            .replace('^T', '^1')
                        )
            results.extend(generated_texts)
        return results

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'MathTextRecognizer':
        return cls(config)

    @staticmethod
    def visualize(image: Image.Image | np.ndarray, prediction: Any):
        log.warning("This method is not required for this subclass")
        pass

    def parse_latex(self, latex_str):
        preprocessed = self._parse_latex(latex_str.replace("<math>", '').replace("</math>", ''))
        # normal_str = LatexNodes2Text().latex_to_text(preprocessed)
        normal_str = self.latex_to_text_clean(preprocessed)
        return normal_str

    @staticmethod
    def latex_to_text_clean(latex_str):
        # Step 1: 将 LaTeX 转换为带有基本格式的文本（保留Unicode上下标等）
        try:
            text = LatexNodes2Text().latex_to_text(latex_str)
        except Exception:
            text = latex_str


        # Step 2: 正则替换所有特殊符号和格式残留
        # 删除上标符号（如 ²³¹ 转换为普通数字）
        text = re.sub(r'[\u00B2\u00B3\u00B9\u2070\u2074-\u207F]',
                      lambda m: chr(ord(m.group()) - 0x2070 + 48) if ord(m.group()) >= 0x2070 else '', text)
        # 删除其他 LaTeX 符号（如 \quad、\frac 转换后的斜杠等）
        # text = re.sub(r'[\\{}]', '', text)  # 删除反斜杠、花括号
        text = re.sub(r'\s+', ' ', text)  # 合并多余空格
        return text.strip()

    def _parse_latex(self, sub_str: str):
        sub_strs = []
        i = 0
        while i < len(sub_str):
            if sub_str[i] == "\\":  # 遇到斜杠表达式，直接跳过，找到下一个非字母为止
                end = i + 1
                while end < len(sub_str) and sub_str[end].isalpha():
                    end += 1
                if end >= len(sub_str) or sub_str[end] != "{":
                    sub_strs.append(sub_str[i:end] + " ")  # 如果表达式后面没括号,保留表达式
                i = end
            elif sub_str[i] == "{":  # 递归
                cnt = 1
                start = i
                i += 1
                while i < len(sub_str) and cnt > 0:
                    if sub_str[i] == '{':
                        cnt += 1
                    elif sub_str[i] == '}':
                        cnt -= 1
                    i += 1
                if cnt == 0:
                    sub_strs.append(self._parse_latex(sub_str[start + 1: i - 1]))
                else:
                    sub_strs.append(sub_str[start:i])
            else:
                sub_strs.append(sub_str[i])
                i += 1
        return ''.join(sub_strs)
