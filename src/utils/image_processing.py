import numpy as np
import math
import cv2
from PIL import Image, ImageDraw
from typing import List, Tuple
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def crop_bboxes(image: Image.Image, bboxes) -> Tuple[List[Image.Image], Image.Image]:
    """裁剪边框同时返回删除框内像素的原图"""
    # Step 1: 裁剪出每个边框内的图像
    cropped_images = crop_rectangles(image, bboxes)

    # Step 2: 创建一个与原图相同大小的新图像用于绘制擦除效果
    erased_image = image.copy()
    draw = ImageDraw.Draw(erased_image)

    # Step 3: 在 erased_image 上将每个 bbox 区域填充为白色
    for bbox in bboxes:
        left, upper, right, lower = bbox
        draw.rectangle([left, upper, right, lower], fill=(255, 255, 255))
    # 返回裁剪图列表和擦除后的图像
    return cropped_images, erased_image


def make_square(image: Image, fill_color=(255, 255, 255)):
    """将输入的图像填充为正方形，默认填充白色像素
    Args:
        image (Image): 原始图像
        fill_color (tuple, optional): 填充的像素颜色. Defaults to (255, 255, 255).
    Returns:
        Image: 正方形图像
    """
    width, height = image.size
    max_dim = max(width, height)
    new_image = Image.new('RGB', (max_dim, max_dim), fill_color)
    offset = ((max_dim - width) // 2, (max_dim - height) // 2)
    new_image.paste(image, offset)
    return new_image, offset


def crop_rectangles(image: Image.Image, bboxes: list) -> List[Image.Image]:
    """
    批量从图像中裁剪矩形框。
    Args:
        image (Image.Image): PIL图像.
        bboxes (list): 裁剪框.
    Returns:
        List[Image.Image]: 裁剪后的图像列表.
    """
    cropped_images = []
    for bbox in bboxes:
        # bbox 格式: (x_min, y_min, x_max, y_max)
        cropped = image.crop(bbox)
        cropped_images.append(cropped)
    return cropped_images


def crop_masks(image: np.ndarray, masks: np.ndarray) -> List[np.ndarray]:
    """
    批量从图像中分割掩码对应区域。
    Args:
        image (np.ndarray): cv图像
        masks (np.ndarray): 掩码矩阵，形状为(h, w, c)

    Returns:
        List[np.ndarray]: 分割掩膜得到的图像
    """
    cropped_images = []
    # 遍历每个掩码通道
    for i in range(masks.shape[-1]):  # 遍历最后一个维度（掩码数量）
        mask = masks[:, :, i]  # 提取单个掩码，形状为 (h, w)
        # 计算裁剪区域
        y, x = np.where(mask > 0)  # 找到所有 mask 像素点
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # 创建一个白色背景图像，并复制原图中 mask 区域的内容
        cropped_white_bg = np.full_like(image[y_min:y_max + 1, x_min:x_max + 1], 255)
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1].copy()

        # 将非 mask 区域设为白色
        for c in range(3):  # 遍历每个通道
            cropped_white_bg[:, :, c] = np.where(
                cropped_mask > 0,
                cropped_image[:, :, c],
                255
            )
        cropped_images.append(cropped_white_bg)
    return cropped_images


def remove_masks(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    将原图中掩膜对应的像素区域用白色填充，返回处理后的图像

    Args:
        image (np.ndarray): 原始图像，形状为(H, W)或(H, W, 3)
        masks (np.ndarray): 掩膜矩阵，形状为(H, W, C)

    Returns:
        np.ndarray: 移除掩膜对应区域的图像
    """
    combined_mask = np.any(masks > 0, axis=-1)  # 合并所有掩膜通道（逻辑或操作）, 自动降维到(H, W)
    combined_mask = combined_mask.astype(np.uint8) * 255  # 转换为二值掩膜
    fill_value = 255 if image.ndim == 2 else [255, 255, 255]  # 根据图像类型确定填充值

    # 创建图像副本并白化掩膜区域
    result = image.copy()
    result[combined_mask.astype(bool)] = fill_value

    return result


def image_strip(img, bg_color=(255, 255, 255)) -> Image.Image:
    """
    去除图像周围的所有指定背景色（默认为白色）像素。
    Args:
        img: 输入的PIL Image对象（三通道彩色图像）。
        bg_color: 背景颜色，默认为白色。
    Returns: 
        Image.Image: 处理后的图像对象。
    """
    # 确保图像是RGB模式
    if img.mode != 'RGB':
        img = img.convert("RGB")

    # 获取图像尺寸
    width, height = img.size

    # 转换图像数据为像素列表
    pixels = img.load()

    # 查找非背景颜色的边界
    left, top, right, bottom = width, height, -1, -1
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            if pixel != bg_color:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    # 如果没有找到非背景颜色，则返回原始图像或空白图像
    if left > right or top > bottom:
        print("Image is fully background color.")
        return Image.new("RGB", (1, 1), (0, 0, 0))  # 返回一个1x1的黑色图像

    # 根据找到的边界裁剪图像
    cropped_img = img.crop((left, top, right + 1, bottom + 1))

    return cropped_img


def stitch_images(images: List[Image.Image], grid_line=True):
    """将输入的图片以网格排列，不改变图像尺寸"""
    if not images:
        return None  # 返回空值或创建空白图

    # 计算网格行列数 (接近平方根比例，行比列多)
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # 获取所有图片最大宽高作为单元格尺寸
    cell_width = max(img.width for img in images)
    cell_height = max(img.height for img in images)

    # 创建大画布 (灰色背景)
    width = cols * cell_width
    height = rows * cell_height
    canvas = Image.new("RGB", (width, height), (200, 200, 200))

    # 将每张图片粘贴到对应网格位置
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * cell_width
        y = row * cell_height

        offset_x = (cell_width - img.width) // 2
        offset_y = (cell_height - img.height) // 2

        canvas.paste(img, (x + offset_x, y + offset_y))

    if grid_line:
        draw = ImageDraw.Draw(canvas)
        for col in range(1, cols):
            x = col * cell_width
            draw.line([(x, 0), (x, height)], fill="gray", width=1)
        for row in range(1, rows):
            y = row * cell_height
            draw.line([(0, y), (width, y)], fill="gray", width=1)

    return canvas


def binarize_image(image_array: np.array, threshold="otsu") -> np.array:
    """
    This function takes a np.array that represents an RGB image and returns
    the binarized image (np.array) by applying the otsu threshold.

    Args:
        image_array (np.array): image
        threshold (str, optional): "otsu" or a float. Defaults to "otsu".

    Returns:
        np.array: binarized image
    """
    grayscale = rgb2gray(image_array)
    if threshold == "otsu":
        threshold = threshold_otsu(grayscale)
    binarized_image_array = grayscale < threshold
    return binarized_image_array