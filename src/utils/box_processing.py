import cv2
import numpy as np
from PIL import Image
from typing import List


def convert_polygon_to_bbox(polygon) -> List[int | float]:
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def convert_mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    polygon = []
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        # 简化轮廓
        epsilon = 0.01 * cv2.arcLength(largest_contour, closed=True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, closed=True)

        polygon = approx.squeeze().tolist()
    return polygon


def convert_mask_to_polygon_special(mask: np.ndarray) -> List[List[int]]:
    """
    输入: 一个二值掩膜 mask (0 和 1 组成的二维数组)
    输出: 一个包含四个点的多边形 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    这四个点分别是: 最左、最右、最上、最下
    """
    # 将掩膜转为 uint8 类型，值为 0-255
    height, width = mask.shape
    mask_uint8 = (mask.astype(np.uint8) * 255)

    # 查找所有轮廓
    contours, _ = cv2.findContours(mask_uint8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []
    # 合并所有轮廓点
    all_points = np.concatenate(contours, axis=0).squeeze()
    # 找出最左、最右、最上、最下四个点
    leftmost = all_points[all_points[:, 0].argmin()].tolist()
    rightmost = all_points[all_points[:, 0].argmax()].tolist()
    topmost = all_points[all_points[:, 1].argmin()].tolist()
    bottommost = all_points[all_points[:, 1].argmax()].tolist()
    # 向各自方向偏移一个像素，并限制在图像范围内
    leftmost[0] = max(leftmost[0] - 1, 0)
    rightmost[0] = min(rightmost[0] + 1, width - 1)
    topmost[1] = max(topmost[1] - 1, 0)
    bottommost[1] = min(bottommost[1] + 1, height - 1)
    # 组合成一个多边形 (顺序为左、右、上、下)
    polygon = [list(map(int, point)) for point in [leftmost, topmost, rightmost, bottommost]]
    return polygon


def convert_polygon_to_mask(polygon: np.ndarray | list, image_size: tuple) -> np.ndarray:
    """
    将多边形转换为二值掩膜。

    参数:
        polygon (np.ndarray or list): 多边形顶点数组，shape 应该是 (N, 2)。
        image_size (tuple): 掩膜的大小，格式为 (height, width)。

    返回:
        np.ndarray: 二值掩膜图像，类型为 np.uint8，0 表示背景，1 表示前景。
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    # 确保 polygon 是 numpy 数组，并且 shape 为 (N, 2)
    if isinstance(polygon, list):
        polygon = np.array(polygon, dtype=np.int32)
    # 使用 cv2.fillPoly 填充多边形区域
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)

    return mask


def convert_mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    将二值掩膜转换为外接矩形边界框 (x_min, y_min, x_max, y_max)。

    参数:
        mask (np.ndarray): 二值掩膜图像，形状为 (H, W)，值为 0（背景）或非零（前景）。

    返回:
        tuple[int, int, int, int] | None:
            (x_min, y_min, x_max, y_max) 表示边界框坐标；
            如果掩膜中无前景像素，返回 None。
    """
    # 找到所有前景像素的位置（返回 (row, col) 即 (y, x)）
    coords = np.argwhere(mask)

    if coords.size == 0:
        return None  # 或可根据需求返回 (0, 0, 0, 0) 等

    # coords 的每一行是 [y, x]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def move_polygon(polygon: List[List], offset: List, img_size: tuple[int, int]):
    """
    移动多边形
    Args:
        polygon (List): 多边形.
        offset (List): 多边形移动的偏移量 (delta_x, delta_y).
        img_size (List): 图像的尺寸 (width, height).
    Returns:
        List: 移动后的多边形.
    """
    new_poly = []
    delta_x, delta_y = offset
    for x, y in polygon:
        # 坐标逆向偏移
        x_adj = x - delta_x
        y_adj = y - delta_y
        # 截断到原始图像边界
        x_final = max(0, min(x_adj, img_size[0]))
        y_final = max(0, min(y_adj, img_size[1]))
        new_poly.append([x_final, y_final])
    return new_poly


def expand_bbox(size: tuple | list, box: tuple | list, scale=0.05, max_pixel=8):
    """
    对边界框进行按比例扩张并裁剪图像。
    
    Args:
        size (tuple | list): 图像的大小，(width, height)。
        box (tuple | list): 原始边界框 (x1, y1, x2, y2)。
        scale (float): 在每个方向上的扩张比例，默认为0.05。
        max_pixel (int): 在每个方向上的最大扩张像素值，默认为 8
    Returns:
        tuple: 扩张后的边界框 (x1, y1, x2, y2)
    """
    width, height = map(int, size)
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # 计算扩张量（基于原宽高比例）
    dx = min(w * scale, max_pixel)
    dy = min(h * scale, max_pixel)

    # 计算新边界框坐标
    new_x1 = max(0, x1 - dx)
    new_y1 = max(0, y1 - dy)
    new_x2 = min(width, x2 + dx)
    new_y2 = min(height, y2 + dy)

    # 四舍五入并确保有效性
    new_box = (
        int(round(new_x1)),
        int(round(new_y1)),
        int(round(new_x2)),
        int(round(new_y2))
    )

    # 处理可能的无效区域（如扩张后宽高为0）
    if new_box[2] <= new_box[0]:
        new_box = (new_box[0], new_box[1], new_box[0] + 1, new_box[3])
    if new_box[3] <= new_box[1]:
        new_box = (new_box[0], new_box[1], new_box[2], new_box[1] + 1)

    return new_box


def is_above(a, b):
    """ a 的最低 y > b 的最高 y 表示 a 在 b 下方 """
    _, a_ymin, _, a_ymax = a
    _, b_ymin, _, b_ymax = b
    return a_ymin > b_ymax


def is_left(a, b):
    """ a 的最大 x < b 的最小 x 表示 a 在 b 左边 """
    a_xmin, _, a_xmax, _ = a
    b_xmin, _, _, _ = b
    return a_xmax < b_xmin


def is_top_left(a, b):
    """
    a 是比 b 更左上的物体吗？
    条件是：a 不在 b 的下面 && a 不在 b 的右边
    即：b 不在 a 上面，且 b 不在 a 左边
    """
    return not is_above(b, a) and not is_left(b, a)


def calculate_iou(box_a, box_b):
    """
    计算两个边界框的 IoU（交并比）

    参数:
        box_a: [x1, y1, x2, y2]，第一个框的坐标（左上角和右下角）
        box_b: [x1, y1, x2, y2]，第二个框的坐标

    返回:
        float: IoU 值，范围 [0, 1]，0 表示无重叠，1 表示完全重合
    """
    # 获取坐标
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # 计算交集区域的坐标
    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    # 计算交集面积
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # 计算各自面积
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    # 计算并集面积
    union_area = area_a + area_b - inter_area

    # 防止除以零
    if union_area == 0:
        return 0.0

    # 计算 IoU
    iou = inter_area / union_area
    return iou


def point_in_box(point, box):
    x1, y1, x2, y2 = box
    x, y = point
    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


def normalize_box(box, w, h):
    x1, y1, x2, y2 = box
    normalized_box = [
        x1 / w,  # x1_norm
        y1 / h,  # y1_norm
        x2 / w,  # x2_norm
        y2 / h  # y2_norm
    ]
    return normalized_box
