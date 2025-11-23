import math
import os.path

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image, ImageOps
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from ultralytics import YOLO

R_SYMBOLS = ['attach', 'cut', 'star', 'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', "R'", "R1'", "R2'", "R3'"]


class RGroupSegmentor():
    def __init__(self, model_path, device="cuda", input_size=(640, 640)):
        self.model = YOLO(model_path)
        self.device = device

    def detect_images(self, images, is_backbone=False):
        results = self.model.predict(source=images, device=self.device, save=False, verbose=False, conf=0.3)
        responses = []

        for result in results:
            items = []
            if result.obb is not None:
                # 遍历每个检测到的对象
                xyxyxyxy = result.obb.xyxyxyxy
                for i in range(len(result.obb)):

                    # 新增取代基符号检测，修改代码避免影响主流程
                    cls_id = result.obb.cls[i].int().item()  # 分类 ID
                    if not is_backbone and cls_id > 2:
                        continue

                    polygon_tensor = xyxyxyxy[i]  # 多边形点 (4, 2)
                    conf = result.obb.conf[i].item()  # 置信度
                    polygon_list = polygon_tensor.cpu().tolist()

                    # 添加为一个 item
                    items.append({
                        "polygon": polygon_list,
                        "confidence": conf,
                        "class_id": cls_id
                    })
            responses.append(items)  # 每张图片的结果是一个 item list

        return responses


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


def erase_polygon_area(image, polygon, color=None):
    """
    输入:
        image: 图像，可以是单通道灰度图或三通道图像 (np.uint8)
        polygon: 四边形的4个顶点坐标组成的数组，形状为 (4, 2)
        color: 要填充的颜色，默认为白色：
               - 灰度图：默认 255
               - 彩色图：默认 (255, 255, 255)

    返回:
        image: 修改后的图像（与输入图像类型一致）
    """
    # 判断图像通道数
    if len(image.shape) == 2:
        is_gray = True
        h, w = image.shape
    elif len(image.shape) == 3 and image.shape[2] == 3:
        is_gray = False
        h, w, _ = image.shape
    else:
        raise ValueError("image 必须是 HxW 的灰度图 或 HxWx3 的彩色图")

    # 自动设置颜色
    if color is None:
        color = 1 if is_gray else (255, 255, 255)

    # 创建掩码并绘制四边形区域
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)

    # 抹除对应区域
    result = image.copy()
    if is_gray:
        result[mask == 255] = color
    else:
        result[mask == 255] = color  # 自动广播到三个通道

    return result


def erase_mask_area(image, mask, color=(255, 255, 255)):
    """
    抹去图像中与输入 mask 对应的区域（mask 中值为 255 的地方）

    参数:
        image (np.ndarray): 输入图像，形状为 (H, W, 3)，np.uint8 类型
        mask (np.ndarray): 二值掩膜图像，形状为 (H, W)，np.uint8 类型，值为 0 或 255
        color (tuple): 要填充的颜色，默认为白色 (BGR 格式)

    返回:
        np.ndarray: 修改后的图像
    """
    # 确保 mask 是单通道且与图像尺寸一致
    assert mask.ndim == 2 and image.shape[:2] == mask.shape, "mask 必须是单通道且与图像大小一致"

    # 创建输出图像副本
    result = image.copy()

    # 将 mask 中为 255 的区域设置为指定颜色
    result[mask == 255] = color

    return result


def expand_rect(polygon, scale_factor=1.25):
    """
    输入:
        quad: 四边形的4个点组成的np数组，形状为 (4, 2)
        binary_mask: 二值化图像，布尔类型，True 表示前景
    输出:
        expanded_rect: 放大后的旋转矩形信息 (center, (width, height), angle)
        new_boxes: 三个矩形的顶点坐标列表 (3 个矩形，每个是 4 个点)
    """
    # 1. 获取最小外接旋转矩形
    rect = cv2.minAreaRect(polygon.astype(np.float32))
    center, (width, height), angle = rect

    # 2. 放大矩形到原来的scale_factor倍
    new_width = width * scale_factor
    new_height = height * scale_factor

    return center, (new_width, new_height), angle


def find_nearest_component(point, mask, threshold, min_area=3):
    """寻找掩膜中离点最近的连通块"""
    # Step 1: 将 mask 转换为 uint8 类型用于连通域分析
    mask_uint8 = (mask * 255).astype(np.uint8)
    # Step 2: 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        # print("没有检测到有效连通块")
        return None

    # Step 3: 遍历所有连通块，计算到 point 的距离
    min_distance = float('inf')
    nearest_label = -1
    px, py = point

    for label in range(1, num_labels):  # 忽略背景 label 0
        if stats[label, cv2.CC_STAT_AREA] < min_area:  # 忽略噪点
            continue
        cx, cy = centroids[label]
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

        if dist < min_distance and dist < threshold:
            min_distance = dist
            nearest_label = label

    # Step 4: 没有找到符合条件的连通块
    if nearest_label == -1:
        # print("没有找到在阈值范围内的连通块")
        return None

    return nearest_label, labels, stats, centroids


def erase_nearest_component(image, mask, point, threshold):
    """
    输入:
        image: 原始图像，np.uint8 类型，形状为 (H, W, 3)
        mask: 二值掩膜图像，np.bool_ 类型，形状为 (H, W)
        point: 点坐标 (x, y)，是整数元组
        threshold: 距离阈值，只有距离小于这个值的连通块才会被删除

    返回:
        result_image: 修改后的图像
    """
    # 复制图像防止修改原图
    result_image = image.copy()

    # 寻找最近连通块
    nearest_component = find_nearest_component(point, mask, threshold)
    if not nearest_component:
        return image

    nearest_label, labels, _, _ = nearest_component
    # 创建只包含最近连通块的掩膜
    nearest_mask = np.zeros_like(mask)
    nearest_mask[labels == nearest_label] = True

    # 抹去图像中对应区域（设为白色）
    result_image[nearest_mask == 1] = (255, 255, 255)

    return result_image


def find_mirror_point(near_edge, base_point, rect, mask):
    """根据靠近键的边（远离图像边界的点）寻找键的位置，如果找到了以键的点作为镜像计算文本替换位置，否则以远离键的点作为替换点"""
    height, width = mask.shape[:2]
    # 根据中心点和偏移点建立旋转矩形选区，选区用于寻找直线
    center, (w, h), angle = rect
    line_length = min(w, h)  # 长度为短边长

    # 1. 先计算单位向量
    x1, y1 = np.mean(near_edge, axis=0).astype(np.int32)
    xc, yc = center
    div_x, div_y = x1 - xc, y1 - yc
    nc_length = math.sqrt(div_x ** 2 + div_y ** 2)
    unit_vector = np.array([div_x / nc_length, div_y / nc_length])
    # 根据单位向量计算选区角点
    new_edge = near_edge + unit_vector.reshape(1, -1) * line_length
    polygon = np.concatenate([near_edge, new_edge[::-1]], axis=0).astype(np.int32)
    poly_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon], (255,))
    # 掩膜选区
    new_mask = np.full((height, width), False, dtype=bool)
    new_mask[poly_mask == 255] = mask[poly_mask == 255]

    # 以box的宽作为寻找阈值，比较简单，并且一般来说都能找到
    nearest_component = find_nearest_component((x1, y1), new_mask, line_length)
    if nearest_component:
        nearest_label, _, _, centroids = nearest_component
    else:
        return base_point  # 如果没有就返回基本点

    # 如果找到了，就计算镜像点（镜像点基于基本点偏移）
    xa, ya = base_point
    xb, yb = centroids[nearest_label]

    # 根据选区内寻找的连通块，计算镜像点
    theta = math.radians(angle)
    # 计算斜率
    m1 = math.tan(theta)
    m2 = -1.0 / m1  # 垂直直线的斜率
    # 计算交点 x
    numerator = yb - m2 * xb - ya + m1 * xa
    denominator = m1 - m2
    x = numerator / denominator
    # 计算 y
    y = m1 * x + (ya - m1 * xa)
    return round(x), round(y)


def putText_with_padding(img, text, position, desire_width, color=(0, 0, 0), pad_color=(255, 255, 255)):
    """
    在图像上绘制文字，若文字超出图像范围，则自动扩展图像（白色填充）后再绘制。

    返回：
        新图像 (numpy array)
    """
    # 确定文本大小
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    desire_width = max(9, int(desire_width))

    # 创建字体
    try:
        font = ImageFont.truetype("Arial Bold", size=desire_width)
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", size=desire_width)
        except OSError:
            # 如果找不到粗体字体，使用默认字体
            font = ImageFont.load_default(size=desire_width)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position
    x, y = x - round(text_width // 2), y - round(text_height // 2)  # 往左上角偏移一半，因为绘制坐标为左上角坐标
    org_x, org_y = x, y
    pad_top = pad_bottom = pad_left = pad_right = 0

    # 检查是否超出边界
    if x + text_width > img.shape[1]:
        pad_right = x + text_width - img.shape[1]
    if y + text_height > img.shape[0]:
        pad_bottom = y + text_height - img.shape[0]
    if y < 0:
        pad_top = -y
    if x < 0:
        pad_left = -x

    # 如果有超出，则先进行 padding
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        pil_img = ImageOps.expand(pil_img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)
        draw = ImageDraw.Draw(pil_img)
    # 更新绘制位置（考虑 padding）
    new_x = org_x + pad_left
    new_y = org_y + pad_top
    # 绘制文字，由于cv不能改字体，用PIL绘制
    draw.text((new_x, new_y), text, font=font, fill=color)
    return np.array(pil_img)


def get_long_edges(box):
    # box: (4, 2) 的点集，顺序是顺时针或逆时针排列的
    points = list(box)

    # 构建边：(p0, p1), (p1, p2), (p2, p3), (p3, p0)
    edges = []
    for i in range(4):
        p0 = points[i]
        p1 = points[(i + 1) % 4]
        dist = np.linalg.norm(p0 - p1)
        edges.append((dist, p0, p1))

    # 按照边长排序，取最长的两条边
    edges.sort(reverse=True, key=lambda x: x[0])

    # 取最长的两条边
    edge1 = (edges[0][1], edges[0][2])
    edge2 = None

    # 找第二条边，确保它和第一条边不是共享点的邻边
    for i in range(1, 4):
        candidate = (edges[i][1], edges[i][2])
        # 判断是否与 edge1 共享点（即是否相邻）
        shared_points = set(map(tuple, [edge1[0], edge1[1]])) & set(map(tuple, candidate))
        if not shared_points:
            edge2 = candidate
            break

    return edge1, edge2


def get_closest_boundary_point(image, point, angle):
    """
    获取过点A、角度为angle的直线与图像边界的交点中距离A最近的点

    Args:
        image: cv2图像 (numpy array)
        point: 点A的坐标 (x, y)
        angle: 角度，单位为度 (0度为水平向右，逆时针为正)

    Returns:
        tuple: 距离点A最近的交点坐标 (x, y)，如果没有交点返回None
    """
    height, width = image.shape[:2]
    x_A, y_A = point

    # 将角度转换为弧度
    angle_rad = math.radians((angle + 90) % 180)

    # 计算直线的方向向量
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # 存储所有交点
    intersections = []

    # 检查与四条边界的交点

    # 1. 与左边界的交点 (x = 0)
    if dx != 0:
        t = (0 - x_A) / dx
        y = y_A + t * dy
        if 0 <= y <= height:
            intersections.append((0, int(y)))
    # 2. 与右边界的交点 (x = width)
    if dx != 0:
        t = (width - x_A) / dx
        y = y_A + t * dy
        if 0 <= y <= height:
            intersections.append((width, int(y)))
    # 3. 与上边界的交点 (y = 0)
    if dy != 0:
        t = (0 - y_A) / dy
        x = x_A + t * dx
        if 0 <= x <= width:
            intersections.append((int(x), 0))
    # 4. 与下边界的交点 (y = height)
    if dy != 0:
        t = (height - y_A) / dy
        x = x_A + t * dx
        if 0 <= x <= width:
            intersections.append((int(x), height))

    # 计算距离点A最近的交点
    min_distance = float('inf')
    closest_point = None

    for intersection in intersections:
        distance = math.sqrt((intersection[0] - x_A) ** 2 + (intersection[1] - y_A) ** 2)
        if 0 < distance < min_distance:  # 排除点A本身
            min_distance = distance
            closest_point = intersection

    return closest_point


def normalize_rect(rect, box):
    """
    获取长边的真实角度
    通过分析矩形的四个顶点来确定长边方向
    """
    (center, (w, h), angle) = rect
    edges = []
    for i in range(2):  # 只两条边即可确定
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        # 边向量
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        # 边的角度（相对于x轴）
        edge_angle = np.degrees(np.arctan2(edge_vec[1], edge_vec[0]))
        # 标准化角度到[0, 180)
        if edge_angle < 0:
            edge_angle += 180
        edges.append((edge_length, edge_angle))

    # 找到最长的边
    longest_edge = max(edges, key=lambda x: x[0])
    long_edge_length = longest_edge[0]
    long_edge_angle = longest_edge[1]

    # 确定长边和短边
    if abs(long_edge_length - max(w, h)) < 1e-6:
        return center, (max(w, h), min(w, h)), long_edge_angle
    else:
        # 处理浮点精度问题
        return center, (max(w, h), min(w, h)), long_edge_angle


def postprocess(image: np.ndarray, polygon: list[list[int]], class_id: int):
    # cut 和 attach 需要使用取代基符号替换，否则无需后处理
    if class_id not in (0, 1):
        return image

    # 裁剪图像白边
    image, offset_x, offset_y = trim_edges(image)
    trimmed_polygon = [(x - offset_x, y - offset_y) for x, y in polygon]

    # 扩张边界框
    rect = expand_rect(np.array(trimmed_polygon), scale_factor=1.25)
    box = np.int32(cv2.boxPoints(rect))

    # 移除边界框内像素并二值化
    image = erase_polygon_area(image, polygon=box)
    binary_image = binarize_image(image, 0.95)

    # 计算长边中点
    edge1, edge2 = get_long_edges(box)
    x1, y1 = np.mean(edge1, axis=0).astype(np.int32)
    x2, y2 = np.mean(edge2, axis=0).astype(np.int32)

    # 计算离边界角点最近的点，作为文本替换点
    (center, (box_height, box_width), angle) = normalize_rect(rect, box)
    boundary_point = get_closest_boundary_point(image, center, angle)
    dist1 = abs(boundary_point[0] - x1) + abs(boundary_point[1] - y1)
    dist2 = abs(boundary_point[0] - x2) + abs(boundary_point[1] - y2)

    if dist1 < dist2:
        replace_point = find_mirror_point(edge1, (x1, y1), rect, binary_image)
        image = erase_nearest_component(image, binary_image, (x1, y1), box_width * 1.2)
    else:
        replace_point = find_mirror_point(edge2, (x2, y2), rect, binary_image)
        image = erase_nearest_component(image, binary_image, (x2, y2), box_width * 1.2)

    # 填充文本，以 X 为R基团字符（易于识别），以box宽度为大小
    # cv2.polylines(image, [box], True, (115, 115, 115), thickness=1)
    # cv2.circle(image, (x1, y1), 1, (255, 0, 0), thickness=-1)
    # cv2.circle(image, base_point, 1, (0, 255, 0), thickness=-1)
    # cv2.circle(image, replace_point, 1, (0, 0, 255), thickness=-1)
    image = putText_with_padding(image, "X", replace_point, desire_width=box_width * 2)
    return image


def trim_edges(image, pad=3):
    """
    裁剪图像边缘的空白区域，保留指定的填充

    Args:
        image: 输入图像 (numpy array)
        pad: 填充像素数，保留的边缘宽度 (int)

    Returns:
        numpy array: 裁剪后的图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = cv2.boundingRect(thresh)

    x_start, y_start = 0, 0
    if w > 0 and h > 0:
        # 获取原图像尺寸
        img_h, img_w = image.shape[:2]

        # 计算带填充的边界，确保不超出原图像范围
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(img_w, x + w + pad)
        y_end = min(img_h, y + h + pad)

        # 裁剪图像
        image = image[y_start:y_end, x_start:x_end]

    return image, x_start, y_start


if __name__ == '__main__':
    dir_path = r"D:\datasets\R\JMC\2024\crops\structures"
    image_path = os.path.join(dir_path, "JMC_2024_02_1421-1446_03_00_13.png")
    img = cv2.imread(image_path)
    img, x_start, y_start = trim_edges(img)
    binary = binarize_image(img) * 255
    cv2.imshow('test', binary.astype(np.uint8))
    cv2.waitKey(0)
