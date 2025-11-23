import cv2
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from typing import Union
from pathlib import Path
from src.utils.image_loader import ImageLoader
from src.postprocessing.rgroup.schema import RGroup

InputType = Union[str, np.ndarray, bytes, Path]

# 使用智能换行算法
wrapper = textwrap.TextWrapper(
    width=60,
    break_long_words=False,  # 不截断长单词
    break_on_hyphens=False,  # 不在连字符处断开
    subsequent_indent=''  # 后续行不缩进
)


def visualize_detection_segmentation(image, result):
    """
        Visualize the combined results of object detection and instance segmentation.

        This function overlays detection bounding boxes and segmentation masks
        on the original image, using distinct colors for different instances.
        It is useful for qualitative analysis of model outputs, especially
        when evaluating correct and error cases in multi-stage workflows.

        Args:
            image (ndarray): The original input image.
            result (PipelineResult): The RGreco output result.
    """
    # 分子分割可视化，此步骤考虑用不同颜色表示主结构与取代基结构
    structures = result.mol_res.get('structures', [])
    for struct in structures:
        if struct['is_candidate']:
            color = (173, 112, 147)  # 深紫
        else:
            color = (155, 108, 219) # 粉红
        for c in range(3):  # 对 RGB 通道分别处理
            image[:, :, c] = np.where(
                struct['mask'] == True,  # 掩膜条件
                image[:, :, c] * 0.5 + color[c] * 0.5,  # 半透明叠加
                image[:, :, c]  # 保留原图
            )

        # x1, y1, x2, y2 = struct['bbox']
        # cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # cut_bond_det 可视化
        for cut in struct.get('cut_det', []):
            image = _draw_polygon_overlay(image, cut['polygon'], (35, 23, 253), alpha=0, thickness=1)  # 红色

    # 表格检测可视化
    table_box = result.table_res.get('table_box', {})
    if table_box:
        image = _draw_polygon_overlay(image, table_box['polygon'], (245, 219, 159), alpha=0.1)  # 蓝色

    # 文本检测可视化
    text_lines = result.text_res.get('text_lines', [])
    if text_lines:
        for text_line in text_lines:
            image = _draw_polygon_overlay(image, text_line.polygon, (95, 202, 253), thickness=1)  # 黄色

    return image


def _draw_polygon_overlay(image, polygon, color=(0, 255, 0), alpha=0.5, thickness=2):
    """
    Draw a polygon on a CV image with optional semi-transparent filling.

    Args:
        image (np.ndarray): The input image (BGR format, as used in OpenCV).
        polygon (np.ndarray or list): Polygon vertices of shape (N, 2), e.g. [[x1, y1], [x2, y2], ...].
        color (tuple): Border color in BGR format, e.g. (0, 255, 0) for green.
        alpha (float): Fill transparency (0~1).
                       0 = fully transparent (no fill),
                       1 = fully opaque fill,
                       >1 or 100% = no fill at all (only border).
        thickness (int): Border line thickness.

    Returns:
        np.ndarray: Image with the polygon overlay drawn.
    """
    # Make a copy to avoid modifying the original image
    overlay = image.copy()
    output = image.copy()

    # Ensure polygon is numpy array of correct shape
    polygon = np.array(polygon, dtype=np.float64)
    polygon = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))

    # Draw filled area if alpha < 1
    if alpha < 1.0:
        cv2.fillPoly(overlay, [polygon], color)
        # Blend filled region with original image
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Draw border (always visible)
    cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=thickness)

    return output


def mol_visualize(mol, rgroup_dict: dict[str: RGroup] = None, legend: str = "",
                  highlight_atoms: list = None, black=True) -> Image.Image:
    """根据输入的mol绘制分子，如果给出分子中存在的R基团，也会在下面以文本形式绘制。

    Args:
        mol (Mol): RDKit中的Mol。
        rgroup_dict (dict[str: RGroup], optional): R基团相关信息. Defaults to {}. e.g. {"R": RGroup(abbr: '', smile='[H]'}。
        legend(str): 分子的图例，一般是分子编号。
        highlight_atoms: 需要高亮的原子编号。
        black (): 是否使用黑白配色

    Returns:
        np.ndarray: 绘制出的图像
    """
    if not isinstance(mol, Chem.Mol):
        mol = Chem.MolFromSmiles('C')
        legend += " ILLEGAL!"

    mol = Chem.Mol(mol)

    if rgroup_dict is None:
        rgroup_dict = {}
    if highlight_atoms is None:
        highlight_atoms = []

    # 计算需要高亮的键
    hit_bonds = []
    if highlight_atoms and len(highlight_atoms) > 1:
        for bond in mol.GetBonds():
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            if begin_atom_idx in highlight_atoms and end_atom_idx in highlight_atoms:
                hit_bonds.append(bond.GetIdx())

    # 修改R基团符号，记录脚注信息
    map_num_cnt = 1
    atoms = mol.GetAtoms()
    for i, atom in enumerate(atoms):
        # 设置编号
        # atom.SetAtomMapNum(map_num_cnt)
        # 获取脚注信息
        mol_file_alias = atom.GetProp('molFileAlias') if atom.HasProp("molFileAlias") else None
        # mol_file_alias = atom.GetProp('dummyLabel') if atom.HasProp("dummyLabel") else None
        if mol_file_alias:
            # atom.SetProp("_displayLabel", mol_file_alias)
            # map_num_cnt += 1

            # 高亮未展开的R基团
            if i not in highlight_atoms:
                highlight_atoms.append(i)

    # 确保所有原子位置合理
    AllChem.Compute2DCoords(mol)

    draw_options = Draw.rdMolDraw2D.MolDrawOptions()
    if black:
        draw_options.useBWAtomPalette()  # 启用黑白模式
    img = Draw.MolToImage(mol, size=(400, 300), legend=legend, kekulize=True,
                          highlightAtoms=highlight_atoms, highlightBonds=hit_bonds, options=draw_options)

    # 创建新画布，扩展底部以添加脚注
    new_height = img.height + 100
    new_img = Image.new('RGB', (img.width, new_height), (255, 255, 255))
    new_img.paste(img, (0, 0))

    # 设置脚注字体
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)  # 使用Arial字体，size14时每个字符约占11*11的像素
    except IOError:
        font = ImageFont.load_default()  # 回退到默认字体

    # 绘制脚注内容，智能换行不切断单词
    text_y = img.height - 16  # 行数超过6时，y起点为0，每行16像素，高100时约6行

    # 生成完整脚注文本
    footer_lines = []
    if rgroup_dict:
        rgroups = list(rgroup_dict.items())
        # 构建自然分隔的文本
        footer_text = " ".join(
            [f"{k}: '{v}'," for k, v in rgroups[:-1]] +
            [f"{rgroups[-1][0]}: '{rgroups[-1][1]}'"] if rgroups else []
        )
        # 动态计算绘制y的起点，使文本垂直居中
        line_num = len(footer_text) // 60
        if line_num < 6:
            text_y += (6 - line_num) * 8  # 8为半行高度
        footer_lines = wrapper.wrap(footer_text)

    # 绘制处理后的文本
    for line in footer_lines:
        line = line.strip().rstrip(',')
        bbox = draw.textbbox((0, 0), line, anchor="lt")  # anchor="lt"表示左上角对齐
        width = int(bbox[2] - bbox[0])  # right - left
        text_x = max(0, (300 - width) // 2)  # 动态计算x起始点，使文本水平居中
        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
        text_y += 16

    return new_img


def ocr_visualize(image: InputType, ocr_results: list):
    """OCR识别结果绘制

    Args:
        image (InputType): 图像或图像路径
        ocr_results (list): OCR识别结果
        [
            [[[351.0, 259.0], [363.0, 259.0], [363.0, 272.0], [351.0, 272.0]], '8', 0.9],
            ...
        ]
    Returns:
        _type_: _description_
    """
    image_loader = ImageLoader()
    image = image_loader.load_image(image)

    draw = ImageDraw.Draw(image)

    # 加载字体（根据需要调整字体路径）
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()

    # 遍历OCR结果并绘制边界框和文本
    for result in ocr_results:
        box = result[0]
        if len(box) != 4:
            continue

        text = result[1]
        confidence = result[2]

        # 绘制边界框
        draw.polygon([(box[0][0], box[0][1]), (box[1][0], box[1][1]),
                      (box[2][0], box[2][1]), (box[3][0], box[3][1])], outline="red")

        # 在边界框上方绘制文本
        draw.text((box[0][0], box[0][1] - 15), text, fill="blue", font=font)
    return image


def _plot_rec_box_with_logic_info(img, logic_points, sorted_polygons):
    """
    :param img
    :param logic_points: [row_start,row_end,col_start,col_end]
    :param sorted_polygons: [xmin,ymin,xmax,ymax]
    :return:
    """
    # 读取原图
    image_loader = ImageLoader()
    img = image_loader.load_cv_img(img)
    img = cv2.copyMakeBorder(
        img, 0, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    # 绘制 polygons 矩形
    for idx, polygon in enumerate(sorted_polygons):
        x0, y0, x1, y1 = polygon[0], polygon[1], polygon[2], polygon[3]
        x0 = round(x0)
        y0 = round(y0)
        x1 = round(x1)
        y1 = round(y1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
        # 增大字体大小和线宽
        font_scale = 0.9  # 原先是0.5
        thickness = 1  # 原先是1
        logic_point = logic_points[idx]
        cv2.putText(
            img,
            f"row: {logic_point[0]}-{logic_point[1]}",
            (x0 + 3, y0 + 8),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
        cv2.putText(
            img,
            f"col: {logic_point[2]}-{logic_point[3]}",
            (x0 + 3, y0 + 18),
            cv2.FONT_HERSHEY_PLAIN,
            font_scale,
            (0, 0, 255),
            thickness,
        )
    return img


def _draw_rectangle(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    img_copy = img.copy()
    for box in boxes.astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img_copy


def _draw_polylines(img: np.ndarray, points) -> np.ndarray:
    img_copy = img.copy()
    for point in points.astype(int):
        point = point.reshape(4, 2)
        cv2.polylines(img_copy, [point.astype(int)], True, (255, 0, 0), 2)
    return img_copy
