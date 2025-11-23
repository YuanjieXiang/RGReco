import logging
import re
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.postprocessing.rgroup import rmol, rannotation
from src.postprocessing.rgroup.schema import Compound, RGroupAbbr
from src.external.surya.recognition.schema import TextLine

log = logging.getLogger(__name__)



def replace_mol_to_text_in_table(image: Image.Image, table_box, bboxes: List):
    """
    将图像中的表格区域内识别出中心点完全位于表格内的分子结构替换为文本 'MOL'

    参数:
        image (Image.Image): PIL 图像，是根据table_box裁剪下来的表格图像
        table_box (list): 表格边界 [x1, y1, x2, y2]
        bboxes (List[list]): 所有文本检测框列表，每个框格式为 [x1, y1, x2, y2]

    返回:
        np.ndarray: 修改后的图像
    """
    # 复制图像以便不影响原图
    # 表格区域坐标

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # 使用Arial字体，size14时每个字符约占11*11的像素
    except IOError:
        font = ImageFont.load_default()  # 回退到默认字体
    t_x1, t_y1, t_x2, t_y2 = table_box

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        text_x = x1 + int(w * 0.25)
        text_y = y1 + int(h * 0.5)

        # 判断是否在表格内
        if t_x1 <= text_x <= t_x2 and t_y1 <= text_y <= t_y2:
            # 直接在原图坐标绘制
            draw.text((text_x, text_y), 'MOL', fill=(0, 0, 0), font=font)

    return image


def get_mol_masks_in_table(table_box, mol_res):
    t_x1, t_y1, t_x2, t_y2 = table_box
    # 设置文本参数
    masks = []
    for item in mol_res:
        bbox = item['bbox']
        mask = item['mask']
        y1, x1, y2, x2 = bbox
        # 插入文本的位置：bbox 左上角 四分之一处
        w = x2 - x1
        h = y2 - y1
        text_x = x1 + int(w * 0.25)
        text_y = y1 + int(h * 0.5)
        # 判断四分之一处是否在 table_box 内部
        if t_x1 <= text_x <= t_x2 and t_y1 <= text_y <= t_y2:
            masks.append(mask)
    if not masks:
        return np.array([])
    masks = np.stack(masks, axis=2)
    return masks


def merge_cmpds(table_cmpds: List[rannotation.Compound], label_cmpds: List[rannotation.Compound]):
    # 4种情况，两者都没有，两者都有，或者只有其中之一，
    if not table_cmpds and not label_cmpds:
        return []
    elif not label_cmpds:
        return table_cmpds
    elif not table_cmpds:
        return label_cmpds
    else:  # 对于两者都有的情况，又要分两者cmpd_id相同和不同的情况（一般来说是不同的，需要互补）
        if len(label_cmpds) < len(table_cmpds):  # 大多数情况，一个label对应多个table_id
            if len(label_cmpds) == 1 and label_cmpds[0].cmpd_id == '-':  # 一对多
                r_groups = label_cmpds[0].r_groups
                for cmpd in table_cmpds:
                    for k, v in r_groups.items():
                        cmpd.r_groups.setdefault(k, v)
            else:  # 多对多
                for label_cmpd in label_cmpds:
                    label_cmpd_id = label_cmpd.cmpd_id
                    r_groups = label_cmpd.r_groups
                    for table_cmpd in table_cmpds:
                        if label_cmpd_id in table_cmpd.cmpd_id:
                            for k, v in r_groups.items():
                                table_cmpd.r_groups.setdefault(k, v)
        else:  # 不应该有的情况，没人会让label中的cmpd数多于table中的cmpd
            log.warning(f"label_cmpds '{label_cmpds}' 数量多于 table_cmpds {table_cmpds}。")
            pass
    return table_cmpds


def split_mol_result(mol_rec_res):
    candidate_mol_res, normal_mol_res, bad_mol_res = [], [], []
    for i, structure in enumerate(mol_rec_res):
        detections = structure.get("cut_det", [])

        smiles = structure['smiles']
        mol = rmol.convert_smiles_to_mol(smiles)
        if mol is None:
            normal_mol_res.append(structure)
            continue

        atoms = mol.GetAtoms()
        if len(atoms) <= 2:
            normal_mol_res.append(structure)
            continue

        if not candidate_mol_res and not detections:
            r_syms, unknown_syms = rmol.get_rsymbols_from_mol(mol)
            if not r_syms and unknown_syms:
                bad_mol_res.append(structure)
                continue

            if r_syms:
                candidate_mol_res.append(structure)
            else:
                normal_mol_res.append(structure)
        else:
            normal_mol_res.append(structure)

    return bad_mol_res, normal_mol_res, candidate_mol_res


def filter_structures_by_box(structures, box):
    """
    过滤 structures 中 polygon 完全在 box 内的项。

    :param structures: list of dict, 每个 dict 包含 'polygon' 键
    :param box: list or tuple, 表示 [x_min, y_min, x_max, y_max]
    :return: 新的结构数组
    """
    x_min, y_min, x_max, y_max = box
    table_structures, other_structures = [], []

    for struct in structures:
        polygon = struct.get('polygon', [])
        is_candidate = struct.get('is_candidate', [])
        if is_candidate:
            continue
        inside = True

        for (x, y) in polygon:
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                inside = False
                break

        if inside:
            table_structures.append(struct)
        else:
            other_structures.append(struct)

    return table_structures, other_structures


def match_text_and_struct(r_sym: str, text_lines: list[TextLine], structures_lines: list[TextLine], structures,
                          max_distance=25):
    cmpds = []

    for a in structures_lines:
        a_x1, a_y1, a_x2, a_y2 = a.bbox
        a_bottom = a_y2  # a的最下端

        best_match = None
        min_distance = float('inf')

        for b in text_lines:
            b_x1, b_y1, b_x2, b_y2 = b.bbox
            b_top = b_y1  # b的最上端

            # 检查条件1: 垂直距离
            vertical_distance = b_top - a_bottom
            if a_y1 > b_y1 or vertical_distance > max_distance:
                continue  # b不在a下方，或距离太远
            # 检查条件2: b的左边界在a的左边界右侧
            if b_x1 <= a_x1:
                continue
            # 检查条件3: b的右边界在a的右边界左侧
            if b_x2 >= a_x2:
                continue

            # 如果满足所有条件，选择距离最近的b
            if vertical_distance < min_distance:
                min_distance = vertical_distance
                best_match = b

        # 如果找到匹配的b，添加到结果中
        if best_match is not None:
            idx = int(a.text[3:])
            group = RGroupAbbr(abbr='-')
            group._smiles = structures[idx]['smiles']
            cmpd = Compound(r_groups={r_sym: group}, cmpd_id=best_match.text)
            cmpds.append(cmpd)

    return cmpds


def natural_sort_key(text):
    """
    将字符串按数字和非数字部分分割，数字部分转为整数
    例如: 'cmpd10' -> ['cmpd', 10]
    """
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', text)]
