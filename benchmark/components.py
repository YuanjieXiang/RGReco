import json
import os
from typing import Dict, Tuple

import tqdm
from rdkit import Chem
from shapely import Polygon

from overall import evaluate_smiles
from src.external.decimer_segmentation import complete_structure_mask as mask_postprocess
from src.utils.box_processing import calculate_iou, convert_polygon_to_bbox, convert_polygon_to_mask
from src.utils.image_loader import ImageLoader
from src.utils.pipeline_processing import *

log = logging.getLogger(__name__)


def group_smi_to_mol(smiles: str):
    mol = None
    if '*' not in smiles and smiles.count('[') == 1 and smiles.count(']') == 1 and smiles.index(']') - smiles.index(
            '[') == 2:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        # 对于转换错误的smi，先添加虚拟原子再转换
        smiles = smiles.replace('[', '').replace(']', '(*)')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 要修改分子，需转为可编辑形式
    mol = Chem.RWMol(mol)

    # 如果有虚拟原子
    if '*' in smiles:
        # 2. 找到虚拟原子 * 和它连接的原子
        dummy_atom_idx = None
        neighbor_atom_idx = None

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # 虚拟原子 [*]
                dummy_atom_idx = atom.GetIdx()
                # 获取连接的邻居（应该是氮）
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 1:
                    neighbor_atom_idx = neighbors[0].GetIdx()
        if neighbor_atom_idx is None:
            return None
        # 3. 修改自由价，删除虚拟原子
        atom_l = mol.GetAtomWithIdx(neighbor_atom_idx)
        atom_l.SetNumRadicalElectrons(atom_l.GetNumRadicalElectrons() + 1)
        mol.RemoveAtom(dummy_atom_idx)

        # 4. 移除虚拟原子后，生成最终分子
        mol = mol.GetMol()
    return mol


def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算Dice系数（也称为F1-score），是常用的掩膜相似度指标

    Args:
        mask1: 第一个掩膜
        mask2: 第二个掩膜

    Returns:
        Dice系数，范围[0, 1]
    """
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)

    if mask1.shape != mask2.shape:
        raise ValueError(f"掩膜尺寸不匹配: {mask1.shape} vs {mask2.shape}")

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 1.0

    return 2.0 * intersection / total


def normalize_cmpd(cmpd: Dict, norm_smi=True) -> Tuple:
    """
    将化合物标准化为可哈希的元组，用于比较和去重
    通过修改是否使用smiles，可分析取代基抽取准确率

    Args:
        norm_smi : 是否对比smiles
        cmpd: 单个化合物字典

    Returns:
        标准化后的元组表示
    """
    cmpd_id = cmpd.get('cmpd_id', '')

    # 将r_groups转换为排序后的元组
    r_groups = cmpd.get('r_groups', {})
    r_groups_items = []

    for r_key in sorted(r_groups.keys()):
        r_value = r_groups[r_key]
        abbr = r_value.get('abbr', '')
        if norm_smi:
            smiles = r_value.get('smiles', '')
            smiles = Chem.MolToSmiles(group_smi_to_mol(smiles))  # 统一SMILES格式
            r_groups_items.append((r_key, abbr, smiles))
        else:
            r_groups_items.append((r_key, abbr))

    return (cmpd_id, tuple(r_groups_items))


def calculate_doc_metrics(pred_cmpds: List[Dict], gold_cmpds: List[Dict], calculate_smi: bool) -> Dict[str, float]:
    """
    计算预测化合物列表相对于金标准列表的评估指标

    Args:
        pred_cmpds: 预测的化合物列表
        gold_cmpds: 金标准化合物列表

    Returns:
        包含precision, recall, f1的字典
    """
    # 标准化化合物为集合
    pred_set = set(normalize_cmpd(cmpd, calculate_smi) for cmpd in pred_cmpds)
    gold_set = set(normalize_cmpd(cmpd, calculate_smi) for cmpd in gold_cmpds)

    return general_compute(pred_set, gold_set)


def calculate_polygon_iou(poly1, poly2):
    """
    poly1, poly2: list of points like [[x1,y1], [x2,y2], ...]
    返回 IoU 值（0~1）
    """
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)

        if not p1.is_valid or not p2.is_valid:
            return 0.0

        intersection = p1.intersection(p2).area
        union = p1.union(p2).area

        if union == 0:
            return 0.0
        return intersection / union
    except Exception as e:
        print(f"IoU 计算错误: {e}")
        return 0.0


def general_compute(pred_set, true_set):
    tp = len(pred_set & true_set)  # 正确识别
    fp = len(pred_set - true_set)  # 误识别
    fn = len(true_set - pred_set)  # 漏识别

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    jaccard = len(pred_set & true_set) / len(pred_set | true_set) if len(pred_set | true_set) > 0 else 0.0

    # 是否完全正确（Set Accuracy）
    set_acc = 1 if pred_set == true_set else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'set_acc': set_acc,
        'tp': tp, 'fp': fp, 'fn': fn
    }


def polygon_compute(predictions, ground_truths, iou_thresh=0.8):
    """
    计算多边形预测的 TP, FP, FN, 并可扩展 Precision/Recall/F1
    predictions: list of polygons, each polygon is a list of [x, y] points
    ground_truths: same format
    iou_thresh: 匹配所需的最小 IoU（常用 0.5）
    """

    if len(ground_truths) == 0:
        # 没有真实多边形 → 所有预测都是 FP
        return {
            'tp': 0,
            'fp': len(predictions),
            'fn': 0,
        }

    if len(predictions) == 0:
        # 没有预测 → 全部是 FN
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(ground_truths),
        }

    # 标记哪些真实多边形已被匹配
    matched_gt = [False] * len(ground_truths)

    tp = 0
    fp = 0

    # 可选：按置信度排序（如果你的 predictions 包含 score）
    # 假设 predictions 是 [(poly, score), ...] 或 dict 形式
    # 这里假设输入是纯坐标列表，不带 score
    # 如果有 score，请先排序：predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    for pred_poly in predictions:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt_poly in enumerate(ground_truths):
            iou = calculate_polygon_iou(pred_poly, gt_poly)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        # 检查是否匹配成功
        if best_iou >= iou_thresh and not matched_gt[best_gt_idx]:
            tp += 1
            matched_gt[best_gt_idx] = True  # 一个 gt 只能匹配一次
        else:
            fp += 1

    fn = len(ground_truths) - sum(matched_gt)  # 未被匹配的真实多边形数量

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_gt': matched_gt
    }


def box_compute(predictions, ground_truths, iou_thresh=0.5):
    """
    计算指定类别的 Precision, Recall, F1
    predictions: list of [x1, y1, x2, y2]
    ground_truths: list of [x1, y1, x2, y2]
    iou_thresh: 判断匹配的 IoU 阈值（默认 0.5）
    """
    matched_gt = [False] * len(ground_truths)  # 标记哪些真实框已被匹配
    if len(ground_truths) == 0:
        # 没有真实框 → 所有预测都是 FP
        tp = 0
        fp = len(predictions)
        fn = 0
    else:
        # 按置信度从高到低排序预测框
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

        tp = 0
        fp = 0

        for pred in predictions:
            x1, y1, x2, y2 = pred
            pred_box = [x1, y1, x2, y2]
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(ground_truths):
                gx1, gy1, gx2, gy2 = gt
                gt_box = [gx1, gy1, gx2, gy2]
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_thresh and not matched_gt[best_gt_idx]:
                tp += 1
                matched_gt[best_gt_idx] = True  # 一个 gt 只能匹配一个 pred
            else:
                fp += 1

        fn = len(ground_truths) - sum(matched_gt)  # 未被匹配的真实框

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'matched_gt': matched_gt
    }


def load_results(label_dir: str, image_dir: str):
    names = [name[:-4] for name in os.listdir(image_dir) if name.endswith('png')]

    results = []
    for name in names:
        json_path = os.path.join(label_dir, f"{name}.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            results.append({})
            continue
        results.append(data)
    return results


def evaluate_mol_seg(res_pred, res_true, image=None):
    pred_structures = res_pred.get('mol_res', {}).get('structures', [])
    gold_structures = res_true.get('mol_res', {}).get('structures', [])
    pred_polygons = [s['polygon'] for s in pred_structures]
    gold_polygons = [s['polygon'] for s in gold_structures]
    res = polygon_compute(pred_polygons, gold_polygons)

    if isinstance(image, np.ndarray):
        image_size = image.shape[:2]
        pred_masks = [mask_postprocess(image, convert_polygon_to_mask(p, image_size)[:, :, None], image_size).squeeze()
                      for p in pred_polygons]
        gold_masks = [mask_postprocess(image, convert_polygon_to_mask(p, image_size)[:, :, None], image_size).squeeze()
                      for p in gold_polygons]
        [s.update(mask=m) for s, m in zip(pred_structures, pred_masks)]
        [s.update(mask=m) for s, m in zip(gold_structures, gold_masks)]
        res_true['masks'] = gold_masks
        # 使用逻辑OR：任何位置有实例就为前景
        pred_mask = np.any(pred_masks, axis=0).astype(np.uint8)
        gold_mask = np.any(gold_masks, axis=0).astype(np.uint8)

        dice = dice_coefficient(pred_mask, gold_mask)
        res['dice'] = dice

    return res


def evaluate_r_group_detect(res_pred, res_true):
    pred_structures = res_pred.get('mol_res', {}).get('structures', [])
    gold_structures = res_true.get('mol_res', {}).get('structures', [])
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for ps, gs in zip(pred_structures, gold_structures):
        pred_cut_det = ps.get('cut_det', [])
        gold_cut_det = gs.get('cut_det', [])

        if pred_cut_det and gold_cut_det:
            pred_boxes = [convert_polygon_to_bbox(s['polygon']) for s in pred_cut_det]
            gold_boxes = [convert_polygon_to_bbox(s['polygon']) for s in gold_cut_det]
            result = box_compute(pred_boxes, gold_boxes)

            # print(f"Result: {result}, \npred_boxes: {pred_boxes}, \ngold_boxes: {gold_boxes}")
            total_tp += result['tp']
            total_fp += result['fp']
            total_fn += result['fn']
    return {'tp': total_tp, 'fp': total_fp, 'fn': total_fn}


def evaluate_mol_rec(res_pred, res_true):
    pred_structures = res_pred.get('mol_res', {}).get('structures', [])
    gold_structures = res_true.get('mol_res', {}).get('structures', [])
    pred_smiles = [s['smiles'] for s in pred_structures]
    is_candidates = [s['is_candidate'] for s in pred_structures]
    gold_smiles = [s['smiles'] for s in gold_structures]
    return evaluate_smiles(pred_smiles, gold_smiles)


def evaluate_doc_parsing(res_pred, res_true):
    """文本识别错误"""
    pred_lines = res_pred.get('compounds', {}).get('cmpds', [])
    gold_lines = res_true.get('compounds', {}).get('cmpds', [])
    metrics = calculate_doc_metrics(gold_lines, pred_lines, True)
    return metrics


def evaluate_doc_rec(res_pred, res_true):
    """文本识别错误"""
    pred_lines = res_pred.get('compounds', {}).get('cmpds', [])
    gold_lines = res_true.get('compounds', {}).get('cmpds', [])
    metrics = calculate_doc_metrics(gold_lines, pred_lines, False)
    return metrics


def evaluate_r_group_text_rec(res_pred, res_true):
    pred_cmpds = res_pred.get('compounds', {}).get('cmpds', [])
    gold_cmpds = res_true.get('compounds', {}).get('cmpds', [])
    pred_smiles, gold_smiles = [], []
    for s in pred_cmpds:
        smis = [Chem.MolToSmiles(group_smi_to_mol(group['smiles'])) for group in s['r_groups'].values() if '|$' not in group['smiles']]
        pred_smiles.extend(smis)
    for s in gold_cmpds:
        smis = [Chem.MolToSmiles(group_smi_to_mol(group['smiles'])) for group in s['r_groups'].values() if '|$' not in group['smiles']]
        gold_smiles.extend(smis)
    return evaluate_smiles(pred_smiles, gold_smiles)


def total_eval(eval_func, pred_path):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_predictions = load_results(pred_path, image_path)
    all_ground_truths = load_results(gold_path, image_path)
    names = [name for name in os.listdir(image_path) if name.endswith('.png')]

    for img_pred, img_true, name in tqdm.tqdm(zip(all_predictions, all_ground_truths, names), total=len(names)):
        # 获取该图像的 TP, FP, FN
        result = eval_func(img_pred, img_true)
        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']

    # 全局 Precision, Recall, F1（基于所有化合物）
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (
                                                                                                     global_precision + global_recall) > 0 else 0.0

    print(f"Global Precision: {global_precision:.3f}")
    print(f"Global Recall:    {global_recall:.3f}")
    print(f"Global F1:        {global_f1:.3f}")


if __name__ == '__main__':
    gold_path = "../evaluate_dataset/gold"
    image_path = "../evaluate_dataset/images"
    image_loader = ImageLoader()

    # eval mol_segmentation
    print("eval mol_segmentation")
    total_eval(eval_func=evaluate_mol_seg, pred_path="../result/0")

    # eval r_group_detection
    print("eval r_group_detection")
    total_eval(eval_func=evaluate_r_group_detect, pred_path="../result/1")

    # eval mol_recognition
    print("eval mol_recognition")
    total_eval(eval_func=evaluate_mol_rec, pred_path="../result/1")

    # eval doc_rec
    print("eval doc_rec")
    total_eval(eval_func=evaluate_doc_rec, pred_path="../result/4")

    # eval doc_parsing
    print("eval doc_parsing")
    total_eval(eval_func=evaluate_doc_parsing, pred_path="../result/4")

    # eval r_group_text_rec
    print("eval r_group_text_rec")
    total_eval(eval_func=evaluate_r_group_text_rec, pred_path="../result/5")
