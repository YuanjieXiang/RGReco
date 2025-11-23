import csv
import json
import os
from collections import Counter
from typing import Tuple

from rdkit import Chem

from src.utils.pipeline_processing import *

log = logging.getLogger(__name__)


def extract_group_symbols(smiles_with_groups: str) -> Tuple[str, List[str]]:
    """
    从带基团符号的SMILES中提取基团信息

    Args:
        smiles_with_groups: 带基团符号的SMILES字符串

    Returns:
        tuple: (纯SMILES字符串, 基团符号列表)
    """
    # 使用正则表达式匹配 |$...$| 模式
    pattern = r'\|\$([^$]*)\$\|'
    match = re.search(pattern, smiles_with_groups)

    if match:
        # 提取基团符号部分
        group_part = match.group(1)
        # 分割基团符号（以分号分隔）
        group_symbols = [g.strip() for g in group_part.split(';') if g.strip()]

        # 移除基团符号部分，得到纯SMILES
        pure_smiles = re.sub(pattern, '', smiles_with_groups).strip()
        if len(group_symbols) == 1:
            group_symbols = []

        return pure_smiles, group_symbols
    else:
        # 没有基团符号
        return smiles_with_groups.strip(), []


def process_smiles_list(smiles_list, isomeric=True):
    """
    标准化 SMILES 列表：去重、去无效、转为 Cano-SMILES
    """
    canonical_set = set()
    valid_smiles = []

    for s in smiles_list:
        try:
            mol = Chem.MolFromSmiles(s.strip())
            if mol is not None:
                # 生成规范 SMILES（保留立体化学）
                cano_smi = Chem.MolToSmiles(mol, isomericSmiles=isomeric)

                pure_smi, group_symbols = extract_group_symbols(s)
                tail = ''.join(sorted(group_symbols))

                if cano_smi not in canonical_set:
                    canonical_set.add(cano_smi)
                    valid_smiles.append(cano_smi + tail)
        except:
            continue  # 跳过无效

    return valid_smiles   # 返回无重复、有效的规范 SMILES 列表


def evaluate_smiles(pred_smiles, true_smiles):
    """
    评估单张图像的识别结果
    """
    # 处理 SMILES 列表（如去空格、标准化等）
    pred_list = process_smiles_list(pred_smiles)
    true_list = process_smiles_list(true_smiles)

    # 使用 Counter 统计频次
    pred_counter = Counter(pred_list)
    true_counter = Counter(true_list)

    # 计算 TP: 每个 item 的最小频次之和（交集）
    tp = sum(min(pred_counter[k], true_counter[k]) for k in (pred_counter & true_counter))

    # 计算 FP: 预测多出的部分
    fp = sum(max(0, pred_counter[k] - true_counter.get(k, 0)) for k in pred_counter)

    # 计算 FN: 真实漏掉的部分
    fn = sum(max(0, true_counter[k] - pred_counter.get(k, 0)) for k in true_counter)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'num': len(pred_smiles)
    }


def load_results(label_dir: str, image_dir: str):
    names = [name[:-4] for name in os.listdir(image_dir) if name.endswith('png')]

    results = []
    for name in names:
        json_path = os.path.join(label_dir, f"{name}.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cmpds = data.get("compounds", {}).get('cmpds', [])
            if not cmpds:
                print(f"未正确完成：{name}")
        except Exception:
            results.append([])
            continue
        smiles_list = [cmpd['smiles'] for cmpd in cmpds]
        results.append({'name': name, 'smiles_list': smiles_list})
    return results


def total_eval():
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_predictions = load_results(pred_path, image_path)
    all_ground_truths = load_results(gold_path, image_path)

    # 用于保存每轮结果
    per_image_results = []

    for idx, (img_pred, img_true) in enumerate(zip(all_predictions, all_ground_truths)):
        result = evaluate_smiles(img_pred['smiles_list'], img_true['smiles_list'])
        tp, fp, fn = result['tp'], result['fp'], result['fn']

        # 累加
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 保存本轮结果（假设 img_pred / img_true 是 dict 且包含 '图像名' 字段）
        image_name = img_pred['name']
        per_image_results.append({
            '图像名': image_name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        })

    # 保存本轮所有图像的详细结果到 CSV（使用全局 i 命名）
    output_file = f"eval_result_{i}.csv"
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = ['图像名', 'TP', 'FP', 'FN', 'Precision', 'Recall']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_results)

    # print(f"📊 评估结果已保存到: {output_file}")

    # 计算全局指标
    global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (
                                                                                                     global_precision + global_recall) > 0 else 0.0

    return global_precision, global_recall, global_f1


if __name__ == '__main__':
    gold_path = "../evaluate_dataset/gold"
    image_path = "../evaluate_dataset/images"
    print('epcho     Precision     Recall    F1\n')
    for i in range(6):
        pred_path = os.path.join("../result", f'{i}')
        precision, recall, f1 = total_eval()
        print(f'{i}         {precision:.3f}      {recall:.3f}       {f1:.3f}\n')