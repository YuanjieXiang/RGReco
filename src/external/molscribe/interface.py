# Copyright (c) 2022 Yujie Qian.
# This code is licensed under the MIT License.
# See the LICENSE file in the project root for full license details.
#
# Modifications made by yuanjie.
# Changes include: Added a substituent symbol detection model to fix misrecognition of R-Group identifier
#
# Original code at: https://github.com/thomas0809/MolScribe

import argparse
from typing import List

import cv2
import torch

from .chemistry import convert_graph_to_smiles
from .dataset import get_transforms
from .model import Encoder, Decoder
from .tokenizer import get_tokenizer

R_DET_SYMBOLS = ['attach', 'cut', 'star', 'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11',
                 'R12', 'Ra', 'Rb', 'Rc', 'Rd', "R'", "R''", "R'''", "R1'", "R2'", "R3'",
                 'X', 'Y', 'Z', 'Q', 'Ar']
VALID_SYMBOLS = [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 20, 21, 22] # 经实验，替换正确率较高的符号，RO可能被识别为R9、
BOND_TYPES = ["", "single", "double", "triple", "aromatic", "solid wedge", "dashed wedge"]


def safe_load(module, module_states):
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    return


class MolScribe:

    def __init__(self, model_path, device=None, num_workers=1):
        """
        MolScribe Interface
        :param model_path: path of the model checkpoint.
        :param device: torch device, defaults to be CPU.
        :param multiprocessing_enabled: uses multiprocessing to parallelize parts of the inference when enabled, defaults to False.
        """
        model_states = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        args = self._get_args(model_states['args'])
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.tokenizer = get_tokenizer(args)
        self.encoder, self.decoder = self._get_model(args, self.tokenizer, self.device, model_states)
        self.transform = get_transforms(args.input_size, augment=False)
        self.num_workers = num_workers

    def _get_args(self, args_states=None):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--encoder', type=str, default='swin_base')
        parser.add_argument('--decoder', type=str, default='transformer')
        parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
        parser.add_argument('--no_pretrained', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true', default=True)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--embed_dim', type=int, default=256)
        parser.add_argument('--enc_pos_emb', action='store_true')
        group = parser.add_argument_group("transformer_options")
        group.add_argument("--dec_num_layers", help="No. of layers in transformer decoder", type=int, default=6)
        group.add_argument("--dec_hidden_size", help="Decoder hidden size", type=int, default=256)
        group.add_argument("--dec_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
        group.add_argument("--dec_num_queries", type=int, default=128)
        group.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
        group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
        group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
        parser.add_argument('--continuous_coords', action='store_true')
        parser.add_argument('--compute_confidence', action='store_true')
        # Data
        parser.add_argument('--input_size', type=int, default=384)
        parser.add_argument('--vocab_file', type=str, default=None)
        parser.add_argument('--coord_bins', type=int, default=64)
        parser.add_argument('--sep_xy', action='store_true', default=True)

        args = parser.parse_args([])
        if args_states:
            for key, value in args_states.items():
                args.__dict__[key] = value
        return args

    def _get_model(self, args, tokenizer, device, states):
        encoder = Encoder(args, pretrained=False)
        args.encoder_dim = encoder.n_features
        decoder = Decoder(args, tokenizer)

        safe_load(encoder, states['encoder'])
        safe_load(decoder, states['decoder'])
        # print(f"Model loaded from {load_path}")

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        return encoder, decoder

    def predict_images(self, input_images: List, return_atoms_bonds=True, return_confidence=False, batch_size=16,
                       r_sym_det_model=None):
        device = self.device
        predictions = []
        self.decoder.compute_confidence = return_confidence

        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx:idx + batch_size]
            images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            images = torch.stack(images, dim=0).to(device)
            with torch.no_grad():
                features, hiddens = self.encoder(images)
                batch_predictions = self.decoder.decode(features, hiddens)
            predictions += batch_predictions

        if r_sym_det_model:
            r_sym_predictions = r_sym_det_model.predict(input_images, is_backbone=True)

            for image, pred, r_sym_pred in zip(input_images, predictions, r_sym_predictions):
                if not r_sym_pred:
                    continue

                # 别名更新
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                _update_atom_aliases(coords, symbols, r_sym_pred, image)

        smiles = [pred['chartok_coords']['smiles'] for pred in predictions]
        node_coords = [pred['chartok_coords']['coords'] for pred in predictions]
        node_symbols = [pred['chartok_coords']['symbols'] for pred in predictions]
        edges = [pred['edges'] for pred in predictions]

        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            node_coords, node_symbols, edges, images=input_images, num_workers=self.num_workers)

        outputs = []
        for smiles, molblock, pred in zip(smiles_list, molblock_list, predictions):
            pred_dict = {"smiles": smiles, "molfile": molblock}
            if return_confidence:
                pred_dict["confidence"] = pred["overall_score"]
            if return_atoms_bonds:
                coords = pred['chartok_coords']['coords']
                symbols = pred['chartok_coords']['symbols']
                # get atoms info
                atom_list = []
                for i, (symbol, coord) in enumerate(zip(symbols, coords)):
                    atom_dict = {"atom_symbol": symbol, "x": coord[0], "y": coord[1]}
                    if return_confidence:
                        atom_dict["confidence"] = pred['chartok_coords']['atom_scores'][i]
                    atom_list.append(atom_dict)
                pred_dict["atoms"] = atom_list
                # get bonds info
                bond_list = []
                num_atoms = len(symbols)
                for i in range(num_atoms - 1):
                    for j in range(i + 1, num_atoms):
                        bond_type_int = pred['edges'][i][j]
                        if bond_type_int != 0:
                            bond_type_str = BOND_TYPES[bond_type_int]
                            bond_dict = {"bond_type": bond_type_str, "endpoint_atoms": (i, j)}
                            if return_confidence:
                                bond_dict["confidence"] = pred["edge_scores"][i][j]
                            bond_list.append(bond_dict)
                pred_dict["bonds"] = bond_list
            outputs.append(pred_dict)
        return outputs

    def predict_image(self, image, return_atoms_bonds=True, return_confidence=False):
        return self.predict_images([
            image], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]

    def predict_image_files(self, image_files: List, return_atoms_bonds=False, return_confidence=False):
        input_images = []
        for path in image_files:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_images.append(image)
        return self.predict_images(
            input_images, return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)

    def predict_image_file(self, image_file: str, return_atoms_bonds=False, return_confidence=False):
        return self.predict_image_files(
            [image_file], return_atoms_bonds=return_atoms_bonds, return_confidence=return_confidence)[0]


def _update_atom_aliases(coords, symbols, detections, image):
    """批量更新原子别名"""
    h, w, c = image.shape
    # 清洗并扩张边界框
    detections = [det for det in detections if det.get('class_id', 0) in VALID_SYMBOLS]
    for det in detections:
        if 'box' not in det:
            det['box'] = convert_polygon_to_bbox(det['polygon'])
            det['box'] = expand_bbox((w, h), det['box'], 0.15)
    if not detections:
        return

    # 提取并清理R原子符号
    for i, (symbol, coord) in enumerate(zip(symbols, coords)):
        if symbol.startswith('[') and symbol.endswith(']'):
            x, y = coord
            x, y = w * x, h * y
            for det in detections:
                x1, y1, x2, y2 = det['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    r_sym = R_DET_SYMBOLS[det['class_id']]
                    symbols[i] = f'[{r_sym}]'


def convert_polygon_to_bbox(polygon) -> List[int | float]:
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


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
