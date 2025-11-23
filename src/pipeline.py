import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import ModelLoader
from src.postprocessing.rgroup import SuryaTableManager
from src.settings import settings
from src.storage.storage import DataStorage, StorageEntry
from src.external.surya.table_rec.schema import TableResult, TableCell, TableCol, TableRow
from src.utils.box_processing import convert_polygon_to_bbox, convert_polygon_to_mask, convert_mask_to_box
from src.utils.image_loader import ImageLoader
from src.utils.image_processing import stitch_images, crop_bboxes
from src.utils.pipeline_processing import *
from src.utils.visualize import mol_visualize, visualize_detection_segmentation
from src.external.decimer_segmentation import complete_structure_mask as mask_postprocess

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    meta: dict | None = None
    mol_res: dict | None = None
    table_res: dict | None = None
    text_res: dict | None = None
    compounds: dict | None = None
    success: bool = False
    _locked_fields = set()

    def __post_init__(self):
        """自动将 None 替换为 {}"""
        for field in ['meta', 'mol_res', 'table_res', 'text_res', 'compounds']:
            value = getattr(self, field)
            if value is None:
                setattr(self, field, {})

    def lock_field(self, field_name: str) -> None:
        """
        设置某个字段的值，并将其锁定，防止后续修改。
        """
        if not hasattr(self, field_name):
            raise AttributeError(f"字段 {field_name} 不存在")
        self._locked_fields.add(field_name)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._locked_fields and key in self.__dict__:
            print(f"🔒 警告：字段 '{key}' 已锁定，修改无效。")
            return
        object.__setattr__(self, key, value)

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        """将结果转换为字典，用于保存"""
        # 分子识别结果
        structures = self.mol_res.get("structures", [])
        mol_list = [
            {"polygon": mol['polygon'], "smiles": mol.get('smiles', ''), "bbox": mol['bbox'],
             "is_candidate": mol.get('is_candidate', False), "cut_det": mol.get('cut_det', [])} for mol in structures]
        # 表格结果
        table_box = self.table_res.get("table_box", {})
        table_res = self.table_res.get("table_rec")
        table_rec_res = {"cells": [], "rows": [], "cols": []}
        if table_box and table_res:
            table_res: TableResult
            cells = table_res.cells
            rows = table_res.rows
            cols = table_res.cols
            for cell in cells:
                info = f"{cell.row_id},{cell.colspan},{cell.within_row_id},{cell.cell_id},{int(cell.is_header)},{cell.rowspan},{cell.col_id}"
                table_rec_res["cells"].append({"info": info, "polygon": cell.polygon})
            for row in rows:
                info = f"{row.row_id},{int(row.is_header)}"
                table_rec_res["rows"].append({"info": info, "polygon": row.polygon})
            for col in cols:
                info = f"{col.col_id},{int(col.is_header)}"
                table_rec_res["cols"].append({"info": info, "polygon": col.polygon})
        # 文本识别结果
        text_lines = [{"text": tl.text, "confidence": tl.confidence, "polygon": tl.polygon} for tl in
                      self.text_res.get("text_lines", [])]

        # 化合物结果
        cmpds = [cmpd.to_dict() for cmpd in self.compounds.get("cmpds", [])]

        return {
            "meta": self.meta,
            "mol_res": {"structures": mol_list},
            "table_res": {"table_box": table_box, "table_rec": table_rec_res},
            "text_res": {"text_lines": text_lines},
            "compounds": {"cmpds": cmpds},
        }

    @classmethod
    def from_dict(cls, data: dict, image_path):
        # meta
        image_size = (data["meta"]["height"], data["meta"]["width"])
        test_image = ImageLoader().load_cv_img(image_path)
        # 分子结果
        mol_res = {}
        structures = data.get("mol_res", {}).get("structures", [])
        mol_list = [
            {"smiles": m["smiles"],
             "polygon": m["polygon"],
             "mask": mask_postprocess(test_image, convert_polygon_to_mask(m["polygon"], image_size)[:, :, None],
                                      image_size).squeeze(),
             "bbox": m['bbox'],
             "confidence": 1,
             "is_candidate": m["is_candidate"],
             "cut_det": m.get("cut_det", [])
             } for m in structures
        ]
        # for mol in mol_list:
        #     mol['bbox'] = convert_mask_to_box(mol['mask'])

        mol_res["structures"] = mol_list
        # 表格结果
        table_res = {}
        if data.get("table_res"):
            if data.get("table_res").get("table_box"):
                table_res["table_box"] = data.get("table_res").get("table_box")
            if data.get("table_res").get("table_rec"):
                table_rec = data.get("table_res").get("table_rec")
                if table_rec.get("cells") and table_rec.get("rows") and table_rec.get("cols"):
                    cell_infos = [list(map(int, cell["info"].split(","))) for cell in table_rec.get("cells")]
                    row_infos = [list(map(int, row["info"].split(","))) for row in table_rec.get("rows")]
                    col_infos = [list(map(int, col["info"].split(","))) for col in table_rec.get("cols")]
                    cells, rows, cols = [], [], []
                    for i, cell in enumerate(table_rec.get("cells")):
                        cell = TableCell(polygon=cell["polygon"], row_id=cell_infos[i][0], colspan=cell_infos[i][1],
                                         within_row_id=cell_infos[i][2],
                                         cell_id=cell_infos[i][3], is_header=cell_infos[i][4], rowspan=cell_infos[i][5],
                                         col_id=cell_infos[i][6])
                        cells.append(cell)
                    for i, row in enumerate(table_rec.get("rows")):
                        row = TableRow(polygon=row["polygon"], row_id=row_infos[i][0], is_header=row_infos[i][1])
                        rows.append(row)
                    for i, col in enumerate(table_rec.get("cols")):
                        col = TableCol(polygon=col["polygon"], col_id=col_infos[i][0], is_header=col_infos[i][1])
                        cols.append(col)
                    table_res["table_rec"] = TableResult(cells=cells, rows=rows, cols=cols, unmerged_cells=[],
                                                         image_bbox=[])
                    # 文本结果
        if data.get("text_res") and data["text_res"].get("text_lines"):
            text_lines = [TextLine(text=tl["text"], polygon=tl["polygon"], confidence=tl["confidence"]) for tl in
                          data["text_res"]["text_lines"]]
        else:
            text_lines = []
        # 化合物结果
        compounds = {}
        for k, v in data.get("compounds", {}).items():
            cmpds = [Compound.from_dict(cmpd) for cmpd in v]
            compounds[k] = cmpds
        return cls(
            meta=data["meta"],
            mol_res=mol_res,
            table_res=table_res,
            text_res={"text_lines": text_lines},
            compounds=compounds
        )


class Pipeline:
    def __init__(self, lazy_load=True, allow_save=True, visualize=True):
        self.allow_save = allow_save
        self.visualize = visualize
        self._storage = DataStorage()
        self.models = ModelLoader(lazy_load=lazy_load)
        self.image_loader = ImageLoader()
        self.result: None | PipelineResult = None
        self.image_path: None | Path = None

    def _reload_image(self, image_path: str) -> np.ndarray:
        self.image_path = Path(image_path)
        image = self.image_loader.load_cv_img(image_path)
        height, width, _ = image.shape
        self.result = PipelineResult(
            meta={"image_path": self.image_path, "height": height, "width": width})  # 重置当前结果
        return image

    def _mol_segment(self, source_image: np.ndarray):
        # 分子分割
        mol_seg_res = self.models.mol_seg.predict([source_image])[0]
        self.result.mol_res["structures"] = mol_seg_res.get("structures", [])

    def _mol_segment_postprocess(self, source_image: np.ndarray) -> List[np.ndarray]:
        masks = [s['mask'] for s in self.result.mol_res["structures"]]
        masks = np.stack(masks, axis=2)
        mol_images, _ = self.models.mol_seg.crop_masks(source_image, masks)
        return mol_images

    def _attachment_point_detection(self, mol_images: List[np.ndarray]):
        cut_detections = self.models.r_group_seg.predict(mol_images)
        for structure, cut_detection in zip(self.result.mol_res["structures"], cut_detections):
            x1, y1, x2, y2 = structure['bbox']
            for cut_det in cut_detection:
                polygon_cut_det = cut_det['polygon']
                cut_det['polygon'] = [[x + x1, y + y1] for x, y in polygon_cut_det]
            structure['cut_det'] = cut_detection

    def _attachment_point_detection_postprocess(self, mol_images: List[np.ndarray]):
        """将连接点替换为R基团符号"""
        for i, structure in enumerate(self.result.mol_res.get("structures", [])):
            detections = structure.get("cut_det", [])
            if isinstance(detections, list) and len(detections) > 0:
                detection = detections[0]  # TODO 处理多连接点
                cls = detection['class_id']
                polygon_cut_det = detection['polygon']
                x1, y1, x2, y2 = structure['bbox']
                polygon = [[point[0] - x1, point[1] - y1] for point in polygon_cut_det]
                replace_image = self.models.r_group_seg.replace_polygon(mol_images[i], polygon, cls)
                mol_images[i] = self.image_loader.load_cv_img(replace_image)

    def _mol_recognize(self, mol_images: list[np.ndarray]):
        mol_rec_res = self.models.mol_rec.predict(mol_images)
        structures = self.result.mol_res["structures"]
        for structure, mol in zip(structures, mol_rec_res):
            structure["smiles"] = mol['smiles']

        if len(mol_images) > 1:
            bad_mol_res, normal_mol_res, candidate_mol_res = split_mol_result(structures)
            for cand in candidate_mol_res:
                cand["is_candidate"] = True
            for normal in normal_mol_res:
                normal["is_candidate"] = False
            self.result.mol_res['structures'] = candidate_mol_res + normal_mol_res
        else:
            structures[0]["is_candidate"] = True

        # 重新识别主结构（利用R基团符号检测模型识别上下标）
        for image, struct in zip(mol_images, structures):
            if struct.get("is_candidate", False):
                smiles = self._mol_re_recognize(image)
                if smiles:
                    struct['smiles'] = smiles

    def _mol_re_recognize(self, image):
        return self.models.mol_rec.predict([image], r_sym_det=self.models.r_group_seg)[0]["smiles"]

    def _mol_recognize_postprocess(self, source_image: np.ndarray):
        structures = self.result.mol_res.get('structures', [])
        if len(structures) == 0:
            return source_image
        remove_masks = np.stack([item['mask'] for item in structures], axis=2)
        _, image_without_mol = self.models.mol_seg.crop_masks(source_image, remove_masks)
        return image_without_mol

    def _table_detect(self, image_without_mol: Image.Image):
        # 表格检测
        table_det_res = self.models.table_det.predict([image_without_mol])[0]
        if not table_det_res or not table_det_res.get('polygon', None):
            log.debug(f"未检测到表格")
            return
        self.result.table_res['table_box'] = table_det_res

    def _table_detect_postprocess(self, image_without_mol: Image.Image):
        """需要注意在文本识别后才修改图像，以提高表格识别准确率"""
        table_box = self.result.table_res.get('table_box', {})
        if table_box and table_box.get('polygon', None):
            table_box = convert_polygon_to_bbox(table_box.get('polygon'))
            mol_res = self.result.mol_res.get('structures', [])
            image_without_mol = replace_mol_to_text_in_table(image_without_mol, table_box,
                                                             [item['bbox'] for item in mol_res])
        return image_without_mol

    def _table_recognize(self, image: Image.Image, table_box):
        # 表格识别
        offset = table_box[0]
        table_box = convert_polygon_to_bbox(table_box)
        table_images, _ = crop_bboxes(image, [table_box])
        table_rec_res = self.models.table_rec.predict(table_images)[0]
        for item in table_rec_res.cells + table_rec_res.rows + table_rec_res.cols:
            item.polygon = [[p[0] + offset[0], p[1] + offset[1]] for p in item.polygon]
        self.result.table_res["table_rec"] = table_rec_res

    def _table_recognize_postprocess(self, table_image: Image.Image):
        # 5. 后处理，结构化的表格中，解析出可能是R基团的列
        table_structures, other_structures = filter_structures_by_box(self.result.mol_res["structures"],
                                                                      convert_polygon_to_bbox(
                                                                          self.result.table_res['table_box'][
                                                                              'polygon']))
        table_manager = SuryaTableManager(self.result.table_res["table_rec"], self.result.text_res["text_lines"],
                                          table_image, self.models.math_rec, table_structures)
        self.result.compounds['table_cmpds'] = table_manager.get_cmpds()
        self.result.table_res['format'] = table_manager.format_table
        # 将表格外的文本和分子结构单独摘出来
        self.result.text_res['other_text_lines'] = table_manager.unmatched_text_lines
        self.result.mol_res['other_structures'] = other_structures

    def _text_recognize(self, text_images: List[Image.Image]):
        # 文本检测与识别，其中可能还包括表格图像的文本识别
        text_rec_res = self.models.text_rec.predict(text_images)
        self.result.text_res['text_lines'] = text_rec_res[0].text_lines

    def _text_recognize_postprocess(self, text_lines, structures):
        # 后处理，从分子结构和表格之外的文本内容种抽取R基团信息
        structure_textlines = [
            TextLine(text=f'MOL{i}', polygon=convert_polygon_to_bbox(struct['polygon']))
            for i, struct in enumerate(structures) if struct.get('is_candidate', True) == False
        ]
        sentences = self.models.text_rec.sort_and_merge_text_lines(text_lines, structure_textlines)  # 转换为每行的具体文本列表
        if sentences:
            self._storage.put_entry(
                StorageEntry('\n'.join(sent for sent in sentences), 'sentence', 'txt', extension='txt',
                             session_name=self.image_path.stem))
        tagged_sentences = [rannotation.Sentence(sent) for sent in sentences]
        # 第一种方法，直接在句子中寻找，如果没找到再用位置关系匹配
        label_cmpds = rannotation.detect_r_group(tagged_sentences)
        for cmpd in label_cmpds:
            for abbr_group in cmpd.r_groups.values():
                if abbr_group.abbr.startswith('MOL') and len(abbr_group.abbr) > 3:
                    abbr_group._smiles = structures[int(abbr_group.abbr[3:])]['smiles']
        if not label_cmpds:
            candidata_smiles = next(s['smiles'] for s in self.result.mol_res['structures'] if s['is_candidate'])
            if "|$" in candidata_smiles and ';' in candidata_smiles:
                r_sym = candidata_smiles.split('|$')[1].split(';', maxsplit=1)[0]
                label_cmpds = match_text_and_struct(r_sym, text_lines, structure_textlines, structures)

        self.result.compounds['label_cmpds'] = label_cmpds

    def _merge_compounds(self):
        cmpds = merge_cmpds(self.result.compounds.get('table_cmpds', []),
                            self.result.compounds.get('label_cmpds', []))
        cmpds.sort(key=lambda x: natural_sort_key(x.cmpd_id))  # 按id排序结果
        self.result.compounds['cmpds'] = cmpds

    def load_result(self):
        json_path = os.path.join(settings.GOLD_PATH, f"{Path(self.image_path).stem}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PipelineResult.from_dict(data, self.image_path)

    def pipeline(self, image_path: str, session_name="result", visualize=False) -> PipelineResult:
        print(f'==========================={session_name}==========================')
        self.visualize = visualize
        result = self._pipeline(image_path)
        self._storage.put_entry(StorageEntry(result.to_dict(), self.image_path.stem, 'json',
                                             session_name=session_name))
        return result

    def _pipeline(self, image_path: str) -> PipelineResult:
        """默认流程

        Args:
            image_path (str): 图像的路径
        
        Returns:
            PipelineResult: 管线运行的信息
        """
        # 读取图片，进行预处理
        image = self._reload_image(image_path)
        if settings.RUN_WITH_GOLD:
            self.result = self.load_result()
            self.result.compounds = {}

        try:
            # 分子分割与处理
            self._mol_segment(image)
            mol_images = self._mol_segment_postprocess(image)

            if len(mol_images) == 0:
                log.error("没找到分子结构，退出流程")
                return self.result
            elif len(mol_images) > 1:
                # R基团分子分割
                self._attachment_point_detection(mol_images)
                self._attachment_point_detection_postprocess(mol_images)

            self._mol_recognize(mol_images)
            image_without_mol = self._mol_recognize_postprocess(image)

            has_candidates = False
            for structure in self.result.mol_res.get('structures', []):
                if structure.get('is_candidate'):
                    has_candidates = True
                    break
            if not has_candidates:
                log.warning("没有检测到需要扩展的分子结构！")
                return self.result
            # 文本识别去除分子结构后的图像
            image_without_mol = self.image_loader.load_image(image_without_mol)
            self._text_recognize([image_without_mol])

            # 表格识别与处理
            self._table_detect(image_without_mol)
            if self.result.table_res.get('table_box', {}).get('polygon', None):
                image_without_mol = self._table_detect_postprocess(image_without_mol)
                table_box = self.result.table_res.get('table_box', {}).get('polygon')
                self._table_recognize(image_without_mol, table_box)
                self._table_recognize_postprocess(image_without_mol)
                if self.allow_save:
                    self._storage.put_entry(
                        StorageEntry(self.result.table_res['format'], "table", 'csv',
                                     session_name=self.image_path.stem))
                self._text_recognize_postprocess(self.result.text_res['other_text_lines'],
                                                 self.result.mol_res['other_structures'])  # other_text_lines 即为表格外的文本
            # 只处理文字
            else:
                self._text_recognize_postprocess(self.result.text_res['text_lines'], self.result.mol_res['structures'])

            # 化合物处理
            self._merge_compounds()
            cmpds = self.result.compounds['cmpds']

            if not cmpds:
                log.info("未发现R基团")
                return self.result

            # 后处理，融合前面得到的结果。
            candidate_smiles = [structure['smiles'] for structure in self.result.mol_res.get('structures', []) if
                                structure['is_candidate']]
            for mol_id, smiles in enumerate(candidate_smiles):
                rgroup_syms, _ = rmol.get_rsymbols_from_smiles(smiles)
                for cmpd in cmpds:
                    cmpd.origin_smi = smiles
                    expanded_mol = rmol.expand_rgroup(smiles, cmpd.r_groups,
                                                      rgroup_syms)  # 如果处理失败，将返回空 TODO 将方法移交给Compound类
                    smi = "" if expanded_mol is None else rmol.convert_mol_to_smiles(expanded_mol)
                    cmpd.smiles = smi
                # 保存结果
                # self._storage.put_entry(StorageEntry([{'origin_smi': smiles}] + cmpds, f'result_{mol_id}', 'json',
                #                                      session_name=self.image_path.stem))

                # 结果可视化并存储
                if self.visualize:
                    root_mol = rmol.convert_smiles_to_mol(smiles)
                    origin_mol_img = mol_visualize(root_mol, legend=self.image_path.stem,
                                                   rgroup_dict={'smiles': smiles})
                    result_mol_images = [origin_mol_img]  # 原图放在第一位
                    for i in range(len(cmpds)):
                        mol_img = mol_visualize(rmol.convert_smiles_to_mol(cmpds[i].smiles),
                                                rgroup_dict=cmpds[i].r_groups, legend=cmpds[i].cmpd_id)
                        result_mol_images.append(mol_img)
                    result_img = stitch_images(result_mol_images)  # 将所有得到的结果绘制为一张大图
                    self._storage.put_entry(
                        StorageEntry(result_img, self.image_path.stem, 'image', extension='png',
                                     session_name="visualize"))

                    image = self.image_loader.load_cv_img(image_path)
                    self._storage.put_entry(
                        StorageEntry(visualize_detection_segmentation(image, self.result), self.image_path.stem, 'image', extension='png',
                                     session_name="det_seg_vis"))

            self.result.success = True
        except Exception:
            logging.exception("Operation failed")
            return self.result
        return self.result
