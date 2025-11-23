import logging
import math
import os
from collections import defaultdict
from typing import List, Iterable

from PIL import Image
from bs4 import BeautifulSoup

from src.models.table_recognize import TableResult
from src.models.text_recognize import MathTextRecognizer
from src.models.text_recognize import TextLine
from src.postprocessing.substituent_text.formatting import convert_sup_sub_to_normal
from .constant import SIMILAR_SYMBOLS, CMPD_SYMBOLS, SPLIT_SIGNS, RGROUP_SYMBOLS
from .schema import RGroupAbbr, Compound

log = logging.getLogger(__name__)


class SuryaTableManager:
    """
        新的表格管理器，用于Surya的表格识别方案：Surya text_det, text_rec, table_det, table_rec
    """

    def __init__(self, table_rec_res: TableResult, text_lines: list[TextLine], source_image: Image.Image,
                 math_ocr_model: MathTextRecognizer = None, structures: list = None):
        """
        初始化表格管理器
        Args:
            table_rec_res (TableResult): 表格识别结果.
            text_lines: list[TextLine]: 文本识别结果.
            source_image (Image.Image): 原始表格图像。
            math_ocr_model (MathTextRecognizer): 数学公式识别模型。
            structures：分子结构列表。
        """
        assert isinstance(math_ocr_model, MathTextRecognizer), "需要数学识别模型！ "

        # 如下初始化方式会修改原始数据，可以根据需求看是否需要复制一份副本
        self.unmatched_text_lines = []
        self.merge_table_ocr_result(table_rec_res, text_lines)
        self.table = table_rec_res
        self._table_map = None
        self.math_ocr_model = math_ocr_model
        self.table_img = source_image
        self.structures = structures if structures else []

        _ = self.table_map  # 初始化表格映射

    @property
    def format_table(self):
        """将表格的具体文本内容提取为一个二维字符串数组"""
        return [[self.get_text_by_cell_id(cell_id) for cell_id in row] for row in self.table_map]

    @property
    def table_map(self):
        """构建表格映射数组，数组中记录的是cell的编号"""
        if not self._table_map:
            cells = self.table.cells
            cells.sort(key=lambda cel: cel.cell_id)
            # 有时模型输出的cell_id并非连续，重置cell_id为连续
            for i, cell in enumerate(cells):
                cell.cell_id = i

            num_col = max(self.table.cols, key=lambda col: col.col_id).col_id + 1
            num_row = max(self.table.rows, key=lambda row: row.row_id).row_id + 1
            table_map = [[-1 for _ in range(num_col)] for _ in range(num_row)]  # 初始化映射表
            for i, cell in enumerate(cells):
                assert cell.cell_id == i, f"cell_id 并非严格递增，需要重新编写逻辑。 {i}_{cells}"  # TODO: remove
                for i in range(cell.rowspan):
                    for j in range(cell.colspan):
                        row_id = cell.row_id + i
                        col_id = cell.col_id + j
                        if row_id < num_row and col_id < num_col:
                            table_map[row_id][col_id] = cell.cell_id
            self._table_map = table_map
        return self._table_map

    def math_ocr(self, texe_line: TextLine, force: bool = False, repair: bool = True) -> bool:
        """
        输入 Texe_line，交给数学文本识别模型重新识别。
        Args:
            texe_line (Texe_line): OCRResult中的文本块，其中记录了文本内容和文本坐标.
            force (bool): 如果为True，当识别结果不同时，强制认可math_ocr的结果
            repair (bool): 是否需要启动自动修复.

        Returns:
            bool: 是否替换了文本内容
        """
        cropped_image = self.table_img.crop(texe_line.bbox)
        generated_text = self.math_ocr_model.predict([cropped_image], parse_latex=True, batch_size=1, repair=repair)[0]
        if generated_text == texe_line.text:
            log.debug("数学识别结果未改变")
            return False
        if repair:
            pass  # TODO: 收集更多运行结果，研究合适方案，尽量提高准确率
        # 识别结果不同的情况下，可以使用force强制接收数学模型，或结果为 R 字符加上下标的情况也可接受
        if force or 'R' in generated_text and ('^' in generated_text or '_' in generated_text):
            generated_text = generated_text.replace('^', '').replace('_', '')
            texe_line.text = generated_text
            return True
        else:
            return False

    def _get_cmpd_and_r_cell_ids(self, head_cell_ids, cmpd_syms, similar_r_syms):
        cmpd_sym_cell_id = self.capture_sym_cell_by_cell_ids(cmpd_syms, head_cell_ids)
        r_sym_cell_id = self.capture_sym_cell_by_cell_ids(similar_r_syms, head_cell_ids)
        return cmpd_sym_cell_id, r_sym_cell_id

    def get_cmpds(self, cmpd_syms: Iterable[str] = CMPD_SYMBOLS, similar_r_syms: Iterable[str] = SIMILAR_SYMBOLS) -> \
            List[Compound]:
        """
        获取化合物编号及R基团对应的列，参数需要是可迭代多次的对象，如list\set\dict，因为会使用余弦相似度，因此不需要严格匹配
        Args:
            cmpd_syms (Iterable[str]): ，可能存在的化合物编号符号
            similar_r_syms (Iterable[str]): 可能存在的R基团符号
        Returns:
            List[Compound]: 记录化合物对应的R基团字典，每个元素对应一个化合物
        """
        cells = self.table.cells

        # 利用字符相似度匹配算法寻找标题行
        half_row_num = min(len(self.table_map) / 2, 5)  # 标题行最多在前一半，并认为不超过4行
        row_now = 0
        cmpd_sym_cell_id, r_sym_cell_id = -1, -1
        while row_now < half_row_num and r_sym_cell_id == -1:  # 有时cmpd会占多行，或者没有，因此以找到R基团符号为终止条件
            head_cell_ids = set(self.table_map[row_now])
            cmpd_sym_cell_id, r_sym_cell_id = self._get_cmpd_and_r_cell_ids(head_cell_ids, cmpd_syms, similar_r_syms)
            row_now += 1

        if cmpd_sym_cell_id == -1 and r_sym_cell_id == -1:
            return []
        elif r_sym_cell_id == -1:
            head_row_id = cells[cmpd_sym_cell_id].row_id
        else:
            head_row_id = cells[r_sym_cell_id].row_id  # 只要找到了R基团对应行，就应该以R基团为准(cmpd有时会跨行，而R基团不会)

        cmpd_sym_str = self.get_text_by_cell_id(cmpd_sym_cell_id) if cmpd_sym_cell_id != -1 else ""  # 记录一下标题字符串

        r_col_ids = []
        combined_r_col_infos = []  # 考虑如 R1,R2 出现在同一个单元格的情况
        all_r_syms = []  # 获取到的R基团符号
        cmpd_sym = self.get_text_by_cell_id(cmpd_sym_cell_id)
        cmpd_col_ids = []  # 如果找到多个cmpd_sym，说明表格有多栏，需要做特殊处理
        head_cell_ids = [cell_id for cell_id in head_cell_ids if cells[cell_id].row_id == head_row_id]
        for cell_id in head_cell_ids:
            if cells[cell_id].text_lines is None:
                cells[cell_id].text_lines = []
            for text_line in cells[cell_id].text_lines:
                self.math_ocr(text_line, force=False)
            text = self.get_text_by_cell_id(cell_id)  # 统一为大写，防止大小写识别错误的问题

            for sp in SPLIT_SIGNS:
                if sp in text:
                    sp_text = text.split(sp, 1)
                    if len(sp_text) == 2 and sp_text[0] in RGROUP_SYMBOLS and sp_text[1] in RGROUP_SYMBOLS:
                        combined_r_col_infos.append(
                            (sp_text[0], sp_text[1], sp, cells[cell_id].col_id))  # 注意这里的结构与r_col_ids不同
                    break

            if text in RGROUP_SYMBOLS:  # R 基团符号
                r_col_ids.append(cells[cell_id].col_id)
                all_r_syms.append(text)
            elif text in ('x', 'z'):  # 修正大小写识别错误
                r_col_ids.append(cells[cell_id].col_id)
                all_r_syms.append(text.upper())
            elif cmpd_sym and text == cmpd_sym:  # cmpd 符号
                cmpd_col_ids.append(cells[cell_id].col_id)

        r_col_ids, all_r_syms = zip(*sorted(zip(r_col_ids, all_r_syms)))  # 排序确保分栏时读取正确

        # 初始化cmpd（化合物）编号列表
        if len(cmpd_col_ids) == 0:  # 如果没有找到编号列，强行定义第一个列为编号列
            cmpd_col_ids = [0]
        elif len(r_col_ids) % len(cmpd_col_ids) != 0 or len(combined_r_col_infos) % len(
                cmpd_col_ids) != 0:  # 确保多栏时R基团符号能被均分
            cmpd_col_ids = cmpd_col_ids[0:]

        cmpd_sym_num = len(cmpd_col_ids)
        row_num = len(self.table_map) - head_row_id - 1  # 每栏的行数
        r_sym_num = len(r_col_ids) // cmpd_sym_num  # R 基团符号的数量
        combined_r_sym_num = len(combined_r_col_infos) // cmpd_sym_num

        cmpds = []  # 最终返回的化合物列表
        for i, cmpd_col_id in enumerate(cmpd_col_ids):
            for j in range(row_num):
                row_id = j + head_row_id + 1
                cmpd_id = self.get_text_by_cell_id(self.table_map[row_id][cmpd_col_id])
                if not cmpd_id or cmpd_sym == cmpd_id:
                    continue
                cmpd_ids = []

                # 一种特殊情况，如 65a-65e，实际上是5个化合物，但共用其它基团（共用的基团不在表格中）
                if '-' in cmpd_id and 0 < cmpd_id.index('-') < len(cmpd_id):
                    l, r = map(str.strip, cmpd_id.split('-', 1))
                    if len(l) == len(r):
                        common_start = os.path.commonprefix([l, r])
                        if len(common_start) == len(l) - 1:
                            s, e = l[-1], r[-1]
                            if ord(e) > ord(s):
                                cmpd_ids = [f"{common_start}{chr(i)}" for i in range(ord(s), ord(e) + 1)]

                if not cmpd_ids:
                    cmpd_ids = [cmpd_id]

                for cmpd_id in cmpd_ids:


                    cmpd = Compound()
                    # 化合物编号
                    cmpd.cmpd_id = cmpd_id
                    r_groups: dict[str, RGroupAbbr] = dict()
                    waiting_r_syms: dict[str, str] = dict()  # 记录跳转，如 R5: abbr=R4
                    # 处理一般 R 基团
                    for k in range(r_sym_num):
                        # R基团
                        r_id = k + (r_sym_num * i)  # 第几个R基团
                        r_col_id = r_col_ids[r_id]  # 对应的列号
                        r_sym = all_r_syms[r_id]  # 对应的R基团符号
                        cell_id = self.table_map[row_id][r_col_id]
                        abbr_text = self.get_text_by_cell_id(cell_id)

                        if abbr_text == "":
                            abbr_group = RGroupAbbr(abbr='/STRUCT')
                            for struct in self.structures:
                                if is_more_than_half_in_box(struct['bbox'], self.table.cells[cell_id].bbox):
                                    abbr_group._smiles = struct.get('smiles', '')
                                    break
                            if not abbr_group.smiles:
                                abbr_group = RGroupAbbr(abbr='')
                        else:
                            abbr_group = RGroupAbbr(abbr=abbr_text)
                        # 尝试解析缩写，如果没解析出结果，进行特殊尝试
                        if not abbr_group.smiles:
                            if abbr_group.abbr == '/STRUCT':
                                pass
                            # 1. 特殊情况：link ，比如 above、=R4
                            elif abbr_group.abbr.lower() in ("same as above", "above"):
                                if len(cmpds) > 0 and r_sym in cmpds[-1].r_groups:
                                    abbr_group = cmpds[-1].r_groups[r_sym]
                            elif abbr_group.abbr in all_r_syms:
                                waiting_r_syms[r_sym] = abbr_group.abbr
                            elif abbr_group.abbr[1:].strip() in all_r_syms:
                                waiting_r_syms[r_sym] = abbr_group.abbr[1:].strip()
                            else:
                                # 2. 特殊情况：识别错误，使用数学OCR的强制替换模式重新识别再转换
                                if cells[cell_id].text_lines:
                                    for text_line in cells[cell_id].text_lines:
                                        self.math_ocr(text_line, force=True)
                                    abbr_group.abbr = self.get_text_by_cell_id(cell_id)
                        r_groups[r_sym] = abbr_group

                    # 处理两个R基团和并在一起的情况
                    for k in range(combined_r_sym_num):
                        r_id = k + (combined_r_sym_num * i)
                        left_r_sym, right_r_sym, sp, r_col_id = combined_r_col_infos[r_id]  # 合并基团的信息
                        cell_id = self.table_map[row_id][r_col_id]
                        abbr_texts = self.get_text_by_cell_id(cell_id).split(sp, 1)
                        if len(abbr_texts) == 1:  # TODO：这种情况一般是两个基团连接在一起的特殊情况，可以特殊处理
                            left_abbr = right_abbr = abbr_texts[0]
                        elif len(abbr_texts) == 2:
                            left_abbr = abbr_texts[0]
                            right_abbr = abbr_texts[1]
                        else:
                            continue

                        left_abbr_group = RGroupAbbr(abbr=left_abbr)
                        right_abbr_group = RGroupAbbr(abbr=right_abbr)
                        r_groups[left_r_sym] = left_abbr_group
                        r_groups[right_r_sym] = right_abbr_group

                    # 处理 link 的情况
                    for waiting_r_sym, link_r_sym in waiting_r_syms.items():
                        if link_r_sym in r_groups:
                            r_groups[waiting_r_sym] = r_groups[link_r_sym]

                    # 有时会出现所有基团的值都为空的情况，显然是错误的，因此过滤掉（如果是-或/这类字符，则记为H）
                    if not r_groups or all(abbr_group.abbr in (None, '') for abbr_group in r_groups.values()):
                        continue
                    cmpd.r_groups = r_groups
                    cmpds.append(cmpd)

        return cmpds

    def capture_sym_cell_by_cell_ids(self, aim_sym_iter: Iterable[str], cell_ids: List[int] | Iterable[int]) -> int:
        """输入需要匹配的字符组和需要检查的cell_id列表，返回第一个匹配的cell_id"""
        for cell_id in cell_ids:
            text = self.get_text_by_cell_id(cell_id)
            if 1 <= len(text) <= 3 and text[0] in aim_sym_iter:  # 处理基团符号
                return cell_id
            if seqs_match(text, aim_sym_iter):  # 处理化合物编号
                return cell_id
        return -1

    def get_text_by_cell_id(self, cell_id: int):
        if cell_id < 0 or cell_id >= len(self.table.cells):
            return ""
        text_lines = self.table.cells[cell_id].text_lines
        if not text_lines:
            return ""
        content = "".join(text_line.text for text_line in text_lines)
        format_content = content.replace('—', '-')
        return format_content

    def merge_table_ocr_result(self, table_result: TableResult, text_lines: list[TextLine]):
        """合并表格识别和OCR识别的结果"""

        def calc_ioa(box_a, box_b):
            """
            计算 box_b 与 box_a 的交集占 box_a 面积的比例（Intersection over Area of A）
            用于判断 box_b 是否大面积覆盖 box_a，避免大框或冗余框导致的误判。
            注意：这不是标准 IoU，而是非对称的 IoA。
            """
            a_x1, a_y1, a_x2, a_y2 = box_a
            b_x1, b_y1, b_x2, b_y2 = box_b
            # 计算交集区域
            x_left = max(a_x1, b_x1)
            y_top = max(a_y1, b_y1)
            x_right = min(a_x2, b_x2)
            y_bottom = min(a_y2, b_y2)
            if x_right < x_left or y_bottom < y_top:
                return 0.0

            intersection = (x_right - x_left) * (y_bottom - y_top)
            # 计算各自面积
            area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
            return intersection / area_a

        cells = table_result.cells
        for text_line in text_lines:
            max_iou = 0.5  # 最低阈值
            matched_cell = None
            text_box = text_line.bbox

            for cell in cells:
                iou = calc_ioa(text_box, cell.bbox)  # 改用IOA计算，即计算文本框有多少部分在cell框内

                # 保留最大IOU的单元格
                if iou > max_iou:
                    max_iou = iou
                    matched_cell = cell

            if matched_cell:
                if matched_cell.text_lines is None:
                    matched_cell.text_lines = []
                matched_cell.text_lines.append(text_line)
            else:
                self.unmatched_text_lines.append(text_line)

        # 排序结果
        for cell in cells:
            if cell.text_lines is not None:
                cell.text_lines.sort(key=lambda t: (t.bbox[1], t.bbox[0]))


class TableManager:
    """
        旧的表格管理器
    """

    def __init__(self, html_content):
        log.warning("该表格识别方案已被废弃，请使用新的方案")
        self.table_array = self.parse_table(html_content)

    def parse_table(self, html_content):
        """将html形式的表格转换为数组，并将所有文本中的空格去掉，如果遇到合并单元格以填充法填充。

        Args:
            html_content (str): html字符串

        Returns:
            list: 表格对应的二维数组
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')

        # 提取所有行
        rows = table.find_all('tr')

        # 计算表格的最大列数
        max_cols = 0
        for row in rows:
            cols = 0
            for cell in row.find_all(['td', 'th']):
                cols += int(cell.get('colspan', 1))  # 考虑colspan
            max_cols = max(max_cols, cols)

        # 初始化二维数组
        result = [[None for _ in range(max_cols)] for _ in range(len(rows))]

        # 填充数据
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row.find_all(['td', 'th']):
                # 跳过已被填充的位置
                while result[row_idx][col_idx] is not None:
                    col_idx += 1
                # 获取单元格的值（去除空格）
                cell_value = cell.text.replace(" ", '')
                # 转换上下标为标准字符
                cell_value = convert_sup_sub_to_normal(cell_value)
                # 获取colspan和rowspan
                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                # 填充主单元格及合并区域
                for i in range(rowspan):
                    for j in range(colspan):
                        if row_idx + i < len(result) and col_idx + j < max_cols:
                            result[row_idx + i][col_idx + j] = cell_value
                # 更新列索引
                col_idx += colspan

        # 如果存在None，应该替换为空字符串
        for i in range(len(result)):  # 遍历每一行（i 是行索引）
            for j in range(len(result[i])):  # 遍历当前行中的每个元素（j 是列索引）
                if result[i][j] is None:  # 如果元素为 None
                    result[i][j] = ""  # 就地替换为空字符串

        return result

    def extract_rcols_by_title(self, similar_r_syms=SIMILAR_SYMBOLS):
        """抽取序号和R基团对应的列，结果以字典返回"""
        # 尝试在前三行寻找标题行
        cnt, title_row_id = 1, -1
        for i, row in enumerate(self.table_array):
            if cnt > 3 or title_row_id != -1:
                break
            for text in row:
                only_alpha = ''.join([char for char in text if char.isalpha()])
                if only_alpha in similar_r_syms or only_alpha in CMPD_SYMBOLS:
                    title_row_id = i
                    break

        if title_row_id == -1:
            return None

        # 在标题行中寻找匹配的字串
        cmpd_col = []
        rgroup_cols = defaultdict(lambda: None)
        title_row = self.table_array[title_row_id]

        log.info(f'title_row: {str(title_row)}')

        for i, title in enumerate(title_row):
            if title is None:
                title = ''
            # 前3列找编号列
            if i < 3 and not cmpd_col:
                low_case_title = title.lower()
                if seqs_match(low_case_title, CMPD_SYMBOLS):
                    for row in self.table_array[title_row_id + 1:]:
                        cmpd_col.append(row[i])

            # 长度在3-5之间，可能有多个R基团
            if 5 >= len(title) >= 3:
                split_symbol = None
                for s in SPLIT_SIGNS:
                    if s in title:
                        split_symbol = s
                        break
                sub_titles = title.split(split_symbol) if split_symbol else []
                if len(sub_titles) == 2 and seqs_match(sub_titles[0], similar_r_syms):
                    col_content = [row[i].split(split_symbol) for row in self.table_array[title_row_id + 1:]]
                    if not rgroup_cols[sub_titles[0]]:
                        rgroup_cols[sub_titles[0]] = [arr[0] if len(arr) == 2 else "" for arr in col_content]
                    else:
                        log.warning(f"重复的R基团字符： {sub_titles[0]}")
                    if not rgroup_cols[sub_titles[1]]:
                        rgroup_cols[sub_titles[1]] = [arr[1] if len(arr) == 2 else "" for arr in col_content]
                    else:
                        log.warning(f"重复的R基团字符： {sub_titles[1]}")
            if len(title) <= 3 and seqs_match(title, similar_r_syms):
                if not rgroup_cols[title]:
                    rgroup_cols[title] = [row[i] for row in self.table_array[title_row_id + 1:]]
                else:
                    log.warning(f"重复的R基团字符： {title}")

        if not rgroup_cols.keys():
            log.debug(cmpd_col)
            return None
        if not cmpd_col:
            cmpd_col = [str(i) for i in range(1, len(self.table_array) - title_row_id)]
        result = rgroup_cols
        result['id'] = cmpd_col
        return result


def seqs_match(seq: str, match_set: Iterable[str]):
    # seq与多个文本进行序列匹配，一个成功即可
    for s in match_set:
        if seq_match(s, seq):
            return True


def cosine_similarity(s1, s2):
    """计算两个字符串的余弦相似度"""
    # log.warning("不建议使用余弦相似度，因为该算法会统计字符共享的情况，这导致 analogue 与 a 的相似度都达到了0.6")
    count1 = {}
    count2 = {}
    for char in s1:
        count1[char] = count1.get(char, 0) + 1
    for char in s2:
        count2[char] = count2.get(char, 0) + 1
    # 计算点积和向量的模长
    dot_product = 0
    norm_s1 = 0
    norm_s2 = 0
    for char in set(s1).union(s2):
        dot_product += count1.get(char, 0) * count2.get(char, 0)
        norm_s1 += count1.get(char, 0) ** 2
        norm_s2 += count2.get(char, 0) ** 2
    norm_s1 = math.sqrt(norm_s1)
    norm_s2 = math.sqrt(norm_s2)
    # 防止除以0
    if norm_s1 == 0 or norm_s2 == 0:
        return 0.0
    return dot_product / (norm_s1 * norm_s2)


def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:  # 处理空字符串
        return 1.0
    return intersection / union


def seq_match(s1: str, s2: str, threshold=0.6):
    """序列匹配，当匹配度达到阈值时返回真，默认阈值为0.6"""
    similarity = jaccard_similarity(s1.lower(), s2.lower())
    log.debug(f'[{s1}:{s2}] Jaccard相似度={similarity:.2f}')
    return similarity >= threshold


def is_more_than_half_in_box(A, B):
    """
    判断 box A 是否有超过 50% 的区域在 box B 内。

    :param A: list or tuple, [x_min, y_min, x_max, y_max]
    :param B: list or tuple, [x_min, y_min, x_max, y_max]
    :return: bool
    """
    # 解包坐标
    a_xmin, a_ymin, a_xmax, a_ymax = A
    b_xmin, b_ymin, b_xmax, b_ymax = B

    # 计算 A 的面积
    a_width = max(0, a_xmax - a_xmin)
    a_height = max(0, a_ymax - a_ymin)
    area_a = a_width * a_height
    if area_a == 0:
        return False  # 如果 A 没有面积，直接返回 False

    # 计算交集区域的坐标
    inter_xmin = max(a_xmin, b_xmin)
    inter_ymin = max(a_ymin, b_ymin)
    inter_xmax = min(a_xmax, b_xmax)
    inter_ymax = min(a_ymax, b_ymax)

    # 计算交集区域的面积
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    area_inter = inter_width * inter_height

    # 判断交集面积是否超过 A 面积的一半
    return area_inter / area_a > 0.5
