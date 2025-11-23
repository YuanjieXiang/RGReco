import os
import csv
from typing import Optional, ClassVar

import torch
import onnxruntime as ort
from pydantic import computed_field
from pydantic_settings import BaseSettings


def load_abbreviation_to_smiles(csv_path):
    """
    从 CSV 文件中读取 abbreviation 和 smiles 列，返回字典 {abbr: smi}

    CSV 格式应包含列：abbreviation, smiles, population
    """
    abbr_to_smi = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbr = row['abbreviation'].strip()
            smi = row['smiles'].strip()
            if abbr and smi:
                abbr_to_smi[abbr] = smi
    return abbr_to_smi


class Setting(BaseSettings):
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 本文件的上层目录

    # result
    RESULT_DIR: str = os.path.join(BASE_DIR, 'result')

    # GENERAL
    RUNNING_MODE: str = "test"  # ui | cli | test, using in app.py
    TORCH_DEVICE: Optional[str] = 'cuda'  # None | cpu | cuda
    ONNX_DEVICE: Optional[str] = 'cuda'  # None | cpu | cuda
    # CLI MODE (single image)
    IMAGE_PATH: Optional[str] = os.path.join(BASE_DIR, "evaluate_dataset/images/JMC_2024_01_213-233_02_01.png")
    # TEST MODE (folder)
    TEST_DIR: str = os.path.join(BASE_DIR, "evaluate_dataset", "images")
    # eval
    GOLD_PATH: str = os.path.join(BASE_DIR, "evaluate_dataset", "gold")
    RUN_WITH_GOLD: bool = True


    # MODEL SETTINGS
    LAZY_LOAD: bool = True  #
    # mol_recognition: Copyright (c) 2022 Yujie Qian. https://github.com/thomas0809/MolScribe
    MOL_REC_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\MolScribe.pth")
    MOL_REC_VISUALIZE: bool = True
    # mol_segmentation: Copyright (c) 2020 Kohulan Rajan. https://github.com/Kohulan/DECIMER-Image-Segmentation
    MOL_SEG_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\mol_seg.onnx")
    MOL_SEG_VISUALIZE: bool = True
    # doc_recognition: Surya: https://github.com/datalab-to/surya
    TABLE_REC_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\table_rec")
    TABLE_REC_VISUALIZE: bool = True
    TABLE_DET_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\table_det")
    TABLE_DET_VISUALIZE: bool = True
    TEXT_REC_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\text_rec")
    TEXT_REC_VISUALIZE: bool = True
    TEXT_DET_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\text_det")
    TEXT_DET_VISUALIZE: bool = False
    MATH_REC_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\math_rec")
    MATH_REC_VISUALIZE: bool = False
    # substituent attachment and R-Group identifier detection
    RGROUP_DET_CHECKPOINT: str = os.path.join(BASE_DIR, r"weights\r_group_det.pt")
    RGROUP_DET_VISUALIZE: bool = True

    # OPSIN: Open Parser for Systematic IUPAC Nomenclature https://github.com/dan2097/opsin
    USE_OPSIN_WEB_API: bool = False

    # abbreviation dictionary
    ABBR_DICT: ClassVar[dict] = load_abbreviation_to_smiles(os.path.join(BASE_DIR, 'data/abbreviations.csv'))

    @computed_field
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        try:
            import torch_xla
            if len(torch_xla.devices()) > 0:
                return "xla"
        except:
            pass

        return "cpu"

    @computed_field
    def ONNX_PROVIDERS(self) -> str:
        # 定义设备检测优先级和映射关系
        DEVICE_PROVIDERS = {
            'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'cpu': ['CPUExecutionProvider']
        }
        if self.ONNX_DEVICE in DEVICE_PROVIDERS:
            return DEVICE_PROVIDERS[self.ONNX_DEVICE]

        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.ONNX_DEVICE = 'cuda'
        else:
            self.ONNX_DEVICE = 'cpu'
        return providers


settings = Setting()
