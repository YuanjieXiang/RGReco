from typing import Dict, Any, List

import numpy as np
from PIL import Image

from src.models.mol_segment import RGroupSegmentor
from src.postprocessing.rgroup.rmol import convert_smiles_to_mol
from src.external.molscribe import MolScribe
from src.utils.image_processing import stitch_images
from src.utils.visualize import mol_visualize
from .common import ModelConfig, BaseModel


class MolRecognizer(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = ModelConfig
        self.model = MolScribe(config.model_path, device=config.device)

    @classmethod
    def load_model(cls, config: ModelConfig) -> 'MolRecognizer':
        return cls(config)

    def predict(self, images: List[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """
        Returns:
            {
                "smiles": smiles,
                "molfile": molblock,
                "confidence": confidence,
                "atoms": atom_list,
                "bonds": bond_list
            }
        """

        # 新增，使用文本识别模型增强分子识别模型识别文本下表的效果
        r_sym_det = None
        if "r_sym_det" in kwargs and isinstance(kwargs["r_sym_det"], RGroupSegmentor):
            r_sym_det = kwargs["r_sym_det"]

        predictions = self.model.predict_images(images, r_sym_det_model=r_sym_det)
        responses = []
        for pred in predictions:
            confidence = pred.get("confidence", 0)
            smiles = pred.get("smiles", '')
            responses.append({'smiles': smiles, "confidence": confidence})
        return responses

    @staticmethod
    def visualize(image: np.ndarray, prediction: Dict[str, Any]) -> Image.Image:
        smiles = prediction.get("smiles", '')
        mol = convert_smiles_to_mol(smiles)
        return stitch_images([Image.fromarray(image), mol_visualize(mol)])


