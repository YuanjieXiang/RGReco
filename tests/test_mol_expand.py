from src.postprocessing.rgroup.rmol import expand_rgroup, convert_smiles_to_mol, convert_mol_to_smiles
from src.postprocessing.rgroup.schema import Compound
from rdkit import Chem
from src.utils.visualize import mol_visualize
from src.utils.image_processing import stitch_images

def expand_test():
    smiles = "*c1cc(CSc2nnc(*)o2)cc([N+](=O)[O-])c1 |$Ar;;;;;;;;;;R1;;;;;;;$|"
    cmpds = [
        {
            'cmpd_id': '4d', 'smiles': '',
            'r_groups': {
                'R1': {'abbr': 'Cl', 'smiles': '[Cl]'},
                'R2': {'abbr': '-CH2CH2COOH', 'smiles': '[CH2]CC(=O)O'},
                'Ar': {'abbr': '', 'smiles': 'FC1=CC=[C]C=C1'}}
         },
        {
            'cmpd_id': '5d', 'smiles': '',
            'r_groups': {
                'R1': {'abbr': 'Cl', 'smiles': '[Cl]'},
                'R2': {'abbr': '-CH3', 'smiles': '[CH3]'},
                'Ar': {'abbr': '', 'smiles': 'FC1=CC=[C]C=C1'}}
         },
        {
            'cmpd_id': '6d', 'smiles': '',
            'r_groups': {
                'R1': {'abbr': 'Cl', 'smiles': '[Cl]'},
                'R2': {'abbr': '-CH2CH3', 'smiles': '[CH2]C'},
                'Ar': {'abbr': '', 'smiles': 'FC1=CC=[C]C=C1'}}
         },
        {
            'cmpd_id': '7d', 'smiles': '',
            'r_groups': {
                'R1': {'abbr': 'Cl', 'smiles': '[Cl]'},
                'R2': {'abbr': '-CH2CH2NH2', 'smiles': '[CH2]CN'},
                'Ar': {'abbr': '', 'smiles': 'FC1=CC=[C]C=C1'}}
         },
        {
            'cmpd_id': '8d', 'smiles': '',
            'r_groups': {
                'R1': {'abbr': 'Cl', 'smiles': '[Cl]'},
                'R2': {'abbr': '-CH2CH2C(O)NH2', 'smiles': '[CH2]CC(N)=O'},
                'Ar': {'abbr': 'TEST', 'smiles': ''}}
         },
    ]
    compounds = [Compound().from_dict(cmpd) for cmpd in cmpds]
    expanded_mols = []
    for cmpd in compounds:
        expanded_mol = expand_rgroup(smiles, cmpd.r_groups)
        expanded_mols.append(expanded_mol)
        smi = "" if expanded_mol is None else convert_mol_to_smiles(expanded_mol)
        cmpd.smiles = smi

    visualize = True
    # 结果可视化并存储
    if visualize:
        root_mol = convert_smiles_to_mol(smiles)
        origin_mol_img = mol_visualize(root_mol, legend='Skeletal Structure', rgroup_dict={"smiles": smiles})
        result_mol_imgs = [origin_mol_img]  # 原图放在第一位
        for i in range(len(compounds)):
            mol_img = mol_visualize(convert_smiles_to_mol(compounds[i].smiles),
                                    rgroup_dict=compounds[i].r_groups, legend=compounds[i].cmpd_id)
            result_mol_imgs.append(mol_img)
        result_img = stitch_images(result_mol_imgs)  # 将所有得到的结果绘制为一张大图
        result_img.save('result.png')
