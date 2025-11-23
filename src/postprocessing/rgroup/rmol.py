import logging

from rdkit import Chem
from rdkit.Chem import rdmolfiles

from src.postprocessing.rgroup.schema import RGroup
from src.postprocessing.substituent_text.parsing import remove_r_sym_in_smile, group_smi_to_mol
from .constant import RGROUP_SYMBOLS

log = logging.getLogger(__name__)

BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 12: Chem.rdchem.BondType.AROMATIC}


def replace_BH2_with_B_sym(mol):
    """有一种基团符号是B，同时也是一种元素，需要做特殊处理"""
    if not isinstance(mol, Chem.Mol):
        return None
        # 匹配甲基
    pattern = Chem.MolFromSmarts("[BH2]")
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return None
    # 编辑分子
    emol = Chem.EditableMol(mol)
    for match in matches:
        target_atom_idx = match[0]

        # 创建虚原子
        new_atom = Chem.Atom("*")
        new_atom.SetProp('molFileAlias', 'B')
        emol.ReplaceAtom(target_atom_idx, new_atom)

        # 删除氢
        h_indices = [n.GetIdx() for n in mol.GetAtomWithIdx(target_atom_idx).GetNeighbors() if n.GetSymbol() == "H"]
        for h_idx in sorted(h_indices, reverse=True):
            emol.RemoveAtom(h_idx)

    # 生成新分子
    new_mol = emol.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def expand_rgroup(smiles: str, rgroup_dict: dict[str, RGroup], rgroup_syms: dict[str: list[int]]):
    """输入分子的mol_block字符串和R基团字典，返回合并后的分子"""
    root_mol = convert_smiles_to_mol(smiles)
    if root_mol is None:
        log.error(f"Error SMILES: {smiles}")
        return None

    if 'B' in rgroup_dict:  # 处理特殊情况，即在R基团符号中发现了B
        root_mol = replace_BH2_with_B_sym(root_mol)

    matched_syms = []
    for r_sym in rgroup_syms:
        if r_sym in rgroup_dict:
            matched_syms.append(r_sym)

    # 如果只有一个符号未匹配，且字典中的key数量与符号数量相等
    if len(rgroup_syms) - len(matched_syms) == 1 and len(rgroup_dict) == len(rgroup_syms):
        # 找出未匹配的符号
        unmatched_sym = None
        for r_sym in rgroup_syms:
            if r_sym not in matched_syms:
                unmatched_sym = r_sym
                break

        # 找出字典中未被匹配的key
        unmatched_key = None
        for key in rgroup_dict:
            if key not in matched_syms:
                unmatched_key = key
                break

        # 如果找到了未匹配的符号和key，进行更新
        if unmatched_sym is not None and unmatched_key is not None:
            # 保存原始值
            original_value = rgroup_dict[unmatched_key]
            # 删除原始key
            del rgroup_dict[unmatched_key]
            # 用新的符号作为key，保持原始值
            rgroup_dict[unmatched_sym] = original_value
            matched_syms.append(unmatched_sym)

    if not matched_syms:
        log.warning("未找到可以扩展的R基团。")
        return root_mol

    # 将匹配到的结果连到主结构上
    mol_w = Chem.RWMol(root_mol)
    atoms_to_remove = []
    for sym in matched_syms:
        smi = rgroup_dict[sym].smiles
        if not smi:
            continue

        mol_r = group_smi_to_mol(smi)
        if mol_r is None:
            continue

        rids = rgroup_syms[sym]
        for rid in rids:
            atom = mol_w.GetAtomWithIdx(rid)
            bonds = atom.GetBonds()
            # 获取主结构上与虚拟原子连接的原子，并移除与虚拟原子连接的键
            adjacent_indices = [bond.GetOtherAtomIdx(rid) for bond in bonds]
            for adjacent_idx in adjacent_indices:
                mol_w.RemoveBond(rid, adjacent_idx)
            # 根据键的类型修改与虚拟原子连接的原子的自由基
            # adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
            bond_types = [bond.GetBondType() for bond in bonds]

            # get indices of atoms of main body that connect to substituent
            bonding_atoms_w = adjacent_indices
            # 将要与主结构连接的原子的编号
            offset = mol_w.GetNumAtoms()
            bonding_atoms_r = [offset + atm.GetIdx() for atm in mol_r.GetAtoms() if atm.GetNumRadicalElectrons() > 0]
            if not bonding_atoms_r:
                log.warning("未找到自由基原子，无法确定成键位置。")
                break

            # combine main body and substituent into a single molecule object
            combo = Chem.CombineMols(mol_w, mol_r)

            # connect substituent to main body with bonds
            mol_w = Chem.RWMol(combo)
            # if len(bonding_atoms_r) == 1:  # substituent uses one atom to bond to main body
            for atm, bond_type in zip(bonding_atoms_w, bond_types):
                mol_w.AddBond(atm, bonding_atoms_r[0], order=bond_type)
            # reset radical electrons
            for atm in bonding_atoms_w:
                mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
            for atm in bonding_atoms_r:
                mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
            atoms_to_remove.append(rid)
    # Remove atom in the end, otherwise the id will change
    # Reverse the order and remove atoms with larger id first
    atoms_to_remove.sort(reverse=True)
    for i in atoms_to_remove:
        mol_w.RemoveAtom(i)
    mol = mol_w.GetMol()

    return mol
    # TODO: 根据实验结果再想办法处理不能匹配的


def convert_smiles_to_mol(smiles):
    """安全的转SMILES为Mol，避免报错"""
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for i, atom in enumerate(mol.GetAtoms()):
                label = atom.GetProp('atomLabel') if atom.HasProp('atomLabel') else None
                if label:
                    Chem.SetAtomAlias(atom, label)

    except:
        return None
    return mol


def convert_mol_to_smiles(mol):
    if mol:
        for i, atom in enumerate(mol.GetAtoms()):
            alias = atom.GetProp('molFileAlias') if atom.HasProp("molFileAlias") else None
            if alias:
                atom.SetProp("atomLabel", alias)
                atom.ClearProp("molFileAlias")
                atom.ClearProp("dummyLabel")
        params = Chem.SmilesWriteParams()
        params.doIsomericSmiles = True  # 是否输出立体化学信息（默认 True）
        params.doCanonical = True  # 是否生成规范化 SMILES（默认 True）
        params.allBondsExplicit = False  # 是否显式输出所有键类型
        params.allHsExplicit = False  # 是否显式显示所有氢原子
        params.kekuleSmiles = False  # 是否使用 Kekulé 表示法（即不显示芳香环）
        params.rootedAtAtom = -1  # 是否指定某个原子为起点
        smi = Chem.MolToCXSmiles(mol, params, rdmolfiles.CXSmilesFields.CX_ATOM_LABELS)
        # print(smi)
        return smi
    return ""


def get_rsymbols_from_mol(mol: Chem.Mol):
    """从分子对象中获取所有R基团相关字符"""
    if mol is None:
        return {}, None

    rgroup_syms = dict()
    unknown_syms = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()  # 获取原子索引
        mol_file_alias = atom.GetProp('molFileAlias') if atom.HasProp("molFileAlias") else None

        # 记录R基团的位置
        if mol_file_alias:
            if mol_file_alias in RGROUP_SYMBOLS or "R" in mol_file_alias:
                if mol_file_alias not in rgroup_syms:
                    rgroup_syms[mol_file_alias] = []
                rgroup_syms[mol_file_alias].append(atom_idx)

            # 不是已知的R基团符号，存储一下，看是什么缩写不能识别
            else:
                unknown_syms.append(mol_file_alias)
                # log.warning(f"发现未正确解析的缩写：{mol_file_alias}, 已记录")
                # with open('molscribe_parse_failed.log', 'a', encoding='utf-8') as file:
                #     file.write(f'{mol_file_alias}\n')
    return rgroup_syms, unknown_syms


def get_rsymbols_from_smiles(smiles: str):
    """获取 smiles 字符串中的所有R字符并返回.

    Args:
        smiles (str): smiles字符串，表示一个完整的分子。

    Returns:
        dict: 记录了R基团和对应的原子编号
    """
    mol = convert_smiles_to_mol(smiles)
    if mol is None:
        log.warning(f"smiles 格式错误。smiles: {smiles}.")
        return {}, None
    return get_rsymbols_from_mol(mol)


def get_rsymbols_from_block(mol_block: str):
    """获取mol_block字符串中的所有R字符并返回。

    Args:
        mol_block (str): rdkit的mol_block字符串，表示一个完整的分子。

    Returns:
        dict: 记录了R基团和对应的原子编号
    """
    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        log.warning(f"mol_block格式错误。mol_block:{mol_block}.")
        return {}
    return get_rsymbols_from_mol(mol)

