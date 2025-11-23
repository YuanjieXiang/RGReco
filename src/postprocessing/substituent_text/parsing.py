import requests
import logging
from rdkit import Chem
from src.external.py2opsin import py2opsin
from .formatting import format
from src.external.molscribe.chemistry import _expand_carbon, _parse_formula, _condensed_formula_list_to_smiles
from src.settings import settings

log = logging.getLogger(__name__)


def parse(raw_group_text: str):
    """解析R基团字符的对外接口，输入R基团字符，返回对应的SMILES字符串，如果解析失败返回None"""
    raw_group_text = raw_group_text.strip()
    if not raw_group_text or raw_group_text in ('-', '/', '\\', '_', '—'):
        return "[H]"  # 字符串为空时，返回氢原子

    if raw_group_text in settings.ABBR_DICT:
        return remove_r_sym_in_smile(settings.ABBR_DICT[raw_group_text])

    chem_groups = format(raw_group_text)
    res = _parse(chem_groups)

    if not res:
        # 解析失败，记录一下
        # with open('parse_failed.log', 'a', encoding='utf-8') as f:
        #     f.write(f'{raw_group_text}\n')
        return None
    return res


def _parse(groups):
    """将输入的基团列表转换为单个的SMILES字符串

    Args:
        groups (list): 基团列表

    Returns:
        str: 基团列表对应的SMIELS字符串
    """
    if not groups:
        return None

    smis = []
    # 分三种情况处理，iupac、formula和复合类型
    for group in groups:
        # 复合类型递归处理
        if isinstance(group, list):
            smi = _parse(group)
        # TODO: iupac型考虑位置和数量信息
        elif group.group_type == 'iupac':
            iupac_str = group.value.strip('-')
            smi = iupac2smiles(iupac_str)
        elif group.group_type == 'formula':
            smi = formula2smiles(group.value)
        else:
            smi = None
        if not smi:
            log.debug((f"Cannot to parse group: {str(group)}"))
            return None
        smis.append(smi)
    try:
        smiles = _combine_smis(smis)
        return smiles
    except Exception as e:
        log.debug(f"发生了一个异常: {e}")
        return None


def _combine_smis(smis):
    """使用RDKit工具将SMILES字符串数组链接起来

    Args:
        smis (list): SMILES字符串数组

    Returns:
        str: SMILES字符串
    """
    if len(smis) == 0:
        return None
    elif len(smis) == 1:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smis[0]))
    else:
        # 从左至右依次连接
        root_mol = group_smi_to_mol(smis[0])
        for smi in smis[1:]:
            # 获取 root_mol 偏右的自由基原子编号
            root_atom_num = root_mol.GetNumAtoms()
            right_radical_electron_idx = -1
            for i in range(root_atom_num):
                atom = root_mol.GetAtomWithIdx(i)
                if atom.GetNumRadicalElectrons() > 0:
                    right_radical_electron_idx = i
            if right_radical_electron_idx == -1:
                log.error("Radical electron shortage")
                return None
            # 获取 mol 偏左的自由基原子编号
            left_radical_electron_idx = -1
            mol = group_smi_to_mol(smi)
            atom_num = mol.GetNumAtoms()
            for i in range(atom_num):
                atom = mol.GetAtomWithIdx(i)
                if atom.GetNumRadicalElectrons() > 0:
                    left_radical_electron_idx = i
                    break
            if left_radical_electron_idx == -1:
                log.error("Radical electron shortage")
                return None
            # 将两个分子放入一个分子对象，转为写模式，并将两个自由基连接连起来
            combine_mol = Chem.RWMol(Chem.CombineMols(root_mol, mol))
            left_radical_electron_idx += root_atom_num
            combine_mol.AddBond(right_radical_electron_idx, left_radical_electron_idx,
                                order=Chem.rdchem.BondType.SINGLE)
            # 减少对应的自由基
            atom_l = combine_mol.GetAtomWithIdx(left_radical_electron_idx)
            atom_l.SetNumRadicalElectrons(atom_l.GetNumRadicalElectrons() - 1)
            atom_r = combine_mol.GetAtomWithIdx(right_radical_electron_idx)
            atom_r.SetNumRadicalElectrons(atom_r.GetNumRadicalElectrons() - 1)

            root_mol = combine_mol
        smiles = Chem.MolToSmiles(root_mol)
        return smiles


def group_smi_to_mol(smiles: str):
    mol = None
    if '*' not in smiles and smiles.count('[') == 1 and smiles.count(']') == 1 and smiles.index(']') - smiles.index('[') == 2:
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


def iupac2smiles(iupac_str, use_web_api=settings.USE_OPSIN_WEB_API, is_rgroup: bool = True):
    """将分子的IUPAC名转换为SMILES字符串。

    Args:
        iupac_str (str): 符合IUPAC命名规范的字符串。
        use_web_api (bool, optional): 是否使用web api, 如果为否则调用Java库. Defaults to False.
        is_rgroup (bool): 是否解析的是基团

    Returns:
        str: IUPAC名对应的SMILES字符串, 如果失败则返回空。
    """
    # 新增，处理IUPAC基团命名不规范的问题
    _convert_func = _opsin_web_api if use_web_api else _opsin_py_api

    smiles = None
    # 第一种情况，应该加取代基后缀，表示这是一个取代基
    if is_rgroup and 'yl' not in iupac_str:
        yl_iupac_str = iupac_str + '-yl'
        smiles = _convert_func(yl_iupac_str)

    if not smiles:
        smiles = _convert_func(iupac_str)

    # 第二种情况，没有主结构导致位置和手性信息无效
    if not smiles and 0 <= iupac_str.find('-') < 3:
        iupac_str = iupac_str.split('-', 1)[1]
        smiles = _convert_func(iupac_str)

    if not smiles:
        log.debug("Failed to convert IUPAC name to SMILES string")
    # return remove_r_sym_in_smile(smiles)
    return smiles


def formula2smiles(formula_str, link_bond_num=1):
    """将分子式转换为SMILES字符串

    Args:
        formula_str (str): 化学分子式字符串。
        link_bond_num (int, optional): 与该基团连接的化学键的数量. Defaults to 1.

    Returns:
        _type_: 分子式对应的SMILES字符串
    """
    formula_list = _expand_carbon(_parse_formula(formula_str))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, link_bond_num, None)
    if not success:
        log.error(f"Cannot parse formula: {formula_str}")
        return ""
    # print(success)
    # mol = Chem.MolFromSmiles(smiles)
    # smiles = Chem.MolToSmiles(mol)  # 标准化会破坏原子顺序，因此不使用
    return smiles


def _opsin_web_api(iupac_name):
    """使用OPSIN官方提供的web api, 将IUPAC名转换为SMILES字符串。

    Args:
        iupac_name (str): 分子对应的IUPAC名。

    Returns:
        str: IUPAC名对应的SMILES字符串。
    """
    try:
        path = "https://opsin.ch.cam.ac.uk/opsin/"  # URL path to the OPSIN API
        apiurl = path + iupac_name + '.json'  # concatenate (join) strings with the '+' operator
        reqdata = requests.get(apiurl)  # get is a method of request data from the OPSIN server
        data = reqdata.json()  # get the downloaded JSON
        # del data['cml']
        smiles = data.get('smiles')
        return smiles if smiles is not None else ""
    except requests.RequestException as e:
        log.error(f"Request to opsin_web_api failed: {e}", exc_info=True)
        return ""


def _opsin_py_api(iupac_name):
    """使用opsin2py库将IUPAC名转换为SMILES字符串。需要Java环境。

    Args:
        iupac_name (str): 分子对应的IUPAC名。

    Returns:
        _type_: IUPAC名对应的SMILES字符串。
    """
    try:
        smiles = py2opsin(iupac_name, output_format='SMILES', allow_acid=True, allow_radicals=True,
                          wildcard_radicals=True)
        return smiles
    except ImportError as e:
        log.error(f"Module py2opsin is not available. {str(e)}")
        return ""
    except Exception as e:
        print(f"Error converting IUPAC name '{iupac_name}' to SMILES: {str(e)}")
        return ""


def remove_r_sym_in_smile(smiles: str):
    if '*' not in smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return ""
    mol = Chem.RWMol(mol)
    atoms_to_remove = []
    # 去除 *，用断裂键代替
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == '*':
            bonds = atom.GetBonds()
            adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
            for adjacent_idx in adjacent_indices:
                mol.RemoveBond(i, adjacent_idx)
            adjacent_atoms = [mol.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
            for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))
            atoms_to_remove.append(i)
    atoms_to_remove.sort(reverse=True)
    for i in atoms_to_remove:
        mol.RemoveAtom(i)
    smiles = Chem.MolToSmiles(mol)
    return smiles


