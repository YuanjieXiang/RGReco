from src.postprocessing.substituent_text.parsing import formula2smiles, iupac2smiles, _parse, _combine_smis, parse, group_smi_to_mol
from src.postprocessing.substituent_text.formatting import _tokenize, _preprocess, format, _combine
from rdkit import Chem


def test_groupsmi(smi):
    mol = group_smi_to_mol(smi)
    smi = Chem.MolToSmiles(mol)
    print(smi)


def test_combine_smis():
    _combine_smis(['[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])', 'Fc1cc[c]cc1'])


def test_parse(input):
    print(parse(input))



def test_formula2smiles(formula_str, link_bond_num=1):
    smi = formula2smiles(formula_str, link_bond_num)
    print(smi)
    # smi = '[C]([H])([H])[C]([H])([H])[C](=[O])[N]([H])([H])'
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    print(smi)


def test_iupac2smiles(iupac_str, use_web_api=False):
    smi = iupac2smiles(iupac_str, use_web_api)
    print(smi)
