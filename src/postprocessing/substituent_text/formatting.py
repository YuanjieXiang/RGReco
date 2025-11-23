import logging
import re
from .constant import ELEMENTS, PREFIXES, ABBRS, CONVERTIBLE_ELEMENTS, SUBSCRIPTS, SUPERSCRIPTS, CONVERTIBLE_FORMULA

log = logging.getLogger(__name__)
replacement_map = {**SUBSCRIPTS, **SUPERSCRIPTS}
trans_table = str.maketrans(replacement_map)


def convert_sup_sub_to_normal(text):
    return text.translate(trans_table)


def _preprocess(input_str):
    """对输入的R基团文本进行预处理规范化，去除不必要的内容，并规范字符类型。

    Args:
        input_str (str): R基团文本

    Returns:
        str: 规范后的R基团文本
    """
    # 更新，处理几种苯环加位置的情况
    if 'C6H4' in input_str and input_str[0].isdigit():
        input_str = input_str.replace('C6H4', '')
        input_str = formula2iupac(input_str) + '-Ph'
    elif input_str.startswith('CH2Ph'):
        input_str = input_str.replace('CH2Ph', '')
        input_str = formula2iupac(input_str) + '-Bn'
    elif input_str.startswith("CH2-p-Ph"):
        input_str = input_str.replace('CH2-p-Ph', '')
        input_str = '4-' + formula2iupac(input_str) + '-Bn'

    input_str = re.sub(r'-+', '-', input_str)  # 合并连续连字符
    # 由于字符串普遍较短，使用replace更快
    input_str = (
        # 替换全角括号为半角，去除空格
        input_str.replace("（", "(")
        .replace("）", ")")
        .replace("∘", "O")
        .replace(" ", "")
        # 手性信息
        .replace("(rac)", "")
        .replace("(R)", "(r)")
        .replace("(S)", "(s)")
        # 顺反异构体
        .replace("(E)", "(e)")
        .replace("(Z)", "(z)")
        # 几种特殊分子式
        .replace("C6H5", "Ph")
        .replace("(Me)", "(CH3)")
        .replace("N3", "azido")
    )
    input_str = re.sub(r'\b(di|tri|tetra)(?=[A-Z])', r'\1-', input_str, flags=re.IGNORECASE)  # 统一取代基数目前缀格式（如 di → di-）
    input_str = re.sub(r'Me(\d+)', r'(CH3)\1', input_str)  # Me3这种情况
    input_str = convert_sup_sub_to_normal(input_str)
    return input_str


def formula2iupac(input_str):
    return input_str.replace('CH3', 'Me').replace('OH', 'hydroxy')


class ChemToken:
    def __init__(self, value, token_type, iupac_name=None, sub_tokens=None):
        self.value = value  # 实际字符串
        self.token_type = token_type  # "iupac", "abbr", "element", "formula_sign", "conj_sign", "prefix", "position", "combi", "other"
        self.iupac_name = iupac_name  # 对缩写适用，转换为iupac后的结果
        self.sub_tokens = sub_tokens  # 对括号情况适用，括号使用递归

    def __str__(self):
        sub_tokens_str = ", ".join(str(token) for token in self.sub_tokens) if self.sub_tokens is not None else '[]'
        return f"value: {self.value}, token_type: {self.token_type}, iupac_name: {self.iupac_name}, sub_tokens: [{sub_tokens_str}]"


def _tokenize(input_str: str):
    """输入基团文本，返回对应的分词结果。采用长序列优先的深搜算法。
    例如:
    OiPr -> [O, i, Pr], 
    (CH2)4-(p-F-Ph) -> [(CH2)4, -, (p-F-Ph)], 其中的两部分会对括号内的内容递归处理

    Args:
        input_str (str): 经过预处理的基团文本

    Returns:
        tokens: 分词后的结果
        success: 是否处理成功
    """
    tokens = []

    def dfs(cur_idx):
        if cur_idx >= len(input_str):
            return True
        # 左括号，需要做递归处理
        if input_str[cur_idx] == '(' or input_str[cur_idx] == '[':
            right = _find_matching_parenthesis(input_str, cur_idx)
            if right == -1:
                return False
            # 如果括号内长度只有1, 则认为是一种不需要处理的iupac前缀, 如(r)\[b]等
            if right - cur_idx == 2:
                end = right + 1
                tokens.append(ChemToken(input_str[cur_idx:end], 'prefix', input_str[cur_idx:end]))
                return dfs(end)

            # 否则递归处理
            sub_tokens, success = _tokenize(input_str[cur_idx + 1:right])
            if not success:
                return False
            # 考虑是分子式时, 括号后的数字
            end = right + 1
            while end < len(input_str) and input_str[end].isdigit():
                end += 1
            tokens.append(ChemToken(input_str[cur_idx:end], 'combi', sub_tokens=sub_tokens))
            return dfs(end)
        # 大写开头Token的处理，可能是iupac名、缩写或元素符号
        elif input_str[cur_idx].isupper():
            end = cur_idx + 1
            while end < len(input_str) and input_str[end].islower():  # 允许后面跟小写字母
                end += 1
            if 4 <= end - cur_idx:  # 有可能是大写开头的iupac名
                # 处理以大写 O 开头的情况，氧原子二价，因此常出现在开头，而O开头的iupac名必以Ox开头
                # TODO：待发现更多类似情况
                sub_str = input_str[cur_idx:end]
                if sub_str and sub_str[0] == 'O' and sub_str[1] != 'x':
                    end = cur_idx + 1
                else:
                    tokens.append(ChemToken(sub_str, "iupac"))
                    success = dfs(end)
                    if not success:
                        tokens.pop()
                    else:
                        return success
                    end = cur_idx + 3  # 缩写的最大长度为3
            for mid in range(end, cur_idx, -1):
                sub_str = input_str[cur_idx:mid]
                if sub_str in ABBRS:  # 能匹配上某种缩写
                    tokens.append(ChemToken(sub_str, "abbr", ABBRS[sub_str]))
                    success = dfs(mid)
                    if not success:
                        tokens.pop()
                    else:
                        return success
                # 匹配化学元素
                if len(sub_str) <= 2:
                    # 可转换为iupac的元素，不允许后面跟数字
                    if mid < len(input_str) and not input_str[mid].isdigit() and sub_str in CONVERTIBLE_ELEMENTS:
                        tokens.append(ChemToken(sub_str, "element", CONVERTIBLE_ELEMENTS[sub_str]))
                        success = dfs(mid)
                        if not success:
                            tokens.pop()
                        else:
                            return success
                    if sub_str in ELEMENTS:
                        # 一般元素，添加上后面跟的数字
                        while mid < len(input_str) and input_str[mid].isdigit():
                            mid += 1
                        tokens.append(ChemToken(input_str[cur_idx:mid], "element"))
                        success = dfs(mid)
                        if not success:
                            tokens.pop()
                        else:
                            return success
        # 小写字母开头，可能是位置缩写、同位素缩写或者iupac名
        elif input_str[cur_idx].islower():
            end = cur_idx + 1
            while end < len(input_str) and input_str[end].islower():  # 只允许后面跟小写字母
                end += 1
            if end - cur_idx == 1 and input_str[cur_idx:end] in PREFIXES:  # 确认是位置或同位素缩写
                prefix = PREFIXES[input_str[cur_idx:end]]  # 需在后面加一个连接符
                if end < len(input_str) and input_str[end] == '-':
                    end += 1
                tokens.append(ChemToken(input_str[cur_idx:end], 'prefix', prefix))
                return dfs(end)
            else:  # iupac名
                tokens.append(ChemToken(input_str[cur_idx:end], 'iupac'))
                return dfs(end)
        # 数字开头，可能为位置信息或氢原子位置（1H, 2H）
        elif input_str[cur_idx].isdigit():
            end = cur_idx + 1
            while end < len(input_str) and (input_str[end].isdigit() or input_str[end] == ',' or input_str[end] == 'H'):
                end += 1
            sub_str = input_str[cur_idx:end]
            token_type = 'iupac' if sub_str.endswith("H") else 'position'
            tokens.append(ChemToken(sub_str, token_type))
            return dfs(end)
        # 分子式中独有的连接符
        elif input_str[cur_idx] == '=':
            tokens.append(ChemToken(input_str[cur_idx:cur_idx + 1], 'formula_sign'))
            return dfs(cur_idx + 1)
        # 一般连接符
        elif input_str[cur_idx] == '-':
            tokens.append(ChemToken(input_str[cur_idx:cur_idx + 1], 'conj_sign'))
            return dfs(cur_idx + 1)
        # 其他字符（已知有', "），无意义
        elif input_str[cur_idx] in ("'", '"', ','):  # 已知的识别错误的字符
            return False
        else:
            tokens.append(ChemToken(input_str[cur_idx:cur_idx + 1], 'other'))
            return dfs(cur_idx + 1)

        return False

    success = dfs(0)
    return tokens, success


def _find_matching_parenthesis(context, start):
    """
    查找给定左括号对应的右括号位置。
    
    参数:
        s (str): 输入字符串。
        start (int): 左括号在字符串中的起始位置。
        
    返回:
        int: 对应的右括号位置；如果没有找到匹配的右括号，返回 -1。
    """
    stack = []
    brackets = {')': '(', ']': '['}  # 匹配的括号对
    if context[start] not in brackets.values():
        return -1  # 确保起始位置是一个左括号
    stack.append((context[start], start))
    for i in range(start + 1, len(context)):
        char = context[i]
        if char in brackets.values():  # 遇到左括号
            stack.append((char, i))
        elif char in brackets:  # 遇到右括号
            if stack and stack[-1][0] == brackets[char]:  # 检查是否匹配
                stack.pop()
                if not stack:  # 如果栈为空，说明找到了匹配的右括号
                    return i
            else:
                return -1  # 不匹配的情况
    return -1  # 没有找到匹配的右括号


class ChemGroup:
    def __init__(self, value, group_type, tokens):
        self.group_type = group_type  # "iupac", "formula"
        self.value = value  # 具体的字符串，能直接用于转换
        self.tokens = tokens  # 原始信息，用于调试

    def __str__(self):
        tokens_str = ", ".join(str(token) for token in self.tokens)
        return f"group_type: {self.group_type}, value: {self.value}, tokens: {tokens_str}"


def _get_token_state(token):
    if token.token_type in ('iupac', 'abbr', 'position', 'prefix'):
        return 'iupac'
    elif (token.token_type == 'element' and token.iupac_name is None) or token.token_type == 'formula_sign':
        return 'formula'
    else:
        return 'middle'


def _combine(tokens):
    """使用贪心算法使得到的分词组成最少数量的集合，需要注意的是这并非万能解，存在无法处理的情况。

    Args:
        tokens (list[ChemToken]): ChemToken的数组，如果有原始文本中有括号，可能还包含嵌套的数组
    """

    def _submit():
        nonlocal assemble, now_state
        if not assemble:
            log.debug(f"empty assemble!")
            return False
        value = ""
        # 组合为分子式，直接连接
        if now_state == "formula":
            group_type = "formula"
            for token in assemble:
                # 更新，如果以(CH3)起始，应找到括号后第一个token，作为主链
                if value.startswith('(CH3)'):
                    value = token.value + value
                else:
                    value += token.value
        # 组合为iupac，注意O和S转iupac需要变成后缀
        else:
            group_type = "iupac"
            suffix = None  # 后缀为-oxy或者-thio
            accept_suffix = False  # 是否允许插入后缀
            for token in assemble:
                # 遇到后缀
                if token.value == 'O' or token.value == 'S':
                    if accept_suffix:
                        value += token.iupac_name
                    elif not suffix:
                        suffix = token.iupac_name
                    else:
                        log.error(f"Continuous Suffix: {str(assemble)}")
                        return False
                # 一般的iupac名
                elif token.token_type == 'iupac':
                    value += token.value
                    accept_suffix = True
                # 前缀应该被转换，但后面不能直接接后缀
                elif token.token_type == 'prefix':
                    value += token.iupac_name
                # 缩写、某种可转换的元素
                elif token.iupac_name is not None:
                    if accept_suffix:
                        value += '-'  # 对中间和末尾的缩写，在前面添加 - 字符，
                    value += token.iupac_name
                    accept_suffix = True
                # 其它字符，TODO 先过滤掉, 后面考虑是否需要特殊处理
                elif token.token_type == 'other':
                    log.debug(f'Other sign {token.value} has been filtered')
                    continue
                # 位置或连字符
                else:
                    if suffix is not None and token.token_type == 'conj_sign':  # 当有正在等待的后缀时，跳过连字符，O-xx的情况
                        continue
                    value += token.value
                    accept_suffix = False

                if accept_suffix and suffix is not None:
                    value += suffix
                    suffix = None
                    accept_suffix = False
            if suffix:
                log.error(f"Superfluous Suffix: {str(assemble)}")
                return False

        # 提交结果
        combined_groups.append(ChemGroup(value, group_type, assemble))
        # 重置收集器和状态
        now_state = 'middle'
        assemble = []
        return True

    combined_groups = []
    now_state = 'middle'
    assemble = []
    for token in tokens:
        token: ChemToken
        # 复合类型表示遇到括号，需递归处理子结构，分成三种情况处理
        if token.token_type == 'combi':
            sub_combined_groups = _combine(token.sub_tokens)
            # 子结构构成多个基团，表示子结构式是复合的
            if len(sub_combined_groups) > 1:
                if not _submit():
                    return []
                # 添加得到的子结构，注意这里默认括号后不会出现数字，所以不考虑
                log.info(f"Combined group: {token.value}")
                combined_groups.append(sub_combined_groups)
            # 子结构只有一个基团，分为iupac和formula进行处理
            elif len(sub_combined_groups) == 1:
                single_group = sub_combined_groups[0]
                single_group: ChemGroup
                # 不能与前面收集的共融，先提交结果
                if now_state != 'middle' and now_state != single_group.group_type:
                    if not _submit():
                        return []

                # 分子式使用原始字符串（便于保留括号后的数字）
                if single_group.group_type == 'formula':
                    assemble.append(ChemToken(token.value, 'element'))
                    now_state = 'formula'
                # iupac使用刚刚获取到的字符串, 但要添加括号
                else:
                    assemble.append(ChemToken(f'({single_group.value})', 'iupac'))
                    now_state = 'iupac'
            else:
                log.error(f'Failed to process the substructure: {token.sub_tokens}')
                return []
        # 其他情况处理（分子、iupac命名）
        else:
            new_state = _get_token_state(token)
            if now_state != 'middle' and new_state != 'middle' and now_state != new_state:
                if not _submit():
                    return []
            if now_state != new_state and new_state != 'middle':
                now_state = new_state
            assemble.append(token)
    _submit()
    return combined_groups


def format(raw_group_text):
    """输入原始基团文本，输出不同的基团数组，用于后续转换为SMILES字符串

    Args:
        raw_group_text (str): 原始基团文本

    Returns:
        list: 基团数组，里面存储了基团的基本信息
    """
    standard_group_text = _preprocess(raw_group_text)  # 预处理，规范文本
    tokens, success = _tokenize(standard_group_text)  # 对规范文本进行分词
    if not success:
        log.debug(f"Unable to format {raw_group_text}")
        for token in tokens:
            log.debug(token)
        return None

    combined_groups = _combine(tokens)  # 对分词进行连接，尽可能的分成最少数量的块
    return combined_groups
