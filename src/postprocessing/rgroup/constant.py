# 用于计算相似度，因此尽可能少
SIMILAR_SYMBOLS = ['R', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar', 'W', 'L']
CMPD_SYMBOLS = {"compound", "cmpd", "id", 'example', 'no'}

# 用于正式匹配R基团
RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', "R'", "R''", "R'''", "R1'", "R2'", "R3'",
                  'X', 'Y', 'Z', 'Q', 'Ar', 'A', 'B', 'E']

R_DET_SYMBOLS = ['attach', 'cut', 'star'] + RGROUP_SYMBOLS

SPLIT_SIGNS = {"/", "|", ",", }
