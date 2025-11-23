from src.pipeline import PipelineResult
import json


def test_load_result(file_path):
    # 打开并加载 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 查看结果
    print(data)
    pr = PipelineResult.from_dict(data)
    pr.lock_field("mol_res")
    print(pr.mol_res)
    pr.mol_res = {}
    print(pr.mol_res)
