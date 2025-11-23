class SafeDict(dict):
    def __getitem__(self, key):
        try:
            # 尝试获取键值
            value = super().__getitem__(key)
        except KeyError:
            # 键不存在时返回 None
            return None
        # 递归处理嵌套字典
        if isinstance(value, dict) and not isinstance(value, SafeDict):
            value = SafeDict(value)
            super().__setitem__(key, value)
        return value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化时递归转换嵌套字典和列表中的字典
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = SafeDict(value)
            elif isinstance(value, list):
                self[key] = [
                    SafeDict(item) if isinstance(item, dict) else item
                    for item in value
                ]