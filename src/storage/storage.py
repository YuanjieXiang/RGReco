import json
import os
import logging
import threading
import io
import re
import cv2
import csv
import numpy as np
from PIL import Image
from queue import Queue
from pathlib import Path
from src.settings import settings
from src.postprocessing.rgroup.schema import Compound

log = logging.getLogger(__name__)


class StorageEntry:
    def __init__(self, data, save_name, data_type="txt", extension=None, session_name="output", save_mode=None):
        """
        初始化存储条目。
        
        :param data: 要存储的数据（可以是字符串、字典或二进制数据）。
        :param session_name: 提交存储请求的会话名称，一般是原图名，作为二级目录。
        :param save_name: 文件保存名称，不需要加后缀。
        :param data_type: 数据类型（默认为 "txt"，支持 "json", "txt", "image"）。
        :param extension: 文件存储类型（默认为空，该参数对json类型无效）。
        :param save_mode: 文件存储的模式，不使用会自动确认，可以选择追加或写入模式。
        """

        self.data = data
        self.session_name = session_name
        self.save_name = save_name
        self.data_type = data_type.lower()  # 文件类型统一转为小写

        # 根据数据类型选择写入模式
        is_binary = self.data_type == "image"
        if save_mode in ('wb', 'w', 'a'):
            self.save_mode = save_mode
        else:
            self.save_mode = 'wb' if is_binary else 'w'
        self.encoding = None if is_binary else 'utf-8'
        self.errors = 'replace' if not is_binary else None
        if not extension:
            if self.data_type == 'json':
                extension = 'json'
            elif self.data_type == 'image':
                extension = 'png'
            elif self.data_type == 'csv':
                extension = 'csv'
            else:
                extension = ''
        self.extension = extension

    def serialize(self):
        """
        根据文件类型序列化数据。

        :return: 序列化后的数据。
        """
        try:
            if self.data_type == "json":
                # 支持字典、列表、字符串等JSON兼容类型 # TODO 简化逻辑
                if isinstance(self.data, list) and len(self.data) > 0 and isinstance(self.data[0], Compound):
                    return json.dumps([cmpd.to_dict() for cmpd in self.data])  # 对Compound做特殊保存
                return json.dumps(self.data, default=str, indent=2)
            elif self.data_type == "txt":
                # 确保文本数据可以转换为字符串
                return str(self.data)
            elif self.data_type == "image":
                return self._serialize_image(self.data)
            elif self.data_type == "csv":
                return self._serialize_csv(self.data)
            else:
                raise ValueError(f"不支持的文件类型: {self.data_type}")
        except Exception as e:
            logging.error(f"序列化失败: {e}")
            raise

    @staticmethod
    def _serialize_csv(csv_data: dict | list):
        """
        将二维数组或字典形式的数据转换为字符串
        Args:
            csv_data (dict | list): 需要存储的数据

        Returns:
            序列化为字符串后的csv数据
        """
        # 创建内存文件对象
        output = io.StringIO()
        # 创建csv写入器
        writer = csv.writer(output, lineterminator='\n')  # 统一使用\n换行

        # 处理不同数据类型
        if isinstance(csv_data, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in csv_data):
            # 二维列表类型数据
            writer.writerows(csv_data)
        elif isinstance(csv_data, (list, tuple)) and all(isinstance(item, dict) for item in csv_data):
            # 字典列表类型数据
            writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        else:
            raise ValueError("CSV数据格式无效，支持二维列表或字典列表")
        return output.getvalue()

    def _serialize_image(self, image_data):
        """
        将图像数据序列化为二进制格式。

        :param image_data: 输入的图像数据（可以是 PIL.Image、cv2 图像或二进制数据）。
        :return: 序列化后的二进制数据。
        """
        if not self.extension:
            self.extension = 'png'  # 默认保存为 PNG 格式

        if isinstance(image_data, bytes):  # 已经是二进制数据
            return image_data
        elif isinstance(image_data, Image.Image):  # PIL.Image 对象
            buffer = io.BytesIO()
            image_data.save(buffer, format=self.extension)
            return buffer.getvalue()
        elif isinstance(image_data, list) or isinstance(image_data, np.ndarray):  # OpenCV 图像（numpy 数组）
            if len(image_data.shape) == 2:  # 单通道图像
                return cv2.imencode(f'.{self.extension}', image_data)[1].tobytes()
            elif len(image_data.shape) == 3 and image_data.shape[2] in [1, 3]:  # 彩色图像
                return cv2.imencode(f'.{self.extension}', image_data)[1].tobytes()
            else:
                raise TypeError("OpenCV 图像数据格式不正确")
        else:
            log.warning(f"image type: {type(image_data)}")
            raise TypeError("图像数据必须为 PIL.Image、cv2 图像或二进制格式")


# 单例模式，辅助存储中间变量
class DataStorage:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._async_queue = Queue()
                cls._instance._start_async_worker()
                cls._instance._initialize(*args, **kwargs)
            return cls._instance

    def _initialize(self, storage_path=os.path.join(settings.BASE_DIR, settings.RESULT_DIR)):
        """初始化方法，用于设置文件输出路径"""
        if not isinstance(storage_path, str):
            raise TypeError("Storage path must be a string.")
        if not storage_path.strip():
            raise ValueError("Storage path cannot be empty or whitespace only.")
        if not os.path.exists(os.path.dirname(storage_path)):
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        if not os.access(os.path.dirname(storage_path), os.W_OK):
            raise PermissionError(f"No write permission for the storage path: {storage_path}")

        self.storage_path = Path(storage_path)
        log.info(f"Singleton DataStorage initialized with storage_path: {self.storage_path}")

    def put_entry(self, storage_entry: StorageEntry):
        """异步存储处理，接受 StorageEntry 对象"""
        if not isinstance(storage_entry, StorageEntry):
            raise TypeError("必须传入 StorageEntry 对象")
        self._async_queue.put(storage_entry)

    def _start_async_worker(self):
        def worker():
            while True:
                try:
                    entry = self._async_queue.get()
                    if entry is None:  # 使用 None 作为停止信号
                        break
                    self._save_to_disk(entry)
                except Exception as e:
                    logging.error(f"工作线程异常: {e}")
                finally:
                    self._async_queue.task_done()  # 标记任务已完成

        self._worker_thread = threading.Thread(target=worker, daemon=False)
        self._worker_thread.start()

    def shutdown(self):
        # 向队列发送停止信号
        self._async_queue.put(None)
        # 等待队列中的所有任务完成
        self._async_queue.join()
        # 等待线程结束(最多等5秒)
        self._worker_thread.join(timeout=5)

    def get_save_dir(self, session_name) -> Path:
        return self.storage_path / self._sanitize_path(session_name)

    def _save_to_disk(self, storage_entry: StorageEntry):
        """按文件类型保存数据（优化编码处理版）"""
        if not self.storage_path:
            raise ValueError("存储路径未正确初始化，请设置有效的存储路径")

        try:
            # 创建带安全字符的路径
            save_dir = self.get_save_dir(storage_entry.session_name)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 生成安全文件名
            safe_save_name = self._sanitize_filename(storage_entry.save_name)
            file_path = save_dir / f"{safe_save_name}.{storage_entry.extension}"

            # 获取序列化数据并验证类型
            serialized_data = storage_entry.serialize()
            self._validate_data_type(serialized_data, storage_entry.data_type)

            # 执行文件写入
            with open(file_path, storage_entry.save_mode, encoding=storage_entry.encoding,
                      errors=storage_entry.errors) as f:
                f.write(serialized_data)
                if storage_entry.save_mode == 'a':
                    f.write('\n')

            logging.info(f"数据成功保存至: {file_path}")

        except Exception as e:
            error_info = f"保存 {storage_entry.data_type} 数据到 {file_path} 失败: {str(e)}"
            log.error(error_info, exc_info=True)
            raise RuntimeError(error_info) from e

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名中的非法字符"""
        # 替换Windows系统保留字符和路径分隔符
        return re.sub(r'[\\/*?:"<>|]', '_', name).strip()

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """清理路径中的非法字符"""
        # 替换路径中的非法字符，保留目录分隔符
        return re.sub(r'[\\*?:"<>|]', '_', path).strip()

    @staticmethod
    def _validate_data_type(data, data_type: str):
        """验证数据类型是否符合预期"""
        is_binary = data_type == 'image'
        if is_binary and not isinstance(data, bytes):
            raise TypeError(f"二进制数据预期为 bytes 类型，实际得到 {type(data)}")
        if not is_binary and not isinstance(data, str):
            raise TypeError(f"文本数据预期为 str 类型，实际得到 {type(data)}")
