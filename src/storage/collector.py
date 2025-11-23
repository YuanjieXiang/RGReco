import inspect
import logging
import os.path
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import List, TypeVar, Type

from src.models import *
from src.settings import settings
from .storage import DataStorage, StorageEntry

log = logging.getLogger(__name__)

MODEL = TypeVar("MODEL", bound=BaseModel)


class MoldeOutputCollector:
    def __init__(self):
        self._session_step = None  # 记录当前会话执行了几次保存操作
        self._active = False  # 记录会话状态避免嵌套
        self._storage = DataStorage()  # 单例存储器
        self._current_name = None  # 当前处理的文件名
        self._register_models()

    @contextmanager
    def capture_session(self, session_name):
        """上下文管理器用于捕获整个pipeline运行"""
        if self._active:
            yield
            return

        save_dir = self._storage.get_save_dir(session_name)
        record_file = save_dir / 'session_record.json'
        except_file = save_dir / 'exception.json'
        if record_file.exists():
            os.remove(record_file)
        if except_file.exists():
            os.remove(except_file)

        self._active = True
        self._current_name = session_name
        self._session_step = 0

        try:
            yield
        except Exception as e:
            # 捕获异常并记录
            self._store_exception(e)
            raise  # 重新抛出异常
        finally:
            self._active = False
            self._current_name = None
            self._session_step = 0

    def _store_results(self, data_entries: List[StorageEntry]):
        """根据当前会话信息加工收到的存储节点，并传入存储队列

        Args:
            data_entries (List[StorageEntry]): 存储节点数组，每个节点都是一个要存储的文件
        """
        if not isinstance(self._storage, DataStorage):
            raise TypeError(
                f"Expected '_storage' to be an instance of DataStorage, but got {type(self._storage).__name__}")

        # 使用当前的会话名作存储目录
        for entry in data_entries:
            entry.session_name = self._current_name
            self._storage.put_entry(entry)

    def _store_exception(self, exception):
        """存储异常信息"""
        if not isinstance(self._storage, DataStorage):
            raise TypeError(
                f"Expected '_storage' to be an instance of DataStorage, but got {type(self._storage).__name__}")

        except_info = {
            'type': type(exception).__name__,
            'message': str(exception),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

        self._storage.put_entry(StorageEntry(except_info, 'exception', 'json', session_name=self._current_name, save_mode='a'))

    def _register_models(self):
        self._register_model(MolSegmentor, "MolSeg", visualize=settings.MOL_SEG_VISUALIZE)
        self._register_model(MolRecognizer, "MolRec", visualize=settings.MOL_REC_VISUALIZE)
        self._register_model(TableDetector, "TableDet", visualize=settings.TABLE_DET_VISUALIZE)
        self._register_model(TableRecognizer, "TableRec", visualize=settings.TABLE_REC_VISUALIZE)
        self._register_model(TextDetector, "TextDet", visualize=settings.TEXT_DET_VISUALIZE)
        self._register_model(TextRecognizer, "TextRec", visualize=settings.TEXT_REC_VISUALIZE)
        # self._register_model(MathTextRecognizer, "MathRec", visualize=settings.MATH_REC_VISUALIZE)
        self._register_model(RGroupSegmentor, "RGroupSeg", visualize=settings.RGROUP_DET_VISUALIZE)

    def _register_model(self, model_cls: Type[MODEL], model_name, visualize=True):
        """动态包装指定的实例方法。"""
        assert issubclass(model_cls, BaseModel), f"Class must be BaseModel."
        original_predict = model_cls.predict

        @wraps(original_predict)
        def wrapped_method(*args, **kwargs):
            if not self._active:  # 捕捉对话时才启用装饰方法
                return original_predict(*args, **kwargs)

            # 前置拦截，获取入参
            sig = inspect.signature(model_cls.predict)
            bound_args = sig.bind(*args, **kwargs)
            inputs = bound_args.arguments
            input_images = inputs["images"]

            # 执行原始调用
            output = original_predict(*args, **kwargs)
            data_entries = []

            # 记录这次运行的基本数据
            self._session_step += 1
            session_data = {
                'step': self._session_step,
                'step_name': model_name,
                'data': output,
                'date_time': datetime.now().isoformat(sep=' ', timespec='seconds'),
            }
            session_entry = StorageEntry(session_data, 'session_record', 'json', save_mode='a')
            data_entries.append(session_entry)

            if visualize:
                for i in range(len(input_images)):
                    if output[i]:
                        output_visualize = model_cls.visualize(input_images[i], output[i])
                        data_entries.append(StorageEntry(output_visualize, f"{self._session_step}.{model_name}_{i}", 'image'))

            # 保存识别文件
            self._store_results(data_entries)

            # TODO: UI展示

            return output

        model_cls.predict = wrapped_method
