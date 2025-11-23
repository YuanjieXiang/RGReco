import argparse
import logging
import os
import weakref

from src.pipeline import Pipeline
from src.settings import settings
from src.storage.storage import DataStorage

# 全局存储：{pipeline_instance -> 原始方法字典}
_original_methods_registry = weakref.WeakKeyDictionary()
# 设置日志等级
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_original_methods(pipeline):
    """
    无侵入地保存 pipeline 的原始方法
    """
    if pipeline in _original_methods_registry:
        return  # 已保存

    original_methods = {
        '_mol_segment': pipeline._mol_segment,
        '_attachment_point_detection': pipeline._attachment_point_detection,
        '_mol_recognize': pipeline._mol_recognize,
        '_mol_re_recognize': pipeline._mol_re_recognize,
        '_text_recognize': pipeline._text_recognize,
        '_table_detect': pipeline._table_detect,
        '_table_recognize': pipeline._table_recognize,
        '_merge_compounds': pipeline._merge_compounds,
    }
    _original_methods_registry[pipeline] = original_methods


def reset_pipeline_methods(pipeline):
    """
    无侵入地恢复 pipeline 的原始方法
    """
    if pipeline not in _original_methods_registry:
        raise ValueError("原始方法未保存，请先调用 save_original_methods(pipeline)")

    original_methods = _original_methods_registry[pipeline]
    for method_name, original_method in original_methods.items():
        setattr(pipeline, method_name, original_method)


def run_with_cli(pipeline: Pipeline, image_path: str):
    # base
    reset_pipeline_methods(pipeline)
    result = pipeline.pipeline(image_path, '0', visualize=True)

    # +GT mol segmentation
    pipeline._mol_segment = lambda *args, **kwargs: None
    result = pipeline.pipeline(image_path, '1')

    # +GT _attachment_point_detection
    pipeline._attachment_point_detection = lambda *args, **kwargs: None
    # result = pipeline.pipeline(image_path, '2')

    # close R-Group identifier detection
    pipeline._mol_re_recognize = lambda *args, **kwargs: None
    # result = pipeline.pipeline(image_path, '3')

    # +GT mol_rec
    pipeline._mol_recognize = lambda *args, **kwargs: None
    # result = pipeline.pipeline(image_path, '4')

    # +GT doc_rec
    pipeline._text_recognize = lambda *args, **kwargs: None
    pipeline._table_detect = lambda *args, **kwargs: None
    pipeline._table_recognize = lambda *args, **kwargs: None
    # result = pipeline.pipeline(image_path, '5')

    # +GT doc parsing & substituent mapping
    pipeline._merge_compounds = lambda *args, **kwargs: None
    # result = pipeline.pipeline(image_path, '6')  # 需特殊处理

def init_settings():
    parser = argparse.ArgumentParser(description="处理命令行参数")
    parser.add_argument("--image_path", type=str, required=False, help="命令行模式时，使用该参数指定图像路径")
    args = parser.parse_args()
    args_dict = vars(args)
    if 'image_path' in args_dict and args_dict['image_path']:
        settings.IMAGE_PATH = args_dict['image_path']


def main():
    init_settings()
    pipeline = Pipeline(lazy_load=True, allow_save=False, visualize=True)
    save_original_methods(pipeline)
    storage = DataStorage()

    try:
        if settings.RUNNING_MODE == 'cli':
            run_with_cli(pipeline, settings.IMAGE_PATH)
        elif settings.RUNNING_MODE == 'test':
            test_dir = settings.TEST_DIR
            names = [name for name in os.listdir(test_dir) if name.endswith('.png')]
            for name in names[:]:
                try:
                    image_path = os.path.join(test_dir, name)
                    run_with_cli(pipeline, image_path)
                except Exception as e:
                    log.exception(f"运行时出错：{e}")
        else:
            log.error('Unknown Mode!')
    except Exception as e:
        log.exception(f"运行时出错：{e}")
    finally:
        storage.shutdown()


if __name__ == '__main__':
    main()
