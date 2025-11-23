import sys
import logging
import os
import argparse
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from src.pipeline import Pipeline
from src.ui.ui_loader import Main, ProcessController
from src.storage.collector import MoldeOutputCollector
from src.storage.storage import DataStorage
from src.settings import settings

# 设置日志等级
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_with_ui(pipeline: Pipeline, collector: MoldeOutputCollector):
    process_controller = ProcessController(pipeline, collector)
    app = QApplication(sys.argv)
    window = Main(process_controller)
    window.show()
    sys.exit(app.exec_())


def run_with_cli(pipeline: Pipeline, collector: MoldeOutputCollector, image_path: str):
    log.info(f"打开图像：{image_path}")
    image_name = Path(image_path).stem

    with collector.capture_session(image_name):  # 启用可视化存储
        result = pipeline.pipeline(image_path, session_name="0", visualize=True)

    result = "管线成功运行完成:)" if result.success else "管线没有成功运行完成:("
    log.info(result)


def run_with_server(pipeline: Pipeline, collector: MoldeOutputCollector):
    # TODO
    pass


def init_settings():
    parser = argparse.ArgumentParser(description="处理命令行参数")
    parser.add_argument("--image_path", type=str, required=False, help="命令行模式时，使用该参数指定图像路径")
    args = parser.parse_args()
    args_dict = vars(args)
    if 'image_path' in args_dict and args_dict['image_path']:
        settings.IMAGE_PATH = args_dict['image_path']


def main():
    init_settings()
    pipeline = Pipeline()
    collector = MoldeOutputCollector()  # 模型输出可视化
    storage = DataStorage()  # 初始化存储器

    try:
        if settings.RUNNING_MODE == 'ui':
            run_with_ui(pipeline, collector)
        elif settings.RUNNING_MODE == 'server':
            run_with_server(pipeline, collector)
        elif settings.RUNNING_MODE == 'cli':
            run_with_cli(pipeline, collector, settings.IMAGE_PATH)
        elif settings.RUNNING_MODE == 'test':
            test_dir = settings.TEST_DIR
            names = [name for name in os.listdir(test_dir) if name.endswith('.png')]
            for name in names[:]:
                try:
                    image_path = os.path.join(test_dir, name)
                    # image_path = os.path.join(test_dir, '2020_01_52-65_07_00.png')
                    run_with_cli(pipeline, collector, image_path)
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
