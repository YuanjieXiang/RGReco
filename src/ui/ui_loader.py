import os
import logging
import threading
import datetime
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from .dialog import Ui_Dialog
from .mainwindow import Ui_MainWindow

log = logging.getLogger(__name__)

# 新增处理控制器（减少耦合）
class ProcessController(QObject):
    result_ready = pyqtSignal(object)  # 处理结果信号
    progress_updated = pyqtSignal(int)  # 进度更新信号
    def __init__(self, pipeline, collector):
        super().__init__()
        self.pipeline = pipeline
        self.collector = collector

    def start_process(self, image_path):
        thread = threading.Thread(
            target=self._async_process,
            args=(image_path,),
            daemon=True
        )
        thread.start()

    def _async_process(self, image_path):
        log.info(f"打开图像：{image_path}")
        image_name = Path(image_path).stem
        # 如已注册，将启用被包装方法(捕获模型输入输出并可视化存储)
        with self.collector.capture_session(image_name):
            success = self.pipeline.pipeline(image_path, visualize=True)
        result = "管线成功运行完成:)" if success else "管线没有成功运行完成:("
        log.info(result)
        # 结果返回给ui程序
        self.result_ready.emit(result)

class Bbox(object):
    def __init__(self):
        self._x1, self._y1 = 0, 0
        self._x2, self._y2 = 0, 0

    @property
    def point1(self):
        return self._x1, self._y1

    @point1.setter
    def point1(self, position: tuple):
        self._x1 = position[0]
        self._y1 = position[1]

    @property
    def point2(self):
        return self._x2, self._y2

    @point2.setter
    def point2(self, position: tuple):
        self._empty = False
        self._x2 = position[0]
        self._y2 = position[1]

    @property
    def bbox(self):
        if self._x1 < self._x2:
            x_min, x_max = self._x1, self._x2
        else:
            x_min, x_max = self._x2, self._x1

        if self._y1 < self._y2:
            y_min, y_max = self._y1, self._y2
        else:
            y_min, y_max = self._y2, self._y1
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    def __str__(self):
        return str(self.bbox)

class ScreenLabel(QLabel):
    signal = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self._press_flag = False
        self._bbox = Bbox()
        self._pen = QPen(Qt.white, 2, Qt.DashLine)
        self._painter = QPainter()
        self._bbox = Bbox()
        
        height = QApplication.desktop().screenGeometry().height()
        width = QApplication.desktop().screenGeometry().width()
        self._pixmap = QPixmap(width, height)
        self._pixmap.fill(QColor(255, 255, 255))
        self.setPixmap(self._pixmap)
        self.setWindowOpacity(0.4)

        self.setAttribute(Qt.WA_TranslucentBackground, True)  # 设置背景颜色为透明

        QShortcut(QKeySequence("esc"), self, self.close)

        self.setWindowFlag(Qt.Tool)  # 不然exec_执行退出后整个程序退出

        # palette = QPalette()
        # palette.
        # self.setPalette()

    def _draw_bbox(self):
        pixmap = self._pixmap.copy()
        self._painter.begin(pixmap)
        self._painter.setPen(self._pen)  # 设置pen必须在begin后
        rect = QRect(*self._bbox.bbox)
        self._painter.fillRect(rect, Qt.SolidPattern)  # 区域不透明
        self._painter.drawRect(rect)  # 绘制虚线框
        self._painter.end()
        self.setPixmap(pixmap)
        self.update()
        self.showFullScreen()

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            log.info(f"鼠标左键：[{QMouseEvent.x()}, {QMouseEvent.y()}]")
            self._press_flag = True
            self._bbox.point1 = [QMouseEvent.x(), QMouseEvent.y()]

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton and self._press_flag:
            log.info("鼠标释放：[{QMouseEvent.x()}, {QMouseEvent.y()}]")
            self._bbox.point2 = [QMouseEvent.x(), QMouseEvent.y()]
            self._press_flag = False
            self.signal.emit(QRect(*self._bbox.bbox))

    def mouseMoveEvent(self, QMouseEvent):
        if self._press_flag:
            # log.info("鼠标移动：", [QMouseEvent.x(), QMouseEvent.y()])
            self._bbox.point2 = [QMouseEvent.x(), QMouseEvent.y()]
            self._draw_bbox()

class ShotDialog(QDialog, Ui_Dialog):
    def __init__(self, parent, rect, process_controller):
        super().__init__()
        # 新增
        self.parent = parent
        self.process_controller = process_controller

        self.setupUi(self)
        self.adjustSize()
        self.setWindowFlag(Qt.FramelessWindowHint)  # 没有窗口栏
        # self.setAttribute(Qt.WA_TranslucentBackground)  # 设置背景透明

        self.pushButton_save.clicked.connect(self.save_local)
        self.pushButton_cancel.clicked.connect(self.close)

        # self.label_shot.setPixmap(QApplication.primaryScreen().grabWindow(0).copy(rect))
        self.setWindowFlag(Qt.Tool)  # 不然exec_执行退出后整个程序退出
        # 自动创建 imgsave 文件夹
        self.save_dir = "imgsave"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 获取屏幕设备像素比
        screen = QApplication.primaryScreen()
        dpr = screen.devicePixelRatio()
        # 调整截图区域为物理像素尺寸
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        physical_rect = QRect(int(x * dpr), int(y * dpr), int(w * dpr), int(h * dpr))
        # 截取高分辨率图像并设置设备像素比
        pixmap = screen.grabWindow(0, physical_rect.x(), physical_rect.y(), physical_rect.width(), physical_rect.height())
        pixmap.setDevicePixelRatio(dpr)  # 关键：保持逻辑尺寸不变，物理像素足够
        self.label_shot.setPixmap(pixmap)

    def get_shot_img(self):
        return self.label_shot.pixmap().toImage()

    def get_shot_bytes(self):
        shot_bytes = QByteArray()
        buffer = QBuffer(shot_bytes)
        buffer.open(QIODevice.WriteOnly)
        shot_img = self.get_shot_img()
        shot_img.save(buffer, 'png')
        return shot_bytes.data()

    def save_local(self):
        # 使用当前时间戳生成唯一文件名
        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        filepath = os.path.join(self.save_dir, filename)

        # 保存截图
        shot_img = self.get_shot_img()
        if shot_img.save(filepath, "PNG", quality=100):
        #  if shot_img.save(filepath, "PNG"):
            QMessageBox.information(self, "保存成功", f"截图已保存到: {filepath}")
            filepath=filepath.replace('\\','/')
            if self.process_controller:
                self.process_controller.start_process(filepath)
            else:
                log.info("没有加载process_controller")
            self.close()  # 关闭截图对话框
            self.parent.show()  # 恢复主窗口，准备下一次截图
        else:
            QMessageBox.critical(self, "保存失败", "无法保存截图，请检查文件夹权限。")


    # def save_to_clipboard(self):
    #     shot_bytes = self.get_shot_bytes()
    #     OpenClipboard()
    #     EmptyClipboard()
    #     SetClipboardData(CF_BITMAP, shot_bytes[14:])
    #     CloseClipboard()
    #     self.close()

    def showMessage(self, message):
        dialog = QDialog()
        dialog.adjustSize()
        text = QLineEdit(message, dialog)
        text.adjustSize()
        dialog.exec_()

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, process_controller: ProcessController=None):
        super().__init__()
        # 新增
        self.process_controller = process_controller
        if process_controller:
            self.process_controller.result_ready.connect(self.show_result)

        self.adjustSize()
        self.setupUi(self)
        self.setWindowTitle("截图工具")
        self.setWindowIcon(QIcon('./icon/cut.png'))
        self.screen = QApplication.primaryScreen()
        # 托盘行为
        self.action_quit = QAction("退出", self, triggered=self.close)
        self.action_show = QAction("主窗口", self, triggered=self.show)
        self.menu_tray = QMenu(self)
        self.menu_tray.addAction(self.action_quit)
        # 设置最小化托盘
        self.tray = QSystemTrayIcon(QIcon('./icon/screenshot.png'), self)  # 创建系统托盘对象
        self.tray.activated.connect(self.shot)  # 设置托盘点击事件处理函数
        self.tray.setContextMenu(self.menu_tray)
        # 快捷键
        QShortcut(QKeySequence("F1"), self, self.shot)
        # 信号与槽
        self.pushButton_shot.clicked.connect(self.shot)
        self.pushButton_exit.clicked.connect(self.close)

    def shot(self):
        """开始截图"""
        self.hide()
        # time.sleep(0.2)  # 保证隐藏窗口
        # pixmap = self.screen.grabWindow(0)
        # painter = QPainter()
        # painter.setOpacity(0.5)
        # painter.begin(pixmap)
        # painter.end()

        self.label = ScreenLabel()
        # self.label.setPixmap(pixmap)
        self.label.showFullScreen()
        self.label.signal.connect(self.callback)

    def callback(self, pixmap):
        """截图完成回调函数"""
        self.label.close()
        del self.label  # del前必须先close
        dialog = ShotDialog(self, pixmap, self.process_controller)
        dialog.exec_()
        if not self.isMinimized():
            self.show()  # 截图完成显示窗口

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange and self.isMinimized():
            self.tray.showMessage("通知", "已最小化到托盘，点击开始截图")
            self.tray.show()
            self.hide()

    def closeEvent(self, event):
        reply = QMessageBox.information(self, "消息", "是否退出程序", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
            log.info("修改配置文件")
        else:
            event.ignore()

    # 新增
    def show_result(self, result):
        """处理结果回调"""
        log.info(f"接受到返回的结果：{result}")