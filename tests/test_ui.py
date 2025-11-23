import sys
from src.ui.ui_loader import Main

from PyQt5.QtWidgets import QApplication

def test_load_ui():
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())