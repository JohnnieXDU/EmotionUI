import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from ui.mainwindow import MainWindow

dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

if __name__ == '__main__':
    app = QApplication([])

    mainWindow = MainWindow()
    mainWindow.show()

    app.exec()