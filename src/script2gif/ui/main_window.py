from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow

from script2gif.infrastructure.app_paths import AppPaths
from script2gif.ui.local_apng_page import LocalApngPage


class MainWindow(QMainWindow):
    def __init__(self, app_paths: AppPaths) -> None:
        super().__init__()
        self.app_paths = app_paths
        self.setWindowTitle("Script2Gif - 本地 APNG")
        self.resize(1280, 820)
        self.local_apng_page = LocalApngPage(self)
        self.setCentralWidget(self.local_apng_page)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.local_apng_page.is_running():
            self.local_apng_page.cancel()
            self.local_apng_page.process.waitForFinished(5000)
        self.local_apng_page.timer.stop()
        event.accept()
