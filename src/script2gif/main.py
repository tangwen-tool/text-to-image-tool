from __future__ import annotations

import logging
import sys

from script2gif.bootstrap import bootstrap


def main() -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    application = QApplication(sys.argv)
    application.setApplicationName("Script2Gif")
    application.setOrganizationName("Script2Gif")
    try:
        result = bootstrap()
        logging.getLogger(__name__).info("application_started mode=local_apng")
        from script2gif.ui.main_window import MainWindow

        window = MainWindow(result.paths)
        window.show()
        return application.exec()
    except Exception as exc:
        QMessageBox.critical(None, "Script2Gif 启动失败", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
