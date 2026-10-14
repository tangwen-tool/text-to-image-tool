"""Exercise the real desktop generation workflow without opening a visible window."""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from script2gif.infrastructure.app_paths import AppPaths  # noqa: E402
from script2gif.ui.main_window import MainWindow  # noqa: E402

root = Path(__file__).resolve().parents[1]
parser = ArgumentParser(description=__doc__)
parser.add_argument("--batch", action="store_true")
parser.add_argument("--render-only", action="store_true")
parser.add_argument("--script-file", type=Path)
options = parser.parse_args()
qa = root / "outputs" / "desktop-qa"
paths = AppPaths(qa / "config", qa / "cache", qa / "logs", qa / "temp")
paths.ensure()
app = QApplication(sys.argv)
font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
if font_path.exists():
    QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei UI", 10))
window = MainWindow(paths)
window.show()
page = window.local_apng_page
page.prompt.setPlainText("真实摄影风格，阳光下海浪缓缓涌向沙滩，镜头固定，自然光线。")
if options.batch:
    page.prompt.setPlainText(
        "真实摄影风格，阳光下海浪涌向沙滩。\n\n真实摄影风格，海面波光粼粼，镜头固定。"
    )
    page.duration.setValue(0.5)
if options.script_file:
    page.prompt.setPlainText(options.script_file.read_text(encoding="utf-8"))
app.processEvents()
window.grab().save(str(qa / "desktop-ready.png"))
page.load_preview(root / "outputs" / "beach-480p.png")
app.processEvents()
window.grab().save(str(qa / "desktop-preview.png"))
if options.render_only:
    raise SystemExit(0)
page.generate()


def check_result():
    if page.is_running() or not page.generate_button.isEnabled():
        return
    poll.stop()
    if any(shot["status"] != "succeeded" for shot in page.shots):
        print(page.status.text(), flush=True)
        app.exit(1)
        return
    assert page.output.parent.parent.name == "desktop"
    assert len(page.frames) == (5 if options.batch else 17)
    assert page.timer.isActive()
    window.grab().save(str(qa / "desktop-generated.png"))
    print(f"Desktop real generation OK: {page.output}", flush=True)
    app.exit(0)


poll = QTimer()
poll.timeout.connect(check_result)
poll.start(100)
QTimer.singleShot(1800000, lambda: (page.cancel(), app.exit(2)))
raise SystemExit(app.exec())
