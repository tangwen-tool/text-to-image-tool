import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from PIL import Image, ImageSequence

from script2gif.application.script_splitter import split_script
from scripts.local_apng import configure_weight_reader, frame_durations

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QTabWidget, QToolBar  # noqa: E402

from script2gif.infrastructure.app_paths import AppPaths  # noqa: E402
from script2gif.ui.local_apng_page import LocalApngPage  # noqa: E402
from script2gif.ui.main_window import MainWindow  # noqa: E402


def wait_until(predicate):
    deadline = time.monotonic() + 15
    while not predicate():
        assert time.monotonic() < deadline, "Desktop task timed out"
        QTest.qWait(20)


def make_page(tmp_path):
    app = QApplication.instance() or QApplication([])
    page = LocalApngPage()
    page.root = tmp_path
    (tmp_path / "scripts").mkdir()
    shutil.copy2(
        Path(__file__).parent / "fixtures" / "fake_apng.py", tmp_path / "scripts" / "local_apng.py"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    page.model_path.setText(str(model))
    page.python_path.setText(sys.executable)
    return app, page


def test_generate_unicode_prompt_and_animated_preview(tmp_path):
    app, page = make_page(tmp_path)
    prompt = '海浪涌向沙滩，镜头保持固定 "自然光"'
    page.prompt.setPlainText(prompt)
    page.generate()
    assert not page.generate_button.isEnabled()
    wait_until(lambda: page.generate_button.isEnabled())
    assert page.status.text() == "生成完成"
    assert len(page.frames) == 2
    assert page.timer.isActive()
    assert json.loads(page.output.with_suffix(".json").read_text())["prompt"] == prompt
    initial_frame = page.frame_index
    wait_until(lambda: page.frame_index != initial_frame)
    page.toggle_playback()
    assert not page.timer.isActive()
    page.close()


def test_cancel_releases_process_and_allows_retry(tmp_path):
    app, page = make_page(tmp_path)
    page.prompt.setPlainText("WAIT")
    page.generate()
    wait_until(lambda: page.process.state() == QProcess.ProcessState.Running)
    page.cancel()
    wait_until(lambda: page.generate_button.isEnabled())
    assert page.process.state() == QProcess.ProcessState.NotRunning
    assert page.status.text().startswith("已取消")
    page.prompt.setPlainText("海浪")
    page.generate()
    wait_until(lambda: page.generate_button.isEnabled())
    assert page.status.text() == "生成完成"
    page.timer.stop()
    page.close()


def test_window_only_contains_local_apng_and_stops_job_on_close(tmp_path):
    app, page = make_page(tmp_path)
    paths = AppPaths(tmp_path / "config", tmp_path / "cache", tmp_path / "logs", tmp_path / "temp")
    window = MainWindow(paths)
    assert window.centralWidget() is window.local_apng_page
    assert not window.findChildren(QTabWidget)
    assert not window.findChildren(QToolBar)
    window.setCentralWidget(page)
    window.local_apng_page = page
    page.prompt.setPlainText("WAIT")
    page.generate()
    wait_until(lambda: page.process.state() == QProcess.ProcessState.Running)
    window.close()
    assert not page.is_running()
    assert not page.timer.isActive()


def test_resolution_duration_and_hidden_seed_are_passed_to_process(tmp_path):
    app, page = make_page(tmp_path)
    assert not hasattr(page, "seed")
    page.preset.setCurrentIndex(1)
    page.duration.setValue(3.0)
    page.prompt.setPlainText("海浪")
    page.generate()
    wait_until(lambda: page.generate_button.isEnabled())
    report = json.loads(page.output.with_suffix(".json").read_text())
    assert (report["width"], report["height"]) == (832, 480)
    assert report["duration"] == 3.0
    assert report["frames"] == 33
    assert report["seed"] == 42
    page.timer.stop()
    page.close()


@pytest.mark.parametrize("seconds", [0.5, 1.0, 1.7, 3.0, 5.0])
def test_apng_export_matches_requested_duration(tmp_path, seconds):
    count = max(5, 4 * math.ceil((seconds * 10 - 1) / 4) + 1)
    durations = frame_durations(count, 10, seconds)
    frames = [Image.new("RGB", (8, 8), (index * 4, 0, 0)) for index in range(count)]
    output = tmp_path / "timing.png"
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    with Image.open(output) as image:
        assert image.n_frames == count
        assert (
            sum(frame.info["duration"] for frame in ImageSequence.Iterator(image)) == seconds * 1000
        )


@pytest.mark.parametrize(
    ("text", "shots"),
    [
        ("", []),
        ("海浪。\n\n海边有人散步。", ["海浪。", "海边有人散步。"]),
        ("第一行\r\n第二行", ["第一行", "第二行"]),
        ("海浪涌来。镜头拉远！", ["海浪涌来。", "镜头拉远！"]),
        ("Waves roll in. The camera pans.", ["Waves roll in.", "The camera pans."]),
    ],
)
def test_script_splitter(text, shots):
    assert split_script(text) == shots


def test_batch_generates_all_shots_and_continues_after_failure(tmp_path):
    app, page = make_page(tmp_path)
    page.prompt.setPlainText("海浪\n\nFAIL\n\n沙滩")
    page.generate()
    wait_until(lambda: page.generate_button.isEnabled())
    assert [shot["status"] for shot in page.shots] == ["succeeded", "failed", "succeeded"]
    assert page.batch_progress.value() == 3
    assert page.results.count() == 3
    manifest = json.loads((page.batch_dir / "batch.json").read_text(encoding="utf-8"))
    assert manifest["shots"][1]["error"]
    assert manifest["shots"][1]["exit_code"] == 1
    assert "Simulated inference error" in Path(manifest["shots"][1]["log"]).read_text()
    assert (page.batch_dir / "shot-001.png").exists()
    assert (page.batch_dir / "shot-003.png").exists()
    page.results.setCurrentRow(0)
    assert page.output.name == "shot-001.png"
    page.timer.stop()
    page.close()


def test_batch_cancel_preserves_completed_and_skips_remaining(tmp_path):
    app, page = make_page(tmp_path)
    page.prompt.setPlainText("海浪\n\nWAIT\n\n沙滩")
    page.generate()
    wait_until(
        lambda: page.shot_index == 1 and page.process.state() == QProcess.ProcessState.Running
    )
    page.cancel()
    wait_until(lambda: page.generate_button.isEnabled())
    assert [shot["status"] for shot in page.shots] == ["succeeded", "cancelled", "cancelled"]
    assert (page.batch_dir / "shot-001.png").exists()
    assert not (page.batch_dir / "shot-003.png").exists()
    assert not page.is_running()
    page.timer.stop()
    page.close()


def test_windows_weights_remain_valid_after_reader_closes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors")
    tensor_io = pytest.importorskip("safetensors.torch")
    weights = {"weight": torch.arange(128, dtype=torch.float32).reshape(16, 8)}
    path = tmp_path / "weights.safetensors"
    tensor_io.save_file(weights, str(path))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(safetensors, "safe_open", safetensors.safe_open)
    monkeypatch.setattr(tensor_io, "safe_open", tensor_io.safe_open)
    monkeypatch.setattr(tensor_io, "load_file", tensor_io.load_file)
    configure_weight_reader()
    loaded = tensor_io.load_file(str(path), backend="mmap")
    assert torch.equal(loaded["weight"].cpu(), weights["weight"])
    loaded["weight"][0, 0] = -1
    assert tensor_io.load_file(str(path))["weight"][0, 0].item() == 0
