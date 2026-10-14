from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageSequence
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from script2gif.application.script_splitter import split_script
from script2gif.infrastructure.json_io import atomic_write_json


class LocalApngPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = Path(__file__).resolve().parents[3]
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.started.connect(self._process_started)
        self.process.errorOccurred.connect(self._process_error)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)
        self.frames = []
        self.frame_index = 0
        self.output = None
        self.pending_output = None
        self.cancelled = False
        self.stdout_buffer = ""
        self.stderr_tail = ""
        self.batch_active = False
        self.batch_dir = None
        self.shots = []
        self.shot_index = 0
        self.batch_parameters = {}

        layout = QHBoxLayout(self)
        controls = QWidget()
        controls.setMaximumWidth(460)
        left = QVBoxLayout(controls)
        left.addWidget(QLabel("剧本"))
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("输入完整剧本")
        self.prompt.setMinimumHeight(150)
        left.addWidget(self.prompt, 1)
        form = QFormLayout()
        self.preset = QComboBox()
        self.preset.addItem("512 × 288", (512, 288, 20))
        self.preset.addItem("832 × 480", (832, 480, 30))
        form.addRow("分辨率", self.preset)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 5.0)
        self.duration.setDecimals(1)
        self.duration.setSingleStep(0.1)
        self.duration.setSuffix(" 秒")
        self.duration.setValue(1.7)
        form.addRow("每镜头时长", self.duration)
        self.model_path = QLineEdit(str(self.root / "models" / "wan2.1-t2v-1.3b"))
        form.addRow("模型目录", self._path_row(self.model_path, "选择模型目录", True))
        default_python = self.root / ".venv-wan" / "Scripts" / "python.exe"
        self.python_path = QLineEdit(
            str(default_python if default_python.exists() else sys.executable)
        )
        form.addRow("推理环境", self._path_row(self.python_path, "选择 Python", False))
        left.addLayout(form)
        row = QHBoxLayout()
        self.generate_button = QPushButton("批量生成 APNG")
        self.generate_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.generate_button.clicked.connect(self.generate)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.generate_button)
        row.addWidget(self.cancel_button)
        left.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        left.addWidget(self.progress)
        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("整体进度：未开始")
        left.addWidget(self.batch_progress)
        self.status = QLabel("就绪")
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        layout.addWidget(controls, 2)

        right = QVBoxLayout()
        self.preview = QLabel("暂无动画")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("background: #202522; color: #e5e7eb;")
        right.addWidget(self.preview, 1)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        right.addWidget(self.result_label)
        self.results = QListWidget()
        right.addWidget(QLabel("镜头结果"))
        self.results.setMaximumHeight(150)
        self.results.setMinimumHeight(80)
        self.results.currentRowChanged.connect(self._select_result)
        right.addWidget(self.results)
        actions = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_button.clicked.connect(self.toggle_playback)
        self.save_button = QPushButton("另存为")
        self.save_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.save_button.clicked.connect(self.save_as)
        self.folder_button = QPushButton("打开输出目录")
        self.folder_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.folder_button.clicked.connect(self.open_folder)
        for button in (self.pause_button, self.save_button, self.folder_button):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addStretch()
        right.addLayout(actions)
        layout.addLayout(right, 3)

    def _path_row(self, edit, title, directory):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton()
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        button.setToolTip(title)

        def choose():
            if directory:
                path = QFileDialog.getExistingDirectory(self, title, edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self, title, edit.text(), "Python (python.exe)"
                )
            if path:
                edit.setText(path)

        button.clicked.connect(choose)
        row.addWidget(button)
        return widget

    def is_running(self):
        return self.batch_active or self.process.state() != QProcess.ProcessState.NotRunning

    def generate(self):
        if self.is_running():
            return
        prompt = self.prompt.toPlainText().strip()
        model = Path(self.model_path.text().strip())
        python = Path(self.python_path.text().strip())
        script = self.root / "scripts" / "local_apng.py"
        if not prompt:
            QMessageBox.warning(self, "缺少描述", "请输入要生成的画面描述。")
            return
        if (
            not (model / "model_index.json").is_file()
            or not python.is_file()
            or not script.is_file()
        ):
            QMessageBox.warning(self, "无法开始", "请检查模型目录、推理环境和本地生成脚本。")
            return
        self.shots = [
            {"prompt": part, "status": "pending", "error": ""} for part in split_script(prompt)
        ]
        width, height, steps = self.preset.currentData()
        seconds = self.duration.value()
        frames = max(5, 4 * math.ceil((seconds * 10 - 1) / 4) + 1)
        name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.batch_dir = self.root / "outputs" / "desktop" / name
        self.batch_parameters = {
            "width": width,
            "height": height,
            "steps": steps,
            "seconds": seconds,
            "frames": frames,
            "model": str(model),
            "python": str(python),
            "script": str(script),
        }
        try:
            self.batch_dir.mkdir(parents=True)
            (self.batch_dir / "script.txt").write_text(prompt, encoding="utf-8")
            if not self._persist_batch():
                return
        except OSError as exc:
            QMessageBox.critical(self, "无法保存批次", str(exc))
            return
        self.results.clear()
        for index in range(len(self.shots)):
            self.results.addItem("")
            self._update_result(index)
        self.shot_index = 0
        self.cancelled = False
        self.batch_active = True
        self.batch_progress.setRange(0, len(self.shots))
        self.batch_progress.setFormat("整体进度：%v/%m")
        self.batch_progress.setValue(0)
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        for control in (self.prompt, self.preset, self.duration, self.model_path, self.python_path):
            control.setEnabled(False)
        self._start_shot()

    def _start_shot(self):
        if not self.batch_active or self.cancelled:
            return
        p = self.batch_parameters
        shot = self.shots[self.shot_index]
        shot["status"] = "running"
        self.pending_output = self.batch_dir / f"shot-{self.shot_index + 1:03d}.png"
        shot["output"] = str(self.pending_output)
        self._update_result(self.shot_index)
        if not self._persist_batch():
            return
        arguments = [
            "-X",
            "faulthandler",
            p["script"],
            "generate",
            "--prompt",
            shot["prompt"],
            "--model-dir",
            p["model"],
            "--output",
            str(self.pending_output),
            "--width",
            str(p["width"]),
            "--height",
            str(p["height"]),
            "--frames",
            str(p["frames"]),
            "--steps",
            str(p["steps"]),
            "--seed",
            "42",
            "--duration",
            str(p["seconds"]),
        ]
        self.stdout_buffer = ""
        self.stderr_tail = ""
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        environment.insert("HF_HUB_OFFLINE", "1")
        environment.insert("TRANSFORMERS_OFFLINE", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.root))
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText(self._shot_message("正在启动本地模型…"))
        self.process.start(p["python"], arguments)

    def cancel(self):
        if self.is_running():
            self.cancelled = True
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消…")
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
            else:
                self._finish_batch(cancelled=True)

    def _process_started(self):
        if self.cancelled:
            self.process.kill()

    def _read_stdout(self):
        self.stdout_buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self.stdout_buffer:
            line, self.stdout_buffer = self.stdout_buffer.split("\n", 1)
            if line.startswith("APNG_EVENT "):
                try:
                    self._event(json.loads(line[len("APNG_EVENT ") :]))
                except (ValueError, KeyError):
                    continue

    def _read_stderr(self):
        self.stderr_tail = (
            self.stderr_tail + bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        )[-12000:]

    def _event(self, event):
        stage = event["stage"]
        messages = {
            "encoding": "正在理解画面描述…",
            "loading": "正在加载画面生成模型…",
            "decoding": "正在解码连续画面…",
            "exporting": "正在导出 APNG…",
        }
        if stage in messages:
            self.status.setText(self._shot_message(messages[stage]))
            self.progress.setRange(0, 0)
        elif stage == "generating":
            self.progress.setRange(0, event["total"])
            self.progress.setValue(event["current"])
            self.status.setText(
                self._shot_message(f"正在生成画面：{event['current']}/{event['total']}")
            )

    def _reset_controls(self):
        self.generate_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        for control in (self.prompt, self.preset, self.duration, self.model_path, self.python_path):
            control.setEnabled(True)

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.cancelled = True
            self._finish_batch(cancelled=True)
            self.status.setText("无法启动推理环境")
            QMessageBox.critical(self, "启动失败", self.process.errorString())

    def _finished(self, exit_code, exit_status):
        self._read_stdout()
        self._read_stderr()
        shot = self.shots[self.shot_index]
        shot["exit_code"] = exit_code
        shot["exit_status"] = exit_status.name
        log_path = self.pending_output.with_suffix(".log")
        try:
            log_path.write_text(self.stderr_tail, encoding="utf-8")
            shot["log"] = str(log_path)
        except OSError:
            pass
        if self.cancelled:
            self._finish_batch(cancelled=True)
            return
        elif exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit:
            try:
                self.load_preview(self.pending_output)
                self.shots[self.shot_index]["status"] = "succeeded"
            except Exception as exc:
                self.shots[self.shot_index].update(status="failed", error=str(exc))
        else:
            message = self.stderr_tail[-3000:] or self.process.errorString()
            prefix = f"推理进程退出码：{exit_code}（{exit_status.name}）\n"
            if "access violation" in message.lower():
                prefix = "本地推理发生 Windows 内存访问异常。\n" + prefix
            if "not enough memory" in message.lower():
                prefix = "系统内存或虚拟内存不足。\n" + prefix
            if "out of memory" in message.lower():
                message = (
                    "显存或内存不足，请关闭占用资源的程序并降低分辨率或缩短时长。\n\n" + message
                )
            self.shots[self.shot_index].update(status="failed", error=prefix + message)
        self._update_result(self.shot_index)
        if self.shots[self.shot_index]["status"] == "succeeded":
            self.results.blockSignals(True)
            self.results.setCurrentRow(self.shot_index)
            self.results.blockSignals(False)
        self.batch_progress.setValue(self.shot_index + 1)
        if not self._persist_batch():
            return
        self.shot_index += 1
        if self.shot_index < len(self.shots):
            QTimer.singleShot(0, self._start_shot)
        else:
            self._finish_batch()

    def _shot_message(self, text):
        return f"镜头 {self.shot_index + 1}/{len(self.shots)} · {text}"

    def _persist_batch(self):
        try:
            atomic_write_json(
                self.batch_dir / "batch.json",
                {"parameters": self.batch_parameters, "shots": self.shots},
            )
        except OSError as exc:
            self.batch_active = False
            self._reset_controls()
            self.status.setText("无法保存批次，已停止生成")
            QMessageBox.critical(self, "保存批次失败", str(exc))
            return False
        return True

    def _update_result(self, index):
        shot = self.shots[index]
        statuses = {
            "pending": "待生成",
            "running": "生成中",
            "succeeded": "已完成",
            "failed": "失败",
            "cancelled": "已取消",
        }
        item = self.results.item(index)
        item.setText(f"{index + 1:02d} · {shot['prompt'][:32]} · {statuses[shot['status']]}")
        item.setToolTip(shot["prompt"] + ("\n" + shot["error"] if shot["error"] else ""))

    def _finish_batch(self, cancelled=False):
        self.batch_active = False
        self._reset_controls()
        if cancelled:
            for index, shot in enumerate(self.shots):
                if shot["status"] in ("pending", "running"):
                    shot["status"] = "cancelled"
                    self._update_result(index)
        succeeded = sum(shot["status"] == "succeeded" for shot in self.shots)
        failed = sum(shot["status"] == "failed" for shot in self.shots)
        if cancelled:
            self.status.setText(f"已取消，保留 {succeeded} 个已完成镜头")
        elif failed:
            self.status.setText(f"批量完成：成功 {succeeded}，失败 {failed}")
        else:
            self.status.setText(
                "生成完成" if len(self.shots) == 1 else f"批量完成：{succeeded} 个镜头"
            )
        self.progress.setValue(100 if not cancelled else 0)
        self._persist_batch()

    def _select_result(self, index):
        if 0 <= index < len(self.shots) and self.shots[index]["status"] == "failed":
            shot = self.shots[index]
            QMessageBox.warning(self, f"镜头 {index + 1} 生成失败", shot["error"])
            return
        if 0 <= index < len(self.shots) and self.shots[index]["status"] == "succeeded":
            try:
                self.load_preview(Path(self.shots[index]["output"]))
            except Exception as exc:
                QMessageBox.warning(self, "无法预览", str(exc))

    def load_preview(self, path):
        decoded = []
        with Image.open(path) as image:
            if not image.is_animated:
                raise ValueError("文件不是 APNG 动画")
            dimensions = image.size
            for frame in ImageSequence.Iterator(image):
                rgba = frame.convert("RGBA")
                raw = rgba.tobytes()
                qimage = QImage(
                    raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888
                ).copy()
                decoded.append(
                    (QPixmap.fromImage(qimage), max(10, int(frame.info.get("duration", 100))))
                )
        self.timer.stop()
        self.frames = decoded
        self.frame_index = 0
        self.output = Path(path)
        seconds = sum(duration for _, duration in decoded) / 1000
        self.result_label.setText(
            f"{dimensions[0]} × {dimensions[1]} · {len(decoded)} 帧 · {seconds:.1f} 秒"
        )
        for button in (self.pause_button, self.save_button, self.folder_button):
            button.setEnabled(True)
        self.pause_button.setText("暂停")
        self._draw_frame()
        self.timer.start(self.frames[0][1])

    def _draw_frame(self):
        if self.frames:
            self.preview.setPixmap(
                self.frames[self.frame_index][0].scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _advance_frame(self):
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self._draw_frame()
        self.timer.setInterval(self.frames[self.frame_index][1])

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_button.setText("播放")
        else:
            self.timer.start(self.frames[self.frame_index][1])
            self.pause_button.setText("暂停")

    def save_as(self):
        if self.output:
            destination, _ = QFileDialog.getSaveFileName(
                self, "保存 APNG", self.output.name, "APNG (*.png *.apng)"
            )
            if destination and Path(destination).resolve() != self.output.resolve():
                try:
                    shutil.copy2(self.output, destination)
                except OSError as exc:
                    QMessageBox.critical(self, "保存失败", str(exc))

    def open_folder(self):
        path = self.batch_dir or (self.output.parent if self.output else None)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._draw_frame()
