from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QSize
from PySide6.QtGui import QImageReader, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


class AnnotationError(RuntimeError):
    """Raised when an annotation source cannot be safely opened or saved."""


@dataclass(frozen=True)
class LabelChoice:
    value: str
    caption: str
    shortcut: str


@dataclass(frozen=True)
class PassConfig:
    key: str
    title: str
    guidance: str
    label_column: str
    labels: tuple[LabelChoice, ...]
    row_matches: Callable[[dict[str, str]], bool]
    after_assign: Callable[[dict[str, str], str], None] | None = None


@dataclass(frozen=True)
class AnnotationProfile:
    key: str
    title: str
    columns: tuple[str, ...]
    expected_row_count: int
    default_annotations: Path
    passes: tuple[PassConfig, ...]
    validate_rows: Callable[[list[dict[str, str]]], None]
    unique_row_key: Callable[[dict[str, str]], object]


@dataclass
class AnnotationDocument:
    profile: AnnotationProfile
    path: Path
    image_dir: Path
    rows: list[dict[str, str]]

    @classmethod
    def load(
        cls, profile: AnnotationProfile, path: Path, image_dir: Path
    ) -> "AnnotationDocument":
        if not path.is_file():
            raise AnnotationError(f"Не найден CSV разметки: {path}")
        if not image_dir.is_dir():
            raise AnnotationError(f"Не найдена папка изображений: {image_dir}")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if tuple(reader.fieldnames or ()) != profile.columns:
                    raise AnnotationError(
                        f"Неверная схема CSV. Ожидаются колонки: {', '.join(profile.columns)}."
                    )
                rows = [dict(row) for row in reader]
        except OSError as error:
            raise AnnotationError(f"Не удалось прочитать CSV: {error}") from error
        except csv.Error as error:
            raise AnnotationError(f"Некорректный CSV: {error}") from error

        if len(rows) != profile.expected_row_count:
            raise AnnotationError(
                f"Ожидается ровно {profile.expected_row_count} изображений, найдено {len(rows)}."
            )
        row_keys: set[object] = set()
        for row_number, row in enumerate(rows, start=2):
            image_id = row.get("image_id", "")
            if not image_id:
                raise AnnotationError(f"Строка {row_number}: пустой image_id.")
            row_key = profile.unique_row_key(row)
            if row_key in row_keys:
                raise AnnotationError(f"Строка {row_number}: повторяется строка разметки для {image_id!r}.")
            row_keys.add(row_key)
            if not (image_dir / image_id).is_file():
                raise AnnotationError(f"Строка {row_number}: не найдено изображение {image_id!r}.")
        profile.validate_rows(rows)
        return cls(profile=profile, path=path, image_dir=image_dir, rows=rows)

    def save(self) -> None:
        temporary_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=self.profile.columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(temporary_path, self.path)
        except OSError as error:
            raise AnnotationError(f"Не удалось сохранить CSV: {error}") from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def indices_for_pass(self, pass_config: PassConfig) -> list[int]:
        return [index for index, row in enumerate(self.rows) if pass_config.row_matches(row)]

    def completed_count(self, pass_config: PassConfig) -> int:
        return sum(
            self.rows[index][pass_config.label_column] != ""
            for index in self.indices_for_pass(pass_config)
        )

    def first_unannotated_index(self, pass_config: PassConfig) -> int | None:
        return next(
            (
                index
                for index in self.indices_for_pass(pass_config)
                if self.rows[index][pass_config.label_column] == ""
            ),
            None,
        )

    def next_unannotated_index(self, pass_config: PassConfig, start_after: int) -> int | None:
        indexes = self.indices_for_pass(pass_config)
        current_position = indexes.index(start_after)
        for offset in range(1, len(indexes) + 1):
            index = indexes[(current_position + offset) % len(indexes)]
            if self.rows[index][pass_config.label_column] == "":
                return index
        return None


class ImageStore:
    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir
        self._cache: dict[str, QPixmap] = {}

    def full_image(self, image_id: str) -> QPixmap:
        if image_id not in self._cache:
            reader = QImageReader(str(self.image_dir / image_id))
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                raise AnnotationError(f"Не удалось открыть изображение: {image_id}")
            self._cache[image_id] = QPixmap.fromImage(image)
        return self._cache[image_id]


class ImagePreview(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(620, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background: #14171c; border: 1px solid #373d47; border-radius: 10px; padding: 12px;"
        )

    def set_image(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self.setText("")
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source.isNull():
            self.setPixmap(QPixmap())
            return
        size = QSize(max(1, self.width() - 24), max(1, self.height() - 24))
        self.setPixmap(self._source.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))


class AnnotationWindow(QMainWindow):
    def __init__(self, document: AnnotationDocument) -> None:
        super().__init__()
        self.document = document
        self.profile = document.profile
        self.image_store = ImageStore(document.image_dir)
        self.active_pass = self._initial_pass()
        self.current_index = self._initial_index(self.active_pass)
        self.value_buttons: dict[str, QPushButton] = {}
        self.button_group: QButtonGroup | None = None

        self.setWindowTitle(self.profile.title)
        self.resize(1360, 940)
        self.setMinimumSize(1040, 760)
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.refresh_view("Готово к разметке.")

    def _initial_pass(self) -> PassConfig:
        return next(
            (item for item in self.profile.passes if self.document.first_unannotated_index(item) is not None),
            self.profile.passes[0],
        )

    def _initial_index(self, pass_config: PassConfig) -> int | None:
        indexes = self.document.indices_for_pass(pass_config)
        if not indexes:
            return None
        return self.document.first_unannotated_index(pass_config) or indexes[0]

    def _build_ui(self) -> None:
        root = QWidget()
        root.setStyleSheet(
            """
            QWidget { background: #101318; color: #f4f5f7; font-family: "Segoe UI"; }
            QPushButton { background: #2a3340; border: 1px solid #445066; border-radius: 8px; padding: 10px 16px; }
            QPushButton:hover:!disabled { background: #334055; }
            QPushButton:checked { background: #295a83; border: 2px solid #72b7ff; }
            QPushButton:disabled { color: #768091; background: #1a1e25; border-color: #2d333d; }
            QComboBox { background: #1d2128; border: 1px solid #445066; border-radius: 6px; padding: 7px 10px; min-width: 240px; }
            QComboBox QAbstractItemView { background: #1d2128; selection-background-color: #295a83; }
            QGroupBox { border: 1px solid #464c55; border-radius: 8px; margin-top: 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            """
        )
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel(self.profile.title)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        header.addWidget(title)
        header.addStretch(1)
        self.pass_selector: QComboBox | None = None
        if len(self.profile.passes) > 1:
            header.addWidget(QLabel("Проверка:"))
            self.pass_selector = QComboBox()
            for item in self.profile.passes:
                self.pass_selector.addItem(item.title, item.key)
            self.pass_selector.setCurrentIndex(self.profile.passes.index(self.active_pass))
            self.pass_selector.currentIndexChanged.connect(self.change_pass)
            header.addWidget(self.pass_selector)
        layout.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(16)
        layout.addLayout(content, stretch=1)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(8)
        self.image_id_label = QLabel()
        self.image_id_label.setAlignment(Qt.AlignCenter)
        self.image_id_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        image_layout.addWidget(self.image_id_label)
        self.preview = ImagePreview()
        image_layout.addWidget(self.preview, stretch=1)
        self.position_label = QLabel()
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setStyleSheet("color: #bdc3cf;")
        image_layout.addWidget(self.position_label)
        content.addWidget(image_panel, stretch=3)

        sidebar = QWidget()
        sidebar.setMaximumWidth(440)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(14)
        content.addWidget(sidebar, stretch=1)
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #d6e9ff;")
        sidebar_layout.addWidget(self.progress_label)
        rules_box = QGroupBox("Правила")
        rules_layout = QVBoxLayout(rules_box)
        self.guidance_label = QLabel()
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setTextFormat(Qt.RichText)
        self.guidance_label.setStyleSheet("color: #d5d9e0; font-weight: normal; line-height: 1.25;")
        rules_layout.addWidget(self.guidance_label)
        sidebar_layout.addWidget(rules_box)
        self.value_box = QGroupBox("Значение")
        self.value_layout = QGridLayout(self.value_box)
        self.value_layout.setContentsMargins(12, 22, 12, 12)
        self.value_layout.setHorizontalSpacing(10)
        self.value_layout.setVerticalSpacing(10)
        sidebar_layout.addWidget(self.value_box)
        sidebar_layout.addStretch(1)
        hint = QLabel("Выбор значения сохраняется сразу и открывает следующее неразмеченное изображение.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9fa8b7; font-size: 11px;")
        sidebar_layout.addWidget(hint)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("← Назад")
        self.previous_button.clicked.connect(lambda: self.move_manual(-1))
        navigation.addWidget(self.previous_button)
        self.next_button = QPushButton("Далее →")
        self.next_button.clicked.connect(lambda: self.move_manual(1))
        navigation.addWidget(self.next_button)
        navigation.addStretch(1)
        layout.addLayout(navigation)
        status = QStatusBar()
        status.showMessage("Стрелки — навигация; метки можно вводить клавишами, указанными на кнопках.")
        self.setStatusBar(status)
        self._rebuild_value_buttons()

    def _rebuild_value_buttons(self) -> None:
        while self.value_layout.count():
            item = self.value_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.value_buttons.clear()
        self.button_group = QButtonGroup(self.value_box)
        self.button_group.setExclusive(True)
        for index, choice in enumerate(self.active_pass.labels):
            button = QPushButton(f"{choice.value}  {choice.caption}\nКлавиша: {choice.shortcut}")
            button.setCheckable(True)
            button.setMinimumHeight(58)
            button.clicked.connect(lambda _checked=False, value=choice.value: self.assign_value(value))
            self.button_group.addButton(button)
            self.value_buttons[choice.value] = button
            self.value_layout.addWidget(button, index // 2, index % 2)

    def change_pass(self, selector_index: int) -> None:
        if self.pass_selector is None:
            return
        key = self.pass_selector.itemData(selector_index)
        new_pass = next((item for item in self.profile.passes if item.key == key), None)
        if new_pass is None or new_pass == self.active_pass:
            return
        self.active_pass = new_pass
        self.current_index = self._initial_index(new_pass)
        self._rebuild_value_buttons()
        self.refresh_view()

    def refresh_view(self, message: str | None = None) -> None:
        indexes = self.document.indices_for_pass(self.active_pass)
        self.guidance_label.setText(self.active_pass.guidance)
        if not indexes:
            self.current_index = None
            self.image_id_label.setText("Нет подходящих изображений")
            self.position_label.setText("Сначала отметьте +1 в первом проходе.")
            self.progress_label.setText(f"{self.active_pass.title}: 0 / 0 размечено")
            self.preview.set_image(QPixmap())
            self.preview.setText("Для этого прохода пока нет доступных изображений.")
            for button in self.value_buttons.values():
                button.setEnabled(False)
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            if message:
                self.statusBar().showMessage(message)
            return
        if self.current_index not in indexes:
            self.current_index = self._initial_index(self.active_pass)
        assert self.current_index is not None
        row = self.document.rows[self.current_index]
        position = indexes.index(self.current_index)
        completed = self.document.completed_count(self.active_pass)
        self.image_id_label.setText(row["image_id"])
        self.position_label.setText(f"Изображение {position + 1} / {len(indexes)}")
        self.progress_label.setText(f"{self.active_pass.title}: {completed} / {len(indexes)} размечено")
        current_value = row[self.active_pass.label_column]
        for value, button in self.value_buttons.items():
            button.setEnabled(True)
            button.blockSignals(True)
            button.setChecked(value == current_value)
            button.blockSignals(False)
        self.previous_button.setEnabled(position > 0)
        self.next_button.setEnabled(position < len(indexes) - 1)
        try:
            self.preview.set_image(self.image_store.full_image(row["image_id"]))
        except AnnotationError as error:
            self.show_error(str(error))
        if message:
            self.statusBar().showMessage(message)

    def assign_value(self, value: str) -> None:
        if self.current_index is None or value not in {choice.value for choice in self.active_pass.labels}:
            return
        row = self.document.rows[self.current_index]
        row[self.active_pass.label_column] = value
        if self.active_pass.after_assign is not None:
            self.active_pass.after_assign(row, value)
        try:
            self.document.save()
        except AnnotationError as error:
            self.show_error(str(error))
            return
        next_index = self.document.next_unannotated_index(self.active_pass, self.current_index)
        if next_index is None:
            self.refresh_view(f"Сохранено. Проверка «{self.active_pass.title}» полностью завершена.")
            return
        self.current_index = next_index
        self.refresh_view("Сохранено.")

    def move_manual(self, direction: int) -> None:
        indexes = self.document.indices_for_pass(self.active_pass)
        if self.current_index is None or not indexes:
            return
        position = indexes.index(self.current_index)
        target = position + direction
        if target < 0 or target >= len(indexes):
            return
        self.current_index = indexes[target]
        self.refresh_view()

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if event.type() == QEvent.KeyPress and isinstance(event, QKeyEvent):
            key = event.text().upper()
            for choice in self.active_pass.labels:
                if key == choice.shortcut.upper():
                    self.assign_value(choice.value)
                    return True
            if event.key() == Qt.Key_Left:
                self.move_manual(-1)
                return True
            if event.key() == Qt.Key_Right:
                self.move_manual(1)
                return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)


def run_profile(profile: AnnotationProfile, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=profile.title)
    parser.add_argument("--annotations", type=Path, default=profile.default_annotations, help="Путь к CSV разметки.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_DIR / "images", help="Папка с исходными изображениями.")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(profile.title)
    try:
        document = AnnotationDocument.load(profile, args.annotations.resolve(), args.image_dir.resolve())
    except AnnotationError as error:
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Ошибка")
        box.setText(str(error))
        box.exec()
        return 1
    window = AnnotationWindow(document)
    window.show()
    return app.exec()
