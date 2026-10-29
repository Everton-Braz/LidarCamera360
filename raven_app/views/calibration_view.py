"""Rig profiles view: pick, edit, create and validate hardware rig profiles."""
import json
import tempfile
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QSizePolicy, QInputDialog,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel, CaptionLabel, StrongBodyLabel,
    PushButton, PrimaryPushButton, ComboBox, LineEdit, DoubleSpinBox, SpinBox, CheckBox,
    PlainTextEdit, TableWidget, FluentIcon, InfoBar, InfoBarPosition, MessageBoxBase,
    SmoothScrollArea
)
from scipy.spatial.transform import Rotation as Rot

from raven_app.i18n import tr
from raven_app.profile_manager import (get_user_profiles_dir, list_available_profiles, resolve_profile,
                                       save_profile_from_calibration, set_default_profile)
from raven_app.rig_profile import (SCANNER_PRESETS, Body, CameraSpec, RigProfile, load_profile, save_profile,
                                   slugify, validate_profile)
from raven_app.camera_models import SUPPORTED_MODELS

# Table columns: camera, body, model, image size, focal/centre, position (cm), mount rotation (deg).
COLUMNS = ('Camera', 'Body', 'Model', 'Width', 'Height', 'fx', 'fy', 'cx', 'cy',
           'X cm', 'Y cm', 'Z cm', 'Roll°', 'Pitch°', 'Yaw°')
_FORWARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_BACKWARD = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def _lens_frame(profile, body_index, camera_index):
    """Rotation of a lens in an upright camera body: the main body's second lens looks backward."""
    return _BACKWARD if body_index == 0 and camera_index == 1 else _FORWARD


def camera_rows(profile):
    """Table rows (strings / numbers) of a profile; rotations shown relative to an upright, forward lens."""
    rows = []
    for b, body in enumerate(profile.bodies):
        for c, camera in enumerate(body.cameras):
            T = np.asarray(camera.T, dtype=float)
            rpy = Rot.from_matrix(T[:3, :3] @ _lens_frame(profile, b, c).T).as_euler('xyz', degrees=True)
            intr = camera.intrinsics
            rows.append([camera.name, body.body_id, camera.model, camera.width, camera.height,
                         intr.get('fx', 0.0), intr.get('fy', 0.0), intr.get('cx', 0.0), intr.get('cy', 0.0),
                         *(T[:3, 3] * 100.0), *rpy])
    return rows


def apply_camera_rows(profile, rows):
    """Write edited table rows back into ``profile`` (same camera order as ``camera_rows``)."""
    index = 0
    for b, body in enumerate(profile.bodies):
        for c, camera in enumerate(body.cameras):
            row = rows[index]
            index += 1
            camera.model = str(row[2]).upper()
            camera.width, camera.height = int(row[3]), int(row[4])
            if camera.model == 'EQUIRECTANGULAR':
                camera.intrinsics = {}
            else:
                camera.intrinsics.update({k: float(v) for k, v in zip(('fx', 'fy', 'cx', 'cy'), row[5:9])})
            T = np.eye(4)
            T[:3, :3] = Rot.from_euler('xyz', [float(v) for v in row[12:15]], degrees=True).as_matrix() \
                @ _lens_frame(profile, b, c)
            T[:3, 3] = np.array([float(v) for v in row[9:12]]) / 100.0
            camera.T = T.tolist()
    return profile


class NewRigDialog(MessageBoxBase):
    """Wizard: scanner, camera from the catalogue and a measured mount -> starting profile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from raven_app.rig_bootstrap import CAMERA_CATALOGUE, CAPTURE_ADVICE, MOUNT_ORIENTATIONS
        self.viewLayout.addWidget(SubtitleLabel(tr('New rig'), self))
        intro = CaptionLabel(tr(
            'Describe your scanner, camera and how the camera sits on the scanner. A ruler measurement '
            'and the closest orientation are enough: the first calibration finds the exact mount and '
            'time offset from the capture itself.'), self)
        intro.setWordWrap(True)
        self.viewLayout.addWidget(intro)
        grid = QGridLayout()
        self.name_edit = LineEdit(self)
        self.name_edit.setPlaceholderText(tr('e.g. Raven + GoPro on pole'))
        self.scanner_combo = ComboBox(self)
        for key, preset in SCANNER_PRESETS.items():
            self.scanner_combo.addItem(preset['model'], userData=key)
        self.camera_combo = ComboBox(self)
        for key, entry in CAMERA_CATALOGUE.items():
            self.camera_combo.addItem(entry['label'], userData=key)
        self.orientation_combo = ComboBox(self)
        for key, entry in MOUNT_ORIENTATIONS.items():
            self.orientation_combo.addItem(tr(entry['label']), userData=key)
        self.offset = [self._spin(-200, 200, ' cm') for _ in range(3)]
        self.rpy = [self._spin(-180, 180, '°') for _ in range(3)]
        self.width_spin = SpinBox(self)
        self.height_spin = SpinBox(self)
        for spin in (self.width_spin, self.height_spin):
            spin.setRange(0, 16384)
            spin.setSpecialValueText(tr('catalogue'))
        rows = [(tr('Profile name'), self.name_edit), (tr('Scanner'), self.scanner_combo),
                (tr('Camera'), self.camera_combo), (tr('Orientation'), self.orientation_combo)]
        for r, (label, widget) in enumerate(rows):
            grid.addWidget(BodyLabel(label, self), r, 0)
            grid.addWidget(widget, r, 1, 1, 3)
        r = len(rows)
        grid.addWidget(BodyLabel(tr('Camera centre X / Y / Z (x forward, y left, z up)'), self), r, 0)
        for i, spin in enumerate(self.offset):
            grid.addWidget(spin, r, 1 + i)
        grid.addWidget(BodyLabel(tr('Extra roll / pitch / yaw'), self), r + 1, 0)
        for i, spin in enumerate(self.rpy):
            grid.addWidget(spin, r + 1, 1 + i)
        grid.addWidget(BodyLabel(tr('Image width / height (0 = catalogue)'), self), r + 2, 0)
        grid.addWidget(self.width_spin, r + 2, 1)
        grid.addWidget(self.height_spin, r + 2, 2)
        self.viewLayout.addLayout(grid)
        advice = CaptionLabel(tr('Capture for the first calibration: ') + tr(CAPTURE_ADVICE), self)
        advice.setWordWrap(True)
        self.viewLayout.addWidget(advice)
        self.yesButton.setText(tr('Create profile'))
        self.cancelButton.setText(tr('Cancel'))
        self.widget.setMinimumWidth(640)

    def _spin(self, low, high, suffix):
        spin = DoubleSpinBox(self)
        spin.setRange(low, high)
        spin.setDecimals(1)
        spin.setSuffix(suffix)
        return spin

    def validate(self):
        return bool(self.name_edit.text().strip())

    def build(self):
        from raven_app.rig_bootstrap import build_profile
        return build_profile(
            self.name_edit.text().strip(), camera=self.camera_combo.currentData(),
            scanner_preset=self.scanner_combo.currentData(),
            offset_cm=tuple(s.value() for s in self.offset),
            orientation=self.orientation_combo.currentData(), extra_rpy_deg=tuple(s.value() for s in self.rpy),
            width=self.width_spin.value() or None, height=self.height_spin.value() or None)


class CalibrationView(QWidget):
    """WinUI Fluent view for hardware rig profiles (scanner + cameras + mount)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CalibrationView")
        self._profile = None
        self._profile_path = None
        self._validation_report = None
        self._validation_result = None
        self._runner = None
        self._init_ui()
        self._populate_profiles_combo()
        self._on_profile_selected(0)

    # -- layout ----------------------------------------------------------------
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.header_title = TitleLabel(tr("Rig Profiles"))
        self.header_subtitle = CaptionLabel(tr(
            "Your scanner, cameras and how they are mounted. The profile is the starting point of every "
            "capture's calibration and the reference that rejects implausible results."))
        self.header_subtitle.setWordWrap(True)
        layout.addWidget(self.header_title)
        layout.addWidget(self.header_subtitle)

        # Profile picker and actions
        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.profile_combo = ComboBox()
        self.profile_combo.setMinimumWidth(320)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        self.btn_new = PrimaryPushButton(tr("New rig..."), icon=FluentIcon.ADD)
        self.btn_new.clicked.connect(self._new_rig)
        self.btn_import = PushButton(tr("Import..."), icon=FluentIcon.DOWNLOAD)
        self.btn_import.clicked.connect(self._import_profile)
        self.btn_export = PushButton(tr("Export..."), icon=FluentIcon.SHARE)
        self.btn_export.clicked.connect(self._export_profile)
        self.btn_default = PushButton(tr("Use as default"), icon=FluentIcon.ACCEPT)
        self.btn_default.clicked.connect(self._set_default)
        for widget in (self.profile_combo, self.btn_new, self.btn_import, self.btn_export, self.btn_default):
            actions.addWidget(widget)
        actions.addStretch()
        layout.addLayout(actions)

        # Profile form
        form_card = CardWidget(self)
        form = QGridLayout(form_card)
        form.setContentsMargins(16, 14, 16, 14)
        form.setHorizontalSpacing(12)
        self.title_profile = SubtitleLabel(tr("Profile"))
        form.addWidget(self.title_profile, 0, 0, 1, 4)
        self.name_edit = LineEdit()
        self.scanner_combo = ComboBox()
        for key, preset in SCANNER_PRESETS.items():
            self.scanner_combo.addItem(preset['model'], userData=key)
        self.scanner_model_edit = LineEdit()
        self.description_edit = LineEdit()
        form.addWidget(BodyLabel(tr("Name")), 1, 0)
        form.addWidget(self.name_edit, 1, 1)
        form.addWidget(BodyLabel(tr("Scanner")), 1, 2)
        form.addWidget(self.scanner_combo, 1, 3)
        form.addWidget(BodyLabel(tr("Scanner model")), 2, 0)
        form.addWidget(self.scanner_model_edit, 2, 1)
        form.addWidget(BodyLabel(tr("Description")), 2, 2)
        form.addWidget(self.description_edit, 2, 3)
        self.status_label = CaptionLabel()
        self.status_label.setWordWrap(True)
        form.addWidget(self.status_label, 3, 0, 1, 4)
        layout.addWidget(form_card)

        # Cameras
        cam_card = CardWidget(self)
        cam_layout = QVBoxLayout(cam_card)
        cam_layout.setContentsMargins(16, 14, 16, 14)
        self.title_cameras = SubtitleLabel(tr("Cameras and mount"))
        cam_layout.addWidget(self.title_cameras)
        hint = CaptionLabel(tr(
            "Position: camera centre in the LiDAR frame. Rotation: relative to an upright camera whose lens "
            "looks forward (the second lens of a dual-fisheye camera looks backward)."))
        hint.setWordWrap(True)
        cam_layout.addWidget(hint)
        self.camera_table = TableWidget(self)
        self.camera_table.setColumnCount(len(COLUMNS))
        self.camera_table.setHorizontalHeaderLabels([tr(c) for c in COLUMNS])
        self.camera_table.verticalHeader().hide()
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.camera_table.setMinimumHeight(170)
        cam_layout.addWidget(self.camera_table)
        cam_actions = QHBoxLayout()
        self.btn_add_aux = PushButton(tr("Add auxiliary camera"), icon=FluentIcon.ADD)
        self.btn_add_aux.clicked.connect(self._add_aux_camera)
        self.btn_remove_aux = PushButton(tr("Remove selected auxiliary camera"), icon=FluentIcon.DELETE)
        self.btn_remove_aux.clicked.connect(self._remove_aux_camera)
        self.btn_save = PrimaryPushButton(tr("Save"), icon=FluentIcon.SAVE)
        self.btn_save.clicked.connect(self._save_profile)
        self.btn_save_as = PushButton(tr("Save as new profile..."), icon=FluentIcon.COPY)
        self.btn_save_as.clicked.connect(self._save_profile_as)
        for widget in (self.btn_add_aux, self.btn_remove_aux):
            cam_actions.addWidget(widget)
        cam_actions.addStretch()
        cam_actions.addWidget(self.btn_save_as)
        cam_actions.addWidget(self.btn_save)
        cam_layout.addLayout(cam_actions)
        self.geometry_label = CaptionLabel()
        self.geometry_label.setWordWrap(True)
        cam_layout.addWidget(self.geometry_label)
        layout.addWidget(cam_card)

        # Validate on a dataset
        val_card = CardWidget(self)
        val_layout = QVBoxLayout(val_card)
        val_layout.setContentsMargins(16, 14, 16, 14)
        self.title_validate = SubtitleLabel(tr("Validate on a capture"))
        val_layout.addWidget(self.title_validate)
        val_hint = CaptionLabel(tr(
            "Choose a processed output folder (images + SLAM). The calibration refines the time offset and the "
            "mount by multi-view colour consistency and keeps the result only if frames it never used agree "
            "better. A new rig needs the coarse search: it finds the mount and clock from how the camera turns."))
        val_hint.setWordWrap(True)
        val_layout.addWidget(val_hint)
        row = QHBoxLayout()
        self.dataset_edit = LineEdit()
        self.dataset_edit.setPlaceholderText(tr("Processed output folder"))
        self.btn_dataset = PushButton(tr("Browse..."), icon=FluentIcon.FOLDER)
        self.btn_dataset.clicked.connect(self._browse_dataset)
        self.coarse_check = CheckBox(tr("Coarse search (new rig)"))
        self.dt_edit = LineEdit()
        self.dt_edit.setPlaceholderText(tr("Time offset (s), optional"))
        self.dt_edit.setMaximumWidth(180)
        self.btn_validate = PrimaryPushButton(tr("Calibrate"), icon=FluentIcon.PLAY)
        self.btn_validate.clicked.connect(self._start_validation)
        self.btn_save_result = PushButton(tr("Save result as profile..."), icon=FluentIcon.SAVE_AS)
        self.btn_save_result.setEnabled(False)
        self.btn_save_result.clicked.connect(self._save_validation_result)
        for widget in (self.dataset_edit, self.btn_dataset, self.dt_edit, self.coarse_check, self.btn_validate,
                       self.btn_save_result):
            row.addWidget(widget)
        val_layout.addLayout(row)
        self.verdict_label = StrongBodyLabel()
        self.verdict_label.setWordWrap(True)
        val_layout.addWidget(self.verdict_label)
        self.validation_log = PlainTextEdit()
        self.validation_log.setReadOnly(True)
        self.validation_log.setMinimumHeight(140)
        self.validation_log.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; font-family: 'Consolas', "
            "'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }")
        val_layout.addWidget(self.validation_log)
        layout.addWidget(val_card)

        # Advanced: raw JSON
        json_card = CardWidget(self)
        json_layout = QVBoxLayout(json_card)
        json_layout.setContentsMargins(16, 14, 16, 14)
        json_head = QHBoxLayout()
        self.title_json = SubtitleLabel(tr("Advanced: profile JSON"))
        self.btn_toggle_json = PushButton(tr("Show"))
        self.btn_toggle_json.clicked.connect(self._toggle_json)
        self.btn_apply_json = PushButton(tr("Apply JSON"), icon=FluentIcon.SYNC)
        self.btn_apply_json.clicked.connect(self._apply_json)
        json_head.addWidget(self.title_json)
        json_head.addStretch()
        json_head.addWidget(self.btn_apply_json)
        json_head.addWidget(self.btn_toggle_json)
        json_layout.addLayout(json_head)
        self.json_edit = PlainTextEdit()
        self.json_edit.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; font-family: 'Consolas', "
            "'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }")
        self.json_edit.setMinimumHeight(320)
        self.json_edit.hide()
        self.btn_apply_json.hide()
        json_layout.addWidget(self.json_edit)
        layout.addWidget(json_card)
        layout.addStretch()

    # -- profiles ----------------------------------------------------------------
    def _populate_profiles_combo(self, select_path=None):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self._available_profiles = list_available_profiles()
        selected = 0
        for i, p in enumerate(self._available_profiles):
            tags = [tr("default")] if p.get('is_default') else []
            if p.get('source') == 'user':
                tags.append(tr("mine"))
            label = p['name'] + (f" ({', '.join(tags)})" if tags else '')
            self.profile_combo.addItem(label, userData=p['path'])
            if select_path and Path(p['path']) == Path(select_path):
                selected = i
        self.profile_combo.setCurrentIndex(selected)
        self.profile_combo.blockSignals(False)
        return selected

    def _on_profile_selected(self, index):
        if not 0 <= index < len(self._available_profiles):
            return
        path = Path(self._available_profiles[index]['path'])
        try:
            self._show_profile(load_profile(path), path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._error(tr("Cannot read profile"), str(exc))

    def _show_profile(self, profile, path=None):
        self._profile = profile
        self._profile_path = Path(path) if path else None
        self.name_edit.setText(profile.name)
        index = self.scanner_combo.findData(profile.scanner.get('preset', 'generic'))
        self.scanner_combo.setCurrentIndex(max(index, 0))
        self.scanner_model_edit.setText(str(profile.scanner.get('model', '')))
        self.description_edit.setText(profile.description)
        self.description_edit.setCursorPosition(0)
        self.description_edit.setToolTip(profile.description)
        self._fill_table(profile)
        self._refresh_status()
        self.json_edit.setPlainText(json.dumps(profile.to_dict(), indent=2))
        bootstrap = (profile.extra.get('bootstrap') or {}).get('needs_coarse_calibration')
        self.coarse_check.setChecked(bool(bootstrap))

    def _fill_table(self, profile):
        rows = camera_rows(profile)
        self.camera_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if c == 2:
                    combo = ComboBox()
                    combo.addItems(list(SUPPORTED_MODELS))
                    combo.setCurrentText(str(value))
                    self.camera_table.setCellWidget(r, c, combo)
                    continue
                # Three decimals, without trailing zeros, so the table fits without scrolling.
                text = value if isinstance(value, str) else (f'{value:.0f}' if c in (3, 4)
                                                             else f'{value:.3f}'.rstrip('0').rstrip('.'))
                item = QTableWidgetItem(text)
                if c < 2:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.camera_table.setItem(r, c, item)

    def _table_rows(self):
        rows = []
        for r in range(self.camera_table.rowCount()):
            row = []
            for c in range(len(COLUMNS)):
                if c == 2:
                    row.append(self.camera_table.cellWidget(r, c).currentText())
                elif c < 2:
                    row.append(self.camera_table.item(r, c).text())
                else:
                    row.append(float(self.camera_table.item(r, c).text().replace(',', '.')))
            rows.append(row)
        return rows

    def form_profile(self):
        """The profile as edited in the form (a copy; raises ValueError on bad input)."""
        if self._profile is None:
            raise ValueError(tr("No profile loaded"))
        profile = RigProfile.from_dict(self._profile.to_dict())
        profile.name = self.name_edit.text().strip() or profile.name
        preset = self.scanner_combo.currentData() or 'generic'
        profile.scanner = {**SCANNER_PRESETS.get(preset, {}), **profile.scanner, 'preset': preset,
                           'model': self.scanner_model_edit.text().strip() or SCANNER_PRESETS[preset]['model']}
        if preset != self._profile.scanner.get('preset'):
            profile.scanner.update(lidar_topic=SCANNER_PRESETS[preset]['lidar_topic'],
                                   imu_topic=SCANNER_PRESETS[preset]['imu_topic'])
        profile.description = self.description_edit.text().strip()
        try:
            apply_camera_rows(profile, self._table_rows())
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(tr("Check the camera table: {error}", error=exc)) from exc
        return profile

    def _refresh_status(self):
        profile = self._profile
        if profile is None:
            return
        source = tr("your profile") if self._profile_path and self._is_user_path(self._profile_path) \
            else tr("bundled profile (saving creates your own copy)")
        errors = validate_profile(profile)
        parts = [source]
        if profile.validated_on:
            parts.append(tr("validated on: ") + ', '.join(str(v.get('dataset')) for v in profile.validated_on[-3:]))
        if (profile.extra.get('bootstrap') or {}).get('needs_coarse_calibration'):
            parts.append(tr("not calibrated yet: run Calibrate with the coarse search on a capture"))
        if errors:
            parts.append(tr("problems: ") + '; '.join(errors))
        self.status_label.setText(' · '.join(parts))
        main = profile.main.cameras
        if len(main) > 1:
            T0, T1 = np.asarray(main[0].T), np.asarray(main[1].T)
            angle = np.degrees(np.linalg.norm(Rot.from_matrix(T0[:3, :3].T @ T1[:3, :3]).as_rotvec()))
            baseline = np.linalg.norm(T0[:3, 3] - T1[:3, 3]) * 100
            self.geometry_label.setText(tr("Lens baseline {baseline:.2f} cm · angle between lenses {angle:.2f}°",
                                           baseline=baseline, angle=angle))
        else:
            self.geometry_label.setText(tr("Lever arm cam0 {lever:.1f} cm",
                                           lever=float(np.linalg.norm(np.asarray(main[0].T)[:3, 3]) * 100)))

    @staticmethod
    def _is_user_path(path):
        try:
            return Path(path).resolve().parent == get_user_profiles_dir().resolve()
        except OSError:
            return False

    def _save_profile(self):
        try:
            profile = self.form_profile()
            if self._profile_path is not None and self._is_user_path(self._profile_path):
                path = self._profile_path
            else:
                path = get_user_profiles_dir() / f'{slugify(profile.profile_id)}.json'
                profile.is_default = False
            path = save_profile(profile, path)
        except (ValueError, OSError) as exc:
            self._error(tr("Profile not saved"), str(exc))
            return
        self._populate_profiles_combo(path)
        self._show_profile(load_profile(path), path)
        self._success(tr("Profile saved"), str(path))

    def _save_profile_as(self):
        name, ok = QInputDialog.getText(self, tr("Save as new profile"), tr("Profile name:"))
        if not ok or not name.strip():
            return
        try:
            profile = self.form_profile()
            profile.name, profile.profile_id, profile.is_default = name.strip(), slugify(name), False
            path = save_profile(profile, get_user_profiles_dir() / f'{profile.profile_id}.json')
        except (ValueError, OSError) as exc:
            self._error(tr("Profile not saved"), str(exc))
            return
        self._populate_profiles_combo(path)
        self._show_profile(load_profile(path), path)
        self._success(tr("Profile saved"), str(path))

    def _new_rig(self):
        dialog = NewRigDialog(self.window())
        if not dialog.exec():
            return
        try:
            profile = dialog.build()
            path = save_profile(profile, get_user_profiles_dir() / f'{profile.profile_id}.json')
        except (ValueError, OSError) as exc:
            self._error(tr("Profile not created"), str(exc))
            return
        self._populate_profiles_combo(path)
        self._show_profile(load_profile(path), path)
        self._success(tr("Profile created"), tr(
            "Process a capture with this profile (rig refinement off), then calibrate it here with the coarse search."))

    def _import_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Import profile or calibration"), "",
                                              "JSON files (*.json);;All files (*.*)")
        if not path:
            return
        try:
            profile = load_profile(path)
            profile.is_default = False
            saved = save_profile(profile, get_user_profiles_dir() / f'{slugify(profile.profile_id)}.json')
        except (ValueError, OSError) as exc:
            self._error(tr("Import failed"), str(exc))
            return
        self._populate_profiles_combo(saved)
        self._show_profile(load_profile(saved), saved)
        self._success(tr("Profile imported"), str(saved))

    def _export_profile(self):
        try:
            profile = self.form_profile()
        except ValueError as exc:
            self._error(tr("Export failed"), str(exc))
            return
        dest, _ = QFileDialog.getSaveFileName(self, tr("Export profile"), f"{profile.profile_id}.json",
                                              "JSON files (*.json)")
        if dest:
            Path(dest).write_text(json.dumps(profile.to_dict(), indent=2), encoding='utf-8')
            self._success(tr("Exported"), dest)

    def _set_default(self):
        if self._profile_path is None:
            return
        try:
            set_default_profile(str(self._profile_path))
        except (ValueError, OSError) as exc:
            self._error(tr("Default not changed"), str(exc))
            return
        if resolve_profile() != self._profile_path:
            self._error(tr("Default not changed"),
                        tr("Another of your profiles is marked as default; save this one as your own copy first."))
            return
        self._populate_profiles_combo(self._profile_path)
        self._success(tr("Default profile"), self._profile.name)

    def _add_aux_camera(self):
        try:
            profile = self.form_profile()
        except ValueError as exc:
            self._error(tr("Camera not added"), str(exc))
            return
        names = {c.name for c in profile.cameras}
        index = 2
        while f'cam{index}' in names:
            index += 1
        name = f'cam{index}'
        from raven_app.rig_bootstrap import CAMERA_CATALOGUE, camera_intrinsics
        entry = CAMERA_CATALOGUE['phone_main']
        T = np.eye(4)
        T[:3, :3] = _FORWARD
        profile.bodies.append(Body(name, [CameraSpec(name, entry['model'], entry['width'], entry['height'],
                                                     camera_intrinsics(entry), T.tolist())],
                                   sync='photometric', dt_tolerance_s=3.0, source='video'))
        self._show_profile(profile, self._profile_path)

    def _remove_aux_camera(self):
        rows = sorted({i.row() for i in self.camera_table.selectedIndexes()})
        if not rows:
            return
        try:
            profile = self.form_profile()
        except ValueError as exc:
            self._error(tr("Camera not removed"), str(exc))
            return
        names = {self.camera_table.item(r, 0).text() for r in rows}
        if any(n in ('cam0', 'cam1') for n in names):
            self._error(tr("Camera not removed"), tr("cam0/cam1 belong to the main camera"))
            return
        profile.bodies = [profile.main] + [
            b for b in profile.auxiliary if not any(c.name in names for c in b.cameras)]
        self._show_profile(profile, self._profile_path)

    def _toggle_json(self):
        visible = not self.json_edit.isVisible()
        if visible:
            try:
                self.json_edit.setPlainText(json.dumps(self.form_profile().to_dict(), indent=2))
            except ValueError:
                pass
        self.json_edit.setVisible(visible)
        self.btn_apply_json.setVisible(visible)
        self.btn_toggle_json.setText(tr("Hide") if visible else tr("Show"))

    def _apply_json(self):
        try:
            profile = RigProfile.from_dict(json.loads(self.json_edit.toPlainText()))
        except (ValueError, KeyError, TypeError) as exc:
            self._error(tr("Invalid profile JSON"), str(exc))
            return
        self._show_profile(profile, self._profile_path)

    # -- validation on a capture -------------------------------------------------
    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, tr("Processed output folder"))
        if path:
            self.dataset_edit.setText(path)

    def validation_args(self, profile_path, report_path):
        """CLI arguments of the calibrate job for the current form state."""
        args = ['calibrate', '--dataset', self.dataset_edit.text().strip(), '--profile', str(profile_path),
                '--report', str(report_path)]
        dt = self.dt_edit.text().strip().replace(',', '.')
        if dt:
            args += ['--dt', str(float(dt))]
        if self.coarse_check.isChecked():
            args.append('--coarse')
        return args

    def _start_validation(self):
        from raven_app.process_runner import ProcessRunner
        dataset = Path(self.dataset_edit.text().strip())
        if not (dataset / 'images' / 'frames.json').is_file():
            self._error(tr("Not a processed capture"), tr("{path} has no images/frames.json", path=dataset))
            return
        try:
            profile = self.form_profile()
            float(self.dt_edit.text().strip().replace(',', '.') or 0)
        except ValueError as exc:
            self._error(tr("Cannot calibrate"), str(exc))
            return
        work = Path(tempfile.mkdtemp(prefix='rig_validate_'))
        profile_path = work / f'{profile.profile_id}.json'
        profile_path.write_text(json.dumps(profile.to_dict(), indent=2), encoding='utf-8')
        self._validation_report = work / 'report.json'
        self._validation_result = None
        self.btn_save_result.setEnabled(False)
        self.validation_log.clear()
        self.verdict_label.setText(tr("Calibrating..."))
        if self._runner is None:
            self._runner = ProcessRunner(self)
            self._runner.log_received.connect(self.validation_log.appendPlainText)
            self._runner.finished.connect(self._on_validation_finished)
        self.btn_validate.setEnabled(False)
        if not self._runner.start_job(self.validation_args(profile_path, self._validation_report),
                                      log_path=work / 'calibrate.log'):
            self.btn_validate.setEnabled(True)
            self._error(tr("Busy"), tr("A calibration is already running"))

    def _on_validation_finished(self, code):
        self.btn_validate.setEnabled(True)
        try:
            summary = json.loads(self._validation_report.read_text(encoding='utf-8'))
        except (OSError, ValueError, AttributeError):
            self.verdict_label.setText(tr("Calibration failed (exit code {code}); see the log.", code=code))
            return
        holdout = summary.get('holdout') or {}
        before = (holdout.get('initial') or {}).get('cost')
        after = (holdout.get('refined') or {}).get('cost')
        accepted = summary.get('quality_status') == 'accepted'
        text = tr("Accepted" if accepted else "Rejected")
        if before is not None and after is not None:
            text += tr(": held-out cost {before:.3f} -> {after:.3f} ({gain:+.1f}%)",
                       before=before, after=after, gain=(1 - after / before) * 100)
        if summary.get('dt_sync_seconds') is not None:
            text += tr(", time offset {dt:.3f} s", dt=summary['dt_sync_seconds'])
        if summary.get('rejection_reason'):
            text += f" — {summary['rejection_reason']}"
        for name, aux in (summary.get('auxiliary') or {}).items():
            text += f" · {name}: {tr(aux.get('quality_status', ''))}"
        self.verdict_label.setText(text)
        if accepted:
            self._validation_result = Path(summary['calibration'])
            self.btn_save_result.setEnabled(True)

    def _save_validation_result(self):
        if self._validation_result is None:
            return
        name, ok = QInputDialog.getText(self, tr("Save result as profile"), tr("Profile name:"),
                                        text=self.name_edit.text().strip())
        if not ok or not name.strip():
            return
        try:
            data = json.loads(self._validation_result.read_text(encoding='utf-8'))
            holdout = (data.get('photometric_report') or {}).get('holdout') or {}
            path = save_profile_from_calibration(
                data, name.strip(), base_profile=str(self._profile_path) if self._profile_path else None,
                validated_on={'dataset': self.dataset_edit.text().strip(),
                              'holdout_cost_before': (holdout.get('initial') or {}).get('cost'),
                              'holdout_cost_after': (holdout.get('refined') or {}).get('cost')})
        except (OSError, ValueError) as exc:
            self._error(tr("Profile not saved"), str(exc))
            return
        self._populate_profiles_combo(path)
        self._show_profile(load_profile(path), path)
        self._success(tr("Profile saved"), str(path))

    # -- helpers -----------------------------------------------------------------
    def _success(self, title, content):
        InfoBar.success(title=title, content=content, orient=Qt.Orientation.Horizontal,
                        position=InfoBarPosition.TOP, duration=3500, parent=self)

    def _error(self, title, content):
        InfoBar.error(title=title, content=content, orient=Qt.Orientation.Horizontal,
                      position=InfoBarPosition.TOP, duration=6000, parent=self)

    def retranslate_ui(self):
        """Update all text elements dynamically when the language changes."""
        self.header_title.setText(tr("Rig Profiles"))
        self.header_subtitle.setText(tr(
            "Your scanner, cameras and how they are mounted. The profile is the starting point of every "
            "capture's calibration and the reference that rejects implausible results."))
        self.title_profile.setText(tr("Profile"))
        self.title_cameras.setText(tr("Cameras and mount"))
        self.title_validate.setText(tr("Validate on a capture"))
        self.title_json.setText(tr("Advanced: profile JSON"))
        self.camera_table.setHorizontalHeaderLabels([tr(c) for c in COLUMNS])
