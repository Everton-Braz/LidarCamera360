"""GPS georeferencing view backed by the comprehensive georeferencing and orthophoto engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyproj
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QDoubleSpinBox, QScrollArea, QSplitter,
)
from qfluentwidgets import (
    CardWidget, TitleLabel, CaptionLabel, BodyLabel, SubtitleLabel,
    LineEdit, PrimaryPushButton, PushButton, CheckBox, PlainTextEdit,
    FluentIcon, InfoBar, InfoBarPosition, ProgressBar, ComboBox, ToolButton,
)

from raven_app.georeference import run_georeference, apply_transform_adjustment, export_geojson, georeference_pointcloud, georeference_colmap
from raven_app.orthophoto import generate_orthophoto
from raven_app.views.map_widget import TileMapWidget
from raven_app.i18n import tr


class GeoreferenceWorker(QThread):
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params: dict, operation=run_georeference):
        super().__init__()
        self.params = params
        self.operation = operation

    def run(self):
        try:
            res = self.operation(**self.params)
            self.result.emit(res)
        except Exception as exc:
            self.error.emit(str(exc))


def _export_adjustment(transform, output, cloud, colmap_dir, export_cloud, export_colmap, export_json, prior_report):
    """Heavy re-export work runs outside the Qt event thread."""
    out_dir = Path(output)
    report = dict(prior_report or {})
    deliverables = dict(report.get('deliverables', {}))
    cloud_result = georeference_pointcloud(cloud, transform, out_dir) if cloud and export_cloud else None
    colmap_result = georeference_colmap(colmap_dir, transform, out_dir) if colmap_dir and export_colmap else None
    if cloud_result:
        deliverables['cloud_export'] = cloud_result
    if colmap_result:
        deliverables['colmap_export'] = colmap_result
    if export_json:
        crs_info = {'horizontal_crs': transform['target_horizontal_crs'],
                    'vertical_crs': transform['target_vertical_crs'],
                    'compound_crs': transform['target_compound_crs'], 'crs_name': transform['crs_name']}
        deliverables['geojson_files'] = export_geojson(out_dir,
            colmap_cameras=colmap_result['cameras'] if colmap_result else None,
            cloud_bounds_utm=cloud_result['bounds_utm'] if cloud_result else None, crs_info=crs_info)
    report.update(transform=transform, deliverables=deliverables)
    (out_dir / 'local_to_geographic.json').write_text(json.dumps(transform, indent=2), encoding='utf-8')
    (out_dir / 'georeference_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


class OrthophotoWorker(QThread):
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cloud_path: str, out_path: str, gsd_m: float, transform_matrix: Optional[list] = None, crs_epsg: Optional[int] = None):
        super().__init__()
        self.cloud_path = cloud_path
        self.out_path = out_path
        self.gsd_m = gsd_m
        self.transform_matrix = np.array(transform_matrix) if transform_matrix else None
        self.crs_epsg = crs_epsg

    def run(self):
        try:
            res = generate_orthophoto(
                self.cloud_path,
                self.out_path,
                gsd_m=self.gsd_m,
                transform_matrix=self.transform_matrix,
                crs_epsg=self.crs_epsg,
            )
            self.result.emit(res)
        except Exception as exc:
            self.error.emit(str(exc))


class GeoreferenceView(QWidget):
    """End-to-end GPS Georeferencing, Satellite Map Visualizer, Point Cloud Adjustment & Orthophoto view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GeoreferenceView")
        self._worker = None
        self._ortho_worker = None
        self._latest_report = None
        self._latest_transform = None
        self._cloud_local_corners = None
        self._transformer = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ==========================================
        # LEFT PANE: Controls, Inputs & Report
        # ==========================================
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(24, 20, 24, 20)
        left_layout.setSpacing(12)

        self.header_title = TitleLabel(tr("GPS Georeferencing & Visualizer"))
        self.header_subtitle = CaptionLabel(tr("Worldwide georeferencing, satellite alignment, location adjustment, and orthophoto generator"))
        left_layout.addWidget(self.header_title)
        left_layout.addWidget(self.header_subtitle)

        # 1. Inputs Card
        inputs_card = CardWidget(left_container)
        form = QGridLayout(inputs_card)
        form.setContentsMargins(14, 12, 14, 12)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)

        inputs_title = SubtitleLabel(tr("Input Data"))
        form.addWidget(inputs_title, 0, 0, 1, 3)

        self.insv_label = BodyLabel(tr("INSV video:"))
        self.insv_input = LineEdit()
        self.insv_input.setPlaceholderText(tr("Select an Insta360 .insv file..."))
        self.insv_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.insv_browse.clicked.connect(lambda: self._browse_file(self.insv_input, tr("Select INSV Video"), "Insta360 Video (*.insv *.mp4)"))

        self.cloud_label = BodyLabel(tr("Point Cloud (optional):"))
        self.cloud_input = LineEdit()
        self.cloud_input.setPlaceholderText(tr("Point cloud to georeference (.pcd, .ply, .las, .laz)..."))
        self.cloud_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.cloud_browse.clicked.connect(lambda: self._browse_file(self.cloud_input, tr("Select Point Cloud"), "Point Cloud (*.pcd *.ply *.las *.laz)"))

        self.colmap_label = BodyLabel(tr("COLMAP Dataset (optional):"))
        self.colmap_input = LineEdit()
        self.colmap_input.setPlaceholderText(tr("Select COLMAP sparse directory or root folder..."))
        self.colmap_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.colmap_browse.clicked.connect(self._browse_colmap_dir)

        self.traj_label = BodyLabel(tr("SLAM Trajectory (optional):"))
        self.traj_input = LineEdit()
        self.traj_input.setPlaceholderText(tr("SLAM trajectory text file (.txt)..."))
        self.traj_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.traj_browse.clicked.connect(lambda: self._browse_file(self.traj_input, tr("Select SLAM Trajectory"), "Text Files (*.txt *.csv)"))

        self.output_label = BodyLabel(tr("Output directory:"))
        self.output_input = LineEdit()
        self.output_input.setPlaceholderText(tr("Target directory for georeferenced deliverables..."))
        self.output_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.output_browse.clicked.connect(self._browse_output)

        rows = (
            (self.insv_label, self.insv_input, self.insv_browse),
            (self.cloud_label, self.cloud_input, self.cloud_browse),
            (self.colmap_label, self.colmap_input, self.colmap_browse),
            (self.traj_label, self.traj_input, self.traj_browse),
            (self.output_label, self.output_input, self.output_browse),
        )
        for row_idx, (label, edit, button) in enumerate(rows, start=1):
            form.addWidget(label, row_idx, 0)
            form.addWidget(edit, row_idx, 1)
            form.addWidget(button, row_idx, 2)
        form.setColumnStretch(1, 1)
        left_layout.addWidget(inputs_card)

        # 2. Options Card
        options_card = CardWidget(left_container)
        opt_layout = QGridLayout(options_card)
        opt_layout.setContentsMargins(14, 12, 14, 12)
        opt_layout.setHorizontalSpacing(8)
        opt_layout.setVerticalSpacing(6)

        options_title = SubtitleLabel(tr("Target Projected CRS & Export Options"))
        opt_layout.addWidget(options_title, 0, 0, 1, 3)

        self.crs_label = BodyLabel(tr("Projected CRS:"))
        self.crs_input = LineEdit()
        self.crs_input.setPlaceholderText(tr("Auto-detect (SIRGAS 2000 / WGS 84 UTM) or EPSG:xxxx"))
        opt_layout.addWidget(self.crs_label, 1, 0)
        opt_layout.addWidget(self.crs_input, 1, 1, 1, 2)

        self.export_geojson_check = CheckBox(tr("Export GeoJSON (GPS track, COLMAP cameras, Point cloud footprint)"))
        self.export_geojson_check.setChecked(True)
        opt_layout.addWidget(self.export_geojson_check, 2, 0, 1, 3)

        self.export_cloud_check = CheckBox(tr("Export georeferenced point clouds (LAZ with CRS WKT, LAS, PCD, PRJ)"))
        self.export_cloud_check.setChecked(True)
        opt_layout.addWidget(self.export_cloud_check, 3, 0, 1, 3)

        self.export_colmap_check = CheckBox(tr("Georeference COLMAP dataset (reproject cameras, export CSV and embed EXIF GPS)"))
        self.export_colmap_check.setChecked(True)
        opt_layout.addWidget(self.export_colmap_check, 4, 0, 1, 3)

        left_layout.addWidget(options_card)

        # 3. Interactive Move / Adjust Location & Orientation Card
        adjust_card = CardWidget(left_container)
        adj_layout = QVBoxLayout(adjust_card)
        adj_layout.setContentsMargins(14, 12, 14, 12)
        adj_layout.setSpacing(8)

        adj_title = SubtitleLabel(tr("Move & Adjust Point Cloud Location / Orientation"))
        adj_layout.addWidget(adj_title)

        # Quick Rotate Buttons: 90 deg CCW, 90 deg CW, 180 deg
        btn_rotate_row = QHBoxLayout()
        self.btn_rot_ccw = PushButton(tr("Rotate 90° CCW (↺)"))
        self.btn_rot_ccw.clicked.connect(self._on_rotate_90_ccw)
        btn_rotate_row.addWidget(self.btn_rot_ccw)

        self.btn_rot_cw = PushButton(tr("Rotate 90° CW (↻)"))
        self.btn_rot_cw.clicked.connect(self._on_rotate_90_cw)
        btn_rotate_row.addWidget(self.btn_rot_cw)

        self.btn_flip_180 = PushButton(tr("Flip 180°"))
        self.btn_flip_180.clicked.connect(self._on_flip_180)
        btn_rotate_row.addWidget(self.btn_flip_180)
        adj_layout.addLayout(btn_rotate_row)

        # Heading (Yaw) and Shift Controls
        coord_grid = QGridLayout()
        coord_grid.setHorizontalSpacing(8)
        coord_grid.setVerticalSpacing(6)

        coord_grid.addWidget(BodyLabel(tr("True North Yaw:")), 0, 0)
        self.yaw_spin = QDoubleSpinBox()
        self.yaw_spin.setRange(-360.0, 360.0)
        self.yaw_spin.setDecimals(2)
        self.yaw_spin.setSingleStep(1.0)
        self.yaw_spin.setValue(0.0)
        self.yaw_spin.setSuffix("°")
        self.yaw_spin.valueChanged.connect(self._on_adjustment_changed)
        coord_grid.addWidget(self.yaw_spin, 0, 1)

        coord_grid.addWidget(BodyLabel(tr("Δ Easting (m):")), 1, 0)
        self.shift_e_spin = QDoubleSpinBox()
        self.shift_e_spin.setRange(-100000.0, 100000.0)
        self.shift_e_spin.setDecimals(2)
        self.shift_e_spin.setSingleStep(1.0)
        self.shift_e_spin.setValue(0.0)
        self.shift_e_spin.setSuffix(" m")
        self.shift_e_spin.valueChanged.connect(self._on_adjustment_changed)
        coord_grid.addWidget(self.shift_e_spin, 1, 1)

        coord_grid.addWidget(BodyLabel(tr("Δ Northing (m):")), 2, 0)
        self.shift_n_spin = QDoubleSpinBox()
        self.shift_n_spin.setRange(-100000.0, 100000.0)
        self.shift_n_spin.setDecimals(2)
        self.shift_n_spin.setSingleStep(1.0)
        self.shift_n_spin.setValue(0.0)
        self.shift_n_spin.setSuffix(" m")
        self.shift_n_spin.valueChanged.connect(self._on_adjustment_changed)
        coord_grid.addWidget(self.shift_n_spin, 2, 1)

        coord_grid.addWidget(BodyLabel(tr("Δ Elevation (m):")), 3, 0)
        self.shift_z_spin = QDoubleSpinBox()
        self.shift_z_spin.setRange(-5000.0, 5000.0)
        self.shift_z_spin.setDecimals(2)
        self.shift_z_spin.setSingleStep(0.5)
        self.shift_z_spin.setValue(0.0)
        self.shift_z_spin.setSuffix(" m")
        coord_grid.addWidget(self.shift_z_spin, 3, 1)

        adj_layout.addLayout(coord_grid)

        # Apply adjusted georeference button
        self.apply_adjustment_btn = PushButton(tr("Apply & Save Georeference to Deliverables"), icon=FluentIcon.SYNC)
        self.apply_adjustment_btn.clicked.connect(self._on_apply_adjustment)
        adj_layout.addWidget(self.apply_adjustment_btn)

        left_layout.addWidget(adjust_card)

        # 4. Generate Orthophoto Card
        ortho_card = CardWidget(left_container)
        ortho_layout = QVBoxLayout(ortho_card)
        ortho_layout.setContentsMargins(14, 12, 14, 12)
        ortho_layout.setSpacing(8)

        ortho_title = SubtitleLabel(tr("High-Resolution Orthophoto Generator"))
        ortho_layout.addWidget(ortho_title)

        ortho_controls = QHBoxLayout()
        ortho_controls.addWidget(BodyLabel(tr("Resolution (GSD):")))
        self.gsd_combo = ComboBox()
        self.gsd_combo.addItem("0.02 m/px (2 cm - Ultra High)", 0.02)
        self.gsd_combo.addItem("0.05 m/px (5 cm - Standard)", 0.05)
        self.gsd_combo.addItem("0.10 m/px (10 cm - Fast)", 0.10)
        self.gsd_combo.setCurrentIndex(1)
        ortho_controls.addWidget(self.gsd_combo)

        self.btn_gen_ortho = PushButton(tr("Generate Orthophoto (GeoTIFF + PNG)"), icon=FluentIcon.PALETTE)
        self.btn_gen_ortho.clicked.connect(self._on_generate_orthophoto)
        ortho_controls.addWidget(self.btn_gen_ortho)
        ortho_layout.addLayout(ortho_controls)

        left_layout.addWidget(ortho_card)

        # 5. Primary Action Buttons
        act_box = QHBoxLayout()
        self.run_button = PrimaryPushButton(tr("Run Full Georeference"), icon=FluentIcon.GLOBE)
        self.run_button.setMinimumHeight(38)
        self.run_button.clicked.connect(self._run)
        act_box.addWidget(self.run_button)

        self.open_output_button = PushButton(tr("Open Output Folder"), icon=FluentIcon.FOLDER)
        self.open_output_button.clicked.connect(self._open_output_folder)
        act_box.addWidget(self.open_output_button)

        self.open_webmap_btn = PushButton(tr("Open Web Map"), icon=FluentIcon.VIEW)
        self.open_webmap_btn.clicked.connect(self._open_web_map)
        act_box.addWidget(self.open_webmap_btn)

        left_layout.addLayout(act_box)

        self.progress_bar = ProgressBar(left_container)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # 6. Report Card
        self.report_card = CardWidget(left_container)
        rep_layout = QVBoxLayout(self.report_card)
        rep_layout.setContentsMargins(14, 12, 14, 12)
        rep_layout.setSpacing(6)

        self.report_title = SubtitleLabel(tr("Georeferencing Report & Deliverables"))
        rep_layout.addWidget(self.report_title)

        self.report_text = PlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(130)
        self.report_text.setPlaceholderText(tr("Georeferencing deliverables and transformation report will appear here."))
        rep_layout.addWidget(self.report_text)
        left_layout.addWidget(self.report_card)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(left_container)
        splitter.addWidget(scroll)

        # ==========================================
        # RIGHT PANE: Interactive Satellite Map Widget
        # ==========================================
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        self.map_widget = TileMapWidget(right_container)
        self.map_widget.center_moved.connect(self._on_map_center_moved)
        right_layout.addWidget(self.map_widget)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)

        main_layout.addWidget(splitter)

    @property
    def is_busy(self) -> bool:
        return bool((self._worker is not None and self._worker.isRunning()) or
                    (self._ortho_worker is not None and self._ortho_worker.isRunning()))

    def set_dataset_context(self, *, insv=None, trajectory=None, output=None, colmap_dir=None, cloud=None):
        """Prefill inputs from the active workflow without starting any work."""
        for widget, value in ((self.insv_input, insv), (self.traj_input, trajectory),
                              (self.output_input, output), (self.colmap_input, colmap_dir),
                              (self.cloud_input, cloud)):
            if value and not widget.text().strip():
                widget.setText(str(value))
        if insv and not self.output_input.text().strip() and Path(str(insv)).parent:
            self.output_input.setText(str(Path(str(insv)).parent / "georeferenced"))

    def closeEvent(self, event):
        if self.is_busy:
            for worker in (self._worker, self._ortho_worker):
                if worker is not None and worker.isRunning():
                    worker.requestInterruption()
            event.ignore()
            return
        super().closeEvent(event)

    # File browsing helpers
    def _browse_file(self, target, title, filt):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if path:
            target.setText(path)
            if not self.output_input.text().strip():
                self.output_input.setText(str(Path(path).parent / "georeferenced"))

    def _browse_colmap_dir(self):
        path = QFileDialog.getExistingDirectory(self, tr("Select COLMAP Dataset Directory"))
        if path:
            self.colmap_input.setText(path)
            if not self.output_input.text().strip():
                self.output_input.setText(str(Path(path).parent / "georeferenced"))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, tr("Select Output Directory"))
        if path:
            self.output_input.setText(path)

    def _open_output_folder(self):
        out_dir = self.output_input.text().strip()
        if out_dir and os.path.exists(out_dir):
            os.startfile(out_dir)

    def _open_web_map(self):
        out_dir = self.output_input.text().strip()
        html_map = Path(out_dir) / "georeference_map.html"
        if html_map.is_file():
            os.startfile(str(html_map))
        else:
            InfoBar.warning(
                title=tr("Map not generated yet"),
                content=tr("Run georeferencing first to generate the interactive web map."),
                parent=self,
            )

    # Quick rotate actions
    def _on_rotate_90_ccw(self):
        """Rotate 90 degrees counterclockwise (adds +90 to yaw)."""
        cur = self.yaw_spin.value()
        new_val = (cur + 90.0) % 360.0
        self.yaw_spin.setValue(new_val)

    def _on_rotate_90_cw(self):
        """Rotate 90 degrees clockwise (subtracts 90 from yaw)."""
        cur = self.yaw_spin.value()
        new_val = (cur - 90.0 + 360.0) % 360.0
        self.yaw_spin.setValue(new_val)

    def _on_flip_180(self):
        """Flip 180 degrees."""
        cur = self.yaw_spin.value()
        new_val = (cur + 180.0) % 360.0
        self.yaw_spin.setValue(new_val)

    def _on_adjustment_changed(self):
        """Live update of footprint on map as yaw or coordinate nudges change."""
        if self._latest_transform is None or self._transformer is None:
            return

        adj_tf = apply_transform_adjustment(
            self._latest_transform,
            absolute_yaw_deg=self.yaw_spin.value(),
            delta_easting_m=self.shift_e_spin.value(),
            delta_northing_m=self.shift_n_spin.value(),
            delta_elevation_m=self.shift_z_spin.value(),
        )

        R_2d = np.array(adj_tf["rotation_matrix"])[:2, :2]
        t_2d = np.array(adj_tf["translation_utm_m"])[:2]

        if self._cloud_local_corners is not None:
            corners_utm = (R_2d @ self._cloud_local_corners.T).T + t_2d
            latlons = []
            for cu in corners_utm:
                lon, lat = self._transformer.transform(cu[0], cu[1])
                latlons.append((lat, lon))
            self.map_widget.set_footprint(latlons, heading_deg=adj_tf["yaw_heading_deg"])

    def _on_map_center_moved(self, lat: float, lon: float):
        """Update center coordinates if user double-clicked on map."""
        pass

    def _on_apply_adjustment(self):
        """Re-export georeferenced deliverables and GeoJSON with adjusted yaw and translation."""
        if self.is_busy:
            return
        out_dir_str = self.output_input.text().strip()
        if not out_dir_str or not os.path.exists(out_dir_str):
            InfoBar.warning(title=tr("No output directory"), content=tr("Run georeference first."), parent=self)
            return

        out_dir = Path(out_dir_str)
        tf_json_path = out_dir / "local_to_geographic.json"
        if not tf_json_path.is_file():
            InfoBar.warning(title=tr("No transform found"), content=tr("Run georeference first."), parent=self)
            return

        cur_tf = json.loads(tf_json_path.read_text(encoding="utf-8"))
        adj_tf = apply_transform_adjustment(
            cur_tf,
            absolute_yaw_deg=self.yaw_spin.value(),
            delta_easting_m=self.shift_e_spin.value(),
            delta_northing_m=self.shift_n_spin.value(),
            delta_elevation_m=self.shift_z_spin.value(),
        )

        cloud = self.cloud_input.text().strip()
        colmap_dir = self.colmap_input.text().strip()
        params = dict(transform=adj_tf, output=out_dir, cloud=cloud, colmap_dir=colmap_dir,
                      export_cloud=self.export_cloud_check.isChecked(),
                      export_colmap=self.export_colmap_check.isChecked(),
                      export_json=self.export_geojson_check.isChecked(), prior_report=self._latest_report)
        self._worker = GeoreferenceWorker(params, operation=_export_adjustment)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker_finished)
        self.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_generate_orthophoto(self):
        """Generate high-resolution georeferenced orthophoto from point cloud."""
        if self.is_busy:
            return
        cloud = self.cloud_input.text().strip()
        if not cloud or not Path(cloud).is_file():
            InfoBar.warning(title=tr("Point cloud required"), content=tr("Please specify a valid point cloud file (.pcd, .ply, .las, .laz) to generate an orthophoto."), parent=self)
            return

        out_dir = self.output_input.text().strip()
        if not out_dir:
            out_dir = str(Path(cloud).parent / "georeferenced")

        out_path = Path(out_dir) / "orthophoto.png"
        gsd_m = float(self.gsd_combo.currentData())

        tf_matrix = self._latest_transform.get("matrix_4x4") if self._latest_transform else None
        epsg = 31984
        if self._latest_transform:
            h_crs = self._latest_transform.get("target_horizontal_crs", "")
            if "EPSG:" in h_crs:
                try:
                    epsg = int(h_crs.split(":")[-1])
                except ValueError:
                    pass

        self.btn_gen_ortho.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._ortho_worker = OrthophotoWorker(cloud, str(out_path), gsd_m, tf_matrix, epsg)
        self._ortho_worker.result.connect(self._on_ortho_finished)
        self._ortho_worker.error.connect(self._on_ortho_error)
        self._ortho_worker.finished.connect(self._ortho_worker_finished)
        self.setEnabled(False)
        self._ortho_worker.start()

    def _ortho_worker_finished(self):
        worker, self._ortho_worker = self._ortho_worker, None
        self.setEnabled(True)
        if worker is not None:
            worker.deleteLater()

    def _on_ortho_finished(self, meta):
        self.btn_gen_ortho.setEnabled(True)
        self.progress_bar.setVisible(False)
        InfoBar.success(
            title=tr("Orthophoto Generated!"),
            content=tr("Dimensions: {w}x{h} px at {gsd:.2f} m/px. Saved GeoTIFF and PNG to output folder.").format(
                w=meta.get("width_px"), h=meta.get("height_px"), gsd=meta.get("gsd_m")
            ),
            duration=6000,
            parent=self,
        )

    def _on_ortho_error(self, err):
        self.btn_gen_ortho.setEnabled(True)
        self.progress_bar.setVisible(False)
        InfoBar.error(title=tr("Orthophoto error"), content=err, parent=self)

    # Primary execution
    def _run(self):
        if self.is_busy:
            return
        insv = self.insv_input.text().strip()
        if not insv or not Path(insv).is_file():
            InfoBar.error(title=tr("Missing INSV video"), content=tr("Please select an existing Insta360 .INSV video file."), parent=self)
            return

        output = self.output_input.text().strip()
        if not output:
            output = str(Path(insv).parent / "georeferenced")
            self.output_input.setText(output)

        cloud = self.cloud_input.text().strip() or None
        colmap_dir = self.colmap_input.text().strip() or None
        trajectory = self.traj_input.text().strip() or None
        crs = self.crs_input.text().strip() or None
        yaw = self.yaw_spin.value()

        params = {
            "insv": insv,
            "output": output,
            "cloud": cloud,
            "colmap_dir": colmap_dir,
            "trajectory": trajectory,
            "crs": crs,
            "yaw_deg": yaw,
            "delta_easting_m": self.shift_e_spin.value(),
            "delta_northing_m": self.shift_n_spin.value(),
            "delta_elevation_m": self.shift_z_spin.value(),
            "generate_orthophoto_file": True,
            "ortho_gsd_m": float(self.gsd_combo.currentData()),
            "export_geojson_files": self.export_geojson_check.isChecked(),
            "export_cloud_files": self.export_cloud_check.isChecked(),
            "export_colmap_files": self.export_colmap_check.isChecked(),
        }

        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.report_text.setPlainText(tr("Executing generic georeferencing and satellite map alignment..."))

        self._worker = GeoreferenceWorker(params)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker_finished)
        self.setEnabled(False)
        self._worker.start()

    def _worker_finished(self):
        self.setEnabled(True)
        worker = self._worker
        self._worker = None
        self.run_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        if worker is not None:
            worker.deleteLater()

    def _on_result(self, report):
        self._latest_report = report
        tf = report.get("transform", {})
        self._latest_transform = tf
        gps_sum = report.get("gps_summary", {})
        ref = gps_sum.get("reference_origin_wgs84", {})
        deliv = report.get("deliverables", {})

        yaw_val = float(tf.get("yaw_heading_deg", 0.0))
        self.yaw_spin.blockSignals(True)
        self.yaw_spin.setValue(yaw_val)
        self.yaw_spin.blockSignals(False)

        # Setup coordinate transformer for map preview
        h_crs = tf.get("target_horizontal_crs", "EPSG:3857")
        try:
            self._transformer = pyproj.Transformer.from_crs(h_crs, "EPSG:4326", always_xy=True)
        except Exception:
            self._transformer = None

        # Load cloud bounds or corners for map widget
        cloud_deliv = deliv.get("cloud_export")
        if cloud_deliv and "bounds_utm" in cloud_deliv:
            b = cloud_deliv["bounds_utm"]
            min_e, max_e = b["min_easting_m"], b["max_easting_m"]
            min_n, max_n = b["min_northing_m"], b["max_northing_m"]
            if self._transformer:
                corners_utm = [(min_e, min_n), (max_e, min_n), (max_e, max_n), (min_e, max_n)]
                latlons = [self._transformer.transform(e, n)[::-1] for e, n in corners_utm]
                self.map_widget.set_footprint(latlons, heading_deg=yaw_val)

        # Load cameras on map
        colmap_deliv = deliv.get("colmap_export")
        if colmap_deliv and "cameras" in colmap_deliv:
            self.map_widget.set_cameras(colmap_deliv["cameras"])

        # Update report display
        self.report_text.setPlainText(json.dumps(report, indent=2, ensure_ascii=False))
        InfoBar.success(
            title=tr("Georeference Completed!"),
            content=tr("Deliverables, orthophoto, and interactive satellite map saved successfully."),
            duration=6000,
            parent=self,
        )

    def _on_error(self, error):
        self.report_text.setPlainText(f"Error:\n{error}")
        InfoBar.error(title=tr("Georeference error"), content=error, parent=self)
