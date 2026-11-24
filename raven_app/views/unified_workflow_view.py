"""Unified processing workflow view."""
import json
import re
import time
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QFrame, QDialog,
    QMessageBox, QSlider, QSizePolicy
)
from qfluentwidgets import (
    CardWidget, ElevatedCardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, DoubleSpinBox, SpinBox, ComboBox, CheckBox, SwitchButton,
    ProgressBar, PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition,
    SmoothScrollArea, FlowLayout
)
from raven_app.process_runner import ProcessRunner, make_run_log_path
from raven_app.workflow_progress import (
    STAGE_NAMES, add_stage_ratio, dataset_times_path, estimate_stage_seconds, load_stage_history,
    load_stage_ratios, machine_rates_path, parse_progress_line, save_stage_history,
    save_stage_ratios, stage_duration_estimates,
)
from raven_app.mask_download import MaskResourceDownload
from raven_app.mask_resources import resource_status, resources_ready
from raven_app.bag_io import detect_bag_topics, find_adjacent_image_bag, find_adjacent_lidar_bag
from raven_app.seed_export import dataset_seed_point_count, seed_point_count
from raven_app.i18n import tr, tr_button
from raven_app.config import DEFAULT_OPERATOR_RADIUS_M, get_openmvs_bin, load_config
from raven_app.openmvs_mesh import has_programs

RIG_REFINE_CHOICES = (
    ("Photometric (recommended)", "photometric"),
    ("Time offset only", "dt"),
    ("Off", "none"),
)

DENSE_QUALITY_CHOICES = (
    ("Draft (1024 px)", "draft"),
    ("Balanced (1664 px)", "balanced"),
    ("High (2496 px)", "high"),
    ("Full (3328 px)", "full"),
)


class UnifiedWorkflowView(QWidget):
    """WinUI Fluent view providing single-screen end-to-end processing.

    Inputs: ROS Bag (.bag) + Insta360 Video (.insv) + Output Folder
    Workflow: Video Extraction -> SLAM -> Gyro Sync -> Colorization -> Deliverables
    Outputs: .PLY, .PCD, and COLMAP Dataset ready for 3DGS Training.
    """

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("UnifiedWorkflowView")
        self.runner = runner
        self._mask_downloader = None
        self.mask_config = None
        self.third_camera_configs = []
        self._next_aux_camera_index = 2
        self._current_log_path = None
        self._workflow_started_at = None
        self._stage_started_at = None
        self._stage_root = None
        self._current_stage = 0
        self._current_stage_skipped = False
        self._stage_fraction = 0.0
        self._stage_fraction_from_log = False
        self._seed_full_point_count = None
        self._seed_actual_export_count = None
        self._seed_actual_export_percent = None
        self._stage_history = [[] for _ in STAGE_NAMES]
        self._stage_estimates = [60.0] * len(STAGE_NAMES)
        self._stage_defaults = [60.0] * len(STAGE_NAMES)
        self._dataset_ratios = [[] for _ in STAGE_NAMES]
        self._machine_ratios = [[] for _ in STAGE_NAMES]
        self._estimate_inputs = None
        self._recorded_stages = set()
        self._log_fraction = 0.0
        self._log_fraction_start = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_workflow_progress)
        self._init_ui()
        self._connect_signals()
        self._refresh_mask_resource_state()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area to comfortably accommodate all cards
        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # ----------------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------------
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        self.header_title = TitleLabel(tr("Unified LiDAR-Camera Studio"))
        self.header_subtitle = CaptionLabel(
            tr("Automated end-to-end pipeline: Inputs Selection → FAST-LIVO2 SLAM → "
               "Time Sync → Point Cloud Colorization → 3DGS Deliverables")
        )
        header_layout.addWidget(self.header_title)
        self.header_subtitle.setWordWrap(True)
        header_layout.addWidget(self.header_subtitle)
        layout.addLayout(header_layout)

        # ----------------------------------------------------------------------
        # 1. Dataset Inputs Card
        # ----------------------------------------------------------------------
        inputs_card = CardWidget(content)
        inputs_layout = QVBoxLayout(inputs_card)
        inputs_layout.setContentsMargins(20, 18, 20, 18)
        inputs_layout.setSpacing(12)

        self.inputs_title = SubtitleLabel(tr("1. Input Datasets & Destination"))
        inputs_layout.addWidget(self.inputs_title)

        # Row 1: Bag File
        bag_row = QHBoxLayout()
        self.bag_label = BodyLabel(tr("LiDAR ROS Bag (.bag):"))
        self.bag_label.setFixedWidth(170)
        self.bag_input = LineEdit()
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.btn_browse_bag = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.btn_browse_bag.clicked.connect(self._browse_bag)
        bag_row.addWidget(self.bag_label)
        bag_row.addWidget(self.bag_input)
        bag_row.addWidget(self.btn_browse_bag)
        inputs_layout.addLayout(bag_row)

        # Row 1.5: Front Camera Bag
        image_bag_row = QHBoxLayout()
        self.image_bag_label = BodyLabel(tr("Front Camera Bag (.bag):"))
        self.image_bag_label.setFixedWidth(170)
        self.image_bag_input = LineEdit()
        self.image_bag_input.setPlaceholderText(tr("Optional IMAGE_*.bag for front camera & VIO..."))
        self.image_bag_input.textChanged.connect(self._on_image_bag_changed)
        self.btn_browse_image_bag = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.btn_browse_image_bag.clicked.connect(self._browse_image_bag)
        self.use_bag_images_chk = CheckBox(tr_button("Use Bag images in Colorization & 3DGS"))
        self.use_bag_images_chk.setChecked(False)
        self.use_bag_images_chk.setToolTip(tr("Extract and use frames from the image bag for point cloud colorization and 3DGS dataset export."))
        self.bag_camera_info_label = CaptionLabel("")
        self.bag_camera_info_label.setStyleSheet("color: #888888;")
        self.bag_camera_info_label.setIndent(170 + inputs_layout.spacing())
        self.bag_camera_info_label.setVisible(False)
        image_bag_row.addWidget(self.image_bag_label)
        image_bag_row.addWidget(self.image_bag_input)
        image_bag_row.addWidget(self.btn_browse_image_bag)
        image_bag_row.addWidget(self.use_bag_images_chk)
        inputs_layout.addLayout(image_bag_row)
        inputs_layout.addWidget(self.bag_camera_info_label)

        # Row 2: INSV Video
        insv_row = QHBoxLayout()
        self.insv_label = BodyLabel(tr("Insta360 Video (.insv):"))
        self.insv_label.setFixedWidth(170)
        self.insv_input = LineEdit()
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 / X6 video (.insv or .mp4)..."))
        self.btn_browse_insv = PushButton(tr("Browse"), icon=FluentIcon.VIDEO)
        insv_row.addWidget(self.insv_label)
        insv_row.addWidget(self.insv_input)
        insv_row.addWidget(self.btn_browse_insv)
        inputs_layout.addLayout(insv_row)

        # Row 2.1: Dual INSV option (Insta360 ONE RS 360)
        dual_insv_opt_row = QHBoxLayout()
        dual_insv_spacer = BodyLabel("")
        dual_insv_spacer.setFixedWidth(170)
        self.dual_insv_chk = CheckBox(tr("Allow select two INSV files (Insta360 ONE RS 360)"))
        self.dual_insv_chk.setChecked(False)
        self.dual_insv_chk.setToolTip(tr("Enable support for dual-file 360 cameras where front and rear lenses are saved in separate .insv files."))
        dual_insv_opt_row.addWidget(dual_insv_spacer)
        dual_insv_opt_row.addWidget(self.dual_insv_chk)
        dual_insv_opt_row.addStretch()
        inputs_layout.addLayout(dual_insv_opt_row)

        # Row 2.2: Secondary INSV file (Rear lens _10_)
        self.secondary_insv_container = QWidget()
        secondary_insv_container_layout = QVBoxLayout(self.secondary_insv_container)
        secondary_insv_container_layout.setContentsMargins(0, 0, 0, 0)
        secondary_insv_container_layout.setSpacing(4)

        secondary_insv_row = QHBoxLayout()
        secondary_insv_label = BodyLabel(tr("Second Lens (.insv):"))
        secondary_insv_label.setFixedWidth(170)
        self.secondary_insv_input = LineEdit()
        self.secondary_insv_input.setPlaceholderText(tr("Secondary .insv file (rear lens _10_)"))
        self.btn_browse_secondary_insv = PushButton(tr("Browse"), icon=FluentIcon.VIDEO)
        secondary_insv_row.addWidget(secondary_insv_label)
        secondary_insv_row.addWidget(self.secondary_insv_input)
        secondary_insv_row.addWidget(self.btn_browse_secondary_insv)
        secondary_insv_container_layout.addLayout(secondary_insv_row)

        secondary_info_row = QHBoxLayout()
        sec_info_spacer = BodyLabel("")
        sec_info_spacer.setFixedWidth(170)
        self.secondary_insv_info_label = CaptionLabel("")
        secondary_info_row.addWidget(sec_info_spacer)
        secondary_info_row.addWidget(self.secondary_insv_info_label)
        secondary_info_row.addStretch()
        secondary_insv_container_layout.addLayout(secondary_info_row)

        self.secondary_insv_container.setVisible(False)
        inputs_layout.addWidget(self.secondary_insv_container)

        # Signal connections for INSV
        self.btn_browse_insv.clicked.connect(self._browse_insv)
        self.btn_browse_secondary_insv.clicked.connect(self._browse_secondary_insv)
        self.dual_insv_chk.stateChanged.connect(self._on_dual_insv_toggled)
        self.insv_input.textChanged.connect(self._on_insv_changed)
        self.secondary_insv_input.textChanged.connect(self._on_secondary_insv_changed)

        # Row 2.5: Auxiliary image/video cameras
        aux_row = FlowLayout()
        self.aux_label = BodyLabel(tr("Auxiliary Camera:"))
        self.aux_label.setFixedWidth(170)
        self.aux_input = LineEdit()
        self.aux_input.setPlaceholderText(tr("No auxiliary cameras configured."))
        self.aux_input.setReadOnly(True)
        self.btn_add_source = PushButton(tr("Add Camera Source..."), icon=FluentIcon.ADD)
        self.btn_add_source.clicked.connect(self._open_add_source_dialog)
        self.btn_clear_source = PushButton(tr("Clear All"), icon=FluentIcon.DELETE)
        self.btn_clear_source.clicked.connect(self._clear_aux_source)
        aux_row.addWidget(self.aux_label)
        aux_row.addWidget(self.aux_input)
        aux_row.addWidget(self.btn_add_source)
        aux_row.addWidget(self.btn_clear_source)
        inputs_layout.addLayout(aux_row)

        self.aux_sources_widget = QWidget(inputs_card)
        self.aux_sources_layout = QVBoxLayout(self.aux_sources_widget)
        self.aux_sources_layout.setContentsMargins(170, 0, 0, 0)
        self.aux_sources_layout.setSpacing(6)
        inputs_layout.addWidget(self.aux_sources_widget)
        self._refresh_aux_sources_ui()

        # Row 3: Output Folder
        out_row = QHBoxLayout()
        self.out_label = BodyLabel(tr("Output Directory:"))
        self.out_label.setFixedWidth(170)
        self.out_input = LineEdit()
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.btn_browse_out = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_label)
        out_row.addWidget(self.out_input)
        out_row.addWidget(self.btn_browse_out)
        inputs_layout.addLayout(out_row)

        layout.addWidget(inputs_card)

        # ----------------------------------------------------------------------
        # 2. Pipeline Configuration Card
        # ----------------------------------------------------------------------
        config_card = CardWidget(content)
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(20, 18, 20, 18)
        config_layout.setSpacing(14)

        self.config_title = SubtitleLabel(tr("2. Pipeline Processing Options"))
        config_layout.addWidget(self.config_title)

        grid = QVBoxLayout()
        grid.setSpacing(10)

        # Scanner Preset Selector
        scanner_row = FlowLayout()
        self.scanner_label = CaptionLabel(tr("Scanner Hardware Preset:"))
        self.scanner_combo = ComboBox()
        self.scanner_combo.addItems([
            tr("Auto-Detect (Eagle / Raven)"),
            tr("Eagle (Livox Mid-360)"),
            tr("Raven (Vanjee 722z)")
        ])
        self.scanner_combo.currentIndexChanged.connect(self._on_preset_changed)
        scanner_row.addWidget(self.scanner_label)
        scanner_row.addWidget(self.scanner_combo)
        grid.addLayout(scanner_row)

        threads_layout = scanner_row
        self.threads_caption = CaptionLabel(tr("CPU Threads:"))
        self.threads_spin = SpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        threads_layout.addWidget(self.threads_caption)
        threads_layout.addWidget(self.threads_spin)

        # SLAM options
        self.lio_switch = SwitchButton(text=tr("Enable LiDAR + IMU Odometry (Fast LIO)"))
        self.lio_switch.setOnText(tr("LiDAR + IMU (LIO)"))
        self.lio_switch.setOffText(tr("LiDAR + camera + IMU (VIO)"))
        self.lio_switch.setChecked(True)
        grid.addWidget(self.lio_switch)

        # Topics
        topics_layout = FlowLayout()
        self.lidar_caption = CaptionLabel(tr("LiDAR Topic:"))
        self.lidar_topic = LineEdit()
        self.lidar_topic.setText("/vanjee_722z")
        self.lidar_topic.setMinimumWidth(200)
        topics_layout.addWidget(self.lidar_caption)
        topics_layout.addWidget(self.lidar_topic)
        self.imu_caption = CaptionLabel(tr("IMU Topic:"))
        self.imu_topic = LineEdit()
        self.imu_topic.setText("/vanjee_imu_packets")
        self.imu_topic.setMinimumWidth(220)
        topics_layout.addWidget(self.imu_caption)
        topics_layout.addWidget(self.imu_topic)
        grid.addLayout(topics_layout)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        config_layout.addLayout(grid)
        config_layout.addWidget(sep)

        # Sync & Colorization row
        sync_row = FlowLayout()
        self.auto_sync_chk = CheckBox(tr("Auto IMU Gyro Cross-Correlation"))
        self.auto_sync_chk.setChecked(True)
        self.auto_sync_chk.stateChanged.connect(self._toggle_auto_sync)
        sync_row.addWidget(self.auto_sync_chk)

        self.dt_caption = CaptionLabel(tr("Manual Δt (s):"))
        self.dt_input = LineEdit()
        self.dt_input.setPlaceholderText(tr("e.g. 1.892"))
        self.dt_input.setFixedWidth(130)
        self.dt_input.setEnabled(False)
        sync_row.addWidget(self.dt_caption)
        sync_row.addWidget(self.dt_input)

        self.fps_caption = CaptionLabel(tr("Extraction FPS:"))
        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))
        sync_row.addWidget(self.fps_caption)
        sync_row.addWidget(self.fps_spin)

        config_layout.addLayout(sync_row)

        # Dataset Trimming & Chunking Controls
        trim_row = FlowLayout()
        self.trim_chk = CheckBox(tr("Trim dataset"))
        self.trim_chk.setToolTip(tr("Trim seconds from beginning and end of dataset synchronously for both bag and video."))
        self.trim_chk.stateChanged.connect(self._toggle_trim_controls)
        self.trim_chk.setMinimumWidth(self.trim_chk.sizeHint().width() + 24)
        trim_row.addWidget(self.trim_chk)

        self.trim_start_label = CaptionLabel(tr("Start at (s):"))
        self.trim_start_spin = DoubleSpinBox()
        self.trim_start_spin.setRange(0.0, 7200.0)
        self.trim_start_spin.setValue(0.0)
        self.trim_start_spin.setSingleStep(1.0)
        self.trim_start_spin.setEnabled(False)
        self.trim_start_spin.setFixedWidth(130)
        trim_row.addWidget(self.trim_start_label)
        trim_row.addWidget(self.trim_start_spin)

        self.trim_end_label = CaptionLabel(tr("Keep until (s, 0 = end):"))
        self.trim_end_spin = DoubleSpinBox()
        self.trim_end_spin.setRange(0.0, 7200.0)
        self.trim_end_spin.setValue(0.0)
        self.trim_end_spin.setSingleStep(1.0)
        self.trim_end_spin.setEnabled(False)
        self.trim_end_spin.setFixedWidth(130)
        trim_row.addWidget(self.trim_end_label)
        trim_row.addWidget(self.trim_end_spin)

        self.btn_export_trim = PushButton(tr("Save Trimmed..."), icon=FluentIcon.CUT)
        self.btn_export_trim.setToolTip(tr("Export the trimmed Bag and Video to a standalone folder."))
        self.btn_export_trim.clicked.connect(self._action_export_trimmed_dataset)
        trim_row.addWidget(self.btn_export_trim)

        self.btn_chunk_dataset = PushButton(tr("Divide in Chunks..."), icon=FluentIcon.TILES)
        self.btn_chunk_dataset.setToolTip(tr("Divide dataset into overlapping chunks with temporal interpolation."))
        self.btn_chunk_dataset.clicked.connect(self._action_open_chunk_dialog)
        config_layout.addLayout(trim_row)

        chunk_row = FlowLayout()
        chunk_row.addWidget(self.btn_chunk_dataset)
        chunk_row.addWidget(CaptionLabel(tr("Chunks:")))
        self.chunk_count_spin = SpinBox()
        self.chunk_count_spin.setRange(1, 99)
        self.chunk_count_spin.setValue(4)
        self.chunk_count_spin.setToolTip(tr("How many chunks to divide the (trimmed) dataset into."))
        chunk_row.addWidget(self.chunk_count_spin)
        chunk_row.addWidget(CaptionLabel(tr("Overlap (s):")))
        self.chunk_overlap_spin = DoubleSpinBox()
        self.chunk_overlap_spin.setRange(0.0, 600.0)
        self.chunk_overlap_spin.setValue(30.0)
        chunk_row.addWidget(self.chunk_overlap_spin)
        config_layout.addLayout(chunk_row)

        # Colorization & Recalibration Method
        method_row = FlowLayout()
        self.method_label = CaptionLabel(tr("Colorization & Recalibration:"))
        method_row.addWidget(self.method_label)
        self.method_combo = ComboBox()
        self.method_combo.addItems([
            tr("Reconstruction-based colorization"),
            tr("Trajectory-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(0)
        method_row.addWidget(self.method_combo)
        config_layout.addLayout(method_row)

        photometric_row = FlowLayout()
        self.photometric_caption = CaptionLabel(tr("Photometric compensation:"))
        self.photometric_combo = ComboBox()
        self.photometric_combo.addItem(tr("Off"), userData="off")
        self.photometric_combo.addItem(tr("Log-linear"), userData="loglinear")
        self.photometric_combo.addItem(tr("PPISP + bilateral grid"), userData="ppisp-bilateral")
        self.photometric_combo.setCurrentIndex(0)
        self.photometric_combo.setFixedWidth(190)
        self.photometric_combo.currentIndexChanged.connect(self._update_photometric_controls)
        self.photometric_params_caption = CaptionLabel(tr("Photometric parameters:"))
        self.photometric_params_input = LineEdit()
        self.photometric_params_input.setPlaceholderText(tr("Optional fitted JSON file"))
        self.photometric_params_btn = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.photometric_params_btn.clicked.connect(self._browse_photometric_params)
        photometric_row.addWidget(self.photometric_caption)
        photometric_row.addWidget(self.photometric_combo)
        photometric_row.addWidget(self.photometric_params_caption)
        photometric_row.addWidget(self.photometric_params_input)
        self.photometric_params_input.setMinimumWidth(240)
        photometric_row.addWidget(self.photometric_params_btn)
        config_layout.addLayout(photometric_row)
        self._update_photometric_controls()

        recalib_row = QHBoxLayout()
        self.recalibrate_chk = CheckBox(tr("Auto-Recalibrate Spatial Extrinsics from SfM Alignment"))
        self.recalibrate_chk.setChecked(True)
        recalib_row.addWidget(self.recalibrate_chk)
        recalib_row.addStretch()
        config_layout.addLayout(recalib_row)

        rig_row = FlowLayout()
        self.rig_profile_label = CaptionLabel(tr("Rig profile:"))
        self.rig_profile_combo = ComboBox()
        self.rig_profile_combo.setMinimumWidth(260)
        self.btn_new_rig = PushButton(tr("New..."), icon=FluentIcon.ADD)
        self.btn_new_rig.setToolTip(tr("Create a new hardware rig profile (wizard)"))
        self.btn_new_rig.clicked.connect(self._on_new_rig)
        self.btn_manage_rigs = PushButton(tr("Manage..."), icon=FluentIcon.TILES)
        self.btn_manage_rigs.setToolTip(tr("Open the Rig Profiles manager to view, edit, or delete profiles"))
        self.btn_manage_rigs.clicked.connect(self._on_manage_rigs)
        self._populate_rig_profiles()
        self.rig_refine_label = CaptionLabel(tr("Rig refinement:"))
        self.rig_refine_combo = ComboBox()
        for text, mode in RIG_REFINE_CHOICES:
            self.rig_refine_combo.addItem(tr(text), userData=mode)
        self.rig_refine_combo.setCurrentIndex(0)
        self.time_window_label = CaptionLabel(tr("Time window (s):"))
        self.time_window_spin = DoubleSpinBox()
        self.time_window_spin.setRange(0.0, 120.0)
        self.time_window_spin.setSingleStep(1.0)
        self.time_window_spin.setValue(6.0)
        self._update_rig_tooltips()
        rig_row.addWidget(self.rig_profile_label)
        rig_row.addWidget(self.rig_profile_combo)
        rig_row.addWidget(self.btn_new_rig)
        rig_row.addWidget(self.btn_manage_rigs)
        rig_row.addWidget(self.rig_refine_label)
        rig_row.addWidget(self.rig_refine_combo)
        rig_row.addWidget(self.time_window_label)
        rig_row.addWidget(self.time_window_spin)
        config_layout.addLayout(rig_row)

        self.revisit_refinement_chk = CheckBox(tr("Global Pose Graph SLAM Optimization (Loop Closure)"))
        self.revisit_refinement_chk.setChecked(False)
        self.revisit_refinement_chk.setToolTip(tr("Eliminates duplicate walls, ghosting, and trajectory drift by optimizing revisit loops with pose-graph SLAM. Optional; adds processing time."))
        config_layout.addWidget(self.revisit_refinement_chk)

        self.remove_dynamic_chk = CheckBox(tr("Remove moving objects and people from the point cloud"))
        self.remove_dynamic_chk.setChecked(False)
        self._update_rig_tooltips()
        config_layout.addWidget(self.remove_dynamic_chk)

        mask_row = FlowLayout()
        self.mask_persons_chk = CheckBox(tr("Generate masks"))
        self.mask_persons_chk.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.mask_model_btn = PushButton(tr("Download model RF-DETR"), icon=FluentIcon.DOWNLOAD)
        self.mask_backend_label = CaptionLabel(tr("Mask backend:"))
        self.mask_backend_combo = ComboBox()
        self.mask_backend_combo.addItem(tr("Vulkan"), userData="vulkan")
        self.mask_backend_combo.addItem(tr("TensorRT"), userData="tensorrt")
        self.mask_backend_combo.setCurrentIndex(0)
        self.mask_backend_combo.currentIndexChanged.connect(self._refresh_mask_resource_state)
        self.mask_backend_combo.currentIndexChanged.connect(self._update_mask_backend_tooltip)
        self._update_mask_backend_tooltip()
        self.mask_model_btn.clicked.connect(self._download_rf_detr)
        self.mask_settings_btn = PushButton(tr("Mask settings..."))
        self.mask_settings_btn.clicked.connect(self._open_mask_settings)
        self.operator_radius_caption = CaptionLabel(tr("Operator removal radius (m):"))
        self.operator_radius_spin = DoubleSpinBox()
        self.operator_radius_spin.setRange(0.0, 5.0)
        self.operator_radius_spin.setDecimals(2)
        self.operator_radius_spin.setSingleStep(0.10)
        self.operator_radius_spin.setValue(DEFAULT_OPERATOR_RADIUS_M)
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.operator_radius_spin.valueChanged.connect(lambda value: self.mask_persons_chk.setChecked(True) if value > 0 else None)
        # The radius only acts with person masks; keep its value while masks are off.
        self.mask_persons_chk.toggled.connect(self.operator_radius_spin.setEnabled)
        self.operator_radius_spin.setEnabled(self.mask_persons_chk.isChecked())
        mask_row.addWidget(self.mask_persons_chk)
        mask_row.addWidget(self.mask_backend_label)
        mask_row.addWidget(self.mask_backend_combo)
        config_layout.addLayout(mask_row)

        mask_controls_row = FlowLayout()
        mask_controls_row.addWidget(self.mask_model_btn)
        mask_controls_row.addWidget(self.mask_settings_btn)
        mask_controls_row.addWidget(self.operator_radius_caption)
        mask_controls_row.addWidget(self.operator_radius_spin)
        config_layout.addLayout(mask_controls_row)

        self.vulkan_chk = CheckBox(tr("Use Vulkan for point-cloud colorization"))
        self.vulkan_chk.setChecked(True)
        self.vulkan_chk.setToolTip(tr("This setting accelerates point-cloud colorization. Choose the RF-DETR mask backend separately above."))
        config_layout.addWidget(self.vulkan_chk)

        self.process_gps_chk = CheckBox(tr("Automatically georeference using INSV GPS (when available)"))
        self.process_gps_chk.setChecked(True)
        self.process_gps_chk.stateChanged.connect(self._toggle_gps_formats)
        config_layout.addWidget(self.process_gps_chk)

        layout.addWidget(config_card)

        # ----------------------------------------------------------------------
        # 3. Deliverables Output Selection Card
        # ----------------------------------------------------------------------
        output_card = ElevatedCardWidget(content)
        output_layout = QVBoxLayout(output_card)
        output_layout.setContentsMargins(20, 18, 20, 18)
        output_layout.setSpacing(10)

        self.output_title = SubtitleLabel(tr("3. Deliverables of this run"))
        self.output_subtitle = CaptionLabel(tr("The project (project.json) is kept in the output folder. Other formats are exported from it afterwards."))
        output_layout.addWidget(self.output_title)
        self.output_subtitle.setWordWrap(True)
        output_layout.addWidget(self.output_subtitle)

        self.chk_ply = CheckBox(tr("Export Colored Point Cloud (.PLY)"))
        self.chk_ply.setChecked(True)
        output_layout.addWidget(self.chk_ply)

        self.chk_pcd = CheckBox(tr("Export Colored Point Cloud (.PCD)"))
        self.chk_pcd.setChecked(False)
        output_layout.addWidget(self.chk_pcd)

        self.chk_colmap = CheckBox(tr("Metric-Scaled 3DGS COLMAP Dataset"))
        self.chk_colmap.setChecked(True)
        output_layout.addWidget(self.chk_colmap)

        dense_layout = QHBoxLayout()
        self.chk_dense = CheckBox(tr("Dense photogrammetry from the fisheye photos (OpenMVS, optional)"))
        self.chk_dense.setChecked(False)
        self.dense_quality_combo = ComboBox()
        for label, value in DENSE_QUALITY_CHOICES:
            self.dense_quality_combo.addItem(tr(label), userData=value)
        self.dense_quality_combo.setCurrentIndex(1)
        dense_layout.addWidget(self.chk_dense)
        dense_layout.addWidget(self.dense_quality_combo)
        self.chk_dense_mesh = CheckBox(tr("Textured mesh (GLB)"))
        self.chk_dense_mesh.setChecked(False)
        dense_layout.addWidget(self.chk_dense_mesh)
        dense_layout.addStretch()
        output_layout.addLayout(dense_layout)

        seed_layout = QGridLayout()
        seed_layout.setColumnStretch(1, 1)
        self.seed_percent_label = BodyLabel(tr("3DGS LiDAR seed export:"))
        self.seed_percent_slider = QSlider(Qt.Orientation.Horizontal, output_card)
        self.seed_percent_slider.setObjectName("seedPercentSlider")
        self.seed_percent_slider.setRange(1, 100)
        self.seed_percent_slider.setValue(100)
        self.seed_percent_slider.setToolTip(tr("Choose the percentage of LiDAR points to include in the 3DGS seed."))
        self.seed_percent_spin = SpinBox(output_card)
        self.seed_percent_spin.setObjectName("seedPercentSpin")
        self.seed_percent_spin.setRange(1, 100)
        self.seed_percent_spin.setValue(100)
        self.seed_percent_spin.setSuffix("%")
        self.seed_percent_spin.setToolTip(tr("Type the percentage of LiDAR points to include in the 3DGS seed."))
        seed_layout.addWidget(self.seed_percent_label, 0, 0)
        seed_layout.addWidget(self.seed_percent_slider, 0, 1)
        seed_layout.addWidget(self.seed_percent_spin, 0, 2)
        output_layout.addLayout(seed_layout)

        self.seed_point_count_label = CaptionLabel("")
        self.seed_point_count_label.setWordWrap(True)
        output_layout.addWidget(self.seed_point_count_label)

        self.georef_formats_label = CaptionLabel(tr("Automatic georeferenced cloud formats:"))
        output_layout.addWidget(self.georef_formats_label)
        georef_formats_layout = QHBoxLayout()
        self.chk_geo_laz = CheckBox(tr("LAZ")); self.chk_geo_laz.setChecked(True)
        self.chk_geo_las = CheckBox(tr("LAS"))
        self.chk_geo_ply = CheckBox(tr("PLY"))
        self.chk_geo_pcd = CheckBox(tr("PCD"))
        self.chk_geo_geojson = CheckBox(tr("GeoJSON")); self.chk_geo_geojson.setChecked(True)
        for checkbox in (self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply, self.chk_geo_pcd, self.chk_geo_geojson):
            georef_formats_layout.addWidget(checkbox)
        georef_formats_layout.addStretch()
        output_layout.addLayout(georef_formats_layout)

        self.gps_formats_label = CaptionLabel(tr("GPS metadata formats (select one or more):"))
        output_layout.addWidget(self.gps_formats_label)
        gps_formats_layout = QHBoxLayout()
        self.chk_gps_geojson = CheckBox(tr("GeoJSON"))
        self.chk_gps_geojson.setChecked(True)
        self.chk_gps_gpx = CheckBox(tr("GPX"))
        self.chk_gps_gpx.setChecked(True)
        self.chk_gps_csv = CheckBox(tr("CSV"))
        self.chk_gps_csv.setChecked(True)
        gps_formats_layout.addWidget(self.chk_gps_geojson)
        gps_formats_layout.addWidget(self.chk_gps_gpx)
        gps_formats_layout.addWidget(self.chk_gps_csv)
        gps_formats_layout.addStretch()
        output_layout.addLayout(gps_formats_layout)
        self._toggle_gps_formats()

        self._extra_widgets = [self.chk_pcd, self.georef_formats_label, self.chk_geo_laz, self.chk_geo_las,
                               self.chk_geo_ply, self.chk_geo_pcd, self.chk_geo_geojson, self.gps_formats_label,
                               self.chk_gps_geojson, self.chk_gps_gpx, self.chk_gps_csv, self.process_gps_chk]
        show_extra = bool(load_config().get("show_extra_formats", False))
        self.process_gps_chk.setChecked(show_extra)
        for widget in self._extra_widgets:
            widget.setVisible(show_extra)

        self.btn_export_project = PushButton(tr("Export from project..."), icon=FluentIcon.SHARE)
        self.btn_export_project.setToolTip(tr("Write PCD, LAZ/LAS, georeferenced cloud, GPS track or trajectory from an existing project."))
        self.btn_export_project.clicked.connect(self._action_export_from_project)
        output_layout.addWidget(self.btn_export_project)

        # Optional (Settings > Deliverables): hand-off to programs the user owns
        self._photogrammetry_runner = ProcessRunner(self)
        self._photogrammetry_runner.finished.connect(self._on_photogrammetry_finished)
        self._photogrammetry_buttons = {}
        photogrammetry_row = QHBoxLayout()
        for program, name in (('realityscan', 'RealityScan'), ('metashape', 'Metashape')):
            button = PushButton(tr("Open in {program}...", program=name), icon=FluentIcon.SHARE)
            button.setToolTip(tr("Export the fisheye photos, masks, poses and LiDAR of a 3DGS dataset, and open them if the program is installed."))
            button.clicked.connect(lambda _=False, p=program: self._action_open_photogrammetry(p))
            button.setVisible(bool(load_config().get("show_photogrammetry_export", False)))
            self._photogrammetry_buttons[program] = button
            photogrammetry_row.addWidget(button)
        photogrammetry_row.addStretch()
        output_layout.addLayout(photogrammetry_row)

        layout.addWidget(output_card)

        # ----------------------------------------------------------------------
        # 4. Action & Status Bar
        # ----------------------------------------------------------------------
        action_card = CardWidget(content)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 14, 20, 14)
        action_layout.setSpacing(10)

        action_header = QVBoxLayout()
        action_buttons = FlowLayout()

        status_box = QVBoxLayout()
        self.status_title = StrongBodyLabel(tr("Status: Ready"))
        self.status_desc = CaptionLabel(tr("Select a LiDAR bag plus a camera source (INSV video or image bag), then click 'Start Unified Workflow'."))
        self.status_desc.setWordWrap(True)
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_header.addLayout(status_box)

        self.btn_run = PrimaryPushButton(tr("Start Unified Workflow"), icon=FluentIcon.PLAY)
        self.btn_run.clicked.connect(self._start_workflow)
        self.btn_cancel = PushButton(tr_button("Cancel & Save"), icon=FluentIcon.CLOSE)
        self.btn_cancel.clicked.connect(self._cancel_workflow)
        self.btn_cancel.setEnabled(False)

        action_buttons.addWidget(self.btn_run)
        action_buttons.addWidget(self.btn_cancel)
        action_header.addLayout(action_buttons)
        action_layout.addLayout(action_header)

        overall_row = QHBoxLayout()
        self.overall_progress_label = BodyLabel(tr("Overall progress: 0%"))
        self.overall_progress_label.setMinimumWidth(140)
        self.overall_progress_bar = ProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.eta_label = CaptionLabel(tr("Estimated time: waiting to start"))
        self.eta_label.setMinimumWidth(120)
        self.eta_label.setWordWrap(True)
        overall_row.addWidget(self.overall_progress_label)
        overall_row.addWidget(self.overall_progress_bar, 1)
        overall_row.addWidget(self.eta_label)
        action_layout.addLayout(overall_row)

        self.current_step_label = StrongBodyLabel(tr("Workflow steps"))
        action_layout.addWidget(self.current_step_label)
        steps_grid = QGridLayout()
        steps_grid.setHorizontalSpacing(14)
        steps_grid.setVerticalSpacing(7)
        self.step_progress_bars = []
        self.step_progress_labels = []
        for index, name in enumerate(STAGE_NAMES):
            column = (index % 2) * 2
            row = index // 2
            label = CaptionLabel(f"{index + 1}. {tr(name)}")
            label.setWordWrap(True)
            bar = ProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setMinimumWidth(60)
            self.step_progress_labels.append(label)
            self.step_progress_bars.append(bar)
            steps_grid.addWidget(label, row, column)
            steps_grid.addWidget(bar, row, column + 1)
        action_layout.addLayout(steps_grid)
        layout.addWidget(action_card)

        # ----------------------------------------------------------------------
        # 5. Live Streaming Console
        # ----------------------------------------------------------------------
        log_card = CardWidget(content)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 12, 16, 12)
        log_layout.setSpacing(8)

        log_header = QHBoxLayout()
        self.log_title = SubtitleLabel(tr("Live Workflow Execution Console"))
        self.btn_clear_log = ToolButton(FluentIcon.DELETE)
        self.btn_clear_log.setToolTip(tr("Clear visible log (saved file remains)"))
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.btn_export_log = ToolButton(FluentIcon.DOWNLOAD)
        self.btn_export_log.setToolTip(tr("Export log..."))
        self.btn_export_log.clicked.connect(self._export_log)
        log_header.addWidget(self.log_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_export_log)
        log_header.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_header)

        self.log_console = PlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(240)
        self.log_console.setMinimumWidth(0)
        self.log_console.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.log_console.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; "
            "font-family: 'Consolas', 'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_console)
        layout.addWidget(log_card)

        scroll.setWidget(content)
        scroll.enableTransparentBackground()
        content.setAutoFillBackground(False)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self.runner.started.connect(self._on_job_started)
        self.runner.finished.connect(self._on_job_finished)
        self.runner.log_received.connect(self._on_log_received)
        self.runner.error_occurred.connect(self._on_error_occurred)
        self.seed_percent_slider.valueChanged.connect(self.seed_percent_spin.setValue)
        self.seed_percent_spin.valueChanged.connect(self.seed_percent_slider.setValue)
        self.seed_percent_spin.valueChanged.connect(self._update_seed_point_estimate)
        self.out_input.textChanged.connect(self._refresh_seed_point_estimate)
        self.chk_colmap.stateChanged.connect(self._toggle_seed_export_controls)
        self.chk_dense.stateChanged.connect(self._toggle_seed_export_controls)
        self._toggle_seed_export_controls()
        self._refresh_seed_point_estimate()

    def _toggle_seed_export_controls(self, state=None):
        enabled = self.chk_colmap.isChecked()
        self.seed_percent_label.setEnabled(enabled)
        self.seed_percent_slider.setEnabled(enabled)
        self.seed_percent_spin.setEnabled(enabled)
        openmvs = get_openmvs_bin()
        found = openmvs.is_file()
        can_mesh = has_programs(openmvs) if found else False
        self.chk_dense.setEnabled(enabled and found)
        self.dense_quality_combo.setEnabled(enabled and found)
        self.chk_dense_mesh.setEnabled(enabled and can_mesh and self.chk_dense.isChecked())
        self.chk_dense_mesh.setToolTip(
            tr("Also turns the dense cloud into a surface, with its texture copied from the fisheye photos. Passers-by masked out of the photos leave no texture.")
            if can_mesh else tr("Needs ReconstructMesh and TextureMesh of the OpenMVS fisheye build, beside DensifyPointCloud."))
        self.chk_dense.setToolTip(
            tr("Builds a dense cloud from each fisheye photo as a whole and fills its gaps with the LiDAR. Written to colmap_3dgs/dense_openmvs.")
            if found else tr("OpenMVS (fisheye build) was not found. Set its folder in Settings > Deliverables."))
        self._update_seed_point_estimate()

    def _refresh_seed_point_estimate(self, text=None):
        output = self.out_input.text().strip()
        self._seed_full_point_count = dataset_seed_point_count(output) if output else None
        self._seed_actual_export_count = None
        self._seed_actual_export_percent = None
        self._update_seed_point_estimate()

    def _update_seed_point_estimate(self, value=None):
        if not self.chk_colmap.isChecked():
            self.seed_point_count_label.setText(tr("3DGS seed export is disabled."))
            return

        percent = self.seed_percent_spin.value()
        total = self._seed_full_point_count
        if total is None:
            self.seed_point_count_label.setText(
                tr("3DGS seed point count is pending until the LiDAR cloud is available ({percent}% selected).")
                .format(percent=percent)
            )
            return

        if self._seed_actual_export_count is not None and self._seed_actual_export_percent == percent:
            count = self._seed_actual_export_count
            text = tr("3DGS seed exported: {count:,} of {total:,} points ({percent}% requested).")
        else:
            count = seed_point_count(total, percent)
            text = tr("Estimated 3DGS seed: {count:,} of {total:,} points ({percent}% selected).")
        self.seed_point_count_label.setText(text.format(count=count, total=total, percent=percent))

    def _on_preset_changed(self, index: int):
        if index == 1:  # Eagle
            self.lidar_topic.setText("/livox/lidar")
            self.imu_topic.setText("/livox/imu")
        elif index == 2:  # Raven
            self.lidar_topic.setText("/vanjee_722z")
            self.imu_topic.setText("/vanjee_imu_packets")
        else:  # Auto-Detect
            bag = self.bag_input.text().strip()
            if bag and Path(bag).is_file():
                self._auto_detect_bag(bag)

    def _auto_detect_bag(self, bag_path: str):
        try:
            detected = detect_bag_topics([bag_path])
            if detected.get('lidar_topic'):
                self.lidar_topic.setText(detected['lidar_topic'])
            if detected.get('imu_topic'):
                self.imu_topic.setText(detected['imu_topic'])
            stype = detected.get('scanner_type', '')
            if stype == 'eagle':
                self.scanner_combo.blockSignals(True)
                self.scanner_combo.setCurrentIndex(1)
                self.scanner_combo.blockSignals(False)
            elif stype == 'raven':
                self.scanner_combo.blockSignals(True)
                self.scanner_combo.setCurrentIndex(2)
                self.scanner_combo.blockSignals(False)
        except Exception:
            pass

    def _toggle_auto_sync(self, state):
        self.dt_input.setEnabled(not self.auto_sync_chk.isChecked())

    def _toggle_trim_controls(self, state):
        enabled = bool(state)
        self.trim_start_spin.setEnabled(enabled)
        self.trim_end_spin.setEnabled(enabled)

    def _secondary_video_for_cut(self):
        """The second lens file when dual-file INSV is on; None lets the cut find a _10_ partner itself."""
        if self.dual_insv_chk.isChecked():
            sec = self.secondary_insv_input.text().strip()
            if sec and Path(sec).is_file():
                return sec
        return None

    def _load_cut_dataset_aux_cameras(self, folder: Path):
        """A chunk or trim folder holds the auxiliary cameras cut with it; load them in place of the current ones."""
        from raven_app.dataset_trim import AUX_CONFIG_NAME
        from raven_app.third_camera import load_third_camera_configs
        folder = Path(folder)
        cfg_path = folder / AUX_CONFIG_NAME
        is_cut = (folder / "trim_manifest.json").is_file() or (folder.parent / "chunks_manifest.json").is_file()
        if not (cfg_path.is_file() and is_cut):
            return
        try:
            configs = load_third_camera_configs(cfg_path)
        except Exception as exc:
            InfoBar.warning(title=tr("Auxiliary cameras not loaded"), content=str(exc), parent=self, duration=5000)
            return
        self.third_camera_configs = list(configs)
        self._refresh_aux_sources_ui()
        InfoBar.info(
            title=tr("Auxiliary cameras loaded"),
            content=tr("{n} auxiliary camera(s) cut with this dataset were loaded from {name}.").format(
                n=len(configs), name=AUX_CONFIG_NAME),
            parent=self, duration=5000)

    def _turn_off_trim_for_cut_dataset(self, bag_path):
        """A bag made by Trim/Chunks is already cut; the trim values on screen belong to the source."""
        from raven_app.dataset_trim import cut_dataset_sync
        if not self.trim_chk.isChecked() or cut_dataset_sync(bag_path) is None:
            return
        self.trim_chk.setChecked(False)
        InfoBar.info(
            title=tr("Trim turned off"),
            content=tr("'{name}' was already cut by Trim/Chunks, so it is processed whole.").format(name=Path(bag_path).name),
            parent=self, duration=6000)

    def _confirm_trim_of_cut_dataset(self, bag_path):
        """Cutting a Trim/Chunks output again is allowed, but usually means the old trim values were left on."""
        from raven_app.dataset_trim import cut_dataset_sync
        if not self.trim_chk.isChecked() or cut_dataset_sync(bag_path) is None:
            return True
        answer = QMessageBox.question(
            self, tr("Cut this dataset again?"),
            tr("'{name}' was already cut by Trim/Chunks. 'Trim dataset' is on, so it would be cut again: "
               "start at {start:.1f}s of the cut files. Yes cuts it again; No processes it whole.").format(
                name=Path(bag_path).name, start=float(self.trim_start_spin.value())),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.No:
            self.trim_chk.setChecked(False)
        return True

    def _action_export_trimmed_dataset(self):
        """Export trimmed bag and video directly to disk."""
        bag = self.bag_input.text().strip()
        insv = self.insv_input.text().strip()
        if not bag or not Path(bag).is_file():
            InfoBar.warning(title=tr("Missing LiDAR Bag"), content=tr("Please select a valid ROS bag file first."), parent=self)
            return
        if not insv or not Path(insv).is_file():
            InfoBar.warning(title=tr("Missing 360 Video"), content=tr("Please select a valid INSV/MP4 video first."), parent=self)
            return
        out_dir = QFileDialog.getExistingDirectory(self, tr("Select Directory to Save Trimmed Dataset"))
        if not out_dir:
            return
        trim_start = float(self.trim_start_spin.value()) if self.trim_chk.isChecked() else 0.0
        trim_end = float(self.trim_end_spin.value()) if self.trim_chk.isChecked() else 0.0
        dt_val = None
        if self.dt_input.text().strip():
            try: dt_val = float(self.dt_input.text().strip())
            except ValueError: pass
        try:
            from raven_app.dataset_trim import trim_dataset_synchronized
            res = trim_dataset_synchronized(
                bag_path=bag, video_path=insv, output_dir=out_dir,
                trim_start_sec=trim_start, trim_end_sec=0.0, end_at_sec=trim_end, dt_sync=dt_val,
                secondary_video_path=self._secondary_video_for_cut(),
                auxiliary_cameras=self._active_aux_camera_configs(),
            )
            InfoBar.success(
                title=tr("Dataset Trimmed Successfully"),
                content=tr("Trimmed bag ({b_dur:.1f}s) and video ({v_dur:.1f}s) saved to {path}").format(
                    b_dur=res['bag_duration'], v_dur=res['video_duration'], path=out_dir
                ), parent=self, duration=6000
            )
        except Exception as e:
            InfoBar.error(title=tr("Trimming Failed"), content=str(e), parent=self, duration=6000)

    def _action_open_chunk_dialog(self):
        """Divide dataset into overlapping chunks with temporal interpolation."""
        bag = self.bag_input.text().strip()
        insv = self.insv_input.text().strip()
        if not bag or not Path(bag).is_file() or not insv or not Path(insv).is_file():
            InfoBar.warning(title=tr("Missing Inputs"), content=tr("Please select both a valid ROS bag and video first."), parent=self)
            return
        out_dir = QFileDialog.getExistingDirectory(self, tr("Select Directory to Save Dataset Chunks"))
        if not out_dir:
            return
        trim_start = float(self.trim_start_spin.value()) if self.trim_chk.isChecked() else 0.0
        trim_end = float(self.trim_end_spin.value()) if self.trim_chk.isChecked() else 0.0
        dt_val = None
        if self.dt_input.text().strip():
            try: dt_val = float(self.dt_input.text().strip())
            except ValueError: pass
        try:
            from raven_app.dataset_trim import split_dataset_into_chunks
            chunks = split_dataset_into_chunks(
                bag_path=bag, video_path=insv, output_dir=out_dir,
                chunk_duration_sec=300.0, overlap_sec=float(self.chunk_overlap_spin.value()),
                num_chunks=int(self.chunk_count_spin.value()),
                trim_start_sec=trim_start, trim_end_sec=0.0, end_at_sec=trim_end, dt_sync=dt_val,
                secondary_video_path=self._secondary_video_for_cut(),
                auxiliary_cameras=self._active_aux_camera_configs(),
            )
            InfoBar.success(
                title=tr("Chunks Created Successfully"),
                content=tr("Generated {count} synchronized overlapping chunks in {path}").format(
                    count=len(chunks), path=out_dir
                ), parent=self, duration=6000
            )
        except Exception as e:
            InfoBar.error(title=tr("Chunking Failed"), content=str(e), parent=self, duration=6000)

    def _action_export_from_project(self):
        from PyQt6.QtWidgets import QFileDialog, QDialog, QDialogButtonBox, QVBoxLayout as _V
        from raven_app.project import EXPORT_FORMATS, EXPORT_LABELS, PROJECT_FILE, export_from_project
        folder = QFileDialog.getExistingDirectory(self, tr("Project folder (contains project.json)"), "")
        if not folder:
            return
        if not (Path(folder) / PROJECT_FILE).is_file():
            InfoBar.error(title=tr("Not a project"), content=tr("project.json was not found in that folder."),
                          parent=self, duration=5000)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("Export from project"))
        box = _V(dialog)
        checks = {}
        for fmt in EXPORT_FORMATS:
            checks[fmt] = CheckBox(tr(EXPORT_LABELS[fmt]))
            box.addWidget(checks[fmt])
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        box.addWidget(buttons)
        if not dialog.exec():
            return
        chosen = [f for f, c in checks.items() if c.isChecked()]
        if not chosen:
            return
        dest = QFileDialog.getExistingDirectory(self, tr("Destination folder"), str(Path(folder) / 'exports'))
        if not dest:
            return
        errors, written = [], 0
        for fmt in chosen:
            try:
                written += sum(len(v) for v in export_from_project(folder, [fmt], dest).values())
            except Exception as exc:
                errors.append(f"{fmt}: {exc}")
        if errors:
            InfoBar.warning(title=tr("Some exports failed"), content="\n".join(errors), parent=self, duration=10000)
        else:
            InfoBar.success(title=tr("Export finished"), content=f"{written} " + tr("file(s) in") + f" {dest}",
                            parent=self, duration=6000)

    def _action_open_photogrammetry(self, program):
        from PyQt6.QtWidgets import QFileDialog
        from raven_app.photogrammetry_export import PROGRAMS, find_program
        if self._photogrammetry_runner.is_busy:
            return
        output = self.out_input.text().strip()
        default = Path(output) / 'colmap_3dgs' if output else None
        folder = QFileDialog.getExistingDirectory(self, tr("3DGS dataset folder (colmap_3dgs)"),
                                                  str(default) if default and default.is_dir() else "")
        if not folder:
            return
        if not (Path(folder) / 'sparse' / '0' / 'images.txt').is_file():
            InfoBar.error(title=tr("Not a 3DGS dataset"), content=tr("sparse/0/images.txt was not found in that folder."),
                          parent=self, duration=5000)
            return
        installed = find_program(program, str(load_config().get(f'{program}_path', '') or '')) is not None
        args = ['photogrammetry-export', '--program', program, '--colmap', folder] + (['--open'] if installed else [])
        self._photogrammetry_job = (PROGRAMS[program], Path(folder) / program, installed)
        if self._photogrammetry_runner.start_job(args, log_path=make_run_log_path(folder, f'{program}_export')):
            for button in self._photogrammetry_buttons.values():
                button.setEnabled(False)
            InfoBar.info(title=tr("Exporting for {program}", program=PROGRAMS[program]),
                         content=tr("The photos are being prepared; this takes a few minutes."), parent=self, duration=6000)

    def _on_photogrammetry_finished(self, code: int):
        for button in self._photogrammetry_buttons.values():
            button.setEnabled(True)
        name, folder, installed = self._photogrammetry_job
        if code:
            InfoBar.error(title=tr("Export for {program} failed", program=name),
                          content=f"{tr('Log:')} {self._photogrammetry_runner.log_path}", parent=self, duration=10000)
        else:
            text = tr("Files in {folder}", folder=str(folder))
            if not installed:
                text += "\n" + tr("{program} was not found on this computer, so nothing was opened.", program=name)
            InfoBar.success(title=tr("Export finished"), content=text, parent=self, duration=8000)

    def _toggle_gps_formats(self, state=None):
        enabled = self.process_gps_chk.isChecked()
        for checkbox in (self.chk_gps_geojson, self.chk_gps_gpx, self.chk_gps_csv,
                         self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply,
                         self.chk_geo_pcd, self.chk_geo_geojson):
            checkbox.setEnabled(enabled)

    def _browse_bag(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROS Bag (.bag)", "", "ROS Bag files (*.bag);;All files (*.*)"
        )
        if path:
            p = Path(path)
            detected = detect_bag_topics([path])
            if not detected.get('has_lidar'):
                cand_lidar = find_adjacent_lidar_bag(p)
                if cand_lidar and cand_lidar.is_file():
                    self.bag_input.setText(str(cand_lidar))
                    self._auto_detect_bag(str(cand_lidar))
                    self.image_bag_input.setText(str(p))
                    InfoBar.info(
                        title=tr("Auto-Assigned LiDAR & Image Bags"),
                        content=tr("'{img}' contains camera images. Automatically assigned '{lidar}' as LiDAR Bag and '{img}' as Front Camera Bag.").format(
                            img=p.name, lidar=cand_lidar.name
                        ),
                        orient=Qt.Orientation.Horizontal,
                        position=InfoBarPosition.TOP,
                        duration=6000,
                        parent=self
                    )
                    if not self.out_input.text().strip():
                        self.out_input.setText(str(cand_lidar.parent / f"{cand_lidar.stem}_studio_output"))
                    return
                else:
                    InfoBar.warning(
                        title=tr("No LiDAR Points in Bag"),
                        content=tr("The selected bag '{name}' does not contain LiDAR point cloud topics.").format(name=p.name),
                        orient=Qt.Orientation.Horizontal,
                        position=InfoBarPosition.TOP,
                        duration=5000,
                        parent=self
                    )

            self.bag_input.setText(path)
            self._auto_detect_bag(path)
            self._load_cut_dataset_aux_cameras(p.parent)
            self._turn_off_trim_for_cut_dataset(p)
            cand = find_adjacent_image_bag(p)
            if cand and cand.is_file() and not self.image_bag_input.text().strip():
                self.image_bag_input.setText(str(cand))
            # Auto-suggest output folder if empty
            if not self.out_input.text().strip():
                self.out_input.setText(str(p.parent / f"{p.stem}_studio_output"))

    def _on_image_bag_changed(self, text: str):
        from pathlib import Path
        p = Path(text.strip())
        if p.is_file():
            from raven_app.bag_io import discover_bag_image_topics
            topics = discover_bag_image_topics(p)
            if topics:
                summary = ", ".join(f"{t['topic']} ({t['msgcount']} frames)" for t in topics)
                self.bag_camera_info_label.setText(tr(f"Detected {len(topics)} image topic(s): {summary}"))
            else:
                self.bag_camera_info_label.setText(tr("No image topics found in this bag."))
        else:
            self.bag_camera_info_label.setText("")
        self.bag_camera_info_label.setVisible(bool(self.bag_camera_info_label.text()))

    def _browse_image_bag(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Select Front Camera Bag (.bag)"), "", tr("ROS Bag files (*.bag);;All files (*.*)")
        )
        if path:
            self.image_bag_input.setText(path)
            self._on_image_bag_changed(path)

    def _browse_insv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Insta360 Video (.insv)", "", "Insta360 Video (*.insv *.mp4);;All files (*.*)"
        )
        if path:
            self.insv_input.setText(path)

    def _browse_secondary_insv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Select Secondary 360 Video (Rear Lens)"), "",
            "Insta360 Video (*.insv *.mp4);;All files (*.*)"
        )
        if path:
            self.secondary_insv_input.setText(path)

    def _on_dual_insv_toggled(self, state):
        is_checked = bool(state)
        self.secondary_insv_container.setVisible(is_checked)
        if not is_checked:
            self.secondary_insv_info_label.setText("")
        else:
            # If primary already has a path, trigger check
            self._on_insv_changed(self.insv_input.text())

    def _on_insv_changed(self, text):
        from raven_app.video import find_partner_insv
        p_str = text.strip()
        if not p_str:
            self.secondary_insv_info_label.setText("")
            return
        p = Path(p_str)
        if not p.is_file():
            return
        name = p.name
        if '_00_' in name or '_10_' in name:
            c0, c1 = find_partner_insv(p)
            if c0 and c1:
                # Both exist: auto-enable dual_insv and populate fields
                if not self.dual_insv_chk.isChecked():
                    self.dual_insv_chk.setChecked(True)
                # Ensure c0 is in insv_input and c1 is in secondary_insv_input
                if str(c0) != p_str:
                    self.insv_input.blockSignals(True)
                    self.insv_input.setText(str(c0))
                    self.insv_input.blockSignals(False)
                if self.secondary_insv_input.text().strip() != str(c1):
                    self.secondary_insv_input.setText(str(c1))
                self.secondary_insv_info_label.setText(
                    tr(f"[+] Auto-paired Dual-File INSV: cam0 ({c0.name}) + cam1 ({c1.name})")
                )
            else:
                # One lens file detected but partner not found in folder (Decision 1A)
                if not self.dual_insv_chk.isChecked():
                    self.dual_insv_chk.setChecked(True)
                missing = name.replace('_00_', '_10_') if '_00_' in name else name.replace('_10_', '_00_')
                self.secondary_insv_info_label.setText(
                    tr(f"[!] Warning: Partner lens file '{missing}' not found in folder. Please browse manually.")
                )

    def _on_secondary_insv_changed(self, text):
        sec_str = text.strip()
        if sec_str and Path(sec_str).is_file():
            prim_str = self.insv_input.text().strip()
            if prim_str and Path(prim_str).is_file():
                self.secondary_insv_info_label.setText(
                    tr(f"[+] Dual-File INSV configured: {Path(prim_str).name} + {Path(sec_str).name}")
                )

    def _open_add_source_dialog(self):
        from raven_app.views.add_source_dialog import AddSourceDialog
        from raven_app.third_camera import ThirdCameraConfig
        camera_name = self._next_aux_camera_name()
        dialog = AddSourceDialog(
            parent=self,
            config=ThirdCameraConfig(camera_name=camera_name),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            if not config.source_path:
                return
            config.camera_name = camera_name
            self.third_camera_configs.append(config)
            self._refresh_aux_sources_ui()

    def _edit_aux_source(self, index):
        from raven_app.views.add_source_dialog import AddSourceDialog
        if not 0 <= index < len(self.third_camera_configs):
            return
        current = self.third_camera_configs[index]
        dialog = AddSourceDialog(parent=self, config=current)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            config.camera_name = current.camera_name
            self.third_camera_configs[index] = config
            self._refresh_aux_sources_ui()

    def _remove_aux_source(self, index):
        if 0 <= index < len(self.third_camera_configs):
            del self.third_camera_configs[index]
            self._refresh_aux_sources_ui()

    def _next_aux_camera_name(self):
        used = {str(config.camera_name).casefold() for config in self.third_camera_configs}
        while f"cam{self._next_aux_camera_index}".casefold() in used:
            self._next_aux_camera_index += 1
        name = f"cam{self._next_aux_camera_index}"
        self._next_aux_camera_index += 1
        return name

    def _refresh_aux_sources_ui(self):
        while self.aux_sources_layout.count():
            item = self.aux_sources_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self.third_camera_configs:
            self.aux_input.clear()
            self.aux_input.setPlaceholderText(tr("No auxiliary cameras configured."))
            self.btn_clear_source.setEnabled(False)
            empty = CaptionLabel(tr("Add one or more image or video sources to include additional camera views."))
            empty.setWordWrap(True)
            self.aux_sources_layout.addWidget(empty)
            return

        self.aux_input.setText(
            tr("{n} auxiliary camera(s) configured").format(n=len(self.third_camera_configs))
        )
        self.btn_clear_source.setEnabled(True)
        for index, config in enumerate(self.third_camera_configs):
            row = QWidget(self.aux_sources_widget)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            source_name = Path(config.source_path).name if config.source_path else tr("No source selected")
            details = BodyLabel(
                f"{config.camera_name}: {source_name} ({config.camera_model}, {config.fps:g} fps)"
            )
            details.setToolTip(str(config.source_path))
            row_layout.addWidget(details, 1)
            edit = PushButton(tr("Configure"))
            edit.clicked.connect(lambda _checked=False, row_index=index: self._edit_aux_source(row_index))
            row_layout.addWidget(edit)
            remove = PushButton(tr("Remove"), icon=FluentIcon.DELETE)
            remove.clicked.connect(lambda _checked=False, row_index=index: self._remove_aux_source(row_index))
            row_layout.addWidget(remove)
            self.aux_sources_layout.addWidget(row)

    def _clear_aux_source(self):
        self.third_camera_configs.clear()
        self._refresh_aux_sources_ui()

    def _active_aux_camera_configs(self):
        return [config for config in self.third_camera_configs
                if config.enabled and str(config.source_path).strip()]

    def _configured_aux_camera_names(self):
        return [config.camera_name for config in self.third_camera_configs
                if str(config.source_path).strip()]

    def _save_aux_camera_bundle(self, path):
        cameras = [config.to_dict() for config in self._active_aux_camera_configs()]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"cameras": cameras}, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _browse_photometric_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Select photometric parameter JSON"), "", "JSON files (*.json);;All files (*.*)"
        )
        if path:
            self.photometric_params_input.setText(path)

    def showEvent(self, event):
        super().showEvent(event)
        self._populate_rig_profiles()

    def _populate_rig_profiles(self, select_path=None):
        """Fill the hardware rig profile selector; the default profile comes first."""
        from raven_app.profile_manager import list_available_profiles
        prev_selected = select_path or self.rig_profile_combo.currentData()
        self.rig_profile_combo.blockSignals(True)
        self.rig_profile_combo.clear()
        profiles = list_available_profiles()
        idx_to_select = 0
        for i, profile in enumerate(profiles):
            label = profile['name'] + (f" ({tr('default')})" if profile.get('is_default') else '')
            self.rig_profile_combo.addItem(label, userData=profile['path'])
            if prev_selected and Path(profile['path']) == Path(prev_selected):
                idx_to_select = i
        if self.rig_profile_combo.count() > 0:
            self.rig_profile_combo.setCurrentIndex(idx_to_select)
        self.rig_profile_combo.blockSignals(False)

    def _on_new_rig(self):
        """Open the New Rig Wizard directly from the workflow view."""
        from raven_app.views.calibration_view import NewRigDialog
        from raven_app.rig_profile import save_profile, user_profiles_dir
        dialog = NewRigDialog(self.window())
        if not dialog.exec():
            return
        try:
            profile = dialog.build()
            path = save_profile(profile, user_profiles_dir() / f'{profile.profile_id}.json')
            self._populate_rig_profiles(select_path=path)
            InfoBar.success(
                title=tr("Profile created"),
                content=tr("Rig profile '{name}' was created and selected.", name=profile.name),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=4000
            )
        except Exception as exc:
            InfoBar.error(
                title=tr("Error creating profile"),
                content=str(exc),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=6000
            )

    def _on_manage_rigs(self):
        """Navigate to the Rig Profiles view (CalibrationView)."""
        main_win = self.window()
        if hasattr(main_win, 'calib_view') and hasattr(main_win, 'switchTo'):
            main_win.switchTo(main_win.calib_view)

    def _update_rig_tooltips(self):
        self.rig_profile_combo.setToolTip(tr(
            "Hardware profile of your scanner + camera mount. It is the starting point of the "
            "per-capture refinement and the reference that rejects implausible calibrations."))
        self.rig_refine_combo.setToolTip(tr(
            "Photometric: refines the time offset and the mount by multi-view colour consistency; "
            "kept only if it improves held-out frames. Time offset only: faster. Off: use the profile as is."))
        self.time_window_spin.setToolTip(tr(
            "Each LiDAR point is coloured only from frames taken within this many seconds of its "
            "scan, so cars or people that moved between passes do not leave ghost colours. 0 disables."))
        if hasattr(self, 'remove_dynamic_chk'):
            self.remove_dynamic_chk.setToolTip(tr(
                "Deletes what moved during the capture before colorization: spots that other scans saw "
                "through are carved out. With person masks, people standing still are also removed where "
                "most views mask them. All exported clouds and the 3DGS seed come out clean."))

    def _rig_args(self):
        args = ["--rig-refine", str(self.rig_refine_combo.currentData()),
                "--time-window", f"{float(self.time_window_spin.value()):g}"]
        if self.remove_dynamic_chk.isChecked():
            args.append("--remove-dynamic")
        profile = self.rig_profile_combo.currentData()
        if profile:
            args.extend(["--profile", str(profile)])
        return args

    def _update_photometric_controls(self, *_):
        enabled = self.photometric_combo.currentData() != "off"
        self.photometric_params_caption.setEnabled(enabled)
        self.photometric_params_input.setEnabled(enabled)
        self.photometric_params_btn.setEnabled(enabled)

    def _download_rf_detr(self):
        if self.runner.is_busy:
            InfoBar.warning(title=tr('Processing is active'), content=tr('Wait for the current task to finish before downloading RF-DETR resources.'), parent=self)
            return
        backend = self.mask_backend_combo.currentData()
        if resources_ready(backend):
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            InfoBar.success(title=tr('RF-DETR is ready'), content=tr('The model and {backend} mask resources are verified.').format(backend=label), parent=self)
            return
        if resource_status(backend) == 'executor-missing':
            InfoBar.error(title=tr('Mask backend unavailable'), content=tr('This build has no executor for the selected mask backend. Choose the other backend, or install a release that bundles it.'), parent=self)
            return
        if backend == 'vulkan':
            prompt = tr('Verify or download the 139 MB RF-DETR model and prepare its Vulkan cache (~130 MB). This uses Vulkan and does not download TensorRT. Resources stay outside the portable app.')
        else:
            prompt = tr('Verify or download the 139 MB RF-DETR model. If needed, download about 1.3 GB of NVIDIA TensorRT runtime; extraction may need up to 3 GB of temporary disk space. Resources stay outside the portable app.')
        answer = QMessageBox.question(
            self, tr('Prepare RF-DETR masks'), prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        downloader = MaskResourceDownload(backend, self)
        self._mask_downloader = downloader
        downloader.progress.connect(self._on_mask_download_progress)
        downloader.completed.connect(self._on_mask_download_completed)
        self.mask_model_btn.setEnabled(False)
        self.mask_backend_combo.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.mask_model_btn.setText(tr('Preparing RF-DETR...'))
        self.mask_model_btn.setIcon(FluentIcon.SYNC)
        if not downloader.start():
            self._on_mask_download_completed(2, tr('Could not start the resource downloader.'))

    def _update_mask_backend_tooltip(self, *_args):
        if self.mask_backend_combo.currentData() == 'vulkan':
            text = tr('Use Vulkan for person-mask inference; no TensorRT runtime is required. Point-cloud colorization has a separate setting.')
        else:
            text = tr('Use NVIDIA TensorRT for person-mask inference; its optional runtime is downloaded separately. Point-cloud colorization has a separate setting.')
        self.mask_backend_combo.setToolTip(text)

    def _on_mask_download_progress(self, resource, percent):
        title = {
            'model': tr('RF-DETR model'),
            'runtime': tr('TensorRT runtime'),
            'install': tr('Installing TensorRT'),
            'vulkan': tr('Preparing Vulkan model'),
        }.get(resource, tr('Preparing RF-DETR'))
        self.mask_model_btn.setText(f'{title}: {percent}%' if percent else f'{title}...')

    def _on_mask_download_completed(self, code, output):
        downloader, self._mask_downloader = self._mask_downloader, None
        self.mask_model_btn.setEnabled(True)
        self.mask_backend_combo.setEnabled(True)
        self.btn_run.setEnabled(True)
        if downloader is not None:
            downloader.deleteLater()
        self._refresh_mask_resource_state()
        backend = self.mask_backend_combo.currentData()
        if code == 0 and resources_ready(backend):
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            InfoBar.success(title=tr('RF-DETR resources ready'), content=tr('The model and {backend} mask resources are verified and saved in the user cache.').format(backend=label), parent=self)
        else:
            detail = output[-700:] if output else tr('Check your internet connection and available disk space, then retry.')
            InfoBar.error(title=tr('RF-DETR download failed'), content=detail, parent=self, duration=8000)

    def _refresh_mask_resource_state(self, *_args):
        if self._mask_downloader is not None:
            return
        backend = self.mask_backend_combo.currentData()
        status = resource_status(backend)
        # Stays clickable when the executor is missing, so the click explains what to do.
        self.mask_model_btn.setEnabled(True)
        if status == 'ready':
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            self.mask_model_btn.setText(tr('RF-DETR ready ({backend})').format(backend=label))
            self.mask_model_btn.setIcon(FluentIcon.ACCEPT)
            self.mask_model_btn.setToolTip(tr('The model, selected mask executor and required resources are verified.'))
        elif status == 'model-missing':
            self.mask_model_btn.setText(tr('Download model RF-DETR'))
            self.mask_model_btn.setIcon(FluentIcon.DOWNLOAD)
            self.mask_model_btn.setToolTip(tr('Download the verified RF-DETR model and prepare the selected mask backend.'))
        elif status == 'runtime-missing':
            self.mask_model_btn.setText(tr('Download TensorRT runtime'))
            self.mask_model_btn.setIcon(FluentIcon.DOWNLOAD)
            self.mask_model_btn.setToolTip(tr('The RF-DETR model is present. Download the optional NVIDIA TensorRT runtime for this backend.'))
        elif status == 'schedule-missing':
            self.mask_model_btn.setText(tr('Prepare Vulkan masks'))
            self.mask_model_btn.setIcon(FluentIcon.SYNC)
            self.mask_model_btn.setToolTip(tr('The model is downloaded. Prepare the Vulkan schedule; TensorRT is not needed.'))
        else:
            self.mask_model_btn.setText(tr('Mask executor missing'))
            self.mask_model_btn.setIcon(FluentIcon.INFO)
            self.mask_model_btn.setToolTip(tr('This build has no executor for the selected mask backend. '
                                              'Choose the other backend, or install a release that bundles it.'))

    def _open_mask_settings(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog
        from raven_app.person_masks import read_mask_config
        output = Path(self.out_input.text().strip()) if self.out_input.text().strip() else None
        config = self.mask_config
        settings_path = output / 'mask_settings.json' if output else None
        if config is None and settings_path and settings_path.is_file():
            try:
                config = read_mask_config(settings_path)
            except (OSError, ValueError) as exc:
                InfoBar.error(title=tr('Invalid mask settings'), content=str(exc),
                              position=InfoBarPosition.TOP, parent=self)
                return
        if config is None:
            from raven_app.person_masks import profile_mask_defaults
            config = profile_mask_defaults(self.rig_profile_combo.currentData())
        sec_insv = None
        if hasattr(self, 'dual_insv_chk') and self.dual_insv_chk.isChecked():
            sec_text = self.secondary_insv_input.text().strip()
            if sec_text and Path(sec_text).is_file():
                sec_insv = Path(sec_text)

        dialog = MaskSettingsDialog(output, config, self,
                                    backend=self.mask_backend_combo.currentData(),
                                    insv_path=self.insv_input.text().strip() or None,
                                    fps=self.fps_spin.value(),
                                    camera_names=self._configured_aux_camera_names(),
                                    auxiliary_sources=self.third_camera_configs,
                                    secondary_insv_path=sec_insv,
                                    profile=self.rig_profile_combo.currentData(),
                                    profile_name=self.rig_profile_combo.currentText())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.mask_config = dialog.settings()
            self.mask_persons_chk.setChecked(True)
            count = sum(sum(map(len, self.mask_config[key].values())) for key in ('rectangles', 'ellipses', 'polygons'))
            self.mask_settings_btn.setText(tr('Mask settings ({n} shapes)').format(n=count))
            if dialog.saved_to_profile:
                InfoBar.success(
                    title=tr('Masks saved to the rig profile'),
                    content=tr('Runs with {name} start from these masks. Saved in {path}').format(
                        name=self.rig_profile_combo.currentText(), path=dialog.saved_to_profile),
                    position=InfoBarPosition.TOP, parent=self, duration=6000)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_input.setText(folder)

    def _clear_log(self):
        self.log_console.clear()

    def _export_log(self):
        source = self._current_log_path
        default_path = str(source) if source else 'workflow.log'
        target, _ = QFileDialog.getSaveFileName(
            self, tr('Export workflow log'), default_path,
            tr('Log files (*.log);;Text files (*.txt);;All files (*.*)')
        )
        if not target:
            return
        try:
            content = source.read_text(encoding='utf-8') if source and source.is_file() else self.log_console.toPlainText()
            Path(target).write_text(content, encoding='utf-8')
        except OSError as exc:
            InfoBar.error(title=tr('Cannot export log'), content=str(exc),
                          position=InfoBarPosition.TOP, parent=self)

    def _start_workflow(self):
        bag = self.bag_input.text().strip()
        insv = self.insv_input.text().strip()
        out = self.out_input.text().strip()

        if not bag or not Path(bag).is_file():
            InfoBar.error(
                title=tr("Invalid LiDAR Bag"),
                content=tr("Please select a valid ROS bag file (.bag)."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        detected_bag = detect_bag_topics([bag])
        if not detected_bag.get('has_lidar'):
            cand_lidar = find_adjacent_lidar_bag(Path(bag))
            hint = tr(" Did you mean '{cand}'?").format(cand=cand_lidar.name) if cand_lidar else ""
            InfoBar.error(
                title=tr("Invalid LiDAR Bag"),
                content=tr("The selected bag '{name}' does not contain any LiDAR point clouds.{hint}").format(
                    name=Path(bag).name, hint=hint
                ),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=6000,
                parent=self
            )
            return

        if not self._confirm_trim_of_cut_dataset(bag):
            return

        has_insv = bool(insv and Path(insv).is_file())
        has_aux = bool(self.third_camera_configs)
        has_bag_images = bool(self.use_bag_images_chk.isChecked() and Path(self.image_bag_input.text().strip()).is_file())
        if not has_insv and not has_aux and not has_bag_images:
            InfoBar.error(
                title=tr("Missing Image Source"),
                content=tr("Please specify a 360 video (.insv/.mp4), an auxiliary camera, or enable and select an Image Bag."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not out:
            InfoBar.error(
                title=tr("Missing Output Folder"),
                content=tr("Please specify an output folder for deliverables."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        self._refresh_seed_point_estimate()

        if self.process_gps_chk.isChecked() and not any(
            checkbox.isChecked() for checkbox in (
                self.chk_gps_geojson, self.chk_gps_gpx, self.chk_gps_csv
            )
        ):
            InfoBar.error(
                title=tr("GPS output format required"),
                content=tr("Select at least one GPS metadata format or disable GPS processing."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if self.process_gps_chk.isChecked() and not any(
            checkbox.isChecked() for checkbox in (
                self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply,
                self.chk_geo_pcd, self.chk_geo_geojson
            )
        ):
            InfoBar.error(
                title=tr("Georeferenced output format required"),
                content=tr("Select at least one georeferenced output format or disable automatic GPS georeferencing."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        method_key = ["sfm", "direct", "all"][self.method_combo.currentIndex()]
        scanner_key = ["auto", "eagle", "raven"][self.scanner_combo.currentIndex()]

        args = [
            "workflow",
            "--bag", bag,
            "--output", out,
            "--scanner", scanner_key,
            "--lidar-topic", self.lidar_topic.text().strip() or "/vanjee_722z",
            "--imu-topic", self.imu_topic.text().strip() or "/vanjee_imu_packets",
            "--threads", str(self.threads_spin.value()),
            "--fps", str(self.fps_spin.value()),
            "--method", method_key
        ]

        if insv and Path(insv).is_file():
            args.extend(["--insv", insv])

        if self.dual_insv_chk.isChecked() and self.secondary_insv_input.text().strip():
            args.extend(["--secondary-insv", self.secondary_insv_input.text().strip()])

        if self.image_bag_input.text().strip():
            args.extend(["--image-bag", self.image_bag_input.text().strip()])
            if self.use_bag_images_chk.isChecked():
                args.append("--use-image-bag")

        args.extend(["--photometric", str(self.photometric_combo.currentData())])
        photometric_params = self.photometric_params_input.text().strip()
        if photometric_params and self.photometric_combo.currentData() != "off":
            args.extend(["--photometric-params", photometric_params])

        if self._active_aux_camera_configs():
            out_p = Path(out)
            out_p.mkdir(parents=True, exist_ok=True)
            cfg_path = out_p / "third_camera.json"
            self._save_aux_camera_bundle(cfg_path)
            args.extend(["--third-camera", str(cfg_path)])

        if self.recalibrate_chk.isChecked():
            args.append("--recalibrate")
        else:
            args.append("--no-recalibrate")
        args.extend(self._rig_args())

        if self.trim_chk.isChecked():
            if self.trim_start_spin.value() > 0:
                args.extend(["--trim-start", str(float(self.trim_start_spin.value()))])
            if self.trim_end_spin.value() > 0:
                args.extend(["--trim-end-at", str(float(self.trim_end_spin.value()))])

        if method_key in ("sfm", "all"):
            args.append("--run-spirula")

        if self.mask_persons_chk.isChecked():
            args.append("--mask-persons")
            args.extend(['--mask-backend', self.mask_backend_combo.currentData()])
            settings_path = Path(out).resolve() / 'mask_settings.json'
            if self.mask_config is not None:
                try:
                    settings_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = settings_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(self.mask_config, indent=2), encoding='utf-8')
                    temporary.replace(settings_path)
                except OSError as exc:
                    InfoBar.error(title=tr('Cannot save mask settings'), content=str(exc),
                                  position=InfoBarPosition.TOP, parent=self)
                    return
            if settings_path.is_file():
                args.extend(['--mask-config', str(settings_path)])

        if self.mask_persons_chk.isChecked():
            args.extend(["--operator-radius", str(self.operator_radius_spin.value())])

        if self.lio_switch.isChecked():
            args.append("--lio")
        else:
            args.append("--vio")

        if self.revisit_refinement_chk.isChecked():
            args.append("--revisit-refinement")
        else:
            args.append("--no-revisit-refinement")

        if not self.auto_sync_chk.isChecked() and self.dt_input.text().strip():
            args.extend(["--dt", self.dt_input.text().strip()])

        if not self.vulkan_chk.isChecked():
            args.append("--no-vulkan")

        if self.chk_ply.isChecked():
            args.append("--export-ply")
        if self.chk_pcd.isChecked():
            args.append("--export-pcd")
        if self.chk_colmap.isChecked():
            args.append("--export-colmap")
            args.extend(["--seed-percent", str(self.seed_percent_spin.value())])
            if self.chk_dense.isEnabled() and self.chk_dense.isChecked():
                args.extend(["--densify", "openmvs", "--dense-quality", self.dense_quality_combo.currentData()])
                if self.chk_dense_mesh.isChecked() and self.chk_dense_mesh.isEnabled():
                    args.extend(["--dense-mesh", "glb"])

        if self.process_gps_chk.isChecked():
            args.append("--process-gps")
            gps_formats = []
            if self.chk_gps_geojson.isChecked():
                gps_formats.append("geojson")
            if self.chk_gps_gpx.isChecked():
                gps_formats.append("gpx")
            if self.chk_gps_csv.isChecked():
                gps_formats.append("csv")
            args.extend(["--gps-formats", *gps_formats])
            geo_formats = [name for name, checkbox in (
                ("laz", self.chk_geo_laz), ("las", self.chk_geo_las),
                ("ply", self.chk_geo_ply), ("pcd", self.chk_geo_pcd),
                ("geojson", self.chk_geo_geojson)) if checkbox.isChecked()]
            if geo_formats:
                args.extend(["--geo-formats", *geo_formats])

        self._prepare_workflow_progress(out, bag, insv, method_key)
        log_path = make_run_log_path(out, f'{Path(out).name}_workflow')
        if self.runner.start_job(
            args, log_path=log_path,
            initial_log=f">>> Starting Unified Workflow: {' '.join(args)}"
        ):
            self._current_log_path = log_path

    def _cancel_workflow(self):
        self.status_title.setText("Status: Cancelling...")
        self.status_desc.setText("Gracefully saving accumulated SLAM and colorization data...")
        self.btn_cancel.setEnabled(False)
        self.runner.cancel()

    def _on_job_started(self):
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status_title.setText("Status: Workflow Running")
        self.status_desc.setText("Executing end-to-end LiDAR-camera processing. SfM runs in the background.")
        self._workflow_started_at = time.monotonic()
        self._stage_started_at = None
        self._current_stage = 0
        self._current_stage_skipped = False
        self._stage_fraction = 0.0
        self._stage_fraction_from_log = False
        self._log_fraction = 0.0
        self._log_fraction_start = None
        self._recorded_stages = set()
        for bar in self.step_progress_bars:
            bar.setValue(0)
        self.overall_progress_bar.setValue(0)
        self.current_step_label.setText(tr("Workflow steps"))
        self._refresh_workflow_progress()
        self._progress_timer.start()

    def _on_job_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._progress_timer.stop()

        if code == 0:
            if self._current_stage:
                self._complete_progress_stage(self._current_stage)
            for bar in self.step_progress_bars:
                bar.setValue(100)
            self.overall_progress_bar.setValue(100)
            self.overall_progress_label.setText(tr("Overall progress: 100%"))
            elapsed = time.monotonic() - self._workflow_started_at if self._workflow_started_at else 0
            self.eta_label.setText(tr("Completed in {time}").format(time=self._format_duration(elapsed)))
            self.current_step_label.setText(tr("All workflow steps completed"))
            self.status_title.setText("Status: Completed Successfully")
            self.status_desc.setText("Deliverables saved in 'deliverables/'. 3DGS dataset in 'colmap_3dgs/'.")
            InfoBar.success(
                title=tr("Unified Workflow Complete"),
                content=tr("Deliverables saved in 'deliverables/'. For 3DGS, select the 'colmap_3dgs' folder."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=6000,
                parent=self
            )
        elif code == 130:
            self.status_title.setText("Status: Cancelled & Flushed")
            self.status_desc.setText("Workflow cancelled by user. Completed steps and partial data are saved.")
        else:
            self.status_title.setText(f"Status: Failed (exit code {code})")
            current = tr(STAGE_NAMES[self._current_stage - 1]) if self._current_stage else tr("preparation")
            self.status_desc.setText(f"Failed during {current}. Check the saved log below.")
            InfoBar.error(
                title=tr("Workflow Failed"),
                content=tr("Execution exited with code {code}.", code=code),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )

    def _on_log_received(self, line: str):
        self.log_console.insertPlainText(line)
        self.log_console.ensureCursorVisible()
        for item in line.splitlines():
            self._update_workflow_progress_from_line(item)
            match = re.search(
                r"3DGS seed export:\s*([\d,]+)\s*/\s*([\d,]+)\s*points\s*\(requested\s*([\d.]+)%",
                item,
                re.IGNORECASE,
            )
            if match:
                self._seed_actual_export_count = int(match.group(1).replace(",", ""))
                self._seed_full_point_count = int(match.group(2).replace(",", ""))
                self._seed_actual_export_percent = float(match.group(3))
                self._update_seed_point_estimate()

    def _prepare_workflow_progress(self, output_dir, bag_path, insv_path, method):
        self._stage_root = Path(output_dir)
        self._stage_history = load_stage_history(self._stage_root)
        self._dataset_ratios = load_stage_ratios(dataset_times_path(self._stage_root))
        self._machine_ratios = load_stage_ratios(machine_rates_path())
        self._estimate_inputs = ((self._stage_root, bag_path, insv_path), dict(
            use_masks=self.mask_persons_chk.isChecked(),
            mask_backend=self.mask_backend_combo.currentData(),
            use_vulkan=self.vulkan_chk.isChecked(),
            export_colmap=self.chk_colmap.isChecked(),
            method=method,
            recalibrate=self.recalibrate_chk.isChecked(),
            fps=self.fps_spin.value(),
            remove_dynamic=self.remove_dynamic_chk.isChecked(),
            rig_refine=str(self.rig_refine_combo.currentData() or 'none'),
        ))
        self._stage_defaults = [0.0] * len(STAGE_NAMES)
        self._reestimate_stages(after_stage=0)
        total = sum(self._stage_estimates)
        self.eta_label.setText(tr("Estimated remaining: about {time}").format(
            time=self._format_duration(total)))
        for index, bar in enumerate(self.step_progress_bars):
            bar.setValue(0)
            bar.setToolTip(tr("Approximate progress; the estimate improves from saved runs."))
            self.step_progress_labels[index].setText(f"{index + 1}. {tr(STAGE_NAMES[index])}")

    def _reestimate_stages(self, after_stage):
        """Recompute the stages after ``after_stage`` from what is on disk now (frames, SLAM cloud)."""
        if not self._estimate_inputs:
            return
        args, kwargs = self._estimate_inputs
        try:
            defaults = estimate_stage_seconds(*args, **kwargs)
        except Exception:
            return
        estimates = stage_duration_estimates(
            defaults, self._stage_history, self._dataset_ratios, self._machine_ratios)
        for index in range(after_stage, len(STAGE_NAMES)):
            self._stage_defaults[index] = defaults[index]
            self._stage_estimates[index] = estimates[index]

    def _update_workflow_progress_from_line(self, line):
        event = parse_progress_line(line)
        if not event:
            return
        if event['kind'] == 'stage':
            stage = min(max(event['stage'], 1), len(STAGE_NAMES))
            name = event.get('name') or STAGE_NAMES[stage - 1]
            if self._current_stage and stage > self._current_stage:
                for finished in range(self._current_stage, stage):
                    self._complete_progress_stage(finished)
                if self._current_stage <= 3 < stage or self._current_stage <= 1 < stage:
                    # Frame count and SLAM cloud size are now known: refresh what is left.
                    self._reestimate_stages(after_stage=stage - 1)
            if stage != self._current_stage:
                self._current_stage = stage
                self._stage_started_at = time.monotonic()
                self._stage_fraction = 0.0
                self._stage_fraction_from_log = False
                self._log_fraction = 0.0
                self._log_fraction_start = None
                self._current_stage_skipped = (
                    stage == 5 and name.casefold().startswith('skipping'))
            self.current_step_label.setText(tr("Step {current}/{total}: {name}").format(
                current=stage, total=len(STAGE_NAMES), name=tr(name)))
            self.status_desc.setText(tr("Current step: {name}").format(name=tr(name)))
        elif event['kind'] == 'stage_fraction' and self._current_stage:
            fraction = event['fraction']
            if self._log_fraction_start is None or fraction < self._log_fraction:
                # A new progress loop inside this step (each loop is timed on its own).
                self._log_fraction_start = (time.monotonic(), fraction)
            self._log_fraction = fraction
            self._stage_fraction_from_log = True
        elif event['kind'] == 'masks_ready' and self._current_stage == 5 and not (
                self._estimate_inputs and self._estimate_inputs[1].get('remove_dynamic')):
            # With moving-object removal on, the long ray-carving pass only starts here.
            self._log_fraction = max(self._log_fraction, 0.99)
            self._stage_fraction_from_log = True
        elif event['kind'] == 'complete':
            for stage in range(1, len(STAGE_NAMES) + 1):
                self._complete_progress_stage(stage)
            self.current_step_label.setText(tr("All workflow steps completed"))
        self._refresh_workflow_progress()

    def _complete_progress_stage(self, stage):
        index = stage - 1
        if index < 0 or index >= len(self.step_progress_bars) or stage in self._recorded_stages:
            return
        # Recorded once per run. (The bar can already read 100 % from a "frames N/N" line.)
        self._recorded_stages.add(stage)
        self.step_progress_bars[index].setValue(100)
        if self._current_stage == stage and self._stage_started_at is not None:
            duration = max(0.0, time.monotonic() - self._stage_started_at)
            if not self._current_stage_skipped:
                self._stage_history[index].append(duration)
                self._stage_history[index] = self._stage_history[index][-5:]
                default = self._stage_defaults[index]
                learned = add_stage_ratio(self._dataset_ratios, index, duration, default)
                add_stage_ratio(self._machine_ratios, index, duration, default)
                if self._stage_root is not None:
                    try:
                        save_stage_history(self._stage_root, self._stage_history)
                        if learned:
                            save_stage_ratios(dataset_times_path(self._stage_root), self._dataset_ratios)
                            save_stage_ratios(machine_rates_path(), self._machine_ratios)
                    except OSError:
                        pass
            self._stage_started_at = None

    def _refresh_workflow_progress(self):
        if self._workflow_started_at is None:
            return
        now = time.monotonic()
        elapsed_total = max(0.0, now - self._workflow_started_at)
        fractions = [1.0 if index + 1 in self._recorded_stages else bar.value() / 100.0
                     for index, bar in enumerate(self.step_progress_bars)]
        current_remaining = 0.0
        if self._current_stage:
            index = self._current_stage - 1
            stage_elapsed = max(0.0, now - (self._stage_started_at or now))
            estimate = max(1.0, self._stage_estimates[index])
            if stage_elapsed > 0.9 * estimate:
                # Running long: keep a little time in hand instead of showing 0 s.
                estimate = stage_elapsed / 0.9
                self._stage_estimates[index] = estimate
            # A "frames N/N" line is not the end of the step: filtering and saving follow it.
            log_fraction = min(0.97, self._log_fraction) if self._stage_fraction_from_log else 0.0
            fraction = max(self._stage_fraction, log_fraction, min(0.95, stage_elapsed / estimate))
            self._stage_fraction = fraction
            fractions[index] = fraction
            current_remaining = estimate * (1.0 - fraction)
            if self._log_fraction_start is not None and self._log_fraction < 1.0:
                started, start_fraction = self._log_fraction_start
                progressed = self._log_fraction - start_fraction
                if progressed > 0.05:
                    loop_remaining = (now - started) * (1.0 - self._log_fraction) / progressed
                    current_remaining = max(current_remaining, loop_remaining)

        weights = self._stage_estimates
        total_weight = max(1.0, sum(weights))
        overall = sum(weight * fraction for weight, fraction in zip(weights, fractions)) / total_weight
        percent = max(0, min(99, round(overall * 100)))
        self.overall_progress_bar.setValue(percent)
        self.overall_progress_label.setText(tr("Overall progress: {percent}%").format(percent=percent))
        for index, (bar, fraction) in enumerate(zip(self.step_progress_bars, fractions)):
            bar.setValue(100 if fraction >= 1.0 else max(0, min(99, round(fraction * 100))))

        if self._current_stage:
            future = sum(weights[self._current_stage:])
            remaining = current_remaining + future
            estimate_text = tr("Estimated left: about {time}").format(
                time=self._format_duration(remaining))
        else:
            remaining = max(0.0, total_weight - elapsed_total)
            estimate_text = tr("Estimated remaining: about {time}").format(
                time=self._format_duration(remaining))
        self.eta_label.setText(f"{self._format_duration(elapsed_total)} elapsed · {estimate_text}")

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _on_error_occurred(self, err: str):
        self.log_console.appendPlainText(f"\n[ERROR] {err}\n")

    def retranslate_ui(self):
        """Update all text elements dynamically when the language changes."""
        self.header_title.setText(tr("Unified LiDAR-Camera Studio"))
        self.header_subtitle.setText(
            tr("Automated end-to-end pipeline: Inputs Selection → FAST-LIVO2 SLAM → "
               "Time Sync → Point Cloud Colorization → 3DGS Deliverables")
        )
        self.inputs_title.setText(tr("1. Input Datasets & Destination"))
        self.bag_label.setText(tr("LiDAR ROS Bag (.bag):"))
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.btn_browse_bag.setText(tr("Browse"))
        self.image_bag_label.setText(tr("Front Camera Bag (.bag):"))
        self.image_bag_input.setPlaceholderText(tr("Optional IMAGE_*.bag for front camera & VIO..."))
        self.btn_browse_image_bag.setText(tr("Browse"))
        self.insv_label.setText(tr("Insta360 Video (.insv):"))
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 / X6 video (.insv or .mp4)..."))
        self.btn_browse_insv.setText(tr("Browse"))
        self.aux_label.setText(tr("Auxiliary Camera:"))
        self.aux_input.setPlaceholderText(tr("No auxiliary cameras configured."))
        self.btn_add_source.setText(tr("Add Camera Source..."))
        self.btn_clear_source.setText(tr("Clear All"))
        self._refresh_aux_sources_ui()
        self.out_label.setText(tr("Output Directory:"))
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.btn_browse_out.setText(tr("Browse"))

        self.config_title.setText(tr("2. Pipeline Processing Options"))
        self.scanner_label.setText(tr("Scanner Hardware Preset:"))
        self.lio_switch.setText(tr("Enable LiDAR + IMU Odometry (Fast LIO)"))
        self.lio_switch.setOnText(tr("LiDAR + IMU (LIO)"))
        self.lio_switch.setOffText(tr("LiDAR + camera + IMU (VIO)"))
        self.threads_caption.setText(tr("CPU Threads:"))
        self.lidar_caption.setText(tr("LiDAR Topic:"))
        self.imu_caption.setText(tr("IMU Topic:"))
        self.auto_sync_chk.setText(tr("Auto IMU Gyro Cross-Correlation"))
        self.dt_caption.setText(tr("Manual Δt (s):"))
        self.fps_caption.setText(tr("Extraction FPS:"))
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))
        self.method_label.setText(tr("Colorization & Recalibration:"))

        cur_idx = self.method_combo.currentIndex()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems([
            tr("Reconstruction-based colorization"),
            tr("Trajectory-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(cur_idx)
        self.method_combo.blockSignals(False)

        self.photometric_caption.setText(tr("Photometric compensation:"))
        photo_idx = self.photometric_combo.currentIndex()
        self.photometric_combo.blockSignals(True)
        self.photometric_combo.setItemText(0, tr("Off"))
        self.photometric_combo.setItemText(1, tr("Log-linear"))
        self.photometric_combo.setItemText(2, tr("PPISP + bilateral grid"))
        self.photometric_combo.setCurrentIndex(photo_idx)
        self.photometric_combo.blockSignals(False)
        self.photometric_params_caption.setText(tr("Photometric parameters:"))
        self.photometric_params_input.setPlaceholderText(tr("Optional fitted JSON file"))
        self.photometric_params_btn.setText(tr("Browse"))
        self._update_photometric_controls()

        self.recalibrate_chk.setText(tr("Auto-Recalibrate Spatial Extrinsics from SfM Alignment"))
        self.rig_profile_label.setText(tr("Rig profile:"))
        self.btn_new_rig.setText(tr("New..."))
        self.btn_manage_rigs.setText(tr("Manage..."))
        self.rig_refine_label.setText(tr("Rig refinement:"))
        for index, (text, _mode) in enumerate(RIG_REFINE_CHOICES):
            self.rig_refine_combo.setItemText(index, tr(text))
        self.time_window_label.setText(tr("Time window (s):"))
        self.remove_dynamic_chk.setText(tr("Remove moving objects and people from the point cloud"))
        self._update_rig_tooltips()
        self.mask_persons_chk.setText(tr("Generate masks"))
        self.mask_persons_chk.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.mask_backend_label.setText(tr("Mask backend:"))
        self.mask_backend_combo.setItemText(0, tr("Vulkan"))
        self.mask_backend_combo.setItemText(1, tr("TensorRT"))
        self._update_mask_backend_tooltip()
        shapes = sum(sum(map(len, self.mask_config[key].values()))
                     for key in ('rectangles', 'ellipses', 'polygons')) if self.mask_config else 0
        self.mask_settings_btn.setText(tr('Mask settings ({n} shapes)').format(n=shapes) if self.mask_config else tr('Mask settings...'))
        if self._mask_downloader is None:
            self._refresh_mask_resource_state()
        self.operator_radius_caption.setText(tr("Operator removal radius (m):"))
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.vulkan_chk.setText(tr("Use Vulkan for point-cloud colorization"))
        self.vulkan_chk.setToolTip(tr("This setting accelerates point-cloud colorization. Choose the RF-DETR mask backend separately above."))
        self.process_gps_chk.setText(tr("Automatically georeference using INSV GPS (when available)"))
        self.output_title.setText(tr("3. Deliverables of this run"))
        self.output_subtitle.setText(tr("The project (project.json) is kept in the output folder. Other formats are exported from it afterwards."))
        self.chk_ply.setText(tr("Export Colored Point Cloud (.PLY)"))
        self.chk_pcd.setText(tr("Export Colored Point Cloud (.PCD)"))
        self.chk_colmap.setText(tr("Metric-Scaled 3DGS COLMAP Dataset"))
        self.chk_dense.setText(tr("Dense photogrammetry from the fisheye photos (OpenMVS, optional)"))
        for index, (label, _) in enumerate(DENSE_QUALITY_CHOICES):
            self.dense_quality_combo.setItemText(index, tr(label))
        self.chk_dense_mesh.setText(tr("Textured mesh (GLB)"))
        for program, name in (('realityscan', 'RealityScan'), ('metashape', 'Metashape')):
            self._photogrammetry_buttons[program].setText(tr("Open in {program}...", program=name))
        self._toggle_seed_export_controls()
        self.seed_percent_label.setText(tr("3DGS LiDAR seed export:"))
        self.seed_percent_slider.setToolTip(tr("Choose the percentage of LiDAR points to include in the 3DGS seed."))
        self.seed_percent_spin.setToolTip(tr("Type the percentage of LiDAR points to include in the 3DGS seed."))
        self._update_seed_point_estimate()
        self.georef_formats_label.setText(tr("Automatic georeferenced cloud formats:"))
        self.gps_formats_label.setText(tr("GPS metadata formats (select one or more):"))
        self.chk_gps_geojson.setText(tr("GeoJSON"))
        self.chk_gps_gpx.setText(tr("GPX"))
        self.chk_gps_csv.setText(tr("CSV"))
        self.btn_run.setText(tr("Start Unified Workflow"))
        self.btn_cancel.setText(tr_button("Cancel & Save"))
        for index, name in enumerate(STAGE_NAMES):
            self.step_progress_labels[index].setText(f"{index + 1}. {tr(name)}")
        self.log_title.setText(tr("Live Workflow Execution Console"))
        self.btn_clear_log.setToolTip(tr("Clear visible log (saved file remains)"))
        self.btn_export_log.setToolTip(tr("Export log..."))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select a LiDAR bag plus a camera source (INSV video or image bag), then click 'Start Unified Workflow'."))
