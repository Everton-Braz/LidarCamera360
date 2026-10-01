import os
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from raven_app.process_runner import ProcessRunner
from raven_app.views.unified_workflow_view import UnifiedWorkflowView


APP = QApplication.instance() or QApplication([])


def _write_ply_header(path: Path, count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n".encode("ascii")
    )


def test_seed_percentage_controls_sync_and_preview_header_count(tmp_path):
    view = UnifiedWorkflowView(ProcessRunner())
    try:
        view.resize(640, 760)
        view.show()
        APP.processEvents()

        assert view.seed_percent_spin.value() == 100
        assert view.seed_percent_slider.value() == 100
        card = view.seed_percent_slider.parentWidget()
        for widget in (view.seed_percent_label, view.seed_percent_slider, view.seed_percent_spin):
            x = widget.mapTo(card, widget.rect().topLeft()).x()
            assert x >= 0
            assert x + widget.width() <= card.width()

        output = tmp_path / "output"
        _write_ply_header(output / "deliverables" / "lidar_colored_test.ply", 1001)
        view.out_input.setText(str(output))
        view.seed_percent_slider.setValue(37)

        assert view.seed_percent_spin.value() == 37
        assert "Estimated 3DGS seed: 370 of 1,001 points" in view.seed_point_count_label.text()

        view.seed_percent_spin.setValue(62)
        assert view.seed_percent_slider.value() == 62
        assert "Estimated 3DGS seed: 620 of 1,001 points" in view.seed_point_count_label.text()
    finally:
        view.close()


def test_seed_count_is_pending_until_cloud_exists_then_uses_export_log(tmp_path):
    view = UnifiedWorkflowView(ProcessRunner())
    try:
        view.seed_percent_spin.setValue(40)
        assert "pending until the LiDAR cloud is available (40% selected)" in view.seed_point_count_label.text()

        view._on_log_received(
            "[+] 3DGS seed export: 400 / 1,000 points (requested 40%; evenly spaced source indices)\n"
        )
        assert "3DGS seed exported: 400 of 1,000 points (40% requested)" in view.seed_point_count_label.text()
    finally:
        view.close()


def test_workflow_passes_seed_percentage_only_when_3dgs_export_is_enabled(tmp_path):
    bag = tmp_path / "scan.bag"
    insv = tmp_path / "camera.insv"
    bag.touch()
    insv.touch()
    output = tmp_path / "output"

    runner = ProcessRunner()
    runner.start_job = Mock(return_value=True)
    view = UnifiedWorkflowView(runner)
    try:
        view.bag_input.setText(str(bag))
        view.insv_input.setText(str(insv))
        view.out_input.setText(str(output))
        view.seed_percent_spin.setValue(38)

        view._start_workflow()
        args = runner.start_job.call_args.args[0]
        assert args[args.index("--seed-percent") + 1] == "38"

        runner.start_job.reset_mock()
        view.chk_colmap.setChecked(False)
        view._start_workflow()
        args = runner.start_job.call_args.args[0]
        assert "--seed-percent" not in args
    finally:
        view.close()
