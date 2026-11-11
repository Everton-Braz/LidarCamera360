import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from raven_app.process_runner import ProcessRunner
from raven_app.views.unified_workflow_view import UnifiedWorkflowView


APP = QApplication.instance() or QApplication([])


def _workflow_args(view, runner):
    view._start_workflow()
    return runner.start_job.call_args.args[0]


def test_workflow_passes_selected_profile_refinement_and_time_window(tmp_path):
    bag = tmp_path / "scan.bag"
    insv = tmp_path / "camera.insv"
    bag.touch()
    insv.touch()
    runner = ProcessRunner()
    runner.start_job = Mock(return_value=True)
    view = UnifiedWorkflowView(runner)
    try:
        view.bag_input.setText(str(bag))
        view.insv_input.setText(str(insv))
        view.out_input.setText(str(tmp_path / "output"))

        args = _workflow_args(view, runner)
        assert args[args.index("--rig-refine") + 1] == "photometric"
        assert args[args.index("--time-window") + 1] == "6"
        default_profile = view.rig_profile_combo.itemData(0)
        assert default_profile.endswith("raven_x4_montagem_1.json")
        assert args[args.index("--profile") + 1] == default_profile
        assert "--remove-dynamic" not in args

        runner.start_job.reset_mock()
        view.remove_dynamic_chk.setChecked(True)
        assert "--remove-dynamic" in _workflow_args(view, runner)

        runner.start_job.reset_mock()
        view.rig_refine_combo.setCurrentIndex(view.rig_refine_combo.findData("none"))
        view.time_window_spin.setValue(0.0)
        if view.rig_profile_combo.count() > 1:
            view.rig_profile_combo.setCurrentIndex(1)
        args = _workflow_args(view, runner)
        assert args[args.index("--rig-refine") + 1] == "none"
        assert args[args.index("--time-window") + 1] == "0"
        assert args[args.index("--profile") + 1] == view.rig_profile_combo.currentData()
    finally:
        view.close()
