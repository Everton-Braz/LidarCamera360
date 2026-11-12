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

        # Buttons to create and manage profiles exist
        assert hasattr(view, "btn_new_rig")
        assert hasattr(view, "btn_manage_rigs")
        assert any("raven_one_rs" in (view.rig_profile_combo.itemData(i) or "")
                   for i in range(view.rig_profile_combo.count()))

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


def test_workflow_rig_actions_and_dialog(tmp_path, monkeypatch):
    runner = ProcessRunner()
    runner.start_job = Mock(return_value=True)
    view = UnifiedWorkflowView(runner)
    try:
        user_dir = tmp_path / "user_profiles"
        user_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("raven_app.rig_profile.user_profiles_dir", lambda: user_dir)
        monkeypatch.setattr("raven_app.profile_manager.get_user_profiles_dir", lambda: user_dir)

        # Test _on_manage_rigs switches tab if window has calib_view
        class FakeWindow:
            def __init__(self):
                self.calib_view = object()
                self.switched = None
            def switchTo(self, target):
                self.switched = target
        fake_win = FakeWindow()
        monkeypatch.setattr(view, "window", lambda: fake_win)
        view._on_manage_rigs()
        assert fake_win.switched is fake_win.calib_view

        # Test _on_new_rig
        from raven_app.rig_bootstrap import build_profile
        test_prof = build_profile('My New Rig', camera='insta360_one_rs', scanner_preset='raven')
        class FakeDialog:
            def __init__(self, parent=None): pass
            def exec(self): return 1
            def build(self): return test_prof
        monkeypatch.setattr("raven_app.views.calibration_view.NewRigDialog", FakeDialog)

        view._on_new_rig()
        assert (user_dir / "my_new_rig.json").is_file()
        assert view.rig_profile_combo.currentData() == str(user_dir / "my_new_rig.json")
    finally:
        view.close()

