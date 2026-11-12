"""Rig profiles page: form <-> profile round trip, saving to the user folder, wizard, calibrate job."""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from raven_app import profile_manager
from raven_app.rig_profile import load_profile
from raven_app.views import calibration_view
from raven_app.views.calibration_view import CalibrationView, NewRigDialog, apply_camera_rows, camera_rows

APP = QApplication.instance() or QApplication([])
BUNDLED = Path(__file__).resolve().parents[1] / 'configs' / 'profiles' / 'raven_x4_montagem_1.json'


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    user = tmp_path / 'user'
    monkeypatch.setattr(profile_manager, 'get_user_profiles_dir', lambda: user)
    monkeypatch.setattr(calibration_view, 'get_user_profiles_dir', lambda: user)
    return user


def test_table_rows_round_trip_the_transforms():
    profile = load_profile(BUNDLED)
    rows = camera_rows(profile)
    assert [r[0] for r in rows] == ['cam0', 'cam1']
    assert rows[0][12] == pytest.approx(-31.6, abs=0.5)       # Raven tilt shows up as roll
    before = [np.asarray(c.T) for c in profile.cameras]
    apply_camera_rows(profile, rows)
    for a, c in zip(before, profile.cameras):
        assert np.allclose(a, c.T, atol=1e-9)


def test_editing_a_bundled_profile_saves_a_user_copy(user_dir):
    view = CalibrationView()
    try:
        assert view.camera_table.rowCount() == 2
        view.name_edit.setText('Raven X4 edited')
        view.camera_table.item(0, 9).setText('5.000')        # cam0 X position, cm
        view._save_profile()
        saved = user_dir / 'raven_x4_montagem_1.json'
        data = json.loads(saved.read_text(encoding='utf-8'))
        assert data['schema'] == 2 and data['name'] == 'Raven X4 edited' and data['is_default'] is False
        assert data['T_lidar_to_cam0_rigid_4x4'][0][3] == pytest.approx(0.05)
        assert json.loads(BUNDLED.read_text(encoding='utf-8'))['name'] != 'Raven X4 edited'
    finally:
        view.close()


def test_auxiliary_cameras_can_be_added_and_removed(user_dir):
    view = CalibrationView()
    try:
        view._add_aux_camera()
        assert view.camera_table.rowCount() == 3
        assert view.camera_table.item(2, 0).text() == 'cam2'
        assert view.form_profile().auxiliary[0].cameras[0].model == 'OPENCV'
        view.camera_table.selectRow(2)
        view._remove_aux_camera()
        assert view.camera_table.rowCount() == 2
    finally:
        view.close()


def test_wizard_builds_a_profile_and_calibrate_uses_the_coarse_search(user_dir, tmp_path):
    host = CalibrationView()
    dialog = NewRigDialog(host)
    dialog.name_edit.setText('Pole GoPro')
    dialog.camera_combo.setCurrentIndex(dialog.camera_combo.findData('gopro_wide'))
    dialog.offset[2].setValue(25.0)
    profile = dialog.build()
    assert profile.profile_id == 'pole_gopro' and profile.main.cameras[0].model == 'OPENCV_FISHEYE'
    assert np.asarray(profile.main.cameras[0].T)[2, 3] == pytest.approx(0.25)

    dialog.close()
    host.close()
    view = CalibrationView()
    try:
        view._show_profile(profile)
        assert view.coarse_check.isChecked()
        view.dataset_edit.setText(str(tmp_path / 'capture'))
        view.dt_edit.setText('3,25')
        args = view.validation_args(tmp_path / 'p.json', tmp_path / 'r.json')
        assert args[:3] == ['calibrate', '--dataset', str(tmp_path / 'capture')]
        assert '--coarse' in args and args[args.index('--dt') + 1] == '3.25'
        assert args[args.index('--report') + 1] == str(tmp_path / 'r.json')
    finally:
        view.close()


def test_user_profile_deletion_and_signal(user_dir, monkeypatch):
    view = CalibrationView()
    try:
        # Bundled profile cannot be deleted
        assert not view.btn_delete.isEnabled()

        signal_emitted = []
        view.profiles_changed.connect(lambda: signal_emitted.append(True))

        # Save as a user profile
        view.name_edit.setText('Custom Test Rig')
        view._save_profile_as_name = 'custom_test_rig'
        monkeypatch.setattr('raven_app.views.calibration_view.QInputDialog.getText', lambda *a, **kw: ('Custom Test Rig', True))
        view._save_profile_as()

        saved = user_dir / 'custom_test_rig.json'
        assert saved.is_file()
        assert view.btn_delete.isEnabled()
        assert len(signal_emitted) >= 1

        # Delete with confirmed dialog
        monkeypatch.setattr('raven_app.views.calibration_view.MessageBox.exec', lambda self: 1)
        signal_emitted.clear()
        view._delete_profile()

        assert not saved.exists()
        assert len(signal_emitted) == 1
        assert not view.btn_delete.isEnabled()
    finally:
        view.close()

