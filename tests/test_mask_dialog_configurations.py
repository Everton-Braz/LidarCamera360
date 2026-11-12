import sys

from PyQt6.QtWidgets import QApplication

# Ensure QApplication exists for GUI tests
app = QApplication.instance() or QApplication(sys.argv)

from raven_app.mask_settings_dialog import MaskSettingsDialog


def _cameras(dlg):
    return [dlg.camera_combo.itemText(i) for i in range(dlg.camera_combo.count())]


def test_mask_dialog_single_insv(tmp_path):
    dummy_insv = tmp_path / "test.insv"
    dummy_insv.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(tmp_path, insv_path=dummy_insv)
    assert _cameras(dlg) == ['cam0', 'cam1']
    assert dlg.secondary_insv_path is None
    dlg.close()


def test_mask_dialog_dual_insv(tmp_path):
    f0 = tmp_path / "VID_20260629_151302_00_013.insv"
    f1 = tmp_path / "VID_20260629_151302_10_013.insv"
    f0.write_bytes(b'dummy0')
    f1.write_bytes(b'dummy1')
    dlg = MaskSettingsDialog(tmp_path, insv_path=f0, secondary_insv_path=f1)
    assert dlg.secondary_insv_path == f1
    assert _cameras(dlg) == ['cam0', 'cam1']
    dlg.close()


def test_border_spin_applies_only_to_fisheye_cameras(tmp_path):
    insv = tmp_path / "test.insv"
    insv.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(tmp_path, insv_path=insv, camera_names=('cam2',))
    dlg.camera_combo.setCurrentText('cam2')
    dlg.border_spin.setValue(7)
    dlg.camera_combo.setCurrentText('cam0')
    dlg.camera_combo.setCurrentText('cam2')
    assert dlg.border_spin.value() == 7
    dlg.close()
