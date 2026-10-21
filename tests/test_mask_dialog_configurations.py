import tempfile
from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication
import sys

# Ensure QApplication exists for GUI tests
app = QApplication.instance() or QApplication(sys.argv)

from raven_app.mask_settings_dialog import MaskSettingsDialog


def test_mask_dialog_single_insv(tmp_path):
    dummy_insv = tmp_path / "test.insv"
    dummy_insv.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(insv_path=dummy_insv)
    cams = [dlg.camera_select.itemText(i) for i in range(dlg.camera_select.count())]
    assert cams == ['cam0', 'cam1']
    assert dlg._is_fisheye_camera('cam0') is True
    assert dlg._is_fisheye_camera('cam1') is True


def test_mask_dialog_dual_insv(tmp_path):
    f0 = tmp_path / "VID_20260629_151302_00_013.insv"
    f1 = tmp_path / "VID_20260629_151302_10_013.insv"
    f0.write_bytes(b'dummy0')
    f1.write_bytes(b'dummy1')
    dlg = MaskSettingsDialog(insv_path=f0, secondary_insv_path=f1)
    assert dlg.secondary_insv_path == f1
    cams = [dlg.camera_select.itemText(i) for i in range(dlg.camera_select.count())]
    assert cams == ['cam0', 'cam1']


def test_mask_dialog_bag_only(tmp_path):
    bag = tmp_path / "IMAGE_test.bag"
    bag.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(image_bag_path=bag, use_image_bag=True)
    cams = [dlg.camera_select.itemText(i) for i in range(dlg.camera_select.count())]
    assert cams == ['cam0']
    assert dlg._is_bag_camera('cam0') is True
    assert dlg._is_fisheye_camera('cam0') is False


def test_mask_dialog_insv_and_bag(tmp_path):
    insv = tmp_path / "test.insv"
    insv.write_bytes(b'dummy')
    bag = tmp_path / "IMAGE_test.bag"
    bag.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(insv_path=insv, image_bag_path=bag, use_image_bag=True)
    cams = [dlg.camera_select.itemText(i) for i in range(dlg.camera_select.count())]
    assert cams == ['cam0', 'cam1', 'cam2']
    assert dlg._is_fisheye_camera('cam0') is True
    assert dlg._is_fisheye_camera('cam1') is True
    assert dlg._is_bag_camera('cam2') is True
    assert dlg._is_fisheye_camera('cam2') is False


def test_cutoff_visibility_toggled(tmp_path):
    insv = tmp_path / "test.insv"
    insv.write_bytes(b'dummy')
    bag = tmp_path / "IMAGE_test.bag"
    bag.write_bytes(b'dummy')
    dlg = MaskSettingsDialog(insv_path=insv, image_bag_path=bag, use_image_bag=True)
    dlg.camera_select.setCurrentText('cam0')
    assert dlg.cutoff_spin.isVisible() is True
    dlg.camera_select.setCurrentText('cam2')
    assert dlg.cutoff_spin.isVisible() is False
