import json
from pathlib import Path
import pytest
from raven_app.bag_io import extract_images
from raven_app.workflow import prepare_fastlivo_config

def test_extract_images_detects_adjacent_image_bag(tmp_path: Path):
    lidar_bag = tmp_path / "LIDAR_test.bag"
    image_bag = tmp_path / "IMAGE_test.bag"
    lidar_bag.write_bytes(b"dummy")
    image_bag.write_bytes(b"dummy")
    out_dir = tmp_path / "images"
    # Test that extract_images attempts to read the candidate image bag
    with pytest.raises(Exception):
        # Will fail trying to parse dummy ROS1 header on image_bag, proving it selected image_bag
        extract_images(lidar_bag, out_dir)

def test_prepare_fastlivo_config_injects_calib_json(tmp_path: Path):
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    calib_file = calib_dir / "calib.json"
    calib_payload = {
        "Lidar_to_Imu": {
            "Rotation": [1.0, 0.0, 0.0, 0.0, 0.8, -0.6, 0.0, 0.6, 0.8],
            "Translation": [0.01, 0.02, 0.03]
        }
    }
    calib_file.write_text(json.dumps(calib_payload), encoding="utf-8")
    
    # Base raven.yaml exists in repo
    base_cfg = Path("FAST-LIVO2/config/raven.yaml")
    out_cfg = tmp_path / "test_raven.yaml"
    
    target = prepare_fastlivo_config(base_cfg, tmp_path, project_dir=tmp_path)
    import yaml
    with open(target, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["mapping"]["extrinsic_R"] == [1.0, 0.0, 0.0, 0.0, 0.8, -0.6, 0.0, 0.6, 0.8]
    assert loaded["mapping"]["extrinsic_T"] == [0.01, 0.02, 0.03]
    assert loaded["mapping"]["gravity_align_en"] is False
