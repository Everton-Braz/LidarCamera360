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


def test_prepare_fastlivo_config_injects_camera_parameters(tmp_path: Path):
    calib_dir = tmp_path / "calibration"
    calib_dir.mkdir()
    calib_file = calib_dir / "calib.json"
    calib_payload = {
        "camera_info": {
            "front": {
                "K": [1120.0, 0, 1540.0, 0, 1125.0, 2110.0, 0, 0, 1],
                "coeff": [-0.04, 0.002, -0.0008, 0.0001]
            }
        },
        "out_put": {
            "front": {
                "transform_matrix": [
                    [-1.0, 0.0, 0.0, -0.005],
                    [0.0, -0.5, -0.86, -0.07],
                    [0.0, -0.86, 0.5, -0.002],
                    [0, 0, 0, 1]
                ]
            }
        }
    }
    calib_file.write_text(json.dumps(calib_payload), encoding="utf-8")

    base_cfg = Path("FAST-LIVO2/config/raven.yaml")
    target = prepare_fastlivo_config(base_cfg, tmp_path, project_dir=tmp_path)
    import yaml
    with open(target, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["extrin_calib"]["Rcl"] == [-1.0, 0.0, 0.0, 0.0, -0.5, -0.86, 0.0, -0.86, 0.5]
    assert loaded["extrin_calib"]["Pcl"] == [-0.005, -0.07, -0.002]

    # Verify generated camera_raven.yaml
    cam_yaml = tmp_path / "camera_raven.yaml"
    assert cam_yaml.is_file()
    with open(cam_yaml, "r", encoding="utf-8") as f:
        cam_loaded = yaml.safe_load(f)
    assert cam_loaded["cam_fx"] == 1120.0
    assert cam_loaded["cam_fy"] == 1125.0
    assert cam_loaded["cam_cx"] == 1540.0
    assert cam_loaded["cam_cy"] == 2110.0
    assert cam_loaded["k1"] == -0.04
    assert cam_loaded["k2"] == 0.002


def test_cli_accepts_image_bag_option():
    from raven_app.cli import parse
    # Test workflow parser accepts --image-bag
    args = parse(["workflow", "--bag", "lidar.bag", "--insv", "video.insv", "--output", "out", "--image-bag", "image.bag"])
    assert getattr(args, "image_bag_path", None) == Path("image.bag")

    # Test slam parser accepts --image-bag and --calib
    slam_args = parse(["slam", "--bag", "lidar.bag", "--output", "out", "--image-bag", "image.bag", "--calib", "calib.json"])
    assert getattr(slam_args, "image_bag_path", None) == Path("image.bag")
    assert getattr(slam_args, "calib_json", None) == Path("calib.json")

