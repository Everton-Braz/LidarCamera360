import pytest
from pathlib import Path
import tempfile
from raven_app.bag_io import find_adjacent_image_bag, find_adjacent_lidar_bag, detect_bag_topics


def test_find_adjacent_image_and_lidar_bags():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        lidar = d / "LIDAR_covent.bag"
        image = d / "IMAGE_covent.bag"
        lidar.touch()
        image.touch()

        # Direct pairing
        assert find_adjacent_image_bag(lidar).resolve() == image.resolve()
        assert find_adjacent_lidar_bag(image).resolve() == lidar.resolve()

        # Never return self
        assert find_adjacent_image_bag(image) is None
        assert find_adjacent_lidar_bag(lidar) is None


def test_find_adjacent_pcl_and_camera_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        pcl = d / "PCL_scan_01.bag"
        img = d / "IMAGE_scan_01.bag"
        pcl.touch()
        img.touch()

        assert find_adjacent_image_bag(pcl).resolve() == img.resolve()
        assert find_adjacent_lidar_bag(img).resolve() == pcl.resolve()


def test_find_adjacent_generic_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        lidar = d / "my_raven_scan.bag"
        img = d / "camera_frames.bag"
        lidar.touch()
        img.touch()

        assert find_adjacent_lidar_bag(img).resolve() == lidar.resolve()


def test_early_bag_validation_rejects_missing_lidar_topics(monkeypatch):
    from raven_app.workflow import execute_unified_workflow

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        img_bag = d / "IMAGE_covent.bag"
        lidar_bag = d / "LIDAR_covent.bag"
        img_bag.touch()
        lidar_bag.touch()
        out = d / "out"

        monkeypatch.setattr(
            "raven_app.bag_io.detect_bag_topics",
            lambda *args, **kwargs: {"has_lidar": False, "has_imu": False, "has_image": True}
        )

        with pytest.raises(ValueError) as exc:
            execute_unified_workflow(img_bag, None, output_dir=out)

        assert "contains no LiDAR point cloud topics" in str(exc.value)
        assert "Did you mean to select 'LIDAR_covent.bag'?" in str(exc.value)

