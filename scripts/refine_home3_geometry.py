"""Create a blocked-CV-gated Home3 alignment and rig sidecar candidate."""
import argparse
import json
from pathlib import Path

from raven_app.timelapse_calibration import align_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dt-hint', type=float)
    args = parser.parse_args(argv)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    alignment_path = output / 'colmap_to_lidar_alignment.json'
    alignment = align_dataset(args.dataset, args.dt_hint,
                              output_path=alignment_path,
                              validate_refinement=True)

    # Reuse the production extrinsic calculator, but pass the candidate
    # alignment explicitly and disable its JSON writes to the source dataset.
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    calibration = pipeline.recalibrate_from_sfm(
        args.dataset.resolve(), alignment=alignment, write_outputs=False)
    calibration_path = output / 'rig_calibration_candidate.json'
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding='utf-8')

    report = {
        'dataset': str(args.dataset.resolve()),
        'alignment_candidate': str(alignment_path),
        'rig_candidate': str(calibration_path),
        'timelapse_calibration_version': alignment.get('timelapse_calibration_version'),
        'trajectory_rmse_cm': alignment.get('trajectory_rmse_cm'),
        'lever_arm_body_m': alignment.get('initial_camera_lever_body_m'),
        'dt_sync_seconds': alignment.get('dt_sync_seconds'),
        'lever_arm_refinement': alignment.get('lever_arm_refinement'),
        'source_dataset_json_written': False,
        'lidar_xyz_changed': False,
    }
    report_path = output / 'geometry_refinement_validation.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = report.get('lever_arm_refinement') or {}
    print(f"Candidate alignment: {alignment_path}")
    print(f"Candidate rig sidecar: {calibration_path}")
    print(f"Refinement accepted: {summary.get('accepted', False)}")
    if 'baseline_untrimmed_rmse_cm' in summary:
        print('Blocked untrimmed RMSE: '
              f"{summary['baseline_untrimmed_rmse_cm']:.3f} -> "
              f"{summary['candidate_untrimmed_rmse_cm']:.3f} cm")
    print(f"Report: {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
