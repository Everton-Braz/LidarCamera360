"""Compare two sets of binary PNG masks (white=keep, black=remove)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


def _mask_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"Mask directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if not files:
        raise ValueError(f"No PNG masks found in: {root}")
    return files


def _read_binary_mask(path: Path, relative_path: str) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read mask {relative_path}: {exc}") from exc
    mask = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Cannot decode PNG mask: {relative_path}")
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be a single-channel grayscale PNG: {relative_path} "
            f"(got shape {mask.shape})"
        )
    values = np.unique(mask)
    if not np.isin(values, (0, 255)).all():
        raise ValueError(
            f"Mask must contain only 0 (remove) and 255 (keep): {relative_path} "
            f"(found values {values[:8].tolist()})"
        )
    return mask == 0


def _scores(reference_removed: np.ndarray, candidate_removed: np.ndarray) -> dict[str, Any]:
    if reference_removed.shape != candidate_removed.shape:
        raise ValueError(
            f"Mask shape mismatch: reference {reference_removed.shape}, "
            f"candidate {candidate_removed.shape}"
        )
    intersection = int(np.count_nonzero(reference_removed & candidate_removed))
    reference_count = int(np.count_nonzero(reference_removed))
    candidate_count = int(np.count_nonzero(candidate_removed))
    union = int(np.count_nonzero(reference_removed | candidate_removed))
    pixels = int(reference_removed.size)
    matching = int(np.count_nonzero(reference_removed == candidate_removed))
    return {
        "width": int(reference_removed.shape[1]),
        "height": int(reference_removed.shape[0]),
        "pixels": pixels,
        "reference_removed_pixels": reference_count,
        "candidate_removed_pixels": candidate_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "matching_pixels": matching,
        # Empty-vs-empty is a perfect match by convention.
        "removal_iou": intersection / union if union else 1.0,
        "pixel_agreement": matching / pixels,
    }


def compare_mask_sets(reference: Path | str, candidate: Path | str) -> dict[str, Any]:
    """Compare matching relative PNG paths and return per-frame and global scores.

    Masks are expected to be binary grayscale PNGs with 255 meaning keep and
    0 meaning remove. The directories must contain precisely the same relative
    PNG paths, and corresponding masks must have identical dimensions.
    """
    reference_root = Path(reference)
    candidate_root = Path(candidate)
    reference_files = _mask_files(reference_root)
    candidate_files = _mask_files(candidate_root)

    reference_names = set(reference_files)
    candidate_names = set(candidate_files)
    missing_candidate = sorted(reference_names - candidate_names)
    missing_reference = sorted(candidate_names - reference_names)
    if missing_candidate or missing_reference:
        details = []
        if missing_candidate:
            details.append("missing from candidate: " + ", ".join(missing_candidate))
        if missing_reference:
            details.append("missing from reference: " + ", ".join(missing_reference))
        raise ValueError("Mask file sets differ (" + "; ".join(details) + ")")

    frames: list[dict[str, Any]] = []
    total_pixels = 0
    total_matching = 0
    total_intersection = 0
    total_union = 0
    total_reference_removed = 0
    total_candidate_removed = 0
    for relative_path in sorted(reference_names):
        reference_removed = _read_binary_mask(reference_files[relative_path], relative_path)
        candidate_removed = _read_binary_mask(candidate_files[relative_path], relative_path)
        if reference_removed.shape != candidate_removed.shape:
            raise ValueError(
                f"Mask shape mismatch for {relative_path}: reference "
                f"{reference_removed.shape}, candidate {candidate_removed.shape}"
            )
        scores = _scores(reference_removed, candidate_removed)
        frames.append({"path": relative_path, **scores})
        total_pixels += scores["pixels"]
        total_matching += scores["matching_pixels"]
        total_intersection += scores["intersection_pixels"]
        total_union += scores["union_pixels"]
        total_reference_removed += scores["reference_removed_pixels"]
        total_candidate_removed += scores["candidate_removed_pixels"]

    return {
        "schema": 1,
        "polarity": "white_keep_black_remove",
        "frame_count": len(frames),
        "reference_directory": str(reference_root.resolve()),
        "candidate_directory": str(candidate_root.resolve()),
        "overall": {
            "pixels": total_pixels,
            "reference_removed_pixels": total_reference_removed,
            "candidate_removed_pixels": total_candidate_removed,
            "intersection_pixels": total_intersection,
            "union_pixels": total_union,
            "matching_pixels": total_matching,
            "removal_iou": total_intersection / total_union if total_union else 1.0,
            "pixel_agreement": total_matching / total_pixels,
        },
        "frames": frames,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare binary white=keep PNG mask sets by relative path."
    )
    parser.add_argument("--reference", type=Path, required=True, help="Reference mask directory")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate mask directory")
    parser.add_argument(
        "--output", type=Path, help="Write JSON report here (default: print to stdout)"
    )
    args = parser.parse_args(argv)

    try:
        report = compare_mask_sets(args.reference, args.candidate)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Compared {report['frame_count']} masks; report: {args.output}")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
