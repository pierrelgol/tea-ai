from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_index import index_dataset
from .geometry import run_geometry_checks
from .gui import launch_gui
from .integrity import run_integrity_checks
from .overlays import export_debug_overlays
from .predictions import run_prediction_checks
from .reports import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Check augmented dataset integrity and geometry")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/augmented"))
    parser.add_argument("--reports-dir", type=Path, default=Path("dataset/augmented/reports"))
    parser.add_argument("--outlier-threshold-px", type=float, default=2.0)
    parser.add_argument("--debug-overlays-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    records = index_dataset(args.dataset_root)
    integrity_issues, integrity_summary = run_integrity_checks(records)
    geometry_metrics, geometry_summary = run_geometry_checks(records, args.outlier_threshold_px)
    model_reports = run_prediction_checks(records, args.predictions_root)

    export_debug_overlays(
        records=records,
        reports_dir=args.reports_dir,
        n_per_split=args.debug_overlays_per_split,
        seed=args.seed,
    )

    write_reports(
        reports_dir=args.reports_dir,
        integrity_issues=integrity_issues,
        integrity_summary=integrity_summary,
        geometry_metrics=geometry_metrics,
        geometry_summary=geometry_summary,
        model_reports=model_reports,
    )

    print(f"checked {len(records)} samples")
    print(f"integrity issues: {integrity_summary['total_issues']}")
    print(f"geometry outliers: {geometry_summary['num_outliers']}")

    if not args.no_gui:
        launch_gui(
            records=records,
            integrity_issues=integrity_issues,
            geometry_metrics=geometry_metrics,
            model_reports=model_reports,
        )
