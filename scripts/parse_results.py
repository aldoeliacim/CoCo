"""
Results Parser and Organizer

This script organizes analysis results into timestamped directories
and creates summary reports.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import json

def organize_latest_results(analysis_dir: Path, results_dir: Path):
    """
    Organize the latest analysis results into timestamped directories.

    Args:
        analysis_dir: Source analysis directory (e.g., out/analysis)
        results_dir: Target results directory (e.g., results)
    """
    if not analysis_dir.exists():
        print(f"❌ Analysis directory not found: {analysis_dir}")
        return False

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d")
    target_dir = results_dir / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)

    # Count items to copy
    items_copied = 0

    # Copy XML files
    xml_source = analysis_dir / "xml"
    if xml_source.exists():
        xml_target = target_dir / "analysis"
        xml_target.mkdir(exist_ok=True)

        for xml_file in xml_source.glob("*.xml"):
            shutil.copy2(xml_file, xml_target / xml_file.name)
            items_copied += 1

    # Copy annotated images
    annotated_source = analysis_dir / "annotated_panels"
    if annotated_source.exists():
        annotated_target = target_dir / "analysis"
        annotated_target.mkdir(exist_ok=True)

        for img_file in annotated_source.glob("*.png"):
            shutil.copy2(img_file, annotated_target / img_file.name)
            items_copied += 1

    # Create a simple summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "source_directory": str(analysis_dir),
        "target_directory": str(target_dir),
        "items_copied": items_copied,
        "xml_files": len(list((target_dir / "analysis").glob("*.xml"))) if (target_dir / "analysis").exists() else 0,
        "image_files": len(list((target_dir / "analysis").glob("*.png"))) if (target_dir / "analysis").exists() else 0
    }

    # Save summary
    summary_file = target_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Results organized to: {target_dir}")
    print(f"✓ Items copied: {items_copied}")
    print(f"✓ Summary saved: {summary_file}")

    return True

def main():
    """Main function for results parsing and organization."""
    parser = argparse.ArgumentParser(
        description="Organize CoCo analysis results into timestamped directories"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Organize the latest analysis results"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("out/analysis"),
        help="Source analysis directory (default: out/analysis)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Target results directory (default: results)"
    )

    args = parser.parse_args()

    if args.latest:
        success = organize_latest_results(args.analysis_dir, args.results_dir)
        if success:
            print("🎉 Results organization completed successfully!")
            sys.exit(0)
        else:
            print("❌ Results organization failed!")
            sys.exit(1)
    else:
        print("Please specify --latest to organize results")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
