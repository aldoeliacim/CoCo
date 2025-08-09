"""
Data Validation for YOLO Training

This script validates annotation quality and dataset integrity.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger_setup import setup_logger

logger = setup_logger("data_validation")


class DataValidator:
    """Validate annotation quality and dataset integrity."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        logger.info(f"Initialized data validator for: {self.dataset_dir}")

    def validate_file_structure(self) -> bool:
        """Validate dataset file structure."""
        logger.info("🔍 Validating file structure...")

        required_dirs = [
            'train/images', 'train/labels',
            'val/images', 'val/labels'
        ]

        required_files = ['dataset.yaml']

        # Check directories
        for dir_path in required_dirs:
            full_path = self.dataset_dir / dir_path
            if not full_path.exists():
                logger.error(f"❌ Missing directory: {dir_path}")
                return False
            logger.info(f"✅ Found directory: {dir_path}")

        # Check files
        for file_path in required_files:
            full_path = self.dataset_dir / file_path
            if not full_path.exists():
                logger.error(f"❌ Missing file: {file_path}")
                return False
            logger.info(f"✅ Found file: {file_path}")

        return True

    def validate_image_label_pairs(self) -> Dict[str, Dict]:
        """Validate that each image has a corresponding label file."""
        logger.info("🔍 Validating image-label pairs...")

        results = {}

        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / split / 'images'
            labels_dir = self.dataset_dir / split / 'labels'

            if not images_dir.exists():
                continue

            # Get all images
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(images_dir.glob(ext))

            # Check for corresponding labels
            matched_pairs = 0
            missing_labels = []
            orphaned_labels = []

            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    matched_pairs += 1
                else:
                    missing_labels.append(img_file.name)

            # Check for orphaned labels
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                for label_file in label_files:
                    # Check if corresponding image exists
                    img_found = False
                    for ext in ['.png', '.jpg', '.jpeg']:
                        img_file = images_dir / f"{label_file.stem}{ext}"
                        if img_file.exists():
                            img_found = True
                            break
                    if not img_found:
                        orphaned_labels.append(label_file.name)

            results[split] = {
                'total_images': len(image_files),
                'matched_pairs': matched_pairs,
                'missing_labels': missing_labels,
                'orphaned_labels': orphaned_labels
            }

            logger.info(f"Split '{split}':")
            logger.info(f"  Images: {len(image_files)}")
            logger.info(f"  Matched pairs: {matched_pairs}")
            if missing_labels:
                logger.warning(f"  Missing labels: {len(missing_labels)}")
            if orphaned_labels:
                logger.warning(f"  Orphaned labels: {len(orphaned_labels)}")

        return results

    def validate_annotation_format(self) -> Dict[str, List]:
        """Validate YOLO annotation format."""
        logger.info("🔍 Validating annotation format...")

        issues = {
            'format_errors': [],
            'bbox_errors': [],
            'class_errors': []
        }

        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_dir / split / 'labels'

            if not labels_dir.exists():
                continue

            label_files = list(labels_dir.glob('*.txt'))

            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]

                    for line_num, line in enumerate(lines, 1):
                        parts = line.split()

                        # Check format (should be: class x_center y_center width height)
                        if len(parts) != 5:
                            issues['format_errors'].append(
                                f"{label_file.name}:{line_num} - Expected 5 values, got {len(parts)}"
                            )
                            continue

                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                        except ValueError as e:
                            issues['format_errors'].append(
                                f"{label_file.name}:{line_num} - Invalid number format: {e}"
                            )
                            continue

                        # Validate class ID (should be 0 for panels)
                        if class_id != 0:
                            issues['class_errors'].append(
                                f"{label_file.name}:{line_num} - Invalid class {class_id}, expected 0"
                            )

                        # Validate bbox coordinates (should be 0-1)
                        for coord_name, coord_value in [
                            ('x_center', x_center), ('y_center', y_center),
                            ('width', width), ('height', height)
                        ]:
                            if not (0 <= coord_value <= 1):
                                issues['bbox_errors'].append(
                                    f"{label_file.name}:{line_num} - {coord_name} {coord_value} out of range [0,1]"
                                )

                except Exception as e:
                    issues['format_errors'].append(f"{label_file.name} - Read error: {e}")

        # Report issues
        for issue_type, issue_list in issues.items():
            if issue_list:
                logger.warning(f"Found {len(issue_list)} {issue_type}:")
                for issue in issue_list[:5]:  # Show first 5
                    logger.warning(f"  {issue}")
                if len(issue_list) > 5:
                    logger.warning(f"  ... and {len(issue_list) - 5} more")
            else:
                logger.info(f"✅ No {issue_type} found")

        return issues

    def validate_image_integrity(self) -> Dict[str, List]:
        """Validate image file integrity."""
        logger.info("🔍 Validating image integrity...")

        issues = {
            'corrupt_images': [],
            'size_issues': [],
            'format_issues': []
        }

        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_dir / split / 'images'

            if not images_dir.exists():
                continue

            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(images_dir.glob(ext))

            for img_file in image_files:
                try:
                    # Try to load image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        issues['corrupt_images'].append(str(img_file))
                        continue

                    # Check size
                    height, width = img.shape[:2]
                    if width < 32 or height < 32:
                        issues['size_issues'].append(f"{img_file.name} - {width}x{height} too small")
                    elif width > 10000 or height > 10000:
                        issues['size_issues'].append(f"{img_file.name} - {width}x{height} too large")

                    # Check format
                    if img.ndim != 3:
                        issues['format_issues'].append(f"{img_file.name} - Not 3-channel")

                except Exception as e:
                    issues['corrupt_images'].append(f"{img_file.name} - {e}")

        # Report issues
        for issue_type, issue_list in issues.items():
            if issue_list:
                logger.warning(f"Found {len(issue_list)} {issue_type}:")
                for issue in issue_list[:3]:  # Show first 3
                    logger.warning(f"  {issue}")
                if len(issue_list) > 3:
                    logger.warning(f"  ... and {len(issue_list) - 3} more")
            else:
                logger.info(f"✅ No {issue_type} found")

        return issues

    def generate_statistics(self) -> Dict:
        """Generate dataset statistics."""
        logger.info("📊 Generating dataset statistics...")

        stats = {
            'splits': {},
            'totals': {
                'images': 0,
                'annotations': 0,
                'panels': 0
            },
            'panel_size_stats': {
                'widths': [],
                'heights': [],
                'areas': []
            }
        }

        for split in ['train', 'val', 'test']:
            split_stats = {
                'images': 0,
                'annotations': 0,
                'panels': 0,
                'avg_panels_per_image': 0
            }

            images_dir = self.dataset_dir / split / 'images'
            labels_dir = self.dataset_dir / split / 'labels'

            if not images_dir.exists():
                continue

            # Count images
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(images_dir.glob(ext))
            split_stats['images'] = len(image_files)

            # Count annotations and panels
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                split_stats['annotations'] = len(label_files)

                total_panels = 0
                for label_file in label_files:
                    try:
                        with open(label_file, 'r') as f:
                            lines = [line.strip() for line in f if line.strip()]
                            panel_count = len(lines)
                            total_panels += panel_count

                            # Collect panel size statistics
                            for line in lines:
                                parts = line.split()
                                if len(parts) == 5:
                                    try:
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        area = width * height

                                        stats['panel_size_stats']['widths'].append(width)
                                        stats['panel_size_stats']['heights'].append(height)
                                        stats['panel_size_stats']['areas'].append(area)
                                    except ValueError:
                                        pass

                    except Exception as e:
                        logger.warning(f"Error reading {label_file}: {e}")

                split_stats['panels'] = total_panels
                if split_stats['images'] > 0:
                    split_stats['avg_panels_per_image'] = total_panels / split_stats['images']

            stats['splits'][split] = split_stats

            # Add to totals
            stats['totals']['images'] += split_stats['images']
            stats['totals']['annotations'] += split_stats['annotations']
            stats['totals']['panels'] += split_stats['panels']

        # Calculate panel size statistics
        if stats['panel_size_stats']['widths']:
            for metric in ['widths', 'heights', 'areas']:
                values = stats['panel_size_stats'][metric]
                stats['panel_size_stats'][f'{metric}_mean'] = np.mean(values)
                stats['panel_size_stats'][f'{metric}_std'] = np.std(values)
                stats['panel_size_stats'][f'{metric}_min'] = np.min(values)
                stats['panel_size_stats'][f'{metric}_max'] = np.max(values)

        return stats

    def run_full_validation(self) -> Dict:
        """Run complete dataset validation."""
        logger.info("🚀 Running full dataset validation")

        validation_results = {
            'structure_valid': False,
            'pair_validation': {},
            'format_issues': {},
            'image_issues': {},
            'statistics': {}
        }

        # Validate structure
        validation_results['structure_valid'] = self.validate_file_structure()

        if not validation_results['structure_valid']:
            logger.error("❌ Structure validation failed, skipping further checks")
            return validation_results

        # Validate pairs
        validation_results['pair_validation'] = self.validate_image_label_pairs()

        # Validate annotation format
        validation_results['format_issues'] = self.validate_annotation_format()

        # Validate image integrity
        validation_results['image_issues'] = self.validate_image_integrity()

        # Generate statistics
        validation_results['statistics'] = self.generate_statistics()

        # Overall assessment
        has_critical_issues = (
            not validation_results['structure_valid'] or
            validation_results['format_issues']['format_errors'] or
            validation_results['image_issues']['corrupt_images']
        )

        if has_critical_issues:
            logger.error("❌ Dataset has critical issues that must be fixed")
        else:
            logger.info("✅ Dataset validation passed")

        return validation_results


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO training dataset")
    parser.add_argument("--dataset", required=True,
                       help="Path to dataset directory")
    parser.add_argument("--output", help="Output file for validation report (JSON)")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: Dataset directory {args.dataset} does not exist")
        return 1

    print("🎯 Dataset Validation")
    print(f"📁 Dataset: {args.dataset}")

    try:
        # Run validation
        validator = DataValidator(args.dataset)
        results = validator.run_full_validation()

        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📄 Validation report saved to: {args.output}")

        # Print summary
        stats = results['statistics']
        print(f"\n📊 Dataset Summary:")
        print(f"  Total images: {stats['totals']['images']}")
        print(f"  Total annotations: {stats['totals']['annotations']}")
        print(f"  Total panels: {stats['totals']['panels']}")

        if stats['totals']['images'] > 0:
            print(f"  Avg panels per image: {stats['totals']['panels']/stats['totals']['images']:.1f}")

        # Check for issues
        total_issues = (
            len(results['format_issues']['format_errors']) +
            len(results['format_issues']['bbox_errors']) +
            len(results['format_issues']['class_errors']) +
            len(results['image_issues']['corrupt_images']) +
            len(results['image_issues']['size_issues']) +
            len(results['image_issues']['format_issues'])
        )

        if total_issues == 0:
            print(f"\n✅ Validation successful - dataset is ready for training!")
            return 0
        else:
            print(f"\n⚠️  Found {total_issues} issues - please review and fix before training")
            return 1

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
