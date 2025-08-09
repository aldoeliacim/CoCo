"""
Dataset Builder for YOLO Training

This script converts human-curated panel annotations into YOLO training format.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import YOLO_PANEL_MODEL_PATH
from src.logger_setup import setup_logger

logger = setup_logger("dataset_builder")


class YOLODatasetBuilder:
    """Build YOLO-format dataset from annotations."""

    def __init__(self, annotations_dir: str, output_dir: str):
        self.annotations_dir = Path(annotations_dir)
        self.output_dir = Path(output_dir)

        # Validate inputs
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")

        # Create output structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized YOLO dataset builder")
        logger.info(f"  Annotations: {self.annotations_dir}")
        logger.info(f"  Output: {self.output_dir}")

    def validate_annotations(self) -> Tuple[List[Path], List[Path]]:
        """
        Validate annotation files and find corresponding images.

        Returns:
            Tuple of (annotation_files, image_files)
        """
        annotation_files = list(self.annotations_dir.glob("*.txt"))
        if not annotation_files:
            raise ValueError(f"No annotation files found in {self.annotations_dir}")

        # Find corresponding images
        approved_dir = self.annotations_dir.parent / "approved"
        if not approved_dir.exists():
            raise FileNotFoundError(f"Approved images directory not found: {approved_dir}")

        image_files = []
        valid_annotations = []

        for ann_file in annotation_files:
            # Look for corresponding image
            image_found = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                img_file = approved_dir / f"{ann_file.stem}{ext}"
                if img_file.exists():
                    image_files.append(img_file)
                    valid_annotations.append(ann_file)
                    image_found = True
                    break

            if not image_found:
                logger.warning(f"No image found for annotation: {ann_file.name}")

        logger.info(f"Found {len(valid_annotations)} valid annotation-image pairs")
        return valid_annotations, image_files

    def create_splits(self, annotation_files: List[Path], image_files: List[Path],
                     train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict[str, Dict]:
        """
        Split data into train/validation/test sets.

        Args:
            annotation_files: List of annotation file paths
            image_files: List of corresponding image file paths
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set

        Returns:
            Dictionary with split information
        """
        if len(annotation_files) != len(image_files):
            raise ValueError("Mismatch between annotation and image file counts")

        # Create paired data
        data_pairs = list(zip(annotation_files, image_files))

        # Shuffle for random split
        import random
        random.seed(42)  # For reproducible splits
        random.shuffle(data_pairs)

        # Calculate split sizes
        n_total = len(data_pairs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Create splits
        train_pairs = data_pairs[:n_train]
        val_pairs = data_pairs[n_train:n_train + n_val]
        test_pairs = data_pairs[n_train + n_val:]

        splits = {
            'train': {
                'pairs': train_pairs,
                'count': len(train_pairs)
            },
            'val': {
                'pairs': val_pairs,
                'count': len(val_pairs)
            },
            'test': {
                'pairs': test_pairs,
                'count': len(test_pairs)
            }
        }

        logger.info(f"Dataset splits:")
        logger.info(f"  Total: {n_total}")
        logger.info(f"  Train: {len(train_pairs)} ({len(train_pairs)/n_total*100:.1f}%)")
        logger.info(f"  Val: {len(val_pairs)} ({len(val_pairs)/n_total*100:.1f}%)")
        logger.info(f"  Test: {len(test_pairs)} ({len(test_pairs)/n_total*100:.1f}%)")

        return splits

    def copy_split_data(self, splits: Dict[str, Dict]) -> None:
        """
        Copy data files to appropriate split directories.

        Args:
            splits: Split information from create_splits()
        """
        for split_name, split_data in splits.items():
            # Create directories
            split_dir = self.output_dir / split_name
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for ann_file, img_file in split_data['pairs']:
                # Copy image
                dst_img = images_dir / img_file.name
                if not dst_img.exists():
                    shutil.copy2(img_file, dst_img)

                # Copy annotation
                dst_ann = labels_dir / ann_file.name
                if not dst_ann.exists():
                    shutil.copy2(ann_file, dst_ann)

            logger.info(f"Copied {split_data['count']} files to {split_name} split")

    def create_dataset_yaml(self) -> str:
        """
        Create YOLO dataset configuration file.

        Returns:
            Path to the created YAML file
        """
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'panel'
            },
            'nc': 1  # number of classes
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created dataset YAML: {yaml_path}")
        return str(yaml_path)

    def validate_yolo_format(self) -> bool:
        """
        Validate that the dataset is in correct YOLO format.

        Returns:
            True if valid, False otherwise
        """
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']

        for dir_path in required_dirs:
            full_path = self.output_dir / dir_path
            if not full_path.exists():
                logger.error(f"Missing required directory: {full_path}")
                return False

            # Check for files
            if dir_path.endswith('images'):
                files = list(full_path.glob('*.png')) + list(full_path.glob('*.jpg'))
            else:
                files = list(full_path.glob('*.txt'))

            if not files:
                logger.error(f"No files found in: {full_path}")
                return False

        # Check YAML file
        yaml_file = self.output_dir / "dataset.yaml"
        if not yaml_file.exists():
            logger.error(f"Dataset YAML file missing: {yaml_file}")
            return False

        logger.info("✅ Dataset validation passed")
        return True

    def generate_statistics(self, splits: Dict[str, Dict]) -> Dict:
        """
        Generate dataset statistics.

        Args:
            splits: Split information from create_splits()

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_images': sum(split['count'] for split in splits.values()),
            'total_panels': 0,
            'splits': {}
        }

        for split_name, split_data in splits.items():
            split_stats = {
                'images': split_data['count'],
                'panels': 0
            }

            # Count panels in annotations
            for ann_file, _ in split_data['pairs']:
                try:
                    with open(ann_file, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        split_stats['panels'] += len(lines)
                except Exception as e:
                    logger.warning(f"Error reading {ann_file}: {e}")

            stats['splits'][split_name] = split_stats
            stats['total_panels'] += split_stats['panels']

        return stats

    def build_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict:
        """
        Build complete YOLO dataset.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set

        Returns:
            Dataset statistics
        """
        logger.info("🚀 Building YOLO dataset")

        # Validate annotations
        annotation_files, image_files = self.validate_annotations()

        # Create splits
        splits = self.create_splits(annotation_files, image_files, train_ratio, val_ratio)

        # Copy data to split directories
        self.copy_split_data(splits)

        # Create dataset YAML
        self.create_dataset_yaml()

        # Validate final format
        if not self.validate_yolo_format():
            raise RuntimeError("Dataset validation failed")

        # Generate statistics
        stats = self.generate_statistics(splits)

        # Save statistics
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("📊 Dataset statistics:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Total panels: {stats['total_panels']}")
        logger.info(f"  Avg panels per image: {stats['total_panels']/stats['total_images']:.1f}")

        logger.info("✅ Dataset build completed successfully")
        return stats


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from annotations")
    parser.add_argument("--annotations", required=True,
                       help="Directory with annotation files")
    parser.add_argument("--output", required=True,
                       help="Output directory for YOLO dataset")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation set ratio (default: 0.2)")

    args = parser.parse_args()

    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be < 1.0")
        return 1

    print("🎯 YOLO Dataset Builder")
    print(f"📁 Annotations: {args.annotations}")
    print(f"💾 Output: {args.output}")
    print(f"📊 Splits: {args.train_ratio:.1f} train, {args.val_ratio:.1f} val, {1-args.train_ratio-args.val_ratio:.1f} test")

    try:
        # Build dataset
        builder = YOLODatasetBuilder(args.annotations, args.output)
        stats = builder.build_dataset(args.train_ratio, args.val_ratio)

        print(f"\n✅ Dataset built successfully!")
        print(f"📄 Dataset config: {Path(args.output) / 'dataset.yaml'}")
        print(f"📊 Statistics: {Path(args.output) / 'dataset_stats.json'}")

        return 0

    except Exception as e:
        print(f"\n❌ Dataset building failed: {e}")
        logger.error(f"Dataset building failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
