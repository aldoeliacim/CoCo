"""
YOLO Model Fine-tuning Script

This script fine-tunes a YOLO model for comic panel detection using
human-curated training data.
"""

import os
import sys
import yaml
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Standardized project imports
import config
from config import setup_logger, YOLO_PANEL_MODEL_PATH, MODELS_DIR, OUT_DIR

from ultralytics import YOLO

logger = setup_logger("train_model")


class YOLOFineTuner:
    """Fine-tune YOLO model for comic panel detection."""

    def __init__(self, base_model_path: str, dataset_path: str, output_path: str):
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

        # Validate inputs
        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found: {self.base_model_path}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized YOLO fine-tuner")
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Output: {self.output_path}")

    def create_dataset_config(self) -> str:
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train',
            'val': 'val',
            'test': 'test',
            'names': {0: 'panel'},
            'nc': 1  # number of classes
        }

        config_path = self.dataset_path / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.info(f"Created dataset config: {config_path}")
        return str(config_path)

    def setup_training_config(self, epochs: int = 100, batch_size: int = 16,
                            learning_rate: float = 0.01, **kwargs) -> Dict:
        """
        Setup training configuration for dry run testing.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            **kwargs: Additional training parameters

        Returns:
            Training configuration dictionary
        """
        config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dataset': str(self.dataset_path),
            'base_model': str(self.base_model_path),
            'output': str(self.output_path),
            **kwargs
        }
        logger.info(f"Training config setup: {config}")
        return config

    def train(self, epochs: int = 100, batch_size: int = 16, img_size: int = 640,
              patience: int = 50, **kwargs) -> Dict:
        """
        Fine-tune the YOLO model.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            img_size: Input image size
            patience: Early stopping patience
            **kwargs: Additional YOLO training arguments

        Returns:
            Training results dictionary
        """
        logger.info("🚀 Starting YOLO fine-tuning")

        # Load base model
        model = YOLO(str(self.base_model_path))
        logger.info(f"Loaded base model: {self.base_model_path}")

        # Create dataset config
        dataset_config = self.create_dataset_config()

        # Training parameters
        train_params = {
            'data': dataset_config,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'patience': patience,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'project': str(self.output_path.parent),
            'name': self.output_path.stem,
            'exist_ok': True,
            **kwargs
        }

        logger.info("Training parameters:")
        for key, value in train_params.items():
            logger.info(f"{key}: {value}")

        # Train model
        try:
            results = model.train(**train_params)

            # Copy best model to final location
            best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
            if best_model_path.exists():
                shutil.copy2(best_model_path, self.output_path)
                logger.info(f"✅ Fine-tuned model saved to: {self.output_path}")
            else:
                logger.error(f"❌ Best model not found at: {best_model_path}")

            return results

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

    def validate(self, test_data: str = None) -> Dict:
        """
        Validate the fine-tuned model.

        Args:
            test_data: Path to test dataset (optional)

        Returns:
            Validation results
        """
        if not self.output_path.exists():
            raise FileNotFoundError(f"Fine-tuned model not found: {self.output_path}")

        logger.info("🔍 Validating fine-tuned model")

        model = YOLO(str(self.output_path))

        # Use dataset config if no test data specified
        test_path = test_data or str(self.dataset_path / "dataset.yaml")

        results = model.val(data=test_path)

        logger.info("Validation results:")
        logger.info(f"mAP50: {results.box.map50:.4f}")
        logger.info(f"mAP50-95: {results.box.map:.4f}")
        logger.info(f"Precision: {results.box.mp:.4f}")
        logger.info(f"Recall: {results.box.mr:.4f}")

        return results


def prepare_dataset_splits(annotations_dir: str, output_dir: str,
                          train_ratio: float = 0.7, val_ratio: float = 0.2) -> Dict[str, List[str]]:
    """
    Split annotations into train/val/test sets.

    Args:
        annotations_dir: Directory with annotation files
        output_dir: Output directory for dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data

    Returns:
        Dictionary with file lists for each split
    """
    annotations_path = Path(annotations_dir)
    output_path = Path(output_dir)

    # Find all annotation files
    annotation_files = list(annotations_path.glob("*.txt"))
    if not annotation_files:
        raise ValueError(f"No annotation files found in {annotations_dir}")

    # Create splits
    import random
    random.shuffle(annotation_files)

    n_total = len(annotation_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = annotation_files[:n_train]
    val_files = annotation_files[n_train:n_train + n_val]
    test_files = annotation_files[n_train + n_val:]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    logger.info(f"Dataset splits:")
    logger.info(f"Total: {n_total}")
    logger.info(f"Train: {len(train_files)}")
    logger.info(f"Val: {len(val_files)}")
    logger.info(f"Test: {len(test_files)}")

    # Create directory structure and copy files
    for split_name, files in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create images and labels subdirectories
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "labels").mkdir(exist_ok=True)

        for annotation_file in files:
            # Copy annotation file
            shutil.copy2(annotation_file, split_dir / "labels" / annotation_file.name)

            # Find and copy corresponding image
            image_extensions = ['.png', '.jpg', '.jpeg']
            for ext in image_extensions:
                image_file = annotations_path.parent / "approved" / f"{annotation_file.stem}{ext}"
                if image_file.exists():
                    shutil.copy2(image_file, split_dir / "images" / image_file.name)
                    break
            else:
                logger.warning(f"No image found for annotation: {annotation_file.name}")

    return {k: [f.stem for f in v] for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO model for comic panel detection")
    parser.add_argument("--annotations", required=True,
                       help="Directory with approved annotations")
    parser.add_argument("--dataset", required=True,
                       help="Output directory for prepared dataset")
    parser.add_argument("--output", required=True,
                       help="Output path for fine-tuned model")
    parser.add_argument("--base-model", default=str(YOLO_PANEL_MODEL_PATH),
                       help="Path to base YOLO model")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation data ratio")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation after training")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.annotations).exists():
        print(f"Error: Annotations directory {args.annotations} does not exist")
        return 1

    if not Path(args.base_model).exists():
        print(f"Error: Base model {args.base_model} does not exist")
        print("Please run: python scripts/models_downloader.py")
        return 1

    print("🎯 Starting YOLO Model Fine-tuning")
    print(f"📁 Annotations: {args.annotations}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"💾 Output: {args.output}")
    print(f"🤖 Base Model: {args.base_model}")

    try:
        # Prepare dataset
        print("\n📋 Preparing dataset splits...")
        splits = prepare_dataset_splits(
            args.annotations,
            args.dataset,
            args.train_ratio,
            args.val_ratio
        )

        # Initialize fine-tuner
        fine_tuner = YOLOFineTuner(args.base_model, args.dataset, args.output)

        # Train model
        print(f"\n🚀 Starting training for {args.epochs} epochs...")
        results = fine_tuner.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            patience=args.patience
        )

        # Validate if requested
        if args.validate:
            print("\n🔍 Running validation...")
            val_results = fine_tuner.validate()

        print(f"\n✅ Fine-tuning completed successfully!")
        print(f"📄 Fine-tuned model saved to: {args.output}")
        print(f"📊 Dataset prepared at: {args.dataset}")

        return 0

    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        logger.error(f"Fine-tuning failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
