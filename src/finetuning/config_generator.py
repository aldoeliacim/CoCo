#!/usr/bin/env python3
"""
YOLO Dataset Configuration Generator

This script generates YAML configuration files for YOLO training.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional


def create_dataset_config(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    class_names: List[str] = None,
    output_path: str = None
) -> str:
    """
    Create YOLO dataset configuration file.

    Args:
        train_dir: Path to training images directory
        val_dir: Path to validation images directory
        test_dir: Optional path to test images directory
        class_names: List of class names (default: ['panel'])
        output_path: Optional output path for YAML file

    Returns:
        Path to created YAML file
    """
    if class_names is None:
        class_names = ['panel']

    # Convert to absolute paths
    train_path = Path(train_dir).resolve()
    val_path = Path(val_dir).resolve()

    config = {
        'path': str(train_path.parent),  # Dataset root directory
        'train': str(train_path.relative_to(train_path.parent)),
        'val': str(val_path.relative_to(val_path.parent)),
        'nc': len(class_names),  # Number of classes
        'names': class_names
    }

    # Add test set if provided
    if test_dir:
        test_path = Path(test_dir).resolve()
        config['test'] = str(test_path.relative_to(train_path.parent))

    # Determine output path
    if output_path is None:
        output_path = train_path.parent / "dataset.yaml"
    else:
        output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Dataset configuration saved to: {output_path}")

    # Print configuration summary
    print("\n📋 Dataset Configuration:")
    print(f"Root: {config['path']}")
    print(f"Train: {config['train']}")
    print(f"Val: {config['val']}")
    if 'test' in config:
        print(f"Test: {config['test']}")
    print(f"Classes: {config['nc']} ({', '.join(config['names'])})")

    return str(output_path)


def create_training_config(
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0005,
    momentum: float = 0.937,
    augmentation: bool = True,
    output_path: str = None
) -> str:
    """
    Create YOLO training configuration file.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
        learning_rate: Initial learning rate
        weight_decay: Weight decay coefficient
        momentum: SGD momentum
        augmentation: Whether to use data augmentation
        output_path: Optional output path for YAML file

    Returns:
        Path to created YAML file
    """
    config = {
        # Training parameters
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'lr0': learning_rate,
        'weight_decay': weight_decay,
        'momentum': momentum,

        # Optimization
        'optimizer': 'SGD',
        'cos_lr': True,  # Cosine learning rate schedule
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Augmentation
        'hsv_h': 0.015 if augmentation else 0,
        'hsv_s': 0.7 if augmentation else 0,
        'hsv_v': 0.4 if augmentation else 0,
        'degrees': 15.0 if augmentation else 0,
        'translate': 0.1 if augmentation else 0,
        'scale': 0.5 if augmentation else 0,
        'shear': 2.0 if augmentation else 0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5 if augmentation else 0,
        'mosaic': 1.0 if augmentation else 0,
        'mixup': 0.1 if augmentation else 0,
        'copy_paste': 0.1 if augmentation else 0,

        # Loss function
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,

        # Validation
        'val': True,
        'save_period': 10,  # Save model every N epochs
        'patience': 50,  # Early stopping patience

        # Other
        'workers': 8,
        'device': '',  # Auto-detect
        'single_cls': True,  # Single class detection
        'rect': False,  # Rectangular training
        'resume': False,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Dataset fraction to use
        'profile': False,
        'freeze': None,  # Layers to freeze
    }

    # Determine output path
    if output_path is None:
        output_path = Path.cwd() / "training_config.yaml"
    else:
        output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Training configuration saved to: {output_path}")

    # Print configuration summary
    print("\n🏋️ Training Configuration:")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch']}")
    print(f"Image size: {config['imgsz']}")
    print(f"Learning rate: {config['lr0']}")
    print(f"Augmentation: {'enabled' if augmentation else 'disabled'}")
    print(f"AMP: {'enabled' if config['amp'] else 'disabled'}")

    return str(output_path)


def main():
    """Example usage of configuration generators."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate YOLO configuration files")
    parser.add_argument("--type", choices=['dataset', 'training', 'both'],
                       default='both', help="Configuration type to generate")
    parser.add_argument("--train-dir", help="Training images directory")
    parser.add_argument("--val-dir", help="Validation images directory")
    parser.add_argument("--test-dir", help="Test images directory (optional)")
    parser.add_argument("--classes", nargs='+', default=['panel'],
                       help="Class names")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for config files")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--no-augmentation", action='store_true',
                       help="Disable data augmentation")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔧 YOLO Configuration Generator")

    if args.type in ['dataset', 'both']:
        if not args.train_dir or not args.val_dir:
            print("❌ Error: --train-dir and --val-dir required for dataset config")
            return 1

        dataset_config_path = create_dataset_config(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            test_dir=args.test_dir,
            class_names=args.classes,
            output_path=output_dir / "dataset.yaml"
        )

    if args.type in ['training', 'both']:
        training_config_path = create_training_config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            learning_rate=args.lr,
            augmentation=not args.no_augmentation,
            output_path=output_dir / "training.yaml"
        )

    print("\n✅ Configuration generation completed!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
