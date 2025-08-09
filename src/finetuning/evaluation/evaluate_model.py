"""
Model Evaluation and Comparison

This script evaluates and compares YOLO models for comic panel detection.
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from config.settings import get_base_settings, MODELS_DIR
from config.settings import OUT_DIR, DATA_DIR
from src.logger_setup import setup_logger

logger = setup_logger("evaluate_model")


class ModelEvaluator:
    """Evaluate and compare YOLO models."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized model evaluator, output: {self.output_dir}")

    def load_model(self, model_path: str) -> YOLO:
        """Load YOLO model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = YOLO(model_path)
        logger.info(f"Loaded model: {model_path}")
        return model

    def evaluate_on_dataset(self, model: YOLO, dataset_path: str) -> Dict:
        """
        Evaluate model on dataset using YOLO's built-in validation.

        Args:
            model: YOLO model instance
            dataset_path: Path to dataset YAML file

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating model on dataset: {dataset_path}")

        try:
            results = model.val(data=dataset_path, save_json=True, save_hybrid=True)

            # Extract key metrics
            metrics = {
                'map50': float(results.box.map50),
                'map50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1': float(results.box.f1),
                'fitness': float(results.fitness)
            }

            logger.info("Evaluation results:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def compare_models(self, models: Dict[str, str], dataset_path: str) -> Dict:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of {model_name: model_path}
            dataset_path: Path to dataset YAML file

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")

        comparison_results = {}

        for model_name, model_path in models.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                model = self.load_model(model_path)
                results = self.evaluate_on_dataset(model, dataset_path)
                comparison_results[model_name] = {
                    'model_path': model_path,
                    'metrics': results,
                    'status': 'success'
                }

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                comparison_results[model_name] = {
                    'model_path': model_path,
                    'metrics': {},
                    'status': 'failed',
                    'error': str(e)
                }

        return comparison_results

    def visualize_comparison(self, comparison_results: Dict, save_path: str = None) -> str:
        """
        Create visualization of model comparison.

        Args:
            comparison_results: Results from compare_models()
            save_path: Optional path to save plot

        Returns:
            Path to saved plot
        """
        # Extract successful results
        successful_results = {
            name: data for name, data in comparison_results.items()
            if data['status'] == 'success'
        }

        if len(successful_results) < 2:
            logger.warning("Need at least 2 successful model evaluations for comparison")
            return None

        # Prepare data for plotting
        model_names = list(successful_results.keys())
        metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1']

        # Create subplot for each metric
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]

            values = [successful_results[name]['metrics'].get(metric, 0) for name in model_names]

            bars = ax.bar(model_names, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 10:
                ax.tick_params(axis='x', rotation=45)

        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "visualizations" / "model_comparison.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison plot saved to: {save_path}")
        return str(save_path)

    def test_on_images(self, model: YOLO, image_dir: str, output_dir: str = None,
                      conf_threshold: float = 0.25) -> Dict:
        """
        Test model on individual images and save visualizations.

        Args:
            model: YOLO model instance
            image_dir: Directory with test images
            output_dir: Directory to save annotated images
            conf_threshold: Confidence threshold for detections

        Returns:
            Test results summary
        """
        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if output_dir is None:
            output_dir = self.output_dir / "test_images"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(image_path.glob(ext))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Testing on {len(image_files)} images")

        results_summary = {
            'total_images': len(image_files),
            'total_detections': 0,
            'avg_detections_per_image': 0,
            'confidence_stats': {
                'mean': 0,
                'std': 0,
                'min': 1,
                'max': 0
            },
            'image_results': {}
        }

        all_confidences = []

        for img_file in image_files:
            logger.info(f"Processing: {img_file.name}")

            # Run inference
            results = model(str(img_file), conf=conf_threshold)
            detections = results[0].boxes

            # Load image for visualization
            img = cv2.imread(str(img_file))
            if img is None:
                logger.warning(f"Failed to load image: {img_file}")
                continue

            img_results = {
                'detections': len(detections) if detections is not None else 0,
                'confidences': []
            }

            # Draw detections
            if detections is not None and len(detections) > 0:
                for i, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = detections.conf[i].item()

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw confidence score
                    label = f"Panel {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(img, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    img_results['confidences'].append(conf)
                    all_confidences.append(conf)

            # Save annotated image
            output_path = output_dir / f"annotated_{img_file.name}"
            cv2.imwrite(str(output_path), img)

            results_summary['image_results'][img_file.name] = img_results
            results_summary['total_detections'] += img_results['detections']

        # Calculate summary statistics
        if results_summary['total_images'] > 0:
            results_summary['avg_detections_per_image'] = (
                results_summary['total_detections'] / results_summary['total_images']
            )

        if all_confidences:
            results_summary['confidence_stats'] = {
                'mean': float(np.mean(all_confidences)),
                'std': float(np.std(all_confidences)),
                'min': float(np.min(all_confidences)),
                'max': float(np.max(all_confidences))
            }

        logger.info("Test results summary:")
        logger.info(f"Total detections: {results_summary['total_detections']}")
        logger.info(f"Avg per image: {results_summary['avg_detections_per_image']:.1f}")
        logger.info(f"Confidence mean: {results_summary['confidence_stats']['mean']:.3f}")

        return results_summary

    def generate_report(self, comparison_results: Dict, test_results: Dict = None) -> str:
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / "reports" / "evaluation_report.json"

        report = {
            'timestamp': str(pd.Timestamp.now()),
            'model_comparison': comparison_results,
            'test_results': test_results or {},
            'summary': {}
        }

        # Generate summary
        successful_models = {
            name: data for name, data in comparison_results.items()
            if data['status'] == 'success'
        }

        if successful_models:
            # Find best model for each metric
            metrics = ['map50', 'map50_95', 'precision', 'recall', 'f1']
            best_models = {}

            for metric in metrics:
                best_score = -1
                best_model = None

                for model_name, data in successful_models.items():
                    score = data['metrics'].get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name

                best_models[metric] = {
                    'model': best_model,
                    'score': best_score
                }

            report['summary']['best_models_by_metric'] = best_models

            # Overall best model (by F1 score)
            if 'f1' in best_models:
                report['summary']['overall_best'] = best_models['f1']['model']

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to: {report_path}")
        return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO models for panel detection")
    parser.add_argument("--models", nargs='+', required=True,
                       help="Model paths to evaluate (format: name:path)")
    parser.add_argument("--dataset", required=True,
                       help="Dataset YAML file for evaluation")
    parser.add_argument("--test-images", help="Directory with test images for visualization")
    parser.add_argument("--output", default=str(OUT_DIR / "finetuning" / "evaluation_results"),
                       help="Output directory for results")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for test images")

    args = parser.parse_args()

    # Parse model specifications
    models = {}
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':', 1)
        else:
            name = Path(model_spec).stem
            path = model_spec
        models[name] = path

    print("🎯 Model Evaluation")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🤖 Models: {list(models.keys())}")
    print(f"💾 Output: {args.output}")

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.output)

        # Compare models
        print("\n📈 Comparing models...")
        comparison_results = evaluator.compare_models(models, args.dataset)

        # Create comparison visualization
        plot_path = evaluator.visualize_comparison(comparison_results)
        if plot_path:
            print(f"📊 Comparison plot saved to: {plot_path}")

        # Test on images if provided
        test_results = None
        if args.test_images and Path(args.test_images).exists():
            print(f"\n🖼️  Testing on images from: {args.test_images}")

            # Use best model or first successful model
            best_model_name = None
            for name, data in comparison_results.items():
                if data['status'] == 'success':
                    best_model_name = name
                    break

            if best_model_name:
                model = evaluator.load_model(models[best_model_name])
                test_results = evaluator.test_on_images(
                    model, args.test_images,
                    conf_threshold=args.conf_threshold
                )
                print(f"🎯 Test results: {test_results['total_detections']} total detections")

        # Generate comprehensive report
        report_path = evaluator.generate_report(comparison_results, test_results)
        print(f"📄 Full report saved to: {report_path}")

        # Print summary
        successful_count = sum(1 for data in comparison_results.values() if data['status'] == 'success')
        print(f"\n✅ Evaluation completed!")
        print(f"Successfully evaluated: {successful_count}/{len(models)} models")

        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    # Import pandas here to avoid dependency issues if not needed
    try:
        import pandas as pd
    except ImportError:
        import datetime as pd
        pd.Timestamp = pd.datetime

    sys.exit(main())
