# YOLO Fine-tuning for Comic Panel Detection

A research framework for domain adaptation of YOLO object detection models to comic panel segmentation. This pipeline implements human-in-the-loop methodology for training data curation and provides comprehensive evaluation tools for comic-specific object detection tasks.

**Academic Context**: EGC thesis research component for comic panel detection optimization
**Principal Investigator**: Aldo Eliacim Alvarez Lemus
**Advisor**: Dr. Gerardo Eugenio Sierra Martínez
**Research Focus**: Domain adaptation of object detection models for comic analysis
**Last Updated**: August 2025

## Research Motivation

Comic panel detection presents unique challenges compared to standard object detection:
- **Irregular Geometries**: Non-rectangular panel shapes and unconventional layouts
- **Artistic Variability**: Diverse art styles, panel borders, and layout conventions
- **Reading Order Dependency**: Sequential panel arrangement affects narrative comprehension
- **Domain Specificity**: General object detection models underperform on comic-specific layouts
- **Fantomas-Specific Challenges**: Unique artistic style and panel layout conventions in the Fantomas corpus

This framework addresses these challenges through domain-adapted training with human-curated annotations specifically optimized for the Fantomas comic collection.

## Enhanced Methodology

### Six-Stage Training Pipeline Architecture

The fine-tuning process implements an enhanced research methodology:

1. **Data Preprocessing**: Image standardization and grayscale conversion for optimal detection
2. **Human-in-the-Loop Curation**: Interactive annotation quality assessment with domain expertise
3. **Dataset Preparation**: YOLO format conversion and train/validation/test splitting
4. **Hyperparameter Optimization**: Domain-specific parameter tuning for comic panel characteristics
5. **Model Training**: Fine-tuned YOLO training with comic-specific adaptations
6. **Comprehensive Evaluation**: Performance assessment with baseline comparison and error analysis

### Advanced Training Strategy

- **Base Model**: Pre-trained YOLO from [mosesb/best-comic-panel-detection](https://huggingface.co/mosesb/best-comic-panel-detection)
- **Domain Adaptation**: Human-curated annotations ensure high-quality Fantomas-specific training data
- **Image Processing**: Optimized grayscale preprocessing pipeline for enhanced panel boundary detection
- **Quality Assurance**: Rigorous validation methodology with confidence scoring
- **Performance Optimization**: Integration with VisionThink for panel validation and reading order determination

## Implementation

### Prerequisites

```bash
# Ensure CoCo research environment is activated
cd /path/to/CoCo
source .venv/bin/activate

# Verify model availability
python scripts/models_downloader.py --test-only
```

### Complete Training Workflow

#### Stage 1: Data Preprocessing

For raw comic image optimization:

```bash
# Image preprocessing for YOLO training
python scripts/dataset_preprocessing.py \
    --input data/fantomas/raw \
    --output data/fantomas/processed/grayscale \
    --preserve-originals \
    --create-manifest \
    --grayscale \
    --normalize
```

Results: Standardized images optimized for object detection training.

#### Stage 2: Human-in-the-Loop Data Curation

Interactive annotation quality assessment:

```bash
# Launch annotation curation interface
python src/finetuning/data_curation.py \
    --input data/fantomas/processed/grayscale \
    --output out/finetuning/curated_data \
    --model data/models/best.pt
```

**Curation Interface Controls:**
- **Approve (✓)**: High-quality panel detections (included in training set)
- **Reject (✗)**: Poor-quality detections (excluded from training)
- **Skip (→)**: Ambiguous cases (no annotation decision)

**Quality Assessment Criteria:**
- **Accept**: Precise panel boundaries, minimal false positives, correct reading order
- **Reject**: Incomplete panels, boundary inaccuracies, excessive noise
- **Methodology**: Prioritize annotation quality over dataset size

#### Stage 3: Dataset Preparation

Convert curated annotations to YOLO training format:

```bash
# Generate YOLO-compatible dataset
python src/finetuning/preprocessing/dataset_builder.py \
    --annotations out/finetuning/curated_data \
    --output out/finetuning/yolo_dataset \
    --train-split 0.8 \
    --val-split 0.15 \
    --test-split 0.05 \
    --class-names panel
```

Output: Properly formatted YOLO dataset with train/validation/test splits.

# Validate dataset quality and annotation consistency
python src/finetuning/preprocessing/data_validation.py
    --dataset out/finetuning/yolo_dataset
    --output out/finetuning/validation_report.json
```

#### Stage 4: Model Training

Execute domain-adapted YOLO fine-tuning:

```bash
# Train fine-tuned model with optimal hyperparameters
python src/finetuning/train_model.py \
    --base-model data/models/best.pt \
    --dataset out/finetuning/yolo_dataset/dataset.yaml \
    --output out/finetuning/trained_models \
    --epochs 100 \
    --batch-size 16 \
    --image-size 640

# Monitor training progress via TensorBoard
tensorboard --logdir out/finetuning/trained_models/runs
```

#### Stage 5: Model Evaluation

Comprehensive performance assessment:

```bash
# Evaluate against baseline models
python src/finetuning/evaluation/evaluate_model.py \
    --dataset out/finetuning/yolo_dataset \
    --models out/finetuning/trained_models/best.pt data/models/best.pt \
    --output out/finetuning/evaluation_results \
    --generate-visualizations
```

Results: Quantitative metrics, performance visualizations, and comparative analysis.

## Performance Analysis

### Baseline Performance Metrics

**Pre-trained Model (General Comics):**
- **mAP@0.5**: 0.75-0.85 (domain-dependent)
- **Precision**: 0.80-0.90
- **Recall**: 0.70-0.85
- **Inference Speed**: 40-100ms per image

**Expected Fine-tuning Improvements:**
- **mAP@0.5**: +5-15% improvement on domain-specific content
- **Precision**: +3-10% enhancement in boundary accuracy
- **Recall**: +5-20% improvement in panel detection completeness
- **False Positive Reduction**: Significant decrease in erroneous detections

### Quality Enhancement Areas

- **Boundary Precision**: More accurate panel edge detection
- **False Positive Reduction**: Fewer background region misclassifications
- **Art Style Adaptation**: Improved performance on specific visual styles
- **Layout Recognition**: Enhanced handling of irregular panel geometries

## Research Configuration

### Training Hyperparameters

Empirically optimized parameters for comic panel detection:

```python
# Optimal configuration for domain adaptation
TRAINING_CONFIG = {
    "epochs": 100,                  # Maximum training iterations
    "batch_size": 16,              # GPU memory dependent
    "image_size": 640,             # Input resolution
    "learning_rate": 0.001,        # Initial learning rate
    "weight_decay": 0.0005,        # L2 regularization strength
    "momentum": 0.937,             # SGD momentum parameter
    "warmup_epochs": 3,            # Learning rate warmup period
    "patience": 50,                # Early stopping threshold
}
```

### Domain-Specific Augmentation

```python
# Comic-optimized augmentation strategy
AUGMENTATION_CONFIG = {
    "hsv_h": 0.015,               # Minimal hue variation (preserve art style)
    "hsv_s": 0.7,                 # Saturation adjustment
    "hsv_v": 0.4,                 # Brightness variation
    "degrees": 0.0,               # Rotation disabled (preserve reading order)
    "translate": 0.1,             # Translation augmentation
    "scale": 0.5,                 # Scale variation
    "shear": 0.0,                 # Shear disabled (maintain panel integrity)
    "perspective": 0.0,           # Perspective disabled
    "flipud": 0.0,                # Vertical flip disabled (reading order)
    "fliplr": 0.5,                # Horizontal flip (conditional)
    "mosaic": 1.0,                # Mosaic augmentation enabled
    "mixup": 0.0,                 # Mixup disabled (detection task)
}
```

## Quality Assurance Methodology

### Dataset Validation Framework

```bash
# Comprehensive dataset integrity verification
python src/finetuning/preprocessing/data_validation.py \
    --dataset out/finetuning/yolo_dataset \
    --check-annotations \
    --check-images \
    --check-distribution \
    --generate-report
```

**Validation Components:**
- Annotation format compliance verification
- Image integrity and quality assessment
- Class distribution statistical analysis
- Bounding box geometric validation
- Dataset split integrity confirmation

### Human Annotation Standards

**Acceptance Criteria:**
- Panel boundaries precisely align with visual comic panel edges
- Minimal overlapping detections (IoU < 0.3 between panels)
- Reading order follows conventional comic narrative flow
- Detection confidence exceeds 0.3 threshold
- Complete panel coverage (no partial detections)

**Rejection Criteria:**
- False positive detections (speech bubbles, characters misclassified as panels)
- Incomplete or inaccurate panel boundary definitions
- Background elements erroneously classified as panels
- Excessive overlapping or duplicate detection instances

## Implementation Structure

```
src/finetuning/
├── README.md                    # Research methodology documentation
├── data_curation.py            # Human-in-the-loop annotation interface
├── train_model.py              # YOLO domain adaptation pipeline
├── config_generator.py         # Training configuration management
├── evaluation/
│   └── evaluate_model.py       # Comprehensive performance assessment
└── preprocessing/
    ├── dataset_builder.py      # YOLO format conversion utilities
    └── data_validation.py      # Dataset integrity validation

out/finetuning/                 # Training outputs (generated)
├── curated_data/               # Human-validated annotations
├── yolo_dataset/               # YOLO training format
├── trained_models/             # Domain-adapted models
└── evaluation_results/         # Performance analysis
```

## Technical Issues and Solutions

### Computational Limitations

**GPU Memory Constraints:**
```bash
# Reduce batch size for memory-limited environments
python src/finetuning/train_model.py --batch-size 8

# Enable automatic mixed precision training
python src/finetuning/train_model.py --amp
```

**Display Interface Issues:**
```bash
# For remote server environments, ensure X11 forwarding
ssh -X username@server

# Verify Tkinter GUI framework availability
python -c "import tkinter; print('Tkinter available')"
```

### Training Optimization

**Convergence Challenges:**
- Ensure minimum dataset size (100+ high-quality annotations)
- Verify balanced class distribution across training examples
- Optimize learning rate and batch size parameters
- Implement sufficient training epoch duration

**Data Limitations:**
- **Limited Annotations**: Prioritize quality over quantity in curation process
- **Augmentation**: Implement robust data augmentation strategies
- **Transfer Learning**: Leverage multiple pre-trained base models
- **Synthetic Data**: Consider procedural data generation techniques

**Large-Scale Training:**
- **Distributed Computing**: Multi-GPU training implementation
- **Efficient Loading**: Optimized data pipeline with parallel workers
- **Progressive Training**: Multi-stage resolution increase training strategy

## Advanced Research Features

### Ensemble Learning Methodology

```bash
# Multi-model training with configuration variations
python src/finetuning/train_model.py --config ensemble_config_1.yaml
python src/finetuning/train_model.py --config ensemble_config_2.yaml

# Ensemble performance evaluation
python src/finetuning/evaluation/evaluate_model.py --ensemble-models model1.pt model2.pt
```

### Custom Loss Function Implementation

The framework supports domain-specific loss functions for comic panel detection:

- **Focal Loss**: Enhanced handling of class imbalance in panel/background distribution
- **IoU-aware Loss**: Improved bounding box regression for irregular panel shapes
- **Aspect Ratio Loss**: Panel geometry-aware training optimization

### Model Export and Deployment

```bash
# Export trained models to various deployment formats
python src/finetuning/train_model.py --export-onnx
python src/finetuning/train_model.py --export-tensorrt
python src/finetuning/train_model.py --export-coreml
```

## Research Applications

This fine-tuning framework enables:

- **Domain-Specific Adaptation**: Custom model training for specific comic publishers or art styles
- **Comparative Studies**: Performance analysis across different comic genres
- **Ablation Studies**: Systematic evaluation of training methodology components
- **Benchmark Development**: Standardized evaluation protocols for comic analysis research

---

**YOLO Fine-tuning Framework**: A systematic approach to domain adaptation for comic panel detection through human-guided machine learning.

The fine-tuned model integrates seamlessly with the main CoCo pipeline:

1. **Model Location**: `data/models/best_fantomas.pt`
2. **Configuration**: Update `config/settings.py` to use fine-tuned model
3. **Evaluation**: Use with existing evaluation scripts
4. **Production**: Deploy through main `main.py` workflow

This finetuning pipeline ensures high-quality, domain-specific comic panel detection through human expertise and systematic validation.
