# Model Documentation

This directory contains the deep learning models utilized by the CoCo comic analysis research framework. All models are automatically managed through the centralized model acquisition system.

**Academic Context**: Models utilized for EGC thesis research on computational comic character analysis
**Principal Investigator**: Aldo Eliacim Alvarez Lemus
**Advisor**: Dr. Gerardo Eugenio Sierra Martínez
**Last Updated**: August 2025

## Model Architecture Overview

### VisionThink-General (Qwen2.5-VL)

**Location**: `VisionThink-General/`
- **Source**: [HuggingFace - Senqiao/VisionThink-General](https://huggingface.co/Senqiao/VisionThink-General)
- **Research Foundation**: [VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning](https://arxiv.org/abs/2507.13348)
- **Architecture**: Qwen2.5-VL base model enhanced with reinforcement learning optimization
- **Parameters**: ~7B parameters
- **Research Application**: Multi-modal content understanding, character analysis, narrative comprehension

**Technical Characteristics**:
- **Enhanced Character Analysis**: Structured prompting for detailed character detection and name extraction
- **Contextual Panel Understanding**: Cross-panel character tracking with narrative context
- **Advanced Natural Language Processing**: Character name extraction with fallback pattern matching
- **Multi-modal Integration**: Unified visual and textual information processing
- **Panel Validation**: Quality assessment and reading order determination

**Research Configuration**:
```python
VISIONTHINK_CONFIG = {
    "max_new_tokens": 800,           # Enhanced for detailed character analysis
    "temperature": 0.3,              # Focused analysis parameter
    "do_sample": True,
    "torch_dtype": "float16",
    "character_analysis_enabled": True,
    "adaptive_resolution": True,
    "multi_turn_analysis": True,
    "character_verification": True
}
```

### YOLO Panel Detection Models

#### Base Model: best.pt
**Location**: `best.pt`
- **Source**: [HuggingFace - mosesb/best-comic-panel-detection](https://huggingface.co/mosesb/best-comic-panel-detection)
- **Architecture**: YOLOv12 optimized for general comic panel detection
- **Training Data**: Multi-source comic panel dataset
- **Research Application**: Baseline panel segmentation for comparative studies

#### Domain-Adapted Model: best_fantomas.pt
**Location**: `best_fantomas.pt` (generated through fine-tuning pipeline)
- **Source**: Custom training via human-in-the-loop methodology (see `src/finetuning/README.md`)
- **Architecture**: YOLOv12 adapted for specific comic domain (Fantomas corpus)
- **Training Data**: Human-curated panel annotations with quality validation
- **Research Application**: Domain-specific panel detection with enhanced accuracy

**Technical Configuration**:
```python
YOLO_CONFIG = {
    "confidence_threshold": 0.3,        # Detection confidence threshold
    "iou_threshold": 0.5,              # Non-maximum suppression threshold
}
```

## Model Selection Strategy

The CoCo framework implements an automatic model selection hierarchy:

1. **Primary**: `best_fantomas.pt` (domain-adapted model, if available)
2. **Fallback**: `best.pt` (general comic panel model)
3. **Acquisition**: Automatic download from HuggingFace if models unavailable

```python
# Model loading logic (from main.py)
FINE_TUNED_MODEL = "data/models/best_fantomas.pt"
BASE_MODEL = "data/models/best.pt"

if os.path.exists(FINE_TUNED_MODEL):
    yolo_model = YOLO(FINE_TUNED_MODEL)  # Domain-adapted model
else:
    yolo_model = YOLO(BASE_MODEL)        # General baseline model
```
```

## Model Management System

### Automated Acquisition

```bash
# Download all required models for research
python scripts/models_downloader.py

# Verify model functionality and integration
python scripts/models_downloader.py --test-only

# Selective model acquisition
python scripts/models_downloader.py --visionthink-only
python scripts/models_downloader.py --yolo-only
```

### Manual Model Installation

For environments with restricted internet access:

```bash
# VisionThink-General manual installation
git lfs install
git clone https://huggingface.co/Senqiao/VisionThink-General data/models/VisionThink-General

# YOLO model manual download
wget https://huggingface.co/mosesb/best-comic-panel-detection/resolve/main/best.pt \
     -O data/models/best.pt
```

## Empirical Performance Analysis

### VisionThink-General Validation Results

**Large-Scale Execution (August 9-10, 2025):**
- **Total Processing**: 141 comic pages over 12.5 hours
- **Model Stability**: Zero crashes or memory errors during extended execution
- **Initialization Time**: 9 seconds average model loading
- **Processing Variance**: 23.4-692.5 seconds per page (content complexity dependent)
- **Success Rate**: 97.9% successful analysis completion

**Content-Specific Performance:**
- **Cover Pages**: 52.0s average (single-pass analysis)
- **Advertisement Pages**: 61.7s average (text-heavy processing)
- **Comic Pages**: 351.4s average (multi-panel processing)

### YOLO Model Performance

**Fine-tuned Model Validation:**
- **Model Used**: `best_fantomas.pt` (domain-adapted for Fantomas corpus)
- **Loading Time**: <1 second initialization
- **Panel Detection**: Successfully applied to 125 comic pages
- **Integration**: Seamless operation with VisionThink pipeline
- **Memory Efficiency**: Minimal VRAM usage throughout 12.5-hour execution


## Research Validation and Testing

### Model Verification

```bash
# Comprehensive model testing
python scripts/models_downloader.py --test-only

# Individual model validation
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; \
           model = Qwen2_5_VLForConditionalGeneration.from_pretrained('data/models/VisionThink-General'); \
           print('VisionThink loaded successfully')"

python -c "from ultralytics import YOLO; \
           model = YOLO('data/models/best.pt'); \
           print('YOLO loaded successfully')"
```

### Memory Requirements

**VisionThink-General:**
- **VRAM**: 4-6GB (float16 precision)
- **System RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~14GB for complete model

**YOLO Models:**
- **VRAM**: 2-4GB during inference
- **System RAM**: 4GB sufficient
- **Storage**: ~50MB per model file

## Integration with CoCo Pipeline

### Centralized Configuration

All model parameters are managed through `config/settings.py`:

```python
# Model paths
VISIONTHINK_MODEL_PATH = MODELS_DIR / "VisionThink-General"
YOLO_PANEL_MODEL_PATH = MODELS_DIR / "best.pt"

# Processing configuration
PROCESSING_CONFIG = {
    "device": "auto",                   # Automatic GPU/CPU selection
    "max_panels_per_page": 50,          # Panel processing limit
    "panel_min_area": 1000,             # Minimum panel area (pixels²)
}
```

### Device Management

The framework implements automatic device selection:

```python
# Automatic device assignment based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Memory-aware model loading
if device == "cuda":
    model = model.to(device, dtype=torch.float16)  # GPU optimization
else:
    model = model.to(device, dtype=torch.float32)  # CPU compatibility
```

## Research Applications and Validation

### Model Testing Framework

The model management system includes comprehensive validation:

```python
# VisionThink validation protocol
✓ Model architecture verification and loading
✓ Tokenizer compatibility and functionality
✓ Image processor integration testing
✓ GPU/CPU device allocation optimization
✓ Sample inference validation

# YOLO validation protocol
✓ Model architecture integrity verification
✓ Input/output tensor shape validation
✓ Inference speed benchmarking
✓ Detection confidence scoring accuracy
✓ Multi-device compatibility testing
```

## Computational Performance

### System Requirements Validation

**Empirically Validated Configuration:**
- **GPU Memory**: Successfully operated on available hardware for 12.5 hours
- **System Memory**: Stable operation during large-scale processing
- **Processing Throughput**: 141 pages processed with 97.9% success rate
- **Storage Impact**: Generated 138 XML files with complete metadata

### VisionThink-General
- **Initialization**: 9 seconds (validated)
- **Memory Footprint**: Stable throughout extended execution
- **Processing Speed**: 23.4-692.5 seconds per page (complexity-dependent)
- **Device Compatibility**: Successfully executed with automatic GPU management

### YOLO Panel Detection
- **Initialization**: <1 second (validated)
- **Memory Footprint**: Minimal impact during concurrent VisionThink operation
- **Processing Speed**: Integrated seamlessly within overall pipeline timing
- **Model Selection**: Successfully prioritized fine-tuned `best_fantomas.pt` model

## Model Selection Hierarchy

The framework implements intelligent model prioritization:

```python
# Automatic model selection priority (highest to lowest)
1. best_fantomas.pt              # Domain-adapted (highest accuracy)
2. best.pt                       # General baseline (reliable performance)
3. HuggingFace fallback          # Remote acquisition (last resort)
```

This strategy ensures optimal performance while maintaining system robustness across deployment environments.

## Research Licensing and Attribution

### VisionThink-General
- **Licensing**: Refer to [original repository](https://huggingface.co/Senqiao/VisionThink-General) for current terms
- **Academic Use**: Generally permitted with appropriate citation
- **Commercial Applications**: Requires careful license review

### YOLO Models
- **Base Model**: Refer to [original repository](https://huggingface.co/mosesb/best-comic-panel-detection) for licensing
- **Ultralytics Framework**: AGPL-3.0 license (commercial licensing available)

## Development and Extension

### Custom Model Integration

For incorporating additional models:

1. Place model artifacts in `data/models/` directory
2. Update configuration parameters in `config/settings.py`
3. Modify model loading procedures in relevant source modules
4. Update automated management scripts for deployment

### Domain Adaptation Pipeline

For creating custom domain-adapted models, utilize the complete training framework:

```bash
# Access comprehensive fine-tuning methodology
cd finetuning/
# Reference finetuning/README.md for detailed research protocols
```

## Technical Implementation Notes

The model management system provides:
- **Automated Acquisition**: Seamless model downloading and validation
- **Graceful Fallbacks**: Robust operation across different model availability scenarios
- **Performance Optimization**: Device-aware memory and processing management
- **Research Reproducibility**: Consistent model versioning and configuration management

---

**CoCo Model Management**: Centralized, automated model lifecycle management supporting reproducible comic analysis research.
- **Use Case**: Perfect for comic analysis where some panels need detailed OCR while others don't

## Model Naming Convention

We use the original model names from their source repositories to maintain consistency and traceability:
- `best.pt` - As provided by mosesb/best-comic-panel-detection
- `VisionThink-General` - As provided by Senqiao
- Standard naming for other models

## VisionThink Two-Round Approach

VisionThink implements a revolutionary two-round analysis paradigm that's perfect for comic analysis:

1. **Round 1 (Low-Resolution)**: Process downscaled image (1/4 visual tokens)
2. **Round 2 (High-Resolution)**: Only if model requests `<UPSCALING_TOKEN>` for fine details

**Benefits for Comic Analysis**:
- Most comic panels don't need high-resolution analysis
- OCR-heavy panels (text, charts) automatically get high-resolution processing
- 50% average token reduction while maintaining quality
- Intelligent decision making per panel

**Example Usage**:
```python
# VisionThink automatically decides resolution needs
analysis = visionthink_analyzer.analyze_panel(panel_image, prompt)
if analysis.get('analysis_rounds') == 2:
    print("High resolution was needed for this panel")
```

## Fine-tuning

The `best.pt` model can be fine-tuned for specific comic styles using our comprehensive pipeline:

### 🤖 **Human-Curated Training (Recommended)**
```bash
# Interactive annotation review and model training
python finetuning/data_curation.py --images-dir data/fantomas/processed/grayscale
python finetuning/train_model.py --base-model data/models/best.pt
```

### � **Model Evaluation**
```bash
# Compare base vs fine-tuned models
python finetuning/evaluation/evaluate_model.py --models "base:data/models/best.pt"
```

### 🚀 **Complete Pipeline**
```bash
# See comprehensive documentation
cat finetuning/README.md
```
