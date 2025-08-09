#!.venv/bin/python
import sys
import os
import argparse
from pathlib import Path

# Standardized project imports
import config
from config.logger import setup_logger
from config.settings import set_log_level

logger = setup_logger("main")

# Global model state to avoid reloading models
_model_state = {
    'loaded': False,
    'visionthink_model': None,
    'tokenizer': None,
    'processor': None,
    'yolo_model': None
}


def xml_exists_for_image(image_file: Path, output_dir: str) -> bool:
    """
    Check if XML output already exists for the given image file.

    Args:
        image_file: Path to the image file
        output_dir: Output directory path

    Returns:
        bool: True if XML file exists, False otherwise
    """
    from config.settings import PROCESSING_CONFIG
    xml_subdir = PROCESSING_CONFIG["output_subdirs"]["xml"]
    xml_filename = f"{image_file.stem}_analysis.xml"
    xml_path = Path(output_dir) / xml_subdir / xml_filename
    return xml_path.exists()


def clean_output_directory(output_dir: str) -> None:
    """
    Clean the output directory by removing all files but preserving directory structure.

    Args:
        output_dir: Path to the output directory to clean
    """
    import shutil

    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"🧹 Output directory does not exist, creating: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        return

    logger.info(f"🧹 Cleaning output directory: {output_path}")

    # Count files and directories before cleaning
    total_files = 0
    total_dirs = 0
    for root, dirs, files in os.walk(output_path):
        total_files += len(files)
        total_dirs += len(dirs)

    logger.debug(f"   Found {total_files} files and {total_dirs} directories")

    try:
        # Remove all contents but preserve the root directory
        for item in output_path.iterdir():
            if item.is_file():
                item.unlink()
                logger.debug(f"   Removed file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                logger.debug(f"   Removed directory: {item.name}")

        logger.info(f"✓ Output directory cleaned successfully")
        logger.debug(f"   Removed {total_files} files and {total_dirs} directories")

    except Exception as e:
        logger.error(f"❌ Failed to clean output directory: {e}")
        raise RuntimeError(f"Failed to clean output directory: {e}")


def load_models():
    """
    Load all required models for the CoCo pipeline.
    Uses global state to avoid reloading models if already loaded.

    Returns:
        dict: Dictionary containing all initialized models with proper labels
    """
    global _model_state

    # Return cached models if already loaded
    if _model_state['loaded']:
        logger.info("📋 Using cached models (already loaded)")
        return {
            'visionthink_model': _model_state['visionthink_model'],
            'tokenizer': _model_state['tokenizer'],
            'processor': _model_state['processor'],
            'yolo_model': _model_state['yolo_model']
        }

    # Import heavy dependencies only when needed
    import torch
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoTokenizer,
        AutoProcessor
    )
    from ultralytics import YOLO
    from config.settings import (
        VISIONTHINK_MODEL_PATH,
        VISIONTHINK_CONFIG,
        YOLO_PANEL_MODEL_PATH
    )

    logger.info("=" * 50)
    logger.info("INITIALIZING MODELS")
    logger.info("=" * 50)

    # 1. VisionThink Model (Qwen2.5-VL) - Main Vision-Language Model
    logger.info("📋 Loading VisionThink-General (Qwen2.5-VL) model...")
    logger.debug(f"Model path: {VISIONTHINK_MODEL_PATH}")

    visionthink_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(VISIONTHINK_MODEL_PATH),
        torch_dtype=getattr(torch, VISIONTHINK_CONFIG["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(VISIONTHINK_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True
    )

    processor = AutoProcessor.from_pretrained(
        str(VISIONTHINK_MODEL_PATH),
        trust_remote_code=True,
        use_fast=True
    )
    logger.info("✓ VisionThink model loaded successfully")

    # 2. YOLO Panel Detection Model - Comic Panel Segmentation
    logger.info("🎯 Loading YOLO panel detection model...")

    # Check for fine-tuned model first
    fine_tuned_model_path = Path("data/models/best_fantomas.pt")
    base_model_path = YOLO_PANEL_MODEL_PATH

    yolo_model = None
    if fine_tuned_model_path.exists():
        logger.debug(f"🚀 Using fine-tuned model: {fine_tuned_model_path}")
        yolo_model = YOLO(str(fine_tuned_model_path))
        logger.info("✓ Fine-tuned YOLO panel model loaded successfully")
    elif base_model_path.exists():
        logger.debug(f"📦 Using base model: {base_model_path}")
        yolo_model = YOLO(str(base_model_path))
        logger.info("✓ Base YOLO panel model loaded successfully")
    else:
        logger.error(f"❌ No YOLO models found locally!")
        logger.error(f"📍 Expected locations:")
        logger.error(f"Fine-tuned: {fine_tuned_model_path}")
        logger.error(f"Base model: {base_model_path}")
        logger.error(f"💡 Please run: python scripts/models_downloader.py")
        raise FileNotFoundError("No YOLO models available. Please download models first.")

    logger.info("=" * 50)
    logger.info("✅ ALL MODELS LOADED SUCCESSFULLY")
    logger.info("=" * 50)

    # Cache models in global state
    _model_state['visionthink_model'] = visionthink_model
    _model_state['tokenizer'] = tokenizer
    _model_state['processor'] = processor
    _model_state['yolo_model'] = yolo_model
    _model_state['loaded'] = True

    return {
        'visionthink_model': visionthink_model,      # Main vision-language model for content analysis
        'tokenizer': tokenizer,                      # Tokenizer for VisionThink model
        'processor': processor,                      # Processor for VisionThink model
        'yolo_model': yolo_model,                   # Panel detection model
    }


def main():
    """Main function to run the Comic Analysis Pipeline."""
    parser = argparse.ArgumentParser(
        description="Comic Analysis Pipeline - Process comic pages for character analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # Process default input directory
  python main.py /path/to/directory                       # Process all pages in directory
  python main.py page1.png                                # Process single page
  python main.py page1.png page2.png page3.png           # Process multiple specific pages
  python main.py -v page1.png page2.png                  # Verbose logging for specific pages
  python main.py -c /comics/fantomas                      # Clean output directory then process comic directory
  python main.py --clean --verbose page1.png page2.png   # Clean output, verbose logging for specific pages
        """
    )

    parser.add_argument(
        "inputs",
        nargs="*",  # Accept 0 or more positional arguments
        help="Input files or directories to process. Can be: single directory, single page, or multiple pages"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean output directory before processing (wipes all previous results)"
    )

    args = parser.parse_args()

    # Import config only after argparse (so --help is fast)
    from config.settings import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, initialize_directories

    # Initialize required directories
    initialize_directories()

    # Use standardized output directory
    output_dir = str(DEFAULT_OUTPUT_DIR)

    # Parse input sources - flexible handling
    input_files = []

    if args.inputs:
        # Process positional arguments
        logger.debug(f"Processing positional arguments: {args.inputs}")
        for input_arg in args.inputs:
            input_path = Path(input_arg)

            if input_path.is_dir():
                # Directory - find all image files
                logger.debug(f"Processing directory: {input_path}")
                dir_files = [f for f in input_path.iterdir()
                           if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
                input_files.extend(sorted(dir_files))
                logger.debug(f"Found {len(dir_files)} image files in {input_path}")

            elif input_path.is_file():
                # Single file - check if it's an image
                if input_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    input_files.append(input_path)
                    logger.debug(f"Added single file: {input_path}")
                else:
                    logger.warning(f"Skipping non-image file: {input_path}")

            else:
                # Path doesn't exist
                logger.error(f"Input path does not exist: {input_path}")
                sys.exit(1)

    else:
        # No input specified - use default directory
        default_dir = Path(DEFAULT_INPUT_DIR)
        logger.debug(f"No input specified, using default directory: {default_dir}")

        if not default_dir.exists():
            logger.error(f"Default input directory does not exist: {default_dir}")
            logger.info("Please specify input files or create the default input directory")
            sys.exit(1)

        dir_files = [f for f in default_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        input_files.extend(sorted(dir_files))
        logger.debug(f"Found {len(dir_files)} image files in default directory")    # Validate we have files to process
    if not input_files:
        logger.error("No image files found to process")
        logger.info("Supported formats: PNG, JPG, JPEG")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in input_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    input_files = unique_files

    # Setup global logging level based on verbose flag
    log_level = "DEBUG" if args.verbose else "INFO"
    set_log_level(log_level)

    logger.info("=" * 60)
    logger.info("Comic Analysis Pipeline Starting")
    logger.info("=" * 60)
    logger.debug(f"Logging level: {log_level}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Input files to process: {len(input_files)}")

    # Show input summary
    if len(input_files) == 1:
        logger.info(f"Processing single file: {input_files[0].name}")
    elif len(input_files) <= 5:
        logger.info(f"Processing files: {', '.join([f.name for f in input_files])}")
    else:
        logger.info(f"Processing {len(input_files)} files:")
        for i, f in enumerate(input_files[:3]):
            logger.info(f"  {i+1}. {f.name}")
        logger.info(f"  ... and {len(input_files) - 3} more files")

    try:
        # Import pipeline functions only when we actually need them
        from analysis import initialize_pipeline, process_page

        # Load models
        models = load_models()

        # Clean output directory if requested (after models loaded to avoid interrupting expensive loading)
        if args.clean:
            logger.info("=" * 40)
            logger.info("CLEANING OUTPUT DIRECTORY")
            logger.info("=" * 40)
            clean_output_directory(output_dir)

        # Initialize pipeline
        initialize_pipeline(
            visionthink_model=models['visionthink_model'],
            tokenizer=models['tokenizer'],
            processor=models['processor'],
            yolo_model=models['yolo_model'],
            output_dir=output_dir
        )

        # Process all input files
        total_pages = len(input_files)
        processed_pages = 0
        failed_pages = 0
        skipped_pages = 0

        logger.info("=" * 40)
        logger.info("STARTING PAGE PROCESSING")
        logger.info("=" * 40)

        for i, image_file in enumerate(input_files, 1):
            try:
                logger.info(f"[{i}/{total_pages}] Processing: {image_file.name}")

                # Ensure the file still exists (in case of race conditions)
                if not image_file.exists():
                    logger.error(f"File no longer exists: {image_file}")
                    failed_pages += 1
                    continue

                # Check if XML output already exists (skip processing for resume functionality)
                if xml_exists_for_image(image_file, output_dir):
                    skipped_pages += 1
                    logger.info(f"⏭️  [{i}/{total_pages}] {image_file.name} skipped (XML already exists)")
                    logger.debug(f"   Existing XML found, skipping to resume processing")
                    continue

                result = process_page(str(image_file))

                if "error" not in result:
                    processed_pages += 1
                    processing_time = result.get('processing_time', 0)
                    page_type = result.get('page_type', 'unknown')
                    panels_detected = result.get('panels_detected', 0)

                    logger.info(f"✓ [{i}/{total_pages}] {image_file.name} completed successfully")
                    logger.debug(f"   Type: {page_type}, Panels: {panels_detected}, Time: {processing_time:.1f}s")
                else:
                    failed_pages += 1
                    logger.error(f"✗ [{i}/{total_pages}] {image_file.name} failed: {result['error']}")

            except KeyboardInterrupt:
                logger.warning(f"Processing interrupted by user at file {i}/{total_pages}")
                break
            except Exception as e:
                failed_pages += 1
                logger.error(f"✗ [{i}/{total_pages}] {image_file.name} failed with exception: {e}")
                logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")

        # Summary results
        results = {
            'total_pages': total_pages,
            'processed_pages': processed_pages,
            'failed_pages': failed_pages,
            'skipped_pages': skipped_pages,
            'success_rate': (processed_pages / total_pages * 100) if total_pages > 0 else 0
        }

        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total pages: {results['total_pages']}")
        logger.info(f"Successfully processed: {results['processed_pages']}")
        logger.info(f"Skipped (already exists): {results['skipped_pages']}")
        logger.info(f"Failed: {results['failed_pages']}")
        logger.info(f"Success rate: {results['success_rate']:.1f}%")

        if results['skipped_pages'] > 0:
            logger.info(f"💡 {results['skipped_pages']} pages were skipped because XML output already exists")
            logger.info("   Use --clean/-c to force reprocessing of all pages")

        if results['failed_pages'] > 0:
            logger.warning("Some pages failed to process. Check the logs for details.")
        elif results['processed_pages'] > 0:
            logger.info("🎉 All new pages processed successfully!")
        elif results['skipped_pages'] == results['total_pages']:
            logger.info("ℹ️  All pages already processed (all XML files exist)")
        else:
            logger.info("✓ Processing completed")

        # Always run parse results (includes moving to timestamped directories)
        logger.info("=" * 40)
        logger.info("ORGANIZING RESULTS")
        logger.info("=" * 40)
        try:
            import subprocess

            # Parse results and move to timestamped directories in one step
            logger.info("📊 Parsing results and organizing into timestamped directories...")
            parse_script = Path(__file__).parent / "scripts" / "parse_results.py"
            subprocess.run([sys.executable, str(parse_script), "--latest"], check=True)
            logger.info("✓ Results parsing and organization completed successfully")

        except Exception as e:
            logger.warning(f"Results parsing or organization failed: {e}")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
