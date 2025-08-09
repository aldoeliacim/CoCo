import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import sys
import cv2
import numpy as np

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

# Standardized project imports
import config
from config import setup_logger

# Add src to path for src-internal imports
sys.path.insert(0, str(Path(__file__).parent))

from xml_validator import XMLValidator
from visionthink import (
    initialize_visionthink,
    identify_page_type,
    validate_panel_detection,
    analyze_panel_with_context,
    create_annotated_image,
    crop_panels
)

logger = setup_logger("analysis")

def identify_page_type(image_path: str) -> str:
    """Identify comic page type using VisionThink model."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized")
        return "comic"

    logger.info(f"🔍 Starting page type identification for: {Path(image_path).name}")

    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        return "comic"

    # Clear GPU memory before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        logger.debug(f"   Image size: {image.size}")

        # Create the prompt for page type identification
        prompt = """Analyze this comic page image and determine its type. Respond with exactly one of these categories:
- cover: Cover page or title page
- comic: Regular comic page with panels and story content
- advertisement: Advertisement or promotional content
- text: Text-heavy page (credits, introduction, etc.)
- illustration: Full-page illustration without panels

Category:"""

        # Generate response
        response = _generate_visionthink_response(image, prompt)
        logger.debug(f"   Raw response: '{response}'")

        # Extract and validate page type
        page_type = _extract_page_type(response)
        logger.debug(f"   Extracted page type: '{page_type}'")

        if page_type in ["cover", "comic", "advertisement", "text", "illustration"]:
            logger.info(f"✅ Page type identified: {page_type}")
            return page_type
        else:
            logger.warning(f"⚠️ Invalid page type '{page_type}', defaulting to 'comic'")
            return "comic"

    except Exception as e:
        logger.error(f"❌ Error in page type identification: {e}")
        return "comic"
    finally:
        # Clear GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _generate_visionthink_response(image: Image.Image, prompt: str) -> str:
    """Generate response from VisionThink model for given image and prompt."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized")
        return "Error: VisionThink not initialized"

    try:
        logger.debug("🔄 Starting VisionThink response generation...")

        # Format as messages for VisionThink model
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process messages using VisionThink's expected format
        from qwen_vl_utils import process_vision_info

        text = _visionthink_state['processor'].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = _visionthink_state['processor'](
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(_visionthink_state['device'])

        # Generate response
        with torch.no_grad():
            generated_ids = _visionthink_state['model'].generate(
                **inputs,
                max_new_tokens=min(config.VISIONTHINK_CONFIG["max_new_tokens"], 500),
                temperature=config.VISIONTHINK_CONFIG["temperature"],
                do_sample=config.VISIONTHINK_CONFIG["do_sample"],
                pad_token_id=_visionthink_state['tokenizer'].eos_token_id,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = _visionthink_state['processor'].batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    except Exception as e:
        logger.error(f"Error generating VisionThink response: {e}")
        return f"Error: {str(e)}"

def _extract_page_type(response: str) -> str:
    """Extract page type from VisionThink response."""
    try:
        response = response.strip().lower()

        # Look for the page type after "category:"
        if "category:" in response:
            category_part = response.split("category:")[-1].strip()
            page_type = category_part.split()[0] if category_part.split() else ""
            page_type = page_type.strip('.,!?;:').lower()

            # Map variations to standard types
            type_mapping = {
                'cover': 'cover',
                'comic': 'comic',
                'advertisement': 'advertisement',
                'ad': 'advertisement',
                'text': 'text',
                'illustration': 'illustration',
                'credits': 'text'
            }
            return type_mapping.get(page_type, 'comic')
        else:
            # Fallback: look for key words anywhere in response
            response_lower = response.lower()
            if any(word in response_lower for word in ['cover', 'title']):
                return 'cover'
            elif any(word in response_lower for word in ['ad', 'advertisement']):
                return 'advertisement'
            elif any(word in response_lower for word in ['text', 'credits']):
                return 'text'
            elif any(word in response_lower for word in ['illustration', 'full-page']):
                return 'illustration'
            else:
                return 'comic'

    except Exception as e:
        logger.debug(f"Error parsing page type response: {e}")
        return 'comic'


# Module-level pipeline state
_pipeline_state = None


def initialize_pipeline(
    visionthink_model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    yolo_model: YOLO,
    output_dir: str = str(config.DEFAULT_OUTPUT_DIR)
) -> None:
    """Initialize the enhanced comic analysis pipeline."""
    global _pipeline_state

    # Device setup first
    device = config.PROCESSING_CONFIG.get("device", "auto")
    if device == "auto":
        device = visionthink_model.device if visionthink_model else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize pipeline state
    _pipeline_state = {
        'models': {
            'visionthink_model': visionthink_model,
            'tokenizer': tokenizer,
            'processor': processor,
            'yolo_model': yolo_model
        },
        'output_dir': Path(output_dir),
        'device': device,
        'xml_validator': XMLValidator(),
        'initialized': True
    }

    # Initialize VisionThink module
    try:
        initialize_visionthink(
            model=visionthink_model,
            tokenizer=tokenizer,
            processor=processor,
            device=device
        )
        logger.info("✅ VisionThink module initialized in pipeline")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VisionThink module: {e}")
        raise RuntimeError(f"VisionThink initialization failed: {e}")

    # Create output directory structure
    base_output_path = Path(output_dir)
    xml_subdir = config.PROCESSING_CONFIG["output_subdirs"]["xml"]
    annotated_subdir = config.PROCESSING_CONFIG["output_subdirs"]["annotated_panels"]

    (_pipeline_state['output_dir'] / xml_subdir).mkdir(parents=True, exist_ok=True)
    (_pipeline_state['output_dir'] / annotated_subdir).mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Comic Pipeline initialized")
    logger.info(f"   🎯 VisionThink analysis enabled")


def detect_panels_with_yolo(image_path: str) -> List[Dict[str, Any]]:
    """Detect panels using fine-tuned YOLO model on processed/grayscale image."""
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("Detecting panels with fine-tuned YOLO...")

    try:
        # Get corresponding grayscale image from processed directory
        raw_image_path = Path(image_path)
        processed_image_path = Path("data/fantomas/processed/grayscale") / raw_image_path.name

        if not processed_image_path.exists():
            logger.warning(f"⚠️ Processed image not found: {processed_image_path}")
            logger.info("   Using original image for panel detection")
            processed_image_path = raw_image_path

        # Load processed (grayscale) image for panel detection
        image = Image.open(processed_image_path).convert('L')
        logger.debug(f"   Using processed image: {processed_image_path}")

        # Run YOLO detection
        yolo_model = _pipeline_state['models']['yolo_model']
        results = yolo_model(
            image,
            conf=config.YOLO_CONFIG["confidence_threshold"],
            iou=config.YOLO_CONFIG["iou_threshold"]
        )

        panels = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Filter out very small panels
                    panel_area = (x2 - x1) * (y2 - y1)
                    if panel_area > config.PROCESSING_CONFIG["panel_min_area"]:
                        panels.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'area': panel_area
                        })

        # Sort panels by position (top to bottom, left to right)
        panels = _sort_panels_reading_order(panels)

        # Limit number of panels per page
        max_panels = config.PROCESSING_CONFIG["max_panels_per_page"]
        if len(panels) > max_panels:
            logger.warning(f"⚠️ Found {len(panels)} panels, limiting to {max_panels}")
            panels = panels[:max_panels]

        logger.debug(f"✓ Detected {len(panels)} panels")
        return panels

    except Exception as e:
        logger.error(f"Error in panel detection: {e}")
        return []


def _sort_panels_reading_order(panels: List[Dict]) -> List[Dict]:
    """Sort panels in reading order (top to bottom, left to right)."""
    return sorted(panels, key=lambda p: (p['bbox'][1], p['bbox'][0]))


def generate_annotated_image(image_path: str, panels: List[Dict], output_dir: str) -> str:
    """Generate annotated image with panel bounding boxes."""
    try:
        # Load original raw image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Draw panel bounding boxes
        for i, panel in enumerate(panels):
            x1, y1, x2, y2 = panel['bbox']
            confidence = panel['confidence']

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add panel number and confidence
            label = f"Panel {i+1} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            # Background rectangle for text
            cv2.rectangle(image, (x1, y1-30), (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Create output path
        image_name = Path(image_path).stem
        output_path = Path(output_dir) / f"{image_name}" / "annotated_panels.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save annotated image
        cv2.imwrite(str(output_path), image)
        logger.info(f"📸 Saved annotated image: {output_path}")

        return str(output_path)

    except Exception as e:
        logger.error(f"Error generating annotated image: {e}")
        return ""


def crop_individual_panels(image_path: str, panels: List[Dict], output_dir: str) -> List[str]:
    """Crop individual panels from raw image and save them."""
    try:
        # Load original raw image
        image = Image.open(image_path).convert('RGB')

        # Create output directory
        image_name = Path(image_path).stem
        panels_dir = Path(output_dir) / f"{image_name}" / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)

        panel_paths = []

        for i, panel in enumerate(panels):
            x1, y1, x2, y2 = panel['bbox']

            # Crop panel
            panel_image = image.crop((x1, y1, x2, y2))

            # Save panel
            panel_path = panels_dir / f"panel_{i+1:02d}.png"
            panel_image.save(panel_path)
            panel_paths.append(str(panel_path))

            logger.debug(f"   Cropped panel {i+1}: {panel_path}")

        logger.info(f"✂️ Cropped {len(panels)} panels to: {panels_dir}")
        return panel_paths

    except Exception as e:
        logger.error(f"Error cropping panels: {e}")
        return []


def assess_panel_confidence(annotated_image_path: str) -> Dict[str, Any]:
    """Use VisionThink to assess panel detection quality and determine panel order."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized")
        return {'confidence_score': 0.5, 'panel_order': [], 'assessment': 'Error: VisionThink not initialized'}

    try:
        logger.info("🔍 Assessing panel detection quality with VisionThink...")

        # Load annotated image
        image = Image.open(annotated_image_path).convert('RGB')

        prompt = """Analyze this comic page with panel detection bounding boxes (green rectangles with panel numbers).

Evaluate the panel detection quality and provide:

1. **PANEL_CONFIDENCE_SCORE**: Rate from 0.0-1.0 how accurately the panels are detected
   - 1.0: Perfect detection, all panels correctly identified
   - 0.8-0.9: Very good, minor issues
   - 0.6-0.7: Good, some panels missed or incorrectly detected
   - 0.4-0.5: Fair, significant detection issues
   - 0.0-0.3: Poor, major problems with panel detection

2. **PANEL_ORDER**: List the correct reading order of panels (e.g., "1,2,3,4" or "1,3,2,4")

3. **ASSESSMENT**: Brief explanation of detection quality and any issues

Format your response as:
PANEL_CONFIDENCE_SCORE: [score]
PANEL_ORDER: [order]
ASSESSMENT: [explanation]"""

        response = _generate_visionthink_response(image, prompt)

        # Parse response
        confidence_score = 0.7  # default
        panel_order = []
        assessment = response

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('PANEL_CONFIDENCE_SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    confidence_score = float(score_str)
                    confidence_score = max(0.0, min(1.0, confidence_score))  # clamp to [0,1]
                except:
                    pass
            elif line.startswith('PANEL_ORDER:'):
                try:
                    order_str = line.split(':', 1)[1].strip()
                    panel_order = [int(x.strip()) for x in order_str.split(',') if x.strip().isdigit()]
                except:
                    pass
            elif line.startswith('ASSESSMENT:'):
                assessment = line.split(':', 1)[1].strip()

        result = {
            'confidence_score': confidence_score,
            'panel_order': panel_order,
            'assessment': assessment,
            'raw_response': response
        }

        logger.info(f"✅ Panel confidence: {confidence_score:.2f}")
        logger.debug(f"   Panel order: {panel_order}")

        return result

    except Exception as e:
        logger.error(f"Error assessing panel confidence: {e}")
        return {
            'confidence_score': 0.5,
            'panel_order': [],
            'assessment': f'Error: {str(e)}',
            'raw_response': ''
        }


def analyze_panel_with_visionthink(panel_image_path: str, panel_index: int, total_panels: int,
                                  page_character_context: List[Dict] = None) -> Dict[str, Any]:
    """Analyze individual panel with VisionThink for detailed character detection."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized")
        return {'error': 'VisionThink not initialized'}

    logger.debug(f"👁️ Analyzing panel {panel_index + 1}/{total_panels}: {Path(panel_image_path).name}")

    try:
        # Load panel image
        panel_image = Image.open(panel_image_path).convert('RGB')

        # Build context from previous panels
        character_context = ""
        if page_character_context:
            recent_chars = page_character_context[-3:]  # Last 3 characters for context
            if recent_chars:
                char_names = [char.get('name', 'Unknown') for char in recent_chars]
                character_context = f"Characters seen in previous panels: {', '.join(char_names)}"

        # Enhanced detailed panel analysis prompt
        prompt = f"""Analyze this comic panel in detail. This is panel {panel_index + 1} of {total_panels}.

{character_context}

Provide a comprehensive analysis with the following structure:

**CHARACTERS:**
For each character visible, provide:
- Name (or descriptive identifier like "bald man", "woman in red", etc.)
- Detailed physical description (clothing, appearance, pose, facial features)
- Current action/activity
- Any dialogue spoken (exact text if visible in speech bubbles)
- Role assessment (primary/secondary/extra based on prominence and panel size)

**SETTING:**
- Location description (indoor/outdoor, specific environment)
- Background details and atmosphere
- Objects and props visible

**MOOD:**
- Emotional tone of the scene
- Tension level or dramatic atmosphere
- Visual mood indicators

**STORY_ELEMENTS:**
- Key plot developments happening in this panel
- Character interactions and relationships
- Narrative significance and context

**DIALOGUE:**
- Extract any visible text/speech bubbles with exact quotes
- Indicate which character is speaking each line
- Note any sound effects or onomatopoeia

Be specific and detailed in your descriptions. Focus on character identification and dialogue extraction."""

        response = _generate_visionthink_response(panel_image, prompt)

        # Enhanced parsing to extract structured data
        parsed_data = _parse_detailed_panel_analysis(response, panel_index)
        parsed_data['image_path'] = panel_image_path

        return parsed_data

    except Exception as e:
        logger.error(f"Error analyzing panel {panel_index}: {e}")
        return {'error': str(e), 'panel_index': panel_index}


def _parse_detailed_panel_analysis(response: str, panel_index: int) -> Dict[str, Any]:
    """Parse VisionThink response into structured panel analysis data."""
    try:
        # Initialize result structure
        result = {
            'panel_index': panel_index,
            'raw_analysis': response,
            'characters': [],
            'setting': '',
            'mood': '',
            'story_elements': '',
            'dialogue': []
        }

        # Split response into sections
        sections = {}
        current_section = None
        lines = response.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith(':**'):
                section_name = line.replace('**', '').replace(':', '').strip().lower()
                current_section = section_name
                sections[current_section] = []
            elif current_section and line:
                sections[current_section].append(line)

        # Extract characters
        if 'characters' in sections:
            characters = _extract_characters_from_section(sections['characters'])
            result['characters'] = characters

        # Extract other sections
        result['setting'] = '\n'.join(sections.get('setting', []))
        result['mood'] = '\n'.join(sections.get('mood', []))
        result['story_elements'] = '\n'.join(sections.get('story_elements', []))

        # Extract dialogue
        if 'dialogue' in sections:
            dialogue_lines = []
            for line in sections['dialogue']:
                if ':' in line:
                    parts = line.split(':', 1)
                    speaker = parts[0].strip()
                    text = parts[1].strip().strip('"\'')
                    dialogue_lines.append({'speaker': speaker, 'text': text})
                elif line.strip():
                    dialogue_lines.append({'speaker': 'Unknown', 'text': line.strip()})
            result['dialogue'] = dialogue_lines

        # If no structured characters found, try to extract from raw response
        if not result['characters']:
            result['characters'] = _extract_characters_fallback(response)

        return result

    except Exception as e:
        logger.debug(f"Error parsing panel analysis: {e}")
        return {
            'panel_index': panel_index,
            'raw_analysis': response,
            'characters': [{'name': 'Character', 'description': 'Analysis parse error', 'role': 'unknown'}],
            'setting': '',
            'mood': '',
            'story_elements': '',
            'dialogue': []
        }


def _extract_characters_from_section(character_lines: List[str]) -> List[Dict[str, str]]:
    """Extract character information from the characters section."""
    characters = []
    current_char = None

    for line in character_lines:
        line = line.strip()
        if line.startswith('-') and 'Name:' in line:
            # Start of new character
            if current_char:
                characters.append(current_char)
            name = line.split('Name:', 1)[1].strip() if 'Name:' in line else 'Unknown'
            current_char = {
                'name': name,
                'description': '',
                'action': '',
                'dialogue': '',
                'role': 'unknown'
            }
        elif current_char and line.startswith('-'):
            # Character attribute
            if 'description:' in line.lower():
                current_char['description'] = line.split(':', 1)[1].strip()
            elif 'action:' in line.lower():
                current_char['action'] = line.split(':', 1)[1].strip()
            elif 'dialogue:' in line.lower():
                current_char['dialogue'] = line.split(':', 1)[1].strip()
            elif 'role:' in line.lower():
                role = line.split(':', 1)[1].strip().lower()
                current_char['role'] = role if role in ['primary', 'secondary', 'extra'] else 'unknown'

    # Add the last character
    if current_char:
        characters.append(current_char)

    return characters


def _extract_characters_fallback(response: str) -> List[Dict[str, str]]:
    """Fallback character extraction from raw response."""
    try:
        characters = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ['character', 'person', 'figure', 'man', 'woman', 'shows']):
                # Extract basic character info
                characters.append({
                    'name': 'Character',
                    'description': line,
                    'action': '',
                    'dialogue': '',
                    'role': 'unknown'
                })

        # Ensure at least one character
        if not characters:
            characters.append({
                'name': 'Unknown',
                'description': 'No clear characters identified in this panel',
                'action': '',
                'dialogue': '',
                'role': 'unknown'
            })

        return characters[:5]  # Limit to 5 characters max

    except Exception as e:
        logger.debug(f"Error in fallback character extraction: {e}")
        return [{'name': 'Unknown', 'description': 'Parse error', 'action': '', 'dialogue': '', 'role': 'unknown'}]


def generate_validated_xml(image_path: str, page_type: str, panel_data: List[Dict],
                          panel_confidence: Dict, character_summary: Dict) -> str:
    """Generate validated XML output for the page."""
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("📄 Generating comprehensive validated XML output...")

    try:
        # Create XML structure with enhanced metadata
        root = ET.Element("comic_page_analysis")
        root.set("version", "2.0")
        root.set("image_file", Path(image_path).name)
        root.set("image_path", str(Path(image_path).relative_to(Path.cwd())))
        root.set("page_type", page_type)
        root.set("analysis_date", datetime.now().isoformat())
        root.set("total_panels", str(len(panel_data)))
        root.set("processing_pipeline", "VisionThink+YOLO")

        # Image metadata
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode
                file_size = Path(image_path).stat().st_size

            img_meta = ET.SubElement(root, "image_metadata")
            img_meta.set("width", str(width))
            img_meta.set("height", str(height))
            img_meta.set("format", format_name or "Unknown")
            img_meta.set("mode", mode or "Unknown")
            img_meta.set("file_size", str(file_size))
        except Exception as e:
            logger.debug(f"Could not extract image metadata: {e}")

        # Panel confidence assessment
        confidence_elem = ET.SubElement(root, "panel_confidence")
        confidence_elem.set("score", str(panel_confidence.get('confidence_score', 0.7)))
        confidence_elem.set("assessment", panel_confidence.get('assessment', 'No assessment available'))
        if panel_confidence.get('panel_order'):
            confidence_elem.set("suggested_order", ','.join(map(str, panel_confidence['panel_order'])))

        # Character summary
        char_summary = ET.SubElement(root, "character_summary")
        total_chars = len(character_summary.get('all_characters', []))
        primary_chars = [c for c in character_summary.get('all_characters', []) if c.get('role') == 'primary']
        secondary_chars = [c for c in character_summary.get('all_characters', []) if c.get('role') == 'secondary']
        extra_chars = [c for c in character_summary.get('all_characters', []) if c.get('role') == 'extra']

        char_summary.set("total_unique", str(total_chars))
        char_summary.set("primary_count", str(len(primary_chars)))
        char_summary.set("secondary_count", str(len(secondary_chars)))
        char_summary.set("extra_count", str(len(extra_chars)))

        # Character details
        if primary_chars:
            primary_elem = ET.SubElement(char_summary, "primary_chars")
            primary_elem.set("count", str(len(primary_chars)))
            for char in primary_chars:
                char_elem = ET.SubElement(primary_elem, "character")
                char_elem.set("name", char.get('name', 'Unknown'))
                char_elem.set("appearances", str(char.get('appearances', 1)))
                char_elem.set("dialogue_lines", str(len(char.get('dialogue', []))))

        if secondary_chars:
            secondary_elem = ET.SubElement(char_summary, "secondary_chars")
            secondary_elem.set("count", str(len(secondary_chars)))
            for char in secondary_chars:
                char_elem = ET.SubElement(secondary_elem, "character")
                char_elem.set("name", char.get('name', 'Unknown'))
                char_elem.set("appearances", str(char.get('appearances', 1)))
                char_elem.set("dialogue_lines", str(len(char.get('dialogue', []))))

        if extra_chars:
            extra_elem = ET.SubElement(char_summary, "extra_chars")
            extra_elem.set("count", str(len(extra_chars)))
            for char in extra_chars:
                char_elem = ET.SubElement(extra_elem, "character")
                char_elem.set("name", char.get('name', 'Unknown'))
                char_elem.set("appearances", str(char.get('appearances', 1)))
                char_elem.set("dialogue_lines", str(len(char.get('dialogue', []))))

        # Panels section
        panels_elem = ET.SubElement(root, "panels")
        panels_elem.set("detection_method", "fine_tuned_yolo")
        panels_elem.set("analysis_method", "visionthink_contextual")

        for i, panel in enumerate(panel_data):
            panel_elem = ET.SubElement(panels_elem, "panel")
            panel_elem.set("number", str(i + 1))
            panel_elem.set("sequence_order", str(i + 1))

            # Characters in this panel
            panel_characters = panel.get('characters', [])
            characters_elem = ET.SubElement(panel_elem, "characters")
            characters_elem.set("count", str(len(panel_characters)))

            for char in panel_characters:
                char_elem = ET.SubElement(characters_elem, "character")
                char_elem.set("name", char.get('name', 'Unknown'))
                char_elem.set("description", char.get('description', ''))
                char_elem.set("role", char.get('role', 'unknown'))

                if char.get('action'):
                    char_elem.set("action", char.get('action'))
                if char.get('dialogue'):
                    char_elem.set("dialogue", char.get('dialogue'))

            # Content analysis
            content_elem = ET.SubElement(panel_elem, "content_analysis")
            if panel.get('setting'):
                setting_elem = ET.SubElement(content_elem, "setting")
                setting_elem.text = panel.get('setting')
            if panel.get('mood'):
                mood_elem = ET.SubElement(content_elem, "mood")
                mood_elem.text = panel.get('mood')
            if panel.get('story_elements'):
                story_elem = ET.SubElement(content_elem, "story_elements")
                story_elem.text = panel.get('story_elements')

            # Dialogue section
            if panel.get('dialogue'):
                dialogue_elem = ET.SubElement(panel_elem, "dialogue")
                for line in panel.get('dialogue', []):
                    line_elem = ET.SubElement(dialogue_elem, "line")
                    line_elem.set("speaker", line.get('speaker', 'Unknown'))
                    line_elem.text = line.get('text', '')

        # Processing metadata
        meta_elem = ET.SubElement(root, "processing_metadata")
        meta_elem.set("analysis_type", "full")

        # Generate XML string
        xml_str = ET.tostring(root, encoding='unicode', method='xml')

        # Format XML with proper indentation
        try:
            import xml.dom.minidom
            dom = xml.dom.minidom.parseString(xml_str)
            formatted_xml = dom.toprettyxml(indent="  ")
            # Remove empty lines
            lines = [line for line in formatted_xml.split('\n') if line.strip()]
            formatted_xml = '\n'.join(lines)
        except:
            formatted_xml = xml_str

        # Save XML to file
        xml_subdir = config.PROCESSING_CONFIG["output_subdirs"]["xml"]
        xml_filename = f"{Path(image_path).stem}_analysis.xml"
        xml_path = _pipeline_state['output_dir'] / xml_subdir / xml_filename

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)

        # Validate XML
        try:
            xml_validator = _pipeline_state['xml_validator']
            is_valid, validation_errors = xml_validator.validate_xml_string(formatted_xml)
            if not is_valid:
                logger.warning(f"⚠️ XML validation warnings: {validation_errors}")
            else:
                logger.debug("✓ XML validation passed")
        except Exception as e:
            logger.debug(f"XML validation error: {e}")

        logger.info(f"📄 Generated XML: {xml_path}")
        return str(xml_path)

    except Exception as e:
        logger.error(f"Error generating XML: {e}")
        raise


def _build_character_summary(panel_data: List[Dict]) -> Dict[str, Any]:
    """Build character summary from all panel data."""
    character_registry = {}

    for panel in panel_data:
        for char in panel.get('characters', []):
            char_name = char.get('name', 'Unknown')

            if char_name not in character_registry:
                character_registry[char_name] = {
                    'name': char_name,
                    'appearances': 0,
                    'dialogue': [],
                    'role': char.get('role', 'unknown'),
                    'descriptions': []
                }

            # Update character data
            character_registry[char_name]['appearances'] += 1

            if char.get('dialogue'):
                character_registry[char_name]['dialogue'].append(char.get('dialogue'))

            if char.get('description'):
                character_registry[char_name]['descriptions'].append(char.get('description'))

            # Update role (prioritize primary > secondary > extra)
            current_role = character_registry[char_name]['role']
            new_role = char.get('role', 'unknown')
            if new_role == 'primary' or (new_role == 'secondary' and current_role != 'primary'):
                character_registry[char_name]['role'] = new_role

    return {
        'all_characters': list(character_registry.values()),
        'character_registry': character_registry
    }


def process_page(image_path: str) -> Dict[str, Any]:
    """
    Enhanced comic page processing workflow with VisionThink analysis.

    Workflow:
    1. Identify page type (VisionThink)
    2. Panel detection (YOLO on processed/grayscale)
    3. Generate annotated image with panel boundaries
    4. Validate panel detection and determine reading order (VisionThink)
    5. Crop individual panels from raw image
    6. Analyze each panel with VisionThink for character detection
    7. Generate comprehensive XML output
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    start_time = time.time()
    image_path = Path(image_path)

    logger.info(f"🎯 Processing: {image_path.name}")
    logger.debug(f"   Full path: {image_path}")
    logger.debug(f"   File size: {image_path.stat().st_size / 1024:.1f} KB")

    results = {
        'image_path': str(image_path),
        'processing_time': 0,
        'page_type': '',
        'panels_detected': 0,
        'panel_confidence': 0.0,
        'annotated_image': '',
        'panel_paths': [],
        'xml_output': ''
    }

    try:
        # Step 0: Page Type Identification using VisionThink
        logger.info("📋 Step 0: Page Type Identification")
        page_type = identify_page_type(str(image_path))
        results['page_type'] = page_type
        logger.info(f"✓ Page type: {page_type}")
        step0_time = time.time()
        logger.debug(f"   Step 0 duration: {step0_time - start_time:.1f}s")

        if page_type == 'comic':
            # Step 1: Panel Detection using fine-tuned YOLO (on processed/grayscale images)
            logger.info("🎯 Step 1: Panel Detection")
            processed_image_path = _get_processed_image_path(image_path)
            panels = detect_panels_with_yolo(str(processed_image_path))
            results['panels_detected'] = len(panels)
            step1_time = time.time()
            logger.debug(f"   Step 1 duration: {step1_time - step0_time:.1f}s")
            logger.info(f"✓ Detected {len(panels)} panels")

            if panels:
                # Step 2: Create Annotated Image with Panel Boundaries
                logger.info("🎨 Step 2: Generate Annotated Image")
                annotated_dir = _pipeline_state['output_dir'] / config.PROCESSING_CONFIG["output_subdirs"]["annotated_panels"]
                annotated_filename = f"{image_path.stem}_annotated.png"
                annotated_image_path = annotated_dir / annotated_filename

                # Create annotated image using VisionThink function
                create_annotated_image(str(image_path), panels, str(annotated_image_path))
                results['annotated_image'] = str(annotated_image_path)
                step2_time = time.time()
                logger.debug(f"   Step 2 duration: {step2_time - step1_time:.1f}s")
                logger.info(f"✓ Annotated image saved: {annotated_image_path}")

                # Step 3: Validate Panel Detection and Determine Reading Order using VisionThink
                logger.info("🎯 Step 3: Panel Validation & Reading Order")
                validation = validate_panel_detection(str(annotated_image_path), panels)
                results['panel_confidence'] = validation['confidence']
                panel_order = validation['panel_order']
                step3_time = time.time()
                logger.debug(f"   Step 3 duration: {step3_time - step2_time:.1f}s")
                logger.info(f"✓ Panel confidence: {validation['confidence']:.2f}")
                if validation['issues']:
                    logger.warning(f"   Issues found: {', '.join(validation['issues'])}")

                # Step 4: Crop Individual Panels in Reading Order
                logger.info("✂️ Step 4: Crop Individual Panels")
                panels_dir = annotated_dir / f"{image_path.stem}" / "panels"
                panel_paths = crop_panels(str(image_path), panels, str(panels_dir), panel_order)
                results['panel_paths'] = panel_paths
                step4_time = time.time()
                logger.debug(f"   Step 4 duration: {step4_time - step3_time:.1f}s")
                logger.info(f"✓ Generated {len(panel_paths)} panel crops")

                # Step 5: Analyze Each Panel with VisionThink for Character Detection
                logger.info("🔍 Step 5: VisionThink Panel Analysis")
                panel_analyses = []
                previous_context = []

                for i, panel_path in enumerate(panel_paths):
                    logger.debug(f"   Analyzing panel {i+1}/{len(panel_paths)}")

                    # Analyze panel with VisionThink (enhanced character detection)
                    analysis = analyze_panel_with_context(
                        panel_path, i, len(panel_paths), previous_context
                    )

                    # Update context for next panels
                    if 'summary' in analysis:
                        previous_context.append(analysis)

                    panel_analyses.append(analysis)
                    logger.debug(f"   Panel {i+1} completed - {len(analysis.get('characters', []))} characters found")

                step5_time = time.time()
                logger.debug(f"   Step 5 duration: {step5_time - step4_time:.1f}s")
                logger.info(f"✓ Analyzed {len(panel_analyses)} panels with VisionThink")

                # Step 6: Generate Comprehensive XML Output
                logger.info("📄 Step 6: Generate Enhanced XML")
                xml_path = generate_enhanced_xml(
                    str(image_path), page_type, panels, panel_analyses,
                    validation, panel_order
                )
                results['xml_output'] = xml_path
                step6_time = time.time()
                logger.debug(f"   Step 6 duration: {step6_time - step5_time:.1f}s")
                logger.info(f"✓ Enhanced XML saved: {xml_path}")

            else:
                logger.warning("⚠ No panels detected")
                # Generate minimal XML for pages without panels
                xml_path = generate_enhanced_xml(str(image_path), page_type, [], [], {}, [])
                results['xml_output'] = xml_path

        else:
            # Handle non-comic pages (covers, ads, etc.)
            logger.info(f"📑 Processing as {page_type} page (minimal analysis)")
            xml_path = generate_enhanced_xml(str(image_path), page_type, [], [], {}, [])
            results['xml_output'] = xml_path

        results['processing_time'] = time.time() - start_time
        logger.info(f"✅ Complete in {results['processing_time']:.2f}s")

    except Exception as e:
        logger.error(f"❌ Error processing page: {e}")
        results['error'] = str(e)

    return results


def _update_character_context(character: Dict[str, Any], page_character_context: List[Dict]):
    """Update the page-level character context for continuity."""
    char_name = character.get('name', 'Unknown')

    # Check if this character is already in context
    existing_char = None
    for i, existing in enumerate(page_character_context):
        if existing.get('name') == char_name:
            existing_char = i
            break

    if existing_char is not None:
        # Update existing character with latest information
        page_character_context[existing_char].update(character)
    else:
        # Add new character
        page_character_context.append(character)

    # Keep only recent characters (max 10 per page for better context)
    if len(page_character_context) > 10:
        page_character_context[:] = page_character_context[-10:]


# Minimal CharacterRegistry implementation (placeholder)
class CharacterRegistry:
    """Minimal character registry implementation for pipeline compatibility."""

    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.characters = {}

    def register_character_appearance(self, **kwargs):
        """Register a character appearance (placeholder implementation)."""
        pass

    def classify_character_roles(self):
        """Return minimal character classification."""
        return {
            "primary_characters": [],
            "secondary_characters": [],
            "extra_characters": []
        }

    def analyze_character_relationships(self):
        """Return minimal relationship analysis."""
        return {"relationships": []}

    def get_statistics(self):
        """Return minimal statistics."""
        return {
            "total_characters": 0,
            "primary_count": 0,
            "secondary_count": 0,
            "extra_count": 0
        }


# Module-level pipeline state
_pipeline_state = None


def initialize_pipeline(
    visionthink_model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
    yolo_model: YOLO,
    output_dir: str = str(config.DEFAULT_OUTPUT_DIR)
) -> None:
    """
    Initialize the enhanced comic analysis pipeline with SOTA character tracking.

    This function sets up the module-level state required for processing comic pages
    with advanced character identification and tracking capabilities.

    Enhanced Pipeline Workflow:
    0. Page Identification → VisionThink determines page type (cover/ad/comic)
    1. Panel Identification → Fine-tuned YOLO detects panels (grayscale)
    2. Advanced Character Analysis → Multi-turn VisionThink with adaptive resolution
    3. Character Registry → Sophisticated character tracking across panels/pages
    4. Character Verification → LLM-as-Judge pattern for character validation
    5. Relationship Analysis → Character interaction and relationship mapping
    6. XML Output → Enhanced structured XML with character data
    """
    global _pipeline_state

    # Device setup first
    device = config.PROCESSING_CONFIG.get("device", "auto")
    if device == "auto":
        device = visionthink_model.device if visionthink_model else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize character registry
    registry_path = Path(output_dir) / "character_registry.json"
    character_registry = CharacterRegistry(registry_path)

    # Initialize pipeline state
    _pipeline_state = {
        'models': {
            'visionthink_model': visionthink_model,
            'tokenizer': tokenizer,
            'processor': processor,
            'yolo_model': yolo_model
        },
        'output_dir': Path(output_dir),
        'device': device,
        'xml_validator': XMLValidator(),
        'character_registry': character_registry,
        'initialized': True
    }

    # Initialize VisionThink module with enhanced functionality
    try:
        initialize_visionthink(
            model=visionthink_model,
            tokenizer=tokenizer,
            processor=processor,
            device=device
        )
        logger.info("✅ VisionThink module initialized in pipeline")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VisionThink module: {e}")
        raise RuntimeError(f"VisionThink initialization failed: {e}")

        # Create output directory structure
    base_output_path = Path(output_dir)
    xml_subdir = config.PROCESSING_CONFIG["output_subdirs"]["xml"]
    panels_subdir = config.PROCESSING_CONFIG["output_subdirs"]["annotated_panels"]
    (_pipeline_state['output_dir'] / xml_subdir).mkdir(parents=True, exist_ok=True)
    (_pipeline_state['output_dir'] / panels_subdir).mkdir(parents=True, exist_ok=True)

    logger.info("🚀 Comic Pipeline initialized")
    logger.info(f"   📊 Character registry: {len(character_registry.characters)} existing characters")
    logger.info(f"   🎯 VisionThink analysis enabled")

def identify_page_type(image_path: str) -> str:
    """
    Step 0: Identify comic page type using VisionThink model.
    """
    if not _pipeline_state or not _pipeline_state.get('initialized'):
        logger.error("❌ Pipeline not initialized - call initialize_pipeline() first")
        return "comic"

    # Use the imported VisionThink function directly
    from visionthink import identify_page_type as vt_identify_page_type
    return vt_identify_page_type(image_path)


def detect_panels_with_yolo(image_path: str) -> List[Dict[str, Any]]:
    """
    Step 1: Use fine-tuned YOLO model to detect panels on grayscale image.

    Returns: List of panel detections with coordinates
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("Detecting panels with fine-tuned YOLO...")

    try:
        # Convert to grayscale for panel detection (as trained)
        image = Image.open(image_path).convert('L')

        # Run YOLO detection
        yolo_model = _pipeline_state['models']['yolo_model']
        results = yolo_model(
            image,
            conf=config.YOLO_CONFIG["confidence_threshold"],
            iou=config.YOLO_CONFIG["iou_threshold"]
        )

        panels = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()

                    # Filter out very small panels
                    panel_area = (x2 - x1) * (y2 - y1)
                    if panel_area > config.PROCESSING_CONFIG["panel_min_area"]:
                        panels.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'area': panel_area
                        })

        # Sort panels by position (top to bottom, left to right)
        panels = _sort_panels_reading_order(panels)

        # Limit number of panels per page
        max_panels = config.PROCESSING_CONFIG["max_panels_per_page"]
        if len(panels) > max_panels:
            logger.warning(f"⚠️ Found {len(panels)} panels, limiting to {max_panels}")
            panels = panels[:max_panels]

        logger.debug(f"✓ Detected {len(panels)} panels")
        return panels

    except Exception as e:
        logger.error(f"Error in panel detection: {e}")
        return []


def analyze_panel_with_context(image_path: str, panel_bbox: List[int], panel_index: int, total_panels: int, page_character_context: List[Dict]) -> Dict[str, Any]:
    """
    Step 2: Panel analysis using VisionThink.
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug(f"🔍 Analyzing panel {panel_index + 1}/{total_panels}")

    try:
        # Use VisionThink for panel analysis
        from visionthink import analyze_panel_with_context as vt_analyze_panel
        return vt_analyze_panel(
            image_path, panel_bbox, panel_index, total_panels, page_character_context
        )

    except Exception as e:
        logger.error(f"Error in panel analysis {panel_index}: {e}")
        return {'error': str(e), 'panel_index': panel_index}


def classify_page_characters(all_panel_data: List[Dict]) -> Dict[str, Any]:
    """
    Step 3: Basic character classification using character registry.
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("👥 Character classification...")

    try:
        # Get character registry
        character_registry = _pipeline_state['character_registry']

        # Update character registry with panel data
        for panel_data in all_panel_data:
            characters = panel_data.get('characters', [])
            panel_id = panel_data.get('panel_id', f"panel_{panel_data.get('panel_index', 0)}")

            for char_data in characters:
                character_registry.register_character_appearance(
                    character_name=char_data.get('name', 'Unknown'),
                    panel_id=panel_id,
                    page_path=panel_data.get('image_path', ''),
                    analysis_data=char_data,
                    resolution_used=panel_data.get('resolution_used', 'adaptive')
                )

        # Get character classification
        classification = character_registry.classify_character_roles()
        relationships = character_registry.analyze_character_relationships()
        stats = character_registry.get_statistics()

        result = {
            **classification,
            'character_relationships': relationships,
            'analysis_statistics': stats,
            'classification_method': 'basic'
        }

        logger.debug(f"✓ Classified {stats.get('total_characters', 0)} characters:")
        logger.debug(f"   Primary: {len(classification.get('primary_characters', []))}")
        logger.debug(f"   Secondary: {len(classification.get('secondary_characters', []))}")
        logger.debug(f"   Extra: {len(classification.get('extra_characters', []))}")

        return result

    except Exception as e:
        logger.error(f"Error in character classification: {e}")
        return {
            'primary_characters': [],
            'secondary_characters': [],
            'extra_characters': [],
            'total_unique_characters': 0,
            'error': str(e)
        }


def generate_validated_xml(image_path: str, page_type: str, panel_data: List[Dict], character_classification: Dict) -> str:
    """
    Step 4: Generate validated XML output for the page.
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("📄 Generating comprehensive validated XML output...")

    try:
        # Create XML structure with enhanced metadata
        root = ET.Element("comic_page_analysis")
        root.set("version", "2.0")
        root.set("image_file", Path(image_path).name)
        root.set("image_path", str(image_path))
        root.set("page_type", page_type)
        root.set("analysis_date", datetime.now().isoformat())
        root.set("total_panels", str(len(panel_data)))
        root.set("processing_pipeline", "VisionThink+YOLO")

        # Add image metadata
        try:
            image = Image.open(image_path)
            image_meta = ET.SubElement(root, "image_metadata")
            image_meta.set("width", str(image.width))
            image_meta.set("height", str(image.height))
            image_meta.set("format", image.format or "unknown")
            image_meta.set("mode", image.mode)
            image_meta.set("file_size", str(Path(image_path).stat().st_size))
            logger.debug(f"   Added image metadata: {image.width}x{image.height}")
        except Exception as e:
            logger.warning(f"   Could not read image metadata: {e}")

        # Add character summary with enhanced details
        char_summary = ET.SubElement(root, "character_summary")
        char_summary.set("total_unique", str(character_classification.get('total_unique_characters', 0)))
        char_summary.set("primary_count", str(len(character_classification.get('primary_characters', []))))
        char_summary.set("secondary_count", str(len(character_classification.get('secondary_characters', []))))
        char_summary.set("extra_count", str(len(character_classification.get('extra_characters', []))))

        # Add detailed character classifications
        for role_type in ['primary_characters', 'secondary_characters', 'extra_characters']:
            role_element = ET.SubElement(char_summary, role_type.replace('_characters', '_chars'))
            characters = character_classification.get(role_type, [])
            role_element.set("count", str(len(characters)))

            for char in characters:
                char_elem = ET.SubElement(role_element, "character")
                char_elem.set("name", char.get('name', 'unnamed'))
                char_elem.set("appearances", str(char.get('appearances', 0)))
                char_elem.set("dialogue_lines", str(char.get('dialogue_lines', 0)))

                # Add additional character attributes if available
                for attr in ['description', 'role', 'significance']:
                    if attr in char and char[attr]:
                        char_elem.set(attr, str(char[attr]))

        # Add comprehensive panel data
        panels_elem = ET.SubElement(root, "panels")
        panels_elem.set("detection_method", "fine_tuned_yolo")
        panels_elem.set("analysis_method", "visionthink_contextual")

        for i, panel in enumerate(panel_data):
            panel_elem = ET.SubElement(panels_elem, "panel")
            panel_elem.set("number", str(i + 1))
            panel_elem.set("sequence_order", str(i + 1))

            # Add panel bounding box if available
            if 'bbox' in panel:
                bbox = panel['bbox']
                panel_elem.set("bbox_x1", str(bbox[0]))
                panel_elem.set("bbox_y1", str(bbox[1]))
                panel_elem.set("bbox_x2", str(bbox[2]))
                panel_elem.set("bbox_y2", str(bbox[3]))
                panel_elem.set("area", str((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))

            # Add panel confidence if available
            if 'confidence' in panel:
                panel_elem.set("detection_confidence", str(round(panel['confidence'], 3)))

            # Add characters in panel with enhanced structure
            if 'characters' in panel and panel['characters']:
                chars_elem = ET.SubElement(panel_elem, "characters")
                chars_elem.set("count", str(len(panel['characters'])))

                for char in panel['characters']:
                    char_elem = ET.SubElement(chars_elem, "character")

                    # Core character attributes
                    char_elem.set("name", char.get('name', 'Unknown character'))
                    char_elem.set("description", char.get('description', ''))
                    char_elem.set("role", char.get('role', 'unknown'))

                    # Additional attributes if present
                    if char.get('action'):
                        char_elem.set("action", char['action'])
                    if char.get('dialogue'):
                        char_elem.set("dialogue", char['dialogue'])

            # Add comprehensive content analysis - mimicking old XML structure
            content_elem = ET.SubElement(panel_elem, "content_analysis")

            # Setting section
            if panel.get('setting'):
                setting_elem = ET.SubElement(content_elem, "setting")
                setting_elem.text = panel['setting']

            # Mood section
            if panel.get('mood'):
                mood_elem = ET.SubElement(content_elem, "mood")
                mood_elem.text = panel['mood']

            # Story elements section
            if panel.get('story_elements'):
                story_elem = ET.SubElement(content_elem, "story_elements")
                story_elem.text = panel['story_elements']

            # Add narrative elements if available
            if any(k in panel for k in ['narrative_function', 'pacing', 'emotional_tone']):
                narrative_elem = ET.SubElement(panel_elem, "narrative_analysis")
                for key in ['narrative_function', 'pacing', 'emotional_tone']:
                    if key in panel and panel[key]:
                        narrative_elem.set(key, str(panel[key]))

        # Add processing metadata
        processing_meta = ET.SubElement(root, "processing_metadata")
        processing_meta.set("analysis_type", "full" if page_type == "comic" else "minimal")

        # Format XML with proper indentation
        logger.debug("   Formatting XML structure...")
        ET.indent(root, space="  ", level=0)
        xml_str = ET.tostring(root, encoding='unicode')

        # Validate and potentially fix XML
        logger.debug("   Validating XML structure...")
        xml_validator = _pipeline_state['xml_validator']
        if xml_validator:
            is_valid, final_xml, correction_method = xml_validator.validate_and_fix(xml_str)
            if is_valid:
                xml_str = final_xml
                if correction_method:
                    logger.debug(f"✓ XML corrected using: {correction_method}")
                logger.debug("✓ XML validation passed")
            else:
                logger.warning(f"⚠ XML validation failed: {final_xml}")
        else:
            logger.debug("   XML validator not available, skipping validation")

        # Save XML with enhanced filename
        xml_filename = f"{Path(image_path).stem}_analysis.xml"
        xml_subdir = config.PROCESSING_CONFIG["output_subdirs"]["xml"]
        xml_path = _pipeline_state['output_dir'] / xml_subdir / xml_filename

        # Ensure output directory exists
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        logger.debug(f"   Saving XML to: {xml_path}")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(f'<!-- Page Type: {page_type} -->\n')
            f.write(xml_str)

        logger.info(f"✓ Enhanced XML saved: {xml_path}")
        logger.debug(f"   XML file size: {xml_path.stat().st_size} bytes")
        return str(xml_path)

    except Exception as e:
        logger.error(f"❌ Error generating XML: {e}")
        logger.debug(f"   Exception details: {type(e).__name__}: {str(e)}")
        return ""


def generate_annotated_images(image_path: str, panels: List[Dict], page_type: str) -> str:
    """
    Generate annotated images with panel bounding boxes and save them.

    Args:
        image_path: Path to the original image
        panels: List of panel detection data with bounding boxes
        page_type: Type of page (comic, cover, etc.)

    Returns:
        Path to saved annotated image
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("🎨 Generating annotated image with panel boundaries...")

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return ""

        # Create copy for annotation
        annotated_image = image.copy()

        # Draw panel bounding boxes
        for i, panel in enumerate(panels):
            if 'bbox' in panel:
                x1, y1, x2, y2 = panel['bbox']
                confidence = panel.get('confidence', 0.0)

                # Draw rectangle
                color = (0, 255, 0)  # Green for panels
                thickness = 3
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)

                # Add panel number and confidence
                label = f"Panel {i+1}"
                if confidence > 0:
                    label += f" ({confidence:.2f})"

                # Calculate label position
                label_y = y1 - 10 if y1 > 30 else y1 + 25

                # Add text background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    annotated_image,
                    (x1, label_y - text_height - 5),
                    (x1 + text_width + 10, label_y + 5),
                    (0, 0, 0),  # Black background
                    -1
                )

                # Add text
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White text
                    2
                )

        # Add page info
        page_info = f"Page Type: {page_type} | Panels: {len(panels)}"
        cv2.putText(
            annotated_image,
            page_info,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),  # White text
            2
        )

        # Add black background for text
        (text_width, text_height), _ = cv2.getTextSize(
            page_info, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )
        cv2.rectangle(
            annotated_image,
            (15, 10),
            (25 + text_width, 50),
            (0, 0, 0),  # Black background
            -1
        )
        cv2.putText(
            annotated_image,
            page_info,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),  # White text
            2
        )

        # Save annotated image
        annotated_filename = f"{Path(image_path).stem}_annotated.png"
        panels_subdir = config.PROCESSING_CONFIG["output_subdirs"]["annotated_panels"]
        annotated_path = _pipeline_state['output_dir'] / panels_subdir / annotated_filename

        # Ensure output directory exists
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        success = cv2.imwrite(str(annotated_path), annotated_image)

        if success:
            logger.info(f"✓ Annotated image saved: {annotated_path}")
            logger.debug(f"   Image size: {annotated_path.stat().st_size / 1024:.1f} KB")
            return str(annotated_path)
        else:
            logger.error("Failed to save annotated image")
            return ""

    except Exception as e:
        logger.error(f"Error generating annotated image: {e}")
        return ""


def generate_panel_crops(image_path: str, panels: List[Dict]) -> List[str]:
    """
    Generate individual panel crop images.

    Args:
        image_path: Path to the original image
        panels: List of panel detection data with bounding boxes

    Returns:
        List of paths to saved panel crop images
    """
    if _pipeline_state is None:
        raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")

    logger.debug("✂️ Generating individual panel crops...")

    crop_paths = []

    try:
        # Load image
        image = Image.open(image_path)

        for i, panel in enumerate(panels):
            if 'bbox' in panel:
                x1, y1, x2, y2 = panel['bbox']

                # Crop panel
                panel_crop = image.crop((x1, y1, x2, y2))

                # Save crop
                crop_filename = f"{Path(image_path).stem}_panel_{i+1:02d}.png"
                panels_subdir = config.PROCESSING_CONFIG["output_subdirs"]["annotated_panels"]
                crop_path = _pipeline_state['output_dir'] / panels_subdir / crop_filename

                # Ensure output directory exists
                crop_path.parent.mkdir(parents=True, exist_ok=True)

                # Save crop
                panel_crop.save(crop_path)
                crop_paths.append(str(crop_path))

                logger.debug(f"   Panel {i+1} crop saved: {crop_path.name}")

        logger.info(f"✓ Generated {len(crop_paths)} panel crops")
        return crop_paths

    except Exception as e:
        logger.error(f"Error generating panel crops: {e}")
        return []


def generate_enhanced_xml(image_path: str, page_type: str, panels: List[Dict],
                         panel_analyses: List[Dict], validation: Dict,
                         panel_order: List[int]) -> str:
    """Generate comprehensive XML output with enhanced VisionThink analysis."""
    try:
        image_path = Path(image_path)

        # Create XML structure
        root = ET.Element("comic_page_analysis")
        root.set("version", "2.0")
        root.set("image_file", image_path.name)
        root.set("image_path", str(image_path))
        root.set("page_type", page_type)
        root.set("analysis_date", datetime.now().isoformat())
        root.set("total_panels", str(len(panels)))
        root.set("processing_pipeline", "VisionThink+YOLO")

        # Add processing comment
        comment = ET.Comment(f" Page Type: {page_type} ")
        root.insert(0, comment)

        # Image metadata
        if image_path.exists():
            from PIL import Image
            with Image.open(image_path) as img:
                metadata = ET.SubElement(root, "image_metadata")
                metadata.set("width", str(img.width))
                metadata.set("height", str(img.height))
                metadata.set("format", img.format or "Unknown")
                metadata.set("mode", img.mode)
                metadata.set("file_size", str(image_path.stat().st_size))

        # Panel validation info
        if validation:
            validation_elem = ET.SubElement(root, "panel_validation")
            validation_elem.set("confidence", f"{validation.get('confidence', 0.0):.2f}")
            validation_elem.set("reading_order", ",".join(map(str, validation.get('panel_order', []))))
            if validation.get("issues"):
                issues_elem = ET.SubElement(validation_elem, "issues")
                issues_elem.text = "; ".join(validation["issues"])

        # Character summary - Enhanced character aggregation
        all_characters = []
        for analysis in panel_analyses:
            all_characters.extend(analysis.get("characters", []))

        char_summary = ET.SubElement(root, "character_summary")

        # Build character registry with proper name handling
        character_registry = {}
        for char in all_characters:
            char_name = char.get("name", "Unknown")

            # Skip truly unknown characters but include named ones (even if generic)
            if char_name in ["Unknown", ""]:
                continue

            if char_name not in character_registry:
                character_registry[char_name] = {
                    "name": char_name,
                    "description": char.get("description", ""),
                    "full_description": char.get("full_description", char.get("description", "")),
                    "appearances": 1,
                    "role": char.get("role", "unknown"),
                    "dialogue": char.get("dialogue", ""),
                    "action": char.get("action", "")
                }
            else:
                # Update existing character
                character_registry[char_name]["appearances"] += 1
                # Keep the most detailed description
                if len(char.get("full_description", "")) > len(character_registry[char_name]["full_description"]):
                    character_registry[char_name]["full_description"] = char.get("full_description", "")
                    character_registry[char_name]["description"] = char.get("description", "")
                # Update role if more specific
                current_role = character_registry[char_name]["role"]
                new_role = char.get("role", "unknown")
                if new_role == "primary" or (new_role == "secondary" and current_role != "primary"):
                    character_registry[char_name]["role"] = new_role

        unique_chars = list(character_registry.values())

        # Classify characters by role
        primary_chars = [char for char in unique_chars if char.get("role", "").lower() == "primary"]
        secondary_chars = [char for char in unique_chars if char.get("role", "").lower() == "secondary"]
        extra_chars = [char for char in unique_chars if char.get("role", "").lower() == "extra"]
        unknown_chars = [char for char in unique_chars if char.get("role", "").lower() == "unknown"]

        char_summary.set("total_unique", str(len(unique_chars)))
        char_summary.set("primary_count", str(len(primary_chars)))
        char_summary.set("secondary_count", str(len(secondary_chars)))
        char_summary.set("extra_count", str(len(extra_chars)))
        char_summary.set("unknown_count", str(len(unknown_chars)))

        # Character details
        if primary_chars:
            primary_elem = ET.SubElement(char_summary, "primary_chars")
            primary_elem.set("count", str(len(primary_chars)))
            for char in primary_chars:
                char_elem = ET.SubElement(primary_elem, "character")
                char_elem.set("name", char.get("name", "Unknown"))
                char_elem.set("description", char.get("full_description", char.get("description", "")))
                char_elem.set("appearances", str(char.get("appearances", 1)))

        if secondary_chars:
            secondary_elem = ET.SubElement(char_summary, "secondary_chars")
            secondary_elem.set("count", str(len(secondary_chars)))
            for char in secondary_chars:
                char_elem = ET.SubElement(secondary_elem, "character")
                char_elem.set("name", char.get("name", "Unknown"))
                char_elem.set("description", char.get("full_description", char.get("description", "")))
                char_elem.set("appearances", str(char.get("appearances", 1)))

        if extra_chars:
            extra_elem = ET.SubElement(char_summary, "extra_chars")
            extra_elem.set("count", str(len(extra_chars)))
            for char in extra_chars:
                char_elem = ET.SubElement(extra_elem, "character")
                char_elem.set("name", char.get("name", "Unknown"))
                char_elem.set("description", char.get("full_description", char.get("description", "")))
                char_elem.set("appearances", str(char.get("appearances", 1)))

        if unknown_chars:
            unknown_elem = ET.SubElement(char_summary, "unknown_chars")
            unknown_elem.set("count", str(len(unknown_chars)))
            for char in unknown_chars:
                char_elem = ET.SubElement(unknown_elem, "character")
                char_elem.set("name", char.get("name", "Unknown"))
                char_elem.set("description", char.get("full_description", char.get("description", "")))
                char_elem.set("appearances", str(char.get("appearances", 1)))

        # Panels section
        panels_elem = ET.SubElement(root, "panels")
        panels_elem.set("detection_method", "fine_tuned_yolo")
        panels_elem.set("analysis_method", "visionthink_enhanced")

        for i, analysis in enumerate(panel_analyses):
            panel_elem = ET.SubElement(panels_elem, "panel")
            panel_elem.set("number", str(i + 1))
            panel_elem.set("sequence_order", str(i + 1))

            # Characters in this panel
            characters = analysis.get("characters", [])
            if characters:
                chars_elem = ET.SubElement(panel_elem, "characters")
                chars_elem.set("count", str(len(characters)))

                for char in characters:
                    char_elem = ET.SubElement(chars_elem, "character")
                    char_elem.set("name", char.get("name", "Unknown"))
                    char_elem.set("description", char.get("full_description", char.get("description", "")))
                    char_elem.set("role", char.get("role", "unknown"))
                    char_elem.set("action", char.get("action", ""))
                    if char.get("dialogue"):
                        char_elem.set("dialogue", char.get("dialogue"))

            # Setting and mood
            if analysis.get("setting"):
                setting_elem = ET.SubElement(panel_elem, "setting")
                setting_elem.text = analysis["setting"]

            if analysis.get("mood"):
                mood_elem = ET.SubElement(panel_elem, "mood")
                mood_elem.text = analysis["mood"]

            # Dialogue
            dialogue_entries = analysis.get("dialogue", [])
            if dialogue_entries:
                dialogue_elem = ET.SubElement(panel_elem, "dialogue")
                for entry in dialogue_entries:
                    line_elem = ET.SubElement(dialogue_elem, "line")
                    line_elem.set("speaker", entry.get("speaker", "Unknown"))
                    line_elem.text = entry.get("text", "")

            # Story elements
            if analysis.get("story_elements"):
                story_elem = ET.SubElement(panel_elem, "story_elements")
                story_elem.text = analysis["story_elements"]

        # Processing metadata
        processing_elem = ET.SubElement(root, "processing_metadata")
        processing_elem.set("analysis_type", "enhanced_visionthink")

        # Save XML
        xml_dir = _pipeline_state["output_dir"] / config.PROCESSING_CONFIG["output_subdirs"]["xml"]
        xml_dir.mkdir(parents=True, exist_ok=True)
        xml_filename = f"{image_path.stem}_analysis.xml"
        xml_path = xml_dir / xml_filename

        # Format and save XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        logger.debug(f"✓ Enhanced XML saved: {xml_path}")
        return str(xml_path)

    except Exception as e:
        logger.error(f"❌ Error generating enhanced XML: {e}")
        raise


def _get_processed_image_path(raw_image_path: Path) -> Path:
    """Get the corresponding processed/grayscale image path for panel detection."""
    # Convert from data/fantomas/raw/ to data/fantomas/processed/grayscale/
    processed_path = raw_image_path.parent.parent / "processed" / "grayscale" / raw_image_path.name

    if processed_path.exists():
        return processed_path
    else:
        logger.warning(f"⚠️ Processed image not found: {processed_path}, using raw image")
        return raw_image_path

