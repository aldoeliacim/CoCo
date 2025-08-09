import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Standardized project imports
import config
from config import setup_logger, VISIONTHINK_CONFIG

logger = setup_logger("visionthink")

# Module-level state for VisionThink models
_visionthink_state = {
    'model': None,
    'tokenizer': None,
    'processor': None,
    'device': None,
    'initialized': False
}


def initialize_visionthink(model: Qwen2_5_VLForConditionalGeneration,
                          tokenizer: AutoTokenizer,
                          processor: AutoProcessor,
                          device: str = "auto"):
    """Initialize VisionThink module with models."""
    global _visionthink_state

    _visionthink_state['model'] = model
    _visionthink_state['tokenizer'] = tokenizer
    _visionthink_state['processor'] = processor
    _visionthink_state['device'] = device
    _visionthink_state['initialized'] = True

    logger.debug("✅ VisionThink module initialized")


def identify_page_type(image_path: str) -> str:
    """Identify comic page type using VisionThink model."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized - call initialize_visionthink() first")
        return "comic"

    logger.info(f"🔍 Starting page type identification for: {image_path}")

    if not os.path.exists(image_path):
        logger.error(f"❌ Image file not found: {image_path}")
        return "comic"

    # Clear GPU memory before processing
    logger.debug("🧹 Clearing GPU cache before processing...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Load and preprocess image
        logger.debug(f"📱 Loading image from: {image_path}")
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
        logger.debug("🤖 Generating VisionThink response...")
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


def analyze_panel_with_context(panel_image_path: str, panel_index: int, total_panels: int,
                             previous_context: List[Dict] = None) -> Dict[str, Any]:
    """Analyze a single panel with VisionThink, focusing on character detection."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized - call initialize_visionthink() first")
        return {'error': 'VisionThink not initialized'}

    logger.debug(f"� Analyzing panel {panel_index + 1}/{total_panels}: {panel_image_path}")

    try:
        # Load panel image
        panel_image = Image.open(panel_image_path).convert('RGB')

        # Build context from previous panels
        context_text = ""
        if previous_context:
            context_text = "Previous panels context:\n"
            for i, ctx in enumerate(previous_context[-3:]):  # Last 3 panels for context
                context_text += f"Panel {i+1}: {ctx.get('summary', '')}\n"

        # Enhanced analysis prompt focusing on character detection
        prompt = f"""Analyze this comic panel in detail. This is panel {panel_index + 1} of {total_panels}.

{context_text}

Provide a comprehensive analysis focusing on CHARACTER DETECTION:

**CHARACTERS:**
For each character visible, provide:
- NAME: Specific name if identifiable, or descriptive identifier
- DESCRIPTION: Detailed physical appearance, clothing, pose
- ACTION: What the character is doing
- DIALOGUE: Any speech or text associated with this character
- ROLE: Primary (main focus), Secondary (supporting), or Extra (background)
- EMOTIONS: Visible emotional state

**SCENE DETAILS:**
- SETTING: Location and environment description
- MOOD: Overall emotional tone and atmosphere
- OBJECTS: Important items or props visible

**DIALOGUE & TEXT:**
- Extract all visible text, speech bubbles, sound effects
- Indicate which character speaks each line

**STORY ELEMENTS:**
- Key plot developments in this panel
- Character interactions and relationships
- Narrative significance

Be specific and detailed, especially about character identification."""

        response = _generate_visionthink_response(panel_image, prompt)

        # Parse the detailed analysis
        analysis_result = _parse_panel_analysis(response, panel_index)

        # Add summary for next panel context
        analysis_result['summary'] = _create_panel_summary(analysis_result)

        logger.debug(f"✓ Panel {panel_index + 1} analysis complete - {len(analysis_result.get('characters', []))} characters found")
        return analysis_result

    except Exception as e:
        logger.error(f"❌ Error analyzing panel {panel_index + 1}: {e}")
        return {
            'error': str(e),
            'panel_index': panel_index,
            'characters': [],
            'setting': 'Analysis failed',
            'mood': 'Unknown',
            'dialogue': [],
            'story_elements': 'Could not analyze panel'
        }


def validate_panel_detection(annotated_image_path: str, panels: List[Dict]) -> Dict[str, Any]:
    """Use VisionThink to validate panel detection quality and determine panel order."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized - call initialize_visionthink() first")
        return {'confidence': 0.0, 'panel_order': list(range(len(panels))), 'issues': ['VisionThink not initialized']}

    logger.debug(f"🎯 Validating panel detection with {len(panels)} panels...")

    try:
        # Load the annotated image
        image = Image.open(annotated_image_path).convert('RGB')

        # Create comprehensive validation prompt
        prompt = f"""Analyze this comic page with {len(panels)} detected panels marked with green rectangles.

Evaluate the panel detection quality:

1. DETECTION ACCURACY:
   - Are all comic panels properly detected?
   - Any missing panels or false positives?
   - Are panel boundaries accurate?

2. READING ORDER:
   - What is the correct reading order of these panels?
   - Number the panels from 1 to {len(panels)} in proper reading sequence

3. CONFIDENCE ASSESSMENT:
   - Rate detection quality from 0.0 to 1.0
   - Consider completeness, accuracy, and boundary precision

Respond in this format:
CONFIDENCE: [0.0-1.0]
READING_ORDER: [comma-separated panel numbers]
ISSUES: [list any problems found]
ANALYSIS: [detailed assessment]"""

        response = _generate_visionthink_response(image, prompt)

        # Parse the validation response
        validation_result = _parse_validation_response(response, len(panels))

        logger.debug(f"✓ Panel validation complete - confidence: {validation_result['confidence']:.2f}")
        return validation_result

    except Exception as e:
        logger.error(f"❌ Error in panel validation: {e}")
        return {
            'confidence': 0.5,  # Default moderate confidence
            'panel_order': list(range(len(panels))),
            'issues': [f'Validation error: {str(e)}'],
            'analysis': 'Could not validate panel detection'
        }


def create_annotated_image(image_path: str, panels: List[Dict], output_path: str) -> str:
    """Create annotated image with panel bounding boxes."""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Draw panel bounding boxes
        for i, panel in enumerate(panels):
            bbox = panel['bbox']
            x1, y1, x2, y2 = bbox

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Add panel number
            label = f"Panel {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            label_x = x1
            label_y = y1 - 10 if y1 > 30 else y1 + label_size[1] + 10

            # Background for text
            cv2.rectangle(image, (label_x, label_y - label_size[1]),
                         (label_x + label_size[0], label_y + 5), (0, 255, 0), -1)
            cv2.putText(image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Save annotated image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

        logger.debug(f"✓ Annotated image saved: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"❌ Error creating annotated image: {e}")
        raise


def crop_panels(image_path: str, panels: List[Dict], output_dir: str, panel_order: List[int] = None) -> List[str]:
    """Crop individual panels from the image."""
    try:
        # Load original image
        image = Image.open(image_path).convert('RGB')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        panel_paths = []
        order = panel_order if panel_order else list(range(len(panels)))

        for i, panel_idx in enumerate(order):
            if panel_idx >= len(panels):
                continue

            panel = panels[panel_idx]
            bbox = panel['bbox']
            x1, y1, x2, y2 = bbox

            # Crop panel
            panel_image = image.crop((x1, y1, x2, y2))

            # Save panel with correct order number
            panel_filename = f"panel_{i+1:02d}.png"
            panel_path = os.path.join(output_dir, panel_filename)
            panel_image.save(panel_path)

            panel_paths.append(panel_path)
            logger.debug(f"✓ Panel {i+1} cropped: {panel_filename}")

        logger.debug(f"✓ All {len(panel_paths)} panels cropped to: {output_dir}")
        return panel_paths

    except Exception as e:
        logger.error(f"❌ Error cropping panels: {e}")
        raise


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


def _extract_simple_characters(response: str) -> List[Dict[str, str]]:
    """Extract simple character information from response."""
    try:
        # Very simple extraction - look for character mentions
        characters = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ['character', 'person', 'figure', 'man', 'woman']):
                # Extract basic character info
                characters.append({
                    'name': 'Character',  # Generic name for now
                    'description': line,
                    'role': 'unknown'
                })

        return characters if characters else [{'name': 'Unknown', 'description': 'No clear characters identified', 'role': 'unknown'}]

    except Exception as e:
        logger.debug(f"Error extracting characters: {e}")
        return [{'name': 'Unknown', 'description': 'Parse error', 'role': 'unknown'}]


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

        # Generate response with simple timeout
        with torch.no_grad():
            generated_ids = _visionthink_state['model'].generate(
                **inputs,
                max_new_tokens=min(VISIONTHINK_CONFIG["max_new_tokens"], 150),
                temperature=VISIONTHINK_CONFIG["temperature"],
                do_sample=VISIONTHINK_CONFIG["do_sample"],
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


def analyze_panel_with_context(image_path: str, panel_bbox: List[int], panel_index: int,
                             total_panels: int, page_character_context: List[Dict] = None) -> Dict[str, Any]:
    """Analyze a single panel with character context."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized - call initialize_visionthink() first")
        return {'error': 'VisionThink not initialized'}

    logger.debug(f"👁️  Analyzing panel {panel_index + 1}/{total_panels}...")

    try:
        # Load full-color image and crop panel
        image = Image.open(image_path)
        x1, y1, x2, y2 = panel_bbox
        panel_image = image.crop((x1, y1, x2, y2))

        # Build context from previous panels
        character_context = ""
        if page_character_context:
            character_context = "Characters seen in previous panels: " + \
                              ", ".join([f"{char.get('name', 'Unknown')}"
                                       for char in page_character_context[-3:]])

        # Simple panel analysis prompt
        prompt = f"""Analyze this comic panel briefly. This is panel {panel_index + 1} of {total_panels}.

{character_context}

Provide a brief analysis including:
- Characters visible (names or descriptions)
- Main action or dialogue
- Scene setting

Keep response concise and focused."""

        response = _generate_visionthink_response(panel_image, prompt)

        # Simple parsing - just return the raw response for now
        return {
            'panel_index': panel_index,
            'raw_analysis': response,
            'characters': _extract_simple_characters(response)
        }

    except Exception as e:
        logger.error(f"Error analyzing panel {panel_index}: {e}")
        return {'error': str(e)}


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


def _extract_simple_characters(response: str) -> List[Dict[str, str]]:
    """Extract simple character information from response."""
    try:
        # Very simple extraction - look for character mentions
        characters = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ['character', 'person', 'figure', 'man', 'woman']):
                # Extract basic character info
                characters.append({
                    'name': 'Character',  # Generic name for now
                    'description': line,
                    'role': 'unknown'
                })

        return characters if characters else [{'name': 'Unknown', 'description': 'No clear characters identified', 'role': 'unknown'}]

    except Exception as e:
        logger.debug(f"Error extracting characters: {e}")
        return [{'name': 'Unknown', 'description': 'Parse error', 'role': 'unknown'}]


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

        # Generate response with simple timeout
        with torch.no_grad():
            generated_ids = _visionthink_state['model'].generate(
                **inputs,
                max_new_tokens=min(VISIONTHINK_CONFIG["max_new_tokens"], 150),
                temperature=VISIONTHINK_CONFIG["temperature"],
                do_sample=VISIONTHINK_CONFIG["do_sample"],
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
def analyze_panel_with_context(image_path: str, panel_bbox: List[int], panel_index: int,
                             total_panels: int, page_character_context: List[Dict] = None) -> Dict[str, Any]:
    """Analyze a single panel with character context."""
    if not _visionthink_state['initialized']:
        logger.error("❌ VisionThink not initialized - call initialize_visionthink() first")
        return {'error': 'VisionThink not initialized'}

    logger.debug(f"👁️  Analyzing panel {panel_index + 1}/{total_panels}...")

    try:
        # Load full-color image and crop panel
        image = Image.open(image_path)
        x1, y1, x2, y2 = panel_bbox
        panel_image = image.crop((x1, y1, x2, y2))

        # Build context from previous panels
        character_context = ""
        if page_character_context:
            character_context = "Characters seen in previous panels: " + \
                              ", ".join([f"{char.get('name', 'Unknown')}"
                                       for char in page_character_context[-3:]])

        # Simple panel analysis prompt
        prompt = f"""Analyze this comic panel briefly. This is panel {panel_index + 1} of {total_panels}.

{character_context}

Provide a brief analysis including:
- Characters visible (names or descriptions)
- Main action or dialogue
- Scene setting

Keep response concise and focused."""

        response = _generate_visionthink_response(panel_image, prompt)

        # Simple parsing - just return the raw response for now
        return {
            'panel_index': panel_index,
            'raw_analysis': response,
            'characters': _extract_simple_characters(response)
        }

    except Exception as e:
        logger.error(f"Error analyzing panel {panel_index}: {e}")
        return {'error': str(e)}


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


def _extract_simple_characters(response: str) -> List[Dict[str, str]]:
    """Extract simple character information from response."""
    try:
        # Very simple extraction - look for character mentions
        characters = []
        lines = response.split('\n')

        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ['character', 'person', 'figure', 'man', 'woman']):
                # Extract basic character info
                characters.append({
                    'name': 'Character',  # Generic name for now
                    'description': line,
                    'role': 'unknown'
                })

        return characters if characters else [{'name': 'Unknown', 'description': 'No clear characters identified', 'role': 'unknown'}]

    except Exception as e:
        logger.debug(f"Error extracting characters: {e}")
        return [{'name': 'Unknown', 'description': 'Parse error', 'role': 'unknown'}]


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

        # Generate response with simple timeout
        with torch.no_grad():
            generated_ids = _visionthink_state['model'].generate(
                **inputs,
                max_new_tokens=min(VISIONTHINK_CONFIG["max_new_tokens"], 150),
                temperature=VISIONTHINK_CONFIG["temperature"],
                do_sample=VISIONTHINK_CONFIG["do_sample"],
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


def _parse_validation_response(response: str, num_panels: int) -> Dict[str, Any]:
    """Parse panel validation response from VisionThink."""
    try:
        result = {
            "confidence": 0.5,
            "panel_order": list(range(num_panels)),
            "issues": [],
            "analysis": response
        }

        # Extract confidence score
        confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.IGNORECASE)
        if confidence_match:
            result["confidence"] = float(confidence_match.group(1))

        # Extract reading order
        order_match = re.search(r"READING_ORDER:\s*([0-9,\s]+)", response, re.IGNORECASE)
        if order_match:
            order_str = order_match.group(1)
            try:
                # Parse comma-separated numbers and convert to 0-based indexing
                order_numbers = [int(x.strip()) - 1 for x in order_str.split(",") if x.strip().isdigit()]
                if len(order_numbers) == num_panels and all(0 <= x < num_panels for x in order_numbers):
                    result["panel_order"] = order_numbers
            except ValueError:
                pass

        return result

    except Exception as e:
        logger.debug(f"Error parsing validation response: {e}")
        return {
            "confidence": 0.5,
            "panel_order": list(range(num_panels)),
            "issues": [f"Parse error: {str(e)}"],
            "analysis": response
        }


def _parse_panel_analysis(response: str, panel_index: int) -> Dict[str, Any]:
    """Parse detailed panel analysis response from VisionThink."""
    try:
        result = {
            "panel_index": panel_index,
            "characters": [],
            "setting": "",
            "mood": "",
            "dialogue": [],
            "story_elements": "",
            "raw_analysis": response
        }

        # Parse characters section
        characters_section = _extract_section(response, "CHARACTERS")
        if characters_section:
            result["characters"] = _parse_characters(characters_section)

        # Parse scene details
        scene_section = _extract_section(response, "SCENE DETAILS")
        if scene_section:
            result["setting"] = _extract_field(scene_section, "SETTING") or ""
            result["mood"] = _extract_field(scene_section, "MOOD") or ""

        # Parse dialogue
        dialogue_section = _extract_section(response, "DIALOGUE & TEXT")
        if dialogue_section:
            result["dialogue"] = _parse_dialogue(dialogue_section)

        # Parse story elements
        story_section = _extract_section(response, "STORY ELEMENTS")
        if story_section:
            result["story_elements"] = story_section.strip()

        return result

    except Exception as e:
        logger.debug(f"Error parsing panel analysis: {e}")
        return {
            "panel_index": panel_index,
            "characters": [{"name": "Unknown", "description": "Parse error", "role": "unknown"}],
            "setting": "Parse error",
            "mood": "Unknown",
            "dialogue": [],
            "story_elements": "Could not parse analysis",
            "raw_analysis": response
        }


def _extract_section(text: str, section_name: str) -> str:
    """Extract a section from structured text."""
    pattern = rf"\*\*{section_name}:\*\*\s*(.*?)(?=\*\*[A-Z\s]+:\*\*|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_field(text: str, field_name: str) -> str:
    """Extract a field from text."""
    pattern = rf"{field_name}:\s*([^\n]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _parse_characters(characters_text: str) -> List[Dict[str, str]]:
    """Parse character information from text with enhanced parsing for both structured and unstructured responses."""
    characters = []

    # First try structured parsing (looking for NAME: indicators)
    char_entries = re.split(r"(?=- NAME:|NAME:)", characters_text)

    structured_found = False
    for entry in char_entries:
        if not entry.strip():
            continue

        # Check if this entry has structured fields
        if "NAME:" in entry or "DESCRIPTION:" in entry:
            structured_found = True
            char_data = {
                "name": _extract_field(entry, "NAME") or "Unknown",
                "description": _extract_field(entry, "DESCRIPTION") or "",
                "action": _extract_field(entry, "ACTION") or "",
                "dialogue": _extract_field(entry, "DIALOGUE") or "",
                "role": _extract_field(entry, "ROLE") or "unknown",
                "emotions": _extract_field(entry, "EMOTIONS") or ""
            }

            # Combine description, action, and emotions for full description
            full_description_parts = [char_data["description"], char_data["action"], char_data["emotions"]]
            char_data["full_description"] = ". ".join([part for part in full_description_parts if part])

            characters.append(char_data)

    # If no structured format found, try to extract from natural language
    if not structured_found or not characters:
        characters = _parse_characters_from_natural_language(characters_text)

    return characters if characters else [{"name": "Unknown", "description": "No characters identified", "role": "unknown"}]


def _parse_characters_from_natural_language(text: str) -> List[Dict[str, str]]:
    """Extract character information from natural language descriptions."""
    characters = []

    # Look for character mentions and descriptions
    sentences = text.split('.')

    # Enhanced character indicators - more comprehensive patterns
    character_indicators = [
        # Named characters
        r'(?:cat|tiger|animal|creature)\s+named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'(?:character|figure|person|man|woman|being)\s+(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'politician\s+named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # General "named X" pattern
        # Dialogue mentions
        r'mentioning\s+["\']([A-Z][a-z]+)["\']',
        r'dialogue.*["\']([A-Z][a-z]+)["\']',
        r'["\']([A-Z][a-z]+)["\'].*dialogue',
        r'speaks?\s+(?:to\s+)?([A-Z][a-z]+)',
        r'(?:says?|calls?)\s+["\']([A-Z][a-z]+)["\']',
        # Context clues
        r'([A-Z][a-z]+)\s*[,:]',  # Names followed by comma or colon
        r'([A-Z][a-z]+)(?:\s+is|appears|stands|sits|speaks|says)',
        r'(?:bearded\s+)?(?:figure|character|person|man|woman)',  # Generic character refs
        r'(?:smaller\s+)?figure\s+with\s+(?:horns|beard)',  # Descriptive characters
    ]

    found_names = set()
    character_descriptions = {}

    # Extract potential character names with case-insensitive matching
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        for pattern in character_indicators:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1]
                if match and len(match) > 1 and match.lower() not in ['the', 'and', 'with', 'that', 'this']:
                    name = match.title()
                    found_names.add(name)
                    # Store the sentence for context
                    if name not in character_descriptions:
                        character_descriptions[name] = []
                    character_descriptions[name].append(sentence.strip())

    # Also look for explicit character descriptions even without names
    character_desc_patterns = [
        r'(bearded\s+figure)',
        r'(smaller\s+figure\s+with\s+horns)',
        r'(figure\s+with\s+horns)',
        r'(demon|devil)',
        r'(character\s+(?:is\s+)?(?:sitting|standing|speaking|reading))',
    ]

    generic_characters = []
    for sentence in sentences:
        for pattern in character_desc_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match:
                    generic_characters.append({
                        'description': match.lower(),
                        'context': sentence.strip()
                    })

    # Create character entries from found names
    if found_names:
        for name in found_names:
            # Get all context for this character
            context_sentences = character_descriptions.get(name, [])
            full_description = ". ".join(context_sentences[:3])  # Limit to avoid too long descriptions

            # Determine role based on prominence and context
            role = "unknown"
            if len(context_sentences) >= 3 or any('primary' in s.lower() or 'main' in s.lower() for s in context_sentences):
                role = "primary"
            elif len(context_sentences) >= 2 or any('important' in s.lower() or 'speaks' in s.lower() for s in context_sentences):
                role = "secondary"
            else:
                role = "extra"

            # Extract dialogue if mentioned
            dialogue = ""
            for sentence in context_sentences:
                dialogue_match = re.search(r'["\']([^"\']+)["\']', sentence)
                if dialogue_match:
                    dialogue = dialogue_match.group(1)
                    break

            characters.append({
                "name": name,
                "description": full_description,
                "full_description": full_description,
                "action": "",
                "dialogue": dialogue,
                "role": role,
                "emotions": ""
            })

    # Add generic character descriptions
    if generic_characters:
        for i, generic_char in enumerate(generic_characters[:2]):  # Limit to 2 generic characters
            char_name = f"Character_{i+1}" if len(generic_characters) > 1 else "Character"
            characters.append({
                "name": char_name,
                "description": generic_char['context'],
                "full_description": generic_char['context'],
                "action": "",
                "dialogue": "",
                "role": "unknown",
                "emotions": ""
            })

    # If no characters found at all, create a single generic entry
    if not characters:
        characters.append({
            "name": "Character",
            "description": text.strip(),
            "full_description": text.strip(),
            "action": "",
            "dialogue": "",
            "role": "unknown",
            "emotions": ""
        })

    return characters
def _parse_dialogue(dialogue_text: str) -> List[Dict[str, str]]:
    """Parse dialogue from text."""
    dialogue_entries = []

    # Look for dialogue patterns
    lines = dialogue_text.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("-"):
            # Try to identify speaker and text
            if ":" in line:
                parts = line.split(":", 1)
                dialogue_entries.append({
                    "speaker": parts[0].strip(),
                    "text": parts[1].strip()
                })
            elif line:
                dialogue_entries.append({
                    "speaker": "Unknown",
                    "text": line
                })

    return dialogue_entries


def _create_panel_summary(analysis: Dict[str, Any]) -> str:
    """Create a brief summary for context in next panels."""
    try:
        chars = analysis.get("characters", [])
        char_names = [char.get("name", "Unknown") for char in chars if char.get("name") != "Unknown"]

        summary_parts = []
        if char_names:
            summary_parts.append(f"Characters: {', '.join(char_names[:3])}")

        setting = analysis.get("setting", "")
        if setting:
            summary_parts.append(f"Setting: {setting[:50]}...")

        return "; ".join(summary_parts) if summary_parts else "Panel analyzed"

    except Exception:
        return "Panel analyzed"

