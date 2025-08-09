"""
XML Validation and Correction Utility
"""

import xml.etree.ElementTree as ET
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standardized project imports
import config
from config import setup_logger

logger = setup_logger("xml_validator")

class XMLValidator:
    def __init__(self, model_name: str = None, enable_llm_correction: bool = False):
        """
        Initialize the XML Validator and Corrector.

        Args:
            model_name (str): The name of the LLM to use for correction (unused - LLM disabled).
            enable_llm_correction (bool): Whether to enable LLM-based XML correction (disabled by default).
        """
        # LLM correction is disabled - only fast ElementTree validation is used
        self.model_name = model_name
        self.enable_llm_correction = enable_llm_correction
        self.tokenizer = None
        self.model = None
        self.model_available = False

        if self.enable_llm_correction:
            self._load_model()

    def _load_model(self):
        """Load the LLM for XML correction."""
        try:
            logger.debug(f"🔄 Loading XML correction model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.model_available = True
            logger.debug("✅ XML correction model loaded successfully.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load XML correction model: {e}")
            logger.debug("📋 Falling back to basic XML validation without LLM correction")
            self.tokenizer = None
            self.model = None
            self.model_available = False

    def validate_and_fix(self, xml_string: str) -> tuple[bool, str, Optional[str]]:
        """
        Validate XML and attempt to fix if invalid, using the most efficient approach.

        1. First tries simple validation (fast)
        2. If invalid, tries basic fixes (fast)
        3. Only uses LLM as last resort (slow)

        Args:
            xml_string (str): The XML content as a string.

        Returns:
            tuple[bool, str, Optional[str]]: (is_valid, final_xml_or_error, correction_method_used)
                - is_valid: True if XML is valid after processing
                - final_xml_or_error: Valid XML string or error message
                - correction_method_used: None/"basic"/"llm" indicating what method was used
        """
        # Step 1: Quick validation check (microseconds)
        is_valid, error_msg = self.validate(xml_string)
        if is_valid:
            logger.info("✅ XML is already valid - no correction needed")
            return True, xml_string, None

        logger.debug(f"⚠️ XML validation failed: {error_msg}")

        # Step 2: Try basic fixes first (milliseconds)
        logger.debug("🔧 Attempting fast basic XML corrections...")
        basic_fixed = self._basic_xml_fix(xml_string)
        if basic_fixed:
            is_valid, _ = self.validate(basic_fixed)
            if is_valid:
                logger.info("✅ XML corrected with basic fixes")
                return True, basic_fixed, "basic"
            else:
                logger.info("⚠️ Basic fixes insufficient")

        # Step 3: LLM correction as last resort (seconds/minutes)
        if self.enable_llm_correction and self.model_available:
            logger.info("🤖 Attempting LLM-based XML correction (this may take time)...")
            llm_fixed = self._llm_fix_xml(xml_string)
            if llm_fixed:
                is_valid, _ = self.validate(llm_fixed)
                if is_valid:
                    logger.info("✅ XML corrected with LLM")
                    return True, llm_fixed, "llm"

        # All correction attempts failed
        logger.error("❌ All XML correction attempts failed")
        return False, f"Could not fix XML: {error_msg}", "failed"

    def validate(self, xml_string: str) -> tuple[bool, str]:
        """
        Validate an XML string using fast built-in parser with enhanced structure validation.

        Args:
            xml_string (str): The XML content as a string.

        Returns:
            tuple[bool, str]: (is_valid, error_message) - True/False and error description if any.
        """
        try:
            # Parse XML structure
            root = ET.fromstring(xml_string)

            # Enhanced validation for comic analysis XML structure
            if root.tag == "comic_page_analysis":
                # Validate required attributes
                required_attrs = ["image_file", "page_type", "analysis_date", "total_panels"]
                missing_attrs = [attr for attr in required_attrs if attr not in root.attrib]
                if missing_attrs:
                    return False, f"Missing required attributes: {missing_attrs}"

                # Validate structure elements
                expected_elements = ["character_summary", "panels", "processing_metadata"]
                found_elements = [child.tag for child in root]

                # Check if at least basic elements are present
                if "character_summary" not in found_elements:
                    logger.debug("⚠️ character_summary element missing but XML is structurally valid")

                # Validate panel structure if comic page
                page_type = root.get("page_type", "")
                if page_type == "comic":
                    panels_elem = root.find("panels")
                    if panels_elem is not None:
                        panel_count = len(panels_elem.findall("panel"))
                        declared_count = int(root.get("total_panels", "0"))
                        if panel_count != declared_count:
                            logger.debug(f"⚠️ Panel count mismatch: declared={declared_count}, found={panel_count}")

                logger.debug("✓ Enhanced XML structure validation passed")

            return True, ""

        except ET.ParseError as e:
            error_msg = f"XML validation failed: {e}"
            logger.debug(f"❌ XML parsing error: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"XML validation error: {e}"
            logger.debug(f"❌ XML validation error: {error_msg}")
            return False, error_msg

    def get_xml_schema_info(self) -> str:
        """
        Get information about the expected XML schema for comic analysis.

        Returns:
            str: XML schema documentation
        """
        schema_info = """
        CoCo Comic Analysis XML Schema (v2.0)

        Root Element: <comic_page_analysis>
        Required Attributes:
        - version: Schema version (e.g., "2.0")
        - image_file: Image filename
        - image_path: Full path to image
        - page_type: cover|comic|advertisement|text|illustration
        - analysis_date: ISO format timestamp
        - total_panels: Number of detected panels
        - processing_pipeline: Analysis method used

        Structure:
        <comic_page_analysis>
          <image_metadata width="..." height="..." format="..." mode="..." file_size="..." />
          <character_summary total_unique="..." primary_count="..." secondary_count="..." extra_count="...">
            <primary_chars count="...">
              <character name="..." appearances="..." dialogue_lines="..." />
            </primary_chars>
            <secondary_chars count="...">...</secondary_chars>
            <extra_chars count="...">...</extra_chars>
          </character_summary>
          <panels detection_method="..." analysis_method="...">
            <panel number="..." sequence_order="..." bbox_x1="..." bbox_y1="..." bbox_x2="..." bbox_y2="..." area="..." detection_confidence="...">
              <characters count="...">
                <character name="..." role="..." dialogue="..." />
              </characters>
              <content_analysis>
                <setting>...</setting>
                <mood>...</mood>
                <story_elements>...</story_elements>
                <dialogue>...</dialogue>
                <action>...</action>
                <visual_elements>...</visual_elements>
              </content_analysis>
              <narrative_analysis narrative_function="..." pacing="..." emotional_tone="..." />
            </panel>
          </panels>
          <processing_metadata pipeline_version="..." yolo_model="..." visionthink_model="..." analysis_type="..." />
        </comic_page_analysis>
        """
        return schema_info.strip()

    def fix_xml(self, invalid_xml: str) -> Optional[str]:
        """
        Legacy method - use validate_and_fix() for better performance.
        Attempt to fix an invalid XML string using the optimal approach.

        Args:
            invalid_xml (str): The invalid XML string.

        Returns:
            Optional[str]: The corrected XML string, or None if correction fails.
        """
        is_valid, result, method = self.validate_and_fix(invalid_xml)
        return result if is_valid else None

    def _llm_fix_xml(self, invalid_xml: str) -> Optional[str]:
        """
        Attempt to fix XML using LLM (expensive operation).

        Args:
            invalid_xml (str): The invalid XML string.

        Returns:
            Optional[str]: The corrected XML string, or None if correction fails.
        """
        if not self.enable_llm_correction or not self.model_available:
            logger.warning("LLM correction not available")
            return None

        prompt = f"""
        The following XML is malformed. Please fix it so it is well-formed and valid.
        Return ONLY the corrected XML content, without any explanations or surrounding text.

        Invalid XML:
        ```xml
        {invalid_xml}
        ```

        Corrected XML:
        """

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=len(invalid_xml) + 512, do_sample=False)
            corrected_xml = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the XML part from the model's response
            import re
            xml_match = re.search(r'```xml\n(.*?)\n```', corrected_xml, re.DOTALL)
            if xml_match:
                corrected_xml = xml_match.group(1).strip()
            else:
                # Fallback if the model doesn't use markdown
                corrected_xml = corrected_xml.split("Corrected XML:")[1].strip()

            return corrected_xml

        except Exception as e:
            logger.error(f"❌ Error during LLM-based XML correction: {e}")
            return None

    def _basic_xml_fix(self, invalid_xml: str) -> Optional[str]:
        """
        Attempt basic XML fixes without using an LLM (fast, rule-based corrections).

        Args:
            invalid_xml (str): The invalid XML string.

        Returns:
            Optional[str]: The corrected XML string, or None if correction fails.
        """
        import re

        try:
            corrected = invalid_xml.strip()

            # Common XML fixes that can be done quickly:

            # 1. Fix unescaped ampersands (but not already escaped ones)
            corrected = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', corrected)

            # 2. Fix unescaped angle brackets in text content
            # This is tricky - we only want to escape < and > that aren't part of tags

            # 3. Fix unclosed tags (basic heuristic for simple cases)
            # Look for opening tags without corresponding closing tags
            opening_tags = re.findall(r'<(\w+)[^>]*(?<!/)>', corrected)
            closing_tags = re.findall(r'</(\w+)>', corrected)

            # Simple case: if we have one opening tag without closing, add it at the end
            unclosed = []
            for tag in opening_tags:
                if tag not in closing_tags:
                    unclosed.append(tag)

            # Add missing closing tags at the end (simple heuristic)
            for tag in reversed(unclosed):
                corrected += f'</{tag}>'

            # 4. Fix missing XML declaration if needed
            if not corrected.strip().startswith('<?xml'):
                corrected = '<?xml version="1.0" encoding="UTF-8"?>\n' + corrected

            # 5. Basic whitespace cleanup
            corrected = re.sub(r'>\s+<', '><', corrected)  # Remove whitespace between tags

            logger.info("🔧 Applied basic XML fixes (character escaping, tag closing, declaration)")

            return corrected

        except Exception as e:
            logger.error(f"❌ Error during basic XML correction: {e}")
            return None

            # Validate the basic fix
            is_valid, error_msg = self.validate(corrected)
            if is_valid:
                logger.info("✅ Basic XML fix successful")
                return corrected
            else:
                logger.warning(f"⚠️ Basic XML fix insufficient: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"❌ Error during basic XML correction: {e}")
            return None

if __name__ == '__main__':
    import re
    from logger_setup import setup_logger

    # Setup logger for testing
    log_dir = Path("./logs_test")
    setup_logger(log_dir, "XMLValidatorTest")

    validator = XMLValidator()

    # --- Test Case 1: Valid XML ---
    valid_xml = "<root><child>Hello</child></root>"
    is_valid, error_msg = validator.validate(valid_xml)
    print(f"Testing valid XML... Valid: {is_valid}")

    # --- Test Case 2: Invalid XML (unclosed tag) ---
    invalid_xml = "<root><child>Hello</root>"
    is_valid, error_msg = validator.validate(invalid_xml)
    print(f"\nTesting invalid XML... Valid: {is_valid}, Error: {error_msg}")

    if validator.model:
        fixed_xml = validator.fix_xml(invalid_xml)
        if fixed_xml:
            print(f"Corrected XML:\n{fixed_xml}")
        else:
            print("Failed to correct XML.")
    else:
        print("Skipping XML correction test as model is not loaded.")

    # Clean up
    import shutil
    shutil.rmtree(log_dir)
    print("\nCleaned up test directory.")
