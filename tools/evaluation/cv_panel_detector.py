"""
Computer Vision Panel Detection

This module implements panel detection using traditional computer vision techniques:
- Edge detection to find panel borders
- Contour analysis to identify rectangular regions
- Heuristic filtering to remove noise and false positives

This approach is more interpretable than deep learning and can work well
for comics with clear panel borders.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import json
import sys

# Standardized project imports
import config
from config import setup_logger, OUT_DIR

logger = setup_logger("cv_panel_detector")


class CVPanelDetector:
    """Computer Vision-based panel detector using edge detection and contours."""

    def __init__(self,
                 min_panel_area: int = 10000,
                 max_panel_area_ratio: float = 0.8,
                 min_aspect_ratio: float = 0.2,
                 max_aspect_ratio: float = 5.0):
        """
        Initialize the CV panel detector.

        Args:
            min_panel_area: Minimum area in pixels for a valid panel
            max_panel_area_ratio: Maximum area as ratio of total image area
            min_aspect_ratio: Minimum width/height ratio
            max_aspect_ratio: Maximum width/height ratio
        """
        self.min_panel_area = min_panel_area
        self.max_panel_area_ratio = max_panel_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def detect_panels(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect panels in a comic page using computer vision.

        Args:
            image_path: Path to the comic page image

        Returns:
            List of detected panels with bounding boxes and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        height, width = image.shape[:2]
        image_area = width * height

        print(f"🔍 Analyzing image: {Path(image_path).name}")
        print(f"📐 Image size: {width}x{height}")

        # Step 1: Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Edge detection with multiple methods
        panels_canny = self._detect_with_canny(gray, image_area)
        panels_morphology = self._detect_with_morphology(gray, image_area)
        panels_contours = self._detect_with_contours(gray, image_area)

        # Step 3: Combine and deduplicate results
        all_panels = panels_canny + panels_morphology + panels_contours
        unique_panels = self._remove_duplicates(all_panels)

        # Step 4: Post-processing and validation
        valid_panels = self._validate_panels(unique_panels, width, height)

        print(f"✓ Found {len(valid_panels)} valid panels")
        return valid_panels

    def _detect_with_canny(self, gray: np.ndarray, image_area: int) -> List[Dict[str, Any]]:
        """Detect panels using Canny edge detection."""
        print("  🔍 Method 1: Canny edge detection")

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        panels = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if it's roughly rectangular (4-8 vertices)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                if area > self.min_panel_area:
                    panels.append({
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'confidence': 0.8,
                        'method': 'canny',
                        'vertices': len(approx)
                    })

        print(f"    Found {len(panels)} candidates")
        return panels

    def _detect_with_morphology(self, gray: np.ndarray, image_area: int) -> List[Dict[str, Any]]:
        """Detect panels using morphological operations."""
        print("  🔍 Method 2: Morphological operations")

        # Create binary image using adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        # Invert so that lines are white
        binary = cv2.bitwise_not(binary)

        # Create kernels for detecting horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines
        panel_borders = cv2.bitwise_or(horizontal_lines, vertical_lines)

        # Find contours in the combined border image
        contours, _ = cv2.findContours(panel_borders, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        panels = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area > self.min_panel_area:
                panels.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'confidence': 0.7,
                    'method': 'morphology'
                })

        print(f"    Found {len(panels)} candidates")
        return panels

    def _detect_with_contours(self, gray: np.ndarray, image_area: int) -> List[Dict[str, Any]]:
        """Detect panels using contour analysis on thresholded image."""
        print("  🔍 Method 3: Contour analysis")

        # Multiple threshold values to catch different panel types
        thresholds = [127, 100, 150, 80, 180]
        all_panels = []

        for thresh_val in thresholds:
            # Apply threshold
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_panel_area:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate how rectangular the contour is
                    rect_area = w * h
                    rectangularity = area / rect_area if rect_area > 0 else 0

                    if rectangularity > 0.7:  # Fairly rectangular
                        all_panels.append({
                            'bbox': [x, y, x + w, y + h],
                            'area': area,
                            'confidence': 0.6 * rectangularity,
                            'method': 'contour',
                            'threshold': thresh_val,
                            'rectangularity': rectangularity
                        })

        print(f"    Found {len(all_panels)} candidates")
        return all_panels

    def _remove_duplicates(self, panels: List[Dict[str, Any]], iou_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Remove duplicate panels based on IoU overlap."""
        if not panels:
            return panels

        print(f"  🔄 Removing duplicates from {len(panels)} panels...")

        # Sort by confidence
        panels_sorted = sorted(panels, key=lambda p: p['confidence'], reverse=True)
        unique_panels = []

        for panel in panels_sorted:
            is_duplicate = False

            for existing in unique_panels:
                iou = self._calculate_iou(panel['bbox'], existing['bbox'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_panels.append(panel)

        print(f"    Kept {len(unique_panels)} unique panels")
        return unique_panels

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _validate_panels(self, panels: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        """Validate and filter panels based on size and aspect ratio constraints."""
        image_area = width * height
        valid_panels = []

        print(f"  ✅ Validating {len(panels)} panels...")

        for i, panel in enumerate(panels):
            x1, y1, x2, y2 = panel['bbox']
            panel_width = x2 - x1
            panel_height = y2 - y1
            area = panel['area']

            # Check area constraints
            area_ratio = area / image_area
            if area_ratio > self.max_panel_area_ratio:
                print(f"    Panel {i+1}: Rejected - too large ({area_ratio:.2%})")
                continue

            if area < self.min_panel_area:
                print(f"    Panel {i+1}: Rejected - too small ({area} pixels)")
                continue

            # Check aspect ratio
            aspect_ratio = panel_width / panel_height if panel_height > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                print(f"    Panel {i+1}: Rejected - bad aspect ratio ({aspect_ratio:.2f})")
                continue

            # Check if panel is within image bounds
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                print(f"    Panel {i+1}: Rejected - out of bounds")
                continue

            # Add additional metadata
            panel.update({
                'panel_number': len(valid_panels) + 1,
                'width': panel_width,
                'height': panel_height,
                'area_ratio': area_ratio,
                'aspect_ratio': aspect_ratio
            })

            valid_panels.append(panel)
            print(f"    Panel {i+1}: ✓ Valid ({panel_width}x{panel_height}, {area_ratio:.2%})")

        return valid_panels

    def create_visualization(self, image_path: str, panels: List[Dict[str, Any]],
                           output_path: str = None) -> str:
        """Create a visualization of detected panels."""
        image = cv2.imread(image_path)

        # Colors for different methods
        method_colors = {
            'canny': (0, 255, 0),      # Green
            'morphology': (255, 0, 0),  # Blue
            'contour': (0, 0, 255),     # Red
            'yolo': (255, 255, 0)       # Cyan
        }

        for panel in panels:
            x1, y1, x2, y2 = panel['bbox']
            method = panel.get('method', 'unknown')
            color = method_colors.get(method, (255, 255, 255))

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            # Draw label
            label = f"P{panel['panel_number']} ({method})"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add legend
        y_offset = 30
        for method, color in method_colors.items():
            cv2.putText(image, method, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        # Save visualization
        if output_path is None:
            # Use proper output directory structure
            output_dir = OUT_DIR / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"cv_panel_detection_{Path(image_path).stem}.png")

        cv2.imwrite(output_path, image)
        print(f"💾 Saved visualization: {output_path}")
        return output_path


def main():
    """Test the CV panel detector."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python cv_panel_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detector = CVPanelDetector()

    try:
        panels = detector.detect_panels(image_path)

        print(f"\n📊 Detection Results:")
        print(f"  Total panels found: {len(panels)}")

        for panel in panels:
            print(f"  Panel {panel['panel_number']}: {panel['method']} method")
            print(f"    BBox: {panel['bbox']}")
            print(f"    Size: {panel['width']}x{panel['height']}")
            print(f"    Confidence: {panel['confidence']:.3f}")

        # Create visualization
        viz_path = detector.create_visualization(image_path, panels)
        print(f"\n✅ Complete! Check visualization: {viz_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
