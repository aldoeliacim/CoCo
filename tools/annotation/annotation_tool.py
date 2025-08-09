"""
Interactive Panel Annotation Tool

This tool allows manual annotation of comic panels to create ground truth data
for training and evaluation. It shows the comic page and lets you draw rectangles
around actual panels.

Usage:
    python annotation_tool.py <image_path>

Controls:
    - Click and drag to draw panel rectangles
    - Press 's' to save annotations
    - Press 'r' to reset/clear all annotations
    - Press 'u' to undo last annotation
    - Press 'q' to quit
    - Press 'h' to show help
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Standardized project imports
import config
from config import setup_logger

logger = setup_logger("annotation_tool")


class PanelAnnotationTool:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Scale image to fit screen if too large
        self.scale_factor = 1.0
        height, width = self.original_image.shape[:2]
        max_height, max_width = 1000, 1400

        if height > max_height or width > max_width:
            scale_h = max_height / height if height > max_height else 1.0
            scale_w = max_width / width if width > max_width else 1.0
            self.scale_factor = min(scale_h, scale_w)

            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.display_image = cv2.resize(self.original_image, (new_width, new_height))
        else:
            self.display_image = self.original_image.copy()

        # Annotation state
        self.panels = []  # List of panel rectangles [(x1, y1, x2, y2), ...]
        self.current_panel = None  # Currently being drawn panel
        self.drawing = False
        self.start_point = None

        # Colors
        self.panel_color = (0, 255, 0)  # Green for panels
        self.current_color = (0, 0, 255)  # Red for current drawing

        # Window setup
        self.window_name = f"Panel Annotation: {self.image_path.name}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"📝 Annotating: {self.image_path.name}")
        print(f"📐 Image size: {width}x{height}")
        if self.scale_factor < 1.0:
            print(f"🔍 Display scaled by {self.scale_factor:.2f}")
        self.show_help()

    def show_help(self):
        """Display help information."""
        print("\n📖 Controls:")
        print("  - Click and drag: Draw panel rectangle")
        print("  - 's': Save annotations")
        print("  - 'r': Reset/clear all annotations")
        print("  - 'u': Undo last annotation")
        print("  - 'q': Quit")
        print("  - 'h': Show this help")
        print()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing panels."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_panel = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                self.current_panel = (*self.start_point, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                end_point = (x, y)

                # Convert to original image coordinates
                orig_start = self.display_to_original(self.start_point)
                orig_end = self.display_to_original(end_point)

                # Ensure proper rectangle format (top-left, bottom-right)
                x1, y1 = min(orig_start[0], orig_end[0]), min(orig_start[1], orig_end[1])
                x2, y2 = max(orig_start[0], orig_end[0]), max(orig_start[1], orig_end[1])

                # Only add if rectangle is large enough
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.panels.append((x1, y1, x2, y2))
                    print(f"✓ Panel {len(self.panels)}: ({x1}, {y1}) → ({x2}, {y2})")

                self.drawing = False
                self.start_point = None
                self.current_panel = None

    def display_to_original(self, point):
        """Convert display coordinates to original image coordinates."""
        x, y = point
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        return (orig_x, orig_y)

    def original_to_display(self, point):
        """Convert original coordinates to display coordinates."""
        x, y = point
        disp_x = int(x * self.scale_factor)
        disp_y = int(y * self.scale_factor)
        return (disp_x, disp_y)

    def draw_annotations(self):
        """Draw all annotations on the display image."""
        image = self.display_image.copy()

        # Draw existing panels
        for i, (x1, y1, x2, y2) in enumerate(self.panels):
            # Convert to display coordinates
            disp_p1 = self.original_to_display((x1, y1))
            disp_p2 = self.original_to_display((x2, y2))

            # Draw rectangle
            cv2.rectangle(image, disp_p1, disp_p2, self.panel_color, 2)

            # Draw panel number
            label = f"Panel {i+1}"
            label_pos = (disp_p1[0], disp_p1[1] - 10 if disp_p1[1] > 20 else disp_p1[1] + 20)
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.panel_color, 2)

        # Draw current panel being drawn
        if self.current_panel:
            start_disp = self.start_point
            end_disp = (self.current_panel[2], self.current_panel[3])
            cv2.rectangle(image, start_disp, end_disp, self.current_color, 2)

        # Add status text
        status_text = f"Panels: {len(self.panels)} | Press 'h' for help"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return image

    def save_annotations(self):
        """Save annotations to JSON file."""
        # Create annotations directory
        annotations_dir = Path("experiments/panel_detection/training_data/annotations")
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Prepare annotation data
        annotation_data = {
            "image_path": str(self.image_path),
            "image_name": self.image_path.name,
            "image_size": {
                "width": self.original_image.shape[1],
                "height": self.original_image.shape[0]
            },
            "panels": [
                {
                    "id": i + 1,
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1)
                }
                for i, (x1, y1, x2, y2) in enumerate(self.panels)
            ],
            "panel_count": len(self.panels),
            "annotation_date": datetime.now().isoformat(),
            "annotator": "manual"
        }

        # Save to file
        output_file = annotations_dir / f"{self.image_path.stem}_annotations.json"
        with open(output_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        print(f"💾 Saved {len(self.panels)} panels to: {output_file}")
        return output_file

    def reset_annotations(self):
        """Clear all annotations."""
        self.panels.clear()
        print("🗑️  Cleared all annotations")

    def undo_last(self):
        """Remove the last annotation."""
        if self.panels:
            removed = self.panels.pop()
            print(f"↶ Undid panel: {removed}")
        else:
            print("❌ No annotations to undo")

    def run(self):
        """Run the annotation tool main loop."""
        while True:
            # Draw and display image
            display_img = self.draw_annotations()
            cv2.imshow(self.window_name, display_img)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.panels:
                    self.save_annotations()
                else:
                    print("❌ No panels to save")
            elif key == ord('r'):
                self.reset_annotations()
            elif key == ord('u'):
                self.undo_last()
            elif key == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()
        print(f"👋 Annotation complete. Final panel count: {len(self.panels)}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python annotation_tool.py <image_path>")
        print("Example: python annotation_tool.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        tool = PanelAnnotationTool(image_path)
        tool.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
