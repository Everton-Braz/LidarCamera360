#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag Marker & Calibration Board Generator (PDF & PNG)
=========================================================
Generates print-ready vector PDF files (A4 1:1 exact scale) and high-res PNGs
for AprilTag markers (family tag36h11) and dual-modality LiDAR-Camera boards.

PDF Output guarantees 100% exact physical print scale:
- Exact 150.0 mm (or custom) black outer tag dimension
- Exact 100.0 mm physical verification ruler
- Crosshair guidelines for 10 mm retroreflective tape strips
- Clean vector graphics (crisp lines at any printer resolution)
- Combined multi-page PDF workbook + individual PDF files

Usage:
  python generate_apriltags.py --count 12 --size 150 --out apriltags_to_print
  python generate_apriltags.py --ids 0,1,2,3,4,5 --size 150 --out apriltags_to_print
"""

import argparse
import os
import sys
import cv2
import numpy as np

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def get_tag_matrix(dictionary, tag_id):
    """
    Extracts the binary module matrix (e.g. 10x10 for tag36h11 with 1-bit border)
    from OpenCV AprilTag dictionary.
    """
    # Generate minimal 1-pixel-per-module image
    # For tag36h11: 6x6 data bits + 2 border bits = 8x8 (or 10x10 with 2-bit border)
    # Let's generate at 80x80 with borderBits=1 to extract the exact grid
    img = cv2.aruco.generateImageMarker(dictionary, int(tag_id), 80, borderBits=1)
    # Downsample to module grid
    # Find grid size
    grid_size = 8  # tag36h11 has 6 data + 2 border = 8x8 modules
    # Let's sample the center of each cell
    step = img.shape[0] / grid_size
    matrix = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for r in range(grid_size):
        for c in range(grid_size):
            y = int((r + 0.5) * step)
            x = int((c + 0.5) * step)
            matrix[r, c] = 1 if img[y, x] > 128 else 0
    return matrix


def draw_vector_scale_ruler(c, start_x, start_y, length_mm=100.0):
    """Draw a precise 100 mm vector measurement ruler on the ReportLab canvas."""
    c.saveState()
    
    # Background box
    c.setFillColor(HexColor("#F8F8FA"))
    c.setStrokeColor(HexColor("#B0B0B8"))
    c.setLineWidth(0.5)
    c.rect(start_x - 3 * mm, start_y - 7 * mm, (length_mm + 6) * mm, 14 * mm, fill=1, stroke=1)
    
    # Title
    c.setFont("Helvetica-Bold", 7.5)
    c.setFillColor(black)
    c.drawCentredString(start_x + (length_mm / 2.0) * mm, start_y + 3.5 * mm,
                        "100 mm (10 cm) Physical Scale Verification Ruler")
    
    # Baseline
    c.setStrokeColor(black)
    c.setLineWidth(0.8)
    c.line(start_x, start_y - 3 * mm, start_x + length_mm * mm, start_y - 3 * mm)
    
    # Ticks
    for m in range(int(length_mm) + 1):
        x = start_x + m * mm
        if m % 10 == 0:
            # 1 cm tick (5 mm high)
            c.setLineWidth(0.8)
            c.line(x, start_y - 3 * mm, x, start_y + 2 * mm)
            c.setFont("Helvetica-Bold", 6.5)
            c.drawCentredString(x, start_y - 6.0 * mm, str(m // 10))
        elif m % 5 == 0:
            # 5 mm tick (3.5 mm high)
            c.setLineWidth(0.6)
            c.line(x, start_y - 3 * mm, x, start_y + 0.5 * mm)
        else:
            # 1 mm tick (2 mm high)
            c.setLineWidth(0.4)
            c.line(x, start_y - 3 * mm, x, start_y - 1.0 * mm)
            
    c.setFont("Helvetica", 5.5)
    c.setFillColor(HexColor("#505050"))
    c.drawString(start_x + length_mm * mm + 1 * mm, start_y - 5.5 * mm, "cm")
    
    c.restoreState()


def draw_vector_calibration_board_pdf(c, dictionary, tag_id, nominal_tag_size_mm=150.0):
    """
    Renders an exact 1:1 A4 vector calibration sheet onto a ReportLab canvas.
    """
    page_w, page_h = A4  # 210 x 297 mm
    
    # Center coordinates of the tag
    center_x = page_w / 2.0
    center_y = page_h / 2.0 + 15.0 * mm  # Slightly above center to fit footer
    
    tag_w = nominal_tag_size_mm * mm
    tag_h = nominal_tag_size_mm * mm
    
    tag_x0 = center_x - tag_w / 2.0
    tag_y0 = center_y - tag_h / 2.0
    
    # 1. Header Banner
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(black)
    c.drawCentredString(page_w / 2.0, page_h - 18 * mm, f"LiDAR-Camera Calibration Target - Tag #{tag_id}")
    
    c.setFont("Helvetica", 8.5)
    c.setFillColor(HexColor("#444444"))
    c.drawCentredString(page_w / 2.0, page_h - 23 * mm,
                        f"Family: tag36h11 | Exact Physical Size: {nominal_tag_size_mm:.1f} mm x {nominal_tag_size_mm:.1f} mm")
    
    # Outer page border
    c.setStrokeColor(HexColor("#D0D0D5"))
    c.setLineWidth(0.8)
    c.rect(10 * mm, 10 * mm, page_w - 20 * mm, page_h - 20 * mm, fill=0, stroke=1)
    
    # 2. Retroreflective Tape / Crosshair Guidelines (10 mm wide)
    tape_w = 10.0 * mm
    tape_ext = 22.0 * mm  # Extension outside the tag
    
    c.setFillColor(HexColor("#EAEAEF"))
    c.setStrokeColor(HexColor("#90909A"))
    c.setLineWidth(0.6)
    
    # Top guide
    c.rect(center_x - tape_w / 2.0, tag_y0 + tag_h, tape_w, tape_ext, fill=1, stroke=1)
    # Bottom guide
    c.rect(center_x - tape_w / 2.0, tag_y0 - tape_ext, tape_w, tape_ext, fill=1, stroke=1)
    # Left guide
    c.rect(tag_x0 - tape_ext, center_y - tape_w / 2.0, tape_ext, tape_w, fill=1, stroke=1)
    # Right guide
    c.rect(tag_x0 + tag_w, center_y - tape_w / 2.0, tape_ext, tape_w, fill=1, stroke=1)
    
    # 3. Draw Vector AprilTag Modules
    matrix = get_tag_matrix(dictionary, tag_id)
    grid_n = matrix.shape[0]
    cell_size = tag_w / float(grid_n)
    
    c.setLineWidth(0)
    for r in range(grid_n):
        for col in range(grid_n):
            # ReportLab Y is from bottom up
            cell_x = tag_x0 + col * cell_size
            cell_y = tag_y0 + (grid_n - 1 - r) * cell_size
            
            if matrix[r, col] == 0:
                c.setFillColor(black)
            else:
                c.setFillColor(white)
            c.rect(cell_x, cell_y, cell_size + 0.05, cell_size + 0.05, fill=1, stroke=0)
            
    # Red center fiducial dot
    c.setStrokeColor(HexColor("#D02020"))
    c.setLineWidth(0.6)
    c.line(center_x - 3 * mm, center_y, center_x + 3 * mm, center_y)
    c.line(center_x, center_y - 3 * mm, center_x, center_y + 3 * mm)
    
    # 4. Draw Precision Physical Scale Ruler (100 mm)
    ruler_w = 100.0
    ruler_x = center_x - (ruler_w / 2.0) * mm
    ruler_y = tag_y0 - tape_ext - 16 * mm
    draw_vector_scale_ruler(c, ruler_x, ruler_y, length_mm=ruler_w)
    
    # 5. Instructions & Printing Guide Box
    box_y = 14 * mm
    box_h = 32 * mm
    c.setFillColor(HexColor("#F4F4F8"))
    c.setStrokeColor(HexColor("#C0C0C8"))
    c.setLineWidth(0.6)
    c.rect(14 * mm, box_y, page_w - 28 * mm, box_h, fill=1, stroke=1)
    
    c.setFont("Helvetica-Bold", 8.0)
    c.setFillColor(black)
    c.drawString(18 * mm, box_y + box_h - 6 * mm, "PRINTING & USAGE INSTRUCTIONS (EXACT 1:1 SCALE):")
    
    instructions = [
        "1. In your PDF reader / print dialog, select 'Actual Size' or '100% Scale' (Do NOT use 'Fit to Page').",
        f"2. Check with a real ruler: the black square must measure exactly {nominal_tag_size_mm:.0f} mm, and the ruler bar must be 10.0 cm.",
        "3. (Optional Dual-Modality): Stick 10 mm wide retroreflective tape onto the shaded crosshair extensions.",
        "4. Mount flat onto rigid backing (wall, foam board, cardboard) facing the scanner.",
        "5. Works with raw fisheye images (Insta360 X4 / Raven JMK7) and LiDAR point clouds.",
    ]
    
    c.setFont("Helvetica", 7.0)
    c.setFillColor(HexColor("#303030"))
    for i, line in enumerate(instructions):
        c.drawString(18 * mm, box_y + box_h - 11.5 * mm - i * 4.2 * mm, line)


def generate_png_board(dictionary, tag_id, nominal_tag_size_mm=150, board_w_px=2480, board_h_px=3508):
    """Generates 300 DPI A4 raster PNG version."""
    px_per_mm = board_w_px / 210.0
    canvas_img = np.full((board_h_px, board_w_px, 3), 255, dtype=np.uint8)
    
    cx, cy = board_w_px // 2, int(board_h_px // 2 - 40 * px_per_mm)
    tag_px = int(nominal_tag_size_mm * px_per_mm)
    
    tag_raw = cv2.aruco.generateImageMarker(dictionary, int(tag_id), tag_px, borderBits=1)
    tag_bgr = cv2.cvtColor(tag_raw, cv2.COLOR_GRAY2BGR)
    
    x1, y1 = cx - tag_px // 2, cy - tag_px // 2
    canvas_img[y1:y1 + tag_px, x1:x1 + tag_px] = tag_bgr
    
    tape_w = int(10 * px_per_mm)
    tape_len = int(22 * px_per_mm)
    
    cv2.rectangle(canvas_img, (cx - tape_w // 2, y1 - tape_len), (cx + tape_w // 2, y1), (230, 230, 235), -1)
    cv2.rectangle(canvas_img, (cx - tape_w // 2, y1 + tag_px), (cx + tape_w // 2, y1 + tag_px + tape_len), (230, 230, 235), -1)
    cv2.rectangle(canvas_img, (x1 - tape_len, cy - tape_w // 2), (x1, cy + tape_w // 2), (230, 230, 235), -1)
    cv2.rectangle(canvas_img, (x1 + tag_px, cy - tape_w // 2), (x1 + tag_px + tape_len, cy + tape_w // 2), (230, 230, 235), -1)
    
    cv2.putText(canvas_img, f"LiDAR-Camera Calibration Target - Tag #{tag_id}", (120, 150),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas_img, f"Family: tag36h11 | Exact Physical Size: {nominal_tag_size_mm} mm x {nominal_tag_size_mm} mm",
                (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
    
    return canvas_img


def main():
    parser = argparse.ArgumentParser(description="Generate Print-Ready AprilTag Calibration Boards in PDF and PNG.")
    parser.add_argument("--out", "-o", default="apriltags_to_print",
                        help="Output directory (default: apriltags_to_print)")
    parser.add_argument("--count", "-n", type=int, default=12,
                        help="Number of sequential tags to generate starting from 0 (default: 12)")
    parser.add_argument("--ids", type=str, default="",
                        help="Comma-separated specific tag IDs (e.g. '0,1,2,3,4,5')")
    parser.add_argument("--size", "-s", type=float, default=150.0,
                        help="Exact physical size in mm of the black tag square (default: 150.0 mm)")
    parser.add_argument("--no-png", action="store_true", help="Skip PNG generation (PDF only)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    if args.ids.strip():
        tag_ids = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    else:
        tag_ids = list(range(args.count))

    print(f"[*] Generating {len(tag_ids)} Print-Ready AprilTag Targets in PDF & PNG...")
    print(f"[*] Physical Tag Dimension: {args.size:.1f} mm x {args.size:.1f} mm (A4 1:1 Scale)")
    print(f"[*] Output directory: {os.path.abspath(args.out)}")

    # 1. Generate Combined Multi-Page PDF
    multi_pdf_path = os.path.join(args.out, f"apriltag_calibration_targets_all_{int(args.size)}mm_A4.pdf")
    c_multi = canvas.Canvas(multi_pdf_path, pagesize=A4)
    
    for tag_id in tag_ids:
        # Draw on multi-page PDF
        draw_vector_calibration_board_pdf(c_multi, dictionary, tag_id, nominal_tag_size_mm=args.size)
        c_multi.showPage()
        
        # 2. Individual PDF
        single_pdf_path = os.path.join(args.out, f"calibration_board_tag36h11_id{tag_id:02d}_{int(args.size)}mm.pdf")
        c_single = canvas.Canvas(single_pdf_path, pagesize=A4)
        draw_vector_calibration_board_pdf(c_single, dictionary, tag_id, nominal_tag_size_mm=args.size)
        c_single.save()
        
        # 3. Individual PNG (optional)
        if not args.no_png:
            png_img = generate_png_board(dictionary, tag_id, nominal_tag_size_mm=args.size)
            png_path = os.path.join(args.out, f"calibration_board_tag36h11_id{tag_id:02d}_{int(args.size)}mm.png")
            cv2.imwrite(png_path, png_img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
        print(f"  [+] Generated Tag #{tag_id:02d}: PDF & PNG")

    c_multi.save()
    print(f"\n[+] Master Multi-Page PDF: {os.path.basename(multi_pdf_path)}")

    # Write Print README
    readme_path = os.path.join(args.out, "README_PRINT.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# AprilTag Printable Targets (PDF & PNG) - 1:1 Exact Physical Scale

Generated on: {len(tag_ids)} targets (IDs {tag_ids[0]} to {tag_ids[-1]})
- **Nominal Tag Size**: {args.size:.1f} mm x {args.size:.1f} mm (Outer black boundary)
- **Paper Size**: ISO A4 (210 mm x 297 mm)

## Recommended Printable File
**`{os.path.basename(multi_pdf_path)}`**
Contains all {len(tag_ids)} targets in one multi-page PDF document.

## Printing Instructions
1. Open the `.pdf` file in Adobe Acrobat, Chrome, or your PDF reader.
2. Select **Actual Size** or **100% Scale** (Do NOT choose *"Fit to Page"* or *"Shrink to Printable Area"*).
3. Print on standard A4 paper.
4. Measure the printed verification ruler at the bottom: the 100 mm bar must measure exactly 10.0 cm.
5. Fix boards to rigid flat surfaces without curling.
""")

    print(f"[DONE] Successfully created vector PDF targets in '{args.out}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
