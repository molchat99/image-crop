import os
import re
from rembg import remove, new_session
from PIL import Image, ImageDraw, ImageFilter
import io

# ─────────────────────────────────────────────
# MODE SELECTOR
# Controls horizontal alignment of the person:
#
# "head_center": center image based on head position
# "black_pixel": align using floor/mat reference (first black pixel)
# ─────────────────────────────────────────────
MODE = "black_pixel"

# ─────────────────────────────────────────────
# TUNING PARAMETERS
# These influence scaling and positioning
# ─────────────────────────────────────────────
PERSON_VERTICAL_OFFSET = 50
# Shifts person vertically AFTER bottom alignment, fixes feet being cut off
# Positive: moves person upward

MASK_FIXED_HEIGHT = 1000
# Fixed height of trapezoid "mat" in final image

EXTRA_SCALE_PX = 100
# Extra pixels added to computed person height (fine-tuning)

SLICE_OFFSET = -50  # adjust (5–30 usually good)
# Offset for the right edge of the mat (necessary to get rid of the pink tape getting extended)

ROOT_FOLDER = "/Volumes/private/10_Data/01_Main/sub-C9941R/sharp"

USE_SUBJECT_FOLDERS = False
# True: expects sub-XXXXXX folders with /jpg_work inside
# False: directly scans ROOT_FOLDER for JPGs

# ─────────────────────────────────────────────

# Load segmentation model (human-specific)
session = new_session("u2net_human_seg")

# helper function to save file with just the cut out person
def save_debug_person(foreground_scaled, output_path):
    """Saves the cropped person (with transparency) as a PNG for debugging."""
    foreground_scaled.save(output_path, "PNG")
    print(f" Debug person saved → {output_path}")

# Cuts out the mat from the image
def get_trapezoid_mask(image_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image_size

    anchor_x = 0
    anchor_y = height

    bottom_left  = (anchor_x + 3403, anchor_y - 0)
    bottom_right = (anchor_x + 5220, anchor_y - 0)
    top_left     = (anchor_x + 3403, anchor_y - 1082)
    top_right    = (anchor_x + 5220, anchor_y - 1082)

    polygon = [bottom_left, bottom_right, top_right, top_left]

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bounding_box = (min(xs), min(ys), max(xs), max(ys))

    return mask, bounding_box

# Centers the fighter by the head
def get_paste_x_head_center(foreground_scaled, alpha_scaled, bbox_scaled,
                             input_width, new_fg_width, paste_y):
    """Script 1: center horizontally by head position."""
    left_s, top_s, right_s, bottom_s = bbox_scaled
    paste_x = (input_width - new_fg_width) // 2

    head_scan_y = top_s + 150
    alpha_row = alpha_scaled.crop((0, head_scan_y, new_fg_width, head_scan_y + 1))
    alpha_row_pixels = list(alpha_row.getdata())
    non_transparent = [x for x, val in enumerate(alpha_row_pixels) if val > 10]

    if non_transparent:
        head_left = min(non_transparent)
        head_right = max(non_transparent)
        head_mid_x_in_fg = (head_left + head_right) // 2

        image_center_x = input_width // 2
        head_abs_x = paste_x + head_mid_x_in_fg
        shift = image_center_x - head_abs_x
        paste_x += shift

        head_abs_x = paste_x + head_mid_x_in_fg
        head_abs_y = paste_y + head_scan_y
        print(f" Horizontal shift: {shift}px to center head")
        print(f" Head center (absolute): x={head_abs_x}, y={head_abs_y}")

    return paste_x

# Centers fighter by gap between mat
def get_paste_x_black_pixel(original_scaled, bounding_box, scale_factor, input_width):
    """Script 2: center horizontally by first black pixel in trapezoid bottom row."""
    rect_left_s, rect_top_s, rect_right_s, rect_bottom_s = [
        int(x * scale_factor) for x in bounding_box
    ]

    trap_region = original_scaled.crop((rect_left_s, rect_top_s, rect_right_s, rect_bottom_s))
    trap_region_rgb = trap_region.convert("RGB")

    trap_w, trap_h = trap_region.size
    bottom_row_y = trap_h - 1
    black_threshold = 30

    black_pixels_found = []
    for x in range(trap_w):
        r, g, b = trap_region_rgb.getpixel((x, bottom_row_y))
        if r < black_threshold and g < black_threshold and b < black_threshold:
            black_pixels_found.append(x)

    image_center_x = input_width // 2

    if black_pixels_found:
        first_black_x = black_pixels_found[0]
        first_black_in_scaled = rect_left_s + first_black_x
        paste_x = image_center_x - first_black_in_scaled
        print(f" Aligned first black pixel to center, paste_x={paste_x}")
    else:
        print(f" No black pixels found in bottom row, falling back to width-center")
        paste_x = (input_width - int(original_scaled.width)) // 2

    return paste_x

# Use rmbeg model to cut out person
def crop_person_to_black_bg(input_path, output_path, filename):

    match = re.search(r'stance-(\d+)', filename)
    if not match:
        raise ValueError(f"Could not find stance-XXX in filename: {filename}")
    stance_cm = int(match.group(1))
    print(f" Stance height from filename: {stance_cm} cm")

    # --- Step 1: Load original image ---
    original = Image.open(input_path).convert("RGBA")

    # --- Step 2: Build the trapezoid mask ---
    trap_mask, bounding_box = get_trapezoid_mask(original.size)

    # --- Step 3: Remove background with rembg ---
    with open(input_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes, session=session)
    foreground = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # --- Step 4: Get person bounding box ---
    alpha = foreground.split()[3]
    bbox = alpha.getbbox()
    if not bbox:
        raise ValueError(f"No person detected in {filename}")

    left, top, right, bottom = bbox
    person_height_px = bottom - top
    person_width_px = right - left
    print(f" Person size: {person_width_px} × {person_height_px} px")

    # --- Step 5: Calculate 16:9 canvas size based on input width ---
    input_width, input_height = original.size
    target_height = int(input_width * 9 / 16)
    print(f" Canvas size: {input_width} × {target_height} px")

    # --- Step 6: Calculate target person height based on stance ratio ---
    screen_height_cm = 225

    target_person_height_px = int((stance_cm / screen_height_cm) * target_height) + EXTRA_SCALE_PX
    print(f" Target person height: {target_person_height_px} px ({stance_cm}/{screen_height_cm} of {target_height}px)")

    # --- Step 7: Scale foreground to match target person height ---
    scale_factor = target_person_height_px / person_height_px
    new_fg_width = int(foreground.width * scale_factor)
    new_fg_height = int(foreground.height * scale_factor)
    foreground_scaled = foreground.resize((new_fg_width, new_fg_height), Image.LANCZOS)
    print(f" Scale factor: {scale_factor:.3f}")

    # --- Step 8: Create black background canvas ---
    black_bg = Image.new("RGBA", (input_width, target_height), (0, 0, 0, 255))
    original_scaled = original.resize((new_fg_width, new_fg_height), Image.LANCZOS)
    trap_mask_scaled = trap_mask.resize((new_fg_width, new_fg_height), Image.LANCZOS)

    # --- Get left/right bounds of trapezoid in scaled coords ---
    rect_left_s, _, rect_right_s, _ = [
        int(x * scale_factor) for x in bounding_box
    ]

    # --- Get person bounding box after scaling ---
    alpha_scaled = foreground_scaled.split()[3]
    bbox_scaled = alpha_scaled.getbbox()
    if not bbox_scaled:
        raise ValueError(f"No person detected after scaling in {filename}")

    left_s, top_s, right_s, bottom_s = bbox_scaled

    # ── Pin person bottom to canvas bottom + vertical offset ────────────────
    paste_y = target_height - bottom_s + PERSON_VERTICAL_OFFSET
    print(f" paste_y (bottom-pinned): {paste_y}  (canvas_h={target_height}, person_bottom={bottom_s})")

    # ── Mat anchored to canvas bottom with fixed height (paste_y is now known) ──
    rect_bottom_s = target_height - paste_y
    rect_top_s = rect_bottom_s - MASK_FIXED_HEIGHT
    rect_h = MASK_FIXED_HEIGHT
    print(f" Mat local coords: top={rect_top_s}, bottom={rect_bottom_s}, height={rect_h}")
    # ────────────────────────────────────────────────────────────────────────

    # ── Centering logic switches here ──
    if MODE == "head_center":
        paste_x = get_paste_x_head_center(
            foreground_scaled, alpha_scaled, bbox_scaled,
            input_width, new_fg_width, paste_y
        )
    elif MODE == "black_pixel":
        paste_x = get_paste_x_black_pixel(
            original_scaled, bounding_box, scale_factor, input_width
        )
    else:
        raise ValueError(f"Unknown MODE: '{MODE}'. Use 'head_center' or 'black_pixel'.")

    # --- Step 9: Paste rectangle with edges stretched in-place ---

    # Left edge fill
    left_edge_x = rect_left_s + paste_x
    if left_edge_x > 0:
        left_col = original_scaled.crop((rect_left_s, rect_top_s, rect_left_s + 1, rect_bottom_s))
        left_fill = left_col.resize((left_edge_x + 1, rect_h), Image.LANCZOS)
        black_bg.paste(left_fill, (0, paste_y + rect_top_s))

    # Right edge fill
    right_edge_x = rect_right_s + paste_x
    if right_edge_x < input_width:
        right_col = original_scaled.crop((
            rect_right_s - 1 - SLICE_OFFSET,
            rect_top_s,
            rect_right_s - SLICE_OFFSET,
            rect_bottom_s
        ))
        right_fill = right_col.resize((input_width - right_edge_x, rect_h), Image.LANCZOS)
        black_bg.paste(right_fill, (right_edge_x, paste_y + rect_top_s))

    # Cut trap mask above rect_top_s so it aligns with the edge fills
    draw_mask = ImageDraw.Draw(trap_mask_scaled)
    draw_mask.rectangle([0, 0, trap_mask_scaled.width, rect_top_s - 1], fill=0)

    # Paste the mat rectangle
    black_bg.paste(original_scaled, (paste_x, paste_y), mask=trap_mask_scaled)

    # --- Step 10: Paste scaled person on top ---
    black_bg.paste(foreground_scaled, (paste_x, paste_y), mask=foreground_scaled.split()[3])

        # --- Debug: save cropped person ---
    #debug_output_path = output_path.replace(".jpg", "_debug.png").replace(".jpeg", "_debug.png")
    #save_debug_person(foreground_scaled, debug_output_path)

    # --- Step 11: Save ---
    black_bg.convert("RGB").save(output_path, "JPEG", quality=95)


def scan_and_crop(root_folder, limit=10):
    processed = 0

    try:
        # ─────────────────────────────────────────
        # MODE 1: Standard subject folder structure
        # ─────────────────────────────────────────
        if USE_SUBJECT_FOLDERS:

            for subject in os.listdir(root_folder):
                subject_path = os.path.join(root_folder, subject)

                if not (os.path.isdir(subject_path) and subject.startswith("sub-")):
                    continue

                jpg_work_path = os.path.join(subject_path, "jpg_work")

                if not os.path.isdir(jpg_work_path):
                    print(f" Skipping {subject} (no jpg_work folder)")
                    continue

                print(f"\n Subject: {subject}")
                print(f" Folder: {jpg_work_path}")
                print(f" Mode: {MODE}")
                print("-" * 50)

                jpg_files = [
                    f for f in os.listdir(jpg_work_path)
                    if f.lower().endswith((".jpg", ".jpeg"))
                ]

                for filename in jpg_files:
                    if processed >= limit:
                        raise StopIteration

                    filepath = os.path.join(jpg_work_path, filename)

                    output_dir = os.path.join(subject_path, "cropped")
                    os.makedirs(output_dir, exist_ok=True)

                    prefix = "cropped" if MODE == "head_center" else "cropped-mat"
                    output_path = os.path.join(output_dir, f"{prefix}_{filename}")

                    print(f" {filename} — processing...")
                    crop_person_to_black_bg(filepath, output_path, filename)

                    processed += 1
                    print(f" Saved -> {output_path} ({processed}/{limit})")

        # ─────────────────────────────────────────
        # MODE 2: Flat folder (no subject structure)
        # ─────────────────────────────────────────
        else:
            print(f"\n Flat folder mode")
            print(f" Folder: {root_folder}")
            print(f" Mode: {MODE}")
            print("-" * 50)

            jpg_files = [
                f for f in os.listdir(root_folder)
                if f.lower().endswith((".jpg", ".jpeg"))
            ]

            output_dir = os.path.join(root_folder, "cropped")
            os.makedirs(output_dir, exist_ok=True)

            for filename in jpg_files:
                if processed >= limit:
                    raise StopIteration

                filepath = os.path.join(root_folder, filename)

                prefix = "cropped" if MODE == "head_center" else "cropped-mat"
                output_path = os.path.join(output_dir, f"{prefix}_{filename}")

                print(f" {filename} — processing...")
                crop_person_to_black_bg(filepath, output_path, filename)

                processed += 1
                print(f" Saved -> {output_path} ({processed}/{limit})")

    except StopIteration:
        pass

    print(f"\nProcessed {processed} image(s).")


if __name__ == "__main__":
    import sys

    if not os.path.isdir(ROOT_FOLDER):
        print(f"Error: '{ROOT_FOLDER}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning for JPG images in: {os.path.abspath(ROOT_FOLDER)}")
    scan_and_crop(ROOT_FOLDER)
    print("\nDone!")
