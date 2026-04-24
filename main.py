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
# Shifts person vertically AFTER bottom alignment
# Positive: moves person upward

MASK_FIXED_HEIGHT = 1000
# Fixed height of trapezoid "mat" in final image

EXTRA_SCALE_PX = 100
# Extra pixels added to computed person height (fine-tuning)

# ─────────────────────────────────────────────

# Load segmentation model (human-specific)
session = new_session("u2net_human_seg")


def save_debug_person(foreground_scaled, output_path):
    """Save cropped person (with transparency) for debugging."""
    foreground_scaled.save(output_path, "PNG")
    print(f" Debug person saved → {output_path}")


def get_trapezoid_mask(image_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Create trapezoid mask representing floor/mat region.

    Returns:
        mask: binary mask (white = keep)
        bounding_box: bounding rectangle of trapezoid

    NOTE: coordinates assume fixed camera setup
    """
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


def get_paste_x_head_center(foreground_scaled, alpha_scaled, bbox_scaled,
                             input_width, new_fg_width, paste_y):
    """
    Align horizontally using head position.

    Strategy:
    - Scan a horizontal line near head
    - Find non-transparent pixels (head edges)
    - Center midpoint in image
    """
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


def get_paste_x_black_pixel(original_scaled, bounding_box, scale_factor, input_width):
    """
    Align horizontally using first black pixel in floor/mat region.

    Assumes:
    - floor is dark/black
    - first black pixel is stable anchor point
    """
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
        print(f" No black pixels found, fallback to center")
        paste_x = (input_width - int(original_scaled.width)) // 2

    return paste_x


def crop_person_to_black_bg(input_path, output_path, filename):
    """
    Full pipeline for one image:
    - extract person
    - scale to real-world size
    - place on standardized canvas
    - align + composite
    """

    # Extract stance height from filename
    match = re.search(r'stance-(\d+)', filename)
    if not match:
        raise ValueError(f"Could not find stance-XXX in filename: {filename}")
    stance_cm = int(match.group(1))
    print(f" Stance height: {stance_cm} cm")

    # Load original image
    original = Image.open(input_path).convert("RGBA")

    # Build trapezoid mask (floor)
    trap_mask, bounding_box = get_trapezoid_mask(original.size)

    # Remove background (segmentation)
    with open(input_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes, session=session)
    foreground = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # Get person bounding box
    alpha = foreground.split()[3]
    bbox = alpha.getbbox()
    if not bbox:
        raise ValueError(f"No person detected in {filename}")

    left, top, right, bottom = bbox
    person_height_px = bottom - top
    print(f" Person height: {person_height_px}px")

    # Create 16:9 canvas
    input_width, input_height = original.size
    target_height = int(input_width * 9 / 16)

    # Scale person based on real-world ratio
    screen_height_cm = 225
    target_person_height_px = int((stance_cm / screen_height_cm) * target_height) + EXTRA_SCALE_PX

    scale_factor = target_person_height_px / person_height_px
    new_fg_width = int(foreground.width * scale_factor)
    new_fg_height = int(foreground.height * scale_factor)

    foreground_scaled = foreground.resize((new_fg_width, new_fg_height), Image.LANCZOS)

    # Create black background
    black_bg = Image.new("RGBA", (input_width, target_height), (0, 0, 0, 255))
    original_scaled = original.resize((new_fg_width, new_fg_height), Image.LANCZOS)
    trap_mask_scaled = trap_mask.resize((new_fg_width, new_fg_height), Image.LANCZOS)

    # Position person (bottom aligned)
    alpha_scaled = foreground_scaled.split()[3]
    bbox_scaled = alpha_scaled.getbbox()
    _, _, _, bottom_s = bbox_scaled

    paste_y = target_height - bottom_s + PERSON_VERTICAL_OFFSET

    # Horizontal alignment
    if MODE == "head_center":
        paste_x = get_paste_x_head_center(
            foreground_scaled, alpha_scaled, bbox_scaled,
            input_width, new_fg_width, paste_y
        )
    else:
        paste_x = get_paste_x_black_pixel(
            original_scaled, bounding_box, scale_factor, input_width
        )

    # Paste floor (mat)
    black_bg.paste(original_scaled, (paste_x, paste_y), mask=trap_mask_scaled)

    # Paste person on top
    black_bg.paste(foreground_scaled, (paste_x, paste_y), mask=alpha_scaled)

    # Save final image
    black_bg.convert("RGB").save(output_path, "JPEG", quality=95)


def scan_and_crop(root_folder, limit=10):
    """
    Iterate over subject folders and process images.

    Expected structure:
        root/sub-XXXX/jpg_work/*.jpg
    """
    processed = 0

    for subject in os.listdir(root_folder):
        subject_path = os.path.join(root_folder, subject)
        if not (os.path.isdir(subject_path) and subject.startswith("sub-")):
            continue

        jpg_work_path = os.path.join(subject_path, "jpg_work")
        if not os.path.isdir(jpg_work_path):
            continue

        jpg_files = [
            f for f in os.listdir(jpg_work_path)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

        for filename in jpg_files:
            if processed >= limit:
                return

            filepath = os.path.join(jpg_work_path, filename)
            output_dir = os.path.join(subject_path, "cropped")
            os.makedirs(output_dir, exist_ok=True)

            prefix = "cropped" if MODE == "head_center" else "cropped-mat"
            output_path = os.path.join(output_dir, f"{prefix}_{filename}")

            print(f"Processing {filename}...")
            crop_person_to_black_bg(filepath, output_path, filename)

            processed += 1

    print(f"\nProcessed {processed} image(s).")


if __name__ == "__main__":
    folder = "/Volumes/private/10_Data/01_Main"

    if not os.path.isdir(folder):
        print(f"Invalid directory: {folder}")
        exit(1)

    print(f"Scanning folder: {folder}")
    scan_and_crop(folder)
    print("Done!")
