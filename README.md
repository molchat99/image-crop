# Image Cropping & Background Processing Pipeline

This script processes JPG images by:

* removing the background using a neural network model
* scaling the person based on real-world height (from filename)
* placing the person onto a standardized black 16:9 canvas
* reconstructing and extending a floor/mat region
* exporting clean, aligned output images

---
![Alt text](images/crop.png)
---

## Installation

Installation can be slightly tricky depending on your environment.

### 1. Recommended setup

Use a **virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
```

### 2. Python version

If you encounter issues:

* try a different Python version
* known working version: **Python 3.11.15**

---

### 3. Install dependencies

If this fails:

```bash
pip install rembg pillow
```

Try installing step by step:

```bash
pip install pillow
pip install rembg
```

---

### 4. rembg (background removal)

This script uses the library:

https://github.com/danielgatis/rembg

It loads the **u2net human segmentation model** automatically.

---

## Input Requirements

### File naming

Images must include **stance height in cm**:

```
stance-180.jpg
stance-165.jpeg
```

This is required for scaling.

---

### Folder structure options

#### Option A — Subject-based structure

```
ROOT_FOLDER/
└── sub-XXXXXX/
    └── jpg_work/
        ├── image1.jpg
        ├── image2.jpg
```

#### Option B — Flat folder (simpler)

```
ROOT_FOLDER/
├── image1.jpg
├── image2.jpg
```

---

## Processing Logic

1. Remove background (person segmentation)
2. Detect person bounding box
3. Scale person based on stance height
4. Create 16:9 output canvas
5. Reconstruct trapezoid floor ("mat")
6. Extend left/right edges of the mat
7. Paste person onto final image

---

## Settings

### Alignment mode

```python
MODE = "black_pixel"
```

* `"head_center"` centers image based on head position
* `"black_pixel"` aligns based on floor/mat (recommended)

---

### Vertical positioning

```python
PERSON_VERTICAL_OFFSET = 50
```

* shifts person **after bottom alignment**
* useful if feet are cut off
* positive → moves person upward

---

### Mat (floor) height

```python
MASK_FIXED_HEIGHT = 1000
```

* fixed height of the trapezoid floor in output image
* same for all images

---

### Scaling fine-tuning

```python
EXTRA_SCALE_PX = 100
```

* adds extra pixels to computed height
* helps compensate for segmentation inaccuracies

---

### Mat edge correction (important)

```python
SLICE_OFFSET = -50
```

* shifts the **right edge sampling region**
* prevents unwanted areas (e.g. pink tape) from being stretched

Typical values: `-30` to `+30`
Adjust depending on artifacts

---

### Folder mode

```python
USE_SUBJECT_FOLDERS = False
```

* `True` expects `sub-XXXXXX/jpg_work/` structure
* `False` directly scans `ROOT_FOLDER`

---

## Usage

Set your input directory:

```python
ROOT_FOLDER = "/path/to/your/images"
```

Run:

```bash
python your_script.py
```

---

## Output

Processed images are saved to:

* Subject mode:

  ```
  sub-XXXXXX/cropped/
  ```

* Flat mode:

  ```
  ROOT_FOLDER/cropped/
  ```

---

## Common Issues

### Wrong scaling

* check filename contains:

  ```
  stance-XXX
  ```

---

### Mat artifacts (e.g. pink tape stretched)

* adjust:

  ```python
  SLICE_OFFSET
  ```

---

## Debugging

You can enable debug output (in script):

```python
save_debug_person(...)
```

This saves the extracted person as PNG.

---
