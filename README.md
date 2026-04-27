Installation can be tricky, a common error was installation of wheels failed. In this case, install dependencies individually in a virtual environment. The python version can also be a source of error, which can be fixed by installing a different version in the env. I'm using Python 3.11.15.

The script uses rembg, a model to remove background from an image. Here's the documentation and how to install: https://github.com/danielgatis/rembg

Settings:

MODE = "black_pixel"
"head_center": center image based on head position
"black_pixel": align using floor/mat reference (first black pixel)

PERSON_VERTICAL_OFFSET = 50
Shifts person vertically AFTER bottom alignment, fixes feet being cut off
Positive: moves person upward

MASK_FIXED_HEIGHT = 1000
Fixed height of trapezoid "mat" in final image

EXTRA_SCALE_PX = 100
Extra pixels added to computed person height (fine-tuning)

SLICE_OFFSET = -50  # adjust (5–30 usually good)
Offset for the right edge of the mat (necessary to get rid of the pink tape getting extended)

USE_SUBJECT_FOLDERS = False
True: expects sub-XXXXXX folders with /jpg_work inside
False: directly scans ROOT_FOLDER for JPGs
