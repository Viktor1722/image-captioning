# AI Image Captioning for Interior Design

This script uses AI to generate detailed captions for interior design images, with a focus on describing room styles and details. Each caption includes the mandatory "[trigger]" word and is saved in a separate text file.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:

```bash
python caption_generator.py
```

2. When prompted, enter the full path to the folder containing your images.

3. The script will:
   - Validate the folder path and check for supported images
   - Process all images in your specified folder
   - Generate AI-powered captions with interior design focus
   - Include the "[trigger]" word in each caption
   - Save captions in individual .txt files in the `dataset` folder

## Supported Image Formats

- .jpg
- .jpeg
- .png
- .webp

## Output

- All caption files will be automatically saved in the `dataset` folder
- Each caption file will have the same name as its corresponding image, but with a .txt extension
- Each caption will start with "[trigger]" and focus on describing the interior design elements of the image

## Notes

- The script uses the BLIP (Bootstrapping Language-Image Pre-training) model for high-quality image captioning
- Make sure you have sufficient disk space for the model (approximately 2GB)
- Processing time may vary depending on your hardware and image size
# image-captioning
