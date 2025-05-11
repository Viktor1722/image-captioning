import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def load_model():
    """Load the BLIP model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_caption(image_path, processor, model):
    """Generate a detailed caption for an image focusing on interior design elements."""
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Create a detailed prompt for interior design focus
    prompt = "Describe this interior space in detail, including the color scheme, furniture style, and overall design aesthetic. Focus on materials, textures, and architectural elements."
    
    # Generate initial caption with the prompt
    inputs = processor(image, text=prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=150, num_beams=5, min_length=50)
    initial_caption = processor.decode(output[0], skip_special_tokens=True)
    
    # Generate a second caption focusing on colors and materials
    prompt_colors = "List the main colors and materials used in this interior space."
    inputs_colors = processor(image, text=prompt_colors, return_tensors="pt")
    output_colors = model.generate(**inputs_colors, max_new_tokens=50, num_beams=5)
    colors_caption = processor.decode(output_colors[0], skip_special_tokens=True)
    
    # Combine the captions and format with the trigger word
    enhanced_caption = f"[trigger] In this interior space: {initial_caption.lower()} The color palette and materials include {colors_caption.lower()}"
    
    return enhanced_caption

def process_images(input_folder):
    """Process all images in the input folder and save captions to text files."""
    # Ensure the dataset folder exists
    dataset_folder = "dataset"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Load the model
    processor, model = load_model()
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(input_folder, filename)
            
            # Generate caption
            caption = generate_caption(image_path, processor, model)
            
            # Create output text file name in the dataset folder
            text_filename = os.path.splitext(filename)[0] + '.txt'
            text_path = os.path.join(dataset_folder, text_filename)
            
            # Save caption to text file
            with open(text_path, 'w') as f:
                f.write(caption)
            
            print(f"Processed {filename} -> {text_filename} (saved in dataset folder)")

if __name__ == "__main__":
    print("Welcome to the Image Captioning Tool!")
    while True:
        # Ask for input folder path
        input_folder = input("\nPlease enter the path to your folder containing images: ").strip()
        
        # Remove quotes if present
        input_folder = input_folder.strip("'\"")
        
        # Check if the folder exists
        if not os.path.exists(input_folder):
            print(f"Error: The folder '{input_folder}' does not exist. Please try again.")
            continue
        
        # Check if the folder contains any supported images
        has_images = any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) 
                        for f in os.listdir(input_folder))
        if not has_images:
            print("Error: No supported images found in the folder. Please make sure your folder contains .jpg, .jpeg, .png, or .webp files.")
            continue
        
        print("\nStarting image captioning process...")
        process_images(input_folder)
        print("\nCaptioning complete! All caption files have been saved in the 'dataset' folder.")
        break 