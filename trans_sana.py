import json
import os
from pathlib import Path


def transform_to_sana_format(input_files, image_dir, output_dir):
    """Transform caption JSON files to Sana training format."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all input files
    captions_by_model = {}
    for input_file in input_files:
        model_name = input_file.split('-')[-1].split('.')[0]  # Extract model name
        with open(input_file, 'r') as f:
            captions_by_model[model_name] = json.load(f)

    # Get all unique image names
    img_names = set()
    for data in captions_by_model.values():
        img_names.update(item["image"] for item in data)
    
    # Create individual JSON files for each image and model
    for img_name in img_names:
        
        for model_name, captions in captions_by_model.items():
            # Find caption for this image
            caption = next((item["caption"] for item in captions if item["image"] == img_name), None)
            if caption:
                # Create output file
                output_file = os.path.join(output_dir, f"{img_name.split('.')[0]}_{model_name}.json")
                output_data = {
                    img_name.split('.')[0]: {
                        model_name: caption
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=4)

    # Create meta_data.json
    meta_data = {
        "name": "sana-finetune",
        "__kind__": "Sana-ImgDataset",
        "img_names": [f"{image_dir}/{img_name}" for img_name in img_names],
        "prompt_dir": output_dir
    }
    
    with open(os.path.join(output_dir, "meta_data.json"), 'w') as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    
    input_files = [
        "T2I-2M-captions-qwen.json",
        "T2I-2M-captions-gemma.json"
    ]
    image_dir = "/home/diaomuxi/dataset_zoo/text-to-image-2M/data_1024_10K"
    output_dir = "/home/diaomuxi/dataset_zoo/sana_data/T2I-2M"


    # input_files = [
    #     "SA-1B-captions-qwen.json",
    #     "SA-1B-captions-gemma.json"
    # ]
    # image_dir = "/home/diaomuxi/dataset_zoo/OpenDataLab___SA-1B/raw/cropped_images/"
    # output_dir = "/home/diaomuxi/dataset_zoo/sana_data/SA-1B"
    
    transform_to_sana_format(input_files, image_dir, output_dir)