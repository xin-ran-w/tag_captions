import os
import json
import asyncio
import aiofiles
from openai import AsyncOpenAI
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import base64
import argparse

# Model name to path mapping
MODEL_MAPPING = {
    "qwen": "/home/diaomuxi/model_zoo/Qwen2.5-VL-7B-Instruct",
    "gemma": "/home/diaomuxi/model_zoo/gemma-3-4b-it",
    "deepseek": "/home/diaomuxi/model_zoo/deepseek-vl2"
}

DATA_MAPPING = {
    "T2I-2M": "/home/diaomuxi/dataset_zoo/text-to-image-2M/data_1024_10K",
    "SA-1B": "/home/diaomuxi/dataset_zoo/OpenDataLab___SA-1B/raw/cropped_images" 
}

async def create_image_caption(client: AsyncOpenAI, image_path: Path, model_name: str) -> Dict:
    try:
        # Convert file path to data URI
        async with aiofiles.open(image_path, "rb") as image_file:
            image_data = base64.b64encode(await image_file.read()).decode('utf-8')
            image_uri = f"data:image/jpeg;base64,{image_data}"

        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed description of this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_uri
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        
        return {
            "image": str(image_path.name),
            "caption": response.choices[0].message.content,
            "model": model_name
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

async def process_batch(client: AsyncOpenAI, image_paths: List[Path], model_name: str, batch_size: int = 5):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        tasks = [create_image_caption(client, img_path, model_name) for img_path in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend([r for r in batch_results if r is not None])
    return results

async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process images with vision model to generate captions')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of images to process in parallel (default: 5)')
    parser.add_argument('--model-name', type=str, default="qwen",
                       choices=list(MODEL_MAPPING.keys()),
                       help='Name of the vision model to use')
    parser.add_argument('--dataset', type=str, default="SA-1B",
                       choices=list(DATA_MAPPING.keys()),
                       help='Name of the dataset images to tag')
    args = parser.parse_args()

    # Initialize AsyncOpenAI client
    port = 30000 # adjust if your local server uses a different port
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="None")
    model_name = MODEL_MAPPING[args.model_name]  # Get the full model path from the mapping

    # Set up paths
    image_dir = Path(DATA_MAPPING[args.dataset])
    output_file = f"{args.dataset}-captions-{args.model_name}.json"

    # Load existing results if the output file already exists
    if os.path.exists(output_file):
        async with aiofiles.open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.loads(await f.read())
    else:
        existing_results = []

    # Get list of image files
    image_files = list(image_dir.glob("*.jpg"))
    
    # Filter out already processed images
    processed_images = {result['image'] for result in existing_results}
    image_files = [img for img in image_files if img.name not in processed_images]

    # Process images in batches with progress bar
    batch_size = args.batch_size
    all_results = existing_results  # Start with existing results
    
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            results = await process_batch(client, batch, model_name, batch_size)
            all_results.extend(results)
            
            # Save intermediate results
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(all_results, indent=2, ensure_ascii=False))
            
            pbar.update(len(batch))

if __name__ == "__main__":
    asyncio.run(main())