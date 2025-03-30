#!/bin/bash

# Run the caption generation script with default settings (llama-3.2-11b model, batch size 5)
# python tag_captions.py

# Alternatively, you can specify model and batch size:
python tag_captions.py --model-name qwen --batch-size 64 --dataset "SA-1B"
# python tag_captions.py --model-name gemma --batch-size 64 --dataset "SA-1B"
# python tag_captions.py --model-name deepseek --batch-size 64 --dataset "SA-1B"