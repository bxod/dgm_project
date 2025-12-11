import torch
import json
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FluxPipeline

# 1. SETUP: Output Folder
os.makedirs("output_images", exist_ok=True)

# 2. LOAD DATA
with open('simple_prompts.json', 'r') as f:
    simple_data = json.load(f)

with open('smart_prompts.json', 'r') as f:
    smart_map = {item['id']: item['smart_prompt'] for item in json.load(f)}

print("Loading SD 1.5 from local files...")
pipe_sd = StableDiffusionPipeline.from_pretrained(
    "./sd-v1-5", 
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

pipe_sd.scheduler = DPMSolverMultistepScheduler.from_config(pipe_sd.scheduler.config)

print("Model loaded successfully!")

SEED = 42

neg_prompt = "low quality, bad quality, blurry, distorted, deformed, ugly, lowres, jpeg artifacts"

generator = torch.Generator("cuda").manual_seed(SEED)

for item in simple_data:
    pid = item['id']
    base_prompt = item['simple_prompt']
    smart_prompt = smart_map.get(pid, base_prompt)
    
    print(f"[{pid}] Generating...")

    # 1. SD Baseline
    # image_base = pipe_sd(
    #     base_prompt,
    #     # negative_prompt=neg_prompt, # <--- ADDED
    #     guidance_scale=7.5,         # <--- EXPLICIT STANDARD 
    #     generator=generator, # <--- CRITICAL
    #     num_inference_steps=25 # Standard for SD 1.5
    # ).images[0]
    # image_base.save(f"output_images/{pid}_sd_baseline.png")
    
    generator = torch.Generator("cuda").manual_seed(SEED)
    
    image_smart = pipe_sd(
        smart_prompt, 
        # negative_prompt=neg_prompt, # <--- ADDED
        guidance_scale=7.5,         # <--- EXPLICIT STANDARD 
        generator=generator, # <--- CRITICAL
        num_inference_steps=25 # Standard for SD 1.5
    ).images[0]
    image_smart.save(f"output_images/{pid}_sd_smart.png")

print("Done.")

