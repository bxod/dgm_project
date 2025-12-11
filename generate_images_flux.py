import torch
import json
from diffusers import FluxPipeline
import os
os.makedirs("output_images", exist_ok=True)

with open('smart_prompts.json', 'r') as f:
    simple_data = json.load(f)
pipe_flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
# pipe_flux.enable_model_cpu_offload()

for item in simple_data:
    pid = item['id']
    prompt = item['smart_prompt']
    
    print(f"Generating FLUX: {pid, prompt}")
    image = pipe_flux(
        prompt,
        height=1024, #1024
        width=1024, #1024
        guidance_scale=3.5, #3.5
        num_inference_steps=25, #50
        max_sequence_length=512,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]
    
    image.save(f"output_images/{pid}_flux_smart.png")