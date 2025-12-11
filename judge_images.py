import torch
import json
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "./qwen2.5-vl-7b" 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_path)


with open('simple_prompts.json', 'r') as f:
    prompts = json.load(f)

results = []

def run_judge(image_path, target_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"""
                You are an objective logic checker. You should check colors, count and spatial positions of objects.
                The requested prompt: "{target_prompt}"
                Does the image perfectly match the text? Answer only "YES" or "NO".
                """}, 
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    score = 1 if "YES" in response[:15].upper() else 0
    return score, response

models = ["sd_baseline", "sd_smart"] #, 

for item in prompts:
    pid = item['id']
    target = item['simple_prompt']
    
    print(f"\n\n\nJudging ID {pid}...")
    
    for m in models:
        img_path = f"output_images/{pid}_{m}.png"
        
        if not os.path.exists(img_path):
            print(f"Skipping missing: {img_path}")
            continue
            
        try:
            score, reason = run_judge(img_path, target)
            print(f"  > {m}: {score} {reason}")
            
            results.append({
                "id": pid,
                "model": m,
                "score": score,
                "reason": reason
            })
        except Exception as e:
            print(f"  > Error judging {img_path}: {e}")

os.makedirs("output_scores", exist_ok=True)
with open('output_scores/final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nEvaluation Complete! Results saved.")

# ---------------------------
# Added Accuracy Report
# ---------------------------
print("\n=== MODEL ACCURACY REPORT ===")
model_scores = {m: {"correct": 0, "total": 0} for m in models}

for r in results:
    model_scores[r["model"]]["total"] += 1
    model_scores[r["model"]]["correct"] += r["score"]

for m in models:
    total = model_scores[m]["total"]
    correct = model_scores[m]["correct"]
    acc = (correct / total * 100) if total > 0 else 0
    print(f"{m}: {acc:.2f}% accuracy ({correct}/{total})")
