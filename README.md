# Enhancing Compositional Faithfulness in Text-to-Image Generation

A research project investigating **Model-Agnostic Prompt Optimization** to solve compositional failures (attribute bleeding, numeracy errors, spatial confusion) in legacy diffusion models like **Stable Diffusion 1.5** without fine-tuning.

-----

## 📌 Overview

Text-to-Image models like SD 1.5 often struggle with complex prompts (e.g., *"a green bench and a red car"* often results in two red objects). This project implements a **"Smart Prompting"** pipeline that algorithmically restructures user prompts into rigid, functional constraints at inference time.

We validate this method using a novel **Soft-VQA (Visual Question Answering)** evaluation metric powered by **Qwen2.5-VL-7B**, measuring both binary accuracy and model confidence.

### Key Features

  * **Smart Prompt Logic:** Transforms natural language into structured constraints (e.g., *"Isolated on white background"*, *"Object A on top. Object B on bottom"*).
  * **Multi-Model Support:** Pipelines for **Stable Diffusion 1.5** (Baseline & Smart) and **FLUX.1-dev** (SOTA).
  * **Automated Evaluation:** A reference-free evaluation pipeline using VLM logits to calculate a "Soft Score" ($0.0 - 1.0$).

-----

## 📊 Key Results

Our structured prompting strategy nearly **doubled** the compositional accuracy of Stable Diffusion 1.5.

| Model | Overall Accuracy | Attribute | Numeracy | Spatial |
| :--- | :--- | :--- | :--- | :--- |
| **SD 1.5 (Baseline)** | 23.33% | 12.00% | 36.00% | 22.00% |
| **SD 1.5 (Smart)** | **42.00%** | **27.00%** | **62.00%** | **37.00%** |
| **FLUX.1-dev** | 72.67% | 92.00% | 79.00% | 47.00% |
| **FLUX.1-dev (Smart)**| 73.33% | 73.00% | 73.00% | **74.00%** |

> **Note:** While FLUX generally prefers natural language, our structured spatial prompts significantly improved its spatial reasoning (+27%).

-----

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Ensure you have `torch`, `diffusers`, `transformers`, `qwen_vl_utils`, and `google-genai` installed.*

3.  **API Setup:**

      * Create a `.env` file or export your Google Gemini API key (for prompt generation):
        ```bash
        export GOOGLE_API_KEY="your_api_key_here"
        ```

-----

## 🚀 Usage

### 1\. Generate Smart Prompts

Use the LLM (Gemini-3-Pro) to rewrite your simple prompts into structured constraints.

```bash
python make_prompts.py
```

*Input:* `simple_prompts.json`  
*Output:* `smart_prompts.json`

### 2\. Generate Images

Run the generation pipeline for both SD 1.5 and FLUX.

```bash
python step2_generate_images.py
```

  * **Hardware Used:**
      * SD 1.5: NVIDIA RTX 3090 (\~1.5s/image)
      * FLUX.1-dev: NVIDIA RTX 6000 Ada (\~20s/image)

### 3\. Run Soft-VQA Evaluation

Evaluate the generated images using Qwen2.5-VL-7B as a judge.

```bash
python step3_evaluate.py
```

*Output:* `output_scores/final_results_soft.json` and a console report table.

-----

## 🧠 Methodology Highlights

### The "Smart" Logic

Instead of adding flowery details (which confuse SD 1.5), we enforced **Token Separation**:

  * **Naive:** "A red car next to a green bench."
  * **Smart:** "A green bench. A red car. Side by side. Distinct colors."

### Soft VQA Scoring

We move beyond binary YES/NO by extracting logits from the VLM:
$$\text{Soft Score} = \frac{e^{\text{logit}(YES)}}{e^{\text{logit}(YES)} + e^{\text{logit}(NO)}}$$

-----

## 👥 Contributors

  * **Bekhzod Shukhratov** - *Methodology, Implementation, Evaluation*