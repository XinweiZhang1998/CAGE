
# [ICML26]On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression

This repository contains the official implementation of **CAGE** on the TextVQA dataset. It targets LLaVA models equipped with VisionZIP inference acceleration.

## 1. Dataset Preparation

1. Download the **Validation set** and **images** from the [TextVQA official website](https://textvqa.org/dataset/).
2. Organize the dataset directory structure as follows:

```text
dataset/
└── TextVQA/
    ├── train_images/
    └── TextVQA_0.5.1_val.json

```

## 2. Inference (Clean Images)

Perform inference on LLaVA with VisionZIP using clean images to establish the baseline performance.

```bash

conda env create -f environment.yml


CUDA_VISIBLE_DEVICES=0 python llava_verify_zip_VQA_Text.py \
  --dominant 54 \
  --contextual 10 \
  --zip

```

> **Note:**
> * Replace `54` and `10` with your specific VisionZIP hyperparameters (dominant/contextual tokens).
> * Results will be saved at: `./output/textvqa_results/textvqa_zip_54_10_val.json`
> 
> 

## 3. Perform Attack (CAGE)

Run the attack script to generate adversarial examples targeting the compressed model.

```bash
CUDA_VISIBLE_DEVICES=0 python attacker_TextVQA_caa.py \
  --results_json    ./output/textvqa_results/textvqa_zip_54_10_val.json \
  --textvqa_root ./dataset/TextVQA \
  --epsilon 2 \
  --alpha 0.5 \
  --steps 100 \
  --sel_layer -2 \
  --lambda_div 0.005

```

**Arguments:**

* `--results_json`: Path to the clean baseline results (generated in Step 2).
* `--epsilon`: Perturbation budget ( norm).
* `--alpha`: Step size for the attack optimization.
* `--steps`: Number of optimization iterations.
* `--sel_layer`: The layer index used for feature selection (e.g., -2).
* `--lambda_div`: Weight for the diversity loss term.

The generated adversarial examples will be saved in: `./attack/CAA_EFD_RDAKL_attn_seed42_TextVQA_[TIMESTAMP]/`

## 4. Verify Attack Performance

Evaluate the attack success rate by testing the generated adversarial examples against the target model.

```bash
CUDA_VISIBLE_DEVICES=0 python llava_verify_zip_VQA_Text.py \
  --dominant 54 \
  --contextual 10 \
  --zip \
  --adversarial \
  --adversarial-dir ./attack/CAA_EFD_RDAKL_attn_seed42_TextVQA_[TIMESTAMP]/

```

> **Important:**
> * Ensure `--adversarial-dir` points to the specific timestamped folder created in **Step 3**.
> * Keep the `--dominant` and `--contextual` parameters consistent with Step 2.
> 
> 
