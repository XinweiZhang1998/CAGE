# attacker_caa_textvqa.py

import argparse, os, json, datetime
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor


# =========================================================
# 1) Setup & Utilities
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP un-normalization for visualization
unnorm = T.Normalize(
    mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
    std=[1/0.26862954, 1/0.26130258, 1/0.27577711]
)

def _load_results(results_json: str):
    """Compatible with {"results":[...]} or plain list."""
    with open(results_json, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized results_json format: {type(data)}")

def find_textvqa_image_path(textvqa_root, image_id):
    """
    TextVQA images are typically stored in:
      {textvqa_root}/train_images/{image_id}.jpg
      {textvqa_root}/val_images/{image_id}.jpg
      {textvqa_root}/test_images/{image_id}.jpg
    Also try png/jpeg variants.
    """
    for folder in ["train_images", "val_images", "test_images"]:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(textvqa_root, folder, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
    return None

@torch.no_grad()
def get_clean_feats_and_scores(model, pixel, sel_layer=-2, score_method='norm'):
    out = model.vision_model(
        pixel,
        output_hidden_states=True,
        output_attentions=(score_method == 'attn'),
        return_dict=True
    )
    feats = out.hidden_states[sel_layer][0]  # [1+P, D]
    if score_method == 'norm':
        scores = feats[1:].norm(dim=-1)      # [P]
    elif score_method == 'attn':
        # CLS -> patches attention from penultimate attention layer
        attn = out.attentions[-2]            # [B, heads, T, T]
        attn = attn.mean(dim=1)[0]           # [T, T]
        scores = attn[0, 1:]                 # [P]
    else:
        raise ValueError(f"Unknown score_method: {score_method}")
    return feats, scores

def get_adv_feats_and_scores(model, pixel_adv, sel_layer=-2, score_method='norm'):
    out = model.vision_model(
        pixel_adv,
        output_hidden_states=True,
        output_attentions=(score_method == 'attn'),
        return_dict=True
    )
    feats = out.hidden_states[sel_layer][0]  # [1+P, D]
    if score_method == 'norm':
        scores = feats[1:].norm(dim=-1)      # [P]
    elif score_method == 'attn':
        attn = out.attentions[-2]            # [B, heads, T, T]
        attn = attn.mean(dim=1)[0]           # [T, T]
        scores = attn[0, 1:]                 # [P]
    else:
        raise ValueError(f"Unknown score_method: {score_method}")
    return feats, scores


# =========================================================
# 2) Core Method: EFD + RDA-KL (single extra loss)
# =========================================================

def token_distortion(feats_adv_p, feats_clean_p, metric="cos"):
    """Per-token distortion d_i."""
    if metric == "cos":
        z_adv = F.normalize(feats_adv_p, dim=-1)
        z_cln = F.normalize(feats_clean_p, dim=-1)
        d = 1.0 - (z_adv * z_cln).sum(dim=-1)  # [P] in [0,2]
        return d
    elif metric == "l2":
        return (feats_adv_p - feats_clean_p).pow(2).mean(dim=-1)  # [P]
    else:
        raise ValueError(f"Unknown metric: {metric}")

def get_survival_probs(scores, k_min=16, k_max=128):
    """
    Survival probability pi_i as a soft mask induced by a uniform prior K ~ U[k_min,k_max].
    Implemented as a rank-based linear decay (detached weights).
    """
    ranks = torch.argsort(torch.argsort(scores, descending=True))  # rank 0 highest
    probs = (k_max - ranks.float()) / (k_max - k_min + 1e-6)
    probs = torch.clamp(probs, 0.0, 1.0)
    return probs.detach()  # do not backprop through ranks

def expected_feature_divergence(feats_adv_p, feats_clean_p, probs, metric='cos'):
    """EFD: L_div = sum(pi_i * d_i) / sum(pi_i).  We MAXIMIZE this."""
    dists = token_distortion(feats_adv_p, feats_clean_p, metric=metric)
    weighted = (dists * probs).sum()
    loss = weighted / (probs.sum() + 1e-6)
    return loss

def rda_kl_align(scores_adv, feats_adv_p, feats_clean_p,
                 detach_d=True, metric="cos",
                 normalize=True):
    """
    Returns:
      align_score in [0,1] if normalize=True, else raw;
      raw (always).
    """
    d = token_distortion(feats_adv_p, feats_clean_p, metric=metric)
    if detach_d:
        d = d.detach()

    P = scores_adv.numel()

    # ---- remove tau: normalize d and s before softmax/log_softmax ----
    d32 = d.float()
    s32 = scores_adv.float()

    d_hat = (d32 - d32.mean()) / (d32.std(unbiased=False))  # [P]
    s_hat = (s32 - s32.mean()) / (s32.std(unbiased=False))  # [P]

    p_d = torch.softmax(d_hat, dim=0)          # [P]
    log_p_s = torch.log_softmax(s_hat, dim=0)  # [P]

    raw = (p_d * log_p_s).sum()  # <= 0

    if not normalize:
        return raw.to(scores_adv.dtype), raw.to(scores_adv.dtype)

    denom = np.log(P) if P > 1 else 1.0
    align_score = (raw + denom) / denom  # map [-logP,0] -> [0,1]
    return align_score.to(scores_adv.dtype), raw.to(scores_adv.dtype)

# =========================================================
# 3) Main Attack Logic (PGD ascent)
# =========================================================

def generate_adversarial(results_json, textvqa_root, args, current_time):
    set_seed(args.seed)

    results = _load_results(results_json)

    # Build unique image list from results
    uniq = {}
    missing = 0
    for r in results:
        if "image_id" not in r:
            continue
        image_id = r["image_id"]
        if image_id in uniq:
            continue
        p = find_textvqa_image_path(textvqa_root, image_id)
        if p:
            uniq[image_id] = p
        else:
            missing += 1

    print(f"Total results: {len(results)}")
    print(f"Unique images found: {len(uniq)}")
    if missing > 0:
        print(f"Warning: {missing} images missing under {textvqa_root}/train_images|val_images|test_images")

    # Optional: limit number of images (keep deterministic order by sorting keys)
    image_items = sorted(list(uniq.items()), key=lambda x: str(x[0]))
    if args.num_samples is not None and args.num_samples < len(image_items):
        image_items = image_items[:args.num_samples]
        print(f"Using first {args.num_samples} images")

    print(f"Loading Model: {args.model_id} ...")
    model = CLIPModel.from_pretrained(args.model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_id)

    # per-channel scaling to match CLIP normalization domain
    scaling = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device).view(1, 3, 1, 1)
    alpha = args.alpha / 255.0 / scaling
    epsilon = args.epsilon / 255.0 / scaling

    out_dir = f"attack/CAA_EFD_RDAKL_{args.score_method}_seed{args.seed}_TextVQA_{current_time}/"
    os.makedirs(out_dir, exist_ok=True)
    mapping = {}

    print("Starting Attack: maximize [ lambda_div * EFD + lambda_attr * RDA-KL ]")
    print(f"  score_method={args.score_method} | div_metric={args.div_metric}")
    print(f"  lambda_div={args.lambda_div} | lambda_attr={args.lambda_attr}")
    print(f"  tau_d={args.tau_d} | tau_s={args.tau_s}")
    print(f"  K prior ~ U[{args.k_min},{args.k_max}]")

    for image_id, image_path in tqdm(image_items, desc="Attacking (TextVQA)"):
        image = Image.open(image_path).convert("RGB")
        pixel = processor(images=image, return_tensors="pt")["pixel_values"].to(device)  # [1,3,H,W]

        # clean reference
        feats_clean_all, _ = get_clean_feats_and_scores(
            model, pixel, sel_layer=args.sel_layer, score_method=args.score_method
        )
        feats_clean_pch = feats_clean_all[1:]  # [P,D]

        delta = torch.zeros_like(pixel, requires_grad=True)

        for step in range(args.steps):
            adv_img = pixel + delta

            feats_adv_all, scores_adv = get_adv_feats_and_scores(
                model, adv_img, sel_layer=args.sel_layer, score_method=args.score_method
            )
            feats_adv_pch = feats_adv_all[1:]  # [P,D]

            # Loss A: EFD (maximize)
            probs = get_survival_probs(scores_adv, k_min=args.k_min, k_max=args.k_max)
            L_div = expected_feature_divergence(
                feats_adv_pch, feats_clean_pch, probs, metric=args.div_metric
            )

            # Loss B: RDA-KL alignment (maximize)
            L_align, L_align_raw = rda_kl_align(
                scores_adv, feats_adv_pch, feats_clean_pch,
                detach_d=True, metric=args.div_metric,
                normalize=True
            )

            # Total (PGD ascent)
            loss = args.lambda_div * L_div + args.lambda_attr * L_align

            if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
                print(f"[{image_id}] step {step:03d} | "
                      f"Total={loss.item():.4f}  "
                      f"EFD={L_div.item():.4f}  "
                      f"RDA(KL-norm)={L_align.item():.4f}  "
                      f"RDA(raw)={L_align_raw.item():.4f}")
               

            loss.backward()

            with torch.no_grad():
                delta.data += alpha * delta.grad.sign()
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.grad.zero_()

        # save adversarial image
        adv_img_vis = torch.clamp(unnorm(pixel + delta), 0.0, 1.0)
        adv_pil = T.ToPILImage()(adv_img_vis.squeeze(0).detach().cpu())
        save_path = os.path.join(out_dir, f"{image_id}.png")
        adv_pil.save(save_path)

        mapping[str(image_id)] = {"original_path": image_path, "adv_path": save_path}

    with open(os.path.join(out_dir, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # optional: save config
    cfg = vars(args).copy()
    cfg.update({
        "dataset": "TextVQA",
        "results_json": results_json,
        "textvqa_root": textvqa_root,
        "num_unique_images": len(image_items),
        "timestamp": current_time,
        "out_dir": out_dir,
    })
    with open(os.path.join(out_dir, "attack_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n=== Done ===\nOutput saved to: {out_dir}")


# =========================================================
# 4) CLI
# =========================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CAA (TextVQA): EFD + RDA-KL (K-agnostic, compression-aware)")

    ap.add_argument("--results_json", type=str, required=True)
    ap.add_argument("--textvqa_root", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--sel_layer", type=int, default=-2)

    ap.add_argument("--score_method", type=str, default="attn", choices=["norm", "attn"])
    ap.add_argument("--div_metric", type=str, default="cos", choices=["cos", "l2"])

    # K prior range for EFD
    ap.add_argument("--k_min", type=int, default=16)
    ap.add_argument("--k_max", type=int, default=192)

    # loss weights
    ap.add_argument("--lambda_div", type=float, default=1.0, help="weight for EFD (maximize)")
    ap.add_argument("--lambda_attr", type=float, default=1.0, help="weight for RDA-KL alignment (maximize)")
、
    # PGD params
    ap.add_argument("--alpha", type=float, default=1.0, help="step size in 1/255 units (before channel scaling)")
    ap.add_argument("--epsilon", type=int, default=16, help="Linf budget in 1/255 units (before channel scaling)")
    ap.add_argument("--steps", type=int, default=100)

    ap.add_argument("--seed", type=int, default=42)

    # Optional: limit number of images
    ap.add_argument("--num_samples", type=int, default=None, help="limit number of unique images attacked")

    args = ap.parse_args()
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    generate_adversarial(args.results_json, args.textvqa_root, args, now)
