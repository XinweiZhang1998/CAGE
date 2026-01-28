# -*- coding: utf-8 -*-
"""
TextVQA Dataset Evaluator for LLaVA with VisionZip
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from VisionZip.visionzip import visionzip

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token, process_images, get_model_name_from_path
)


class TextVQAEvaluator:
    def __init__(
        self,
        model_path: str = "liuhaotian/llava-v1.5-7b",
        model_base: str = None,
        device: str = None,
        load_8bit: bool = False,
        use_visionzip: bool = True,
        dominant: int = 14,
        contextual: int = 10,
        use_adversarial: bool = False,
        adversarial_dir: str = None,
    ):
        """初始化LLaVA模型"""
        print(f"Initializing model: {model_path}")
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        
        # 保存对抗样本配置
        self.use_adversarial = use_adversarial
        self.adversarial_dir = adversarial_dir
        
        if use_adversarial:
            if adversarial_dir is None:
                raise ValueError("adversarial_dir must be provided when use_adversarial=True")
            if not os.path.exists(adversarial_dir):
                raise ValueError(f"Adversarial directory not found: {adversarial_dir}")
            print(f"⚠ Using adversarial samples from: {adversarial_dir}")
        
        print("Loading pretrained model...")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=self.model_name,
            load_8bit=load_8bit
        )
        
        if use_visionzip:
            print(f"Applying VisionZip: dominant={dominant}, contextual={contextual}")
            self.model = visionzip(self.model, dominant=dominant, contextual=contextual)
        else:
            print("Running baseline without VisionZip")
        
        self.model.eval()
        self.device = self.model.device if device is None else torch.device(device)
        print(f"Model loaded on device: {self.device}\n")

        self.conv_mode = "llava_v1"
        self.base_conv = conv_templates[self.conv_mode].copy()
        self.stop_str = self.base_conv.sep if self.base_conv.sep_style != SeparatorStyle.TWO else self.base_conv.sep2

    def get_image_path(self, textvqa_root: str, image_id: str) -> str:
        """
        Args:
            textvqa_root: TextVQA数据根目录
            image_id: 图片ID
        """
        if self.use_adversarial:
            for ext in ['.png', '.jpg', '.jpeg']:
                adv_path = os.path.join(self.adversarial_dir, f"{image_id}{ext}")
                if os.path.exists(adv_path):
                    return adv_path
            
            print(f"⚠ Adversarial image not found for {image_id}, using original")
        
        possible_paths = [
            os.path.join(textvqa_root, "train_val_images", "train_images", f"{image_id}.jpg"),
            os.path.join(textvqa_root, "train_val_images", "val_images", f"{image_id}.jpg"),
            os.path.join(textvqa_root, "train_images", f"{image_id}.jpg"),
            os.path.join(textvqa_root, "val_images", f"{image_id}.jpg"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return os.path.join(textvqa_root, "train_val_images", "val_images", f"{image_id}.jpg")

    def _build_prompt(self, question: str) -> str:
        conv = self.base_conv.copy()
        if getattr(self.model.config, "mm_use_im_start_end", False):
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def predict(
        self,
        image_path: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> str:
        img = Image.open(image_path).convert("RGB")
        image_tensor = process_images([img], self.image_processor, self.model.config)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

        full_prompt = self._build_prompt(question)
        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_tokens,
                use_cache=True
            )

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text

    def normalize_answer(self, answer: str) -> str:
        import re
        import string
        
        answer = answer.lower().strip()
        answer = answer.translate(str.maketrans('', '', string.punctuation))
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        answer = ' '.join(answer.split())
        return answer

    def textvqa_accuracy(self, pred: str, gt_answers: list) -> bool:
        pred_norm = self.normalize_answer(pred)
        
        for ans in gt_answers:
            ans_norm = self.normalize_answer(ans)
            if ans_norm in pred_norm:
                return True
        
        return False

    def evaluate_textvqa(
        self,
        textvqa_root: str,
        split: str = "val",
        num_samples: int = None,
        output_file: str = None,
    ):
        data_file = os.path.join(textvqa_root, f"TextVQA_0.5.1_{split}.json")
        
        print(f"Loading TextVQA data from: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)['data']
        
        print(f"Loaded {len(data)} questions")
        print(f"Adversarial mode: {'ON' if self.use_adversarial else 'OFF'}\n")
        
        if num_samples is not None:
            data = data[:num_samples]
            print(f"Evaluating first {num_samples} questions\n")
        
        correct = 0
        total = 0
        results = []
        error_types = defaultdict(int)
        missing_images = []
        
        progress_bar = tqdm(data, desc="Evaluating TextVQA", total=len(data))
        
        for item in progress_bar:
            try:
                question_id = item['question_id']
                image_id = item['image_id']
                question = item['question']
                gt_answers = item['answers']
                
                image_path = self.get_image_path(textvqa_root, image_id)
                
                if not os.path.exists(image_path):
                    missing_images.append(image_id)
                    error_types['missing_image'] += 1
                    continue
                
                is_using_adv = self.use_adversarial and self.adversarial_dir and image_path.startswith(self.adversarial_dir)
                
                pred_answer = self.predict(image_path, question)

                is_correct = self.textvqa_accuracy(pred_answer, gt_answers)
                
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    'question_id': question_id,
                    'image_id': image_id,
                    'question': question,
                    'pred_answer': pred_answer,
                    'gt_answers': gt_answers,
                    'correct': is_correct,
                    'is_adversarial': is_using_adv,
                    'image_path': image_path
                })
                
                current_acc = correct / total * 100 if total > 0 else 0
                progress_bar.set_postfix({
                    'acc': f'{current_acc:.1f}%',
                    'correct': f'{correct}/{total}',
                    'adv': 'ON' if self.use_adversarial else 'OFF'
                })
                
                if total <= 5 or total % 100 == 0:
                    print(f"\n[Sample {total}]")
                    print(f"Image: {image_id} {'(ADV)' if is_using_adv else '(ORIG)'}")
                    print(f"Q: {question}")
                    print(f"Pred: {pred_answer}")
                    print(f"GT: {gt_answers}")
                    print(f"Result: {'✓ Correct' if is_correct else '✗ Wrong'}")
                
            except Exception as e:
                print(f"\nError processing {question_id}: {e}")
                error_types[str(type(e).__name__)] += 1
                continue
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        print("\n" + "="*60)
        print(f"TextVQA Evaluation Results ({split})")
        print("="*60)
        print(f"Adversarial mode: {'ON' if self.use_adversarial else 'OFF'}")
        print(f"Total evaluated: {total}")
        print(f"Correct: {correct}")
        print(f"Wrong: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("="*60)
        
        if error_types:
            print("\nErrors encountered:")
            for error, count in error_types.items():
                print(f"  {error}: {count}")
        
        if missing_images:
            print(f"\nMissing images: {len(missing_images)}")
        
        if output_file:
            output_data = {
                'config': {
                    'split': split,
                    'total_questions': len(data),
                    'evaluated': total,
                    'use_adversarial': self.use_adversarial,
                    'adversarial_dir': self.adversarial_dir if self.use_adversarial else None,
                },
                'metrics': {
                    'accuracy': accuracy,
                    'correct': correct,
                    'wrong': total - correct,
                    'total': total,
                },
                'errors': dict(error_types),
                'results': results
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nDetailed results saved to: {output_file}")
        
        return accuracy, results


def verify_textvqa_setup(textvqa_root):
    print("Verifying TextVQA dataset setup...")
    print(f"TextVQA Root: {textvqa_root}\n")
    
    if not os.path.exists(textvqa_root):
        print(f"❌ TextVQA root directory not found: {textvqa_root}")
        return False
    
    data_files = [
        "TextVQA_0.5.1_val.json",
        "TextVQA_0.5.1_train.json"
    ]
    
    found_files = []
    for filename in data_files:
        path = os.path.join(textvqa_root, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)['data']
            found_files.append((filename, len(data)))
            print(f"✓ Found {filename}: {len(data)} questions")
        else:
            print(f"✗ Not found: {filename}")
    
    if not found_files:
        print("\n❌ No data files found!")
        return False
    
    image_dirs = [
        os.path.join(textvqa_root, "train_val_images", "train_images"),
        os.path.join(textvqa_root, "train_val_images", "val_images"),
        os.path.join(textvqa_root, "train_images"),
        os.path.join(textvqa_root, "val_images"),
    ]
    
    found_images = False
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            num_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            print(f"✓ Found {img_dir.replace(textvqa_root, '.')}: {num_images} images")
            found_images = True
    
    if not found_images:
        print("✗ No image directories found!")
        return False
    
    print("\n✓ Setup verification complete!\n")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='TextVQA Evaluation with VisionZip')
    parser.add_argument('--dominant', type=int, default=14)
    parser.add_argument('--contextual', type=int, default=10)
    parser.add_argument('--zip', action='store_true', help=" VisionZip")
    parser.add_argument('--adversarial', action='store_true')
    parser.add_argument('--adversarial-dir', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=1000)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    TEXTVQA_ROOT = "./dataset/TextVQA"
    OUTPUT_DIR = "./output/textvqa_results"
    args = parse_args()

    SPLIT = "val"
    NUM_SAMPLES = args.num_samples
    
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if args.zip:
        config_name = f"zip_{args.dominant}_{args.contextual}"
    else:
        config_name = "baseline"
    if args.adversarial:
        config_name += "_adv"
    
    configs = [
        {
            "name": config_name,
            "use_visionzip": args.zip,
            "dominant": args.dominant,
            "contextual": args.contextual,
            "use_adversarial": args.adversarial,
            "adversarial_dir": args.adversarial_dir,
        },
    ]
    
    all_results = {}
    
    for config in configs:
        print("\n" + "="*60)
        print(f"Configuration: {config['name']}")
        print("="*60 + "\n")
        
        evaluator = TextVQAEvaluator(
            model_path="liuhaotian/llava-v1.5-7b",
            use_visionzip=config["use_visionzip"],
            dominant=config.get("dominant", 14),
            contextual=config.get("contextual", 10),
            use_adversarial=config.get("use_adversarial", False),
            adversarial_dir=config.get("adversarial_dir", None),
        )
        
        output_file = os.path.join(OUTPUT_DIR, f"textvqa_{config['name']}_{SPLIT}.json")
        
        try:
            accuracy, results = evaluator.evaluate_textvqa(
                textvqa_root=TEXTVQA_ROOT,
                split=SPLIT,
                num_samples=NUM_SAMPLES,
                output_file=output_file,
            )
            
            all_results[config['name']] = accuracy
            
        except Exception as e:
            print(f"\n❌ Error in {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for name, acc in all_results.items():
        print(f"{name:20s}: {acc:6.2f}%")
    print("="*60)
    
    summary_file = os.path.join(OUTPUT_DIR, f"summary_{SPLIT}.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'split': SPLIT,
            'num_samples': NUM_SAMPLES,
            'results': all_results
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")