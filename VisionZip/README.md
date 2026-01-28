<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/visionzip.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# VisionZip: Longer is Better but Not Necessary in Vision Language Models


[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-light)](https://arxiv.org/abs/2412.04467)
[![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Discussion-orange)](https://huggingface.co/papers/2412.04467)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/dvlab-research/VisionZip/blob/main/LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Chat-red.svg)](http://202.104.135.156:7860/)
[![Demo](https://img.shields.io/badge/Demo-Visualize%20-green)](http://202.104.135.156:11030/)
<a href='https://huggingface.co/spaces/Senqiao/VisionZip'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Video](#video)
4. [Demo](#demo)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Evaluation](#evaluation)
8. [Examples](#examples)
9. [Citation](#citation)
10. [Acknowledgement](#acknowledgement)
11. [License](#license)
      
## News
- [x] [2025.07.20] Released [VisionThink](https://github.com/dvlab-research/VisionThink) — an exploration of **Efficient Reasoning VLM**, maintaining strong OCR performance via RL-driven token reduction.
- [x] [2025.05.26] VisionZip for Qwen2.5VL is now released! See details [here](https://github.com/dvlab-research/VisionZip/tree/main/Qwen2_5_VL).
- [x] [2025.02.27] VisionZip has been accepted by **CVPR 2025**. :rocket:
- [x] [2024.12.28] With support from Hugging Face, we add our demo on the [Hugging Face Space](https://huggingface.co/spaces/Senqiao/VisionZip), allowing for easy comparison of output results across different model sizes.
- [x] [2024.12.16] Due to positive feedback on our demo, we have released the VisionZip [Demo-Chat](http://202.104.135.156:7860/) code in a new branch, 'demo-chat'.
- [x] [2024.12.05] We add an [Usage-Video](https://youtu.be/9GNIJy4U6-k?si=jcWIJ2O0IjB4aamm), providing a step-by-step guide on how to use the demo.
- [x] [2024.12.05] We add a new [Demo-Chat](http://202.104.135.156:7860/), where users can manually select visual tokens to send to the LLM and observe how different visual tokens affect the final response. We believe this will further enhance the analysis of VLM interpretability.
- [x] [2024.11.30] We release [Paper](https://arxiv.org/abs/2412.04467) and this GitHub repo, including code for LLaVA.

**VisionZip: Longer is Better but Not Necessary in Vision Language Models [[Paper](https://arxiv.org/abs/2412.04467)]** <br />
[Senqiao Yang](https://scholar.google.com/citations?user=NcJc-RwAAAAJ),
[Yukang Chen](https://scholar.google.com/citations?user=6p0ygKUAAAAJ),
[Zhuotao Tian](https://scholar.google.com/citations?user=mEjhz-IAAAAJ),
[Chengyao Wang](https://scholar.google.com.hk/citations?user=1pZcoqgAAAAJ),
[Jingyao Li](https://scholar.google.com/citations?user=mqrKmvcAAAAJ),
[Bei Yu](https://scholar.google.com/citations?user=tGneTm4AAAAJ),
[Jiaya Jia](https://scholar.google.com/citations?user=XPAkzTEAAAAJ)<br />

## Highlights
<p align="center" width="80%">
<img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/Teaser.png" alt="Stanford-Alpaca" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

1. Our VisionZip achieves state-of-the-art performance among efficient VLM methods. By retaining only **10%** of visual tokens, it achieves nearly **95%** of the performance in training-free mode.
2. VisionZip can be applied during the inference stage (without incurring any additional training cost), the efficient tuning stage (to achieve better results), and the training stage (**almost no performance degradation，saving 2× memory and 2× training time**).
3. VisionZip significantly reduces the prefilling time and the total inference time (with KV cache enabled).
4. Why does this simple, text-agnostic method outperform text-relevant methods? We conduct an in-depth analysis in the [paper](https://arxiv.org/abs/2412.04467) and provide a [demo](http://202.104.135.156:7860/) to visualize these findings.
5. Since VisionZip is a text-agnostic method that reduces visual tokens before input into the LLM, it can adapt to **any** existing LLM acceleration algorithms and is applicable to any task that a vanilla VLM can perform, such as multi-turn conversations.

## Video
<p align="center" width="80%">
  <a href="https://youtu.be/sytaAzmxxpo?si=IieArmQ7YNf2dVyM" target="_blank">
    <img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/VisionZip-youtube-video.png" alt="Stanford-Alpaca" style="width: 80%; min-width: 300px; display: block; margin: auto;">
  </a>
</p>

## Demo
### Speed Improvement
The input [video](https://www.youtube.com/watch?v=I7c1etV7D7g
) is about the Titanic, and the question is, "What’s the video talking about?"


<p align="center" width="80%">
  <a href="https://www.youtube.com/watch?v=I7c1etV7D7g" target="_blank">
    <img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/titanic.png" alt="Stanford-Alpaca" style="width: 80%; min-width: 300px; display: block; margin: auto;">
  </a>
</p>

It is important to note that the left side shows the vanilla model, which encodes only 16 frames, while the right side shows our VisionZip, which, despite encoding **32 frames**, is still **twice** as fast as the vanilla model.


<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/speed.gif" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


### Visualize Redundancy and Misalignment
<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/gradio.png" alt="Stanford-Alpaca" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

Explore the visual redundancy and feature misalignment in the above [Demo](http://202.104.135.156:7860/). To run it locally, use the following command:
```
python gradio_demo.py 
```

### Observe How Different Visual Tokens Impact the Final Response
This [Demo-Chat](http://202.104.135.156:7860/) lets users to manually select which visual tokens to send to the LLM and observe how different visual tokens affect the final response.


## Installation
Our code is easy to use.

1. Install the [LLaVA](https://github.com/haotian-liu/LLaVA) environment.

2. For formal usage, you can install the package from PyPI by running the following command:
```
pip install visionzip
```

For development, you can install the package by cloning the repository and running the following command:
```
git clone https://github.com/dvlab-research/VisionZip
cd VisionZip
pip install -e .
```

## Quick Start
```Python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from visionzip import visionzip
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
## VisoinZip retains 54 dominant tokens and 10 contextual tokens
model = visionzip(model, dominant=54, contextual=10)
```



## Evaluation
The evaluation code follows the structure of [LLaVA](https://github.com/haotian-liu/LLaVA) or [Lmms-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). After loading the model, simply add two lines as shown below:

```python
## Load LLaVA Model (code from llava.eval.model_vqa_loader)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
## add VisionZip
from visionzip import visionzip
model = visionzip(model, dominant=54, contextual=10)
```

## Examples
### Multi-turn Conversations
VisionZip, which extracts text-agnostic tokens, is better suited for multi-turn dialogue.

<p align="center"> <img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/conversation.png" width="80%"> </p>

### Longer Videos with More Frames
VisionZip reduces the number of visual tokens per frame, allowing more frames to be processed. This improves the model's ability to understand longer videos.
<p align="center"> <img src="https://raw.githubusercontent.com/dvlab-research/VisionZip/main/imgs/longer-video.png" width="80%"> </p>

## Citation
If you find this project useful in your research, please consider citing:

> **Note:** *VisionThink* is our new exploration of Efficient Reasoning VLMs, designed to maintain strong OCR capabilities via RL-driven token reduction. Check it out [here](https://github.com/dvlab-research/VisionThink).

```
@article{yang2024visionzip,
  title={VisionZip: Longer is Better but Not Necessary in Vision Language Models},
  author={Yang, Senqiao and Chen, Yukang and Tian, Zhuotao and Wang, Chengyao and Li, Jingyao and Yu, Bei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2412.04467},
  year={2024}
}
@article{yang2025visionthink,
  title={VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning},
  author={Yang, Senqiao and Li, Junyi and Lai, Xin and Yu, Bei and Zhao, Hengshuang and Jia, Jiaya},
  journal={arXiv preprint arXiv:2507.13348},
  year={2025}
}
```

## Acknowledgement
- This work is built upon [LLaVA](https://llava-vl.github.io/), [mini-Gemini](https://github.com/dvlab-research/MGM), [Lmms-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), and [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA). We thank them for their excellent open-source contributions.

- We also thank [StreamingLLM](https://github.com/mit-han-lab/streaming-llm), [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), and others for their contributions, which have provided valuable insights.

## License
- VisionZip is licensed under the Apache License 2.0. 
