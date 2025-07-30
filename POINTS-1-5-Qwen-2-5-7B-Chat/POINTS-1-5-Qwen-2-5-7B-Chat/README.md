---
license: apache-2.0
language:
- en
- zh
base_model:
- Qwen/Qwen2.5-7B-Instruct
---
## POINTS-1-5-Qwen-2-5-7B-Chat

### Introduction

We are excited to release the latest update of WePOINTS series, namely POINTS1.5, a much stronger model than POINTS and integrating recent advancement in vision-language model and new techniques proposed by researchers from WeChat AI.
Notably, POINTS-1-5-Qwen-2-5-7B-Chat ranks **first** on the [OpenCompass](https://rank.opencompass.org.cn/leaderboard-multimodal) leaderboard among all models under 10B.

<p align="center">
        🏠 <a href="https://github.com/WePOINTS/WePOINTS">Github</a>&nbsp&nbsp |  &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2412.08443">Paper</a> &nbsp&nbsp  </a>
</p>

### What's new in POINTS1.5?

**Key Innovations**

1. **Native Dynamic High Resolution**: In line with the recent trend in vision-language models, we have replaced the original CLIP vision encoder with a NaViT-style vision encoder. This new encoder can process images at various resolutions without the need for splitting.

2. **Bilingual Support**: Most of the pre-training and visual instruction tuning datasets in POINTS are in English. In this update, we have added support for Chinese, with plans to include more languages in the future. For the pre-training stage, we followed the strategy proposed by POINTS and created an additional 1 million Chinese pre-training datasets. For the visual instruction tuning stage, we supplemented the original English dataset used in POINTS with a series of Chinese visual instruction tuning datasets sourced from the open-source community. We also collected images and generated corresponding textual question-and-answer pairs using a combination of manual and automated methods. These visual instruction tuning datasets cover various domains, such as optical character recognition and general conversation.

2. **Quality Control**: We conducted a series of quality control tests on both the pre-training and visual instruction tuning datasets. For instance, we filtered the pre-training dataset using perplexity, following the strategy proposed in POINTS. For the visual instruction tuning datasets, we implemented a combination of filtering strategies, such as removing samples with grammatical errors.

3. **Model Soup**: In line with POINTS, we also applied model soup techniques to further enhance performance.


<div style="display: flex; justify-content: space-between; gap: 5px;">
  <img src="https://github.com/user-attachments/assets/a2fd1f54-e36c-45ea-870e-b5be07310e29" alt="model development" style="width: 48%;"/>
  <img src="https://github.com/user-attachments/assets/c1c5c55e-bcce-4187-b167-084868be99d8" alt="model architecture" style="width: 48%;"/>
</div>


### Prepare the environment

```bash
pip install torch>=2.4.1
pip install transformers>=4.46.3
git clone https://github.com/WePOINTS/WePOINTS.git
cd WePOINTS
pip install -e .
```


### How to use POINTS1.5?

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from wepoints.utils.images import Qwen2ImageProcessorForPOINTSV15
import torch
from PIL import Image
import requests
from io import BytesIO


model_path = 'WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat'
model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 device_map='cuda') 
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
image_processor = Qwen2ImageProcessorForPOINTSV15.from_pretrained(model_path)


image_url = 'https://github.com/user-attachments/assets/83258e94-5d61-48ef-a87f-80dd9d895524'
response = requests.get(image_url)
image_data = BytesIO(response.content)
pil_image = Image.open(image_data)
pil_image = pil_image.save('image.jpg')
prompt = 'please describe the image in detail'

content = [
        dict(type='image', image='image.jpg'),
        dict(type='text', text=prompt)
    ]
messages = [
        {
            'role': 'user',
            'content': content
        }
    ]
generation_config = {
        'max_new_tokens': 1024,
        'temperature': 0.0,
        'top_p': 0.0,
        'num_beams': 1,
    }
response = model.chat(
    messages,
    tokenizer,
    image_processor,
    generation_config
)
print(response)
```

### Evaluation

| Benchmark | Qwen2-VL-7B | POINTS-7B | POINTS1.5-7B |
| :-------: | :----------: | :-------------: | :----: |
| MMBench-TEST-avg      | 81.0 | 78.0 | 80.7 |
| MMStar                | 60.7 | 60.9 | 61.1 |
| MMMU                  | 53.7 | 51.4 | 53.8 |
| MathVista             | 61.4 | 63.0 | 66.4 |
| HallucinationBench    | 50.4 | 45.6 | 50.0 |
| AI2D                  | 83.0 | 81.2 | 81.4 |
| OCRBench              | 84.3 | 71.7 | 82.3 | 
| MMVet                 | 61.8 | 47.9 | 62.2 |
| Average               | 67.0 | 62.5 | 67.4 |

All results are taken from [OpenCompass](https://rank.opencompass.org.cn/leaderboard-multimodal)


### License Agreement

All of our open-source models are licensed under the Apache-2.0 license.


### Citation

If you find our work helpful, feel free to cite us:

```
@article{liu2024points1,
  title={POINTS1. 5: Building a Vision-Language Model towards Real World Applications},
  author={Liu, Yuan and Tian, Le and Zhou, Xiao and Gao, Xinyu and Yu, Kavio and Yu, Yang and Zhou, Jie},
  journal={arXiv preprint arXiv:2412.08443},
  year={2024}
}

@article{liu2024points,
  title={POINTS: Improving Your Vision-language Model with Affordable Strategies},
  author={Liu, Yuan and Zhao, Zhongyin and Zhuang, Ziyuan and Tian, Le and Zhou, Xiao and Zhou, Jie},
  journal={arXiv preprint arXiv:2409.04828},
  year={2024}
}

@article{liu2024rethinking,
  title={Rethinking Overlooked Aspects in Vision-Language Models},
  author={Liu, Yuan and Tian, Le and Zhou, Xiao and Zhou, Jie},
  journal={arXiv preprint arXiv:2405.11850},
  year={2024}
}
```