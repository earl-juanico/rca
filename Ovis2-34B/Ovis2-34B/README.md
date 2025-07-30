---
license: apache-2.0
datasets:
- AIDC-AI/Ovis-dataset
library_name: transformers
tags:
- MLLM
pipeline_tag: image-text-to-text
language:
- en
- zh
---

# Ovis2-34B
<div align="center">
  <img src=https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png width="30%"/>
</div>

## Introduction
[GitHub](https://github.com/AIDC-AI/Ovis) | [Paper](https://arxiv.org/abs/2405.20797) 

We are pleased to announce the release of **Ovis2**, our latest advancement in multi-modal large language models (MLLMs). Ovis2 inherits the innovative architectural design of the Ovis series, aimed at structurally aligning visual and textual embeddings. As the successor to Ovis1.6, Ovis2 incorporates significant improvements in both dataset curation and training methodologies.

**Key Features**:

- **Small Model Performance**: Optimized training strategies enable small-scale models to achieve higher capability density, demonstrating cross-tier leading advantages.

- **Enhanced Reasoning Capabilities**: Significantly strengthens Chain-of-Thought (CoT) reasoning abilities through the combination of instruction tuning and preference learning.

- **Video and Multi-Image Processing**: Video and multi-image data are incorporated into training to enhance the ability to handle complex visual information across frames and images.

- **Multilingual Support and OCR**: Enhances multilingual OCR beyond English and Chinese and improves structured data extraction from complex visual elements like tables and charts.

<div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/XB-vgzDL6FshrSNGyZvzc.png" width="100%" />
</div>

## Model Zoo

| Ovis MLLMs |           ViT           |          LLM          |                      Model Weights                      |                           Demo                           |
|:-----------|:-----------------------:|:---------------------:|:-------------------------------------------------------:|:--------------------------------------------------------:|
| Ovis2-1B   | aimv2-large-patch14-448 | Qwen2.5-0.5B-Instruct | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-1B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-1B)  |
| Ovis2-2B   | aimv2-large-patch14-448 | Qwen2.5-1.5B-Instruct | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-2B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-2B)  |
| Ovis2-4B   | aimv2-huge-patch14-448  |  Qwen2.5-3B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-4B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-4B)  |
| Ovis2-8B   | aimv2-huge-patch14-448  |  Qwen2.5-7B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-8B)  | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-8B)  |
| Ovis2-16B  | aimv2-huge-patch14-448  | Qwen2.5-14B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-16B) | [Space](https://huggingface.co/spaces/AIDC-AI/Ovis2-16B) |
| Ovis2-34B  |  aimv2-1B-patch14-448   | Qwen2.5-32B-Instruct  | [Huggingface](https://huggingface.co/AIDC-AI/Ovis2-34B) |                            -                             |

## Performance
We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), as employed in the OpenCompass [multimodal](https://rank.opencompass.org.cn/leaderboard-multimodal) and [reasoning](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning) leaderboard, to evaluate Ovis2.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/M1XRFbeNbfe1lEvt9WF-j.png)

### Image Benchmark
| Benchmark                    | Qwen2.5-VL-72B   | InternVL2.5-38B-MPO   | InternVL2.5-26B-MPO   |   Ovis1.6-27B | LLaVA-OV-72B          | Ovis2-16B   | Ovis2-34B   |
|:-----------------------------|:----------------:|:---------------------:|:---------------------:|:-------------:|:---------------------:|:-----------:|:-----------:|
| MMBench-V1.1<sub>test</sub>  | **87.8**         | 85.4                  | 84.2                  |          82.2 | 84.4                  | 85.6        | 86.6        |
| MMStar                       | **71.1**         | 70.1                  | 67.7                  |          63.5 | 65.8                  | 67.2        | 69.2        |
| MMMU<sub>val</sub>           | **67.9**         | 63.8                  | 56.4                  |          60.3 | 56.6                  | 60.7        | 66.7        |
| MathVista<sub>testmini</sub> | 70.8             | 73.6                  | 71.5                  |          70.2 | 68.4                  | 73.7        | **76.1**    |
| HallusionBench               | 58.8             | **59.7**              | 52.4                  |          54.1 | 47.9                  | 56.8        | 58.8        |
| AI2D                         | 88.2             | 87.9                  | 86.2                  |          86.6 | 86.2                  | 86.3        | **88.3**    |
| OCRBench                     | 88.1             | 89.4                  | **90.5**              |          85.6 | 74.1                  | 87.9        | 89.4        |
| MMVet                        | 76.7             | 72.6                  | 68.1                  |          68   | 60.6                  | 68.4        | **77.1**    |
| MMBench<sub>test</sub>       | **88.2**         | 86.4                  | 85.4                  |          84.6 | 85.6                  | 87.1        | 87.8        |
| MMT-Bench<sub>val</sub>      | 69.1             | 69.1                  | 65.7                  |          68.2 | -                     | 69.2        | **71.2**    |
| RealWorldQA                  | **75.9**         | 74.4                  | 73.7                  |          72.7 | 73.9                  | 74.1        | 75.6        |
| BLINK                        | 62.3             | **63.2**              | 62.6                  |          48   | -                     | 59.0        | 60.1        |
| QBench                       | -                | 76.1                  | 76.0                  |          77.7 | -                     | 79.5        | **79.8**    |
| ABench                       | -                | 78.6                  | **79.4**              |          76.5 | -                     | **79.4**    | 78.7        |
| MTVQA                        | -                | **31.2**              | 28.7                  |          26.5 | -                     | 30.3        | 30.6        |

### Video Benchmark
| Benchmark           | Qwen2.5-VL-72B | InternVL2.5-38B | InternVL2.5-26B | LLaVA-OneVision-72B | Ovis2-16B | Ovis2-34B     |
|:--------------------|:--------------:|:---------------:|:---------------:|:-------------------:|:---------:|:-------------:|
| VideoMME(wo/w-subs) | **73.3/79.1**  | 70.7 / 73.1     | 66.9 / 69.2     | 66.2/69.5           | 70.0/74.4 | 71.2/75.6     |
| MVBench             | 70.4           | 74.4            | **75.2**        | 59.4                | 68.6      | 70.3          |
| MLVU(M-Avg/G-Avg)   | 74.6/-         | 75.3/-          | 72.3/-          | 68.0/-              | 77.7/4.44 | **77.8**/4.59 |
| MMBench-Video       | **2.02**       | 1.82            | 1.86            | -                   | 1.92      | 1.98          |
| TempCompass         | 74.8           | -               | -               | -                   | 74.16     | **75.97**     |

### Reasoning Benchmark
![image/png](https://cdn-uploads.huggingface.co/production/uploads/658a8a837959448ef5500ce5/aGc1DZLPEeERd0jGW9JRA.png)

## Usage
Below is a code snippet demonstrating how to run Ovis with various input types. For additional usage instructions, including inference wrapper and Gradio UI, please refer to [Ovis GitHub](https://github.com/AIDC-AI/Ovis?tab=readme-ov-file#inference).
```bash
pip install torch==2.4.0 transformers==4.46.2 numpy==1.25.0 pillow==10.3.0
pip install flash-attn==2.7.0.post2 --no-build-isolation
```
```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-34B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# single-image input
image_path = '/data/images/example_1.jpg'
images = [Image.open(image_path)]
max_partition = 9
text = 'Describe the image.'
query = f'<image>\n{text}'

## cot-style input
# cot_suffix = "Provide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
# image_path = '/data/images/example_1.jpg'
# images = [Image.open(image_path)]
# max_partition = 9
# text = "What's the area of the shape?"
# query = f'<image>\n{text}\n{cot_suffix}'

## multiple-images input
# image_paths = [
#     '/data/images/example_1.jpg',
#     '/data/images/example_2.jpg',
#     '/data/images/example_3.jpg'
# ]
# images = [Image.open(image_path) for image_path in image_paths]
# max_partition = 4
# text = 'Describe each image.'
# query = '\n'.join([f'Image {i+1}: <image>' for i in range(len(images))]) + '\n' + text

## video input (require `pip install moviepy==1.0.3`)
# from moviepy.editor import VideoFileClip
# video_path = '/data/videos/example_1.mp4'
# num_frames = 12
# max_partition = 1
# text = 'Describe the video.'
# with VideoFileClip(video_path) as clip:
#     total_frames = int(clip.fps * clip.duration)
#     if total_frames <= num_frames:
#         sampled_indices = range(total_frames)
#     else:
#         stride = total_frames / num_frames
#         sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
#     frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
#     frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
# images = frames
# query = '\n'.join(['<image>'] * len(images)) + '\n' + text

## text-only input
# images = []
# max_partition = None
# text = 'Hello'
# query = text

# format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
if pixel_values is not None:
    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
pixel_values = [pixel_values]

# generate output
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'Output:\n{output}')
```

<details>
<summary>Batch Inference</summary>

```python
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-34B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# preprocess inputs
batch_inputs = [
    ('/data/images/example_1.jpg', 'What colors dominate the image?'),
    ('/data/images/example_2.jpg', 'What objects are depicted in this image?'),
    ('/data/images/example_3.jpg', 'Is there any text in the image?')
]

batch_input_ids = []
batch_attention_mask = []
batch_pixel_values = []

for image_path, text in batch_inputs:
    image = Image.open(image_path)
    query = f'<image>\n{text}'
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image], max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    batch_input_ids.append(input_ids.to(device=model.device))
    batch_attention_mask.append(attention_mask.to(device=model.device))
    batch_pixel_values.append(pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device))

batch_input_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_input_ids], batch_first=True,
                                                  padding_value=0.0).flip(dims=[1])
batch_input_ids = batch_input_ids[:, -model.config.multimodal_max_length:]
batch_attention_mask = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_attention_mask],
                                                       batch_first=True, padding_value=False).flip(dims=[1])
batch_attention_mask = batch_attention_mask[:, -model.config.multimodal_max_length:]

# generate outputs
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(batch_input_ids, pixel_values=batch_pixel_values, attention_mask=batch_attention_mask,
                                **gen_kwargs)

for i in range(len(batch_inputs)):
    output = text_tokenizer.decode(output_ids[i], skip_special_tokens=True)
    print(f'Output {i + 1}:\n{output}\n')
```
</details>

## Citation
If you find Ovis useful, please consider citing the paper
```
@article{lu2024ovis,
  title={Ovis: Structural Embedding Alignment for Multimodal Large Language Model},
  author={Shiyin Lu and Yang Li and Qing-Guo Chen and Zhao Xu and Weihua Luo and Kaifu Zhang and Han-Jia Ye},
  year={2024},
  journal={arXiv:2405.20797}
}
```

## License
This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) (SPDX-License-Identifier: Apache-2.0).

## Disclaimer
We used compliance-checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to the complexity of the data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.