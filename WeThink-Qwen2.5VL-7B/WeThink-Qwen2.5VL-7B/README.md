---
license: apache-2.0
tags:
- Reinforcement Learning
- Visual-langauge Reasoning
---
# Model Card for WeThink-Qwen2.5VL-7B

Repository: https://github.com/yangjie-cv/WeThink

Paper: https://arxiv.org/abs/2506.07905



## 🏆 Performance Highlights
**WeThink-Qwen2.5VL-7B** achieves:
- 🥇 **1st place** on [OpenCompass Multimodal Reasoning Leaderboard](https://rank.opencompass.org.cn/leaderboard-multimodal-reasoning/?m=REALTIME)
- 🏅 **5th place** on [OpenCompass Multi-modal Academic Leaderboard](https://rank.opencompass.org.cn/leaderboard-multimodal/?m=REALTIME)  
*(As of May 30th, 2025)*


## 🚀 Quick Start
### Inference

```bash
git clone https://github.com/yangjie-cv/WeThink
cd WeThink
python inference.py
```
💡 ​​Note​​: System prompt is required during inference. 

### 📊 Evaluation
We have integrated WeThink-Qwen2.5VL-7B into the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). Please follow its [Quickstart guide](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) to evaluate WeThink-Qwen2.5VL-7B on various benchmarks.


## Citation
```
@misc{yang2025wethink,
      title={WeThink: Toward General-purpose Vision-Language Reasoning via Reinforcement Learning}, 
      author={Jie Yang and Feipeng Ma and Zitian Wang and Dacheng Yin and Kang Rong and Fengyun Rao and Ruimao Zhang},
      year={2025},
      eprint={2506.07905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.07905}, 
}
```