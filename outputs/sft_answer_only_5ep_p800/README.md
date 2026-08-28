---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
model_name: sft_answer_only_5ep_p800
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for sft_answer_only_5ep_p800

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with SFT.

### Framework versions

- PEFT 0.18.1
- TRL: 0.29.0
- Transformers: 4.57.1
- Pytorch: 2.5.1+cu118
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```