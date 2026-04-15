# Falcon 3

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/TII - 2024 - Falcon 3.html`
- Source URL: https://huggingface.co/tiiuae/Falcon3-7B-Instruct
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

Hugging Face

- Models

- Datasets

- Spaces

- Buckets new

- Docs

- Enterprise

- Pricing

- Log In

- Sign Up

/

Falcon3-7B-Instruct

like 78

Follow

Transformers

Safetensors

4 languages

llama

falcon3

conversational

text-generation-inference

License: falcon-llm-license

Model card Files Files and versions

xet

12

Deploy

Use this model

- Falcon3-7B-Instruct

- Model Details

- Getting started

- Benchmarks

- Useful links

- Technical Report

- Citation

# Falcon3-7B-Instruct

Falcon3 family of Open Foundation Models is a set of pretrained and instruct LLMs ranging from 1B to 10B.

This repository contains the Falcon3-7B-Instruct. It achieves state of art results (at the time of release) on reasoning, language understanding, instruction following, code and mathematics tasks.
Falcon3-7B-Instruct supports 4 languages (english, french, spanish, portuguese) and a context length up to 32K.

## Model Details

- Architecture

- Transformer based causal decoder only architecture

- 28 decoder blocks

- Grouped query attention (GQA) for faster inference: 12 query heads and 4 key value heads

- Wider head dimension: 256

- High RoPE value to support long context understanding: 1000042

- Uses SwiGLU and RMSNorm

- 32K context length

- 131K vocab size

- Pretrained on 14 Teratokens of datasets comprising of web, code, STEM, high quality and mutlilingual data using 1024 H100 GPU chips

- Postrained on 1.2 million samples of STEM, conversations, code, safety and function call data

- Supports EN, FR, ES, PT

- Developed by Technology Innovation Institute

- License: TII Falcon-LLM License 2.0

- Model Release Date: December 2024

## Getting started

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tiiuae/Falcon3-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
 model_name,
 torch_dtype="auto",
 device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many hours in one day?"
messages = [
 {"role": "system", "content": "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible."},
 {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
 messages,
 tokenize=False,
 add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
 **model_inputs,
 max_new_tokens=1024
)
generated_ids = [
 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Benchmarks

We report the official HuggingFace leaderboard normalized evaluations Open LLM Leaderboard Evaluation Results in the following table.

Benchmark
 Llama-3.1-8B-Instruct
 Qwen2.5-7B-Instruct
 Falcon3-7B-Instruct

IFEval
 78.56
 75.85
 76.12

BBH (3-shot)
 29.89
 34.89
 37.92

MATH Lvl-5 (4-shot)
 19.34
 0.00
 31.87

GPQA (0-shot)
 2.35
 5.48
 8.05

MUSR (0-shot)
 8.41
 8.45
 21.17

MMLU-PRO (5-shot)
 30.68
 36.52
 34.30

Also, we report in the following table our internal pipeline benchmarks.

- We use lm-evaluation harness.

- We report raw scores obtained by applying chat template and fewshot_as_multiturn.

- We use same batch-size across all models.

Category
 Benchmark
 Llama-3.1-8B-Instruct
 Qwen2.5-7B-Instruct
 Falcon3-7B-Instruct

General
 MMLU (5-shot)
 68.2
 73.5
 70.5

MMLU-PRO (5-shot)
 36.4
 43.1
 40.7

IFEval
 78.8
 74.7
 76.5

Math
 GSM8K (5-shot)
 82.6
 72.0
 81.4

GSM8K (8-shot, COT)
 85.4
 76.6
 79.7

MATH Lvl-5 (4-shot)
 15.4
 -
 29.4

Reasoning
 Arc Challenge (25-shot)
 58.6
 57.8
 62.6

GPQA (0-shot)
 33.5
 32
 31.9

GPQA (0-shot, COT)
 9.6
 13.8
 22.3

MUSR (0-shot)
 38.6
 41
 46.4

BBH (3-shot)
 48.6
 54.1
 52.4

CommonSense Understanding
 PIQA (0-shot)
 78.9
 73.7
 78.8

SciQ (0-shot)
 80.2
 50.9
 94.7

Winogrande (0-shot)
 -
 -
 70.4

OpenbookQA (0-shot)
 46.2
 42.4
 45.8

Instructions following
 MT-Bench (avg)
 7.9
 8.5
 8.4

Alpaca (WC)
 26.6
 31.5
 26.1

Tool use
 BFCL AST (avg)
 90.6
 91.4
 89.5

## Useful links

- View our release blogpost.

- Feel free to join our discord server if you have any questions or to interact with our researchers and developers.

## Technical Report

Coming soon....

## Citation

If Falcon3 family were helpful to your work, feel free to give us a cite.

```
@misc{Falcon3,
 title = {The Falcon 3 family of Open Models},
 author = {TII Team},
 month = {December},
 year = {2024}
}
```

Downloads last month 20,536

Safetensors

Model size

7B params

Tensor type

BF16

·

Chat template

Files info

Inference Providers NEW

Text Generation

This model isn't deployed by any Inference Provider. 🙋 Ask for provider support

## Model tree for tiiuae/Falcon3-7B-Instruct

Base model

Finetuned

this model

Adapters

Finetunes

Merges

Quantizations

## Spaces using tiiuae/Falcon3-7B-Instruct 40

tiiuae/Falcon3-demo

💥

pliny-the-prompter/obliteratus

🏆

eduagarcia/open_pt_llm_leaderboard

👀

BaggerOfWords/MOSAIC

🔓

dromero14521/obliteratus

💥

cw4444/obliteratus

💥

GrozniMilitq/obliteratus

💥

dumbordumber/obliteratus

## Collection including tiiuae/Falcon3-7B-Instruct

#### Falcon3

Collection

Falcon3 family of Open Foundation Models is a set of pretrained and instruct LLMs ranging from 1B to 10B parameters. • 40 items • Updated Nov 6, 2025 • 93

System theme

Company

Website
