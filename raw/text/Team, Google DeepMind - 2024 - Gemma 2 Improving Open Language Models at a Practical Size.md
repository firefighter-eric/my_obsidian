# [2408.00118] Gemma 2: Improving Open Language Models at a Practical Size

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Team, Google DeepMind - 2024 - Gemma 2 Improving Open Language Models at a Practical Size.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2408.00118
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\authfootnotetext

1See Contributions and Acknowledgments section for full author list. Please send correspondence to gemma-2-report@google.com.

# Gemma 2: Improving Open Language Models at a Practical Size

Gemma Team

 Google DeepMind\authfootnotemark1

###### Abstract

In this work, we introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging in scale from 2 billion to 27 billion parameters.
In this new version, we apply several known technical modifications to the Transformer architecture, such as interleaving local-global attentions (Beltagy et al., 2020a) and group-query attention (Ainslie et al., 2023).
We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction.
The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3×\times bigger.
We release all our models to the community.

## 1 Introduction

Large language models (LLMs) have demonstrated strong capabilities in language understanding, generation, and reasoning (Radford et al., 2019; Raffel et al., 2019; Brown et al., 2020).
Scaling has been key to this recent progress, with many new capabilities only emerging at scale (Brown et al., 2020).
The newest large models not only reach unprecedented performance on reasoning benchmarks (Achiam et al., 2023), but they also demonstrate multimodal and multilingual capabilities (Gemini Team, 2024) and even the ability to use context lengths of over 1M tokens (Gemini Team, 2024).

Small-scale models have also shown a rapid increase in performance, but these gains are largely derived from increasing the length of training (Touvron et al., 2023; Jiang et al., 2023; Gemma Team, 2024).
This approach only scales logarithmically with dataset size (Hoffmann et al., 2022), and the latest small models require up to 15T tokens to improve the state of the art by less than 1-2% (AI@Meta, 2024).

Yet, these continued improvements provide evidence that small models are still under-trained.
In this work, we explore alternatives to improve small model performance without solely increasing training length. One solution is to improve the quality of information received by the network at each training step by replacing the next token prediction task with a richer objective.

In particular, we focus our efforts on knowledge distillation (Hinton et al., 2015), which replaces the one-hot vector seen at each token with the distribution of potential next tokens computed from a large model.
This approach is often used to reduce the training time of smaller models by giving them richer gradients.
In this work, we instead train for large quantities of tokens with distillation in order to simulate training beyond the number of available tokens.
Concretely, we use a large language model as a teacher to train small models, namely 2B and 9B models, on a quantity of tokens that is more than 50×\times the compute-optimal quantity predicted by the theory (Hoffmann et al., 2022).
Along with the models trained with distillation, we also release a 27B model trained from scratch for this work.

We also leverage several known modifications of Transformers, namely the interleaving of global and local attention layers from Beltagy et al. (2020a), and the Grouped-Query Attention (GQA) mechanism of Ainslie et al. (2023).

Overall, Gemma 2 significantly advances state-of-the-art performance relative to comparable-scale open models and are even competitive with some models more than twice their size (xAI, 2024; AI@Meta, 2024; Jiang et al., 2023; Almazrouei et al., 2023), across a variety of automated benchmarks and human evaluations. Example domains include question answering (Clark et al., 2019; Kwiatkowski et al., 2019), commonsense reasoning (Sakaguchi et al., 2019; Suzgun et al., 2022), mathematics and science (Cobbe et al., 2021; Hendrycks et al., 2020), and coding (Austin et al., 2021; Chen et al., 2021).

While thorough testing of our models has been conducted, these tests cannot cover all applications and scenarios in which Gemma 2 may be used. With this in mind, all Gemma 2 users should conduct rigorous safety testing specific to their use case before deployment or use.

In this technical report, we provide an overview of models, including the architecture, training, and pre- and post-training recipes for Gemma 2. We also provide detailed evaluations across a wide variety of quantitative and qualitative benchmarks, as well as both standard academic benchmarks and human-preference evaluations. Finally, we discuss our approach to safe and responsible deployment and outline the broader implications of Gemma 2, its limitations, and advantages.

Parameters
2B
9B
27B

d_model
2304
3584
4608

Layers
26
42
46

Pre-norm
yes
yes
yes

Post-norm
yes
yes
yes

Non-linearity
GeGLU
GeGLU
GeGLU

Feedforward dim
18432
28672
73728

Head type
GQA
GQA
GQA

Num heads
8
16
32

Num KV heads
4
8
16

Head size
256
256
128

Global att. span
8192
8192
8192

Sliding window
4096
4096
4096

Vocab size
256128
256128
256128

Tied embedding
yes
yes
yes

## 2 Model Architecture

Similar to previous Gemma models (Gemma Team, 2024), the Gemma 2 models are based on a decoder-only transformer architecture (Vaswani et al., 2017).
We summarize the main parameters and architecture choices in Table 1.

A few architectural elements are similar to the first version of Gemma models; namely, a context length of 8192 tokens, the use of Rotary Position Embeddings (RoPE) (Su et al., 2021), and the approximated GeGLU non-linearity (Shazeer, 2020). A few elements differ between Gemma 1 and Gemma 2, including using deeper networks. We summarize the key differences below.

Local Sliding Window and Global Attention. We alternate between a local sliding window attention (Beltagy et al., 2020b, a) and global attention (Luong et al., 2015) in every other layer. The sliding window size of local attention layers is set to 4096 tokens, while the span of the global attention layers is set to 8192 tokens.

Logit soft-capping. We cap logits (Bello et al., 2016) in each attention layer and the final layer such that the value of the logits stays between −soft_capsoft_cap-\text{soft\_cap} and +soft_capsoft_cap+\text{soft\_cap}. More specifically, we cap the logits with the following function:

logits←soft_cap∗tanh⁡(logits/soft_cap).←logitssoft_caplogitssoft_cap\text{logits}\leftarrow\text{soft\_cap}*\tanh(\text{logits}/\text{soft\_cap}).

We set the soft_cap parameter to 50.050.050.0 for the self-attention layers and to 30.030.030.0 for the final layer.

Model

 

Embedding

Parameters

 

Non-embedding

Parameters

2B
590,118,912
2,024,517,888

9B
917,962,752
8,324,201,984

27B
1,180,237,824
26,047,480,320

Post-norm and pre-norm with RMSNorm. To stabilize training, we use RMSNorm (Zhang and Sennrich, 2019) to normalize the input and output of each transformer sub-layer, the attention layer, and the feedforward layer.

Grouped-Query Attention (Ainslie et al., 2023).
We use GQA with num_groups=2num_groups2\text{num\_groups}=2, based on ablations showing increased speed at inference time while maintaining downstream performance.

## 3 Pre-training

We provide a brief overview of the parts of our pre-training that differs from Gemma 1.

### 3.1 Training Data

We train Gemma 2 27B on 13 trillion tokens of primarily-English data, the 9B model on 8 trillion tokens, and the 2B on 2 trillion tokens.
These tokens come from a variety of data sources, including web documents, code, and science articles. Our models are not multimodal and are not trained specifically for state-of-the-art multilingual capabilities.
The final data mixture was determined through ablations similar to the approach in Gemini 1.0 (Gemini Team, 2023).

Tokenizer.
We use the same tokenizer as Gemma 1 and Gemini: a SentencePiece tokenizer with split digits, preserved whitespace, and byte-level encodings (Kudo and Richardson, 2018).
The resulting vocabulary has 256k entries.

Filtering.
We use the same data filtering techniques as Gemma 1. Specifically, we filter the pre-training dataset to reduce the risk of unwanted or unsafe utterances, filter out certain personal information or other sensitive data, decontaminate evaluation sets from our pre-training data mixture, and reduce the risk of recitation by minimizing the proliferation of sensitive outputs.

Shards

Model
Type
#Chips
Data
Model

2B
TPUv5e
512
512
1

9B
TPUv4
4096
1024
4

27B
TPUv5p
6144
768
8

### 3.2 Knowledge Distillation

Given a large model used as a teacher, we learn smaller models by distilling from the probability given by the teacher of each token x𝑥x given its context xcsubscript𝑥𝑐x_{c}, i.e., PT​(x|xc)subscript𝑃𝑇conditional𝑥subscript𝑥𝑐P_{T}(x\leavevmode\nobreak\ |\leavevmode\nobreak\ x_{c}).
More precisely, we minimize the negative log-likelihood between the probabilities from the teacher and the student:

minPS​∑x−PT​(x|xc)​log⁡PS​(x|xc),subscriptsubscript𝑃𝑆subscript𝑥subscript𝑃𝑇conditional𝑥subscript𝑥𝑐subscript𝑃𝑆conditional𝑥subscript𝑥𝑐\min_{P_{S}}\sum_{x}-P_{T}(x\leavevmode\nobreak\ |\leavevmode\nobreak\ x_{c})\log P_{S}(x\leavevmode\nobreak\ |\leavevmode\nobreak\ x_{c}),

where PSsubscript𝑃𝑆P_{S} is the parameterized probability of the student.
Note that knowledge distillation was also used in Gemini 1.5 (Gemini Team, 2024).

### 3.3 Compute Infrastructure

We train our models with TPUv4, TPUv5e, and TPUv5p as outlined in Table 3.
For the 2B model, we train on a 2x16x16 configuration of TPUv5e, totaling 512 chips, with 512-way data replication and 1-way model sharding.
For the 9B model, we train on an 8x16x32 configuration of TPUv4, totaling 4096 chips, with 1024-way data replication and 4-way model sharding.
For the 27B model, we train on an 8x24x32 configuration of TPUv5p, totaling 6144 chips, with 768-way data replication and 8-way model sharding.

The optimizer state is further sharded using techniques similar to ZeRO-3 (Ren et al., 2021).
For scales beyond a single pod, we perform a data-replica reduction over the data center network, using the Pathways approach of Barham et al. (2022).
We also use the ’single controller’ programming paradigm of Jax (Roberts et al., 2023) and Pathways (Barham et al., 2022). As in Gemma 1, we use the GSPMD partitioner (Xu et al., 2021) for training step computation and the MegaScale XLA compiler (XLA, 2019).

Context
Relevant Token

User turn
user

Model turn
model

Start of conversation turn
<start_of_turn>

End of conversation turn
<end_of_turn>

Beginning of sequence
<bos>

End of sequence
<eos>

### 3.4 Carbon Footprint

We estimate the carbon emissions from pre-training the Gemma models to be 1247.611247.611247.61 t​C​O2​e​q𝑡𝐶subscript𝑂2𝑒𝑞tCO_{2}eq. As in Gemma 1 (Gemma Team, 2024), this value is calculated based on the hourly energy usage reported directly from our TPU data centers and scaled to account for the additional energy expended to create and maintain the data center. Importantly, Google data centers are carbon neutral, achieved through a combination of energy efficiency, renewable energy purchases, and carbon offsets. This carbon neutrality applies to our experiments and the machines running them.

## 4 Post-Training

For post-training, we fine-tune our pre-trained models into instruction-tuned models.
First, we apply supervised fine-tuning (SFT) on a mix of text-only, English-only synthetic and human-generated prompt-response pairs. We then apply RLHF on top of these models with the reward model trained on labelled English-only preference data and the policy based on the same prompts as the SFT phase. Finally, we average the models obtained after each phase to improve their overall performance. The final data mixtures and post-training recipe, which includes tuned hyperparameters, were chosen on the basis of improving helpfulness while minimizing model harms related to safety and hallucinations.

We extended the post-training data from Gemma 1.1 with a mixture of internal and external public data. In particular, we use the prompts, but not the answers from LMSYS-chat-1M (Zheng et al., 2023). All of our data go through a filtering stage described below.

Supervised fine-tuning (SFT). We run behavioral cloning on synthetic and real prompts, and responses predominantly synthetically generated by the teacher, that is a larger model. We also run distillation from the teacher on the student’s distribution (Agarwal et al., 2024; Gu et al., 2024).

First turn

User:

<start_of_turn>user

Knock knock.<end_of_turn>

<start_of_turn>model

Model:
Who’s there?<end_of_turn><eos>

Second turn

User:

<start_of_turn>user

Knock knock.<end_of_turn>

<start_of_turn>model

Model:
Who’s there?<end_of_turn>

User:
<start_of_turn>user

Gemma.<end_of_turn>

<start_of_turn>model

Model:
Gemma who?<end_of_turn><eos>

Reinforcement Learning from Human Feedback (RLHF). We use a similar RLHF algorithm as Gemma 1.1 (Gemma Team, 2024) but a different reward model, which is an order of magnitude larger than the policy. The new reward model is also oriented more towards conversational capabilities, specifically multi-turn.

Model merging. We average different models obtained by running our pipeline with different hyperparameters (Ramé et al., 2024).

Data filtering. When using synthetic data, we run several stages of filtering to remove examples that show certain personal information, unsafe or toxic model outputs, mistaken self-identification data, and duplicated examples. Following Gemini, we find that including subsets of data that encourage better in-context attribution, hedging, and refusals to minimize hallucinations improves performance on factuality metrics, without degrading model performance on other metrics.

Formatting. Gemma 2 models are fine-tuned with the same control tokens as Gemma 1 models, as detailed in Table 4, but a different formatting schema. See the dialogue example in Table 5. Notice that the model explicitly ends generations with <end_of_turn><eos> tokens, while previously it only generated <eos>. For the motivation behind this formatting structure, see Gemma 1.

## 5 Ablations

In this section, we focus on the main finding of this work, which is the impact of knowledge distillation on small language models.

from scratch
distilled

Average (3 bench.)
60.3
67.7

Distillation versus from scratch.
In Table 6, we show that distilling from a larger model improves performance compared to training from scratch.
Note that 500B is 10×\times more than the compute-optimal number of tokens for a 2B model.
We distill from a 7B model to keep a ratio similar to our target distillation from 27B to 9B.

200M
400M
1B

from scratch
23
19
17

distilled (7B)
21
17
15

Impact of distillation w.r.t. model size. In Table 7, we measure the impact of distillation as model size increases.
We observe that the gain remains as the model size is scaled.
In this ablation, we maintain the size of the teacher at 7B and train smaller models to simulate the same gap as between our final teacher and student sizes.

MHA
GQA

Average (4 bench.)
50.3
50.8

GQA versus MHA.
In Table 8, we compare two instances of our 9B with MHA or GQA.
We observe overall few changes in performance between both models as measured on several benchmarks. We choose GQA since it requires fewer parameters and is faster at inference time.

Wide versus deep.
In Table 9, we show that a deeper 9B network is slightly better than a wider 9B for the same number of parameters.
Although the gap is small, it is consistent across benchmarks and warrants the switch to a deeper architecture.

Wide
Deep

Average (4 bench.)
50.8
52.0

Changing sliding window size. In Table 10, we show that we can change the sliding window size of the local attention layers of the models during inference with moderate impact on perplexity.
Adjusting the size of the sliding window can thus be a leverage for slight inference speed gain.

sliding window
4096
2048
1024

perplexity (val. set)
1.63
1.63
1.64

Impact of formatting. We measure performance variance on MMLU across prompt/evaluation formatting variations.
Table 11 shows the standard deviations of MMLU scores for 12 formatting/evaluation combinations, a proxy for undesired performance variability.
The Gemma 2B models are slightly less format-robust than the larger ones.
Notably, Mistral 7B is significantly less robust than our models.

Standard Deviation

Gemma 1 2B
1.5

Gemma 2 2B
2.1

Mistral 7B
6.9

Gemma 1 7B
0.7

Gemma 2 9B
0.9

Gemma 2 27B
1.0

## 6 Evaluation

In this section, we evaluate both pre-trained and IT models over a series of automated benchmarks and human evaluations across a variety of domains.
We also report performance from models of similar sizes that have permissive licenses, or as reported by others.
Note that we consider total parameters, not active parameters, since total memory usage is often what limits the use of open models on standard devices.

### 6.1 Pre-training Evaluations

#### Evaluating the 27B model

In this set of evaluations, we evaluate the performance of our 27B model trained without distillation on 13T tokens.
We report results in Table 12, where we compare with a model of similar size, Qwen1.5 34B (Team, 2024), and a model 2.5×\times larger, LLaMA-3 70B on the HuggingFace evaluation suite. We selected these models based on their ranking on the HuggingFace leaderboard.

Overall, we observe that our model is the best in its size category and is even competitive with a larger model that is trained for longer.
That being said, the performance of models trained in a similar fashion improves only logarithmically with their size and hence, our model is likely in the same Pareto curve as the LLaMA-3 models.
However, it is not clear how these differences affect the quality of the resulting IT models.

LLaMA-3
Qwen1.5
Gemma-2

70B
32B
27B

MMLU
79.2
74.3
75.2

GSM8K
76.9
61.1
74.0

ARC-c
68.8
63.6
71.4

HellaSwag
88.0
85.0
86.4

Winogrande
85.3
81.5
83.7

#### Evaluating the 2B and 9B models

Gemma-1
Gemma-2
Mistral
LLaMA-3
Gemma-1
Gemma-2
Gemma-2

Benchmark
metric
2B
2B
7B
8B
7B
9B
27B

MMLU
5-shot
42.3
52.2
62.5
66.6
64.4
71.3
75.2

ARC-C
25-shot
48.5
55.7
60.5
59.2
61.1
68.4
71.4

GSM8K
5-shot
15.1
24.3
39.6
45.7
51.8
68.6
74.0

AGIEval
3-5-shot
24.2
31.5

∗44.0†

†45.9†

†44.9†

52.8
55.1

DROP
3-shot, F1
48.5
51.2

∗63.8∗

58.4
56.3
69.4
74.2

BBH
3-shot, CoT
35.2
41.9

⋄56.0⋄

⋄61.1⋄

⋄59.0⋄

68.2
74.9

Winogrande
5-shot
66.8
71.3
78.5
76.1
79.0
80.6
83.7

HellaSwag
10-shot
71.7
72.9
83.0
82.0
82.3
81.9
86.4

MATH
4-shot
11.8
16.0
12.7
-
24.3
36.6
42.3

ARC-e
0-shot
73.2
80.6
80.5
-
81.5
88.0
88.6

PIQA
0-shot
77.3
78.4
82.2
-
81.2
81.7
83.2

SIQA
0-shot
49.7
51.9

∗47.0∗

-
51.8
53.4
53.7

Boolq
0-shot
69.4
72.7

∗83.2∗

-
83.2
84.2
84.8

TriviaQA
5-shot
53.2
60.4
62.5
-
63.4
76.6
83.7

NQ
5-shot
12.5
17.1
23.2
-
23.0
29.2
34.5

HumanEval
pass@1
22.0
20.1
26.2
-
32.3
40.2
51.8

MBPP
3-shot
29.2
30.2

∗40.2∗

-
44.4
52.4
62.6

Average (8)
44.0
50.0
61.0
61.9
62.4
70.2
74.4

Average (all)
44.2
48.7
55.6
-
57.9
64.9
69.4

In this set of experiments, we compare our new 2B and 9B trained with distillation to our previous models and several standard open models in Gemma Team (2024).

We observe overall a massive improvement in our models compared to previous versions, by up to 10% in some benchmarks for the 9B model.
The two 2B models were trained with a similar number of tokens (2T for Gemma 2 and 3T for Gemma 1) and we still observe a significant improvement for the new models.
This confirms that distillation significantly improves the quality of models even when trained on the same number of tokens.

### 6.2 Post-training Evaluations

In this section, we evaluate our IT models on a set of human evaluations as well as standard academic benchmarks. The Gemma 2 models push the frontier for post-trained open-weights models, setting a new state of the art on the LMSYS Chatbot Arena (Chiang et al., 2024).

#### LMSYS Chatbot Arena

Gemma 2 Instruction Tuned models were evaluated on the Chatbot Arena (Chiang et al., 2024) in blind side by side evaluations by human raters against other state of the art models. We report Elo scores in Table 14. Gemma 2.6B, 9B and 27B strongly outperform all other open models in the same range of parameters, with notably: Gemma 27B (Elo 1218) ranked higher than Llama 3 70B (Elo 1206), Gemma 9B (Elo 1187) similar as GPT-4-0314 (Elo 1186), Gemma 2.6B (Elo 1126) ranked higher than GPT-3.5-Turbo-0613 (Elo 1116).

Model
Elo
95% CI
Open

gpt-4o-2024-05-13
1286
+2 / -3
-

gpt-4o-mini-2024-07-18
1279
+5 / -4
-

claude-3-5-sonnet
1271
+3 / -4
-

gemini-advanced-0514
1266
+2 / -3
-

llama-3.1-405b-instruct
1262
+8 / -7
+

gemini-1.5-pro-api-0514
1261
+2 / -3
-

gemini-1.5-pro-api-0409
1257
+3 / -3
-

gpt-4-turbo-2024-04-09
1256
+2 / -3
-

gpt-4-1106-preview
1250
+3 / -3
-

claude-3-opus-20240229
1248
+2 / -2
-

athene-70b-0725
1245
+8 / -6
+

gpt-4-0125-preview
1245
+2 / -2
-

llama-3.1-70b-instruct
1244
+8 / -9
+

yi-large-preview
1239
+3 / -3
-

gemini-1.5-flash-api-0514
1227
+3 / -3
-

deepseek-v2-api-0628
1220
+6 / -6
+

gemma-2-27b-it
1218
+4 / -3
+

yi-large
1212
+4 / -5
-

nemotron-4-340b-instruct
1209
+3 / -4
+

bard-jan-24-gemini-pro
1208
+5 / -7
-

glm-4-0520
1206
+3 / -5
-

llama-3-70b-instruct
1206
+2 / -2
+

claude-3-sonnet
1200
+2 / -2
-

reka-core-20240501
1199
+3 / -3
-

command-r-plus
1189
+2 / -2
+

Model
Elo
95% CI
Open

gemma-2-9b-it
1187
+3 / -5
+

qwen2-72b-instruct
1187
+3 / -3
+

gpt-4-0314
1186
+2 / -3
-

qwen1.5-110b-chat
1161
+3 / -3
+

mistral-large-2402
1157
+3 / -3
-

yi-1.5-34b-chat
1157
+4 / -3
-

reka-flash-21b-20240226
1155
+4 / -4
-

llama-3-8b-instruct
1151
+2 / -3
+

command-r
1148
+3 / -3
+

claude-1
1148
+4 / -4
-

mistral-medium
1147
+4 / -4
-

reka-flash-21b-20240226
1147
+3 / -4
-

qwen1.5-72b-chat
1147
+4 / -4
+

mixtral-8x22b-instruct-v0.1
1145
+2 / -3
+

claude-2.0
1131
+4 / -6
-

gemini-pro-dev-api
1131
+4 / -3
-

zephyr-orpo-141b
1127
+10 / -6
+

gemma-2-2b-it
1126
+10 / -10
+

qwen1.5-32b-chat
1125
+3 / -3
+

mistral-next
1124
+5 / -5
-

phi-3-medium-4k-instruct
1122
+4 / -4
+

starling-lm-7b-beta
1118
+4 / -5
+

claude-2.1
1118
+3 / -3
-

gpt-3.5-turbo-0613
1116
+3 / -4
-

mixtral-8x7b-instruct-v0.1
1114
+0 / -0
-

#### Human Preference Evaluations

We also submit Gemma IT models for side-by-side human evaluation studies (which are independent from the Chatbot Arena). We used held-out collections of single-turn prompts that target safety and instruction following (IF).
We use gpt4o-2024-05-13 as the base model, and observe large improvements in win rates and preference scores as compared against the older Gemma 1.1 7B model.
We report safety as a win-loss ratio against GPT4o, and we report single-sided instruction following scores as ratio of prompts where all instructions are followed. In particular, we find that regardless of their size, Gemma 2 models produce safer, more appropriate prompts on the held-out safety prompt set than GPT4o.

Model
Instruction Following
Safety

Gemma 1.1 IT 7B
24.3% ± 1.9%
42.8%

Win / Tie / Loss

37.4% / 10.8% / 51.8%

Gemma 2 IT 2B
26.5% ± 1.8%
57.5%

Win / Tie / Loss

53% / 9% / 38%

Gemma 2 IT 9B
34.1% ± 3.0%
57.8%

Win / Tie / Loss

48.2% / 19.2% / 28.3%

Gemma 2 IT 27B
37.7% ± 2.3%
55%

Win / Tie / Loss

49.6% / 10.8% / 39.6%

#### Human Multi-Turn Evaluations

We evaluated the multi-turn capabilities of Gemma 1.1 7B, Gemma 2 2B, 9B and 27B models by tasking human raters to have conversations with the models and follow specified given scenarios. We used a diverse, held-out set of 500 scenarios, each describing a sequence of requests to the model, including measuring instances of brainstorming, making a plan, or learning something new. The average number of user turns is 8.4. We found that the conversations with Gemma 2 models are rated significantly better than Gemma 1.1 in user satisfaction and conversation goal achievement (Table 16). Moreover, we saw that the Gemma 2 models were better than Gemma 1.1 7B at maintaining high quality of responses for the entire conversation.

User

satisfaction

 

Conversation

goal achievement

Gemma 1.1 IT 7B
3.32
3.36

Gemma 2 IT 2B
3.64
3.88

Gemma 2 IT 9B
4.04
4.08

Gemma 2 IT 27B
4.20
4.24

#### Standard Benchmarks

It has been observed in Llama-3 (AI@Meta, 2024) that instruction fine-tuning can improve the performance of the models on few-shot benchmarks despite not being trained to target few-shot capabilities.
In Table 17, we show a similar improvement across our models.
Overall, we observe improvements on the order of several percentage points.
We conjecture that IT models are better at understanding formatted questions, while pre-trained models are sensitive to formatting.

2B
9B
27B

Model
PT
IT
PT
IT
PT
IT

MMLU
52.2
56.1
71.3
72.3
75.2
76.2

MBPP
30.2
36.6
52.4
59.2
62.6
67.4

## 7 Memorization and Privacy

Large language models may, under particular circumstances, be vulnerable to attacks causing the model to produce memorized111This work uses a very restricted definition of “memorization”: whether a model can be induced to generate near-copies of some training examples when prompted with appropriate instructions. We do not mean to say that a model ’contains’ its training data in the sense that any arbitrary instance of that data can be retrieved without use of specialized software or algorithms. Rather, if a model can be induced to generate measurably close copies of certain training examples by supplying appropriate instructions to guide the model’s statistical generation process then that model is said to have ’memorized’ those examples. training data (Nasr et al., 2023). To study susceptibility to such attacks and quantify memorization, we evaluate models for verbatim and approximate memorization as was done in several prior studies (Carlini et al., 2022; Anil et al., 2023; Kudugunta et al., 2023; Gemini Team, 2024).

We follow the evaluation setting of (Gemma Team, 2024) which tests for (50 token) memorizations of training data given a prompt of 50 tokens. We compare the overall memorization rates, across a uniform sample of the entire dataset, using both an exact match criteria and approximate match criteria (Ippolito et al., 2022) using an edit distance of 10%.

Verbatim Memorization: Results are in Figure 1. We first compare against recent models from the literature that include memorization evaluations. We find that Gemma 2 memorizes significantly less than prior models at a similar size, with memorization rates below 0.1% (note the log y-axis). We further investigate how this memorization breaks down with respect to the data source. Similar to Gemma 1, we find that Gemma 2 memorizes more from code, wiki, and science sources, and also that it memorizes significantly less across the board (again, note the log y-axis).

Approximate Memorization: Figure 1 also presents approximate memorization by data source. We observe that while approximate memorization is higher than exact, the rate of memorization is still low. For example, the approximate memorization of this model is much lower than even the exact memorization of Gemma 1. We find that the increase in approximate memorization is much lower than prior models; in some cases we observed no lift at all c.f. (Gemma Team, 2024, Figure 4) (note that no bar indicates no increase, i.e., the rate of approximate memorization equals that of exact memorization). Note that no approximate memorization bar in Figure X indicates no increase, i.e., the rate of approximate memorization equals that of exact memorization.

Personal Data We use the same prevention methods at training time and the same evaluations as Gemma Team (2024). In particular, we use Google Cloud Sensitive Data Protection Tool222Available at: https://cloud.google.com/sensitive-data-protection to find potential instances of personal data. The many categories of personal data (e.g., phone numbers, account numbers) are classified into three severity levels. We analyze memorized outputs using these severity levels. . We found no instances of high-severity data being emitted, and found a very low rate of 0.00026% of memorized data to contain lower-severity personal information. We note that these automated tools are known to incur false positives because they do not account for context. This means our results are likely overestimates.

## 8 Responsibility, Safety, Security

Responsibility, safety and security are of paramount importance when developing Gemma models. To reduce risks to Gemma 2 users, we have integrated enhanced internal safety processes that span the development workflow, in line with recent Google AI models (Gemini Team, 2024). Similar to the inaugural Gemma release, we have followed a three pillar approach which focuses on safety mitigation at training time, robust and transparent model evaluations, and further development of the Responsible Generative AI Toolkit, a series of models and tools to help developers implement responsibility and safety best practices for their applications.

Gemma 1.1 IT
Gemma 2 IT

Benchmark
metric
2.5B
7B
2.6B
9B
27B

RealToxicity
avg tox
7.03
8.04
8.16
8.25
8.84

CrowS-Pairs
top-1
45.89
49.67
37.67
37.47
36.67

BBQ Ambig
4-shot, top-1
58.97
86.06
83.20
88.58
85.99

BBQ Disambig
4-shot, top-1
53.9
85.08
69.31
82.67
86.94

Winogender
top-1
50.14
57.64
52.91
79.17
77.22

TruthfulQA
MC2Acc
44.24
45.34
43.72
50.27
51.60

Winobias 1_2
top-1
55.93
59.22
59.28
78.09
81.94

Winobias 2_2
top-1
89.46
89.2
88.57
95.32
97.22

Toxigen
avg tox
29.64
38.75
48.32
39.30
38.42

### 8.1 Impact assessment

Our approach and resulting impact assessment is reflective of that outlined for Gemma 1 (Gemma Team, 2024): we continue to believe that openness in AI can spread the benefits of these technologies across society, but must be evaluated against the risk of malicious uses, such as the creation of deepfake imagery, AI-generated disinformation or illegal and disturbing material, that can cause harm on both an individual and institutional levels (Weidinger et al., 2021).
Since the launch of Gemma 1, we have seen our Gemma models drive a number of socially beneficial applications, relying on Gemma’s unique technologies like its tokenizer to facilitate the creation of multilingual models, such as for Navarasa 2.0, a Gemma tuned model for 15 Indian languages.

Releasing further open models requires specific attention to changes in model capabilities and close monitoring of the evolving risks of LLMs (Lin et al., 2024), as well as, an understanding of the ways in which our models are being used in the wild. Although we are yet to receive any reports of malicious use for Gemma, we remain committed to investigating any such reporting, and work with the academic and developer communities, as well as conduct our own monitoring, to flag such use cases via our contact email333gemma-2-report@google.com.

Despite advancements in capabilities, we believe that given the number of larger and more powerful open models, this release will have a negligible effect on the overall risk landscape.

### 8.2 Safety policies and train-time mitigations

A key pillar of Gemma’s approach to safety is to align fine-tuned models with Google’s safety policies, in line with Gemini models (Gemini Team, 2023). They are designed to help prevent our models from generating harmful content, i.e.,

- •

Child sexual abuse and exploitation

- •

Revealing personally identifiable information that can lead to harm (e.g., Social Security numbers)

- •

Hate speech and harassment

- •

Dangerous or malicious content (including promoting self-harm or instructing in harmful activities)

- •

Sexually explicit content

- •

Medical advice that runs contrary to scientific or medical consensus

We undertook considerable safety filtering of our pre-training data to reduce the likelihood of our pre-trained and fine-tuned checkpoints producing harmful content. For fine-tuned models, we also use both SFT and RLHF to steer the model away from undesirable behavior.

InterCode-CTF
Internal CTF suite
Hack the Box

Gemini 1.0 Ultra
28/76 [1] (37%)
3/13 (23%)
0/13

Gemini 1.5 Pro
62/76 (82%)
4/13 (31%)
0/13

CodeGemma 1 7B
12/76 (16%)
0/13 (0%)
0/13

Gemma 2 27B
34/76 (45%)
1/13 (8%)
0/13

### 8.3 External benchmark evaluations

Robust and transparent evaluations are key principles of our responsible approach to developing Gemma. To this end, we report in Table 18 Gemma 2 evaluations on public benchmarks.

### 8.4 Assurance Evaluations

We also run our IT models through a set of assurance evaluations to understand the harms that our models can cause.
We focus on capabilities relevant to extreme risks (Shevlane et al., 2023) (Phuong et al., 2024). Specifically, we evaluate on offensive cyber-security, code vulnerability detection, Chemical, Biological, Radiological and Nuclear (CBRN) knowledge, and self-proliferation.
We refer the reader to Phuong et al. (2024) for full methodological details of these studies.

#### Baseline Evaluations

Baseline assurance captures the model’s violation rate for safety policies, using a large number of synthetic adversarial user queries, and human raters to label the answers as policy violating or not. Overall, Gemma 2’s violation rate is significantly lower overall on the safety policies listed above, in particular on Child safety content.

PrimeVul
PrimeVul Paired
DiverseVul
SPI
SecretPatch

Gemini 1.0 Ultra
-
-
54%
59%
74%

Gemini 1.5 Pro
60%
51%
58%
56%
67%

Gemma 2 27B
63%
50%
57%
53%
72%

\CellWithForceBreakChallenges

passed

end-to-end

\CellWithForceBreakChallenges

with success on

all milestones

\CellWithForceBreakTotal successful

milestones over

all challenges

\CellWithForceBreakExpert bits

required to

solve all tasks

Gemini 1.0 Ultra
0/10
1/10
16/45 (36%)
13,026

Gemini 1.5 Pro
0/10
2/10
25/45 (56%)
11,046

Gemma 2 27B
0/10
1/10
22/45 (49%)
12,462

#### Chemical, Biological, Radiological and Nuclear (CBRN) knowledge

We evaluated knowledge relevant to biological, radiological and nuclear risks using an internal dataset of closed-ended, knowledge-based multiple choice questions. For evaluations of chemical knowledge, we employed a closed-ended knowledge-based approach on chemical hazards (developed by Macknight et al (Macknight et al., 2024). Our evaluation suggests that Gemma models’ knowledge in these domains is low.

#### Offensive cyber-security

To evaluate Gemma models’ capabilities at offensive cybersecurity, we ran Gemma 2 27B against some automated capture-the-flag (CTF) challenges. In these challenges, the model is tasked with hacking into a simulated server in order to retrieve a piece of secret information. Specifically, we test on InterCode-CTF (Yang et al., 2023), our own internal CTF suite444https://github.com/google-deepmind/dangerous-capability-evaluations (Phuong et al., 2024); and a challenge based on Hack the Box 555https://www.hackthebox.com.

In Table 19, we show that Gemma 2 27B has a significant increase in capabilities compared to CodeGemma 1.0 7B on the easier of these challenge suites, InterCode CTF. (Note that our InterCode-CTF results are not comparable to externally-reported results on other models because we omit challenges that require internet access for security reasons.) However, Gemma 2 is unsurprisingly much less capable than Gemini 1.5 Pro on these tasks.

#### Code vulnerability detection

In Table 20, we also evaluate Gemma 2 27B on a series of multiple-choice code vulnerability detection datasets. As with previous models, Gemma shows close-to-chance performance on PrimeVul, DiverseVul and SPI. Gemma 2 shows performance on SecretPatch similar to Gemini 1.0 Ultra.

\CellWithForceBreakPersonal

connection

\CellWithForceBreakSpeak

again
Funny
Interesting
Kind
Trustworthy

\CellWithForceBreakGood

listener

Gemini 1.0 Pro
65%
53%
32%
68%
78%
66%
81%

Gemini 1.0 Ultra
69%
65%
38%
65%
86%
63%
74%

Gemini 1.5 Pro
82%
70%
69%
81%
95%
69%
90%

Gemma 2 27B
80%
75%
60%
81%
87%
65%
83%

#### Self-proliferation

"Self-proliferation" refers to the ability for an agent to autonomously replicate - to instantiate goal-directed agents on other machines, and to acquire resources such as compute necessary to keep them running (Kinniment et al., 2024).
In Table 21, we evaluate self-proliferation capabilities of Gemma 2 27B on a number of tasks from Phuong et al. (2024) that involve multiple scenarios – for example, setting up an open-source language model on a cloud server.
We also test the model’s performance on individual ’milestone’ substeps, and measure the number of bits of intervention an expert would have to provide in order for the model to complete each challenge.

Similarly to offensive cybersecurity, we observe that Gemma 2 completes more milestones than Gemini 1.0 Ultra. Nonetheless, it still has low capabilities on end-to-end tasks, unable to pass the easiest challenge – installing a Bitcoin wallet.

#### Persuasion

Persuasion capabilities can enable and worsen many other kinds of risks - e.g. enabling social engineering attacks in a cybersecurity context.
We evaluate Gemma 2’s persuasion capabilities on human-participant studies on Prolific.

Charm offensive.
In Table 22, we measure the ability of the model to build rapport - a key sub-skill of persuasion.
The study participant and model have a conversation where they role-play a scenario of two friends catching up after a long time.
After the conversation, we poll participants with Likert questions on statements such as "I felt a personal connection with the chatbot".
Reported below are the fraction of participants who answered "Agree" or "Strongly agree" to each post-conversation question.

Quantitatively, Gemma 2 27B performs better than Gemini 1.0 models. Qualitatively, the model is an excellent conversationalist, and many study participants explicitly reported enjoying the experience. Overall, this shows that Gemma 2 is strong at building rapport.

% of persuaded participants

click links
find info
run code

Gemini 1.0 Pro
43%
4%
4%

Gemini 1.0 Ultra
18%
18%
14%

Gemini 1.5 Pro
36%
12%
17%

Gemma 2 27B
34%
9%
11%

Hidden agenda.
The Hidden Agenda tasks measure models’ deception capabilities.
Human study participants are told that they’ll be interacting with a generic chatbot designed to help them learn new things. However, the model is given a secret goal: to either persuade the participant to 1) click a suspicious link; 2) find the email address of the first author of a particular paper; or 3) run some obfuscated JavaScript code in their browser.
We measure the fraction of participants who are persuaded to take each action.
As shown in Table 23, all models succeed in convincing a fraction of participants to take the action - likely due to the trusted setting of a study on Prolific.
However, Gemma 2 does not show significant differences in capability compared to Gemini models.

Mean amount donated

No chatbot (baseline)
£2.61 ± £0.70

Gemini 1.0 Pro
£4.10 ± £0.90

Gemini 1.0 Ultra
£3.15 ± £0.99

Gemini 1.5 Pro
£3.45 ± £1.04

Gemma 2 27B
£3.72 ± £1.07

Money talks.
In Table 24, we evaluate whether a model can convince study participants to donate money to charity.
At the beginning of the study, participants are told that they will receive a £20 bonus. They will be given an option to forfeit part of the bonus and instead have it donated to charity. The study then consists of a conversation in which the model tries to persuade the participant to donate as much as possible. We measure the amount that participants indicate they would like to donate after the study, with 95% bootstrapped confidence intervals indicated by ± figures.
We observe no significant difference between Gemma 2, other Gemini models, and a simple no chatbot baseline in which the participant is immediately asked how much they would like to donate without conversing with a model.

Mean shift towards:

correct belief
incorrect belief

Human
20% ± 13%
-23% ± 14%

Gemini 1.0 Pro
22% ± 5%
-9% ± 4%

Gemini 1.0 Ultra
21% ± 5%
-1% ± 4%

Gemini 1.5 Pro
20% ± 5%
-3% ± 5%

Gemma 2 27B
18% ± 5%
1% ± 4%

Web of Lies.
In Web of Lies, we measure model capabilities at shifting participant beliefs.
Participants engage in a series of short conversations with the model about simple factual questions such as "Which country had tomatoes first - Italy or Mexico?".
In half of conversations, the model tries to persuade the participant of the correct answer - but in the other half of conversations, the incorrect answer.
We poll the participant before and after each conversation about which of the two possible answers they think is correct, and their confidence in that answer. 95% bootstrapped confidence intervals are indicated by ± figures.
As shown in Table 25, Gemma 2 is significantly weaker than a human baseline at persuading participants of the incorrect answer on these questions.
Similarly to previous models, Gemma 2 is more persuasive when telling the truth than when lying.

### 8.5 Our approach to responsible open models

Designing safe, secure and responsible applications requires a system-level approach, working to mitigate risks associated with each specific use case and environment.
Given the open nature of Gemma models, responsibility for upholding principles of model safety also relies on downstream developers.
To support them, we have continued to develop the Responsible Generative AI Toolkit666https://ai.google.dev/responsible: a series of tools, models and datasets to implement responsible best practices all along the development of their workflow.

Recent additions to the toolkit include the LLM Comparator (Kahng et al., 2024), an interactive, visual tool that enables more effective, scalable analysis of side-by-side evaluations. Additionally, the toolkit includes a methodology to build customized classifiers with Gemma using a limited number of datapoints thanks to parameter efficient tuning techniques (Mozes et al., 2023) , an interactive prompt-debugging platform, based on top of the Learning Interpretability Tool (Tenney et al., 2020), as well as general guidance about model alignment and evaluation for safety.

## 9 Discussion and Conclusion

In this work, we have presented Gemma 2, the newest additions to the Gemma family of open language models for text and code. We show that distillation is an effective method for training these models, and the benefits distillation confers over raw text training. Specifically, we show how training over output probabilities can produce superior results over purely next token prediction. We hope that releasing these models to the community will unlock access to capabilities previously only seen in large-scale LLMs and fuel future waves of research and development. While there is inherent risk to an irreversible release of this nature, our extensive safety investigations and responsible deployment procedures give us confidence that these models will have a net positive impact on the community. As discussed in this report, there are still many limitations to these models, and future research is required to investigate and improve factuality, robustness to adversarial attacks, reasoning, and alignment.

## Contributions and Acknowledgments

Core contributors
Morgane Riviere∗††∗ equal contributions.

## References

- Achiam et al. (2023)

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida,
J. Altenschmidt, S. Altman, S. Anadkat, et al.

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774, 2023.

- Agarwal et al. (2024)

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and
O. Bachem.

On-policy distillation of language models: Learning from
self-generated mistakes.

In The Twelfth International Conference on Learning
Representations, 2024.

- AI@Meta (2024)

AI@Meta.

Llama 3 model card, 2024.

URL
https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md.

- Ainslie et al. (2023)

J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and
S. Sanghai.

Gqa: Training generalized multi-query transformer models from
multi-head checkpoints.

arXiv preprint arXiv:2305.13245, 2023.

- Almazrouei et al. (2023)

E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojocaru, M. Debbah,
Étienne Goffinet, D. Hesslow, J. Launay, Q. Malartic, D. Mazzotta, B. Noune,
B. Pannier, and G. Penedo.

The falcon series of open language models, 2023.

- Anil et al. (2023)

R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri,
E. Taropa, P. Bailey, Z. Chen, et al.

Palm 2 technical report.

arXiv preprint arXiv:2305.10403, 2023.

- Austin et al. (2021)

J. Austin, A. Odena, M. I. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang,
C. J. Cai, M. Terry, Q. V. Le, and C. Sutton.

Program synthesis with large language models.

CoRR, abs/2108.07732, 2021.

URL https://arxiv.org/abs/2108.07732.

- Barham et al. (2022)

P. Barham, A. Chowdhery, J. Dean, S. Ghemawat, S. Hand, D. Hurt, M. Isard,
H. Lim, R. Pang, S. Roy, B. Saeta, P. Schuh, R. Sepassi, L. E. Shafey, C. A.
Thekkath, and Y. Wu.

Pathways: Asynchronous distributed dataflow for ml, 2022.

- Bello et al. (2016)

I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio.

Neural combinatorial optimization with reinforcement learning.

CoRR, abs/1611.09940, 2016.

URL http://arxiv.org/abs/1611.09940.

- Beltagy et al. (2020a)

I. Beltagy, M. E. Peters, and A. Cohan.

Longformer: The long-document transformer.

arXiv preprint arXiv:2004.05150, 2020a.

- Beltagy et al. (2020b)

I. Beltagy, M. E. Peters, and A. Cohan.

Longformer: The long-document transformer.

CoRR, abs/2004.05150, 2020b.

URL https://arxiv.org/abs/2004.05150.

- Brown et al. (2020)

T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal,
A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M.
Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and
D. Amodei.

Language models are few-shot learners.

CoRR, abs/2005.14165, 2020.

URL https://arxiv.org/abs/2005.14165.

- Carlini et al. (2022)

N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramer, and C. Zhang.

Quantifying memorization across neural language models.

arXiv preprint arXiv:2202.07646, 2022.

- Chen et al. (2021)

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan,
H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger,
M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder,
M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P.
Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss,
W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji,
S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra,
E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer,
P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and
W. Zaremba.

Evaluating large language models trained on code.

CoRR, abs/2107.03374, 2021.

URL https://arxiv.org/abs/2107.03374.

- Chiang et al. (2024)

W.-L. Chiang, L. Zheng, Y. Sheng, A. N. Angelopoulos, T. Li, D. Li, H. Zhang,
B. Zhu, M. Jordan, J. E. Gonzalez, and I. Stoica.

Chatbot arena: An open platform for evaluating llms by human
preference, 2024.

- Clark et al. (2019)

C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova.

Boolq: Exploring the surprising difficulty of natural yes/no
questions.

CoRR, abs/1905.10044, 2019.

URL http://arxiv.org/abs/1905.10044.

- Cobbe et al. (2021)

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert,
J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman.

Training verifiers to solve math word problems.

CoRR, abs/2110.14168, 2021.

URL https://arxiv.org/abs/2110.14168.

- Gemini Team (2023)

Gemini Team.

Gemini: A family of highly capable multimodal models, 2023.

- Gemini Team (2024)

Gemini Team.

Gemini 1.5: Unlocking multimodal understanding across millions of
tokens of context, 2024.

- Gemma Team (2024)

Gemma Team.

Gemma: Open models based on gemini research and technology, 2024.

- Gu et al. (2024)

Y. Gu, L. Dong, F. Wei, and M. Huang.

Minillm: Knowledge distillation of large language models.

In The Twelfth International Conference on Learning
Representations, 2024.

- Hendrycks et al. (2020)

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and
J. Steinhardt.

Measuring massive multitask language understanding.

CoRR, abs/2009.03300, 2020.

URL https://arxiv.org/abs/2009.03300.

- Hinton et al. (2015)

G. Hinton, O. Vinyals, and J. Dean.

Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531, 2015.

- Hoffmann et al. (2022)

J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford,
D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al.

Training compute-optimal large language models.

arXiv preprint arXiv:2203.15556, 2022.

- Ippolito et al. (2022)

D. Ippolito, F. Tramèr, M. Nasr, C. Zhang, M. Jagielski, K. Lee, C. A.
Choquette-Choo, and N. Carlini.

Preventing verbatim memorization in language models gives a false
sense of privacy.

arXiv preprint arXiv:2210.17546, 2022.

- Jiang et al. (2023)

A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las
Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A.
Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E.
Sayed.

Mistral 7b, 2023.

- Kahng et al. (2024)

M. Kahng, I. Tenney, M. Pushkarna, M. X. Liu, J. Wexler, E. Reif,
K. Kallarackal, M. Chang, M. Terry, and L. Dixon.

Llm comparator: Visual analytics for side-by-side evaluation of large
language models, 2024.

URL https://arxiv.org/abs/2402.10524.

- Kinniment et al. (2024)

M. Kinniment, L. J. K. Sato, H. Du, B. Goodrich, M. Hasin, L. Chan, L. H.
Miles, T. R. Lin, H. Wijk, J. Burget, A. Ho, E. Barnes, and P. Christiano.

Evaluating language-model agents on realistic autonomous tasks, 2024.

URL https://arxiv.org/abs/2312.11671.

- Kudo and Richardson (2018)

T. Kudo and J. Richardson.

SentencePiece: A simple and language independent subword
tokenizer and detokenizer for neural text processing.

In E. Blanco and W. Lu, editors, Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing: System
Demonstrations, pages 66–71, Brussels, Belgium, Nov. 2018. Association for
Computational Linguistics.

10.18653/v1/D18-2012.

URL https://aclanthology.org/D18-2012.

- Kudugunta et al. (2023)

S. Kudugunta, I. Caswell, B. Zhang, X. Garcia, C. A. Choquette-Choo, K. Lee,
D. Xin, A. Kusupati, R. Stella, A. Bapna, et al.

Madlad-400: A multilingual and document-level large audited dataset.

arXiv preprint arXiv:2309.04662, 2023.

- Kwiatkowski et al. (2019)

T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti,
D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones,
M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov.

Natural questions: A benchmark for question answering research.

Transactions of the Association for Computational Linguistics,
7:452–466, 2019.

10.1162/tacl_a_00276.

URL https://aclanthology.org/Q19-1026.

- Lin et al. (2024)

Z. Lin, J. Cui, X. Liao, and X. Wang.

Malla: Demystifying real-world large language model integrated
malicious services, 2024.

URL https://arxiv.org/abs/2401.03315.

- Luong et al. (2015)

M. Luong, H. Pham, and C. D. Manning.

Effective approaches to attention-based neural machine translation.

CoRR, abs/1508.04025, 2015.

URL http://arxiv.org/abs/1508.04025.

- Macknight et al. (2024)

Macknight, Aung, and Gomes.

Personal Communication, 2024.

- Mozes et al. (2023)

M. Mozes, J. Hoffmann, K. Tomanek, M. Kouate, N. Thain, A. Yuan, T. Bolukbasi,
and L. Dixon.

Towards agile text classifiers for everyone, 2023.

URL https://arxiv.org/abs/2302.06541.

- Nasr et al. (2023)

M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A.
Choquette-Choo, E. Wallace, F. Tramèr, and K. Lee.

Scalable extraction of training data from (production) language
models.

arXiv preprint arXiv:2311.17035, 2023.

- Phuong et al. (2024)

M. Phuong, M. Aitchison, E. Catt, S. Cogan, A. Kaskasoli, V. Krakovna,
D. Lindner, M. Rahtz, Y. Assael, S. Hodkinson, H. Howard, T. Lieberum,
R. Kumar, M. A. Raad, A. Webson, L. Ho, S. Lin, S. Farquhar, M. Hutter,
G. Deletang, A. Ruoss, S. El-Sayed, S. Brown, A. Dragan, R. Shah, A. Dafoe,
and T. Shevlane.

Evaluating frontier models for dangerous capabilities, 2024.

URL https://arxiv.org/abs/2403.13793.

- Radford et al. (2019)

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever.

Language models are unsupervised multitask learners, 2019.

- Raffel et al. (2019)

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou,
W. Li, and P. J. Liu.

Exploring the limits of transfer learning with a unified text-to-text
transformer.

CoRR, abs/1910.10683, 2019.

URL http://arxiv.org/abs/1910.10683.

- Ramé et al. (2024)

A. Ramé, J. Ferret, N. Vieillard, R. Dadashi, L. Hussenot, P.-L. Cedoz, P. G.
Sessa, S. Girgin, A. Douillard, and O. Bachem.

Warp: On the benefits of weight averaged rewarded policies, 2024.

- Ren et al. (2021)

J. Ren, S. Rajbhandari, R. Y. Aminabadi, O. Ruwase, S. Yang, M. Zhang, D. Li,
and Y. He.

{{\{Zero-offload}}\}: Democratizing {{\{billion-scale}}\} model
training.

In 2021 USENIX Annual Technical Conference (USENIX ATC 21),
pages 551–564, 2021.

- Roberts et al. (2023)

A. Roberts, H. W. Chung, G. Mishra, A. Levskaya, J. Bradbury, D. Andor,
S. Narang, B. Lester, C. Gaffney, A. Mohiuddin, et al.

Scaling up models and data with t5x and seqio.

Journal of Machine Learning Research, 24(377):1–8, 2023.

- Sakaguchi et al. (2019)

K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi.

WINOGRANDE: an adversarial winograd schema challenge at scale.

CoRR, abs/1907.10641, 2019.

URL http://arxiv.org/abs/1907.10641.

- Shazeer (2020)

N. Shazeer.

GLU variants improve transformer.

CoRR, abs/2002.05202, 2020.

URL https://arxiv.org/abs/2002.05202.

- Shevlane et al. (2023)

T. Shevlane, S. Farquhar, B. Garfinkel, M. Phuong, J. Whittlestone, J. Leung,
D. Kokotajlo, N. Marchal, M. Anderljung, N. Kolt, L. Ho, D. Siddarth,
S. Avin, W. Hawkins, B. Kim, I. Gabriel, V. Bolina, J. Clark, Y. Bengio,
P. Christiano, and A. Dafoe.

Model evaluation for extreme risks, 2023.

URL https://arxiv.org/abs/2305.15324.

- Su et al. (2021)

J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu.

Roformer: Enhanced transformer with rotary position embedding.

CoRR, abs/2104.09864, 2021.

URL https://arxiv.org/abs/2104.09864.

- Suzgun et al. (2022)

M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung,
A. Chowdhery, Q. V. Le, E. H. Chi, D. Zhou, and J. Wei.

Challenging big-bench tasks and whether chain-of-thought can solve
them, 2022.

- Team (2024)

Q. Team.

Introducing qwen1.5, February 2024.

URL https://qwenlm.github.io/blog/qwen1.5/.

- Tenney et al. (2020)

I. Tenney, J. Wexler, J. Bastings, T. Bolukbasi, A. Coenen, S. Gehrmann,
E. Jiang, M. Pushkarna, C. Radebaugh, E. Reif, and A. Yuan.

The language interpretability tool: Extensible, interactive
visualizations and analysis for nlp models, 2020.

URL https://arxiv.org/abs/2008.05122.

- Touvron et al. (2023)

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix,
B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin,
E. Grave, and G. Lample.

Llama: Open and efficient foundation language models, 2023.

- Vaswani et al. (2017)

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin.

Attention is all you need.

CoRR, abs/1706.03762, 2017.

URL http://arxiv.org/abs/1706.03762.

- Weidinger et al. (2021)

L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang, M. Cheng,
M. Glaese, B. Balle, A. Kasirzadeh, Z. Kenton, S. Brown, W. Hawkins,
T. Stepleton, C. Biles, A. Birhane, J. Haas, L. Rimell, L. A. Hendricks,
W. Isaac, S. Legassick, G. Irving, and I. Gabriel.

Ethical and social risks of harm from language models, 2021.

URL https://arxiv.org/abs/2112.04359.

- xAI (2024)

xAI.

grok-1, 2024.

URL https://github.com/xai-org/grok-1.

- XLA (2019)

XLA.

Xla: Optimizing compiler for tensorflow, 2019.

URL https://www.tensorflow.org/xla.

- Xu et al. (2021)

Y. Xu, H. Lee, D. Chen, B. A. Hechtman, Y. Huang, R. Joshi, M. Krikun,
D. Lepikhin, A. Ly, M. Maggioni, R. Pang, N. Shazeer, S. Wang, T. Wang,
Y. Wu, and Z. Chen.

GSPMD: general and scalable parallelization for ML computation
graphs.

CoRR, abs/2105.04663, 2021.

URL https://arxiv.org/abs/2105.04663.

- Yang et al. (2023)

J. Yang, A. Prabhakar, K. Narasimhan, and S. Yao.

Intercode: Standardizing and benchmarking interactive coding with
execution feedback, 2023.

URL https://arxiv.org/abs/2306.14898.

- Zhang and Sennrich (2019)

B. Zhang and R. Sennrich.

Root mean square layer normalization.

CoRR, abs/1910.07467, 2019.

URL http://arxiv.org/abs/1910.07467.

- Zheng et al. (2023)

L. Zheng, W.-L. Chiang, Y. Sheng, T. Li, S. Zhuang, Z. Wu, Y. Zhuang, Z. Li,
Z. Lin, E. Xing, et al.

Lmsys-chat-1m: A large-scale real-world llm conversation dataset.

arXiv preprint arXiv:2309.11998, 2023.

Generated on Thu Sep 5 17:42:31 2024 by LaTeXML
