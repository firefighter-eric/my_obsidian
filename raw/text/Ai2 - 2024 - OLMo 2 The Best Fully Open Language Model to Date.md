# OLMo 2: The Best Fully Open Language Model to Date

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Ai2 - 2024 - OLMo 2 The Best Fully Open Language Model to Date.html`
- Source URL: https://allenai.org/blog/olmo2
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

- Open models

### Open models

- Olmo

- Tülu 3

- Molmo

- Playground

- Language models

- Multimodal models

- Evaluation frameworks

- Open data

- Applications

### AI for science

- Asta

- AstaBench

- Research with Asta

- Asta leaderboards

- Semantic Scholar

- All projects

### AI for the planet

- OlmoEarth

- EarthRanger

- Skylight

- Climate Modeling

- All projects

### AI for robotics

- Embodied AI

- Research

### Research

- Latest

- Papers

- Research principles

- News

- Institute

### Institute

- About

- Careers

- Media center

# OLMo 2: The best fully open language model to date

November 26, 2024

Ai2

Share

ModelsTech ReportAPIDemo

Since the release of the first OLMo in February 2024, we’ve seen rapid growth in the open language model ecosystem, and a narrowing of the performance gap between open and proprietary models. OLMo-0424 saw a notable boost in downstream performance relative to our first release in February. We were also excited by increasing participation in fully open model development, notably including LLM360’s Amber, M-A-P’s Neo models, and DCLM’s baseline models. In September, we released OLMoE, a mixture-of-experts model and the first among its fully open peers to be on the Pareto frontier of performance and size.

Because fully open science requires more than just open weights, we are excited to share a new round of OLMo updates–including weights, data, code, recipes, intermediate checkpoints, and instruction–tuned models—with the broader language modeling community!

# Announcing OLMo 2

We introduce OLMo 2, a new family of 7B and 13B models trained on up to 5T tokens. These models are on par with or better than equivalently sized fully open models, and competitive with open-weight models such as Llama 3.1 on English academic benchmarks.

We achieved this through investing in our core model development competencies, while diving deep into often under-discussed but critical aspects of model development, including:

- Training Stability. Long model training runs can be plagued by training instabilities and loss spikes, which are known to correlate with lower final model performance. Our forthcoming technical report will discuss techniques we used to improve the stability of long pretraining runs, which were critical to ensuring performance of the final trained model.

- Staged Training: Interventions during Late Pretraining. Pretraining is slow and expensive, which moves us to seek solutions to overcome knowledge or capability deficiencies discovered during the course of long training runs. We discuss the role of learning rate annealing and data curriculum as interventions that can be applied late in the pretraining process to “patch” model capabilities that weren’t successfully acquired earlier in training.

- State-of-the-art Post-training Recipes. We applied our state-of-the-art post-training methodology from Tülu 3 to OLMo 2 models, to create OLMo 2-Instruct models. Play with OLMo 2-Instruct-13B, our most capable OLMo 2 model, in the Ai2 playground.

- Actionable Evaluation Framework. For OLMo 2, we established clear performance goals and task scaling laws and designed an evaluation framework (Open Language Modeling Evaluation System, OLMES) that helped guide improvement through development stages. OLMES consists of a suite of 20 evaluation benchmarks to assess models’ core capabilities such as knowledge recall and commonsense, general, and mathematical reasoning, transformed to maximize signal-to-noise ratio for assessing modeling improvements in small-scale experiments.

We present a summary of key aspects of OLMo 2 below; a technical report will be available soon.

We compare OLMo 2 to other open models using a selection of tasks in OLMES. We group benchmarks into development, which we tracked during OLMo development (e.g., ARC Challenge, HellaSwag, WinoGrande, MMLU, DROP, and Natural Questions) and unseen, on which we did not compute metrics until model development was completed (e.g., AGIEval, MMLU Pro, GSM8k, TriviaQA).

We compare OLMo 2 to a set of baseline models, which we group into three families:

- Open weight models: models released with only their final checkpoint, and very limited to no information about their training data and recipe is known;

- Partially open models: models released with weights, and most of the data (or details necessary to reproduce them)are either released or known;

- Fully open models: models released with weights, training data, code, and evaluation in full, and thus can be fully inspected and reproduced.

First, we find that OLMo 2 7B and 13B are the best fully-open models to-date, often outperforming open weight models of equivalent size. Not only do we observe a dramatic improvement in performance across all tasks compared to our earlier OLMo 0424 model but, notably, OLMo 2 7B outperforms LLama-3.1 8B and OLMo 2 13B outperforms Qwen 2.5 7B despite its lower total training FLOPs. The OLMo 2 models sit at the Pareto frontier of training FLOPs vs model average performance (see figure above).

Overall, we find that gains observed on development metrics largely translate to our unseen evaluation suite. Of course, we have no guarantee that tasks we consider unseen during development of OLMo 2 are not part of the development set of other models we compare to. Nevertheless, we think it should be standard practice for model developers to keep a subset of evaluation tasks unseen; further, we encourage other open-weight model developers to clearly state which tasks were used as reference during model development.

# Pretraining OLMo 2

OLMo 2’s architecture is similar to the first OLMo but with several key changes to improve training stability such as switching from a nonparametric layer norm to RMSNorm (Zhang and Sennrich 2019) reordering the layer norm as in Liu et al (2022) and employing QK-Norm from Dehghani et al (2023), and replacing absolute positional embeddings with rotary positional embedding as in Su et al (2023). We also employ Z-loss regularization as seen in Wortsman et al (2023) and the Chameleon paper, and an improved initialization that better preserves the scale of activations and gradients across layers. Further details will be discussed in our forthcoming technical report.

OLMo 2 is pretrained in two stages, using a curriculum approach similar to Blakeney et al (2024).

In the first stage, which covers over 90% of the total pretraining budget, we use the OLMo-Mix-1124, a collection of approximately 3.9 trillion tokens sourced from DCLM, Dolma, Starcoder, and Proof Pile II. OLMo 2 7B is trained for approximately one epoch on this dataset, while OLMo 2 13B is trained for 1.2 epochs up to 5T tokens.

In the second stage, we curate a mixture of (a) web data that has been filtered for high quality and (b) a collection of domain-specific high-quality data (academic content, Q&A forums, instruction data, and math workbooks, both synthetic and human generated). This collection is available as Dolmino-Mix-1124. In total, it consists of 843 billion tokens, which we sample to create 3 mixes of 50 billion, 100 billion, and 300 billion tokens each, each of which has 50% data from (a) and (b).

For OLMo 2 7B, we train 3 copies from the final stage 1 checkpoint on the 50B mix with different data order. Consistent with previous OLMo versions, we anneal the learning rate linearly to zero from where it left off after Stage 1 for each of these. Then, we merge them together to obtain the final base checkpoint using a technique called model souping (Wortsman et al, 2022). For OLMo 2 13B, we repeat this process, but create three models using 100B tokens, and one other model using 300B tokens. They are merged to create the final 13B base checkpoint.

# Making OLMo 2 Instruct

Last week, we released Tülu 3, our family of state-of-the-art, fully-open post-trained models alongside data, code, recipes and more. These recipes combine multiple types of training techniques, including supervised finetuning (SFT) on model prompt completions, preference tuning with DPO, and reinforcement learning with verifiable rewards (RLVR). We applied our best recipe to the OLMo 2 models, and evaluated them on the Tülu 3 evaluation suite implemented in OLMES, which consists of benchmarks assessing models’ instruction-following, knowledge recall, and math and general reasoning capabilities.

Our Instruct variants of OLMo 2 are competitive with the best open-weight models, with OLMo 2 13B Instruct outperforming Qwen 2.5 14B instruct, Tülu 3 8B, and Llama 3.1 8B instruct models.

We are excited to find that the Tülu 3 recipe can be largely applied to OLMo 2 models without the need for expensive customizations. For example, we removed models from our completions pool to remove any restrictions on the use of model outputs for derivative models. Additionally, we updated the preference data to incorporate on-policy completions generated by our OLMo 2 models. Otherwise, the supervised finetuning (SFT) mix and preference tuning process remain largely unchanged. Most of the changes at these first two stages are differences in the learning rates. For the final stage, Reinforcement Learning with Verifiable Rewards (RLVR), we also saw consistent improvements across key evaluations such as GSM8K and MATH for both the 7B and 13B models. For more details on Tülu 3, check out our technical blog post.

# Artifacts

- Demo: playground.allenai.org

- Paper: arxiv.org/abs/2501.00656

- OLMo 2 base models:

- allenai/OLMo-2-1124-7B

- allenai/OLMo-2-1124-13B

- OLMo 2 instruct models:

- allenai/OLMo-2-1124-7B-Instruct

- allenai/OLMo-2-1124-13B-Instruct

- Pretraining dataset:

- Stage 1: allenai/olmo-mix-1124

- Stage 2: allenai/dolmino-mix-1124

- Post-training dataset:

- Tülu 3 SFT Mix: allenai/tulu-3-sft-olmo-2-mixture

- Preference data:

- For OLMo 2 7B: allenai/olmo-2-1124-7b-preference-mix

- For OLMo 2 13B: allenai/olmo-2-1124-13b-preference-mix

- RLVR mix: allenai/RLVR-GSM-MATH-IF-Mixed-Constraints

- OLMo 2 HuggingFace Collection

## Subscribe to receive monthly updates about the latest Ai2 news.

First Name

Last Name

Email

Sign up

Contact us

Questions about our work, or need support with one of our technologies?

Get in touch

Resources

- Media center

- Documentation

- Careers

- Team directory

Community

- Discord

- Reddit

- X/Twitter

- GitHub

- Hugging Face

- LinkedIn

- Bluesky

- Threads

Legal

- Terms of use

- Privacy policy

- DMCA policy

- Business code of conduct

- Responsible use

© The Allen Institute for Artificial Intelligence - All Rights Reserved.
