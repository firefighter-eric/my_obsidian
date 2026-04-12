# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- Source HTML: `raw/html/Yu et al. - 2025 - DAPO An Open-Source LLM Reinforcement Learning System at Scale.html`
- Source URL: https://arxiv.org/html/2503.14476
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

- 1 Introduction

- 2 Preliminary

- 2.1 Proximal Policy Optimization (PPO)

- 2.2 Group Relative Policy Optimization (GRPO)

- 2.3 Removing KL Divergence

- 2.4 Rule-based Reward Modeling

- 3 DAPO

- 3.1 Raise the Ceiling: Clip-Higher

- 3.2 The More the Merrier: Dynamic Sampling

- 3.3 Rebalancing Act: Token-Level Policy Gradient Loss

- 3.4 Hide and Seek: Overlong Reward Shaping

- 3.5 Dataset Transformation

- 4 Experiments

- 4.1 Training Details

- 4.2 Main Results

- 4.3 Training Dynamics

- 4.4 Case Study

- 5 Conclusion

- 6 Dataset Transformation

- 7 Supplementary Case

1]ByteDance Seed 2Institute for AI Industry Research (AIR), Tsinghua University
3]The University of Hong Kong
4]SIA-Lab of Tsinghua AIR and ByteDance Seed
\contributionFull author list in Contributions

# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

(March 17, 2025)

###### Abstract

Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results.
We propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model.
Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework 111https://github.com/volcengine/verl, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.

\correspondence

, 
\checkdata[Project Page]https://dapo-sia.github.io/

## 1 Introduction

Test-time scaling such as OpenAI’s o1 [1] and DeepSeek’s R1 [2] brings a profound paradigm shift to Large Language Models (LLMs) [3, 4, 5, 6, 7]. Test-time scaling enables longer Chain-of-Thought thinking and induces sophisticated reasoning behaviors, which makes the models superior in competitive math and coding tasks like AIME and Codeforces.

The central technique driving the revolution is large-scale Reinforcement Learning (RL), which elicits complex reasoning behaviors such as self-verification and iterative refinement. However, the actual algorithm and key recipe for scalable RL training remains a myth, hidden from technical reports of existing reasoning models [1, 2, 8, 9, 10, 11]. In this paper, we reveal significant obstacles in large-scale RL training and open-source a scalable RL system with fully open-sourced algorithm, training code and dataset that provides democratized solutions with industry-level RL results.

We experiment over Qwen2.5-32B [12] as the pretrained model for RL. In our initial GRPO run, we achieved only 30 points on AIME — a performance significantly below DeepSeek’s RL (47 points). A thorough analysis reveals that the naive GRPO baseline suffers from several key issues such as entropy collapse, reward noise, and training instability. The broader community has encountered similar challenges in reproducing DeepSeek’s results [13, 14, 15, 16, 17, 18, 19] suggesting that critical training details may have been omitted in the R1 paper that are required to develop an industry-level, large-scale, and reproducible RL system.

To close this gap, we release an open-source state-of-the-art system for large-scale LLM RL, which achieves 50 points on AIME 2024 based on Qwen2.5-32B model, outperforming previous state-of-the-art results achieved by DeepSeek-R1-Zero-Qwen-32B [2] (47 points) using 50% training steps (Figure 1). We propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm, and introduce 4 key techniques to make RL shine in the long-CoT RL scenario. Details are presented in Section 3.

- 1.

Clip-Higher, which promotes the diversity of the system and avoids entropy collapse;

- 2.

Dynamic Sampling, which improves training efficiency and stability;

- 3.

Token-Level Policy Gradient Loss, which is critical in long-CoT RL scenarios;

- 4.

Overlong Reward Shaping, which reduces reward noise and stabilizes training.

Our implementation is based on verl [20]. By fully releasing our state-of-the-art RL system including training code and data, we aim to reveal valuable insights to large-scale LLM RL that benefit the larger community.

## 2 Preliminary

### 2.1 Proximal Policy Optimization (PPO)

PPO [21] introduces a clipped surrogate objective for policy optimization. By constraining the policy updates within a proximal region of the previous policy using clip, PPO stabilizes training and improves sample efficiency. Specifically, PPO updates the policy by maximizing the following objective:

𝒥PPO⁢(θ)=𝔼(q,a)∼𝒟,o≤t∼πθold(⋅∣q)⁢[min⁡(πθ⁢(ot∣q,o<t)πθold⁢(ot∣q,o<t)⁢A^t,clip⁢(πθ⁢(ot∣q,o<t)πθold⁢(ot∣q,o<t),1−ε,1+ε)⁢A^t)],\displaystyle\mathcal{J}_{\text{PPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D}%
,o_{\leq t}\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)}\Bigg{[}\min\Bigg{(}%
\frac{\pi_{\theta}(o_{t}\mid q,o_{<t})}{\pi_{\theta_{\text{old}}}(o_{t}\mid q,%
o_{<t})}\hat{A}_{t},\ \text{clip}\Bigg{(}\frac{\pi_{\theta}(o_{t}\mid q,o_{<t}%
)}{\pi_{\theta_{\text{old}}}(o_{t}\mid q,o_{<t})},1-\varepsilon,1+\varepsilon%
\Bigg{)}\hat{A}_{t}\Bigg{)}\Bigg{]},caligraphic_J start_POSTSUBSCRIPT PPO end_POSTSUBSCRIPT ( italic_θ ) = blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , italic_o start_POSTSUBSCRIPT ≤ italic_t end_POSTSUBSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT [ roman_min ( divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT ) end_ARG over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT , clip ( divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT < italic_t end_POSTSUBSCRIPT ) end_ARG , 1 - italic_ε , 1 + italic_ε ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT ) ] ,

(1)

where (q,a)𝑞𝑎(q,a)( italic_q , italic_a ) is a question-answer pair from the data distribution 𝒟𝒟\mathcal{D}caligraphic_D, ε𝜀\varepsilonitalic_ε is the clipping range of importance sampling ratio, and A^tsubscript^𝐴𝑡\hat{A}_{t}over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT is an estimator of the advantage at time step t𝑡titalic_t. Given the value function V𝑉Vitalic_V and the reward function R𝑅Ritalic_R, A^tsubscript^𝐴𝑡\hat{A}_{t}over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT is computed using the Generalized Advantage Estimation (GAE) [22]:

A^tGAE⁢(γ,λ)=∑l=0∞(γ⁢λ)l⁢δt+l,superscriptsubscript^𝐴𝑡GAE𝛾𝜆superscriptsubscript𝑙0superscript𝛾𝜆𝑙subscript𝛿𝑡𝑙\displaystyle\hat{A}_{t}^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(%
\gamma\lambda)^{l}\delta_{t+l},over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_t end_POSTSUBSCRIPT start_POSTSUPERSCRIPT GAE ( italic_γ , italic_λ ) end_POSTSUPERSCRIPT = ∑ start_POSTSUBSCRIPT italic_l = 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT ∞ end_POSTSUPERSCRIPT ( italic_γ italic_λ ) start_POSTSUPERSCRIPT italic_l end_POSTSUPERSCRIPT italic_δ start_POSTSUBSCRIPT italic_t + italic_l end_POSTSUBSCRIPT ,

(2)

where

δl=Rl+γ⁢V⁢(sl+1)−V⁢(sl),0≤γ,λ≤1.formulae-sequencesubscript𝛿𝑙subscript𝑅𝑙𝛾𝑉subscript𝑠𝑙1𝑉subscript𝑠𝑙formulae-sequence0𝛾𝜆1\delta_{l}=R_{l}+\gamma V(s_{l+1})-V(s_{l}),\quad 0\leq\gamma,\lambda\leq 1.italic_δ start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT = italic_R start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT + italic_γ italic_V ( italic_s start_POSTSUBSCRIPT italic_l + 1 end_POSTSUBSCRIPT ) - italic_V ( italic_s start_POSTSUBSCRIPT italic_l end_POSTSUBSCRIPT ) , 0 ≤ italic_γ , italic_λ ≤ 1 .

(3)

### 2.2 Group Relative Policy Optimization (GRPO)

Compared to PPO, GRPO eliminates the value function and estimates the advantage in a group-relative manner. For a specific question-answer pair (q,a)𝑞𝑎(q,a)( italic_q , italic_a ), the behavior policy πθoldsubscript𝜋subscript𝜃old\pi_{\theta_{\text{old}}}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT samples a group of G𝐺Gitalic_G individual responses {oi}i=1Gsuperscriptsubscriptsubscript𝑜𝑖𝑖1𝐺\{o_{i}\}_{i=1}^{G}{ italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT. Then, the advantage of the i𝑖iitalic_i-th response is calculated by normalizing the group-level rewards {Ri}i=1Gsuperscriptsubscriptsubscript𝑅𝑖𝑖1𝐺\{R_{i}\}_{i=1}^{G}{ italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT:

A^i,t=ri−mean⁢({Ri}i=1G)std⁢({Ri}i=1G).subscript^𝐴𝑖𝑡subscript𝑟𝑖meansuperscriptsubscriptsubscript𝑅𝑖𝑖1𝐺stdsuperscriptsubscriptsubscript𝑅𝑖𝑖1𝐺\hat{A}_{i,t}=\frac{r_{i}-\text{mean}(\{R_{i}\}_{i=1}^{G})}{\text{std}(\{R_{i}%
\}_{i=1}^{G})}.over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT = divide start_ARG italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - mean ( { italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ) end_ARG start_ARG std ( { italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ) end_ARG .

(4)

Similar to PPO, GRPO adopts a clipped objective, together with a directly imposed KL penalty term:

𝒥GRPO⁢(θ)subscript𝒥GRPO𝜃\displaystyle\mathcal{J}_{\text{GRPO}}(\theta)caligraphic_J start_POSTSUBSCRIPT GRPO end_POSTSUBSCRIPT ( italic_θ )
=𝔼(q,a)∼𝒟,{oi}i=1G∼πθold(⋅∣q)\displaystyle=\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{%
\theta_{\text{old}}}(\cdot\mid q)}= blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT

(5)

[1G∑i=1G1|oi|∑t=1|oi|(min(ri,t(θ)A^i,t,clip(ri,t(θ),1−ε,1+ε)A^i,t)−βDKL(πθ||πref))],\displaystyle\Bigg{[}\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_%
{i}|}\Bigg{(}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,%
t}(\theta),1-\varepsilon,1+\varepsilon\Big{)}\hat{A}_{i,t}\Big{)}-\beta D_{%
\text{KL}}(\pi_{\theta}||\pi_{\text{ref}})\Bigg{)}\Bigg{]},[ divide start_ARG 1 end_ARG start_ARG italic_G end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT divide start_ARG 1 end_ARG start_ARG | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT ( roman_min ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , clip ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ε , 1 + italic_ε ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ) - italic_β italic_D start_POSTSUBSCRIPT KL end_POSTSUBSCRIPT ( italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT | | italic_π start_POSTSUBSCRIPT ref end_POSTSUBSCRIPT ) ) ] ,

where

ri,t⁢(θ)=πθ⁢(oi,t∣q,oi,<t)πθold⁢(oi,t∣q,oi,<t).subscript𝑟𝑖𝑡𝜃subscript𝜋𝜃conditionalsubscript𝑜𝑖𝑡𝑞subscript𝑜𝑖absent𝑡subscript𝜋subscript𝜃oldconditionalsubscript𝑜𝑖𝑡𝑞subscript𝑜𝑖absent𝑡r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text%
{old}}}(o_{i,t}\mid q,o_{i,<t})}.italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) = divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT italic_i , < italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT italic_i , < italic_t end_POSTSUBSCRIPT ) end_ARG .

(6)

It is also worth noting that GRPO computes the objective at the sample-level. To be exact, GRPO first calculates the mean loss within each generated sequence, before averaging the loss of different samples. As we will be discussing in Section 3.3, such difference may have an impact on the performance of the algorithm.

(a) Accuracies on AIME.

(b) Entropy of actor model.

### 2.3 Removing KL Divergence

The KL penalty term is used to regulate the
divergence between the online policy and the frozen reference policy.
In the RLHF scenario [23], the goal of RL is to align the model behavior without diverging too far from the initial model.
However, during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary. Therefore, we will exclude the KL term from our proposed algorithm.

### 2.4 Rule-based Reward Modeling

The use of reward model usually suffers from the reward hacking problem [24, 25, 26, 27, 28, 29].
Instead, we directly use the final accuracy of a verifiable task as the outcome reward, computed using the following rule:

R⁢(y^,y)={1,is_equivalent⁢(y^,y)−1,otherwise𝑅^𝑦𝑦cases1is_equivalent^𝑦𝑦1otherwiseR(\hat{y},y)=\begin{cases}1,&\texttt{is\_equivalent}(\hat{y},y)\\
-1,&\text{otherwise}\end{cases}italic_R ( over^ start_ARG italic_y end_ARG , italic_y ) = { start_ROW start_CELL 1 , end_CELL start_CELL is_equivalent ( over^ start_ARG italic_y end_ARG , italic_y ) end_CELL end_ROW start_ROW start_CELL - 1 , end_CELL start_CELL otherwise end_CELL end_ROW

(7)

where y𝑦yitalic_y is the ground-truth answer and y^^𝑦\hat{y}over^ start_ARG italic_y end_ARG is the predicted answer.
This is proved to be an effective approach to activating the base model’s reasoning capability, as shown in multiple domains such as automated theorem proving [30, 31, 32, 33], computer programming [34, 35, 36, 37], and mathematics competition [2].

## 3 DAPO

We propose the Decouple Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm. DAPO samples a group of outputs {oi}i=1Gsuperscriptsubscriptsubscript𝑜𝑖𝑖1𝐺\{o_{i}\}_{i=1}^{G}{ italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT for each question q𝑞qitalic_q paired with the answer a𝑎aitalic_a, and optimizes the policy via the following objective:

𝒥DAPO⁢(θ)=subscript𝒥DAPO𝜃absent\displaystyle\mathcal{J}_{\text{DAPO}}(\theta)=caligraphic_J start_POSTSUBSCRIPT DAPO end_POSTSUBSCRIPT ( italic_θ ) =
𝔼(q,a)∼𝒟,{oi}i=1G∼πθold(⋅∣q)\displaystyle\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{%
\theta_{\text{old}}}(\cdot\mid q)}blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT

(8)

[1∑i=1G|oi|⁢∑i=1G∑t=1|oi|min⁡(ri,t⁢(θ)⁢A^i,t,clip⁢(ri,t⁢(θ),1−εlow,1+εhigh)⁢A^i,t)]delimited-[]1superscriptsubscript𝑖1𝐺subscript𝑜𝑖superscriptsubscript𝑖1𝐺superscriptsubscript𝑡1subscript𝑜𝑖subscript𝑟𝑖𝑡𝜃subscript^𝐴𝑖𝑡clipsubscript𝑟𝑖𝑡𝜃1subscript𝜀low1subscript𝜀highsubscript^𝐴𝑖𝑡\displaystyle\Bigg{[}\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{%
|o_{i}|}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,t}(%
\theta),1-{\varepsilon_{\text{low}}},1+{\varepsilon_{\text{high}}}\Big{)}\hat{%
A}_{i,t}\Big{)}\Bigg{]}[ divide start_ARG 1 end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT roman_min ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , clip ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , 1 + italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ) ]

s.t.
0<|{oi∣is_equivalent⁢(a,oi)}|<G,0conditional-setsubscript𝑜𝑖is_equivalent𝑎subscript𝑜𝑖𝐺\displaystyle 0<\Big{|}\{o_{i}\mid\texttt{is\_equivalent}(a,o_{i})\}\Big{|}<G,0 < | { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ is_equivalent ( italic_a , italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } | < italic_G ,

where

ri,t⁢(θ)=πθ⁢(oi,t∣q,oi,<t)πθold⁢(oi,t∣q,oi,<t),A^i,t=Ri−mean⁢({Ri}i=1G)std⁢({Ri}i=1G).formulae-sequencesubscript𝑟𝑖𝑡𝜃subscript𝜋𝜃conditionalsubscript𝑜𝑖𝑡𝑞subscript𝑜𝑖absent𝑡subscript𝜋subscript𝜃oldconditionalsubscript𝑜𝑖𝑡𝑞subscript𝑜𝑖absent𝑡subscript^𝐴𝑖𝑡subscript𝑅𝑖meansuperscriptsubscriptsubscript𝑅𝑖𝑖1𝐺stdsuperscriptsubscriptsubscript𝑅𝑖𝑖1𝐺r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text%
{old}}}(o_{i,t}\mid q,o_{i,<t})},\quad\hat{A}_{i,t}=\frac{R_{i}-\text{mean}(\{%
R_{i}\}_{i=1}^{G})}{\text{std}(\{R_{i}\}_{i=1}^{G})}.italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) = divide start_ARG italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT italic_i , < italic_t end_POSTSUBSCRIPT ) end_ARG start_ARG italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ∣ italic_q , italic_o start_POSTSUBSCRIPT italic_i , < italic_t end_POSTSUBSCRIPT ) end_ARG , over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT = divide start_ARG italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT - mean ( { italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ) end_ARG start_ARG std ( { italic_R start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ) end_ARG .

(9)

The full algorithm can be found in Algorithm 1. In this section, we will introduce the key techniques associated with DAPO.

### 3.1 Raise the Ceiling: Clip-Higher

In our initial experiments using naive PPO [21] or GRPO [38], we observed the entropy collapse phenomenon: the entropy of the policy decreases quickly as training progresses (Figure 2(b)). The sampled responses of certain groups tend to be nearly identical. This indicates limited exploration and early deterministic policy, which can hinder the scaling process.

We propose the Clip-Higher strategy to address this issue. Clipping over the importance sampling ratio is introduced in Clipped Proximal Policy Optimization (PPO-Clip) [21] to restrict the trust region and enhance the stability of RL.
We identify that the upper clip can restrict the exploration of the policy, where making an ‘exploitation’ token more probable is much easier yet the probability of an unlikely ‘exploration’ token is too tightly bounded to be uplifted.

Concretely, when ε=0.2𝜀0.2\varepsilon=0.2italic_ε = 0.2 (the default value of most algorithms) and A^i,t>0subscript^𝐴𝑖𝑡0\hat{A}_{i,t}>0over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT > 0 (the system tries to increase the probability), consider two actions with probabilities πθold⁢(oi∣q)=0.01subscript𝜋subscript𝜃oldconditionalsubscript𝑜𝑖𝑞0.01\pi_{\theta_{\text{old}}}(o_{i}\mid q)=0.01italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ italic_q ) = 0.01 and 0.90.90.90.9. The upper bounds of the increased probabilities πθ⁢(oi∣q)subscript𝜋𝜃conditionalsubscript𝑜𝑖𝑞\pi_{\theta}(o_{i}\mid q)italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ italic_q ) are 0.0120.0120.0120.012 and 1.081.081.081.08, respectively (πθold⋅(1+ϵ))⋅subscript𝜋subscript𝜃old1italic-ϵ\left(\pi_{\theta_{\text{old}}}\cdot(1+\epsilon)\right)( italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ⋅ ( 1 + italic_ϵ ) ).
This implies that ‘exploitation’ tokens with a higher probability (e.g., 0.9) are not constrained to get even extremely larger probabilities like 0.999. Conversely, for low-probability ‘exploration’ tokens, achieving a non-trivial increase in probability is considerably more challenging.
Empirically, we also observe that the mean probability of up-clipped tokens is low: πθ⁢(oi∣q)<0.2subscript𝜋𝜃conditionalsubscript𝑜𝑖𝑞0.2\pi_{\theta}(o_{i}\mid q)<0.2italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT ( italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ italic_q ) < 0.2 (Figure 3(a)). This finding supports our intuition that the upper clipping threshold indeed restricts the probability increase of low-probability ‘exploration’ tokens, thereby potentially constraining the exploration of the system.

Adhering to the Clip-Higher strategy, we decouple the lower and higher clipping range as εlowsubscript𝜀low\varepsilon_{\text{low}}italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT and εhighsubscript𝜀high\varepsilon_{\text{high}}italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT, as highlighted in Equation 10:

𝒥DAPO⁢(θ)=subscript𝒥DAPO𝜃absent\displaystyle\mathcal{J}_{\text{DAPO}}(\theta)=caligraphic_J start_POSTSUBSCRIPT DAPO end_POSTSUBSCRIPT ( italic_θ ) =
𝔼(q,a)∼𝒟,{oi}i=1G∼πθold(⋅∣q)\displaystyle\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{%
\theta_{\text{old}}}(\cdot\mid q)}blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT

(10)

[1∑i=1G|oi|⁢∑i=1G∑t=1|oi|min⁡(ri,t⁢(θ)⁢A^i,t,clip⁢(ri,t⁢(θ),1−εlow,1+εhigh)⁢A^i,t)]delimited-[]1superscriptsubscript𝑖1𝐺subscript𝑜𝑖superscriptsubscript𝑖1𝐺superscriptsubscript𝑡1subscript𝑜𝑖subscript𝑟𝑖𝑡𝜃subscript^𝐴𝑖𝑡clipsubscript𝑟𝑖𝑡𝜃1subscript𝜀low1subscript𝜀highsubscript^𝐴𝑖𝑡\displaystyle\Bigg{[}\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{%
|o_{i}|}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,t}(%
\theta),1-{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}%
\varepsilon_{\text{low}}},1+{\color[rgb]{1,0,0}\definecolor[named]{%
pgfstrokecolor}{rgb}{1,0,0}\varepsilon_{\text{high}}}\Big{)}\hat{A}_{i,t}\Big{%
)}\Bigg{]}[ divide start_ARG 1 end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT roman_min ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , clip ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , 1 + italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ) ]

s.t.
0<|{oi∣is_equivalent⁢(a,oi)}|<G.0conditional-setsubscript𝑜𝑖is_equivalent𝑎subscript𝑜𝑖𝐺\displaystyle 0<\Big{|}\{o_{i}\mid\texttt{is\_equivalent}(a,o_{i})\}\Big{|}<G.0 < | { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ is_equivalent ( italic_a , italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } | < italic_G .

We increase the value of εhighsubscript𝜀high\varepsilon_{\text{high}}italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT to leave more room for the increase of low-probability tokens. As shown in Figure 2, this adjustment effectively enhances the policy’s entropy and facilitates the generation of more diverse samples.
We keep εlowsubscript𝜀low\varepsilon_{\text{low}}italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT as it is, because increasing it will suppress the probability of these tokens to 00, resulting in the collapse of the sampling space.

(a) Mean up-clipped probability.

(b) The proportion of samples with an accuracy of 1.

### 3.2 The More the Merrier: Dynamic Sampling

Existing RL algorithm suffers from the gradient-decreasing problem when some prompts have accuracy equal to 1. For example for GRPO, if all outputs {oi}i=1Gsuperscriptsubscriptsubscript𝑜𝑖𝑖1𝐺\{o_{i}\}_{i=1}^{G}{ italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT of a particular prompt are correct and receive the same reward, the resulting advantage for this group is zero. A zero advantage results in zero policy gradients, shrinking the magnitude and increasing the noise sensitivity of the batch gradient, thereby degrading sample efficiency. Empirically, the number of samples with accuracy equal to 1 continues to increase, as shown in Figure 3(b). This means that the effective number of prompts in each batch keeps decreasing, which can lead to larger variance in gradient and dampens the gradient signals for model training.

To this end, we propose to over-sample and filter out prompts with the accuracy equal to 1 and 0 as illustrated in Equation 11, leaving all prompts in the batch with effective gradients and keeping a consistent number of prompts. The sampling cost for each batch is dynamic. Before training, we keep sampling until the batch is fully filled with samples whose accuracy is neither 0 nor 1.

𝒥DAPO⁢(θ)=subscript𝒥DAPO𝜃absent\displaystyle\mathcal{J}_{\text{DAPO}}(\theta)=caligraphic_J start_POSTSUBSCRIPT DAPO end_POSTSUBSCRIPT ( italic_θ ) =
𝔼(q,a)∼𝒟,{oi}i=1G∼πθold(⋅∣q)\displaystyle\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{%
\theta_{\text{old}}}(\cdot\mid q)}blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT

(11)

[1∑i=1G|oi|⁢∑i=1G∑t=1|oi|min⁡(ri,t⁢(θ)⁢A^i,t,clip⁢(ri,t⁢(θ),1−εlow,1+εhigh)⁢A^i,t)]delimited-[]1superscriptsubscript𝑖1𝐺subscript𝑜𝑖superscriptsubscript𝑖1𝐺superscriptsubscript𝑡1subscript𝑜𝑖subscript𝑟𝑖𝑡𝜃subscript^𝐴𝑖𝑡clipsubscript𝑟𝑖𝑡𝜃1subscript𝜀low1subscript𝜀highsubscript^𝐴𝑖𝑡\displaystyle\Bigg{[}\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{%
|o_{i}|}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,t}(%
\theta),1-{\varepsilon_{\text{low}}},1+{\varepsilon_{\text{high}}}\Big{)}\hat{%
A}_{i,t}\Big{)}\Bigg{]}[ divide start_ARG 1 end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT roman_min ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , clip ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , 1 + italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ) ]

s.t.
0<|{oi∣is_equivalent⁢(a,oi)}|<G.0conditional-setsubscript𝑜𝑖is_equivalent𝑎subscript𝑜𝑖𝐺\displaystyle{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0%
}0<\Big{|}\{o_{i}\mid\texttt{is\_equivalent}(a,o_{i})\}\Big{|}<G}.0 < | { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ is_equivalent ( italic_a , italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } | < italic_G .

Note that this strategy does not necessarily impede training efficiency, because the generation time is typically dominated by the generation of long-tail samples if the RL system is synchronized and the generation stage is not pipelined. Besides, we find that with dynamic sampling the experiment achieves the same performance faster as shown in Figure 6.

### 3.3 Rebalancing Act: Token-Level Policy Gradient Loss

The original GRPO algorithm employs a sample-level loss calculation, which involves first averaging the losses by token within each sample and then aggregating the losses across samples. In this approach, each sample is assigned an equal weight in the final loss computation. However, we find that this method of loss reduction introduces several challenges in the context of long-CoT RL scenarios.

Since all samples are assigned the same weight in the loss calculation, tokens within longer responses (which contain more tokens) may have a disproportionately lower contribution to the overall loss, which can lead to two adverse effects.
First, for high-quality long samples, this effect can impede the model’s ability to learn reasoning-relevant patterns within them.
Second, we observe that excessively long samples often exhibit low-quality patterns such as gibberish and repetitive words. Thus, sample-level loss calculation, due to its inability to effectively penalize those undesirable patterns in long samples, leads to an unhealthy increase in entropy and response length, as shown in
Figure 4(a) and Figure 4(b).

(a) Entropy of actor model’s generation probabilities.

(b) Average length of actor model-generated responses

We introduce a Token-level Policy Gradient Loss in the long-CoT RL scenario to address the above limitations:

𝒥DAPO⁢(θ)=subscript𝒥DAPO𝜃absent\displaystyle\mathcal{J}_{\text{DAPO}}(\theta)=caligraphic_J start_POSTSUBSCRIPT DAPO end_POSTSUBSCRIPT ( italic_θ ) =
𝔼(q,a)∼𝒟,{oi}i=1G∼πθold(⋅∣q)\displaystyle\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{%
\theta_{\text{old}}}(\cdot\mid q)}blackboard_E start_POSTSUBSCRIPT ( italic_q , italic_a ) ∼ caligraphic_D , { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ ∣ italic_q ) end_POSTSUBSCRIPT

(12)

[1∑i=1G|oi|⁢∑i=1G∑t=1|oi|min⁡(ri,t⁢(θ)⁢A^i,t,clip⁢(ri,t⁢(θ),1−εlow,1+εhigh)⁢A^i,t)],delimited-[]1superscriptsubscript𝑖1𝐺subscript𝑜𝑖superscriptsubscript𝑖1𝐺superscriptsubscript𝑡1subscript𝑜𝑖subscript𝑟𝑖𝑡𝜃subscript^𝐴𝑖𝑡clipsubscript𝑟𝑖𝑡𝜃1subscript𝜀low1subscript𝜀highsubscript^𝐴𝑖𝑡\displaystyle\Bigg{[}\frac{1}{\color[rgb]{1,0,0}\definecolor[named]{%
pgfstrokecolor}{rgb}{1,0,0}\sum_{i=1}^{G}|o_{i}|}{\color[rgb]{1,0,0}%
\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i%
}|}}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,t}(\theta%
),1-{\varepsilon_{\text{low}}},1+{\varepsilon_{\text{high}}}\Big{)}\hat{A}_{i,%
t}\Big{)}\Bigg{]},[ divide start_ARG 1 end_ARG start_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_ARG ∑ start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∑ start_POSTSUBSCRIPT italic_t = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT | italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT | end_POSTSUPERSCRIPT roman_min ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT , clip ( italic_r start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ( italic_θ ) , 1 - italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT , 1 + italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT ) over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT ) ] ,

s.t.
0<|{oi∣is_equivalent⁢(a,oi)}|<G.0conditional-setsubscript𝑜𝑖is_equivalent𝑎subscript𝑜𝑖𝐺\displaystyle 0<\Big{|}\{o_{i}\mid\texttt{is\_equivalent}(a,o_{i})\}\Big{|}<G.0 < | { italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ∣ is_equivalent ( italic_a , italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT ) } | < italic_G .

In this setting, longer sequences can have more influence on the overall gradient update compared to shorter sequences.
Moreover, from the perspective of individual tokens, if a particular generation pattern can lead to an increase or decrease in reward, it will be equally prompted or suppressed, regardless of the length of the response in which it appears.

### 3.4 Hide and Seek: Overlong Reward Shaping

In RL training, we typically set a maximum length for generation, with overlong samples truncated accordingly. We find that improper reward shaping for truncated samples can introduce reward noise and significantly disrupt the training process.

By default, we assign a punitive reward to truncated samples.
This approach may introduce noise into the training process, as a sound reasoning process can be penalized solely due to its excessive length. Such penalties can potentially confuse the model regarding the validity of its reasoning process.

To investigate the impact of this reward noise, we first apply an Overlong Filtering strategy which masks the loss of truncated samples. We find that this approach significantly stabilizes training and enhances performance, as demonstrated in Figure 5.

(a) Performance on AIME.

(b) Entropy of actor model.

Algorithm 1   DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization

Input initial policy model πθsubscript𝜋𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT; reawrd model R𝑅Ritalic_R; task prompts 𝒟𝒟\mathcal{D}caligraphic_D; hyperparameters ε𝚕𝚘𝚠,ε𝚑𝚒𝚐𝚑subscript𝜀𝚕𝚘𝚠subscript𝜀𝚑𝚒𝚐𝚑\varepsilon_{\mathtt{low}},\varepsilon_{\mathtt{high}}italic_ε start_POSTSUBSCRIPT typewriter_low end_POSTSUBSCRIPT , italic_ε start_POSTSUBSCRIPT typewriter_high end_POSTSUBSCRIPT

1: for step = 1,…,M do

2:     Sample a batch 𝒟bsubscript𝒟𝑏\mathcal{D}_{b}caligraphic_D start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT from 𝒟𝒟\mathcal{D}caligraphic_D

3:     Update the old policy model πθo⁢l⁢d←πθ←subscript𝜋subscript𝜃𝑜𝑙𝑑subscript𝜋𝜃\pi_{\theta_{old}}\leftarrow\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT italic_o italic_l italic_d end_POSTSUBSCRIPT end_POSTSUBSCRIPT ← italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT

4:     Sample G outputs {oi}i=1G∼πθold(⋅|q)\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|q){ italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT ∼ italic_π start_POSTSUBSCRIPT italic_θ start_POSTSUBSCRIPT old end_POSTSUBSCRIPT end_POSTSUBSCRIPT ( ⋅ | italic_q ) for each question q∈𝒟b𝑞subscript𝒟𝑏q\in\mathcal{D}_{b}italic_q ∈ caligraphic_D start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT

5:     Compute rewards {ri}i=1Gsuperscriptsubscriptsubscript𝑟𝑖𝑖1𝐺\{r_{i}\}_{i=1}^{G}{ italic_r start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT } start_POSTSUBSCRIPT italic_i = 1 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT italic_G end_POSTSUPERSCRIPT for each sampled output oisubscript𝑜𝑖o_{i}italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT by running R𝑅Ritalic_R

6:     Filter out oisubscript𝑜𝑖o_{i}italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT and add the remaining to the dynamic sampling buffer (Dynamic Sampling Equation 11)

7:     if buffer size nb<Nsubscript𝑛𝑏𝑁n_{b}<Nitalic_n start_POSTSUBSCRIPT italic_b end_POSTSUBSCRIPT < italic_N:

8:          continue

9:     For each oisubscript𝑜𝑖o_{i}italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT in the buffer, compute A^i,tsubscript^𝐴𝑖𝑡\hat{A}_{i,t}over^ start_ARG italic_A end_ARG start_POSTSUBSCRIPT italic_i , italic_t end_POSTSUBSCRIPT for the t-th token of oisubscript𝑜𝑖o_{i}italic_o start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT (Equation 9)

10:    for iteration = 1, …, μ𝜇\muitalic_μ do

11:         Update the policy model πθsubscript𝜋𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT by maximizing the DAPO objective (Equation 8)

Output πθsubscript𝜋𝜃\pi_{\theta}italic_π start_POSTSUBSCRIPT italic_θ end_POSTSUBSCRIPT

Furthermore, we propose Soft Overlong Punishment (Equation 13), a length-aware penalty mechanism designed to shape the reward for truncated samples.
Specifically, when the response length exceeds the predefined maximum value, we define a punishment interval. Within this interval, the longer the response, the greater the punishment it receives.
This penalty is added to the original rule-based correctness reward, thereby signaling to the model to avoid excessively long responses.

Rlength⁢(y)={0,|y|≤Lmax−Lcache(Lmax−Lcache)−|y|Lcache,Lmax−Lcache<|y|≤Lmax−1,Lmax<|y|subscript𝑅length𝑦cases0𝑦subscript𝐿maxsubscript𝐿cachesubscript𝐿maxsubscript𝐿cache𝑦subscript𝐿cachesubscript𝐿maxsubscript𝐿cache𝑦subscript𝐿max1subscript𝐿max𝑦R_{\text{length}}(y)=\begin{cases}0,&|y|\leq L_{\text{max}}-L_{\text{cache}}\\
\frac{(L_{\text{max}}-L_{\text{cache}})-|y|}{L_{\text{cache}}},&L_{\text{max}}%
-L_{\text{cache}}<|y|\leq L_{\text{max}}\\
-1,&L_{\text{max}}<|y|\end{cases}italic_R start_POSTSUBSCRIPT length end_POSTSUBSCRIPT ( italic_y ) = { start_ROW start_CELL 0 , end_CELL start_CELL | italic_y | ≤ italic_L start_POSTSUBSCRIPT max end_POSTSUBSCRIPT - italic_L start_POSTSUBSCRIPT cache end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL divide start_ARG ( italic_L start_POSTSUBSCRIPT max end_POSTSUBSCRIPT - italic_L start_POSTSUBSCRIPT cache end_POSTSUBSCRIPT ) - | italic_y | end_ARG start_ARG italic_L start_POSTSUBSCRIPT cache end_POSTSUBSCRIPT end_ARG , end_CELL start_CELL italic_L start_POSTSUBSCRIPT max end_POSTSUBSCRIPT - italic_L start_POSTSUBSCRIPT cache end_POSTSUBSCRIPT < | italic_y | ≤ italic_L start_POSTSUBSCRIPT max end_POSTSUBSCRIPT end_CELL end_ROW start_ROW start_CELL - 1 , end_CELL start_CELL italic_L start_POSTSUBSCRIPT max end_POSTSUBSCRIPT < | italic_y | end_CELL end_ROW

(13)

### 3.5 Dataset Transformation

Our dataset is sourced from the web and official competition homepages through a combination of web scraping and manual annotation.
The answers of math dataset typically come in a variety of formats, such as expression, formula and number, which makes it challenging to design comprehensive rules to parse them.
To provide accurate reward signals using rules and minimize errors introduced by formula parsers, inspired by AIME, we select and transform the answers into integers, which are easy to parse.
For example, if the original answer is expressed in the form of a+bc𝑎𝑏𝑐\frac{a+\sqrt{b}}{c}divide start_ARG italic_a + square-root start_ARG italic_b end_ARG end_ARG start_ARG italic_c end_ARG, we instruct the LLM to modify the question so that the expected answer becomes a+b+c𝑎𝑏𝑐a+b+citalic_a + italic_b + italic_c.
After selection and transformation, we obtained the DAPO-Math-17K dataset, which consists of 17K prompts, each paired with an integer as the answer.

## 4 Experiments

### 4.1 Training Details

In this work, we focus specifically on mathematical tasks to evaluate our algorithm, which can be readily transferred to other tasks. We adopt the verl framework [20] for training. We use naive GRPO [38] as our baseline algorithm and estimate advantages using group reward normalization.

For hyper-parameters, we utilize the AdamW [39] optimizer with a constant learning rate of 1×10−61superscript1061\times 10^{-6}1 × 10 start_POSTSUPERSCRIPT - 6 end_POSTSUPERSCRIPT, incorporating a linear warm-up over 20 rollout steps.
For rollout, the prompt batch size is 512 and we sample 16 responses for each prompt. For training, the mini-batch size is set to 512, i.e., 16 gradient updates for each rollout step. For Overlong Reward Shaping, we set the expected maximum length as 16,384 tokens and allocate additional 4,096 tokens as the soft punish cache. Therefore, the maximum number of tokens for generation is set to 20,480 tokens.
As for the Clip-Higher mechanism, we set the clipping parameter εlowsubscript𝜀low\varepsilon_{\text{low}}italic_ε start_POSTSUBSCRIPT low end_POSTSUBSCRIPT to 0.2 and εhighsubscript𝜀high\varepsilon_{\text{high}}italic_ε start_POSTSUBSCRIPT high end_POSTSUBSCRIPT to 0.28, which effectively balance the trade-off between exploration and exploitation.
For evaluation on AIME, we repeat the evaluation set for 32 times and report avg@32 for results stability. The inference hyperparameters of evaluation are set to temperature 1.0 and topp 0.7.

### 4.2 Main Results

Experiments on AIME 2024 demonstrate that DAPO has successfully trained the Qwen-32B Base model into a powerful reasoning model, achieving performance superior to DeepSeek’s experiments on Qwen2.5-32B using the R1 approach.
In Figure 1, we observe a substantial improvement of performance on AIME 2024, with accuracy increasing from near 00% to 50%. Notably, this improvement is achieved with only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B.

We analyze the contributions of each training technique in our methodology, as detailed in Table 1.
The observed improvements demonstrate the effectiveness of these techniques in RL training, each contributing several accuracy points in AIME 2024.
Notably, given the vanilla GRPO setting, only 30% accuracy can be reached by training from a Qwen2.5-32B base model.

For token-level loss, although it brings less performance improvement, we find it enhances training stability and makes the length increase more healthily.

When applying Dynamic Sampling, although more data needs to be sampled due to the filtering out of zero-gradient data, the overall training time is not significantly affected.
As shown in Figure 6, although the number of sampling instances increases, the model’s convergence time is even reduced, due to fewer training steps required.

Model
AIME24avg@32subscriptAIME24avg@32\textbf{AIME24}_{\text{avg@32}}AIME24 start_POSTSUBSCRIPT avg@32 end_POSTSUBSCRIPT

DeepSeek-R1-Zero-Qwen-32B
47

Naive GRPO
30

+ Overlong Filtering
36

+ Clip-Higher
38

+ Soft Overlong Punishment
41

+ Token-level Loss
42

+ Dynamic Sampling (DAPO)
50

### 4.3 Training Dynamics

Reinforcement learning on large language models is not only a cutting-edge research direction but also an intrinsically complex systems engineering challenge, characterized by the interdependence of its various subsystems. Modifications to any single subsystem can propagate through the system, leading to unforeseen consequences due to the intricate interplay among these components. Even seemingly minor changes in initial conditions, such as variations in data and hyperparameters, can amplify through iterative reinforcement learning processes, yielding substantial deviations in outcomes. This complexity often confronts researchers with a dilemma: even after meticulous analysis and well-founded expectations that a modification will enhance specific aspects of the training process, the actual results frequently diverge from the anticipated trajectory. Therefore, monitoring of key intermediate results during experimentation is essential for swiftly identifying the sources of discrepancies and, ultimately, for refining the system.

(a) Mean response length.

(b) Reward score.

(c) Generation entropy.

(d) Mean probability.

- •

The Length of Generated Responses is a metric closely related to training stability and performance, as shown in Figure 7(a). The increase in length provides the model with a larger space for exploration, allowing more complex reasoning behaviors to be sampled and gradually reinforced through training. However, it is important to note that length does not always maintain a continuous upward trend during training. In some considerable periods, it can exhibit a trend of stagnation or even decline, which has also been demonstrated in [2]. We typically use length in conjunction with validation accuracy as indicators to assess whether an experiment is deteriorating.

- •

The Dynamics of Reward during training has always been one of the crucial monitoring indicators in reinforcement learning, as shown in Figure 7(b). In the majority of our experiments, the trend of reward increase is relatively stable and does not fluctuate or decline significantly due to adjustments in experimental settings. This indicates that, given a reliable reward signal, language models can robustly fit the distribution of training set. However, we find that the final reward on the training set often exhibits little correlation with the accuracy on the validation set, which indicates overfitting to the training set.

- •

The Entropy of the Actor Model and Generation Probability are related to the model’s exploration capability and are key metrics that we closely monitor in our experiments. Intuitively, the model’s entropy needs to be maintained within an appropriate range. An excessively low entropy indicates that the probability distribution is overly sharp, leading to a loss of exploration capability. Conversely, an excessively high entropy is often associated with issues of over-exploration such as gibberish and repetitive generation. For the generation probability, the situation is exactly the opposite. As demonstrated in Section 3.1, by applying the Clip-Higher strategy, we effectively addressed the issue of entropy collapse. In subsequent experiments, we find that maintaining a slow upward trend in entropy is conducive to the improvement of model performance, shown in Figure 7(c) and Figure 7(d).

### 4.4 Case Study

Question:

Given that the base A⁢B⁢C𝐴𝐵𝐶ABCitalic_A italic_B italic_C of the tetrahedron S−A⁢B⁢C𝑆𝐴𝐵𝐶S-ABCitalic_S - italic_A italic_B italic_C is an equilateral triangle, the projection H𝐻Hitalic_H of point A𝐴Aitalic_A on the face S⁢B⁢C𝑆𝐵𝐶SBCitalic_S italic_B italic_C is the orthocenter of △⁢S⁢B⁢C△𝑆𝐵𝐶\triangle SBC△ italic_S italic_B italic_C, the dihedral angle H−A⁢B−C𝐻𝐴𝐵𝐶H-AB-Citalic_H - italic_A italic_B - italic_C is 30∘superscript3030^{\circ}30 start_POSTSUPERSCRIPT ∘ end_POSTSUPERSCRIPT, and S⁢A=2𝑆𝐴2SA=2italic_S italic_A = 2, find the volume of the tetrahedron. The answer is in the form km𝑘𝑚\frac{k}{m}divide start_ARG italic_k end_ARG start_ARG italic_m end_ARG, so give the value of k+m𝑘𝑚k+mitalic_k + italic_m.

Response:

Let us consider the given problem: We need to determine …

…

Next, let S𝑆Sitalic_S have coordinates (x0,y0,z0)subscript𝑥0subscript𝑦0subscript𝑧0(x_{0},y_{0},z_{0})( italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT , italic_z start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ). Since S⁢A=2𝑆𝐴2SA=2italic_S italic_A = 2, the distance formula gives us (x0)2+(y0)2+(z0)2=2superscriptsubscript𝑥02superscriptsubscript𝑦02superscriptsubscript𝑧022\sqrt{(x_{0})^{2}+(y_{0})^{2}+(z_{0})^{2}}=2square-root start_ARG ( italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + ( italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + ( italic_z start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT ) start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT end_ARG = 2, so we can write x02+y02+z02=4superscriptsubscript𝑥02superscriptsubscript𝑦02superscriptsubscript𝑧024x_{0}^{2}+y_{0}^{2}+z_{0}^{2}=4italic_x start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_y start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT + italic_z start_POSTSUBSCRIPT 0 end_POSTSUBSCRIPT start_POSTSUPERSCRIPT 2 end_POSTSUPERSCRIPT = 4. Now, remember that H𝐻Hitalic_H is the orthogonal projection of

…

However, wait a moment, let’s rethink about the dihedral angle involving planes in a more thoughtful geometric way. Consider the plane α1=A⁢B⁢Csubscript𝛼1𝐴𝐵𝐶\alpha_{1}=ABCitalic_α start_POSTSUBSCRIPT 1 end_POSTSUBSCRIPT = italic_A italic_B italic_C, which lies entirely on the xy coordinate plane (as z=0𝑧0z=0italic_z = 0). Let the plane α2=S⁢B⁢Csubscript𝛼2𝑆𝐵𝐶\alpha_{2}=SBCitalic_α start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT = italic_S italic_B italic_C. The point A𝐴Aitalic_A projected perpendicularly to plane α2subscript𝛼2\alpha_{2}italic_α start_POSTSUBSCRIPT 2 end_POSTSUBSCRIPT lands on H𝐻Hitalic_H. The line l=A⁢B𝑙𝐴𝐵l=ABitalic_l = italic_A italic_B …

…

During the RL training process, we observe an interesting phenomenon: the reasoning patterns of the actor model evolve dynamically over time. Specifically, the algorithm not only reinforces existing reasoning patterns that facilitate correct problem-solving but also gradually gives rise to entirely new modes of reasoning that were initially absent.
This finding reveals the adaptability and exploration capability of RL algorithms and offers new insights into the learning mechanisms of the model.

For example, in the early stages of model training, there was virtually no occurrence of checking and reflecting on previous reasoning steps.
However, as training progresses, the model exhibits distinct behaviors of reflection and backtracking, as shown in Table 2. This observation sheds light on further exploration into interpreting the emergence of reasoning abilities during RL, which we leave for future research.

## 5 Conclusion

In this paper, we release a fully open-sourced system for large-scale LLM RL, including algorithm, code infrastructure, and dataset. The system achieves state-of-the-art large-scale LLM RL performance (AIME 50 using Qwen-32B pretrained model). We propose the Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) algorithm, and introduce 4 key techniques to make RL powerfully effective and efficient in the long-CoT RL scenario.
Additionally, by open-sourcing the training code and dataset, we provide the broader research community and society with practical access to a scalable reinforcement learning solution, enabling all to benefit from these advancements.

## Contributions

Project Lead

Qiying Yu1,2,4

Algorithm

Qiying Yu1,2,4, Zheng Zhang1, Ruofei Zhu1, Yufeng Yuan1, Xiaochen Zuo1, Yu Yue1

Infrastructure∗

Weinan Dai1,2,4, Tiantian Fan1, Gaohong Liu1, Juncai Liu1, Lingjun Liu1, Xin Liu1, Haibin Lin1, Zhiqi Lin1, Bole Ma1, Guangming Sheng1,3, Yuxuan Tong1,2,4, Qiying Yu1,2,4, Chi Zhang1, Mofan Zhang1, Ru Zhang1, Wang Zhang1, Hang Zhu1, Jinhua Zhu1

∗Last-Name in Alphabetical Order

Dataset

Jiaze Chen1, Jiangjie Chen1,4, Chengyi Wang1, Hongli Yu1,2,4, Yuxuan Song1,2,4, Xiangpeng Wei1, Qiying Yu1,2,4

Supervision

Hao Zhou2,4, Jingjing Liu2,4, Wei-Ying Ma2,4, Ya-Qin Zhang2,4, Lin Yan1,4, Mu Qiao1,4, Yonghui Wu1, Mingxuan Wang1,4

Affiliation

1ByteDance Seed

2Institute for AI Industry Research (AIR), Tsinghua University

3The University of Hong Kong

4SIA-Lab of Tsinghua AIR and ByteDance Seed

## Acknowledgments

We thank Zhengyin Du, Shengding Hu, Kai Shen, Tianyang Zhan, Zhen Xiao, Renjie Zheng, Li Han, Kaihua Jiang as well as other colleagues at ByteDance for their support for the DAPO project.

## References

- [1]

OpenAI.

Learning to reason with llms, 2024.

- [2]

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948, 2025.

- [3]

OpenAI.

GPT4 technical report.

arXiv preprint arXiv:2303.08774, 2023.

- [4]

Anthropic.

Claude 3.5 sonnet, 2024.

- [5]

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

Language models are few-shot learners.

Advances in neural information processing systems, 33:1877–1901, 2020.

- [6]

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.

Palm: Scaling language modeling with pathways.

Journal of Machine Learning Research, 24(240):1–113, 2023.

- [7]

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al.

Deepseek-v3 technical report.

arXiv preprint arXiv:2412.19437, 2024.

- [8]

XAI.

Grok 3 beta — the age of reasoning agents, 2024.

- [9]

Google DeepMind.

Gemini 2.0 flash thinking, 2024.

- [10]

Qwen.

Qwq-32b: Embracing the power of reinforcement learning, 2024.

- [11]

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.

Kimi k1. 5: Scaling reinforcement learning with llms.

arXiv preprint arXiv:2501.12599, 2025.

- [12]

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al.

Qwen2. 5 technical report.

arXiv preprint arXiv:2412.15115, 2024.

- [13]

Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, et al.

An empirical study on eliciting and improving r1-like reasoning models.

arXiv preprint arXiv:2503.04548, 2025.

- [14]

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang.

Open-reasoner-zero: An open source approach to scaling reinforcement learning on the base model.

https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero, 2025.

- [15]

Jian Hu.

Reinforce++: A simple and efficient approach for aligning large language models.

arXiv preprint arXiv:2501.03262, 2025.

- [16]

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al.

Process reinforcement through implicit rewards.

arXiv preprint arXiv:2502.01456, 2025.

- [17]

Jung Hyun Lee, June Yong Yang, Byeongho Heo, Dongyoon Han, and Kang Min Yoo.

Token-supervised value models for enhancing mathematical reasoning capabilities of large language models.

arXiv preprint arXiv:2407.12863, 2024.

- [18]

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux.

Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment.

arXiv preprint arXiv:2410.01679, 2024.

- [19]

Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan.

What’s behind ppo’s collapse in long-cot? value optimization holds the secret.

arXiv preprint arXiv:2503.01491, 2025.

- [20]

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu.

Hybridflow: A flexible and efficient rlhf framework.

arXiv preprint arXiv:2409.19256, 2024.

- [21]

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347, 2017.

- [22]

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel.

High-dimensional continuous control using generalized advantage estimation, 2018.

- [23]

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe.

Training language models to follow instructions with human feedback.

In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022.

- [24]

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.

Concrete problems in ai safety, 2016.

- [25]

Tom Everitt, Victoria Krakovna, Laurent Orseau, Marcus Hutter, and Shane Legg.

Reinforcement learning with a corrupted reward channel, 2017.

- [26]

Victoria Krakovna, Jonathan Uesato, Vladimir Mikulik, Matthew Rahtz, Tom Everitt, Ramana Kumar, Zac Kenton, Jan Leike, and Shane Legg.

Specification gaming: the flip side of ai ingenuity, 2020.

- [27]

Tom Everitt, Marcus Hutter, Ramana Kumar, and Victoria Krakovna.

Reward tampering problems and solutions in reinforcement learning: A causal influence diagram perspective, 2021.

- [28]

Leo Gao, John Schulman, and Jacob Hilton.

Scaling laws for reward model overoptimization, 2022.

- [29]

Lilian Weng.

Reward hacking in reinforcement learning.

lilianweng.github.io, Nov 2024.

- [30]

Stanislas Polu and Ilya Sutskever.

Generative language modeling for automated theorem proving, 2020.

- [31]

Trieu H Trinh, Yuhuai Wu, Quoc V Le, He He, and Thang Luong.

Solving olympiad geometry without human demonstrations.

Nature, 625(7995):476–482, 2024.

- [32]

Trieu Trinh and Thang Luong.

Alphageometry: An olympiad-level ai system for geometry, 2024.

- [33]

AlphaProof and AlphaGeometry Teams.

Ai achieves silver-medal standard solving international mathematical olympiad problems, 2024.

- [34]

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi.

Coderl: Mastering code generation through pretrained models and deep reinforcement learning.

Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

- [35]

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao.

Reflexion: Language agents with verbal reinforcement learning, 2023.

- [36]

Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou.

Teaching large language models to self-debug, 2023.

- [37]

Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Quentin Carbonneaux, Taco Cohen, and Gabriel Synnaeve.

Rlef: Grounding code llms in execution feedback with reinforcement learning, 2025.

- [38]

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Y Wu, and Daya Guo.

Deepseekmath: Pushing the limits of mathematical reasoning in open language models.

arXiv preprint arXiv:2402.03300, 2024.

- [39]

Ilya Loshchilov and Frank Hutter.

Decoupled weight decay regularization.

In International Conference on Learning Representations, 2019.

\beginappendix

## 6 Dataset Transformation

Here is an example of data transformation, where we demonstrate the original problem, the transformed problem, and the model outputs during the transformation process (the original is in markdown text, which has been rendered for readability).

To ensure the model can complete this task with high accuracy, our goal is to have it perform comprehensive reasoning while avoiding hallucinations. We achieve this by providing a clear thinking framework that encourages thorough reasoning. Specifically, we employ Chain of Thought (CoT) reasoning along with four well-defined steps: extract the answer format, rewrite the problem statement, solve the modified problem and provide an integer as the final answer. For each step, we provide few-shot solutions or detailed guidelines to guide the model’s reasoning.

Through this approach, we have observed that in most cases, the LLM can generate reformulations with both format and quality that are satisfactory.

## 7 Supplementary Case

Question:

Among the 900 residents of Aimeville, there are 195 who own a diamond ring, 367 who own a set of golf clubs, and 562 who own a garden spade. In addition, each of the 900 residents owns a bag of candy hearts. There are 437 residents who own exactly two of these things, and 234 residents who own exactly three of these things. Find the number of residents of Aimeville who own all four of these things.

Response:

Step by Step Solution:

1. Let’s define the following variables:

Let A𝐴Aitalic_A be the set of residents who own a diamond ring.

Let B𝐵Bitalic_B be the set of residents who own a set of golf clubs.

Let C𝐶Citalic_C be the set of residents who own a garden spade.

Let D𝐷Ditalic_D be the set of residents who own a bag of candy hearts. Since all 900 residents own a bag of candy hearts, |D|=900𝐷900|D|=900| italic_D | = 900.

Let aisubscript𝑎𝑖a_{i}italic_a start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT denote the number of residents who own exactly i𝑖iitalic_i out of the 4 items.

Let’s use the principle of inclusion-exclusion for the union of 4 sets A,B,C,D𝐴𝐵𝐶𝐷A,B,C,Ditalic_A , italic_B , italic_C , italic_D.

…

a4=−219−4=2194=54.75subscript𝑎42194219454.75a_{4}=\frac{-219}{-4}=\frac{219}{4}=54.75italic_a start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT = divide start_ARG - 219 end_ARG start_ARG - 4 end_ARG = divide start_ARG 219 end_ARG start_ARG 4 end_ARG = 54.75.

Since a4subscript𝑎4a_{4}italic_a start_POSTSUBSCRIPT 4 end_POSTSUBSCRIPT, the number of people owning all four items, must be a whole number, our current approach needs to be reconsidered in terms of the combinatorial counting of people owning different combinations of the items.
Instead of directly using the inclusion-exclusion principle, we can use a different approach involving the number of people owning combinations of the items A𝐴Aitalic_A, B𝐵Bitalic_B, C𝐶Citalic_C, and D𝐷Ditalic_D.

Let xisubscript𝑥𝑖x_{i}italic_x start_POSTSUBSCRIPT italic_i end_POSTSUBSCRIPT be the number of people owning exactly i𝑖iitalic_i item among A𝐴Aitalic_A, B𝐵Bitalic_B, C𝐶Citalic_C (excluding the item D𝐷Ditalic_D which everyone owns).

Then the total number of people owning exactly k𝑘kitalic_k items among the A𝐴Aitalic_A, B𝐵Bitalic_B, C𝐶Citalic_C, and D𝐷Ditalic_D can be broken down:

…

Generated on Tue May 20 01:38:52 2025 by LaTeXML
