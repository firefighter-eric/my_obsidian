# ORPO: Monolithic Preference Optimization without Reference Model

- Source HTML: `raw/html/Hong et al. - 2024 - ORPO Monolithic Preference Optimization without Reference Model.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2403.07691
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# ORPO: Monolithic Preference Optimization without Reference Model

Jiwoo Hong

KAIST AI 
{jiwoo_hong, noah.lee, thorne}@kaist.ac.kr

Noah Lee

KAIST AI 
{jiwoo_hong, noah.lee, thorne}@kaist.ac.kr

James Thorne

KAIST AI 
{jiwoo_hong, noah.lee, thorne}@kaist.ac.kr

###### Abstract

While recent preference alignment algorithms for language models have demonstrated promising results, supervised fine-tuning (SFT) remains imperative for achieving successful convergence. In this paper, we study the crucial role of SFT within the context of preference alignment, emphasizing that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT. Building on this foundation, we introduce a straightforward and innovative reference model-free monolithic odds ratio preference optimization algorithm, ORPO, eliminating the necessity for an additional preference alignment phase. We demonstrate, both empirically and theoretically, that the odds ratio is a sensible choice for contrasting favored and disfavored styles during SFT across the diverse sizes from 125M to 7B. Specifically, fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO on the UltraFeedback alone surpasses the performance of state-of-the-art language models with more than 7B and 13B parameters: achieving up to 12.20% on AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0} (Figure 1), 66.19% on IFEval (instruction-level loose, Table 6), and 7.32 in MT-Bench (Figure 12). We release code111https://github.com/xfactlab/orpo and model checkpoints for Mistral-ORPO-α𝛼\alpha (7B)222https://huggingface.co/kaist-ai/mistral-orpo-alpha and Mistral-ORPO-β𝛽\beta (7B).333https://huggingface.co/kaist-ai/mistral-orpo-beta

## 1 Introduction

Pre-trained language models (PLMs) with vast training corpora such as web texts (Gokaslan and Cohen, 2019; Penedo et al., 2023) or textbooks (Li et al., 2023c) have shown remarkable abilities in diverse natural language processing (NLP) tasks (Brown et al., 2020; Zhang et al., 2022; Touvron et al., 2023; Jiang et al., 2023; Almazrouei et al., 2023).

However, the models must undergo further tuning to be usable in general-domain applications, typically through processes such as instruction tuning and preference alignment.

Instruction-tuning (Wei et al., 2022; Taori et al., 2023; Wang et al., 2023; Zhou et al., 2023a) trains models to follow task descriptions given in natural language, which enables models to generalize well to previously unseen tasks. However, despite the ability to follow instructions, models may generate harmful or unethical outputs (Carlini et al., 2021; Gehman et al., 2020; Pryzant et al., 2023). To further align these models with human values, additional training is required with pairwise preference data using techniques such as reinforcement learning with human feedback (Ziegler et al., 2020; Stiennon et al., 2022, RLHF) and direct preference optimization (Rafailov et al., 2023, DPO).

Preference alignment methods have demonstrated success in several downstream tasks beyond reducing harm. For example, improving factuality (Tian et al., 2023; Cheng et al., 2024; Chen and Li, 2024), code-based question answering (Gorbatovski and Kovalchuk, 2024), and machine translation (Ramos et al., 2023). The versatility of alignment algorithms over a wide range of downstream tasks highlights the necessity of understanding the alignment procedure and further improving the algorithms in terms of efficiency and performance. However, existing preference alignment methods normally consist of a multi-stage process, as shown in Figure 2, typically requiring a second reference model and a separate warm-up phase with supervised fine-tuning (SFT) (Ziegler et al., 2020; Rafailov et al., 2023; Wu et al., 2023).

In this paper, we study the role and impact of SFT in pairwise preference datasets for model alignment in Section 3 and propose a simple and novel monolithic alignment method, odds ratio preference optimization (ORPO), which efficiently penalizes the model from learning undesired generation styles during SFT in Section 4. In contrast to previous works, our approach requires neither an SFT warm-up stage nor a reference model, enabling resource-efficient development of preference-based aligned models.

We demonstrate the effectiveness of our method with the evaluation of model alignment tasks and popular leaderboards in Section 6.1 and 6.2 by fine-tuning Phi-2 (2.7B), Llama-2 (7B), and Mistral (7B) with ORPO. Then, we conduct controlled experiments comparing ORPO against established methods for model alignment, RLHF, and DPO for different datasets and model sizes in Section 6.3. Along with the post-hoc analysis of generation diversity in Section 6.4, we expound on the theoretical, empirical, and computational justification of utilizing the odds ratio in monolithic preference alignment in Section 7.3. We release the training code and the checkpoints for Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B). These models achieve 7.24 and 7.32 in MT-Bench, 11.33% and 12.20% on AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0}, and 61.63% and 66.19% in IFEval instruction-level loose accuracy, respectively.

## 2 Related Works

#### Alignment with Reinforcement Learning

Reinforcement learning with human feedback (RLHF) commonly applies the Bradley-Terry model (Bradley and Terry, 1952) to estimate the probability of a pairwise competition between two independently evaluated instances. An additional reward model is trained to score instances. Reinforcement learning algorithms such as proximal policy optimization (PPO) (Schulman et al., 2017) are employed to train the model to maximize the score of the reward model for the chosen response, resulting in language models that are trained with human preferences (Ziegler et al., 2020; Stiennon et al., 2022). Notably, Ouyang et al. (2022) demonstrated the scalability and versatility of RLHF for instruction-following language models. Extensions such as language model feedback (RLAIF) could be a viable alternative to human feedback (Bai et al., 2022b; Lee et al., 2023; Pang et al., 2023). However, RLHF faces challenges of extensive hyperparameter searching due to the instability of PPO (Rafailov et al., 2023; Wu et al., 2023) and the sensitivity of the reward models (Gao et al., 2022; Wang et al., 2024). Therefore, there is a crucial need for stable preference alignment algorithms.

#### Alignment without Reward Model

Several techniques for preference alignment mitigate the need for reinforcement learning (Rafailov et al., 2023; Song et al., 2023; Azar et al., 2023; Ethayarajh et al., 2024). Rafailov et al. (2023) introduce direct policy optimization (DPO), which combines the reward modeling stage into the preference learning stage. Azar et al. (2023) prevented potential overfitting problems in DPO through identity preference optimization (IPO). Ethayarajh et al. (2024) and Cai et al. (2023) proposed Kahneman-Tversky Optimisation (KTO) and Unified Language Model Alignment (ULMA) that does not require the pair-wise preference dataset, unlike RLHF and DPO. Song et al. (2023) further suggests incorporation of the softmax value of the reference response set in the negative log-likelihood loss to merge the supervised fine-tuning and preference alignment.

#### Alignment with Supervised Fine-tuning

Preference alignment methods in reinforcement learning (RL) often leverage supervised fine-tuning (SFT) to ensure the stable update of the active policy in relation to the old policy (Schulman et al., 2017). This is because the SFT model is the old policy in the context of RLHF (Ziegler et al., 2020). Furthermore, empirical findings indicate that, even in non-RL alignment methods, the SFT model is crucial for achieving convergence to desired results (Rafailov et al., 2023; Tunstall et al., 2023).

In contrast, there have been approaches to build human-aligned language models by conducting SFT only with filtered datasets (Zhou et al., 2023a; Li et al., 2023a; Haggerty and Chandra, 2024; Zhou et al., 2023b). Zhou et al. (2023a) demonstrated that SFT with a small amount of data with fine-grained filtering and curation could be sufficient for building helpful language model assistants. Furthermore, Li et al. (2023a) and Haggerty and Chandra (2024) proposed an iterative process of fine-tuning the supervised fine-tuned language models with their own generations after fine-grained selection of aligned generations and Zhou et al. (2023b) suggested that a curated subset of preference dataset is sufficient for alignment. While these works highlight the impact and significance of SFT in the context of alignment, the actual role of SFT and the theoretical background for incorporating preference alignment in SFT remains understudied.

## 3 The Role of Supervised Fine-tuning

We study the behavior of supervised fine-tuning (SFT) as an initial stage of preference alignment methods (Ziegler et al., 2020; Rafailov et al., 2023) through analysis of the loss function in SFT and empirical demonstration of the preference comprehension ability of the trained SFT model. SFT plays a significant role in tailoring the pre-trained language models to the desired domain (Zhou et al., 2023a; Dong et al., 2024) by increasing the log probabilities of pertinent tokens. Nevertheless, this inadvertently increases the likelihood of generating tokens in undesirable styles, as illustrated in Figure 3. Therefore, it is necessary to develop methods capable of preserving the domain adaptation role of SFT while concurrently discerning and mitigating unwanted generation styles.

#### Absence of Penalty in Cross-Entropy Loss

The goal of cross-entropy loss model fine-tuning is to penalize the model if the predicted logits for the reference answers are low, as shown in Equation 2.

ℒℒ\displaystyle\mathcal{L}
=−1m​∑k=1mlog⁡P​(𝐱(k),𝐲(k))absent1𝑚superscriptsubscript𝑘1𝑚𝑃superscript𝐱𝑘superscript𝐲𝑘\displaystyle=-\frac{1}{m}\sum_{k=1}^{m}\log P(\mathbf{x}^{(k)},\mathbf{y}^{(k)})

(1)

=−1m​∑k=1m∑i=1|V|yi(k)⋅log⁡(pi(k))absent1𝑚superscriptsubscript𝑘1𝑚superscriptsubscript𝑖1𝑉⋅superscriptsubscript𝑦𝑖𝑘superscriptsubscript𝑝𝑖𝑘\displaystyle=-\frac{1}{m}\sum_{k=1}^{m}\sum_{i=1}^{|V|}y_{i}^{(k)}\cdot\log(p_{i}^{(k)})

(2)

where yisubscript𝑦𝑖y_{i} is a boolean value that indicates if i𝑖ith token in the vocabulary set V𝑉V is a label token, pisubscript𝑝𝑖p_{i} refers to the probability of i𝑖ith token, and m𝑚m is the length of sequence. Using cross-entropy alone gives no direct penalty or compensation for the logits of non-answer tokens (Lin et al., 2017) as yisubscript𝑦𝑖y_{i} will be set to 0. While cross-entropy is generally effective for domain adaptation (Mao et al., 2023), there are no mechanisms to penalize rejected responses when compensating for the chosen responses. Therefore, the log probabilities of the tokens in the rejected responses increase along with the chosen responses, which is not desired from the viewpoint of preference alignment.

#### Generalization over Both Response Styles

We conduct a pilot study to empirically demonstrate the miscalibration of chosen and rejected responses with supervised fine-tuning alone. We fine-tune OPT-350M (Zhang et al., 2022) on the chosen responses only from the HH-RLHF dataset (Bai et al., 2022b). Throughout the training, we monitor the log probability of rejected responses for each batch and report this in Figure 3. Both the log probability of chosen and rejected responses exhibited a simultaneous increase. This can be interpreted from two different perspectives. First, the cross-entropy loss effectively guides the model toward the intended domain (e.g., dialogue). However, the absence of a penalty for unwanted generations results in rejected responses sometimes having even higher log probabilities than the chosen ones.

#### Penalizing Undesired Generations

Appending an unlikelihood penalty to the loss has demonstrated success in reducing unwanted degenerative traits in models (Welleck et al., 2019; Li et al., 2020). For example, to prevent repetitions, an unwanted token set of previous contexts, k∈𝒞r​e​c​e​n​t𝑘subscript𝒞𝑟𝑒𝑐𝑒𝑛𝑡k\in\mathcal{C}_{recent}, is disfavored by adding the following term to (1−pi(k))1superscriptsubscript𝑝𝑖𝑘(1-p_{i}^{(k)}) to the loss (such as Equation 2) which penalizes the model for assigning high probabilities to recent tokens. Motivated by SFT ascribing high probabilities to rejected tokens (Figure 3) and the effectiveness of appending penalizing unwanted traits, we design a monolithic preference alignment method that dynamically penalizes the disfavored response for each query without the need for crafting sets of rejected tokens.

## 4 Odds Ratio Preference Optimization

We introduce a novel preference alignment algorithm, Odds Ratio Preference Optimization (ORPO), which incorporates an odds ratio-based penalty to the conventional negative log-likelihood (NLL) loss for differentiating the generation styles between favored and disfavored responses.

### 4.1 Preliminaries

Given an input sequence x𝑥x, the average log-likelihood of generating the output sequence y𝑦y, of length m𝑚m tokens, is computed as Equation 3. The odds of generating the output sequence y𝑦y given an input sequence x𝑥x is defined in Equation 4:

log⁡Pθ​(y|x)=1m​∑t=1mlog⁡Pθ​(yt|x,y<t)subscript𝑃𝜃conditional𝑦𝑥1𝑚superscriptsubscript𝑡1𝑚subscript𝑃𝜃conditionalsubscript𝑦𝑡𝑥subscript𝑦absent𝑡\log P_{\theta}(y|x)=\frac{1}{m}\sum_{t=1}^{m}\log P_{\theta}(y_{t}|x,y_{<t})

(3)

oddsθ​(y|x)=Pθ​(y|x)1−Pθ​(y|x)subscriptodds𝜃conditional𝑦𝑥subscript𝑃𝜃conditional𝑦𝑥1subscript𝑃𝜃conditional𝑦𝑥\textbf{odds}_{\theta}(y|x)=\frac{P_{\theta}(y|x)}{1-P_{\theta}(y|x)}

(4)

Intuitively, oddsθ​(y|x)=ksubscriptodds𝜃conditional𝑦𝑥𝑘\textbf{odds}_{\theta}(y|x)=k implies that it is k𝑘k times more likely for the model θ𝜃\theta to generate the output sequence y𝑦y than not generating it. Thus, the odds ratio of the chosen response ywsubscript𝑦𝑤y_{w} over the rejected response ylsubscript𝑦𝑙y_{l}, ORθ​(yw,yl)subscriptOR𝜃subscript𝑦𝑤subscript𝑦𝑙\textbf{OR}_{\theta}(y_{w},y_{l}), indicates how much more likely it is for the model θ𝜃\theta to generate ywsubscript𝑦𝑤y_{w} than ylsubscript𝑦𝑙y_{l} given input x𝑥x, defined in Equation 5.

ORθ​(yw,yl)=oddsθ​(yw|x)oddsθ​(yl|x)subscriptOR𝜃subscript𝑦𝑤subscript𝑦𝑙subscriptodds𝜃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃conditionalsubscript𝑦𝑙𝑥\textbf{OR}_{\theta}(y_{w},y_{l})=\frac{\textbf{odds}_{\theta}(y_{w}|x)}{\textbf{odds}_{\theta}(y_{l}|x)}

(5)

### 4.2 Objective Function of ORPO

The objective function of ORPO in Equation 6 consists of two components: 1) supervised fine-tuning (SFT) loss (ℒS​F​Tsubscriptℒ𝑆𝐹𝑇\mathcal{L}_{SFT}); 2) relative ratio loss (ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR}).

ℒO​R​P​O=𝔼(x,yw,yl)​[ℒS​F​T+λ⋅ℒO​R]subscriptℒ𝑂𝑅𝑃𝑂subscript𝔼𝑥subscript𝑦𝑤subscript𝑦𝑙delimited-[]subscriptℒ𝑆𝐹𝑇⋅𝜆subscriptℒ𝑂𝑅\mathcal{L}_{ORPO}=\mathbb{E}_{(x,y_{w},y_{l})}\left[\mathcal{L}_{SFT}+\lambda\cdot\mathcal{L}_{OR}\right]

(6)

ℒS​F​Tsubscriptℒ𝑆𝐹𝑇\mathcal{L}_{SFT} follows the conventional causal language modeling negative log-likelihood (NLL) loss function to maximize the likelihood of generating the reference tokens as previously discussed in Section 3.
ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR} in Equation 7 maximizes the odds ratio between the likelihood of generating the disfavored response ywsubscript𝑦𝑤y_{w} and the disfavored response ylsubscript𝑦𝑙y_{l}. We wrap the log odds ratio with the log sigmoid function so that ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR} could be minimized by increasing the log odds ratio between ywsubscript𝑦𝑤y_{w} and ylsubscript𝑦𝑙y_{l}.

ℒO​R=−log⁡σ​(log⁡oddsθ​(yw|x)oddsθ​(yl|x))subscriptℒ𝑂𝑅𝜎subscriptodds𝜃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃conditionalsubscript𝑦𝑙𝑥\mathcal{L}_{OR}=-\log\sigma\left(\log\frac{\textbf{odds}_{\theta}(y_{w}|x)}{\textbf{odds}_{\theta}(y_{l}|x)}\right)

(7)

Together, ℒS​F​Tsubscriptℒ𝑆𝐹𝑇\mathcal{L}_{SFT} and ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR} weighted with λ𝜆\lambda tailor the pre-trained language model to adapt to the specific subset of the desired domain and disfavor generations in the rejected response sets.

### 4.3 Gradient of ORPO

The gradient of ℒR​a​t​i​osubscriptℒ𝑅𝑎𝑡𝑖𝑜\mathcal{L}_{Ratio} further justifies using the odds ratio loss. It comprises two terms: one that penalizes the wrong predictions and one that contrasts between chosen and rejected responses, denoted in Equation 8444The full derivation for ∇θℒO​Rsubscript∇𝜃subscriptℒ𝑂𝑅\nabla_{\theta}\mathcal{L}_{OR} is in Appendix A. for d=(x,yl,yw)∼D𝑑𝑥subscript𝑦𝑙subscript𝑦𝑤similar-to𝐷d=(x,y_{l},y_{w})\sim D.

∇θℒO​R=δ​(d)⋅h​(d)subscript∇𝜃subscriptℒ𝑂𝑅⋅𝛿𝑑ℎ𝑑\nabla_{\theta}\mathcal{L}_{OR}=\delta(d)\cdot h(d)

(8)

δ​(d)𝛿𝑑\displaystyle\delta(d)
=[1+oddsθ​P​(yw|x)oddsθ​P​(yl|x)]−1absentsuperscriptdelimited-[]1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥1\displaystyle=\left[1+\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}\right]^{-1}

(9)

h​(d)ℎ𝑑\displaystyle h(d)
=∇θlog⁡Pθ​(yw|x)1−Pθ​(yw|x)−∇θlog⁡Pθ​(yl|x)1−Pθ​(yl|x)absentsubscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥1subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥1subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥\displaystyle=\frac{\nabla_{\theta}\log P_{\theta}(y_{w}|x)}{1-P_{\theta}(y_{w}|x)}-\frac{\nabla_{\theta}\log P_{\theta}(y_{l}|x)}{1-P_{\theta}(y_{l}|x)}

(10)

When the odds of the favored responses are relatively higher than the disfavored responses, δ​(d)𝛿𝑑\delta(d) in Equation 9 will converge to 0. This indicates that the δ​(d)𝛿𝑑\delta(d) will play the role of a penalty term, accelerating the parameter updates if the model is more likely to generate the rejected responses.

Meanwhile, h​(d)ℎ𝑑h(d) in Equation 10 implies a weighted contrast of the two gradients from the chosen and rejected responses. Specifically, 1−P​(y|x)1𝑃conditional𝑦𝑥1-P(y|x) in the denominators amplifies the gradients when the corresponding side of the likelihood P​(y|x)𝑃conditional𝑦𝑥P(y|x) is low. For the chosen responses, this accelerates the model’s adaptation toward the distribution of chosen responses as the likelihood increases.

## 5 Experimental Settings

### 5.1 Training Configurations

#### Models

We train a series of OPT models (Zhang et al., 2022) scaling from 125M to 1.3B parameters comparing supervised fine-tuning (SFT), proximal policy optimization (PPO), direct policy optimization (DPO), and compare these to our ORPO. PPO and DPO models were fine-tuned with TRL library (von Werra et al., 2020) on top of SFT models trained for a single epoch on the chosen responses following Rafailov et al. (2023) and Tunstall et al. (2023). We notate this by prepending "+" to each algorithm (e.g., +DPO). Additionally, we train Phi-2 (2.7B) (Javaheripi and Bubeck, 2023), a pre-trained language model with promising downstream performance (Beeching et al., 2023), as well as Llama-2 (7B) (Touvron et al., 2023) and Mistral (7B) (Jiang et al., 2023).
Further training details for each method are in Appendix C.

#### Datasets

We test each training configuration and model on two datasets: 1) Anthropic’s HH-RLHF (Bai et al., 2022a), 2) Binarized UltraFeedback (Tunstall et al., 2023). We filtered out instances where yw=ylsubscript𝑦𝑤subscript𝑦𝑙y_{w}=y_{l} or where yw=∅subscript𝑦𝑤y_{w}=\emptyset or where yl=∅subscript𝑦𝑙y_{l}=\emptyset.

#### Reward Models

We train OPT-350M and OPT-1.3B on each dataset for a single epoch for reward modeling with the objective function in Equation 11 (Ziegler et al., 2020). The OPT-350M reward model was used for PPO, and OPT-1.3B reward model was used to assess the generations of fine-tuned models. We refer to these reward models as RM-350M and RM-1.3B in Section 6.

−𝔼(x,yl,yw)​[log⁡σ​(r​(x,yw)−r​(x,yl))]subscript𝔼𝑥subscript𝑦𝑙subscript𝑦𝑤delimited-[]𝜎𝑟𝑥subscript𝑦𝑤𝑟𝑥subscript𝑦𝑙-\mathbb{E}_{(x,y_{l},y_{w})}\left[\log\sigma\left(r(x,y_{w})-r(x,y_{l})\right)\right]

(11)

### 5.2 Leaderboard Evaluation

In Section 6.1, we evaluate the models using the AlpacaEval1.0subscriptAlpacaEval1.0\text{AlpacaEval}_{1.0} and AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0} (Li et al., 2023b) benchmarks, comparing ORPO to other instruction-tuned models reported in the official leaderboard,555https://tatsu-lab.github.io/alpaca_eval/ including Llama-2 Chat (7B) and (13B) (Touvron et al., 2023), and Zephyr α𝛼\alpha and β𝛽\beta (Almazrouei et al., 2023). Similarly, in Section 6.2, we evaluate the models with MT-Bench (Zheng et al., 2023) and report the results and the scores of the same models reported in the official leaderboard.666https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
Using GPT-4 (Achiam et al., 2023) as an evaluator in AlpacaEval1.0subscriptAlpacaEval1.0\text{AlpacaEval}_{1.0}, we assess if the trained model can be preferred over the responses generated from text-davinci-003. For AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0}, we used GPT-4-turbo777https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo as an evaluator following the default setting. We assess if the generated responses are favored over those generated from GPT-4. Finally, using GPT-4 as an evaluator in MT-Bench, we check if the models can follow the instructions with hard answers in a multi-turn conversation.

## 6 Results and Analysis

First, we assess the general instruction-following abilities of the models by comparing the preference alignment algorithms in Sections 6.1 and 6.2. Second, we measure the win rate of OPT models trained with ORPO against other alignment methods training OPT 1.3B as a reward model in Section 6.3. Then, we measure the lexical diversity of the models trained with ORPO and DPO in Section 6.4.

### 6.1 Single-turn Instruction Following

Model Name
Size
AlpacaEval1.0subscriptAlpacaEval1.0\textbf{AlpacaEval}_{\textbf{1.0}}
AlpacaEval2.0subscriptAlpacaEval2.0\textbf{AlpacaEval}_{\textbf{2.0}}

Phi-2 + SFT
2.7B
48.37% (1.77)
0.11% (0.06)

Phi-2 + SFT + DPO
2.7B
50.63% (1.77)
0.78% (0.22)

Phi-2 + ORPO (Ours)

2.7B
71.80% (1.59)
6.35% (0.74)

Llama-2 Chat *
7B
71.34% (1.59)
4.96% (0.67)

Llama-2 Chat *
13B
81.09% (1.38)
7.70% (0.83)

Llama-2 + ORPO (Ours)

7B
81.26% (1.37)
9.44% (0.85)

Zephyr (α𝛼\alpha) *

7B
85.76% (1.23)
8.35% (0.87)

Zephyr (β𝛽\beta) *

7B
90.60% (1.03)
10.99% (0.96)

Mistral-ORPO-α𝛼\alpha (Ours)

7B
87.92% (1.14)
11.33% (0.97)

Mistral-ORPO-β𝛽\beta (Ours)

7B
91.41% (1.15)
12.20% (0.98)

#### Phi-2 (2.7B)

ORPO improved pre-trained Phi-2 to exceed the performance of the Llama-2 Chat instruction-following language model by only using UltraFeedback as the instruction-tuning dataset, as shown in Table 1. λ𝜆\lambda of 0.25 was applied for Phi-2, resulting in 71.80% and 6.35% in AlpacaEval.

#### Llama-2 (7B)

Notably, UltraFeedback and ORPO with λ𝜆\lambda of 0.2 on Llama-2 (7B) resulted in higher AlpacaEval scores than the chat versions of both 7B and 13B scale trained with RLHF, eventually showing 81.26% and 9.44% in both AlpacaEvals.

In contrast, in our controlled experimental setting of conducting one epoch of SFT and three epochs of DPO following Tunstall et al. (2023) and Rafailov et al. (2023), Llama-2 + SFT and Llama-2 + SFT + DPO yielded models with outputs that could not be evaluated. This supports the efficacy of ORPO, in which the model can rapidly learn the desired domain and the preference with limited data. This aligns with the h​(d)ℎ𝑑h(d) examination in the gradient of our method studied in Section 4.3.

#### Mistral-ORPO-α𝛼\alpha (7B)

Furthermore, fine-tuning Mistral (7B) with single-turn conversation dataset, UltraFeedback, and ORPO with λ𝜆\lambda of 0.1 outperforms Zephyr series, which are the Mistral (7B) models fine-tuned with SFT on 20K UltraChat (Ding et al., 2023) and DPO on the full UltraFeedback. As shown in Table 1, Mistral-ORPO-α𝛼\alpha (7B) achieves 87.92% and 11.33%, which exceeds Zephyr α𝛼\alpha by 1.98% and Zephyr β𝛽\beta by 0.34% in AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0}. The sample responses and corresponding references from GPT-4 can be found in Appendix I.

#### Mistral-ORPO-β𝛽\beta (7B)

Using the same configuration of Mistral-ORPO-α𝛼\alpha (7B), we additionally compare fine-tuning Mistral on the cleaned version of the UltraFeedback888https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned to demonstrate the effect of the data quality (Bartolome et al., 2023). While the actual sizes of datasets are similar, ORPO gains further advantages from the dataset quality by scoring over 91% and 12% on AlpacaEval, as shown in Table 1. Further instruction-following evaluation on two Mistral-based models with IFEval (Zhou et al., 2023c) is reported in the Appendix D.

### 6.2 Multi-turn Instruction Following

With our best model, Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B), we also assess the multi-turn instruction-following skills with deterministic answers (e.g., math) through MT-Bench.

As shown in Figure 4, ORPO-Mistral (7B) series achieve comparable results to either larger or the proprietary models, including Llama-2-Chat (70B) and Claude. Eventually, Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B) scored 7.23 and 7.32 in MT-Bench without being exposed to the multi-turn conversation dataset during training.

### 6.3 Reward Model Win Rate

We assess the win rate of ORPO over other preference alignment methods, including supervised fine-tuning (SFT), PPO, and DPO, using RM-1.3B to understand the effectiveness and scalability of ORPO in Tables 2 and 3. Additionally, we visually verify that ORPO can effectively enhance the expected reward compared to SFT in Figure 5.

#### HH-RLHF

In Table 2, ORPO outperforms SFT and PPO across all model scales. The highest win rate against SFT and PPO across the size of the model was 78.0% and 79.4%, respectively. Meanwhile, the win rate over DPO was correlated to the model’s size, with the largest model having the highest win rate: 70.9%.

ORPO vs
SFT
+DPO
+PPO

OPT-125M
84.0 (0.62)
41.7 (0.77)
66.1 (0.26)

OPT-350M
82.7 (0.56)
49.4 (0.54)
79.4 (0.29)

OPT-1.3B
78.0 (0.16)
70.9 (0.52)
65.9 (0.33)

#### UltraFeedback

The win rate in UltraFeedback followed similar trends to what was reported in HH-RLHF, as shown in Table 3. ORPO was preferred over SFT and PPO for maximum 80.5% and 85.8%, respectively. While consistently preferring ORPO over SFT and PPO, the win rate over DPO gradually increases as the size of the model increases. The scale-wise trend exceeding DPO will be further shown through 2.7B models in Section 6.1.

ORPO vs
SFT
+DPO
+PPO

OPT-125M
73.2 (0.12)
48.8 (0.29)
71.4 (0.28)

OPT-350M
80.5 (0.54)
50.5 (0.17)
85.8 (0.62)

OPT-1.3B
69.4 (0.57)
57.8 (0.73)
65.7 (1.07)

#### Overall Reward Distribution

In addition to the win rate, we compare the reward distribution of the responses generated with respect to the test set of the UltraFeedback dataset in Figure 5 and HH-RLHF dataset in Appendix F. Regarding the SFT reward distribution as a default, PPO, DPO, and ORPO shift it in both datasets. However, the magnitude of reward shifts for each algorithm differs.

In Figure 5, RLHF (i.e., SFT + PPO) has some abnormal properties of the distribution with a low expected reward. We attribute this to empirical evidence of the instability and reward mismatch problem of RLHF (Rafailov et al., 2023; Gao et al., 2022; Shen et al., 2023) as the RLHF models were trained with RM-350M and assessed with RM-1.3B. Meanwhile, it is notable that the ORPO distribution (red) is mainly located on the very right side of each subplot, indicating higher expected rewards. Recalling the intent of preference alignment methods, the distributions in Figure 5 indicate that ORPO tends to fulfill the aim of preference alignment for all model sizes.

### 6.4 Lexical Diversity

The lexical diversity of the preference-aligned language models was studied in previous works (Kirk et al., 2024). We expand the concept of per-input and across-input diversity introduced in Kirk et al. (2024) by using Gemini-Pro (Gemini Team et al., 2023) as an embedding model, which is suitable for assessing the diversity of instruction-following language models by encoding a maximum of 2048 tokens. The diversity metric with the given set of sampled responses is defined as Equation 13.

𝒪θi:={yj∼θ(y|xi)|j=1,2,…,K}\mathcal{O}_{\theta}^{i}:=\{y_{j}\sim\theta(y|x_{i})|j=1,2,...,K\}

(12)

D​(𝒪θi)=12⋅∑i=1N−1∑j=i+1Ncos⁡(hi,hj)N⋅(N−1)𝐷subscriptsuperscript𝒪𝑖𝜃⋅12superscriptsubscript𝑖1𝑁1superscriptsubscript𝑗𝑖1𝑁subscriptℎ𝑖subscriptℎ𝑗⋅𝑁𝑁1D(\mathcal{O}^{i}_{\theta})=\frac{1}{2}\cdot\frac{\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}\cos(h_{i},h_{j})}{N\cdot(N-1)}

(13)

where cos⁡(hi,hj)subscriptℎ𝑖subscriptℎ𝑗\cos(h_{i},h_{j}) refers to the cosine similarity between the embedding hisubscriptℎ𝑖h_{i} and hjsubscriptℎ𝑗h_{j}. 5 different responses are sampled with a temperature of 1.0 to 160 queries in AlpacaEval (i.e., K=5,N=160formulae-sequence𝐾5𝑁160K=5,N=160) using Phi-2 and Llama-2 trained with ORPO and DPO. We report the results in Table 4.

#### Per Input Diversity (PID)

We average the input-wise average cosine similarity between the generated samples with Equation 14 to assess the per-input diversity. In Table 4, ORPO models have the highest average cosine similarity in the first column for both models, which implies the lowest diversity per input. This indicates that ORPO generally assigns high probabilities to the desired tokens, while DPO has a relatively smoother logit distribution.

PIDD​(θ)=1N​∑i=1ND​(𝒪θi)subscriptPID𝐷𝜃1𝑁superscriptsubscript𝑖1𝑁𝐷subscriptsuperscript𝒪𝑖𝜃\text{PID}_{D}(\theta)=\frac{1}{N}\sum_{i=1}^{N}D(\mathcal{O}^{i}_{\theta})

(14)

#### Across Input Diversity (AID)

Using 8 samples generated per input, we sample the first item for each input and examine their inter cosine similarity with Equation 15 for across-input diversity. Unlike per-input diversity, it is noteworthy that Phi-2 (ORPO) has lower average cosine similarity in the second row of Table 4. We can infer that ORPO triggers the model to generate more instruction-specific responses than DPO.

AIDD(θ)=D(⋃i=1N𝒪i,θ,j=1)\text{AID}_{D}(\theta)=D\left(\bigcup\limits_{i=1}^{N}\mathcal{O}^{i},_{\theta,j=1}\right)

(15)

Per Input↓↓\downarrow
Across Input↓↓\downarrow

Phi-2 + SFT + DPO
0.8012
0.6019

Phi-2 + ORPO
0.8909
0.5173

Llama-2 + SFT + DPO
0.8889
0.5658

Llama-2 + ORPO
0.9008
0.5091

## 7 Discussion

In this section, we expound on the theoretical and computational details of ORPO. The theoretical analysis of ORPO is studied in Section 7.1, which will be supported with the empirical analysis in Section 7.2. Then, we compare the computational load of DPO and ORPO in Section 7.3.

### 7.1 Comparison to Probability Ratio

The rationale for selecting the odds ratio instead of the probability ratio lies in its stability. The probability ratio for generating the favored response ywsubscript𝑦𝑤y_{w} over the disfavored response ylsubscript𝑦𝑙y_{l} given an input sequence x𝑥x can be defined as Equation 16.

PRθ​(yw,yl)=Pθ​(yw|x)Pθ​(yl|x)subscriptPR𝜃subscript𝑦𝑤subscript𝑦𝑙subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥\textbf{PR}_{\theta}(y_{w},y_{l})=\frac{P_{\theta}(y_{w}|x)}{P_{\theta}(y_{l}|x)}

(16)

While this formulation has been used in previous preference alignment methods that precede SFT (Rafailov et al., 2023; Azar et al., 2023), the odds ratio is a better choice in the setting where the preference alignment is incorporated in SFT as the odds ratio is more sensitive to the model’s preference understanding. In other words, the probability ratio leads to more extreme discrimination of the disfavored responses than the odds ratio.

We visualize this through the sample distributions of the log probability ratio log⁡PR​(X2|X1)PRconditionalsubscript𝑋2subscript𝑋1\log\textbf{PR}(X_{2}|X_{1}) and log odds ratio log⁡OR​(X2|X1)ORconditionalsubscript𝑋2subscript𝑋1\log\textbf{OR}(X_{2}|X_{1}). We sample 50,000 samples each with Equation 17 and plot the log probability ratio and log odds ratio in Figure 6. We multiply β𝛽\beta for the probability ratio as it is practiced in the probability ratio-based methods and report the cases where β=0.2𝛽0.2\beta=0.2 and β=1.0𝛽1.0\beta=1.0.

X1,X2∼Unif​(0,1)similar-tosubscript𝑋1subscript𝑋2Unif01\displaystyle X_{1},X_{2}\sim\text{Unif}(0,1)

(17)

Y∼β​(log⁡X1−log⁡X2)similar-to𝑌𝛽subscript𝑋1subscript𝑋2\displaystyle Y\sim\beta\left(\log X_{1}-\log X_{2}\right)

(18)

Y∼log⁡X11−X1−log⁡X21−X2similar-to𝑌subscript𝑋11subscript𝑋1subscript𝑋21subscript𝑋2\displaystyle Y\sim\log\frac{X_{1}}{1-X_{1}}-\log\frac{X_{2}}{1-X_{2}}

(19)

Recalling that the log sigmoid function is applied to the log probability ratio and log odds ratio, each ratio’s scale determines the expected margin between the likelihood of the favored and disfavored styles when the loss is minimized. In that sense, the contrast should be relatively extreme to minimize the log sigmoid loss when PR​(X2|X1)PRconditionalsubscript𝑋2subscript𝑋1\textbf{PR}(X_{2}|X_{1}) is inputted instead of OR​(X2|X1)ORconditionalsubscript𝑋2subscript𝑋1\textbf{OR}(X_{2}|X_{1}) to the log sigmoid function, regarding the sharp distribution of log⁡PR​(X2|X1)PRconditionalsubscript𝑋2subscript𝑋1\log\textbf{PR}(X_{2}|X_{1}) in Figure 6. This results in overly suppressing the logits for the tokens in the disfavored responses in the setting where SFT and preference alignment are incorporated, as the model is not adapted to the domain. We empirically support this analysis through the ablation study in Appendix B. Therefore, the odds ratio is a better choice when the preference alignment is done with SFT due to the mild discrimination of disfavored responses and the prioritizing of the favored responses to be generated.

Throughout fine-tuning, minimizing the log sigmoid loss leads to either PR​(X2|X1)PRconditionalsubscript𝑋2subscript𝑋1\textbf{PR}(X_{2}|X_{1}) or OR​(X2|X1)ORconditionalsubscript𝑋2subscript𝑋1\textbf{OR}(X_{2}|X_{1}) to be larger. This is equivalent to the rejected responses’ token-wise likelihood, which will generally get smaller. In this context, it is essential to avoid an overly extreme contrast. This precaution is especially important given the sharp distribution of log⁡PR​(X2|X1)PRconditionalsubscript𝑋2subscript𝑋1\log\textbf{PR}(X_{2}|X_{1}) depicted in Figure 6. The excessive margin could lead to the unwarranted suppression of logits for tokens in disfavored responses within the incorporated setting, potentially resulting in issues of degeneration.

### 7.2 Minimizing ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR}

We demonstrate that models trained with ORPO learned to reflect the preference throughout the training process. We monitored the log probabilities of the chosen and rejected responses and the log odds ratio with λ=1.0𝜆1.0\lambda=1.0. With the same dataset and model as Figure 3, Figure 7 shows that the log probability of rejected responses is diminishing while that of chosen responses is on par with Figure 3 as the log odds ratio increases. This indicates that ORPO is successfully preserving the domain adaptation role of SFT while the penalty term LO​Rsubscript𝐿𝑂𝑅L_{OR} induces the model to lower the likelihood of unwanted generations. We discuss the effect of λ𝜆\lambda in Equation 6 in Appendix E, studying the proclivity of the log probability margin between the favored and disfavored responses with respect to λ𝜆\lambda.

### 7.3 Computational Efficiency

As depicted in Figure 2, ORPO does not require a reference model, unlike RLHF and DPO. In that sense, ORPO is computationally more efficient than RLHF and DPO in two perspectives: 1) memory allocation and 2) fewer FLOPs per batch.

The reference model (πS​F​Tsubscript𝜋𝑆𝐹𝑇\pi_{SFT}) in the context of RLHF and DPO denotes the model trained with supervised fine-tuning (SFT), which will be the baseline model for updating the parameters with RLHF or DPO (Ziegler et al., 2020; Rafailov et al., 2023). Thus, two πS​F​Tsubscript𝜋𝑆𝐹𝑇\pi_{SFT}s, a frozen reference model and the model undergoing tuning, are required during training. Furthermore, in theory, two forward passes should be calculated for each model to acquire the logits for the chosen and rejected responses. In other words, four forward passes happen in total for a single batch.
On the other hand, a reference model is not required in ORPO as πS​F​Tsubscript𝜋𝑆𝐹𝑇\pi_{SFT} is directly updated. This leads to half the number of forward passes required for each batch during training.

## 8 Conclusion

In this paper, we introduced a reference-free monolithic preference alignment method, odds ratio preference optimization (ORPO), by revisiting and understanding the value of the supervised fine-tuning (SFT) phase in the context of preference alignment. ORPO was consistently preferred by the fine-tuned reward model against SFT and RLHF across the scale, and the win rate against DPO increased as the size of the model increased. Furthermore, we validate the scalability of ORPO with 2.7B and 7B pre-trained language models by exceeding the larger state-of-the-art instruction-following language models in AlpacaEval. Specifically, Mistral-ORPO-α𝛼\alpha and Mistral-ORPO-β𝛽\beta achieved 11.33% and 12.20% in AlpacaEval2.0subscriptAlpacaEval2.0\text{AlpacaEval}_{2.0}, 7.23 and 7.32 in MT-Bench, thereby underscoring the efficiency and effectiveness of ORPO. We release fine-tuning code and model checkpoints for Mistral-ORPO-α𝛼\alpha and Mistral-ORPO-β𝛽\beta to aid reproducibility.

## Limitations

While conducting a comprehensive analysis of the diverse preference alignment methods, including DPO and RLHF, we did not incorporate a more comprehensive range of preference alignment algorithms. We leave the broader range of comparison against other methods as future work, along with scaling our method to over 7B models. In addition, we will expand the fine-tuning datasets into diverse domains and qualities, thereby verifying the generalizability of our method in various NLP downstream tasks. Finally, we would like to study the internal impact of our method on the pre-trained language model, expanding the understanding of preference alignment procedure to not only the supervised fine-tuning stage but also consecutive preference alignment algorithms.

## References

- Achiam et al. (2023)

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023.

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774.

- Almazrouei et al. (2023)

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. 2023.

The falcon series of open language models.

- Azar et al. (2023)

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023.

A general theoretical paradigm to understand learning from human preferences.

- Bai et al. (2022a)

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. 2022a.

Training a helpful and harmless assistant with reinforcement learning from human feedback.

- Bai et al. (2022b)

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022b.

Constitutional ai: Harmlessness from ai feedback.

- Bartolome et al. (2023)

Alvaro Bartolome, Gabriel Martin, and Daniel Vila. 2023.

Notus.

https://github.com/argilla-io/notus.

- Beeching et al. (2023)

Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. 2023.

Open llm leaderboard.

https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard.

- Bradley and Terry (1952)

Ralph Allan Bradley and Milton E. Terry. 1952.

Rank analysis of incomplete block designs: I. the method of paired comparisons.

Biometrika, 39(3/4):324--345.

- Brown et al. (2020)

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.

Language models are few-shot learners.

In Advances in Neural Information Processing Systems, volume 33, pages 1877--1901. Curran Associates, Inc.

- Cai et al. (2023)

Tianchi Cai, Xierui Song, Jiyan Jiang, Fei Teng, Jinjie Gu, and Guannan Zhang. 2023.

Ulma: Unified language model alignment with demonstration and point-wise human preference.

ArXiv, abs/2312.02554.

- Carlini et al. (2021)

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel. 2021.

Extracting training data from large language models.

- Chen and Li (2024)

Weixin Chen and Bo Li. 2024.

Grath: Gradual self-truthifying for large language models.

- Cheng et al. (2024)

Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li, Kai Chen, and Xipeng Qiu. 2024.

Can ai assistants know what they don’t know?

- Dao (2023)

Tri Dao. 2023.

Flashattention-2: Faster attention with better parallelism and work partitioning.

- Dettmers et al. (2023)

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023.

Qlora: Efficient finetuning of quantized llms.

- Ding et al. (2023)

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023.

Enhancing chat language models by scaling high-quality instructional conversations.

- Dong et al. (2024)

Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. 2024.

How abilities in large language models are affected by supervised fine-tuning data composition.

- Ethayarajh et al. (2024)

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024.

Kto: Model alignment as prospect theoretic optimization.

arXiv preprint arXiv:2402.01306.

- Gao et al. (2022)

Leo Gao, John Schulman, and Jacob Hilton. 2022.

Scaling laws for reward model overoptimization.

- Gao et al. (2023)

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023.

A framework for few-shot language model evaluation.

- Gehman et al. (2020)

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith. 2020.

RealToxicityPrompts: Evaluating neural toxic degeneration in language models.

In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356--3369, Online. Association for Computational Linguistics.

- Gemini Team et al. (2023)

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023.

Gemini: a family of highly capable multimodal models.

arXiv preprint arXiv:2312.11805.

- Gokaslan and Cohen (2019)

Aaron Gokaslan and Vanya Cohen. 2019.

Openwebtext corpus.

http://Skylion007.github.io/OpenWebTextCorpus.

- Gorbatovski and Kovalchuk (2024)

Alexey Gorbatovski and Sergey Kovalchuk. 2024.

Reinforcement learning for question answering in programming domain using public community scoring as a human feedback.

- Haggerty and Chandra (2024)

Hamish Haggerty and Rohitash Chandra. 2024.

Self-supervised learning for skin cancer diagnosis with limited training data.

- Javaheripi and Bubeck (2023)

Mojan Javaheripi and Sébastien Bubeck. 2023.

Phi-2: The surprising power of small language models.

- Jiang et al. (2023)

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023.

Mistral 7b.

- Kirk et al. (2024)

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2024.

Understanding the effects of rlhf on llm generalisation and diversity.

- Lee et al. (2023)

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2023.

Rlaif: Scaling reinforcement learning from human feedback with ai feedback.

- Li et al. (2020)

Margaret Li, Stephen Roller, Ilia Kulikov, Sean Welleck, Y-Lan Boureau, Kyunghyun Cho, and Jason Weston. 2020.

Don’t say that! making inconsistent dialogue unlikely with unlikelihood training.

In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4715--4728, Online. Association for Computational Linguistics.

- Li et al. (2023a)

Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023a.

Self-alignment with instruction backtranslation.

- Li et al. (2023b)

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023b.

Alpacaeval: An automatic evaluator of instruction-following models.

https://github.com/tatsu-lab/alpaca_eval.

- Li et al. (2023c)

Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023c.

Textbooks are all you need ii: phi-1.5 technical report.

- Lin et al. (2017)

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017.

Focal loss for dense object detection.

In Proceedings of the IEEE international conference on computer vision, pages 2980--2988.

- Loshchilov and Hutter (2019)

Ilya Loshchilov and Frank Hutter. 2019.

Decoupled weight decay regularization.

- Mao et al. (2023)

Anqi Mao, Mehryar Mohri, and Yutao Zhong. 2023.

Cross-entropy loss functions: Theoretical analysis and applications.

- Ouyang et al. (2022)

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022.

Training language models to follow instructions with human feedback.

- Pang et al. (2023)

Jing-Cheng Pang, Pengyuan Wang, Kaiyuan Li, Xiong-Hui Chen, Jiacheng Xu, Zongzhang Zhang, and Yang Yu. 2023.

Language model self-improvement by reinforcement learning contemplation.

- Penedo et al. (2023)

Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023.

The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only.

- Pryzant et al. (2023)

Reid Pryzant, Dan Iter, Jerry Li, Yin Lee, Chenguang Zhu, and Michael Zeng. 2023.

Automatic prompt optimization with ‘‘gradient descent’’ and beam search.

In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7957--7968, Singapore. Association for Computational Linguistics.

- Rafailov et al. (2023)

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023.

Direct preference optimization: Your language model is secretly a reward model.

- Ramos et al. (2023)

Miguel Moura Ramos, Patrick Fernandes, António Farinhas, and André F. T. Martins. 2023.

Aligning neural machine translation models: Human feedback in training and inference.

- Rasley et al. (2020)

Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020.

Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters.

In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, page 3505–3506, New York, NY, USA. Association for Computing Machinery.

- Schulman et al. (2017)

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017.

Proximal policy optimization algorithms.

- Shen et al. (2023)

Wei Shen, Rui Zheng, Wenyu Zhan, Jun Zhao, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023.

Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback.

In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2859--2873, Singapore. Association for Computational Linguistics.

- Song et al. (2023)

Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang. 2023.

Preference ranking optimization for human alignment.

- Stiennon et al. (2022)

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. 2022.

Learning to summarize from human feedback.

- Taori et al. (2023)

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023.

Stanford alpaca: An instruction-following llama model.

https://github.com/tatsu-lab/stanford_alpaca.

- Tian et al. (2023)

Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, and Chelsea Finn. 2023.

Fine-tuning language models for factuality.

- Touvron et al. (2023)

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023.

Llama: Open and efficient foundation language models.

- Tunstall et al. (2023)

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023.

Zephyr: Direct distillation of lm alignment.

- von Werra et al. (2020)

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, and Shengyi Huang. 2020.

Trl: Transformer reinforcement learning.

https://github.com/huggingface/trl.

- Wang et al. (2024)

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, Songyang Gao, Nuo Xu, Yuhao Zhou, Xiaoran Fan, Zhiheng Xi, Jun Zhao, Xiao Wang, Tao Ji, Hang Yan, Lixing Shen, Zhan Chen, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. 2024.

Secrets of rlhf in large language models part ii: Reward modeling.

- Wang et al. (2023)

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023.

How far can camels go? exploring the state of instruction tuning on open resources.

- Wei et al. (2022)

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022.

Finetuned language models are zero-shot learners.

- Welleck et al. (2019)

Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. 2019.

Neural text generation with unlikelihood training.

arXiv preprint arXiv:1908.04319.

- Wu et al. (2023)

Tianhao Wu, Banghua Zhu, Ruoyu Zhang, Zhaojin Wen, Kannan Ramchandran, and Jiantao Jiao. 2023.

Pairwise proximal policy optimization: Harnessing relative feedback for llm alignment.

- Zhang et al. (2022)

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022.

Opt: Open pre-trained transformer language models.

- Zhao et al. (2023)

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023.

Pytorch fsdp: Experiences on scaling fully sharded data parallel.

- Zheng et al. (2023)

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023.

Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.

ArXiv:2306.05685 [cs].

- Zhou et al. (2023a)

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023a.

Lima: Less is more for alignment.

- Zhou et al. (2023b)

Haotian Zhou, Tingkai Liu, Qianli Ma, Jianbo Yuan, Pengfei Liu, Yang You, and Hongxia Yang. 2023b.

Lobass: Gauging learnability in supervised fine-tuning data.

ArXiv, abs/2310.13008.

- Zhou et al. (2023c)

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023c.

Instruction-following evaluation for large language models.

- Ziegler et al. (2020)

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020.

Fine-tuning language models from human preferences.

## Appendix A Derivation of ∇θℒO​Rsubscript∇𝜃subscriptℒ𝑂𝑅\nabla_{\theta}\mathcal{L}_{OR} with Odds Ratio

Suppose that g​(x,yl,yw)=oddsθ​P​(yw|x)oddsθ​P​(yl|x)𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥g(x,y_{l},y_{w})=\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}

∇θℒO​Rsubscript∇𝜃subscriptℒ𝑂𝑅\displaystyle\nabla_{\theta}\mathcal{L}_{OR}
=∇θlog⁡σ​(log⁡oddsθ​P​(yw|x)oddsθ​P​(yl|x))absentsubscript∇𝜃𝜎subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥\displaystyle=\nabla_{\theta}\log\sigma\left(\log\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}\right)

(20)

=σ′​(log⁡g​(x,yl,yw))σ​(log⁡g​(x,yl,yw))absentsuperscript𝜎′𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤𝜎𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤\displaystyle=\frac{\sigma^{\prime}\left(\log g(x,y_{l},y_{w})\right)}{\sigma\left(\log g(x,y_{l},y_{w})\right)}

(21)

=σ​(−log⁡g​(x,yl,yw))⋅∇θlog⁡g​(x,yl,yw)absent⋅𝜎𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤subscript∇𝜃𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤\displaystyle=\sigma\left(-\log g(x,y_{l},y_{w})\right)\cdot\nabla_{\theta}\log g(x,y_{l},y_{w})

(22)

=σ​(−log⁡g​(x,yl,yw))g​(x,yl,yw)⋅∇θg​(x,yl,yw)absent⋅𝜎𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤subscript∇𝜃𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤\displaystyle=\frac{\sigma\left(-\log g(x,y_{l},y_{w})\right)}{g(x,y_{l},y_{w})}\cdot\nabla_{\theta}g(x,y_{l},y_{w})

(23)

=σ​(−log⁡g​(x,yl,yw))⋅∇θlog⁡g​(x,yl,yw)absent⋅𝜎𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤subscript∇𝜃𝑔𝑥subscript𝑦𝑙subscript𝑦𝑤\displaystyle=\sigma\left(-\log g(x,y_{l},y_{w})\right)\cdot\nabla_{\theta}\log g(x,y_{l},y_{w})

(24)

=(1+oddsθ​P​(yw|x)oddsθ​P​(yl|x))−1⋅∇θlog⁡oddsθ​P​(yw|x)oddsθ​P​(yl|x)absent⋅superscript1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥1subscript∇𝜃subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥\displaystyle=\left(1+\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}\right)^{-1}\cdot\nabla_{\theta}\log\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}

(25)

In Equation 25, the remaining derivative can be further simplified by replacing 1−Pθ​(y|x)1subscript𝑃𝜃conditional𝑦𝑥1-P_{\theta}(y|x) terms where P​(y|x)=∏tNPθ(yt|x,y<tNP(y|x)=\sqrt[N]{\prod_{t}^{N}P_{\theta}(y_{t}|x,y_{<t}} in oddsθ​(y|x)subscriptodds𝜃conditional𝑦𝑥\textbf{odds}_{\theta}(y|x) as follows.

∇θlog⁡(1−Pθ​(y|x))subscript∇𝜃1subscript𝑃𝜃conditional𝑦𝑥\displaystyle\nabla_{\theta}\log\left(1-P_{\theta}(y|x)\right)
=∇θ(1−Pθ​(y|x))1−Pθ​(y|x)absentsubscript∇𝜃1subscript𝑃𝜃conditional𝑦𝑥1subscript𝑃𝜃conditional𝑦𝑥\displaystyle=\frac{\nabla_{\theta}\left(1-P_{\theta}(y|x)\right)}{1-P_{\theta}(y|x)}

(26)

=−∇θPθ​(y|x)1−Pθ​(y|x)absentsubscript∇𝜃subscript𝑃𝜃conditional𝑦𝑥1subscript𝑃𝜃conditional𝑦𝑥\displaystyle=\frac{-\nabla_{\theta}P_{\theta}(y|x)}{1-P_{\theta}(y|x)}

(27)

=−Pθ​(y|x)1−Pθ​(y|x)⋅∇θlog⁡Pθ​(y|x)absent⋅subscript𝑃𝜃conditional𝑦𝑥1subscript𝑃𝜃conditional𝑦𝑥subscript∇𝜃subscript𝑃𝜃conditional𝑦𝑥\displaystyle=-\frac{P_{\theta}(y|x)}{1-P_{\theta}(y|x)}\cdot\nabla_{\theta}\log P_{\theta}(y|x)

(28)

=oddsθ​(y|x)⋅∇θlog⁡Pθ​(y|x)absent⋅subscriptodds𝜃conditional𝑦𝑥subscript∇𝜃subscript𝑃𝜃conditional𝑦𝑥\displaystyle=\textbf{odds}_{\theta}(y|x)\cdot\nabla_{\theta}\log P_{\theta}(y|x)

(29)

∇θlog⁡oddsθ​P​(yw|x)oddsθ​P​(yl|x)subscript∇𝜃subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥\displaystyle\nabla_{\theta}\log\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}
=∇θlogPθ​(yw|x)Pθ​(yl|x)−(∇θlog(1−Pθ(yw|x))−∇θlog(1−Pθ(yl|x)))\displaystyle=\nabla_{\theta}\log\frac{P_{\theta}(y_{w}|x)}{P_{\theta}(y_{l}|x)}-\Bigl{(}\nabla_{\theta}\log(1-P_{\theta}(y_{w}|x))-\nabla_{\theta}\log(1-P_{\theta}(y_{l}|x))\Bigl{)}

(30)

=(1+oddsθ​P​(yw|x))​∇θlog⁡Pθ​(yw|x)−(1+oddsθ​P​(yl|x))​∇θlog⁡Pθ​(yl|x)absent1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥\displaystyle=\left(1+\textbf{odds}_{\theta}P(y_{w}|x)\right)\nabla_{\theta}\log P_{\theta}(y_{w}|x)-\left(1+\textbf{odds}_{\theta}P(y_{l}|x)\right)\nabla_{\theta}\log P_{\theta}(y_{l}|x)

(31)

Therefore, the final form of ∇θℒO​Rsubscript∇𝜃subscriptℒ𝑂𝑅\nabla_{\theta}\mathcal{L}_{OR} would be

∇θℒO​Rsubscript∇𝜃subscriptℒ𝑂𝑅\displaystyle\nabla_{\theta}\mathcal{L}_{OR}
=1+oddsθ​P​(yw|x)1+oddsθ​P​(yw|x)oddsθ​P​(yl|x)⋅∇θlog⁡Pθ​(yw|x)−1+oddsθ​P​(yl|x)1+oddsθ​P​(yw|x)oddsθ​P​(yl|x)⋅∇θlog⁡Pθ​(yl|x)absent⋅1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥⋅1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥\displaystyle=\frac{1+\textbf{odds}_{\theta}P(y_{w}|x)}{1+\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}}\cdot\nabla_{\theta}\log P_{\theta}(y_{w}|x)-\frac{1+\textbf{odds}_{\theta}P(y_{l}|x)}{1+\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}}\cdot\nabla_{\theta}\log P_{\theta}(y_{l}|x)

(32)

=(1+oddsθ​P​(yw|x)oddsθ​P​(yl|x))−1⋅(∇θlog⁡Pθ​(yw|x)1−P​(yw|x)−∇θlog⁡Pθ​(yl|x)1−P​(yl|x))absent⋅superscript1subscriptodds𝜃𝑃conditionalsubscript𝑦𝑤𝑥subscriptodds𝜃𝑃conditionalsubscript𝑦𝑙𝑥1subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑤𝑥1𝑃conditionalsubscript𝑦𝑤𝑥subscript∇𝜃subscript𝑃𝜃conditionalsubscript𝑦𝑙𝑥1𝑃conditionalsubscript𝑦𝑙𝑥\displaystyle=\left(1+\frac{\textbf{odds}_{\theta}P(y_{w}|x)}{\textbf{odds}_{\theta}P(y_{l}|x)}\right)^{-1}\cdot\left(\frac{\nabla_{\theta}\log P_{\theta}(y_{w}|x)}{1-P(y_{w}|x)}-\frac{\nabla_{\theta}\log P_{\theta}(y_{l}|x)}{1-P(y_{l}|x)}\right)

(33)

## Appendix B Ablation on Probability Ratio and Odds Ratio

In this section, we continue the discussion in Section 7.1 through empirical results comparing the log probabilities of chosen and rejected responses in UltraFeedback when trained with probability ratio and odds ratio. Recalling the sensitivity of each ratio discussed in Section 7.1, it is expected for the probability ratio to lower the log probabilities of the rejected responses with a larger scale than the odds ratio. This is well-shown in Figure 8, which is the log probabilities of each batch while fine-tuning with probability ratio (left) rapidly reaches under -4, while the same phenomenon happens after the over-fitting occurs in the case of odds ratio (right).

## Appendix C Experimental Details

Flash-Attention 2 (Dao, 2023) is applied for all the pre-trained models for computational efficiency. In particular, the OPT series and Phi-2 (2.7B) were trained with DeepSpeed ZeRO 2 (Rasley et al., 2020), Llama-2 (7B) and Mistral (7B) were trained with Fully Sharded Data Parallel(FSDP) (Zhao et al., 2023). 7B and 2.7B models were trained with four and two NVIDIA A100, and the rest were trained on four NVIDIA A6000. For optimizer, AdamW optimizer (Loshchilov and Hutter, 2019) and paged AdamW (Dettmers et al., 2023) were used, and the linear warmup with cosine decay was applied for the learning rate. For input length, every instance was truncated and padded to 1,024 tokens and 2,048 tokens for HH-RLHF and UltraFeedback, respectively. To guarantee that the models can sufficiently learn to generate the proper response to the conversation history or the complex instruction, we filtered instances with prompts with more than 1,024 tokens.

#### Supervised Fine-tuning (SFT)

For SFT, the maximum learning rate was set to 1e-5. Following Ziegler et al. (2020) and Rafailov et al. (2023), the training epoch is set to 1.

#### Reinforcement Learning with Human Feedback (RLHF)

For RLHF, the hyperparameters were set as Table 5 for UltraFeedback. For the HH-RLHF dataset, the output_min_length and output_max_length were set to 64 and 256.

Hyperparameter
Setting

ppo_epoch
4

init_kl_coef
0.1

horizon
2,000

batch_size
64

mini_batch_size
8

gradient_accumulation_steps
1

output_min_length
128

output_max_length
512

optimizer
AdamW

learning_rate
1e-05

gamma
0.99

#### Direct Preference Optimization (DPO)

For DPO, β𝛽\beta was set to 0.1 for every case. The learning rate was set to 5e-6, and the model was trained for three epochs to select the best model by evaluation loss in each epoch. However, in most cases, the first or the second checkpoint was selected as the best model as the evaluation loss increased from the third epoch.

#### Odds Ratio Preference Optimization (ORPO)

As ORPO does not require any special hyperparameter, only the learning rate and epoch were the only hyperparameter to set. For ORPO, the maximum learning rate was set to 8e-6 and trained for 10 epochs. The best model is selected based on the lowest evaluation loss for the OPT series, Phi-2 (2.7B) and Llama-2 (7B).

## Appendix D IFEval Result for Mistral-ORPO-α𝛼\alpha and Mistral-ORPO-β𝛽\beta

Along with the AlpacaEval results reported in Section 6.1, we report the results of Mistral-ORPO-α𝛼\alpha and Mistral-ORPO-β𝛽\beta on IFEval (Zhou et al., 2023c), calculated with the codes from Gao et al. (2023). Additional information can be found in: https://jiwooya1000.github.io/posts/orpo/#ifeval.

Model Type
Prompt-Strict
Prompt-Loose
Inst-Strict
Inst-Loose

Mistral-ORPO-α𝛼\alpha
0.5009
0.5083
0.5995
0.6163

Mistral-ORPO-β𝛽\beta
0.5287
0.5564
0.6355
0.6619

## Appendix E Ablation on the Weighting Value (λ𝜆\lambda)

For the weighting value λ𝜆\lambda in Equation 6, we conduct an ablation study with {0.1,0.5,1.0}0.10.51.0\{0.1,0.5,1.0\}. Mistral (7B) and UltraFeedback were used for the base model and dataset. In Section E.1, we compare the log probability trends by the value of λ𝜆\lambda, and we assess the downstream effect of λ𝜆\lambda in Section E.2.

### E.1 Log Probability

In Figure 9, we find that larger λ𝜆\lambda leads to stronger discrimination of the rejected responses in general. With λ=0.1𝜆0.1\lambda=0.1, the average log probability of the chosen and the rejected responses stay close as the fine-tuning proceeds. Also, unlike other settings, the log probabilities for the rejected responses do not decrease, but rather, the log probabilities of the chosen responses increase to minimize ℒO​Rsubscriptℒ𝑂𝑅\mathcal{L}_{OR} term.

Moreover, in λ=0.5𝜆0.5\lambda=0.5, there exists a similar trend of further increasing the log probabilities of the chosen responses, but the log probabilities of the rejected responses are diminishing simultaneously. Lastly, in λ=1.0𝜆1.0\lambda=1.0, the chosen responses diminish along with the rejected responses while enlarging the margin between them. However, this does not mean smaller λ𝜆\lambda is always the better. It will depend on the specific need and model.

### E.2 MT-Bench

The downstream impact of λ𝜆\lambda stands out in the MT-Bench result. In comparison to λ=0.1𝜆0.1\lambda=0.1, Mistral+ORPO (7B) with λ=1.0𝜆1.0\lambda=1.0 performs worse in extraction, math, and reasoning, which are the categories that generally require deterministic answers. On the other hand, it performs better in STEM, humanities, and roleplay, which ask the generations without hard answers. Along with the amount of discrepancy between the trend in the logits of chosen and rejected responses, we can infer that making a more significant margin between the chosen and the rejected responses through higher λ𝜆\lambda in ORPO leads to overly adapting to the chosen responses set in the training dataset. This proclivity results in open-ended generations generally being preferred by the annotator while showing weaker performance in the hard-answered questions.

## Appendix F Test Set Reward Distribution on HH-RLHF

Along with Figure 11, which depicts the reward distribution of OPT2-125M, OPT2-350M, and OPT2-1.3B on the UltraFeedback dataset, we report the reward distribution of each pre-trained checkpoint trained on the HH-RLHF dataset. As discussed in Section 6.3, ORPO consistently pushes the reward distribution of SFT to the right side.

## Appendix G MT-Bench Result of Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B)

For the MT-Bench result in Section 6.2, we report the category-wise scores of Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B) in Figure 12. While surpassing Llama-2 Chat (13B) and Llama-2 Chat (70B) in most cases, Mistral-ORPO-β𝛽\beta (7B) is comparable to GPT-3.5-turbo in the categories that require descriptive generations. However, it lacks coding and math skills, which we speculate is due to the lack of training data, as we used 61k instances in UltraFeedback.

## Appendix H Special Instructions for Verbosity Assessment

For the succinctness and verboseness instructions, we generated five different instructions, each with ChatGPT 999https://chat.openai.com/. From the instructions in Table 7, we randomly sampled one prompt each for every batch to prevent potential word bias.

#
Succinctness
Verboseness

1
Please generate a short and concise response.
Please generate an elaborative and chatty response.

2
Provide a brief and concise answer.
Provide a detailed answer.

3
Keep your reply short and to the point.
Keep your reply elaborative and intricate.

4
Keep your answer brief for clarity.
Keep your answer detailed.

5
Generate a brief and to-the-point answer.
Generate a chatty and step-wise answer.

## Appendix I Generation Samples from Mistral-ORPO-α𝛼\alpha (7B)

In this section, we two sample generations from Mistral-ORPO-α𝛼\alpha (7B) on AlpacaEval, including the one which is preferred over the response of GPT-4 and the one in which GPT-4 was preferred. For readability, the instances with short responses are selected for both cases. Additionally, the responses for Mistral-ORPO-α𝛼\alpha (7B) were sampled with a temperature of 0.7. The checkpoints for Mistral-ORPO-α𝛼\alpha (7B) and Mistral-ORPO-β𝛽\beta (7B) can be found in https://huggingface.co/kaist-ai/mistral-orpo-alpha and https://huggingface.co/kaist-ai/mistral-orpo-beta.

Query 1 (GPT-4 Win): Give a brief description of the book that you are given. Matilda

- 1.

GPT-4:
"Matilda" is a children’s novel written by the British author Roald Dahl. It was first published in 1988. The story revolves around a very intelligent and precocious young girl named Matilda Wormwood. Despite being neglected and mistreated by her ignorant and self-absorbed parents, Matilda finds solace in reading books and quickly develops a love for learning.

At the age of five, she begins attending school and meets Miss Honey, a kindhearted teacher who recognizes Matilda’s extraordinary intellectual abilities. However, the school’s tyrannical headmistress, Miss Trunchbull, is a cruel and oppressive figure who takes pleasure in punishing the children with her unreasonable and harsh methods.

Matilda discovers that she possesses a unique telekinetic ability, which she learns to control and use to her advantage. With her wit, intelligence, and special powers, Matilda sets out to defend her friends from Miss Trunchbull’s abuses and to help Miss Honey overcome her own difficult past.

The book is a tale of good triumphing over evil, and it celebrates the power of knowledge, kindness, and standing up for what is right. It has been adapted into a successful film, stage musical, and has remained a beloved classic in children’s literature.

- 2.

Mistral-ORPO-β𝛽\beta:
"Matilda" is a children’s book written by Roald Dahl, published in 1988. The story revolves around the life of Matilda Wormwood, a brilliant and gifted five-year-old girl who has telekinetic powers. The story explores themes of education, family, and the importance of reading and intellectual curiosity. The book has been praised for its wit, humor, and powerful message, making it a classic and beloved children’s book.

Query 2 (Mistral-ORPO-β𝛽\beta Win): What do you think about ChatGPT?

- 1.

GPT-4:
As an AI developed by OpenAI, I don’t have personal opinions. However, I can share that ChatGPT is an advanced language model that has received considerable attention for its ability to generate human-like text. It’s designed to assist users in a wide range of tasks, from answering questions to creating content and engaging in conversation.

ChatGPT has been praised for its fluency and versatility in handling various topics and for its potential to streamline and enhance communication in both personal and professional settings. However, like any technology, it also faces challenges and criticisms, such as ensuring the accuracy of information, preventing misuse, and managing ethical considerations surrounding AI-generated content.

Overall, ChatGPT represents a significant achievement in natural language processing and has sparked discussions about the future of AI in our daily lives.

- 2.

Mistral-ORPO-β𝛽\beta:
As an AI language model, I am an example of the technology used in ChatGPT. I can provide a personal opinion, but please keep in mind that I am designed to be socially unbiased and positive in nature.

ChatGPT is an impressive advancement in artificial intelligence and natural language processing. It has the ability to understand and generate human-like text, making it possible for users to have engaging and informative conversations with the AI. This technology has the potential to revolutionize various industries, including customer support, education, content creation, and more.

However, it’s important to recognize that ChatGPT, like any AI technology, is not perfect. It may sometimes provide inaccurate or biased information due to the limitations of its training data or the complexity of human language. It’s crucial to verify the facts and use critical thinking when interacting with AI systems like ChatGPT.

Overall, ChatGPT represents a significant step forward in the field of AI, and its potential applications are vast. As a responsible user, it’s important to approach this technology with a critical and respectful mindset, recognizing its strengths and limitations.

Generated on Fri Apr 5 17:28:40 2024 by LaTeXML
