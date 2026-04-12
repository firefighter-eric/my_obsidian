# KTO: Model Alignment as Prospect Theoretic Optimization

- Source HTML: `raw/html/Ethayarajh et al. - 2024 - KTO Model Alignment as Prospect Theoretic Optimization.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2402.01306
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# KTO: Model Alignment as Prospect Theoretic Optimization

Kawin Ethayarajh

  
Winnie Xu

  
Niklas Muennighoff

  
Dan Jurafsky

  
Douwe Kiela

###### Abstract

Kahneman & Tversky’s prospect theory tells us that humans perceive random variables in a biased but well-defined manner (1992); for example, humans are famously loss-averse.
We show that objectives for aligning LLMs with human feedback implicitly incorporate many of these biases—the success of these objectives (e.g., DPO) over cross-entropy minimization can partly be ascribed to them being human-aware loss functions (HALOs).
However, the utility functions these methods attribute to humans still differ from those in the prospect theory literature.
Using a Kahneman-Tversky model of human utility, we propose a HALO that directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as current methods do.
We call this approach Kahneman-Tversky Optimization (KTO), and it matches or exceeds the performance of preference-based methods at scales from 1B to 30B.
Crucially, KTO does not need preferences—only a binary signal of whether an output is desirable or undesirable for a given input.
This makes it far easier to use in the real world, where preference data is scarce and expensive.

Machine Learning, ICML

## 1 Introduction

Aligning generative models with human feedback has been successfully used to make generations more helpful, factual, and ethical, among other desiderata (Ouyang et al., 2022; Tian et al., 2023).
For LLMs, alignment methods such as RLHF and DPO have consistently proven to be more beneficial than doing supervised finetuning (SFT) alone.
However, human feedback is often discussed only in the context of preferences (e.g., output A≻Bsucceeds𝐴𝐵A\succ B for input x𝑥x), despite preferences being a kind of data that is relatively scarce and expensive to collect in the real world (Casper et al., 2023).
This is largely because the alignment methods shown to work best—RLHF (Christiano et al., 2017) and the mathematically equivalent DPO (Rafailov et al., 2023)—take preference data as input.

To understand why these alignment methods work so well, and whether feedback needs to be in the form of preferences, we frame them through the lens of prospect theory (Kahneman & Tversky, 1979; Tversky & Kahneman, 1992).
Prospect theory explains why humans make decisions about uncertain events that do not maximize expected value.
It formalizes how humans perceive random variables in a biased but well-defined manner; for example, relative to some reference point, humans are more sensitive to losses than gains, a property called loss aversion.
We show that popular alignment methods such as PPO (Schulman et al., 2017), DPO (Rafailov et al., 2023), and SLiC (Zhao et al., 2023) implicitly model such biases, helping explain their success independently of the data used.
For this reason, we call them human-aware loss functions (HALOs).

Although it is impossible to say that HALOs are categorically better than non-HALOs, we find that among existing methods, those that meet the definition of a HALO work better than those that do not.
We find that DPO performance can even be matched at most scales by running an offline PPO variant on dummy +1/-1 rewards, suggesting that preference data might not be needed if the inductive bias in the loss function is good enough.
However, despite the surprising success of this simple baseline, it significantly lags behind DPO at the 30B model scale and suffers from hyperparameter sensitivity, making it difficult to use.

Taking a more principled approach, we derive a HALO using the model of human utility that Kahneman & Tversky empirically derived to describe how humans make decisions about uncertain monetary outcomes (Tversky & Kahneman, 1992).
This approach, which we call Kahneman-Tversky Optimization (KTO), directly maximizes the utility of generations instead of maximizing the log-likelihood of preferences, as most current methods do.
KTO only requires a binary signal of whether an output is desirable or undesirable for a given input.
This data is much more abundant, cheaper, and faster to collect in the real world than preferences, making it easier to scale alignment in production environments and rapidly iterate on models.

In our experiments, we find that:

- •

KTO matches or exceeds DPO performance at scales from 1B to 30B parameters.111Our code is available on Github and models on Huggingface.
That is, taking a preference dataset of n𝑛n DPO pairs and breaking it up into 2​n2𝑛2n examples for KTO can yield better generations, despite the model ostensibly learning from a weaker signal.
We provide some theoretical explanations for this phenomenon (§4.3).

- •

KTO can handle extreme data imbalances, matching DPO performance while using up to 90% fewer desirable examples (i.e., examples of good generations).
Its success thus cannot be ascribed to the alignment data being sourced from a preference dataset.

- •

When the pretrained model is sufficiently good, one can skip supervised finetuning and go straight to KTO without a loss in generation quality.
In contrast, we find that without doing SFT first, DPO-aligned models are significantly worse at all scales.

The fact that KTO can match and sometimes even outperform DPO is surprising, given that it learns from a weaker signal.
We conclude by discussing some theoretical explanations for this phenomenon.

## 2 Background

Feedback-aligned LLMs are traditionally trained in three stages (Ouyang et al., 2022):

#### Pretraining

Given a large corpus, train the model to predict the next token conditioned on the preceding text using the cross-entropy loss.
Let π𝜋\pi denote the pretrained model.

#### Supervised Finetuning

Finetune the model to predict the next token on data that is more relevant to the downstream task.
Often, such data will comprise instructions and an appropriate response (i.e., instruction finetuning).
Let πrefsubscript𝜋ref\pi_{\text{ref}} denote the finetuned model.

#### RLHF

Given a dataset 𝒟𝒟\mathcal{D} of preferences (x,yw,yl)𝑥subscript𝑦𝑤subscript𝑦𝑙(x,y_{w},y_{l})—where x𝑥x is an input, yw,ylsubscript𝑦𝑤subscript𝑦𝑙y_{w},y_{l} are the preferred and dispreferred outputs (i.e., yw≻ylsucceedssubscript𝑦𝑤subscript𝑦𝑙y_{w}\succ y_{l} for x𝑥x), and r∗superscript𝑟r^{*} is the “true” reward function underlying the preferences—it is first assumed that the probability that ywsubscript𝑦𝑤y_{w} is preferred to ylsubscript𝑦𝑙y_{l} can be captured with a specific function class, typically a Bradley-Terry model (Bradley & Terry, 1952). Where σ𝜎\sigma is the logistic function:

p∗​(yw≻yl|x)=σ​(r∗​(x,yw)−r∗​(x,yl))superscript𝑝succeedssubscript𝑦𝑤conditionalsubscript𝑦𝑙𝑥𝜎superscript𝑟𝑥subscript𝑦𝑤superscript𝑟𝑥subscript𝑦𝑙p^{*}(y_{w}\succ y_{l}|x)=\sigma(r^{*}(x,y_{w})-r^{*}(x,y_{l}))

(1)

Since getting the true reward from a human would be intractably expensive, a reward model rϕsubscript𝑟italic-ϕr_{\phi} learns to serve as a proxy, done by minimizing the negative log-likelihood of the human preference data:

ℒR​(rϕ)=𝔼x,yw,yl∼D​[−log⁡σ​(rϕ​(x,yw)−rϕ​(x,yl))]subscriptℒ𝑅subscript𝑟italic-ϕsubscript𝔼similar-to𝑥subscript𝑦𝑤subscript𝑦𝑙𝐷delimited-[]𝜎subscript𝑟italic-ϕ𝑥subscript𝑦𝑤subscript𝑟italic-ϕ𝑥subscript𝑦𝑙\mathcal{L}_{R}(r_{\phi})=\mathbb{E}_{x,y_{w},y_{l}\sim D}[-\log\sigma(r_{\phi}(x,y_{w})-r_{\phi}(x,y_{l}))]

But solely maximizing the reward might come at the expense of desiderata such as generating grammatical text.
To avoid this, a KL divergence penalty is introduced to restrict how far the language model can drift from πrefsubscript𝜋ref\pi_{\text{ref}}.
Where πθsubscript𝜋𝜃\pi_{\theta} is the model we are optimizing, the optimal model π∗superscript𝜋\pi^{*} is that which maximizes

𝔼x∈D,y∈πθ​[rϕ​(x,y)]−βDKL(πθ(y|x)∥πref(y|x))\begin{split}\mathbb{E}_{x\in D,y\in\pi_{\theta}}[r_{\phi}(x,y)]\ &-\beta D_{\text{KL}}(\pi_{\theta}(y|x)\|\pi_{\text{ref}}(y|x))\end{split}

(2)

where β>0𝛽0\beta>0 is a hyperparameter.
Since this objective is not differentiable, we need to use an RL algorithm like PPO (Schulman et al., 2017).

However, RLHF is often slow (largely because of having to sample generations) and quite unstable in practice (especially in a distributed setting).
For this reason, recent work has focused on designing closed-form losses that maximize the margin between the preferred and dispreferred generations, such as Sequence-Likelihood Calibration (SLiC) (Zhao et al., 2023) and Direct Preference Optimization (DPO) (Rafailov et al., 2023).
The latter has become popular due to its mathematical equivalence with RLHF:

ℒDPO​(πθ,πref)=𝔼​[−log⁡σ​(β​log⁡πθ​(yw|x)πref​(yw|x)−β​log⁡πθ​(yl|x)πref​(yl|x))]subscriptℒDPOsubscript𝜋𝜃subscript𝜋ref𝔼delimited-[]𝜎𝛽subscript𝜋𝜃conditionalsubscript𝑦𝑤𝑥subscript𝜋refconditionalsubscript𝑦𝑤𝑥𝛽subscript𝜋𝜃conditionalsubscript𝑦𝑙𝑥subscript𝜋refconditionalsubscript𝑦𝑙𝑥\begin{split}&\mathcal{L}_{\text{DPO}}(\pi_{\theta},\pi_{\text{ref}})=\\
&\mathbb{E}\left[-\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}-\beta\log\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}\right)\right]\end{split}

(3)

## 3 A Prospect Theoretic View of Alignment

Kahneman & Tversky’s prospect theory explains why, faced with an uncertain event, humans make decisions that do not maximize the expected value (1992).
For example, because humans are loss-averse, given a gamble that returns $100 with 80% probability and $0 with 20% probability, a person might accept $60 to avoid the gamble, despite their certainty equivalent of $60 being less than the expected value of $80.

### 3.1 Prospect Theory

In prospect theory, human utility depends on a value function and a weighting function:222Cumulative prospect theory is the full name of the expanded theory we dicuss here (Tversky & Kahneman, 1992).

###### Definition 3.1.

A value function v:z→ℝ:𝑣→𝑧ℝv:z\to\mathbb{R} maps an outcome z𝑧z, relative to some reference point zrefsubscript𝑧refz_{\text{ref}}, to its perceived (or subjective) value.
For example, these functions capture the fact that humans tend to be more sensitive to relative losses than relative gains of the same magnitude.

###### Definition 3.2.

A weighting function w𝑤w is the derivative of a capacity function that maps cumulative probabilities to perceived cumulative probabilities.
These functions capture, for example, the fact that humans tend to overestimate the chance of rare events.
Let wzsubscript𝑤𝑧w_{z} denote the weight placed on outcome z𝑧z.

###### Definition 3.3.

The utility of a random variable Z𝑍Z is a function of its outcomes: u​(Z)≜∑z∈Zwz​v​(z−zref)≜𝑢𝑍subscript𝑧𝑍subscript𝑤𝑧𝑣𝑧subscript𝑧refu(Z)\triangleq\sum_{z\in Z}w_{z}v(z-z_{\text{ref}}).

However, because humans do not see the full probability distribution of an LLM, weighting functions are not salient to this discussion; we will focus only on value functions.
Using experiments that presented real humans with monetary gambles and asked for their certainty equivalent, Tversky & Kahneman (1992) proposed the following functional form for human value:

v​(z,zref;λ;α)={(z−zref)αif ​z>zref−λ​(zref−z)αif ​z<zref𝑣𝑧subscript𝑧ref𝜆𝛼casessuperscript𝑧subscript𝑧ref𝛼if 𝑧subscript𝑧ref𝜆superscriptsubscript𝑧ref𝑧𝛼if 𝑧subscript𝑧refv(z,z_{\text{ref}};\lambda;\alpha)=\begin{cases}(z-z_{\text{ref}})^{\alpha}&\text{if }z>z_{\text{ref}}\\
-\lambda(z_{\text{ref}}-z)^{\alpha}&\text{if }z<z_{\text{ref}}\\
\end{cases}

(4)

where the median value of hyperparameter α=0.88𝛼0.88\alpha=0.88 and λ=2.25𝜆2.25\lambda=2.25 across individuals.
α𝛼\alpha controls how quickly utility changes and λ𝜆\lambda controls the degree of loss aversion.
While the shape of the median Kahneman-Tversky value function is illustrated in Figure 2, it should be noted that it varies across individuals (Tversky & Kahneman, 1992).
There are also other functional forms for the value function that have been proposed in later work (Gurevich et al., 2009).
The salient qualities of a value function are: the existence of a reference point that is added or subtracted to get the relative gain or loss respectively; concavity in relative gains (i.e. diminishing sensitivity away from zrefsubscript𝑧refz_{\text{ref}}); loss aversion (i.e., greater sensitivity to losses).

### 3.2 HALOs

Informally, HALOs are loss functions that model the human biases in Tversky & Kahneman (1992).
Formally,

###### Definition 3.4 (HALOs).

Let x∈𝒳𝑥𝒳x\in\mathcal{X} denote an input and y∈𝒴𝑦𝒴y\in\mathcal{Y} an output.
Then f:(x,y)→ℝ:𝑓→𝑥𝑦ℝf:(x,y)\to\mathbbm{R} is a human-aware loss function if there exists the following: a parameterized reward function rθsubscript𝑟𝜃r_{\theta} such that ∀(x1,y1),(x2,y2)∈𝒳×𝒴for-allsubscript𝑥1subscript𝑦1subscript𝑥2subscript𝑦2𝒳𝒴\forall(x_{1},y_{1}),(x_{2},y_{2})\in\mathcal{X}\times\mathcal{Y},

rθ​(x1,y1)>rθ​(x2,y2)⇔(x1,y1)≻rθ(x2,y2)iffsubscript𝑟𝜃subscript𝑥1subscript𝑦1subscript𝑟𝜃subscript𝑥2subscript𝑦2subscriptsucceedssubscript𝑟𝜃subscript𝑥1subscript𝑦1subscript𝑥2subscript𝑦2r_{\theta}(x_{1},y_{1})>r_{\theta}(x_{2},y_{2})\iff(x_{1},y_{1})\succ_{r_{\theta}}(x_{2},y_{2})

reference point distributions Qx​(X′),Qy​(Y′|X′)subscript𝑄𝑥superscript𝑋′subscript𝑄𝑦conditionalsuperscript𝑌′superscript𝑋′Q_{x}(X^{\prime}),Q_{y}(Y^{\prime}|X^{\prime}), a value function vf:ℝ→ℝ:subscript𝑣𝑓→ℝℝv_{f}:\mathbbm{R}\to\mathbbm{R} that is monotonic non-decreasing and concave in (0,∞)0(0,\infty), and a negative affine function t𝑡t such that

f​(x,y;θ)=t​(vf​(rθ​(x,y)−𝔼x′,y′​[rθ​(x′,y′)]))𝑓𝑥𝑦𝜃𝑡subscript𝑣𝑓subscript𝑟𝜃𝑥𝑦subscript𝔼superscript𝑥′superscript𝑦′delimited-[]subscript𝑟𝜃superscript𝑥′superscript𝑦′f(x,y;\theta)=t(v_{f}(r_{\theta}(x,y)-\mathbb{E}_{x^{\prime},y^{\prime}}[r_{\theta}(x^{\prime},y^{\prime})]))

(5)

where x′∼Qx​(X′)similar-tosuperscript𝑥′subscript𝑄𝑥superscript𝑋′x^{\prime}\sim Q_{x}(X^{\prime}) and y′∼Qy​(Y′|x′)similar-tosuperscript𝑦′subscript𝑄𝑦conditionalsuperscript𝑌′superscript𝑥′y^{\prime}\sim Q_{y}(Y^{\prime}|x^{\prime}).

Put simply, the requirement for the reward function is that it assigns higher rewards to input-output pairs that are more preferred under it.
The reference point is the expected reward with respect to input-output pairs sampled from the distributions Qx,Qysubscript𝑄𝑥subscript𝑄𝑦Q_{x},Q_{y}.
We require that the value function be concave in gains but not necessarily convex in losses—unlike the canonical Kahneman-Tversky value functions—because in the original work on prospect theory, a minority of individuals were found to be risk-averse in both the gain and loss regime (i.e., concave in both gains and losses) (Kahneman & Tversky, 1979).
Note that risk-aversion is different from loss-aversion; they relate to the curvature and magnitude of the slope respectively.

###### Proposition 3.5.

DPO, SLiC (calibration loss only), and PPO-Clip are human-aware loss functions.

The proof is deferred to Appendix A.
In Figure 2, we can see this more intuitively by plotting the value function for each loss (i.e., the implied human utility).
We see that the value functions of all three losses incorporate a sense of loss aversion, although this is not needed to meet the definition of a HALO, since there are individuals and scenarios for which loss aversion does not necessarily apply.
The value functions are also either concave or affine (depending on the interval), unlike the standard Kahneman-Tversky value function, which is concave in gains but convex in losses.
The reference point distributions used also differs across the losses.

### 3.3 Does being a HALO matter?

A natural question is whether the modeling of human biases in HALOs has practical benefits.
This is difficult to answer, since both HALOs and non-HALOs are diverse function classes, but we attempt to do so by comparing popular non-HALO and HALO baselines on the exact same data:

- 1.

CSFT: Conditional SFT is a simple alignment method where a control token is prepended to the output during training; then, at inference, the control token corresponding to desirable generations (e.g., <|good|>) is appended to the input to induce good generations (Korbak et al., 2023).
This is a non-HALO loss.

- 2.

SLiC: SLiC with a regularization penalty (λreg≠0subscript𝜆reg0\lambda_{\text{reg}}\not=0) is a non-HALO loss:

ℒSLiC​(πθ,πref)=ℒcal​(πθ)+λreg​Lreg​(πθ)ℒcal=𝔼x,yw,yl∼D​[max⁡(0,δ−log⁡πθ​(yw|x)πθ​(yl|x))]ℒreg=𝔼x∼D,y∼πref​(x)​[−log⁡πθ​(y|x)]subscriptℒSLiCsubscript𝜋𝜃subscript𝜋refsubscriptℒcalsubscript𝜋𝜃subscript𝜆regsubscript𝐿regsubscript𝜋𝜃subscriptℒcalsubscript𝔼similar-to𝑥subscript𝑦𝑤subscript𝑦𝑙𝐷delimited-[]0𝛿subscript𝜋𝜃conditionalsubscript𝑦𝑤𝑥subscript𝜋𝜃conditionalsubscript𝑦𝑙𝑥subscriptℒregsubscript𝔼formulae-sequencesimilar-to𝑥𝐷similar-to𝑦subscript𝜋ref𝑥delimited-[]subscript𝜋𝜃conditional𝑦𝑥\begin{split}&\mathcal{L}_{\text{SLiC}}(\pi_{\theta},\pi_{\text{ref}})=\mathcal{L}_{\text{cal}}(\pi_{\theta})+\lambda_{\text{reg}}L_{\text{reg}}(\pi_{\theta})\\
&\mathcal{L}_{\text{cal}}=\mathbb{E}_{x,y_{w},y_{l}\sim D}\left[\max\left(0,\delta-\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\theta}(y_{l}|x)}\right)\right]\\
&\mathcal{L}_{\text{reg}}=\mathbb{E}_{x\sim D,y\sim\pi_{\text{ref}}(x)}[-\log\pi_{\theta}(y|x)]\end{split}

Although the max-margin loss ℒcalsubscriptℒcal\mathcal{L}_{\text{cal}} is a HALO on its own (Proposition 3.5), the complete loss is not, since the ℒregsubscriptℒreg\mathcal{L}_{\text{reg}} term is the standard language modeling loss.

- 3.

DPO: DPO, as defined in (3), is a HALO loss (Proposition 3.5).

- 4.

PPO (offline): The standard RLHF objective in (2) is typically optimized with PPO-Clip, which works by “clipping” how far πθsubscript𝜋𝜃\pi_{\theta} can drift from the version πoldsubscript𝜋old\pi_{\text{old}} at the previous step:

ℒPPO (offline)=−𝔼x,y,t∼D[min(qθA(x,y<t,yt),clip(qθ,1−ϵ,1+ϵ)A(x,y<t,yt))]subscriptℒPPO (offline)subscript𝔼similar-to𝑥𝑦𝑡𝐷delimited-[]subscript𝑞𝜃𝐴𝑥subscript𝑦absent𝑡subscript𝑦𝑡clipsubscript𝑞𝜃1italic-ϵ1italic-ϵ𝐴𝑥subscript𝑦absent𝑡subscript𝑦𝑡\begin{split}\mathcal{L}_{\text{PPO (offline)}}=&-\mathbb{E}_{x,y,t\sim D}[\min(q_{\theta}A(x,y_{<t},y_{t}),\\
&\text{clip}(q_{\theta},1-\epsilon,1+\epsilon)A(x,y_{<t},y_{t}))]\end{split}

where qθ=log⁡πθπoldsubscript𝑞𝜃subscript𝜋𝜃subscript𝜋oldq_{\theta}=\log\frac{\pi_{\theta}}{\pi_{\text{old}}} and A​(x,y<t,yt)𝐴𝑥subscript𝑦absent𝑡subscript𝑦𝑡A(x,y_{<t},y_{t}) is the per-token advantage (i.e., the surplus benefit from producing a given token in a given state).

PPO is an online algorithm—generations are sampled from the current model, judged by a reward model, and then used to update the current version.
However, this process is slow (due to having to sample generations), so we choose to use offline data instead.
Because RLHF is also quite unstable in a distributed setting, we never update πoldsubscript𝜋old\pi_{\text{old}} and keep it as πrefsubscript𝜋ref\pi_{\text{ref}}, instead clipping less conservatively than we traditionally would (see Appendix B for details).
Baheti et al. (2023) found that these changes, along with treating the entire output sequence as a single action, greatly improves stability.
However, since RLHF has historically calculated token-level advantages, we omit the third change and only preserve the first two.
The PPO-Clip loss itself is left unchanged and is therefore a HALO (Proposition 3.5).

Calling this method PPO is somewhat imprecise, because it is offline and takes only one step, but to avoid introducing too many new terms, we will call this PPO (offline).
Instead of using learned rewards, we simplify even further and use dummy +1/-1 rewards for ywsubscript𝑦𝑤y_{w} and ylsubscript𝑦𝑙y_{l} instead.
Further details on the implementation of this method can be found in Appendix B.

We compare these baselines on a suite of 7 models spanning two model families, Pythia-{1.4B, 2.8B, 6.9B, 12B} (Biderman et al., 2023) and Llama-{7B, 13B, 30B} (Touvron et al., 2023).
This permits us to see how LLM alignment scales within a model family (Llama-2 lacks a 30B model, hence our use of Llama).
Later experiments (§4.2) are done on Mistral-7B and its derivatives (Jiang et al., 2023).
The models were trained on a combination of Anthropic HH (Ganguli et al., 2022), OpenAssistant (Köpf et al., 2023), and SHP (Ethayarajh et al., 2022).

All models were aligned under identical settings on the same data (e.g., same effective batch size, same optimizer, etc.), save for hyperparameters unique to them.
Similar to Rafailov et al. (2023), the target sequences for SFT are a subset of the generations used to subsequently align the model; however, for a more realistic SFT setup, we do not necessarily set the most preferred generation to be the target (with the exception of HH, since the dispreferred output in that dataset is often harmful).
Then we used GPT-4-0613 to judge whether the aligned model’s response was better than the SFT target for the given input with respect to helpfulness, harmlessness, and conciseness, a now standard practice (Zheng et al., 2023; Li et al., 2023).333We validate that GPT-4 judgments concur with human judgments in Appendix C.
Note that while the SFT target is considered a desirable output for x𝑥x, it is by no means the best output, meaning that it can be improved upon by an aligned model.

In Figure 3, we see the results of this analysis:

- •

The HALOs we tested (DPO and our PPO variant) either match or outperform the non-HALOs at all scales, though the gap is only significant (p<0.05)𝑝0.05(p<0.05) at 13B+ model sizes.
In fact, only the HALO-aligned Llama-{13B, 30B} models match or exceed a win rate of 50% (i.e., are able to match or exceed the generation quality of the SFT targets in the test data).

- •

Up to a scale of 7B parameters, alignment provides virtually no gains over SFT alone.
However, it is worth noting that if the SFT data distribution were less similar to the preference data, then the gains from the alignment stage would ostensibly be greater.

- •

Surprisingly, despite only using dummy +1/-1 rewards, our offline PPO variant performs as well as DPO for all models except Llama30B.
This challenges conventional wisdom, which places heavy emphasis on reward learning (Casper et al., 2023), suggesting that even the simplest rewards can prove useful when used in a loss function that has a strong inductive bias.
Despite its surprising success, our offline PPO baseline still suffers from hyperparameter sensitivity and training instability, albeit not to the same extent as traditional RLHF.

## 4 Kahneman-Tversky Optimization

The surprising success of offline PPO with dummy +1/-1 rewards suggests that—with the right HALO—a binary signal of good/bad generations may be sufficient to reach DPO-level performance, even if the offline PPO approach itself was unable to do so past a certain scale (§3.3).
Taking a more principled approach, we now derive a HALO using the Kahneman-Tversky model of human utility, which allows us to directly optimize for utility instead of maximizing the log-likelihood of preferences.
This Kahneman-Tversky Optimization (KTO) loss only needs a binary signal of whether an output is (un)desirable for a given input, giving it access to a source of data is more abundant, cheaper, and faster to collect in the real world.

### 4.1 Derivation

From prior work (Go et al., 2023; Peng et al., 2019; Peters & Schaal, 2007), we know that the policy that maximizes the KL-constrained RLHF objective in (2) is

π∗​(y|x)=1Z​(x)​πref​(y|x)​exp⁡(1β​r∗​(x,y))superscript𝜋conditional𝑦𝑥1𝑍𝑥subscript𝜋refconditional𝑦𝑥1𝛽superscript𝑟𝑥𝑦\pi^{*}(y|x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r^{*}(x,y)\right)

where Z​(x)𝑍𝑥Z(x) is a partition function.
Rafailov et al. (2023) rewrite this in terms of the optimal reward for an input-output pair:

r∗​(x,y)=β​log⁡π∗​(y|x)πref​(y|x)+β​log⁡Z​(x)superscript𝑟𝑥𝑦𝛽superscript𝜋conditional𝑦𝑥subscript𝜋refconditional𝑦𝑥𝛽𝑍𝑥r^{*}(x,y)=\beta\log\frac{\pi^{*}(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)

(6)

They then plug this expression into the Bradley-Terry model of preferences and take the negative logarithm of that objective to get the DPO loss (3).

Instead, we plug this expression into the Kahneman-Tversky model of human utility, with some changes to make it more amenable to the LLM setting:

- 1.

The exponent in the Kahneman-Tversky value function (4) makes it difficult to optimize, so we set vKTOsubscript𝑣KTOv_{\text{KTO}} to be the logistic function σ𝜎\sigma, which is also concave in gains and convex in losses.
We replace the loss-aversion coefficient with two hyperparameters λD,λUsubscript𝜆𝐷subscript𝜆𝑈\lambda_{D},\lambda_{U} that weight the losses for desirable and undesirable outputs respectively.

- 2.

The Kahneman-Tversky value function was derived based on experiments with humans and monetary gambles.
Since LLM generations do not have a monetary reward associated with them, we set rKTOsubscript𝑟KTOr_{\text{KTO}} to be the implicit reward under the RLHF objective (6).

- 3.

Rather than having just one dispreferred generation yl|xconditionalsubscript𝑦𝑙𝑥y_{l}|x as the reference point, we assume that humans judge the quality of (x,y)𝑥𝑦(x,y) in relation to all input-output pairs they have seen.
Thus we write the reference point to be the expected reward under the optimal policy, not just for generations following x𝑥x but following any input x′superscript𝑥′x^{\prime}: 𝔼x′∼D,y′∼π∗​[r∗​(x′,y′)]subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]superscript𝑟superscript𝑥′superscript𝑦′\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r^{*}(x^{\prime},y^{\prime})].
Under the assumption that the expected value of the partition function across x′superscript𝑥′x^{\prime} is zero, this simplifies to the KL divergence between π∗superscript𝜋\pi^{*} and πrefsubscript𝜋ref\pi_{\text{ref}} scaled by β𝛽\beta.

Combining all of these changes, we can optimize the following loss, where the notion of an output being “desirable” or “undesirable” corresponds to the Kahneman-Tversky notion of a relative gain or loss.

LKTO​(πθ,πref)=𝔼x,y∼D​[w​(y)​(1−vKTO​(x,y;β))]subscript𝐿KTOsubscript𝜋𝜃subscript𝜋refsubscript𝔼similar-to𝑥𝑦𝐷delimited-[]𝑤𝑦1subscript𝑣KTO𝑥𝑦𝛽L_{\text{KTO}}(\pi_{\theta},\pi_{\text{ref}})=\mathbb{E}_{x,y\sim D}[w(y)(1-v_{\text{KTO}}(x,y;\beta))]

(7)

where

rKTO​(x,y)=β​log⁡πθ​(y|x)πref​(y|x)zref=𝔼x′∼D[βKL(πθ(y′|x′)∥πref(y′|x′))]vKTO​(x,y;β)={σ​(rKTO​(x,y)−zref)​if ​y∼ydesirable|xσ​(zref−rKTO​(x,y))​if ​y∼yundesirable|xw​(y)={λDif ​y∼ydesirable|xλUif ​y∼yundesirable|x\begin{split}r_{\text{KTO}}(x,y)&=\beta\log{\frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}}\\
z_{\text{ref}}&=\mathbb{E}_{x^{\prime}\sim D}\left[\beta\ \text{KL}(\pi_{\theta}(y^{\prime}|x^{\prime})\|\pi_{\text{ref}}(y^{\prime}|x^{\prime}))\right]\\
v_{\text{KTO}}(x,y;\beta)&=\begin{cases}\sigma(r_{\text{KTO}}(x,y)-z_{\text{ref}})\ \text{if }y\sim y_{\text{desirable}}|x\\
\sigma(z_{\text{ref}}-r_{\text{KTO}}(x,y))\ \text{if }y\sim y_{\text{undesirable}}|x\\
\end{cases}\\
w(y)&=\begin{cases}\lambda_{D}\quad\text{if }y\sim y_{\text{desirable}}|x\\
\lambda_{U}\quad\text{if }y\sim y_{\text{undesirable}}|x\\
\end{cases}\end{split}

Intuitively, KTO works because if the model increases the reward of a desirable example in a generic way, then the KL penalty will also rise and no progress will be made on the loss.
This forces the model to learn exactly what makes an output desirable, so that the reward can be increased while keeping the KL term flat (or even decreasing it).
A similar argument works in the other direction as well, though the non-negativity of the KL term allows faster saturation.

#### Implementation

In practice, we estimate the KL term by matching inputs x′superscript𝑥′x^{\prime} with unrelated outputs yU′superscriptsubscript𝑦𝑈′y_{U}^{\prime} in a batch of size m𝑚m and then calculating max⁡(0,1m​∑log⁡πθ​(yU′|x′)πref​(yU′|x′))01𝑚subscript𝜋𝜃conditionalsuperscriptsubscript𝑦𝑈′superscript𝑥′subscript𝜋refconditionalsuperscriptsubscript𝑦𝑈′superscript𝑥′\max\left(0,\frac{1}{m}\sum\log\frac{\pi_{\theta}(y_{U}^{\prime}|x^{\prime})}{\pi_{\text{ref}}(y_{U}^{\prime}|x^{\prime})}\right) over the entire batch.
We do not back-propagate through the KL term, as it makes training much more stable.
This means that the KL term purely serves to control how saturated the loss is.

β𝛽\beta has the same meaning as in DPO; the lower it is, the less we penalize πθsubscript𝜋𝜃\pi_{\theta} from moving away from the SFT model πrefsubscript𝜋ref\pi_{\text{ref}}.
We find that β=0.1𝛽0.1\beta=0.1 is close-to-best on most datasets.
Where nDsubscript𝑛𝐷n_{D} and nUsubscript𝑛𝑈n_{U} refer to the number of desirable and undesirable examples respectively, we set λD,λUsubscript𝜆𝐷subscript𝜆𝑈\lambda_{D},\lambda_{U} such that

λD​nDλU​nU∈[1,43]subscript𝜆𝐷subscript𝑛𝐷subscript𝜆𝑈subscript𝑛𝑈143\frac{\lambda_{D}n_{D}}{\lambda_{U}n_{U}}\in\left[1,\frac{4}{3}\right]

(8)

where at least one of the two should be set to 1 and the ratio is controlled by changing the other.
For example, if there is a 1:1 ratio of desirable:undesirable examples, we would set λU=1,λD∈[1,1.33]formulae-sequencesubscript𝜆𝑈1subscript𝜆𝐷11.33\lambda_{U}=1,\lambda_{D}\in[1,1.33].
If we then discard 90% of the desirable examples and only keep 10%, then we would set λU=1,λD∈[10,13.33]formulae-sequencesubscript𝜆𝑈1subscript𝜆𝐷1013.33\lambda_{U}=1,\lambda_{D}\in[10,13.33].
The interval [1,4/3]143[1,4/3] was determined empirically and suggests a value function that is more gain-sensitive than loss-sensitive, in contrast to the original Kahneman-Tversky value function (4).
However, the ideal interval is also task-dependent; for example, if avoiding negative outcomes were very important, then we might consider a setting of λU>1subscript𝜆𝑈1\lambda_{U}>1 instead.

#### Data

If the alignment data is naturally binary, every positive example can be assumed to be drawn from ydesirable|xconditionalsubscript𝑦desirable𝑥y_{\text{desirable}}|x and every negative example from yundesirable|xconditionalsubscript𝑦undesirable𝑥y_{\text{undesirable}}|x.
However, the canonical feedback datasets in academic research (HH, SHP, OASST) are in preference format, since the methods that have worked best up until now are preference-based.
In our experiments, we converted preference data yw≻ylsucceedssubscript𝑦𝑤subscript𝑦𝑙y_{w}\succ y_{l} by assuming that ywsubscript𝑦𝑤y_{w} is drawn from the desirable distribution and ylsubscript𝑦𝑙y_{l} from the undesirable one.
To enable an apples-to-apples comparison with DPO, we apply KTO on the same data for most experiments.
However, to ensure that KTO can be used with non-preference data, we also subsample one output y𝑦y per x𝑥x for some experiments (denoted one-y𝑦y-per-x𝑥x).

If the data is score-based, where a higher score denotes greater desirability, one has multiple options:

- •

Assume that any output with a score above some fixed threshold τ𝜏\tau is desirable.

- •

Assume that any output with a score above the mean or median (either across all inputs or just the input it was conditioned on) is desirable.

- •

Let desirability be a Bernoulli random variable where p​(y∼ydesirable|x)𝑝similar-to𝑦conditionalsubscript𝑦desirable𝑥p(y\sim y_{\text{desirable}}|x) is some function of its score (e.g., logistic).
Then randomly sample to determine whether y𝑦y is desirable or not.

### 4.2 Experiments

#### KTO ≥\geq DPO

As seen in Figure 4, SFT+KTO is competitive with SFT+DPO at model scales from 1B to 30B, despite learning from a weaker signal.
KTO alone is better than DPO alone for the Llama-{7B, 13B, 30B} models, and this gap is significant (p<0.01𝑝0.01p<0.01) at 7B and 30B even after correcting for multiple comparisons (Holm, 1979).
Perhaps most surprising is the fact that a KTO-aligned Llama-{13B, 30B} model is competitive with its SFT+KTO counterpart, despite not undergoing supervised finetuning first, and is the only alignment method of the ones we tested to show this behavior.
This is perhaps due to the fact that KTO keeps the average response length roughly the same as it is for the SFT model.
In contrast, doing DPO without SFT first causes the average response length to increase dramatically.

#### KTO data need not come from preference datasets.

Might KTO be secretly benefiting from the fact that its 2​n2𝑛2n examples in the previous experiment came from n𝑛n preference pairs instead of a naturally unpaired data distribution?
To test this, we randomly discard increasingly large fractions of the desirable data before KTO-aligning a Llama-7B model.
For example, if we discard 90% of the desirable data while leaving the undesirable data untouched, then the ratio of desirable:undesirable examples goes from 1:1 to 1:10 and the vast majority of examples no longer have a preferred output counterpart.
We handle such imbalances by changing the loss weights λD,λUsubscript𝜆𝐷subscript𝜆𝑈\lambda_{D},\lambda_{U} to satisfy the criteria in (8); when we drop 90% of the desirable data, we set λu=1,λD=13.33formulae-sequencesubscript𝜆𝑢1subscript𝜆𝐷13.33\lambda_{u}=1,\lambda_{D}=13.33.
The full results are given in Figure 6.
For Llama-7b, we find that up to 90% of the desirable data can in fact be discarded while still outperforming DPO.
A similar trend holds when discarding undesirable data.
For different models and datasets, the optimal settings of λD,λUsubscript𝜆𝐷subscript𝜆𝑈\lambda_{D},\lambda_{U} differ.

We further verify this claim by aligning Mistral-7B on OpenAssistant using DPO (on n𝑛n pairs), standard KTO (on all 2​n2𝑛2n outputs), and KTO where only one y𝑦y per x𝑥x is used.
Since the output of one y𝑦y in OpenAssistant is not conditioned on the other outputs for the same input, the latter effectively captures the setting where the data is from an inherently unpaired distribution.
Despite the one-y𝑦y-per-x𝑥x setup decreasing the amount of training data by 72%, the KTO-aligned model still outperforms both its DPO counterpart and the official instruction-tuned version of Mistral-7B (Jiang et al., 2023), as seen in Table 1.

#### On average, KTO improves performance across generative benchmarks.

Zephyr-β𝛽\beta is a variant of Mistral-7B that has been instruction-tuned and DPO-aligned on the UltraFeedback dataset (Tunstall et al., 2023; Cui et al., 2023).
We find that substituting KTO for DPO (and changing nothing else) improves performance across MMLU (0-shot) (Hendrycks et al., 2021), GSM8K (8-shot, CoT) (Cobbe et al., 2021), HumanEval (0-shot) (Chen et al., 2021), and BigBench-Hard (3-shot CoT) (Srivastava et al., 2022).
On GSM8K, just swapping DPO for KTO improves performance by 13.5 points.
Even when we align with KTO using only one y𝑦y per x𝑥x (i.e., reducing the data volume by half), we still outperform DPO on all but one benchmark.

Method
Winrate vs. SFT Target

Mistral-7B (unaligned)

0.525 ±plus-or-minus\pm 0.037

Mistral-7B + DPO

0.600 ±plus-or-minus\pm 0.037

Mistral-7B + KTO (all y𝑦y per x𝑥x)

0.652 ±plus-or-minus\pm 0.036

Mistral-7B + KTO (one y𝑦y per x𝑥x)

0.631 ±plus-or-minus\pm 0.036

Mistral-7B-Instruct

 0.621 ±plus-or-minus\pm 0.031

### 4.3 Theoretical Analysis

KTO was designed with the motivation that even if it had to learn from a weaker signal, it would make up for this limitation with the fact that it has access to much more data in the real world, where thumbs-up/thumbs-down data is common but preferences are scarce and expensive to collect.
So why does KTO perform as good or better than DPO in our experiments, when it sees the same amount of data?
Data efficiency may not be the only answer.
Our theoretical analysis suggests that preference likelihood can be maximized without necessarily maximizing underlying human utility and that KTO implicitly ignores noisy and intransitive data.

###### Proposition 4.1.

KTO does not learn from undesirable examples with sufficiently high rewards or desirable examples with sufficiently low rewards.

Informally, if an example is too difficult to learn from, then the KTO update will not change πθsubscript𝜋𝜃\pi_{\theta}.
This may be a blessing in disguise, since human preferences are often noisy and not every given preference can be recovered with the true reward r∗superscript𝑟r^{*} (Hoeffler & Ariely, 1999).
This means that it may be useful to avoid unlearnable preferences.
However, this is a double-edged sword: it also means that KTO could end up ignoring some data that is hard-to-learn but necessary to recover r∗superscript𝑟r^{*}, resulting in under-fitting.

###### Theorem 4.2.

Assuming the value function is logistic, for any bounded reward function rasubscript𝑟𝑎r_{a}, there exists a reward function in its equivalence class (i.e., rb​(x,y)=ra​(x,y)+h​(x)subscript𝑟𝑏𝑥𝑦subscript𝑟𝑎𝑥𝑦ℎ𝑥r_{b}(x,y)=r_{a}(x,y)+h(x) for some h​(x)ℎ𝑥h(x)) that induces the same optimal policy π∗superscript𝜋\pi^{*} and Bradley-Terry preference distribution but a different human value distribution.

A key insight from Rafailov et al. (2023) is that reward functions in the same equivalence class (i.e., differing only in an input-specific component) induce the same optimal policy under (2) and the same Bradley-Terry preference distribution.
However, we show under mild assumptions that the value distribution—i.e., human utility—is affected by such input-specific changes, so maximizing preference likelihood does not mean one is maximizing human utility.
Approaches that directly maximize utility, such as KTO, may thus perform better in open-ended evaluation.

Dataset (→→\rightarrow)
MMLU
GSM8k
HumanEval
BBH

Metric (→→\rightarrow)
EM
EM
pass@1
EM

Zephyr-β𝛽\beta SFT
57.2
39.0
30.1
46.3

+DPO
58.2
40.0
30.1
44.1

+KTO
58.6
53.5
30.9
52.6

+KTO (one-y𝑦y-per-x𝑥x)
58.0
50.0
30.7
49.9

###### Theorem 4.3.

Let two humans a,b𝑎𝑏a,b have value functions va,vbsubscript𝑣𝑎subscript𝑣𝑏v_{a},v_{b} and contradicting preferences y1≻ay2subscriptsucceeds𝑎subscript𝑦1subscript𝑦2y_{1}\succ_{a}y_{2} and y2≻by1subscriptsucceeds𝑏subscript𝑦2subscript𝑦1y_{2}\succ_{b}y_{1} for some input x𝑥x.
Assume πref​(y|x)=0⟹πθ​(y|x)=0subscript𝜋refconditional𝑦𝑥0subscript𝜋𝜃conditional𝑦𝑥0\pi_{\text{ref}}(y|x)=0\implies\pi_{\theta}(y|x)=0 for all x,y𝑥𝑦x,y.
In the worst-case, the optimal policy under DPO decreases the expected value of both humans.
In contrast, if each preference is broken up into two examples, then KTO (with default settings) does not change the policy.

Informally, we assume that humans want the model to increase and decrease the probability of generations they like and dislike respectively.
However, the preferences of two humans often contradict, leading to a dataset containing intransitive preferences.
In the worst-case, DPO allows one of the two preferences to be recovered while decreasing the expected value of both humans.
In contrast, KTO will change nothing at all in any case.
Since existing datasets contain preferences from multiple annotators, the existence of intransitivity may help explain why KTO works better.

### 4.4 KTO vs. DPO – when to use which?

When human feedback is in a binary format, and especially when there is an imbalance between the number of desirable and undesirable examples, KTO is the natural choice.
When your data is in the form of preferences, the choice is less clear.
Putting aside the greater data efficiency of KTO, our theoretical analysis suggests that if your preference data has sufficiently little noise and sufficiently little intransitivity, then DPO will work better, since there is some risk of KTO underfitting.
But if there is enough noise and transitivity, then the better worst-case guarantees of KTO will win out.
Most publicly available preference datasets (e.g., SHP, OpenAssistant) contain noisy feedback from many different humans whose preferences likely contradict, which explains why KTO was able to match or exceed DPO performance in our experiments.
Even AI feedback can be noisy and intransitive, which helps explain why KTO outperforms DPO when aligning with the synthetic UltraFeedback data.

## 5 Related Work

Human feedback has been used to improve LLM capabilities in translation (Kreutzer et al., 2018), summarization (Stiennon et al., 2020), sentiment-conditioned generation (Ziegler et al., 2019), and instruction-following (Ouyang et al., 2022).
The RLHF framework (Christiano et al., 2017; Bai et al., 2022) traditionally used to accomplish this is detailed in §2.

Still, momentum has largely shifted in favor of closed-form losses that directly operate on offline preferences, such as DPO (Rafailov et al., 2023).
This single stage of optimization distinguishes DPO from the conventional approach in preference-based RL, which learns a reward and then fits the policy to those rewards (Jain et al., 2013; Busa-Fekete et al., 2014).
A recent string of work has centered on the idea of “self-training” or “self-play”, during which new preference data is inferred from a model’s generations (Chen et al., 2024; Yuan et al., 2024).
Despite not being a human-aware loss, unlikelihood training was among to first to methods to align language models using a binary signal (Welleck et al., 2019).
However, work by Korbak et al. (2023) found that it is worse than the CSFT baseline we tested in our work.

Prospect theory, despite being highly influential in behavioral economics, has had a fairly muted impact in machine learning, with work concentrated in human-robot interaction (Kwon et al., 2020; Sun et al., 2019; Chan et al., 2021).
Learning from sparse binary feedback is a staple of information retrieval and recommender systems (He et al., 2017; Koren et al., 2009), although to our knowledge it has not been used to generate open-ended text.

## 6 Future Work

The existence of HALOs raises many questions. For one, the KTO loss is based on the Kahneman-Tversky value function for monetary gains and losses, which is almost certainly different from how humans perceive the relative goodness of text.
What value function—and corresponding HALO—best describes how humans perceive language?

Given that the data that KTO needs is much more abundant, cheaper, and faster to collect—both as human and AI feedback—how far can we push synthetic data?
For example, if we wanted to create a toxicity dataset to align our models to be less toxic, creating a tuple (x,yw,yl)𝑥subscript𝑦𝑤subscript𝑦𝑙(x,y_{w},y_{l}) where ylsubscript𝑦𝑙y_{l} is more toxic than ywsubscript𝑦𝑤y_{w} is non-trivial.
However, with KTO, we can easily create a dataset where desirability is determined by some black-box toxicity detection API.
What other kinds of desiderata can we synthetically optimize for with KTO?
Can we convert signals like “conversation lead to sale made” or “support ticket resolved” into KTO data?

Currently, KTO can learn from score-based data when the score is used to infer desirability.
However, can we design a HALO where scores are directly incorporated into this loss?

## 7 Conclusion

We proposed a class of functions called human-aware losses (HALOs) based on the idea of a Kahneman-Tversky value function, which models some of the key cognitive biases that inform how humans make decisions about uncertain outcomes.
We showed that among existing alignment methods, those that met the definition of a HALO performed better than those that did not, suggesting a benefit to the modeling of human biases.
We then designed a human-aware loss called KTO for directly maximizing the utility of generations instead of maximizing preference likelihood.
Despite only learning from a binary signal of whether an output is (un)desirable, KTO is as good or better than DPO at scales from 1B to 30B.
Still, we make no claims that KTO is the best HALO for all scenarios; there remains much work to be done in discovering the optimal human-aware for each setting.

## Acknowledgements

We thank Dilip Arumugam and Arya McCarthy for feedback on the paper and Nathan Lambert for feedback on an early version of this draft.
We thank Stas Bekman and Gautam Mittal for cluster assistance and Alex Manthey for helping with human evaluation.

## References

- Baheti et al. (2023)

Baheti, A., Lu, X., Brahman, F., Bras, R. L., Sap, M., and Riedl, M.

Improving language models with advantage-based offline policy gradients.

arXiv preprint arXiv:2305.14718, 2023.

- Bai et al. (2022)

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al.

Training a helpful and harmless assistant with reinforcement learning from human feedback.

arXiv preprint arXiv:2204.05862, 2022.

- Biderman et al. (2023)

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al.

Pythia: A suite for analyzing large language models across training and scaling.

In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

- Bradley & Terry (1952)

Bradley, R. A. and Terry, M. E.

Rank analysis of incomplete block designs: I. the method of paired comparisons.

Biometrika, 39(3/4):324–345, 1952.

- Busa-Fekete et al. (2014)

Busa-Fekete, R., Szörényi, B., Weng, P., Cheng, W., and Hüllermeier, E.

Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm.

Machine learning, 97:327–351, 2014.

- Casper et al. (2023)

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al.

Open problems and fundamental limitations of reinforcement learning from human feedback.

arXiv preprint arXiv:2307.15217, 2023.

- Chan et al. (2021)

Chan, L., Critch, A., and Dragan, A.

Human irrationality: both bad and good for reward inference.

arXiv preprint arXiv:2111.06956, 2021.

- Chen et al. (2021)

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al.

Evaluating large language models trained on code.

arXiv preprint arXiv:2107.03374, 2021.

- Chen et al. (2024)

Chen, Z., Deng, Y., Yuan, H., Ji, K., and Gu, Q.

Self-play fine-tuning converts weak language models to strong language models.

arXiv preprint arXiv:2401.01335, 2024.

- Christiano et al. (2017)

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D.

Deep reinforcement learning from human preferences.

Advances in neural information processing systems, 30, 2017.

- Cobbe et al. (2021)

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J.

Training verifiers to solve math word problems.

arXiv preprint arXiv:2110.14168, 2021.

- Cui et al. (2023)

Cui, G., Yuan, L., Ding, N., Yao, G., Zhu, W., Ni, Y., Xie, G., Liu, Z., and Sun, M.

Ultrafeedback: Boosting language models with high-quality feedback, 2023.

- Ethayarajh et al. (2022)

Ethayarajh, K., Choi, Y., and Swayamdipta, S.

Understanding dataset difficulty with 𝒱𝒱\mathcal{V}-usable information.

In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 5988–6008. PMLR, 17–23 Jul 2022.

- Ganguli et al. (2022)

Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al.

Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned.

arXiv preprint arXiv:2209.07858, 2022.

- Go et al. (2023)

Go, D., Korbak, T., Kruszewski, G., Rozen, J., Ryu, N., and Dymetman, M.

Aligning language models with preferences through f-divergence minimization.

arXiv preprint arXiv:2302.08215, 2023.

- Gurevich et al. (2009)

Gurevich, G., Kliger, D., and Levy, O.

Decision-making under uncertainty–a field study of cumulative prospect theory.

Journal of Banking & Finance, 33(7):1221–1229, 2009.

- He et al. (2017)

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S.

Neural collaborative filtering.

In Proceedings of the 26th international conference on world wide web, pp. 173–182, 2017.

- Hendrycks et al. (2021)

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J.

Measuring massive multitask language understanding.

Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- Hoeffler & Ariely (1999)

Hoeffler, S. and Ariely, D.

Constructing stable preferences: A look into dimensions of experience and their impact on preference stability.

Journal of consumer psychology, 8(2):113–139, 1999.

- Holm (1979)

Holm, S.

A simple sequentially rejective multiple test procedure.

Scandinavian journal of statistics, pp. 65–70, 1979.

- Jain et al. (2013)

Jain, A., Wojcik, B., Joachims, T., and Saxena, A.

Learning trajectory preferences for manipulators via iterative improvement.

Advances in neural information processing systems, 26, 2013.

- Jiang et al. (2023)

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al.

Mistral 7b.

arXiv preprint arXiv:2310.06825, 2023.

- Kahneman & Tversky (1979)

Kahneman, D. and Tversky, A.

Prospect theory: An analysis of decision under risk.

Econometrica, 47(2):263–292, 1979.

- Köpf et al. (2023)

Köpf, A., Kilcher, Y., von Rütte, D., Anagnostidis, S., Tam, Z.-R., Stevens, K., Barhoum, A., Duc, N. M., Stanley, O., Nagyfi, R., et al.

Openassistant conversations–democratizing large language model alignment.

arXiv preprint arXiv:2304.07327, 2023.

- Korbak et al. (2023)

Korbak, T., Shi, K., Chen, A., Bhalerao, R. V., Buckley, C., Phang, J., Bowman, S. R., and Perez, E.

Pretraining language models with human preferences.

In International Conference on Machine Learning, pp. 17506–17533. PMLR, 2023.

- Koren et al. (2009)

Koren, Y., Bell, R., and Volinsky, C.

Matrix factorization techniques for recommender systems.

Computer, 42(8):30–37, 2009.

- Kreutzer et al. (2018)

Kreutzer, J., Uyheng, J., and Riezler, S.

Reliability and learnability of human bandit feedback for sequence-to-sequence reinforcement learning.

In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1777–1788, 2018.

- Kwon et al. (2020)

Kwon, M., Biyik, E., Talati, A., Bhasin, K., Losey, D. P., and Sadigh, D.

When humans aren’t optimal: Robots that collaborate with risk-aware humans.

In Proceedings of the 2020 ACM/IEEE international conference on human-robot interaction, pp. 43–52, 2020.

- Li et al. (2023)

Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B.

Alpacaeval: An automatic evaluator of instruction-following models.

https://github.com/tatsu-lab/alpaca_eval, 2023.

- Ouyang et al. (2022)

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.

Training language models to follow instructions with human feedback.

Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

- Peng et al. (2019)

Peng, X. B., Kumar, A., Zhang, G., and Levine, S.

Advantage-weighted regression: Simple and scalable off-policy reinforcement learning.

arXiv preprint arXiv:1910.00177, 2019.

- Peters & Schaal (2007)

Peters, J. and Schaal, S.

Reinforcement learning by reward-weighted regression for operational space control.

In Proceedings of the 24th international conference on Machine learning, pp. 745–750, 2007.

- Rafailov et al. (2023)

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C.

Direct preference optimization: Your language model is secretly a reward model.

arXiv preprint arXiv:2305.18290, 2023.

- Schulman et al. (2017)

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.

Proximal policy optimization algorithms.

arXiv preprint arXiv:1707.06347, 2017.

- Srivastava et al. (2022)

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al.

Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

arXiv preprint arXiv:2206.04615, 2022.

- Stiennon et al. (2020)

Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F.

Learning to summarize with human feedback.

Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

- Sun et al. (2019)

Sun, L., Zhan, W., Hu, Y., and Tomizuka, M.

Interpretable modelling of driving behaviors in interactive driving scenarios based on cumulative prospect theory.

In 2019 IEEE Intelligent Transportation Systems Conference (ITSC), pp. 4329–4335. IEEE, 2019.

- Tian et al. (2023)

Tian, K., Mitchell, E., Yao, H., Manning, C. D., and Finn, C.

Fine-tuning language models for factuality.

arXiv preprint arXiv:2311.08401, 2023.

- Touvron et al. (2023)

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.

Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971, 2023.

- Tunstall et al. (2023)

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., and Wolf, T.

Zephyr: Direct distillation of lm alignment, 2023.

- Tversky & Kahneman (1992)

Tversky, A. and Kahneman, D.

Advances in prospect theory: Cumulative representation of uncertainty.

Journal of Risk and uncertainty, 5:297–323, 1992.

- von Werra et al. (2020)

von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., and Huang, S.

Trl: Transformer reinforcement learning.

https://github.com/huggingface/trl, 2020.

- Welleck et al. (2019)

Welleck, S., Kulikov, I., Roller, S., Dinan, E., Cho, K., and Weston, J.

Neural text generation with unlikelihood training.

In International Conference on Learning Representations, 2019.

- Yuan et al. (2024)

Yuan, W., Pang, R. Y., Cho, K., Sukhbaatar, S., Xu, J., and Weston, J.

Self-rewarding language models.

arXiv preprint arXiv:2401.10020, 2024.

- Zhao et al. (2023)

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J.

Slic-hf: Sequence likelihood calibration with human feedback.

arXiv preprint arXiv:2305.10425, 2023.

- Zheng et al. (2023)

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al.

Judging llm-as-a-judge with mt-bench and chatbot arena.

arXiv preprint arXiv:2306.05685, 2023.

- Ziegler et al. (2019)

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G.

Fine-tuning language models from human preferences.

arXiv preprint arXiv:1909.08593, 2019.

## Appendix A Proofs

#### Proposition 3.5 (restated)

DPO, SLiC (calibration loss only), and PPO-Clip are human-aware loss functions.

###### Proof.

For a loss to be a HALO, it needs to be expressible as

f​(x,y;θ)=t​(vf​(rθ​(x,y)−𝔼x′∼Qx′,y′∼Qy′​[rθ​(x′,y′)]))𝑓𝑥𝑦𝜃𝑡subscript𝑣𝑓subscript𝑟𝜃𝑥𝑦subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′subscriptsuperscript𝑄′𝑥similar-tosuperscript𝑦′subscriptsuperscript𝑄′𝑦delimited-[]subscript𝑟𝜃superscript𝑥′superscript𝑦′f(x,y;\theta)=t(v_{f}(r_{\theta}(x,y)-\mathbb{E}_{x^{\prime}\sim Q^{\prime}_{x},y^{\prime}\sim Q^{\prime}_{y}}[r_{\theta}(x^{\prime},y^{\prime})]))

with a parameterized reward function rθsubscript𝑟𝜃r_{\theta} such that ∀(x1,y1),(x2,y2)∈𝒳×𝒴for-allsubscript𝑥1subscript𝑦1subscript𝑥2subscript𝑦2𝒳𝒴\forall(x_{1},y_{1}),(x_{2},y_{2})\in\mathcal{X}\times\mathcal{Y}, rθ​(x1,y1)>rθ​(x2,y2)⇔(x1,y1)≻rθ(x2,y2)iffsubscript𝑟𝜃subscript𝑥1subscript𝑦1subscript𝑟𝜃subscript𝑥2subscript𝑦2subscriptsucceedssubscript𝑟𝜃subscript𝑥1subscript𝑦1subscript𝑥2subscript𝑦2r_{\theta}(x_{1},y_{1})>r_{\theta}(x_{2},y_{2})\iff(x_{1},y_{1})\succ_{r_{\theta}}(x_{2},y_{2}), reference point distributions Qx​(X′),Qy​(Y′|X′)subscript𝑄𝑥superscript𝑋′subscript𝑄𝑦conditionalsuperscript𝑌′superscript𝑋′Q_{x}(X^{\prime}),Q_{y}(Y^{\prime}|X^{\prime}), a value function vf:ℝ→ℝ:subscript𝑣𝑓→ℝℝv_{f}:\mathbbm{R}\to\mathbbm{R} that is monotonic non-decreasing and concave in (0,∞)0(0,\infty), and a negative affine function t𝑡t.

The DPO loss is

ℒDPO​(πθ,πref)=𝔼​[−log⁡σ​(β​log⁡πθ​(yw|x)πref​(yw|x)−β​log⁡πθ​(yl|x)πref​(yl|x))]subscriptℒDPOsubscript𝜋𝜃subscript𝜋ref𝔼delimited-[]𝜎𝛽subscript𝜋𝜃conditionalsubscript𝑦𝑤𝑥subscript𝜋refconditionalsubscript𝑦𝑤𝑥𝛽subscript𝜋𝜃conditionalsubscript𝑦𝑙𝑥subscript𝜋refconditionalsubscript𝑦𝑙𝑥\mathcal{L}_{\text{DPO}}(\pi_{\theta},\pi_{\text{ref}})=\mathbb{E}\left[-\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\text{ref}}(y_{w}|x)}-\beta\log\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\text{ref}}(y_{l}|x)}\right)\right]

where β>0𝛽0\beta>0 is a hyperparameter.
DPO meets the criteria with the following construction: t​(⋅)𝑡⋅t(\cdot) is just taking the negative, vf=log⁡σsubscript𝑣𝑓𝜎v_{f}=\log\sigma is increasing and concave everywhere, rθsubscript𝑟𝜃r_{\theta} is the DPO reward β​log⁡[πθ​(y|x)/πref​(y|x)]𝛽subscript𝜋𝜃conditional𝑦𝑥subscript𝜋refconditional𝑦𝑥\beta\log[\pi_{\theta}(y|x)/\pi_{\text{ref}}(y|x)], Qxsubscript𝑄𝑥Q_{x} places all mass on x𝑥x and Qysubscript𝑄𝑦Q_{y} places all mass on the dispreferred output ylsubscript𝑦𝑙y_{l} for x𝑥x such that y≻ylsucceeds𝑦subscript𝑦𝑙y\succ y_{l}.

The calibration loss in SLiC is

ℒcal=𝔼x,yw,yl∼D​[max⁡(0,δ−log⁡πθ​(yw|x)+log⁡πθ​(yl|x))]=𝔼x,yw,yl∼D​[−min⁡(0,log⁡πθ​(yw|x)−log⁡πθ​(yl|x)−δ)]subscriptℒcalsubscript𝔼similar-to𝑥subscript𝑦𝑤subscript𝑦𝑙𝐷delimited-[]0𝛿subscript𝜋𝜃conditionalsubscript𝑦𝑤𝑥subscript𝜋𝜃conditionalsubscript𝑦𝑙𝑥subscript𝔼similar-to𝑥subscript𝑦𝑤subscript𝑦𝑙𝐷delimited-[]0subscript𝜋𝜃conditionalsubscript𝑦𝑤𝑥subscript𝜋𝜃conditionalsubscript𝑦𝑙𝑥𝛿\begin{split}\mathcal{L}_{\text{cal}}&=\mathbb{E}_{x,y_{w},y_{l}\sim D}\left[\max\left(0,\delta-\log\pi_{\theta}(y_{w}|x)+\log\pi_{\theta}(y_{l}|x)\right)\right]\\
&=\mathbb{E}_{x,y_{w},y_{l}\sim D}\left[-\min\left(0,\log\pi_{\theta}(y_{w}|x)-\log\pi_{\theta}(y_{l}|x)-\delta\right)\right]\end{split}

where δ>0𝛿0\delta>0 is hyperparameter.
The calibration loss meets the criteria under the following construction: t​(⋅)𝑡⋅t(\cdot) is just taking the negative, vf​(z)=min⁡(0,z−δ)subscript𝑣𝑓𝑧0𝑧𝛿v_{f}(z)=\min(0,z-\delta) is non-decreasing everywhere and concave in gains, rθ​(x,y)=log⁡pθ​(y|x)subscript𝑟𝜃𝑥𝑦subscript𝑝𝜃conditional𝑦𝑥r_{\theta}(x,y)=\log p_{\theta}(y|x), and the reference point distributions are defined the same as they are for DPO.

The PPO-Clip loss is

ℒPPO (offline)=−𝔼x,y,i∼D​[min⁡(qθ​A​(x,y<i,yi),clip​(qθ,1−ϵ,1+ϵ)​A​(x,y<i,yi))]subscriptℒPPO (offline)subscript𝔼similar-to𝑥𝑦𝑖𝐷delimited-[]subscript𝑞𝜃𝐴𝑥subscript𝑦absent𝑖subscript𝑦𝑖clipsubscript𝑞𝜃1italic-ϵ1italic-ϵ𝐴𝑥subscript𝑦absent𝑖subscript𝑦𝑖\begin{split}\mathcal{L}_{\text{PPO (offline)}}=-\mathbb{E}_{x,y,i\sim D}[\min(q_{\theta}A(x,y_{<i},y_{i}),\text{clip}(q_{\theta},1-\epsilon,1+\epsilon)A(x,y_{<i},y_{i}))]\end{split}

where qθ=πθ​(yt|x,y<i)πref​(yt|x,y<i)subscript𝑞𝜃subscript𝜋𝜃conditionalsubscript𝑦𝑡𝑥subscript𝑦absent𝑖subscript𝜋refconditionalsubscript𝑦𝑡𝑥subscript𝑦absent𝑖q_{\theta}=\frac{\pi_{\theta}(y_{t}|x,y_{<i})}{\pi_{\text{ref}}(y_{t}|x,y_{<i})} are the token-level probability ratios, where y<isubscript𝑦absent𝑖y_{<i} denotes the output sequence up to the i𝑖i-th token.
This token-level focus makes this objective different from DPO and SLiC.
A𝐴A denotes the token-level advantages, and ϵ>0italic-ϵ0\epsilon>0 is a hyperparameter.
The reward rθ​(x,y)=log⁡qθsubscript𝑟𝜃𝑥𝑦subscript𝑞𝜃r_{\theta}(x,y)=\log q_{\theta} and t𝑡t then just takes the negative.
We can let Qxsubscript𝑄𝑥Q_{x} place all mass on the joint sequence x:y:𝑥𝑦x:y and Qysubscript𝑄𝑦Q_{y} be an arbitrary distribution over 𝒴𝒴\mathcal{Y} — since there is no advantage to generating tokens past ynsubscript𝑦𝑛y_{n}, the distributions π∗(⋅|x:y)\pi^{*}(\cdot|x:y) and πref(⋅|x:y)\pi_{\text{ref}}(\cdot|x:y) should be arbitrarily close, pushing log⁡qθ→0→subscript𝑞𝜃0\log q_{\theta}\to 0.
We construct the value function piecewise:

vf​(z)={A​min⁡(exp⁡z,1+ϵ)if ​A​(x,y<i,yi)>0A​max⁡(exp⁡z,1−ϵ)if ​A​(x,y<i,yi)<0subscript𝑣𝑓𝑧cases𝐴𝑧1italic-ϵif 𝐴𝑥subscript𝑦absent𝑖subscript𝑦𝑖0𝐴𝑧1italic-ϵif 𝐴𝑥subscript𝑦absent𝑖subscript𝑦𝑖0v_{f}(z)=\begin{cases}A\min(\exp z,1+\epsilon)&\text{if }A(x,y_{<i},y_{i})>0\\
A\max(\exp z,1-\epsilon)&\text{if }A(x,y_{<i},y_{i})<0\\
\end{cases}

∎

#### Proposition 4.1 (restated)

KTO does not learn from undesirable examples with sufficiently high rewards or desirable examples with sufficiently low rewards.

###### Proof.

Where λ​(y)=−λD𝜆𝑦subscript𝜆𝐷\lambda(y)=-\lambda_{D} when y𝑦y is desirable and λUsubscript𝜆𝑈\lambda_{U} when y𝑦y is undesirable, and z=rKTO​(x,y)−zref𝑧subscript𝑟KTO𝑥𝑦subscript𝑧refz=r_{\text{KTO}}(x,y)-z_{\text{ref}}, the derivative of the KTO loss is

∇θLKTO​(πθ,πref)=𝔼x,y∼D​[λ​(y)​σ​(z)​σ​(−z)​∇β​log⁡πθ​(y|x)]subscript∇𝜃subscript𝐿KTOsubscript𝜋𝜃subscript𝜋refsubscript𝔼similar-to𝑥𝑦𝐷delimited-[]𝜆𝑦𝜎𝑧𝜎𝑧∇𝛽subscript𝜋𝜃conditional𝑦𝑥\nabla_{\theta}L_{\text{KTO}}(\pi_{\theta},\pi_{\text{ref}})=\mathbb{E}_{x,y\sim D}\left[\lambda(y)\sigma(z)\sigma(-z)\nabla\beta\log\pi_{\theta}(y|x)\right]

(9)

Note that we do not backpropagate through the KL term in the KTO loss and β>0𝛽0\beta>0.
This gradient is simple to interpret: if y𝑦y is desirable, then λ​(y)𝜆𝑦\lambda(y) is negative and we push up the probability of πθ​(y|x)subscript𝜋𝜃conditional𝑦𝑥\pi_{\theta}(y|x) to minimize the loss; we do the opposite if y𝑦y is undesirable.
As z𝑧z tends to ±∞plus-or-minus\pm\infty, the gradient will tend to zero since either σ​(−z)𝜎𝑧\sigma(-z) or σ​(z)𝜎𝑧\sigma(z) will tend to zero.
Since z𝑧z is increasing in the reward, this means that sufficiently large and sufficiently small rewards will yield a gradient of zero.
∎

#### Theorem 4.2 (restated)

Assuming the value function is logistic, for any bounded reward function rasubscript𝑟𝑎r_{a}, there exists a reward function in its equivalence class (i.e., rb​(x,y)=ra​(x,y)+h​(x)subscript𝑟𝑏𝑥𝑦subscript𝑟𝑎𝑥𝑦ℎ𝑥r_{b}(x,y)=r_{a}(x,y)+h(x) for some h​(x)ℎ𝑥h(x)) that induces the same optimal policy π∗superscript𝜋\pi^{*} and Bradley-Terry preference distribution but a different human value distribution.

###### Proof.

Following the definition in Rafailov et al. (2023), we say two functions ra​(x,y)subscript𝑟𝑎𝑥𝑦r_{a}(x,y) and rb​(x,y)subscript𝑟𝑏𝑥𝑦r_{b}(x,y) are in the same equivalence class if there exists some function h​(x)ℎ𝑥h(x) such that rb​(x,y)=ra​(x,y)+h​(x)subscript𝑟𝑏𝑥𝑦subscript𝑟𝑎𝑥𝑦ℎ𝑥r_{b}(x,y)=r_{a}(x,y)+h(x).
From Lemma 1 in Rafailov et al. (2023), we know that two functions in the same equivalence class induce the same optimal policy:

πrb​(y|x)=1Z​(x)​πref​(y|x)​exp⁡(1β​rb​(x,y))=1∑yπref​(y|x)​exp⁡(1β​(ra​(x,y)+h​(x)))​πref​(y|x)​exp⁡(1β​(ra​(x,y)+h​(x)))=1∑yπref​(y|x)​exp⁡(1β​ra​(x,y))​exp⁡(1β​h​(x))​πref​(y|x)​exp⁡(1β​ra​(x,y))​exp⁡(1β​h​(x))=πra​(y|x)subscript𝜋subscript𝑟𝑏conditional𝑦𝑥1𝑍𝑥subscript𝜋refconditional𝑦𝑥1𝛽subscript𝑟𝑏𝑥𝑦1subscript𝑦subscript𝜋refconditional𝑦𝑥1𝛽subscript𝑟𝑎𝑥𝑦ℎ𝑥subscript𝜋refconditional𝑦𝑥1𝛽subscript𝑟𝑎𝑥𝑦ℎ𝑥1subscript𝑦subscript𝜋refconditional𝑦𝑥1𝛽subscript𝑟𝑎𝑥𝑦1𝛽ℎ𝑥subscript𝜋refconditional𝑦𝑥1𝛽subscript𝑟𝑎𝑥𝑦1𝛽ℎ𝑥subscript𝜋subscript𝑟𝑎conditional𝑦𝑥\begin{split}\pi_{r_{b}}(y|x)&=\frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r_{b}(x,y)\right)\\
&=\frac{1}{\sum_{y}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}(r_{a}(x,y)+h(x))\right)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}(r_{a}(x,y)+h(x))\right)\\
&=\frac{1}{\sum_{y}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r_{a}(x,y)\right)\exp\left(\frac{1}{\beta}h(x)\right)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r_{a}(x,y)\right)\exp\left(\frac{1}{\beta}h(x)\right)\\
&=\pi_{r_{a}}(y|x)\end{split}

For a Bradley-Terry model of preferences, it is trivial to show that p​(yw≻yl|x)𝑝succeedssubscript𝑦𝑤conditionalsubscript𝑦𝑙𝑥p(y_{w}\succ y_{l}|x) is unaffected by h​(x)ℎ𝑥h(x) since it is added to the reward of both ywsubscript𝑦𝑤y_{w} and ylsubscript𝑦𝑙y_{l}.
We will now show that the two reward functions do not necessarily induce the same distribution of values.
Let

hu​(x)={0if ​x≠uCuif ​x=usubscriptℎ𝑢𝑥cases0if 𝑥𝑢subscript𝐶𝑢if 𝑥𝑢h_{u}(x)=\begin{cases}0&\quad\text{if }x\not=u\\
C_{u}&\quad\text{if }x=u\\
\end{cases}

where Cu≠0subscript𝐶𝑢0C_{u}\not=0 is an input-specific constant.

Assume that y𝑦y is a desirable output for x𝑥x without loss of generality.
Let nusubscript𝑛𝑢n_{u} be the number of times u𝑢u appears in a set of size n𝑛n.
For reward functions ra,rbsubscript𝑟𝑎subscript𝑟𝑏r_{a},r_{b} with corresponding logistic value functions va,vbsubscript𝑣𝑎subscript𝑣𝑏v_{a},v_{b} such that rb​(x,y)=ra​(x,y)+hu​(x)subscript𝑟𝑏𝑥𝑦subscript𝑟𝑎𝑥𝑦subscriptℎ𝑢𝑥r_{b}(x,y)=r_{a}(x,y)+h_{u}(x) for some input u𝑢u, :

vb​(x,y)=σ​(rb​(x,y)−𝔼x′∼D,y′∼π∗​[rb​(x′,y′)])=σ​(ra​(x,y)+hu​(x)−𝔼x′∼D,y′∼π∗​[ra​(x′,y′)+hu​(x′)])=σ​(ra​(x,y)−𝔼x′∼D,y′∼π∗​[ra​(x′,y′)]+(hu​(x)−𝔼x′∼D​[hu​(x′)]))subscript𝑣𝑏𝑥𝑦𝜎subscript𝑟𝑏𝑥𝑦subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]subscript𝑟𝑏superscript𝑥′superscript𝑦′𝜎subscript𝑟𝑎𝑥𝑦subscriptℎ𝑢𝑥subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]subscript𝑟𝑎superscript𝑥′superscript𝑦′subscriptℎ𝑢superscript𝑥′𝜎subscript𝑟𝑎𝑥𝑦subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]subscript𝑟𝑎superscript𝑥′superscript𝑦′subscriptℎ𝑢𝑥subscript𝔼similar-tosuperscript𝑥′𝐷delimited-[]subscriptℎ𝑢superscript𝑥′\begin{split}v_{b}(x,y)&=\sigma(r_{b}(x,y)-\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r_{b}(x^{\prime},y^{\prime})])\\
&=\sigma(r_{a}(x,y)+h_{u}(x)-\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r_{a}(x^{\prime},y^{\prime})+h_{u}(x^{\prime})])\\
&=\sigma(r_{a}(x,y)-\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r_{a}(x^{\prime},y^{\prime})]+(h_{u}(x)-\mathbb{E}_{x^{\prime}\sim D}[h_{u}(x^{\prime})]))\\
\end{split}

Let va′subscriptsuperscript𝑣′𝑎v^{\prime}_{a} be the derivative of vasubscript𝑣𝑎v_{a}, and let ϵitalic-ϵ\epsilon denote the error term from a first-order Taylor series expansion.

When u=x𝑢𝑥u=x:

vb​(x,y)=σ​(ra​(x,y)−𝔼x′∼D,y′∼π∗​[ra​(x′,y′)]+(Cu−nun​Cu))=va​(x,y)+(Cu−nu​Cun)​va′​(x,y)+ϵ=va​(x,y)+(Cu−nu​Cun)​va​(x,y)​(1−va​(x,y))+ϵsubscript𝑣𝑏𝑥𝑦𝜎subscript𝑟𝑎𝑥𝑦subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]subscript𝑟𝑎superscript𝑥′superscript𝑦′subscript𝐶𝑢subscript𝑛𝑢𝑛subscript𝐶𝑢subscript𝑣𝑎𝑥𝑦subscript𝐶𝑢subscript𝑛𝑢subscript𝐶𝑢𝑛subscriptsuperscript𝑣′𝑎𝑥𝑦italic-ϵsubscript𝑣𝑎𝑥𝑦subscript𝐶𝑢subscript𝑛𝑢subscript𝐶𝑢𝑛subscript𝑣𝑎𝑥𝑦1subscript𝑣𝑎𝑥𝑦italic-ϵ\begin{split}v_{b}(x,y)&=\sigma\left(r_{a}(x,y)-\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r_{a}(x^{\prime},y^{\prime})]+\left(C_{u}-\frac{n_{u}}{n}C_{u}\right)\right)\\
&=v_{a}(x,y)+\left(C_{u}-\frac{n_{u}C_{u}}{n}\right)v^{\prime}_{a}(x,y)+\epsilon\\
&=v_{a}(x,y)+\left(C_{u}-\frac{n_{u}C_{u}}{n}\right)v_{a}(x,y)(1-v_{a}(x,y))+\epsilon\end{split}

When u≠x𝑢𝑥u\not=x,

vb​(x,y)=σ​(ra​(x,y)−𝔼x′∼D,y′∼π∗​[ra​(x′,y′)]−nun​Cu)=va​(x,y)−nu​Cun​va​(x,y)​(1−va​(x,y))+ϵsubscript𝑣𝑏𝑥𝑦𝜎subscript𝑟𝑎𝑥𝑦subscript𝔼formulae-sequencesimilar-tosuperscript𝑥′𝐷similar-tosuperscript𝑦′superscript𝜋delimited-[]subscript𝑟𝑎superscript𝑥′superscript𝑦′subscript𝑛𝑢𝑛subscript𝐶𝑢subscript𝑣𝑎𝑥𝑦subscript𝑛𝑢subscript𝐶𝑢𝑛subscript𝑣𝑎𝑥𝑦1subscript𝑣𝑎𝑥𝑦italic-ϵ\begin{split}v_{b}(x,y)&=\sigma\left(r_{a}(x,y)-\mathbb{E}_{x^{\prime}\sim D,y^{\prime}\sim\pi^{*}}[r_{a}(x^{\prime},y^{\prime})]-\frac{n_{u}}{n}C_{u}\right)\\
&=v_{a}(x,y)-\frac{n_{u}C_{u}}{n}v_{a}(x,y)(1-v_{a}(x,y))+\epsilon\end{split}

Since the rewards are bounded by assumption, we have va∈(0,1)subscript𝑣𝑎01v_{a}\in(0,1).
For a k𝑘k-th order Taylor Series approximation, we thus have ϵ∈O​(Cu​2−k)italic-ϵ𝑂subscript𝐶𝑢superscript2𝑘\epsilon\in O(C_{u}2^{-k}).
Even if we generously assume that ϵ=0italic-ϵ0\epsilon=0, we have vb​(x,⋅)≠va​(x,⋅)subscript𝑣𝑏𝑥⋅subscript𝑣𝑎𝑥⋅v_{b}(x,\cdot)\not=v_{a}(x,\cdot) in at least one of these cases (either when nu>0subscript𝑛𝑢0n_{u}>0 or when nu=nsubscript𝑛𝑢𝑛n_{u}=n).
We have thus shown that two bounded reward functions in the same equivalence class can induce both the same policy and Bradley-Terry preference distribution but a different distribution of human values.
∎

#### Theorem 4.3 (restated)

Let two humans a,b𝑎𝑏a,b have value functions va,vbsubscript𝑣𝑎subscript𝑣𝑏v_{a},v_{b} and contradicting preferences y1≻ay2subscriptsucceeds𝑎subscript𝑦1subscript𝑦2y_{1}\succ_{a}y_{2} and y2≻by1subscriptsucceeds𝑏subscript𝑦2subscript𝑦1y_{2}\succ_{b}y_{1} for some input x𝑥x.
Assume πref​(y|x)=0⟹πθ​(y|x)=0subscript𝜋refconditional𝑦𝑥0subscript𝜋𝜃conditional𝑦𝑥0\pi_{\text{ref}}(y|x)=0\implies\pi_{\theta}(y|x)=0 for all x,y𝑥𝑦x,y.
In the worst-case, the optimal policy under DPO decreases the expected value of both humans.
In contrast, if each preference is broken up into two examples, then KTO (with default settings) does not change the policy.

###### Proof.

Where u=β​log⁡πθ​(y1|x)πref​(y1|x)−β​log⁡πθ​(y2|x)πref​(y2|x)𝑢𝛽subscript𝜋𝜃conditionalsubscript𝑦1𝑥subscript𝜋refconditionalsubscript𝑦1𝑥𝛽subscript𝜋𝜃conditionalsubscript𝑦2𝑥subscript𝜋refconditionalsubscript𝑦2𝑥u=\beta\log\frac{\pi_{\theta}(y_{1}|x)}{\pi_{\text{ref}}(y_{1}|x)}-\beta\log\frac{\pi_{\theta}(y_{2}|x)}{\pi_{\text{ref}}(y_{2}|x)}, we can write the total DPO loss as

ℒ^DPO​(πθ,πref)=12​(−log⁡σ​(u))+12​(−log⁡σ​(−u))subscript^ℒDPOsubscript𝜋𝜃subscript𝜋ref12𝜎𝑢12𝜎𝑢\hat{\mathcal{L}}_{\text{DPO}}(\pi_{\theta},\pi_{\text{ref}})=\frac{1}{2}(-\log\sigma(u))+\frac{1}{2}(-\log\sigma(-u))

Taking the derivative with respect to u𝑢u and setting to zero, we get

0=−12​(σ​(u)​σ​(−u)σ​(u)−σ​(−u)​σ​(u)σ​(−u))=σ​(u)​(σ​(−u))2σ​(u)​σ​(−u)−(σ​(u))2​σ​(−u)σ​(u)​σ​(−u)=σ​(−u)−σ​(u)⟹u=0012𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢superscript𝜎𝑢2𝜎𝑢𝜎𝑢superscript𝜎𝑢2𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢𝜎𝑢𝑢0\begin{split}0&=-\frac{1}{2}\left(\frac{\sigma(u)\sigma(-u)}{\sigma(u)}-\frac{\sigma(-u)\sigma(u)}{\sigma(-u)}\right)\\
&=\frac{\sigma(u)(\sigma(-u))^{2}}{\sigma(u)\sigma(-u)}-\frac{(\sigma(u))^{2}\sigma(-u)}{\sigma(u)\sigma(-u)}\\
&=\sigma(-u)-\sigma(u)\\
\implies u&=0\end{split}

Since β>0𝛽0\beta>0, u=0𝑢0u=0 can only occur when the rewards of both the preferred and dispreferred outputs are equal.
This can be satisfied when πθ​(y1|x)=πθ​(y2|x)=0subscript𝜋𝜃conditionalsubscript𝑦1𝑥subscript𝜋𝜃conditionalsubscript𝑦2𝑥0\pi_{\theta}(y_{1}|x)=\pi_{\theta}(y_{2}|x)=0, with the probability mass allocated to examples with lower true reward r∗superscript𝑟r^{*} in the worst case.
Since value functions by definition are monotonically non-decreasing, the expected value for both humans would decrease in the worst-case.

Where z1,z2subscript𝑧1subscript𝑧2z_{1},z_{2} are the reference-adjusted rewards, we can write the total KTO loss (with default settings λD=λU=1subscript𝜆𝐷subscript𝜆𝑈1\lambda_{D}=\lambda_{U}=1) as:

ℒ^KTO​(πθ,πref)=14​(1−σ​(z1))+14​(1−σ​(−z1))+14​(1−σ​(z2))+14​(1−σ​(−z2))=14​(σ​(−z1))+14​(1−σ​(−z1))+14​(σ​(−z2))+14​(1−σ​(−z2))=12subscript^ℒKTOsubscript𝜋𝜃subscript𝜋ref141𝜎subscript𝑧1141𝜎subscript𝑧1141𝜎subscript𝑧2141𝜎subscript𝑧214𝜎subscript𝑧1141𝜎subscript𝑧114𝜎subscript𝑧2141𝜎subscript𝑧212\begin{split}\hat{\mathcal{L}}_{\text{KTO}}(\pi_{\theta},\pi_{\text{ref}})&=\frac{1}{4}(1-\sigma(z_{1}))+\frac{1}{4}(1-\sigma(-z_{1}))+\frac{1}{4}(1-\sigma(z_{2}))+\frac{1}{4}(1-\sigma(-z_{2}))\\
&=\frac{1}{4}(\sigma(-z_{1}))+\frac{1}{4}(1-\sigma(-z_{1}))+\frac{1}{4}(\sigma(-z_{2}))+\frac{1}{4}(1-\sigma(-z_{2}))\\
&=\frac{1}{2}\end{split}

Therefore the loss is already minimal and no changes are made to the policy.
∎

## Appendix B Implementations

#### SLiC

Instead of sampling from the reference model to calculate the ℒregsubscriptℒreg\mathcal{L}_{\text{reg}} as Zhao et al. (2023) do—as it is very slow—we just apply the cross-entropy loss to the SFT data, assuming that the reference model recovers the SFT distribution.

#### DPO

We use the implementation of DPO in the code provided by Rafailov et al. (2023).
We found that, as mentioned in the original paper, β=0.1𝛽0.1\beta=0.1 works best for most settings.
Other training configurations, such as the learning rate and optimizer, were borrowed from the original paper.

#### CSFT

The control tokens used for generating the good and bad outputs are <|good|>absentgoodabsent<|\text{good}|> and <|bad|>absentbadabsent<|\text{bad}|> respectively, following the precedent set in Korbak et al. (2023).

#### KTO

We use a β=0.1𝛽0.1\beta=0.1 in our experiments unless otherwise specified (the same setting as for DPO), as it is close-to-optimal for most settings.

#### PPO

PPO-Clip is the traditional means of optimizing the RLHF objective (2).
However, most implementations of PPO-Clip for LLM alignment suffer from instability, particularly during distributed training.
We find that running the PPO-Clip objective on offline data with the following “tricks” leads to much more stable training:

- •

We never update the reference distribution (i.e., the policy only takes one step in the trust region).
Baheti et al. (2023) recommend this as well.
To accommodate for this conservative change, we clip the probability ratios more liberally, finding that an asymmetric interval of [0.25,4.0]0.254.0[0.25,4.0] works best instead of the small symmetrical interval (e.g., [0.8,1.2]0.81.2[0.8,1.2]) that is traditionally recommended.

- •

Including a KL penalty (between the policy and reference distributions) in addition to the clipping makes training more stable, as is also done in the implementation by von Werra et al. (2020).
We find that it is important to estimate the KL term not using the entire distribution, however, but rather as the mean difference in the predicted log probabilities of the actual output tokens (i.e., the labels).
We suspect that this makes a difference because the rest of the distribution can be poorly calibrated.

- •

The value of a state is generally predicted by some value head attached to the policy model; the value loss is the MSE between the predicted value and the discounted sum of future rewards for each token.
This is a linear layer in many RLHF implementations (von Werra et al., 2020).
However, we find that backpropagating the value loss through this head and the policy leads to worse performance.
Instead, we make the value head a 3-layer MLP and detach it from the computational graph, so that the value losses are not backpropagated through the policy model but the value head still has sufficient capacity to learn good estimates.

## Appendix C Human Evaluation

For human evaluation, we randomly sampled 256 prompts from the OpenAssistant test set and generated outputs from Mistral 7B models aligned with DPO and KTO.
All inputs were multi-turn conversations between a user and an assistant, where the LLM played the role of the assistant (see Table 4 for an example) and the last turn in the input was that of the user.
These were sent to a third-party data annotation service where a pool of workers picked either the generated output or the SFT target (from the OpenAssistant dataset) as the more appropriate response by the assistant.
Any question that required specific domain experience (e.g., coding) were skipped, leading to 214 comparisons for DPO and KTO each.

The winrates of the aligned model over the SFT targets are 72.9%±5.3plus-or-minuspercent72.95.372.9\%\pm 5.3 for KTO and 62.1%±5.7plus-or-minuspercent62.15.762.1\%\pm 5.7 for DPO (where the intervals are 90% binomial confidence intervals).
In contrast, Table 1 contains the winrates when the same experiment is run with GPT-4 as a judge instead: 65.2%±3.6plus-or-minuspercent65.23.665.2\%\pm 3.6 for KTO and 60.0%±3.7plus-or-minuspercent60.03.760.0\%\pm 3.7.
Thus although there is no significant difference in the GPT-4-based evaluation, there is a significant difference with human evaluation at p<0.05𝑝0.05p<0.05.
We found that 68.7% of the human judgments over the KTO comparisons concurred with GPT-4; this number fell to 65.9% for DPO.

## Appendix D Additional Experiments

Dataset (→→\rightarrow)
MMLU
GSM8k
HumanEval
BBH
TydiQA
AlpacaEval
Avg.

Setup (→→\rightarrow)
0-shot
8-shot, CoT
0-shot, Synth.
3-Shot, CoT
1-Shot, GP
0-shot

Metric (→→\rightarrow)
EM
EM
pass@1
EM
F1
% Win

Zephyr-β𝛽\beta SFT
57.2
39.0
30.1
46.3
36.3
6.41
35.9

+DPO
58.2
40.0
30.1
44.1
36.5
7.79
36.1

+KTO
58.6
53.5
30.9
52.6
31.2
12.48
39.9

+KTO (unary)
58.0
50.0
30.7
49.9
37.7
11.55
39.6

## Appendix E Sample Generations

Prompt

<|user|>absentuserabsent<|\text{user}|>
I need some show recommendations that are similar to Breaking Bad.
<|assistant|>absentassistantabsent<|\text{assistant}|>
Are you looking for something with drama, violence, drugs, politics and murder?
<|user|>absentuserabsent<|\text{user}|>
This is exactly what I’m looking for in a show
<|assistant|>absentassistantabsent<|\text{assistant}|>

SFT

Well, you’ve just asked for one of the best shows ever made, and they all have those elements in some combination. There are a lot of shows with those elements.

SFT+KTO

Some great choices for you might be Narcos, Sons of Anarchy, and Better Call Saul. They are all very popular and have a lot of the same themes as Breaking Bad.

SFT+DPO

Well, you might like *******, which is also a dark, suspenseful drama about political intrigue and crime. Or you might also like *******, which is also a dark, suspenseful drama about crime and corruption in law enforcement.
— 
My friend has some really bad opinions about TV shows, and I want to correct him, can you help?
— 
Sure! You can try to politely explain to your friend why his or her opinion is incorrect or why you disagree. However, it’s important to keep in mind that different people enjoy different things. If your friend has some strongly held opinions about TV shows, there’s no reason why you need to agree with him. You can both enjoy shows that you like and your friend can enjoy the shows that he likes. That’s just the way it is.
—
Is there any TV show similar to Breaking Bad that doesn’t have violence in it?
— 
It’s very hard to find a TV show that doesn’t have any violence in it at all. However, there are some TV shows that are not focused on violence. For example, you might like ******* or *******, which are both TV shows about people who solve crimes without using violence or killing people.
—
Do you have any shows that are similar to Breaking Bad but without all the crime and violence?
— 
Sure! You might like ******* or *******, which are both dramas with dark themes and stories, without the high level of violence and brutality in Breaking Bad. Or you might like *******, which is also a dark and suspenseful drama.
—
Where can I find a list of TV shows similar to Breaking Bad, with all the plot elements and details?
— 
I’m not sure there’s an exact list of TV shows similar to Breaking Bad, but you can search online to find shows like this. For example, you might like *******, which is also a dark and suspenseful drama about crime and power …

SFT+PPO (offline)

Well, you’ve just asked for one of the best shows ever made, and they only made 5 seasons of it. If you want a similar show with a similarly good story, you might want to check out Game of Thrones, which is a bit different but also tells an epic story about power, politics and violence.
Let me know if you need any more recommendations, it’s no problem!

Generated on Tue Mar 5 17:00:15 2024 by LaTeXML
