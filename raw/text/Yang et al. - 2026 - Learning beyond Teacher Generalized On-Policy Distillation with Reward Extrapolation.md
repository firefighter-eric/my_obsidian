# Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation

- Source HTML: `raw/html/Yang et al. - 2026 - Learning beyond Teacher Generalized On-Policy Distillation with Reward Extrapolation.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2602.12125
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Learning beyond Teacher: Generalized On-Policy Distillation
with Reward Extrapolation

Wenkai Yang1, , Weijie Liu2, Ruobing Xie2, Kai Yang2, 
Saiyong Yang2, Yankai Lin1,
1Gaoling School of Artificial Intelligence, Renmin University of China 
2LLM Department, Tencent
🖂 {wenkaiyang,yankailin}@ruc.edu.cn

Work done during an internship at Tencent.Corresponding author.

###### Abstract

On-policy distillation (OPD), which aligns the student with the teacher’s logit distribution on student-generated trajectories, has demonstrated strong empirical gains in improving student performance and often outperforms off-policy distillation and reinforcement learning (RL) paradigms. In this work, we first theoretically show that OPD is a special case of dense KL-constrained RL where the reward function and the KL regularization are always weighted equally and the reference model can by any model. Then, we propose the Generalized On-Policy Distillation (G-OPD) framework, which extends the standard OPD objective by introducing a flexible reference model and a reward scaling factor that controls the relative weight of the reward term against the KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we derive two novel insights: (1) Setting the reward scaling factor to be greater than 1 (i.e., reward extrapolation), which we term ExOPD, consistently improves over standard OPD across a range of teacher-student size pairings. In particular, in the setting where we merge the knowledge from different domain experts, obtained by applying domain-specific RL to the same student model, back into the original student, ExOPD enables the student to even surpass the teacher’s performance boundary and outperform the domain teachers. (2) Building on ExOPD, we further find that in the strong-to-weak distillation setting (i.e., distilling a smaller student from a larger teacher), performing reward correction by choosing the reference model as the teacher’s base model before RL yields a more accurate reward signal and further improves distillation performance. However, this choice assumes access to the teacher’s pre-RL variant and incurs more computational overhead. We hope our work offers new insights for future research on OPD.111Code is available at https://github.com/RUCBM/G-OPD.

(a) The empirical effectiveness of our method ExOPD compared with off-policy distillation (SFT), standard OPD, and the weight-extrapolation method ExPO (Zheng et al., 2025) in multi-teacher and strong-to-weak distillation settings (results averaged over 4 math reasoning and 3 code generation benchmarks). (a) When merging multiple domain experts—obtained by applying domain-specific RL to the same base model—back into the original base model, ExOPD is the only method that yields a unified student that consistently outperforms all domain teachers. (b) ExOPD also yields significant improvements over standard OPD when distilling a smaller student from a larger teacher. Moreover, applying reward correction in ExOPD can further boost distillation performance (Figure 14(a)).

## 1 Introduction

Recently, on-policy distillation (OPD) (Agarwal et al., 2024; Yang et al., 2025a; Lu and Lab, 2025) has emerged as an effective post-training paradigm for improving capabilities of Large Language Models (LLMs). Unlike prior off-policy distillation methods (Taori et al., 2023; Guha et al., 2025) that train the student on teacher-generated trajectories, OPD allows the student to learn from the teacher’s supervision (i.e., predicted logits) on student-generated tokens. Previous studies have shown that OPD can not only serve as a promising multi-task post-training paradigm to (near-)losslessly merge the capabilities acquired by different RL variants across domains back into the original base model (Xiao et al., 2026), but also be effective and efficient in distilling the capabilities of a larger teacher into a smaller student (Gu et al., 2024; Yang et al., 2025a).

Despite its empirical effectiveness, a mechanistic understanding of OPD remains limited in the field, leaving its full potential under-explored. In this work, we bridge this gap by establishing a theoretical connection between OPD and dense reinforcement learning (RL), and by extending standard OPD into a generalized formulation.

First, we make derivations to show that OPD is essentially a special case of the standard dense RL with Kullback–Leibler (KL) constraint, where the token-level reward function is always weighted equally with the KL regularization and the reference model can be chosen arbitrarily. Building on this insight, we generalize the OPD objective to a more universal formulation by further introducing a reward scaling factor that controls the relative weight of the reward term against the KL regularization, in addition to the flexible reference model. We refer to this generalized formulation as the Generalized On-Policy Distillation (G-OPD) framework.

Based on the G-OPD framework, we theoretically analyze how the reward scaling factor and the choice of reference model affect distillation effectiveness across different settings, supported by comprehensive experiments in both math reasoning and code generation domains.
In the first setting, the teacher is obtained by applying domain-specific RL to the student, and the reference model is naturally fixed to the student’s initial state. We show that (1) when the reward scaling factor lies in (0,1)(0,1) (i.e., reward interpolation), the distilled student exhibits behaviors (e.g., performance and response length) that fall between the reference and teacher models; (2) when the reward scaling factor is greater than 11 (i.e., reward extrapolation), the student can learn beyond the teacher’s capability boundary and outperform teacher in domain tasks. We refer to the reward extrapolation variant as ExOPD. We further show that ExOPD extends well to the multi-teacher distillation setting, enabling a unified student to surpass all domain teachers. Second, we study the strong-to-weak distillation setting, where a smaller student is distilled from a larger teacher. In this setting, we demonstrate that replacing the reference model from the student’s initial policy to the teacher’s pre-RL variant (i.e., reward correction) in ExOPD yields a more accurate reward signal and further improves distillation performance. However, the limitations of this practice are that it assumes access to an additional model (the teacher’s pre-RL variant) and incurs more computational cost on computing the log-probabilities of the larger reference model. Despite these limitations, ExOPD and ExOPD with reward correction significantly outperform standard OPD in the strong-to-weak distillation setting.

## 2 Related Work

Off-Policy Distillation. Knowledge distillation (KD) (Hinton et al., 2015) is a widely used technique for transferring knowledge from a domain expert (teacher) to a student model.
Most prior studies focus on off-policy distillation, where the student is trained on trajectories generated by the teacher, either by aligning the student’s logits distribution with the teacher’s via a Kullback–Leibler (KL) divergence loss on token logits (Sanh et al., 2019; Kim and Rush, 2016; Guo et al., 2025b), or by directly performing supervised fine-tuning (SFT) with a cross-entropy loss on the teacher-generated tokens (Taori et al., 2023; Zhou et al., 2023; Guha et al., 2025). This practice has been shown to effectively improve the student model across a broad range of capabilities (Ding et al., 2023; Yang et al., 2025e; Ye et al., 2025b) in the LLM era.

On-Policy Distillation.
By sampling trajectories from the student and aligning the student with the teacher’s logit distribution on each token of these student-generated trajectories, on-policy distillation (OPD) (Agarwal et al., 2024; Gu et al., 2024) realizes dense on-policy learning.
Empirically, OPD has been shown to achieve faster and more effective distillation than off-policy distillation (Yang et al., 2025a; Lu and Lab, 2025). Recent OPD studies have explored distillation across different model families (Patiño et al., 2025), developed black-box on-policy distillation methods that do not require access to the teacher’s logits (Ye et al., 2025a), and investigated the self-distillation paradigm that leverage the LLM’s in-context capabilities to distill textual context information into its parameters (Yang et al., 2025c; Hübotter et al., 2026; Shenfeld et al., 2026; Zhao et al., 2026; Penaloza et al., 2026).

## 3 Methodology

### 3.1 Preliminaries

In this section, we start with a brief review of relevant preliminaries.

Off-Policy Distillation.
Let DD denote the input distribution, and let π𝜽\pi_{\bm{\theta}} and π∗\pi^{*} denote the student and teacher policies, respectively. The general form of Knowledge Distillation (KD) (Hinton et al., 2015) can be written as

𝒥KD(𝜽)=min𝜽𝔼𝒙∼D,𝒚∼π∗(⋅|𝒙)[𝒟KL(π∗(𝒚|𝒙)∥π𝜽(𝒚|𝒙))],\mathcal{J}_{\text{KD}}(\bm{\theta})=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi^{*}(\cdot|\bm{x})}\Big[\mathcal{D}_{\mathrm{KL}}\!\big(\pi^{*}(\bm{y}|\bm{x})\,\big\|\,\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big)\Big],

(1)

where 𝒟KL\mathcal{D}_{\mathrm{KL}} denotes the Kullback–Leibler (KL) divergence loss.
In the era of LLMs, obtaining the teacher’s full output distribution (e.g., logits) is often expensive or even infeasible. As a result, KD is commonly implemented as supervised fine-tuning (SFT) of the student on trajectories generated by the teacher. Though effective, the major drawback of this paradigm is its off-policy nature: the student is trained to imitate the teacher’s behavior, rather than to learn from reward signals induced by its own actions. As a result, it may fail to adapt and generalize from its own experience at test time, when faced with similar problems.

On-Policy RL. We use π𝜽\pi_{\bm{\theta}} to denote the policy model to be optimized. The RL objective can be formulated as

𝒥RL​(𝜽)=max𝜽⁡𝔼𝒙∼D,𝒚∼π𝜽(⋅|x)​[r​(𝒙,𝒚)−β​𝒟KL​(π𝜽∥πref)].\mathcal{J}_{\text{RL}}(\bm{\theta})=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|x)}\Big[r(\bm{x},\bm{y})-\beta\mathcal{D}_{\mathrm{KL}}(\pi_{\bm{\theta}}\,\|\,\pi_{\mathrm{ref}})\Big].

(2)

In the above formulation, the trajectories yy are sampled from the current policy model, making the training remain on-policy. r​(𝒙,𝒚)r(\bm{x},\bm{y}) is the reward function that measures the quality of a response sequence 𝒚=(y1,⋯,yT)\bm{y}=(y_{1},\cdots,y_{T}) to a query 𝒙\bm{x}. Depending on the setting, it can be either (i) a parameterized neural reward model trained on the specific preference data for open-domain alignment (Cai et al., 2024; Dong et al., 2024; Liu et al., 2025a), or (ii) a rule-based, deterministic outcome verifier commonly used in verifiable LLM reasoning tasks (Guo et al., 2025a; Hu et al., 2025; Liu and Zhang, 2025; Yang et al., 2025b). 𝒟KL​(π𝜽∥πref)\mathcal{D}_{\mathrm{KL}}(\pi_{\bm{\theta}}\,\|\,\pi_{\mathrm{ref}}) prevents the policy model π𝜽\pi_{\bm{\theta}} from drifting too far from a reference model πref\pi_{\mathrm{ref}}, and the coefficient β\beta controls the strength of this constraint. To solve Eq. (2), a common approach is to apply policy gradient (Sutton et al., 1998), updating the policy parameters using an estimated gradient of the form

∇𝜽𝒥RL​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1TAt​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)],\nabla_{\bm{\theta}}\mathcal{J}_{\text{RL}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\;\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}A_{t}\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big],

(3)

where AtA_{t} is the relative advantage of token yty_{t} over a baseline value. In practice, the reward signal in RL is often sparse: the policy model only receives a reward at the final token after the response is completed, which may make optimization inefficient and ineffective (Cui et al., 2025).

On-Policy Distillation. On-Policy Distillation (OPD) (Agarwal et al., 2024; Gu et al., 2024; Lu and Lab, 2025) inherits the on-policy nature of policy training and the advantage of dense credit assignment, making it an efficient post-training paradigm (Yang et al., 2025a; Xiao et al., 2026). The main idea of OPD is to let the student generate its own trajectories, and then minimize the reverse KL divergence between the student and the teacher π∗\pi^{*} on those student-generated trajectories:

𝒥OPD(𝜽)=min𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[𝒟KL(π𝜽(𝒚|𝒙)∥π∗(𝒚|𝒙))].\mathcal{J}_{\text{OPD}}(\bm{\theta})=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\mathcal{D}_{\mathrm{KL}}\!\Big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi^{*}(\bm{y}|\bm{x})\Big)\Big].

(4)

Notice that in Eq. (4), the trajectories 𝒚\bm{y} are generated by the policy model itself, resulting in the on-policy training. Also, we can get the gradient of OPD as222Detailed derivations are in Appendix A.

∇𝜽𝒥OPD​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1T(∑t′=tT(log⁡π𝜽​(yt′|𝒙,𝒚<t′)−log⁡π∗​(yt′|𝒙,𝒚<t′)))​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)].\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}\Big(\sum_{t^{{}^{\prime}}=t}^{T}\big(\log\pi_{\bm{\theta}}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})-\log\pi^{*}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})\big)\Big)\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big].

(5)

In practice, current studies (Lu and Lab, 2025; Xiao et al., 2026) use a discount factor of 0 (focus on next-token optimization only) and approximate the gradient as

∇𝜽𝒥OPD​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1T(log⁡π𝜽​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)].\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}\big(\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big)\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big].

(6)

Comparing Eq. (6) with Eq. (3), we can see that −(log⁡π𝜽​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))-\big(\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big) can be regarded as the token-level advantage in OPD, thereby providing dense credit assignment for each token-level action.

### 3.2 Generalized On-Policy Distillation

In this section, we first start from Eq. (4) and derive a generalized formulation of OPD.

First, we re-formulate the OPD objective (Xiao et al., 2025) as

𝒥OPD​(𝜽)\displaystyle\mathcal{J}_{\text{OPD}}(\bm{\theta})
=min𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[𝒟KL(π𝜽(𝒚|𝒙)∥π∗(𝒚|𝒙))]\displaystyle=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi^{*}(\bm{y}|\bm{x})\big)\Big]

(7)

=min𝜽⁡𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙)]\displaystyle=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\Big]

=max𝜽⁡𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[log⁡π∗​(𝒚|𝒙)−log⁡π𝜽​(𝒚|𝒙)]\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\log\pi^{*}(\bm{y}|\bm{x})-\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big]

=max𝜽⁡𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[(log⁡π∗​(𝒚|𝒙)−log⁡πref​(𝒚|𝒙))−(log⁡π𝜽​(𝒚|𝒙)−log⁡πref​(𝒚|𝒙))]\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\big(\log\pi^{*}(\bm{y}|\bm{x})-\log\pi_{\mathrm{ref}}(\bm{y}|\bm{x})\big)-\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi_{\mathrm{ref}}(\bm{y}|\bm{x})\big)\Big]

=max𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[logπ∗​(𝒚|𝒙)πref​(𝒚|𝒙)−𝒟KL(π𝜽(𝒚|𝒙)∥πref(𝒚|𝒙))].\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\log\frac{\pi^{*}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})}-\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi_{\mathrm{ref}}(\bm{y}|\bm{x})\big)\Big].

Therefore, we have the following remark:

From the above remark, we establish the connection between OPD and RL. However, we emphasize that OPD differs from standard RL in the following key respects:

(1) Dense rewards. As discussed above, in standard RL the model typically receives an effective reward only at the final token, while the rewards for all other tokens are zero:

rtR​L={0t=1,⋯,T−1,Outcome Rewardt=T.r_{t}^{RL}=\begin{cases}0&t=1,\cdots,T-1,\\
\text{Outcome Reward}&t=T.\end{cases}

(8)

However, in OPD, each token-level action receives an effective reward

rtO​P​D=log⁡π∗​(yt|𝒙,𝒚<t)πref​(yt|𝒙,𝒚<t),t=1,⋯,T.r_{t}^{OPD}=\log\frac{\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})}{\pi_{\mathrm{ref}}(y_{t}|\bm{x},\bm{y}_{<t})},\quad t=1,\cdots,T.

(9)

This token-level reward takes essentially the same form as the implicit reward defined in Rafailov et al. (2023). Implicit reward is initially derived from the closed-form solution of Eq. (2), which can be written as

r​(𝒙,𝒚)=β​log⁡π𝜽​(𝒚|𝒙)πref​(𝒚|𝒙)+β​log⁡Z​(𝒙), where ​Z​(𝒙)=∑𝒚πref​(𝒚|𝒙)​exp⁡(1β​r​(𝒙,𝒚)).r(\bm{x},\bm{y})=\beta\log\frac{\pi_{\bm{\theta}}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})}+\beta\log Z(\bm{x}),\text{ where }Z(\bm{x})=\sum_{\bm{y}}\pi_{\text{ref}}(\bm{y}|\bm{x})\exp(\frac{1}{\beta}r(\bm{x},\bm{y})).

(10)

As we can see, since log⁡Z​(𝒙)\log Z(\bm{x}) is a constant depending only on 𝒙\bm{x}, log⁡π𝜽​(𝒚|𝒙)πref​(𝒚|𝒙)\log\frac{\pi_{\bm{\theta}}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})} can be regarded as a well-defined proxy of the true reasoning reward, and this idea is adopted in previous studies (Yuan et al., 2024; Cui et al., 2025; Yang et al., 2025d; Liu et al., 2025c) to provide dense supervision for RL. However, in OPD, the implicit reward
log⁡π∗​(𝒚|𝒙)πref​(𝒚|𝒙)\log\frac{\pi^{*}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})}
does not require π∗\pi^{*} to be obtained by applying RL starting from πref\pi_{\mathrm{ref}}. In fact, π∗\pi^{*} and πref\pi_{\mathrm{ref}} can even be models of different sizes. Nevertheless, this reward function still captures the log-probability shift from the reference (πref\pi_{\mathrm{ref}}) distribution to the expert (π∗\pi^{*}) distribution, and thus provides a meaningful training signal.

(2) Fixed weighting between the reward function and the KL regularization. As revealed in the remark, in OPD, the reward term and the KL regularization are always weighted equally. In what follows, we present and discuss our generalized OPD formulation by introducing a reward scaling factor that allows us to adjust the relative weight of the reward term against the KL regularization.

(3) Flexible choice of the reference model. In RL (i.e., Eq. (2)), the reference model is typically initialized as the policy model’s starting checkpoint. However, we note that in OPD (i.e., Eq. (11)), the introduced reference model can be any model, since this choice does not affect the final simplification of the objective back to its original form in Eq. (4). In what follows, we discuss how different choices of πref\pi_{\mathrm{ref}} affect our proposed generalized OPD framework. By default, the reference model is selected as the student’s initial policy.

From the above discussion, we can see that OPD offers two key advantages over RL—dense reward signals and a flexible choice of reference model—yet it fixes the relative weighting between the reward function and the KL regularization to 1:11:1. This motivates us to follow Eq. (2) and generalize the original OPD objective in Eq. (4) into a general dense RL objective with a flexible KL constraint, by introducing both a third reference model and an additional reward scaling factor λ\lambda:

𝒥G-OPD​(𝜽)\displaystyle\mathcal{J}_{\text{G-OPD}}(\bm{\theta})
=max𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[λlogπ∗​(𝒚|𝒙)πref​(𝒚|𝒙)−𝒟KL(π𝜽(𝒚|𝒙)∥πref(𝒚|𝒙))].\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\lambda}\log\frac{\pi^{*}(\bm{y}|\bm{x})}{{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\mathrm{ref}}}(\bm{y}|\bm{x})}-\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\pi_{\mathrm{ref}}}(\bm{y}|\bm{x})\big)\Big].

(11)

The above Eq. (11) presents our Generalized On-Policy Distillation (G-OPD) formulation, where λ\lambda controls the relative weight of the reward term against the KL regularization in the objective, and is essential 1β\frac{1}{\beta} in Eq. (2). As we can see, compared to RL, G-OPD enables dense credit assignment and a more flexible choice of reference model; compared to OPD, it further allows more general control over the reward weight. In the following, we discuss in detail about the two crucial components, λ\lambda and πref\pi_{\mathrm{ref}}, in G-OPD.

##### Reward interpolation and extrapolation in G-OPD.

The optimal solution to G-OPD in Eq. (11) satisfies that

log⁡π𝜽​(𝒚|𝒙)\displaystyle\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})
=λ​log⁡π∗​(𝒚|𝒙)+(1−λ)​log⁡πref​(𝒚|𝒙)\displaystyle=\lambda\log\pi^{*}(\bm{y}|\bm{x})+(1-\lambda)\log\pi_{\mathrm{ref}}(\bm{y}|\bm{x})

(12)

=log⁡π∗​(𝒚|𝒙)+(λ−1)​(log⁡π∗​(𝒚|𝒙)−log⁡πref​(𝒚|𝒙)).\displaystyle=\log\pi^{*}(\bm{y}|\bm{x})+(\lambda-1)(\log\pi^{*}(\bm{y}|\bm{x})-\log\pi_{\mathrm{ref}}(\bm{y}|\bm{x})).

This reveals that, (1) when 0<λ<10<\lambda<1, G-OPD encourages the student model’s log-probability distribution to match a linear interpolation between that of the teacher and reference models. This can also be interpreted as replacing the reward rr in Eq. (7) with λ⋅r+(1−λ)⋅0\lambda\cdot r+(1-\lambda)\cdot 0. Therefore, we refer to this case as reward interpolation. We conjecture that, under this setting, the student trained with G-OPD may exhibit behavior (e.g., performance, response length, etc.) that lies between the reference model and the standard OPD with λ=1\lambda=1. (2) When λ>1\lambda>1, G-OPD encourages the student’s log-probability distribution to go beyond matching the teacher’s log-probabilities by additionally fitting an extra shift term (λ−1)​(log⁡π∗−log⁡πref)(\lambda-1)(\log\pi^{*}-\log\pi_{\mathrm{ref}}). From the perspective of rewards, G-OPD with λ>1\lambda>1 performs an extrapolation of the reward function’s weight in Eq. (7); thus, we refer to this regime as reward extrapolation. We wonder whether reward extrapolation can outperform standard OPD, and in a special case, when the teachers are domain experts obtained by applying RL to the same student (Xiao et al., 2026) in different domains, can reward extrapolation in G-OPD distill a unified student that surpasses all the domain teachers?

##### Reward correction in strong-to-weak distillation.

When the reward scaling factor λ≠1\lambda\neq 1, different choices of the reference model πref\pi_{\mathrm{ref}} in Eq. (11) lead to different objectives. Based on distillation settings, in the following, we discuss the choices of πref\pi_{\mathrm{ref}} in two cases: (1) One application of G-OPD is to merge the capabilities of several experts, each obtained by applying domain-specific RL starting from the same base model, back into the original base model (Xiao et al., 2026). In this setting, πref\pi_{\mathrm{ref}} is naturally chosen as the original base model, and the reward function in G-OPD is exactly the implicit reward defined in Eq. (10). (2) Another distillation setting is strong-to-weak distillation (Yang et al., 2025a), i.e., distilling a large teacher into a smaller student. In this case, πref\pi_{\mathrm{ref}} admits two choices: (i) the student’s base model, πbasestudent\pi_{\mathrm{base}}^{\mathrm{student}}, which corresponds to the default setting where we only have access to π∗\pi^{*} and πbasestudent\pi_{\mathrm{base}}^{\mathrm{student}}; or (ii) the teacher expert’s pre-RL base model, πbaseteacher\pi_{\mathrm{base}}^{\mathrm{teacher}} (i.e., the teacher before post-training), assuming it is available. To compare these two choices, we first rewrite the G-OPD objective into an equivalent form:

𝒥G-OPD​(𝜽)\displaystyle\mathcal{J}_{\text{G-OPD}}(\bm{\theta})
=max𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[λlogπ∗​(𝒚|𝒙)πref​(𝒚|𝒙)−𝒟KL(π𝜽(𝒚|𝒙)∥πref(𝒚|𝒙))]\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\lambda\log\frac{\pi^{*}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})}-\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi_{\mathrm{ref}}(\bm{y}|\bm{x})\big)\Big]

(13)

=max𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[(λ−1)logπ∗​(𝒚|𝒙)πref​(𝒚|𝒙)−𝒟KL(π𝜽(𝒚|𝒙)∥π∗(𝒚|𝒙))].\displaystyle=\max_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[(\lambda-1)\log\frac{\pi^{*}(\bm{y}|\bm{x})}{\pi_{\mathrm{ref}}(\bm{y}|\bm{x})}-\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi^{*}(\bm{y}|\bm{x})\big)\Big].

Now, under the same KL regularization strength, we can see that choosing πref=πbaseteacher\pi_{\mathrm{ref}}=\pi_{\mathrm{base}}^{\mathrm{teacher}} is more reasonable. The reason is that the reward log⁡π∗πbaseteacher\log\frac{\pi^{*}}{\pi_{\mathrm{base}}^{\mathrm{teacher}}} corresponds to the implicit reward induced by the teacher’s RL post-training, and is thus well-defined according to Eq. (10). In contrast, log⁡π∗πbasestudent\log\frac{\pi^{*}}{\pi_{\mathrm{base}}^{\mathrm{student}}} can be noisier, since there exists fundamental gap between the internal knowledge and capacity of teacher and student base models. Therefore, in the strong-to-weak distillation setting, we think that applying a reward correction to the default reward log⁡π∗πbasestudent\log\frac{\pi^{*}}{\pi_{\mathrm{base}}^{\mathrm{student}}}—by adding log⁡πbasestudentπbaseteacher\log\frac{\pi_{\mathrm{base}}^{\mathrm{student}}}{\pi_{\mathrm{base}}^{\mathrm{teacher}}} to obtain log⁡π∗πbaseteacher\log\frac{\pi^{*}}{\pi_{\mathrm{base}}^{\mathrm{teacher}}}—can lead to better distillation performance. The limitations, however, are that this requires access to πbaseteacher\pi_{\mathrm{base}}^{\mathrm{teacher}} and incurs additional computation, since computing log⁡πbaseteacher\log\pi_{\mathrm{base}}^{\mathrm{teacher}} requires more cost than computing log⁡πbasestudent\log\pi_{\mathrm{base}}^{\mathrm{student}}.

Finally, the approximated gradient of G-OPD can be written as

∇𝜽𝒥G-OPD​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1TAtG-OPD​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)],\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{G-OPD}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}A_{t}^{\text{G-OPD}}\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big],

(14)

where AtG-OPD=(log⁡π𝜽​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))+(λ−1)​(log⁡πref​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))A_{t}^{\text{G-OPD}}=\big(\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big)+(\lambda-1)\big(\log\pi_{\text{ref}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big).

## 4 Experiments and Analysis

In this section, we conduct a series of extensive experiments on math reasoning and code generation tasks to analyze the properties of the proposed G-OPD framework and assess the effectiveness of ExOPD. We begin with preliminary experiments on same-size teacher-student pairs in Section 4.1.2, where we investigate the impact of the reward scaling factor within G-OPD. We then explore the effectiveness of ExOPD in the multi-teacher distillation setting in Section 4.1.3. Finally, we present experimental results in the strong-to-weak distillation setting in Section 4.2.

### 4.1 Experiments with Same-Sized Student and Teacher

Here, we consider the scenario where the domain teachers are reinforced models derived from the student through domain-specific RL.

#### 4.1.1 Experimental Settings

Base Model. We primarily conduct experiments using the Qwen3-4B-Non-Thinking (Yang et al., 2025a) model. The student model is initialized as Qwen3-4B-Non-Thinking, while the domain teachers are derived by applying RL separately to Qwen3-4B-Non-Thinking on domain-specific data.

Training Datasets. We filter the DeepMath (He et al., 2025) dataset to select 57K samples with a difficulty level greater than or equal to 6 to form the math RL data, and use Eurus-RL-Code (Cui et al., 2025) as the code RL data, which consists of 25K samples. We then apply RL to the base model on two datasets separately to get domain teachers, Qwen3-4B-Non-Thinking-RL-Math and Qwen3-4B-Non-Thinking-RL-Code. The distillation data is the same as the RL data.

Training Settings. We apply Group Relative Policy Optimization (GRPO) (Shao et al., 2024) to obtain domain teachers. In RL, a reward of 1.0 is given when the final answer is correct in math reasoning or when all unit tests pass in code generation; otherwise, the reward is 0.0. Detailed training hyper-parameters in GRPO are in Appendix B. After this, we implement G-OPD on the original student model (i.e., Qwen3-4B-Non-Thinking) with different reward scaling factors λ∈{0.0,0.25,0.5,0.75,1.0,1.25,1.5}\lambda\in\{0.0,0.25,0.5,0.75,1.0,1.25,1.5\}. Note that λ=0.0\lambda=0.0 corresponds to the initial state Qwen3-4B-Non-Thinking, and λ=1.0\lambda=1.0 corresponds to standard OPD. The reference model here is fixed naturally as Qwen3-4B-Non-Thinking. Detailed training hyper-parameters in G-OPD are in Appendix B. In both GRPO and G-OPD, we implement token-level rollout correction (Liu et al., 2025b) to mitigate training-inference mismatch. Our experiments are based on verl (Sheng et al., 2024) framework.

Evaluation. For the evaluation of math reasoning, we select four competition-level benchmarks: AIME24 (AI-MO, 2024), AIME25 (OpenCompass, 2025), HMMT25 (February) (Balunović et al., 2025), and HMMT25 (November) (Balunović et al., 2025). For the evaluation of code generation, we select three test sets: HumanEval+, MBPP+ (Liu et al., 2023), and LiveCodeBench (v6 only, February 2025∼\simMay 2025) (Jain et al., 2024). In all evaluations, we set the temperature to 1.0, top-p to 1.0, and the maximum generation length to 16,384.
On each math reasoning benchmark, we sample 32 solutions for each problem; whereas each code generation benchmark, we sample 4 solutions per problem. We then report the average accuracy of each model on each benchmark. We adopt Math-Verify333https://github.com/huggingface/Math-Verify as a rule-based verifier to validate answer correctness for math reasoning benchmarks.

(a) Trends in the average number of tokens and the average accuracy of the on-policy distilled models across different benchmarks under varying reward scaling factors. The teacher for math reasoning tasks is Qwen3-4B-Non-Thinking-RL-Math, while the teacher for code generation tasks is Qwen3-4B-Non-Thinking-RL-Code.

#### 4.1.2 Results of Single-Teacher Distillation

We first explore the impact of reward scaling factor λ\lambda in G-OPD in the same-sized single-teacher distillation setting as the preliminary experiments (i.e., distilling Qwen3-4B-Non-Thinking-RL-Math or Qwen3-4B-Non-Thinking-RL-Code back into Qwen3-4B-Non-Thinking). The evaluation results in math reasoning and code generation domains are in Figure 3 and Figure 3 respectively. We also visualize the relationship between accuracy and response length of each model in Figure 9(a) for deep analysis.

We can draw the following conclusions: (1) Standard OPD can fully recover the post-training behavior. As we can see, the student produced by OPD closely matches the evaluation accuracy and response length of the domain teacher. (2) Reward interpolation (0<λ<10<\lambda<1) produces a student whose behavior (performance and response length) lies between the base model and the teacher model. Also, both the performance and response length increase monotonically as λ\lambda grows, approaching the behavior of the teacher. This property can be leveraged to achieve budget-controlled reasoning (Yang et al., 2025e; Liang et al., 2026). (3) Reward extrapolation (λ>1\lambda>1) outperforms standard OPD and has the potential to produce a student that surpasses the domain teacher. As observed, ExOPD with appropriate reward extrapolation (i.e., λ=1.25\lambda=1.25) consistently outperforms OPD and the domain teacher in all settings (also see Table 2), while excessive reward extrapolation (i.e., λ=1.5\lambda=1.5) may lead to instability and degrade performance. This can be explained by the fact that continuously increasing λ\lambda introduces the risk of the student hacking the implicit reward in Eq. (9), by aggressively fitting the peak of the log ratio, even if some tokens have excessively large log ratios due to bias. Furthermore, we can see that the response lengths of the students produced by ExOPD continue to increase, which may be due to the length bias issue of the implicit reward (Yang et al., 2025d).

To demonstrate that the improvement of ExOPD over the teacher is not due to less training of the teacher, we compare the evaluation performance of ExOPD and the teacher after an additional 100 steps of RL training. The results in Table 1 show that the teacher with more continued RL training show smaller improvement compared to ExOPD with fewer steps. We also demonstrate the generalizability and effectiveness of ExOPD when the teacher models are trained with sufficient RL steps (i.e., 1200 RL steps), with the corresponding results provided in Appendix C.

Method

AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)

00Avg.00

Teacher

58.0
54.6
32.5
38.9
46.0

+ continued RL (100 steps)
60.9+2.960.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.9}}}
55.6+0.555.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.5}}}
32.8+0.332.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
38.4−0.538.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.5}}}
46.9+0.946.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.9}}}

ExOPD (50 steps)
62.7+4.7\textbf{62.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+4.7}}}
56.1+1.5\textbf{56.1}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.5}}}
33.9+1.4\textbf{33.9}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}
39.3+0.4\textbf{39.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
48.0+2.0\textbf{48.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.0}}}

#### 4.1.3 Results of Multi-Teacher Distillation

Based on above analysis, we conduct experiments in the multi-teacher distillation setting, where we aim to merge the capabilities from different domain teachers, obtained by applying domain-specific RL to the same base model, into the original base model through OPD (Xiao et al., 2026). This has been demonstrated to be an effective new multi-task post-training paradigm. Specifically, the domain teachers are the above RL variants Qwen3-4B-Non-Thinking-RL-Math/Code, and the student model is Qwen3-4B-Non-Thinking. From the preliminary results in Section 4.1.2, we can see that λ=1.25\lambda=1.25 in ExOPD consistently leads to better performance than OPD. Thus, in all subsequent experiments, we fix λ=1.25\lambda=1.25 for ExOPD without any further specific tuning.
Besides OPD, we also compare against two baselines: (1) Supervised fine-tuning (SFT), which trains the student on the teachers’ generated trajectories via Cross-Entropy Loss. We ensure that the number of trajectories used for SFT is consistent with those in OPD and ExOPD. More details can be found in Appendix B. (2) ExPO (Zheng et al., 2025), a weight extrapolation method. We implement ExPO by first averaging the weights of all domain teachers, then extrapolating the weights against the student model using an extrapolation factor α\alpha, which is tuned from {0.25,0.5}\{0.25,0.5\} following the recommendations. For a fair comparison, we downweight the sample size of the math RL data to match that of the code RL data in both OPD and ExOPD here, ensuring that each domain has the same sample size.

Method

Math Reasoning
Code Generation

AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)

00Avg.00

HumanEval+

0MBPP+0

00LCB00

00Avg.00

Teacher

58.0
54.6
32.5
38.9
46.0
86.0
70.2
27.3
61.2

Student

21.5
21.9
10.0

08.0
15.4
74.7
64.7
17.9
52.4

Single-Teacher Distillation

ExPO
58.7+0.758.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.7}}}
55.2+0.655.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.6}}}
32.4−0.132.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.1}}}
37.0−1.937.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.9}}}
45.8−0.245.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.2}}}
84.8−1.284.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.2}}}
70.2+0.070.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
28.0+0.728.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.7}}}
61.0−0.261.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.2}}}

OPD
60.7+2.760.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.7}}}
55.0+0.455.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
32.4−0.132.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.1}}}
37.9−1.037.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.0}}}
46.5+0.546.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.5}}}
85.2−0.885.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.8}}}
69.9−0.369.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.3}}}
27.3+0.027.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
60.8−0.360.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.3}}}

ExOPD
62.7+4.7\textbf{62.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+4.7}}}
56.1+1.5\textbf{56.1}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.5}}}
33.9+1.4\textbf{33.9}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}
39.3+0.4\textbf{39.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
48.0+2.0\textbf{48.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.0}}}
86.9+0.9\textbf{86.9}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.9}}}
70.7+0.5\textbf{70.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.5}}}
28.6+1.3\textbf{28.6}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.3}}}
62.1+0.9\textbf{62.1}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.9}}}

Multi-Teacher Distillation

SFT
58.5+0.558.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.5}}}
53.3−1.353.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.3}}}
30.7−1.830.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.8}}}
34.8−4.134.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-4.1}}}
44.3−1.744.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.7}}}
86.4+0.486.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
69.6−0.669.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.6}}}
26.4−0.926.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.9}}}
60.8−0.460.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.4}}}

ExPO
57.5−0.557.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.5}}}
54.5−0.154.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.1}}}
31.7−0.831.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.8}}}
36.3−2.636.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-2.6}}}
45.0−1.045.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.0}}}
86.7+0.7\textbf{86.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.7}}}
72.0+1.8\textbf{72.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.8}}}
29.0+1.7\textbf{29.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.7}}}
62.6+1.4\textbf{62.6}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}

OPD
60.6+2.660.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.6}}}
54.1−0.554.1\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.5}}}
32.5+0.032.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
38.3−0.638.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.6}}}
46.4+0.446.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
84.6−1.484.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.4}}}
69.5−0.769.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.7}}}
27.6+0.327.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
60.6−0.660.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.6}}}

ExOPD
61.0+3.0\textbf{61.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+3.0}}}
56.0+1.4\textbf{56.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}
34.4+1.9\textbf{34.4}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.9}}}
39.2+0.3\textbf{39.2}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
47.7+1.7\textbf{47.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.7}}}
86.3+0.386.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
70.6+0.470.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
29.0+1.7\textbf{29.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.7}}}
62.0+0.862.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.8}}}

The results of multi-teacher distillation are shown in Table 2. As we can see, SFT produces a sub-optimal student, while the performance ceiling of OPD is typically bounded by the teachers. ExPO, though training-free, cannot ensure that the weight-extrapolated student consistently surpasses all domain teachers, lacking good controllability.
However, our method ExOPD consistently outperforms OPD and is the only method that produces a unified student capable of surpassing both domain teachers on all benchmarks.

Furthermore, we analyze the training dynamics of ExOPD compared to OPD to gain a deeper understanding of ExOPD. We put the comparison in Figure 12(a). ExOPD achieves higher training rewards but makes the student generate longer response lengths, which is consistent with the evaluation results shown in Figure 9(a). We also observe that the response entropy of the student trained by ExOPD is higher than that trained by OPD. We attribute this to the fact that the former tends to generate longer responses, increasing the response diversity.

### 4.2 Experiments in the Strong-to-Weak Distillation Setting

Another practical usage of OPD is for strong-to-weak distillation (Yang et al., 2025a), i.e., distilling capabilities from a larger teacher into a smaller student. Thus, in this section, we explore the effectiveness of ExOPD and the additional reward correction practice in the strong-to-weak distillation setting.

#### 4.2.1 Experimental Settings

We select Qwen3-30B-A3B-Instruct-2507 as the teacher model and perform distillation on Qwen3-1.7B-Non-Thinking and Qwen3-4B-Non-Thinking, respectively. We primarily conduct experiments in the math reasoning domain, where the training and evaluation datasets are the same as those used in Section 4.1. The training details are in Appendix B. In ExOPD, we first conduct experiments in the default setting (Section 4.2.2), where we assume the availability of only two models: the student base model and the stronger teacher model. Thus, in this default setting, we set the reference model in ExOPD to the student base model. We also explore the effectiveness of the reward correction technique in Section 4.2.3, where we assume extra access to the teacher’s pre-RL variant, which serves as the reference model in ExOPD. We compare ExOPD against standard OPD and off-policy distillation (SFT).

Method

0AIME240

0AIME250

HMMT25 (Feb.)
HMMT25 (Nov.)

00Avg.00

Teacher
74.7
62.8
44.2
57.2
59.7

Student: Qwen3-1.7B-Non-Thinking

Base
12.3
11.4

06.8

04.5

08.8

SFT
18.1
20.5

09.2

06.3
13.5

OPD
33.0
28.7
15.7
14.9
23.1

ExOPD
37.3+4.3\textbf{37.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+4.3}}}
31.5+2.8\textbf{31.5}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.8}}}
16.2+0.5\textbf{16.2}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.5}}}
16.5+1.6\textbf{16.5}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.6}}}
25.4+2.3\textbf{25.4}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.3}}}

Student: Qwen3-4B-Non-Thinking

Base
21.5
21.9
10.0

08.0
15.4

SFT
45.4
40.9
22.4
31.6
35.1

OPD
55.0
48.0
29.8
37.7
42.6

ExOPD
58.7+3.7\textbf{58.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+3.7}}}
50.8+2.8\textbf{50.8}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.8}}}
33.0+3.2\textbf{33.0}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+3.2}}}
38.8+1.1\textbf{38.8}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.1}}}
45.3+2.7\textbf{45.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+2.7}}}

#### 4.2.2 Results of Strong-to-Weak Distillation

The results in default strong-to-weak distillation setting are put in Table 3. The main conclusion is that ExOPD can bring significant improvements in strong-to-weak distillation, outperforming off-policy distillation and standard OPD by a large margin. The results reveal that, although the implicit reward log⁡π∗πbasestudent\log\frac{\pi^{*}}{\pi_{\text{base}}^{\text{student}}} may contain noise due to the intrinsic knowledge gap and distribution bias between the small and large models, extrapolating the rewards can still push the limits of OPD in strong-to-weak distillation.

#### 4.2.3 Reward Correction in Strong-to-Weak Distillation

As shown above, the default ExOPD with the reference model fixed as the student base model can already bring significant improvement over OPD. However, as discussed in Remark 3.2, setting the reference model to the teacher’s pre-RL variant—if available—may further enhance the distillation performance. Here, we conduct experiments to validate this analysis. Specifically, since we cannot get the pre-RL variant of Qwen3-30B-A3B-Instruct-2507, we choose our trained Qwen3-4B-Non-Thinking-RL-Math/Code as the teachers and take Qwen3-4B-Non-Thinking as the pre-RL variant. The student model is Qwen3-1.7B-Non-Thinking.

The comparison results are displayed in Figure 14(a). The results validate the effectivenss of the reward correction practice, which consistently boosts the performance of ExOPD. However, we reiterate that reward correction requires access to πbaseteacher\pi_{\text{base}}^{\text{teacher}} and incurs higher computational cost, since it requires computing log-probabilities under a larger reference model than in the default ExOPD.

(a) Effect of reward correction in the strong-to-weak distillation setting.

## 5 Conclusion and Discussion

In this work, we conduct an in-depth analysis of the on-policy distillation paradigm. We first establish an interesting connection between OPD and dense KL-constrained RL. Building on this insight, we propose a generalized OPD framework (G-OPD) by introducing (i) a flexible reference model for the implicit reward function and (ii) a reward scaling factor that controls the relative weight of the reward term versus KL regularization. Through comprehensive experiments on math reasoning and code generation tasks, we provide several novel insights: (1) Appropriate reward extrapolation (i.e., setting the reward scaling factor to be larger than 1) can improve OPD performance, and in same-sized multi-teacher distillation it enables learning a unified student that surpasses all domain-specific teachers. We refer to this variant as ExOPD. (2) Moreover, in strong-to-weak distillation, replacing the student’s initial policy with the teacher’s pre-RL policy as the reference model can further boost the performance of ExOPD.

Regarding future work, we believe it is practical to explore: (1) validating the generalizability of ExOPD on larger-scale models; (2) assessing the robustness of ExOPD in multi-teacher distillation with a broader and more diverse set of domain teachers; and (3) evaluating the effectiveness of ExOPD for on-policy distillation across different model families.

## References

- R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem (2024)
On-policy distillation of language models: learning from self-generated mistakes.

In The twelfth international conference on learning representations,

Cited by: §1,
§2,
§3.1.

- AI-MO (2024)
AIME 2024.

Note: https://huggingface.co/datasets/AI-MO/aimo-validation-aime

Cited by: §4.1.1.

- M. Balunović, J. Dekoninck, I. Petrov, N. Jovanović, and M. Vechev (2025)
MathArena: evaluating llms on uncontaminated math competitions.

 SRI Lab, ETH Zurich.

External Links: Link

Cited by: §4.1.1.

- Z. Cai, M. Cao, H. Chen, K. Chen, K. Chen, X. Chen, X. Chen, Z. Chen, Z. Chen, P. Chu, et al. (2024)
Internlm2 technical report.

arXiv preprint arXiv:2403.17297.

Cited by: §3.1.

- G. Cui, L. Yuan, Z. Wang, H. Wang, W. Li, B. He, Y. Fan, T. Yu, Q. Xu, W. Chen, et al. (2025)
Process reinforcement through implicit rewards.

arXiv preprint arXiv:2502.01456.

Cited by: §3.1,
§3.2,
§4.1.1.

- N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou (2023)
Enhancing chat language models by scaling high-quality instructional conversations.

In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,

 pp. 3029–3051.

Cited by: §2.

- H. Dong, W. Xiong, B. Pang, H. Wang, H. Zhao, Y. Zhou, N. Jiang, D. Sahoo, C. Xiong, and T. Zhang (2024)
RLHF workflow: from reward modeling to online rlhf.

arXiv preprint arXiv:2405.07863.

Cited by: §3.1.

- Y. Gu, L. Dong, F. Wei, and M. Huang (2024)
MiniLLM: knowledge distillation of large language models.

In The Twelfth International Conference on Learning Representations,

External Links: Link

Cited by: §1,
§2,
§3.1.

- E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague, et al. (2025)
OpenThoughts: data recipes for reasoning models.

arXiv preprint arXiv:2506.04178.

Cited by: §1,
§2.

- D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. (2025a)
Deepseek-r1: incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948.

Cited by: §3.1.

- Y. Guo, W. Yang, Z. Sun, N. Ding, Z. Liu, and Y. Lin (2025b)
Learning to focus: causal attention distillation via gradient-guided token pruning.

arXiv preprint arXiv:2506.07851.

Cited by: §2.

- Z. He, T. Liang, J. Xu, Q. Liu, X. Chen, Y. Wang, L. Song, D. Yu, Z. Liang, W. Wang, et al. (2025)
Deepmath-103k: a large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning.

arXiv preprint arXiv:2504.11456.

Cited by: §4.1.1.

- G. Hinton, O. Vinyals, and J. Dean (2015)
Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531.

Cited by: §2,
§3.1.

- J. Hu, Y. Zhang, Q. Han, D. Jiang, X. Zhang, and H. Shum (2025)
Open-reasoner-zero: an open source approach to scaling up reinforcement learning on the base model.

arXiv preprint arXiv:2503.24290.

Cited by: §3.1.

- J. Hübotter, F. Lübeck, L. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K. Buening, C. Guestrin, et al. (2026)
Reinforcement learning via self-distillation.

arXiv preprint arXiv:2601.20802.

Cited by: §2.

- N. Jain, K. Han, A. Gu, W. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica (2024)
Livecodebench: holistic and contamination free evaluation of large language models for code.

arXiv preprint arXiv:2403.07974.

Cited by: §4.1.1.

- Y. Kim and A. M. Rush (2016)
Sequence-level knowledge distillation.

In Proceedings of the 2016 conference on empirical methods in natural language processing,

 pp. 1317–1327.

Cited by: §2.

- K. Liang, C. Bai, X. Xu, C. Tang, S. Lee, W. Liu, S. Yang, and Y. Wu (2026)
ORBIT: on-policy exploration-exploitation for controllable multi-budget reasoning.

arXiv preprint arXiv:2601.08310.

Cited by: §4.1.2.

- C. Y. Liu, L. Zeng, Y. Xiao, J. He, J. Liu, C. Wang, R. Yan, W. Shen, F. Zhang, J. Xu, et al. (2025a)
Skywork-reward-v2: scaling preference data curation via human-ai synergy.

arXiv preprint arXiv:2507.01352.

Cited by: §3.1.

- J. Liu, Y. Li, Y. Fu, J. Wang, Q. Liu, and Y. Shen (2025b)
External Links: Link

Cited by: §4.1.1.

- J. Liu, C. S. Xia, Y. Wang, and L. Zhang (2023)
Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation.

In Thirty-seventh Conference on Neural Information Processing Systems,

External Links: Link

Cited by: §4.1.1.

- J. Liu and L. Zhang (2025)
Code-r1: reproducing r1 for code with reliable rewards.

Note: https://github.com/ganler/code-r1

Cited by: §3.1.

- X. Liu, K. Wang, Y. Wu, F. Huang, Y. Li, J. Zhang, and J. Jiao (2025c)
Agentic reinforcement learning with implicit step rewards.

arXiv preprint arXiv:2509.19199.

Cited by: §3.2.

- K. Lu and T. M. Lab (2025)
On-policy distillation.

Thinking Machines Lab: Connectionism.

Note: https://thinkingmachines.ai/blog/on-policy-distillation

External Links: Document

Cited by: Appendix A,
§1,
§2,
§3.1,
§3.1.

- OpenCompass (2025)
AIME 2025.

Note: https://huggingface.co/datasets/opencompass/AIME2025

Cited by: §4.1.1.

- C. M. Patiño, K. Rasul, Q. Gallouédec, B. Burtenshaw, S. Paniego, V. Srivastav, T. Frere, E. Beeching, L. Tunstall, L. von Werra, and T. Wolf (2025)
Unlocking on-policy distillation for any model family.

Cited by: §2.

- E. Penaloza, D. Vattikonda, N. Gontier, A. Lacoste, L. Charlin, and M. Caccia (2026)
Privileged information distillation for language models.

arXiv preprint arXiv:2602.04942.

Cited by: §2.

- R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023)
Direct preference optimization: your language model is secretly a reward model.

Advances in neural information processing systems 36, pp. 53728–53741.

Cited by: §3.2.

- V. Sanh, L. Debut, J. Chaumond, and T. Wolf (2019)
DistilBERT, a distilled version of bert: smaller, faster, cheaper and lighter.

arXiv preprint arXiv:1910.01108.

Cited by: §2.

- Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)
Deepseekmath: pushing the limits of mathematical reasoning in open language models.

arXiv preprint arXiv:2402.03300.

Cited by: §4.1.1.

- I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal (2026)
Self-distillation enables continual learning.

arXiv preprint arXiv:2601.19897.

Cited by: §2.

- G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu (2024)
HybridFlow: a flexible and efficient rlhf framework.

arXiv preprint arXiv: 2409.19256.

Cited by: §4.1.1.

- R. S. Sutton, A. G. Barto, et al. (1998)
Reinforcement learning: an introduction.

Vol. 1, MIT press Cambridge.

Cited by: §3.1.

- R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto (2023)
Alpaca: a strong, replicable instruction-following model.

Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html 3 (6), pp. 7.

Cited by: §1,
§2.

- B. Xiao, B. Xia, B. Yang, B. Gao, B. Shen, C. Zhang, C. He, C. Lou, F. Luo, G. Wang, et al. (2026)
MiMo-v2-flash technical report.

arXiv preprint arXiv:2601.02780.

Cited by: Appendix A,
§1,
§3.1,
§3.1,
§3.2,
§3.2,
§4.1.3.

- T. Xiao, Y. Yuan, M. Li, Z. Chen, and V. G. Honavar (2025)
On a connection between imitation learning and RLHF.

In The Thirteenth International Conference on Learning Representations,

External Links: Link

Cited by: §3.2.

- A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. (2025a)
Qwen3 technical report.

arXiv preprint arXiv:2505.09388.

Cited by: §1,
§2,
§3.1,
§3.2,
§4.1.1,
§4.2.

- W. Yang, J. Chen, Y. Lin, and J. Wen (2025b)
Deepcritic: deliberate critique with large language models.

arXiv preprint arXiv:2505.00662.

Cited by: §3.1.

- W. Yang, Y. Lin, J. Zhou, and J. Wen (2025c)
Distilling rule-based knowledge into large language models.

In Proceedings of the 31st International Conference on Computational Linguistics,

 pp. 913–932.

Cited by: §2.

- W. Yang, W. Liu, R. Xie, Y. Guo, L. Wu, S. Yang, and Y. Lin (2025d)
Laser: reinforcement learning with last-token self-rewarding.

arXiv preprint arXiv:2510.14943.

Cited by: §3.2,
§4.1.2.

- W. Yang, S. Ma, Y. Lin, and F. Wei (2025e)
Towards thinking-optimal scaling of test-time compute for llm reasoning.

arXiv preprint arXiv:2502.18080.

Cited by: §2,
§4.1.2.

- T. Ye, L. Dong, Z. Chi, X. Wu, S. Huang, and F. Wei (2025a)
Black-box on-policy distillation of large language models.

arXiv preprint arXiv:2511.10643.

Cited by: §2.

- Y. Ye, Z. Huang, Y. Xiao, E. Chern, S. Xia, and P. Liu (2025b)
Limo: less is more for reasoning.

arXiv preprint arXiv:2502.03387.

Cited by: §2.

- L. Yuan, W. Li, H. Chen, G. Cui, N. Ding, K. Zhang, B. Zhou, Z. Liu, and H. Peng (2024)
Free process rewards without process labels.

arXiv preprint arXiv:2412.01981.

Cited by: §3.2.

- S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F. Chen, and A. Grover (2026)
Self-distilled reasoner: on-policy self-distillation for large language models.

arXiv preprint arXiv:2601.18734.

Cited by: §2.

- C. Zheng, Z. Wang, H. Ji, M. Huang, and N. Peng (2025)
Model extrapolation expedites alignment.

In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),

 pp. 1025–1041.

Cited by: 2(a),
§4.1.3.

- C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al. (2023)
Lima: less is more for alignment.

Advances in Neural Information Processing Systems 36, pp. 55006–55021.

Cited by: §2.

## Appendix A Detailed Math Derivations

Here, we make mathematical derivations to calculate the expected gradients of OPD objective in Eq. (4).

Since

𝒥OPD​(𝜽)\displaystyle\mathcal{J}_{\text{OPD}}(\bm{\theta})
=min𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)[𝒟KL(π𝜽(𝒚|𝒙)∥π∗(𝒚|𝒙))]\displaystyle=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\mathcal{D}_{\mathrm{KL}}\!\big(\pi_{\bm{\theta}}(\bm{y}|\bm{x})\,\big\|\,\pi^{*}(\bm{y}|\bm{x})\big)\Big]

(15)

=min𝜽⁡𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙)].\displaystyle=\min_{\bm{\theta}}\;\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\Big].

We can get

∇𝜽𝒥OPD​(𝜽)\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})
=∇𝜽𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙)]\displaystyle=\nabla_{\bm{\theta}}\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\Big]

(16)

=∇𝜽𝔼𝒙​[∑𝒚π𝜽​(𝒚|𝒙)​(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))]\displaystyle=\nabla_{\bm{\theta}}\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)\Big]

=𝔼𝒙​[∇𝜽​∑𝒚π𝜽​(𝒚|𝒙)​(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\nabla_{\bm{\theta}}\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)\Big]

=𝔼𝒙​[∑𝒚(∇𝜽π𝜽​(𝒚|𝒙))​(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))+∑𝒚π𝜽​(𝒚|𝒙)​∇𝜽log⁡π𝜽​(𝒚|𝒙)].\displaystyle=\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\big(\nabla_{\bm{\theta}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big)\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)+\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big].

Notice that

𝔼𝒙​[∑𝒚π𝜽​(𝒚|𝒙)​∇𝜽log⁡π𝜽​(𝒚|𝒙)]\displaystyle\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big]
=𝔼𝒙​[∑𝒚π𝜽​(𝒚|𝒙)​∇𝜽π𝜽​(𝒚|𝒙)π𝜽​(𝒚|𝒙)]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\frac{\nabla_{\bm{\theta}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})}{\pi_{\bm{\theta}}(\bm{y}|\bm{x})}\Big]

(17)

=𝔼𝒙​[∑𝒚∇𝜽π𝜽​(𝒚|𝒙)]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\nabla_{\bm{\theta}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big]

=𝔼𝒙​[∇𝜽​∑𝒚π𝜽​(𝒚|𝒙)]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\nabla_{\bm{\theta}}\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big]

=𝔼𝒙​[∇𝜽1]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\nabla_{\bm{\theta}}1\Big]

=0.\displaystyle=0.

Therefore, Eq. (16) can be reduced to

∇𝜽𝒥OPD​(𝜽)\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})
=𝔼𝒙​[∑𝒚∇𝜽π𝜽​(𝒚|𝒙)​(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\nabla_{\bm{\theta}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)\Big]

(18)

=𝔼𝒙​[∑𝒚π𝜽​(𝒚|𝒙)​∇𝜽log⁡π𝜽​(𝒚|𝒙)​(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))]\displaystyle=\mathbb{E}_{\bm{x}}\Big[\sum_{\bm{y}}\pi_{\bm{\theta}}(\bm{y}|\bm{x})\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)\Big]

=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[(log⁡π𝜽​(𝒚|𝒙)−log⁡π∗​(𝒚|𝒙))​∇𝜽log⁡π𝜽​(𝒚|𝒙)]\displaystyle=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\big(\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})-\log\pi^{*}(\bm{y}|\bm{x})\big)\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(\bm{y}|\bm{x})\Big]

=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1T∑t′=1T(log⁡π𝜽​(yt′|𝒙,𝒚<t′)−log⁡π∗​(yt′|𝒙,𝒚<t′))​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)].\displaystyle=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}\sum_{t^{{}^{\prime}}=1}^{T}\big(\log\pi_{\bm{\theta}}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})-\log\pi^{*}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})\big)\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big].

Now let’s denote

Δt′=(log⁡π𝜽​(yt′|𝒙,𝒚<t′)−log⁡π∗​(yt′|𝒙,𝒚<t′)),\Delta_{t^{{}^{\prime}}}=\big(\log\pi_{\bm{\theta}}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})-\log\pi^{*}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})\big),

and consider each term 𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[Δt′​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)]\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\Delta_{t^{{}^{\prime}}}\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big] where t′<tt^{{}^{\prime}}<t:

𝔼𝒙,𝒚​[Δt′​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)]\displaystyle\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big]
=𝔼𝒙,𝒚​[𝔼yt​[Δt′​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)|𝒙,𝒚<t]]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\mathbb{E}_{y_{t}}\Big[\Delta_{t^{{}^{\prime}}}\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\big|\bm{x},\bm{y}_{<t}\Big]\Big]

(19)

=𝔼𝒙,𝒚​[Δt′​𝔼yt​[∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)|𝒙,𝒚<t]]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\mathbb{E}_{y_{t}}\Big[\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\big|\bm{x},\bm{y}_{<t}\Big]\Big]

=𝔼𝒙,𝒚​[Δt′​𝔼yt∼π𝜽(⋅|𝒙,𝒚<t)​[∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)]]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\mathbb{E}_{y_{t}\sim\pi_{\bm{\theta}}(\cdot|\bm{x},\bm{y}_{<t})}\big[\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\big]\Big]

=𝔼𝒙,𝒚​[Δt′​∑yt∇𝜽π𝜽​(yt|𝒙,𝒚<t)]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\sum_{y_{t}}\nabla_{\bm{\theta}}\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big]

=𝔼𝒙,𝒚​[Δt′​∇𝜽​∑ytπ𝜽​(yt|𝒙,𝒚<t)]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\nabla_{\bm{\theta}}\sum_{y_{t}}\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big]

=𝔼𝒙,𝒚​[Δt′​∇𝜽1]\displaystyle=\mathbb{E}_{\bm{x},\bm{y}}\Big[\Delta_{t^{{}^{\prime}}}\nabla_{\bm{\theta}}1\Big]

=0.\displaystyle=0.

Therefore, Eq. (18) can be reduced to

∇𝜽𝒥OPD​(𝜽)\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})
=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1T(∑t′=tT(log⁡π𝜽​(yt′|𝒙,𝒚<t′)−log⁡π∗​(yt′|𝒙,𝒚<t′)))​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)].\displaystyle=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}\Big(\sum_{t^{{}^{\prime}}=t}^{T}\big(\log\pi_{\bm{\theta}}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})-\log\pi^{*}(y_{t^{{}^{\prime}}}|\bm{x},\bm{y}_{<t^{{}^{\prime}}})\big)\Big)\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big].

(20)

In practice, recent studies (Lu and Lab, 2025; Xiao et al., 2026) use a discount factor of 0 and approximate the gradient as

∇𝜽𝒥OPD​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1T(log⁡π𝜽​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)].\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{OPD}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}\big(\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big)\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big].

(21)

Similarly, the approximated gradient of G-OPD in Eq. (11) can be written as

∇𝜽𝒥G-OPD​(𝜽)=𝔼𝒙∼D,𝒚∼π𝜽(⋅|𝒙)​[∑t=1TAtG-OPD​∇𝜽log⁡π𝜽​(yt|𝒙,𝒚<t)],\displaystyle\nabla_{\bm{\theta}}\mathcal{J}_{\text{G-OPD}}(\bm{\theta})=\mathbb{E}_{\bm{x}\sim D,\bm{y}\sim\pi_{\bm{\theta}}(\cdot|\bm{x})}\Big[\sum_{t=1}^{T}A_{t}^{\text{G-OPD}}\,\nabla_{\bm{\theta}}\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})\Big],

(22)

where AtG-OPD=(log⁡π𝜽​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))+(λ−1)​(log⁡πref​(yt|𝒙,𝒚<t)−log⁡π∗​(yt|𝒙,𝒚<t))A_{t}^{\text{G-OPD}}=\big(\log\pi_{\bm{\theta}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big)+(\lambda-1)\big(\log\pi_{\text{ref}}(y_{t}|\bm{x},\bm{y}_{<t})-\log\pi^{*}(y_{t}|\bm{x},\bm{y}_{<t})\big).

## Appendix B Detailed Training Settings

The training hyper-parameters in math and code RL training are put in Table 5 and Table 5 respectively.

The training hyper-parameters in G-OPD in both domains are in Table 7. In preliminary experiments, we find that under the same prompt size ∗* rollout nn conditions, setting a larger prompt size leads to smoother convergence. The number of optimization steps for G-OPD in all experiments with same-size teacher-student pairs (Section 4.1) is set to 50, while it is set to 100 for experiments in the strong-to-weak distillation setting (Section 4.2). We find that further increasing the number of distillation steps may degrade generalization performance due to overfitting.

The training hyper-parameters in SFT are in Table 7. We make sure the number of trajectories to each problem generated by the teacher in SFT is consistent with that generated by the student in OPD and ExOPD. We keep the number of optimization steps consistent with the corresponding G-OPD experiment for fair comparison.

Table 4: Training hyper-parameters of GRPO in math RL.

Hyper-parameter
Value

Train Batch Size
128

Micro Batch Size
128

Rollout nn

8

Maximum Prompt Length
2048

Maximum Response Length
16,384

Temperature
1.0

Top-p
1.0

LR
1×10−61\times 10^{-6}

Optimization Steps
500

KL Coefficient
0.0

Table 5: Training hyper-parameters of GRPO in code RL.

Hyper-parameter
Value

Train Batch Size
128

Micro Batch Size
128

Rollout nn

8

Maximum Prompt Length
2048

Maximum Response Length
8192

Temperature
1.0

Top-p
1.0

LR
1×10−61\times 10^{-6}

Optimization Steps
300

KL Coefficient
0.0

Table 6: Training hyper-parameters of G-OPD in both math and code domains.

Hyper-parameter
Value

Batch Size
1024

Rollout nn

1

Maximum Prompt Length
2048

Maximum Response Length
16,384

Temperature
1.0

Top-p
1.0

LR
1×10−51\times 10^{-5}

Table 7: Training hyper-parameters of SFT in both math and code domains.

Hyper-parameter
Value

Batch Size
1024

Maximum Sequence Length
32,768

Warm-up Ratio
0.05

LR
1×10−51\times 10^{-5}

## Appendix C Results of Distillation from Domain Teachers with Sufficient RL Trainings

Here, we show the on-policy distillation results when the domain teachers are trained with sufficient RL steps (i.e., 1200 steps). The experimental settings are the same as that in the Section 4.1. The results in Table 8 demonstrate the generalizability and effectiveness of ExOPD in this case.

Method

Math Reasoning
Code Generation

AIME24
AIME25
HMMT25 (Feb.)
HMMT25 (Nov.)

00Avg.00

HumanEval+

0MBPP+0

00LCB00

00Avg.00

Teacher

68.2
59.3
37.3
42.9
51.9
88.9
72.5
28.0
63.1

Student

21.5
21.9
10.0

08.0
15.4
74.7
64.7
17.9
52.4

Single-Teacher Distillation

OPD
68.3+0.168.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.1}}}
58.7−0.658.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.6}}}
38.7+1.4\textbf{38.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}
41.2−1.741.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.7}}}
51.7−0.251.7\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.2}}}
89.3+0.489.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.4}}}
71.3−1.271.3\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.2}}}
28.0+0.028.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
62.9−0.262.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.2}}}

ExOPD
68.4+0.2\textbf{68.4}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.2}}}
59.2−0.1\textbf{59.2}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.1}}}
38.2+0.938.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.9}}}
42.8−0.1\textbf{42.8}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.1}}}
52.2+0.3\textbf{52.2}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
89.9+1.0\textbf{89.9}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.0}}}
73.7+1.2\textbf{73.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.2}}}
29.3+1.3\textbf{29.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.3}}}
64.3+1.2\textbf{64.3}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.2}}}

Multi-Teacher Distillation

OPD
68.2+0.068.2\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
60.2+0.9\textbf{60.2}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.9}}}
38.5+1.2\textbf{38.5}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.2}}}
40.8−2.140.8\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-2.1}}}
51.9+0.051.9\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.0}}}
86.4−2.586.4\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-2.5}}}
72.1−0.472.1\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.4}}}
27.6−0.427.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.4}}}
62.0−1.162.0\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-1.1}}}

ExOPD
70.1+1.9\textbf{70.1}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.9}}}
59.6+0.359.6\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.3}}}
37.5+0.237.5\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.2}}}
42.7−0.2\textbf{42.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0.6484375,0.0546875,0.3125}\definecolor[named]{pgfstrokecolor}{rgb}{0.6484375,0.0546875,0.3125}-0.2}}}
52.5+0.6\textbf{52.5}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.6}}}
89.5+0.6\textbf{89.5}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+0.6}}}
73.9+1.4\textbf{73.9}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.4}}}
29.7+1.7\textbf{29.7}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.7}}}
64.4+1.3\textbf{64.4}\mathrlap{{}_{\scriptscriptstyle{\color[rgb]{0,0.58984375,0}\definecolor[named]{pgfstrokecolor}{rgb}{0,0.58984375,0}+1.3}}}

## Appendix D Prompt Templates

We show the prompt templates used in our experiments in the end.

Conversion to HTML had a Fatal error and exited abruptly. This document may be truncated or damaged.

Generated on Thu Mar 5 16:05:43 2026 by LaTeXML
