# Alabdulmohsin et al. - 2023 - Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Alabdulmohsin et al. - 2023 - Getting ViT in Shape Scaling Laws for Compute-Optimal Model Design.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2305.13035
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

††

⋆Significant technical contributions.

# Getting ViT in Shape:
Scaling Laws for Compute-Optimal Model Design

Ibrahim Alabdulmohsin⋆,  Xiaohua Zhai⋆,  Alexander Kolesnikov,  Lucas Beyer⋆ 
Google DeepMind 
Zürich, Switzerland 
{ibomohsin,xzhai,akolesnikov,lbeyer}@google.com

###### Abstract

Scaling laws have been recently employed to derive compute-optimal model size (number of parameters) for a given compute duration. We advance and refine such methods to infer compute-optimal model shapes, such as width and depth, and successfully implement this in vision transformers. Our shape-optimized vision transformer, SoViT, achieves results competitive with models that exceed twice its size, despite being pre-trained with an equivalent amount of compute. For example, SoViT-400m/14 achieves 90.3% fine-tuning accuracy on ILSRCV2012, surpassing the much larger ViT-g/14 and approaching ViT-G/14 under identical settings, with also less than half the inference cost. We conduct a thorough evaluation across multiple tasks, such as image classification, captioning, VQA and zero-shot transfer, demonstrating the effectiveness of our model across a broad range of domains and identifying limitations. Overall, our findings challenge the prevailing approach of blindly scaling up vision models and pave a path for a more informed scaling.

## 1 Introduction

The de-facto approach for improving performance of vision and language models today is scale: large models are trained on more data for longer [64, 43, 24, 19, 80, 23, 13, 16]. Empirically, it has been observed that the benefit of scale often follows a predictable power law in which the performance f​(x)𝑓𝑥f(x) (e.g. error rate or log-perplexity) satisfies f​(x)∼β​x−c+ε∞similar-to𝑓𝑥𝛽superscript𝑥𝑐subscript𝜀f(x)\sim\beta x^{-c}+\varepsilon_{\infty} for some β,c>0𝛽𝑐0\beta,c>0 as one varies the scaling dimension x𝑥x (e.g. data or model size), if the remaining dimensions are not bottlenecks [34, 39, 27, 26, 3, 1]. Here, ε∞subscript𝜀\varepsilon_{\infty} is the irreducible loss.

However, the simple power-law relation becomes more complicated when compute is considered. In this case, power laws are observed only along the compute-optimal frontier. Otherwise, scaling up the model size for a fixed compute budget can deteriorate performance (see [39, 35] and Figure 4). Since one often has a fixed compute budget in mind (e.g. available hardware and time), one should pick the model size that maximizes performance subject to the compute budget constraint, which may imply not training until convergence. Indeed, this approach was used successfully in the recent Chinchilla [35] that outperformed its predecessor Gopher [55] despite being 4×4\times smaller in size.

Unfortunately, in both [39] and [35] among others, the “size” of a model is equated with its parameter count, with no special consideration for model “shape dimensions”, such as “depth” or “width”. The rationale behind this choice follows from the surprising observation that the transformer shape had little impact on its scaling behavior in language modeling (LM) when performance is measured upstream (e.g. using log-perplexity) [39, 32, 33]. Nevertheless, follow-up analysis suggests that shape plays a pivotal role in other domains, such as in machine translation [47] and also in language modeling for downstream performance [66], with recent works even advocating for extreme aspect ratios, such as a single wide attention layer [12].

In vision, in particular, much earlier works using convolutional neural networks (CNNs) pointed out that the parameter count is indeed a poor predictor of performance. For example, scaling all dimensions [64, 43, 5] in ResNets [29] is more effective than scaling a single dimension such as depth alone. In addition, scaling width [79] is often more effective than depth, especially for small models [36, 58, 75]. Hence, optimizing the “shape” of transformers seems worthwhile.

In this work, we present SoViT: a shape-optimized vision transformer [24] that matches the performance of much larger models despite being pre-trained with equal compute. It is derived from a recipe we introduce for optimizing the shape of neural architectures, such as their depth and width.
A principled approach for scaling multiple dimensions is advantageous because although one can scale dimensions via brute-force search, this requires extensive computation and often remains sub-optimal [64]. Our recipe allows us to extrapolate without having to conduct an extensive set of experiments. For example, after only 115 experments, we identify a scaling strategy in ViT for all three dimensions: width (internal representation), depth, and MLP size. For comparison, [35] requires over 400 experiments to optimize a single dimension (the parameter count) alone.

One major finding is that small vision models can perform on par with larger ones with the same compute if we optimize their shape. In language, recent works have demonstrated the value of scaled-down architectures, such as the Chinchilla model [35] discussed earlier — a 70B parameter model that outperforms the 280B-parameter Gopher [55] and 175B-parameter GPT3 [13] — as well as LLaMA with its 13B parameter variant outperforming GPT3 on most benchmarks [69]. By introducing SoViT, we establish this phenomenon in vision as well.

Figure 1 summarizes how the various shape dimensions are scaled in SoViT (see Section 3 for derivation). The MLP dimension is scaled faster than depth, which in turn is scaled faster than width. When summarized by their parameter count (rightmost plot), compute-optimal ViTs are smaller than was previously used. With this scaling strategy, we find the shape of a ViT for the compute-equivalent of ViT-g/14 [80] pretrained on 16B JFT images [63]. We call this 2.5×2.5\times smaller model SoViT-400m/14. It achieves 90.3% fine-tuning accuracy on ILSRCV2012 [22] and 82.2% zero-shot accuracy in the locked-image text tuning (LiT) setup [81]. We further evaluate SoViT-400m/14 on captioning, VQA and panoptic segmentation and highlight some results in Figure 2.

Statement of Contribution. In summary, our contribution is to:

- •

Introduce a new method for optimizing the shape of neural networks, such as their depth and width. Our technique expands and improves previous methods by optimizing multiple shape dimensions jointly while requiring significantly fewer experiments.

- •

Demonstrate the effectiveness of scaled-down architectures in vision. We optimize ViT for the compute-equivalent of ViT-g/14, leading to a smaller, faster model of equal quality.

- •

Present new qualitative insights for scaling vision transformers, such as on how to scale individual shape dimensions and how optimal ViT shapes vary across domains.

- •

Conduct extensive evaluation across tasks like image classification, image captioning, VQA, zero-shot classification and panoptic segmentation, identifying both gains and limitations.

## 2 Related Work

Optimizing training for compute has received a significant amount of attention in recent years, partly due to the financial and environmental costs of training large models [52, 55].
However, conflicting results are sometimes reported. For example, in language modeling, [39] argues that the model size should be scaled faster than the data size, implying it is compute optimal to “undertrain” large models. Similar conclusions are found in [47]. On the other hand, [35] argues that the model size should be scaled uniformly with the data size, and highlights that transformers were not trained long enough, leading to some recent efforts [69] “overtraining” their models instead. Our analysis for ViT in Section 4 agrees partially with the latter result.

Scaling the size of vision transformers has led to remarkable results achieving, for instance, 90.4% top-1 accuracy on ImageNet (ILSRCV2012) with 2 billion parameters [80] and 90.9% top-1 accuracy with 4 billion parameters [15]. When scaled to 22 billion parameters, ViT exhibits state-of-the-art alignment to human visual perception in terms of shape/texture bias, among other findings [21].

Despite the clear benefit of scale, there has been little investigation into optimally scaling the shape of ViTs. [66] suggest preferentially increasing depth before scaling other dimensions uniformly. For ViT, however, they only consider small ViT-S and ViT-B models and the reported accuracy improvement comes with an increase in FLOPs of up to ×4absent4\times 4, making it difficult to draw conclusions about the suggested shape’s quality. In contrast [12] recommend scaling width over depth, but the authors do not observe any improvement when applying their strategy to ViT.

Our analysis draws inspiration from “compound scaling” in MobileNet [36] and EfficientNet [64], while differing in significant ways. EfficientNet uses an exhaustive grid search to determine the optimal architecture for a fixed increase in compute (e.g. ×2absent2\times 2). Afterwards, each dimension is scaled up by the same ratio with every subsequent increase in compute. In contrast, we expand scaling laws to simultaneously account for model size and compute beyond the efficient frontier and leverage them to derive the optimal scaling exponents for each dimension separately, as outlined in Section 3.

Throughout our analysis, we use downstream metrics, e.g. ImageNet 10-shot error, when measuring performance instead of upstream metrics. This follows recent reports arguing that upstream performance may not reflect downstream performance in language and vision [65, 80].

We use GFLOPs as a proxy for compute since it is hardware-agnostic and correlates well with actual wall-clock core-hours (see Figure 4). However, GFLOPs can have limitations [5, 20] and may not be a perfect predictor for the metric of interest (e.g. core hours) in all model and hardware types. Note that we focus on scaling the shape of the architecture, not on improving its training protocol, which can be similarly beneficial [5, 67, 62, 68].

## 3 Scaling Strategy

Notation. We begin with a formal description of the problem. We represent a neural architecture as a tuple 𝐱=(𝐱1,𝐱2,…,𝐱D)∈ℕD𝐱subscript𝐱1subscript𝐱2…subscript𝐱𝐷superscriptℕ𝐷\mathbf{x}=(\mathbf{x}_{1},\mathbf{x}_{2},\ldots,\mathbf{x}_{D})\in\mathbb{N}^{D} containing D𝐷D shape dimensions, such as width, depth and MLP size. We denote compute such as GFLOPs by 𝐭𝐭\mathbf{t}. We designate f:ℕD×ℝ+→ℝ:𝑓→superscriptℕ𝐷superscriptℝℝf:\mathbb{N}^{D}\times\mathbb{R}^{+}\to\mathbb{R} a performance metric of interest, such as downstream ImageNet 10-shot error rate. Specifically, f​(𝐱,𝐭)𝑓𝐱𝐭f(\mathbf{x},\mathbf{t}) results from (pre)-training an architecture 𝐱𝐱\mathbf{x} for a fixed compute budget 𝐭𝐭\mathbf{t}. We always assume that f𝑓f corresponds to a loss, meaning lower values are better.

The goal of optimizing shape for fixed compute 𝐭𝐭\mathbf{t} is to identify 𝐱⋆superscript𝐱⋆\mathbf{x}^{\star} (depending on 𝐭𝐭\mathbf{t}) such that:

f​(𝐱⋆,𝐭)−infx∈ℕDf​(x,𝐭)≤ϵ,𝑓superscript𝐱⋆𝐭subscriptinfimum𝑥superscriptℕ𝐷𝑓𝑥𝐭italic-ϵf(\mathbf{x}^{\star},\mathbf{t})-\inf_{x\in\mathbb{N}^{D}}f(x,\mathbf{t})\;\leq\;\epsilon,

(1)

for some small tolerance ϵ>0italic-ϵ0\epsilon>0.
Due to modeling assumptions, approximations, and the finite possible number of experiments conducted, we cannot hope for ϵ=0italic-ϵ0\epsilon=0 and have to tolerate a small excess loss.

Single Dimension. As demonstrated in Figure 3, the shape of a pretrained vision transformer has an impact on its downstream performance. To determine an optimal shape scaling strategy, we begin by considering both compute 𝐭𝐭\mathbf{t} and a single shape dimension 𝐱ksubscript𝐱𝑘\mathbf{x}_{k} for k∈[D]𝑘delimited-[]𝐷k\in[D], such as depth. In prior works, optimizing a single dimension 𝐱ksubscript𝐱𝑘\mathbf{x}_{k} for compute involves running a large number of experiments in order to identify the Pareto optimal frontier, from which power laws on 𝐱ksubscript𝐱𝑘\mathbf{x}_{k} or 𝐭𝐭\mathbf{t} are derived [39, 35]. Since this is expensive, we propose the following joint functional form instead:

fk​(𝐱k,𝐭)∼αk​𝐱k−ak+(βk​𝐱kbk+ξk)​𝐭−c+εk,similar-tosubscript𝑓𝑘subscript𝐱𝑘𝐭subscript𝛼𝑘superscriptsubscript𝐱𝑘subscript𝑎𝑘subscript𝛽𝑘superscriptsubscript𝐱𝑘subscript𝑏𝑘subscript𝜉𝑘superscript𝐭𝑐subscript𝜀𝑘f_{k}(\mathbf{x}_{k},\,\mathbf{t})\sim\alpha_{k}\mathbf{x}_{k}^{-a_{k}}+(\beta_{k}\mathbf{x}_{k}^{b_{k}}+\xi_{k})\,\mathbf{t}^{-c}+\varepsilon_{k},

(2)

where αk,ak,βk,bk,c,ξk,εk>0subscript𝛼𝑘subscript𝑎𝑘subscript𝛽𝑘subscript𝑏𝑘𝑐subscript𝜉𝑘subscript𝜀𝑘0\alpha_{k},a_{k},\beta_{k},b_{k},c,\xi_{k},\varepsilon_{k}>0. Here, fksubscript𝑓𝑘f_{k} focuses on the dimension k𝑘k alone and assumes that all other shape dimensions j≠k𝑗𝑘j\neq k are sufficiently large such that they do not constitute a bottleneck. We also assume that data is unlimited so that there is no risk of overfitting. We estimate the parameters in (2) by minimizing the relative error. In (2), aksubscript𝑎𝑘a_{k} are scaling exponents when varying the corresponding shape dimension in the compute-unbounded regime, c𝑐c is the data scaling exponent, while bksubscript𝑏𝑘b_{k} relates to the impact of the model shape on compute.

Our argument for this particular functional form is six-fold:

- I.

If compute is unbounded, we recover the familiar power law relation on model size fk​(𝐱k)∼αk​𝐱k−ak+εksimilar-tosubscript𝑓𝑘subscript𝐱𝑘subscript𝛼𝑘superscriptsubscript𝐱𝑘subscript𝑎𝑘subscript𝜀𝑘f_{k}(\mathbf{x}_{k})\sim\alpha_{k}\mathbf{x}_{k}^{-a_{k}}+\varepsilon_{k} [34, 2, 38, 39]. In addition, increasing the model size xksubscript𝑥𝑘x_{k} while keep the data size fixed does not imply that fk​(𝐱k,𝐭)→εk→subscript𝑓𝑘subscript𝐱𝑘𝐭subscript𝜀𝑘f_{k}(\mathbf{x}_{k},\,\mathbf{t})\to\varepsilon_{k} because 𝐱kbsuperscriptsubscript𝐱𝑘𝑏\mathbf{x}_{k}^{b} can increase faster than 𝐭csuperscript𝐭𝑐\mathbf{t}^{c} in (2).

- II.

For any fixed model size, the relation above reduces to the power law fk​(𝐭)∼A​𝐭−c+Bsimilar-tosubscript𝑓𝑘𝐭𝐴superscript𝐭𝑐𝐵f_{k}(\mathbf{t})\sim A\mathbf{t}^{-c}+B, where A=βk​𝐱kbk+ξk𝐴subscript𝛽𝑘superscriptsubscript𝐱𝑘subscript𝑏𝑘subscript𝜉𝑘A=\beta_{k}\mathbf{x}_{k}^{b_{k}}+\xi_{k} and B=αk​𝐱k−ak+εk𝐵subscript𝛼𝑘superscriptsubscript𝐱𝑘subscript𝑎𝑘subscript𝜀𝑘B=\alpha_{k}\mathbf{x}_{k}^{-a_{k}}+\varepsilon_{k}. Since the model size is fixed, 𝐭𝐭\mathbf{t} is proportional to the size of the data. Such data scaling laws have been demonstrated extensively in various domains [1, 2, 3, 27, 34, 39, 59, 80].

- III.

For fixed compute, the relation w.r.t. 𝐱ksubscript𝐱𝑘\mathbf{x}_{k} is non-monotone, quasiconvex (see Appendix A), in agreement with empirical measurements [39, 35]. See IsoFlop curves in Figure 4.

- IV.

Arguments for power law behavior using space partitioning suggest that the exponent c𝑐c is independent of the shape dimension. In particular, c=Θ​(1/d)𝑐Θ1𝑑c=\Theta(1/d), where d𝑑d is the intrinsic dimension of the data manifold [2, 38, 59]. From this, we conclude that assuming the functional form in (2) for every shape dimension separately cannot lead to any contradictions since this assumption is satisfied by the decomposable loss:

f​(𝐱,𝐭)=∑kαk​𝐱k−ak+∑kβk​𝐱kbk​𝐭−c+ξ​𝐭−c+ε∞,𝑓𝐱𝐭subscript𝑘subscript𝛼𝑘superscriptsubscript𝐱𝑘subscript𝑎𝑘subscript𝑘subscript𝛽𝑘superscriptsubscript𝐱𝑘subscript𝑏𝑘superscript𝐭𝑐𝜉superscript𝐭𝑐subscript𝜀f(\mathbf{x},\mathbf{t})=\sum_{k}\alpha_{k}\mathbf{x}_{k}^{-a_{k}}+\sum_{k}\beta_{k}\mathbf{x}_{k}^{b_{k}}\mathbf{t}^{-c}+\xi\mathbf{t}^{-c}+\varepsilon_{\infty},

(3)

for some constants ξ,ε∞>0𝜉subscript𝜀0\xi,\varepsilon_{\infty}>0.

- V.

When optimizing the shape dimension 𝐱ksubscript𝐱𝑘\mathbf{x}_{k} for fixed compute 𝐭𝐭\mathbf{t}, the optimal value 𝐱k⋆superscriptsubscript𝐱𝑘⋆\mathbf{x}_{k}^{\star} is:

𝐱k⋆=(αk​ak​𝐭cβk​bk)1bk+ak=O​(𝐭sk),where: ​sk=cbk+ak.formulae-sequencesuperscriptsubscript𝐱𝑘⋆superscriptsubscript𝛼𝑘subscript𝑎𝑘superscript𝐭𝑐subscript𝛽𝑘subscript𝑏𝑘1subscript𝑏𝑘subscript𝑎𝑘𝑂superscript𝐭subscript𝑠𝑘where: subscript𝑠𝑘𝑐subscript𝑏𝑘subscript𝑎𝑘\mathbf{x}_{k}^{\star}=\left(\frac{\alpha_{k}\,a_{k}\,\mathbf{t}^{c}}{\beta_{k}b_{k}}\right)^{\frac{1}{b_{k}+a_{k}}}=O\left(\mathbf{t}^{s_{k}}\right),\quad\text{where: }s_{k}=\frac{c}{b_{k}+a_{k}}.

(4)

Recall that the scaling exponent sksubscript𝑠𝑘s_{k} in (4) is positive because ak,bk,c>0subscript𝑎𝑘subscript𝑏𝑘𝑐0a_{k},b_{k},c>0. Using the relation (4), we rearrange the terms in Eq. (2), and obtain the scaling law for model performance along the compute-optimal frontier (Appendix A):

fk​(𝐱k,t)=F​𝐱k−ak+G​𝐭−c+εk,(in the compute-optimal frontier)subscript𝑓𝑘subscript𝐱𝑘𝑡𝐹superscriptsubscript𝐱𝑘subscript𝑎𝑘𝐺superscript𝐭𝑐subscript𝜀𝑘(in the compute-optimal frontier)f_{k}(\mathbf{x}_{k},t)=F\mathbf{x}_{k}^{-a_{k}}+G\mathbf{t}^{-c}+\varepsilon_{k},\quad\quad\text{(in the compute-optimal frontier)}

(5)

for some constants F𝐹F and G𝐺G, which is a sum of power law terms involving the model size and compute. Indeed, this decomposition has been demonstrated to hold within the compute-optimal frontier by [39] and [35].

- VI.

Eq. (2) fits empirical measurements and extrapolates accurately as well, see Figure 4.

Multiple Dimensions. Next, we expand upon the previous approach by incorporating multiple dimensions. To reiterate, our method involves both a functional form (2) and a novel procedure. Our procedure significantly decreases the number of large-scale experiments required to identify compute-optimal architectures, by an order of magnitude compared to prior work [35].

Star Sweep – Conducting a brute-force grid search to estimate scaling parameters across all dimensions is expensive, since it requires O​(2D)𝑂superscript2𝐷O(2^{D}) experiments to cover the search space. Instead, we demonstrate that a “star sweep” is sufficient: (1) starting from a large model 𝐱(c)superscript𝐱𝑐\mathbf{x}^{(c)} (the star center), we vary a single dimension k∈[D]𝑘delimited-[]𝐷k\in[D] at a time in an exponentially-spaced grid, such that all values are much smaller than 𝐱k(c)subscriptsuperscript𝐱𝑐𝑘\mathbf{x}^{(c)}_{k}. In our experiments, for instance, we optimize three shape parameters: width, depth, and MLP dim (see Section 4 for a brief definition of each dimension). Our star center is 𝐱(c)=(1968, 40, 6144)superscript𝐱𝑐1968406144\mathbf{x}^{(c)}=(1968,\,40,\,6144); i.e. has width 1968, depth 40, and MLP dim 6144. When varying MLP dim in the star sweep, we use the grid (1088, 1360, 1728, 2160, 2592, 3072)108813601728216025923072(1088,\,1360,\,1728,\,2160,\,2592,\,3072), corresponding to about 20% increase in each step, while fixing width to 1968 and depth to 40. We do this to ensure that other dimensions do not form a bottleneck when estimating the parameters in (2). This gives us the scaling exponents sksubscript𝑠𝑘s_{k} in (4).

Grid Sweep – The second stage is a grid sweep for small models trained for short compute. Depending on the number of shape dimensions involved, the cost of running this grid sweep can be negligible. Its goal is to identify a single architecture 𝐱(0)superscript𝐱0\mathbf{x}^{(0)} that lies in the Pareto optimal frontier for small compute as illustrated in Figure 3. This is important since a suboptimal 𝐱(0)superscript𝐱0\mathbf{x}^{(0)} can significantly skew results [5]. Our grid sweep identifies 𝐱(0)superscript𝐱0\mathbf{x}^{(0)} to be (608, 10, 928)60810928(608,\,10,\,928), the blue star in Figure 3. The advantage of this step is to absorb the leading coefficients in 𝐱k⋆=O​(𝐭sk)superscriptsubscript𝐱𝑘⋆𝑂superscript𝐭subscript𝑠𝑘\mathbf{x}_{k}^{\star}=O(\mathbf{t}^{s_{k}}) in (4) so that the star sweep focuses on estimating the exponents sksubscript𝑠𝑘s_{k} alone. We demonstrate in Figure 5 that the scaling exponents sksubscript𝑠𝑘s_{k} are robust to the choice of the evaluation metric f𝑓f. In Appendix B.3, we discuss important considerations that were taken into account during this analysis.

Scaling. Finally, we scale all dimensions jointly. Starting from the small compute-optimal architecture 𝐱(0)superscript𝐱0\mathbf{x}^{(0)} and the amount of compute 𝐭(0)superscript𝐭0\mathbf{t}^{(0)} it is optimal for, suppose we increase compute by a factor τ>1𝜏1\tau>1 (i.e. the new compute is τ​𝐭(0)𝜏superscript𝐭0\tau\,\mathbf{t}^{(0)}). By treating this increment τ𝜏\tau as a sequence of D𝐷D smaller increments of size τwksuperscript𝜏subscript𝑤𝑘\tau^{w_{k}} each with ∑kwk=1subscript𝑘subscript𝑤𝑘1\sum_{k}w_{k}=1, an increase in compute by a factor of τ𝜏\tau is accompanied by an increase in every shape dimension k𝑘k by a factor of τwksuperscript𝜏subscript𝑤𝑘\tau^{w_{k}}, respectively. In this work, the adopt the simplest strategy of setting wk=1/Dsubscript𝑤𝑘1𝐷w_{k}=1/D, but acknowledge that more sophisticated approaches might lead to better results.

## 4 Shape-optimized ViT

We implement the scaling strategy in Section 3 in vision transformers [24] pretrained on JFT-3B, a proprietary dataset with about 30k classes and around 3 billion examples [80], using the Adam optimizer [41]. As mentioned in Section 3, we focus on optimizing three shape dimensions: width (size of internal representation), depth (number of encoder blocks) and MLP dim (hidden dimension). Following [43, 24, 80], we remove near-duplicate examples between upstream JFT-3B data and all the downstream train and test sets. Appendix B contains the full set of hyper-parameters used in the experiments, including full details about the star and grid sweeps described in Section 3. We fix the patch size in our analysis to 14×14141414\times 14, but study “flexifying” to arbitrary sequence lengths following [7] in Section 5.5.

As an evaluation metric f𝑓f, we consider two domains: (1) image classification, with ImageNet linear 10-shot error rate as the metric, and (2) image-to-text LiT-decoding following [8]. In the latter case, the evaluation metric f𝑓f is an average of four perplexity scores: COCO captioning, optical character recognition (OCR), and question answering (VQAv2 and GQA). Refer to [8] for details about the LiT-decoder setup. By considering such distinct domains, our goal is to identify similarities and differences (if any) in how to optimally scale the shape of vision transformers (ViT).

### 4.1 Image Classification

We use the aforementioned star center 𝐱(c)=(1968, 40, 6144)superscript𝐱𝑐1968406144\mathbf{x}^{(c)}=(1968,\,40,\,6144) as our starting point. To estimate the scaling exponents sksubscript𝑠𝑘s_{k} in (4) for each dimension separately, we vary width in the grid (608, 768, 928, 1088, 1328, 1648)608768928108813281648(608,\,768,\,928,\,1088,\,1328,\,1648), depth in the grid (8, 10, 12, 16, 20, 24)81012162024(8,\,10,\,12,\,16,\,20,\,24), and MLP dim in the grid (1088, 1360, 1728, 2160, 2592, 3072)108813601728216025923072(1088,\,1360,\,1728,\,2160,\,2592,\,3072). As discussed in Section 3, we use an exponential spacing with all values being much smaller than in the star center 𝐱(c)superscript𝐱𝑐\mathbf{x}^{(c)}. Following [24], we evaluate quality using few-shot linear transfer by using pre-trained models to extract features and fitting a linear regression head mapping them to the one-hot encoding of the target labels.

The individual scaling exponents we find are sdepth≈0.45subscript𝑠depth0.45s_{\text{depth}}\approx 0.45, swidth≈0.22subscript𝑠width0.22s_{\text{width}}\approx 0.22, and sMLP≈0.6subscript𝑠MLP0.6s_{\text{MLP}}\approx 0.6. Importantly, these exponents are quite robust to the choice of the metric. As shown in Figure 5, changing the metric from ImageNet 10-shot to either 5-shot or 25-shot can change the best-fit estimate of the other exponents ak,bk,cksubscript𝑎𝑘subscript𝑏𝑘subscript𝑐𝑘a_{k},b_{k},c_{k} in (2) but the scaling exponent sksubscript𝑠𝑘s_{k} is relatively unchanged, since it is formed as a ratio over other exponents. In addition, the data scaling exponent c𝑐c appears to be independent of the choice of the shape dimension. As mentioned earlier, this is consistent with space partitioning arguments for power law scaling [2, 38, 59].

The estimated scaling exponents sksubscript𝑠𝑘s_{k} point to the following picture:

- I.

MLP dimension should be scaled faster than depth, and depth faster than width.

- II.

The size of ViT, as quantified by its parameter count, is scaled more slowly than the allocated compute. More precisely, for every increment in compute by a factor of 101010, the parameter count of the optimized model shape increases by a factor of ≈2.5absent2.5\approx 2.5.

- III.

As demonstrated in Figure 1, small ViT models can match the performance of much larger ones when their shape and training duration are jointly optimized for the available compute.

We validate these predictions by optimizing the shape of ViT for the compute-equivalent of ViT-g/14 when the latter is pretrained on 16 billion JFT-3B examples as done in [80]. The resulting model, SoViT-400m/14, is significantly smaller and faster, yet equally competitive. It has a width of 1152, depth 27, and MLP dim 4304. Fine-tuning it on ImageNet results in a 90.3% top-1 accuracy, see Figure 2. Section 5 presents various other evaluations.

In Figure 6, we also optimize the shape of ViT for the compute-equivalent of ViT-B/14 pretrained on 4 billion examples of JFT-3B using Imagenet 10-shot error rate as an evaluation metric, resulting in SoViT-150m/14. It has a width of 880, depth 18, and MLP dim 2320.
As shown in Figure 6, optimizing the shape of ViT leads to a significant improvement in performance, from 76.6% in ViT-B/14 to 78.5% in SoViT-150m/14 when both are trained for the same amount of compute. We also vary the optimized shape by decreasing/increasing one dimension at a time and retraining the corresponding model while keeping the total compute fixed. As shown in Figure 6, small deviations from the predicted optimal shape can lead to a notable drop in performance, especially for width since it has the smallest scaling exponent (see Figure 5). We also include in Figure 6 (left) a comparison with a model, denoted B-150m, which has the same shape as ViT-B/14 but the same size as SoViT-150m/14. This confirms that while optimizing the model size improves performance, optimizing the shape improves it even further.

Importantly, the model shapes in Figure 6 bear no resemblance to those observed during the star or grid sweeps. To recall, the star sweep is centered around an architecture 𝐱(c)superscript𝐱𝑐\mathbf{x}^{(c)} whose shape dimensions are significantly larger than in ViT-B/14, whereas the grid sweep pretrains models that are substantially smaller and for only 600M examples. The ability of our strategy to accurately identify a near-optimal model shape within this context underscores its robust extrapolation capability.

### 4.2 Multitask Decoder

Besides image classification, there has been a significant interest in multimodal applications, mostly fueled by the convergence across language and vision on the transformer architecture [72, 24]. In particular, an encoder-decoder transformer with an autoregressive decoder is a popular choice because it allows reusing pretrained image encoders. We repeat the analysis conducted in Section 4.1 to optimize the shape of the image encoder, while fixing the decoder architecture to two layers as was used in [8]. Further details are provided in Appendix C. As an evaluation metric f𝑓f, we use the average of four perplexity scores: COCO captioning [48, 14], OCR [50], VQAv2 [28] and GQA [37], without normalization since they share a similar scale.
For the learning rate and weight decay hyper-parameters, we conduct a sweep where we vary the learning rate in {10−3, 3×10−4, 10−4}superscript1033superscript104superscript104\{10^{-3},\,3\times 10^{-4},\,10^{-4}\} and the weight decay in {3×10−4, 10−4, 3×10−5}3superscript104superscript1043superscript105\{3\times 10^{-4},\,10^{-4},\,3\times 10^{-5}\}. We pick the largest learning rate and the corresponding weight decay that result in a stable training run (i.e. smooth training loss curve and gradient norms) for both the largest and smallest image encoder architectures. From this, a learning rate of 3×10−43superscript1043\times 10^{-4} and a weight decay of 10−4superscript10410^{-4} are selected.

Using this analysis, the derived scaling exponents are approximately 0.25,0.490.250.490.25,0.49 and 0.620.620.62 for width, depth and MLP size, respectively. Hence, whereas the optimal shape dimensions in small architectures can be quite different between image classification and multitask decoding, as shown in Figure 3, the scaling exponents are nearly identical, so the same scaling recipe is used in both domains.

## 5 Evaluations

Overview. 
We now evaluate SoViT-400M in various contexts to verify whether it broadly matches ViT-g/14’s performance, or only in the ILSRCV2012 10-shot metric it was optimized for.
The settings we cover are few-shot, frozen linear probes on ImageNet, zero-shot transfer, image-language multitasking including captioning, OCR, and question answering, as well as panoptic segmentation. In each of these settings, we compare SoViT-400m/14 to ViT-L/16 and a ViT-g/14, all trained on the

Compute. 
Experiments are executed on Tensor Processing Units (TPU). SoViT-400m/14 is pretrained on 40 billion examples, which amounts to 9T GFLOPs and 230K TPUv3 core-hours.
ViT-g/14 was pretrained on 16 billion examples, corresponding to 9T GFLOPs and 210K TPUv3 core-hours.

### 5.1 Image Classification

We verify classification performance in three common and widely useful setups: full fine-tuning, linear probes on the frozen model, and few-shot linear classification.

Model

Pretraining

Size

ImageNet variant

Input

Params

FLOPs

Val [57]

ReaL [6]

v2 [56]

SoViT-400m/14

JFT-3B

2242

0428 M

0221 G

88.9

90.3

80.7

ViT-L/16 [80]

JFT-3B

3842

0303 M

0383 G

88.5

90.4

80.4

SoViT-400m/14

JFT-3B

3842

0428 M

0672 G

90.0

90.9

83.2

ViT-g/14 [80]

JFT-3B

5182

1011 M

3208 G

90.2

90.9

-

SoViT-400m/14

JFT-3B

5182

0428 M

1374 G

90.3

91.0

83.4

ViT-G/14 [80]

JFT-3B

5182

1882 M

5668 G

90.4

90.8

83.3

SwinV2-G [49]

IN-21k + 70M

6402

3000 M

-

90.2

-

84.0

CoAtNet-6 [19]

JFT-3B

5122

1470 M

1521 G

90.4

-

-

MAE→→\rightarrowWSP [61]

IG-3B

5182

1890 M

5679 G

89.7

90.9

83.0

CoCa [77]

JFT-3B + ALIGN-1.8B

5762

2100 M

-

91.0

-

-

Fine-tuning on ImageNet. 
Pre-trained image encoders are most commonly [18] evaluated by fine-tuning them on the ILSVRC2012 classification task.
The detailed fine-tuning settings are provided in Appendix E.
One important aspect is to increase image resolution [70] as a way of further increasing the capacity of the pre-trained model during fine-tuning [43].
Table 1 shows the performance of SoViT-400m/14 in comparison with ViT-L/16, ViT-g/14 fine-tuned at various resolutions, along with a few more representative models from the literature.
The results confirm that SoViT-400m/14 achieves the goal of matching ViT-g/14 while being significantly smaller.

Val

ReaL

v2

-R

-A

Obj

L/16

86.7

90.0

78.5

88.9

67.8

63.5

SoViT

88.2

90.3

80.6

89.0

76.4

68.7

g/14

88.4

90.2

80.8

90.3

76.6

67.7

Linear probing on ImageNet. 
The quality of the pre-trained representation learned by the model is often more directly assessed by performing linear probes, meaning learning a linear classifier on top of unmodified, frozen output features from the model.
We present results of this evaluation on the full ImageNet-1k [57] dataset in Table 2, including robustness evaluations of the learned probe according to ReaL [6], ImageNet-v2 [56], ImageNet-Renditions [30], ImageNet-Adversarial [31], and ObjectNet [4] testsets.
SoViT-400m/14 is generally on par with ViT-g/14 despite its smaller output width.

Broad few-shot linear transfer. 
We follow [24, 80] and evaluate a closed-form linear regression probe for 10-shot classification across a wide range of classification tasks in Table 3. Again, SoViT-400m/14 performs on-par with ViT-g/14 across the board.

INet [22]

CIFAR100 [46]

Pets [51]

Birds [74]

Caltech [25]

Cars [45]

Colorectal [40]

DTD [17]

UC [76]

ViT-L/16

81.5

82.2

97.0

97.1

89.9

93.8

79.4

72.0

96.3

SoViT-400m/14

84.1

86.7

97.6

88.8

91.3

93.6

81.5

72.5

97.7

ViT-g/14

84.0

87.2

97.4

88.5

89.3

93.9

78.9

74.1

98.2

### 5.2 Contrastive image-text tuning

Next, we follow the locked-image text tuning (LiT) recipe [81] on the WebLI dataset [15] to add zero-shot classification abilities to the pre-trained ViT-L/16, SoViT-400m/14 and ViT-g/14 image encoders. In this setup, a new text encoder is trained using the contrastive image-text matching objective [54]. See Appendix D for details. Table 4 (second column) shows that SoViT-400m/14 is competitive with ViT-g/14, and substantially better than ViT-L/16.

### 5.3 Multitask Decoding

We also evaluate the three pretrained ViT models in multitask decoding as described in Section 4.2, where we follow the setup studied in [8]. We fix the decoder architecture to two layers since it was found to perform well [8]. For evaluation, we report COCO CIDEr [48, 14, 73], OCR [50], VQAv2 [28] and GQA [37] accuracy and log-perplexity. In brief, the CIDEr score measures the similarity between a generated caption and reference captions, considering n𝑛n-gram statistics, OCR evaluates optical character recognition, whereas both VQAv2 and GQA are question-answering evaluations. Results are summarized in Table 4. SoViT-400M performs on par with ViT-g/14.

### 5.4 Panoptic Segmentation

Additionally, we evaluate SoViT-400m/14 on panoptic segmentation [42], which is a challenging dense scene understating task by closely following the setup in UViM [44]. At a high level, UViM panoptic segmentation model consists of a visual image encoder and a decoder which maps the image representation to an intermediate code.
The code is later decoded to the panoptic segmentation mask using a fixed VQVAE [71] model, which was pretrained on panoptic masks [44]. In our experiments we initialize UViM’s image encoder with ViT-L/16, SoViT-400m/14 and ViT-g/14.

Following [44], we train the UViM model using the COCO panoptic dataset (with 512×512512512512\times 512 input resolution) and report the PQ metric. We achieve 43.5, 43.7 and 44.8 PQ points for ViT-L/16, SoViT-400m/14 and ViT-g/14 respectively. Our results indicate that dense segmentation tasks can be a limitation of the proposed optimal model shape, and a different model shape might be derived in this domain. We leave this investigation for future work.

### 5.5 Flexifying SoViT-400M

Finally, since we do not include the patch size (sequence length) as part of the shape optimization, we verify that this is not a limitation by flexifying [7] SoViT-400m/14 on ILSVRC2012 for 300 epochs.
The performance of the resulting FlexiSoViT-400m is shown in Fig 7 as green curve when varying the patch-size at inference time. A few reference ViT models from Table 1 and [80] are added, confirming that SoViT-400m maintains a clear advantage. It is worth noting that flexifying does not rule out that other patch sizes could be compute-optimal. It merely demonstrates that SoViT-400M continues to perform quite well for other patch sizes when it is flexified.

Model

ImgNet

OCR-VQA [50]

GQA [37]

VQAv2 [28]

COCO Capt. [14]

Zero-shot

Acc [%]

Log-PPL

Acc [%]

Log-PPL

Acc [%]

Log-PPL

CIDEr

Log-PPL

ViT-L/16

79.9

48.3

17.9

55.3

24.9

66.4

20.9

120

28.7

SoViT-400M

82.2

52.9

15.3

56.0

23.9

67.7

20.9

125

28.1

ViT-g/14

82.4

52.5

15.9

58.0

22.5

68.8

21.5

126

27.9

## 6 Conclusion

In conclusion, we introduce an efficient method for optimizing the shape of neural architectures and successfully apply it to vision transformers. Our analysis demonstrates that smaller models, trained at their optimal architecture shape for the right amount of compute, can match much larger models.

## Acknowledgments and Disclosure of Funding

We thank Mostafa Dehghani, Andreas Steiner, Daniel Keysers, Neil Houlsby, Sam Smith, David Schneider-Joseph, Rodolphe Jenatton and the anonymous reviewers for their valuable feedback and discussions.
We also thank the Google DeepMind unit at large for providing a supportive research environment.
We use the big_vision codebase [10, 9] for conducting experiments in this project.

## ArXiv Version History

Version 1: Original version.
Version 2: Layout fixes. Add missing citations to ImageNet-R,-A and ObjectNet.
Version 3: Provided the full shape of SoViT-150m/14. Added details to Appendix B.3 about the grid sweep, a missing citation to the CIDEr score, and further discussions to Figures 3 and 6. Included a brief explanation of the image-to-text evaluation metrics in Section 5.3, the scaling exponents in Section 3 and the shape dimensions in Section 4. Fixed typos.
Version 4: Fixed wall-clock time pre-training duration (TPUv3 core-hours) of SoViT-400m.
Version 5: Fixed typos. Added brief explanations to Section 3.

## References

- Alabdulmohsin et al., [2022]

Alabdulmohsin, I., Neyshabur, B., and Zhai, X. (2022).

Revisiting neural scaling laws in language and vision.

In Advances in neural information processing systems (NeurIPS).

- Bahri et al., [2021]

Bahri, Y., Dyer, E., Kaplan, J., Lee, J., and Sharma, U. (2021).

Explaining neural scaling laws.

arXiv preprint arXiv:2102.06701.

- Bansal et al., [2022]

Bansal, Y., Ghorbani, B., Garg, A., Zhang, B., Krikun, M., Cherry, C.,
Neyshabur, B., and Firat, O. (2022).

Data scaling laws in NMT: The effect of noise and architecture.

arXiv preprint arXiv:2202.01994.

- Barbu et al., [2019]

Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum,
J., and Katz, B. (2019).

Objectnet: A large-scale bias-controlled dataset for pushing the
limits of object recognition models.

Advances in neural information processing systems, 32.

- Bello et al., [2021]

Bello, I., Fedus, W., Du, X., Cubuk, E. D., Srinivas, A., Lin, T.-Y., Shlens,
J., and Zoph, B. (2021).

Revisiting resnets: Improved training and scaling strategies.

Advances in neural information processing systems (NeurIPS).

- Beyer et al., [2020]

Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., and van den Oord, A.
(2020).

Are we done with imagenet?

CoRR, abs/2006.07159.

- [7]

Beyer, L., Izmailov, P., Kolesnikov, A., Caron, M., Kornblith, S., Zhai, X.,
Minderer, M., Tschannen, M., Alabdulmohsin, I., and Pavetic, F. (2023a).

Flexivit: One model for all patch sizes.

In CVPR.

- [8]

Beyer, L., Wan, B., Madan, G., Pavetic, F., Steiner, A., Kolesnikov, A., Pinto,
A. S., Bugliarello, E., Wang, X., Yu, Q., Chen, L.-C., and Zhai, X. (2023b).

A study of autoregressive decoders for multi-tasking in computer
vision.

- [9]

Beyer, L., Zhai, X., and Kolesnikov, A. (2022a).

Better plain vit baselines for imagenet-1k.

- [10]

Beyer, L., Zhai, X., and Kolesnikov, A. (2022b).

Big vision.

https://github.com/google-research/big_vision.

- Boyd and Vandenberghe, [2004]

Boyd, S. P. and Vandenberghe, L. (2004).

Convex optimization.

Cambridge university press.

- Brown et al., [2022]

Brown, J. R., Zhao, Y., Shumailov, I., and Mullins, R. D. (2022).

Wide attention is the way forward for transformers.

arXiv preprint arXiv:2210.00640.

- Brown et al., [2020]

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P.,
Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020).

Language models are few-shot learners.

Advances in neural information processing systems (NeurIPS).

- Chen et al., [2015]

Chen, X., Fang, H., Lin, T.-Y., Vedantam, R., Gupta, S., Dollár, P., and
Zitnick, C. L. (2015).

Microsoft coco captions: Data collection and evaluation server.

arXiv preprint arXiv:1504.00325.

- Chen et al., [2022]

Chen, X., Wang, X., Changpinyo, S., Piergiovanni, A., Padlewski, P., Salz, D.,
Goodman, S., Grycner, A., Mustafa, B., Beyer, L., Kolesnikov, A., Puigcerver,
J., Ding, N., Rong, K., Akbari, H., Mishra, G., Xue, L., Thapliyal, A.,
Bradbury, J., Kuo, W., Seyedhosseini, M., Jia, C., Ayan, B. K., Riquelme, C.,
Steiner, A., Angelova, A., Zhai, X., Houlsby, N., and Soricut, R. (2022).

Pali: A jointly-scaled multilingual language-image model.

- Chowdhery et al., [2022]

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A.,
Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. (2022).

Palm: Scaling language modeling with pathways.

arXiv preprint arXiv:2204.02311.

- Cimpoi et al., [2014]

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. (2014).

Describing textures in the wild.

In Proceedings of the IEEE Conf. on Computer Vision and
Pattern Recognition (CVPR).

- Code, [2023]

Code, P. W. (2023).

Papers With Code: ImageNet Benchmark.

https://paperswithcode.com/sota/image-classification-on-imagenet.

[Online; accessed 16-May-2023].

- Dai et al., [2021]

Dai, Z., Liu, H., Le, Q. V., and Tan, M. (2021).

Coatnet: Marrying convolution and attention for all data sizes.

Advances in neural information processing systems (NeurIPS).

- Dehghani et al., [2022]

Dehghani, M., Arnab, A., Beyer, L., Vaswani, A., and Tay, Y. (2022).

The efficiency misnomer.

In ICLR.

- Dehghani et al., [2023]

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J.,
Steiner, A., Caron, M., Geirhos, R., Alabdulmohsin, I., Jenatton, R., Beyer,
L., Tschannen, M., Arnab, A., Wang, X., Riquelme, C., Minderer, M.,
Puigcerver, J., Evci, U., Kumar, M., van Steenkiste, S., Elsayed, G. F.,
Mahendran, A., Yu, F., Oliver, A., Huot, F., Bastings, J., Collier, M. P.,
Gritsenko, A., Birodkar, V., Vasconcelos, C., Tay, Y., Mensink, T.,
Kolesnikov, A., Pavetić, F., Tran, D., Kipf, T., Lučić, M., Zhai, X.,
Keysers, D., Harmsen, J., and Houlsby, N. (2023).

Scaling vision transformers to 22 billion parameters.

- Deng et al., [2009]

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009).

Imagenet: A large-scale hierarchical image database.

In Conference on Computer Vision and Pattern Recognition
(CVPR).

- Devlin et al., [2018]

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2018).

Bert: Pre-training of deep bidirectional transformers for language
understanding.

arXiv preprint arXiv:1810.04805.

- Dosovitskiy et al., [2020]

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.
(2020).

An image is worth 16x16 words: Transformers for image recognition at
scale.

International Conference on Representation Learning (ICLR).

- Fei-Fei et al., [2004]

Fei-Fei, L., Fergus, R., and Perona, P. (2004).

Learning generative visual models from few training examples: An
incremental bayesian approach tested on 101 object categories.

Conference on Computer Vision and Pattern Recognition (CVPR)
Workshops.

- Ghorbani et al., [2021]

Ghorbani, B., Firat, O., Freitag, M., Bapna, A., Krikun, M., Garcia, X.,
Chelba, C., and Cherry, C. (2021).

Scaling laws for neural machine translation.

arXiv preprint arXiv:2109.07740.

- Gordon et al., [2021]

Gordon, M. A., Duh, K., and Kaplan, J. (2021).

Data and parameter scaling laws for neural machine translation.

In Conference on Empirical Methods in Natural Language
Processing.

- Goyal et al., [2017]

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. (2017).

Making the v in vqa matter: Elevating the role of image understanding
in visual question answering.

In CVPR.

- He et al., [2016]

He, K., Zhang, X., Ren, S., and Sun, J. (2016).

Deep residual learning for image recognition.

In Conference on Computer Vision and Pattern Recognition
(CVPR).

- [30]

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai,
R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., and Gilmer, J.
(2021a).

The many faces of robustness: A critical analysis of
out-of-distribution generalization.

ICCV.

- [31]

Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., and Song, D. (2021b).

Natural adversarial examples.

CVPR.

- Henighan et al., [2020]

Henighan, T., Kaplan, J., Katz, M., Chen, M., Hesse, C., Jackson, J., Jun, H.,
Brown, T. B., Dhariwal, P., Gray, S., et al. (2020).

Scaling laws for autoregressive generative modeling.

arXiv preprint arXiv:2010.14701.

- Hernandez et al., [2021]

Hernandez, D., Kaplan, J., Henighan, T., and McCandlish, S. (2021).

Scaling laws for transfer.

arXiv preprint arXiv:2102.01293.

- Hestness et al., [2017]

Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H.,
Patwary, M., Ali, M., Yang, Y., and Zhou, Y. (2017).

Deep learning scaling is predictable, empirically.

arXiv preprint arXiv:1712.00409.

- Hoffmann et al., [2022]

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford,
E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., et al. (2022).

Training compute-optimal large language models.

In Advances in neural information processing systems (NeurIPS).

- Howard et al., [2017]

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T.,
Andreetto, M., and Adam, H. (2017).

MobileNets: Efficient convolutional neural networks for mobile
vision applications.

arXiv preprint arXiv:1704.04861.

- Hudson and Manning, [2019]

Hudson, D. A. and Manning, C. D. (2019).

GQA: a new dataset for compositional question answering over
real-world images.

arXiv preprint arXiv:1902.09506.

- Hutter, [2021]

Hutter, M. (2021).

Learning curve theory.

arXiv preprint arXiv:2102.04074.

- Kaplan et al., [2020]

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R.,
Gray, S., Radford, A., Wu, J., and Amodei, D. (2020).

Scaling laws for neural language models.

arXiv preprint arXiv:2001.08361.

- Kather et al., [2016]

Kather, J. N., Weis, C.-A., Bianconi, F., Melchers, S. M., Schad, L. R.,
Gaiser, T., Marx, A., and Z"ollner, F. G. (2016).

Multi-class texture analysis in colorectal cancer histology.

Scientific reports, 6:27988.

- Kingma and Ba, [2014]

Kingma, D. P. and Ba, J. (2014).

Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980.

- Kirillov et al., [2019]

Kirillov, A., He, K., Girshick, R., Rother, C., and Dollar, P. (2019).

Panoptic segmentation.

In Conference on Computer Vision and Pattern Recognition
(CVPR).

- Kolesnikov et al., [2020]

Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., and
Houlsby, N. (2020).

Big transfer (BiT): General visual representation learning.

In European Conference on Computer Vision (ECCV).

- Kolesnikov et al., [2022]

Kolesnikov, A., Susano Pinto, A., Beyer, L., Zhai, X., Harmsen, J., and
Houlsby, N. (2022).

UViM: A unified modeling approach for vision with learned guiding
codes.

Advances in neural information processing systems (NeurIPS).

- Krause et al., [2013]

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. (2013).

3d object representations for fine-grained categorization.

In 4th International IEEE Workshop on 3D Representation and
Recognition (3dRR-13), Sydney, Australia.

- Krizhevsky, [2009]

Krizhevsky, A. (2009).

Learning multiple layers of features from tiny images.

Technical report.

- Li et al., [2020]

Li, Z., Wallace, E., Shen, S., Lin, K., Keutzer, K., Klein, D., and Gonzalez,
J. (2020).

Train big, then compress: Rethinking model size for efficient
training and inference of transformers.

In International Conference on Machine Learning (ICML).

- Lin et al., [2014]

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D.,
Dollár, P., and Zitnick, C. L. (2014).

Microsoft coco: Common objects in context.

In ECCV.

- Liu et al., [2022]

Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang,
Z., Dong, L., et al. (2022).

Swin transformer v2: Scaling up capacity and resolution.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 12009–12019.

- Mishra et al., [2019]

Mishra, A., Shekhar, S., Singh, A. K., and Chakraborty, A. (2019).

OCR-VQA: Visual question answering by reading text in images.

In ICDAR.

- Parkhi et al., [2012]

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. (2012).

Cats and dogs.

In IEEE Conference on Computer Vision and Pattern Recognition.

- Patterson et al., [2021]

Patterson, D., Gonzalez, J., Le, Q., Liang, C., Munguia, L.-M., Rothchild, D.,
So, D., Texier, M., and Dean, J. (2021).

Carbon emissions and large neural network training.

arXiv preprint arXiv:2104.10350.

- Pham et al., [2021]

Pham, H., Dai, Z., Ghiasi, G., Kawaguchi, K., Liu, H., Yu, A. W., Yu, J., Chen,
Y.-T., Luong, M.-T., Wu, Y., et al. (2021).

Combined scaling for zero-shot transfer learning.

arXiv preprint arXiv:2111.10050.

- Radford et al., [2021]

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry,
G., Askell, A., Mishkin, P., Clark, J., et al. (2021).

Learning transferable visual models from natural language
supervision.

In ICML.

- Rae et al., [2021]

Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F.,
Aslanides, J., Henderson, S., Ring, R., Young, S., et al. (2021).

Scaling language models: Methods, analysis & insights from training
gopher.

arXiv preprint arXiv:2112.11446.

- Recht et al., [2019]

Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. (2019).

Do imagenet classifiers generalize to imagenet?

CoRR, abs/1902.10811.

- Russakovsky et al., [2014]

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z.,
Karpathy, A., Khosla, A., Bernstein, M. S., Berg, A. C., and Fei-Fei, L.
(2014).

Imagenet large scale visual recognition challenge.

CoRR, abs/1409.0575.

- Sandler et al., [2018]

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018).

MobileNetV2: Inverted residuals and linear bottlenecks.

In Conference on Computer Vision and Pattern Recognition
(CVPR).

- Sharma and Kaplan, [2022]

Sharma, U. and Kaplan, J. (2022).

Scaling laws from the data manifold dimension.

Journal of Machine Learning Research, 23.

- Shazeer and Stern, [2018]

Shazeer, N. and Stern, M. (2018).

Adafactor: Adaptive learning rates with sublinear memory cost.

In International Conference on Machine Learning (ICML).

- Singh et al., [2023]

Singh, M., Duval, Q., Alwala, K. V., Fan, H., Aggarwal, V., Adcock, A., Joulin,
A., Dollár, P., Feichtenhofer, C., Girshick, R., et al. (2023).

The effectiveness of mae pre-pretraining for billion-scale
pretraining.

arXiv preprint arXiv:2303.13496.

- Steiner et al., [2022]

Steiner, A. P., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., and
Beyer, L. (2022).

How to train your vit? data, augmentation, and regularization in
vision transformers.

Transactions on Machine Learning Research.

- Sun et al., [2017]

Sun, C., Shrivastava, A., Singh, S., and Gupta, A. (2017).

Revisiting unreasonable effectiveness of data in deep learning era.

In International Conference on Computer Vision (ICCV).

- Tan and Le, [2019]

Tan, M. and Le, Q. (2019).

EfficientNet: Rethinking model scaling for convolutional neural
networks.

In International Conference on Machine Learning (ICML).

- [65]

Tay, Y., Dehghani, M., Abnar, S., Chung, H. W., Fedus, W., Rao, J., Narang, S.,
Tran, V. Q., Yogatama, D., and Metzler, D. (2022a).

Scaling laws vs model architectures: How does inductive bias
influence scaling?

arXiv preprint arXiv:2207.10551.

- [66]

Tay, Y., Dehghani, M., Rao, J., Fedus, W., Abnar, S., Chung, H. W., Narang, S.,
Yogatama, D., Vaswani, A., and Metzler, D. (2022b).

Scale efficiently: Insights from pre-training and fine-tuning
transformers.

In International Conference on Representation Learning (ICLR).

- Touvron et al., [2021]

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jégou,
H. (2021).

Training data-efficient image transformers & distillation through
attention.

In International Conference on Machine Learning (ICML).

- Touvron et al., [2022]

Touvron, H., Cord, M., and Jégou, H. (2022).

DeiT III: Revenge of the ViT.

In Computer Vision–ECCV 2022: 17th European Conference, Tel
Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV, pages 516–533.
Springer.

- Touvron et al., [2023]

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix,
T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin,
A., Grave, E., and Lample, G. (2023).

Llama: Open and efficient foundation language models.

- Touvron et al., [2019]

Touvron, H., Vedaldi, A., Douze, M., and Jégou, H. (2019).

Fixing the train-test resolution discrepancy.

Advances in neural information processing systems, 32.

- Van Den Oord et al., [2017]

Van Den Oord, A., Vinyals, O., et al. (2017).

Neural discrete representation learning.

Advances in neural information processing systems (NeurIPS).

- Vaswani et al., [2017]

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., and Polosukhin, I. (2017).

Attention is all you need.

Advances in neural information processing systems (NeurIPS).

- Vedantam et al., [2015]

Vedantam, R., Lawrence Zitnick, C., and Parikh, D. (2015).

Cider: Consensus-based image description evaluation.

In CVPR.

- Welinder et al., [2010]

Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., and
Perona, P. (2010).

Caltech-UCSD Birds 200.

Technical Report CNS-TR-2010-001, California Institute of Technology.

- Wu et al., [2019]

Wu, Z., Shen, C., and Van Den Hengel, A. (2019).

Wider or deeper: Revisiting the resnet model for visual recognition.

Pattern Recognition.

- Yang and Newsam, [2010]

Yang, Y. and Newsam, S. (2010).

Bag-of-visual-words and spatial extensions for land-use
classification.

In ACM SIGSPATIAL International Conference on Advances in
Geographic Information Systems (ACM GIS).

- Yu et al., [2022]

Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., and Wu, Y.
(2022).

CoCa: Contrastive captioners are image-text foundation models.

arXiv preprint arXiv:2205.01917.

- Yuan et al., [2021]

Yuan, L., Chen, D., Chen, Y.-L., Codella, N., Dai, X., Gao, J., Hu, H., Huang,
X., Li, B., Li, C., et al. (2021).

Florence: A new foundation model for computer vision.

arXiv preprint arXiv:2111.11432.

- Zagoruyko and Komodakis, [2016]

Zagoruyko, S. and Komodakis, N. (2016).

Wide residual networks.

arXiv preprint arXiv:1605.07146.

- [80]

Zhai, X., Kolesnikov, A., Houlsby, N., and Beyer, L. (2022a).

Scaling vision transformers.

In Conference on Computer Vision and Pattern Recognition
(CVPR).

- [81]

Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., and
Beyer, L. (2022b).

LiT: Zero-shot transfer with locked-image text tuning.

In CVPR, pages 18123–18133.

## Appendix A Scaling Laws Analysis

In this appendix, we present proofs of two claims in the paper. First, we show that (2) is quasiconvex on its first argument 𝐱ksubscript𝐱𝑘\mathbf{x}_{k}. Second, we derive (5).

### A.1 Quasiconvexity Proof

We assume throughout the proof that ak,bksubscript𝑎𝑘subscript𝑏𝑘a_{k},b_{k} are strictly positive, otherwise fk​(𝐱k,𝐭)subscript𝑓𝑘subscript𝐱𝑘𝐭f_{k}(\mathbf{x}_{k},\mathbf{t}) is a monotone function on its first argument and the statement holds trivially.

To establish the quasiconvexity of fk​(𝐱k,𝐭)subscript𝑓𝑘subscript𝐱𝑘𝐭f_{k}(\mathbf{x}_{k},\,\mathbf{t}) in (2), we observe that:

∂fk∂𝐱k=−αk​ak​𝐱k−(1+ak)+βk​bk​𝐭−c​𝐱kbk−1≐−A​𝐱k−(1+ak)+B​𝐱bk−1.subscript𝑓𝑘subscript𝐱𝑘subscript𝛼𝑘subscript𝑎𝑘superscriptsubscript𝐱𝑘1subscript𝑎𝑘subscript𝛽𝑘subscript𝑏𝑘superscript𝐭𝑐superscriptsubscript𝐱𝑘subscript𝑏𝑘1approaches-limit𝐴superscriptsubscript𝐱𝑘1subscript𝑎𝑘𝐵superscript𝐱subscript𝑏𝑘1\frac{\partial f_{k}}{\partial\mathbf{x}_{k}}=-\alpha_{k}a_{k}\mathbf{x}_{k}^{-(1+a_{k})}+\beta_{k}b_{k}\mathbf{t}^{-c}\mathbf{x}_{k}^{b_{k}-1}\doteq-A\mathbf{x}_{k}^{-(1+a_{k})}+B\mathbf{x}^{b_{k}-1}.

Setting the derivative to zero gives the unique solution in ℝ+superscriptℝ\mathbb{R}^{+}:

𝐱^=(AB)1ak+bk.^𝐱superscript𝐴𝐵1subscript𝑎𝑘subscript𝑏𝑘\hat{\mathbf{x}}=\left(\frac{A}{B}\right)^{\frac{1}{a_{k}+b_{k}}}.

At the limit 𝐱k→∞→subscript𝐱𝑘\mathbf{x}_{k}\to\infty, the term involving 𝐱k−aksuperscriptsubscript𝐱𝑘subscript𝑎𝑘\mathbf{x}_{k}^{-a_{k}} vanishes and we have the asymptotic relation:

fk​(𝐱k,𝐭)∼βk​𝐭−c​𝐱kbk,similar-tosubscript𝑓𝑘subscript𝐱𝑘𝐭subscript𝛽𝑘superscript𝐭𝑐superscriptsubscript𝐱𝑘subscript𝑏𝑘f_{k}(\mathbf{x}_{k},\mathbf{t})\sim\beta_{k}\mathbf{t}^{-c}\mathbf{x}_{k}^{b_{k}},

which is an increasing function since bk>0subscript𝑏𝑘0b_{k}>0. Since x^^𝑥\hat{x} is the only point in ℝ+superscriptℝ\mathbb{R}^{+} where ∂fk/∂𝐱k=0subscript𝑓𝑘subscript𝐱𝑘0{\partial f_{k}}/{\partial\mathbf{x}_{k}}=0, we conclude that f​(𝐱k,𝐭)𝑓subscript𝐱𝑘𝐭f(\mathbf{x}_{k},\mathbf{t}) is monotone increasing for all 𝐱k≥x^subscript𝐱𝑘^𝑥\mathbf{x}_{k}\geq\hat{x}.

Similarly, when 𝐱k→0+→subscript𝐱𝑘superscript0\mathbf{x}_{k}\to 0^{+}, we have:

fk​(𝐱k,𝐭)∼αk​𝐱k−ak,similar-tosubscript𝑓𝑘subscript𝐱𝑘𝐭subscript𝛼𝑘superscriptsubscript𝐱𝑘subscript𝑎𝑘f_{k}(\mathbf{x}_{k},\mathbf{t})\sim\alpha_{k}\mathbf{x}_{k}^{-a_{k}},

which is monotone decreasing. Therefore, f′​(𝐱k,𝐭)≤0superscript𝑓′subscript𝐱𝑘𝐭0f^{\prime}(\mathbf{x}_{k},\mathbf{t})\leq 0 for all 𝐱k≤x^subscript𝐱𝑘^𝑥\mathbf{x}_{k}\leq\hat{x}. Combining both results implies that fk​(x,𝐭)subscript𝑓𝑘𝑥𝐭f_{k}(x,\mathbf{t}) is monotone decreasing in the domain x∈(0,x^)𝑥0^𝑥x\in(0,\hat{x}) and is monotone increasing in the domain x∈(x^,∞)𝑥^𝑥x\in(\hat{x},\infty).

A function f​(y)𝑓𝑦f(y) is said to be quasi-convex if for any y1subscript𝑦1y_{1} and y2subscript𝑦2y_{2} in its domain and any λ∈[0,1]𝜆01\lambda\in[0,1], one has [11]:

f​(λ​y1+(1−λ)​y2)≤max⁡{f​(y1),f​(y2)}.𝑓𝜆subscript𝑦11𝜆subscript𝑦2𝑓subscript𝑦1𝑓subscript𝑦2f(\lambda y_{1}+(1-\lambda)y_{2})\leq\max\left\{f(y_{1}),\,f(y_{2})\right\}.

(6)

Suppose for the purpose of obtaining a contradiction that fk​(𝐱k,𝐭)subscript𝑓𝑘subscript𝐱𝑘𝐭f_{k}(\mathbf{x}_{k},\mathbf{t}) is not quasiconvex on its first argument. Then, there exists two points y1,y2∈ℝ+subscript𝑦1subscript𝑦2superscriptℝy_{1},y_{2}\in\mathbb{R}^{+} and λ∈[0,1]𝜆01\lambda\in[0,1] such that the above condition is violated. Let y^=λ​y1+(1−λ)​y2^𝑦𝜆subscript𝑦11𝜆subscript𝑦2\hat{y}=\lambda y_{1}+(1-\lambda)y_{2}. But, then, by the mean-value theorem, there must exist two points c1∈[y1,y^]subscript𝑐1subscript𝑦1^𝑦c_{1}\in[y_{1},\hat{y}] and c2∈[y^,y2]subscript𝑐2^𝑦subscript𝑦2c_{2}\in[\hat{y},y_{2}] where:

fk′​(c1)superscriptsubscript𝑓𝑘′subscript𝑐1\displaystyle f_{k}^{\prime}(c_{1})
=f​(y^)−f​(y1)y^−y1≥0absent𝑓^𝑦𝑓subscript𝑦1^𝑦subscript𝑦10\displaystyle=\frac{f(\hat{y})-f(y_{1})}{\hat{y}-y_{1}}\geq 0

fk′​(c2)superscriptsubscript𝑓𝑘′subscript𝑐2\displaystyle f_{k}^{\prime}(c_{2})
=f​(y2)−f​(y^)y2−y^≤0,absent𝑓subscript𝑦2𝑓^𝑦subscript𝑦2^𝑦0\displaystyle=\frac{f(y_{2})-f(\hat{y})}{y_{2}-\hat{y}}\leq 0,

with c2>c1subscript𝑐2subscript𝑐1c_{2}>c_{1}. This implies that c1≥x^subscript𝑐1^𝑥c_{1}\geq\hat{x} and c2≤x^subscript𝑐2^𝑥c_{2}\leq\hat{x}, which is a contradiction. Therefore, fk​(𝐱k,𝐭)subscript𝑓𝑘subscript𝐱𝑘𝐭f_{k}(\mathbf{x}_{k},\mathbf{t}) is quasi-convex on its first argument.

### A.2 Derivation of (5)

Rearranging the expression in (4), we have:

(βk​bkαk​ak)​(𝐱k⋆)bk+ak=𝐭csubscript𝛽𝑘subscript𝑏𝑘subscript𝛼𝑘subscript𝑎𝑘superscriptsuperscriptsubscript𝐱𝑘⋆subscript𝑏𝑘subscript𝑎𝑘superscript𝐭𝑐\left(\frac{\beta_{k}b_{k}}{\alpha_{k}a_{k}}\right)\left(\mathbf{x}_{k}^{\star}\right)^{b_{k}+a_{k}}=\mathbf{t}^{c}

From this and (2), we obtain:

fk​(𝐱k⋆,𝐭)=αk​(𝐱k⋆)−ak+βk​(𝐱k⋆)bk​(αk​akβk​bk​(𝐱k⋆)bk+ak)+ξk​𝐭−c+εk,subscript𝑓𝑘superscriptsubscript𝐱𝑘⋆𝐭subscript𝛼𝑘superscriptsuperscriptsubscript𝐱𝑘⋆subscript𝑎𝑘subscript𝛽𝑘superscriptsuperscriptsubscript𝐱𝑘⋆subscript𝑏𝑘subscript𝛼𝑘subscript𝑎𝑘subscript𝛽𝑘subscript𝑏𝑘superscriptsuperscriptsubscript𝐱𝑘⋆subscript𝑏𝑘subscript𝑎𝑘subscript𝜉𝑘superscript𝐭𝑐subscript𝜀𝑘\displaystyle f_{k}(\mathbf{x}_{k}^{\star},\,\mathbf{t})=\alpha_{k}(\mathbf{x}_{k}^{\star})^{-a_{k}}+\beta_{k}(\mathbf{x}_{k}^{\star})^{b_{k}}\left(\frac{\alpha_{k}a_{k}}{\beta_{k}b_{k}\,(\mathbf{x}_{k}^{\star})^{b_{k}+a_{k}}}\right)+\xi_{k}\mathbf{t}^{-c}+\varepsilon_{k},

where we plugged in the last expression. Simplifying yields (5) for some constants F,G≥0𝐹𝐺0F,G\geq 0.

## Appendix B Shape Optimization

### B.1 Hyper-parameters

Image Resolution

224 ×\times 224

Batch size

128

Preprocessing

Rescale(-1, 1)

Augmentation

InceptionCrop, Left-Right Flip

Optimizer

AdaFactor [60]

Gradient Clipping

1.0

Learning Rate

8e-4

Label Smoothing

0

Weight Decay

0.03 ×\times 8e-4

Schedule

Reverse SQRT, 10K Warmup steps, 50K Cooldown steps

Table 5 provides the set of hyperparameters used in the star and grid sweeps. We use a small batch size of 128 here in order to train multiple models in parallel on small hardware topologies.

### B.2 Star Sweep

In the star sweep, we use the center 𝐱(c)=(1968, 40, 6144)superscript𝐱𝑐1968406144\mathbf{x}^{(c)}=(1968,\,40,\,6144) as our starting point. To estimate the scaling exponents sksubscript𝑠𝑘s_{k} in (4) for each dimension separately, we vary width in the grid (608, 768, 928, 1088, 1328, 1648)608768928108813281648(608,\,768,\,928,\,1088,\,1328,\,1648), depth in the grid (8, 10, 12, 16, 20, 24)81012162024(8,\,10,\,12,\,16,\,20,\,24), and MLP dim in the grid (1088, 1360, 1728, 2160, 2592, 3072)108813601728216025923072(1088,\,1360,\,1728,\,2160,\,2592,\,3072). We train each model on 500K, 1M, and 2M steps. We always fix the patch size to 14×14141414\times 14 and the number of attention heads to 16.

### B.3 Grid Sweep

In the grid sweep, we pretrain each architecture on 600M examples. We use the cross-product of:

- 1.

width: 416, 512, 608, 768416512608768416,\,512,\,608,\,768

- 2.

depth: 6, 8, 10, 126810126,\,8,\,10,\,12

- 3.

MLP Size: 768, 928, 1088, 136076892810881360768,\,928,\,1088,\,1360

Some important considerations to be taken into account include:

- •

When designing the grid sweep, we made sure that the compute-optimal model selected lies strictly in the interior of the grid, not on its boundary. This is because if it lies at the boundary (e.g. its depth is the maximum depth used in the grid), one cannot determine if it is compute-optimal or if increasing that dimension will yield even better models. This can be an iterative process, in which additional grid points are added to the sweep if necessary.

- •

When identifying the model, we ensured that it is compute-optimal for a good range of compute (not only at some isolated point). Since the model is now compute-optimal for a range of compute budgets, we select as a starting point in our recipe the least compute it is optimal for. For example, if a model is compute-optimal for computes ranging from 1 TFLOPs to 2 TFLOPs, we use 1 TFLOPS in our recipe. In other words, we err on the side of caution, giving preference to larger models as we scale up the vision transformer (ViT).

- •

Generally, the grid sweep should be tightly packed; e.g. with increments of 20% only in each dimension. By contrast, increments in the star sweep should be large in order to identify the scaling exponents reliably.

- •

Even though the goal in the grid sweep is to identify a “small” architecture that is compute-optimal for a “small” amount of compute, the amount of compute used in the analysis should be large enough for results to be reliable and for power laws to take effect. That is why in our experiments, we use >100​Mabsent100M>100\mathrm{M} training examples in the grid sweep as opposed, for instance, to using only a few million examples.

## Appendix C Multitask Decoding Setup

Image Resolution

224 ×\times 224

Batch size

512

Preprocessing

Rescale(-1, 1), ResizeSmall(256), CentralCrop(224)

Augmentation

InceptionCrop(224), Left-Right Flip

Optimizer

AdaFactor [60]

Epochs

50

Gradient Clipping

1.0

Label Smoothing

0.1

Learning Rate

3e-4

Weight Decay

1e-4

Schedule

Cosine, 10% Warmup period

Vocabulary Size

32k

Encoder Dropout Rate

0

Decoder Dropout Rate

0.1

Table 6 summarizes the hyperparameter settings for the multitask decoding setup in Section 4.2 and Section 5.3. We always fix the decoder to 2 layers since it generally performs well [8].

## Appendix D LiT Training Setup

Image Resolution

224 ×\times 224

Batch size

32K

Preprocessing

Rescale(-1, 1)

Augmentation

None

Optimizer

AdaFactor [60]

Total Examples

900M

Gradient Clipping

1.0

Learning Rate

1e-3

Weight Decay

1e-4

Schedule

Cosine, 20% Warmup period

Vocabulary Size

32k

Bias Init

-10

Temperature Init

10

Internal Representation

1,152

Table 7 summarizes the hyperparameter settings for the locked-image text turning (LiT) setup, which is used to report zero-shot classification accuracy in Table 4. We use a large batch size of 32K in this setup because it improves the performance of contrastive training [53].

## Appendix E Transfer to ImageNet-1k

### E.1 Full model fine-tuning

Table 8 lists the settings for the ImageNet-1k fine-tuning results presented in Table 1 in the main paper.
The only three settings which differ across resolutions are learningrate decay, random augment and mixup strenghts.
We did explore various learningrates, training durations (mostly shorter) as well as Polyak averaging, although the same setting shown in the table appears to be best across the board.
Finally, we list various other settings which we did not explore. We simply used good default values from experience.

Full model fine-tuning

224 px

384 px

518 px

Learning rate decay

0.85

0.9

0.9

Random augment

-

2,10

2,10

Mixup

-

0.2

0.2

Training duration

50 k steps (20 epochs)

Learning rate

0.03

Polyak averaging (EMA)

-

Optimizer

SGD with 0.9 Momentum

Gradient clipping

1.0

Weight decay

-

Batch size

512

Learning rate schedule

Cosine with 500 steps linear warmup

Image crop

inception_crop (RandomResize)

Random flip

Horizontal

Loss

Sigmoid cross-entropy [6]

Head init

kernel=0, bias=-6.9

Train and minival splits

train[:98%] and train[98%:]

### E.2 Linear probe on frozen encoder

We take the image representation at the pre-logits, i.e. the 1152-dimensional vector that comes out of the MAP-head and feeds right into the linear classification layer.
For each of ViT-L/16, SoViT-400m/14 and ViT-g/14, we perform a grid-search over the following settings, and select the best-performing model on minival (2% of train) to be reported in Table 2:
Augmentation: resize(256)|random_crop(224) vs. inception_crop(224),
learning rate: 0.001, 0.0003, 0.0001,
epochs: 1, 3, 10,
weight decay: 0.0001, None.
It should be noted that we keep various other settings to “known good defaults” based on prior explorations with similar models (i.e. plain ViTs).
Table 9 summarizes key settings.

Linear probe at 224 px

ViT-L/16

SoViT-400m/14

ViT-g/14

Learning rate

0.001

0.0003

0.001

Weight decay

0.0001

-

-

Training duration

24.7 k steps (10 epochs)

Image crop

resize(256)|random_crop(224)

Random augment

-

Mixup

0.1

Learning rate decay

-

Polyak averaging (EMA)

-

Optimizer

SGD with 0.9 Momentum

Gradient clipping

-

Batch size

512

Learning rate schedule

Cosine with 10% linear warmup

Random flip

Horizontal

Loss

Sigmoid cross-entropy [6]

Head init

kernel=0, bias=-6.9

Train and minival splits

train[:99%] and train[99%:]

Generated on Thu Feb 29 06:32:07 2024 by LaTeXML
