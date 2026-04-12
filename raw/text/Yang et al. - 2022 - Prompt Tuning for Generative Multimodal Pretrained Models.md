# Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Yang et al. - 2022 - Prompt Tuning for Generative Multimodal Pretrained Models.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2208.02532
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Prompt Tuning for Generative Multimodal Pretrained Models

Hao Yang∗, Junyang Lin∗, An Yang, Peng Wang, Chang Zhou, Hongxia Yang 
DAMO Academy, Alibaba Group 
{yh351016, junyang.ljy, ya235025, zheluo.wp, ericzhou.zc, yang.yhx}@alibaba-inc.com

###### Abstract

Prompt tuning has become a new paradigm for model tuning and it has demonstrated success in natural language pretraining and even vision pretraining.
In this work, we explore the transfer of prompt tuning to multimodal pretraining, with a focus on generative multimodal pretrained models, instead of contrastive ones.
Specifically, we implement prompt tuning on the unified sequence-to-sequence pretrained model adaptive to both understanding and generation tasks.
Experimental results demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning and surpass other light-weight tuning methods.
Besides, in comparison with finetuned models, the prompt-tuned models demonstrate improved robustness against adversarial attacks.
We further figure out that experimental factors, including the prompt length, prompt depth, and reparameteratization, have great impacts on the model performance, and thus we empirically provide a recommendation for the setups of prompt tuning.
Despite the observed advantages, we still find some limitations in prompt tuning, and we correspondingly point out the directions for future studies. Codes are available at https://github.com/OFA-Sys/OFA

## 1 Introduction

Recent years have witnessed the great success of large-scale pretraining based on large models and big data in natural language processing (NLP) (Radford et al., 2018; Devlin et al., 2019; Yang et al., 2019; Liu et al., 2019; Raffel et al., 2020; Brown et al., 2020) and computer vision (Chen et al., 2020b, a, c; Chen and He, 2021; Bao et al., 2021; He et al., 2021b).
Mirroring the success of BERT-like models (Devlin et al., 2019), researchers have found that pretraining can level up the downstream performance of cross-modal representation learning algorithms by a large margin (Chen et al., 2020d; Lu et al., 2019; Su et al., 2020; Tan and Bansal, 2019).
Recent advances show that this idea is compatible with sequence-to-sequence (Seq2Seq) learning, and the Seq2Seq-based multimodal pretrained model can adapt to both understanding and generation tasks, and even achieve the state-of-the-art performance in a series of downstream tasks (Cho et al., 2021; Wang et al., 2021, 2022).

Despite the great success of large-scale pretrained models across multiple domains, training such models requires a large amount of computation costs.
The conventional finetuning is though effective in gaining high performance yet suffers from low training efficiency, especially when the pretrained model is of large scale in model size.
Brown et al. (2020) introduced the idea of prompt to encourage the model to generate the correct answer with a manual prompt of task instruction or a demonstration of several task examples, without further training to tune the model parameters.
This is often regarded as “in-context learning”, as the model generates responses based on the given context.
It helps large-scale pretrained language models achieve unprecedented performance in few-shot and zero-shot learning (Brown et al., 2020; Chowdhery et al., 2022; Sanh et al., 2021; Wei et al., 2021).
Inspired by this idea, researchers have moved forward to a new paradigm called prompt tuning (Li and Liang, 2021; Liu et al., 2021c; Lester et al., 2021; Liu et al., 2021a).
In comparison with finetuning, prompt tuning only tunes pretrained models by a trivial amount of parameters (e.g., 1%).
Prompt tuning freezes most parameters of the pretrained model and only tunes several prompt embeddings, as well as the output layer if necessary. Recent advances have shown that prompt tuning can help pretrained models achieve comparable performance with finetuning across different NLP downstream tasks, including natural language understanding and generation (Liu et al., 2021b; He et al., 2021a).
Such significant achievements have attracted attention of the research community of large pretrained models.

In the domains other than NLP, recent studies have also demonstrated the effectiveness of prompt tuning.
Jia et al. (2022) demonstrated that visual prompt tuning could surpass finetuning across a series of tasks, and its advantages in training efficiency were significant.
In cross-modal representation learning, the research in prompt tuning mainly focuses on the CLIP-like models (Radford et al., 2021).
CLIP is a contrastive-learning-based multimodal pretrained model, pretrained on large-scale image-text pairs.
CLIP is able to achieve outstanding performance in zero-shot image classification by turning labels to textual prompts with manual prompt templates.
To enhance the performance, Radford et al. (2021) proposed prompt ensembling by handcrafting a number of prompt templates.
However, as creating hard prompts is tedious, researchers turned to the application of soft prompts for CLIP (Rao et al., 2021; Zhou et al., 2021, 2022) or the incorporation of adapters (Gao et al., 2021; Zhang et al., 2021).
Except for the implementation on CLIP-like models, another line of work is the application of image prompts to pretrained language models for multimodal representation learning (Yao et al., 2021b; Tsimpoukelli et al., 2021).
Though the large-scale pretrained laguage model is frozen in the process of downstream transfer, it can adapt to the few-shot learning scenarios of multimodal downstream task.
Be that as it may, prompt tuning for the popular generative multimodal pretrained models, including BERT-like models and encoder-decoder pretrained models for cross-modal representation learning, is still unexplored.
Yao et al. (2022) matched the tuning paradigm to the pretraining one with manual prompts.
Yet it is still unknown whether the light-weight prompt tuning can also be effective for the generative multimodal pretrained model.

This work fills in the void and takes the lead to explore prompt tuning for the generative multimodal pretrained models.
The objective of this study is to investigate whether prompt tuning is effective for the downstream transfer of generative multimodal pretrained models, and how it benefits large pretrained models in comparison with the conventional finetuning.
To be specific, we implement the simple but effective prefix tuning, one of the most popular prompt tuning methods, on the generative multimodal pretrained model. Prefix tuning owns the advantage of simplicity but at the same time is able to achieve remarkable performance in either natural language understanding or generation (Li and Liang, 2021; Liu et al., 2021b).
In comparison with finetuning, the number of tunable parameters for prompt tuning is much smaller (~1%), leading to fewer computation costs, e.g., memory.

Through extensive experiments we observe that the light-weight prompt tuning is able to help the pretrained model achieve comparable performance with finetuning across 444 multimodal downstream tasks, spanning from understanding to generation.
To analyze the difference between finetuning and prompt tuning, we follow the assumption that prompt tuning with most parameters in the pretrained model frozen should induce model robustness.
We experiment on the tuning methods with adversarial attack and observe phenomena consistent with the hypothesis.
To make a step further, this study delves into the implementation details and investigate whether experimental factors like the prompt length, prompt depth, and reparameterization could saliently influence the final downstream performance.
We find that in general a longer prompt length (longer than 202020 tokens) is a preferable choice, and our experiments show that 646464 should be favored in most cases as a longer prompt sequence will not only increase the computation costs but also incur performance degradation.
Also, we show that reparameterizaton with additional trainable parameters cannot introduce significant improvements in downstream performance.
Finally, we reflect on the method and illustrate its defects of computation costs and training instabilities, and correspondingly, we point out some directions for the future work.

In the following sections, we briefly review the related work, deliver an introduction to prompt tuning for generative multimodal pretrained models, and report the experimental results and analysis. Lastly, we discuss the problems of prompt tuning in this scenario, point out the future work, and finally conclude this work.

## 2 Related Work

In this section, we include the review of multimodal pretraining as well as prompt tuning. We first review the studies in the two main lines of multimodal pretraining, namely generative pretraining and contrastive pretraining, and we then review researches of prompt-based learning in both NLP and cross-modal representation learning.

### 2.1 Multimodal Pretraining

The rise of vision & language pretraining started from the transfer of BERT (Devlin et al., 2019) to cross-modal representation learning. A series of studies (Lu et al., 2019; Su et al., 2020; Tan and Bansal, 2019; Chen et al., 2020d; Li et al., 2019) introduced BERT to multimodal pretraining.
The encoder-decoder framework for multimodal pretraining has recently raised attention, as a number of encoder-decoder models achieved state-of-the-art performance in the cross-modal understanding and generation tasks (Wang et al., 2021, 2022; Yu et al., 2022).
Besides, such framework allows the unification of tasks to sequence-to-sequence learning format and thus allows multitask pretraining with manual prompts (Cho et al., 2021; Wang et al., 2022).
This leads to our motivation that prompt tuning should be a perfect combination with the recent unified multimodal pretrained model and it can unleash the power of pretrained models with much fewer computation costs than the conventional finetuning.

Another trend in multimodal pretraining is contrastive learning.
The most typical constrastive pretrained model is CLIP (Radford et al., 2021).
It uses a Vision Transformer (ViT) (Dosovitskiy et al., 2021) or ResNet (He et al., 2016; Tan and Le, 2019) as the image encoder and a transformer model as the text encoder, and trains the two encoders jointly with contrastive loss (van den Oord et al., 2018).
Note that this model is pretrained on extremely large-scale data of image-text pairs.
Following CLIP, a series of studies demonstrated the success of this route of contrastive-learning-based pretraining on large-scale data (Jia et al., 2021; Yao et al., 2021a).
CLIP can achieve remarkable performance in cross-modal retrieval. What makes it really attractive is its strong performance in zeroshot classification with prompt ensembling, i.e., ensembling the outputs of the model with a handful of handcrafted prompts as the inputs.
This started the research of prompt in multimodal pretraining.

### 2.2 Prompt-based Learning

Brown et al. (2020) illustrated that large-scale pretrained models can learn from the context and perform few-shot and zero-shot learning with the prompts of task instruction or a few task examples.
Instead of using hard prompts by handcrafting, Li and Liang (2021) demonstrated that only tuning soft prompt embeddings at each layer is sufficient for the pretrained model to achieve competitive performance in natural language generation, and later a number of studies showed that prompt tuning can be essentially effective for low-resource scenarios (Liu et al., 2021c; Gu et al., 2022; Sun et al., 2022b) and it can even achieve comparable performance with finetuning (Lester et al., 2021; Liu et al., 2021b).
Following this trend, a series of modification to prompts and adapters (Hu et al., 2022; He et al., 2021a; Jiang et al., 2022; Sun et al., 2022a) for improvements in performance or training efficiency have emerged and made prompt tuning a heated topic in the whole NLP community.

Recent prompt tuning methods for multimodal pretrained models mostly serve for CLIP-like models (Zhou et al., 2021, 2022; Rao et al., 2021). Similarly, researchers tried to incorporate adapters to CLIP and also achieved satisfactory performance (Gao et al., 2021; Zhang et al., 2021).
Except for prompt tuning for CLIP-like models, another line of work explored visual prompts for frozen language models.
Tsimpoukelli et al. (2021) showed that when there is a powerful large pretrained language model, a visual encoder for prompt tuning is sufficient for multimodal few-shot learning.
To take a step forward, Alayrac et al. (2022) proposed Flamingo, a colossal multimodal model that enables in-context learning.
It could achieve state-of-the-art performance in a series of cross-modal downstream tasks in either few-shot or full-shot learning scenarios.
Such tremendous success indicates the strong potential of prompt tuning in multimodal pretraining.
In this work, we focus on an unexplored topic, prompt tuning for generative multimodal pretrained model.

## 3 Method

This section introduces the details of our proposed method. It provides the detailed implementation of prompt tuning on a unified generative multimodal pretrained model. The overall framework is illustrated in Figure 1.

### 3.1 Preliminaries

We select the unified sequence-to-sequence framework as it unifies understanding and generation tasks, and we specifically implement prompt tuning on the recent open-sourced state-of-the-art model OFA (Wang et al., 2022).
In brief, it is built with a Transformer-based (Vaswani et al., 2017) encoder-decoder framework.

Both the encoder and decoder consist of Transformer layers. To be more specific, an encoder layer consists of a multi-head self attention and a point-wise Feed-Forward Network (FFN).
To build a connection between the encoder and decoder, the Transformer decoder layer additionally contains a cross-attention module in comparison with the encoder layer.
The cross-attention is essentially multi-head attention, where the keys K𝐾K and values V𝑉V are the transformation of the encoder output states, instead of the inputs.
Such architecture can handle tasks that provide inputs of the sequence-to-sequence format.

In this work, we focus on prompt tuning for the transfer of the multimodal pretrained model. We leave the prompt learning in the stage of pretraining to the future work.

### 3.2 Prompt Tuning for Generative Multimodal Pretrained Models

In the following, we introduce our implementation details of prompt tuning on the sequence-to-sequence multimodal pretrained model. Note that our method can extend to other generative multimodal pretrained models, e.g., BERT-like models.

#### Basic Implementation

We focus on implementing prefix tuning (Li and Liang, 2021; Liu et al., 2021b) based on its outstanding performance in either natural language understanding or generation.
In comparison with the other prompt tuning methods, e.g., P-Tuning (Liu et al., 2021c), Prompt Tuning (Lester et al., 2021), PPT (Gu et al., 2022), adding soft prompt embeddings to each layer demonstrates enhanced training stability and improved downstream task performance even on relatively small models.
Specifically, for the encoder and decoder, we add tunable prompt embeddings to each layer.
Formally, we refer the pretrained model to a function ℳ​(⋅)ℳ⋅\mathcal{M}(\cdot), and the generation function of the prompt embeddings to 𝒢​(⋅)𝒢⋅\mathcal{G}(\cdot).
The formulation is demonstrated below:
\linenomathAMS

y𝑦\displaystyle y
=ℳ​(𝒢​(L,l),x),absentℳ𝒢𝐿𝑙𝑥\displaystyle=\mathcal{M}(\mathcal{G}(L,l),x),

(1)

where x𝑥x refers to the multimodal inputs, L𝐿L refers to the number of layers, and l𝑙l refers to the prompt length, which should be predefined by a hyperparameter.
At each layer, we prefix soft prompt embeddings p(i)superscript𝑝𝑖p^{(i)} to the input hidden states h(i)superscriptℎ𝑖h^{(i)}
Note that we only prefix prompt embeddings at Transformer layers.
In the simplest practice, the prompt generator 𝒢𝒢\mathcal{G} is a sparse embedding matrix of ℝL×l×hsuperscriptℝ𝐿𝑙ℎ\mathbb{R}^{L\times l\times h}, and we select the corresponding embedding at the i𝑖i-th index and the j𝑗j-th layer as the prompt embedding. Below we provide an illustration of some more complex implementations, and we compare those methods in this study.

In the downstream tuning process, we only tune the newly added prompt embeddings at each layer and keep the parameters of the large pretrained model frozen. Therefore, while there are only a small amount of parameters that need to be updated, e.g., 1%, the computation costs are far fewer than those of finetuning.

#### Reparameterization

Except for the simplest implementation of adding a sparse embedding matrix at each layer, a more complex one should be adding an encoder, e.g., an MLP layer, to reparameterize prompt embeddings.
We also investigate the influence of reparameterization in this context.

#### Prompt Length

Similar to previous studies (Li and Liang, 2021; Liu et al., 2021b), we find that the length of prompt embeddings make a great difference in different downstream tasks.
In this study, we investigate how this factor imposes influence on model performance in different downstream tasks.

#### Prompt Depth

To investigate the impacts of the place of prompt embedding insertion, we delve into the issue of prompt depth. Specifically, we simplify it to adding prompt embeddings to the encoder or decoder only, as well as to both modules.

## 4 Experiments

Model
RefCOCO
RefCOCO+
RefCOCOg
SNLI-VE
COCO Captions
VQA

val
testA
testB
val
testA
testB
val-u
test-u
dev
test
B@4
M
C
S
test-dev
test-std

Base-size Models

Finetuning
88.48
90.67
83.30
81.39
87.15
74.29
82.29
82.31
89.30
89.20
41.00
30.90
138.2
24.20
78.00
78.10

Prompt Tuning
84.53
85.21
77.36
76.34
81.44
67.68
75.61
76.57
88.18
88.59
39.70
30.10
134.2
23.50
74.31
74.47

Large-size Models

Finetuning
90.05
92.93
85.26
85.80
89.87
79.22
85.89
86.55
90.30
90.20
42.40
31.50
142.2
24.50
80.40
80.70

Prompt Tuning
90.05
92.31
85.59
84.54
89.40
77.77
85.27
85.89
90.04
90.12
41.81
31.51
141.4
24.42
78.30
78.53

To validate the effectiveness of prompt tuning for multimodal pretrained models, we conduct experiments on the conventional cross-modal tasks.
Specifically, we experiment on cross-modal understanding and generation, including referring expression comprehension, visual entailment, image captioning, and visual question answering (VQA).
We use the mostly-used base-size and large-size models for the experiments, whose sizes are around 180180180M and 470470470M respectively. We provide more details about the experimental setups in the Appendix A.1.

### 4.1 Datasets & Metrics

#### Referring Expression Comprehension

We conduct experiments on the 333 subtasks of referring expression comprehension, namely RefCOCO, RefCOCO+, and RefCOCOg (Yu et al., 2016; Mao et al., 2016).
This task requires the model to generate a correct bounding box that answers the given text query on a provided image.
We use Acc@0.5 as the evaluation metric.

#### Image Captioning

We evaluate the image captioning capability of our method on the Microsoft COCO Image Captioning dataset (Chen et al., 2015).
In this task, the model should generate a description that corresponds to the information of the given image.
We use BLEU@4 (Papineni et al., 2002), METEOR (Lavie and Agarwal, 2007), CIDEr (Vedantam et al., 2015), and SPICE (Anderson et al., 2016) as the evaluation metrics.

#### Visual Entailment

To evaluate the performance of entailment, we implement the experiments on SNLI-VE (Xie et al., 2019).
Given an image and a text, the model should figure out their relations, whether they are entailment, contradiction, or neutrality.
We follow the setups in (Wang et al., 2022) and add the given premise to the input. We use accuracy as the evaluation metric.

#### VQA

We implement our experiments on VQA 2.0 (Antol et al., 2015; Goyal et al., 2017).
This task requires the model to generate the correct answer based on an image and a question about certain information on the image.
Following Wang et al. (2022), we use the all-candidate evaluation, which requires the model to generate a probability for each candidate among the 3,12931293,129 most frequent answers.
We use accuracy as the evaluation metric.

### 4.2 Experimental Results

Below we provide the detailed experiment results, including the comparison of prompt tuning and finetuning, as well as prompt tuning and other parameter-efficient tuning methods.

Method
RefCOCO
RefCOCO+
RefCOCOg
SNLI-VE
COCO Captions
VQA

val
testA
testB
val
testA
testB
val-u
test-u
dev
test
B@4
M
C
S
test-dev
test-std

Bitfit
89.61
92.20
84.91
82.60
88.08
75.16
84.66
84.68
89.70
89.42
41.02
30.92
138.8
24.23
78.23
78.44

Adapter
90.01
92.30
85.02
83.79
88.93
76.09
85.10
85.45
89.84
89.78
41.38
31.16
139.5
24.30
78.27
78.47

Prompt Tuning
90.05
92.31
85.59
84.54
89.40
77.77
85.27
85.89
90.04
90.12
41.81
31.51
141.4
24.42
78.30
78.53

#### Comparison with Finetuning

We demonstrate the experimental results of the 444 tasks in Table 1.
In general, for the base-size model, prompt tuning underperforms finetuning by significant margins, but for the large-size model, prompt tuning is able to achieve comparable performance.
To be more specific, in the evaluation of referring expression comprehension, for the base-size model, prompt tuning significantly underperforms finetuning by lagging behind a large margin of 5.645.645.64 on average across RefCOCO, RefCOCO+, and RefCOCOg, but for the large-size model, prompt tuning only slightly underperforms finetuning by a small margin of 0.590.590.59.
In the evaluation of visual entailment, the gap between the algorithms is closer, which is around 0.170.170.17.
In the evaluation with the CIDEr score on image captioning, for the base-size model, prompt tuning underperforms finetuning by a margin of 4.04.04.0, but for the large-size model, the performance gap is only 0.80.80.8.
In the evaluation of VQA, for the base-size model the performance gap is 3.633.633.63 between prompt tuning and finetuning, and for the large-size model the gap is 2.172.172.17 on the test-std set. Different from the other tasks, even in the experiments on the large-size model, the gap is still significant.
We hypothesize that it is still necessary to search a better hyperparameter setup for this task due to the sensitivity of prompt tuning to hyperparameters.

#### Comparison with Other Parameter-Efficient Tuning Methods

We additionally add a comparison with two parameter-efficient tuning methods, namely Adapter (Houlsby et al., 2019) and BitFit (Zaken et al., 2022) to test whether prompt tuning is the best solution of light-weight transfer.
Table 2 shows the results of different light-weight tuning methods implemented on the aforementioned datasets.
In all the downstream tasks, prompt tuning surpasses the performance of Adapter and BitFit.

Method
RefCOCO
RefCOCO+
RefCOCOg
SNLI-VE
COCO Captions
VQA

val
testA
testB
val
testA
testB
val-u
test-u
dev
test
B@4
M
C
S
test-dev
test-std

Encoder
89.48
91.71
84.98
84.50
89.22
77.71
85.07
85.58
89.64
89.70
41.39
31.08
141.1
24.34
78.10
78.26

Decoder
88.90
91.28
84.32
83.46
88.24
76.82
84.54
85.02
88.56
88.71
40.08
30.43
140.8
24.06
77.84
78.03

Encoder+Decoder
90.05
92.31
85.59
84.54
89.40
77.77
85.27
85.89
90.04
90.12
41.81
31.51
141.4
24.42
78.30
78.53

Method
RefCOCO
RefCOCO+
RefCOCOg
SNLI-VE
COCO Captions
VQA 
xqx

val
testA
testB
val
testA
testBxq
val-u
test-u
dev
test
B@4
M
C
S
test-dev
test-std

Prompt Tuning
90.05
92.31
85.59
84.54
89.40
77.77
85.27
85.89
90.04
90.12
41.81
31.51
141.4
24.42
78.30
78.53

MLP
90.12
92.56
85.63
84.83
89.65
77.94
85.42
86.01
89.98
90.02
41.67
31.48
140.7
24.40
78.26
78.48

### 4.3 Analyses

In this section, we move forward to analyzing prompt tuning in multimodal pretraining.
Specifically, we examine the robustness of prompt tuning based on the assumption that keeping most parameters of the pretrained model frozen should lead to improved robustness to adversarial attack.
Also, we evaluate how different setups of prompt tuning, say the prompt length, the depth of prompt, and reparameterization, influence the downstream performance, and try to provide a recommended setup for consistently better performance.

#### Robustness Analysis

To test whether the pretrained model with prompt tuning for downstream transfer is robust, we conduct experiments of adversarial attack for the examination.
Adversarial attack was first proposed in computer vision, which revealed the vulnerability of deep learning models. The most common adversarial attack methods in computer vision are gradient-based methods, such as FGSM (Goodfellow et al., 2014), PGD (Madry et al., 2017), MIM (Dong et al., 2017) and SI (Lin et al., 2019).
Most of the typical unimodal adversarial attack on tasks are gradient-based methods.
Among them, we select FGSM, which requires only one step of gradient computation on text and image embeddings.
Experimental results are demonstrated in Figure 3.
Prompt tuning consistently demonstrates better robustness in comparison with finetuning across all tasks.
This confirms our hypothesis and also shows one significant advantage of prompt tuning not reflected in the standard evaluation.
In practice, if model vulnerability is a issue that matters, we recommend the application of prompt tuning for the enhanced robustness without significant performance degradation.

#### Prompt Length

To study the effects of the prompt length on the final downstream performance, we evaluate the prompt tuning performance on the downstream tasks with a prompt length selected from {10,16,30,64,100,120}10163064100120\{10,16,30,64,100,120\}.
As shown in Figure 2, a general tendency is that a longer prompt length with more parameters to tune can encourage improvements in downstream performance across the tasks.
However, we observe diminishing marginal utility and a prompt too long may even negatively impact the performance.
Although the best prompt length for tasks are different, we empirically advise that the length of 646464 tokens can achieve a better performance on average. See Appendix A.2 for more details.

#### Prompt Depth

As we base our implementation on the encoder-decoder model, we intuitively assume that where to insert prompt embeddings matters the performance.
To simplify this issue, in our practice, we evaluate the performance of inserting prompts to the encoder only, to the decoder only, or to both the encoder and decoder.
Experimental results are demonstrated in Table 3.
We find that it is best to insert prompts to every layer of the whole Transformer model, though compared with the other alternatives it is less computation-efficient.
In the comparison between insertion to the encoder only and to the decoder only, we observe that the former solution leads to a significantly better results across multiple downstream tasks.
This suggests that the insertion of prompts to the bottom layers might contribute more to the success of downstream transfer.

#### Reparameterization

Empirically, directly updating the trainable embeddings leads to unstable optimization and a slight drop in performance.
Prior work usually leveraged an encoder, e.g., an MLP (Li and Liang, 2021), to reparameterize the trainable embeddings. We evaluate the performance of reparameterization, and we demonstrate the experimental results in Table 4. For some datasets (e.g., RefCOCO and RefCOCOg), MLP brings consistent improvements.
For the others, MLP leads to relatively negative impacts (e.g., SNLI-VE and VQA).
Thus we cannot come to a conclusion about which should be a preferable one. To achieve better performance on a specific dataset, it is still necessary to make an attempt on both methods.

## 5 Discussion

Apparently we have to admit that prompt tuning still cannot replace finetuning, and here we try to illustrate some of its limitations and point out some directions for future work.

A salient problem is its slow convergence.
Though prompt tuning has noticeable advantages in training efficiency,
it costs at least 404040 epochs for prompt tuning to achieve the nearly best performance on some datasets (e.g., RefCOCO).
A larger number of training epochs may incur more computation costs though prompt tuning has an advantage in training efficiency compared with finetuning.
We demonstrate more details in Appendix A.2.
This indicates that finding a better solution for fast and stable convergence is also important besides reaching comparable or even improved performance over the conventional finetuning.

Another common defect of prompt tuning is the difficulty in searching for a suitable hyperparamter setup. The hyperparameter tuning experience in finetuning is not suitable for prompt tuning. Fortunately, we find that prompt tuning for generative multimodal pretrained models is not as sensitive to hyperparameters as prompt tuning for pretrained language models. We provide details of hyperparameter setups in Appendix A.1.

Despite the aforementioned limitations, prompt tuning demonstrates significantly better robustness against adversarial attack.
In the future, we should pay more attention to this merit and find ways to leverage it.

## 6 Conclusion

In this work, we explore prompt tuning for generative multimodal pretrained models.
Through extensive experiments, we demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning with much fewer parameters to tune (e.g., 1%), and it can surpass other light-weight tuning methods, e.g., Adapter and BitFit.
Through our analysis, we figure out a significant advantage of prompt tuning about its robustness against adversarial attack.
Furthermore, we provide a comprehensive analysis about the influence of prompt tuning setups, including the prompt length, prompt depth, and reparameterization.
Potentially prompt tuning can be an alternative to finetuning, but still, there are some salient limitations in this method, e.g., slow convergence and training instabilities.
We hope that future studies in this field can alleviate the aforementioned problems and thus promote the application of prompt tuning.

## References

- Alayrac et al. (2022)

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr,
Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds,
Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina
Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew
Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo
Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. 2022.

Flamingo: a visual language model for few-shot learning.

CoRR, abs/2204.14198.

- Anderson et al. (2016)

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. 2016.

SPICE: semantic propositional image caption evaluation.

In ECCV 2016, volume 9909 of Lecture Notes in Computer
Science, pages 382–398. Springer.

- Antol et al. (2015)

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra,
C. Lawrence Zitnick, and Devi Parikh. 2015.

VQA: visual question answering.

In ICCV 2015, pages 2425–2433. IEEE Computer Society.

- Bao et al. (2021)

Hangbo Bao, Li Dong, and Furu Wei. 2021.

Beit: Bert pre-training of image transformers.

arXiv preprint arXiv:2106.08254.

- Brown et al. (2020)

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom
Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens
Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott
Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.

Language models are few-shot learners.

In NeurIPS 2020.

- Chen et al. (2020a)

Ting Chen, Simon Kornblith, Kevin Norouzi, Mohammad Swersky, and Geoffrey
Hinton. 2020a.

Big self-supervised models are strong semi-supervised learners.

In NeurIPS 2020, pages 10466–10478. PMLR.

- Chen et al. (2020b)

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
2020b.

A simple framework for contrastive learning of visual
representations.

In ICML 2020, pages 1597–1607. PMLR.

- Chen et al. (2020c)

Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. 2020c.

Improved baselines with momentum contrastive learning.

arXiv preprint arXiv:2003.04297.

- Chen et al. (2015)

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta,
Piotr Dollár, and C. Lawrence Zitnick. 2015.

Microsoft COCO captions: Data collection and evaluation server.

CoRR, abs/1504.00325.

- Chen and He (2021)

Xinlei Chen and Kaiming He. 2021.

Exploring simple siamese representation learning.

In CVPR 2021, pages 15750–15758.

- Chen et al. (2020d)

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan,
Yu Cheng, and Jingjing Liu. 2020d.

UNITER: universal image-text representation learning.

In ECCV 2020, volume 12375 of Lecture Notes in
Computer Science, pages 104–120. Springer.

- Cho et al. (2021)

Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. 2021.

Unifying vision-and-language tasks via text generation.

In ICML 2021, volume 139 of Proceedings of Machine
Learning Research, pages 1931–1942. PMLR.

- Chowdhery et al. (2022)

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra,
Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian
Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez,
Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran,
Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob
Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm
Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia,
Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David
Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David
Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai,
Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica
Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi
Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei,
Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah
Fiedel. 2022.

Palm: Scaling language modeling with pathways.

CoRR, abs/2204.02311.

- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

BERT: pre-training of deep bidirectional transformers for language
understanding.

In NAACL-HLT 2019, pages 4171–4186. Association for
Computational Linguistics.

- Dong et al. (2017)

Yinpeng Dong, Fangzhou Liao, Tianyu Pang, Hang Su, Jun Zhu, Xiaolin Hu, and
Jianguo Li. 2017.

Boosting adversarial attacks with momentum.

CoRR, abs/1710.06081.

- Dosovitskiy et al. (2021)

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021.

An image is worth 16x16 words: Transformers for image recognition at
scale.

In ICLR 2021. OpenReview.net.

- Gao et al. (2021)

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang,
Hongsheng Li, and Yu Qiao. 2021.

Clip-adapter: Better vision-language models with feature adapters.

CoRR, abs/2110.04544.

- Goodfellow et al. (2014)

Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014.

Explaining and harnessing adversarial examples.

CoRR, abs/1412.6572.

- Goyal et al. (2017)

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh.
2017.

Making the V in VQA matter: Elevating the role of image
understanding in visual question answering.

In CVPR 2017, pages 6325–6334. IEEE Computer Society.

- Gu et al. (2022)

Yuxian Gu, Xu Han, Zhiyuan Liu, and Minlie Huang. 2022.

PPT: pre-trained prompt tuning for few-shot learning.

In ACL 2022, pages 8410–8423. Association for Computational
Linguistics.

- He et al. (2021a)

Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham
Neubig. 2021a.

Towards a unified view of parameter-efficient transfer learning.

CoRR, abs/2110.04366.

- He et al. (2021b)

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross
Girshick. 2021b.

Masked autoencoders are scalable vision learners.

arXiv preprint arXiv:2111.06377.

- He et al. (2016)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016.

Deep residual learning for image recognition.

In CVPR 2016, pages 770–778.

- Houlsby et al. (2019)

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.

Parameter-efficient transfer learning for NLP.

In ICML 2019, volume 97 of Proceedings of Machine
Learning Research, pages 2790–2799. PMLR.

- Hu et al. (2022)

Shengding Hu, Ning Ding, Huadong Wang, Zhiyuan Liu, Jingang Wang, Juanzi Li,
Wei Wu, and Maosong Sun. 2022.

Knowledgeable prompt-tuning: Incorporating knowledge into prompt
verbalizer for text classification.

In ACL 2022, pages 2225–2240. Association for Computational
Linguistics.

- Jia et al. (2021)

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V
Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. 2021.

Scaling up visual and vision-language representation learning with
noisy text supervision.

arXiv preprint arXiv:2102.05918.

- Jia et al. (2022)

Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge J. Belongie,
Bharath Hariharan, and Ser-Nam Lim. 2022.

Visual prompt tuning.

CoRR, abs/2203.12119.

- Jiang et al. (2022)

Yuezihan Jiang, Hao Yang, Junyang Lin, Hanyu Zhao, An Yang, Chang Zhou, Hongxia
Yang, Zhi Yang, and Bin Cui. 2022.

Instance-wise prompt tuning for pretrained language models.

CoRR, abs/2206.01958.

- Lavie and Agarwal (2007)

Alon Lavie and Abhaya Agarwal. 2007.

METEOR: an automatic metric for MT evaluation with high levels of
correlation with human judgments.

In WMT@ACL 2007, pages 228–231. Association for Computational
Linguistics.

- Lester et al. (2021)

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.

The power of scale for parameter-efficient prompt tuning.

In EMNLP 2021, pages 3045–3059. Association for
Computational Linguistics.

- Li et al. (2019)

Gen Li, Nan Duan, Yuejian Fang, Daxin Jiang, and Ming Zhou. 2019.

Unicoder-vl: A universal encoder for vision and language by
cross-modal pre-training.

CoRR, abs/1908.06066.

- Li and Liang (2021)

Xiang Lisa Li and Percy Liang. 2021.

Prefix-tuning: Optimizing continuous prompts for generation.

In ACL/IJCNLP 2021, pages 4582–4597. Association for
Computational Linguistics.

- Lin et al. (2019)

Jiadong Lin, Chuanbiao Song, Kun He, Liwei Wang, and John E. Hopcroft. 2019.

Nesterov accelerated gradient and scale invariance for adversarial
attacks.

CoRR, abs/1908.06281.

- Liu et al. (2021a)

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and
Graham Neubig. 2021a.

Pre-train, prompt, and predict: A systematic survey of prompting
methods in natural language processing.

CoRR, abs/2107.13586.

- Liu et al. (2021b)

Xiao Liu, Kaixuan Ji, Yicheng Fu, Zhengxiao Du, Zhilin Yang, and Jie Tang.
2021b.

P-tuning v2: Prompt tuning can be comparable to fine-tuning
universally across scales and tasks.

CoRR, abs/2110.07602.

- Liu et al. (2021c)

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and
Jie Tang. 2021c.

GPT understands, too.

CoRR, abs/2103.10385.

- Liu et al. (2019)

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019.

Roberta: A robustly optimized BERT pretraining approach.

CoRR, abs/1907.11692.

- Lu et al. (2019)

Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019.

Vilbert: Pretraining task-agnostic visiolinguistic representations
for vision-and-language tasks.

In NeurIPS 2019, pages 13–23.

- Madry et al. (2017)

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and
Adrian Vladu. 2017.

Towards deep learning models resistant to adversarial attacks.

CoRR, abs/1706.06083.

- Mao et al. (2016)

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L. Yuille, and
Kevin Murphy. 2016.

Generation and comprehension of unambiguous object descriptions.

In CVPR 2016, pages 11–20. IEEE Computer Society.

- Papineni et al. (2002)

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.

Bleu: a method for automatic evaluation of machine translation.

In ACL 2002, pages 311–318.

- Radford et al. (2021)

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021.

Learning transferable visual models from natural language
supervision.

In ICML 2021, volume 139 of Proceedings of Machine
Learning Research, pages 8748–8763. PMLR.

- Radford et al. (2018)

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018.

Improving language understanding by generative pre-training.

- Raffel et al. (2020)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020.

Exploring the limits of transfer learning with a unified text-to-text
transformer.

J. Mach. Learn. Res., 21:140:1–140:67.

- Rao et al. (2021)

Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang,
Jie Zhou, and Jiwen Lu. 2021.

Denseclip: Language-guided dense prediction with context-aware
prompting.

CoRR, abs/2112.01518.

- Sanh et al. (2021)

Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika,
Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja,
Manan Dey, M. Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza
Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta,
Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen,
Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj,
Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan
Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and
Alexander M. Rush. 2021.

Multitask prompted training enables zero-shot task generalization.

CoRR, abs/2110.08207.

- Su et al. (2020)

Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai.
2020.

VL-BERT: pre-training of generic visual-linguistic representations.

In ICLR 2020. OpenReview.net.

- Sun et al. (2022a)

Tianxiang Sun, Zhengfu He, Hong Qian, Xuanjing Huang, and Xipeng Qiu.
2022a.

Bbtv2: Pure black-box optimization can be comparable to gradient
descent for few-shot learning.

CoRR, abs/2205.11200.

- Sun et al. (2022b)

Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, and Xipeng Qiu.
2022b.

Black-box tuning for language-model-as-a-service.

In ICML 2022, volume 162 of Proceedings of Machine
Learning Research, pages 20841–20855. PMLR.

- Tan and Bansal (2019)

Hao Tan and Mohit Bansal. 2019.

LXMERT: learning cross-modality encoder representations from
transformers.

In EMNLP-IJCNLP 2019, pages 5099–5110. Association for
Computational Linguistics.

- Tan and Le (2019)

Mingxing Tan and Quoc V. Le. 2019.

Efficientnet: Rethinking model scaling for convolutional neural
networks.

In ICML 2019, volume 97 of Proceedings of Machine
Learning Research, pages 6105–6114. PMLR.

- Tsimpoukelli et al. (2021)

Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals,
and Felix Hill. 2021.

Multimodal few-shot learning with frozen language models.

In NeurIPS 2021, pages 200–212.

- van den Oord et al. (2018)

Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018.

Representation learning with contrastive predictive coding.

CoRR, abs/1807.03748.

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.

Attention is all you need.

In NeurIPS 2017, pages 5998–6008.

- Vedantam et al. (2015)

Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. 2015.

Cider: Consensus-based image description evaluation.

In CVPR 2015, pages 4566–4575. IEEE Computer Society.

- Wang et al. (2022)

Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma,
Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022.

Unifying architectures, tasks, and modalities through a simple
sequence-to-sequence learning framework.

CoRR, abs/2202.03052.

- Wang et al. (2021)

Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao.
2021.

Simvlm: Simple visual language model pretraining with weak
supervision.

CoRR, abs/2108.10904.

- Wei et al. (2021)

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian
Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2021.

Finetuned language models are zero-shot learners.

CoRR, abs/2109.01652.

- Xie et al. (2019)

Ning Xie, Farley Lai, Derek Doran, and Asim Kadav. 2019.

Visual entailment: A novel task for fine-grained image
understanding.

CoRR, abs/1901.06706.

- Yang et al. (2019)

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov,
and Quoc V. Le. 2019.

Xlnet: Generalized autoregressive pretraining for language
understanding.

In NeurIPS 2019, pages 5754–5764.

- Yao et al. (2021a)

Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan
Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. 2021a.

FILIP: fine-grained interactive language-image pre-training.

CoRR, abs/2111.07783.

- Yao et al. (2022)

Yuan Yao, Qianyu Chen, Ao Zhang, Wei Ji, Zhiyuan Liu, Tat-Seng Chua, and
Maosong Sun. 2022.

PEVL: position-enhanced pre-training and prompt tuning for
vision-language models.

CoRR, abs/2205.11169.

- Yao et al. (2021b)

Yuan Yao, Ao Zhang, Zhengyan Zhang, Zhiyuan Liu, Tat-Seng Chua, and Maosong
Sun. 2021b.

CPT: colorful prompt tuning for pre-trained vision-language models.

CoRR, abs/2109.11797.

- Yu et al. (2022)

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and
Yonghui Wu. 2022.

Coca: Contrastive captioners are image-text foundation models.

CoRR, abs/2205.01917.

- Yu et al. (2016)

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg.
2016.

Modeling context in referring expressions.

In ECCV 2016, volume 9906 of Lecture Notes in Computer
Science, pages 69–85. Springer.

- Zaken et al. (2022)

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. 2022.

Bitfit: Simple parameter-efficient fine-tuning for transformer-based
masked language-models.

In ACL 2022, pages 1–9. Association for Computational
Linguistics.

- Zhang et al. (2021)

Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai,
Yu Qiao, and Hongsheng Li. 2021.

Tip-adapter: Training-free clip-adapter for better vision-language
modeling.

CoRR, abs/2111.03930.

- Zhou et al. (2021)

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2021.

Learning to prompt for vision-language models.

CoRR, abs/2109.01134.

- Zhou et al. (2022)

Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022.

Conditional prompt learning for vision-language models.

CoRR, abs/2203.05557.

## Appendix A Appendix

### A.1 Experimental Setups

Referring Expression Comprehension Referring expression comprehension requires models to locate an image region described by a language query. We perform experiments on RefCOCO (Yu et al., 2016), RefCOCO+ (Yu et al., 2016), and RefCOCOg (Mao et al., 2016).
We report the standard metric Acc@0.5 on the validation and test sets.
For finetuning, the batch
size is set to 128128128, the learning rate is set to 0.030.030.03, and the prompt length varies from 101010–120120120.

Visual Entailment Visual entailment requires the model to evaluate the semantic relation between the given image and text, i.e., entailment, neutrality, or contradiction. We perform experiments on the SNLI-VE (Xie et al., 2019) dataset.
We report accuracy on both dev and test sets.
The model is finetuned with a learning rate of 0.030.030.03 and a batch size of 128128128. The prompt length varies from 101010–120120120.

Image Captioning Image captioning is a standard vision & language task that requires models to generate an appropriate and fluent caption for an image.
We report BLEU@4 (Papineni et al., 2002), METEOR (Lavie and Agarwal, 2007), CIDEr (Vedantam et al., 2015), and SPICE (Anderson et al., 2016) scores on the Karpathy test split.
We finetune the model with a learning rate of 0.030.030.03, a batch size of 256256256, and a prompt length varying from 101010–120120120.
We only finetune the model with cross-entropy loss, without further CIDEr optimization.

Visual Question Answering Visual question answering (Antol et al., 2015; Goyal et al., 2017) is a cross-modal task that requires the models to answer the question given an image.
We conduct experiments on VQA 2.0 and report the score on the test-std set.
For finetuning, the batch size is set to 256256256 and the learning rate is set to 0.030.030.03. Exponential Moving Average (EMA) with a decay rate of 0.99990.99990.9999 is employed in finetuning. The prompt length varies from 101010–120120120.

### A.2 Additional Experimental Results

In this section, we provide more experimental results for comprehensive understanding of the performance of prompt tuning.

Below we summarize the detailed performance of prompt tuning on the downstream tasks in the conditions of different prompt lengths. See Table 6. On average, a prompt length of 646464 helps achieve the best average performance in the downstream tasks.

Method
Fintuning
Prompt Tuning

RefCOCO
40.00
77.44

SNLI-VE
80.96
164.48

COCO Captions
29.60
16.16

VQA
616.16
455.52

Length
10
16
32
64
100
120

Score
91.84
91.29
91.94
92.29
92.10
91.93

To evaluate the training efficiency of different methods, we experiment on the base model OFA of different sizes, spanning from 939393M to 930930930M paramters.
Figure 4 demonstrates their performance in efficiency by evaluating their used time of processing 100100100 samples.
We find that prompt tuning consistently performs better than finetuning in training efficiency.
For the huge-size model, it can perform around 222 times faster than finetuning.
However, based on our observation, the advantage in training efficiency does not lead to less required computation resource.
Table 5 lists the detailed computation resource consumption of both finetuning and prompt tuning.
Specifically, we compute the computation resource consumption by calculating the GPU-hours of finetuning and prompt tuning on different tasks.
We find that for image captioning and VQA, prompt tuning consumes less resource, but for the other tasks prompt tuning adversely consumes more.
It reflects that for tasks similar to pretraining tasks, especially those with more data in the pretraining stage, prompt tuning is able to outperform finetuning, but for others, prompt tuning even incurs more carbon footprints.
This indicates that the real computation resource consumption for downstream transfer should be an important issue in the field of prompt tuning and the solution to this problem can further the developments of the application.

Generated on Wed Mar 13 19:21:48 2024 by LaTeXML
