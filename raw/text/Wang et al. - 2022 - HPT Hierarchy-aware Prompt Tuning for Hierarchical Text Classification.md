# Wang et al. - 2022 - HPT Hierarchy-aware Prompt Tuning for Hierarchical Text Classification

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Wang et al. - 2022 - HPT Hierarchy-aware Prompt Tuning for Hierarchical Text Classification.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2204.13413
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Hpt: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification

Zihan Wang1222Equal contribution.  Peiyi Wang1222Equal contribution.  Tianyu Liu2  Binghuai Lin2 
Yunbo Cao2  Zhifang Sui1  Houfeng Wang1111Corresponding author.

1 MOE Key Laboratory of Computational Linguistics, Peking University, China 
2 Tencent Cloud Xiaowei 
{wangzh9969, wangpeiyi9979}@gmail.com; {szf, wanghf}@pku.edu.cn
{rogertyliu, binghuailin, yunbocao}@tencent.com

###### Abstract

Hierarchical text classification (HTC) is a challenging subtask of multi-label classification due to its complex label hierarchy.
Recently, the pretrained language models (PLM)
have been widely adopted in HTC through a fine-tuning paradigm.
However, in this paradigm, there exists a huge gap between the classification tasks with sophisticated label hierarchy and the masked language model (MLM) pretraining tasks of PLMs and thus the potentials of PLMs can not be fully tapped.
To bridge the gap, in this paper, we propose Hpt, a Hierarchy-aware Prompt Tuning method to handle HTC from a multi-label MLM perspective.
Specifically, we construct a dynamic virtual template and label words that take the form of soft prompts to fuse the label hierarchy knowledge and introduce a zero-bounded multi-label cross entropy loss to harmonize the objectives of HTC and MLM.
Extensive experiments show Hpt achieves state-of-the-art performances on 333 popular HTC datasets and is adept at handling the imbalance and low resource situations. Our code is available at https://github.com/wzh9969/HPT.

## 1 Introduction

Hierarchical text classification (HTC) aims to categorize a text into a set of labels with a structured class hierarchy (commonly modeled as a tree) Silla and Freitas (2011).
HTC is a multi-label text classification problem, where the classification result corresponds to one or more paths of the hierarchy Zhou et al. (2020).
The major challenge of HTC is to model the large-scale, imbalanced, and structured label hierarchy Mao et al. (2019).

As shown in Figure 1(a), existing state-of-the-art HTC models Zhou et al. (2020); Deng et al. (2021); Chen et al. (2021); Zhao et al. (2021) separately extract text and label hierarchy features by utilizing text and graph encoders, and then fuse the two sources of features into a final representation for text classification.
Specifically, Chen et al. (2021) takes the advantages of powerful pretrained language models (PLMs) in HTC through a fine-tuning paradigm, where they use PLMs as the text encoder.
In this paradigm, the PLMs are trained to inference with complex label hierarchy.

Despite the success of the fine-tuning paradigm, some recent studies suggest that it may suffer from distinct training strategies in the pretraining and fine-tuning stages, which restrains the finetuned models to take full advantage of knowledge in PLMs Chen et al. (2022).
Therefore, a new paradigm known as prompt tuning is proposed to bridge the gap between the downstream tasks and the pretraining tasks of PLMs, which can tap the full potential of PLMs.
By warping the text
(e.g., “x”)
into the model input
(e.g., “x is [MASK]” )
and taming the PLMs to complete the masked cloze test,
prompt tuning has achieved promising performances on the flat text classification where labels have no hierarchy Shin et al. (2020).

How about the performances of the prompt tuning in HTC?
In the pilot study, we test flat prompt tuning methods on HTC and surprisingly find that they are even comparable with the state-of-the-art models in HTC.
This result suggests that the expressive power of PLMs has been undermined in the prior HTC methods due to the pretraining-finetuning gap.
Although the flat prompt tuning methods have somewhat narrowed the gap, there still remain two challenges while combining PLMs with HTC.

- 1.

hierarchy and flat gap.
Labels of HTC lie on a sophisticated hierarchy while MLM pretraining and flat prompt tuning do not take label hierarchy into consideration.

- 2.

multi-label and multi-class gap.
HTC is a multi-label classification problem where the output labels are interconnected with a hierarchy while MLM pretraining is formulated as multi-class classification.

To bridge these two gaps, as shown in Figure 1(b), we propose a hierarchy-aware prompt tuning (Hpt) method that solves HTC from a multi-label MLM perspective.
In detail, to bridge the hierarchy and flat gap,
we incorporate the label hierarchy knowledge into soft prompts with continuous representation.
Specifically, we incorporate the depth and width information in the label hierarchy into different virtual template words, which is helpful to alleviate the label imbalance problem as verified by our experiments.
To bridge the multi-label and multi-class gap, we transform HTC into a multi-label MLM problem by a zero bounded multi-label cross entropy loss which continually seeks to increase the score of the correct label
and decrease the score of the incorrect labels.

We summarize our contributions as follows:

- •

We propose a hierarchy-aware prompt tuning (Hpt) method for hierarchical text classification. To the best of our knowledge, this is the first investigation on flat and hierarchical prompt tuning in HTC.

- •

We summarize two challenging gaps between HTC and masked language modeling (MLM). To bridge these gaps, we transform HTC into a hierarchy-aware multi-label MLM problem.

- •

Extensive experiments demonstrate that our proposed model achieves the new state-of-the-art results on three popular datasets, and is adept at handling label imbalance and low resource situations.

## 2 Related Work

### 2.1 Hierarchical Text Classification

Hierarchical text classification (HTC) is a challenge task due to its large-scale, imbalanced, and structured label hierarchy Mao et al. (2019).
Existing work for HTC could be categorized into local and global approaches based on their ways of utilizing the label hierarchy Zhou et al. (2020): local approaches build classifiers for each node or level while the global ones build only one classifier for the entire graph. Although early works on HTC mainly focus on local approaches Wehrmann et al. (2018); Shimura et al. (2018); Banerjee et al. (2019), global approaches soon become mainstream.
The early global approaches neglect the hierarchical structure of labels and view the problem as a flat multi-label classification Johnson and Zhang (2015). Later on, some work try to coalesce the label structure by meta-learning Wu et al. (2019), reinforcement learning Mao et al. (2019), and attention module Zhang et al. (2021). Although such methods can capture the hierarchical information, Zhou et al. (2020) demonstrate that encode the holistic label structure directly by a structure encoder can further improve performance. Following this research, a bunch of models tries to study how the hierarchy should interact with the text. Both Chen et al. (2020) and Chen et al. (2021) embed word and label hierarchy jointly in a same space. Deng et al. (2021) constrains label representation with information maximization. Zhao et al. (2021) designs a self-adaption fusion strategy to extract features from text and label. Wang et al. (2022) adopts contrastive learning to directly inject the hierarchical knowledge into text encoder.

### 2.2 Prompt tuning

Prompt tuning Schick and Schütze (2021) aims to transform the downstream NLP task into the pretraining task of the pretrained language models (PLM), which can bridge their gap and better utilize PLM.
The most popular pretraining task of PLM is MLM Devlin et al. (2019), which masks some words in the input text and requires PLM to recover these masked words.
The prompt tuning methods can be broadly divided into 222 categories:
(1) Hard prompt Gao et al. (2021); Schick and Schütze (2021).
The hard prompt methods select template and label words from the vocabulary of PLM, which require carefully manual designing.
(2) Soft prompt Hambardzumyan et al. (2021); Qin and Eisner (2021). Soft prompt methods first create some continuous vectors as template and label embeddings, and then find the best prompt using the training examples, which eliminate the need for manually-designed prompts.

## 3 Preliminaries

### 3.1 Problem Definition

For each hierarchical text classification (HTC) dataset, we have a predefined label hierarchy ℋ=(𝒴,E)ℋ𝒴𝐸\mathcal{H}=(\mathcal{Y},E), where 𝒴𝒴\mathcal{Y} is the label set (also the node set of ℋℋ\mathcal{H}) and E𝐸E is the edge set.
In HTC, given an input text x, the models aim to categorise it into a label set Y⊆𝒴𝑌𝒴Y\subseteq\mathcal{Y}.
Specifically, we focus on a setting where every node except the root has one and only one father so that the hierarchy can be simplified as a tree-like structure. In this case, labels can be organized into layers where labels in the same layer have the same depth in the tree.
The predicted label set Y𝑌Y corresponds to one or more paths in ℋℋ\mathcal{H}.

### 3.2 Vanilla Fine Tuning for HTC

Given an input text x, the vanilla Fine Tuning method first converts it to “[CLS] x [SEP]” as the model input, and then utilizes the PLM to encode it.
After that, it utilizes 𝐡[CLS]subscript𝐡[CLS]\mathbf{h_{\texttt{[CLS]}}}, the hidden state of “[CLS]”, to predict the labels of the input text.
Previous methods Chen et al. (2021); Wang et al. (2022) based on the PLM all follow this fine tuning paradigm.

### 3.3 Prompt Tuning for HTC

To bridge the gap between the pretraining task and the downstream tasks, prompt tuning has been proposed.
We adopt 222 typical flat text classification prompt methods to HTC.

##### Hard Prompt

For a text “x”, hard prompt first applies a template and fills the input into it. For HTC, we choose “[CLS] x [SEP] The text is about [MASK] [SEP]” as template. The PLM is then asked to predict the “[MASK]” slot, which outputs a score for every word in the vocabulary. A verbalizer is then selected for each label to represent its meaning: the score of filling that verbalizer into the “[MASK]” slot is the prediction score on according label. We select the head word (the root word on the dependency tree) of the label name as verbalizer to represent according label.

##### Soft Prompt

For a text “x”, soft prompt append a fixed number of learnable virtual template words to the text (i.e., “[CLS] x [SEP] [V1] [V2] … [V8] [MASK] [SEP]” in case of 888) as template. During training, the PLM learns to predict the “[MASK]” slot as well as tunes virtual template words.
For HTC, we create a learnable label embedding as verbalizer for each hierarchical label.

Since HTC is a multi-label classification problem, following previous works,
both the vanilla fine tuning and 222 typical prompt tuning methods finally conduct a multiple binary classification. The output of PLM is normalized by sigmoid instead of the original softmax to predict on each label and the loss function is changed to binary cross entropy.

Although we can modify these 222 typical prompt tuning methods for HTC, the essence of this challenge has not been considered. As mentioned, the existing prompt methods experience two major gaps when migrating to HTC:

- 1.

Hierarchy and flat gap. Both soft prompt and hard prompt do not take labels into account until prediction, and PLM views all candidate words as equal.
Previous works suggest that incorporating label dependency instead of modeling them as flat classification is essential for alleviating the label imbalance Gopal and Yang (2013).

- 2.

Multi-label and multi-class gap. Previous works on HTC view the problem as multiple binary classification but MLM is designed for multi-class classification. Prompting aims to bridge the gap between pretraining and fine-tuning but the gap still exists if we use the sigmoid normalization and binary cross entropy loss functions for HTC during fine-tuning.

## 4 Methodology

In this section, we introduce a hierarchy-aware prompt tuning method to solve HTC from a multi-label MLM perspective.

### 4.1 Hierarchy-aware Prompt

To bridge the hierarchy and flat gap,
we create the prompt with the label hierarchy constraint and injection.

#### 4.1.1 Hierarchy Constraint

To incorporate the label hierarchy, we propose a layer-wise prompt. Since the label hierarchy is a tree, we construct templates based on the depth of the hierarchy.
Given a predefined label hierarchy ℋ=(𝒴,E)ℋ𝒴𝐸\mathcal{H}=(\mathcal{Y},E) with a depth of L𝐿L and input text x, the template is “[CLS] x [SEP] [V1] [PRED] [V2] [PRED] … [VL] [PRED] [SEP]”. Instead of a fixed number of template words as soft prompt, we have a dynamic template which has template words (from [V1] to [VL]) the same number as hierarchy layers. We use a special [PRED] token for label prediction, indicating a multi-label predication.

We use BERT Devlin et al. (2019) as text encoder, which first embeds input tokens to embedding space:

𝐓=[𝐱1,…,𝐱N,𝐭1,𝐞P,…,𝐭L,𝐞P]𝐓subscript𝐱1…subscript𝐱𝑁subscript𝐭1subscript𝐞𝑃…subscript𝐭𝐿subscript𝐞𝑃\mathbf{T}=[\mathbf{x}_{1},\dots,\mathbf{x}_{N},\mathbf{t}_{1},\mathbf{e}_{P},\dots,\mathbf{t}_{L},\mathbf{e}_{P}]

(1)

where 𝐗=[𝐱1,…,𝐱N]𝐗subscript𝐱1…subscript𝐱𝑁\mathbf{X}=[\mathbf{x}_{1},\dots,\mathbf{x}_{N}] is word embeddings and 𝐞Psubscript𝐞𝑃\mathbf{e}_{P} is the embedding of [PRED], which is initialized by the [MASK] token of BERT. {𝐭𝐢}i=1Lsuperscriptsubscriptsubscript𝐭𝐢𝑖1𝐿\{\mathbf{\mathbf{t}_{i}}\}_{i=1}^{L} are layer-wise template embeddings. Similar to soft prompt, template embeddings are randomly initialized and are learned through training. Here we omit special tokens of BERT ([CLS] and [SEP]) for clarity.

BERT then encodes 𝐓𝐓\mathbf{T} to achieve the hidden states:

𝐇=[𝐡1,…,𝐡N,𝐡t1,𝐡P1,…,𝐡tL,𝐡PL]𝐇subscript𝐡1…subscript𝐡𝑁subscript𝐡subscript𝑡1superscriptsubscript𝐡𝑃1…subscript𝐡subscript𝑡𝐿superscriptsubscript𝐡𝑃𝐿\mathbf{H}=[\mathbf{h}_{1},\dots,\mathbf{h}_{N},\mathbf{h}_{t_{1}},\mathbf{h}_{P}^{1},\dots,\mathbf{h}_{t_{L}},\mathbf{h}_{P}^{L}]

(2)

where 𝐡Pisuperscriptsubscript𝐡𝑃𝑖\mathbf{h}_{P}^{i} is the hidden state of the i𝑖i-th 𝐞Psubscript𝐞𝑃\mathbf{e}_{P}, which corresponds to the i𝑖i-th layer of the label hierarchy.

For verbalizer, we create a learnable virtual label word visubscript𝑣𝑖v_{i} for each label yisubscript𝑦𝑖y_{i} and initialize its embedding 𝐯isubscript𝐯𝑖\mathbf{v}_{i} with the averaging embedding of its corresponding tokens. Instead of predicting all labels in one slot, as shown in the green part of Figure 2, we divide labels into different groups according to their layers and constrain [PRED] to only predict labels on one layer. To this end, each template word [Vi] is followed by a [PRED] token for predictions on the i𝑖i-th layer. By splitting predictions into different slots, the model may learn better about the dependency between labels across different layers and somewhat solve the label imbalance.

Formally, for 𝐡Pmsuperscriptsubscript𝐡𝑃𝑚\mathbf{h}_{P}^{m}, we define a verbalizer VerbmsubscriptVerb𝑚{\rm Verb}_{m} as follows:

Verbm(yi)={vi,yi∈𝒩m∅,Others\displaystyle{\rm Verb}_{m}(y_{i})=\left\{\begin{aligned} &v_{i},&y_{i}\in\mathcal{N}_{m}\\
&\varnothing,&{\rm Others}\\
\end{aligned}\right.

(3)

where 𝒩msubscript𝒩𝑚\mathcal{N}_{m} is the label set of the m𝑚m-th layer and ∅\varnothing denotes that there is no label word for labels at other layers.

#### 4.1.2 Hierarchy Injection

The hierarchy constraint only introduces depth of labels but lacks their connectivity. To make full use of the label hierarchy in an MLM-manner,
we further inject the per-layer label hierarchy knowledge into template embedding.

As shown in the blue part of Figure 2,
a K𝐾K-layer stacked Graph Attention Network (GAT) Kipf and Welling (2017) is adopted to model the label hierarchy.
Given a node u𝑢u at the k𝑘k-th GAT layer, the information interaction and aggregation operation is defined as follows:

𝐠u(k+1)=ReLU​(∑v∈𝒩​(u)​⋃{u}1cu​𝐖(k)​𝐠v(k))superscriptsubscript𝐠𝑢𝑘1ReLUsubscript𝑣𝒩𝑢𝑢1subscript𝑐𝑢superscript𝐖𝑘superscriptsubscript𝐠𝑣𝑘\mathbf{g}_{u}^{(k+1)}=\mathrm{ReLU}(\sum_{v\in\mathcal{N}(u)\bigcup\{u\}}\frac{1}{c_{u}}\mathbf{W}^{(k)}\mathbf{g}_{v}^{(k)})

(4)

where 𝒩​(u)𝒩𝑢\mathcal{N}(u) denotes the neighbors for node u𝑢u , cusubscript𝑐𝑢c_{u} is a normalization constant and 𝐖(l)∈ℝdm×dmsuperscript𝐖𝑙superscriptℝsubscript𝑑𝑚subscript𝑑𝑚\mathbf{W}^{(l)}\in\mathbb{R}^{d_{m}\times d_{m}} is the trainable parameter.

To achieve per layer knowledge for our layer-wise prompt, we create L𝐿L virtual nodes t1,…,tLsubscript𝑡1…subscript𝑡𝐿t_{1},\dots,t_{L} (colored in yellow) and connect tisubscript𝑡𝑖t_{i} with all label nodes at the i𝑖i-th layer in ℋℋ\mathcal{H}. In this way, these virtual nodes can aggregate information from a certain hierarchical level through artificial connections.
For the first GAT layer, we adopt the virtual label word 𝐯𝐢subscript𝐯𝐢\mathbf{v_{i}} for node yi∈𝒴subscript𝑦𝑖𝒴y_{i}\in\mathcal{Y} as its node feature and assign template embedding 𝐭𝐢subscript𝐭𝐢\mathbf{\mathbf{t}_{i}} to virtual node tisubscript𝑡𝑖t_{i} as its node feature.

GAT is then applied to the new graph and it outputs representations 𝐠tiKsuperscriptsubscript𝐠subscript𝑡𝑖𝐾\mathbf{g}_{t_{i}}^{K} for virtual node tisubscript𝑡𝑖t_{i}, which has gathered knowledge from the i𝑖i-th layer. We utilize a residual connection to achieve the i𝑖i-th graph template embedding:

𝐭′i=𝐭i+𝐠tiKsubscriptsuperscript𝐭′𝑖subscript𝐭𝑖superscriptsubscript𝐠subscript𝑡𝑖𝐾\mathbf{t^{\prime}}_{i}=\mathbf{t}_{i}+\mathbf{g}_{t_{i}}^{K}

(5)

where the new template embedding with hierarchy knowledge, 𝐭′isubscriptsuperscript𝐭′𝑖\mathbf{t^{\prime}}_{i}, is injected into BERT replacing 𝐭isubscript𝐭𝑖\mathbf{t}_{i} in Equation 1.

### 4.2 Zero-bounded Multi-label Cross Entropy Loss

Since hierarchical text classification is a multi-label classification problem,
previous methods Zhou et al. (2020); Chen et al. (2021); Zhao et al. (2021) mainly regard HTC as a multiple binary classification problem and utilize the binary cross entropy (BCE) as their loss function:

ℒB​C​E=−∑iC(yi​log​(syi)+(1−yi)​log​(1−syi))subscriptℒ𝐵𝐶𝐸superscriptsubscript𝑖𝐶subscript𝑦𝑖logsubscript𝑠subscript𝑦𝑖1subscript𝑦𝑖log1subscript𝑠subscript𝑦𝑖\mathcal{L}_{BCE}=-\sum_{i}^{C}(y_{i}{\rm log}(s_{y_{i}})+(1-y_{i}){\rm log}(1-s_{y_{i}}))

(6)

where syisubscript𝑠subscript𝑦𝑖s_{y_{i}} is the predicted sigmoid score of the label yisubscript𝑦𝑖y_{i} for the input.
As illustrate in Equation 6, BCE ignores the correlation between labels.
In contrast, the masked language modeling
is a multi-class classification task, which is optimized with the cross entropy (CE) loss:

ℒC​E=−log​esyt∑i=1Cesyi=log​(1+∑i=1,i≠tCesyi−syt)subscriptℒ𝐶𝐸logsuperscript𝑒subscript𝑠subscript𝑦𝑡superscriptsubscript𝑖1𝐶superscript𝑒subscript𝑠subscript𝑦𝑖log1superscriptsubscriptformulae-sequence𝑖1𝑖𝑡𝐶superscript𝑒subscript𝑠subscript𝑦𝑖subscript𝑠subscript𝑦𝑡\begin{split}\mathcal{L}_{CE}&=-{\rm log}\frac{e^{s_{y_{t}}}}{\sum_{i=1}^{C}e^{s_{y_{i}}}}\\
&={\rm log}(1+\sum_{i=1,i\neq t}^{C}e^{s_{y_{i}}-s_{y_{t}}})\end{split}

(7)

where ytsubscript𝑦𝑡y_{t} is the gold label for the input.
As shown in Equation 7, CE forces the score of the gold label is greater than all other labels, which directly models the label correlation.

To harmonize their objectives and bridge this multi-label and multi-class gap,
in this paper, instead of calculating the score of each label separately, we expect the scores of all target labels are greater than all non-target labels. We use a multi-label cross entropy (MLCE) loss Sun et al. (2020); Su (2020):

ℒM​L​C​E=log​(1+∑yi∈𝒩n∑yj∈𝒩pesyi−syj)subscriptℒ𝑀𝐿𝐶𝐸log1subscriptsubscript𝑦𝑖superscript𝒩𝑛subscriptsubscript𝑦𝑗superscript𝒩𝑝superscript𝑒subscript𝑠subscript𝑦𝑖subscript𝑠subscript𝑦𝑗\mathcal{L}_{MLCE}={\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{n}}\sum_{y_{j}\in\mathcal{N}^{p}}e^{s_{y_{i}}-s_{y_{j}}})

(8)

where 𝒩psuperscript𝒩𝑝\mathcal{N}^{p} and 𝒩nsuperscript𝒩𝑛\mathcal{N}^{n} are the target and non-target label set of the input text.

However, Equation 8 is impracticable since we cannot know the number of target labels during inference even if the positive (target) labels and negative (other) labels are separated.
To fix this glitch, following Su (2020), we introduce an anchor label with a constant score 00 in MLCE and hope that the scores of the target labels and the non-target labels are all greater and less than 00 respectively. Thus, we form a zero-bounded multi-label cross entropy (ZMLCE) loss:

ℒZ​M​L​C​E=log(1+∑yi∈𝒩n∑yj∈𝒩pesyi−syj+∑yi∈𝒩nesyi−0+∑yj∈𝒩pe0−syj)=log​(1+∑yi∈𝒩nesyi)+log​(1+∑yi∈𝒩pe−syi)subscriptℒ𝑍𝑀𝐿𝐶𝐸log1subscriptsubscript𝑦𝑖superscript𝒩𝑛subscriptsubscript𝑦𝑗superscript𝒩𝑝superscript𝑒subscript𝑠subscript𝑦𝑖subscript𝑠subscript𝑦𝑗subscriptsubscript𝑦𝑖superscript𝒩𝑛superscript𝑒subscript𝑠subscript𝑦𝑖0subscriptsubscript𝑦𝑗superscript𝒩𝑝superscript𝑒0subscript𝑠subscript𝑦𝑗log1subscriptsubscript𝑦𝑖superscript𝒩𝑛superscript𝑒subscript𝑠subscript𝑦𝑖log1subscriptsubscript𝑦𝑖superscript𝒩𝑝superscript𝑒subscript𝑠subscript𝑦𝑖\begin{split}&\mathcal{L}_{ZMLCE}={\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{n}}\sum_{y_{j}\in\mathcal{N}^{p}}e^{s_{y_{i}}-s_{y_{j}}}\\
&+\sum_{y_{i}\in\mathcal{N}^{n}}e^{s_{y_{i}}-0}+\sum_{y_{j}\in\mathcal{N}^{p}}e^{0-s_{y_{j}}})\\
&={\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{n}}e^{s_{y_{i}}})+{\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{p}}e^{-s_{y_{i}}})\\
\end{split}

(9)

To be consistent with the hierarchy constraint, we adopt ZMLCE at each label hierarchy layer for the layer-wise prediction.
Formally, for the m𝑚m-th layer with scores predicted by 𝐡Pmsuperscriptsubscript𝐡𝑃𝑚\mathbf{h}_{P}^{m}, we add layer constraints as follow:

ℒZ​M​L​C​Em=log​(1+∑yi∈𝒩mnesyi)+log​(1+∑yi∈𝒩mpe−syi)superscriptsubscriptℒ𝑍𝑀𝐿𝐶𝐸𝑚log1subscriptsubscript𝑦𝑖subscriptsuperscript𝒩𝑛𝑚superscript𝑒subscript𝑠subscript𝑦𝑖log1subscriptsubscript𝑦𝑖subscriptsuperscript𝒩𝑝𝑚superscript𝑒subscript𝑠subscript𝑦𝑖\begin{split}\mathcal{L}_{ZMLCE}^{m}=&{\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{n}_{m}}e^{s_{y_{i}}})\\
&+{\rm log}(1+\sum_{y_{i}\in\mathcal{N}^{p}_{m}}e^{-s_{y_{i}}})\end{split}

(10)

where syi=𝐯iT​𝐡Pm+bi​msubscript𝑠subscript𝑦𝑖superscriptsubscript𝐯𝑖𝑇superscriptsubscript𝐡𝑃𝑚subscript𝑏𝑖𝑚s_{y_{i}}=\mathbf{v}_{i}^{T}\mathbf{h}_{P}^{m}+b_{im} and bi​msubscript𝑏𝑖𝑚b_{im} is a learnable bias term.
𝒩mpsuperscriptsubscript𝒩𝑚𝑝\mathcal{N}_{m}^{p} and 𝒩mnsuperscriptsubscript𝒩𝑚𝑛\mathcal{N}_{m}^{n} are the target and non-target label set at the m𝑚m-th layer for the input text respectively.

We keep the original MLM loss as BERT pretraining and the final loss ℒa​l​lsubscriptℒ𝑎𝑙𝑙\mathcal{L}_{all} is the sum of ZMLCE losses at different layers and the MLM loss:

ℒa​l​l=∑m=1LℒZ​M​L​C​Em+ℒM​L​Msubscriptℒ𝑎𝑙𝑙superscriptsubscript𝑚1𝐿superscriptsubscriptℒ𝑍𝑀𝐿𝐶𝐸𝑚subscriptℒ𝑀𝐿𝑀\mathcal{L}_{all}=\sum_{m=1}^{L}\mathcal{L}_{ZMLCE}^{m}+\mathcal{L}_{MLM}

(11)

We randomly mask 151515% words of the text to compute the MLM loss ℒM​L​Msubscriptℒ𝑀𝐿𝑀\mathcal{L}_{MLM}.
During inference, we select labels with scores greater than 00 as our prediction.
A comparison between our method and existing prompt methods is in Appendix B.

## 5 Experiments

### 5.1 Experiment Setup

##### Datasets and Evaluation Metrics

We experiment on Web-of-Science (WOS) Kowsari et al. (2017), NYTimes (NYT) Sandhaus (2008), and RCV1-V2 Lewis et al. (2004) datasets for analysis.
The statistic details are illustrated in Table 4.
We follow the data processing of previous work Zhou et al. (2020); Chen et al. (2021) and measure the experimental results with Macro-F1 and Micro-F1.

##### Baselines

For systematic comparisons, we introduce a variety of hierarchical text classification baselines and compare Hpt with two typical prompt learning methods.
1) TextRCNN Lai et al. (2015). A simple network of bidirectional GRU followed by CNN. It is a traditional text classification model adopted by HiAGM, HTCInfoMax, and HiMatch as their text encoder.
2) BERT Devlin et al. (2019). A widely used pretrained language model that can serve as a text encoder. Among previous work, only HiMatch introduces BERT as text encoder so that we implement other baselines with BERT replaced.
3) HiAGM Zhou et al. (2020). HiAGM exploits the prior probability of label dependencies through Graph Convolution Network and applies soft attention over text feature and label feature for the mixed feature.
4) HTCInfoMax Deng et al. (2021). HTCInfoMax improves HiAGM by maximizing text-label mutual information and matching the label feature to a prior distribution.
5) HiMatch Chen et al. (2021). HiMatch views the problem as a semantic matching problem and matches the relationship between the text semantics and the label semantics.
6) HGCLR Wang et al. (2022). HGCLR regulates BERT representation by contrastive learning and introduces a new graph encoder.

##### Implement Details

We implement our model using PyTorch with an end-to-end fashion. Following previous work Chen et al. (2021), we use bert-base-uncased as our base architecture. We use a single layer of GAT for hierarchy injection. The batch size is set to 161616. The optimizer is Adam with a learning rate of 3​e−53superscript𝑒53e^{-5}. We train the model with train set and evaluate on development set after every epoch and stop training if the Macro-F1 does not increase for 555 epochs. All of the hyperparameters have not been tuned. For baseline models, we follow the hyperparameter tuning procedure in their original paper. We use a length of 888 template words for soft prompt in accordance with Hpt.

### 5.2 Main Results

Table 1 illustrates our main results.
As is shown, “HardPrompt” and “SoftPrompt” outperform the vanilla fine tuning BERT on all 333 datasets and achieve a comparable result with the state-of-the-art method on RCV1-V2.
This result shows the superiority of the prompt tuning paradigm since it adapts HTC to BERT to some extent.

By bridging the gaps between HTC and MLM, our Hpt achieves new state-of-the-art results on all 333 datasets.
Comparing to HiMatch Chen et al. (2021), our model introduces no extra parameter so that these improvements demonstrate that Hpt can better utilize the pretrained language model.
Although HGCLR Wang et al. (2022) introduces a new graph encoder, our model achieves consistent improvements on all datasets with a simple GAT.
In addition, the depths of the label hierarchy for WOS, RCV1-V2, and NYT are 222, 444, and 888 respectively, which can reflect the respective difficulty of the label hierarchy.
Hpt outperforms both the baseline BERT and HGCLR by increasing margins on WOS, RCV1-V2, and NYT respectively, showing that hierarchy-aware prompt can better handle more difficult label hierarchy.

Model
WOS (Depth 2)
RCV1-V2 (Depth 4)
NYT (Depth 8)

Micro-F1
Macro-F1
Micro-F1
Macro-F1
Micro-F1
Macro-F1

TextRCNN Zhou et al. (2020)

83.55
76.99
81.57
59.25
70.83
56.18

HiAGM Zhou et al. (2020)

85.82
80.28
83.96
63.35
74.97
60.83

HTCInfoMax Deng et al. (2021)

85.58
80.05
83.51
62.71
-
-

HiMatch Chen et al. (2021)

86.20
80.53
84.73
64.11
-
-

BERT Wang et al. (2022)

85.63
79.07
85.65
67.02
78.24
66.08

BERT+HiAGMWang et al. (2022)

86.04
80.19
85.58
67.93
78.64
66.76

BERT+HTCInfoMaxWang et al. (2022)

86.30
79.97
85.53
67.09
78.75
67.31

BERT+HiMatch Chen et al. (2021)

86.70
81.06
86.33
68.66
-
-

HGCLR Wang et al. (2022)

87.11
81.20
86.49
68.31
78.86
67.96

BERT+HardPrompt (Ours)
86.39
80.43
86.78
68.78
79.45
67.99

BERT+SoftPrompt (Ours)
86.57
80.75
86.53
68.34
78.95
68.21

Hpt (Ours)
87.16
81.93
87.26
69.53
80.42
70.42

Ablation Models
Micro-F1
Macro-F1

Hpt
80.49
71.07

r.m. hierarchy constraint
80.32
70.58

r.m. hierarchy injection
80.41
69.71

r.p. BCE loss
79.74
70.40

r.m. MLM loss
80.16
70.78

with random connection
80.12
69.42

### 5.3 Ablation Study

To illustrate the effect of our proposed mechanisms, we conduct ablation studies by removing one component of our model at a time. We test on NYT dataset in this and following sections because it has the most complicated label hierarchy and it can better demonstrate how our method reacts to the hierarchy.

After removing the hierarchy constraint, the template has only one [PRED] token and the model needs to recover all label words according to its hidden state.
As shown in Table 2, the Micro-F1 and Macro-F1 drop slightly, which shows the effectiveness of our layer-wise prompt.
By removing the hierarchy injection (i.e., remove Equation 5), the model cannot access the connectivity of the label hierarchy and drops 1.361.361.36 on Macro-F1. From this decline, we can see that the hierarchy injection is essential for the performance of labels with few instances. By incorporating an extra structural encoder, the model can learn label features from training instances from other classes based on the hierarchical dependencies between them. As a result, the hierarchy injection significantly boosts the performance of scarce classes.
At last, both the performances of using BCE loss instead of ZMLCE loss (r.p. BCE loss) and removing MLM loss (r.m. MLM loss) drop, which shows it is important to bridge the gap of optimizing objectives between HTC and MLM.

To further illustrate the effectiveness of the hierarchy injection, we test our model with random connection. As a reminder, during hierarchy injection, we connect virtual nodes with according labels with the same depth. Random connection adds random connections based on that connection. For each label, it connects the label to another virtual node randomly.

As in the last row of Table 2, the variant with random connection drops over 1% on Macro-F1 score. This result illustrates that connections that violate the label hierarchy have adverse effects. The destructiveness of a contradicting input like random connection even outweighs removing the hierarchy completely (r.m. hierarchy injection), reflecting that the proposed Hpt indeed gains instructive information from the label hierarchy. More discussions on the connection of virtual nodes are elaborate in Appendix C. Ablation results on other datasets are in Appendix D.

### 5.4 Interpreting on Representation Space

In this section, we hope to intuitively show how the label hierarchy is incorporated and what the prompt has learned.
The virtual label words are learned in the same space as word embedding, so they can be interpreted by their similarities with meaningful words.
Therefore,
we illustrate the top 888 nearest words of 222 labels in the NYT dataset, National Hockey League (NHL) and News and Features (NF).
As shown in table 3, despite some meaningless words, the model indeed learns some interpretable features. For NHL, the label words of Hpt consist of the semantic of football, which is the brother node of Hockey (the father node of NHL) in the label hierarchy.
For NF, the label words of Hpt consist of the semantic of theatre, which is the father node of NF.
After removing the hierarchy knowledge (r.m. hierarchy), these semantics disappear from label words of NHL and NF.
These results intuitively show Hpt incorporates the hierarchy knowledge into the pretrained language model and bridges the gap between HTC and MLM.

Top 8 nearest words

Label

(different layers separated by ‘/’)

Hpt

Hpt (r.m. hierarchy)

[1] hockey
[2] league
[1] hockey
[2] national

[3] national
[4] 2011
[3] league
[4] 2012

[5] 2013
[6] ##^
[5] 2008
[6] 1996

News/Sports/Hockey/

National Hockey League

[7] 2012
[8] football

[7] 2010
[8] 2014

[1] features
[2] .
[1] .
[2] features

[3] and
[4] the
[3] and
[4] the

[5] theatre

[6] ;
[5] ,
[6] ;

Features/Theater/

News and Features

[7] ,
[8] news
[7] of
[8] news

### 5.5 Results on Imbalanced Hierarchy

(a)

(b)

One of the key challenges of hierarchical text classification is the imbalanced label hierarchy. In this section, we analyze how our model resolves the issue of imbalance on the development set of NYT.

For HTC, the imbalance can be viewed from two perspectives. For one, number of labels at different depths of the hierarchy is imbalanced. As shown in Figure 3a, medium layers (depth 3 and 4) have more labels than deep or shallow layers, where all models have poor performances. Comparing with other baselines, Hpt mainly boost the performance of medium levels.
For another, instance of each label is various. Take NYT dataset as an example, the ratio of the maximum and minimum amount of training samples of a label is over 100100100. In Figure 3b, we cluster labels into 555 bunches depending on their amounts of training samples. Our model largely improves the performance of labels with few training instances, showing that our method can alleviate the long-tail effect to some extent.

### 5.6 Results on Low Resource Setting

To further evaluate the potential of our method, we conduct experiments in low-resource settings. Since the problem is multi-label, the commonly used N-way K-shot setting is hard to define so we simply sample 10%percent1010\% of training data.
As previous HTC works do not consider the low resource setting (LRS), we reproduce baseline models in LRS. Besides less training data, other settings follow the main experiment.

The comparison of LRS experimental results is shown in Figure 4.
Among baseline methods, prompt-based models outperform non-prompt-based models on 333 datasets, which shows the advantages of prompt methods in LRS.
Our model outperforms all baseline models and has better stability (lower standard deviation) on all 333 LRS datasets.
Comparing with the full resource setting (FRS) (i.e., main results), the performance gap between Hpt and other baselines increases on the LRS.
For example, on RCV1-V2, Hpt outperforms “BERT+HTCinfoMAX” 2.132.132.13 and 6.096.096.09 Macro-F1 scores in FRS and LRS, respectively, which shows the potential of our method.

## 6 Conclusion

In this paper, we propose a hierarchy-aware prompt tuning (Hpt) method to bridge the gaps between HTC and MLM.
To bridge the hierarchy and flat gap, Hpt incorporates the label hierarchy knowledge into virtual template and label words.
To bridge the multi-label and multi-class gap, Hpt introduces a zero-bounded multi-label cross entropy loss to harmonize the objectives of HTC and MLM.
Hpt transforms HTC into a hierarchy-aware multi-label MLM task, which can better tap the potential of the pretrained language model in HTC.
Extensive experiments show that our method achieves state-of-the-art performances on 333 popular HTC datasets, and is adept at handling the imbalance and low resource situations.

## Limitations

Prompting methods needs pretrained language model as backbone. Our work is based on the masked language model (MLM) task but it is not a universal component of PLM. As a result, our approach is only applicable to PLMs which incorporate MLM. Despite such limited choices, comparing to other HTC works which adopt PLM as a replaceable text encoder, our approach takes more advantage of PLMs by considering how they are trained. Another limitation is the constraint of maximum sequence length. Although the length limitation of PLM is extensively existed, our approach needs extra tokens for template, and that further shortens the length of input text. Even so, the experiment results indicate that our method performs better than the raw PLM so that this sacrifice is worthy. Notice that the length of our template is proportional to the depth of the label hierarchy, so Hpt may fail to datasets with extreme hierarchy depth.

## Acknowledgements

We thank all the anonymous reviewers for their constructive feedback.
The work is supported by National Natural Science Foundation of China under Grant No.62036001, PKU-Baidu Fund (No. 2020BD021) and NSFC project U19A2065.

## References

- Banerjee et al. (2019)

Siddhartha Banerjee, Cem Akkaya, Francisco Perez-Sorrosal, and Kostas
Tsioutsiouliklis. 2019.

Hierarchical transfer
learning for multi-label text classification.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 6295–6300, Florence, Italy.
Association for Computational Linguistics.

- Chen et al. (2020)

Boli Chen, Xin Huang, Lin Xiao, Zixin Cai, and Liping Jing. 2020.

Hyperbolic interaction model for hierarchical multi-label classification.

In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 34, pages 7496–7503.

- Chen et al. (2021)

Haibin Chen, Qianli Ma, Zhenxi Lin, and Jiangyue Yan. 2021.

Hierarchy-aware label semantics matching network for hierarchical text
classification.

In Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 4370–4379,
Online. Association for Computational Linguistics.

- Chen et al. (2022)

Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei
Huang, Luo Si, and Huajun Chen. 2022.

Knowprompt: Knowledge-aware prompt-tuning with synergistic optimization for
relation extraction.

In Proceedings of the ACM Web Conference 2022, pages
2778–2788.

- Deng et al. (2021)

Zhongfen Deng, Hao Peng, Dongxiao He, Jianxin Li, and Philip Yu. 2021.

HTCInfoMax: A global model for hierarchical text classification via
information maximization.

In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 3259–3265, Online. Association for Computational
Linguistics.

- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

BERT: Pre-training of
deep bidirectional transformers for language understanding.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4171–4186,
Minneapolis, Minnesota. Association for Computational Linguistics.

- Gao et al. (2021)

Tianyu Gao, Adam Fisch, and Danqi Chen. 2021.

Making
pre-trained language models better few-shot learners.

In Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 3816–3830,
Online. Association for Computational Linguistics.

- Gopal and Yang (2013)

Siddharth Gopal and Yiming Yang. 2013.

Recursive
regularization for large-scale classification with hierarchical and graphical
dependencies.

In Proceedings of the 19th ACM SIGKDD international conference
on Knowledge discovery and data mining, pages 257–265.

- Hambardzumyan et al. (2021)

Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. 2021.

WARP:
Word-level Adversarial ReProgramming.

In Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 4921–4933,
Online. Association for Computational Linguistics.

- Johnson and Zhang (2015)

Rie Johnson and Tong Zhang. 2015.

Effective use of word
order for text categorization with convolutional neural networks.

In Proceedings of the 2015 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 103–112, Denver, Colorado. Association for
Computational Linguistics.

- Kipf and Welling (2017)

Thomas N. Kipf and Max Welling. 2017.

Semi-supervised
classification with graph convolutional networks.

In 5th International Conference on Learning Representations
(ICLR).

- Kowsari et al. (2017)

Kamran Kowsari, Donald E Brown, Mojtaba Heidarysafa, Kiana Jafari Meimandi,
Matthew S Gerber, and Laura E Barnes. 2017.

Hdltex: Hierarchical deep learning for text classification.

In 2017 16th IEEE international conference on machine learning
and applications (ICMLA), pages 364–371. IEEE.

- Lai et al. (2015)

Siwei Lai, Liheng Xu, Kang Liu, and Jun Zhao. 2015.

Recurrent
convolutional neural networks for text classification.

In Proceedings of the Twenty-Ninth AAAI Conference on
Artificial Intelligence, pages 2267–2273.

- Lewis et al. (2004)

David D Lewis, Yiming Yang, Tony Russell-Rose, and Fan Li. 2004.

Rcv1: A new benchmark collection for text categorization research.

Journal of machine learning research, 5(Apr):361–397.

- Mao et al. (2019)

Yuning Mao, Jingjing Tian, Jiawei Han, and Xiang Ren. 2019.

Hierarchical text
classification with reinforced label assignment.

In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 445–455, Hong Kong,
China. Association for Computational Linguistics.

- Qin and Eisner (2021)

Guanghui Qin and Jason Eisner. 2021.

Learning
how to ask: Querying lms with mixtures of soft prompts.

In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 5203–5212.

- Sandhaus (2008)

Evan Sandhaus. 2008.

The new york times annotated corpus.

Linguistic Data Consortium, Philadelphia, 6(12):e26752.

- Schick and Schütze (2021)

Timo Schick and Hinrich Schütze. 2021.

Exploiting
cloze-questions for few-shot text classification and natural language
inference.

In Proceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics: Main Volume, pages
255–269.

- Shimura et al. (2018)

Kazuya Shimura, Jiyi Li, and Fumiyo Fukumoto. 2018.

HFT-CNN: Learning
hierarchical category structure for multi-label short text categorization.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 811–816, Brussels, Belgium. Association
for Computational Linguistics.

- Shin et al. (2020)

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer
Singh. 2020.

AutoPrompt: Eliciting Knowledge from Language Models with
Automatically Generated Prompts.

In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 4222–4235, Online. Association
for Computational Linguistics.

- Silla and Freitas (2011)

Carlos N Silla and Alex A Freitas. 2011.

A survey of hierarchical classification across different application
domains.

Data Mining and Knowledge Discovery, 22(1):31–72.

- Su (2020)

Jianlin Su. 2020.

Extending “softmax+cross entropy” to multi-label classification
problem.

https://spaces.ac.cn/archives/7359.

- Sun et al. (2020)

Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang,
and Yichen Wei. 2020.

Circle loss: A unified perspective of pair similarity optimization.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 6398–6407.

- Wang et al. (2022)

Zihan Wang, Peiyi Wang, Lianzhe Huang, Xin Sun, and Houfeng Wang. 2022.

Incorporating
hierarchy into text encoder: a contrastive learning approach for hierarchical
text classification.

In Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 7109–7119,
Dublin, Ireland. Association for Computational Linguistics.

- Wehrmann et al. (2018)

Jonatas Wehrmann, Ricardo Cerri, and Rodrigo Barros. 2018.

Hierarchical multi-label classification networks.

In International Conference on Machine Learning, pages
5075–5084. PMLR.

- Wu et al. (2019)

Jiawei Wu, Wenhan Xiong, and William Yang Wang. 2019.

Learning to learn and
predict: A meta-learning approach for multi-label classification.

In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 4354–4364, Hong Kong,
China. Association for Computational Linguistics.

- Zhang et al. (2021)

Xinyi Zhang, Jiahao Xu, Charlie Soh, and Lihui Chen. 2021.

La-hcn: Label-based attention for hierarchical multi-label text
classification neural network.

Expert Systems with Applications, page 115922.

- Zhao et al. (2021)

Rui Zhao, Xiao Wei, Cong Ding, and Yongqi Chen. 2021.

Hierarchical multi-label text classification: Self-adaption semantic
awareness network integrating text topic and label level information.

In International Conference on Knowledge Science, Engineering
and Management, pages 406–418. Springer.

- Zhou et al. (2020)

Jie Zhou, Chunping Ma, Dingkun Long, Guangwei Xu, Ning Ding, Haoyu Zhang,
Pengjun Xie, and Gongshen Liu. 2020.

Hierarchy-aware global model for hierarchical text classification.

In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 1106–1117, Online. Association for
Computational Linguistics.

## Appendix A Data Statistics

Dataset
|Y|𝑌|Y|
Depth
Avg(|yi|subscript𝑦𝑖|y_{i}|)
Train
Dev
Test

WOS
141
2
2.0
30,070
7,518
9,397

NYT
166
8
7.6
23,345
5,834
7,292

RCV1-V2
103
4
3.24
20,833
2,316
781,265

## Appendix B Example of Different Prompt Methods

We provide some detailed examples here to explain the difference between our Hpt with existing prompt methods.

Templates of hard prompt, soft prompt and Hpt are illustrated in Table 5. x is the original text and [CLS] and [SEP] are special tokens of BERT. [V1] to [VN] in soft prompt are N𝑁N virtual template words which are learnable embeddings, and the number N𝑁N is predefined. Our method has L𝐿L virtual template words. They are output embeddings of graph encoder as in Equation 5 and L𝐿L is the number of hierarchy layers.
Our method uses a special token [PRED] for multi-label prediction (Section 4.2), whereas hard and soft prompt use the same [MASK] token as BERT, which is proposed for single-label predictions.

Method
Template

Hard prompt

[CLS] x [SEP] The text is

about [MASK] [SEP]

Soft prompt

[CLS] x [SEP] [V1]

[V2] … [VN] [MASK] [SEP]

Hpt

[CLS] x [SEP] [V1] [PRED] [V2]

[PRED] … [VL] [PRED] [SEP]

## Appendix C Discussion on Different Connections of Hierarchy Injection

During hierarchy injection, we connect virtual nodes with according labels with same depth, but this connection is not unique. Besides random connection, we further test our model with a variant. Depth increasing connects a virtual nodes with labels on the same and shallower layers, i.e., virtual node tisubscript𝑡𝑖t_{i} connects with all label nodes on 1st to i𝑖i-th layers. Figure 5 is an illustration of theses two connections.

As in the third row of Table 6, variant with depth increasing behaves similarly to the original one. This observation illustrates that the impact of the connection of virtual nodes is not significant as long as it contains logical hierarchical information. Comparing with random connection which violates the label hierarchy and has adverse effects, this result reflects that the proposed Hpt is aware of the label hierarchy on the secondary side.

(a) Depth increasing

(b) Random connection

Ablation Models
Micro-F1
Macro-F1

Hpt
80.49
71.07

r.m. hierarchy injection
80.41
69.71

with depth increasing
80.48
70.95

with random connection
80.12
69.42

## Appendix D Ablation results on WebOfScience and RCV1-V2

Ablation Models
Micro-F1
Macro-F1

Hpt
87.88
81.68

r.m. hierarchy constraint
87.34
81.27

r.m. hierarchy injection
87.58
81.54

r.p. BCE loss
87.17
80.78

r.m. MLM loss
87.22
81.36

with random connection
87.56
81.42

Ablation Models
Micro-F1
Macro-F1

Hpt
88.37
70.12

r.m. hierarchy constraint
87.62
69.04

r.m. hierarchy injection
87.57
68.53

r.p. BCE loss
87.79
68.12

r.m. MLM loss
87.83
69.76

with random connection
88.22
68.86

The hierarchy of WOS dataset only has two layer so that structural information of WOS is weak. So, in Table 7, removing or disturbing such information have little influence.

After replacing ZMLCE loss with BCE loss, Macro-F1 decreases dramatically on all datasets. Although BCE loss indeed can solve the multi-label problem, ZMLCE loss is a better choice theoretically and experimentally.

Generated on Mon Mar 11 11:24:34 2024 by LaTeXML
