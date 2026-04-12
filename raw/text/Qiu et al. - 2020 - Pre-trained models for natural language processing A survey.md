# Qiu et al. - 2020 - Pre-trained models for natural language processing A survey

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Qiu et al. - 2020 - Pre-trained models for natural language processing A survey.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2003.08271
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\ArticleType

Invited Review\Year2020
\MonthMarch
\Volxx
\Noxx
\BeginPage1 \EndPage25

xpqiu@fudan.edu.cn

\AuthorMark

QIU XP, et al.

\AuthorCitation

QIU XP, et al

# Pre-trained Models for Natural Language Processing: A Survey

Xipeng Qiu

  
Tianxiang Sun

  
Yige Xu

  
Yunfan Shao

  
Ning Dai

  
Xuanjing Huang

School of Computer Science, Fudan University, Shanghai 200433, China

Shanghai Key Laboratory of Intelligent Information Processing, Shanghai 200433, China

###### Abstract

Recently, the emergence of pre-trained models (PTMs)111PTMs are also known as pre-trained language models (PLMs). In this survey, we use PTMs for NLP instead of PLMs to avoid confusion with the narrow concept of probabilistic (or statistical) language models. has brought natural language processing (NLP) to a new era.
In this survey, we provide a comprehensive review of PTMs for NLP. We first briefly introduce language representation learning and its research progress. Then we systematically categorize existing PTMs based on a taxonomy from four different perspectives. Next, we describe how to adapt the knowledge of PTMs to downstream tasks. Finally, we outline some potential directions of PTMs for future research. This survey is purposed to be a hands-on guide for understanding, using, and developing PTMs for various NLP tasks.

###### keywords:

## 1 Introduction

With the development of deep learning, various neural networks have been widely used to solve Natural Language Processing (NLP) tasks, such as convolutional neural networks (CNNs) [kalchbrenner2014convolutional, kim2014convolutional, gehring2017convolutional], recurrent neural networks (RNNs) [sutskever2014sequence, liu2016recurrent], graph-based neural networks (GNNs) [socher2013recursive, tai2015improved, DBLP:conf/naacl/MarcheggianiBT18] and attention mechanisms [bahdanau2014neural, vaswani2017transformer].
One of the advantages of these neural models is their ability to alleviate the feature engineering problem. Non-neural NLP methods usually heavily rely on the discrete handcrafted features, while neural methods usually use low-dimensional and dense vectors (aka. distributed representation) to implicitly represent the syntactic or semantic features of the language. These representations are learned in specific NLP tasks. Therefore, neural methods make it easy for people to develop various NLP systems.

Despite the success of neural models for NLP tasks, the performance improvement may be less significant compared to the Computer Vision (CV) field. The main reason is that current datasets for most supervised NLP tasks are rather small (except machine translation). Deep neural networks usually have a large number of parameters, which make them overfit on these small training data and do not generalize well in practice. Therefore, the early neural models for many NLP tasks were relatively shallow and usually consisted of only 1∼similar-to\sim3 neural layers.

Recently, substantial work has shown that pre-trained models (PTMs), on the large corpus can learn universal language representations, which are beneficial for downstream NLP tasks and can avoid training a new model from scratch. With the development of computational power, the emergence of the deep models (i.e., Transformer [vaswani2017transformer]), and the constant enhancement of training skills, the architecture of PTMs has been advanced from shallow to deep.
The first-generation PTMs aim to learn good word embeddings. Since these models themselves are no longer needed by downstream tasks, they are usually very shallow for computational efficiencies, such as Skip-Gram [mikolov2013word2vec] and GloVe [DBLP:conf/emnlp/PenningtonSM14]. Although these pre-trained embeddings can capture semantic meanings of words, they are context-free and fail to capture higher-level concepts in context, such as polysemous disambiguation, syntactic structures, semantic roles, anaphora. The second-generation PTMs focus on learning contextual word embeddings, such as CoVe [mccan2017learn], ELMo [peters2018elmo], OpenAI GPT [radford2018improving] and BERT [devlin2019bert]. These learned encoders are still needed to represent words in context by downstream tasks. Besides, various pre-training tasks are also proposed to learn PTMs for different purposes.

The contributions of this survey can be summarized as follows:

- 1.

Comprehensive review. We provide a comprehensive review of PTMs for NLP, including background knowledge, model architecture, pre-training tasks, various extensions, adaption approaches, and applications.

- 2.

New taxonomy. We propose a taxonomy of PTMs for NLP, which categorizes existing PTMs from four different perspectives: 1) representation type, 2) model architecture; 3) type of pre-training task; 4) extensions for specific types of scenarios.

- 3.

Abundant resources. We collect abundant resources on PTMs, including
open-source implementations of PTMs, visualization tools, corpora, and paper lists.

- 4.

Future directions. We discuss and analyze the limitations of existing PTMs. Also, we suggest possible future research directions.

The rest of the survey is organized as follows. Section 2 outlines the background concepts and commonly used notations of PTMs.
Section 3 gives a brief overview of PTMs and clarifies the categorization of PTMs. Section LABEL:sec:extension provides extensions of PTMs.
Section LABEL:sec:adapt discusses how to transfer the knowledge of PTMs to downstream tasks.
Section LABEL:sec:resources gives the related resources on PTMs.
Section LABEL:sec:app
presents a collection of applications across various NLP tasks.
Section LABEL:sec:future discusses the current challenges and suggests future
directions. Section LABEL:sec:conclusion summarizes the paper.

## 2 Background

### 2.1 Language Representation Learning

As suggested by bengio2013representation, a good representation should express general-purpose priors that are not task-specific but would be likely to be useful
for a learning machine to solve AI-tasks.
When it comes to language, a good representation should capture the implicit linguistic rules and common sense knowledge hiding in text data, such as lexical meanings, syntactic structures, semantic roles, and even pragmatics.

The core idea of distributed representation is to describe the meaning of a piece of text by low-dimensional real-valued vectors. And each dimension of the vector has no corresponding sense, while the whole represents a concrete concept.
Figure 1 illustrates the generic neural architecture for NLP. There are two kinds of word embeddings: non-contextual and contextual embeddings. The difference between them is whether the embedding for a word dynamically changes according to the context it appears in.

##### Non-contextual Embeddings

The first step of representing language is to map discrete language symbols into a distributed embedding space. Formally, for each word (or sub-word) x𝑥x in a vocabulary 𝒱𝒱\mathcal{V}, we map it to a vector 𝐞x∈ℝDefragmentse𝑥RfragmentsD𝑒\mathbf{e}_{x}\in\mathbb{R}^{D_{e}} with a lookup table 𝐄∈ℝDe×|𝒱|fragmentsERfragmentsD𝑒|V|\mathbf{E}\in\mathbb{R}^{D_{e}\times|\mathcal{V}|}, where DefragmentsD𝑒D_{e} is a hyper-parameter indicating the dimension of token embeddings. These embeddings are trained on task data along with other model parameters.

There are two main limitations to this kind of embeddings. The first issue is that the embeddings are static. The embedding for a word does is always the same regardless of its context. Therefore, these non-contextual embeddings fail to model polysemous words.
The second issue is the out-of-vocabulary problem. To tackle this problem, character-level word representations or sub-word representations are widely used in many NLP tasks, such as CharCNN [kim2016character], FastText [DBLP:journals/tacl/BojanowskiGJM17] and Byte-Pair Encoding (BPE) [DBLP:conf/acl/SennrichHB16a].

(a) Convolutional Model

(b) Recurrent Model

(c) Fully-Connected Self-Attention Model

##### Contextual Embeddings

To address the issue of polysemous and the context-dependent nature of words, we need
distinguish the semantics of words in different contexts. Given a text x1,x2,⋯,xTfragmentsx1,x2,⋯,x𝑇x_{1},x_{2},\cdots,x_{T} where each token xt∈𝒱fragmentsx𝑡Vx_{t}\in\mathcal{V} is a word or sub-word, the contextual representation of xtfragmentsx𝑡x_{t} depends on the whole text.

[𝐡1,𝐡2,⋯,𝐡T]=fenc(x1,x2,⋯,xT),fragments[h1,h2,⋯,h𝑇]fenc(x1,x2,⋯,x𝑇),\displaystyle[\mathbf{h}_{1},\mathbf{h}_{2},\cdots,\mathbf{h}_{T}]=f_{\mathrm{enc}}(x_{1},x_{2},\cdots,x_{T}),

(1)

where fenc(⋅)fragmentsfenc(⋅)f_{\mathrm{enc}}(\cdot) is neural encoder, which is described in Section 2.2, 𝐡tfragmentsh𝑡\mathbf{h}_{t} is called contextual embedding or dynamical embedding of token xtfragmentsx𝑡x_{t} because of the contextual information included in.

### 2.2 Neural Contextual Encoders

Most of the neural contextual encoders can be classified into two categories: sequence models and non-sequence models. Figure 2 illustrates three representative architectures.

#### 2.2.1 Sequence Models

Sequence models usually capture local context of a word in sequential order.

##### Convolutional Models

Convolutional models take the embeddings of words in the input sentence and capture the meaning of a word by aggregating the local information from its neighbors by convolution operations [kim2014convolutional].

##### Recurrent Models

Recurrent models capture the contextual representations of words with short memory, such as LSTMs [DBLP:journals/neco/HochreiterS97] and GRUs [chung2014empirical].
In practice, bi-directional LSTMs or GRUs are used to collect information from both sides of a word, but its performance is often affected by the long-term dependency problem.

#### 2.2.2 Non-Sequence Models

Non-sequence models learn the contextual representation with a pre-defined tree or graph structure between words, such as the syntactic structure or semantic relation. Some popular non-sequence models include Recursive NN [socher2013recursive], TreeLSTM [tai2015improved, zhu2015long], and GCN [kipf2017semi].

Although the linguistic-aware graph structure can provide useful inductive bias, how to build a good graph structure is also a challenging problem. Besides, the structure depends heavily on expert knowledge or external NLP tools, such as the dependency parser.

##### Fully-Connected Self-Attention Model

In practice, a more straightforward way is to use a fully-connected graph to model the relation of every two words and let the model learn the structure by itself. Usually, the connection weights are dynamically computed by the self-attention mechanism, which implicitly indicates the connection between words.
A successful instance of fully-connected self-attention model is the Transformer [vaswani2017transformer, lin2021survey], which also needs other supplement modules, such as positional embeddings, layer normalization, residual connections and position-wise feed-forward network (FFN) layers.

#### 2.2.3 Analysis

Sequence models learn the contextual representation of the word with locality bias and are hard to capture the long-range interactions between words. Nevertheless, sequence models are usually easy to train and get good results for various NLP tasks.

In contrast, as an instantiated fully-connected self-attention model, the Transformer can directly model the dependency between every two words in a sequence, which is more powerful and suitable to model long range dependency of language.
However, due to its heavy structure and less model bias, the Transformer usually requires a large training corpus and is easy to overfit on small or modestly-sized datasets [radford2018improving, guo2019star].

Currently, the Transformer has become the mainstream architecture of PTMs due to its powerful capacity.

### 2.3 Why Pre-training?

With the development of deep learning, the number of model parameters has increased rapidly. The much larger dataset is needed to fully train model parameters and prevent overfitting.
However, building large-scale labeled datasets is a great challenge for most NLP tasks due to the extremely expensive annotation costs, especially for syntax and semantically related tasks.

In contrast, large-scale unlabeled corpora are relatively easy to construct. To leverage the huge unlabeled text data, we can first learn a good representation from them and then use these representations for other tasks. Recent studies have demonstrated significant performance gains on many NLP tasks with the help of the representation extracted from the PTMs on the large unannotated corpora.

The advantages of pre-training can be summarized as follows:

- 1.

Pre-training on the huge text corpus can learn universal language representations and help with the downstream tasks.

- 2.

Pre-training provides a better model initialization, which usually leads to a better generalization performance and speeds up convergence on the target task.

- 3.

Pre-training can be regarded as a kind of regularization to avoid overfitting on small data [DBLP:journals/jmlr/ErhanBCMVB10].

### 2.4 A Brief History of PTMs for NLP

Pre-training has always been an effective strategy to learn the parameters of deep neural networks, which are then fine-tuned on downstream tasks.
As early as 2006, the breakthrough of deep learning came with greedy layer-wise unsupervised pre-training followed by supervised fine-tuning [hinton2006reducing].
In CV, it has been in practice to pre-train models on the huge ImageNet corpus, and then fine-tune further on smaller data for different tasks.
This is much better than a random initialization because the model learns general image features, which can then be used in various vision tasks.

In NLP, PTMs on large corpus have also been proved to be beneficial for the downstream NLP tasks, from the shallow word embedding to deep neural models.

#### 2.4.1 First-Generation PTMs: Pre-trained Word Embeddings

Representing words as dense vectors has a long history [hinton1986distributed]. The “modern” word embedding is introduced in pioneer work of neural network language model (NNLM) [bengio2003neural].
DBLP:journals/jmlr/CollobertWBKKK11 showed that the pre-trained word embedding on the unlabelled data could significantly improve many NLP tasks. To address the computational complexity, they learned word embeddings with pairwise ranking task instead of language modeling.
Their work is the first attempt to obtain generic word embeddings useful for other tasks from unlabeled data.
mikolov2013word2vec showed that there is no need for deep neural networks to build good word embeddings. They propose two shallow architectures: Continuous Bag-of-Words (CBOW) and Skip-Gram (SG) models.
Despite their simplicity, they can still learn high-quality word embeddings to capture the latent syntactic and semantic similarities among words.
Word2vec is one of the most popular implementations of these models and makes the pre-trained word embeddings accessible for different tasks in NLP.
Besides, GloVe [DBLP:conf/emnlp/PenningtonSM14] is also a widely-used model for obtaining pre-trained word embeddings, which are computed by global word-word co-occurrence statistics from a large corpus.

Although pre-trained word embeddings have been shown effective in NLP tasks, they are context-independent and mostly trained by shallow models. When used on a downstream task, the rest of the whole model still needs to be learned from scratch.

During the same time period, many researchers also try to learn embeddings of paragraph, sentence or document, such as paragraph vector [le2014distributed], Skip-thought vectors [kiros2015skip], Context2Vec [melamud-etal-2016-context2vec].
Different from their modern successors, these sentence embedding models try to encode input sentences into a fixed-dimensional vector representation, rather than the contextual representation for each token.

#### 2.4.2 Second-Generation PTMs: Pre-trained Contextual Encoders

Since most NLP tasks are beyond word-level, it is natural to pre-train the neural encoders on sentence-level or higher. The output vectors of neural encoders are also called contextual word embeddings since they represent the word semantics depending on its context.

dai2015semi proposed the first successful instance of PTM for NLP. They initialized LSTMs with a language model (LM) or a sequence autoencoder, and found the pre-training can improve the training and generalization of LSTMs in many text classification tasks.
liu2016recurrent pre-trained a shared LSTM encoder with LM and fine-tuned it under the multi-task learning (MTL) framework. They found the pre-training and fine-tuning can further improve the performance of MTL for several text classification tasks. ramachandran2017unsupervised found the Seq2Seq models can be significantly improved by unsupervised pre-training. The weights of both encoder and decoder are initialized with pre-trained weights of two language models and then fine-tuned with labeled data.
Besides pre-training the contextual encoder with LM, mccan2017learn pre-trained a deep LSTM encoder from an attentional sequence-to-sequence model with machine translation (MT). The context vectors (CoVe) output by the pre-trained encoder can improve the performance of a wide variety of common NLP tasks.

Since these precursor PTMs, the modern PTMs are usually trained with larger scale corpora, more powerful or deeper architectures (e.g., Transformer), and new pre-training tasks.

peters2018elmo pre-trained 2-layer LSTM encoder with a bidirectional language model (BiLM), consisting of a forward LM and a backward LM.
The contextual representations output by the pre-trained BiLM, ELMo (Embeddings from Language Models), are shown to bring large improvements on a broad range of NLP tasks.
akbik2018contextual captured word meaning with contextual string embeddings pre-trained with character-level LM.
However, these two PTMs are usually used as a feature extractor to produce the contextual word embeddings, which are fed into the main model for downstream tasks. Their parameters are fixed, and the rest parameters of the main model are still trained from scratch.
ULMFiT (Universal Language Model Fine-tuning) [DBLP:conf/acl/RuderH18] attempted to fine-tune pre-trained LM for text classification (TC) and achieved state-of-the-art results on six widely-used TC datasets. ULMFiT consists of 3 phases: 1) pre-training LM on general-domain data; 2) fine-tuning LM on target data; 3) fine-tuning on the target task.
ULMFiT also investigates some effective fine-tuning strategies, including discriminative fine-tuning, slanted triangular learning rates, and gradual unfreezing.

More recently, the very deep PTMs have shown their powerful ability in learning universal language representations: e.g., OpenAI GPT (Generative Pre-training) [radford2018improving] and BERT (Bidirectional Encoder Representation from Transformer) [devlin2019bert]. Besides LM, an increasing number of self-supervised tasks (see Section 3.1) is proposed to make the PTMs capturing more knowledge form large scale text corpora.

Since ULMFiT and BERT, fine-tuning has become the mainstream approach to adapt PTMs for the downstream tasks.

## 3 Overview of PTMs

The major differences between PTMs are the usages of contextual encoders, pre-training tasks, and purposes. We have briefly introduced the architectures of contextual encoders in Section 2.2. In this section, we focus on the description of pre-training tasks and give a taxonomy of PTMs.

### 3.1 Pre-training Tasks

The pre-training tasks are crucial for learning the universal representation of language. Usually, these pre-training tasks should be challenging and have substantial training data. In this section, we summarize the pre-training tasks into three categories: supervised learning, unsupervised learning, and self-supervised learning.

- 1.

Supervised learning (SL) is to learn a function that maps an input to an output based on training data consisting of input-output pairs.

- 2.

Unsupervised learning (UL) is to find some intrinsic knowledge from unlabeled data, such as clusters, densities, latent representations.

- 3.

Self-Supervised learning (SSL) is a blend of supervised learning and unsupervised learning222Indeed, it is hard to clearly distinguish the unsupervised learning and self-supervised learning. For clarification, we refer “unsupervised learning” to the learning
without human-annotated supervised labels. The purpose of “self-supervised learning” is to learn the general knowledge from data rather than standard unsupervised objectives, such as density estimation..
The learning paradigm of SSL is entirely the same as supervised learning, but the labels of training data are generated automatically.
The key idea of SSL is to predict any part of the input from other parts in some form. For example, the masked language model (MLM) is a self-supervised task that attempts to predict the masked words in a sentence given the rest words.

In CV, many PTMs are trained on large supervised training sets like ImageNet.
However, in NLP, the datasets of most supervised tasks are not large enough to train a good PTM. The only exception is machine translation (MT). A large-scale MT dataset, WMT 2017, consists of more than 7 million sentence pairs.
Besides, MT is one of the most challenging tasks in NLP, and an encoder pre-trained on MT can benefit a variety of downstream NLP tasks.
As a successful PTM, CoVe [mccan2017learn] is an encoder pre-trained on MT task and improves a wide variety of common NLP tasks: sentiment analysis (SST, IMDb), question classification (TREC), entailment (SNLI), and question answering (SQuAD).

In this section, we introduce some widely-used pre-training tasks in existing PTMs. We can regard these tasks as self-supervised learning.
Table 1 also summarizes their loss functions.

Task
Loss Function

Description

LM
ℒLM=−∑t=1Tlogp(xt|𝐱<t)fragmentsLLMfragmentst1𝑇p(x𝑡|xfragmentst)\displaystyle\mathcal{L}_{\textrm{\tiny LM}}=-\sum_{t=1}^{T}\log p(x_{t}|\mathbf{x}_{<t})

𝐱<t=x1,x2,⋯,xt−1fragmentsxfragmentstx1,x2,⋯,xfragmentst1\mathbf{x}_{<t}=x_{1},x_{2},\cdots,x_{t-1}.

MLM
ℒMLM=−∑x^∈m(𝐱)logp(x^|𝐱∖m(𝐱))fragmentsLMLMfragments^𝑥m(x)p(^𝑥|xfragmentsm(x))\displaystyle\mathcal{L}_{\textrm{\tiny MLM}}=-\sum_{\hat{x}\in m(\mathbf{x})}\log p\Big{(}\hat{x}|\mathbf{x}_{\setminus m(\mathbf{x})}\Big{)}

m(𝐱)fragmentsm(x)m(\mathbf{x}) and 𝐱∖m(𝐱)fragmentsxfragmentsm(x)\mathbf{x}_{\setminus m(\mathbf{x})} denote the masked words from 𝐱𝐱\mathbf{x} and the rest words respectively.

Seq2Seq MLM
ℒS2SMLM=−∑t=ijlogp(xt|𝐱∖𝐱i:j,𝐱i:t−1)fragmentsLS2SMLMfragmentsti𝑗p(x𝑡|xfragmentsxfragmentsi:j,xfragmentsi:t1)\displaystyle\mathcal{L}_{\textrm{\tiny S2SMLM}}=-\sum_{t=i}^{j}\log p\Big{(}x_{t}|\mathbf{x}_{\setminus\mathbf{x}_{i:j}},\mathbf{x}_{i:t-1}\Big{)}

𝐱i:jfragmentsxfragmentsi:j\mathbf{x}_{i:j} denotes an masked n-gram span from i𝑖i to j𝑗j in 𝐱𝐱\mathbf{x}.

PLM
ℒPLM=−∑t=1Tlogp(zt|𝐳<t)fragmentsLPLMfragmentst1𝑇p(z𝑡|zfragmentst)\displaystyle\mathcal{L}_{\textrm{\tiny PLM}}=-\sum_{t=1}^{T}\log p(z_{t}|\mathbf{z}_{<t})

𝐳=perm(𝐱)fragmentszperm(x)\mathbf{z}=perm(\mathbf{x}) is a permutation of 𝐱𝐱\mathbf{x} with random order.

DAE
ℒDAE=−∑t=1Tlogp(xt|𝐱^,𝐱<t)fragmentsLDAEfragmentst1𝑇p(x𝑡|^𝐱,xfragmentst)\displaystyle\mathcal{L}_{\textrm{\tiny DAE}}=-\sum_{t=1}^{T}\log p(x_{t}|\hat{\mathbf{x}},\mathbf{x}_{<t})

𝐱^^𝐱\hat{\mathbf{x}} is randomly perturbed text from 𝐱𝐱\mathbf{x}.

DIM
ℒDIM=s(𝐱^i:j,𝐱i:j)−log∑𝐱~i:j∈𝒩s(𝐱^i:j,𝐱~i:j)fragmentsLDIMs(^𝐱fragmentsi:j,xfragmentsi:j)fragments~𝐱fragmentsi:jNs(^𝐱fragmentsi:j,~𝐱fragmentsi:j)\displaystyle\mathcal{L}_{\textrm{\tiny DIM}}=s(\hat{\mathbf{x}}_{i:j},\mathbf{x}_{i:j})-\log\sum_{\tilde{\mathbf{x}}_{i:j}\in\mathcal{N}}s(\hat{\mathbf{x}}_{i:j},\tilde{\mathbf{x}}_{i:j})

𝐱i:jfragmentsxfragmentsi:j\mathbf{x}_{i:j} denotes an n-gram span from i𝑖i to j𝑗j in 𝐱𝐱\mathbf{x}, 𝐱^i:jfragments^𝐱fragmentsi:j\hat{\mathbf{x}}_{i:j} denotes a sentence masked at position i𝑖i to j𝑗j, and 𝐱~i:jfragments~𝐱fragmentsi:j\tilde{\mathbf{x}}_{i:j} denotes a randomly-sampled negative n-gram from corpus.

NSP/SOP
ℒNSP/SOP=−logp(t|𝐱,𝐲)fragmentsLNSP/SOPp(t|x,y)\displaystyle\mathcal{L}_{\textrm{\tiny NSP/SOP}}=-\log p(t|\mathbf{x},\mathbf{y})

t=1fragmentst1t=1 if 𝐱𝐱\mathbf{x} and 𝐲𝐲\mathbf{y} are continuous segments from corpus.

RTD
ℒRTD=−∑t=1Tlogp(yt|𝐱^)fragmentsLRTDfragmentst1𝑇p(y𝑡|^𝐱)\displaystyle\mathcal{L}_{\textrm{\tiny RTD}}=-\sum_{t=1}^{T}\log p(y_{t}|\hat{\mathbf{x}})

yt=𝟏(x^t=xt)fragmentsy𝑡1(^𝑥𝑡x𝑡)y_{t}=\mathbf{1}(\hat{x}_{t}=x_{t}), 𝐱^^𝐱\hat{\mathbf{x}} is corrupted from 𝐱𝐱\mathbf{x}.

- 111

𝐱=[x1,x2,⋯,xT]fragmentsx[x1,x2,⋯,x𝑇]\mathbf{x}=[x_{1},x_{2},\cdots,x_{T}] denotes a sequence.

#### 3.1.1 Language Modeling (LM)

The most common unsupervised task in NLP is probabilistic language modeling (LM), which is a classic probabilistic density estimation problem.
Although LM is a general concept, in practice, LM often refers in particular to auto-regressive LM or unidirectional LM.

Given a text sequence 𝐱1:T=[x1,x2,⋯,xT]fragmentsxfragments1:T[x1,x2,⋯,x𝑇]\mathbf{x}_{1:T}=[x_{1},x_{2},\cdots,x_{T}], its joint probability p(x1:T)fragmentsp(xfragments1:T)p(x_{1:T}) can be decomposed as

p(𝐱1:T)=∏t=1Tp(xt|𝐱0:t−1),fragmentsp(xfragments1:T)productfragmentst1𝑇p(x𝑡|xfragments0:t1),\displaystyle p(\mathbf{x}_{1:T})=\prod_{t=1}^{T}p(x_{t}|\mathbf{x}_{0:t-1}),

(2)

where x0fragmentsx0x_{0} is special token indicating the begin of sequence.

The conditional probability p(xt|𝐱0:t−1)fragmentsp(x𝑡|xfragments0:t1)p(x_{t}|\mathbf{x}_{0:t-1}) can be modeled by a probability distribution over the vocabulary given linguistic context 𝐱0:t−1fragmentsxfragments0:t1\mathbf{x}_{0:t-1}.
The context 𝐱0:t−1fragmentsxfragments0:t1\mathbf{x}_{0:t-1} is modeled by neural encoder fenc(⋅)fragmentsfenc(⋅)f_{\mathrm{enc}}(\cdot), and the conditional probability is

p(xt|𝐱0:t−1)=gLM(fenc(𝐱0:t−1)),fragmentsp(x𝑡|xfragments0:t1)gLM(fenc(xfragments0:t1)),\displaystyle p(x_{t}|\mathbf{x}_{0:t-1})=g_{\mathrm{LM}}\Big{(}f_{\mathrm{enc}}(\mathbf{x}_{0:t-1})\Big{)},

(3)

where gLM(⋅)fragmentsgLM(⋅)g_{\mathrm{LM}}(\cdot) is prediction layer.

Given a huge corpus, we can train the entire network with maximum likelihood estimation (MLE).

A drawback of unidirectional LM is that the representation of each token encodes only the leftward context tokens and itself. However, better contextual
representations of text should encode contextual information from both directions. An improved solution is bidirectional LM (BiLM), which consists of two unidirectional LMs: a forward left-to-right LM and a backward right-to-left LM.
For BiLM, DBLP:conf/emnlp/BaevskiELZA19 proposed a two-tower model that the forward tower operates the left-to-right LM and the backward tower operates the right-to-left LM.

#### 3.1.2 Masked Language Modeling (MLM)

Masked language modeling (MLM) is first proposed by doi:10.1177/107769905303000401 in the literature, who referred to this as a Cloze task. devlin2019bert adapted this task as a novel pre-training task to overcome the drawback of the standard unidirectional LM. Loosely speaking, MLM first masks out some tokens from the input sentences and then trains the model to predict the masked tokens by the rest of the tokens. However, this pre-training method will create a mismatch between the pre-training phase and the fine-tuning phase because the mask token does not appear during the fine-tuning phase. Empirically, to deal with this issue, devlin2019bert used a special [MASK] token 80% of the time, a random token 10% of the time and the original token 10% of the time to perform masking.

##### Sequence-to-Sequence MLM (Seq2Seq MLM)

MLM is usually solved as classification problem.
We feed the masked sequences to a neural encoder whose output vectors are further fed into a softmax classifier to predict the masked token.
Alternatively, we can use encoder-decoder (aka. sequence-to-sequence) architecture for MLM, in which the encoder is fed a masked sequence, and the decoder sequentially produces the masked tokens in auto-regression fashion.
We refer to this kind of MLM as sequence-to-sequence MLM (Seq2Seq MLM), which is used in MASS [DBLP:conf/icml/SongTQLL19] and T5 [raffel2019t5]. Seq2Seq MLM can benefit the Seq2Seq-style downstream tasks, such as question answering, summarization, and machine translation.

##### Enhanced Masked Language Modeling (E-MLM)

Concurrently, there are multiple research proposing different enhanced versions of MLM to further improve on BERT.
Instead of static masking, RoBERTa [liu2019roberta] improves BERT by dynamic masking.

UniLM [DBLP:conf/nips/00040WWLWGZH19, bao2020unilmv2] extends the task of mask prediction on three types of language modeling tasks: unidirectional, bidirectional, and sequence-to-sequence prediction.
XLM [DBLP:conf/nips/ConneauL19] performs MLM on a concatenation of parallel bilingual sentence pairs, called Translation Language Modeling (TLM).
SpanBERT [joshi2019spanbert] replaces MLM with Random Contiguous Words Masking and Span Boundary Objective (SBO) to integrate structure information into pre-training, which requires the system to predict masked spans based on span boundaries. Besides, StructBERT [wang2020structbert] introduces the Span Order Recovery task to further incorporate language structures.

Another way to enrich MLM is to incorporate external knowledge (see Section LABEL:sec:ptms-knowledge).

#### 3.1.3 Permuted Language Modeling (PLM)

Despite the wide use of the MLM task in pre-training, yang2019xlnet claimed that some special tokens used in the pre-training of MLM, like [MASK], are absent when the model is applied on downstream tasks, leading to a gap between pre-training and fine-tuning. To overcome this issue, Permuted Language Modeling (PLM) [yang2019xlnet] is a pre-training objective to replace MLM. In short, PLM is a language modeling task on a random permutation of input sequences. A permutation is randomly sampled from all possible permutations. Then some of the tokens in the permuted sequence are chosen as the target, and the model is trained to predict these targets, depending on the rest of the tokens and the natural positions of targets. Note that this permutation does not affect the natural positions of sequences and only defines the order of token predictions. In practice, only the last few tokens in the permuted sequences are predicted, due to the slow convergence. And a special two-stream self-attention is introduced for target-aware representations.

#### 3.1.4 Denoising Autoencoder (DAE)

Denoising autoencoder (DAE) takes a partially corrupted input and aims to recover the original undistorted input. Specific to language, a sequence-to-sequence model, such as the standard Transformer, is used to reconstruct the original text. There are several ways to corrupt text [lewis2019bart]:

(1) Token Masking: Randomly sampling tokens from the input and replacing them with [MASK] elements.

(2) Token Deletion: Randomly deleting tokens from the input. Different from token masking, the model needs to decide the positions of missing inputs.

(3) Text Infilling: Like SpanBERT, a number of text spans are sampled and replaced with a single [MASK] token. Each span length is drawn from a Poisson distribution (λ=3fragmentsλ3\lambda=3). The model needs to predict how many tokens are missing from a span.

(4) Sentence Permutation: Dividing a document into sentences based on full stops and shuffling these sentences in random order.

(5) Document Rotation: Selecting a token uniformly at random and rotating the document so that it begins with that token. The model needs to identify the real start position of the document.

#### 3.1.5 Contrastive Learning (CTL)

Contrastive learning [saunshi2019theoretical]
assumes some observed pairs of text that are more semantically similar than randomly sampled
text. A score function s(x,y)fragmentss(x,y)s(x,y) for text pair (x,y)fragments(x,y)(x,y) is learned to minimize the objective function:

ℒCTL=𝔼x,y+,y−[−logexp(s(x,y+))exp(s(x,y+))+exp(s(x,y−))],fragmentsLCTLEfragmentsx,y,y[fragments(s(x,y))fragments(s(x,y))(s(x,y))],\displaystyle\mathcal{L}_{\textrm{\tiny CTL}}=\mathbb{E}_{x,y^{+},y^{-}}\Big{[}-\log\frac{\exp\big{(}s(x,y^{+})\big{)}}{\exp\big{(}s(x,y^{+})\big{)}+\exp\big{(}s(x,y^{-})\big{)}}\Big{]},

(4)

where (x,y+)fragments(x,y)(x,y^{+}) are a similar pair and y−fragmentsyy^{-}
is presumably dissimilar to x𝑥x. y+fragmentsyy^{+} and y−fragmentsyy^{-} are typically called positive and negative sample. The score function s(x,y)fragmentss(x,y)s(x,y) is often computed by a learnable neural encoder in two ways: s(x,y)=fenc(x)Tfenc(y)fragmentss(x,y)ffragmentsenc(x)Tffragmentsenc(y)s(x,y)=f_{\mathrm{enc}(x)}^{\mathrm{\scriptscriptstyle T}}f_{\mathrm{enc}(y)} or s(x,y)=fenc(x⊕y)fragmentss(x,y)fenc(xdirect-sumy)s(x,y)=f_{\mathrm{enc}}(x\oplus y).

The idea behind CTL is “learning by comparison”.
Compared to LM, CTL usually has less computational complexity and therefore is desirable alternative training criteria for PTMs.

DBLP:journals/jmlr/CollobertWBKKK11 proposed pairwise ranking task to distinguish real and fake phrases. The model needs to predict a higher score for a legal phrase than an incorrect phrase obtained by replacing its central word with a random word.
mnih2013learning trained word embeddings efficiently with Noise-Contrastive Estimation (NCE) [gutmann2010noise], which trains a binary classifier to distinguish real and fake samples. The idea of NCE is also used in the well-known word2vec embedding [mikolov2013word2vec].

We briefly describe some recently proposed CTL tasks in the following paragraphs.

##### Deep InfoMax (DIM)

Deep InfoMax (DIM) [DBLP:conf/iclr/HjelmFLGBTB19] is originally proposed for images, which improves the quality of the representation by maximizing the mutual information between an image
representation and local regions of the image.

kong2019mutual applied DIM to language representation learning.
The global representation of a sequence x𝑥x is defined to be the hidden state of the first token (assumed to be a special start of sentence symbol) output by contextual encoder fenc(𝐱)fragmentsfenc(x)f_{\mathrm{enc}}(\mathbf{x}).
The objective of DIM is to assign a higher score for
fenc(𝐱i:j)Tfenc(𝐱^i:j)fragmentsfenc(xfragmentsi:j)Tfenc(^𝐱fragmentsi:j)f_{\mathrm{enc}}(\mathbf{x}_{i:j})^{\mathrm{\scriptscriptstyle T}}f_{\mathrm{enc}}(\hat{\mathbf{x}}_{i:j}) than fenc(𝐱~i:j)Tfenc(𝐱^i:j)fragmentsfenc(~𝐱fragmentsi:j)Tfenc(^𝐱fragmentsi:j)f_{\mathrm{enc}}(\tilde{\mathbf{x}}_{i:j})^{\mathrm{\scriptscriptstyle T}}f_{\mathrm{enc}}(\hat{\mathbf{x}}_{i:j}),
where 𝐱i:jfragmentsxfragmentsi:j\mathbf{x}_{i:j} denotes an n-gram333n𝑛n is drawn from a Gaussian distribution
𝒩(5,1)fragmentsN(5,1)\mathcal{N}(5,1) clipped at 1 (minimum length) and 10 (maximum length). span from i𝑖i to j𝑗j in 𝐱𝐱\mathbf{x}, 𝐱^i:jfragments^𝐱fragmentsi:j\hat{\mathbf{x}}_{i:j} denotes a sentence masked at position i𝑖i to j𝑗j, and 𝐱~i:jfragments~𝐱fragmentsi:j\tilde{\mathbf{x}}_{i:j} denotes a randomly-sampled negative n-gram from corpus.

##### Replaced Token Detection (RTD)

Replaced Token Detection (RTD) is the same as NCE but predicts whether a token is replaced given its surrounding context.

CBOW with negative sampling (CBOW-NS) [mikolov2013word2vec] can be viewed as a simple version of RTD, in which the negative samples are randomly sampled from vocabulary with simple proposal distribution.

ELECTRA [clark2020electra] improves RTD by utilizing a generator to replacing some tokens of a sequence. A generator G𝐺G and a discriminator D𝐷D are trained following a two-stage procedure: (1) Train only the generator with MLM task for n1fragmentsn1n_{1} steps; (2) Initialize the weights of the discriminator with the weights of the generator. Then train the discriminator with a discriminative task for n2fragmentsn2n_{2} steps, keeping G𝐺G frozen. Here the discriminative task indicates justifying whether the input token has been replaced by G𝐺G or not.
The generator is thrown after pre-training, and only the discriminator will be fine-tuned on downstream tasks.

RTD is also an alternative solution for the mismatch problem. The network sees [MASK] during pre-training but not when being fine-tuned in downstream tasks.

Similarly, WKLM [xiong2019pretrain] replaces words on the entity-level instead of token-level. Concretely, WKLM replaces entity mentions with names of other entities of the same type and train the models to distinguish whether the entity has been replaced.

0\float@count-1\float@count\e@alloc@chardef\forest@temp@box\float@count=dj\e@ch@ck\float@count\e@alloc@chardef\e@alloc@chardef\float@count0\float@count-1\float@count\e@alloc@chardef\forest@temp@box\float@count=

PTMs

GPT [radford2018improving], GPT-2 [radford2019language],GPT-3 [Brown2020GPT3]

MASS [DBLP:conf/icml/SongTQLL19], BART [lewis2019bart],T5 [raffel2019t5], XNLG [chi2019cross], mBART [liu2020multilingual]

MT

CoVe [mccan2017learn]

LM

MLM

TLM

Seq2Seq MLM

PLM

XLNet [yang2019xlnet]

DAE

BART [lewis2019bart]

CTL

RTD

NSP

SOP

Conversion to HTML had a Fatal error and exited abruptly. This document may be truncated or damaged.

Generated on Sat Mar 9 07:30:39 2024 by LaTeXML
