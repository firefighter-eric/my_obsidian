# Liu, Lapata - 2020 - Text summarization with pretrained encoders

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Liu, Lapata - 2020 - Text summarization with pretrained encoders.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/1908.08345
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Text Summarization with Pretrained Encoders

Yang Liu

  
Mirella Lapata 
Institute for Language, Cognition and Computation

School of Informatics, University of Edinburgh

yang.liu2@ed.ac.uk, mlap@inf.ed.ac.uk

###### Abstract

Bidirectional Encoder Representations from Transformers
(Bert; Devlin et al. 2019) represents the latest
incarnation of pretrained language models which have recently
advanced a wide range of natural language processing tasks. In
this paper, we showcase how Bert can be usefully applied
in text summarization and propose a general framework for both
extractive and abstractive models. We introduce a novel
document-level encoder based on Bert which is able to
express the semantics of a document and obtain representations
for its sentences. Our extractive model is built on top of this
encoder by stacking several inter-sentence Transformer layers.
For abstractive summarization, we propose a new fine-tuning
schedule which adopts different optimizers for the encoder and
the decoder as a means of alleviating the mismatch between the
two (the former is pretrained while the latter is not). We also
demonstrate that a two-staged fine-tuning approach can further
boost the quality of the generated summaries. Experiments on
three datasets show that our model achieves state-of-the-art
results across the board in both extractive and abstractive
settings.111Our code is available at
https://github.com/nlpyang/PreSumm.

## 1 Introduction

Language model pretraining has advanced the state of the art in many
NLP tasks ranging from sentiment analysis, to question answering,
natural language inference, named entity recognition, and textual
similarity. State-of-the-art pretrained models include ELMo
Peters et al. (2018), GPT Radford et al. (2018), and more
recently Bidirectional Encoder Representations from Transformers
(Bert; Devlin et al. 2019). Bert combines both
word and sentence representations in a single very large Transformer
Vaswani et al. (2017); it is pretrained on vast amounts of text,
with an unsupervised objective of masked language modeling and
next-sentence prediction and can be fine-tuned with various
task-specific objectives.

In most cases, pretrained language models have been employed as
encoders for sentence- and paragraph-level natural language
understanding problems Devlin et al. (2019) involving various
classification tasks (e.g., predicting whether any two sentences are
in an entailment relationship; or determining the completion of a
sentence among four alternative sentences).
In this paper, we examine
the influence of language model pretraining on text
summarization. Different from previous tasks, summarization requires
wide-coverage natural language understanding going beyond the meaning
of individual words and sentences.
The aim is to condense a
document into a shorter version while preserving most of its
meaning. Furthermore, under abstractive modeling formulations, the task
requires language generation capabilities in order to create summaries
containing novel words and phrases not featured in the source text, while extractive summarization is often defined as a binary
classification task with labels indicating whether a text span
(typically a sentence) should be included in the summary.

We explore the potential of Bert for text summarization under
a general framework encompassing both extractive and abstractive
modeling paradigms. We propose a novel document-level encoder based
on Bert which is able to encode a document and obtain
representations for its sentences. Our extractive model is built on
top of this encoder by stacking several inter-sentence Transformer
layers to capture document-level features for extracting sentences.
Our abstractive model adopts an encoder-decoder architecture,
combining the same pretrained Bert encoder with a
randomly-initialized Transformer decoder
Vaswani et al. (2017). We design a new training schedule which
separates the optimizers of the encoder and the decoder in order to
accommodate the fact that the former is pretrained while the latter
must be trained from scratch. Finally, motivated by previous work
showing that the combination of extractive and abstractive objectives
can help generate better summaries Gehrmann et al. (2018), we
present a two-stage approach where the encoder is fine-tuned twice,
first with an extractive objective and subsequently on the abstractive
summarization task.

We evaluate the proposed approach on three single-document news
summarization datasets representative of different writing
conventions (e.g., important information is concentrated at the
beginning of the document or distributed more evenly throughout)
and summary styles (e.g., verbose vs. more telegraphic; extractive
vs. abstractive). Across datasets, we experimentally show that the
proposed models achieve state-of-the-art results under both
extractive and abstractive settings. Our contributions in this
work are three-fold: a) we highlight the importance of document
encoding for the summarization task; a variety of recently
proposed techniques aim to enhance summarization performance via
copying mechanisms Gu et al. (2016); See et al. (2017); Nallapati et al. (2017),
reinforcement learning Narayan et al. (2018b); Paulus et al. (2018); Dong et al. (2018),
and multiple communicating encoders Celikyilmaz et al. (2018). We
achieve better results with a minimum-requirement model without
using any of these mechanisms; b) we showcase ways to effectively
employ pretrained language models in summarization under both
extractive and abstractive settings; we would expect any
improvements in model pretraining to translate in better
summarization in the future; and c) the proposed models can be
used as a stepping stone to further improve summarization
performance as well as baselines against which new proposals are
tested.

## 2 Background

### 2.1 Pretrained Language Models

Pretrained language models Peters et al. (2018); Radford et al. (2018); Devlin et al. (2019); Dong et al. (2019); Zhang et al. (2019) have recently emerged as a key technology
for achieving impressive gains in a wide variety of natural language
tasks. These models extend the idea of word embeddings
by learning contextual representations from large-scale corpora using
a language modeling objective. Bidirectional Encoder Representations
from Transformers (Bert; Devlin et al. 2019) is a new
language representation model which is trained with a masked language
modeling and a “next sentence prediction” task on a corpus
of 3,300M words.

The general architecture of Bert is shown in the left part
of Figure 1. Input text is first
preprocessed by inserting two special tokens. [cls] is
appended to the beginning of the text; the output representation of
this token is used to aggregate information from the whole sequence
(e.g., for classification tasks). And token [sep] is
inserted after each sentence as an indicator of sentence boundaries.
The modified text is then represented as a sequence of tokens
X=[w1,w2,⋯,wn]𝑋subscript𝑤1subscript𝑤2⋯subscript𝑤𝑛X=[w_{1},w_{2},\cdots,w_{n}]. Each token wisubscript𝑤𝑖w_{i} is assigned three kinds of
embeddings: token embeddings indicate the meaning of each
token, segmentation embeddings are used to discriminate between
two sentences (e.g., during a sentence-pair classification task) and
position embeddings indicate the position of each token within
the text sequence. These three embeddings are summed to a single
input vector xisubscript𝑥𝑖x_{i} and fed to a bidirectional Transformer with
multiple layers:

h~l=LN​(hl−1+MHAtt​(hl−1))superscript~ℎ𝑙LNsuperscriptℎ𝑙1MHAttsuperscriptℎ𝑙1\displaystyle\tilde{h}^{l}=\mathrm{LN}(h^{l-1}+\mathrm{MHAtt}(h^{l-1}))

(1)

hl=LN​(h~l+FFN​(h~l))superscriptℎ𝑙LNsuperscript~ℎ𝑙FFNsuperscript~ℎ𝑙\displaystyle h^{l}=\mathrm{LN}(\tilde{h}^{l}+\mathrm{FFN}(\tilde{h}^{l}))

(2)

where h0=xsuperscriptℎ0𝑥h^{0}=x are the input vectors; LNLN\mathrm{LN} is the layer
normalization operation Ba et al. (2016); MHAttMHAtt\mathrm{MHAtt} is
the multi-head attention operation Vaswani et al. (2017);
superscript l𝑙l indicates the depth of the stacked layer. On the
top layer, Bert will generate an output vector tisubscript𝑡𝑖t_{i} for
each token with rich contextual information.

Pretrained language models are usually used to enhance performance
in language understanding tasks. Very recently, there have been
attempts to apply pretrained models to various generation
problems Edunov et al. (2019); Rothe et al. (2019). When
fine-tuning for a specific task, unlike ELMo whose parameters are
usually fixed, parameters in Bert are jointly
fine-tuned with additional task-specific parameters.

### 2.2 Extractive Summarization

Extractive summarization systems create a summary by identifying (and
subsequently concatenating) the most important sentences in a
document.
Neural models consider extractive summarization as a sentence
classification problem: a neural encoder creates sentence
representations and a classifier predicts which sentences should be
selected as summaries.
SummaRuNNer Nallapati et al. (2017) is one of the
earliest neural approaches adopting an encoder based on Recurrent
Neural Networks. Refresh Narayan et al. (2018b) is a
reinforcement learning-based system trained by globally optimizing the
ROUGE metric. More recent work achieves higher performance with more
sophisticated model structures. Latent
Zhang et al. (2018) frames extractive summarization as a latent
variable inference problem; instead of maximizing the likelihood of
“gold” standard labels, their latent model directly maximizes the
likelihood of human summaries given selected sentences.
Sumo
Liu et al. (2019) capitalizes on the notion of structured attention to
induce a multi-root dependency tree representation of the document
while predicting the output summary.
NeuSum Zhou et al. (2018) scores and selects sentences
jointly and represents the state of the art in extractive
summarization.

### 2.3 Abstractive Summarization

Neural approaches to abstractive summarization conceptualize the task
as a sequence-to-sequence problem,
where an encoder maps a sequence of tokens in the source document
𝒙=[x1,…,xn]𝒙subscript𝑥1…subscript𝑥𝑛\bm{x}=[x_{1},...,x_{n}] to a sequence of continuous
representations 𝒛=[z1,…,zn]𝒛subscript𝑧1…subscript𝑧𝑛\bm{z}=[z_{1},...,z_{n}], and a decoder then
generates the target summary 𝒚=[y1,…,ym]𝒚subscript𝑦1…subscript𝑦𝑚\bm{y}=[y_{1},...,y_{m}]
token-by-token, in an auto-regressive manner, hence modeling the
conditional probability: p​(y1,…,ym|x1,…,xn)𝑝subscript𝑦1…conditionalsubscript𝑦𝑚subscript𝑥1…subscript𝑥𝑛p(y_{1},...,y_{m}|x_{1},...,x_{n}).

Rush et al. (2015) and Nallapati et al. (2016) were among
the first to apply the neural encoder-decoder architecture to text
summarization. See et al. (2017) enhance this model with a
pointer-generator network (PTgen) which allows it to copy words
from the source text, and a coverage mechanism (Cov) which
keeps track of words that have been summarized.
Celikyilmaz et al. (2018) propose an abstractive system where
multiple agents (encoders) represent the document together with a
hierarchical attention mechanism (over the agents) for decoding.
Their Deep Communicating Agents (DCA) model is trained
end-to-end with reinforcement learning. Paulus et al. (2018) also
present a deep reinforced model (DRM) for abstractive
summarization which handles the coverage problem with an
intra-attention mechanism where the decoder attends over previously
generated words. Gehrmann et al. (2018) follow a bottom-up
approach (BottomUp); a content selector first determines
which phrases in the source document should be part of the summary, and
a copy mechanism is applied only to preselected phrases during
decoding. Narayan et al. (2018a) propose an abstractive model which is
particularly suited to extreme summarization (i.e., single sentence
summaries), based on convolutional neural networks and additionally
conditioned on topic distributions (TConvS2S).

## 3 Fine-tuning Bert for Summarization

### 3.1 Summarization Encoder

Although Bert has been used to fine-tune various NLP tasks,
its application to summarization is not as straightforward. Since
Bert is trained as a masked-language model, the output
vectors are grounded to tokens instead of sentences, while in
extractive summarization, most models manipulate sentence-level
representations. Although segmentation embeddings represent different
sentences in Bert, they only apply to sentence-pair inputs,
while in summarization we must encode and manipulate multi-sentential
inputs. Figure 1 illustrates our proposed
Bert architecture for Summarization (which we call
BertSum).

In order to represent individual sentences, we insert
external [cls] tokens at the start of each sentence, and
each [cls] symbol collects features for the sentence
preceding it. We also use interval segment embeddings to
distinguish multiple sentences within a document. For s​e​n​ti𝑠𝑒𝑛subscript𝑡𝑖sent_{i} we
assign segment embedding EAsubscript𝐸𝐴E_{A} or EBsubscript𝐸𝐵E_{B} depending on whether i𝑖i
is odd or even. For example, for document [s​e​n​t1,s​e​n​t2,s​e​n​t3,s​e​n​t4,s​e​n​t5]𝑠𝑒𝑛subscript𝑡1𝑠𝑒𝑛subscript𝑡2𝑠𝑒𝑛subscript𝑡3𝑠𝑒𝑛subscript𝑡4𝑠𝑒𝑛subscript𝑡5[sent_{1},sent_{2},sent_{3},sent_{4},sent_{5}], we would assign embeddings [EA,EB,EA,EB,EA]subscript𝐸𝐴subscript𝐸𝐵subscript𝐸𝐴subscript𝐸𝐵subscript𝐸𝐴[E_{A},E_{B},E_{A},E_{B},E_{A}]. This way, document representations are learned
hierarchically where lower Transformer layers represent adjacent
sentences, while higher layers, in combination with
self-attention, represent multi-sentence discourse.

Position embeddings in the original Bert model have a maximum
length of 512; we overcome this limitation by adding more position
embeddings that are initialized randomly and fine-tuned with other
parameters in the encoder.

### 3.2 Extractive Summarization

Let d𝑑d denote a document containing sentences [s​e​n​t1,s​e​n​t2,⋯,s​e​n​tm]𝑠𝑒𝑛subscript𝑡1𝑠𝑒𝑛subscript𝑡2⋯𝑠𝑒𝑛subscript𝑡𝑚[sent_{1},sent_{2},\cdots,sent_{m}], where s​e​n​ti𝑠𝑒𝑛subscript𝑡𝑖sent_{i} is the i𝑖i-th sentence in the
document. Extractive summarization can be defined as the task of
assigning a label yi∈{0,1}subscript𝑦𝑖01y_{i}\in\{0,1\} to each s​e​n​ti𝑠𝑒𝑛subscript𝑡𝑖sent_{i}, indicating
whether the sentence should be included in the summary. It is assumed
that summary sentences represent the most important content of the
document.

With BertSum, vector tisubscript𝑡𝑖t_{i} which is the vector of the i𝑖i-th
[cls] symbol from the top layer can be used as the
representation for s​e​n​ti𝑠𝑒𝑛subscript𝑡𝑖sent_{i}. Several inter-sentence Transformer
layers are then stacked on top of Bert outputs, to capture
document-level features for extracting summaries:

h~l=LN​(hl−1+MHAtt​(hl−1))superscript~ℎ𝑙LNsuperscriptℎ𝑙1MHAttsuperscriptℎ𝑙1\displaystyle\tilde{h}^{l}=\mathrm{LN}(h^{l-1}+\mathrm{MHAtt}(h^{l-1}))

(3)

hl=LN​(h~l+FFN​(h~l))superscriptℎ𝑙LNsuperscript~ℎ𝑙FFNsuperscript~ℎ𝑙\displaystyle h^{l}=\mathrm{LN}(\tilde{h}^{l}+\mathrm{FFN}(\tilde{h}^{l}))

(4)

where h0=PosEmb​(T)superscriptℎ0PosEmb𝑇h^{0}=\mathrm{PosEmb}(T); T𝑇T denotes the sentence vectors
output by BertSum, and function PosEmbPosEmb\mathrm{PosEmb} adds
sinusoid positional embeddings Vaswani et al. (2017) to T𝑇T,
indicating the position of each sentence.

The final output layer is a sigmoid classifier:

y^i=σ​(Wo​hiL+bo)subscript^𝑦𝑖𝜎subscript𝑊𝑜superscriptsubscriptℎ𝑖𝐿subscript𝑏𝑜\hat{y}_{i}=\sigma(W_{o}h_{i}^{L}+b_{o})

(5)

where hiLsubscriptsuperscriptℎ𝐿𝑖h^{L}_{i} is the vector for s​e​n​ti𝑠𝑒𝑛subscript𝑡𝑖sent_{i} from the top layer (the
L𝐿L-th layer ) of the Transformer. In experiments, we implemented
Transformers with L=1,2,3𝐿123L=1,2,3 and found that a Transformer with
L=2𝐿2L=2 performed best.
We name this model BertSumExt.

The loss of the model is the binary classification entropy of
prediction y^isubscript^𝑦𝑖\hat{y}_{i} against gold label yisubscript𝑦𝑖y_{i}. Inter-sentence
Transformer layers are jointly fine-tuned with BertSum.
We use the Adam optimizer with β1=0.9subscript𝛽10.9\beta_{1}=0.9, and
β2=0.999subscript𝛽20.999\beta_{2}=0.999). Our learning rate schedule
follows Vaswani et al. (2017) with warming-up (warmup=10,000warmup10000\operatorname{\operatorname{warmup}}=10,000):

lr=2e−3⋅min(step,−0.5step⋅warmup)−1.5lr=2e^{-3}\cdot\min{}(\operatorname{\operatorname{step}}{}^{-0.5},\operatorname{\operatorname{step}}{}\cdot\operatorname{\operatorname{warmup}}{}^{-1.5})

Datasets
# docs (train/val/test)
avg. doc length
 avg. summary length
% novel bi-grams

words
sentences
words
sentences
in gold summary

 CNN
90,266/1,220/1,093
760.50
33.98
45.70
3.59
52.90

DailyMail
196,961/12,148/10,397
653.33
29.33
54.65
3.86
52.16

NYT
96,834/4,000/3,452
800.04
35.55
45.54
2.44
54.70

XSum
204,045/11,332/11,334
431.07
19.77
23.26
1.00
83.31

### 3.3 Abstractive Summarization

We use a standard encoder-decoder framework for abstractive
summarization See et al. (2017). The encoder is the pretrained
BertSum and the decoder is a 6-layered Transformer
initialized randomly. It is conceivable that there is a mismatch
between the encoder and the decoder, since the former is pretrained
while the latter must be trained from scratch. This can make
fine-tuning unstable; for example, the encoder might overfit the data
while the decoder underfits, or vice versa. To circumvent this, we
design a new fine-tuning schedule which separates the optimizers of
the encoder and the decoder.

We use two Adam optimizers with β1=0.9subscript𝛽10.9\beta_{1}=0.9 and
β2=0.999subscript𝛽20.999\beta_{2}=0.999 for the encoder and the decoder, respectively, each
with different warmup-steps and learning rates:

l​rℰ=l​r~ℰ⋅min⁡(s​t​e​p−0.5,step⋅warmupℰ−1.5)𝑙subscript𝑟ℰ⋅subscript~𝑙𝑟ℰ𝑠𝑡𝑒superscript𝑝0.5⋅stepsuperscriptsubscriptwarmupℰ1.5\displaystyle lr_{\mathcal{E}}=\tilde{lr}_{\mathcal{E}}\cdot\min(step^{-0.5},\operatorname{\operatorname{step}}\cdot\operatorname{\operatorname{warmup}}_{\mathcal{E}}^{-1.5})

(6)

l​r𝒟=l​r~𝒟⋅min⁡(s​t​e​p−0.5,step⋅warmup𝒟−1.5)𝑙subscript𝑟𝒟⋅subscript~𝑙𝑟𝒟𝑠𝑡𝑒superscript𝑝0.5⋅stepsuperscriptsubscriptwarmup𝒟1.5\displaystyle lr_{\mathcal{D}}=\tilde{lr}_{\mathcal{D}}\cdot\min(step^{-0.5},\operatorname{\operatorname{step}}\cdot\operatorname{\operatorname{warmup}}_{\mathcal{D}}^{-1.5})

(7)

where l​r~ℰ=2​e−3subscript~𝑙𝑟ℰ2superscript𝑒3\tilde{lr}_{\mathcal{E}}=2e^{-3}, and
warmupℰ=20,000subscriptwarmupℰ20000\operatorname{\operatorname{warmup}}_{\mathcal{E}}=20,000 for the encoder and
l​r~𝒟=0.1subscript~𝑙𝑟𝒟0.1\tilde{lr}_{\mathcal{D}}=0.1, and warmup𝒟=10,000subscriptwarmup𝒟10000\operatorname{\operatorname{warmup}}_{\mathcal{D}}=10,000 for
the decoder. This is based on the assumption that the pretrained
encoder should be fine-tuned with a smaller learning rate and smoother
decay (so that the encoder can be trained with more accurate gradients when the decoder is becoming stable).

In addition, we propose a two-stage fine-tuning approach, where we
first fine-tune the encoder on the extractive summarization task
(Section 3.2) and then fine-tune it on the abstractive
summarization task (Section 3.3). Previous
work Gehrmann et al. (2018); Li et al. (2018) suggests that using
extractive objectives can boost the performance of abstractive
summarization. Also notice that this two-stage approach is
conceptually very simple, the model can take advantage of information
shared between these two tasks, without fundamentally changing its
architecture. We name the default abstractive model
BertSumAbs and the two-stage fine-tuned model
BertSumExtAbs.

## 4 Experimental Setup

In this section, we describe the summarization datasets used in our
experiments and discuss various implementation details.

### 4.1 Summarization Datasets

We evaluated our model on three benchmark datasets, namely the
CNN/DailyMail news highlights dataset Hermann et al. (2015),
the New York Times Annotated Corpus (NYT; Sandhaus 2008),
and XSum Narayan et al. (2018a). These datasets represent different summary
styles ranging from highlights to very brief one sentence
summaries. The summaries also vary with respect to the type of
rewriting operations they exemplify (e.g., some showcase more cut
and paste operations while others are genuinely
abstractive). Table 1 presents statistics on
these datasets (test set); example (gold-standard) summaries are
provided in the supplementary material.

#### CNN/DailyMail

contains news articles and associated
highlights, i.e., a few bullet points giving a brief overview of
the article. We used the standard splits
of Hermann et al. (2015) for training, validation, and
testing (90,266/1,220/1,093 CNN documents and
196,961/12,148/10,397 DailyMail documents). We did not anonymize
entities. We first split sentences with the Stanford CoreNLP
toolkit Manning et al. (2014) and pre-processed the
dataset following See et al. (2017). Input documents were
truncated to 512 tokens.

#### NYT

contains 110,540 articles with abstractive
summaries. Following Durrett et al. (2016), we split these
into 100,834/9,706 training/test examples, based on the date of
publication (the test set contains all articles published from
January 1, 2007 onward). We used 4,000 examples from the training
as validation set. We also followed their filtering procedure,
documents with summaries less than 50 words were removed from the
dataset. The filtered test set (NYT50) includes 3,452 examples.
Sentences were split with the Stanford CoreNLP toolkit
Manning et al. (2014) and pre-processed following
Durrett et al. (2016). Input documents were truncated
to 800 tokens.

#### XSum

contains 226,711 news articles accompanied with a
one-sentence summary, answering the question “What is this
article about?”. We used the splits of Narayan et al. (2018a) for
training, validation, and testing (204,045/11,332/11,334) and
followed the pre-processing introduced in their work. Input
documents were truncated to 512 tokens.

Aside from various statistics on the three datasets,
Table 1 also reports the proportion of novel
bi-grams in gold summaries as a measure of their
abstractiveness. We would expect models with extractive biases to
perform better on datasets with (mostly) extractive summaries, and
abstractive models to perform more rewrite operations on datasets
with abstractive summaries. CNN/DailyMail and NYT are somewhat
extractive, while XSum is highly abstractive.

### 4.2 Implementation Details

For both extractive and abstractive settings, we used PyTorch,
OpenNMT Klein et al. (2017) and the
‘bert-base-uncased’222https://git.io/fhbJQ version of
Bert to implement BertSum. Both source and
target texts were tokenized with Bert’s subwords tokenizer.

#### Extractive Summarization

All extractive models were trained for 50,000 steps on 3 GPUs (GTX
1080 Ti) with gradient accumulation every two steps. Model
checkpoints were saved and evaluated on the validation set every
1,000 steps. We selected the top-3 checkpoints based on the
evaluation loss on the validation set, and report the averaged
results on the test set.
We used a greedy
algorithm similar to Nallapati et al. (2017) to obtain an
oracle summary for each document to train extractive models. The algorithm generates an oracle
consisting of multiple sentences which maximize the ROUGE-2 score
against the gold summary.

When predicting summaries for a new document, we first use the
model to obtain the score for each sentence. We then rank these
sentences by their scores from highest to lowest, and select the
top-3 sentences as the summary.

During sentence selection we use Trigram Blocking to
reduce redundancy Paulus et al. (2018). Given summary S𝑆S and
candidate sentence c𝑐c, we skip c𝑐c if there exists a trigram
overlapping between c𝑐c and S𝑆S. The intuition is similar to
Maximal Marginal Relevance (MMR; Carbonell and Goldstein 1998); we
wish to minimize the similarity between the sentence being
considered and sentences which have been already selected as part
of the summary.

#### Abstractive Summarization

In all abstractive models, we applied dropout (with
probability 0.10.10.1) before all linear layers; label
smoothing (Szegedy et al., 2016) with smoothing
factor 0.10.10.1 was also used. Our Transformer decoder has
768 hidden units and the hidden size for all feed-forward layers
is 2,048. All models were trained for 200,000 steps on 4 GPUs
(GTX 1080 Ti) with gradient accumulation every five steps. Model
checkpoints were saved and evaluated on the validation set every
2,500 steps. We selected the top-3 checkpoints based on their
evaluation loss on the validation set, and report the averaged
results on the test set.

During decoding we used beam search (size 5), and tuned the
α𝛼\alpha for the length penalty (Wu et al., 2016) between 0.60.60.6
and 1 on the validation set; we decode until an end-of-sequence
token is emitted and repeated trigrams are
blocked Paulus et al. (2018). It is worth noting that our
decoder applies neither a copy nor a coverage mechanism
See et al. (2017), despite their popularity in abstractive
summarization. This is mainly because we focus on building a
minimum-requirements model and these mechanisms may introduce
additional hyper-parameters to tune. Thanks to the subwords
tokenizer, we also rarely observe issues with out-of-vocabulary
words in the output; moreover, trigram-blocking produces diverse
summaries managing to reduce repetitions.

Model
R1
R2
RL

Oracle
 52.59
31.24
48.87

Lead-3
 40.42
17.62
36.67

Extractive

SummaRuNNer Nallapati et al. (2017)

 39.60
16.20
35.30

Refresh Narayan et al. (2018b)

 40.00
18.20
36.60

Latent
Zhang et al. (2018)

 41.05
18.77
37.54

NeuSum Zhou et al. (2018)

 41.59
19.01
37.98

Sumo Liu et al. (2019)

 41.00
18.40
37.20

TransformerExt

 40.90
18.02
37.17

Abstractive

PTGen See et al. (2017)

 36.44
15.66
33.42

PTGen+Cov See et al. (2017)

 39.53
17.28
36.38

DRM (Paulus et al., 2018)

 39.87
15.82
36.90

BottomUp Gehrmann et al. (2018)

 41.22
18.68
38.34

DCA
Celikyilmaz et al. (2018)

 41.69
19.47
37.92

TransformerAbs

 40.21
17.76
37.09

Bert-based

BertSumExt
 43.25
20.24
39.63

BertSumExt w/o interval embeddings
 43.20
20.22
39.59

BertSumExt (large)
 43.85
20.34
39.90

BertSumAbs
 41.72
19.39
38.76

BertSumExtAbs
 42.13
19.60
39.18

Model
R1
R2
RL

Oracle
49.18
33.24
46.02

Lead-3
39.58
20.11
35.78

Extractive

Compress Durrett et al. (2016)

42.20
24.90
—

Sumo Liu et al. (2019)

42.30
22.70
38.60

TransformerExt

41.95
22.68
38.51

Abstractive

PTGen See et al. (2017)

42.47
25.61
—

PTGen + Cov See et al. (2017)

43.71
26.40
—

DRM Paulus et al. (2018)

42.94
26.02
—

TransformerAbs

35.75
17.23
31.41

Bert-based

BertSumExt
46.66
26.35
42.62

BertSumAbs
48.92
30.84
45.41

BertSumExtAbs
49.02
31.02
45.55

## 5 Results

### 5.1 Automatic Evaluation

We evaluated summarization quality automatically using ROUGE Lin (2004). We report unigram and bigram overlap (ROUGE-1
and ROUGE-2) as a means of assessing informativeness and the longest
common subsequence (ROUGE-L) as a means of assessing
fluency.

Table 2 summarizes our results on the CNN/DailyMail
dataset. The first block in the table includes the results of an
extractive Oracle system as an upper bound. We also present the Lead-3 baseline
(which simply selects the first three sentences in a document).

The second block in the table includes various extractive models
trained on the CNN/DailyMail dataset (see
Section 2.2 for an overview). For comparison to our
own model, we also implemented a non-pretrained Transformer baseline
(TransformerExt) which uses the same architecture as
BertSumExt, but with fewer parameters. It is randomly initialized
and only trained on the summarization task. TransformerExt
has 6 layers, the hidden size is 512, and the feed-forward filter size
is 2,048. The model was trained with same settings as
in Vaswani et al. (2017).

The third block in Table 2 highlights the performance
of several abstractive models on the CNN/DailyMail dataset (see
Section 2.3 for an overview). We also include an
abstractive Transformer baseline (TransformerAbs) which has
the same decoder as our abstractive BertSum models; the encoder
is a 6-layer Transformer with 768 hidden size and 2,048 feed-forward
filter size.

The fourth block reports results with fine-tuned Bert models:
BertSumExt and its two variants (one without interval
embeddings, and one with the large version of Bert),
BertSumAbs, and BertSumExtAbs. Bert-based
models outperform the Lead-3 baseline which is not a
strawman; on the CNN/DailyMail corpus it is indeed superior to
several extractive
Nallapati et al. (2017); Narayan et al. (2018b); Zhou et al. (2018) and
abstractive models See et al. (2017). Bert models collectively
outperform all previously proposed extractive and abstractive systems,
only falling behind the Oracle upper bound. Among
Bert variants, BertSumExt performs best which is not
entirely surprising; CNN/DailyMail summaries are somewhat extractive
and even abstractive models are prone to copying sentences from the
source document when trained on this dataset See et al. (2017). Perhaps
unsurprisingly we observe that larger versions of Bert lead
to performance improvements and that interval embeddings bring only
slight gains.

Table 3 presents results on the NYT dataset. Following
the evaluation protocol in Durrett et al. (2016), we use
limited-length ROUGE Recall, where predicted summaries are truncated
to the length of the gold summaries. Again, we report the performance
of the Oracle upper bound and Lead-3 baseline. The
second block in the table contains previously proposed extractive
models as well as our own Transformer baseline. Compress
Durrett et al. (2016) is an ILP-based model which combines
compression and anaphoricity constraints. The third block includes
abstractive models from the literature, and our Transformer baseline.
Bert-based models are shown in the fourth block. Again, we
observe that they outperform previously proposed approaches. On this dataset, abstractive Bert
models generally perform better compared to BertSumExt,
almost approaching Oracle performance.

Model
R1
R2
RL

Oracle
29.79
8.81
22.66

Lead
16.30
1.60
11.95

Abstractive

PTGen See et al. (2017)

29.70
9.21
23.24

PTGen+Cov See et al. (2017)

28.10
8.02
21.72

TConvS2S Narayan et al. (2018a)

31.89
11.54
25.75

TransformerAbs

29.41
9.77
23.01

Bert-based

BertSumAbs
38.76
16.33
31.15

BertSumExtAbs
38.81
16.50
31.27

Table 4 summarizes our results on the XSum
dataset. Recall that summaries in this dataset are highly
abstractive (see Table 1) consisting of a
single sentence conveying the gist of the document. Extractive
models here perform poorly as corroborated by the low performance
of the Lead baseline (which simply selects the leading
sentence from the document), and the Oracle (which
selects a single-best sentence in each document) in
Table 4. As a result, we do not report results
for extractive models on this dataset.
The second block in Table 4 presents the results
of various abstractive models taken from Narayan et al. (2018a) and also
includes our own abstractive Transformer baseline. In the third block
we show the results of our Bert summarizers which again are
superior to all previously reported models (by a wide margin).

1
0.1
0.01
0.001

 2e-2
50.69
9.33
10.13
19.26

2e-3
37.21
8.73
9.52
16.88

### 5.2 Model Analysis

#### Learning Rates

Recall that our abstractive model uses
separate optimizers for the encoder and decoder. In Table 5
we examine whether the combination of different learning rates
(l​r~ℰsubscript~𝑙𝑟ℰ\tilde{lr}_{\mathcal{E}} and l​r~𝒟subscript~𝑙𝑟𝒟\tilde{lr}_{\mathcal{D}}) is indeed
beneficial. Specifically, we report model perplexity on the
CNN/DailyMail validation set for varying encoder/decoder learning
rates. We can see that the model performs best with
l​r~ℰ=2​e−3subscript~𝑙𝑟ℰ2𝑒3\tilde{lr}_{\mathcal{E}}=2e-3 and l​r~𝒟=0.1subscript~𝑙𝑟𝒟0.1\tilde{lr}_{\mathcal{D}}=0.1.

#### Position of Extracted Sentences

In addition to the evaluation based on ROUGE, we also analyzed in more
detail the summaries produced by our model. For the extractive
setting, we looked at the position (in the source document) of the
sentences which were selected to appear in the
summary. Figure 2 shows the proportion of selected summary
sentences which appear in the source document at positions 1, 2, and so on. The
analysis was conducted on the CNN/DailyMail dataset for Oracle
summaries, and those produced by BertSumExt and the
TransformerExt. We can see that Oracle summary
sentences are fairly smoothly distributed across documents, while
summaries created by TransformerExt mostly concentrate on the first
document sentences. BertSumExt outputs are more similar to Oracle
summaries, indicating that with the pretrained encoder, the model
relies less on shallow position features, and learns deeper document
representations.

#### Novel N-grams

We also analyzed the output of
abstractive systems by calculating the proportion of novel n-grams that
appear in the summaries but not in the source texts. The results
are shown in Figure 3. In the CNN/DailyMail
dataset, the proportion of novel n-grams in automatically
generated summaries is much lower compared to reference summaries,
but in XSum, this gap is much smaller. We also observe that on
CNN/DailyMail, BertExtAbs produces less novel n-ngrams
than BertAbs, which is not surprising. BertExtAbs
is more biased towards selecting sentences from the source
document since it is initially trained as an extractive model.

The supplementary material includes examples of system output and
additional ablation studies.

(a) CNN/DailyMail Dataset

(b) XSum dataset

### 5.3 Human Evaluation

In addition to automatic evaluation, we also evaluated system output
by eliciting human judgments. We report experiments following a
question-answering (QA)
paradigm (Clarke and Lapata, 2010; Narayan et al., 2018b) which
quantifies the degree to which summarization models retain key
information from the document. Under this paradigm, a set of questions
is created based on the gold summary under the assumption that it
highlights the most important document content. Participants are then
asked to answer these questions by reading system summaries alone
without access to the article. The more questions a system can answer,
the better it is at summarizing the document as a whole.

Moreover, we also assessed the overall
quality of the summaries produced by abstractive systems which due to
their ability to rewrite content may produce disfluent or
ungrammatical output. Specifically, we followed the Best-Worst
Scaling Kiritchenko and Mohammad (2017) method where participants were
presented with the output of two systems (and the original document)
and asked to decide which one was better according to the criteria of
Informativeness, Fluency, and Succinctness.

Extractive
CNN/DM
NYT

 Lead
42.5†
36.2†

NeuSum
42.2†
—

Sumo
41.7†
38.1†

Transformer
37.8†
32.5†

BertSum
58.9
41.9

Both types of evaluation were conducted on the Amazon Mechanical
Turk platform. For the CNN/DailyMail and NYT datasets we used the
same documents (20 in total) and questions from previous work
Narayan et al. (2018b); Liu et al. (2019). For XSum, we randomly
selected 20 documents (and their questions) from the release
of Narayan et al. (2018a). We elicited 3 responses per HIT. With regard to
QA evaluation, we adopted the scoring mechanism from
Clarke and Lapata (2010); correct answers were marked with a
score of one, partially correct answers with 0.5, and zero
otherwise. For quality-based evaluation, the rating of each system
was computed as the percentage of times it was chosen as better
minus the times it was selected as worse. Ratings thus range from
-1 (worst) to 1 (best).

CNN/DM
NYT
XSum

Abstractive
QA
Rank
QA
Rank
QA
Rank

 Lead
42.5†
—
36.2†
—
9.20†
—

PTGen
33.3†
-0.24†
30.5†
-0.27†
23.7†
-0.36†

BottomUp
40.6†
-0.16†
—
—
—
—

TConvS2S
—
—
—
—
52.1
 -0.20†

Gold
—
0.22†
—
 0.33†
—
 0.38†

BertSum
56.1
0.17
41.8
-0.07
57.5
0.19

Results for extractive and abstractive systems are shown in
Tables 6 and 7, respectively. We compared
the best performing BertSum model in each setting (extractive
or abstractive) against various state-of-the-art systems (whose output
is publicly available), the Lead baseline, and the
Gold standard as an upper bound. As shown in both tables
participants overwhelmingly prefer the output of our model against
comparison systems across datasets and evaluation paradigms. All
differences between BertSum and comparison models are
statistically significant (p<0.05𝑝0.05p<0.05), with the exception of
TConvS2S (see Table 7; XSum) in the QA
evaluation setting.

## 6 Conclusions

In this paper, we showcased how pretrained Bert can be
usefully applied in text summarization. We introduced a novel
document-level encoder and proposed a general framework for both
abstractive and extractive summarization. Experimental results across
three datasets show that our model achieves state-of-the-art results
across the board under automatic and human-based evaluation
protocols. Although we mainly focused on document encoding for
summarization, in the future, we would like to take advantage the
capabilities of Bert for language generation.

## Acknowledgments

This research is supported by a Google
PhD Fellowship to the first author. We gratefully acknowledge the
support of the European Research Council (Lapata, award number
681760, “Translating Multiple Modalities into Text”). We would
also like to thank Shashi Narayan for providing us with the XSum dataset.

## References

- Ba et al. (2016)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.

Layer normalization.

arXiv preprint arXiv:1607.06450.

- Carbonell and Goldstein (1998)

Jaime G Carbonell and Jade Goldstein. 1998.

The use of MMR and diversity-based reranking for reodering
documents and producing summaries.

In Proceedings of the 21st Annual International ACL SIGIR
Conference on Research and Development in Information Retrieval, pages
335–336, Melbourne, Australia.

- Celikyilmaz et al. (2018)

Asli Celikyilmaz, Antoine Bosselut, Xiaodong He, and Yejin Choi. 2018.

Deep communicating
agents for abstractive summarization.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 1662–1675, New Orleans,
Louisiana.

- Clarke and Lapata (2010)

James Clarke and Mirella Lapata. 2010.

Discourse constraints for document compression.

Computational Linguistics, 36(3):411–441.

- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

BERT: Pre-training of
deep bidirectional transformers for language understanding.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4171–4186,
Minneapolis, Minnesota.

- Dong et al. (2019)

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao,
Ming Zhou, and Hsiao-Wuen Hon. 2019.

Unified language model pre-training for natural language
understanding and generation.

arXiv preprint arXiv:1905.03197.

- Dong et al. (2018)

Yue Dong, Yikang Shen, Eric Crawford, Herke van Hoof, and Jackie Chi Kit
Cheung. 2018.

BanditSum:
Extractive summarization as a contextual bandit.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 3739–3748, Brussels, Belgium.

- Durrett et al. (2016)

Greg Durrett, Taylor Berg-Kirkpatrick, and Dan Klein. 2016.

Learning-based
single-document summarization with compression and anaphoricity constraints.

In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 1998–2008,
Berlin, Germany.

- Edunov et al. (2019)

Sergey Edunov, Alexei Baevski, and Michael Auli. 2019.

Pre-trained language
model representations for language generation.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4052–4059,
Minneapolis, Minnesota.

- Gehrmann et al. (2018)

Sebastian Gehrmann, Yuntian Deng, and Alexander Rush. 2018.

Bottom-up abstractive
summarization.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 4098–4109, Brussels, Belgium.

- Gu et al. (2016)

Jiatao Gu, Zhengdong Lu, Hang Li, and Victor O.K. Li. 2016.

Incorporating copying
mechanism in sequence-to-sequence learning.

In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 1631–1640,
Berlin, Germany. Association for Computational Linguistics.

- Hermann et al. (2015)

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will
Kay, Mustafa Suleyman, and Phil Blunsom. 2015.

Teaching machines to read and comprehend.

In Advances in Neural Information Processing Systems, pages
1693–1701.

- Kiritchenko and Mohammad (2017)

Svetlana Kiritchenko and Saif Mohammad. 2017.

Best-worst scaling more
reliable than rating scales: A case study on sentiment intensity annotation.

In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers), pages 465–470,
Vancouver, Canada.

- Klein et al. (2017)

Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander Rush.
2017.

OpenNMT: Open-source toolkit for neural machine translation.

In Proceedings of ACL 2017, System Demonstrations, pages
67–72, Vancouver, Canada.

- Li et al. (2018)

Wei Li, Xinyan Xiao, Yajuan Lyu, and Yuanzhuo Wang. 2018.

Improving neural
abstractive document summarization with explicit information selection
modeling.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 1787–1796, Brussels, Belgium.

- Lin (2004)

Chin-Yew Lin. 2004.

ROUGE: A package
for automatic evaluation of summaries.

In Text Summarization Branches Out, pages 74–81, Barcelona,
Spain.

- Liu et al. (2019)

Yang Liu, Ivan Titov, and Mirella Lapata. 2019.

Single document
summarization as tree induction.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 1745–1755,
Minneapolis, Minnesota.

- Manning et al. (2014)

Christopher Manning, Mihai Surdeanu, John Bauer, Jenny Finkel, Steven Bethard,
and David McClosky. 2014.

The Stanford
CoreNLP natural language processing toolkit.

In Proceedings of 52nd Annual Meeting of the Association for
Computational Linguistics: System Demonstrations, pages 55–60, Baltimore,
Maryland.

- Nallapati et al. (2017)

Ramesh Nallapati, Feifei Zhai, and Bowen Zhou. 2017.

SummaRuNNer: A recurrent neural network based sequence model
for extractive summarization of documents.

In Proceedings of the 31st AAAI Conference on Artificial
Intelligence, pages 3075–3081, San Francisco, California.

- Nallapati et al. (2016)

Ramesh Nallapati, Bowen Zhou, Cicero dos Santos, Çağlar
Gu̇lçehre, and Bing Xiang. 2016.

Abstractive text
summarization using sequence-to-sequence RNNs and beyond.

In Proceedings of The 20th SIGNLL Conference on Computational
Natural Language Learning, pages 280–290, Berlin, Germany.

- Narayan et al. (2018a)

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018a.

Don’t give me the
details, just the summary! topic-aware convolutional neural networks for
extreme summarization.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 1797–1807, Brussels, Belgium.

- Narayan et al. (2018b)

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018b.

Ranking sentences for
extractive summarization with reinforcement learning.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 1747–1759, New Orleans,
Louisiana.

- Paulus et al. (2018)

Romain Paulus, Caiming Xiong, and Richard Socher. 2018.

A deep reinforced model for abstractive summarization.

In Proceedings of the 6th International Conference on Learning
Representations, Vancouver, Canada.

- Peters et al. (2018)

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer. 2018.

Deep contextualized
word representations.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans,
Louisiana.

- Radford et al. (2018)

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018.

Improving language understanding by generative pre-training.

In CoRR, abs/1704.01444, 2017.

- Rothe et al. (2019)

Sascha Rothe, Shashi Narayan, and Aliaksei Severyn. 2019.

Leveraging pre-trained checkpoints for sequence generation tasks.

arXiv preprint arXiv:1907.12461.

- Rush et al. (2015)

Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015.

A neural attention
model for abstractive sentence summarization.

In Proceedings of the 2015 Conference on Empirical Methods in
Natural Language Processing, pages 379–389, Lisbon, Portugal.

- Sandhaus (2008)

Evan Sandhaus. 2008.

The New York Times Annotated Corpus.

Linguistic Data Consortium, Philadelphia, 6(12).

- See et al. (2017)

Abigail See, Peter J. Liu, and Christopher D. Manning. 2017.

Get to the point:
Summarization with pointer-generator networks.

In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 1073–1083,
Vancouver, Canada.

- Szegedy et al. (2016)

Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew
Wojna. 2016.

Rethinking the inception architecture for computer vision.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), pages 2818–2826, Las Vegas, Nevada.

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.

Attention is all you need.

In Advances in Neural Information Processing Systems, pages
5998–6008.

- Wu et al. (2016)

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016.

Google’s neural machine translation system: Bridging the gap between
human and machine translation.

In arXiv preprint arXiv:1609.08144.

- Zhang et al. (2018)

Xingxing Zhang, Mirella Lapata, Furu Wei, and Ming Zhou. 2018.

Neural latent
extractive document summarization.

In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 779–784, Brussels, Belgium.

- Zhang et al. (2019)

Xingxing Zhang, Furu Wei, and Ming Zhou. 2019.

HIBERT: Document
level pre-training of hierarchical bidirectional transformers for document
summarization.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 5059–5069, Florence, Italy.
Association for Computational Linguistics.

- Zhou et al. (2018)

Qingyu Zhou, Nan Yang, Furu Wei, Shaohan Huang, Ming Zhou, and Tiejun Zhao.
2018.

Neural document
summarization by jointly learning to score and select sentences.

In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 654–663,
Melbourne, Australia.

Generated on Sat Mar 2 15:01:22 2024 by LaTeXML
