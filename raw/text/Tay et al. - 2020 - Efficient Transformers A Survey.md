# Tay et al. - 2020 - Efficient Transformers A Survey

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Tay et al. - 2020 - Efficient Transformers A Survey.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2009.06732
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Efficient Transformers: A Survey

\nameYi Tay \emailyitay@google.com
\addr
Google Research
\AND\nameMostafa Dehghani \emaildehghani@google.com
\addr
Google Research, Brain team
\AND\nameDara Bahri \emaildbahri@google.com
\addr
Google Research
\AND\nameDonald Metzler \emailmetzler@google.com
\addr
Google Research

###### Abstract

Transformer model architectures have garnered immense interest lately due to their effectiveness across a range of domains like language, vision and reinforcement learning. In the field of natural language processing for example, Transformers have become an indispensable staple in the modern deep learning stack. Recently, a dizzying number of “X-former” models have been proposed - Reformer, Linformer, Performer, Longformer, to name a few - which improve upon the original Transformer architecture, many of which make improvements around computational and memory efficiency. With the aim of helping the avid researcher navigate this flurry, this paper characterizes a large and thoughtful selection of recent efficiency-flavored “X-former” models, providing an organized and comprehensive overview of existing work and models across multiple domains.

Keywords: 
Deep Learning, Natural Language Processing, Transformer Models, Attention Models, Neural Networks

## 1 Introduction

Transformers (Vaswani et al., 2017) are a formidable force in the modern deep learning stack. Transformers are pervasive and have made tremendous impact in many fields such as language understanding (Devlin et al., 2018; Brown et al., 2020; Raffel et al., 2019) and image processing (Parmar et al., 2018; Carion et al., 2020). As such, it is only natural that a wealth of research has been dedicated to making fundamental improvements to the model over the past few years (Dehghani et al., 2018; So et al., 2019; Ahmed et al., 2017). This immense interest has also spurred research into more efficient variants of the model (Kitaev et al., 2020; Roy et al., 2020; Beltagy et al., 2020; Katharopoulos et al., 2020; Tay et al., 2020b; Wang et al., 2020c; Rae et al., 2020; Choromanski et al., 2020b; Dai et al., 2020; Correia et al., 2019; Sukhbaatar et al., 2019a; Vyas et al., 2020).

There has been such a surge of Transformer model variants proposed recently, that researchers and practitioners alike may find it challenging to keep pace with the rate of innovation. As of this writing and this manuscript’s first draft (circa August 2020), there have been nearly a dozen new efficiency-focused models proposed in just the past 6 months. Thus, a survey of the existing literature is both beneficial for the community and quite timely.

The self-attention mechanism is a key defining characteristic of Transformer models. The mechanism can be viewed as a graph-like inductive bias that connects all tokens in a sequence with a relevance-based pooling operation. A well-known concern with self-attention is the quadratic time and memory complexity, which can hinder model scalability in many settings. There has been an overwhelming influx of model variants proposed recently that address this problem. We hereinafter name this class of models “efficient Transformers”.

The efficiency of a model can be interpreted in a variety of ways. It might refer to the memory footprint of the model, which is of importance when the memory of accelerators on which the model is running is limited. Efficiency might also refer to computational costs, e.g. the number of FLOPs, both during training and inference. In particular, for on-device applications, models often must operate within a highly constrained computational budget. Throughout this survey, we refer to the efficiency of Transformers both in terms of memory and computation. We are especially interested in how such models perform when they are applied to large inputs.

Efficient self-attention models are crucial in applications that model long sequences. For example, documents, images, and videos are all often composed of a relatively large number of pixels or tokens. Efficiency in processing long sequences is therefore paramount for widespread adoption of Transformers.

This survey sets out to provide a comprehensive overview of the recent advances made in this class of models. We are primarily interested in modeling advances and architectural innovations that improve the general efficiency of Transformers, including but not limited to tackling the quadratic complexity issue of the self-attention mechanism or reducing the computation costs by means such as pooling and/or sparsity. We also briefly discuss general improvements and other efficiency improvements such as parameter sharing.

We propose a taxonomy of efficient Transformer models, characterizing them by their technical innovation and primary use case. Specifically, we review Transformer models that have applications in both language and vision domains, attempting to consolidate the literature across the spectrum. We also provide a detailed walk-through of many of these models and draw connections between them.

##### Author notes on the updated version (December 2021)

This manuscript went through a round of revision in December 2021 (approximately a year and 4 months later after the first manuscript was written). The main changes involve adding our discussions to better reflect the state of research at this current point of time (new models, new paradigms) and also accurately reflect the current meta trends surrounding this research area. A retrospective section is posed near the end of the paper. See Appendix for a meaningful change log of what has happened as we transitioned to V2 of this survey.

##### Author notes on the updated version (March 2022)

We wanted to post the update to arxiv in Jan but forgot about it. We lightly revised it again in Mar by adding newer SOTA sparse models such as ST-MoE-32B (Zoph et al., 2022).

## 2 Background on Transformers

This section provides an overview of the well-established Transformer architecture (Vaswani et al., 2017). Transformers are multi-layered architectures formed by stacking Transformer blocks on top of one another.

Transformer blocks are characterized by a multi-head self-attention mechanism, a position-wise feed-forward network, layer normalization (Ba et al., 2016) modules and residual connectors. The input to the Transformer model is often a tensor of shape ℝB×ℝNsuperscriptℝ𝐵superscriptℝ𝑁\mathbb{R}^{B}\times\mathbb{R}^{N}, where B𝐵B is the batch size, N𝑁N the sequence length.

The input first passes through an embedding layer that converts each one-hot token representation into a dm​o​d​e​lsubscript𝑑𝑚𝑜𝑑𝑒𝑙d_{model} dimensional embedding, i.e., ℝB×ℝN×ℝdm​o​d​e​lsuperscriptℝ𝐵superscriptℝ𝑁superscriptℝsubscript𝑑𝑚𝑜𝑑𝑒𝑙\mathbb{R}^{B}\times\mathbb{R}^{N}\times\mathbb{R}^{d_{model}}. The new tensor is then additively composed with positional encodings and passed through a multi-headed self-attention module. Positional encodings can take the form of a sinusoidal input (as per (Vaswani et al., 2017)) or be trainable embeddings.

The inputs and output of the multi-headed self-attention module are connected by residual connectors and a layer normalization layer. The output of the multi-headed self-attention module is then passed to a two-layered feed-forward network which has its inputs/outputs similarly connected in a residual fashion with layer normalization. The sub-layer residual connectors with layer norm is expressed as:

X=LayerNorm​(FS​(X))+X𝑋LayerNormsubscript𝐹𝑆𝑋𝑋\displaystyle X=\text{LayerNorm}(F_{S}(X))+X

where FSsubscript𝐹𝑆F_{S} is the sub-layer module which is either the multi-headed self-attention or the position-wise feed-forward layers.

### 2.1 Multi-Head Self-Attention

The Transformer model leverages a multi-headed self-attention mechanism. The key idea behind the mechanism is for each element in the sequence to learn to gather from other tokens in the sequence. The operation for a single head is defined as:

Ah=Softmax​(α​Qh​Kh⊤)​Vh,subscript𝐴ℎSoftmax𝛼subscript𝑄ℎsuperscriptsubscript𝐾ℎtopsubscript𝑉ℎ\displaystyle A_{h}=\text{Softmax}(\alpha Q_{h}K_{h}^{\top})V_{h},

where X𝑋X is a matrix in ℝN×dsuperscriptℝ𝑁𝑑\mathbb{R}^{N\times d}, α𝛼\alpha is a scaling factor that is typically set to 1d1𝑑\frac{1}{\sqrt{d}}, Qh=X​𝑾q,Kh=X​𝑾kformulae-sequencesubscript𝑄ℎ𝑋subscript𝑾𝑞subscript𝐾ℎ𝑋subscript𝑾𝑘Q_{h}=X\bm{W}_{q},K_{h}=X\bm{W}_{k} and Vh=X​𝑾vsubscript𝑉ℎ𝑋subscript𝑾𝑣V_{h}=X\bm{W}_{v} are linear transformations applied on the temporal dimension of the input sequence, 𝑾q,𝑾k,𝑾v∈ℝd×dHsubscript𝑾𝑞subscript𝑾𝑘subscript𝑾𝑣superscriptℝ𝑑𝑑𝐻\bm{W}_{q},\bm{W}_{k},\bm{W}_{v}\in\mathbb{R}^{d\times\frac{d}{H}} are the weight matrices (parameters) for the query, key, and value projections that project the input X𝑋X to an output tensor of d𝑑d dimensions, and NHsubscript𝑁𝐻N_{H} is the number of heads. Softmax is applied row-wise.

The outputs of heads A1​⋯​AHsubscript𝐴1⋯subscript𝐴𝐻A_{1}\cdots A_{H} are concatenated together and passed into a dense layer. The output Y𝑌Y can thus be expressed as Y=𝑾o​[A1​⋯​AH]𝑌subscript𝑾𝑜delimited-[]subscript𝐴1⋯subscript𝐴𝐻Y=\bm{W}_{o}[A_{1}\cdots A_{H}], where 𝑾osubscript𝑾𝑜\bm{W}_{o} is an output linear projection. Note that the computation of A𝐴A is typically done in a parallel fashion by considering tensors of ℝB×ℝN×ℝH×ℝdHsuperscriptℝ𝐵superscriptℝ𝑁superscriptℝ𝐻superscriptℝ𝑑𝐻\mathbb{R}^{B}\times\mathbb{R}^{N}\times\mathbb{R}^{H}\times\mathbb{R}^{\frac{d}{H}} and computing the linear transforms for all heads in parallel.

The attention matrix A=Q​K⊤𝐴𝑄superscript𝐾topA=QK^{\top} is chiefly responsible for learning alignment scores between tokens in the sequence. In this formulation, the dot product between each element/token in the query (Q𝑄Q) and key (K𝐾K) is taken. This drives the self-alignment process in self-attention whereby tokens learn to gather from each other.

### 2.2 Position-wise Feed-forward Layers

The outputs of the self-attention module are then passed into a two-layered feed-forward network with ReLU activations. This feed-forward layer operates on each position independently. This is expressed as follows:

F2​(R​e​L​U​(F1​(XA)))subscript𝐹2𝑅𝑒𝐿𝑈subscript𝐹1subscript𝑋𝐴\displaystyle F_{2}(ReLU(F_{1}(X_{A})))

where F1subscript𝐹1F_{1} and F2subscript𝐹2F_{2} are feed-forward functions of the form W​x+b𝑊𝑥𝑏Wx+b.

### 2.3 Putting it all together

Each Transformer block can be expressed as:

XAsubscript𝑋𝐴\displaystyle X_{A}
=LayerNorm​(MultiheadAttention​(X,X))+XabsentLayerNormMultiheadAttention𝑋𝑋𝑋\displaystyle=\text{LayerNorm}(\text{MultiheadAttention}(X,X))+X

XBsubscript𝑋𝐵\displaystyle X_{B}
=LayerNorm​(PositionFFN​(XA))+XAabsentLayerNormPositionFFNsubscript𝑋𝐴subscript𝑋𝐴\displaystyle=\text{LayerNorm}(\text{PositionFFN}(X_{A}))+X_{A}

where X𝑋X is the input of the Transformer block and XBsubscript𝑋𝐵X_{B} is the output of the Transformer block. Note that the MultiheadAttention​()MultiheadAttention\text{MultiheadAttention}() function accepts two argument tensors, one for query and the other for key-values. If the first argument and second argument is the same input tensor, this is the MultiheadSelfAttention mechanism.

### 2.4 On the compute cost of Transformers

The computation costs of Transformers is derived from multiple factors. Firstly, the memory and computational complexity required to compute the attention matrix is quadratic in the input sequence length, i.e., N×N𝑁𝑁N\times N. In particular, the Q​K⊤𝑄superscript𝐾topQK^{\top} matrix multiplication operation alone consumes N2superscript𝑁2N^{2} time and memory. This restricts the overall utility of self-attentive models in applications which demand the processing of long sequences. Memory restrictions are tend to be applicable more to training (due to gradient updates) and are generally of lesser impact on inference (no gradient updates). The quadratic cost of self-attention impacts speed111We would like to emphasize that complexity does not always translate to real world throughput or latency. A model of linear complexity can be slower than a model with quadratic complexity in practice. in both training and inference. The compute costs of the self-attention mechanism contributes partially to the overall compute cost of the Transformer. A non-trivial amount of compute still stems from the two layer feed-forward layers at every Transformer block (approximately half the compute time and/or FLOPs). The complexity of the FFN is linear with respect to sequence length but is generally still costly. Hence, a large portion of recent work have explored sparsity (Lepikhin et al., 2020; Fedus et al., 2021) as a means to scale up the FFN without incurring compute costs. Efficient attention and efficient models are generally orthogonal - although some efficient attention methods explicitly aim to reduce the sequence length (Dai et al., 2020) and as a result also save computation costs in both aspects. Efficiency and computational costs is generally a complicated affair and we would suggest readers peruse (Dehghani et al., 2021) for more details on trade-offs, intricacies etc.

### 2.5 Transformer Mode

It is important to note the differences in how the Transformer blocks are used. Transformers can primarily be used in three ways, namely: (1) encoder-only (e.g., for classification), (2) decoder-only (e.g., for language modeling), and (3) encoder-decoder (e.g., for machine translation). In encoder-decoder mode, there are usually multiple multi-headed self-attention modules, including a standard self-attention in both the encoder and the decoder, along with an encoder-decoder cross-attention that allows the decoder to utilize information from the encoder.
This influences the design of the self-attention mechanism. In the encoder mode, there is no restriction or constraint that the self-attention mechanism has to be causal, i.e., dependent solely on the present and past tokens. In the encoder-decoder setting, self-attention used in the decoder (i.e. across decoding positions) must be causal since each auto-regressive decoding step can only depend on previous tokens, whereas the self-attention used in the encoder need not. Fulfilling this requirement can prove challenging for many efficient self-attention designs.

The mode of usage of a Transformer model generally depends on the target application. Given an input sequence, the sequence is typically passed through an encoder stack. At this stage, there might be too options. For multi-class classification, a linear layer with Softmax outputs typically projects the sequence representation down to the number of classes. In the case of BERT (Devlin et al., 2018), this is a [CLS] token that is appended to the start of the sequence as a prefix. Recent work has also explored the usage of Encoder-Decoder architectures for classification, such as T5 (Raffel et al., 2019). Decoder-only models are typically used for generation and are trained using a language modeling objective (of predicting the next token). Due to the nature of the loss, these models are often superior for open ended generation (Brown et al., 2020). A decoder-only model needs to be causal and a upper triangular mask needs to be applied to prevent tokens from peeping into the future. We refer interested readers to (Raffel et al., 2019) for more detailed descriptions of the various Transformer modes.

### 2.6 Applications

Transformers have a wide range of applications ranging from language to vision, speech and reinforcement learning. It was initially introduced within the context of sequence to sequence machine translation in NLP. Following which, most of the applications of Transformers have been within the context of language - given the concurrent advance of pretrained models such as BERT (Devlin et al., 2018). Many early improvements to this line of efficient transformers is therefore focused on language processing applications (Beltagy et al., 2020; Ainslie et al., 2020). For historical reasons, this survey paper leans slightly towards language. However, it is also worth noting that a substantial amount of papers considered in our survey also considers multimodal applications whereby a sequence processor is required. For example Roy et al. (2020); Choromanski et al. (2020b); Tay et al. (2020b); Child et al. (2019) considers generative modeling task on images or other modalities such as proteins.

## 3 A Survey of Efficient Transformer Models

In this section, we provide a high-level overview of efficient Transformer models. We begin by presenting a characterization of the different models. Table 1 lists the efficient Transformers released to date while Figure 2 presents a graphical overview of several key efficient Transformer models.

Model / Paper
Complexity
Decode
Class

Memory Compressed (Liu et al., 2018)

𝒪​(Nc2)𝒪superscriptsubscript𝑁𝑐2\mathcal{O}(N_{c}^{2})
✓✓\checkmark
FP+M

Image Transformer (Parmar et al., 2018)

𝒪(N.m)\mathcal{O}(N.m)
✓✓\checkmark
FP

Set Transformer (Lee et al., 2019)

𝒪​(k​N)𝒪𝑘𝑁\mathcal{O}(kN)
✗
M

Transformer-XL (Dai et al., 2019)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
RC

Sparse Transformer (Child et al., 2019)

𝒪​(N​N)𝒪𝑁𝑁\mathcal{O}(N\sqrt{N})
✓✓\checkmark
FP

Reformer (Kitaev et al., 2020)

𝒪​(N​log⁡N)𝒪𝑁𝑁\mathcal{O}(N\log N)
✓✓\checkmark
LP

Routing Transformer (Roy et al., 2020)

𝒪(NN)\mathcal{O}(N\sqrt{N)}
✓✓\checkmark
LP

Axial Transformer (Ho et al., 2019)

𝒪​(N​N)𝒪𝑁𝑁\mathcal{O}(N\sqrt{N})
✓✓\checkmark
FP

Compressive Transformer (Rae et al., 2020)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
RC

Sinkhorn Transformer (Tay et al., 2020b)

𝒪​(B2)𝒪superscript𝐵2\mathcal{O}(B^{2})
✓✓\checkmark
LP

Longformer (Beltagy et al., 2020)

𝒪​(n​(k+m))𝒪𝑛𝑘𝑚\mathcal{O}(n(k+m))
✓✓\checkmark
FP+M

ETC (Ainslie et al., 2020)

𝒪​(Ng2+N​Ng)𝒪superscriptsubscript𝑁𝑔2𝑁subscript𝑁𝑔\mathcal{O}(N_{g}^{2}+NN_{g})
✗
FP+M

Synthesizer (Tay et al., 2020a)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
LR+LP

Performer (Choromanski et al., 2020a)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✓✓\checkmark
KR

Funnel Transformer (Dai et al., 2020)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
FP+DS

Linformer (Wang et al., 2020c)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✗
LR

Linear Transformers (Katharopoulos et al., 2020)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✓✓\checkmark
KR

Big Bird (Zaheer et al., 2020)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✗
FP+M

Random Feature Attention (Peng et al., 2021)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✓✓\checkmark
KR

Long Short Transformers (Zhu et al., 2021)

𝒪​(k​N)𝒪𝑘𝑁\mathcal{O}(kN)
✓✓\checkmark
FP + LR

Poolingformer (Zhang et al., 2021)

𝒪​(N)𝒪𝑁\mathcal{O}(N)
✗
FP+M

Nyströmformer (Xiong et al., 2021b)

𝒪​(k​N)𝒪𝑘𝑁\mathcal{O}(kN)
✗
M+DS

Perceiver (Jaegle et al., 2021)

𝒪​(k​N)𝒪𝑘𝑁\mathcal{O}(kN)
✓✓\checkmark
M+DS

Clusterformer (Wang et al., 2020b)

𝒪​(N​log⁡N)𝒪𝑁𝑁\mathcal{O}(N\log N)
✗
LP

Luna (Ma et al., 2021)

𝒪​(k​N)𝒪𝑘𝑁\mathcal{O}(kN)
✓
M

TokenLearner (Ryoo et al., 2021)

𝒪​(k2)𝒪superscript𝑘2\mathcal{O}(k^{2})
✗
DS

Adaptive Sparse Transformer (Correia et al., 2019)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓
Sparse

Product Key Memory (Lample et al., 2019)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

Switch Transformer (Fedus et al., 2021)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

ST-MoE (Zoph et al., 2022)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

GShard (Lepikhin et al., 2020)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

Scaling Transformers (Jaszczur et al., 2021)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

GLaM (Du et al., 2021)

𝒪​(N2)𝒪superscript𝑁2\mathcal{O}(N^{2})
✓✓\checkmark
Sparse

### 3.1 A Taxonomy of Efficient Transformers

This section outlines a general taxonomy of efficient Transformer models, characterized by their core techniques and primary use case. While, the primary goal of most of these models is to improve the memory complexity if the self-attention mechanism, we also include methods that improve the general efficiency of the Transformer architecture.

- •

Fixed Patterns (FP) - The earliest modifications to self-attention simply sparsifies the attention matrix by limiting the field of view to fixed, predefined patterns such as local windows and block patterns of fixed strides.

- –

Blockwise Patterns The simplest example of this technique in practice is the blockwise (or chunking) paradigm which considers blocks of local receptive fields by chunking input sequences into fixed blocks. Examples of models that do this include Blockwise (Qiu et al., 2019) and/or Local Attention (Parmar et al., 2018). Chunking input sequences into blocks reduces the complexity from N2superscript𝑁2N^{2} to B2superscript𝐵2B^{2} (block size) with B<<Nmuch-less-than𝐵𝑁B<<N, significantly reducing the cost. These blockwise or chunking methods serve as a basis for many more complex models.

- –

Strided Patterns Another approach is to consider strided attention patterns, i.e., only attending at fixed intervals. Models such as Sparse Transformer (Child et al., 2019) and/or Longformer (Beltagy et al., 2020) employ strided or “dilated” windows.

- –

Compressed Patterns - Another line of attack here is to use some pooling operator to down-sample the sequence length to be a form of fixed pattern. For instance, Compressed Attention (Liu et al., 2018) uses strided convolution to effectively reduce the sequence length.

- •

Combination of Patterns (CP) - The key idea of combined222We note that this is also often referred to as factorization approaches, e.g., in Child et al. (2019). We decide to refer to this class of models as combination approaches because (1) it is a better fit to what these models are actually doing and (2) to avoid confusion with matrix factorization or low-rank approaches. approaches is to improve coverage by combining two or more distinct access patterns. For example, the Sparse Transformer (Child et al., 2019) combines strided and local attention by assigning half of its heads to each pattern. Similarly, Axial Transformer (Ho et al., 2019) applies a sequence of self-attention computations given a high dimensional tensor as input, each along a single axis of the input tensor. In essence, the combination of patterns reduces memory complexity in the same way that fixed patterns does. The difference, however, is that the aggregation and combinaton of multiple patterns improves the overall coverage of the self-attention mechanism.

- •

Learnable Patterns (LP) - An extension to fixed, pre-determined pattern are learnable ones. Unsurprisingly, models using learnable patterns aim to learn the access pattern in a data-driven fashion. A key characteristic of learning patterns is to determine a notion of token relevance and then assign tokens to buckets or clusters (Vyas et al., 2020; Wang et al., 2020b). Notably, Reformer (Kitaev et al., 2020) introduces a hash-based similarity measure to efficiently cluster tokens into chunks. In a simlar vein, the Routing Transformer (Roy et al., 2020) employs online k𝑘k-means clustering on the tokens. Meanwhile, the Sinkhorn Sorting Network (Tay et al., 2020b) exposes the sparsity in attention weights by learning to to sort blocks of the input sequence. In all these models, the similarity function is trained end-to-end jointly with the rest of the network. The key idea of learnable patterns is still to exploit fixed patterns (chunked patterns). However, this class of methods learns to sort/cluster the input tokens - enabling a more optimal global view of the sequence while maintaining the efficiency benefits of fixed patterns approaches.

- •

Neural Memory - Another prominent method is to leverage a learnable side memory module that can access multiple tokens at once. A common form is global neural333We use the term neural here to refer to a representation-like memory that is often manifested in the model. memory which is able to access the entire sequence. The global tokens act as a form of model memory that learns to gather from input sequence tokens. This was first introduced in Set Transformers (Lee et al., 2019) as the inducing points method. These parameters are often interpreted as “memory” and are used as a form of temporary context for future processing. This can be thought of as a form of parameter attention (Sukhbaatar et al., 2019b). Global memory tokens are also used in ETC (Ainslie et al., 2020) and Longformer (Beltagy et al., 2020). With a limited amount of neural memory (or inducing points), we are able to perform a preliminary pooling-like operation of the input sequence to compress the input sequence - a neat trick to have at one’s disposal when designing efficient self-attention modules.

- •

Low-Rank Methods - Another emerging technique is to improve efficiency by leveraging low-rank approximations of the self-attention matrix. The key idea is to assume low-rank structure in the N×N𝑁𝑁N\times N matrix. The Linformer (Wang et al., 2020c) is a classic example of this technique, as it projects the length dimension of keys and values to a lower-dimensional representation (N→k→𝑁𝑘N\rightarrow k). It is easy to see that the low-rank method ameliorates the memory complexity problem of self-attention because the N×N𝑁𝑁N\times N matrix is now decomposed to N×k𝑁𝑘N\times k.

- •

Kernels - Another recently popular method to improve the efficiency of Transformers is to view the attention mechanism through kernelization. The usage of kernels (Katharopoulos et al., 2020; Choromanski et al., 2020a) enable clever mathematical re-writing of the self-attention mechanism to avoid explicitly computing the N×N𝑁𝑁N\times N matrix. Since kernels are a form of approximation of the attention matrix, they can be also viewed as a type of low-rank approach (Choromanski et al., 2020a). Examples of recent work in this area include Performers, Linear Transformers and Random Feature Attention (RFA, (Peng et al., 2021))

- •

Recurrence - A natural extension to the blockwise method is to connect these blocks via recurrence. Transformer-XL (Dai et al., 2019) proposed a segment-level recurrence mechanism that connects multiple segments and blocks. These models can, in some sense, be viewed as fixed pattern models. However, we decided to create its own category due to its deviation from other block / local approaches.

- •

Downsampling - Another popular method of reducing computation cost is to reduce the resolution of the sequence, hence reducing computation costs by a commensurate factor. Examples of this class of models include Perceiver (Jaegle et al., 2021), Funnel Transformers (Dai et al., 2020), Swin Transformer (Liu et al., 2021b), and Charformer (Tay et al., 2021c) models. Notably, there might also be some form of overlap of this class of models with models that leverage memory tokens as models such as Set Transformer can also be viewed as a form of downsampling, albeit within the attention mechanism. The recent Nyströmformer (Xiong et al., 2021b), on the surface, may seem like a low-rank or kernal-based approach. However, it is actually a downsampling approach where the ‘landmarks‘ are simply strided based pooling - in similar spirit to Set Transformer, Funnel Transformer or Perceiever.

- •

Sparse Models and Conditional Computation - While not targeted specifically at the attention modules, sparse models sparsely activate a subset of the parameters which generally improves the parameter to FLOPs ratio. Examples of this class of model includes Switch Transformers (Fedus et al., 2021), ST-MoE (Zoph et al., 2022), GShard (Lepikhin et al., 2020), Product-Key Memory Layers (Lample et al., 2019). Within the scope of our studied models, sparse models typically operate on an adaptive basis in which the sparsity is typically learned (via mixture-of-experts like mechanism). Within this context, we can also consider sparsification of attention weights to fall under this paradigm. For this reason, we believe there is a close connection to fixed or learned patterns in attention. However, we believe that the emergence of an entire research direction (Roller et al., 2021; Lewis et al., 2021; Lepikhin et al., 2020; Du et al., 2021) based on sparse efficient should warrant a new category of efficient Transformers.

We note that these buckets are a broad characterization of the different efficient Transformer models. In reality, there is no sharp boundary between the buckets as models may be comprised of multiple technical innovations. For example, the k𝑘k-means clustering in Routing Transformer (Roy et al., 2020) can also be interpreted as a form of global model memory approach, since one can view the centroids as parameterized model memory. In Reformer, however, clustering is used to learn the sparsity pattern of the attention weights. Additionally, pooling (Liu et al., 2018) can be also interpreted as a form of model memory mechanism. We also note that the recent xformer models (circa December 2021) have started adopting some form of two-staged attention mechanism. Many times, these attention mechanisms explicitly combine one or more flavours of the above, e.g., local windows and then memory in Poolingformer (Zhang et al., 2021), or Long Short Transformers (Zhu et al., 2021) that utilize low rank attention with fixed windows (e.g., a combination of local attention with Linformer-like inductive bias).

### 3.2 Detailed Walk-through of Efficient Transformer Models

This section delves into the details of several key efficient Transformer models, discussing their pros, cons, and unique talking points. The goal here is not to exhaustively detail all such models, but rather to cover a representative sample of models.

##### Structure of this section

We begin by discussing local and fixed patterns models such as the Memory Compressed Transformer (Liu et al., 2018) and Image Transformer (Parmar et al., 2018). We then discuss the Set Transformers (Lee et al., 2019), an early approach for utilizing global model memory. Following which, we move on to models that utilize combinations of patterns such as Sparse Transformers (Child et al., 2019), CCNet (Huang et al., 2019), and Axial Transformers (Ho et al., 2019). Next, we discuss Longformer (Beltagy et al., 2020) and ETC (Ainslie et al., 2020), as examples of memory-based Sparse Transformer approaches. Our detailed walkthrough then moves on to models that incorporate learnable patterns (LP) such as Routing Transformers (Roy et al., 2020), Reformer (Kitaev et al., 2020) and Sinkhorn Transformers (Tay et al., 2020b). After which, we introduce Linformer (Wang et al., 2020c) and Synthesizers (Tay et al., 2020a), models that can be considered low-rank factorization approaches. We then discuss models based on kernel approaches such as Performer (Choromanski et al., 2020a) and Linear Transformers (Katharopoulos et al., 2020). Following which, we discuss the models that are based on segment-based recurrence such as Transformer-XL (Dai et al., 2019) and Compressive Transformers (Rae et al., 2020). Finally, we discuss the family of Sparse models which primarily leverage Mixture-of-Experts (MoE) type architectures and conditional computation to achieve computational efficiency. The logical flow of this section is aimed to be loosely chronological instead of categorically organized (with the exception of certain buckets like recurrence or sparsity that are more orthogonal approaches). We believe this is pedagogically helpful.

#### 3.2.1 Memory Compressed Transformer

Memory Compressed Transformer (Liu et al., 2018) is one of the early attempts at modifying Transformers to better handle longer sequences. The modification introduced by Memory Compressed Transformers is in two folds: localizing the attention span and using memory compressed attention.

##### Local Attention Span

A straightforward solution for dealing with long sequences in Transformers is to limit the attention span to a local neighborhood. Liu et al. (2018) proposed dividing the input sequence into blocks of similar length so that self-attention can be computed within each block independently. This keeps the cost of attention per block constant, thus the number of activations scales linearly with the input length.

##### Memory-compressed Attention

The idea behind memory compressed attention is to reduce the number of keys and values using a strided convolution, while the queries remain unchanged. This leads to a reduction in the size of the attention matrix as well as the attention computations based on a compression factor that depends on the kernel size and the strides of the convolution. Memory compressed attention lets the model exchange the information globally across the input sequence as opposed to local attention.

##### Computation and Memory Complexity

For a block size of b𝑏b, the computational and memory cost of self-attention in each block is 𝒪​(b2)𝒪superscript𝑏2\mathcal{O}(b^{2}). Given there are n/b𝑛𝑏n/b blocks, the computational and memory cost of local attention is 𝒪(b.n)\mathcal{O}(b.n). For memory-compressed attention, applying a convolution with kernel size and strides of k𝑘k, the computational and memory cost of the attention mechanism reduces to 𝒪​(n⋅n/k)𝒪⋅𝑛𝑛𝑘\mathcal{O}(n\cdot n/k).

#### 3.2.2 Image Transformer

Image Transformer (Parmar et al., 2018), inspired by convolutional neural networks, restricts the receptive field of self-attention to only local neighborhoods. This helps the model scale up to process larger batch sizes while keeping the likelihood loss tractable. Besides the efficiency, adapting the notion of locality can be a desirable inductive bias for processing images. Image Transformer offers the encoder-decoder architecture, where the encoder generates a contextualized representation for every pixel-channel in the inputs and the decoder autoregressively generates one channel per pixel at each time step.

##### Localized Attention Span

Limiting the receptive field to a local neighborhood (Parmar et al., 2018, 2019) addresses the issues with the computational and memory costs of running global self-attention on large inputs, but changing the neighborhood per query position would prohibit packing the computations of the self-attention into two matrix multiplications. To avoid that, Image Transformer proposes partitioning the inputs into “query blocks” and their associated “memory blocks“, where for all queries from a single query block, the model attends to the same memory block.
There are two different schemes for choosing query blocks and their associated memory block neighborhoods: 1-dimensional local attention and 2-dimensional local attention. Here we briefly explain these schemes in the decoder case.

(a) 1-dimensional local attention

(b) 2-dimensional local attention

For the 1-dimensional local attention, the image is flattened in the raster order444Given a 2D image as a grid of pixels, the horizontally left-to-right scanning of pixels, line-by-line, creates a raster order. and partitioned into non-overlapping query blocks Q𝑄Q of length lqsubscript𝑙𝑞l_{q}, and for each query block, a memory block M𝑀M is built from the same pixels in the Q𝑄Q as well as a fixed number of pixels, lmsubscript𝑙𝑚l_{m}, generated before the query pixel. In 2-dimensional local attention, pixels are generated in raster order.
For the 2-dimensional local attention, the image is partitioned into multiple non-overlapping rectangular query blocks of length lq=wq×hqsubscript𝑙𝑞subscript𝑤𝑞subscriptℎ𝑞l_{q}=w_{q}\times h_{q}. The memory block extends the query block to the top, left hmsubscriptℎ𝑚h_{m} and wmsubscript𝑤𝑚w_{m} pixels and to the right wmsubscript𝑤𝑚w_{m} pixels, so lm=(wq×qh)+2×(hm+wm)subscript𝑙𝑚subscript𝑤𝑞subscript𝑞ℎ2subscriptℎ𝑚subscript𝑤𝑚l_{m}=(w_{q}\times q_{h})+2\times(h_{m}+w_{m}).
The query pixel can attend to all other pixels. In the 2-dimensional local attention, pixels in the image are generated one query block after another. Generated blocks are in raster order, as well as generated pixels inside every block.

##### Computational and Memory Complexity

In Image Transformer, the attention matrix has the shape of lq×msubscript𝑙𝑞𝑚l_{q}\times m, where lqsubscript𝑙𝑞l_{q} is the chosen length for the query blocks and M𝑀M is the length of the memory block (which is in fact lq+lmsubscript𝑙𝑞subscript𝑙𝑚l_{q}+l_{m}). Given that memory blocks do not overlap, we have to compute n×lq𝑛subscript𝑙𝑞n\times l_{q} attention matrices. Thus the memory and computational complexity of Image Transformer is 𝒪​(n⋅m)𝒪⋅𝑛𝑚\mathcal{O}(n\cdot m).

##### Restrictions

Image Transformer, and in general restricting the context in the attention mechanism to a local neighborhood, can decrease the cost of memory and computation at the price of losing the global receptive field. This can be an issue where global information is required to solve the task. Also, local-attention has quadratic complexity with respect to the region length, thereby introducing an extra hyper-parameter in the trade-off between performance and computational complexity.

#### 3.2.3 Set Transformer

The Set Transformer (Lee et al., 2019) adapts the Transformer model for set-input problems - that is, problems wherein the input is a set of features and the output is some function of this set (and is thereby invariant to the permutation, or ordering, of the input features). The Set Transformer leverages attention to capture interactions between elements of the input set. Furthermore, it applies the idea of inducing points from the sparse Gaussian process literature to reduce the complexity of attention from quadratic to linear in the size of the input set.

Problems involving sets of objects often have a permutation invariance property: the target value for the set is the same regardless of the order of the objects in the set. Zaheer et al. (2017) proved that all permutation-invariant functions can be represented by the following functional form:

network​({x1,…,xN})=ρ​(pool​({ϕ​(x1),…,ϕ​(xN)})),networksubscript𝑥1…subscript𝑥𝑁𝜌poolitalic-ϕsubscript𝑥1…italic-ϕsubscript𝑥𝑁\displaystyle\text{network}\left(\{x_{1},\dots,x_{N}\}\right)=\rho\left(\text{pool}\left(\{\phi(x_{1}),\dots,\phi(x_{N})\}\right)\right),

where the pooling function pool is a simple summation and ϕitalic-ϕ\phi and ρ𝜌\rho are continuous functions. This form can be interpreted as the composition of an encoder ϕitalic-ϕ\phi and decoder ρ​(pool​(⋅))𝜌pool⋅\rho\left(\text{pool}(\cdot)\right).
While this form is a universal approximator in the space of permutation-invariant functions, it is unclear how well such models fit tasks in practice. The Set Transformer proposes a solution that can be viewed as an encoder and pooled decoder, but where, unlike the form given above, the encoder and decoder can attend to input elements individually and the pooling function is parameterized.

##### Attention Blocks

The model introduces the following constructs: “Multihead Attention Block” (MAB), “Set Attention Block” (SAB), “Induced Set Attention Block” (ISAB), and “Pooling by Multihead Attention” (PMA). They are defined as follows.

MAB​(𝐗,𝐘)MAB𝐗𝐘\displaystyle\mathbf{\textbf{MAB}(X,Y)}
:=LayerNorm​(H+rFF​(H)),assignabsentLayerNorm𝐻rFF𝐻\displaystyle:=\text{LayerNorm}\left(H+\text{rFF}(H)\right),

H𝐻\displaystyle H
:=LayerNorm​(X+MultiheadAttention​(X,Y)),assignabsentLayerNorm𝑋MultiheadAttention𝑋𝑌\displaystyle:=\text{LayerNorm}\left(X+\text{MultiheadAttention}(X,Y)\right),

SAB​(𝐗)SAB𝐗\displaystyle\mathbf{\textbf{SAB}(X)}
:=MAB​(X,X),assignabsentMAB𝑋𝑋\displaystyle:=\text{MAB}(X,X),

ISAB𝐦​(𝐗)subscriptISAB𝐦𝐗\displaystyle\mathbf{\textbf{ISAB}_{m}(X)}
:=MAB​(X,MAB​(Im,X)).assignabsentMAB𝑋MABsubscript𝐼𝑚𝑋\displaystyle:=\text{MAB}\left(X,\text{MAB}(I_{m},X)\right).

PMA𝐤​(𝐗)subscriptPMA𝐤𝐗\displaystyle\mathbf{\textbf{PMA}_{k}(X)}
:=MAB​(Sk,rFF​(X)).assignabsentMABsubscript𝑆𝑘rFF𝑋\displaystyle:=\text{MAB}\left(S_{k},\text{rFF}(X)\right).

Here, X∈ℝN×d𝑋superscriptℝ𝑁𝑑X\in\mathbb{R}^{N\times d} represents N𝑁N d𝑑d-dimensional input/outputs stacked row-wise and rFF is a parameterized feed-forward layer that operates on each row of its input matrix separately. Im∈ℝm×dsubscript𝐼𝑚superscriptℝ𝑚𝑑I_{m}\in\mathbb{R}^{m\times d} represents m𝑚m trainable d𝑑d-dimensional “inducing points” while Sk∈ℝk×dsubscript𝑆𝑘superscriptℝ𝑘𝑑S_{k}\in\mathbb{R}^{k\times d} represent k𝑘k trainable d𝑑d-dimensional “seed vectors” (with k𝑘k set to 111 except when k>1𝑘1k>1 correlated outputs are needed).
The Set Transformer’s encoder is just N𝑁N layers of either SAB or ISAB (with N𝑁N often set to 222 in practice) while its decoder is given by:

Decoder​(𝐗):=rFF​(SAB​(PMAk​(X))).assignDecoder𝐗rFFSABsubscriptPMA𝑘𝑋\displaystyle\mathbf{\textbf{Decoder}(X)}:=\text{rFF}\left(\text{SAB}\left(\text{PMA}_{k}(X)\right)\right).

It is straightforward to see that both ISAB and SAB are permutation equivariant - in other words, if the input is permuted in some way then the corresponding output of the block is permuted in exactly the same way. Meanwhile, the pooling layer PMA is permutation invariant. Since functional composition, i.e. layering, preserves these properties, the Set Transformer encoder-decoder combination is permutation invariant.

##### Efficiency

We can understand the m𝑚m inducing points Imsubscript𝐼𝑚I_{m} learned in each ISAB layer as a form of static model memory. In addition to reducing the 𝒪​(N​n2)𝒪𝑁superscript𝑛2\mathcal{O}(Nn^{2}) complexity of the self-attending SAB layer to 𝒪​(N​m​n)𝒪𝑁𝑚𝑛\mathcal{O}(Nmn), a reduction particularly valuable when the input set is large, the inducing points effectively encode some global structure that helps explain its inputs. For example, in the problem of amortized clustering, where one attempts to learn to map an input set of points to the centers of clusters of points inside the set, the inducing points learned could be appropriately distributed so that the encoder can effectively compare query elements with each other implicitly via their proximity to the inducing points.

The trainable k𝑘k seeds Sksubscript𝑆𝑘S_{k} used in the pooling layer PMAksubscriptPMA𝑘\text{PMA}_{k} can be viewed as static model memory in a similar light, reducing the memory and runtime complexity of the architecture.

(a) Transformer

(b) Sparse Transformer

#### 3.2.4 Sparse Transformer

The Sparse Transformer (Child et al., 2019) presents a simple initial attempt to reduce the quadratic complexity of the standard self-attention mechanism. The key idea is to reduce the dense attention matrix to a sparse version by only computing attention on a sparse number of qi,kjsubscript𝑞𝑖subscript𝑘𝑗q_{i},k_{j} pairs. Sparse Transformer employs fixed attention patterns which are defined by strides and local neighborhoods. Computation is factorized, wherein local and stride patterns are split amongst the heads.

##### Local Attention Heads

Half of the heads in the Sparse Transformer are dedicated to local attention.

A^i​j={Qi(K)j⊤),if ​⌊j/N⌋=⌊i/N⌋0otherwise\displaystyle\hat{A}_{ij}=\begin{cases}Q_{i}(K)_{j}^{\top}),&\text{if }\lfloor{{j}/{N}}\rfloor=\lfloor{i/{N}}\rfloor\\
0&\text{otherwise}\end{cases}

where Ai​jsubscript𝐴𝑖𝑗A_{ij} is the attention weight of qi,kjsubscript𝑞𝑖subscript𝑘𝑗q_{i},k_{j} and ⌊⌋\lfloor\>\rfloor denote the floor operation. In this case, we only compute the attention if ⌊j/N⌋=⌊i/N⌋𝑗𝑁𝑖𝑁\lfloor{{j}/{N}}\rfloor=\lfloor{i/{N}}\rfloor (within the same block).

##### Strided Attention Heads

The other half of the heads are dedicated to fixed strided patterns. Concretely,

A^i​j={Qi(K)j⊤),if ​(i−j)modN=00otherwise\displaystyle\hat{A}_{ij}=\begin{cases}Q_{i}(K)_{j}^{\top}),&\text{if }(i-j)\mod N=0\\
0&\text{otherwise}\end{cases}

The final result of the factorized sparse attention is visualized in Figure 4. We refer interested to (Yun et al., 2020) for some additional theoretical analysis about the expressiveness of the Sparse attention mechanism.

##### Parameter and Memory Complexity

The modification in the self-attention mechanism does not alter the parameter costs of the model since the model still retains the Q,K,V𝑄𝐾𝑉Q,K,V transforms from the original Transformer model. The memory complexity of the attention layer is reduced from 𝒪​(n2)𝒪superscript𝑛2\mathcal{O}(n^{2}) to 𝒪​(n​log⁡n)𝒪𝑛𝑛\mathcal{O}(n\log n) .

##### Restrictions

The Sparse Transformer implementation requires custom GPU kernels to implement a specific block-sparse variant of matrix-matrix-multiplication and cannot be easily implemented on other hardware such as TPUs.

#### 3.2.5 Axial Transformer

Axial Transformer (Ho et al., 2019; Weissenborn et al., 2019) uses factorization in a simple yet effective setup for the self-attention mechanism to process large inputs that are organized as multidimensional tensors. Instead of applying attention to the flattened version of the input, Axial Transformer simply applies multiple attentions, each along a single axis of the input tensor. Each attention, in fact, mixes information along a particular axis, while keeping information along other axes independent. Since the length of any single axis is typically much smaller than the total number of elements, Axial Transformer significantly saves computation and memory.

Axial Transformer offers an encoder-decoder architecture. For the decoding, to be able to implement the causal mask, Axial Transformer combines axial attentions with shift operations. For instance, for a model on 2-dimensional tensors, pixels are generated in raster order and to do that, first, the model encodes all pixels through an unmasked row and unmasked column attention. Then, for each row, the model applies an unmasked row and masked column attention to integrate the previously sampled rows. Finally, the model shifts the encoded representation up to make sure the conditioning information satisfies causality, and runs a masked row-attention to sample a new row in the image.

An advantage of Axial Transformer over similar methods like Sparse Transformer is that while it provides the global receptive field, it is straightforward to implement and does not require a custom kernel for an efficient implementation.

##### Computational and Memory Complexity

In terms of memory and computational complexity, on a square image of size N𝑁N, Axial Transformer performs the attention computation in 𝒪​(n​n)𝒪𝑛𝑛\mathcal{O}(n\sqrt{n}), which saves 𝒪​(n)𝒪𝑛\mathcal{O}(\sqrt{n}) over normal self-attention. For instance, with on square image with N𝑁N pixels, organized in a b×b𝑏𝑏b\times b grid, Axial Transformer runs b𝑏b attention sequences of length b𝑏b, which is of complexity 𝒪(b.b2)\mathcal{O}(b.b^{2}). In a more general case, for a d𝑑d-dimensional tensor of shape N=N1/d×…×N1/d𝑁superscript𝑁1𝑑…superscript𝑁1𝑑N=N^{1/d}\times\ldots\times N^{1/d}, Axial Transformer saves a 𝒪​(N(d−1)/d)𝒪superscript𝑁𝑑1𝑑\mathcal{O}(N^{(d-1)/d}) factor of resources over standard self-attention.

#### 3.2.6 Longformer

Longformer (Beltagy et al., 2020) is a variant of Sparse Transformer.
Its key distinction compared to Sparse Transformer is “Dilated Sliding Windows”, which can enable better long-range coverage without sacrificing sparsity. This is achieved by increasing the receptive fields by having gaps in the attention patterns. The Longformer also gradually increases the receptive field as the model goes deeper, dedicating lower levels for modeling local patterns and upper levels for modeling global patterns.

##### Global Attention

For classification tasks, Longformer adopts global memory tokens that have access to all input sequences.

##### Parameter and Memory Complexity

The complexity of the model is reduced from 𝒪​(n2)𝒪superscript𝑛2\mathcal{O}(n^{2}) to 𝒪​(n​k)𝒪𝑛𝑘\mathcal{O}(nk) where k𝑘k is the size of the window. When using global attention, the Longformer creates another set of query-key-value
projections for this global attention, doubling the cost of the parameters at the attention layer.

#### 3.2.7 Extended Transformer Construction (ETC)

The ETC model (Ainslie et al., 2020) is another variation in the Sparse Transformer family. It introduces a new global-local attention mechanism. There are four components to this new attention mechanism, namely (1) global-to-global (g2g), global-to-local (g2l), local-to-global (l2g) and local-to-local (l2l). Aside from the original input to the model, ETC introduces ngsubscript𝑛𝑔n_{g} auxiliary tokens as a prefix to the original input sequence. These tokens are regarded as global tokens and take part in global-to-∗* and ∗*-to-global attention. The local-to-local component acts as the local attention with a fixed radius of k𝑘k. Overall, ETC is quite similar to Longformer in the way it introduces global auxiliary tokens. These tokens are trainable parameters and can be interpreted as a form of model memory that pools across the sequence to collect global sequence information.

##### Memory and Parameter Complexity

The memory complexity of the ETC model is 𝒪​(ng2+ng​N)𝒪superscriptsubscript𝑛𝑔2subscript𝑛𝑔𝑁\mathcal{O}(n_{g}^{2}+n_{g}N), where ngsubscript𝑛𝑔n_{g} is the number of global tokens and N𝑁N is the input sequence length.

##### Restrictions

Intuitively, it is easy to observe that ETC cannot be used for auto-regressive decoding. This is because we are not able to compute causal masks because of the global attention.

#### 3.2.8 BigBird

The BigBird model (Zaheer et al., 2020) is another Transformer for modeling longer sequences and is primarily built on top of ETC (Ainslie et al., 2020). The Big Bird model is comprised of several key components, namely (1) global tokens, (2) random attention (queries attend to random keys), and (3) fixed patterns (local sliding windows).

##### Global Attention

Fundamentally, the idea of using global model memory can be traced all the way back to Longformer/ETC and Set Transformer model. Notably, the global model memory in Big Bird is extended to contain tokens within the sequence, instead of simply parameterized model memory. The authors call this the ‘internal transformer construction (ITC)’ in which a subset of indices is selected as global tokens. This can be interpreted as a model-memory-based approach.

##### Sliding Window Attention

The window-ed attention was first proposed in early local-based attention models (Image Transformer, Compressed Attention and/or Sparse Transformer). In BigBird, each query attends to w/2𝑤2w/2 tokens to the left and w/2𝑤2w/2 tokens to the right. This corresponds to a fixed pattern (FP) approach.

##### Random Attention

Finally, each query attends to r𝑟r random keys. This pattern is fixed.

##### Memory and Parameter Complexity

The memory complexity of the self-attention is linear, i.e., 𝒪​(n)𝒪𝑛\mathcal{O}(n). The BigBird model does not introduce new parameters beyond the Transformer model.

##### Restrictions

Similar to ETC, the BigBird model cannot be used to autoregressively decode. Hence, qualifying it as an encoder-only model.

#### 3.2.9 Routing Transformer

The Routing Transformer (Roy et al., 2020) is a content-based sparse attention mechanism. It proposes a clustering-based attention mechanism that learns the attention sparsity in a data driven fashion. The first step is to project Q𝑄Q and K𝐾K into a routing matrix R𝑅R of dimensions n×d𝑛𝑑n\times d.

R=Q​WR+K​WR𝑅𝑄subscript𝑊𝑅𝐾subscript𝑊𝑅\displaystyle R=QW_{R}+KW_{R}

(1)

where WRsubscript𝑊𝑅W_{R} is a d×d𝑑𝑑d\times d orthonormal projection matrix.

##### k𝑘k-means Clustering

The R𝑅R matrix undergoes k𝑘k-means clustering with a series of parameterized cluster centroids u1,u2​⋯​cksubscript𝑢1subscript𝑢2⋯subscript𝑐𝑘u_{1},u_{2}\cdots c_{k}. The k𝑘k-means in Routing Transformer is trained in an online fashion. To ensure a similar number of tokens in each cluster, the model initializes n𝑛\sqrt{n} clusters, computes each token’s distance against the cluster centroid, and takes an equal top-k𝑘k for each centroid. Since the cluster centroids are trainable parameters, this is also reminiscent of the all-attention layer proposed by (Sukhbaatar et al., 2019b).

##### Routing Strategy

The routing strategy is then defined as:

Xi′=∑j∈Ci,j≤iAi​j​Vjsubscriptsuperscript𝑋′𝑖subscriptformulae-sequence𝑗subscript𝐶𝑖𝑗𝑖subscript𝐴𝑖𝑗subscript𝑉𝑗\displaystyle X^{\prime}_{i}=\sum_{j\in C_{i},j\leq i}A_{ij}V_{j}

(2)

where Cisubscript𝐶𝑖C_{i} is the cluster that vector Risubscript𝑅𝑖R_{i} is assigned to. In other words, the token at i𝑖i only attends to tokens in the same cluster.

##### Memory and Parameter Complexity

The Routing Transformer introduces additional parameters in the clustering mechanism, namely k×d𝑘𝑑k\times d centroid vectors and a Wrsubscript𝑊𝑟W_{r} projection matrix. The memory complexity is 𝒪​(n1.5)𝒪superscript𝑛1.5\mathcal{O}(n^{1.5}).

#### 3.2.10 Reformer

Reformer (Kitaev et al., 2020) is another efficient attention model based on locality sensitive hashing (LSH). Reformer also introduces reversible Transformer layers, which contribute to further reducing its memory footprint.

##### LSH Attention

The LSH attention introduces parameter-sharing between query and keys. It hashes the query-keys into buckets using a random-projection based hashing function. The key idea is that nearby vectors should obtain a similar hash while distant vectors should not, hence being termed as ‘locality sensitive’. To perform hashing, a random matrix R∈ℝk×b/2𝑅superscriptℝ𝑘𝑏2R\in\mathbb{R}^{k\times b/2} is first introduced. Next, The hashing function is defined as:

h​(x)=arg max​([x​R;−x​R])ℎ𝑥arg max𝑥𝑅𝑥𝑅\displaystyle h(x)=\text{arg max}([xR;-xR])

(3)

where [;][;] is the concatenation of two vectors. For all queries, attention is computed if and only if the query and key hashes match, i.e., h​(qi)=h​(kj)ℎsubscript𝑞𝑖ℎsubscript𝑘𝑗h(q_{i})=h(k_{j}). In other words, attention is computed amongst query and keys if they fall in the same hash bucket. In order to maintain causal masking, Reformer assigns and maintains a position index for every query and key. It is therefore able to compare if each query key comparison is auto-regressively valid.

##### Memory Efficiency with LSH Attention

The key idea behind LSH attention is to classify tokens into buckets and then process them bucket by bucket in a chunked fashion. To this end, queries are first sorted by bucket number and then by sequence order within the same bucket. During computation, tokens only attend to the same bucket in its own chunk and previous chunk. The chunking and sorted bucketing techniques help to improve the overall efficiency of the Reformer model.

##### Parameter and Memory Complexity

The memory complexity of Reformer is 𝒪​(n​log⁡n)𝒪𝑛𝑛\mathcal{O}(n\log n). In terms of parameter costs, Reformer shares queries and keys, which reduces the cost of the QKV transforms by a third. The random projections are not trainable parameters and hence do not incur parameter costs. Overall, Reformer has fewer parameters than vanilla Transformers. The reversible layers in Reformer also reduce the memory consumption during training by enabling activations to be reconstructed from the next layer’s. This reduces memory cost since this eliminates the need to store activations for all layers during backpropagation.

#### 3.2.11 Sinkhorn Transformers

This section introduces the Sparse Sinkhorn Transformer (Tay et al., 2020b). The Sinkhorn Transformer belongs to the family of learned patterns. This model is a chunked/blocked model that learns sparse patterns by re-sorting the input key and values in a block-wise fashion and then applying local block-based attention.

Ai​j={(Qi​ψS​(K)j⊤),if​⌊j/N⌋=⌊i/N⌋0otherwisesubscript𝐴𝑖𝑗casessubscript𝑄𝑖subscript𝜓𝑆superscriptsubscript𝐾𝑗topif𝑗𝑁𝑖𝑁0otherwise\displaystyle A_{ij}=\begin{cases}(Q_{i}\psi_{S}(K)_{j}^{\top}),&\text{if}\lfloor{{j}/{N}}\rfloor=\lfloor{i/{N}}\rfloor\\
0&\text{otherwise}\end{cases}

where ψSsubscript𝜓𝑆\psi_{S} applies a sorting operator on the sequence length dimension.

##### Sorting Network

The sorting operator is parameterized by a meta sorting network. Let X𝑋X be the input sequence of dimension N×d𝑁𝑑N\times d.

ψS​(X)=ϕS​(FS​(BlockSum​(X)))​BlockShape​(X)subscript𝜓𝑆𝑋subscriptitalic-ϕ𝑆subscript𝐹𝑆BlockSum𝑋BlockShape𝑋\psi_{S}(X)=\phi_{S}(F_{S}(\textsc{BlockSum}(X)))\>\textsc{BlockShape}(X)

(4)

where FS(.)F_{S}(.) is a parameterized function such as a two layer feed-forward network with ReLU activation. The output of FS(.)F_{S}(.) is a tensor of nB×nBsubscript𝑛𝐵subscript𝑛𝐵n_{B}\times n_{B}. The BlockSum function learns the sum embeddings of local blocks. The BlockShape function reshapes the input tensor into ℝN×d→ℝnB×b×d→superscriptℝ𝑁𝑑superscriptℝsubscript𝑛𝐵𝑏𝑑\mathbb{R}^{N\times d}\rightarrow\mathbb{R}^{n_{B}\times b\times d}. Here, we note that N=nB×b𝑁subscript𝑛𝐵𝑏N=n_{B}\times b, where b𝑏b is the size of the block and nBsubscript𝑛𝐵n_{B} is the number of total blocks.

##### Sinkhorn Sorting

ϕitalic-ϕ\phi is the Sinkhorn balancing operator (Sinkhorn, 1964; Adams and Zemel, 2011) which converts the nB×nBsubscript𝑛𝐵subscript𝑛𝐵n_{B}\times n_{B} matrix into a soft permutation matrix. Specifically, a series of row- and column-wise normalizations are applied on the matrix output of FS​BlockSum​(X)subscript𝐹𝑆BlockSum𝑋F_{S}\text{BlockSum}(X). For the sake of brevity, we do not delve into details of this operation. Further details can be found at Adams and Zemel (2011); Tay et al. (2020b).

##### Parameter and Memory Complexity

The memory complexity of the Sinkhorn Transformer is 𝒪​(b2)𝒪superscript𝑏2\mathcal{O}(b^{2}) where b𝑏b is the block size and b=NNb𝑏𝑁subscript𝑁𝑏b=\frac{N}{N_{b}}. Additional parameter costs are incurred from the meta sorting network FS(.)F_{S}(.). The number of additional parameters is therefore 2​d22superscript𝑑22d^{2} when a two layer ReLU network is used as the sorting network.

#### 3.2.12 Linformer

Linformer (Wang et al., 2020c) is an efficient Transformer based on the idea of low-rank self-attention.

##### Low-Rank Projections on Length Dimensions

Linformer projects the N×d𝑁𝑑N\times d dimensional keys and values to k×d𝑘𝑑k\times d dimensions using additional projection layers. Note that this is a reduction on the length dimension instead of the key and value dimensions. This can
Given the newly projected keys (K′superscript𝐾′K^{\prime}) and values (V′superscript𝑉′V^{\prime}), the Q​K′𝑄superscript𝐾′QK^{\prime} matrix is now (N×k)𝑁𝑘(N\times k) dimensions instead of (N×N)𝑁𝑁(N\times N). The attention matrix Softmax​(Q​K′)Softmax𝑄superscript𝐾′\text{Softmax}(QK^{\prime}) multiplies with V′∈ℝk×dsuperscript𝑉′superscriptℝ𝑘𝑑V^{\prime}\in\mathbb{R}^{k\times d} to result in an output tensor of dimensions N×d𝑁𝑑N\times d. To some extent, Linformer is reminiscent of depth-wise convolutions (Kaiser et al., 2017). A projection on the length dimension causes mixing of sequence information (dimension-wise) in a single transformation. Hence, it is non-trivial to maintain causal masking and/or prevent mixing of past and future information when computing attention scores. The formulation of Linformer (for each attention head) can be expressed as:

S​o​f​t​m​a​x​(1dk​X​WiQ​(Ei​X​WiK))⋅Fi​X​WiV⋅𝑆𝑜𝑓𝑡𝑚𝑎𝑥1subscript𝑑𝑘𝑋subscriptsuperscript𝑊𝑄𝑖subscript𝐸𝑖𝑋superscriptsubscript𝑊𝑖𝐾subscript𝐹𝑖𝑋superscriptsubscript𝑊𝑖𝑉\displaystyle Softmax(\frac{1}{\sqrt{d_{k}}}XW^{Q}_{i}(E_{i}XW_{i}^{K}))\cdot F_{i}XW_{i}^{V}

(5)

where WQ,K,Vsuperscript𝑊𝑄𝐾𝑉W^{Q,K,V} are the default linear transformation of X𝑋X into queries (as per vanilla Transformer) and Ei,Fisubscript𝐸𝑖subscript𝐹𝑖E_{i},F_{i} are additional k×N𝑘𝑁k\times N projection of the key and values into k×d𝑘𝑑k\times d tensors.

##### Parameter and Memory Complexity

The memory complexity of Linformer is 𝒪​(n)𝒪𝑛\mathcal{O}(n). There is only a minimal parameter costs of the Linformer due to the extra N×k𝑁𝑘N\times k length projections. If k𝑘k is sufficiently small, there is negligible parameter costs incurred.

#### 3.2.13 Performer

The Performer (Choromanski et al., 2020a, b) model is characterized by its Generalized Attention mechanism and its usage of random Kernels.

##### Generalized Attention

The generalized attention entangles Qi,Kjsubscript𝑄𝑖subscript𝐾𝑗Q_{i},K_{j} with a kernel function K𝐾K. The attention matrix in Performer is computed via:

A=[g​(Qi⊤)​K​(Qi⊤​Kj⊤)​h​(Kj⊤)]𝐴delimited-[]𝑔superscriptsubscript𝑄𝑖top𝐾superscriptsubscript𝑄𝑖topsuperscriptsubscript𝐾𝑗topℎsuperscriptsubscript𝐾𝑗top\displaystyle A=[g(Q_{i}^{\top})K(Q_{i}^{\top}K_{j}^{\top})h(K_{j}^{\top})]

(6)

where K(.)K(.) is a kernel function that maps d×d𝑑𝑑d\times d to a scalar value ℝℝ\mathbb{R} and g,h𝑔ℎg,h are functions that map d𝑑d to a scalar value ℝℝ\mathbb{R}.

##### Fast Attention via Orthogonal Random Features (FAVOR)

The above computation is still quadratic in complexity. Hence, the Performer leverages approximation tricks to avoid storing and computing the N×N𝑁𝑁N\times N attention matrix. It leverages orthogonal random features (ORF) for doing so. The final attention output Y𝑌Y of the Performer is described as follows:

Y=D^−1​(Q′​((K′)⊤​V))𝑌superscript^𝐷1superscript𝑄′superscriptsuperscript𝐾′top𝑉\displaystyle Y=\hat{D}^{-1}(Q^{\prime}((K^{\prime})^{\top}V))

(7)

where D^=diag​(Q′​((K′)⊤​1N))^𝐷diagsuperscript𝑄′superscriptsuperscript𝐾′topsubscript1𝑁\hat{D}=\text{diag}(Q^{\prime}((K^{\prime})^{\top}1_{N})), Q′=DQ​ϕ​(Q⊤)⊤superscript𝑄′subscript𝐷𝑄italic-ϕsuperscriptsuperscript𝑄toptopQ^{\prime}=D_{Q}\phi(Q^{\top})^{\top}, and K′=DK​ϕ​(K⊤)⊤superscript𝐾′subscript𝐷𝐾italic-ϕsuperscriptsuperscript𝐾toptopK^{\prime}=D_{K}\phi(K^{\top})^{\top}. Note that DQ=g​(Qi⊤),DK=h​(Ki⊤)formulae-sequencesubscript𝐷𝑄𝑔superscriptsubscript𝑄𝑖topsubscript𝐷𝐾ℎsuperscriptsubscript𝐾𝑖topD_{Q}=g(Q_{i}^{\top}),D_{K}=h(K_{i}^{\top}). The function ϕ​(x)italic-ϕ𝑥\phi(x) is defined as:

ϕ​(X)=cM​f​(W​x+b)⊤italic-ϕ𝑋𝑐𝑀𝑓superscript𝑊𝑥𝑏top\displaystyle\phi(X)=\frac{c}{\sqrt{M}}f(Wx+b)^{\top}

(8)

where c>0𝑐0c>0 is a constant, W∈ℝM×d𝑊superscriptℝ𝑀𝑑W\in\mathbb{R}^{M\times d} is a random feature matrix and M𝑀M is the dimensionality of this matrix that controls the number of random features. We are able to see that we do not explicitly compute A=Q​K⊤𝐴𝑄superscript𝐾topA=QK^{\top} and hence avoid paying the N2superscript𝑁2N^{2} cost. For rigorous theoretical analysis and further details, we refer interested readers to (Choromanski et al., 2020a).

##### Parameter/Memory Complexity and Compute Costs

The complexity of the bi-directional FAVOR algorithm is 𝒪​(M​d+N​d+M​N)𝒪𝑀𝑑𝑁𝑑𝑀𝑁\mathcal{O}(Md+Nd+MN) where M𝑀M is the dimensionality of the random features. It is worth noting that the unidirectional variations cannot be causally masked in an efficient linear-time fashion. As such, during training, running unidirectional (causal) implementation of kernel-based attention on an autoregressive task can be several times slower than vanilla Transformer during parallelized training due to the need to do a left to right pass (i.e., scan operation) in similar spirit to Recurrent neural networks. Since many autoregressive tasks trained via parallelization and teacher forcing, this makes training Performer on a generative task prohibitively slow. In order for KV to be causally masked efficiently, one would have to manifest the d×d𝑑𝑑d\times d KV matrix at every time step - recovering a quadratic complexity model. We feel this is one of the intricate points that highlight how efficient memory complexity might not equate a faster or more efficient model in practice. We highlight that this only happens during autoregressive training. The inference-time for incremental decoding, however, would benefit from a speed up.

#### 3.2.14 Linear Transformer

The Linear Transformer (Katharopoulos et al., 2020) improves the complexity of self-attention from quadratic to linear by using a kernel-based formulation of self-attention and the associative property of matrix products. Furthermore, it reduces attention with causal masking (which is used in auto-regressive decoding) to a linear-time, constant memory recurrent neural network (RNN). The model has been shown to improve inference speeds up to three orders of magnitude without much loss in predictive performance. Linear Transformers are similar to Performers with the exception of the kernel function and therefore also suffer from the same drawbacks (unable to be parallelized across the time dimension during training in an autoregressive teacher forced setting).

The method rests on the simple but powerful observation that the accumulated value Vi′superscriptsubscript𝑉𝑖′V_{i}^{\prime} for the query Qisubscript𝑄𝑖Q_{i} in position i𝑖i can be written as:

Vi′superscriptsubscript𝑉𝑖′\displaystyle V_{i}^{\prime}
=∑j=1psim​(Qi,Kj)​Vj∑j=1psim​(Qi,Kj).absentsuperscriptsubscript𝑗1𝑝simsubscript𝑄𝑖subscript𝐾𝑗subscript𝑉𝑗superscriptsubscript𝑗1𝑝simsubscript𝑄𝑖subscript𝐾𝑗\displaystyle=\frac{\sum_{j=1}^{p}\text{sim}(Q_{i},K_{j})V_{j}}{\sum_{j=1}^{p}\text{sim}(Q_{i},K_{j})}.

Here, p=N𝑝𝑁p=N in full, unmasked attention and p=i𝑝𝑖p=i in the case of causal masking. Now, in usual softmax attention, sim​(q,k)=exp⁡(qT​kd)sim𝑞𝑘superscript𝑞𝑇𝑘𝑑\text{sim}(q,k)=\exp\left(\frac{q^{T}k}{\sqrt{d}}\right). Linear Transformer, however, expresses the similarity as a kernel function. That is, sim​(q,k):=ϕ​(q)T​ϕ​(k)assignsim𝑞𝑘italic-ϕsuperscript𝑞𝑇italic-ϕ𝑘\text{sim}(q,k):=\phi(q)^{T}\phi(k), where ϕitalic-ϕ\phi is a, possibly high-dimensional, feature map. With this choice,
we can rewrite Vi′superscriptsubscript𝑉𝑖′V_{i}^{\prime} as:

Vi′superscriptsubscript𝑉𝑖′\displaystyle V_{i}^{\prime}
=ϕ​(Qi)T​Spϕ​(Qi)T​Zp,absentitalic-ϕsuperscriptsubscript𝑄𝑖𝑇subscript𝑆𝑝italic-ϕsuperscriptsubscript𝑄𝑖𝑇subscript𝑍𝑝\displaystyle=\frac{\phi(Q_{i})^{T}S_{p}}{\phi(Q_{i})^{T}Z_{p}},

Spsubscript𝑆𝑝\displaystyle S_{p}
:=∑j=1pϕ​(Kj)​VjT,assignabsentsuperscriptsubscript𝑗1𝑝italic-ϕsubscript𝐾𝑗superscriptsubscript𝑉𝑗𝑇\displaystyle:=\sum_{j=1}^{p}\phi(K_{j})V_{j}^{T},

Zpsubscript𝑍𝑝\displaystyle Z_{p}
:=∑j=1pϕ​(Kj).assignabsentsuperscriptsubscript𝑗1𝑝italic-ϕsubscript𝐾𝑗\displaystyle:=\sum_{j=1}^{p}\phi(K_{j}).

For unmasked attention, since p=N𝑝𝑁p=N we only need to compute SNsubscript𝑆𝑁S_{N} and ZNsubscript𝑍𝑁Z_{N} once and we reuse them for the computation at every position 0≤i≤N0𝑖𝑁0\leq i\leq N. For causal attention, the Sisubscript𝑆𝑖S_{i}’s and Zisubscript𝑍𝑖Z_{i}’s can be viewed as states of an RNN that are updated by the following recurrence relations:

Sisubscript𝑆𝑖\displaystyle S_{i}
=Si−1+ϕ​(Ki)​ViT,absentsubscript𝑆𝑖1italic-ϕsubscript𝐾𝑖superscriptsubscript𝑉𝑖𝑇\displaystyle=S_{i-1}+\phi(K_{i})V_{i}^{T},

Zisubscript𝑍𝑖\displaystyle Z_{i}
=Zi−1+ϕ​(Ki)absentsubscript𝑍𝑖1italic-ϕsubscript𝐾𝑖\displaystyle=Z_{i-1}+\phi(K_{i})

with initial condition S0=Z0=0subscript𝑆0subscript𝑍00S_{0}=Z_{0}=0.
If the dimension of the key, query, and values are all d𝑑d and the cost to compute ϕitalic-ϕ\phi is 𝒪​(c)𝒪𝑐\mathcal{O}(c), then the overall run-time complexity of Linear Transformer is 𝒪​(N​c​d)𝒪𝑁𝑐𝑑\mathcal{O}{(Ncd)}. The authors choose

ϕ​(x)=elu​(x)+1,italic-ϕ𝑥elu𝑥1\displaystyle\phi(x)=\text{elu}(x)+1,

where elu​(⋅)elu⋅\text{elu}(\cdot) denotes the exponential linear unit (Clevert et al., 2015). With this choice of feature map, c=d𝑐𝑑c=d and the end-to-end complexity of the model is 𝒪​(N​d2)𝒪𝑁superscript𝑑2\mathcal{O}(Nd^{2}).

#### 3.2.15 Synthesizers

Synthesizer models (Tay et al., 2020a) are an attempt to study and investigate the true importance of conditioning within the self-attention mechanism and are also the first attempts at unconditional token-mixing. In Tay et al. (2020a), the authors study a synthetic self-attention module in which attention weights are approximated instead of being computed by pairwise dot products. Synthesizers are only implicitly related to efficient Transformers and can be considered more as a MLP-Mixer (Tolstikhin et al., 2021). However, the factorized variants can be considered a low-rank efficient Transformer model.

##### Dense Synthesizers

In the Dense Synthesizer, each token xisubscript𝑥𝑖x_{i} is projected to a vector of length N𝑁N using a two-layered non-linear feed-forward network. The computation of the attention matrix A𝐴A is described as:

A=W2​(σR​(W1​(X)+b))+b𝐴subscript𝑊2subscript𝜎𝑅subscript𝑊1𝑋𝑏𝑏\displaystyle A=W_{2}(\sigma_{R}(W_{1}(X)+b))+b

(9)

where X∈ℝN×d𝑋superscriptℝ𝑁𝑑X\in\mathbb{R}^{N\times d} is the input sequence, W2∈ℝd×N,W1∈ℝd×dformulae-sequencesubscript𝑊2superscriptℝ𝑑𝑁subscript𝑊1superscriptℝ𝑑𝑑W_{2}\in\mathbb{R}^{d\times N},W_{1}\in\mathbb{R}^{d\times d}, and σRsubscript𝜎𝑅\sigma_{R} is the ReLU activation function. Given A𝐴A, the output of the Synthetic Dense function is computed as:

Y=Softmax​(A)​G​(X).𝑌Softmax𝐴𝐺𝑋\displaystyle Y=\text{Softmax}(A)G(X).

(10)

where G​(X)𝐺𝑋G(X) is another parameterized function ℝN×d→ℝN×d→superscriptℝ𝑁𝑑superscriptℝ𝑁𝑑\mathbb{R}^{N\times d}\rightarrow\mathbb{R}^{N\times d}.

##### Random Synthesizers

Another variant of the Synthesizer model uses random matrices for A𝐴A. In this case, the output can be expressed by:

Y=Softmax​(R)​G​(X).𝑌Softmax𝑅𝐺𝑋\displaystyle Y=\text{Softmax}(R)G(X).

(11)

where R∈ℝN×N𝑅superscriptℝ𝑁𝑁R\in\mathbb{R}^{N\times N} is a trainable and/or non-trainable matrix. In Tay et al. (2020a), the authors show that Random Synthesizers achieve competitive performance.

##### Factorized Variants

The Dense and Random Synthesizers also come with factorized variants that consider a low-rank structure of the attention matrix. For factorized random Synthesizer can be written as:

Y=Softmax​(R1​R2⊤)​G​(X).𝑌Softmaxsubscript𝑅1superscriptsubscript𝑅2top𝐺𝑋\displaystyle Y=\text{Softmax}(R_{1}R_{2}^{\top})G(X).

(12)

where R1,R2∈ℝN×ksubscript𝑅1subscript𝑅2superscriptℝ𝑁𝑘R_{1},R_{2}\in\mathbb{R}^{N\times k}. On the other hand, the Dense Synthesizer can be factorized as follows:

A=HB​(B)∗HC​(C)​where​B,C=FB​(Xi),FC​(Xi),formulae-sequence𝐴subscript𝐻𝐵𝐵subscript𝐻𝐶𝐶where𝐵𝐶subscript𝐹𝐵subscript𝑋𝑖subscript𝐹𝐶subscript𝑋𝑖\displaystyle A=H_{B}(B)*H_{C}(C)\>\>\text{where}\>\>B,C=F_{B}(X_{i}),F_{C}(X_{i}),

(13)

where FB(.)F_{B}(.) projects onto b𝑏b dimensions and FC(.)F_{C}(.) projects Xisubscript𝑋𝑖X_{i} onto c𝑐c dimensions with c×b=N𝑐𝑏𝑁c\times b=N. HB,HCsubscript𝐻𝐵subscript𝐻𝐶H_{B},H_{C} are tile and repeat functions respectively.

##### Parameter and Memory Complexity

For Random Synthesizers that adopt a non-trainable R𝑅R, there is no need to store N2superscript𝑁2N^{2} activations at this layer. For the trainable Random Synthesizer, the memory complexity and parameter complexity remains as N2superscript𝑁2N^{2}. However, there is no need to compute N2superscript𝑁2N^{2} dot products, reducing the computational costs significantly. The Factorized Random Synthesizers reduce the parameter costs to 2​(N×k)2𝑁𝑘2(N\times k).

#### 3.2.16 Transformer-XL

The Transformer-XL model (Dai et al., 2019) relies on segment-based recurrence. Segment-based recurrence can be considered an orthogonal approach to the other techniques discussed since it does not explicitly sparsify the dense self-attention matrix. Instead, it connects adjacent blocks with a recurrent mechanism.

##### Segment Recurrence

The recurrent mechanism in Transformer-XL is described as:

𝒉~τ+1n−1subscriptsuperscript~𝒉𝑛1𝜏1\displaystyle\tilde{\bm{h}}^{n-1}_{\tau+1}
=[SG​(𝒉τn−1)⊙𝒉τ+1n−1]absentdelimited-[]direct-productSGsubscriptsuperscript𝒉𝑛1𝜏subscriptsuperscript𝒉𝑛1𝜏1\displaystyle=[\text{SG}(\bm{h}^{n-1}_{\tau})\odot\bm{h}^{n-1}_{\tau+1}]

(14)

qτ+1n,kτ+1n,vτ+1nsubscriptsuperscript𝑞𝑛𝜏1subscriptsuperscript𝑘𝑛𝜏1subscriptsuperscript𝑣𝑛𝜏1\displaystyle q^{n}_{\tau+1},k^{n}_{\tau+1},v^{n}_{\tau+1}
=𝒉τ+1n−1​𝑾q⊤,𝒉~τ+1n−1​𝑾k⊤,𝒉~τ+1n−1​𝑾v⊤absentsubscriptsuperscript𝒉𝑛1𝜏1subscriptsuperscript𝑾top𝑞subscriptsuperscript~𝒉𝑛1𝜏1subscriptsuperscript𝑾top𝑘subscriptsuperscript~𝒉𝑛1𝜏1subscriptsuperscript𝑾top𝑣\displaystyle=\bm{h}^{n-1}_{\tau+1}\bm{W}^{\top}_{q}\>,\>\tilde{\bm{h}}^{n-1}_{\tau+1}\bm{W}^{\top}_{k}\>,\>\tilde{\bm{h}}^{n-1}_{\tau+1}\bm{W}^{\top}_{v}

(15)

𝒉τ+1nsubscriptsuperscript𝒉𝑛𝜏1\displaystyle\bm{h}^{n}_{\tau+1}
=Transformer​(qτ+1n,kτ+1n,vτ+1n)absentTransformersubscriptsuperscript𝑞𝑛𝜏1subscriptsuperscript𝑘𝑛𝜏1subscriptsuperscript𝑣𝑛𝜏1\displaystyle=\text{Transformer}(q^{n}_{\tau+1},k^{n}_{\tau+1},v^{n}_{\tau+1})

(16)

where SG() is the stop gradient function, ⊙direct-product\odot is the concatenation of two sequences along the length dimension. Notably, the keys and values are conditioned on the previous sequence length 𝒉~τ+1n−1subscriptsuperscript~𝒉𝑛1𝜏1\tilde{\bm{h}}^{n-1}_{\tau+1} instead of 𝒉τ+1n−1subscriptsuperscript𝒉𝑛1𝜏1\bm{h}^{n-1}_{\tau+1}

##### Relative Positional Encodings

Transformer-XL introduces novel relative position encodings. In this scheme, absolute positional encodings are not added to the content embeddings. Instead, they are only considered while computing attention weights where they can be replaced with relative position encodings. Since the relative position encodings are not directly relevant to the efficiency of the model, we refer interested readers to Dai et al. (2019) for more details.

#### 3.2.17 Compressive Transformers

Compressive Transformers (Rae et al., 2020) are a natural extension of the Transformer-XL model. The key idea behind the Compressive Transformer is to maintain a fine-grained memory of past segment activations. This is unlike Transformer-XL, which discards past activations as it moves across segments.

##### Model Memory

The Compressive Transformer is characterized by a dual model memory system - a primary model memory and a secondary compressed model memory. It maintains a model memory with nmsubscript𝑛𝑚n_{m} memory slots and nc​msubscript𝑛𝑐𝑚n_{cm} compressive memory slots. Whenever the model accepts a new input segment, the oldest nssubscript𝑛𝑠n_{s} activations in the primary model memory are moved to the compressed model memory where a compression function is applied.

##### Compression

These memories are compressed with a variety of compression functions such as (1) mean/max pooling (2) 1D convolutions, (3) dilated convolutions, and (4) most used (e.g., sorted by usage of attention).

##### Memory Reconstruction

In order to better retain memories over long sequences, the Compressive Transformer implements an auto-encoding loss that learns to reconstruct the original memory from its compressed version, i.e., La​e=‖old_mem−g​(new_cm(i))‖superscript𝐿𝑎𝑒normold_mem𝑔superscriptnew_cm𝑖L^{ae}=||\text{old\_mem}-g(\text{new\_cm}^{(i)})|| where g(.):ℝnsc×d→ℝns×dg(.):\mathbb{R}^{\frac{n_{s}}{c}\times d}\rightarrow\mathbb{R}^{n_{s}\times d} is a parameterized function. A second attention reconstruction is a lossy re-construct that attempts to reconstruct the attention over model memory instead of the lossless reconstruction of the model memory itself.

#### 3.2.18 Sparse Models

In this section we describe the family of Sparse models. Sparse models typically achieve a high parameter to FLOP ratio by sparsely activating a subset of parameters or activations. It is good to note that while most of the works within the scope of this survey deals with efficient attention, the scope of sparse models goes beyond the attention module and is generally applied more frequently to the feed forward layers (Lepikhin et al., 2020; Fedus et al., 2021). In this section, we discuss the prime variant for Sparse models, i.e., the Mixture-of-Experts based Sparse models which includes models such as GShard (Lepikhin et al., 2020), Switch Transformer (Fedus et al., 2021) and GLaM (Du et al., 2021).

##### Mixture-of-Experts

The key idea behind MoE is to route token xisubscript𝑥𝑖x_{i} to a set of selected experts determined by a routing function. The routing function typically computed a linear combination over experts using the softmax function and can be interpreted as a form of gating mechanism. The top-k gate values are then selected for each token xisubscript𝑥𝑖x_{i} and the final output of that layer is determined by a linear combination of selected top-k experts. This MoE layer remains foundational and fundamental to many MoE architectures, with the exception of certain implementation details. For example, Switch uses a top-1 routing strategy while GShard uses a group-level top-2 gating.

## 4 Discussion

This section explores the state of research pertaining to this class of efficient models.

### 4.1 On Evaluation

While the field is bustling with new Transformer models, there is not an easy way to compare these models side by side. Many research papers select their own benchmarks to showcase the abilities of the proposed model. This is also coupled with different hyperparameter settings like model sizes and configurations which can make it difficult to correctly attribute the reason for the performance gains.
Moreover, some papers conflate this with pretraining (Devlin et al., 2018) which makes it even harder to distinguish the relative performance of these different models. It is still a mystery to which fundamental efficient Transformer block one should consider using.

On one hand, there are multiple models that focus on generative modeling, showcasing the ability of the proposed Transformer unit on auto-regressive modeling of sequences. To this end, Sparse Transformers (Child et al., 2019), Adaptive Transformers (Correia et al., 2019), Routing Transformers (Roy et al., 2020) and Reformers (Kitaev et al., 2020) are mainly focused on generative modeling tasks. These benchmarks typically involve language modeling and/or pixel-wise image generation on datasets such as wikitext (Merity et al., 2017), and/or ImageNet (Deng et al., 2009) / CIFAR (Krizhevsky et al., 2009). Models that use segment based recurrence such as Transformer-XL and Compressive Transformers are also focused on long-range language modeling tasks such as PG-19.

On one hand, a collection of models is mainly focused on encoding-only tasks such as question answering, reading comprehension and or selections from the GLUE benchmark. For example, the ETC model (Ainslie et al., 2020) only runs experiments on question answering benchmarks such as NaturalQuestions (Kwiatkowski et al., 2019) or TriviaQA (Joshi et al., 2017). On the other hand, the Linformer (Wang et al., 2020c) focuses on subsets of the GLUE (Wang et al., 2018) benchmark. This split is very natural and intuitive, since models like ETC and Linformer cannot be used in an auto-regressive fashion. This exacerbates the challenges associated with comparing these encoder-only models with the other models.

There are models that focus on a balance of both. Longformer (Beltagy et al., 2020) tries to balance this by running benchmarks on both generative modeling and encoder-only tasks. The Sinkhorn Transformer (Tay et al., 2020b) compares on both generative modeling tasks as well as encoding only tasks.

Additionally, it is also worth noting that, although Seq2Seq machine translation (MT) was one of the problems that popularized Transformer models, not many of these efficient Transformer models are evaluated on MT tasks. This is likely because sequence lengths in MT are not long enough to warrant the usage of these models.

While generative modeling, GLUE tasks and/or question answering appear to be the common evaluation benchmarks adopted by many of these tasks, there are several niche benchmarks that a small isolated number of papers choose to evaluate on. For starters, the Performer model (Choromanski et al., 2020a) evaluates on masked language modeling on proteins, deviating from serious head-on comparisons with other efficient Transformer models. The Linear Transformer (Katharopoulos et al., 2020) also evaluates on speech recognition, which is a rare benchmark amongst this group of papers.

There have been recent attempts to unify evaluation on Efficient Transformers, namely Long Range Arena, i.e., LRA, (Tay et al., 2021a) that benchmarked 10 different xformer variants on long range modeling tasks. It is good to note that LRA was designed for evaluating Transformers in encoder-only mode and do not consider generative (or autoregressive tasks) that require causal masking.

### 4.2 On Model Design Trends

When matching our broad categorization against the timeline of the introduction of these models, we are able to see the trend that the community is taking towards designing efficient Transformer models. Early work in this area has primarilyy been focused on more intuitive and simple approaches such as fixed patterns. To this end, most early work in this area is based on block/local patterns such as Image Transformer (Parmar et al., 2018), Compressed Attention (Liu et al., 2018), Blockwise Transformer (Qiu et al., 2019) or the local windows in Sparse Transformer (Child et al., 2019).

The paradigm of factorizing various fixed patterns was first introduced in Child et al. (2019) and CCNet (Huang et al., 2019). Around this same time, we start to observe early traces of model-memory-based approaches from both the inducing point methods in the Set Transformer (Lee et al., 2019) or global nodes in the Star Transformer (Guo et al., 2019a) model.

We observe the next wave of models comes in the form of learnable sparsity patterns. Reformer (Kitaev et al., 2020) and Routing Transformers (Roy et al., 2020) are very similar in the sense that they are models that learn to cluster/bucket tokens before performing attention. The key difference is the means to the end whereby Reformer uses a hashing function while the Routing Transformer uses online k𝑘k-means for cluster assignment. In parallel, Sinkhorn Transformers (Tay et al., 2020b) are also based on the idea of sorting, albeit at the block level. These three models largely follow a similar paradigm of re-arranging sequences for efficient computation of attention scores.

Next, we then observe several extensions that are largely built off the Sparse Transformer paradigm. The ETC (Ainslie et al., 2020) and Longformer (Beltagy et al., 2020) models are very similar ideas that are fundamentally Sparse Transformer extensions. These models incorporate the notion of a global model memory, which is reminiscent of the Set Transformer’s inducing point method or the global model memory of the Star Transformer. Modifications to strides, such as using dilated windows was also proposed in the Longformer work.

The most recent wave of models we’ve been seeing is models that are based on low-rank approximation or kernel methods, e.g., models such as Low-Rank Transformer (Winata et al., 2020), Linformer (Wang et al., 2020c), Performer (Choromanski et al., 2020a) and/or Linear Transformers (Katharopoulos et al., 2020). Although due to the state of evaluation and the high parallelism of research, it is quite unclear if this low-rank or kernel paradigm is actually better than the learnable pattern (LP) or model memory based efficient Transformer models.

More recently, there have been more models that propose a two-pronged or two-step attention mechanism combining models from different techniques. The Long Short Transformer (Zhu et al., 2021) is a dynamic form of Linformer combined with Fixed Pattern attention mechanisms. On the other hand, models like Poolingformer also explicitly construct a two-level attention mechanism with techniques reminiscent of memory-based approaches and local attention. Scatter Brain is a new work (Chen et al., 2021) attempts to unify sparse (fixed pattern) attention with low-rank attention. Two stage attention mechanisms are also proposed by Luna (Ma et al., 2021)

On the side, it is important to note that the recurrent based models (Transformer-XL and Compressive Transformers) seem to operate orthogonally and are not as directly comparable to the other models. We also observe that Sparse models (Lepikhin et al., 2020; Fedus et al., 2021) that are not only applicable to attention modules, are also recently emerging and becoming more popular and have demonstrated considerable success in the recent months (Du et al., 2021).

### 4.3 Brief Discussion on Orthogonal Efficiency Efforts

While this paper is mainly focused on (1) the computational and memory complexity of the self-attention module and (2) sparsity and adaptive computation, we briefly summarize several orthogonal efforts that may also contribute to model efficiency, scalability, and overall usability of Transformer models.

- •

Weight Sharing - Sharing parameters of the Transformer models would help in reducing overall model size. The Universal Transformers (Dehghani et al., 2018) tie attention and transition weights across layers. Similarly, Albert (Lan et al., 2019) does the same parameter sharing across layers. On the other hand, the Quaternion Transformer (Tay et al., 2019) proposes a weight sharing scheme inspired by Hamilton products that locally shares the components in the linear transformation layers.

- •

Quantization / Mixed Precision - Learning mixed precision models has the potential to improve memory costs. Q-BERT (Shen et al., 2020) is a model that quantizes Transformer models to ultra-low precision. Meanwhile mixed precision training (Ott et al., 2019) is a highly popular technique to reduced the memory costs of training Transformers. Fan et al. (2020) applies Quantization Aware training to Transformer models.

- •

Inference-time Efficiency and Network Pruning - Multiple research directions explore improving the Transformer efficiency at inference time. One prime example is network model. An example is to prune attention heads during inference (Voita et al., 2019; Michel et al., 2019). This has shown to have minimal degradation of performance on downstream tasks. On the other hand, Lagunas et al. (2021) proposes a “block” pruning approach which can make a Transformer 2.4x faster with little loss in predictive performance on language tasks. Another line of work involved fast exit during inference which allows us to exit compute if the model is confident of its predictions (Schuster et al., 2021).

- •

Knowledge Distillation -
Knowledge distillation (KD) (Hinton et al., 2015) has been a useful technique for transfering the knowledge learned from a larger teacher model to a smaller student model. The smaller model can then be efficiently deployed into production. There have been many attempts to distill large Transformer models. For example, DistilBERT (Sanh et al., 2019), task-specific distillation (Tang et al., 2019) and TinyBERT (Jiao et al., 2019).

- •

Neural Architecture Search (NAS) -
Searching for more efficient Transformer architectures is also a common strategy. Guo et al. (2019b) proposed Neural Architecture Transformer (NAT), using NAS to search for more compact and efficient Transformers by removing redundant operations. Wang et al. (2020a) proposed HAT (Hardware-aware Transformers), a method that leverages NAS and uses hardware efficiency feedback as a reward signal.

- •

Task Adapters - This line of research has been primarily focused on the problem of fine-tuning large Transformer on T𝑇T tasks and aiming to reuse parameters across a variety of tasks. The key idea is that task adapters (Houlsby et al., 2019) enable reuse of parameters across tasks and reuse the need of serving T𝑇T models in production - resulting in overall parameter savings. A modest number of models have been proposed, such as PALS (Stickland and Murray, 2019), MAD-X (Pfeiffer et al., 2020) and HyperGrid (Tay et al., 2020c).

- •

Alternative Architectures - A considerable amount of effort have gone into designing Transformer alternatives. Amongst the many alternatives considered, a prominent line of emerging research belongs to the family of MLP Mixers (Tolstikhin et al., 2021). Different mixing operations have been proposed, such as the G-MLP (Liu et al., 2021a), FNet (Lee-Thorp et al., 2021). Synthesizers (Tay et al., 2020a), although commonly referred to as an efficient attention method, is also an early manifestation of the mixer line of work, as the random matrices similarly act as an unconditioned mixing operation. A recent promising line of work, based on Structured State Spaces (Gu et al., 2021) also demonstrated very promising results on long range modeling. Lastly, convolutional models are generally more efficient than Transformers since convolutional kernels operate on a fixed, small local neighborhood around the input token. Tay et al. (2021b) shows that, when pre-trained, these more efficient convolutional models can sometimes match the predictive performance of Transformer ones.

### 4.4 A Retrospective on the Past Year and Future Research Directions

With our timely V2 update of this survey (updated December 2021), we present retrospective thoughts about how the field has evolved over the past year or so. From the time of last update, it is undeniable that more xformer variants have emerged to offer more efficient alternatives for vanilla Transformers.

Notably, examples of these include Nyströmformer (Xiong et al., 2021b), Perceiver (Jaegle et al., 2021), RFA (Peng et al., 2021), Luna (Ma et al., 2021) and Long Short Transformer (Zhu et al., 2021). There were also other notable models that sprung up around the time when this manuscript was published and narrowly missed the inclusion in the first edition (e.g., Funnel Transformer (Dai et al., 2020)). Amongst all the new xformer variants, it is good to note that most do not stray away from the fundamental concepts presented in the first version. Our taxonomy and categorization was more or less broad enough to capture many of these models as they use fundamental ideas that are already present in existing work and therefore can be categorized appropriately.
Many works can be thought of explicit combinations of existing techniques (two-staged or combination of two method classes) or improvements over existing methods (dynamic formulation of Linformer’s low rank projection or better kernels for Linear Transformers). Even though many existing ‘memory’ models utilize a form of downsampling to achieve a speed and efficiency gain, we erected a new categoriation of ‘downsampling’ to better reflect this new emerging trend (Dai et al., 2020; Jaegle et al., 2021; Tay et al., 2021c; Ryoo et al., 2021).

Over the past year, it is evident that a lot of research investment have been poured into making quadratic attention scalable, in terms of complexity, or sometimes memory.
At this juncture, it is good to ponder about real tangible need for linear-time attention. Many applications even in language and vision are still dominated by vanilla Transformers with quadratic attention and none of these xformer variants have caught on as the defacto standard. There might be multiple explanations from multiple angles for this phenomena. Firstly, linear attention (e.g., Performer) models struggle to be competitive on common benchmarks, as noted from multiple sources (Xiong et al., 2021a; Anonymous, 2021b).

It is good to note that, apart from toy setups or specific domains and problems, they have never been battle-tested against common paradigms like pretrain-and-finetuning only up till recently. Meanwhile, local attention models based on fixed and/or learned patterns such as Sparse Transformers (Child et al., 2019), Longformer (Beltagy et al., 2020), ETC (Ainslie et al., 2020) or BigBird (Zaheer et al., 2020) have seen more reasonable usage, especially within the areas of long context question answering.
However, the high intrinsic implementation complexity of methods such as in ETC (Ainslie et al., 2020) (substantially increases code complexity by having so many different directions of attention), Swin Transformer (Liu et al., 2021b) or Longformer (Beltagy et al., 2020) (requiring custom CUDA kernels and thus making it prohibitive on hardware such as TPUs) might be reasons why these models have yet to found themselves serving as a good, simple-to-use drop-in Transformer replacement.

As noted by (Rabe and Staats, 2021), for applications that require to flex on sequence length and memory needs time to time, it might be suffice to ‘just sequentially process it’ even if that might not be inherently as satisfying as finding a theoretical approximate. In parallel, (Xiong et al., 2021a) suggests that local attention, when done right, can be a really tough baseline to beat.

A notable fact about the barrage of efficient attention models is the overloading of the term efficient. It is commonly misunderstood that efficient attention models always imply that the Transformer is fast. The truth is that many of these efficient attention models, owing to their innovation constraints, may make the model much slower. Moreover, many linear attention models do not observe any speed or memory gain at all if the sequence length is short. Many of them have extraordinarily painful requirements to achieve causal masking (or TPU packing) (Choromanski et al., 2020b; Peng et al., 2021; Wang et al., 2020c) and often have to substantially trade-off throughput for linear complexity. On the other hand, some models cannot be packed or causally masked at all. More notes and discussions about this efficiency misnomer can be found in this paper (Dehghani et al., 2021) which we encourage readers to also peruse.

This update also extends the original scope of efficient attention based xformer models to sparse models even if they did not necessarily target the attention modules. We believe that sparse models were a necessarily addition to the scope to this paper given its recent signs of promise (Fedus et al., 2021; Du et al., 2021; Zoph et al., 2022). A special note was made to recognize the work done in alternative architectures in the past year (in the section on orthogonal directions). Mixer type architectures (Tolstikhin et al., 2021) have garnered some interest in computer vision but seem to not perform well on language (Anonymous, 2021a). Meanwhile, alternative models based on Structured State Spaces such as S4 (Gu et al., 2021) have solved the hardest Path-X task in the Long Range Arena benchmark (Tay et al., 2021a). It should be exciting to see how a model such as S4 would perform at scale, and under pretrained conditions.

As the year comes to a close and as we reflect back on the amazing advances made by the community, we begin to ponder about the future of efficient transfomers and what the ideal transformer model should look like. We think that the ideal xformer should hopefully take care of the quadratic memory problem, while retaining universality (e.g., do well on most tasks and not only on long range tasks). The ideal xformer should also not trade-off speed for memory and should not sacrifice the ability to be TPU-packed and/or causal masked. It should ideally be simple and not make use of rigid hard-coding or over-excessive engineering, i.e., it should be elegant and scale well. Ideally, efficiency would be baked right into the next generation of Transformers instead of always having a side variant that one could use for long context tasks. While we cannot explicitly point at any of the xformer variants as the definitive one that have solved the efficiency problem in Transformers, we are optimistic that, given about the pace of advance, the true xformer will emerge eventually. It is then a question of whether that new xformer will still be a Transformer.

## 5 Conclusion

In this paper we surveyed the literature on efficient Transformer models especially pertaining to the quadratic complexity of the self-attention module. We provided a taxonomy and high-level abstraction of the core techniques employed in these class of new models. We characterized the existing models based on techniques and provided a comprehensive walkthrough of several of the efficient Transformer models. Finally, we discussed the evaluation landscape of these models along with the design trends of these models. We ended of with a brief discussion of other parallel orthogonal efforts that may improve the efficiency of Transformer models in general. Note: This survey may be revised again bi-annually or annually. Feel free to send feedback to our email address. While we may not reply to all, we certainly would read them. We also welcome anonymous feedback to https://forms.gle/kqjmhSDEQrmL4Egk6.

Acknowledgments

The authors would like to send the numerous authors who send us feedback via email. We tried our best to incorporate most of the suggestions as we sat fit. We also thank Tamas Sarlos for feedback on this manuscript.

## References

- Adams and Zemel (2011)

Ryan Prescott Adams and Richard S Zemel.

Ranking via sinkhorn propagation.

arXiv preprint arXiv:1106.1925, 2011.

- Ahmed et al. (2017)

Karim Ahmed, Nitish Shirish Keskar, and Richard Socher.

Weighted transformer network for machine translation.

arXiv preprint arXiv:1711.02132, 2017.

- Ainslie et al. (2020)

Joshua Ainslie, Santiago Ontanon, Chris Alberti, Philip Pham, Anirudh Ravula,
and Sumit Sanghai.

Etc: Encoding long and structured data in transformers.

Proceedings of EMNLP, 2020.

- Anonymous (2021a)

Anonymous.

Remixers: A mixer-transformer architecture with compositional
operators for natural language understanding.

ACL RR 2021 September Submission, 2021a.

- Anonymous (2021b)

Anonymous.

Scaling laws vs model architectures: How does inductive bias
influence scaling? an extensive empirical study on language tasks.

ACL Rolling Review, September, 2021b.

- Ba et al. (2016)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton.

Layer normalization.

arXiv preprint arXiv:1607.06450, 2016.

- Beltagy et al. (2020)

Iz Beltagy, Matthew E Peters, and Arman Cohan.

Longformer: The long-document transformer.

Proceedings of EMNLP, 2020.

- Brown et al. (2020)

Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell,
et al.

Language models are few-shot learners.

arXiv preprint arXiv:2005.14165, 2020.

- Carion et al. (2020)

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander
Kirillov, and Sergey Zagoruyko.

End-to-end object detection with transformers.

arXiv preprint arXiv:2005.12872, 2020.

- Chen et al. (2021)

Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher
Ré.

Scatterbrain: Unifying sparse and low-rank attention.

In Thirty-Fifth Conference on Neural Information Processing
Systems, 2021.

- Child et al. (2019)

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.

Generating long sequences with sparse transformers.

arXiv preprint arXiv:1904.10509, 2019.

- Choromanski et al. (2020a)

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Jared
Davis, Tamas Sarlos, David Belanger, Lucy Colwell, and Adrian Weller.

Masked language modeling for proteins via linearly scalable
long-context transformers.

Proceedings of ICLR, 2020a.

- Choromanski et al. (2020b)

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song,
Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin,
Lukasz Kaiser, et al.

Rethinking attention with performers.

Proceedings of ICLR, 2020b.

- Clevert et al. (2015)

Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter.

Fast and accurate deep network learning by exponential linear units
(elus).

Proceedings of ICLR 2016, 2015.

- Correia et al. (2019)

Gonçalo M Correia, Vlad Niculae, and André FT Martins.

Adaptively sparse transformers.

Proceedings of EMNLP, 2019.

- Dai et al. (2019)

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan
Salakhutdinov.

Transformer-xl: Attentive language models beyond a fixed-length
context.

Proceedings of ACL, 2019.

- Dai et al. (2020)

Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V Le.

Funnel-transformer: Filtering out sequential redundancy for efficient
language processing.

Proceedings of NeurIPS, 2020.

- Dehghani et al. (2018)

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz
Kaiser.

Universal transformers.

Proceedings of ICLR, 2018.

- Dehghani et al. (2021)

Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, and Yi Tay.

The efficiency misnomer.

arXiv preprint arXiv:2110.12894, 2021.

- Deng et al. (2009)

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.

Imagenet: A large-scale hierarchical image database.

In 2009 IEEE conference on computer vision and pattern
recognition, pages 248–255. Ieee, 2009.

- Devlin et al. (2018)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

Bert: Pre-training of deep bidirectional transformers for language
understanding.

Proceedings of NAACL, 2018.

- Du et al. (2021)

Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong
Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam
Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster,
Marie Pellat, Kevin Robinson, Kathy Meier-Hellstern, Toju Duke, Lucas Dixon,
Kun Zhang, Quoc V Le, Yonghui Wu, Zhifeng Chen, and Claire Cui.

Glam: Efficient scaling of language models with mixture-of-experts,
2021.

- Fan et al. (2020)

Angela Fan, Pierre Stock, Benjamin Graham, Edouard Grave, Remi Gribonval, Herve
Jegou, and Armand Joulin.

Training with quantization noise for extreme fixed-point compression.

arXiv preprint arXiv:2004.07320, 2020.

- Fedus et al. (2021)

William Fedus, Barret Zoph, and Noam Shazeer.

Switch transformers: Scaling to trillion parameter models with simple
and efficient sparsity.

arXiv preprint arXiv:2101.03961, 2021.

- Gu et al. (2021)

Albert Gu, Karan Goel, and Christopher Ré.

Efficiently modeling long sequences with structured state spaces.

Proceedings of NeurIPS, 2021.

- Guo et al. (2019a)

Qipeng Guo, Xipeng Qiu, Pengfei Liu, Yunfan Shao, Xiangyang Xue, and Zheng
Zhang.

Star-transformer.

Proceedings of NAACL, 2019a.

- Guo et al. (2019b)

Yong Guo, Yin Zheng, Mingkui Tan, Qi Chen, Jian Chen, Peilin Zhao, and Junzhou
Huang.

Nat: Neural architecture transformer for accurate and compact
architectures.

In Advances in Neural Information Processing Systems, pages
737–748, 2019b.

- Hinton et al. (2015)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.

Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531, 2015.

- Ho et al. (2019)

Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans.

Axial attention in multidimensional transformers.

arXiv preprint arXiv:1912.12180, 2019.

- Houlsby et al. (2019)

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.

Parameter-efficient transfer learning for nlp.

Proceedings of ICML, 2019.

- Huang et al. (2019)

Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, and Wenyu
Liu.

Ccnet: Criss-cross attention for semantic segmentation.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 603–612, 2019.

- Jaegle et al. (2021)

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin
Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan
Shelhamer, et al.

Perceiver io: A general architecture for structured inputs &
outputs.

arXiv preprint arXiv:2107.14795, 2021.

- Jaszczur et al. (2021)

Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, Łukasz Kaiser,
Wojciech Gajewski, Henryk Michalewski, and Jonni Kanerva.

Sparse is enough in scaling transformers.

Advances in Neural Information Processing Systems, 34, 2021.

- Jiao et al. (2019)

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang
Wang, and Qun Liu.

Tinybert: Distilling bert for natural language understanding.

arXiv preprint arXiv:1909.10351, 2019.

- Joshi et al. (2017)

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer.

Triviaqa: A large scale distantly supervised challenge dataset for
reading comprehension.

In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics, Vancouver, Canada, July 2017. Association for
Computational Linguistics.

- Kaiser et al. (2017)

Lukasz Kaiser, Aidan N Gomez, and Francois Chollet.

Depthwise separable convolutions for neural machine translation.

Proceedings of ICLR, 2017.

- Katharopoulos et al. (2020)

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François
Fleuret.

Transformers are rnns: Fast autoregressive transformers with linear
attention.

arXiv preprint arXiv:2006.16236, 2020.

- Kitaev et al. (2020)

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya.

Reformer: The efficient transformer.

In International Conference on Learning Representations, 2020.

URL https://openreview.net/forum?id=rkgNKkHtvB.

- Krizhevsky et al. (2009)

Alex Krizhevsky, Geoffrey Hinton, et al.

Learning multiple layers of features from tiny images.

2009.

- Kwiatkowski et al. (2019)

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey,
Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang,
Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov.

Natural questions: a benchmark for question answering research.

Transactions of the Association of Computational Linguistics,
2019.

- Lagunas et al. (2021)

François Lagunas, Ella Charlaix, Victor Sanh, and Alexander M Rush.

Block pruning for faster transformers.

Proceedings of EMNLP 2021, 2021.

- Lample et al. (2019)

Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic
Denoyer, and Hervé Jégou.

Large memory layers with product keys.

arXiv preprint arXiv:1907.05242, 2019.

- Lan et al. (2019)

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and
Radu Soricut.

Albert: A lite bert for self-supervised learning of language
representations.

Proceedings of ICLR, 2019.

- Lee et al. (2019)

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye
Teh.

Set transformer: A framework for attention-based
permutation-invariant neural networks.

In International Conference on Machine Learning, pages
3744–3753, 2019.

- Lee-Thorp et al. (2021)

James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, and Santiago Ontanon.

Fnet: Mixing tokens with fourier transforms.

arXiv preprint arXiv:2105.03824, 2021.

- Lepikhin et al. (2020)

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping
Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen.

Gshard: Scaling giant models with conditional computation and
automatic sharding.

arXiv preprint arXiv:2006.16668, 2020.

- Lewis et al. (2021)

Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer.

Base layers: Simplifying training of large, sparse models.

arXiv preprint arXiv:2103.16716, 2021.

- Liu et al. (2021a)

Hanxiao Liu, Zihang Dai, David R So, and Quoc V Le.

Pay attention to mlps.

Proceedings of NeurIPS, 2021a.

- Liu et al. (2018)

Peter J Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz
Kaiser, and Noam Shazeer.

Generating wikipedia by summarizing long sequences.

Proceedings of ICLR, 2018.

- Liu et al. (2021b)

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and
Baining Guo.

Swin transformer: Hierarchical vision transformer using shifted
windows.

arXiv preprint arXiv:2103.14030, 2021b.

- Ma et al. (2021)

Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and
Luke Zettlemoyer.

Luna: Linear unified nested attention.

In Proceedings of NeurIPS 2021, 2021.

- Merity et al. (2017)

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher.

Pointer sentinel mixture models.

Proceedings of ICLR, 2017.

- Michel et al. (2019)

Paul Michel, Omer Levy, and Graham Neubig.

Are sixteen heads really better than one?

Proceedings of NeurIPS, 2019.

- Ott et al. (2019)

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng,
David Grangier, and Michael Auli.

fairseq: A fast, extensible toolkit for sequence modeling.

arXiv preprint arXiv:1904.01038, 2019.

- Parmar et al. (2018)

Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Łukasz Kaiser, Noam Shazeer,
Alexander Ku, and Dustin Tran.

Image transformer.

Proceedings of ICML 2018, 2018.

- Parmar et al. (2019)

Niki Parmar, Prajit Ramachandran, Ashish Vaswani, Irwan Bello, Anselm Levskaya,
and Jon Shlens.

Stand-alone self-attention in vision models.

In Advances in Neural Information Processing Systems, pages
68–80, 2019.

- Peng et al. (2021)

Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A Smith, and
Lingpeng Kong.

Random feature attention.

Proceedings of ICLR, 2021.

- Pfeiffer et al. (2020)

Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder.

Mad-x: An adapter-based framework for multi-task cross-lingual
transfer.

Proceedings of EMNLP, 2020.

- Qiu et al. (2019)

Jiezhong Qiu, Hao Ma, Omer Levy, Scott Wen-tau Yih, Sinong Wang, and Jie Tang.

Blockwise self-attention for long document understanding.

arXiv preprint arXiv:1911.02972, 2019.

- Rabe and Staats (2021)

Markus N Rabe and Charles Staats.

Self-attention does not need o (n^ 2) memory.

arXiv preprint arXiv:2112.05682, 2021.

- Rae et al. (2020)

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and
Timothy P. Lillicrap.

Compressive transformers for long-range sequence modelling.

In International Conference on Learning Representations, 2020.

URL https://openreview.net/forum?id=SylKikSYDH.

- Raffel et al. (2019)

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li, and Peter J Liu.

Exploring the limits of transfer learning with a unified text-to-text
transformer.

Journal of Machine Learning Research, 2020, 2019.

- Roller et al. (2021)

Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston.

Hash layers for large sparse models.

arXiv preprint arXiv:2106.04426, 2021.

- Roy et al. (2020)

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier.

Efficient content-based sparse attention with routing transformers.

Proceedings of TACL, 2020.

- Ryoo et al. (2021)

Michael S. Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia
Angelova.

Tokenlearner: Adaptive space-time tokenization for videos.

In Advances in Neural Information Processing Systems
(NeurIPS), 2021.

- Sanh et al. (2019)

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.

Distilbert, a distilled version of bert: smaller, faster, cheaper and
lighter.

arXiv preprint arXiv:1910.01108, 2019.

- Schuster et al. (2021)

Tal Schuster, Adam Fisch, Tommi Jaakkola, and Regina Barzilay.

Consistent accelerated inference via confident adaptive transformers.

In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing, pages 4962–4979, Online and Punta Cana,
Dominican Republic, November 2021. Association for Computational Linguistics.

URL https://aclanthology.org/2021.emnlp-main.406.

- Shen et al. (2020)

Sheng Shen, Zhen Dong, Jiayu Ye, Linjian Ma, Zhewei Yao, Amir Gholami,
Michael W Mahoney, and Kurt Keutzer.

Q-bert: Hessian based ultra low precision quantization of bert.

2020.

- Sinkhorn (1964)

Richard Sinkhorn.

A relationship between arbitrary positive matrices and doubly
stochastic matrices.

The annals of mathematical statistics, 35(2):876–879, 1964.

- So et al. (2019)

David R So, Chen Liang, and Quoc V Le.

The evolved transformer.

Proceedings of ICML, 2019.

- Stickland and Murray (2019)

Asa Cooper Stickland and Iain Murray.

Bert and pals: Projected attention layers for efficient adaptation in
multi-task learning.

Proceedings of ICML, 2019.

- Sukhbaatar et al. (2019a)

Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin.

Adaptive attention span in transformers.

arXiv preprint arXiv:1905.07799, 2019a.

- Sukhbaatar et al. (2019b)

Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Herve Jegou, and Armand
Joulin.

Augmenting self-attention with persistent memory.

arXiv preprint arXiv:1907.01470, 2019b.

- Tang et al. (2019)

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin.

Distilling task-specific knowledge from bert into simple neural
networks.

arXiv preprint arXiv:1903.12136, 2019.

- Tay et al. (2019)

Yi Tay, Aston Zhang, Luu Anh Tuan, Jinfeng Rao, Shuai Zhang, Shuohang Wang, Jie
Fu, and Siu Cheung Hui.

Lightweight and efficient neural natural language processing with
quaternion networks.

Proceedings of ACL, 2019.

- Tay et al. (2020a)

Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, and Che Zheng.

Synthesizer: Rethinking self-attention in transformer models.

Proceedings of ICML, 2021, 2020a.

- Tay et al. (2020b)

Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, and Da-Cheng Juan.

Sparse sinkhorn attention.

Proceedings of ICML, 2020b.

- Tay et al. (2020c)

Yi Tay, Zhe Zhao, Dara Bahri, Donald Metzler, and Da-Cheng Juan.

Hypergrid: Efficient multi-task transformers with grid-wise
decomposable hyper projections.

Proceedings of ICLR, 2020c.

- Tay et al. (2021a)

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham,
Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler.

Long range arena: A benchmark for efficient transformers.

Proceedings of ICLR, 2021a.

- Tay et al. (2021b)

Yi Tay, Mostafa Dehghani, Jai Gupta, Dara Bahri, Vamsi Aribandi, Zhen Qin, and
Donald Metzler.

Are pre-trained convolutions better than pre-trained transformers?

arXiv preprint arXiv:2105.03322, 2021b.

- Tay et al. (2021c)

Yi Tay, Vinh Q Tran, Sebastian Ruder, Jai Gupta, Hyung Won Chung, Dara Bahri,
Zhen Qin, Simon Baumgartner, Cong Yu, and Donald Metzler.

Charformer: Fast character transformers via gradient-based subword
tokenization.

arXiv preprint arXiv:2106.12672, 2021c.

- Tolstikhin et al. (2021)

Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai,
Thomas Unterthiner, Jessica Yung, Andreas Peter Steiner, Daniel Keysers,
Jakob Uszkoreit, et al.

Mlp-mixer: An all-mlp architecture for vision.

In Thirty-Fifth Conference on Neural Information Processing
Systems, 2021.

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In Advances in neural information processing systems, pages
5998–6008, 2017.

- Voita et al. (2019)

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov.

Analyzing multi-head self-attention: Specialized heads do the heavy
lifting, the rest can be pruned.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 5797–5808, Florence, Italy, July 2019.
Association for Computational Linguistics.

doi: 10.18653/v1/P19-1580.

URL https://aclanthology.org/P19-1580.

- Vyas et al. (2020)

Apoorv Vyas, Angelos Katharopoulos, and François Fleuret.

Fast transformers with clustered attention.

Proceedings of NeurIPS, 2020.

- Wang et al. (2018)

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel
Bowman.

GLUE: A multi-task benchmark and analysis platform for natural
language understanding.

In Proceedings of the 2018 EMNLP Workshop BlackboxNLP:
Analyzing and Interpreting Neural Networks for NLP, pages 353–355,
Brussels, Belgium, November 2018. Association for Computational Linguistics.

doi: 10.18653/v1/W18-5446.

URL https://aclanthology.org/W18-5446.

- Wang et al. (2020a)

Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan, and
Song Han.

Hat: Hardware-aware transformers for efficient natural language
processing.

arXiv preprint arXiv:2005.14187, 2020a.

- Wang et al. (2020b)

Shuohang Wang, Luowei Zhou, Zhe Gan, Yen-Chun Chen, Yuwei Fang, Siqi Sun,
Yu Cheng, and Jingjing Liu.

Cluster-former: Clustering-based sparse transformer for long-range
dependency encoding.

Proceedings of ACL-IJCNLP (Findings), 2020b.

- Wang et al. (2020c)

Sinong Wang, Belinda Li, Madian Khabsa, Han Fang, and Hao Ma.

Linformer: Self-attention with linear complexity.

arXiv preprint arXiv:2006.04768, 2020c.

- Weissenborn et al. (2019)

Dirk Weissenborn, Oscar Täckström, and Jakob Uszkoreit.

Scaling autoregressive video models.

Proceedings of ICLR, 2019.

- Winata et al. (2020)

Genta Indra Winata, Samuel Cahyawijaya, Zhaojiang Lin, Zihan Liu, and Pascale
Fung.

Lightweight and efficient end-to-end speech recognition using
low-rank transformer.

In ICASSP 2020-2020 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 6144–6148. IEEE, 2020.

- Xiong et al. (2021a)

Wenhan Xiong, Barlas Oğuz, Anchit Gupta, Xilun Chen, Diana Liskovich,
Omer Levy, Wen-tau Yih, and Yashar Mehdad.

Simple local attentions remain competitive for long-context tasks.

arXiv preprint arXiv:2112.07210, 2021a.

- Xiong et al. (2021b)

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung,
Yin Li, and Vikas Singh.

Nystr\\\backslash” omformer: A nystr\\\backslash” om-based algorithm
for approximating self-attention.

Proceedings of AAAI, 2021b.

- Yun et al. (2020)

Chulhee Yun, Yin-Wen Chang, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J
Reddi, and Sanjiv Kumar.

o​(n)𝑜𝑛o(n) connections are expressive enough: Universal
approximability of sparse transformers.

Proceedings of NeurIPS, 2020.

- Zaheer et al. (2017)

Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R
Salakhutdinov, and Alexander J Smola.

Deep sets.

In Advances in neural information processing systems, pages
3391–3401, 2017.

- Zaheer et al. (2020)

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti,
Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al.

Big bird: Transformers for longer sequences.

Proceedings of NeurIPS, 2020.

- Zhang et al. (2021)

Hang Zhang, Yeyun Gong, Yelong Shen, Weisheng Li, Jiancheng Lv, Nan Duan, and
Weizhu Chen.

Poolingformer: Long document modeling with pooling attention.

Proceedings of ICML, 2021.

- Zhu et al. (2021)

Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima
Anandkumar, and Bryan Catanzaro.

Long-short transformer: Efficient transformers for language and
vision.

Advances in Neural Information Processing Systems, 34, 2021.

- Zoph et al. (2022)

Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam
Shazeer, and William Fedus.

Designing effective sparse expert models.

arXiv preprint arXiv:2202.08906, 2022.

Generated on Thu Mar 7 05:02:48 2024 by LaTeXML
