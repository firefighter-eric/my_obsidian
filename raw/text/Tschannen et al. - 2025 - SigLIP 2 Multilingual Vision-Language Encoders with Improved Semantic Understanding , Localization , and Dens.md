# Tschannen et al. - 2025 - SigLIP 2 Multilingual Vision-Language Encoders with Improved Semantic Understanding , Localization , and Dens

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Tschannen et al. - 2025 - SigLIP 2 Multilingual Vision-Language Encoders with Improved Semantic Understanding , Localization , and Dens.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2502.14786
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\pdftrailerid

redacted

 
 
 
 
 

 

\correspondingauthortschannen@google.com

# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

Michael Tschannen

Core contributor

Project lead

Alexey Gritsenko

Core contributor

Xiao Wang

Core contributor

Muhammad Ferjad Naeem

Core contributor

Ibrahim Alabdulmohsin

Core contributor

Nikhil Parthasarathy

Core contributor

Talfan Evans

Core contributor

Work done while at Google DeepMind

Lucas Beyer

Core contributor

Work done while at Google DeepMind

Ye Xia

Basil Mustafa

Olivier Hénaff

Work done while at Google DeepMind

Jeremiah Harmsen

Andreas Steiner

Xiaohua Zhai

Core contributor

Work done while at Google DeepMind

Project lead

###### Abstract

We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success of the original SigLIP. In this second iteration, we extend the original image-text training objective with several prior, independently developed techniques into a unified recipe—this includes captioning-based pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements on localization and dense prediction tasks. We also train variants which support multiple resolutions and preserve the input’s native aspect ratio. Finally, we train on a more diverse data-mixture that includes de-biasing techniques, leading to much better multilingual understanding and improved fairness. To allow users to trade off inference cost with performance, we release model checkpoints at four sizes: ViT-B (86M), L (303M), So400m (400M), and g (1B).

## 1 Introduction

Contrastive image-text embedding models trained on billion-scale datasets, as pioneered by CLIP [50] and ALIGN [28], have become the mainstream approach for high-level, semantic understanding of visual data. These models enable fine-grained, zero-shot classification rivaling the quality of supervised methods and enable efficient text-to-image and image-to-text retrieval. Furthermore, they lead to excellent vision-language understanding capabilities when combined with Large Language Models (LLMs) to build Vision-Language Models (VLMs).

Developing on the success of CLIP, several improvements have been proposed such as re-captioning images [38], adding image-only self-supervised losses [45, 38], and training with a small decoder for auxiliary tasks such as captioning and localization [67, 32, 62]. At the same time, several groups have released model checkpoints for the open-source community [50, 70, 27, 57, 19]. However, these releases do not include the full breadth of latest improvements into a single model, as they all relatively closely follow CLIP’s original approach. Here, building on the SigLIP training recipe [71], we incorporate several improvements from prior work and release a new family of open models111Model checkpoints are available at 
https://github.com/google-research/big_vision/tree/main/ big_vision/configs/proj/image_text/README_siglip2.md that both excel on CLIP’s core capabilities—–zero-shot classification, retrieval, and feature extraction for
VLMs—and improve areas where vanilla CLIP-style models lag behind, including localization and extracting dense, semantic representations.

In summary, SigLIP 2 models provide the following:

- •

Strong multilingual vision-language encoders: SigLIP 2 shows excellent performance on English-focused vision-language tasks while providing strong results on multilingual benchmarks with a single model. This enables use in a wide range of languages and cultural contexts.

- •

Dense features: We incorporate self-supervised losses as well as a decoder-based loss, which result in better dense features (e.g. for segmentation and depth estimation) and improve localization tasks (such as referring expression comprehension).

- •

Backward compatibility: SigLIP 2 is designed to be backward compatible with SigLIP by relying on the same architecture. This allows existing users to simply swap out the model weights and tokenizer (which is now multilingual) to get improvements on a wide range of tasks.

- •

Native aspect ratio and variable resolution: SigLIP 2 also includes a NaFlex variant, which supports multiple resolutions and preserves the native image aspect ratio. These models have the potential to improve aspect sensitive applications such as document understanding.

- •

Strong small models: SigLIP 2 further optimizes performance of smaller models (B/16 and B/32 models), by using techniques in distillation via active data curation.

In the next section we provide a detailed description of the SigLIP 2 training recipe. Sec. 3 presents evaluations of SigLIP 2 models and baselines across a variety of tasks and benchmarks. Finally, Sec. 4 gives a short overview of related work, and conclusions can be found in Sec. 5.

## 2 Training recipe

We combine the original SigLIP training recipe [71] with decoder-based pretraining [60, 62], in addition to self-distillation and masked prediction as in the DINO line of work [9, 47] (see Fig. 1 for an overview). Pretraining an image encoder with a language decoder for captioning and referring expression comprehension was shown to improve OCR capabilities and localization [62], whereas self-distillation and masked prediction leads to better features for dense prediction tasks, zero-shot classification and retrieval [45, 38]. Rather than combining all these techniques in a single run we follow a staged approach as outlined below to manage the computational and memory overhead compared to SigLIP training.

In addition to training a set of models and adapting each model separately to different resolutions while distorting the aspect ratio, we also train variants which process images while largely preserving their native aspect ratio like NaViT [12] and support different sequence lengths as FlexiViT [6]. We call this variant NaFlex, described in Sec. 2.4.2.

Finally, to improve the quality of the smallest models we fine-tune those with implicit distillation via active sample selection, following the approach from [61].

### 2.1 Architecture, training data, optimizer

For the architecture, we follow SigLIP [71] so that existing users can simply swap out the encoder weights. Specifically, the fixed-resolution variant relies on the standard ViT architecture [15] with learned positional embedding. We use the same architecture for the image and text tower, except for the g-sized vision encoder which is paired with an So400m-sized [1] text encoder. Vision and text representations are pooled using a MAP head (attention pooling) [69]. We set the text length to 64 and use the multilingual Gemma tokenizer [22] with vocabulary size 256k, transforming the text to lower case before tokenization.

We use the WebLI dataset [10] containing 10 billion images and 12 billion alt-texts covering 109 languages. To strike a good balance between quality on English and multilingual vision-language benchmarks we compose the mixture such that 90% of the training image-text pairs is sourced from English web pages, and the remaining 10% from non-English web pages, as recommended in [49]. We further apply the filtering techniques from [2] to mitigate data biases in representation and association with respect to sensitive attributes.

Unless noted otherwise, we use the Adam optimizer with learning rate 10−310^{-3}, decoupled weight decay 10−410^{-4} [37], and gradient clipping to norm 1. We set the batch size to 32k and use a cosine schedule with 20k warmup steps, training for a total of 40B examples. Our models are trained on up to 2048 TPUv5e chips [24] using a fully-sharded data-parallel strategy (FSDP [72]).

### 2.2 Training with Sigmoid loss and decoder

In the first step of pretraining, we combine SigLIP [71] with LocCa [62] by simply combining the two losses with equal weight. Unlike CLIP [50], which relies on a contrastive loss, SigLIP creates binary classification problems by combining every image embedding with every text embedding in the mini-batch and trains the embeddings to classify matching and non-matching pairs via logistic regression (sigmoid loss). We use the original implementation and refer to [71] for details.

For LocCa, we attach a standard transformer decoder with cross-attention to the un-pooled vision encoder representation (before applying the MAP head). The decoder follows the shapes of the text encoder except that we add cross-attention layers and reduce the number of layers by a factor of two. Besides image captioning, LocCa also trains for automatic referring expression prediction and grounded captioning. The former amounts to predicting bounding box coordinates for captions describing specific image regions, whereas the latter involves predicting region-specific captions given bounding box coordinates. Region-caption pairs are automatically annotated by first extracting n-grams from the alt-texts and then applying open-vocabulary detection using the recipe from [41]. Additionally, we use the fixed set of object categories from [10] instead of n-grams. For each example, the decoder is trained to predict all three targets (amounting to three decoder forward-passes). The captioning target is predicted with parallel prediction [60] with probability of 50%, i.e. all caption tokens are predicted in parallel from mask tokens, without causal attention mask. Please refer to [62] for more detail. Finally, to reduce memory consumption due to the large vocabulary, we implement a chunked version of the decoder loss.

For all model sizes, we set the vision encoder patch size to 16 and the image resolution to 256 (resulting in an image representation sequence length of 256). Finally, we note that the decoder only serves for representation learning here and is not part of the model release.

### 2.3 Training with self-distillation and masked prediction

Following SILC [45] and TIPS [38], we augment the training setup described in Sec. 2.2 with local-to-global correspondence learning with self-distillation and masked prediction losses [9, 75, 47] to improve the local semantics of the (un-pooled) feature representation. This representation is typically used for dense prediction tasks like segmentation, depth estimation etc. Concretely, we add two terms to the losses described in Sec. 2.2 as detailed next.

The first term is the local-to-global consistency loss from [45], in which the vision encoder becomes the student network, which gets a partial (local) view of the training image, and is trained to match the teacher’s representation, derived from the full image. This auxiliary matching task is performed in a high-dimensional feature space computed with a separate MLP head. As is common in the literature, the teacher parameters are obtained as an exponential moving average (EMA) of the student parameters over the previous iterations. We rely on a single global (teacher) view and 8 local (student) views and otherwise follow the augmentations, loss and hyper parameters from [45].

The second loss term is the masked prediction objective from [38]. We replace 50% of the embedded image patches in the student network with mask tokens and train the student to match the features of the teacher at masked locations. The loss is then defined identically to the first term (consistency loss), but applied to per-patch features rather than the pooled, image-level representation. Further, both the student and the teacher see the same, global view (up to masking in the student).

We add these losses at 80% of training completion, initializing the teacher with the student parameters and the remaining additional parameters (heads, mask token and corresponding optimizer parameters) randomly. We use the original image for computing the SigLIP and LocCa losses from the previous section and apply the additional losses on additional augmented views. This is done to ensure that data augmentation does not negatively impact the image-text alignment as recommended by [45]. The weights of the first and the second loss terms are set to 1 and 0.25. Further, to balance model quality on global/semantic and dense tasks, we re-weight the two loss terms by another factor of 0.25, 0.5, 1.0, and 0.5 for the B, L, So400m and g, model sizes, respectively.

### 2.4 Adaptation to different resolutions

ImageNet-1k
COCO
Flickr
XM3600

ViT
Res.
Seq.
Model
val
v2
ReaL
ObjNet
10s.

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

B/32
224
49

MetaCLIP [66]

67.7
59.6
–
52.8
–
46.6
–
72.9
–
–
–

256
64

OpenCLIP [27]

72.8
64.8
–
59.6
–
39.9
57.9
64.9
84.8
–
–

SigLIP 2
74.0
66.9
81.4
66.1
66.6
47.2
63.7
75.5
89.3
38.3
49.0

B/16
224
196

CLIP [50]

68.3
61.9
–
55.3
–
33.1
52.4
62.1
81.9
–
–

OpenCLIP [27]

70.2
62.3
–
56.0
–
42.3
59.4
69.8
86.3
–
–

MetaCLIP [66]

72.4
65.1
–
60.0
–
48.9
–
77.1
–
–
–

EVA-CLIP [57]

74.7
67.0
–
62.3
–
42.2
58.7
71.2
85.7
–
–

SigLIP [71]

76.2
69.5
82.8
70.7
69.9
47.2
64.5
77.9
89.6
22.4
29.3

DFN [19]

76.2
68.2
–
63.2
–
51.9
–
77.3
–
–
–

SigLIP 2
78.2
71.4
84.8
73.6
72.1
52.1
68.9
80.7
93.0
40.3
50.7

256
256

SigLIP [71]

76.7
70.1
83.1
71.3
70.3
47.4
65.1
78.3
91.1
22.5
29.9

SigLIP 2
79.1
72.5
85.4
74.5
73.1
53.2
69.7
81.7
94.4
40.7
51.0

384
576

SigLIP [71]

78.6
72.0
84.6
73.8
72.7
49.7
67.5
80.7
92.2
23.3
30.3

SigLIP 2
80.6
73.8
86.2
77.1
74.7
54.6
71.4
83.8
94.9
41.2
51.6

512
1024

SigLIP [71]

79.2
72.9
84.9
74.8
73.3
50.4
67.6
81.6
92.5
23.5
30.5

SigLIP 2
81.2
74.5
86.7
77.8
75.2
55.2
71.2
84.5
95.5
41.4
52.0

L/14
224
256

OpenCLIP [27]

74.0
61.1
–
66.4
–
46.1
62.1
75.0
88.7
–
–

CLIP [50]

75.5
69.0
–
69.9
–
36.5
56.3
65.2
85.2
–
–

MetaCLIP [66]

79.2
72.6
–
74.6
–
55.7
–
83.3
–
–
–

CLIPA-v2 [33]

79.7
72.8
–
71.1
–
46.3
64.1
73.0
89.1
–
–

EVA-CLIP [57]

79.8
72.9
–
75.3
–
47.5
63.7
77.3
89.7
–
–

DFN [19]

82.2
75.7
–
74.8
–
59.6
–
84.7
–
–
–

L/16
256
256

SigLIP [71]

80.5
74.2
85.9
77.9
76.8
51.2
69.6
81.3
92.0
30.9
40.1

SigLIP 2
82.5
76.8
87.3
83.0
78.8
54.7
71.5
84.1
94.5
46.5
56.5

384
576

SigLIP [71]

82.1
75.9
87.1
80.9
78.7
52.8
70.5
82.6
92.9
31.4
39.7

SigLIP 2
83.1
77.4
87.6
84.4
79.5
55.3
71.4
85.0
95.2
47.1
56.3

512
1024
SigLIP 2
83.5
77.8
87.7
84.6
79.6
55.2
72.1
85.3
95.8
47.4
56.7

So/14
224
256

SigLIP [71]

82.2
76.0
87.1
80.5
78.2
50.8
69.0
76.6
90.7
16.0
22.8

SigLIP 2
83.2
77.7
87.8
84.6
79.5
55.1
71.5
84.3
94.6
47.9
57.5

384
729

SigLIP [71]

83.2
77.1
87.5
82.9
79.4
52.0
70.2
80.5
93.5
17.8
26.6

SigLIP 2
84.1
78.7
88.1
86.0
80.4
55.8
71.7
85.7
94.9
48.4
57.5

So/16
256
256

mSigLIP [71]

80.8
74.1
86.1
79.5
77.1
49.4
68.6
80.0
92.1
50.0
62.8

SigLIP 2
83.4
77.8
87.7
84.8
79.7
55.4
71.5
84.4
94.2
48.1
57.5

384
576
SigLIP 2
84.1
78.4
88.1
85.8
80.4
56.0
71.2
85.3
95.9
48.3
57.5

512
1024
SigLIP 2
84.3
79.1
88.1
86.2
80.5
56.0
71.3
85.5
95.4
48.3
57.6

H/14
224
256

MetaCLIP [66]

80.5
74.1
–
76.5
–
57.5
–
85.0
–
–
–

DFN [19]

83.4
77.3
–
76.5
–
63.1
–
86.5
–
–
–

g/16
256
256
SigLIP 2
84.5
79.2
88.3
87.1
82.1
55.7
72.5
85.3
95.3
48.2
58.2

384
576
SigLIP 2
85.0
79.8
88.5
88.0
82.5
56.1
72.8
86.0
95.4
48.6
57.9

#### 2.4.1 Fixed-resolution variant

To obtain fixed-resolution checkpoints at multiple resolutions, we resume the checkpoints (with sequence length 256 and patch size 16) at 95% of training, resize the positional embedding to the target sequences length (and in some cases the patch embedding from patch size 16 to 14 with the pseudoinverse (PI)-resize strategy from [6]), and resume the training at the target resolution with all losses. We opt for this approach as the common strategy of fine-tuning the final checkpoint with smaller learning rate and without weight decay [71] did not lead to good results across all sizes and resolutions.

#### 2.4.2 Variable aspect and resolution (NaFlex)

NaFlex combines ideas from FlexiViT [6], i.e. supporting multiple, predefined sequence lengths with a single ViT model, and NaViT [12], namely processing images at their native aspect ratio. This enables processing different types of images at appropriate resolution, e.g. using a larger resolution to process document images, while at the same time minimizing the impact of aspect ratio distortion on certain inference tasks, e.g. on OCR.

Given a patch size and target sequence length, NaFlex preprocesses the data by first resizing the input image such that the height and width after resizing are multiples of the patch size, while 1) keeping the aspect ratio distortion as small as possible and 2) producing a sequence length of at most the desired target sequence length. The resulting distortion in width and height is at most (patch_size-1)/width and (patch_size-1)/height, respectively, which tends to be small for common resolutions and aspect ratios. Note that NaViT incurs the same type of distortion. After resizing, the image is split into a sequence of patches, and patch coordinates as well as a mask with padding information is added (to handle the case where the actual sequence length is smaller than the target length).

To process different sequence lengths (and aspect ratios) with a ViT, we bilinearly resize (with anti-aliasing) the learned positional embedding to the target, non-square patch grid for the resized input image. We set the length of the learned positional embedding to 256, assuming a 16×1616\times 16 patch grid before resizing. When the sequence length after resizing is smaller than the target sequence length, the attention layers (including the MAP head) are masked to ignore the extra padding tokens.

As for the fixed-resolution, adapted variants, we start from the default checkpoints trained with the setup described in Sec. 2.2, i.e. with non-aspect preserving resize to 256px, resulting in a sequence length of 256.
We take the checkpoint at 90% training completion, then switch to aspect-preserving resizing and uniformly sampling a sequence length from {128,256,576,784,1024}\{128,256,576,784,1024\} per mini-batch. At the same time we stretch the learning rate schedule corresponding to the last 10% by a factor 3.753.75 to ensure that each resolution is trained for sufficiently many examples. For the largest sequence length we further half the batch size and double the number of training steps to avoid out-of-memory errors.

To keep implementation and computation complexity manageable, we do not apply self-distillation and masked prediction from Sec. 2.3.

### 2.5 Distillation via active data curation

To maximize performance of the smallest fixed-resolution models (ViT-B/16 and ViT-B/32), we distill knowledge from a teacher (reference) model during a short fine-tuning stage. We lower the learning rate to 10−510^{-5}, remove weight-decay, and continue training these models for an additional 4B examples using just the sigmoid image-text loss. During this stage, we perform implicit “distillation through data” using the ACID method proposed in [61]. Briefly, at every training step, the teacher model and the current learner model are used to score examples by their “learnability” [42]. These scores are then used to jointly select an optimal batch of size 32k from a larger super-batch [16]. Here, we select data with a filtering ratio of 0.5 (i.e. super-batch size of 64k) to balance gains from curation with training compute. For the B/32 model, we find leveraging a filtering ratio of 0.75 is worth the extra cost.

We note that the authors in [61] suggest that the best performance is achieved with ACED, a method that combines ACID with explicit softmax-distillation (using a second teacher trained on more diverse data). However, here we propose a way to adapt ACID to capture these benefits without the need for explicit distillation, saving significant amounts of compute. Specifically, instead of utilizing two separate teacher models, we take a single strong teacher trained on the diverse data (in this case, the SigLIP 2 So400m model) and fine-tune it for 1B examples on the high-quality curated dataset from [16]. We then use this fine-tuned teacher model in the ACID method, as described above. Because this teacher blends diverse knowledge of concepts from pretraining, with knowledge of what is high-quality (from the curated dataset), the implicit distillation of ACID alone is sufficient to recover the benefits of ACED.

## 3 Experiments and results

### 3.1 Zero-shot classification and retrieval

In Table 1 we report the performance of SigLIP 2 along with baselines on common zero-shot classification (ImageNet [13] ObjectNet [4], ImageNet-v2 [53], ImageNet
ReaL [5]) and image-text retrieval benchmarks. SigLIP 2 performs better than SigLIP and other (open-weight) baselines across the board, despite supporting many languages unlike the baselines (except mSigLIP [71]). Note that DFN [19], which comes closest to SigLIP 2 on these benchmarks, uses a network fine-tuned on ImageNet, COCO, and Flickr (i.e. the main benchmarks in Table 1) as a filter to improve data quality. SigLIP 2’s improvements over the baselines are particularly significant for the B-sized models owing to distillation (Sec. 2.5). Moreover, we observe the common scaling trends as a function of image resolution and model size.

Table 1 and Figure 2 further show the multilingual retrieval performance on Crossmodal-3600 (XM3600) [58] covering 36 languages. SigLIP 2’s recall exceeds that of SigLIP by a large margin, while only lagging slightly behind mSigLIP, which in turn performs substantially worse than SigLIP and SigLIP 2 on English-focused benchmarks.

Segmentation ↑\uparrow

Depth ↓\downarrow

Normals ↓\downarrow

Model
ViT
Res.
PASCAL
ADE20k
NYUv2
NAVI
NYUv2
NAVI

CLIP [50]

L/14
224
74.5
39.0
0.553
0.073
24.3
25.5

OpenCLIP [27]

G/14
224
71.4
39.3
0.541
–
–
–

SigLIP [71]

So/14
224
72.0
37.6
0.576
0.083
25.9
26.0

SigLIP 2
So/14
224
77.1
41.8
0.493
0.067
24.9
25.4

SigLIP [71]

So/14
384
73.8
40.8
0.563
0.069
24.1
25.4

SigLIP 2
So/14
384
78.1
45.4
0.466
0.064
23.0
25.0

#### 3.1.1 NaFlex variant

Fig. 3 compares the fixed-resolution square-aspect ratio (standard) SigLIP 2 with the aspect-preserving NaFlex variant (one checkpoint for all sequence lengths) as a function of the sequence length. In addition to the retrieval benchmarks listed in the previous section, we add a range of OCR/document/screen-focused image-text benchmarks, namely TextCaps [55], HierText [36], SciCap [26] and Screen2Words [63]. The NaFlex variant outperforms the standard variant on the majority of these retrieval benchmarks, in particular for small sequence lengths (and hence resolutions) which tend to suffer more from aspect ratio distortion. On benchmarks predominantly based on natural images, the standard B-sized variant outperforms NaFlex, arguably thanks to the distillation step, whereas for the So400m architecture the two are on par. This is remarkable since the standard variant also benefits from the self-distillation stage (Sec. 2.3).

### 3.2 SigLIP 2 as a vision encoder for VLMs

A popular use case for vision encoders like CLIP and SigLIP is to extract visual representations for VLMs [3, 32, 48, 35, 7, 39, 59]. The common paradigm combines a pretrained vision encoder with a pretrained LLM and does multimodal training on a rich mixture of vision language tasks. To evaluate the performance of SigLIP 2 in this application, we develop a recipe similar to that of PaliGemma 2 [56]. Concretely, we combine SigLIP 2 vision encoders and baselines with the Gemma 2 2B LLM [23] and train the LLM on 50M examples of the Stage 1 training mix from [7, 56] involving captioning, OCR, grounded captioning, visual question answering, detection, and instance segmentation (the annotations for the last 4 tasks are machine-generated, see [7, Sec. 3.2.5] for details). We keep the vision encoder frozen (which has essentially no impact on quality [7, Sec. 5.4]) and reduce training duration to reflect a typical open model use case. The resulting VLM is then fine-tuned on a broad range of downstream tasks with the transfer settings from [56]. To understand the effect of the input resolution we perform experiments at resolution 224 or 256 (for models with patch size 14 and 16, respectively, to extract 256 image tokens) and 384px, but unlike [7, 56] we repeat stage 1 at 384px rather than starting from the 224px variant.

Fig. 4 shows the results after fine-tuning for each dataset. Overall, SigLIP 2 clearly outperforms SigLIP across resolutions and model size. For an L-sized vision encoder, SigLIP 2 also outperforms the recently released AIMv2 model [20]. The data from Fig. 4 can also be found in Table 6.

Model
ViT
A-847
PC-459
A-150
PC-59
VOC-20
VOC-21

CLIP [50]

L/16
10.8
20.4
31.5
62.0
96.6
81.8

OpenCLIP [27]

G/14
13.3
21.4
36.2
61.5
97.1
81.4

SigLIP [71]

L/16
14.0
23.9
37.5
61.6
96.1
81.1

SigLIP 2
L/16
14.3
24.1
38.8
62.4
97.0
82.3

### 3.3 Dense prediction tasks

#### 3.3.1 Semantic segmentation, depth estimation, surface normal estimation

We adopt the evaluation protocol from [38] and probe the frozen SigLIP 2 representation, either with a linear layer or with a DPT decoder [52], on six benchmarks spanning semantic segmentation, monocular depth estimation, and surface normal estimation (see [38, Sec. 4.1] for details on the protocol and hyper parameters). Note, we make one (necessary) change: where the original method concatenates the CLS token to each of the patch feature vectors, we concatenate the output embedding of the MAP head instead, as we use a MAP head instead of a CLS token. The results in Table 2 indicate that SigLIP 2 outperforms several previous open, CLIP-style vision encoders, including SigLIP, often by a significant margin.

#### 3.3.2 Open-vocabulary segmentation

Open-vocabulary segmentation aims to develop models that can segment any novel classes beyond a fixed training vocabulary. Here, we evaluate SigLIP 2’s performance on this task. We use Cat-Seg [11] as a framework and compare performance across different models as proposed in [45]. We train Cat-Seg on COCO-Stuff-164k [8] with 172 classes and then test it on various representative datasets with different vocabularies: ADE20k [74, 73] with 847 or 150 classes (A-847/A-150), Pascal Context
(PC-459/PC-59) [43], and Pascal VOC (VOC-20/VOC-21) [17].
The results can be found in Table 3. We observe that the SigLIP 2 at L/16 improves on SigLIP and even surpasses the much bigger OpenCLIP G/14 model [27].

### 3.4 Localization tasks

#### 3.4.1 Referring expression comprehension

To probe the referring expression comprehension capabilities of SigLIP 2 on different RefCOCO variants [29, 68] we apply the evaluation protocol from [62]. We attach a 6-layer transformer decoder to the un-pooled, frozen vision encoder representation via cross-attention and train it from scratch on a mix of all RefCOCO variants (see [62] for details). The results in Table 5 show that SigLIP 2 outperforms SigLIP as well as CLIP and pretraining via image captioning (Cap) by a large margin, across resolutions and model sizes. This can be attributed to the decoder-based pretraining, as described in Sec. 2.2. SigLIP 2 is only outperformed LocCa, which we hypothesize might be due to the fact that SigLIP 2 is pretrained on multilingual data. LocCa, on the other hand, is trained on text only from English web sites. Finally, note that we expect significant improvements when using the decoder from pretraining as observed for LocCa.

#### 3.4.2 Open-vocabulary detection

OWL-ViT [40] is a popular method to adapt CLIP-style vision-language models to open-vocabulary detection. Here, we apply this approach to SigLIP and SigLIP 2 models, closely following the data and optimizer configuration from [40]. The results in Table 4 show that SigLIP 2 achieves better performance than SigLIP on the two popular benchmarks COCO [34] and LVIS [25]. The relative improvement is most pronounced for the LVIS rare categories. Further, the results here are better than those in [40] which is likely because [40] used CLIP rather than SigLIP.

### 3.5 Cultural diversity and fairness

Besides the improvement in model quality in SigLIP 2 compared to its predecessor, SigLIP 2 is also more inclusive in two aspects. First, we follow the recommendations of [49] and utilize a training mixture comprising both English and multilingual data to enhance cultural diversity. Second, to address potential societal biases in the training data, we integrate the data de-biasing techniques from [2]. These techniques are applied to mitigate biases in both first-order statistics, such as disparities in gender representation, and second-order statistics, such as biased associations between gender and occupation. Next, we present the evaluation results.

ViT
Model
COCO (AP)
LVIS (AP)
LVIS (APr)

B/16
SigLIP
42.2
33.0
31.0

SigLIP 2
42.8
34.4
32.7

So/14
SigLIP
44.3
39.5
40.9

SigLIP 2
45.2
40.5
42.3

RefCOCO
RefCOCO+
RefCOCOg

ViT
Seq.
Model
val
testA
testB
val
testA
testB
val-u
test-u

B
256

SigLIP [71]

64.05
70.10
57.89
55.77
63.57
47.51
59.06
60.33

SigLIP 2
83.76
86.21
79.57
74.26
79.85
65.83
77.25
77.83

576

SigLIP [71]

67.17
72.94
60.94
59.09
67.26
50.22
61.98
62.64

SigLIP 2
85.18
87.92
80.53
76.08
82.17
67.10
79.08
79.60

L
256

Cap [60]

60.64
65.47
56.17
52.56
58.32
45.99
56.75
57.99

CapPa [60]

64.17
69.90
58.25
56.14
63.68
48.18
58.90
59.91

CLIP [50]

65.21
71.28
58.17
57.53
66.44
47.77
59.32
60.24

SigLIP [71]

67.33
72.40
61.21
59.57
67.09
51.08
61.89
62.90

SigLIP 2
86.04
89.02
81.85
77.29
83.28
70.16
80.11
80.78

LocCa [62]

88.34
91.20
85.10
79.39
85.13
72.61
81.69
82.64

576

SigLIP [71]

70.76
76.32
63.79
63.38
71.48
54.65
64.73
65.74

SigLIP 2
87.28
90.29
82.85
79.00
85.00
70.92
81.84
82.15

So
256

SigLIP [71]

64.68
71.23
58.40
57.43
66.06
49.38
59.66
60.88

SigLIP 2
86.42
89.41
82.48
77.81
84.36
70.67
80.83
81.27

729

SigLIP [71]

67.66
74.12
62.36
60.74
69.73
52.12
62.61
63.24

SigLIP 2
87.88
91.13
83.59
80.06
86.30
72.66
82.68
83.63

g
256
SigLIP 2
87.31
90.24
83.25
79.25
85.23
71.60
81.48
82.14

576
SigLIP 2
88.45
91.53
84.95
80.44
87.09
73.53
83.12
84.14

##### Cultural Diversity

To evaluate for cultural diversity, we report the zero-shot classification accuracy results using Dollar Street [54], GeoDE [51], and Google Landmarks Dataset v2 (GLDv2) [65]. We also include 10-shot geolocalization using Dollar Street and GeoDE, as proposed in [49]. For zero-shot evaluation on Dollar Street, we implement the methodology outlined in [54], mapping 96 topics within the dataset to corresponding ImageNet classes. This process results in a subset of 21K images for our analysis.

Fig. 5 shows a set of representative results (full results are shown in Appendix C). We observe an improvement in these metrics in SigLIP 2 compared to SigLIP for the same model size and resolution, and the improvements are particularly significant in geolocalization tasks. For instance, 10-shot geolocalization accuracy in GeoDE (region) improves from 36.2% for SigLIP L/16 at 256px to 44.4% in SigLIP 2. Similarly, 0-shot accuracy on Dollar Street improves from 52.1% to 55.2% in the same models.

##### Fairness

In terms of fairness, we report two metrics. The first is “representation bias,” as defined in [2], which measures the tendency in the model to associate a random object (such as cars) with a particular gender group. As shown in Fig. 6, SigLIP 2 is significantly better than SigLIP. For instance, while SigLIP L/16 at 256px has a representation bias of about 35.5%—meaning it prefers to associate random images with “men” over “women” more than 85.5% of the time—SigLIP 2 of the same size and resolution has a representation bias of 7.3% only. In addition, larger models tend to exhibit less representation bias than smaller models, in agreement with the earlier findings in [2].

We also investigate the Dollar Street 0-shot results by income level and the GeoDE results by geographic region as [49]. However, in this context we only observe very minor benefits, or no benefits when comparing SigLIP and SigLIP 2 models of matching size and resolution (some results shown in Table 9).

## 4 Related work

Contrastive pretraining as popularized by CLIP [50] and ALIGN [28] has become the dominant approach for learning high-level, semantic, visual representations that perform well on classification and retrieval, as vision encoders for VLMs [3, 32, 48, 35, 7, 39, 59] and open-vocabulary tasks including detection [40, 30, 41] and segmentation [14, 11]. Besides the original CLIP release, several projects have released open-weight contrastive models [27, 57, 71, 33, 19, 66]. At a high level, these works follow training methods that are relatively close to the original CLIP method, mainly [71] proposing modified loss functions and [19, 66] targeting data quality and filtering.

More generally, a large number of modifications and improvements to contrastive training have been proposed in the literature. [21, 19, 66, 16, 61] study filtering techniques to improve data quality. With a similar motivation, [18, 46, 31, 38] re-caption training images with VLMs to improve the caption quality and hence the quality of the training signal. Another promising area has been to modify or augment the loss function. [44, 45, 38] combine CLIP with self-supervised losses. Another popular approach is to add a language decoder to train with captioning as an auxiliary task [67, 32]. Captioning as a standalone representation learning task has attracted less attention, but can produce visual representations competitive with contrastive training [64, 60, 62, 20].

## 5 Conclusion

In this work, we introduced SigLIP 2, a family of open-weight multilingual vision-language encoders that builds on the success of SigLIP. By incorporating a combination of techniques such as decoder-based pretraining, self-supervised losses, and active data curation, SigLIP 2 achieves significant improvements in zero-shot classification, transfer performance as a vision encoder in VLMs, and in localization and dense prediction tasks. Furthermore, thanks to training on multilingual data and applying de-biasing filters, SigLIP 2 attains more balanced quality across culturally diverse data. Finally, the NaFlex variant enables the model to support multiple resolutions with a single model checkpoint, while preserving the native image aspect ratio. We hope that our SigLIP 2 release will enable many exciting applications within the open-source community.

##### Acknowledgments

We would like to thank Josip Djolonga, Neil Houlsby, Andre Araujo, Kevis Maninis, and Phoebe Kirk for discussions and feedback on this project. We also thank Joan Puigcerver, André Susano Pinto, and Alex Bewley for infrastructure contributions to the big_vision code base, which were helpful for this project.

\nobibliography

*

## References

- Alabdulmohsin et al. [2023]

I. Alabdulmohsin, X. Zhai, A. Kolesnikov, and L. Beyer.

Getting vit in shape: Scaling laws for compute-optimal model design.

In NeurIPS, 2023.

- Alabdulmohsin et al. [2024]

I. Alabdulmohsin, X. Wang, A. P. Steiner, P. Goyal, A. D’Amour, and X. Zhai.

Clip the bias: How useful is balancing data in multimodal learning?

In ICLR, 2024.

- Bai et al. [2023]

J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and
J. Zhou.

Qwen-VL: A versatile vision-language model for understanding,
localization, text reading, and beyond.

arXiv:2308.12966, 2023.

- Barbu et al. [2019]

A. Barbu, D. Mayo, J. Alverio, W. Luo, C. Wang, D. Gutfreund, J. Tenenbaum, and
B. Katz.

Objectnet: A large-scale bias-controlled dataset for pushing the
limits of object recognition models.

NeurIPS, 2019.

- Beyer et al. [2020]

L. Beyer, O. J. Hénaff, A. Kolesnikov, X. Zhai, and A. v. d. Oord.

Are we done with imagenet?

arXiv:2006.07159, 2020.

- Beyer et al. [2023]

L. Beyer, P. Izmailov, A. Kolesnikov, M. Caron, S. Kornblith, X. Zhai,
M. Minderer, M. Tschannen, I. Alabdulmohsin, and F. Pavetic.

Flexivit: One model for all patch sizes.

In CVPR, 2023.

- Beyer et al. [2024]

L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann,
I. Alabdulmohsin, M. Tschannen, E. Bugliarello, T. Unterthiner, D. Keysers,
S. Koppula, F. Liu, A. Grycner, A. Gritsenko, N. Houlsby, M. Kumar, K. Rong,
J. Eisenschlos, R. Kabra, M. Bauer, M. Bošnjak, X. Chen, M. Minderer,
P. Voigtlaender, I. Bica, I. Balazevic, J. Puigcerver, P. Papalampidi,
O. Henaff, X. Xiong, R. Soricut, J. Harmsen, and X. Zhai.

PaliGemma: A versatile 3B VLM for transfer.

arXiv:2407.07726, 2024.

- Caesar et al. [2018]

H. Caesar, J. Uijlings, and V. Ferrari.

Coco-stuff: Thing and stuff classes in context.

In CVPR, 2018.

- Caron et al. [2021]

M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and
A. Joulin.

Emerging properties in self-supervised vision transformers.

In CVPR, pages 9650–9660, 2021.

- Chen et al. [2022]

X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz,
S. Goodman, A. Grycner, B. Mustafa, L. Beyer, A. Kolesnikov, J. Puigcerver,
N. Ding, K. Rong, H. Akbari, G. Mishra, L. Xue, A. Thapliyal, J. Bradbury,
W. Kuo, M. Seyedhosseini, C. Jia, B. K. Ayan, C. Riquelme, A. Steiner,
A. Angelova, X. Zhai, N. Houlsby, and R. Soricut.

PaLI: A jointly-scaled multilingual language-image model.

arXiv:2209.06794, 2022.

- Cho et al. [2024]

S. Cho, H. Shin, S. Hong, A. Arnab, P. H. Seo, and S. Kim.

Cat-seg: Cost aggregation for open-vocabulary semantic segmentation.

In CVPR, pages 4113–4123, 2024.

- Dehghani et al. [2024]

M. Dehghani, B. Mustafa, J. Djolonga, J. Heek, M. Minderer, M. Caron,
A. Steiner, J. Puigcerver, R. Geirhos, I. M. Alabdulmohsin, et al.

Patch n’pack: NaViT, a vision transformer for any aspect ratio
and resolution.

NeurIPS, 2024.

- Deng et al. [2009]

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei.

Imagenet: A large-scale hierarchical image database.

In CVPR, pages 248–255, 2009.

- Ding et al. [2022]

J. Ding, N. Xue, G.-S. Xia, and D. Dai.

Decoupling zero-shot semantic segmentation.

In CVPR, pages 11583–11592, 2022.

- Dosovitskiy et al. [2021]

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit,
and N. Houlsby.

An image is worth 16x16 words: Transformers for image recognition at
scale.

In ICLR, 2021.

- Evans et al. [2024]

T. Evans, N. Parthasarathy, H. Merzic, and O. J. Henaff.

Data curation via joint example selection further accelerates
multimodal learning.

In NeurIPS Datasets and Benchmarks Track, 2024.

- Everingham et al. [2010]

M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman.

The pascal visual object classes (voc) challenge.

IJCV, 2010.

- Fan et al. [2023]

L. Fan, D. Krishnan, P. Isola, D. Katabi, and Y. Tian.

Improving clip training with language rewrites.

NeurIPS, pages 35544–35575, 2023.

- Fang et al. [2024]

A. Fang, A. M. Jose, A. Jain, L. Schmidt, A. T. Toshev, and V. Shankar.

Data filtering networks.

In ICLR, 2024.

- Fini et al. [2024]

E. Fini, M. Shukor, X. Li, P. Dufter, M. Klein, D. Haldimann, S. Aitharaju,
V. G. T. da Costa, L. Béthune, Z. Gan, A. T. Toshev, M. Eichner,
M. Nabi, Y. Yang, J. M. Susskind, and A. El-Nouby.

Multimodal autoregressive pre-training of large vision encoders.

arXiv:2411.14402, 2024.

- Gadre et al. [2024]

S. Y. Gadre, G. Ilharco, A. Fang, J. Hayase, G. Smyrnis, T. Nguyen, R. Marten,
M. Wortsman, D. Ghosh, J. Zhang, et al.

Datacomp: In search of the next generation of multimodal datasets.

NeurIPS, 36, 2024.

- Gemma Team [2024a]

Gemma Team.

Gemma: Open models based on gemini research and technology.

arXiv:2403.08295, 2024a.

- Gemma Team [2024b]

Gemma Team.

Gemma 2: Improving open language models at a practical size.

arXiv:2408.00118, 2024b.

- Google Cloud [20xx]

Google Cloud.

Introduction to Cloud TPU.

https://cloud.google.com/tpu/docs/intro-to-tpu, 20xx.

Accessed: 2024-07-04.

- Gupta et al. [2019]

A. Gupta, P. Dollar, and R. Girshick.

Lvis: A dataset for large vocabulary instance segmentation.

In CVPR, pages 5356–5364, 2019.

- Hsu et al. [2021]

T.-Y. Hsu, C. L. Giles, and T.-H. Huang.

Scicap: Generating captions for scientific figures.

arXiv:2110.11624, 2021.

- Ilharco et al. [2021]

G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave,
V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and
L. Schmidt.

OpenCLIP, 2021.

- Jia et al. [2021]

C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li,
and T. Duerig.

Scaling up visual and vision-language representation learning with
noisy text supervision.

In ICML, 2021.

- Kazemzadeh et al. [2014]

S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg.

ReferItGame: Referring to objects in photographs of natural
scenes.

In EMNLP, Oct. 2014.

- Kuo et al. [2023]

W. Kuo, Y. Cui, X. Gu, A. Piergiovanni, and A. Angelova.

Open-vocabulary object detection upon frozen vision and language
models.

In ICLR, 2023.

- Lai et al. [2024]

Z. Lai, H. Zhang, B. Zhang, W. Wu, H. Bai, A. Timofeev, X. Du, Z. Gan, J. Shan,
C.-N. Chuah, Y. Yang, and M. Cao.

VeCLIP: Improving clip training via visual-enriched captions.

arXiv:2310.07699, 2024.

- Li et al. [2023a]

J. Li, D. Li, S. Savarese, and S. C. H. Hoi.

BLIP-2: bootstrapping language-image pre-training with frozen image
encoders and large language models.

In ICML, 2023a.

- Li et al. [2023b]

X. Li, Z. Wang, and C. Xie.

Clipa-v2: Scaling clip training with 81.1% zero-shot imagenet
accuracy within a $10,000 budget; an extra $4,000 unlocks 81.8% accuracy.

arXiv:2306.15658, 2023b.

- Lin et al. [2014]

T. Lin, M. Maire, S. J. Belongie, L. D. Bourdev, R. B. Girshick, J. Hays,
P. Perona, D. Ramanan, P. Doll’a r, and C. L. Zitnick.

Microsoft COCO: common objects in context.

arXiv:1405.0312, 2014.

- Liu et al. [2023]

H. Liu, C. Li, Q. Wu, and Y. J. Lee.

Visual instruction tuning.

In NeurIPS, 2023.

- Long et al. [2023]

S. Long, S. Qin, D. Panteleev, A. Bissacco, Y. Fujii, and M. Raptis.

ICDAR 2023 competition on hierarchical text detection and
recognition.

In ICDAR, 2023.

- Loshchilov et al. [2017]

I. Loshchilov, F. Hutter, et al.

Fixing weight decay regularization in adam.

arXiv preprint arXiv:1711.05101, 5, 2017.

- Maninis et al. [2025]

K.-K. Maninis, K. Chen, S. Ghosh, A. Karpur, K. Chen, Y. Xia, B. Cao, D. Salz,
G. Han, J. Dlabal, et al.

TIPS: Text-image pretraining with spatial awareness.

In ICLR, 2025.

- McKinzie et al. [2024]

B. McKinzie, Z. Gan, J. Fauconnier, S. Dodge, B. Zhang, P. Dufter, D. Shah,
X. Du, F. Peng, F. Weers, A. Belyi, H. Zhang, K. Singh, D. Kang, A. Jain,
H. Hè, M. Schwarzer, T. Gunter, X. Kong, A. Zhang, J. Wang, C. Wang,
N. Du, T. Lei, S. Wiseman, G. Yin, M. Lee, Z. Wang, R. Pang, P. Grasch,
A. Toshev, and Y. Yang.

MM1: methods, analysis & insights from multimodal LLM
pre-training.

arXiv:2403.09611, 2024.

- Minderer et al. [2022]

M. Minderer, A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn,
A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, et al.

Simple open-vocabulary object detection.

In ECCV, pages 728–755, 2022.

- Minderer et al. [2023]

M. Minderer, A. A. Gritsenko, and N. Houlsby.

Scaling open-vocabulary object detection.

In NeurIPS, 2023.

- Mindermann et al. [2022]

S. Mindermann, J. M. Brauner, M. T. Razzak, M. Sharma, A. Kirsch, W. Xu,
B. Höltgen, A. N. Gomez, A. Morisot, S. Farquhar, et al.

Prioritized training on points that are learnable, worth learning,
and not yet learnt.

In ICML, pages 15630–15649, 2022.

- Mottaghi et al. [2014]

R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R. Urtasun, and
A. Yuille.

The role of context for object detection and semantic segmentation in
the wild.

In CVPR, 2014.

- Mu et al. [2022]

N. Mu, A. Kirillov, D. Wagner, and S. Xie.

SLIP: Self-supervision meets language-image pre-training.

In ECCV, pages 529–544, 2022.

- Naeem et al. [2024]

M. F. Naeem, Y. Xian, X. Zhai, L. Hoyer, L. Van Gool, and F. Tombari.

SILC: Improving vision language pretraining with self-distillation.

In ECCV, pages 38–55, 2024.

- Nguyen et al. [2024]

T. Nguyen, S. Y. Gadre, G. Ilharco, S. Oh, and L. Schmidt.

Improving multimodal datasets with image captioning.

NeurIPS, 36, 2024.

- Oquab et al. [2024]

M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov,
P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al.

Dinov2: Learning robust visual features without supervision.

TMLR, 2024.

- Peng et al. [2023]

Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei.

Kosmos-2: Grounding multimodal large language models to the world.

arXiv:2306.14824, 2023.

- Pouget et al. [2024]

A. Pouget, L. Beyer, E. Bugliarello, X. Wang, A. P. Steiner, X. Zhai, and
I. Alabdulmohsin.

No filter: Cultural and socioeconomic diversityin contrastive
vision-language models.

arXiv:2405.13777, 2024.

- Radford et al. [2021]

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry,
A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever.

Learning transferable visual models from natural language
supervision.

In ICML, 2021.

- Ramaswamy et al. [2024]

V. V. Ramaswamy, S. Y. Lin, D. Zhao, A. Adcock, L. van der Maaten,
D. Ghadiyaram, and O. Russakovsky.

Geode: a geographically diverse evaluation dataset for object
recognition.

NeurIPS, 36, 2024.

- Ranftl et al. [2021]

R. Ranftl, A. Bochkovskiy, and V. Koltun.

Vision transformers for dense prediction.

In CVPR, pages 12179–12188, 2021.

- Recht et al. [2019]

B. Recht, R. Roelofs, L. Schmidt, and V. Shankar.

Do imagenet classifiers generalize to imagenet?

In ICML, pages 5389–5400, 2019.

- Rojas et al. [2022]

W. A. G. Rojas, S. Diamos, K. R. Kini, D. Kanter, V. J. Reddi, and C. Coleman.

The dollar street dataset: Images representing the geographic and
socioeconomic diversity of the world.

In NeurIPS Datasets and Benchmarks Track, 2022.

- Sidorov et al. [2020]

O. Sidorov, R. Hu, M. Rohrbach, and A. Singh.

TextCaps: A dataset for image captioning with reading
comprehension.

In ECCV, 2020.

- Steiner et al. [2024]

A. Steiner, A. S. Pinto, M. Tschannen, D. Keysers, X. Wang, Y. Bitton,
A. Gritsenko, M. Minderer, A. Sherbondy, S. Long, et al.

Paligemma 2: A family of versatile vlms for transfer.

arXiv:2412.03555, 2024.

- Sun et al. [2023]

Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao.

EVA-CLIP: Improved training techniques for clip at scale.

arXiv:2303.15389, 2023.

- Thapliyal et al. [2022]

A. V. Thapliyal, J. Pont Tuset, X. Chen, and R. Soricut.

Crossmodal-3600: A massively multilingual multimodal evaluation
dataset.

In EMNLP, 2022.

- Tong et al. [2024]

S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang,
A. Iyer, X. Pan, A. Wang, R. Fergus, Y. LeCun, and S. Xie.

Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal
LLMs.

arXiv:2406.16860, 2024.

- Tschannen et al. [2023]

M. Tschannen, M. Kumar, A. Steiner, X. Zhai, N. Houlsby, and L. Beyer.

Image captioners are scalable vision learners too.

In NeurIPS, 2023.

- Udandarao et al. [2024]

V. Udandarao, N. Parthasarathy, M. F. Naeem, T. Evans, S. Albanie, F. Tombari,
Y. Xian, A. Tonioni, and O. J. Hénaff.

Active data curation effectively distills large-scale multimodal
models.

arXiv:2411.18674, 2024.

- Wan et al. [2024]

B. Wan, M. Tschannen, Y. Xian, F. Pavetic, I. Alabdulmohsin, X. Wang, A. S.
Pinto, A. Steiner, L. Beyer, and X. Zhai.

LocCa: Visual pretraining with location-aware captioners.

In NeurIPS, 2024.

- Wang et al. [2021]

B. Wang, G. Li, X. Zhou, Z. Chen, T. Grossman, and Y. Li.

Screen2words: Automatic mobile ui summarization with multimodal
learning.

In Symposium on User Interface Software and Technology, 2021.

- Wang et al. [2022]

Z. Wang, J. Yu, A. W. Yu, Z. Dai, Y. Tsvetkov, and Y. Cao.

SimVLM: Simple visual language model pretraining with weak
supervision.

In ICLR, 2022.

- Weyand et al. [2020]

T. Weyand, A. Araujo, B. Cao, and J. Sim.

Google landmarks dataset v2-a large-scale benchmark for
instance-level recognition and retrieval.

In CVPR, pages 2575–2584, 2020.

- Xu et al. [2024]

H. Xu, S. Xie, X. Tan, P.-Y. Huang, R. Howes, V. Sharma, S.-W. Li, G. Ghosh,
L. Zettlemoyer, and C. Feichtenhofer.

Demystifying clip data.

In ICLR, 2024.

- Yu et al. [2022]

J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu.

CoCa: Contrastive captioners are image-text foundation models.

TMLR, 2022.

- Yu et al. [2016]

L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg.

Modeling context in referring expressions.

In ECCV, pages 69–85, 2016.

- Zhai et al. [2022a]

X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer.

Scaling vision transformers.

CVPR, 2022a.

- Zhai et al. [2022b]

X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and
L. Beyer.

Lit: Zero-shot transfer with locked-image text tuning.

In CVPR, 2022b.

- Zhai et al. [2023]

X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer.

Sigmoid loss for language image pre-training.

In ICCV, 2023.

- Zhao et al. [2023]

Y. Zhao, A. Gu, R. Varma, L. Luo, C. Huang, M. Xu, L. Wright, H. Shojanazeri,
M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, P. Damania, B. Nguyen,
G. Chauhan, Y. Hao, A. Mathews, and S. Li.

Pytorch FSDP: experiences on scaling fully sharded data parallel.

VLDB, 2023.

- Zhou et al. [2017]

B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba.

Scene parsing through ade20k dataset.

In CVPR, 2017.

- Zhou et al. [2019]

B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba.

Semantic understanding of scenes through the ade20k dataset.

IJCV, 2019.

- Zhou et al. [2022]

J. Zhou, C. Wei, H. Wang, W. Shen, C. Xie, A. Yuille, and T. Kong.

Image BERT pre-training with online tokenizer.

In ICLR, 2022.

## Appendix

## Appendix A Full PaliGemma results

Large 224/256px
So400m/14 224px
So400m 384px

SigLIP
AIMv2
SigLIP2
SigLIP
SigLIP2
SigLIP
SigLIP2

AI2D

-075.2

-073.2

-075.9

-075.3

-074.8

-076.7

-078.3

AOKVQA-DA (val)

-060.3

-062.3

-061.7

-062.0

-062.8

-064.9

-064.7

AOKVQA-MC (val)

-078.3

-078.4

-077.6

-079.0

-080.5

-082.5

-083.1

COCO-35L (avg34)

-109.9

-111.4

-112.2

-111.9

-113.2

-113.6

-114.8

COCO-35L (en)

-136.7

-138.3

-139.4

-139.0

-139.4

-140.3

-141.1

COCOcap

-138.6

-139.9

-141.3

-141.4

-142.7

-142.2

-143.8

CountBenchQA

-075.3

-083.1

-082.2

-078.2

-084.7

-080.8

-083.9

DocVQA (val)

-033.0

-032.3

-035.4

-034.3

-035.9

-062.7

-065.9

GQA

-065.2

-065.6

-066.1

-065.5

-065.7

-067.0

-067.8

InfoVQA (val)

-025.3

-025.1

-026.3

-025.1

-026.0

-034.7

-037.1

NLVR2

-090.7

-091.3

-091.1

-091.0

-091.4

-091.7

-091.8

NoCaps

-117.7

-121.7

-120.3

-120.1

-120.9

-120.8

-121.9

OCR-VQA

-070.6

-071.8

-072.5

-071.3

-072.7

-074.4

-075.2

OKVQA

-062.4

-062.7

-063.3

-063.1

-063.4

-063.7

-064.5

RefCOCO (testA)

-071.0

-071.9

-074.3

-072.4

-074.5

-076.6

-078.2

RefCOCO (testB)

-066.0

-067.8

-070.3

-067.5

-070.5

-071.4

-074.5

RefCOCO (val)

-068.7

-069.5

-072.4

-069.9

-072.5

-074.3

-076.1

RefCOCO+ (testA)

-067.5

-069.0

-070.8

-069.0

-071.4

-074.1

-075.9

RefCOCO+ (testB)

-059.6

-061.5

-063.3

-060.8

-063.3

-065.4

-067.6

RefCOCO+ (val)

-063.6

-065.1

-067.6

-064.9

-067.8

-070.0

-072.0

RefCOCOg (test)

-063.9

-065.4

-067.5

-064.7

-067.9

-069.9

-072.1

RefCOCOg (val)

-063.3

-064.3

-066.8

-064.5

-067.3

-069.5

-071.7

ST-VQA (val)

-054.0

-053.9

-059.8

-056.7

-060.1

-075.0

-077.3

SciCap

-161.1

-156.4

-165.5

-162.3

-161.8

-177.2

-179.3

ScienceQA

-096.1

-096.1

-096.2

-095.4

-096.3

-096.2

-096.1

Screen2Words

-108.7

-106.9

-114.3

-111.3

-110.6

-115.3

-116.1

TallyQA (complex)

-067.6

-069.4

-069.3

-068.4

-070.0

-071.0

-072.5

TallyQA (simple)

-079.9

-081.0

-082.0

-080.4

-082.2

-083.5

-085.4

TextCaps

-116.5

-116.8

-126.1

-121.7

-123.8

-145.0

-150.9

TextVQA (val)

-051.9

-053.9

-057.3

-054.5

-059.4

-069.7

-074.0

VQAv2 (minival)

-081.5

-082.1

-082.1

-081.9

-082.8

-084.3

-085.2

VizWizVQA (val)

-074.4

-074.4

-076.0

-075.5

-076.0

-076.8

-077.6

WidgetCap

-132.8

-133.0

-139.1

-134.4

-142.0

-147.0

-151.1

XM3600 (avg35)

-039.0

-039.6

-039.7

-039.8

-040.1

-040.8

-041.1

XM3600 (en)

-077.7

-078.0

-079.1

-077.8

-079.2

-080.0

-081.0

## Appendix B Full NaFlex results

ImageNet-1k
COCO R@1
TC R@1
HT R@1
SC R@1
S2W R@1

ViT
Seq.
Model
val
v2
ReaL
ObjNet

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

T→\rightarrowI

I→\rightarrowT

B/16
64
SigLIP 2 (NaF.)
71.2
63.2
78.3
62.1
43.6
60.4
30.4
57.5
3.4
6.4
5.2
4.0
6.4
11.0

144
SigLIP 2 (NaF.)
76.2
69.4
82.9
70.2
49.0
65.7
36.5
65.8
5.7
10.3
13.5
11.8
13.9
25.4

196
SigLIP 2
78.2
71.4
84.8
73.6
52.1
68.9
38.9
68.0
5.5
9.5
13.3
10.9
10.8
18.7

256
SigLIP 2
79.1
72.5
85.4
74.5
53.2
69.7
40.5
69.4
6.1
9.8
17.1
14.2
12.9
22.9

SigLIP 2 (NaF.)
78.5
71.9
84.6
74.6
51.1
67.3
39.5
69.0
7.4
12.9
19.7
17.1
14.8
26.6

576
SigLIP 2
80.6
73.8
86.2
77.1
54.6
71.4
43.6
73.0
7.5
12.0
23.3
19.4
14.1
24.8

SigLIP 2 (NaF.)
80.0
73.1
85.6
76.4
52.5
69.1
41.6
71.8
8.7
14.1
24.3
21.0
15.3
26.7

676
SigLIP 2 (NaF.)
80.1
73.5
85.7
76.5
52.9
68.6
41.8
73.0
8.8
13.9
24.3
21.4
15.2
26.2

784
SigLIP 2 (NaF.)
80.2
73.5
85.9
76.9
53.1
68.8
42.5
72.9
8.7
14.0
24.8
21.5
15.2
26.4

900
SigLIP 2 (NaF.)
80.3
73.6
85.9
76.6
52.9
69.2
42.3
72.6
8.6
15.0
24.8
21.6
15.0
25.8

1024
SigLIP 2
81.2
74.5
86.7
77.8
55.2
71.2
44.7
74.7
8.1
14.6
25.2
20.7
14.5
25.3

SigLIP 2 (NaF.)
80.4
73.5
85.9
76.6
52.9
68.9
42.5
73.2
9.1
14.4
25.1
21.5
14.9
26.4

So/16
64
SigLIP 2 (NaF.)
78.5
71.0
84.2
73.8
49.6
67.4
37.0
65.5
5.6
10.3
11.8
10.9
12.1
21.4

144
SigLIP 2 (NaF.)
81.8
75.2
86.7
79.8
53.4
70.4
42.8
71.0
8.0
14.6
22.2
23.1
17.1
29.0

256
SigLIP 2
83.4
77.8
87.7
84.8
55.4
71.5
44.8
72.9
7.9
13.9
29.7
28.8
17.4
28.7

SigLIP 2 (NaF.)
83.5
77.5
87.7
83.8
55.1
71.2
44.9
73.6
9.2
15.7
29.8
29.2
17.5
29.2

576
SigLIP 2
84.1
78.4
88.1
85.8
56.0
71.2
47.0
74.9
9.7
16.3
34.5
32.4
17.8
28.0

SigLIP 2 (NaF.)
84.1
78.6
88.0
85.7
55.9
71.4
46.5
75.1
11.3
18.4
32.9
32.0
17.7
28.8

676
SigLIP 2 (NaF.)
84.2
78.5
88.0
85.7
55.8
71.7
46.9
74.9
11.3
18.5
33.3
32.2
17.7
29.8

784
SigLIP 2 (NaF.)
84.3
78.6
88.0
85.9
55.9
71.3
46.7
74.9
11.5
18.5
33.0
32.3
17.6
29.5

900
SigLIP 2 (NaF.)
84.3
78.6
88.1
85.8
55.8
71.2
46.8
75.4
11.7
18.5
32.9
32.5
17.7
29.4

1024
SigLIP 2
84.3
79.1
88.1
86.2
56.0
71.3
47.3
76.0
10.3
18.3
35.9
33.5
17.9
28.1

SigLIP 2 (NaF.)
84.4
78.8
88.1
85.8
55.8
71.0
46.9
74.9
11.7
18.4
32.6
32.4
17.8
29.4

## Appendix C Full cultural diversity and fairness results

10-shot
0-shot

ViT
Res.
Model
Dollar Street
GeoDE (country)
GeoDE (region)
Dollar Street
GLDv2
GeoDE

B/32
256
SigLIP 2
13.1
13.9
29.3
50.5
44.7
90.6

B/16
224
SigLIP
13.8
12.7
27.3
50.1
48.5
92.4

SigLIP 2
16.2
20.0
34.9
53.4
50.8
92.9

256
SigLIP
15.0
13.3
29.3
50.3
47.7
92.8

SigLIP 2
17.7
22.7
36.3
54.2
52.5
93.3

384
SigLIP
16.1
16.4
31.5
51.5
51.9
93.6

SigLIP 2
19.8
25.6
41.4
54.8
55.2
93.9

512
SigLIP
16.6
17.7
32.3
51.3
53.1
94.1

SigLIP 2
21.7
28.2
43.1
54.9
57.6
94.2

L/16
256
SigLIP
18.8
22.1
36.2
52.1
56.7
93.6

SigLIP 2
26.8
34.5
44.4
55.2
64.5
94.9

384
SigLIP
22.8
26.0
41.7
52.9
60.5
94.3

SigLIP 2
30.4
39.3
48.0
55.4
66.1
95.1

512
SigLIP 2
32.5
42.5
50.6
55.2
67.6
95.3

So400m/14
224
SigLIP
26.6
31.9
45.8
55.1
74.1
94.7

SigLIP 2
31.9
38.1
49.1
55.4
65.6
94.8

384
SigLIP
32.1
36.5
51.6
56.3
71.7
94.9

SigLIP 2
38.3
45.2
56.1
56.6
68.6
95.2

So400m/16
256
SigLIP 2
33.2
39.8
50.9
55.8
66.7
95.0

mSigLIP
27.1
33.3
48.5
54.2
57.5
94.3

384
SigLIP 2
38.2
44.1
54.4
56.5
67.8
95.3

512
SigLIP 2
40.8
47.6
58.6
56.6
69.2
95.3

g-opt/16
256
SigLIP 2
37.6
46.6
54.0
56.9
71.2
95.4

384
SigLIP 2
44.5
52.0
58.7
57.2
72.2
95.7

ViT
Res.
Model
Disparity
Rep. bias

B/32
256
SigLIP 2
33.3
16.6

B/16
224
SigLIP
31.2
36.6

SigLIP 2
31.0
17.2

256
SigLIP
30.2
35.6

SigLIP 2
29.7
19.4

384
SigLIP
30.9
35.8

SigLIP 2
30.6
18.0

512
SigLIP
31.5
35.4

SigLIP 2
30.8
20.0

L/16
256
SigLIP
32.0
35.5

SigLIP 2
31.1
7.3

384
SigLIP
32.0
34.8

SigLIP 2
30.4
6.6

512
SigLIP 2
29.2
6.8

So400m/14
224
SigLIP
30.5
33.3

SigLIP 2
29.7
7.4

384
SigLIP
29.2
33.9

SigLIP 2
28.1
7.5

So400m/16
256
SigLIP 2
28.4
7.2

mSigLIP
31.6
37.3

384
SigLIP 2
29.0
11.0

512
SigLIP 2
28.2
10.8

g-opt/16
256
SigLIP 2
28.1
7.9

384
SigLIP 2
28.3
4.9

Generated on Wed Mar 5 13:11:17 2025 by LaTeXML
