# Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition

- Source HTML: `raw/html/Maruani et al. - 2026 - Illustrator's Depth Monocular Layer Index Prediction for Image Decomposition.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2511.17454
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Illustrator’s Depth: Monocular Layer Index Prediction for Image Decomposition

Nissim Maruani∗

Inria, UCA

  
Peiying Zhang

CityUHK

  
Siddhartha Chaudhuri

Adobe Research

  
Matthew Fisher

Adobe Research

  
Nanxuan Zhao

Adobe Research

  
Vladimir G. Kim

Adobe Research

  
Pierre Alliez

Inria, UCA

  
Mathieu Desbrun

Inria/X, IP Paris

  
Wang Yifan

Adobe Research

###### Abstract

We introduce Illustrator’s Depth, a novel definition of depth that addresses a key challenge in digital content creation: decomposing flat images into editable, ordered layers. Inspired by an artist’s compositional process, illustrator’s depth infers a layer index for each pixel, forming an interpretable image decomposition through a discrete, globally consistent ordering of elements optimized for editability.
We also propose and train a neural network using a curated dataset of layered vector graphics to predict layering directly from raster inputs. Our layer index inference unlocks a range of powerful downstream applications. In particular, it significantly outperforms state-of-the-art baselines for image vectorization while also enabling high-fidelity text-to-vector-graphics generation, automatic 3D relief generation from 2D images, and intuitive depth-aware editing. By reframing depth from a physical quantity to a creative abstraction, illustrator’s depth prediction offers a new foundation for editable image decomposition.

## 1 Introduction

The organization of a digital artwork into a stack of layers is a fundamental concept in creative software. This paradigm, common to both vector-based and raster graphics tools, is central to the creative process as it allows for the independent manipulation and editing of individual compositional elements. This layering is also inherently related to the physical depth of objects within a scene, in that closer elements obscure those that are farther away.

While recent neural architectures can efficiently and accurately predict monocular depth from images [yang_depth_2024, bochkovskii_depth_2025] or compute panoptic segmentations [kirillov_panoptic_2019, ravi_sam_2025], they are unable to decompose input illustrations or images into useful, ordered layers for three main reasons.
First, illustrative layers differ fundamentally from physical depths: important visual elements such as shadows may be placed above the objects on which they are cast, and non-orthogonal flat surfaces with overlapping physical depth gradients may nevertheless be mapped to discrete, sortable layers (see dominoes in Fig. 2).
Second, because illustrations typically appear on flat media (i.e., book pages, posters, or paintings), monocular depth estimation models are explicitly trained to ignore them (see t-shirt in Fig. 2).
Third, illustrative layering also differs from plain panoptic segmentation: the grouping and structuring of segmented regions of an input are key to the editability of the layer decomposition. An illustrator’s notion of layer depth is thus a subtle mix between segmentation and depth ordering to facilitate both design and editing.

Although rarely acknowledged or articulated as such, layer inference is a core challenge in vectorization that impacts numerous downstream applications by offering an intuitive layer decomposition enhancing editing capabilities. Existing state-of-the-art methods, however, remain limited in scope, either handling only simple inputs [wu_layerpeeler_2025, song_layertracer_2025], or relying on brittle heuristics [pun_vtracer_2025, zhao_less_2025, law_image_2025, ma_towards_2022, zhou_segmentation-guided_2025, hirschorn_optimize_2024] which do not consistently yield useful results. In the raster domain, several approaches have explored transparent layer extraction or generation [zhang_transparent_2024, leonardis_objectdrop_2024, pu_art_2025, lee_generative_2025, yang_generative_2025], yet these operate exclusively at the object level. To the best of our knowledge, no existing technique can achieve fine-grained, detailed image layer decomposition.

We introduce Illustrator’s Depth, a new concept designed to address these challenges by providing a novel way to represent the structural layering of vector graphics. Specifically, we define the illustrator’s depth of an image as the inverse mapping from each pixel to its corresponding layer index in its digital mockup, effectively capturing the spatial and compositional ordering of the artwork.
We infer illustrators’ depth from arbitrary images automatically by leveraging a Depth Pro based neural network [bochkovskii_depth_2025] trained on a large, curated SVG dataset.
Our model operates in a feed-forward manner to predict pixel-level layer indices, enabling a wide range of applications such as image editing and depth-aware vector graphics manipulation.

More specifically, we present a number of contributions:

- •

We introduce the notion of Illustrator’s Depth and train a network to predict it, enabling fast layer decomposition;

- •

We show that incorporating our model into standard vectorization pipelines yields consistently layered SVGs with state-of-the-art visual fidelity;

- •

We propose a novel method for evaluating layer quality in vector graphics by rasterizing the predicted illustrator’s depth and assessing its consistency with the ground truth;

- •

We demonstrate that coupling our pipeline with Text2Img models substantially enhances the generation of high-quality, editable vector illustrations from text;

- •

Finally, we showcase other applications of illustrator’s depth in layer-based segmentation, depth-aware object insertion, tactile graphics creation, and artwork analysis.

(a) Input Image
empty

(b) Illustrator’s Depth 
(Ours)

(c) Physical Depth
(Depth Pro [bochkovskii_depth_2025])

## 2 Related Work

##### Monocular depth estimation (MDE)

Classical learning approaches for depth estimation from images [saxena_make3d_2009, zhang_monocular_2015, hoiem_recovering_2007, rezaeirowshan_monocular_2016, eigen_depth_2014, fu_deep_2018] have evolved into strong backbones trained on diverse data [ranftl_towards_2020, farooq_bhat_adabins_2021, yang_depth_2024, bochkovskii_depth_2025].
They yield finely detailed and continuous (relative or metric) depth maps that serve as robust physical priors. Yet, they remain blind to content without true volume, like printed posters or patterns on clothing.
In contrast, our objective is to produce a new kind of depth prioritizing user editability over metric prediction.

##### Layered depth for view synthesis

Layered Depth Images store multiple depth samples per ray to model occlusions [shade_ldi_1998, dhamo2019peeking], while Multiplane Images approximate scenes by many fronto-parallel planes for novel-view rendering [zhou_stereomag_2018, mildenhall_llff_2019]. These abstractions excel at detecting disocclusions and synthesizing new views, but they produce multi-sample or multi-plane depths, not a single discrete index per pixel that a designer can restack. Furthermore, they focus on physical depth like MDE, unlike our illustrator’s depth which focuses on layer index prediction.

##### Amodal / instance / panoptic segmentation

Moving from geometry to semantics, segmentation families group regions by categories, but do not encode geometric ordering. Standard instance and panoptic methods provide high-quality visible masks [he_maskrcnn_2017, kirillov_panoptic_2019, kirillov_panoptic_fpn_2019, cheng_panopticdeeplab_2020, cheng_mask2former_2022, kirillov_sam_2023, ravi2025sam] without global per-pixel depth ordering. Amodal instance and amodal-panoptic formulations extend masks to occluded regions (for countable “thing” categories, typically), while “stuff” categories remain modal; representative datasets and models include [zhu2017semantic, xiao2021amodal, qi_kins_2019, mohan2022amodal]. Occlusion-aware and amodal transformers refine completion and boundary reasoning [lee2022instance, tran_aisformer_2022, ke_bilayer_2021, dhamo2019peeking], yet supervision and metrics remain instance-centric or pairwise. None imposes a single, transitive ordering across all pixels, which is the target of our globally-consistent ordinal layer map.

##### Generative decompositions for editing

Inspired by traditional approaches [richardt2014vectorising, tan2016decomposing, favreau2017photo2clipart], editing-focused decompositions produce per-subject RGBA layers to facilitate local edits. Examples include real-time human matting [lin_real-time_2021], generative pipelines that output editable layers for subjects and effects [lee_generative_2025, yang_generative_2025], and atlas-based video methods that unwrap scenes into a few textures with an alpha channel for temporal consistency [lopes_learned_2019, law_image_2025]. These layers are effective for targeted edits but are independent and not constrained to a global, per-pixel depth order. Instead, we seek a single “illustrator’s depth” map that provides an coherent ordering of all pixels in order to facilitate further editing.

##### Layering in vectorization

An obvious application of our layer index estimation is vectorization: given our per-pixel ordinal map, standard raster-to-vector pipelines can group paths by layer and export edit-ready stacks. Existing systems based on heuristics or optimization [ma_towards_2022, hirschorn_optimize_2024, pun_vtracer_2025, law_image_2025, zhou_segmentation-guided_2025] often fail to infer a clean, useful layering. Learning-based approaches [lopes_learned_2019, reddy_im2vec_2021, rodriguez_starvector_2025, rodriguez_rendering-aware_2025, yang_omnisvg_2025] can, in principle, learn layer order from examples, but their training often compounds all the steps of the vectorization process (including Bézier control points), resulting in frequent reconstruction failures on complex inputs. Very recent works explore explicit layer predictions for better editing [wu_layerpeeler_2025, song_layertracer_2025], but remain limited in the amount of paths and details they generate. Instead, our layer index prediction provides a supervised signal for ordering itself, allowing traditional vectorizers to assemble SVGs in a manner most useful for further editing.

## 3 Method

(a) Input

(b) GT

(c) Ours

(d) Dep.A.-v2

(e) DepthPro

We now introduce our notion of illustrator’s depth in Sec. 3.1, before describing our dataset curation in Sec. 3.2, and finally presenting our neural network implementation and training in Sec. 3.3.
Evaluation tests and ablation studies will be presented and discussed at length in Sec. 4.

### 3.1 Illustrator’s Depth

From an input illustration II, represented as a raster H×WH\!\times\!W RGB image, we refer to its illustrator’s depth as the mapping from each image pixel of the input to a layer index i∈{1​…​N}i\!\in\!\{1...N\}.
Conceptually, this map represents how an artist might have structured the image as a composition of NN separate layers (see Fig. 3), each corresponding to a different element or object drawn at a particular depth. Thus,
illustrator’s depth provides a per-pixel layer assignment that captures an interpretable notion of structural depth implicit in the artist’s compositional workflow, which can then be directly leveraged for editing purposes.
This paper proposes predicting this mapping from an image using a neural network and a curated training set of layered compositions, yielding an illustrator’s depth image Dθ​(I)∈ℝH×WD_{\theta}(I)\!\in\!\mathbb{R}^{H\!\times\!W} where depth is treated as a continuous value rather than a discrete layer index as it still captures relative ordering while allowing straightforward binning into discrete values if necessary.

### 3.2 Curating a Training Dataset

Training our network to predict Illustrator’s Depth requires a large-scale dataset of images paired with their ground-truth layer structure. Scalable Vector Graphics (SVG) files are an ideal source for this data, as they are inherently composed of layered vector paths that define the stacking order of a composition. We leverage this property by developing a three-stage data preparation pipeline: first, we source a suitable dataset of layered SVGs; second, we curate it to reduce ambiguity; and finally, we rasterize the vector files into corresponding image and depth map pairs for training.

##### Data sourcing

While SVGs provide a structural foundation, the quality of their layering is crucial. Many SVG datasets, while visually correct when rendered, contain disorganized or programmatically generated layers that do not reflect an artist’s intent. Yet, effective learning depends on a dataset with intuitively and consistently structured compositions. After reviewing existing options, we selected the MMSVG-Illustration dataset [yang_omnisvg_2025], which features SVGs where elements are layered in a consistent and meaningful way, with layers systematically organized from the lowest index for the background to the highest index for the foreground, and outline strokes always placed above their corresponding color fills for instance.

##### Data curation

Even a high-quality dataset like MMSVG contains inherent ambiguities that can hinder learning. Artistic layering is often subjective; for instance, multiple distinct objects might logically share the same depth level, and different artists may have different layering habits. This variability can create a noisy training signal. To normalize these variations and create a more consistent ground truth, we perform two curation steps. First, we merge consecutive layers that share the same RGB color to simplify the structure. Second, we identify and exclude ambiguous cases where non-consecutive layers of the same color overlap in the final rendered image, as this significantly improves training stability.

##### Ground-truth rasterization

Once the SVG dataset is curated, the final step is to generate the rasterized image-depth pairs for training. For each curated SVG file, we generate its corresponding RGB input image II and ground-truth illustrator’s depth map D​(I)D(I) of size H×WH\!\times\!W through a custom rasterization process.
First, we create a temporary version of the SVG where each layer’s original color is replaced by a unique color representing its layer index ii in base 256: the index is thus encoded across the RGB channels via

(imod256,⌊i/256⌋mod256,⌊i/2562⌋mod256).\left(i\bmod 256,\lfloor i/256\rfloor\bmod 256,\lfloor i/256^{2}\rfloor\bmod 256\right).\vskip-5.69054pt

We then rasterize this modified SVG; the resulting “false color” image is converted back into a per-pixel integer depth map using the formula D​(I)=R+256⋅G+2562⋅B.D\!\left(I\right)\!=\!R+256\!\cdot\!G+256^{2}\!\cdot\!B.
This encoding strategy allows us to efficiently represent a large number of layers with virtually no additional data loading overhead.
All the resulting pairs {Ik,D​(Ik)}k\{I_{k},D(I_{k})\}_{k} of images and their illustrator’s depths form our training dataset.

Rasterized RGB

Rasterized Depth

Rasterized RGB

Rasterized Depth

(a) GT

(b) Ours + [selinger_potrace_2003, pun_vtracer_2025]

(c) Vtracer [pun_vtracer_2025]

(d) L.I.M. [zhao_less_2025]

(e) LIVE [ma_towards_2022]

(f) Starvector [rodriguez_starvector_2025]

(g) O-SVG [yang_omnisvg_2025]

### 3.3 Neural Network & Training

##### Model

Predicting illustrator’s depth requires reasoning about object boundaries, occlusion, and grouping.
While distinct from physical depth estimation, this task benefits immensely from the powerful priors learned by state-of-the-art monocular depth estimation (MDE) models. In particular, we find that Depth Pro [bochkovskii_depth_2025], built on Dino-v2 [oquab_dinov2_2024] and equipped with a multi-scale encoder, provides a robust feature extractor that allows our model to generalize well from our training set of simple vector graphics to complex, artistic images, as we will demonstrate in Sec. 4. We initialize our model with Depth Pro’s pre-trained weights, leveraging its learned understanding of geometry and occlusion as a crucial prior for our task to enable broad generalization.

##### Scale-invariant loss function

In natural images, distant objects correspond to large physical depth values, which are inherently more challenging to estimate accurately. Therefore, most MDE models [yang_depth_2024, bochkovskii_depth_2025, ranftl_towards_2020] learn inverse depth values 1/d1/d, prioritizing the accuracy of foreground objects over distant ones. In contrast, illustrations are composed in a structured, layer-wise manner from background to foreground, where depth values typically range from 11 to NN. In this setting, estimating the illustrator’s depth is not inherently harder for background layers than for foreground ones. Instead of learning in disparity space, we thus train our model to predict discrete ground-truth layer indices (1,…,N1,...,N) directly, assigning equal importance to all image layers (please see ablation studies in Sec. 4.1). Our primary objective, however, is to recover the correct relative ordering of these layers rather than their absolute index values. To focus the training on this relative structure, and remain robust to the potentially large range of NN, we adopt a scale-invariant normalization scheme similar to MiDaS [ranftl_towards_2020]. For any depth map DD, we compute its median mm and mean absolute deviation ss, and normalize each depth value dd as d^≔(d−m)/s\hat{d}\!\coloneqq\!{(d\!-\!m)}/{s}. We then train the network using a Mean Absolute Error (MAE) loss on these normalized maps, i.e., using the loss:

LMAE​(D​(I),Dθ​(I))=|D^​(I)−D^θ​(I)|¯.L_{\scriptscriptstyle\text{MAE}}(D(I),D_{\theta}(I))=\overline{|\hat{D}(I)-\hat{D}_{\theta}(I)|}.\vskip-4.2679pt

(1)

##### Training

The network is trained on our SVG dataset using standard training practices, including data augmentation (color jitter, random inversion, random blur) and a cosine learning rate schedule. Additionally, we follow [bochkovskii_depth_2025] by emplying two distinct learning rates for the encoder (DINO-v2 [oquab_dinov2_2024]) and the CNN-based decoder. Details are provided in Sec. 4, the Supplementary Material, and the code.

##### Post-processing

As our network outputs pixel-wise illustrator’s depth estimates, optional post-processing can be applied to derive discrete layer indices. Depending on the target application, two common strategies are advisable: (1) direct segmentation of depth values using binning or thresholding, and (2) clustering in RGB space followed by assigning each cluster its median depth. We typically adopt the first strategy for raster image processing (Sec. 4.4), whereas the second is better suited for vectorization tasks (Secs. 4.2 and 4.3) where inputs typically exhibit color-consistent regions. In the latter case, clusters with similar colors and depths can be further merged to simplify the resulting SVG paths (see Fig. 5). Notably, even without post-processing, our predicted illustrator’s depth maps are visually coherent and structurally clean, see Figs. 1, 2, 4 and 8.

## 4 Experiments and Applications

In this section, we outline our training setup in Sec. 4.1 and benchmark our model against state-of-the-art monocular depth estimators.
Then we demonstrate a variety of applications of illustrator’s depth.
First, we embed our trained model into a vectorization pipeline (Sec. 4.2), which outperforms state-of-the-art methods, and show how it enables a creative, fully editable workflow when paired with generative image models (Sec. 4.3).
We then showcase diverse raster-based editing tools enhanced by our predicted illustrator’s depths (Sec. 4.4), including relief generation for tactile graphics and
layer-wise decomposition.

### 4.1 Predicting Illustrator’s Depth

Order ↑\uparrow

MAE ↓\downarrow

MSE ↓\downarrow

Depth Pro [bochkovskii_depth_2025]

0.636
1.44
4.76

Depth Anything-v2 [yang_depth_2024]

0.791
1.16
3.58

Ours
0.987
0.12
0.26

Depth prior

initialization

Data

cleaning

Direct index

training

✓
✓
0.903
0.51
1.17

✓

✓
0.905
0.53
1.21

✓
✓

0.980
0.50
1.88

✓
✓
✓
0.981
0.16
0.29

##### Training

As detailed in Sec. 3, our model is trained on the MMSVG-Illustration dataset [yang_omnisvg_2025]. Following data cleaning and rasterization to a resolution of 1536×15361536\!\times\!1536, the dataset comprises approximately 100K consistently layered SVG images, with 80%80\% allocated for training and 20%20\% reserved for evaluation. In line with [zhao_less_2025], we randomly select 100 images for quantitative analysis — see Supplementary Material for results on the SVGX-Core dataset. Training is done for 40 epochs on 8 Nvidia®​ A100 GPUs, with a cosine learning rate schedule, a max learning rate of 5⋅10−65\cdot\!10^{-6}, and a batch size of 8.

Layering Quality
Visual Fidelity

Method
Layering Prior
Order ↑\uparrow

MAE ↓\downarrow

MSE ↓\downarrow

Path Number ↓\downarrow

MSE (×10−2\times 10^{-2}) ↓\downarrow

SSIM ↑\uparrow

LPIPS ↓\downarrow

Vtracer [pun_vtracer_2025]

Heuristics
0.689
2.58
15.67
3.65
0.023
0.994
0.022

Less Is More [zhao_less_2025]

Heuristics
0.746
2.43
21.10
5.54
0.663
0.961
0.043

LIVE [ma_towards_2022]

Optimization-based
0.838
4.88
96.91
8.62
0.297
0.946
0.053

Starvector [rodriguez_starvector_2025]

Data-driven
0.918
1.52
9.75
0.53
9.123
0.858
0.302

OmniSVG [yang_omnisvg_2025]

Data-driven
0.925
1.31
8.08
0.54
9.997
0.830
0.317

Ours + [pun_vtracer_2025, selinger_potrace_2003]

Data-driven
0.987
0.46
2.09
0.16
0.018
0.997
0.005

##### Baselines

We compare our approach with two state-of-the-art monocular depth estimation (MDE) methods, Depth Pro [bochkovskii_depth_2025] and Depth Anything-v2 [yang_depth_2024].

##### Metrics

We evaluate performance by rendering illustrator’s depth maps from ground-truth SVGs as described in Sec. 3.2.
Since each method produces depth estimates in its own scale, we first normalize all predicted depth maps using the procedure described in Sec. 3.3 prior to computing Mean Squared Error (MSE) and Mean Absolute Error (MAE). While both MSE and MAE assess pixel-wise depth accuracy, many of our target applications require a globally consistent layer ordering rather than precise depth values. Therefore, following Zhang et al. [zhang_monocular_2015], we further evaluate depth ordering consistency by randomly sampling pixel pairs from the ground truth and predictions, and checking whether their relative depth order is preserved (see Supplementary Material for details). The resulting depth ordering consistency metric (abbreviated as Order in Tabs. 1-3) measures the percentage of correctly ordered pixel pairs, providing a complementary measure of global depth consistency.

##### Results

While related, physical depth and illustrator’s depth do capture fundamentally different concepts (Fig. 2). Standard MDE models, trained to predict real-world geometry, struggle to recover correct layer ordering in illustrations (Fig. 4); our model, purposely trained to infer layer indices, achieves markedly better results, outperforming all baselines by a wide margin (Tab. 1). Inference takes less than one second on current GPUs as reported in [bochkovskii_depth_2025].

##### Ablation studies

We conduct a series of ablation studies to validate our design choices discussed in Sec. 3. As detailed in Tab. 2, both Depth Pro initialization (leveraging a physical depth prior from weights learned on millions of images) and data cleaning (removing inconsistencies and ambiguities in ground-truth layers) boost the depth ordering consistency
quite sharply.
Although training directly with layer indices (1,…,N)(1,...,N) instead of disparity space (1/d)(1/d) yields comparable global ordering scores, it facilitates a more balanced optimization between foreground and background layers: this results in better depth transitions and a clear advantage across all evaluation metrics; see the Supplementary Material for additional qualitative evaluations.

### 4.2 Vectorization

Image vectorization, which consists in converting raster images to vector graphics, is a particularly straightforward application of illustrator’s depth.

##### Pipeline

Our model integrates seamlessly into existing vectorization pipelines such as VTracer [pun_vtracer_2025], where we replace area-based sorting heuristics with our predicted illustrator’s depth. We first compute color clusters, sort them using our layer index prediction, inpaint layers to fill holes and bridge gaps (with, e.g., Scikit-Image [van2014scikit]), before vectorizing each layer with potrace [selinger_potrace_2003].
The whole process, including our illustrator’s depth prediction, only takes seconds.

##### Baselines

We benchmark our pipeline against key state-of-the-art approaches, based on simple area heuristics (VTracer [pun_vtracer_2025]) or more advanced cluster-sorting strategies (Less Is More [zhao_less_2025]), optimization methods (LIVE [ma_towards_2022]), and LLM-based tools (StarVector [rodriguez_starvector_2025], OmniSVG [yang_omnisvg_2025]).

##### Metrics

Vectorization demands both compactness and accuracy for best editability. We thus measure layering quality using the depth ordering consistency (Order), mean squared error (MSE), and mean absolute error (MAE), as well as path count errors |N−N~|/N|N\!-\!\tilde{N}|/N to compare the number of paths in ground-truth (NN) vs. reconstructed (N~\tilde{N}) SVGs.
We then evaluate visual fidelity by measuring the rasterized output compared the input using MSE in RGB space, Structural Similarity Index Measure (SSIM) [zhou_wang_image_2004], and Learned Perceptual Image Patch Similarity (LPIPS) [zhang_unreasonable_2018].

##### Results

Although most vectorization methods produce outputs that look quite close to the input raster images, visualizing their layer indices in false colors reveals substantial differences in layering quality (Fig. 5).
Methods relying on heuristics such as VTracer [pun_vtracer_2025] and Less is More [zhao_less_2025] frequently misorder layers; for instance, spiral binding holes in the calendar in Fig. 5 are incorrectly positioned on top despite belonging to the background. Optimization-based LIVE [ma_towards_2022] introduces spurious layers and shapes, while LLM-based approaches [rodriguez_starvector_2025, yang_omnisvg_2025] often fail (sometimes, spectacularly) to achieve full reconstruction.
In contrast, our pipeline is able to faithfully reconstruct the input while producing layer indices close to the ground truth.
Additionally, quantitative results from Tab. 3 confirm these observations:
our method matches VTracer’s reconstruction fidelity while outperforming all SOTA competitors in layer-index accuracy.
Interestingly, our layering evaluation reveals a clear divide between methods excelling at reconstruction but weak in layering (VTracer, Less is More) and those with opposite strengths (Starvector, OmniSVG).
Our approach thus combines the power of traditional vectorizers with the quality of data-driven layer index prediction, enabling state-of-the-art performance on both fronts.

### 4.3 Text-to-Vector-Graphics Generation

The creation of high-quality vector graphics remains a challenging problem. Direct generation techniques, such as those employing Score Distillation Sampling (SDS) [zhang_text--vector_2024, polaczek_neuralsvg_2025] or Large Language Models (LLMs) [rodriguez_starvector_2025, rodriguez_rendering-aware_2025, yang_omnisvg_2025], have not yet matched the visual fidelity achieved by state-of-the-art text-to-image generative models. Here again, our illustrator’s depth neural prediction can dramatically help in obtaining high-quality editable illustrations.

##### Pipeline

Leveraging recent advances in high-quality image generation [labs_flux1_2025, google_gemini_2025], we first generate vector-style raster images (prompts are detailed in the Supplementary Material). These raster images are subsequently transformed into structured, editable, and layered SVG using our specialized vectorization pipeline described in Sec. 4.2.

##### Results

Fig. 6 presents examples generated via Flux [labs_flux1_2025] and postprocessed with illustrator’s depth. The resulting SVG illustrations exhibit high visual complexity and coherent layer organization, facilitating the intuitive grouping and editing of individual elements (see supplementary video). Our vectorization can be similarly integrated to Nano Banana [google_gemini_2025] to offer a more advanced, multi-stage generative workflow as illustrated in Fig. 7.
We also show comparisons with Neural Path Representation [zhang_text--vector_2024], NeuralSVG [polaczek_neuralsvg_2025], and LayerTracer [song_layertracer_2025], in the Supplementary Material.

(a) Input Image

(b) Illustrator’s Depth

(c) Relief, 3D rendering

### 4.4 Beyond Vector Graphics

Despite being trained exclusively on depth data generated from simple SVG images, our model demonstrates a remarkable ability to generalize beyond this narrow scope. It successfully infers illustrator’s depth across highly diverse inputs, from complex illustrations and artistic renderings, to even natural images, due to our use of pretrained priors [bochkovskii_depth_2025, oquab_dinov2_2024] learned from millions of images.
This section showcases two practical applications leveraging this strong generalization.
Additional qualitative results and discussions of failure cases are provided in Figs.1 & 2, and in the Supplementary Material.

#### 4.4.1 Automatic Relief Generation From a Single Image

##### Task

Relief is a sculptural method where elements remain attached to a solid background to give the impression that the sculpture has been raised above the background. Bas-relief, a shallow form of this technique, is widely applied, from coinage to architectural ornament [zhang_computer-assisted_2019]. Current methods for generating 3D reliefs from 2D images are fundamentally limited by their reliance on user-defined depth ordering [reichinger_high-quality_2011]. We eliminate this user interaction entirely by leveraging the fully automated output of our model.

##### Pipeline

Given an input image, our system first generates a pixel-wise illustrator’s depth map dθ​(i,j)d_{\theta}(i,j). This depth is then directly used to build a triangulated surface by transforming each pixel into a vertex with 3D coordinates (i,j,dθ​(i,j))(i,j,d_{\theta}(i,j)), and triangulating adjacent vertices.

##### Results

The resulting mesh easily integrates into any 3D application as illustrated in Fig. 8. Crucially, our illustrator’s depth transforms flat paintings into 3D objects without any manual annotation, offering an alternative, intuitive, and tangible interaction with works of art.

#### 4.4.2 Depth-Based Editing

(a) Input Image

(b) I. D.

(c) Foreground

(d) Editing

##### Task

Raster image editing relies on the composition of multiple layers. Existing segmentation tools, however sophisticated they have become, often fall short because they perform based on ambiguous requests: if a user clicks a face, do they mean the face, the whole character, or the entire foreground? Our work helps resolve this ambiguity, as enriching input images with our predicted illustrator’s depth dramatically facilitates layer separation.

##### Pipeline

Illustrator’s depth is easily leveraged to inform segmentation:
based on a user-defined threshold value tt adjustable in realtime via a slider, an image can be split into two layers, one (foreground) defined as illustrator’s depths satisfying D​[i,j]>tD[i,j]\!>\!t and one (background) for all others.
More generally, any binning strategy into NN layers, found through a quick analysis of the entire map DD or derived manually, provides a decomposition into layers by ranges of illustrator’s depths, which can be directly uploaded in raster graphic editors to allow for direct editing.

##### Results

Illustrator’s depth within the context of raster image editing provides a robust mechanism for selective element isolation as demonstrated in Fig. 9. Paired with any inpainting model such as [rombach_high-resolution_2022], our method can produce NN overlapping layers to allow for parallax effects for instance, see Fig. 10. Additional examples can be found in the accompanying Supplementary Material and video.

## 5 Conclusion and Future Work

We introduced Illustrator’s Depth, a novel concept that augments image pixels with additional layer indices, enabling straightforward decomposition into an edit-ready stack. Trained on a curated dataset of SVG files, our network can infer illustrator’s depth across a wide range of inputs, ranging from simple icons to complex raster graphics.
We demonstrated that our method achieves SOTA performance in image vectorization and facilitates a number of downstream tasks beyond vector graphics, such as text-to-vector generation, interactive editing, and relief generation.

Although our current model is trained specifically for this task with a curated dataset of SVGs, the rapid advancement of vision models toward one-shot and zero-shot generalization [wiedemer_video_2025] suggests a near-future where illustrator’s depth could be inferred directly from natural prompts, without explicit training. Beyond its current technical form, we believe that the underlying concept of illustrator’s depth will remain relevant across a variety of creative domains: by shifting the notion of depth from a physical metric to a layer-based ready-to-edit abstraction, our work introduces a new paradigm for intelligent creative tools to better assist the artistic process. Illustrator’s depth transforms image decomposition from a mere technical challenge into a creative and assistive foundation for the next generations of computational art and design systems.

## 6 Acknowledgments

This work was supported by the French government through the 3IA Cote d’Azur Investments in the project managed by the National Research Agency (ANR-23-IACL-0001), Ansys, and a Choose France Inria chair.

\thetitle

Supplementary Material

This supplementary material provides additional details, results, and comparisons to complement our CVPR paper on Illustrator’s Depth.

## 7 Evaluation on SVGX

While we trained our model on (a curated subset of) the MMSVG-Illustration dataset [yang_omnisvg_2025], we also evaluated our layer index predictions on the SVGX-Core-250k dataset curated by [xing_empowering_2025] for completeness. Similar to MMSVG, we randomly select 100 images for quantitative analysis. As shown in Tab. 4 and Fig. 11, our model demonstrates strong generalization and maintains excellent performance.

(a) Input Image

(b) Ground Truth

(c) Ours

Order ↑\uparrow

MAE ↓\downarrow

MSE ↓\downarrow

MMSVG
0.987
0.12
0.26

SVGX-Core-250k
0.984
0.16
0.53

## 8 Ablation Studies

We present additional qualitative results in Fig. 12 to complement the quantitative findings reported in Table 2 of the main paper. While data cleaning and the use of depth priors lead to pronounced improvements, the choice of layer indices vs. disparity space (dd vs. 1/d1/d) yields more subtle effects, yet still provides noticeable gains in these examples.

(a) GT

(b) W/o Data Cleaning

(c) Ours

(d) GT

(e) W/ Direct Index

(f) Ours

(g) GT

(h) W/o Depth Prior

(i) Ours

## 9 Details on our vectorization pipeline

(a) GT

(b) 9 clusters

(c) Ours

(d) 5 Clusters

Our vectorization tests were performed on a pipeline combining VTracer [pun_vtracer_2025] and Potrace [selinger_potrace_2003] with our contributions. Specifically, illustrator’s depth based vectorization is achieved as follows:

- 1.

We find color-constant clusters using VTracer (Fig. 13(b)). The values of several hyper-parameters are important, such as filter_speckle to suppress noise, color_precision and layer_difference to accurately split the image in distinct regions. All of them are provided in our code.

- 2.

Instead of relying on VTracer’s heuristics to sort the clusters, we leverage our predicted illustrator’s depth (Fig. 13(c)) for layering, assigning the cluster’s depth order to the median of the predicted depth for each cluster.

- 3.

Cluster grouping is important to ensure a well-layered, compact output. After sorting, we further merge layers with neighboring indices if their RGB colors are within a certain threshold τ=0.05\tau\!=\!0.05 in the L2L^{2} norm. This results in an ordered clustering image C∈[1,…​N]H×WC\!\in\![1,...N]^{\scriptscriptstyle H\times W} (Fig. 13(d)).

- 4.

While it doesn’t affect the final rendering, filling holes and bridging gaps yields simpler, overlapping layers that are compact and easy to edit. Given a cluster with index nn, we create a binary mask 𝟏C​[i,j]>n\mathbf{1}_{C{[i,j]}>n}, and inpaint the missing regions of 𝟏C​[i,j]⁣=⁣=n\mathbf{1}_{C{[i,j]}==n} using off-the-shelf algorithms (see Sec. 10 and Fig. 14).

- 5.

This layer collection is then vectorized with Potrace [selinger_potrace_2003] and assembled to form the final vector graphics.

## 10 Inpainting

While not part of our contributions, we also show examples of inpainting strategies once our layer index prediction has been generated, see Fig. 14. For vector graphics, we rely on fast, off-the-shelf algorithms provided by Scikit-image [van2014scikit]. We experimented with two variants in order to fill the missing regions: one that interpolates using the nearest unmasked point, and another based on biharmonic interpolation. Depending on the application, users may prefer one approach over the other: the biharmonic method produces smoother curves, whereas the closest-point interpolation yields sharper, crisper boundaries (see Fig. 14). We generally use the latter in our code due to its faster computational time.
While this simple hole-filling approach is sufficient for most vector graphics, data-driven inpainting may be desired for more involved applications, including raster image editing: here again, leveraging off-the-shelf inpainting models (see Fig. 3) offers a solution that doesn’t require any additional training.

(a) GT

(b) Illustrator’s Depth

(c) Scikit Biharmonic Inpainting

(d) Scikit Closest Point Inpainting

## 11 Prompts for vector-styled images

The main paper shows two text-to-image examples, one using FLUX [labs_flux1_2025] and one using Nano Banana [google_gemini_2025].
For FLUX (Fig. 6 in the main paper), we found that, given a desired object to be drawn (underlined below), a mix of positive and negative prompts provides clean vector-styled raster images that are easy to process with our pipeline; for instance,

⬇

{ "prompt": "Vector graphics of a simple cheetah head.",

"prompt_2": "Vector graphics of a simple cheetah head. SVG file. Filled shapes, minimalist design. Abstract.",

"negative_prompt": "Gradient, 3D. Small details. Fineline details.",

"negative_prompt_2": "Gradient, 3D. Small details. Fineline details.",

"num_inference_steps": 28,

"num_images_per_prompt": 1

}

For Nano Banana (Fig. 7 in the main paper), we simply prompt the model through:

⬇

{ "prompt": "Vector graphic illustration of a cat. SVG style, blue background. Smooth, flowy shapes."

}

## 12 Comparison with Text2Vector Generators

In addition to the results discussed in
Sec. 4.3
of the main paper, more examples of text-to-vector-graphics generations are given in Fig. 15. Both text-to-vector generations using Neural Path Representations [zhang_text--vector_2024] and NeuralSVG [polaczek_neuralsvg_2025] are based on Score Distillation Sampling (SDS) that relies on a pretrained diffusion model to backpropagate gradients to Bézier curve parameters. Consequently, their generated illustrations are relatively simple and lack fine details (we reproduce the images provided in their articles in Fig. 15). Although LayerTracer [song_layertracer_2025] employs its own custom diffusion model, it exhibits similar limitations, producing simple emoji-like graphics; note that we used the prompting setup provided in their public repository. In contrast, our method can decompose any output into layered SVG representations, effectively decoupling generation from vectorization — and thus fully leveraging the capabilities of modern generative models. Our modular pipeline, compatible with both Flux [labs_flux1_2025] and Nano Banana [google_gemini_2025], produces detail-rich vector illustrations within seconds (see Sec. 11 for the full prompt configurations).

(a) Ours + [google_gemini_2025]

(b) LayerTracer [song_layertracer_2025]

(c) Ours + [labs_flux1_2025]

(d) LayerTracer [song_layertracer_2025]

(e) Ours + [labs_flux1_2025]

(f) Neural Paths[zhang_text--vector_2024]

(g) Ours + [labs_flux1_2025]

(h) NeuralSVG [polaczek_neuralsvg_2025]

## 13 Failure cases

##### Texture artifacts

Since our model is trained on clean SVG data, a failure case arises when the input image contains canvas textures or defects. These issues can be easily mitigated by using a generative model (e.g., Nano Banana [google_gemini_2025]) to clean the image before applying our method (see Fig. 16).

##### Incorrect ordering

Like any machine learning model, ours can occasionally make mistakes (see Fig. 17, bottom row). Quantitatively, such errors are rare: as shown in Tab. 1, over 98%98\% of randomly sampled pixel pairs are correctly ordered in our experiment with MMSVG.

##### Foreground Focus

Our training set primarily contains single objects over white backgrounds. Consequently, the model sometimes neglects background elements, which may be undesirable in certain scenarios (see Fig. 17, top row). Future work could address this limitation by training on more complex or synthetic SVG datasets that include background elements.

(a) Input Image

(b) Illustrator’s Depth

(c) Input Image (cleaned)

(d) Illustrator’s Depth

(a) Input Image

(b) Illustrator’s Depth

## 14 Depth ordering consistency metric

To compute the depth ordering consistency from a ground-truth illustrator’s depth map DD and a predicted map DθD_{\theta}, we adapt the approach of [zhang_monocular_2015] and proceed as follows:

- 1.

we uniformly sample H×W50\frac{H\times W}{50} random pairs of pixel locations (i,j)(i,j) and (k,l)(k,l) and keep only those corresponding to two different layers in DD, i.e., such that D​[i,j]≠D​[k,l]D[i,j]\!\neq\!D[k,l];

- 2.

we then check whether the relative ordering is preserved by comparing the signs of (D​[i,j]−D​[k,l])(D[i,j]\!-\!D[k,l]) and (Dθ​[i,j]−Dθ​[k,l])(D_{\theta}[i,j]\!-\!D_{\theta}[k,l]).

- 3.

Finally, we compute the average consistency score s¯\bar{s} over all pairs by the ratio of preserved ordering over total number of pixel pairs.

This formulation quantifies how effectively the predicted illustrator’s depth maintains correct relative depths, independent of absolute scale. This metric, inherently stochastic as it relies on randomly sampled pixel pairs from the image, exhibits strong stability: sampling 50,00050,000 pairs on 1536×15361536\times 1536 images yielded no significant variations in our experiments (see Fig. 18). And as Tabs. 1-3 from the original paper demonstrate, it offers a complementary measure of layering quality.

## 15 Additional results

For completeness, we also provide a histogram of the number of layers present in our curated training dataset in Fig. 19, as well as a figure demonstrating another potential use of our illustrator’s depth
in Fig. 20, where a painting is automatically turned into a multi-layered pop-up card.

Generated on Fri Dec 5 13:30:44 2025 by LaTeXML
