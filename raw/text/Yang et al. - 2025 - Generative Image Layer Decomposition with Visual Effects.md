# Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects

- Source HTML: `raw/html/Yang et al. - 2025 - Generative Image Layer Decomposition with Visual Effects.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2411.17864
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Generative Image Layer Decomposition with Visual Effects

Jinrui Yang1,2,∗ 
Qing Liu2 
Yijun Li2 
Soo Ye Kim2 
Daniil Pakhomov2 
Mengwei Ren2 
Jianming Zhang2 
Zhe Lin2 
Cihang Xie1 
Yuyin Zhou1 
1UC Santa Cruz
 
2Adobe Research 
https://rayjryang.github.io/LayerDecomp

###### Abstract

Recent advancements in large generative models, particularly diffusion-based methods, have significantly enhanced the capabilities of image editing. However, achieving precise control over image composition tasks remains a challenge.
Layered representations, which allow for independent editing of image components, are essential for user-driven content creation, yet existing approaches often struggle to decompose image into plausible layers with accurately retained transparent visual effects such as shadows and reflections. We propose LayerDecomp, a generative framework for image layer decomposition which outputs photorealistic clean backgrounds and high-quality transparent foregrounds with faithfully preserved visual effects.
To enable effective training, we first introduce a dataset preparation pipeline that automatically scales up simulated multi-layer data with synthesized visual effects. To further enhance real-world applicability, we supplement this simulated dataset with camera-captured images containing natural visual effects.
Additionally, we propose a consistency loss which enforces the model to learn accurate representations for the transparent foreground layer when ground-truth annotations are not available.
Our method achieves superior quality in layer decomposition, outperforming existing approaches in object removal and spatial editing tasks across several benchmarks and multiple user studies, unlocking various creative possibilities for layer-wise image editing.
The project page is https://rayjryang.github.io/LayerDecomp.
00footnotetext: *This work was done when Jinrui Yang was a research intern at Adobe Research.

## 1 Introduction

The rapid advancement of large-scale text-to-image diffusion models [33, 7, 3] has greatly improved image editing capabilities, with recent studies [15, 36, 52, 5] demonstrating promising results by training on large-scale datasets of captioned images. However, achieving precise control for user-driven image composition tasks remains challenging.
Layered representations, which decompose image components into independently editable layers, are essential for precise user-driven content creation. Most visual content editing
software and workflows are layer-based, relying heavily on transparent or layered elements to compose and create content. Despite this, few existing approaches have explored layer-based representation for image editing in depth. Recently, LayerDiffusion [49] is proposed to generate transparent layer representations from text inputs; however, this approach is not well-suited for image-to-image editing tasks. Meanwhile, MULAN [39] presents a multi-layer annotated dataset for controllable generation, but it often fails to preserve essential visual effects in the right layers, limiting its adaptability for seamless downstream edits.

This work aims to fill these gaps by decomposing an input image into two layers: a highly plausible clean background layer, and a high-quality transparent foreground layer that retains natural visual effects associated with the target. These decomposed layers will support layer-constrained content modification and allow seamless blending for harmonious re-composition. Additionally, the faithfully preserved natural visual effects in the foreground layer will benefit related research areas, such as shadow detection and shadow generation.
However, the lack of publicly available multi-layer datasets with realistic visual effects, such as shadows and reflections, poses a significant challenge for training high-quality layer-wise decomposition models. How to accurately decompose images into layers and learn correct representation for visual effects in the foreground layer without ground-truth data is the key problem to solve.

To address these challenges, we first design a dataset preparation pipeline to collect data from two sources: (1) simulated image triplets consisting of composite images blended from random background and unrelated transparent foreground with generated shadow, allowing us to create a large-scale training set with ground-truth for both branches; and (2) camera-captured images for a scene with and without the target foreground object, ensuring the model can effectively adapt to real-world scenarios.
Building on this dataset, we introduce Layer Decomposition with Visual Effects (LayerDecomp), a generative training framework that enables large-scale pretrained diffusion transformers to effectively decompose images into editable layers with correct representation of visual effects.
The key to our model training lies in a consistency loss, which ensures faithful retention of natural visual effects within the foreground layer while maintaining background coherence.
Specifically, for real-world data where visual effect annotations are not available, it is not feasible to directly compute diffusion loss on the foreground layer. Instead, LayerDecomp enforces consistency between the original input image and the re-composite result, blended from the two predicted layers, to ensure the model learns correct representation for the transparent foreground layer.

As shown in Figure 1, LayerDecomp can effectively decompose an input image into a clean background and a transparent foreground with preserved visual effects. This capability supports downstream editing applications without requiring additional model training or inference. Furthermore, our method exhibits superior layer decomposition quality, outperforming existing approaches in both object removal tasks and object spatial editing tasks across various benchmarks and in multiple user studies.

In summary, our contributions are as follows:

- •

We introduce a scalable pipeline to generate large-scale simulated multi-layer data with paired ground-truth visual effects for training layer decomposition models.

- •

We present LayerDecomp, a generative training framework that leverages both simulated and real-world data to enable robust layer-wise decomposition with accurate visual effect representation through a consistency loss, facilitating high-quality, training-free downstream edits.

- •

LayerDecomp surpasses existing state-of-the-arts in maintaining visual integrity during layer decomposition, excelling in object removal and object spatial editing, and enabling more creative layer-wise editing.

## 2 Related Works

### 2.1 Image Editing

Image editing methods can be broadly categorized into two groups: multi-task editing and local editing. Multi-task editing allows for a wide array of image modifications based on high-level inputs, such as user instructions. For example, Emu-Edit [36] offers a flexible interface for diverse image edits. Other methods, including InstructPix2Pix [4], Prompt2Prompt [11], InstructAny2Pix [18], MagicBrush [48], Imagic [17], PhotoSwap [10], UltraEdit [52], HQ-Edit [14], MGIE [9], and OmniGen [44], focus on fine-grained, user-specific edits. These approaches achieve high precision by aligning closely with user instructions, often through fine-tuned models. However, due to the absence of region references, these methods struggle with accurately locating objects and preserving the integrity of unrelated regions.

Local image editing focuses on tasks like inpainting and object insertion, typically guided by masks or reference images. Early GAN-based methods, such as CMGAN [54], LAMA [38], CoModGAN [53], ProFill [46], CRFill [47], and DeepFillv2 [45], use latent space manipulation for inpainting missing regions.
Leveraging the success of text-to-image diffusion models, Repaint [23], SDEdit [25], ControlNet [50], BrushNet [16], and Blended Diffusion [2] combine text guidance with masks or references for customized image inpainting.
Further, ObjectStitch [37], HDPainter [24], and PowerPaint [55] extend this to versatile object editing with text prompts.
Beyond local object editing, recent research also focus on spatial editing techniques that allow more interactive control over object positioning and transformations. Methods like MagicFixup [1], DiffusionHandle [28], DragGAN [27], DiffEditor [26], DesignEdit [15], and DragAnything [43] highlight a growing emphasis on user-driven, spatially aware editing. However, preserving object appearance and background integrity during spatial edits (e.g., moving and resizing) remains challenging, particularly when complex visual effects like shadows and reflections are involved.

### 2.2 Image Layer representation

Obtaining high-quality image layer representations is crucial for implementing accurate and diverse editing objectives. This process typically involves image decomposition, layer extraction, and image matting. PACO [22] provides a fine-grained dataset with mask annotations for parts and attributes of common objects. However, these object representations indicate only the regions of parts and objects, lacking transparent layers for flexible editing.
MAGICK [6] offers a large-scale matting dataset generated by diffusion models, while MULAN [39] creates RGBA layers from COCO [19] and Laion Aesthetics 6.5 [35] using off-the-shelf detection, segmentation, and inpainting models; however, it does not capture visual effects, limiting its applicability for direct editing.
LayerDiffusion [49] and Alfie [32] generate transparent image layers from text prompts, facilitating layer blending. However, text-driven generation limits object identity control in image-specific editing tasks.
Once the image decomposition is performed, users typically need to execute image composition to achieve a cohesive final image. Existing methods like ObjectDrop [41] require model fine-tuning to restore visual effects, which can disrupt the original image’s appearance. In contrast, LayerDecomp models these effects during training, inherently preserving them to produce a seamless, harmonious composite image without extra fine-tuning.

## 3 Approach

### 3.1 Overview of LayerDecomp Framework

As shown in Fig 2, the LayerDecomp framework builds upon Diffusion Transformers (DiTs) [30] to denoise multi-layer image outputs in the latent space encoded by the VAE encoders gϕRGB​(⋅)superscriptsubscript𝑔italic-ϕRGB⋅g_{\phi}^{\text{RGB}}(\cdot) and gψRGBA​(⋅)superscriptsubscript𝑔𝜓RGBA⋅g_{\psi}^{\text{RGBA}}(\cdot). Specifically, the DiT model fθ​(⋅)subscript𝑓𝜃⋅f_{\theta}(\cdot) takes two types of conditional input 𝐜=(𝐲comp,𝐲obj)𝐜subscript𝐲compsubscript𝐲obj\mathbf{c}\!=\!(\mathbf{y}_{\text{comp}},\mathbf{y}_{\text{obj}}), which are the latents of the original composite image, and the decomposition object mask, respectively, i.e., 𝐜=(𝐲comp,𝐲obj)=(gϕRGB​(𝐈compRGB),gϕRGB​(𝐌obj))𝐜subscript𝐲compsubscript𝐲objsuperscriptsubscript𝑔italic-ϕRGBsubscriptsuperscript𝐈RGBcompsuperscriptsubscript𝑔italic-ϕRGBsubscript𝐌obj\mathbf{c}\!\!\!=\!\!\!(\mathbf{y}_{\text{comp}},\mathbf{y}_{\text{obj}})\!\!=\!\!(g_{\phi}^{\text{RGB}}(\mathbf{I}^{\text{RGB}}_{\text{comp}}),g_{\phi}^{\text{RGB}}(\mathbf{M}_{\text{obj}})). By taking the conditional image embeddings, the model targets to denoise the noisy latent 𝐱t=(𝐱tbg,𝐱tfg)subscript𝐱𝑡superscriptsubscript𝐱𝑡bgsuperscriptsubscript𝐱𝑡fg\mathbf{x}_{t}\!\!=\!\!(\mathbf{x}_{t}^{\text{bg}},\mathbf{x}_{t}^{\text{fg}}), to recover the latents of the clean background image and the transparent foreground layer 𝐱0=(𝐱0bg,𝐱0fg)=(gϕRGB​(𝐈bgRGB),gψRGBA​(𝐈fgRGBA))subscript𝐱0superscriptsubscript𝐱0bgsuperscriptsubscript𝐱0fgsuperscriptsubscript𝑔italic-ϕRGBsubscriptsuperscript𝐈RGBbgsuperscriptsubscript𝑔𝜓RGBAsubscriptsuperscript𝐈RGBAfg\mathbf{x}_{0}\!\!=\!\!(\mathbf{x}_{0}^{\text{bg}},\mathbf{x}_{0}^{\text{fg}})\!=\!(g_{\phi}^{\text{RGB}}(\mathbf{I}^{\text{RGB}}_{\text{bg}}),g_{\psi}^{\text{RGBA}}(\mathbf{I}^{\text{RGBA}}_{\text{fg}})). The training loss follows the standard denoising diffusion loss [13, 20]:

ℒdm=𝔼t∼𝒰​({1,…,T}),ϵ,𝐱t​[‖ϵθ​(𝐱t;𝐜,t)−ϵ‖22],subscriptℒdmsubscript𝔼similar-to𝑡𝒰1…𝑇bold-italic-ϵsubscript𝐱𝑡delimited-[]superscriptsubscriptnormsubscriptbold-italic-ϵ𝜃subscript𝐱𝑡𝐜𝑡bold-italic-ϵ22\displaystyle\mathcal{L}_{\mathrm{dm}}=\mathbb{E}_{t\sim\mathcal{U}(\{1,...,T\}),\bm{\epsilon},\mathbf{x}_{t}}\left[\|\bm{\epsilon}_{\theta}(\mathbf{x}_{t};\mathbf{c},t)-\bm{\epsilon}\|_{2}^{2}\right],

(1)

s.t.𝐱0∼qdata​(𝐱0),ϵ∼𝒩​(𝟎,I),𝐱t=αt​𝐱0+1−αt​ϵ.formulae-sequence𝑠𝑡formulae-sequencesimilar-tosubscript𝐱0subscript𝑞datasubscript𝐱0formulae-sequencesimilar-tobold-italic-ϵ𝒩0𝐼subscript𝐱𝑡subscript𝛼𝑡subscript𝐱01subscript𝛼𝑡bold-italic-ϵ\displaystyle s.t.~{}~{}\mathbf{x}_{0}\!\sim\!q_{\text{data}}(\mathbf{x}_{0}),\bm{\epsilon}\!\sim\!\mathcal{N}(\bm{0},I),\mathbf{x}_{t}=\sqrt{\alpha_{t}}\mathbf{x}_{0}+\sqrt{1-\alpha_{t}}\bm{\epsilon}.

Specifically, the noisy input 𝐱tsubscript𝐱𝑡\mathbf{x}_{t} and image conditions 𝐜𝐜\mathbf{c} are initially divided into non-overlapping patches and converted into patch embeddings. The patch embeddings of each type of images, such as background, foreground or any conditions, are added with a corresponding type embedding and then concatenated into a sequence. Subsequently, the model follows the standard DiT architecture, where the patch embeddings are processed through multiple transformer blocks. With such design, the image conditions provide comprehensive contextual information through the self-attention in the transformer blocks to enhance the denoising, and the loss is only computed on the positions corresponding to the noisy latents.

Note that the latent of the foreground image needs to be encoded with RGBA channels, we leverage an RGBA-VAE fine-tuned from the original VAE by following LayerDiffusion [49] which makes minimal disturbance to the original latent space. Moreover, as the foreground image is not always available, e.g., in the case of real-world camera-captured data, the noisy input and output corresponding to the foreground are masked out from ℒdmsubscriptℒdm\mathcal{L}_{\mathrm{dm}} computation if 𝐈fgRGBAsubscriptsuperscript𝐈RGBAfg\mathbf{I}^{\text{RGBA}}_{\text{fg}} is absent in the training stage.

### 3.2 Consistency Loss for Visual Effects Learning

To handle cases where real-world data lacks ground-truth annotations, we introduce a consistency loss that enables the learning of natural visual effects in the transparent foreground layer without explicit annotation. Intuitively, as shown in Fig 2, the consistency loss is applied in the decoded pixel space to encourage the predicted foreground can faithfully reconstruct the composite input after blending with the background layers.

More specifically, given a composite image
𝐈compRGBsubscriptsuperscript𝐈RGBcomp\mathbf{{I}}^{\text{RGB}}_{\text{comp}}, at any denoising timestep t𝑡t, we reparameterize our model prediction back to the estimation of the clean latent 𝐱0subscript𝐱0\mathbf{x}_{0} as:

𝐱^0​(𝐱t;𝐜,t)=1αt​(𝐱t−1−αt⋅ϵθ​(𝐱t;𝐜,t)).subscript^𝐱0subscript𝐱𝑡𝐜𝑡1subscript𝛼𝑡subscript𝐱𝑡⋅1subscript𝛼𝑡subscriptbold-italic-ϵ𝜃subscript𝐱𝑡𝐜𝑡\hat{\mathbf{x}}_{0}(\mathbf{x}_{t};\mathbf{c},t)=\frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t}-\sqrt{1-\alpha_{t}}\cdot\bm{\epsilon}_{\theta}(\mathbf{x}_{t};\mathbf{c},t)\right).

(2)

Given Eq. 2, we compute the estimated 𝐱^0​(𝐱t;𝐜,t)=(𝐱^0bg,𝐱^0fg)subscript^𝐱0subscript𝐱𝑡𝐜𝑡superscriptsubscript^𝐱0bgsuperscriptsubscript^𝐱0fg\hat{\mathbf{x}}_{0}(\mathbf{x}_{t};\mathbf{c},t)=(\hat{\mathbf{x}}_{0}^{\text{bg}},\hat{\mathbf{x}}_{0}^{\text{fg}}) of background and foreground at time step t𝑡t and decode them into pixel space to get 𝐈^bgRGB=hϕ′RGB​(𝐱^0bg)subscriptsuperscript^𝐈RGBbgsuperscriptsubscriptℎsuperscriptitalic-ϕ′RGBsuperscriptsubscript^𝐱0bg\mathbf{\hat{I}}^{\text{RGB}}_{\text{bg}}=h_{\phi^{\prime}}^{\text{RGB}}(\hat{\mathbf{x}}_{0}^{\text{bg}}) and 𝐈^fgRGBA=hψ′RGBA​(𝐱^0fg)subscriptsuperscript^𝐈RGBAfgsuperscriptsubscriptℎsuperscript𝜓′RGBAsuperscriptsubscript^𝐱0fg\mathbf{\hat{I}}^{\text{RGBA}}_{\text{fg}}=h_{\psi^{\prime}}^{\text{RGBA}}(\hat{\mathbf{x}}_{0}^{\text{fg}}), via the decoder of RGB-VAE and RGBA-VAE, respectively. The results are combined through alpha blending to produce the estimated composite image 𝐈^compRGB=𝒜​(𝐈^bgRGB,𝐈^fgRGBA)subscriptsuperscript^𝐈RGBcomp𝒜subscriptsuperscript^𝐈RGBbgsubscriptsuperscript^𝐈RGBAfg\mathbf{\hat{I}}^{\text{RGB}}_{\text{comp}}=\mathcal{A}(\mathbf{\hat{I}}^{\text{RGB}}_{\text{bg}},\mathbf{\hat{I}}^{\text{RGBA}}_{\text{fg}}). The consistency loss is thus:

ℒconsist=𝔼t​∑i=1H∑j=1W|𝐈compRGB​(i,j)−𝐈^compRGB​(i,j)|,subscriptℒconsistsubscript𝔼𝑡superscriptsubscript𝑖1𝐻superscriptsubscript𝑗1𝑊subscriptsuperscript𝐈RGBcomp𝑖𝑗subscriptsuperscript^𝐈RGBcomp𝑖𝑗\mathcal{L}_{\text{consist}}=\mathbb{E}_{t}\sum_{i=1}^{H}\sum_{j=1}^{W}\left|\mathbf{{I}}^{\text{RGB}}_{\text{comp}}(i,j)-\mathbf{\hat{I}}^{\text{RGB}}_{\text{comp}}(i,j)\right|,

(3)

where H𝐻H and W𝑊W indicates the height and the width of the composite image, respectively.

The consistency loss enables LayerDecomp to learn faithful representations of transparent visual effects in the foreground layer, which is essential for accurately decomposing natural shadows and reflections in real-world data, especially in the absence of ground-truth annotations.

### 3.3 Dataset Preparation

To effectively train LayerDecomp, we curated a hybrid dataset that combines simulated and real-world data. Ideally, training LayerDecomp requires image triplets: an input image 𝐈compRGBsubscriptsuperscript𝐈RGBcomp\mathbf{I}^{\text{RGB}}_{\text{comp}}, a transparent foreground layer containing the target object and its visual effects 𝐈fgRGBAsubscriptsuperscript𝐈RGBAfg\mathbf{I}^{\text{RGBA}}_{\text{fg}}, and a background image without the foreground object 𝐈bgRGBsubscriptsuperscript𝐈RGBbg\mathbf{I}^{\text{RGB}}_{\text{bg}}. While collecting natural triplet images with specialized devices or through manual annotation might be feasible, it is costly and impractical for large-scale data needs. Conversely, synthesizing such triplet data directly with generative models presents significant challenges. Observations from existing approaches, such as HQ-Edit [14] and LayerDiffusion [49], indicate that generative models often inadvertently modify areas outside the target foreground, making it difficult to produce truly aligned image layers with consistent content. Additionally, accurately representing transparent visual effects, such as shadows and reflections in the foreground layer with an alpha channel for transparency, remains unexplored in existing works. To address these limitations, we developed a simulated data pipeline to create triplet images and supplemented it with a smaller portion of real-world 𝐈compRGBsubscriptsuperscript𝐈RGBcomp\mathbf{I}^{\text{RGB}}_{\text{comp}} and 𝐈bgRGBsubscriptsuperscript𝐈RGBbg\mathbf{I}^{\text{RGB}}_{\text{bg}} pairs to enhance robustness.

Simulated Data: To create image triplets, we first collected a large-scale object assets consisting of unoccluded foreground objects with synthesized shadows. We used entity segmentation [31] to select “thing” objects from natural images, and applied depth estimation to infer occlusion relations to exclude incomplete objects. We then applied a shadow synthesis method [40] to generate a shadow intensity map for each object on a white background. By integrating the intensity map into the alpha channel, we obtained comprehensive object assets in RGBA format. During training, we adjusted the scale and position of each foreground asset to align with the properties of a randomly selected background image 𝐈bgRGBsubscriptsuperscript𝐈RGBbg\mathbf{I}^{\text{RGB}}_{\text{bg}}, resulting in a finalized foreground layer 𝐈fgRGBAsubscriptsuperscript𝐈RGBAfg\mathbf{I}^{\text{RGBA}}_{\text{fg}}. By blending the two layers together, we obtained a composite image 𝐈compRGBsubscriptsuperscript𝐈RGBcomp\mathbf{I}^{\text{RGB}}_{\text{comp}}, completing the triplet data needed for model training. Although the composite results may lack fully realistic geometry and harmonized content, this approach enables large-scale training and allows the model to learn the appropriate representations for the two output layers in the decomposition task.

Camera-Captured Data: We also include a small set of real-world camera-captured image pairs, denoted as 𝐈comRGBsubscriptsuperscript𝐈RGBcom\mathbf{I}^{\text{RGB}}_{\text{com}} and 𝐈bgRGBsubscriptsuperscript𝐈RGBbg\mathbf{I}^{\text{RGB}}_{\text{bg}}, similar to the counterfactual dataset proposed by ObjectDrop [41]. The real-world data enhances the model’s ability to generalize to natural images containing authentic shadows and a broader range of visual effects, such as reflections, which are crucial for accurate foreground-background decomposition in real-world tasks.

## 4 Experiments

Implementation. LayerDecomp is finetuned from a 5 billion-parameter DiT model pre-trained for text-to-image generation. For layer decomposition task, the text encoder is dropped and the model does not take text input. RGBA-VAE is finetuned from the DiT VAE using a combination of L1 loss, GAN loss, and perceptual loss [51]. It takes images with 512×512512512512\times 512 resolution and encode them into 64×64646464\times 64 latent feature maps. Image type embedding is learnable through linear layers. Our simulated dataset is built from a large corpus of stock images, and our camera-captured dataset consists of 6,00060006,000 image pairs. During training, we use the Adam optimizer and set the learning rate at 1​e​-​51𝑒-51e\text{-}5. Training is conducted with a total batch size of 128128128 on 161616 A100 GPUs for 80,0008000080,000 iterations. During inference, all results are generated using DDIM sampling with 505050 steps.

Model

PSNR ↑↑\uparrow

LPIPS ↓↓\downarrow

FID ↓↓\downarrow

CLIP-FID ↓↓\downarrow

BG
Comp
BG
Comp
BG
Comp
BG
Comp

V0subscript𝑉0V_{0}:RGB-only

28.21
-
0.0732
-
21.00
-
4.551
-

V1subscript𝑉1V_{1}:V0+limit-fromsubscript𝑉0V_{0}+RGBA FG (obj.)

28.2828.2828.28
27.5327.5327.53
0.07080.07080.0708
0.06490.06490.0649
18.4818.4818.48
18.8318.8318.83
2.4872.4872.487
2.3292.3292.329

V2subscript𝑉2V_{2}:V0+limit-fromsubscript𝑉0V_{0}+RGBA FG (obj.+v.e.)

28.5628.5628.56
28.6628.6628.66
0.06910.06910.0691
0.06120.06120.0612
17.9917.9917.99
16.8716.8716.87
2.5392.5392.539
2.1722.1722.172

Ours:V2+ℒconsistsubscript𝑉2subscriptℒconsistV_{2}+\mathcal{L}_{\text{consist}}

29.2729.2729.27
30.5330.5330.53
0.06180.06180.0618
0.04940.04940.0494
16.0416.0416.04
12.7512.7512.75
1.8131.8131.813
1.5641.5641.564

RORD [34]

MULAN [39]

DESOBAv2 [21]

Model
PSNR ↑↑\uparrow
LPIPS ↓↓\downarrow
FID ↓↓\downarrow
CLIP-FID ↓↓\downarrow
PSNR ↑↑\uparrow
LPIPS ↓↓\downarrow
FID ↓↓\downarrow
CLIP-FID ↓↓\downarrow
PSNRm ↑↑\uparrow
SSIMm ↑↑\uparrow

CNI [50]

20.45L​22.01Tsuperscript20.45𝐿superscript22.01𝑇20.45^{L}22.01^{T}
0.235L​0.182Tsuperscript0.235𝐿superscript0.182𝑇0.235^{L}0.182^{T}
50.40L​53.71Tsuperscript50.40𝐿superscript53.71𝑇50.40^{L}53.71^{T}
8.853L​9.262Tsuperscript8.853𝐿superscript9.262𝑇8.853^{L}9.262^{T}
17.7917.7917.79
0.3210.3210.321
65.0365.0365.03
9.3969.3969.396
36.94Lsuperscript36.94𝐿36.94^{L}
0.491Lsuperscript0.491𝐿0.491^{L}

SDI [33]

19.88L​20.81Tsuperscript19.88𝐿superscript20.81𝑇19.88^{L}20.81^{T}
0.205L​0.166Tsuperscript0.205𝐿superscript0.166𝑇0.205^{L}0.166^{T}
53.73L​56.28Tsuperscript53.73𝐿superscript56.28𝑇53.73^{L}56.28^{T}
11.38L​11.10Tsuperscript11.38𝐿superscript11.10𝑇11.38^{L}11.10^{T}
16.0416.0416.04
0.3030.3030.303
65.7465.7465.74
11.5411.5411.54
34.21Lsuperscript34.21𝐿34.21^{L}
0.527Lsuperscript0.527𝐿0.527^{L}

PP [55]

20.88L​21.26Tsuperscript20.88𝐿superscript21.26𝑇20.88^{L}21.26^{T}
0.231L​0.201Tsuperscript0.231𝐿superscript0.201𝑇0.231^{L}0.201^{T}
39.48L​56.56Tsuperscript39.48𝐿superscript56.56𝑇39.48^{L}56.56^{T}
8.596L​11.32Tsuperscript8.596𝐿superscript11.32𝑇8.596^{L}11.32^{T}
17.1717.1717.17
0.3140.3140.314
55.8055.8055.80
9.9889.9889.988
29.33Lsuperscript29.33𝐿29.33^{L}
0.369Lsuperscript0.369𝐿0.369^{L}

Ours
24.56L​24.79Tsuperscript24.56𝐿superscript24.79𝑇\mathbf{24.56}^{L}\mathbf{24.79}^{T}
0.133L​0.132Tsuperscript0.133𝐿superscript0.132𝑇\mathbf{0.133}^{L}\mathbf{0.132}^{T}
21.77L​21.73Tsuperscript21.77𝐿superscript21.73𝑇\mathbf{21.77}^{L}\mathbf{21.73}^{T}
5.735L​5.778Tsuperscript5.735𝐿superscript5.778𝑇\mathbf{5.735}^{L}\mathbf{5.778}^{T}
19.1319.13\mathbf{19.13}
0.2440.244\mathbf{0.244}
39.2639.26\mathbf{39.26}
6.3326.332\mathbf{6.332}
38.57Tsuperscript38.57𝑇\mathbf{38.57}^{T}
0.640Tsuperscript0.640𝑇\mathbf{0.640}^{T}

### 4.1 Ablations

To quantitatively assess the advantages of incorporating visual effects in the foreground layer and the effectiveness of our proposed consistency loss, we construct a held-out evaluation dataset of 635635635 images for ablation studies. This dataset includes camera-captured composite images with the corresponding backgrounds. Visual examples will be provided in the supplementary materials.
The quality of the decomposed background layers can be directly evaluated using standard metrics, such as PSNR, LPIPS [51], FID [12, 29], and CLIP-FID [29]. To further evaluate the quality of the decomposed foreground layers, we apply alpha blending to re-composite the background and foreground together, comparing the result with the original composite image. As shown in Tab. 1, compared to a naïve DiT baseline which outputs only an RGB background, adding an RGBA foreground layer not only enables decomposition but also improves background quality. Incorporating visual effects in the foreground layer and introducing a consistency loss further enhance model performance. This demonstrates the decomposition task may implicitly improve the model’s understanding of the input scene, leading to superior results in both layers’ predictions.

### 4.2 Comparison on Object Removal

To assess the quality of the background layers predicted by LayerDecomp, we evaluate the model on the object removal task, comparing it to several state-of-the-art approaches, including mask-based methods (ControlNet Inpainting [50], SD-XL Inpainting [33], PowerPaint [55], ObjectDrop [41]) and instruction-driven models (Emu-Edit [36], MGIE [9], and OmniGen [44]).

Quantitative evaluation among mask-based inpainting methods is conducted on three public benchmarks: RORD [34], a real-world object removal dataset collected from video data with human-labeled loose and tight object masks; MULAN [39], a synthesized multi-layer dataset that provides instance-wise RGBA decompositions for COCO and LAION images; and DESOBAv2 [21], a real-world image dataset with shadow mask annotations and synthesized image pairs where instance shadows are removed. Standard metrics, including PSNR, LPIPS, FID, and CLIP-FID, are used on RORD and MULAN, while regional similarity such as masked PSNR and masked SSIM are used for shadow removal in DESOBAv2. More details of dataset preparation will be included in the supplementary materials.

Most existing mask-based object removal approaches cannot automatically detect and remove visual effects associated with the target object, necessitating a loose mask input. Indeed, as shown in Tab. 2, using a loose mask introduces more inpainting area and thus hurts PSNR and LPIPS, but significantly improves result fidelity (i.e., FID and CLIP-FID) for all compared methods. In contrast, LayerDecomp demonstrates robustness to mask tightness and outperforms all methods by a large margin. For shadow removal on DESOBAv2, LayerDecomp surpasses other methods even without utilizing the shadow annotation included in the loose mask. As shown in Fig. 3, LayerDecomp provides substantial improvements over existing methods, generating more photorealistic background layers with fewer artifacts and minimal visual effect residues. Comparing with ObjectDrop, a leading model in object removal that can automatically eliminate visual effect with a tight input mask, we applied LayerDecomp to the example images released by their work. As shown in Fig. 4, LayerDecomp achieves similar high-quality backgrounds with effective object removal. Additionally, LayerDecomp decomposes the foreground with realistic visual effects, enabling further editing capabilities not supported by ObjectDrop. More visual comparisons will be included in the supplementary materials.

Methods

Removal
Effectiveness

Result
Plausibility

Background
Integrity

Overall

Emu-Edit [36]

5.00%percent5.005.00\%
4.38%percent4.384.38\%
3.33%percent3.333.33\%
4.79%percent4.794.79\%

Ours
57.08%percent57.0857.08\%
77.92%percent77.9277.92\%
76.25%percent76.2576.25\%
83.54%percent83.5483.54\%

OmniGen [44]

4.07%percent4.074.07\%
3.15%percent3.153.15\%
2.96%percent2.962.96\%
3.89%percent3.893.89\%

Ours
67.04%percent67.0467.04\%
80.56%percent80.5680.56\%
84.63%percent84.6384.63\%
87.78%percent87.7887.78\%

To compare with instruction-driven methods, we conduct a user study on 606060 randomly selected images from the Emu-Edit Remove Set [36]. We ask 171717 independent researchers to compare results from LayerDecomp and an existing method, focusing on three quality aspects: removal effectiveness, result plausibility, and background integrity. In cases where both methods achieve satisfactory results, users could mark it as a “tie”. For a fair comparison, we use a grounding model to generate text-based masks to input to LayerDecomp. In total, 306030603060 data points are collected in this study. As shown in Tab. 3, LayerDecomp is clearly preferred in at least 83%percent8383\% of the testing cases for overall quality. Example object removal results are visualized in Fig. 5. While Emu-Edit and MGIE struggle to fully remove the target object, OmniGen is more effective but does not reliably preserve the background integrity. Our model, in contrast, successfully removes the target object while preserving most background fine details.

### 4.3 Comparison on Object Spatial Editing

We further compare LayerDecomp for object spatial editing tasks, including object moving and resizing, against several state-of-the-art methods: DiffEditor [26], DragAnything [42], DesignEdit [15], and Diffusion Handles [28]. For user study, we select 151515 editing samples using web images for each task, and ask 232323 independent researcher to compare results from LayerDecomp and an existing method, focusing on three quality aspects: edit effectiveness, result plausibility, and content integrity. In this study, 207020702070 data points are collected. As shown in Tab. 4, LayerDecomp is at least 87%percent8787\% more preferred in the spatial editing tasks, featuring superior plausibility in editing results. As shown in Fig. 6, LayerDecomp enables seamless spatial editing for objects in various scenes. By preserving intact visual effects in the transparent foreground layer, shadows and reflections move naturally with the editing targets, allowing harmonious re-composition to be effortlessly achieved through alpha blending. When compared to DiffusionHandle [28] and DesignEdit [15] on their released examples, LayerDecomp demonstrates comparable results in most scenarios, including graphic design examples without requiring additional fine-tuning.

Methods

Edit
Effectiveness

Result
Plausibility

Content
Integrity

Overall

Moving

DesignEdit [15]

1.67%percent1.671.67\%
1.11%percent1.111.11\%
1.11%percent1.111.11\%
2.22%percent2.222.22\%

Ours
71.67%percent71.6771.67\%
90.56%percent90.5690.56\%
77.22%percent77.2277.22\%
94.44%percent94.4494.44\%

DragAnything [42]

1.82%percent1.821.82\%
1.21%percent1.211.21\%
2.42%percent2.422.42\%
1.21%percent1.211.21\%

Ours
67.88%percent67.8867.88\%
93.33%percent93.3393.33\%
95.15%percent95.1595.15\%
96.36%percent96.3696.36\%

Resizing

DesignEdit [15]

2.22%percent2.222.22\%
3.89%percent3.893.89\%
3.33%percent3.333.33\%
3.89%percent3.893.89\%

Ours
69.44%percent69.4469.44\%
82.78%percent82.7882.78\%
71.67%percent71.6771.67\%
87.22%percent87.2287.22\%

DiffEditor [26]

1.11%percent1.111.11\%
1.11%percent1.111.11\%
1.21%percent1.211.21\%
1.11%percent1.111.11\%

Ours
95.15%percent95.1595.15\%
96.36%percent96.3696.36\%
92.21%percent92.2192.21\%
96.36%percent96.3696.36\%

### 4.4 Multi-Layer Decomposition and Editing

LayerDecomp can be applied sequentially to an original input image with different instance masks, decomposing multiple layers along with their visual effects, as examples shown in Fig. 7.
This process enables creative and complex layer-based editing for each individual layer, including spatial manipulation, recoloring, and filtering. Once editing is complete, the re-composition of all layers maintains a natural and realistic appearance, as demonstrated in Fig. 7.

## 5 Discussions

In conclusion, our model achieves high-quality image layer decomposition and outperforms existing state-of-the-art methods in object removal and spatial editing across multiple benchmarks and user studies. By decomposing images into a photorealistic background an a transparent foreground with faithfully preserved visual effects, our model unlocks various creative possibilities for layer-wise image editing.
The proposed consistency loss enables effective learning of accurate representations for the transparent foreground layer with visual effects, even in the absence of ground-truth data which is challenging to collect from real-world images. While our current dataset preparation pipeline focuses primarily on common visual effects, such as shadows and reflections, extending this work to include other effects, such as smoke and mist, remains an exciting avenue for future exploration.

## References

- Alzayer et al. [2024]

Hadi Alzayer, Zhihao Xia, Xuaner Zhang, Eli Shechtman, Jia-Bin Huang, and Michael Gharbi.

Magic fixup: Streamlining photo editing by watching dynamic videos.

arXiv preprint arXiv:2403.13044, 2024.

- Avrahami et al. [2022]

Omri Avrahami, Dani Lischinski, and Ohad Fried.

Blended diffusion for text-driven editing of natural images.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18208–18218, 2022.

- Betker et al. [2023]

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al.

Improving image generation with better captions.

Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

- Brooks et al. [2022]

Tim Brooks, Aleksander Holynski, and Alexei A Efros.

Instructpix2pix: Learning to follow image editing instructions.

arXiv preprint arXiv:2211.09800, 2022.

- Brooks et al. [2023]

Tim Brooks, Aleksander Holynski, and Alexei A Efros.

Instructpix2pix: Learning to follow image editing instructions.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023.

- Burgert et al. [2024]

Ryan D Burgert, Brian L Price, Jason Kuen, Yijun Li, and Michael S Ryoo.

Magick: A large-scale captioned dataset from matting generated images using chroma keying.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22595–22604, 2024.

- Esser et al. [2024]

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al.

Scaling rectified flow transformers for high-resolution image synthesis.

In Forty-first International Conference on Machine Learning, 2024.

- Facebook AI [2024]

Facebook AI.

Emu edit test set generations.

https://huggingface.co/datasets/facebook/emu_edit_test_set_generations, 2024.

Accessed: 2024-11-20.

- Fu et al. [2024]

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan.

Guiding instruction-based image editing via multimodal large language models.

In International Conference on Learning Representations (ICLR), 2024.

- Gu et al. [2024]

Jing Gu, Yilin Wang, Nanxuan Zhao, Tsu-Jui Fu, Wei Xiong, Qing Liu, Zhifei Zhang, He Zhang, Jianming Zhang, HyunJoon Jung, et al.

Photoswap: Personalized subject swapping in images.

In Advances in Neural Information Processing Systems, 2024.

- Hertz et al. [2022]

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or.

Prompt-to-prompt image editing with cross attention control.

arXiv preprint arXiv:2208.01626, 2022.

- Heusel et al. [2017]

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.

Gans trained by a two time-scale update rule converge to a local nash equilibrium.

Advances in neural information processing systems, 30, 2017.

- Ho et al. [2020]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

Advances in neural information processing systems, 33:6840–6851, 2020.

- Hui et al. [2024]

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie.

Hq-edit: A high-quality dataset for instruction-based image editing.

arXiv preprint arXiv:2404.09990, 2024.

- Jia et al. [2024]

Yueru Jia, Yuhui Yuan, Aosong Cheng, Chuke Wang, Ji Li, Huizhu Jia, and Shanghang Zhang.

Designedit: Multi-layered latent decomposition and fusion for unified & accurate image editing.

arXiv preprint arXiv:2403.14487, 2024.

- Ju et al. [2024]

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu.

Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion.

arXiv preprint arXiv:2403.06976, 2024.

- Kawar et al. [2023]

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani.

Imagic: Text-based real image editing with diffusion models.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.

- Li et al. [2023]

Shufan Li, Harkanwar Singh, and Aditya Grover.

Instructany2pix: Flexible visual editing via multimodal instruction following.

arXiv preprint arXiv:2312.06738, 2023.

- Lin et al. [2014]

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.

Microsoft coco: Common objects in context.

In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

- Lipman et al. [2022]

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

Flow matching for generative modeling.

arXiv preprint arXiv:2210.02747, 2022.

- Liu et al. [2023]

Qingyang Liu, Jianting Wang, and Li Niu.

Desobav2: Towards large-scale real-world dataset for shadow generation.

arXiv preprint arXiv:2308.09972, 2023.

- Liu et al. [2024]

Zhengzhe Liu, Qing Liu, Chirui Chang, Jianming Zhang, Daniil Pakhomov, Haitian Zheng, Zhe Lin, Daniel Cohen-Or, and Chi-Wing Fu.

Object-level scene deocclusion.

In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.

- Lugmayr et al. [2022]

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool.

Repaint: Inpainting using denoising diffusion probabilistic models.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022.

- Manukyan et al. [2023]

Hayk Manukyan, Andranik Sargsyan, Barsegh Atanyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi.

Hd-painter: high-resolution and prompt-faithful text-guided image inpainting with diffusion models.

arXiv preprint arXiv:2312.14091, 2023.

- Meng et al. [2021]

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon.

Sdedit: Guided image synthesis and editing with stochastic differential equations.

arXiv preprint arXiv:2108.01073, 2021.

- Mou et al. [2024]

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang.

Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8488–8497, 2024.

- Pan et al. [2023]

Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt.

Drag your gan: Interactive point-based manipulation on the generative image manifold.

In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–11, 2023.

- Pandey et al. [2024]

Karran Pandey, Paul Guerrero, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, and Niloy J Mitra.

Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7695–7704, 2024.

- Parmar et al. [2022]

Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu.

On aliased resizing and surprising subtleties in gan evaluation.

In CVPR, 2022.

- Peebles and Xie [2023]

William Peebles and Saining Xie.

Scalable diffusion models with transformers.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

- Qi et al. [2022]

Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Philip Torr, Zhe Lin, and Jiaya Jia.

Open world entity segmentation.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7):8743–8756, 2022.

- Quattrini et al. [2024]

Fabio Quattrini, Vittorio Pippi, Silvia Cascianelli, and Rita Cucchiara.

Alfie: Democratising rgba image generation with no $$$.

arXiv preprint arXiv:2408.14826, 2024.

- Rombach et al. [2022]

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

High-resolution image synthesis with latent diffusion models.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- Sagong et al. [2022]

Min-Cheol Sagong, Yoon-Jae Yeo, Seung-Won Jung, and Sung-Jea Ko.

Rord: A real-world object removal dataset.

In BMVC, page 542, 2022.

- Schuhmann et al. [2022]

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al.

Laion-5b: An open large-scale dataset for training next generation image-text models.

Advances in Neural Information Processing Systems, 35:25278–25294, 2022.

- Sheynin et al. [2024]

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman.

Emu edit: Precise image editing via recognition and generation tasks.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

- Song et al. [2023]

Yizhi Song, Zhifei Zhang, Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, and Daniel Aliaga.

Objectstitch: Object compositing with diffusion model.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18310–18319, 2023.

- Suvorov et al. [2022]

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky.

Resolution-robust large mask inpainting with fourier convolutions.

In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2149–2159, 2022.

- Tudosiu et al. [2024]

Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, and Sarah Parisot.

Mulan: A multi layer annotated dataset for controllable text-to-image generation.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22413–22422, 2024.

- under review [2024]

under review.

Metashadow: Object-centered shadow detection, removal, and synthesis.

In under review, 2024.

- Winter et al. [2024]

Daniel Winter, Matan Cohen, Shlomi Fruchter, Yael Pritch, Alex Rav-Acha, and Yedid Hoshen.

Objectdrop: Bootstrapping counterfactuals for photorealistic object removal and insertion.

arXiv preprint arXiv:2403.18818, 2024.

- Wu et al. [2024]

Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang.

Draganything: Motion control for anything using entity representation.

In European Conference on Computer Vision, pages 331–348. Springer, 2024.

- Wu et al. [2025]

Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang.

Draganything: Motion control for anything using entity representation.

In European Conference on Computer Vision, pages 331–348. Springer, 2025.

- Xiao et al. [2024]

Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu.

Omnigen: Unified image generation.

arXiv preprint arXiv:2409.11340, 2024.

- Yu et al. [2019]

Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang.

Free-form image inpainting with gated convolution.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 4471–4480, 2019.

- Zeng et al. [2020]

Yu Zeng, Zhe Lin, Jimei Yang, Jianming Zhang, Eli Shechtman, and Huchuan Lu.

High-resolution image inpainting with iterative confidence feedback and guided upsampling.

In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIX 16, pages 1–17. Springer, 2020.

- Zeng et al. [2021]

Yu Zeng, Zhe Lin, Huchuan Lu, and Vishal M Patel.

Cr-fill: Generative image inpainting with auxiliary contextual reconstruction.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 14164–14173, 2021.

- Zhang et al. [2024]

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su.

Magicbrush: A manually annotated dataset for instruction-guided image editing.

Advances in Neural Information Processing Systems, 36, 2024.

- Zhang and Agrawala [2024]

Lvmin Zhang and Maneesh Agrawala.

Transparent image layer diffusion using latent transparency.

arXiv preprint arXiv:2402.17113, 2024.

- Zhang et al. [2023]

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

Adding conditional control to text-to-image diffusion models.

In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

- Zhang et al. [2018]

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.

The unreasonable effectiveness of deep features as a perceptual metric.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

- Zhao et al. [2024]

Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang.

Ultraedit: Instruction-based fine-grained image editing at scale.

arXiv preprint arXiv:2407.05282, 2024.

- Zhao et al. [2021]

Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I Chang, and Yan Xu.

Large scale image completion via co-modulated generative adversarial networks.

In International Conference on Learning Representations (ICLR), 2021.

- Zheng et al. [2022]

Haitian Zheng, Zhe Lin, Jingwan Lu, Scott Cohen, Eli Shechtman, Connelly Barnes, Jianming Zhang, Ning Xu, Sohrab Amirghodsi, and Jiebo Luo.

Image inpainting with cascaded modulation gan and object-aware training.

In European Conference on Computer Vision, pages 277–296. Springer, 2022.

- Zhuang et al. [2023]

Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen.

A task is worth one word: Learning with task prompts for high-quality versatile image inpainting.

arXiv preprint arXiv:2312.03594, 2023.

\thetitle

Supplementary Material

Model

FID ↓↓\downarrow

CLIP-FID ↓↓\downarrow

V0subscript𝑉0V_{0}:RGB-only

-
-

V1subscript𝑉1V_{1}:V0+limit-fromsubscript𝑉0V_{0}+RGBA FG (obj.)

45.75845.75845.758
3.7563.7563.756

V2subscript𝑉2V_{2}:V0+limit-fromsubscript𝑉0V_{0}+RGBA FG (obj.+v.e.)

45.12345.12345.123
3.7393.7393.739

Ours:V2+ℒconsistsubscript𝑉2subscriptℒconsistV_{2}+\mathcal{L}_{\text{consist}}

44.26044.26044.260
3.1733.1733.173

## 6 Additional Results for the Ablation Study

Test Set Details. The test set is a held-out subset of our camera-captured data consisting of 635 image pairs (composite image and background image). To construct this dataset, we manually collected real-world examples comprising photos of scenes captured before and after the removal of an object, while ensuring all other elements in the scene remain unchanged. We also manually labeled the binary object mask for the removal target. As illustrated in Fig. 8, the dataset encompasses both indoor and outdoor scenarios, effectively reflecting real-world phenomena such as shadows and reflections. This test set allows us to evaluate not only the quality of the decomposed background naturally but also the quality of the foreground. By re-compositing the background and foreground layer output, we can effectively assess the fidelity and visual coherence of the foreground, including the visual effect components.

Qualitative Comparison. To more intuitively demonstrate the effectiveness of our design in LayerDecomp, in addition to the quantitative analysis represented in Table 1 of the main manuscript, we provide more visual results of the four model variants in Fig. 9. For each variant, we present the decomposed background and foreground layers, along with the re-composited image obtained by alpha blending the two layers. For the RGB-only model (V0subscript𝑉0V_{0}), which lacks an RGBA foreground, we show only the decomposed background for reference. From the visual comparison among V1subscript𝑉1V_{1}, V2subscript𝑉2V_{2}, and “Ours”, it is evident that our method, which leverages consistency loss to explicitly model visual effects in the foreground layer, produces: (i) background layers with cleaner removal and less artifacts, (ii) foreground layers with more accurate extraction of transparent visual effects, resulting in re-composited results that are more plausible and realistic.

Quantitative Comparison. To more comprehensively evaluate the quality of the decomposed foreground, we randomly move/resize the foreground prediction and then re-composite it onto the decomposed background to evaluate the fidelity of the resulting image. Specifically, there are three parameters to randomly adjust: Δ​XΔ𝑋\Delta{X}, Δ​YΔ𝑌\Delta{Y}, and Δ​SΔ𝑆\Delta{S}. Δ​X∈[−0.3,+0.3]Δ𝑋0.30.3\Delta{X}\in[-0.3,+0.3] and Δ​Y∈[−0.3,+0.3]Δ𝑌0.30.3\Delta{Y}\in[-0.3,+0.3] specify the horizontal and vertical location changes as proportions of the input dimensions, while Δ​S∈[0.5,1.5]Δ𝑆0.51.5\Delta{S}\in[0.5,1.5] specifies a scaling ratio w.r.t. the original size. For each image, we randomly select three parameters and apply the same adjustment to all model variants’ foreground prediction. The FID and CLIP-FID of the randomly re-composite images are reported in Table 5. Comparing with other model variants, leveraging consistency loss to explicitly model visual effects in the foreground layer indeed improves re-composition quality.

Model

FID ↓↓\downarrow

CLIP-FID ↓↓\downarrow

Emu-Edit [36]

47.55547.55547.555
6.7116.7116.711

OmniGen [44]

48.11648.11648.116
6.2836.2836.283

Ours
38.99838.99838.998
5.6225.6225.622

## 7 Additional Results for the Mask-Based Object Removal Experiment

Benchmarks Details. Here, we provide more details for the mask-based object removal benchmarks used to calculate the metrics presented in Table 2 of the main manuscript.

- •

RORD [34]: We randomly select 1,029 images from the original test set to reduce data redundancy caused by sampling from the same video. The dataset provides both manually labeled loose masks and tight masks for the real-world object removal task. The average area of the loose mask is 3.70 times that of the tight mask in each image. As shown in Fig. 10, RORD includes diverse indoor and outdoor scenes, featuring removal of various target objects with soft shadows or reflections in real-world settings.

- •

MULAN [39]: We randomly select 1,000 images from MULAN-COCO for our evaluation. For each image, the dataset provides multiple object layers in RGBA format, and we select the object in the top most layer as the removal target. To reduce hallucination problems in traditional inpainting methods caused by tight object mask, we further dilate the object mask by 10 pixels. As shown in Fig. 11, MULAN data also includes diverse indoor and outdoor scenes, featuring object removal in more cluttered settings.

- •

DESOBAv2 [21]: There are 750 images in the test set including binary object masks and paired shadow masks. We use the binary object masks as tight mask to input to LayerDecomp and merge the object mask and the corresponding shadow mask to create loose mask to input to other inpainting methods. Similarly, to reduce hallucination problems in traditional inpainting methods, the loose masks are further dilated by 10 pixels. The average area of the loose mask is 2.35 times that of the tight mask. As shown in Fig. 12, DESOBAv2 mostly features outdoor scenes with hard object shadows cast on surfaces with different materials and textures, adding more challenges to decompositing the visual effects.

More Visual Results.
More visual comparison with ControlNet Inpainting [50], SD-XL Inpainting [33], and PowerPaint [55] on the three public benchmarks is provided in Fig. 10, Fig. 11, and Fig. 12. It can be observed that, with the assistance of the loose mask, the three baselines are able to remove most parts of the target object. However, they struggle to eliminate it entirely and face challenges in removing the shadows associated with the target object. Additionally, achieving photorealistic background completion in human plausible style remains a significant challenge. In contrast, our model, using only the tight mask, performs consistently better across a wide range of data sources.

## 8 Additional Results for the Instruction-Driven Object Removal Experiment

Qualitative Comparison. Fig. 13 presents additional comparison results with instruction-driven methods on the object removal task on Emu-Edit Remove Set [36, 8]. Beyond showcasing the superior object removal performance of our model, these results further highlight its enhanced background integrity and completion capabilities.

Quantitative Comparison. We also perform a quantitative comparison with instruction-driven methods, specifically Emu-Edit [36] and OmniGen [44]. Using the released generation results from Emu-Edit Remove Set [8], we evaluate the performance based on FID and CLIP-FID metrics. For a fairer comparison, we use text-based masks as input to our model. As shown in Table 6, our model outperforms existing approaches by a large margin.

## 9 More Image Layer Decomposition Results from LayerDecomp

As shown in Fig. 14, we provide comprehensive visualization results from various data sources, including web images, public datasets, and the held-out test set. These results demonstrate that our model is robust across diverse scenarios.

Generated on Thu Dec 5 13:59:03 2024 by LaTeXML
