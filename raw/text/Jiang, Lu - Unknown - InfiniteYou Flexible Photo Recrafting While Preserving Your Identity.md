# Jiang, Lu - Unknown - InfiniteYou Flexible Photo Recrafting While Preserving Your Identity

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Jiang, Lu - Unknown - InfiniteYou Flexible Photo Recrafting While Preserving Your Identity.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2503.16418
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# InfiniteYou: Flexible Photo Recrafting While Preserving Your Identity

Liming Jiang    Qing Yan    Yumin Jia    Zichuan Liu    Hao Kang    Xin Lu

ByteDance Intelligent Creation

Project Page: https://bytedance.github.io/InfiniteYou

###### Abstract

Achieving flexible and high-fidelity identity-preserved image generation remains formidable, particularly with advanced Diffusion Transformers (DiTs) like FLUX. We introduce InfiniteYou (InfU), one of the earliest robust frameworks leveraging DiTs for this task. InfU addresses significant issues of existing methods, such as insufficient identity similarity, poor text-image alignment, and low generation quality and aesthetics. Central to InfU is InfuseNet, a component that injects identity features into the DiT base model via residual connections, enhancing identity similarity while maintaining generation capabilities. A multi-stage training strategy, including pretraining and supervised fine-tuning (SFT) with synthetic single-person-multiple-sample (SPMS) data, further improves text-image alignment, ameliorates image quality, and alleviates face copy-pasting. Extensive experiments demonstrate that InfU achieves state-of-the-art performance, surpassing existing baselines. In addition, the plug-and-play design of InfU ensures compatibility with various existing methods, offering a valuable contribution to the broader community. Code and model: https://github.com/bytedance/InfiniteYou.

## 1 Introduction

Identity-preserved image generation aims to recraft a photograph of a specific person using free-form text descriptions while preserving facial identity. This task is challenging but highly beneficial.
Previous methods [54, 51, 14] have been mainly developed on U-Net [42]-based text-to-image diffusion models [46, 17, 41], such as Stable Diffusion XL (SDXL) [38]. However, due to the limited generation capacity of the base model, the quality of generated images remains inadequate.
Recent strides in Diffusion Transformers (DiTs) [37, 12] have made remarkable progress in content creation.
In particular, the latest releases of state-of-the-art rectified flow DiTs, such as FLUX [26] and Stable Diffusion 3.5 [12], showcase incredible image generation quality.
Consequently, it is crucial to explore solutions that can leverage the immense potential of DiTs for downstream applications like identity-preserved image generation.

DiT-based identity-preserved image generation remains formidable due to several factors: the absence of customized module designs, difficulties in model scaling, and a lack of high-quality data. Thus, effective solutions for this task using state-of-the-art rectified flow [33, 4] DiTs, such as FLUX [26], are currently scarce in both academia and industry.
PuLID-FLUX, derived from PuLID [14], presented an initial attempt to develop an identity-preserved image generation model based on FLUX.
Other open-source efforts, including FLUX.1-dev IP-Adapters from InstantX [20] and XLabs-AI [3], are not tailored for human facial identities.
Nevertheless, existing methods fall short in three key aspects: 1) The identity similarity is not sufficient; 2) The text-image alignment and editability are poor, and the face copy-paste issue is evident; 3) The generation capability of FLUX is largely compromised, resulting in lower image quality and aesthetic appeal.

To address the aforementioned challenges, we propose a simple yet effective framework for identity-preserved image generation, namely InfiniteYou (InfU). This framework is designed to be systematic and robust, enabling flexible text-based photo re-creation for diverse identities, races, and age groups across various scenarios.
We introduce InfuseNet, a generalization of ControlNet [56], which ingests identity information along with the controlling conditions.
The projected identity features are injected by InfuseNet into the DiT base model through residual connections, thereby disentangling text and identity injections.
InfuseNet is both scalable and compatible, harnessing the powerful generation capabilities of DiTs.
The scale-up injection network and the delicate architecture design, equipped with large-scale model training, effectively enhance identity similarity.
To improve text-image alignment, image quality, and aesthetic appeal, we employ a multi-stage training strategy, including pretraining and supervised fine-tuning (SFT) [21, 49]. The SFT stage utilizes carefully designed synthetic single-person-multiple-sample (SPMS) data generation, leveraging our pretrained model itself and various off-the-shelf modules.
This strategy enhances the quantity, quality, aesthetics, and text-image alignment of the training data, thus improving overall model performance and alleviating the face copy-paste issue.
Thanks to the InfuseNet design, identity information is injected purely via residual connections between DiT blocks, unlike conventional practices [54, 51, 14] that directly modify attention [50] layers via IP-Adapter (IPA).
Consequently, the generation capability of the base model remains largely intact, allowing for the generation of high-quality and aesthetically pleasing images.
Moreover, InfU is plug-and-play and readily compatible with many other methods or plugins, thus offering significant value to the broader community.

Comprehensive experiments show that the proposed InfU framework achieves state-of-the-art performance
(see Figure 1), significantly surpassing existing baselines in identity similarity, text-image alignment, and overall image quality. Our main contributions are summarized as follows:

- •

We propose InfiniteYou (InfU), a versatile and robust DiT-based framework for flexible identity-preserved image generation under various scenarios.

- •

We introduce InfuseNet, a generalization of ControlNet, which effectively injects identity features into the DiT base model via residual connections, enhancing identity similarity with minimal impact on generation capabilities.

- •

We employ a multi-stage training strategy, including pretraining and supervised fine-tuning (SFT), using synthetic single-person-multiple-sample (SPMS) data generation. This approach significantly improves text-image alignment, image quality, and aesthetic appeal.

- •

The InfU module features a desirable plug-and-play design, compatible with many existing methods, thus providing a valuable contribution to the broader community.

## 2 Related Work

Text-to-image Diffusion Transformers (DiTs).
Diffusion models [46, 17, 47, 41] have become a standard paradigm given their incredible capability to produce high-fidelity images. Text-to-image generation aims to synthesize images through the denoising diffusion process [17, 41] from a Gaussian distribution given textual descriptions.
Conventional methods [41, 38, 45, 30] are based on U-Net [42], where the text representation is extracted by CLIP [39].
Recent DiTs [37], based on Vision Transformers (ViTs) [11, 9] and typically using text embeddings encoded by T5 [40] in addition to CLIP, have exhibited even higher generation capacity compared to U-Net architectures.
The latest releases of DiTs with rectified flows (RFs) [33, 4], such as Stable Diffusion 3.5 [12], Playground V3 [32], Recraft V3 [2], and FLUX.1 [26], have further shown their unprecedented generation quality.
The progress made by these DiTs naturally stimulates the development of customized approaches in their downstream applications. This highlights the significance of our work, reforming the architecture from U-Nets to DiTs for identity-preserved image generation.

Identity-preserved image generation.
Tuning-based methods for dentity-preserved image generation include [13, 43, 18, 44, 24]. However, their practical significance is hindered by low efficiency and high tuning cost due to their specificity to individual identities.
Therefore, tuning-free methods have become the mainstream practice for this task. A series of efforts, such as IP-Adapter [54], FastComposer [52], Photomaker [29], InstantID [51], FlashFace [57], Arc2Face [36], Imagine Yourself [15], and PuLID [14], typically employ additional trainable modules as adapters to inject identity information.
After training, these approaches can generate customized images without additional tuning, even when ingesting new subject samples. However, these methods are mainly developed for U-Net-based Stable Diffusion [41] and SDXL [38], where the limited generative capability of the base model inevitably constrains the quality of the generated images.
The remarkable achievements of DiTs highlight the importance of base model replacement, which remains underexplored.
PuLID-FLUX [14] made an initial attempt based on IP-Adapter trained with alignment and identity losses [14]. Other open-source efforts, including FLUX.1-dev IP-Adapters from InstantX [20] and XLabs-AI [3], were devised but not tailored for human faces.
Nevertheless, existing methods still face limitations: insufficient identity similarity, poor text-image alignment, face copy-paste issues, and compromised generation quality.
The proposed InfU effectively addresses these shortcomings.

## 3 Methodology

### 3.1 Preliminary

Simulation-free training of flows.
Following [12], generative models are defined to establish a transformation between samples x1x_{1} drawn from a noise distribution p1p_{1} to samples x0x_{0} drawn from a data distribution p0p_{0}, formulated through an ordinary differential equation (ODE),

d​yt=vΘ​(yt,t)​d​t,dy_{t}=v_{\Theta}\left(y_{t},t\right)\,dt,

(1)

where the velocity vv is parameterized by the neural network weights Θ\Theta. The previous work [7] proposed solving Eq. 1 directly using differentiable ODE solvers.
However, this method is computationally intensive, particularly for large neural network structures that parameterize vΘ​(yt,t)v_{\Theta}\left(y_{t},t\right).
A more efficient approach is to directly regress a vector field utu_{t} that defines a probability path [31] between p0p_{0} and p1p_{1}.
To formulate such a vector field utu_{t}, a forward process is defined that corresponds to a probability path ptp_{t} between p0p_{0} and p1=𝒩​(0,1)p_{1}=\mathcal{N}\left(0,1\right), expressed as

zt=at​x0+bt​ϵ,where​ϵ∼𝒩​(0,I).z_{t}=a_{t}x_{0}+b_{t}\epsilon,\quad\text{where}\;\epsilon\sim\mathcal{N}(0,I).

(2)

For a0=1,b0=0,a1=0a_{0}=1,b_{0}=0,a_{1}=0, and b1=1b_{1}=1, the marginals

pt​(zt)\displaystyle p_{t}\left(z_{t}\right)
=𝔼ϵ∼𝒩​(0,I)​pt​(zt|ϵ),\displaystyle=\mathbb{E}_{\epsilon\sim\mathcal{N}\left(0,I\right)}p_{t}\left(z_{t}|\epsilon\right),

(3)

align with the data and noise distributions. A marginal vector field utu_{t} can generate the marginal probability paths ptp_{t} using conditional vector fields ut(⋅|ϵ)u_{t}\left(\cdot|\epsilon\right):

ut​(z)=𝔼ϵ∼𝒩​(0,I)​ut​(z|ϵ)​pt​(z|ϵ)pt​(z).\displaystyle u_{t}\left(z\right)=\mathbb{E}_{\epsilon\sim\mathcal{N}\left(0,I\right)}u_{t}\left(z|\epsilon\right)\frac{p_{t}\left(z|\epsilon\right)}{p_{t}\left(z\right)}.

(4)

It is intractable to regress utu_{t} directly due to the marginalization in Eq. 4. Thus, we switch to a simple and tractable objective, i.e., Conditional Flow Matching [31, 12]:

ℒC​F​M=𝔼t,pt​(z|ϵ),p​(ϵ)∥vΘ(z,t)−ut(z|ϵ)∥22.\displaystyle\mathcal{L}_{CFM}=\mathbb{E}_{t,p_{t}\left(z|\epsilon\right),p\left(\epsilon\right)}\left\|v_{\Theta}\left(z,t\right)-u_{t}\left(z|\epsilon\right)\right\|_{2}^{2}.

(5)

Rectified Flow.
Rectified flows (RFs) [33, 4] define the forward process as straight paths between the data distribution and a standard Gaussian distribution, i.e.,

zt=(1−t)​x0+t​ϵ,z_{t}=\left(1-t\right)x_{0}+t\epsilon,

(6)

where ϵ∼𝒩​(0,I)\epsilon\sim\mathcal{N}\left(0,I\right). The network output directly parameterizes the velocity vΘv_{\Theta}.
We use ℒC​F​M\mathcal{L}_{CFM} (Eq. 5) as the loss objective.
Different flow trajectories and samplers are defined in [12], including logit-normal sampling, which we also employ in our model training.

Text-to-image DiTs.
Our general setup follows Stable Diffusion 3.5 [12] and FLUX [26], derived from Latent Diffusion Models (LDM) [41] for training text-to-image models in the latent space of a pretrained autoencoder.
Apart from encoding images into latent representations, we also encode the text conditioning ctextc_{\mathrm{text}} using pretrained, frozen text models.
We use FLUX [26] as our DiT base model, which uses T5-XXL [40] and CLIP [39] for text encoding.
FLUX uses a multimodal diffusion backbone, i.e., MMDiT [12]. Unlike traditional DiTs [37], MMDiT uses two separate sets of weights for the two modalities, given that text and image embeddings are conceptually different.
This setup is equivalent to having two independent Transformers for each modality, but combines the sequences via joint attention to ensure that both representations work in their own space while considering each other.
FLUX also applies several single DiT blocks [9] after MMDiT blocks.

In addition to text-conditional image generation, the proposed InfU method also injects human facial identity information cidc_{\mathrm{id}} to accommodate additional modalities.

### 3.2 Network Architecture

Conventional approaches [54, 51] for this task were primarily developed for U-Net-based diffusion models like SDXL [38]. However, the image quality generated by these methods remains inadequate (see Figure 2 (a)).
The significantly better performance of FLUX than SDXL in background clarity, human topology, small face quality, and overall appeal pinpoints the importance of DiT-based solutions. The proposed InfU is inspired by these efforts while presenting a novel solution built on DiTs. We focus on the development and comparison of DiT-based approaches due to their evident superiority over SDXL, as demonstrated.

Unlike common practices [54, 51, 14] that modify attention [50] layers via IP-Adapter (IPA) [54] to inject identity information, we observe the non-optimality of IPA and avoid using it. As shown in Figure 2 (b), IPA typically introduces side effects, such as degradation in text-image alignment, image quality and aesthetics.
We deduce that directly modifying the attention layers significantly compromises the generative capability of the base model. In addition, injecting text and identity information at the same positions (i.e., attention layers) may bring potential entanglement and conflict, thus harming overall performance. Therefore, we propose a novel alternative solution without IPA, mitigating these issues while maintaining high identity similarity.

The proposed InfU framework is illustrated in Figure 3. The DiT base model (e.g., FLUX) remains frozen during training and serves as the main branch for image generation. It ingests a noise map sampled from a standard Gaussian distribution, along with features from both the identity image and text prompt inputs, to generate an image that adheres to the text description while preserving the human facial identity through multiple denoising steps.
The text prompt is embedded by a frozen text encoder and then fed into the base model through attention layers [12]. Below, we detail our mechanism for injecting identity information.

We introduce InfuseNet, an important branch that injects identity and control signals (see Figure 3).
InfuseNet shares a similar structure with the DiT base model but contains fewer Transformer blocks.
We denote MM as the number of DiT blocks in the base model and NN as the number of DiT blocks in InfuseNet. We have M=N⋅iM=N\cdot i, where ii is the multiplication factor.
An optional control image, such as a five-facial-keypoint image, can be input into InfuseNet to control the generation position of the subject. If no control is needed, a pure black image can be used instead.
The identity image is encoded by a frozen face identity encoder into identity embeddings, which are fed into a projection network. This network projects the identity features and sends them to InfuseNet through attention layers, similar to how text features are handled in the DiT base model. InfuseNet then predicts the output residual connections of the DiT base model, contributing to the final image synthesis. Specifically, DiT block jj in InfuseNet predicts the residuals of the following DiT blocks in the base model:

(j−1)⋅i+1,(j−1)⋅i+2,…,j⋅i.\left(j-1\right)\cdot i+1,\;\;\left(j-1\right)\cdot i+2,\;\;\ldots,\;\;j\cdot i.

(7)

During training, the projection network and InfuseNet are trainable (using ℒC​F​M\mathcal{L}_{CFM} in Eq. 5), while other modules remain frozen. The proposed InfuseNet can be seen as a generalization of ControlNet [56], capable of ingesting more modalities to influence the generation process via residual connections. This residual injection of identity features is distinct from the text injection through attention layers, which effectively separates text and identity inputs, thereby reducing potential entanglement and conflict.
Thanks to this pure residual injection design that does not rely on IPA, the generative capability of the base model is less compromised, resulting in higher quality and improved text-image alignment.
InfuseNet is also based on DiT, and its similar architecture with the base model ensures scalability and compatibility.
The scalable network design and large-scale training enhance identity similarity.

### 3.3 Multi-Stage Training Strategy

Despite the robust network design of InfU, challenges in text-image alignment, generation aesthetics, and image quality degradation remain, especially in certain hard cases. This issue is critical for state-of-the-art approaches, necessitating a general solution to facilitate future research.

We devise a multi-stage training strategy, including pretraining and supervised fine-tuning (SFT) [21, 49].
This strategy improves the quantity, quality, aesthetics, and text-image alignment of training data, thereby enhancing overall model performance w.r.t. the above problems. The training strategy is formulated in the following steps (see Figure 4).

Step 1: We collect and filter real single-person-single-sample (SPSS) data from several human portrait datasets. The data, though not highly aesthetic or high-quality, can be used for stage-1 pretraining of our InfU model, following standard training practices [54, 51]. Using real SPSS data, we employ a single authentic portrait image as both the source identity image and the generation target image to learn reconstruction during training.

Step 2: After stage-1 pretraining of the InfU model, we conduct stage-1 model inference to evaluate image generation performance without any plugins, such as LoRA [18]. While the facial identity similarity of the generated results is satisfactory, there remains room for improvement in text-image alignment, generation aesthetics, and image quality.

Step 3: We then equip the stage-1 trained InfU model with a series of useful off-the-shelf modules, such as aesthetic modules/LoRAs, enhancement LoRAs, face swap modules [6], and other pre-/post-processing tools, etc. Although time-consuming and cumbersome, this process enables the model to generate synthetic data with much higher quality and aesthetics. We intentionally formulate the data as single-person-multiple-sample (SPMS), where a real face image serves as the source identity image and the synthetic data serves as the generation target image.

Step 4: The synthetic SPMS data is subsequently fed into the stage-1 trained InfU model for stage-2 supervised fine-tuning (SFT). Leveraging the properties of SPMS, we use real face data as the source identity and the paired high-quality synthetic data as the generation target for model training. Other training settings remain similar to those in stage 1. This SFT enables the model to learn the high quality and aesthetics of the synthetic data while retaining identity similarity with the real face input.

Step 5: After stage-2 SFT, the InfU model is ready for final inference and deployment. Without any plugins, the text-image alignment, generation aesthetics, and image quality of the generated results are significantly improved, while maintaining high facial identity similarity.

## 4 Experiments

### 4.1 Settings

Implementation details.
We implement our InfiniteYou (InfU) framework using PyTorch and leverage the Hugging Face Diffusers library.
The DiT base model is FLUX.1-dev [26].
We set the multiplication factor i=4i=4 for InfuseNet.
The projection network is derived from [54], with the token number of the projected identity feature set to 88.
All experiments are conducted using FSDP [59] on NVIDIA H100 GPUs, each with 8080GB VRAM. We use the AdamW [35] optimizer with β1=0.9\beta_{1}=0.9 and β2=0.999\beta_{2}=0.999. The weight decay is set to 0.010.01.
We employ Conditional Flow Matching [31, 12] (Eq. 5) as the loss function with logit-normal sampling [12] of rf/lognorm(0.00, 1.00).
For stage-1 pretraining, the model is trained using an initial learning rate of 2×10−52\times 10^{-5} on 128128 GPUs. The total batch size is set to 512512, and stage-1 training spans 300300k iterations.
For stage-2 supervised fine-tuning, the model is trained with an initial learning rate of 1×10−51\times 10^{-5} on 6464 GPUs, with a total batch size of 256256. All other settings remain unchanged.

Datasets.
For stage-1 pretraining, we use a total of nine open source datasets, including VGGFace2 [5], MillionCelebs [58], CelebA [34], CelebV-HQ [60], FFHQ [22], VFHQ [53], EasyPortrait [25], CelebV-Text [55], CosmicManHQ-1.0 [28], as well as several high-quality internal datasets.
We perform careful data pre-processing and filtering, removing images with low-quality small faces, multiple faces, watermarks, or NSFW content. The data is pre-processed for training using aspect ratio bucketing [1].
The total amount of single-person single-sample (SPSS) real data for stage-1 pre-training reaches 4343 million, which we consider sufficient for large-scale training of identity-preserved image generation models.
For stage-2 supervised fine-tuning, the total quantity of single-person-multiple-sample (SPMS) synthetic data is 22 million. All data is generated by the stage-1 pretrained InfU model itself, equipped with useful off-the-shelf modules (see Section 3.3). High-quality synthetic data are also carefully processed and filtered to obtain image pairs with normal poses, high ID resemblance, and good aesthetics, ensuring their usefulness.
In addition, we observe that training the model with a mixture of captions from multiple sources, e.g., humans, small captioning models, and large vision-language models (VLMs), is beneficial. Besides the original captions in the datasets, we employ BLIP-2 [27] and InternVL2 [8] to obtain text captions from diverse sources for training.

Baselines.
Since InfU is based on DiT (e.g., FLUX), we compare it with the most relevant and state-of-the-art DiT-based approach, PuLID-FLUX [14]. Other open-source efforts, including FLUX.1-dev IP-Adapters from InstantX [20] and XLabs-AI [3], are not tailored for human faces. We select the one from InstantX as a representative baseline of this series for a more comprehensive comparison. Other conventional methods based on SDXL display much lower image quality due to the limitation of the base model (see Figure 2) and are thus not fairly comparable.

Evaluation.
We conduct evaluations on a portrait benchmark created by GPT-4o [19], comprising 200200 prompts and corresponding gender information. This benchmark covers a variety of cases, including different prompt lengths, face sizes, views, scenes, ages, races, complexities, etc. We selected 1515 representative identity samples and paired their gender with all appropriate prompts, resulting in 1,4971,497 testing outputs for systematic evaluations.
We apply three representative and useful evaluation metrics, i.e., ID Loss [10], CLIPScore [16], and PickScore [23].
ID Loss is defined as 1−CosSim​(IDgen,IDref)1-\mathrm{CosSim}\left(\mathrm{ID}_{\text{gen}},\mathrm{ID}_{\text{ref}}\right), where CosSim\mathrm{CosSim} is cosine similarity, and IDgen\mathrm{ID}_{\text{gen}} and IDref\mathrm{ID}_{\text{ref}} are the generated and reference identity images, respectively. A lower ID Loss means higher similarity. We follow the original papers to use CLIPScore and PickScore. A higher CLIPScore indicates better text-image alignment, and a higher PickScore signifies better image quality and aesthetics.

### 4.2 Main Results

Qualitative comparison.
The qualitative comparison results are shown in Figure 5.
The identity similarity of the results generated by FLUX.1-dev IP-Adapter (IPA) [20] is inadequate. Besides, the text-image alignment and generation quality are inferior to other methods.
PuLID-FLUX [14] generates images with decent identity similarity. However, it suffers from poor text-image alignment (Column 11, 22, 44), and the image quality (e.g., bad hands in Column 55) and aesthetic appeal are degraded, indicating a large compromise in the generative capability of the base model. In addition, the face copy-paste issue is evident in the results generated by PuLID-FLUX (Column 55). In comparison, the proposed InfU outperforms the baselines across all dimensions.

Quantitative comparison.
The quantitative comparative results are shown in Table 1.
Our method achieves the lowest ID Loss, indicating th best identity similarity. As mentioned, the existing release of FLUX.1-dev IPA [20] is not tailored for human faces, resulting in much worse identity similarity than other methods.
In addition, our method obtains a significantly higher CLIPScore, demonstrating superior text-image alignment. Notably, the improvement in CLIPScore is substantial, reducing the gap to the upper-bound performance of FLUX.1-dev on our test set (0.3340.334) by 66.7%66.7\%.
Furthermore, our approach produces the best PickScore, suggesting that the overall image quality and generation aesthetics of InfU surpass all baselines.

Method

ID Loss↓\downarrow

CLIPScore↑\uparrow

PickScore↑\uparrow

FLUX.1-dev IPA [20]

0.772

0.243

0.204

PuLID-FLUX [14]

0.225

0.286

0.212

InfU (Ours)

0.209

0.318

0.221

User study.
We conducted a user study on InfU and the most competitive baseline, PuLID-FLUX [14]. Participants were asked to evaluate 7070 sets of samples. The study included 1616 participants from diverse backgrounds (e.g., QA professionals, researchers, engineers, designers, etc., from different countries) to reduce personal understanding bias. The best selection rate of our method reached 72.8%72.8\% in overall performance (in aspects of identity similarity, text-image alignment, image quality, and generation aesthetics), and PuLID-FLUX was 27.2%27.2\%. This suggests that our results are significantly better w.r.t. human preference.

Plug-and-play properties.
The proposed InfU method features a desirable plug-and-play design, compatible with many existing methods.
It naturally supports base model replacement with any variants of FLUX.1-dev, such as FLUX.1-schnell [26] for more efficient generation (e.g., in 4 steps, Figure 6 (a)).
The compatibility with off-the-shelf ControlNets [56] and LoRAs [18] provides additional controllability and flexibility for customized tasks (Figure 6 (b)(c)(d)).
Notably, our compatibility with OminiControl [48] extends the potential of InfU for multi-concept personalization, such as interacted identity (ID) and object personalized generation (Figure 6 (e)).
Although employing IP-Adapter (IPA) [54] with our method for identity injection is suboptimal (see Section 4.3), InfU is readily compatible with IPA for stylization of personalized images, producing decent results when injecting style references via IPA (Figure 6 (f)).
Our plug-and-play feature can extend to even more approaches beyond those mentioned, providing valuable contributions to the broader community.

### 4.3 Ablation Studies

Method

ID Loss↓\downarrow

CLIPScore↑\uparrow

PickScore↑\uparrow

w/o multi-stage training

0.172

0.292

0.212

w/o SPMS

0.368

0.303

0.220

w/ IPA

0.180

0.241

0.199

full method

0.209

0.318

0.221

We primarily conduct ablation studies on our core contributions of the multi-stage training strategy and the identity injection design. Since InfuseNet is indispensable, we highlight the importance of using InfuseNet solely, without incorporating IPA that could introduce negative impacts.

The results are shown in Table 2.
Without stage-2 supervised fine-tuning (SFT), InfU can generate images with even higher identity similarity. However, text-image alignment degrades, and image quality and aesthetic appeal worsen. We deduce that the SPMS synthetic data introduces slightly more difficulty in learning identity, yet significantly improves other aspects.
Using single-person-single-sample (SPSS) synthetic data instead of SPMS in stage-2 SFT (w/o SPMS) results in a significant drop in identity similarity, as well as degraded text-image alignment and image quality.
We deduce that SPSS synthetic data may weaken the function of InfuseNet by directly learning a reconstruction of synthetic data rather than transforming reference real data into synthetic data. This may lead to fitting back to the base model’s distribution without sufficient data diversity.
These results emphasize the importance of the multi-stage training strategy and the construction of the SPMS format.
If we employ IPA together with InfuseNet for identity injection (distinct from stylization), text-image alignment, image quality, and aesthetics substantially deteriorate, despite a slight improvement in identity similarity (still worse than our stage-1 model). This underscores the non-optimality and negative effects of IPA.

## 5 Conclusion

We introduced InfU, a novel framework for identity-preserved image generation with advanced DiTs. InfU addresses key limitations of existing methods in identity similarity, text-image alignment, overall image quality, and generation aesthetics. Central to our framework is InfuseNet, which enhances identity preservation while maintaining generative capabilities.
The multi-stage training strategy further improves our overall performance.
Comprehensive experiments showed that InfU outperforms state-of-the-art baselines.
Moreover, InfU is plug-and-play and compatible with various methods, contributing significantly to the broader community.
InfU sets a new benchmark in this field, showcasing the immense potential of integrating DiTs for advanced personalized generation.
Future work may explore enhancements in scalability and efficiency, as well as expanding the application of InfU to other domains.

Limitations and societal impact.
Despite promising results, the identity similarity and overall quality of InfU could be further improved. Potential solutions include additional model scaling and an enhanced InfuseNet design.
On another note, InfU may raise concerns about its potential to facilitate high-quality fake media synthesis. However, we believe that developing robust media forensics approaches can serve as effective safeguards.

Acknowledgments.
We sincerely acknowledge the insightful discussions from Stathi Fotiadis, Min Jin Chong, Xiao Yang, Tiancheng Zhi, Jing Liu, and Xiaohui Shen.
We genuinely appreciate the help from Jincheng Liang and Lu Guo with our user study and qualitative evaluation.

## References

- [1]

NovelAI Aspect Ratio Bucketing.

https://github.com/NovelAI/novelai-aspect-ratio-bucketing.

Accessed: 2023-02-16.

- AI [2024a]

Recraft AI.

Recraft V3 release: Recraft introduces a revolutionary ai model that thinks in design language.

https://www.recraft.ai/blog/recraft-introduces-a-revolutionary-ai-model-that-thinks-in-design-language, 2024a.

Accessed: 2024-11-01.

- AI [2024b]

XLabs AI.

flux-ip-adapter-v2.

https://huggingface.co/XLabs-AI/flux-ip-adapter-v2, 2024b.

Accessed: 2024-10-25.

- Albergo and Vanden-Eijnden [2022]

Michael S Albergo and Eric Vanden-Eijnden.

Building normalizing flows with stochastic interpolants.

arXiv preprint, arXiv:2209.15571, 2022.

- Cao et al. [2018]

Qiong Cao, Li Shen, Weidi Xie, Omkar M Parkhi, and Andrew Zisserman.

VGGFace2: A dataset for recognising faces across pose and age.

In FG, 2018.

- Chen et al. [2020]

Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge.

SimSwap: An efficient framework for high fidelity face swapping.

In ACMMM, 2020.

- Chen et al. [2018]

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud.

Neural ordinary differential equations.

In NeurIPS, 2018.

- Chen et al. [2024]

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al.

InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks.

In CVPR, 2024.

- Dehghani et al. [2023]

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al.

Scaling vision transformers to 22 billion parameters.

In ICML, 2023.

- Deng et al. [2019]

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou.

ArcFace: Additive angular margin loss for deep face recognition.

In CVPR, 2019.

- Dosovitskiy et al. [2021]

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.

An image is worth 16x16 words: Transformers for image recognition at scale.

In ICLR, 2021.

- Esser et al. [2024]

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al.

Scaling rectified flow transformers for high-resolution image synthesis.

In ICML, 2024.

- Gal et al. [2022]

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or.

An image is worth one word: Personalizing text-to-image generation using textual inversion.

arXiv preprint, arXiv:2208.01618, 2022.

- Guo et al. [2024]

Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He.

PuLID: Pure and lightning id customization via contrastive alignment.

In NeurIPS, 2024.

- He et al. [2024]

Zecheng He, Bo Sun, Felix Juefei-Xu, Haoyu Ma, Ankit Ramchandani, Vincent Cheung, Siddharth Shah, Anmol Kalia, Harihar Subramanyam, Alireza Zareian, et al.

Imagine yourself: Tuning-free personalized image generation.

arXiv preprint, arXiv:2409.13346, 2024.

- Hessel et al. [2021]

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.

CLIPScore: a reference-free evaluation metric for image captioning.

In EMNLP, 2021.

- Ho et al. [2020]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

In NeurIPS, 2020.

- Hu et al. [2022]

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.

LoRA: Low-rank adaptation of large language models.

In ICLR, 2022.

- Hurst et al. [2024]

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al.

Gpt-4o system card.

arXiv preprint, arXiv:2410.21276, 2024.

- InstantX [2024]

InstantX.

FLUX.1-dev-IP-Adapter.

https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter, 2024.

Accessed: 2024-11-01.

- Jiang et al. [2024]

Xiaohu Jiang, Yixiao Ge, Yuying Ge, Dachuan Shi, Chun Yuan, and Ying Shan.

Supervised fine-tuning in turn improves visual foundation models.

arXiv preprint, arXiv:2401.10222, 2024.

- Karras et al. [2019]

Tero Karras, Samuli Laine, and Timo Aila.

A style-based generator architecture for generative adversarial networks.

In CVPR, 2019.

- Kirstain et al. [2023]

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy.

Pick-a-pic: An open dataset of user preferences for text-to-image generation.

In NeurIPS, 2023.

- Kumari et al. [2023]

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu.

Multi-concept customization of text-to-image diffusion.

In CVPR, 2023.

- Kvanchiani et al. [2023]

Karina Kvanchiani, Elizaveta Petrova, Karen Efremyan, Alexander Sautin, and Alexander Kapitanov.

EasyPortrait–face parsing and portrait segmentation dataset.

arXiv preprint, arXiv:2304.13509, 2023.

- Labs [2024]

Black Forest Labs.

FLUX.1 release: Announcing black forest labs.

https://blackforestlabs.ai/announcing-black-forest-labs/, 2024.

Accessed: 2024-08-01.

- Li et al. [2023]

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.

BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.

In ICML, 2023.

- Li et al. [2024a]

Shikai Li, Jianglin Fu, Kaiyuan Liu, Wentao Wang, Kwan-Yee Lin, and Wayne Wu.

CosmicMan: A text-to-image foundation model for humans.

arXiv preprint, arXiv:2404.01294, 2024a.

- Li et al. [2024b]

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan.

PhotoMaker: Customizing realistic human photos via stacked id embedding.

In CVPR, 2024b.

- Lin et al. [2024]

Shanchuan Lin, Anran Wang, and Xiao Yang.

Sdxl-lightning: Progressive adversarial diffusion distillation.

arXiv preprint, arXiv:2402.13929, 2024.

- Lipman et al. [2023]

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

Flow matching for generative modeling.

In ICLR, 2023.

- Liu et al. [2024]

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li.

Playground v3: Improving text-to-image alignment with deep-fusion large language models.

arXiv preprint, arXiv:2409.10695, 2024.

- Liu et al. [2022]

Xingchao Liu, Chengyue Gong, and Qiang Liu.

Flow straight and fast: Learning to generate and transfer data with rectified flow.

arXiv preprint, arXiv:2209.03003, 2022.

- Liu et al. [2015]

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.

Deep learning face attributes in the wild.

In ICCV, 2015.

- Loshchilov and Hutter [2019]

Ilya Loshchilov and Frank Hutter.

Decoupled weight decay regularization.

In ICLR, 2019.

- Papantoniou et al. [2024]

Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos Zafeiriou.

Arc2Face: A foundation model for id-consistent human faces.

In ECCV, 2024.

- Peebles and Xie [2023]

William Peebles and Saining Xie.

Scalable diffusion models with transformers.

In ICCV, 2023.

- Podell et al. [2023]

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach.

SDXL: Improving latent diffusion models for high-resolution image synthesis.

arXiv preprint, arXiv:2307.01952, 2023.

- Radford et al. [2021]

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.

Learning transferable visual models from natural language supervision.

In ICML, 2021.

- Raffel et al. [2020]

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.

Exploring the limits of transfer learning with a unified text-to-text transformer.

JMLR, 21:1–67, 2020.

- Rombach et al. [2022]

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

High-resolution image synthesis with latent diffusion models.

In CVPR, 2022.

- Ronneberger et al. [2015]

Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

U-Net: Convolutional networks for biomedical image segmentation.

In MICCAI, 2015.

- Ruiz et al. [2023]

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman.

DreamBooth: Fine tuning text-to-image diffusion models for subject-driven generation.

In CVPR, 2023.

- Ruiz et al. [2024]

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman.

HyperDreamBooth: Hypernetworks for fast personalization of text-to-image models.

In CVPR, 2024.

- Sauer et al. [2024]

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach.

Adversarial diffusion distillation.

In ECCV, 2024.

- Sohl-Dickstein et al. [2015]

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.

Deep unsupervised learning using nonequilibrium thermodynamics.

In ICML, 2015.

- Song et al. [2020]

Jiaming Song, Chenlin Meng, and Stefano Ermon.

Denoising diffusion implicit models.

arXiv preprint, arXiv:2010.02502, 2020.

- Tan et al. [2024]

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang.

OminiControl: Minimal and universal control for diffusion transformer.

arXiv preprint, arXiv:2411.15098, 2024.

- Touvron et al. [2023]

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.

Llama 2: Open foundation and fine-tuned chat models.

arXiv preprint, 2307.09288, 2023.

- Vaswani et al. [2017]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In NeurIPS, 2017.

- Wang et al. [2024]

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu.

InstantID: Zero-shot identity-preserving generation in seconds.

arXiv preprint, arXiv:2401.07519, 2024.

- Xiao et al. [2024]

Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han.

FastComposer: Tuning-free multi-subject image generation with localized attention.

IJCV, pages 1–20, 2024.

- Xie et al. [2022]

Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan.

VFHQ: A high-quality dataset and benchmark for video face super-resolution.

In CVPRW, 2022.

- Ye et al. [2023]

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang.

IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models.

arXiv preprint, arXiv:2308.06721, 2023.

- Yu et al. [2023]

Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu.

CelebV-Text: A large-scale facial text-video dataset.

In CVPR, 2023.

- Zhang et al. [2023]

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

Adding conditional control to text-to-image diffusion models.

In ICCV, 2023.

- Zhang et al. [2024]

Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, Zhi-Fan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo.

FlashFace: Human image personalization with high-fidelity identity preservation.

arXiv preprint, arXiv:2403.17008, 2024.

- Zhang et al. [2020]

Yaobin Zhang, Weihong Deng, Mei Wang, Jiani Hu, Xian Li, Dongyue Zhao, and Dongchao Wen.

Global-local GCN: Large-scale label noise cleansing for face recognition.

In CVPR, 2020.

- Zhao et al. [2023]

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al.

PyTorch FSDP: experiences on scaling fully sharded data parallel.

arXiv preprint, arXiv:2304.11277, 2023.

- Zhu et al. [2022]

Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy.

CelebV-HQ: A large-scale video facial attributes dataset.

In ECCV, 2022.

Generated on Sat Apr 5 17:04:46 2025 by LaTeXML
