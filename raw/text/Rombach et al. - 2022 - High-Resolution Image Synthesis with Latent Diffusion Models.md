# Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models

- Source HTML: `raw/html/Rombach et al. - 2022 - High-Resolution Image Synthesis with Latent Diffusion Models.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2112.10752
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# High-Resolution Image Synthesis with Latent Diffusion Models

Robin Rombach1   Andreas Blattmann1 ∗  Dominik Lorenz1   Patrick Esser    Björn Ommer1 
1Ludwig Maximilian University of Munich & IWR, Heidelberg
University, Germany   Runway ML
https://github.com/CompVis/latent-diffusion

The first two authors contributed equally to this
work.

###### Abstract

By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond.
Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining.
However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to
sequential evaluations.
To enable DM training on limited computational resources while retaining their quality and flexibility,
we apply them in the latent space of powerful pretrained autoencoders.
In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point
between complexity reduction and detail preservation,
greatly boosting visual fidelity.
By introducing cross-attention layers into the model architecture,
we turn diffusion models into powerful and flexible
generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner.
Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image
inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including
text-to-image synthesis, unconditional image generation and super-resolution, while
significantly reducing computational requirements compared to pixel-based DMs.

## 1 Introduction

Input

ours (f=4𝑓4f=4)
PSNR: 27.427.427.4 R-FID: 0.580.580.58

DALL-E (f=8𝑓8f=8)
PSNR: 22.822.822.8 R-FID: 32.0132.0132.01

VQGAN
(f=16𝑓16f=16)
PSNR: 19.919.919.9 R-FID: 4.984.984.98

Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. Especially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based models, potentially containing billions of parameters in autoregressive (AR) transformers [66, 67].
In contrast, the promising results of GANs [27, 3, 40] have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. Recently, diffusion models [82], which are built from a hierarchy of denoising autoencoders, have shown to achieve impressive results in image synthesis [30, 85] and beyond [45, 7, 48, 57], and define the state-of-the-art in class-conditional image synthesis [15, 31] and super-resolution [72]. Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization [85] or stroke-based synthesis [53], in contrast to other types of generative models [46, 69, 19].
Being likelihood-based models, they do not exhibit mode-collapse and training
instabilities as GANs and, by heavily exploiting parameter sharing,
they can model highly complex distributions of natural images without involving
billions of parameters as in AR
models [67].

##### Democratizing High-Resolution Image Synthesis

DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data [16, 73].
Although the reweighted variational objective [30]
aims to address this
by undersampling the initial denoising steps,
DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images.
As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [15])
and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days [15] on a single A100 GPU.
This has two consequences for the research community and users in general:
Firstly, training such a model
requires massive computational resources only available to a small fraction of the field,
and leaves a huge carbon footprint [65, 86].
Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must
run sequentially for a large number of steps (e.g. 25 - 1000 steps in [15]).

To increase the accessibility of this powerful model class and at the same time
reduce its significant resource consumption, a method is needed that
reduces the computational complexity for both training and sampling.
Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.

##### Departure to Latent Space

Our approach starts with the analysis of already trained diffusion models in pixel space:
Fig. 2 shows the rate-distortion trade-off of a trained model. As with any likelihood-based model, learning can be roughly divided into two stages: First is a perceptual compression stage which removes high-frequency details
but still learns little semantic variation.
In the second stage, the actual generative model learns the semantic and conceptual composition of the data (semantic compression).
We thus aim to first find a perceptually equivalent, but computationally more suitable space, in which we will train diffusion models for high-resolution image synthesis.

Following common practice [96, 67, 23, 11, 66],
we separate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptually equivalent to the data space.
Importantly, and in contrast to previous work [23, 66],
we do not need to
rely on excessive spatial compression, as we train DMs in the learned latent space, which
exhibits better scaling properties with respect to the spatial dimensionality.
The reduced complexity also provides efficient image generation from the latent space with a single network pass.
We dub the resulting model class Latent Diffusion Models (LDMs).

A notable
advantage of this approach is that we need to train the universal
autoencoding stage only once and can therefore reuse it for multiple DM
trainings or to explore possibly completely different tasks
[81].
This enables efficient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks.
For the latter, we design an architecture that connects transformers to the DM’s UNet backbone [71]
and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.

In sum, our work makes the following contributions:

(i) In contrast to purely transformer-based approaches [23, 66], our
method scales more graceful to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work (see Fig. 1) and (b) can be efficiently applied to high-resolution synthesis of megapixel images.

(ii) We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution)
and datasets while significantly lowering computational costs.
Compared to pixel-based diffusion approaches, we also significantly decrease inference costs.

(iii) We show that, in contrast to previous work [93] which learns both an encoder/decoder architecture and a score-based prior simultaneously, our
approach does not require a delicate weighting of reconstruction and generative abilities.
This ensures extremely faithful reconstructions and requires very little regularization of the latent space.

(iv) We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a
convolutional fashion and render large, consistent images of ∼10242similar-toabsentsuperscript10242\sim 1024^{2} px.

(v) Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training.
We use it to train class-conditional, text-to-image and layout-to-image models.

(vi) Finally, we release pretrained latent diffusion and autoencoding models at https://github.com/CompVis/latent-diffusion
which might be reusable for a various tasks besides
training of DMs [81].

## 2 Related Work

Generative Models for Image Synthesis
The high dimensional nature of images presents distinct challenges to generative modeling.
Generative Adversarial Networks (GAN) [27]
allow for efficient sampling of high resolution images with good perceptual quality [3, 42], but are difficult to optimize [54, 2, 28] and struggle to capture the full data distribution
[55].
In contrast, likelihood-based methods emphasize good density estimation which renders optimization more well-behaved.
Variational autoencoders (VAE) [46]
and flow-based models [18, 19] enable efficient synthesis of high resolution images [9, 92, 44], but sample quality is not on par with GANs.
While autoregressive models (ARM) [95, 94, 6, 10] achieve strong performance in density
estimation, computationally demanding architectures
[97] and a sequential sampling process limit them to low resolution images.
Because pixel based representations of images contain barely
perceptible, high-frequency details [16, 73], maximum-likelihood training spends a
disproportionate amount of capacity on modeling them, resulting in
long training times.
To scale to higher resolutions,
several two-stage approaches [101, 67, 23, 103]
use ARMs to model a compressed latent image space instead of raw pixels.

Recently, Diffusion Probabilistic Models (DM) [82], have achieved state-of-the-art results in density estimation [45] as well as in sample quality [15]. The generative power of these models stems from a natural fit to the inductive biases of image-like data when their underlying neural backbone is implemented as a UNet [71, 30, 85, 15].
The best synthesis quality is usually achieved when a reweighted objective [30]
is used for training. In this case, the DM corresponds to a lossy compressor and allow to trade image quality for compression capabilities.
Evaluating and optimizing these models in pixel space, however, has the downside of low inference speed and very high training costs.
While the former can be partially adressed by advanced sampling strategies [84, 75, 47] and hierarchical approaches [31, 93], training on high-resolution image data always requires to calculate expensive gradients.
We adress both drawbacks with our proposed LDMs, which work on a compressed latent space of lower dimensionality.
This renders training computationally cheaper and speeds up inference with
almost no reduction in synthesis quality (see
Fig. 1).

Two-Stage Image Synthesis

To mitigate the shortcomings of individual generative approaches, a lot of research [11, 70, 23, 103, 101, 67] has gone into combining the strengths of different methods into more efficient and performant models via a two stage approach. VQ-VAEs [101, 67] use autoregressive models to learn an expressive prior over a discretized latent space.
[66] extend this approach to text-to-image generation by learning a joint distributation over discretized image and text representations.
More generally, [70] uses conditionally invertible networks to provide a generic transfer between latent spaces of diverse domains.
Different from VQ-VAEs, VQGANs [23, 103] employ a first stage with an adversarial and perceptual objective to scale autoregressive transformers to larger images.
However, the high compression rates required for feasible ARM training, which introduces billions of trainable parameters [66, 23], limit the overall performance of such approaches and less compression comes at the price of high computational cost [66, 23].
Our work prevents such trade-offs, as our proposed LDMs scale more gently to higher dimensional latent spaces due to their convolutional backbone. Thus, we are free to choose the level of compression which optimally mediates between learning a powerful first stage, without leaving too much perceptual compression up to the generative diffusion model while guaranteeing high-fidelity reconstructions (see Fig. 1).

While approaches to jointly [93] or separately [80]
learn an encoding/decoding model together with a score-based prior exist,
the former still require a difficult weighting between reconstruction and generative capabilities [11] and are outperformed by our approach (Sec. 4), and the latter focus on highly structured images such as human faces.

## 3 Method

To lower the computational demands of training diffusion models towards high-resolution image synthesis,
we observe
that although diffusion models
allow
to ignore perceptually irrelevant details by undersampling the corresponding loss terms [30],
they still require costly function evaluations in pixel space,
which causes
huge demands in computation time and energy resources.

We propose to circumvent this drawback by introducing an explicit separation
of the compressive from the generative learning phase (see
Fig. 2).
To achieve this, we utilize an autoencoding model which learns a space that is perceptually equivalent to the image space,
but offers significantly reduced computational complexity.

Such an approach offers several advantages: (i)
By leaving the high-dimensional image space, we
obtain DMs which are computationally much more efficient
because sampling is performed on a low-dimensional space.
(ii)
We exploit the inductive bias of DMs inherited from their UNet architecture
[71], which makes them particularly effective for data with spatial
structure and therefore
alleviates the need for aggressive, quality-reducing compression levels as required by previous
approaches [23, 66].
(iii)
Finally, we obtain general-purpose compression models whose latent space
can be used to train multiple generative models
and which can also be utilized
for other downstream applications such as single-image CLIP-guided synthesis [25].

### 3.1 Perceptual Image Compression

Our perceptual compression model
is based on previous work [23] and
consists of an autoencoder trained by combination of a perceptual
loss [106] and a patch-based [33] adversarial objective [20, 23, 103].
This ensures that the reconstructions are
confined to the image manifold by enforcing local realism and avoids bluriness introduced by relying solely on pixel-space losses
such as L2subscript𝐿2L_{2} or L1subscript𝐿1L_{1} objectives.

More precisely, given an image x∈ℝH×W×3𝑥superscriptℝ𝐻𝑊3x\in\mathbb{R}^{H\times W\times 3} in RGB space,
the encoder ℰℰ\mathcal{E} encodes x𝑥x into a latent representation z=ℰ​(x)𝑧ℰ𝑥z=\mathcal{E}(x), and the decoder 𝒟𝒟\mathcal{D}
reconstructs the image from the latent, giving x~=𝒟​(z)=𝒟​(ℰ​(x))~𝑥𝒟𝑧𝒟ℰ𝑥\tilde{x}=\mathcal{D}(z)=\mathcal{D}(\mathcal{E}(x)), where z∈ℝh×w×c𝑧superscriptℝℎ𝑤𝑐z\in\mathbb{R}^{h\times w\times c}.
Importantly, the encoder downsamples the image by a factor f=H/h=W/w𝑓𝐻ℎ𝑊𝑤f=H/h=W/w, and we investigate different downsampling
factors f=2m𝑓superscript2𝑚f=2^{m}, with m∈ℕ𝑚ℕm\in\mathbb{N}.

In order to avoid arbitrarily
high-variance
latent spaces, we experiment with two different kinds of regularizations. The
first variant, KL-reg., imposes a slight KL-penalty towards a standard
normal on the learned latent, similar to
a VAE [46, 69], whereas VQ-reg.
uses a vector quantization layer [96] within
the decoder. This model can be interpreted as a VQGAN [23] but with the quantization layer
absorbed by the decoder. Because our subsequent DM
is designed to work with the two-dimensional structure of
our learned latent space z=ℰ​(x)𝑧ℰ𝑥z=\mathcal{E}(x), we can use relatively mild
compression rates and achieve very good reconstructions. This is in contrast to
previous works [23, 66],
which relied on an arbitrary 1D ordering of the learned space z𝑧z to model its
distribution autoregressively and thereby ignored much of the
inherent structure of z𝑧z.
Hence, our compression model preserves details of x𝑥x better (see Tab. 8).
The full objective and training details can be found in the supplement.

### 3.2 Latent Diffusion Models

Diffusion Models [82] are probabilistic models designed to learn a data distribution p​(x)𝑝𝑥p(x) by
gradually denoising a normally distributed variable, which corresponds to learning
the reverse process of a fixed Markov Chain of length T𝑇T.
For image synthesis, the most successful models [30, 15, 72]
rely on a reweighted variant of the variational lower bound on p​(x)𝑝𝑥p(x), which mirrors denoising score-matching [85].
These models can be interpreted as an equally weighted sequence of denoising autoencoders ϵθ​(xt,t);t=1​…​Tsubscriptitalic-ϵ𝜃subscript𝑥𝑡𝑡𝑡1…𝑇\epsilon_{\theta}(x_{t},t);\,t=1\dots T,
which are trained to predict a denoised variant of their input xtsubscript𝑥𝑡x_{t}, where xtsubscript𝑥𝑡x_{t} is a noisy version of the input x𝑥x.
The corresponding objective can be simplified to (Sec. B)

LD​M=𝔼x,ϵ∼𝒩​(0,1),t​[‖ϵ−ϵθ​(xt,t)‖22],subscript𝐿𝐷𝑀subscript𝔼formulae-sequencesimilar-to𝑥italic-ϵ𝒩01𝑡delimited-[]superscriptsubscriptnormitalic-ϵsubscriptitalic-ϵ𝜃subscript𝑥𝑡𝑡22L_{DM}=\mathbb{E}_{x,\epsilon\sim\mathcal{N}(0,1),t}\Big{[}\|\epsilon-\epsilon_{\theta}(x_{t},t)\|_{2}^{2}\Big{]}\,,

(1)

with t𝑡t uniformly sampled from {1,…,T}1…𝑇\{1,\dots,T\}.

Generative Modeling of Latent Representations
With our trained perceptual compression models consisting of ℰℰ\mathcal{E} and 𝒟𝒟\mathcal{D},
we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are
abstracted away.
Compared to the high-dimensional pixel space, this space is more suitable
for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data
and (ii) train in a lower dimensional, computationally much more efficient space.

Unlike previous work that relied on autoregressive, attention-based
transformer models in a highly compressed, discrete latent space [66, 23, 103],
we can take advantage of image-specific inductive biases that our model offers.
This includes the ability to build the underlying UNet primarily from 2D convolutional layers,
and further focusing the objective on the perceptually most relevant bits using
the reweighted bound, which now reads

LL​D​M:=𝔼ℰ​(x),ϵ∼𝒩​(0,1),t​[‖ϵ−ϵθ​(zt,t)‖22].assignsubscript𝐿𝐿𝐷𝑀subscript𝔼formulae-sequencesimilar-toℰ𝑥italic-ϵ𝒩01𝑡delimited-[]superscriptsubscriptnormitalic-ϵsubscriptitalic-ϵ𝜃subscript𝑧𝑡𝑡22L_{LDM}:=\mathbb{E}_{\mathcal{E}(x),\epsilon\sim\mathcal{N}(0,1),t}\Big{[}\|\epsilon-\epsilon_{\theta}(z_{t},t)\|_{2}^{2}\Big{]}\,.

(2)

The neural backbone ϵθ​(∘,t)subscriptitalic-ϵ𝜃𝑡\epsilon_{\theta}(\circ,t) of our model
is realized as a time-conditional UNet [71].
Since the forward process is fixed, ztsubscript𝑧𝑡z_{t} can be efficiently obtained from
ℰℰ\mathcal{E} during training,
and samples from p(zp(z) can be decoded to image space with a single pass through 𝒟𝒟\mathcal{D}.

CelebAHQ
FFHQ
LSUN-Churches
LSUN-Beds
ImageNet

### 3.3 Conditioning Mechanisms

Similar to other types of generative
models [56, 83], diffusion models
are in principle capable of modeling conditional distributions of the form
p​(z|y)𝑝conditional𝑧𝑦p(z|y).
This can be implemented with
a conditional denoising autoencoder ϵθ​(zt,t,y)subscriptitalic-ϵ𝜃subscript𝑧𝑡𝑡𝑦\epsilon_{\theta}(z_{t},t,y) and
paves the way to controlling the synthesis process through
inputs y𝑦y such as text [68], semantic maps [61, 33] or other image-to-image translation tasks [34].

In the context of image synthesis, however, combining the
generative power of DMs with other types of conditionings beyond class-labels [15]
or blurred variants
of the input image [72]
is so far an under-explored area of research.

We turn DMs into more flexible conditional image generators by
augmenting their underlying UNet backbone with the cross-attention mechanism [97],
which
is
effective for learning attention-based models of various input modalities [36, 35].
To pre-process y𝑦y from various modalities (such as language prompts) we introduce a domain specific
encoder τθsubscript𝜏𝜃\tau_{\theta} that projects y𝑦y to an intermediate representation τθ​(y)∈ℝM×dτsubscript𝜏𝜃𝑦superscriptℝ𝑀subscript𝑑𝜏\tau_{\theta}(y)\in\mathbb{R}^{M\times d_{\tau}}, which
is then mapped to the intermediate layers of the UNet via a cross-attention
layer implementing Attention​(Q,K,V)=softmax​(Q​KTd)⋅VAttention𝑄𝐾𝑉⋅softmax𝑄superscript𝐾𝑇𝑑𝑉\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^{T}}{\sqrt{d}}\right)\cdot V, with

Q=WQ(i)⋅φi​(zt),K=WK(i)⋅τθ​(y),V=WV(i)⋅τθ​(y).formulae-sequence𝑄⋅subscriptsuperscript𝑊𝑖𝑄subscript𝜑𝑖subscript𝑧𝑡formulae-sequence𝐾⋅subscriptsuperscript𝑊𝑖𝐾subscript𝜏𝜃𝑦𝑉⋅subscriptsuperscript𝑊𝑖𝑉subscript𝜏𝜃𝑦Q=W^{(i)}_{Q}\cdot\varphi_{i}(z_{t}),\;K=W^{(i)}_{K}\cdot\tau_{\theta}(y),\;V=W^{(i)}_{V}\cdot\tau_{\theta}(y).

Here, φi​(zt)∈ℝN×dϵisubscript𝜑𝑖subscript𝑧𝑡superscriptℝ𝑁subscriptsuperscript𝑑𝑖italic-ϵ\varphi_{i}(z_{t})\in\mathbb{R}^{N\times d^{i}_{\epsilon}} denotes a (flattened) intermediate representation of the UNet implementing
ϵθsubscriptitalic-ϵ𝜃\epsilon_{\theta} and WV(i)∈ℝd×dϵisubscriptsuperscript𝑊𝑖𝑉superscriptℝ𝑑subscriptsuperscript𝑑𝑖italic-ϵW^{(i)}_{V}\in\mathbb{R}^{d\times d^{i}_{\epsilon}}, WQ(i)∈ℝd×dτsubscriptsuperscript𝑊𝑖𝑄superscriptℝ𝑑subscript𝑑𝜏W^{(i)}_{Q}\in\mathbb{R}^{d\times d_{\tau}} & WK(i)∈ℝd×dτsubscriptsuperscript𝑊𝑖𝐾superscriptℝ𝑑subscript𝑑𝜏W^{(i)}_{K}\in\mathbb{R}^{d\times d_{\tau}} are learnable projection matrices [97, 36]. See Fig. 3 for a visual depiction.

Based on image-conditioning pairs, we then learn the conditional LDM via

LL​D​M:=𝔼ℰ​(x),y,ϵ∼𝒩​(0,1),t​[‖ϵ−ϵθ​(zt,t,τθ​(y))‖22],assignsubscript𝐿𝐿𝐷𝑀subscript𝔼formulae-sequencesimilar-toℰ𝑥𝑦italic-ϵ𝒩01𝑡delimited-[]superscriptsubscriptnormitalic-ϵsubscriptitalic-ϵ𝜃subscript𝑧𝑡𝑡subscript𝜏𝜃𝑦22L_{LDM}:=\mathbb{E}_{\mathcal{E}(x),y,\epsilon\sim\mathcal{N}(0,1),t}\Big{[}\|\epsilon-\epsilon_{\theta}(z_{t},t,\tau_{\theta}(y))\|_{2}^{2}\Big{]}\,,

(3)

where
both τθsubscript𝜏𝜃\tau_{\theta} and ϵθsubscriptitalic-ϵ𝜃\epsilon_{\theta} are jointly optimized via Eq. 3.
This conditioning mechanism is flexible as τθsubscript𝜏𝜃\tau_{\theta} can be parameterized with domain-specific experts,
e.g. (unmasked) transformers [97] when y𝑦y are
text prompts (see Sec. 4.3.1)

## 4 Experiments

Text-to-Image Synthesis on LAION. 1.45B Model.

’A street sign that reads
“Latent Diffusion” ’

’A zombie in the
style of Picasso’

’An image of an animal
half mouse half octopus’

’An illustration of a slightly
conscious neural network’

’A painting of a
squirrel eating a burger’

’A watercolor painting of a
chair that looks like an octopus’

’A shirt with the inscription:
“I love generative models!” ’

LDMs provide means to flexible and computationally tractable diffusion based image synthesis
of various image modalities, which we empirically show in the following.
Firstly, however, we analyze the gains of our models compared to pixel-based diffusion models in both training and inference.
Interestingly, we find that LDMs trained in VQ-regularized latent spaces sometimes achieve better sample quality, even though the reconstruction capabilities of VQ-regularized first stage models slightly fall behind those of their continuous counterparts, cf. Tab. 8.
A visual comparison between the effects of first stage regularization schemes on LDM training and their generalization abilities to resolutions >2562absentsuperscript2562>256^{2} can be found in Appendix D.1. In E.2 we list details on architecture, implementation, training and evaluation for all results presented in this section.

### 4.1 On Perceptual Compression Tradeoffs

This section analyzes the behavior of our LDMs with different downsampling
factors f∈{1,2,4,8,16,32}𝑓12481632f\in\{1,2,4,8,16,32\} (abbreviated as LDM-f𝑓f, where
LDM-1 corresponds to pixel-based DMs).
To obtain a
comparable test-field, we fix the computational resources
to a
single NVIDIA A100 for all experiments in this section and train all models for
the same number of steps and with the same number of parameters.

Tab. 8 shows hyperparameters and reconstruction performance of the first stage models used for the LDMs compared in this section.
Fig. 6 shows sample quality as a function of training
progress for 2M steps of class-conditional models on the
ImageNet [12] dataset. We see that, i) small
downsampling factors for LDM-{{\{1,2}}\} result in slow training
progress, whereas ii) overly large values of f𝑓f cause stagnating fidelity
after comparably few training steps. Revisiting the analysis above
(Fig. 1 and 2) we
attribute this to i) leaving most of perceptual compression to the diffusion
model and ii) too strong first stage compression resulting in information loss
and thus limiting the achievable quality. LDM-{{\{4-16}}\}
strike a good balance between efficiency and perceptually faithful results,
which manifests in a significant FID [29] gap of 38 between pixel-based diffusion (LDM-1) and LDM-8 after 2M training steps.

In Fig. 7, we compare models trained on
CelebA-HQ [39] and ImageNet in terms
sampling speed for different numbers of denoising steps with the DDIM
sampler [84] and plot it against FID-scores [29].
LDM-{{\{4-8}}\} outperform models
with unsuitable ratios of perceptual and conceptual compression.
Especially compared to
pixel-based LDM-1, they achieve much lower FID scores while
simultaneously significantly increasing sample throughput.
Complex datasets such as ImageNet require reduced compression rates to avoid
reducing quality.
In summary, LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results.

CelebA-HQ 256×256256256256\times 256

FFHQ 256×256256256256\times 256

Method
FID ↓↓\downarrow
Prec. ↑↑\uparrow
Recall ↑↑\uparrow

Method
FID ↓↓\downarrow
Prec. ↑↑\uparrow
Recall ↑↑\uparrow

DC-VAE [63]
15.8
-
-

ImageBART [21]
9.57
-
-

VQGAN+T. [23] (k=400)
10.2
-
-

U-Net GAN (+aug) [77]
10.9 (7.6)
-
-

PGGAN [39]
8.0
-
-

UDM [43]
5.54
-
-

LSGM [93]
7.22
-
-

StyleGAN [41]
4.16
0.71
0.46

UDM [43]
7.16
-
-

ProjectedGAN[76]
3.08
0.65
0.46

LDM-4 (ours, 500-s†)
5.11
0.72
0.49

LDM-4 (ours, 200-s)
4.98
0.73
0.50

LSUN-Churches 256×256256256256\times 256

LSUN-Bedrooms 256×256256256256\times 256

Method
FID ↓↓\downarrow
Prec. ↑↑\uparrow
Recall ↑↑\uparrow

Method
FID ↓↓\downarrow
Prec. ↑↑\uparrow
Recall ↑↑\uparrow

DDPM [30]
7.89
-
-

ImageBART [21]
5.51
-
-

ImageBART[21]
7.32
-
-

DDPM [30]
4.9
-
-

PGGAN [39]
6.42
-
-

UDM [43]
4.57
-
-

StyleGAN[41]
4.21
-
-

StyleGAN[41]
2.35
0.59
0.48

StyleGAN2[42]
3.86
-
-

ADM [15]
1.90
0.66
0.51

ProjectedGAN[76]
1.59
0.61
0.44

ProjectedGAN[76]
1.52
0.61
0.34

LDM-8∗ (ours, 200-s)
4.02
0.64
0.52

LDM-4 (ours, 200-s)
2.95
0.66
0.48

Text-Conditional Image Synthesis

Method
FID ↓↓\downarrow
IS↑↑\uparrow
Nparamssubscript𝑁paramsN_{\text{params}}

CogView† [17]
27.10
18.20
4B
self-ranking, rejection rate 0.017

LAFITE† [109]
26.94
26.02
75M

GLIDE∗ [59]
12.24
-
6B
277 DDIM steps, c.f.g. [32] s=3𝑠3s=3

Make-A-Scene∗ [26]
11.84
-
4B
c.f.g for AR models [98] s=5𝑠5s=5

LDM-KL-8
23.31
20.03±0.33plus-or-minus0.33\pm\text{0.33}
1.45B
250 DDIM steps

LDM-KL-8-G∗
12.63
30.29±0.42plus-or-minus0.42\pm\text{0.42}
1.45B
250 DDIM steps, c.f.g. [32] s=1.5𝑠1.5s=1.5

### 4.2 Image Generation with Latent Diffusion

We train unconditional models of 2562superscript2562256^{2} images on
CelebA-HQ [39], FFHQ [41],
LSUN-Churches and -Bedrooms [102]
and evaluate the i) sample quality and ii) their coverage of the data manifold using ii) FID [29] and ii) Precision-and-Recall [50]. Tab. 1 summarizes our results. On CelebA-HQ, we report a new state-of-the-art FID of 5.115.115.11, outperforming
previous likelihood-based models as well as GANs. We also outperform
LSGM [93] where a latent diffusion model is
trained jointly together with the first stage.
In contrast, we train diffusion models in a fixed space and avoid the
difficulty of weighing reconstruction quality against learning the prior over
the latent space, see Fig. 1-2.

We outperform prior diffusion based approaches on all but the LSUN-Bedrooms
dataset, where our score is close to
ADM [15], despite utilizing half its
parameters and requiring 4-times less train resources (see Appendix E.3.5).
Moreover, LDMs consistently improve upon GAN-based methods in Precision and Recall, thus confirming the advantages of their mode-covering likelihood-based training objective over adversarial approaches.
In Fig. 4 we also show qualitative results on each dataset.

### 4.3 Conditional Latent Diffusion

#### 4.3.1 Transformer Encoders for LDMs

By introducing cross-attention based conditioning into LDMs
we open them up for various
conditioning modalities previously unexplored for diffusion models.
For text-to-image image modeling, we train a 1.45B parameter KL-regularized LDM conditioned on language prompts on LAION-400M [78].
We employ the BERT-tokenizer [14] and implement τθsubscript𝜏𝜃\tau_{\theta} as a transformer [97] to infer a latent code which
is mapped into the UNet via (multi-head) cross-attention (Sec. 3.3).
This combination of domain specific experts for learning a language representation and visual synthesis results in a powerful model, which generalizes well to complex, user-defined text prompts, cf. Fig. 8 and 5. For quantitative analysis, we follow prior work and evaluate text-to-image generation on the MS-COCO [51] validation set, where our model improves upon powerful AR [66, 17] and GAN-based [109] methods, cf. Tab. 2. We note that applying classifier-free diffusion guidance [32] greatly boosts sample quality, such that the guided LDM-KL-8-G is on par with the recent state-of-the-art AR [26] and diffusion models [59] for text-to-image synthesis, while substantially reducing parameter count.
To further analyze the flexibility of the cross-attention based conditioning mechanism we also train models to synthesize images based on semantic layouts on OpenImages [49],
and finetune on COCO [4], see Fig. 8.
See Sec. D.3 for the quantitative evaluation and implementation details.

Lastly, following prior work [15, 3, 23, 21], we evaluate our best-performing class-conditional ImageNet models with f∈{4,8}𝑓48f\in\{4,8\} from Sec. 4.1 in Tab. 3, Fig. 4 and
Sec. D.4. Here we outperform the state of the art diffusion model ADM [15] while significantly reducing computational requirements and parameter count, cf. Tab 18.

Method
FID↓↓\downarrow
IS↑↑\uparrow
Precision↑↑\uparrow
Recall↑↑\uparrow
Nparamssubscript𝑁paramsN_{\text{params}}

BigGan-deep [3]
6.95
203.6±2.6plus-or-minus2.6\pm\text{2.6}
0.87
0.28
340M
-

ADM [15]
10.94
100.98
0.69
0.63
554M
250 DDIM steps

ADM-G [15]
4.59
186.7
0.82
0.52
608M
250 DDIM steps

LDM-4 (ours)
10.56
103.49±1.24plus-or-minus1.24\pm\text{1.24}
0.71
0.62
400M
250 DDIM steps

LDM-4-G (ours)
3.60
247.67±5.59plus-or-minus5.59\pm\text{5.59}
0.87
0.48
400M
250 steps, c.f.g [32], s=1.5𝑠1.5s=1.5

#### 4.3.2 Convolutional Sampling Beyond 2562superscript2562256^{2}

By concatenating spatially aligned conditioning information to the input of
ϵθsubscriptitalic-ϵ𝜃\epsilon_{\theta}, LDMs can serve
as efficient general-purpose image-to-image translation models.
We use this to train models for semantic synthesis, super-resolution (Sec. 4.4) and inpainting (Sec. 4.5).
For semantic synthesis, we use images of landscapes paired with semantic maps [61, 23]
and concatenate downsampled versions of the semantic maps with the latent image
representation of a f=4𝑓4f=4 model (VQ-reg., see Tab. 8).
We train on an input resolution of 2562superscript2562256^{2} (crops from 3842superscript3842384^{2}) but find that our model generalizes to larger resolutions
and can
generate images up to the megapixel regime when evaluated in a convolutional
manner (see Fig. 9).
We exploit this behavior to also apply the super-resolution models in
Sec. 4.4 and the inpainting models in
Sec. 4.5 to generate large images between 5122superscript5122512^{2} and 10242superscript102421024^{2}.
For this application, the signal-to-noise ratio (induced by the scale of the latent space)
significantly affects the results.
In Sec. D.1 we illustrate this
when learning an LDM on (i) the latent space as provided by a f=4𝑓4f=4 model (KL-reg., see Tab. 8),
and (ii) a rescaled version, scaled by the component-wise standard deviation.

The latter, in combination with classifier-free guidance [32], also enables the direct synthesis of
>2562absentsuperscript2562>256^{2} images for the text-conditional LDM-KL-8-G as in Fig. 13.

### 4.4 Super-Resolution with Latent Diffusion

LDMs can be efficiently trained for super-resolution by diretly conditioning on low-resolution images via concatenation (cf. Sec. 3.3).
In a first experiment, we follow SR3 [72] and fix the image degradation to a bicubic interpolation with 4×4\times-downsampling and train on ImageNet following SR3’s data processing pipeline. We use the f=4𝑓4f=4 autoencoding model pretrained on OpenImages (VQ-reg., cf. Tab. 8) and concatenate the low-resolution conditioning y𝑦y and the inputs to the UNet, i.e. τθsubscript𝜏𝜃\tau_{\theta} is the identity.
Our qualitative and quantitative results (see Fig. 10 and Tab. 5) show competitive performance and LDM-SR outperforms SR3 in FID while SR3 has a better IS.
A simple image regression model achieves the highest PSNR and SSIM scores;
however these metrics do not align well with human perception [106] and favor blurriness over imperfectly aligned high frequency details [72].
Further, we conduct a user study comparing the pixel-baseline with LDM-SR. We follow SR3 [72] where human subjects were shown a low-res image in between two high-res images and asked for preference. The results in Tab. 4 affirm the good performance of LDM-SR.
PSNR and SSIM can be pushed by using a post-hoc guiding mechanism [15] and we implement this image-based guider via a perceptual loss, see Sec. D.6.

bicubic

LDM-SR

SR3

SR on ImageNet

Inpainting on Places

User Study
Pixel-DM (f​1𝑓1f1)
LDM-4

LAMA [88]
LDM-4

Task 1: Preference vs GT ↑↑\uparrow
16.0%
30.4%

13.6%
21.0%

Task 2: Preference Score ↑↑\uparrow
29.4%
70.6%

31.9%
68.1%

Since the bicubic degradation process does not generalize well to images which do not follow this pre-processing, we also train a generic model, LDM-BSR, by using more diverse degradation. The results are shown in Sec. D.6.1.

Method
FID ↓↓\downarrow
IS ↑↑\uparrow
PSNR ↑↑\uparrow
SSIM ↑↑\uparrow
Nparamssubscript𝑁paramsN_{\text{params}}
[sampless](∗)[\frac{\text{samples}}{s}](^{*})

Image Regression [72]
15.2
121.1
27.9
0.801
625M
N/A

SR3 [72]
5.2
180.1
26.4
0.762
625M
N/A

LDM-4 (ours, 100 steps)
2.8†/4.8‡
166.3
24.4±plus-or-minus\pm3.8
0.69±plus-or-minus\pm0.14
169M
4.62

emphLDM-4 (ours, big, 100 steps)
2.4†/4.3‡
174.9
24.7±plus-or-minus\pm4.1
0.71±plus-or-minus\pm0.15
552M
4.5

LDM-4 (ours, 50 steps, guiding)
4.4†/6.4‡
153.7
25.8±plus-or-minus\pm3.7
0.74±plus-or-minus\pm0.12
184M
0.38

### 4.5 Inpainting with Latent Diffusion

Inpainting is the task of filling masked regions of an image with new content either
because parts of the image are are corrupted or to
replace existing but undesired content within the image. We evaluate how our
general approach for conditional image generation compares to more specialized,
state-of-the-art approaches for this task. Our evaluation follows the protocol
of LaMa[88], a recent inpainting model that introduces a specialized
architecture relying on Fast Fourier Convolutions[8].
The exact training & evaluation protocol on Places[108] is described in Sec. E.2.2.

We first analyze the effect of different design choices for
the first stage.

train throughput
sampling throughput†
train+val
FID@2k

Model (reg.-type)
samples/sec.
@256
@512
hours/epoch
epoch 6

LDM-1 (no first stage)
0.11
0.26
0.07
20.66
24.74

LDM-4 (KL, w/ attn)
0.32
0.97
0.34
7.66
15.21

LDM-4 (VQ, w/ attn)
0.33
0.97
0.34
7.04
14.99

LDM-4 (VQ, w/o attn)
0.35
0.99
0.36
6.66
15.95

input
result

In particular, we compare the inpainting efficiency of LDM-1
(i.e. a pixel-based conditional DM) with
LDM-4, for both KL and VQ regularizations,
as well as VQ-LDM-4 without any attention in the first stage
(see Tab. 8), where the latter reduces GPU memory for decoding at high resolutions. For comparability, we fix the number of parameters for all models.
Tab. 6 reports the training and sampling throughput
at resolution 2562superscript2562256^{2} and 5122superscript5122512^{2}, the total training time in hours
per epoch and the FID score on the validation split after six epochs. Overall,
we observe a speed-up of at least 2.7×2.7\times between pixel- and latent-based
diffusion models while improving FID scores by a factor of at least 1.6×1.6\times.

The comparison with other inpainting approaches in Tab. 7
shows that our model with attention improves the overall image quality as
measured by FID over that of [88]. LPIPS between the unmasked
images and our samples is slightly higher than that of [88].
We attribute this to [88] only producing a
single result which tends to recover more of an average image
compared to the diverse results produced by our LDM
cf. Fig. 21.
Additionally in a user study (Tab. 4) human subjects favor our results over those of [88].

Based on these initial results, we also trained a larger diffusion model
(big in Tab. 7) in the
latent space of the VQ-regularized first stage without attention.
Following [15], the UNet of this diffusion model uses attention layers on
three levels of its feature hierarchy, the BigGAN [3] residual block
for up- and downsampling and has 387M parameters instead of 215M.
After training, we noticed a discrepancy in the quality of samples produced at
resolutions 2562superscript2562256^{2} and 5122superscript5122512^{2}, which we hypothesize to be caused by the
additional attention modules. However, fine-tuning the model for half an epoch
at resolution 5122superscript5122512^{2} allows the model to adjust to the new feature statistics
and sets a new state of the art FID on image inpainting (big, w/o attn, w/
ft in Tab. 7, Fig. 11.).

40-50% masked
All samples

Method
FID ↓↓\downarrow
LPIPS ↓↓\downarrow
FID ↓↓\downarrow
LPIPS ↓↓\downarrow

LDM-4 (ours, big, w/ ft)
9.39
0.246±plus-or-minus\pm 0.042
1.50
0.137±plus-or-minus\pm 0.080

LDM-4 (ours, big, w/o ft)
12.89
0.257±plus-or-minus\pm 0.047
2.40
0.142±plus-or-minus\pm 0.085

LDM-4 (ours, w/ attn)
11.87
0.257±plus-or-minus\pm 0.042
2.15
0.144±plus-or-minus\pm 0.084

LDM-4 (ours, w/o attn)
12.60
0.259±plus-or-minus\pm 0.041
2.37
0.145±plus-or-minus\pm 0.084

LaMa[88]†
12.31
0.243±plus-or-minus\pm 0.038
2.23
0.134±plus-or-minus\pm 0.080

LaMa[88]
12.0
0.24±plus-or-minus\pm 0.000
2.21
0.14±plus-or-minus\pm 0.000

CoModGAN[107]
10.4
0.26±plus-or-minus\pm 0.000
1.82
0.15±plus-or-minus\pm 0.000

RegionWise[52]
21.3
0.27±plus-or-minus\pm 0.000
4.75
0.15±plus-or-minus\pm 0.000

DeepFill v2[104]
22.1
0.28±plus-or-minus\pm 0.000
5.20
0.16±plus-or-minus\pm 0.000

EdgeConnect[58]
30.5
0.28±plus-or-minus\pm 0.000
8.37
0.16±plus-or-minus\pm 0.000

## 5 Limitations & Societal Impact

##### Limitations

While LDMs significantly reduce computational requirements compared to pixel-based approaches,
their sequential sampling process is still slower than that of GANs.
Moreover, the use of LDMs can be questionable when high precision is required:
although the loss of image quality is very small in our f=4𝑓4f=4 autoencoding models (see Fig. 1),
their reconstruction capability can become a bottleneck for tasks that require fine-grained accuracy in pixel space.
We assume that our superresolution models (Sec. 4.4) are already somewhat limited in this respect.

##### Societal Impact

Generative models for media like imagery are a double-edged sword: On the one hand, they enable various creative applications,
and in particular approaches like ours that reduce the cost of training and inference have the potential
to facilitate access to this technology and democratize its exploration.
On the other hand, it also means that it becomes easier to create and disseminate manipulated data or spread misinformation and spam.
In particular, the deliberate manipulation of images (“deep fakes”) is a common problem in this context,
and women in particular are disproportionately affected by it [13, 24].

Generative models can also reveal their training data [5, 90],
which is of great concern when the data contain sensitive or personal information
and were collected without explicit consent.
However, the extent to which this also applies to DMs of images is not yet fully understood.

Finally, deep learning modules tend to reproduce or exacerbate biases that are already present in the data [91, 38, 22].
While diffusion models achieve better coverage of the data distribution than e.g. GAN-based approaches,
the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective
misrepresents the data remains an important research question.

For a more general, detailed discussion of the ethical considerations of deep generative models, see e.g. [13].

## 6 Conclusion

We have presented latent diffusion models, a simple and efficient way to significantly improve both the
training and sampling efficiency of denoising diffusion models without
degrading their quality. Based on this and our cross-attention
conditioning mechanism, our experiments could demonstrate favorable results
compared to state-of-the-art methods across a wide range of conditional image
synthesis tasks without task-specific architectures.
††This work has been supported by the German Federal Ministry for Economic Affairs and Energy within the project ’KI-Absicherung - Safe AI for automated driving’ and by the German Research Foundation (DFG) project 421703927.

## References

- [1]

Eirikur Agustsson and Radu Timofte.

NTIRE 2017 challenge on single image super-resolution: Dataset and
study.

In 2017 IEEE Conference on Computer Vision and Pattern
Recognition Workshops, CVPR Workshops 2017, Honolulu, HI, USA, July 21-26,
2017, pages 1122–1131. IEEE Computer Society, 2017.

- [2]

Martin Arjovsky, Soumith Chintala, and Léon Bottou.

Wasserstein gan, 2017.

- [3]

Andrew Brock, Jeff Donahue, and Karen Simonyan.

Large scale GAN training for high fidelity natural image synthesis.

In Int. Conf. Learn. Represent., 2019.

- [4]

Holger Caesar, Jasper R. R. Uijlings, and Vittorio Ferrari.

Coco-stuff: Thing and stuff classes in context.

In 2018 IEEE Conference on Computer Vision and Pattern
Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages
1209–1218. Computer Vision Foundation / IEEE Computer Society, 2018.

- [5]

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel
Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar
Erlingsson, et al.

Extracting training data from large language models.

In 30th USENIX Security Symposium (USENIX Security 21), pages
2633–2650, 2021.

- [6]

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and
Ilya Sutskever.

Generative pretraining from pixels.

In ICML, volume 119 of Proceedings of Machine Learning
Research, pages 1691–1703. PMLR, 2020.

- [7]

Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William
Chan.

Wavegrad: Estimating gradients for waveform generation.

In ICLR. OpenReview.net, 2021.

- [8]

Lu Chi, Borui Jiang, and Yadong Mu.

Fast fourier convolution.

In NeurIPS, 2020.

- [9]

Rewon Child.

Very deep vaes generalize autoregressive models and can outperform
them on images.

CoRR, abs/2011.10650, 2020.

- [10]

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever.

Generating long sequences with sparse transformers.

CoRR, abs/1904.10509, 2019.

- [11]

Bin Dai and David P. Wipf.

Diagnosing and enhancing VAE models.

In ICLR (Poster). OpenReview.net, 2019.

- [12]

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li.

Imagenet: A large-scale hierarchical image database.

In CVPR, pages 248–255. IEEE Computer Society, 2009.

- [13]

Emily Denton.

Ethical considerations of generative ai.

AI for Content Creation Workshop, CVPR, 2021.

- [14]

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

BERT: pre-training of deep bidirectional transformers for language
understanding.

CoRR, abs/1810.04805, 2018.

- [15]

Prafulla Dhariwal and Alex Nichol.

Diffusion models beat gans on image synthesis.

CoRR, abs/2105.05233, 2021.

- [16]

Sander Dieleman.

Musings on typicality, 2020.

- [17]

Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang
Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang.

Cogview: Mastering text-to-image generation via transformers.

CoRR, abs/2105.13290, 2021.

- [18]

Laurent Dinh, David Krueger, and Yoshua Bengio.

Nice: Non-linear independent components estimation, 2015.

- [19]

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.

Density estimation using real NVP.

In 5th International Conference on Learning Representations,
ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track
Proceedings. OpenReview.net, 2017.

- [20]

Alexey Dosovitskiy and Thomas Brox.

Generating images with perceptual similarity metrics based on deep
networks.

In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle
Guyon, and Roman Garnett, editors, Adv. Neural Inform. Process. Syst.,
pages 658–666, 2016.

- [21]

Patrick Esser, Robin Rombach, Andreas Blattmann, and Björn Ommer.

Imagebart: Bidirectional context with multinomial diffusion for
autoregressive image synthesis.

CoRR, abs/2108.08827, 2021.

- [22]

Patrick Esser, Robin Rombach, and Björn Ommer.

A note on data biases in generative models.

arXiv preprint arXiv:2012.02516, 2020.

- [23]

Patrick Esser, Robin Rombach, and Björn Ommer.

Taming transformers for high-resolution image synthesis.

CoRR, abs/2012.09841, 2020.

- [24]

Mary Anne Franks and Ari Ezra Waldman.

Sex, lies, and videotape: Deep fakes and free speech delusions.

Md. L. Rev., 78:892, 2018.

- [25]

Kevin Frans, Lisa B. Soros, and Olaf Witkowski.

Clipdraw: Exploring text-to-drawing synthesis through language-image
encoders.

ArXiv, abs/2106.14843, 2021.

- [26]

Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv
Taigman.

Make-a-scene: Scene-based text-to-image generation with human priors.

CoRR, abs/2203.13131, 2022.

- [27]

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David
Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio.

Generative adversarial networks.

CoRR, 2014.

- [28]

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron
Courville.

Improved training of wasserstein gans, 2017.

- [29]

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter.

Gans trained by a two time-scale update rule converge to a local nash
equilibrium.

In Adv. Neural Inform. Process. Syst., pages 6626–6637, 2017.

- [30]

Jonathan Ho, Ajay Jain, and Pieter Abbeel.

Denoising diffusion probabilistic models.

In NeurIPS, 2020.

- [31]

Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi,
and Tim Salimans.

Cascaded diffusion models for high fidelity image generation.

CoRR, abs/2106.15282, 2021.

- [32]

Jonathan Ho and Tim Salimans.

Classifier-free diffusion guidance.

In NeurIPS 2021 Workshop on Deep Generative Models and
Downstream Applications, 2021.

- [33]

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.

Image-to-image translation with conditional adversarial networks.

In CVPR, pages 5967–5976. IEEE Computer Society, 2017.

- [34]

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.

Image-to-image translation with conditional adversarial networks.

2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5967–5976, 2017.

- [35]

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch,
Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan
Shelhamer, Olivier J. Hénaff, Matthew M. Botvinick, Andrew Zisserman,
Oriol Vinyals, and João Carreira.

Perceiver IO: A general architecture for structured inputs
&outputs.

CoRR, abs/2107.14795, 2021.

- [36]

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and
João Carreira.

Perceiver: General perception with iterative attention.

In Marina Meila and Tong Zhang, editors, Proceedings of the 38th
International Conference on Machine Learning, ICML 2021, 18-24 July 2021,
Virtual Event, volume 139 of Proceedings of Machine Learning Research,
pages 4651–4664. PMLR, 2021.

- [37]

Manuel Jahn, Robin Rombach, and Björn Ommer.

High-resolution complex scene synthesis with transformers.

CoRR, abs/2105.06458, 2021.

- [38]

Niharika Jain, Alberto Olmo, Sailik Sengupta, Lydia Manikonda, and Subbarao
Kambhampati.

Imperfect imaganation: Implications of gans exacerbating biases on
facial data augmentation and snapchat selfie lenses.

arXiv preprint arXiv:2001.09528, 2020.

- [39]

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.

Progressive growing of gans for improved quality, stability, and
variation.

CoRR, abs/1710.10196, 2017.

- [40]

Tero Karras, Samuli Laine, and Timo Aila.

A style-based generator architecture for generative adversarial
networks.

In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401–4410,
2019.

- [41]

T. Karras, S. Laine, and T. Aila.

A style-based generator architecture for generative adversarial
networks.

In 2019 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 2019.

- [42]

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and
Timo Aila.

Analyzing and improving the image quality of stylegan.

CoRR, abs/1912.04958, 2019.

- [43]

Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

Score matching model for unbounded data score.

CoRR, abs/2106.05527, 2021.

- [44]

Durk P Kingma and Prafulla Dhariwal.

Glow: Generative flow with invertible 1x1 convolutions.

In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
and R. Garnett, editors, Advances in Neural Information Processing
Systems, 2018.

- [45]

Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho.

Variational diffusion models.

CoRR, abs/2107.00630, 2021.

- [46]

Diederik P. Kingma and Max Welling.

Auto-Encoding Variational Bayes.

In 2nd International Conference on Learning Representations,
ICLR, 2014.

- [47]

Zhifeng Kong and Wei Ping.

On fast sampling of diffusion probabilistic models.

CoRR, abs/2106.00132, 2021.

- [48]

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro.

Diffwave: A versatile diffusion model for audio synthesis.

In ICLR. OpenReview.net, 2021.

- [49]

Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R. Uijlings, Ivan Krasin,
Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Tom Duerig,
and Vittorio Ferrari.

The open images dataset V4: unified image classification, object
detection, and visual relationship detection at scale.

CoRR, abs/1811.00982, 2018.

- [50]

Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen,
and Timo Aila.

Improved precision and recall metric for assessing generative models.

CoRR, abs/1904.06991, 2019.

- [51]

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B.
Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and
C. Lawrence Zitnick.

Microsoft COCO: common objects in context.

CoRR, abs/1405.0312, 2014.

- [52]

Yuqing Ma, Xianglong Liu, Shihao Bai, Le-Yi Wang, Aishan Liu, Dacheng Tao, and
Edwin Hancock.

Region-wise generative adversarial imageinpainting for large missing
areas.

ArXiv, abs/1909.12507, 2019.

- [53]

Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano
Ermon.

Sdedit: Image synthesis and editing with stochastic differential
equations.

CoRR, abs/2108.01073, 2021.

- [54]

Lars M. Mescheder.

On the convergence properties of GAN training.

CoRR, abs/1801.04406, 2018.

- [55]

Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein.

Unrolled generative adversarial networks.

In 5th International Conference on Learning Representations,
ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track
Proceedings. OpenReview.net, 2017.

- [56]

Mehdi Mirza and Simon Osindero.

Conditional generative adversarial nets.

CoRR, abs/1411.1784, 2014.

- [57]

Gautam Mittal, Jesse H. Engel, Curtis Hawthorne, and Ian Simon.

Symbolic music generation with diffusion models.

CoRR, abs/2103.16091, 2021.

- [58]

Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, and Mehran Ebrahimi.

Edgeconnect: Generative image inpainting with adversarial edge
learning.

ArXiv, abs/1901.00212, 2019.

- [59]

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin,
Bob McGrew, Ilya Sutskever, and Mark Chen.

GLIDE: towards photorealistic image generation and editing with
text-guided diffusion models.

CoRR, abs/2112.10741, 2021.

- [60]

Anton Obukhov, Maximilian Seitzer, Po-Wei Wu, Semen Zhydenko, Jonathan Kyl, and
Elvis Yu-Jing Lin.

High-fidelity performance metrics for generative models in pytorch,
2020.

Version: 0.3.0, DOI: 10.5281/zenodo.4957738.

- [61]

Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu.

Semantic image synthesis with spatially-adaptive normalization.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, 2019.

- [62]

Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and Jun-Yan Zhu.

Semantic image synthesis with spatially-adaptive normalization.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), June 2019.

- [63]

Gaurav Parmar, Dacheng Li, Kwonjoon Lee, and Zhuowen Tu.

Dual contradistinctive generative autoencoder.

In IEEE Conference on Computer Vision and Pattern Recognition,
CVPR 2021, virtual, June 19-25, 2021, pages 823–832. Computer Vision
Foundation / IEEE, 2021.

- [64]

Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu.

On buggy resizing libraries and surprising subtleties in fid
calculation.

arXiv preprint arXiv:2104.11222, 2021.

- [65]

David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel
Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean.

Carbon emissions and large neural network training.

CoRR, abs/2104.10350, 2021.

- [66]

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec
Radford, Mark Chen, and Ilya Sutskever.

Zero-shot text-to-image generation.

CoRR, abs/2102.12092, 2021.

- [67]

Ali Razavi, Aäron van den Oord, and Oriol Vinyals.

Generating diverse high-fidelity images with VQ-VAE-2.

In NeurIPS, pages 14837–14847, 2019.

- [68]

Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele,
and Honglak Lee.

Generative adversarial text to image synthesis.

In ICML, 2016.

- [69]

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra.

Stochastic backpropagation and approximate inference in deep
generative models.

In Proceedings of the 31st International Conference on
International Conference on Machine Learning, ICML, 2014.

- [70]

Robin Rombach, Patrick Esser, and Björn Ommer.

Network-to-network translation with conditional invertible neural
networks.

In NeurIPS, 2020.

- [71]

Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

U-net: Convolutional networks for biomedical image segmentation.

In MICCAI (3), volume 9351 of Lecture Notes in
Computer Science, pages 234–241. Springer, 2015.

- [72]

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and
Mohammad Norouzi.

Image super-resolution via iterative refinement.

CoRR, abs/2104.07636, 2021.

- [73]

Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.

Pixelcnn++: Improving the pixelcnn with discretized logistic mixture
likelihood and other modifications.

CoRR, abs/1701.05517, 2017.

- [74]

Dave Salvator.

NVIDIA Developer Blog.

https://developer.nvidia.com/blog/getting-immediate-speedups-with-a100-tf32,
2020.

- [75]

Robin San-Roman, Eliya Nachmani, and Lior Wolf.

Noise estimation for generative diffusion models.

CoRR, abs/2104.02600, 2021.

- [76]

Axel Sauer, Kashyap Chitta, Jens Müller, and Andreas Geiger.

Projected gans converge faster.

CoRR, abs/2111.01007, 2021.

- [77]

Edgar Schönfeld, Bernt Schiele, and Anna Khoreva.

A u-net based discriminator for generative adversarial networks.

In 2020 IEEE/CVF Conference on Computer Vision and Pattern
Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages
8204–8213. Computer Vision Foundation / IEEE, 2020.

- [78]

Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk,
Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran
Komatsuzaki.

Laion-400m: Open dataset of clip-filtered 400 million image-text
pairs, 2021.

- [79]

Karen Simonyan and Andrew Zisserman.

Very deep convolutional networks for large-scale image recognition.

In Yoshua Bengio and Yann LeCun, editors, Int. Conf. Learn.
Represent., 2015.

- [80]

Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon.

D2C: diffusion-denoising models for few-shot conditional
generation.

CoRR, abs/2106.06819, 2021.

- [81]

Charlie Snell.

Alien Dreams: An Emerging Art Scene.

https://ml.berkeley.edu/blog/posts/clip-art/, 2021.

[Online; accessed November-2021].

- [82]

Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya
Ganguli.

Deep unsupervised learning using nonequilibrium thermodynamics.

CoRR, abs/1503.03585, 2015.

- [83]

Kihyuk Sohn, Honglak Lee, and Xinchen Yan.

Learning structured output representation using deep conditional
generative models.

In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett,
editors, Advances in Neural Information Processing Systems, volume 28.
Curran Associates, Inc., 2015.

- [84]

Jiaming Song, Chenlin Meng, and Stefano Ermon.

Denoising diffusion implicit models.

In ICLR. OpenReview.net, 2021.

- [85]

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano
Ermon, and Ben Poole.

Score-based generative modeling through stochastic differential
equations.

CoRR, abs/2011.13456, 2020.

- [86]

Emma Strubell, Ananya Ganesh, and Andrew McCallum.

Energy and policy considerations for modern deep learning research.

In The Thirty-Fourth AAAI Conference on Artificial
Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of
Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium
on Educational Advances in Artificial Intelligence, EAAI 2020, New York,
NY, USA, February 7-12, 2020, pages 13693–13696. AAAI Press, 2020.

- [87]

Wei Sun and Tianfu Wu.

Learning layout and style reconfigurable gans for controllable image
synthesis.

CoRR, abs/2003.11571, 2020.

- [88]

Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova,
Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong
Park, and Victor S. Lempitsky.

Resolution-robust large mask inpainting with fourier convolutions.

ArXiv, abs/2109.07161, 2021.

- [89]

Tristan Sylvain, Pengchuan Zhang, Yoshua Bengio, R. Devon Hjelm, and Shikhar
Sharma.

Object-centric image generation from layouts.

In Thirty-Fifth AAAI Conference on Artificial Intelligence,
AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial
Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in
Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021,
pages 2647–2655. AAAI Press, 2021.

- [90]

Patrick Tinsley, Adam Czajka, and Patrick Flynn.

This face does not exist… but it might be yours! identity leakage
in generative models.

In Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision, pages 1320–1328, 2021.

- [91]

Antonio Torralba and Alexei A Efros.

Unbiased look at dataset bias.

In CVPR 2011, pages 1521–1528. IEEE, 2011.

- [92]

Arash Vahdat and Jan Kautz.

NVAE: A deep hierarchical variational autoencoder.

In NeurIPS, 2020.

- [93]

Arash Vahdat, Karsten Kreis, and Jan Kautz.

Score-based generative modeling in latent space.

CoRR, abs/2106.05931, 2021.

- [94]

Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, koray kavukcuoglu, Oriol
Vinyals, and Alex Graves.

Conditional image generation with pixelcnn decoders.

In Advances in Neural Information Processing Systems, 2016.

- [95]

Aäron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu.

Pixel recurrent neural networks.

CoRR, abs/1601.06759, 2016.

- [96]

Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu.

Neural discrete representation learning.

In NIPS, pages 6306–6315, 2017.

- [97]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In NIPS, pages 5998–6008, 2017.

- [98]

Rivers Have Wings.

Tweet on Classifier-free guidance for autoregressive models.

https://twitter.com/RiversHaveWings/status/1478093658716966912,
2022.

- [99]

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
and Jamie Brew.

Huggingface’s transformers: State-of-the-art natural language
processing.

CoRR, abs/1910.03771, 2019.

- [100]

Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat.

VAEBM: A symbiosis between variational autoencoders and
energy-based models.

In 9th International Conference on Learning Representations,
ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

- [101]

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas.

Videogpt: Video generation using VQ-VAE and transformers.

CoRR, abs/2104.10157, 2021.

- [102]

Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao.

LSUN: construction of a large-scale image dataset using deep
learning with humans in the loop.

CoRR, abs/1506.03365, 2015.

- [103]

Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander
Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu.

Vector-quantized image modeling with improved vqgan, 2021.

- [104]

Jiahui Yu, Zhe L. Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang.

Free-form image inpainting with gated convolution.

2019 IEEE/CVF International Conference on Computer Vision
(ICCV), pages 4470–4479, 2019.

- [105]

K. Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte.

Designing a practical degradation model for deep blind image
super-resolution.

ArXiv, abs/2103.14006, 2021.

- [106]

Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang.

The unreasonable effectiveness of deep features as a perceptual
metric.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), June 2018.

- [107]

Shengyu Zhao, Jianwei Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I-Chao
Chang, and Yan Xu.

Large scale image completion via co-modulated generative adversarial
networks.

ArXiv, abs/2103.10428, 2021.

- [108]

Bolei Zhou, Àgata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio
Torralba.

Places: A 10 million image database for scene recognition.

IEEE Transactions on Pattern Analysis and Machine Intelligence,
40:1452–1464, 2018.

- [109]

Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu,
Jiuxiang Gu, Jinhui Xu, and Tong Sun.

LAFITE: towards language-free training for text-to-image
generation.

CoRR, abs/2111.13792, 2021.

Appendix

’A painting of the last supper by Picasso.’

’An oil painting of a latent space.’

’An epic painting of Gandalf the Black
summoning thunder and lightning in the mountains.’

’A sunset over a mountain range, vector image.’

## Appendix A Changelog

Here we list changes between this version (https://arxiv.org/abs/2112.10752v2) of the paper and the previous version, i.e. https://arxiv.org/abs/2112.10752v1.

- •

We updated the results on text-to-image synthesis in Sec. 4.3 which were obtained by training a new, larger model (1.45B parameters). This also includes a new comparison to very recent competing methods on this task that were published on arXiv at the same time as ([59, 109]) or after ([26]) the publication of our work.

- •

We updated results on class-conditional synthesis on ImageNet in Sec. 4.1, Tab. 3 (see also Sec. D.4) obtained by retraining the model with a larger batch size.
The corresponding qualitative results in Fig. 26 and Fig. 27 were also updated. Both the updated text-to-image and the class-conditional model now use classifier-free guidance [32] as a measure to increase visual fidelity.

- •

We conducted a user study (following the scheme suggested by Saharia et al [72]) which provides additional evaluation for our inpainting (Sec. 4.5) and superresolution models (Sec. 4.4).

- •

Added Fig. 5 to the main paper, moved Fig. 18 to the appendix, added Fig. 13 to the appendix.

## Appendix B Detailed Information on Denoising Diffusion Models

Diffusion models can be specified in terms of a signal-to-noise ratio
SNR​(t)=αt2σt2SNR𝑡superscriptsubscript𝛼𝑡2superscriptsubscript𝜎𝑡2\text{SNR}(t)=\frac{\alpha_{t}^{2}}{\sigma_{t}^{2}} consisting
of sequences (αt)t=1Tsuperscriptsubscriptsubscript𝛼𝑡𝑡1𝑇(\alpha_{t})_{t=1}^{T} and (σt)t=1Tsuperscriptsubscriptsubscript𝜎𝑡𝑡1𝑇(\sigma_{t})_{t=1}^{T} which, starting from
a data sample x0subscript𝑥0x_{0}, define a forward diffusion process q𝑞q as

q​(xt|x0)=𝒩​(xt|αt​x0,σt2​𝕀)𝑞conditionalsubscript𝑥𝑡subscript𝑥0𝒩conditionalsubscript𝑥𝑡subscript𝛼𝑡subscript𝑥0superscriptsubscript𝜎𝑡2𝕀q(x_{t}|x_{0})=\mathcal{N}(x_{t}|\alpha_{t}x_{0},\sigma_{t}^{2}\mathbb{I})

(4)

with the Markov structure for s<t𝑠𝑡s<t:

q​(xt|xs)𝑞conditionalsubscript𝑥𝑡subscript𝑥𝑠\displaystyle q(x_{t}|x_{s})
=𝒩​(xt|αt|s​xs,σt|s2​𝕀)absent𝒩conditionalsubscript𝑥𝑡subscript𝛼conditional𝑡𝑠subscript𝑥𝑠superscriptsubscript𝜎conditional𝑡𝑠2𝕀\displaystyle=\mathcal{N}(x_{t}|\alpha_{t|s}x_{s},\sigma_{t|s}^{2}\mathbb{I})

(5)

αt|ssubscript𝛼conditional𝑡𝑠\displaystyle\alpha_{t|s}
=αtαsabsentsubscript𝛼𝑡subscript𝛼𝑠\displaystyle=\frac{\alpha_{t}}{\alpha_{s}}

(6)

σt|s2superscriptsubscript𝜎conditional𝑡𝑠2\displaystyle\sigma_{t|s}^{2}
=σt2−αt|s2​σs2absentsuperscriptsubscript𝜎𝑡2superscriptsubscript𝛼conditional𝑡𝑠2superscriptsubscript𝜎𝑠2\displaystyle=\sigma_{t}^{2}-\alpha_{t|s}^{2}\sigma_{s}^{2}

(7)

Denoising diffusion models are generative models p​(x0)𝑝subscript𝑥0p(x_{0}) which revert this
process with a similar Markov structure running backward in time, i.e. they are
specified as

p​(x0)=∫zp​(xT)​∏t=1Tp​(xt−1|xt)𝑝subscript𝑥0subscript𝑧𝑝subscript𝑥𝑇superscriptsubscriptproduct𝑡1𝑇𝑝conditionalsubscript𝑥𝑡1subscript𝑥𝑡p(x_{0})=\int_{z}p(x_{T})\prod_{t=1}^{T}p(x_{t-1}|x_{t})

(8)

The evidence lower bound (ELBO) associated with this model then decomposes over
the discrete time steps as

−logp(x0)≤𝕂𝕃(q(xT|x0)|p(xT))+∑t=1T𝔼q​(xt|x0)𝕂𝕃(q(xt−1|xt,x0)|p(xt−1|xt))-\log p(x_{0})\leq\mathbb{KL}(q(x_{T}|x_{0})|p(x_{T}))+\sum_{t=1}^{T}\mathbb{E}_{q(x_{t}|x_{0})}\mathbb{KL}(q(x_{t-1}|x_{t},x_{0})|p(x_{t-1}|x_{t}))

(9)

The prior p​(xT)𝑝subscript𝑥𝑇p(x_{T}) is typically choosen as a standard normal distribution and
the first term of the ELBO then depends only on the final signal-to-noise ratio
SNR​(T)SNR𝑇\text{SNR}(T). To minimize the remaining terms, a common choice to parameterize
p​(xt−1|xt)𝑝conditionalsubscript𝑥𝑡1subscript𝑥𝑡p(x_{t-1}|x_{t}) is to specify it in terms of the true posterior
q​(xt−1|xt,x0)𝑞conditionalsubscript𝑥𝑡1subscript𝑥𝑡subscript𝑥0q(x_{t-1}|x_{t},x_{0}) but with the unknown x0subscript𝑥0x_{0} replaced by an estimate
xθ​(xt,t)subscript𝑥𝜃subscript𝑥𝑡𝑡x_{\theta}(x_{t},t) based on the current step xtsubscript𝑥𝑡x_{t}. This gives [45]

p​(xt−1|xt)𝑝conditionalsubscript𝑥𝑡1subscript𝑥𝑡\displaystyle p(x_{t-1}|x_{t})
≔q​(xt−1|xt,xθ​(xt,t))≔absent𝑞conditionalsubscript𝑥𝑡1subscript𝑥𝑡subscript𝑥𝜃subscript𝑥𝑡𝑡\displaystyle\coloneqq q(x_{t-1}|x_{t},x_{\theta}(x_{t},t))

(10)

=𝒩​(xt−1|μθ​(xt,t),σt|t−12​σt−12σt2​𝕀),absent𝒩conditionalsubscript𝑥𝑡1subscript𝜇𝜃subscript𝑥𝑡𝑡superscriptsubscript𝜎conditional𝑡𝑡12superscriptsubscript𝜎𝑡12superscriptsubscript𝜎𝑡2𝕀\displaystyle=\mathcal{N}(x_{t-1}|\mu_{\theta}(x_{t},t),\sigma_{t|t-1}^{2}\frac{\sigma_{t-1}^{2}}{\sigma_{t}^{2}}\mathbb{I}),

(11)

where the mean can be expressed as

μθ​(xt,t)=αt|t−1​σt−12σt2​xt+αt−1​σt|t−12σt2​xθ​(xt,t).subscript𝜇𝜃subscript𝑥𝑡𝑡subscript𝛼conditional𝑡𝑡1superscriptsubscript𝜎𝑡12superscriptsubscript𝜎𝑡2subscript𝑥𝑡subscript𝛼𝑡1superscriptsubscript𝜎conditional𝑡𝑡12superscriptsubscript𝜎𝑡2subscript𝑥𝜃subscript𝑥𝑡𝑡\mu_{\theta}(x_{t},t)=\frac{\alpha_{t|t-1}\sigma_{t-1}^{2}}{\sigma_{t}^{2}}x_{t}+\frac{\alpha_{t-1}\sigma_{t|t-1}^{2}}{\sigma_{t}^{2}}x_{\theta}(x_{t},t).

(12)

In this case, the sum of the ELBO simplify to

∑t=1T𝔼q​(xt|x0)𝕂𝕃(q(xt−1|xt,x0)|p(xt−1)=∑t=1T𝔼𝒩​(ϵ|0,𝕀)12(SNR(t−1)−SNR(t))∥x0−xθ(αtx0+σtϵ,t)∥2\sum_{t=1}^{T}\mathbb{E}_{q(x_{t}|x_{0})}\mathbb{KL}(q(x_{t-1}|x_{t},x_{0})|p(x_{t-1})=\sum_{t=1}^{T}\mathbb{E}_{\mathcal{N}(\epsilon|0,\mathbb{I})}\frac{1}{2}(\text{SNR}(t-1)-\text{SNR}(t))\|x_{0}-x_{\theta}(\alpha_{t}x_{0}+\sigma_{t}\epsilon,t)\|^{2}

(13)

Following [30], we use the reparameterization

ϵθ​(xt,t)=(xt−αt​xθ​(xt,t))/σtsubscriptitalic-ϵ𝜃subscript𝑥𝑡𝑡subscript𝑥𝑡subscript𝛼𝑡subscript𝑥𝜃subscript𝑥𝑡𝑡subscript𝜎𝑡\epsilon_{\theta}(x_{t},t)=(x_{t}-\alpha_{t}x_{\theta}(x_{t},t))/\sigma_{t}

(14)

to express the reconstruction term as a denoising objective,

‖x0−xθ​(αt​x0+σt​ϵ,t)‖2=σt2αt2​‖ϵ−ϵθ​(αt​x0+σt​ϵ,t)‖2superscriptnormsubscript𝑥0subscript𝑥𝜃subscript𝛼𝑡subscript𝑥0subscript𝜎𝑡italic-ϵ𝑡2superscriptsubscript𝜎𝑡2superscriptsubscript𝛼𝑡2superscriptnormitalic-ϵsubscriptitalic-ϵ𝜃subscript𝛼𝑡subscript𝑥0subscript𝜎𝑡italic-ϵ𝑡2\|x_{0}-x_{\theta}(\alpha_{t}x_{0}+\sigma_{t}\epsilon,t)\|^{2}=\frac{\sigma_{t}^{2}}{\alpha_{t}^{2}}\|\epsilon-\epsilon_{\theta}(\alpha_{t}x_{0}+\sigma_{t}\epsilon,t)\|^{2}

(15)

and the reweighting, which assigns each of the terms the same weight and
results in Eq. (1).

## Appendix C Image Guiding Mechanisms

Samples 2562superscript2562256^{2}

Guided Convolutional Samples 5122superscript5122512^{2}

Convolutional Samples 5122superscript5122512^{2}

An intriguing feature of diffusion models is that unconditional models can be conditioned at test-time [85, 82, 15].
In particular, [15] presented an algorithm to guide both unconditional and conditional models
trained on the ImageNet dataset with a classifier log⁡pΦ​(y|xt)subscript𝑝Φconditional𝑦subscript𝑥𝑡\log p_{\Phi}(y|x_{t}), trained on each xtsubscript𝑥𝑡x_{t} of the diffusion process.
We directly build on this formulation and introduce post-hoc image-guiding:

For an epsilon-parameterized model with fixed variance, the guiding algorithm as introduced in [15] reads:

ϵ^←ϵθ​(zt,t)+1−αt2​∇ztlog⁡pΦ​(y|zt).←^italic-ϵsubscriptitalic-ϵ𝜃subscript𝑧𝑡𝑡1superscriptsubscript𝛼𝑡2subscript∇subscript𝑧𝑡subscript𝑝Φconditional𝑦subscript𝑧𝑡\hat{\epsilon}\leftarrow\epsilon_{\theta}(z_{t},t)+\sqrt{1-\alpha_{t}^{2}}\;\nabla_{z_{t}}\log p_{\Phi}(y|z_{t})\;.

(16)

This can be interpreted as an update correcting the “score” ϵθsubscriptitalic-ϵ𝜃\epsilon_{\theta} with a conditional distribution
log⁡pΦ​(y|zt)subscript𝑝Φconditional𝑦subscript𝑧𝑡\log p_{\Phi}(y|z_{t}).

So far, this scenario has only been applied to single-class classification models. We re-interpret the guiding distribution pΦ​(y|T​(𝒟​(z0​(zt))))subscript𝑝Φconditional𝑦𝑇𝒟subscript𝑧0subscript𝑧𝑡p_{\Phi}(y|T(\mathcal{D}(z_{0}(z_{t}))))
as a general purpose image-to-image translation task given a target image y𝑦y,
where T𝑇T can be any differentiable transformation adopted to the image-to-image translation task at hand,
such as the identity, a downsampling operation or similar.

As an example, we can assume a Gaussian guider with fixed variance σ2=1superscript𝜎21\sigma^{2}=1, such that

log⁡pΦ​(y|zt)=−12​‖y−T​(𝒟​(z0​(zt)))‖22subscript𝑝Φconditional𝑦subscript𝑧𝑡12subscriptsuperscriptnorm𝑦𝑇𝒟subscript𝑧0subscript𝑧𝑡22\log p_{\Phi}(y|z_{t})=-\frac{1}{2}\|y-T(\mathcal{D}(z_{0}(z_{t})))\|^{2}_{2}

(17)

becomes a L2subscript𝐿2L_{2} regression objective.

Fig. 14 demonstrates how this formulation can serve as an upsampling mechanism
of an unconditional model trained on 2562superscript2562256^{2} images, where unconditional samples of size 2562superscript2562256^{2} guide the
convolutional synthesis of 5122superscript5122512^{2} images
and T𝑇T is a 2×2\times bicubic downsampling.
Following this motivation,
we also experiment with a perceptual similarity guiding and replace the L2subscript𝐿2L_{2} objective with
the LPIPS [106] metric, see Sec. 4.4.

## Appendix D Additional Results

### D.1 Choosing the Signal-to-Noise Ratio for High-Resolution Synthesis

KL-reg, w/o rescaling
KL-reg, w/ rescaling
VQ-reg, w/o rescaling

As discussed in Sec. 4.3.2, the signal-to-noise ratio induced by the variance of the latent
space (i.e. Var(z)/σt2Var(z)subscriptsuperscript𝜎2𝑡\text{Var(z)}/\sigma^{2}_{t}) significantly affects the results for convolutional sampling.
For example, when training a LDM directly in the latent space of a KL-regularized model (see Tab. 8),
this ratio is very high, such that the model allocates a lot of semantic detail early on in the reverse denoising process.
In contrast, when rescaling the latent space by the component-wise standard deviation of the latents as described in Sec. G,
the SNR is descreased. We illustrate the effect on convolutional sampling for semantic image synthesis
in Fig. 15. Note that the VQ-regularized space has a variance close to 111,
such that it does not have to be rescaled.

### D.2 Full List of all First Stage Models

We provide a complete list of various autoenconding models trained on the OpenImages dataset in Tab. 8.

f𝑓f
|𝒵|𝒵|\mathcal{Z}|
c𝑐c
R-FID ↓↓\downarrow
R-IS ↑↑\uparrow
PSNR ↑↑\uparrow
PSIM ↓↓\downarrow
SSIM ↑↑\uparrow

16 VQGAN [23]
16384
256
4.98
–
19.9 ±3.4plus-or-minus3.4\pm 3.4
1.83 ±0.42plus-or-minus0.42\pm 0.42
0.51 ±0.18plus-or-minus0.18\pm 0.18

16 VQGAN [23]
1024
256
7.94
–
19.4 ±3.3plus-or-minus3.3\pm 3.3
1.98 ±0.43plus-or-minus0.43\pm 0.43
0.50 ±0.18plus-or-minus0.18\pm 0.18

8 DALL-E [66]
8192
-
32.01
–
22.8 ±2.1plus-or-minus2.1\pm 2.1
1.95 ±0.51plus-or-minus0.51\pm 0.51
0.73 ±0.13plus-or-minus0.13\pm 0.13

32
16384
16
31.83
40.40 ±1.07plus-or-minus1.07\pm 1.07
17.45 ±2.90plus-or-minus2.90\pm 2.90
2.58 ±0.48plus-or-minus0.48\pm 0.48
0.41 ±0.18plus-or-minus0.18\pm 0.18

16
16384
8
5.15
144.55 ±3.74plus-or-minus3.74\pm 3.74
20.83 ±3.61plus-or-minus3.61\pm 3.61
1.73 ±0.43plus-or-minus0.43\pm 0.43
0.54 ±0.18plus-or-minus0.18\pm 0.18

8
16384
4
1.14
201.92 ±3.97plus-or-minus3.97\pm 3.97
23.07 ±3.99plus-or-minus3.99\pm 3.99
1.17 ±0.36plus-or-minus0.36\pm 0.36
0.65 ±0.16plus-or-minus0.16\pm 0.16

8
256
4
1.49
194.20 ±3.87plus-or-minus3.87\pm 3.87
22.35 ±3.81plus-or-minus3.81\pm 3.81
1.26 ±0.37plus-or-minus0.37\pm 0.37
0.62 ±0.16plus-or-minus0.16\pm 0.16

4
8192
3
0.58
224.78 ±5.35plus-or-minus5.35\pm 5.35
27.43 ±4.26plus-or-minus4.26\pm 4.26
0.53 ±0.21plus-or-minus0.21\pm 0.21
0.82 ±0.10plus-or-minus0.10\pm 0.10

4†
8192
3
1.06
221.94 ±4.58plus-or-minus4.58\pm 4.58
25.21 ±4.17plus-or-minus4.17\pm 4.17
0.72 ±0.26plus-or-minus0.26\pm 0.26
0.76 ±0.12plus-or-minus0.12\pm 0.12

4
256
3
0.47
223.81 ±4.58plus-or-minus4.58\pm 4.58
26.43 ±4.22plus-or-minus4.22\pm 4.22
0.62 ±0.24plus-or-minus0.24\pm 0.24
0.80 ±0.11plus-or-minus0.11\pm 0.11

2
2048
2
0.16
232.75 ±5.09plus-or-minus5.09\pm 5.09
30.85 ±4.12plus-or-minus4.12\pm 4.12
0.27 ±0.12plus-or-minus0.12\pm 0.12
0.91 ±0.05plus-or-minus0.05\pm 0.05

2
64
2
0.40
226.62 ±4.83plus-or-minus4.83\pm 4.83
29.13 ±3.46plus-or-minus3.46\pm 3.46
0.38 ±0.13plus-or-minus0.13\pm 0.13
0.90 ±0.05plus-or-minus0.05\pm 0.05

32
KL
64
2.04
189.53 ±3.68plus-or-minus3.68\pm 3.68
22.27 ±3.93plus-or-minus3.93\pm 3.93
1.41 ±0.40plus-or-minus0.40\pm 0.40
0.61 ±0.17plus-or-minus0.17\pm 0.17

32
KL
16
7.3
132.75 ±2.71plus-or-minus2.71\pm 2.71
20.38 ±3.56plus-or-minus3.56\pm 3.56
1.88 ±0.45plus-or-minus0.45\pm 0.45
0.53 ±0.18plus-or-minus0.18\pm 0.18

16
KL
16
0.87
210.31 ±3.97plus-or-minus3.97\pm 3.97
24.08 ±4.22plus-or-minus4.22\pm 4.22
1.07 ±0.36plus-or-minus0.36\pm 0.36
0.68 ±0.15plus-or-minus0.15\pm 0.15

16
KL
8
2.63
178.68 ±4.08plus-or-minus4.08\pm 4.08
21.94 ±3.92plus-or-minus3.92\pm 3.92
1.49 ±0.42plus-or-minus0.42\pm 0.42
0.59 ±0.17plus-or-minus0.17\pm 0.17

8
KL
4
0.90
209.90 ±4.92plus-or-minus4.92\pm 4.92
24.19 ±4.19plus-or-minus4.19\pm 4.19
1.02 ±0.35plus-or-minus0.35\pm 0.35
0.69 ±0.15plus-or-minus0.15\pm 0.15

4
KL
3
0.27
227.57 ±4.89plus-or-minus4.89\pm 4.89
27.53 ±4.54plus-or-minus4.54\pm 4.54
0.55 ±0.24plus-or-minus0.24\pm 0.24
0.82 ±0.11plus-or-minus0.11\pm 0.11

2
KL
2
0.086
232.66 ±5.16plus-or-minus5.16\pm 5.16
32.47 ±4.19plus-or-minus4.19\pm 4.19
0.20 ±0.09plus-or-minus0.09\pm 0.09
0.93 ±0.04plus-or-minus0.04\pm 0.04

### D.3 Layout-to-Image Synthesis

layout-to-image synthesis on the COCO dataset

COCO256×256256256256\times 256
OpenImages 256×256256256256\times 256
OpenImages 512×512512512512\times 512

Method
FID↓↓\downarrow
FID↓↓\downarrow
FID↓↓\downarrow

LostGAN-V2 [87]
42.55
-
-

OC-GAN [89]
41.65
-
-

SPADE [62]
41.11
-
-

VQGAN+T [37]
56.58
45.33
48.11

LDM-8 (100 steps, ours)
42.06†
-
-

LDM-4 (200 steps, ours)
40.91∗
32.02
35.80

Here we provide the quantitative evaluation and additional samples for our layout-to-image models from Sec. 4.3.1. We train a model on the COCO [4] and one on the OpenImages [49] dataset, which we subsequently additionally finetune on COCO. Tab 9 shows the result. Our COCO model reaches the performance of recent state-of-the art models in layout-to-image synthesis, when following their training and evaluation protocol [89]. When finetuning from the OpenImages model, we surpass these works. Our OpenImages model surpasses the results of Jahn et al [37] by a margin of nearly 11 in terms of FID. In Fig. 16 we show additional samples of the model finetuned on COCO.

### D.4 Class-Conditional Image Synthesis on ImageNet

Tab. 10 contains the results for our class-conditional LDM measured in FID and Inception score (IS).
LDM-8 requires significantly fewer parameters and compute requirements (see Tab. 18) to achieve very competitive performance. Similar to previous work, we can further boost the performance by training a classifier on each noise scale and guiding with it, see Sec. C.
Unlike the pixel-based methods, this classifier is trained very cheaply in latent space.
For additional qualitative results, see Fig. 26 and Fig. 27.

Method
FID↓↓\downarrow
IS↑↑\uparrow
Precision↑↑\uparrow
Recall↑↑\uparrow
Nparamssubscript𝑁paramsN_{\text{params}}

SR3 [72]
11.30
-
-
-
625M
-

ImageBART [21]
21.19
-
-
-
3.5B
-

ImageBART [21]
7.44
-
-
-
3.5B
0.05 acc. rate∗

VQGAN+T [23]
17.04
70.6±1.8plus-or-minus1.8\pm\text{1.8}
-
-
1.3B
-

VQGAN+T [23]
5.88
304.8±3.6plus-or-minus3.6\pm\text{3.6}
-
-
1.3B
0.05 acc. rate∗

BigGan-deep [3]
6.95
203.6±2.6plus-or-minus2.6\pm\text{2.6}
0.87
0.28
340M
-

ADM [15]
10.94
100.98
0.69
0.63
554M
250 DDIM steps

ADM-G [15]
4.59
186.7
0.82
0.52
608M
250 DDIM steps

ADM-G,ADM-U [15]
3.85
221.72
0.84
0.53
n/a
2 ×\times 250 DDIM steps

CDM [31]
4.88
158.71±2.26plus-or-minus2.26\pm\text{2.26}
-
-
n/a
2 ×\times 100 DDIM steps

LDM-8 (ours)
17.41
72.92±2.6plus-or-minus2.6\pm\text{2.6}
0.65
0.62
395M
200 DDIM steps, 2.9M train steps, batch size 64

LDM-8-G (ours)
8.11
190.43±2.60plus-or-minus2.60\pm\text{2.60}
0.83
0.36
506M
200 DDIM steps, classifier scale 10, 2.9M train steps, batch size 64

LDM-8 (ours)
15.51
79.03±1.03plus-or-minus1.03\pm\text{1.03}
0.65
0.63
395M
200 DDIM steps, 4.8M train steps, batch size 64

LDM-8-G (ours)
7.76
209.52±4.24plus-or-minus4.24\pm\text{4.24}
0.84
0.35
506M
200 DDIM steps, classifier scale 10, 4.8M train steps, batch size 64

LDM-4 (ours)
10.56
103.49±1.24plus-or-minus1.24\pm\text{1.24}
0.71
0.62
400M
250 DDIM steps, 178K train steps, batch size 1200

LDM-4-G (ours)
3.95
178.22±2.43plus-or-minus2.43\pm\text{2.43}
0.81
0.55
400M
250 DDIM steps, unconditional guidance [32] scale 1.25, 178K train steps, batch size 1200

LDM-4-G (ours)
3.60
247.67±5.59plus-or-minus5.59\pm\text{5.59}
0.87
0.48
400M
250 DDIM steps, unconditional guidance [32] scale 1.5, 178K train steps, batch size 1200

### D.5 Sample Quality vs. V100 Days (Continued from Sec. 4.1)

For the assessment of sample quality over the training progress in Sec. 4.1, we reported FID and IS scores as a function of train steps. Another possibility is to report these metrics over the used resources in V100 days. Such an analysis is additionally provided in Fig. 17, showing qualitatively similar results.

### D.6 Super-Resolution

Method
FID ↓↓\downarrow
IS ↑↑\uparrow
PSNR ↑↑\uparrow
SSIM ↑↑\uparrow

Image Regression [72]
15.2
121.1
27.9
0.801

SR3 [72]
5.2
180.1
26.4
0.762

LDM-4 (ours, 100 steps)
2.8†/4.8‡
166.3
24.4±plus-or-minus\pm3.8
0.69±plus-or-minus\pm0.14

LDM-4 (ours, 50 steps, guiding)
4.4†/6.4‡
153.7
25.8±plus-or-minus\pm3.7
0.74±plus-or-minus\pm0.12

LDM-4 (ours, 100 steps, guiding)
4.4†/6.4‡
154.1
25.7±plus-or-minus\pm3.7
0.73±plus-or-minus\pm0.12

LDM-4 (ours, 100 steps, +15 ep.)
2.6† / 4.6‡
169.76±plus-or-minus\pm5.03
24.4±plus-or-minus\pm3.8
0.69±plus-or-minus\pm0.14

Pixel-DM (100 steps, +15 ep.)
5.1† / 7.1‡
163.06±plus-or-minus\pm4.67
24.1±plus-or-minus\pm3.3
0.59±plus-or-minus\pm0.12

For better comparability between LDMs and diffusion models in pixel space, we extend our analysis from Tab. 5 by comparing a diffusion model trained for the same number of steps and with a comparable number 111It is not possible to exactly match both architectures since the diffusion model operates in the pixel space of parameters to our LDM. The results of this comparison are shown in the last two rows of Tab. 11 and demonstrate that LDM achieves better performance while allowing for significantly faster sampling.
A qualitative comparison is given in Fig. 20 which shows random samples from both LDM and the diffusion model in pixel space.

#### D.6.1 LDM-BSR: General Purpose SR Model via Diverse Image Degradation

bicubic
LDM-SR
LDM-BSR

To evaluate generalization of our LDM-SR, we apply it both on synthetic LDM samples from a class-conditional ImageNet model (Sec. 4.1) and images crawled from the internet.
Interestingly, we observe that LDM-SR, trained only with a bicubicly
downsampled conditioning as in [72], does
not generalize well to images which do not follow this pre-processing.
Hence, to obtain a superresolution model for a wide range of real world images, which can contain complex superpositions of camera noise, compression artifacts, blurr and interpolations, we replace the bicubic downsampling operation in LDM-SR with the degration pipeline from [105]. The BSR-degradation process is a degradation pipline which applies JPEG compressions noise, camera sensor noise, different image interpolations for downsampling, Gaussian blur kernels and Gaussian noise in a random order to an image. We found that using the bsr-degredation process with the original parameters as in [105] leads to a very strong degradation process. Since a more moderate degradation process seemed apppropiate for our application, we adapted the parameters of the bsr-degradation (our adapted degradation process can be found in our code base at https://github.com/CompVis/latent-diffusion).
Fig. 18 illustrates the effectiveness of this approach by directly comparing LDM-SR with LDM-BSR. The latter produces images much sharper than the models confined to a
fixed pre-processing, making it suitable for real-world applications.
Further results of LDM-BSR are shown on LSUN-cows in Fig. 19.

bicubic
LDM-BSR

input
GT

Pixel Baseline #1

Pixel Baseline #2

LDM #1

LDM #2

## Appendix E Implementation Details and Hyperparameters

### E.1 Hyperparameters

We provide an overview of the hyperparameters of all trained LDM models in Tab. 12, Tab. 13, Tab. 14 and Tab. 15.

CelebA-HQ 256×256256256256\times 256
FFHQ 256×256256256256\times 256
LSUN-Churches 256×256256256256\times 256
LSUN-Bedrooms 256×256256256256\times 256

f𝑓f
4
4
8
4

z𝑧z-shape
64×64×36464364\times 64\times 3
64×64×36464364\times 64\times 3
-
64×64×36464364\times 64\times 3

|𝒵|𝒵|\mathcal{Z}|
8192
8192
-
8192

Diffusion steps
1000
1000
1000
1000

Noise Schedule
linear
linear
linear
linear

Nparamssubscript𝑁paramsN_{\text{params}}
274M
274M
294M
274M

Channels
224
224
192
224

Depth
2
2
2
2

Channel Multiplier
1,2,3,4
1,2,3,4
1,2,2,4,4
1,2,3,4

Attention resolutions
32, 16, 8
32, 16, 8
32, 16, 8, 4
32, 16, 8

Head Channels
32
32
24
32

Batch Size
48
42
96
48

Iterations∗
410k
635k
500k
1.9M

Learning Rate
9.6e-5
8.4e-5
5.e-5
9.6e-5

LDM-1
LDM-2
LDM-4
LDM-8
LDM-16
LDM-32

z𝑧z-shape
256×256×32562563256\times 256\times 3
128×128×21281282128\times 128\times 2
64×64×36464364\times 64\times 3
32×32×43232432\times 32\times 4
16×16×81616816\times 16\times 8
88×8×328883288\times 8\times 32

|𝒵|𝒵|\mathcal{Z}|
-
2048
8192
16384
16384
16384

Diffusion steps
1000
1000
1000
1000
1000
1000

Noise Schedule
linear
linear
linear
linear
linear
linear

Model Size
396M
391M
391M
395M
395M
395M

Channels
192
192
192
256
256
256

Depth
2
2
2
2
2
2

Channel Multiplier
1,1,2,2,4,4
1,2,2,4,4
1,2,3,5
1,2,4
1,2,4
1,2,4

Number of Heads
1
1
1
1
1
1

Batch Size
7
9
40
64
112
112

Iterations
2M
2M
2M
2M
2M
2M

Learning Rate
4.9e-5
6.3e-5
8e-5
6.4e-5
4.5e-5
4.5e-5

Conditioning
CA
CA
CA
CA
CA
CA

CA-resolutions
32, 16, 8
32, 16, 8
32, 16, 8
32, 16, 8
16, 8, 4
8, 4, 2

Embedding Dimension
512
512
512
512
512
512

Transformers Depth
1
1
1
1
1
1

LDM-1
LDM-2
LDM-4
LDM-8
LDM-16
LDM-32

z𝑧z-shape
256×256×32562563256\times 256\times 3
128×128×21281282128\times 128\times 2
64×64×36464364\times 64\times 3
32×32×43232432\times 32\times 4
16×16×81616816\times 16\times 8
88×8×328883288\times 8\times 32

|𝒵|𝒵|\mathcal{Z}|
-
2048
8192
16384
16384
16384

Diffusion steps
1000
1000
1000
1000
1000
1000

Noise Schedule
linear
linear
linear
linear
linear
linear

Model Size
270M
265M
274M
258M
260M
258M

Channels
192
192
224
256
256
256

Depth
2
2
2
2
2
2

Channel Multiplier
1,1,2,2,4,4
1,2,2,4,4
1,2,3,4
1,2,4
1,2,4
1,2,4

Attention resolutions
32, 16, 8
32, 16, 8
32, 16, 8
32, 16, 8
16, 8, 4
8, 4, 2

Head Channels
32
32
32
32
32
32

Batch Size
9
11
48
96
128
128

Iterations∗
500k
500k
500k
500k
500k
500k

Learning Rate
9e-5
1.1e-4
9.6e-5
9.6e-5
1.3e-4
1.3e-4

Task
Text-to-Image
Layout-to-Image
Class-Label-to-Image
Super Resolution
Inpainting
Semantic-Map-to-Image

Dataset
LAION
OpenImages
COCO
ImageNet
ImageNet
Places
Landscapes

f𝑓f
8
4
8
4
4
4
8

z𝑧z-shape
32×32×43232432\times 32\times 4
64×64×36464364\times 64\times 3
32×32×43232432\times 32\times 4
64×64×36464364\times 64\times 3
64×64×36464364\times 64\times 3
64×64×36464364\times 64\times 3
32×32×43232432\times 32\times 4

|𝒵|𝒵|\mathcal{Z}|
-
8192
16384
8192
8192
8192
16384

Diffusion steps
1000
1000
1000
1000
1000
1000
1000

Noise Schedule
linear
linear
linear
linear
linear
linear
linear

Model Size
1.45B
306M
345M
395M
169M
215M
215M

Channels
320
128
192
192
160
128
128

Depth
2
2
2
2
2
2
2

Channel Multiplier
1,2,4,4
1,2,3,4
1,2,4
1,2,3,5
1,2,2,4
1,4,8
1,4,8

Number of Heads
8
1
1
1
1
1
1

Dropout
-
-
0.1
-
-
-
-

Batch Size
680
24
48
1200
64
128
48

Iterations
390K
4.4M
170K
178K
860K
360K
360K

Learning Rate
1.0e-4
4.8e-5
4.8e-5
1.0e-4
6.4e-5
1.0e-6
4.8e-5

Conditioning
CA
CA
CA
CA
concat
concat
concat

(C)A-resolutions
32, 16, 8
32, 16, 8
32, 16, 8
32, 16, 8
-
-
-

Embedding Dimension
1280
512
512
512
-
-
-

Transformer Depth
1
3
2
1
-
-
-

### E.2 Implementation Details

#### E.2.1 Implementations of τθsubscript𝜏𝜃\tau_{\theta} for conditional LDMs

For the experiments on text-to-image and layout-to-image (Sec. 4.3.1) synthesis,
we implement the conditioner τθsubscript𝜏𝜃\tau_{\theta} as an unmasked transformer which processes
a tokenized version of the input y𝑦y and produces an output ζ:=τθ​(y)assign𝜁subscript𝜏𝜃𝑦\zeta:=\tau_{\theta}(y),
where ζ∈ℝM×dτ𝜁superscriptℝ𝑀subscript𝑑𝜏\zeta\in\mathbb{R}^{M\times d_{\tau}}.
More specifically, the transformer is implemented from N𝑁N transformer blocks consisting of
global self-attention layers, layer-normalization and
position-wise MLPs as follows222adapted from https://github.com/lucidrains/x-transformers:

ζ←TokEmb​(y)+PosEmb(y)←𝜁TokEmb𝑦PosEmb(y)\displaystyle\zeta\leftarrow\text{TokEmb}(y)+\text{PosEmb(y)}

(18)

for ​i=1,…,N::for 𝑖1…𝑁absent\displaystyle\text{for }i=1,\dots,N:

ζ1←LayerNorm​(ζ)←subscript𝜁1LayerNorm𝜁\displaystyle\quad\zeta_{1}\leftarrow\text{LayerNorm}(\zeta)

(19)

ζ2←MultiHeadSelfAttention​(ζ1)+ζ←subscript𝜁2MultiHeadSelfAttentionsubscript𝜁1𝜁\displaystyle\quad\zeta_{2}\leftarrow\text{MultiHeadSelfAttention}(\zeta_{1})+\zeta

(20)

ζ3←LayerNorm​(ζ2)←subscript𝜁3LayerNormsubscript𝜁2\displaystyle\quad\zeta_{3}\leftarrow\text{LayerNorm}(\zeta_{2})

(21)

ζ←MLP​(ζ3)+ζ2←𝜁MLPsubscript𝜁3subscript𝜁2\displaystyle\quad\zeta\leftarrow\text{MLP}(\zeta_{3})+\zeta_{2}

(22)

ζ←LayerNorm​(ζ)←𝜁LayerNorm𝜁\displaystyle\zeta\leftarrow\text{LayerNorm}(\zeta)

(23)

With ζ𝜁\zeta available, the conditioning is mapped into the UNet via the cross-attention mechanism
as depicted in Fig. 3. We modify the “ablated UNet” [15] architecture
and replace the self-attention layer with a shallow (unmasked) transformer consisting
of T𝑇T blocks with alternating layers of (i) self-attention, (ii) a position-wise MLP and
(iii) a cross-attention layer; see Tab. 16. Note that without (ii) and (iii), this
architecture is equivalent to the “ablated UNet”.

While it would be possible to increase the representational power of τθsubscript𝜏𝜃\tau_{\theta}
by additionally conditioning on the time step t𝑡t, we do not pursue this choice as it reduces the speed of inference.
We leave a more detailed analysis of this modification to future work.

For the text-to-image model, we rely on a publicly available333https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast tokenizer
[99].
The layout-to-image model discretizes the spatial locations of the bounding boxes
and encodes each box as a (l,b,c)𝑙𝑏𝑐(l,b,c)-tuple, where l𝑙l denotes the (discrete) top-left and b𝑏b the
bottom-right position. Class information is contained in c𝑐c. 
See Tab. 17 for the hyperparameters of τθsubscript𝜏𝜃\tau_{\theta} and Tab. 13 for those of the UNet for both of the above tasks.

Note that the class-conditional model as described in Sec. 4.1 is also implemented via cross-attention, where τθsubscript𝜏𝜃\tau_{\theta} is a single learnable
embedding layer with a dimensionality of 512, mapping classes y𝑦y to
ζ∈ℝ1×512𝜁superscriptℝ1512\zeta\in\mathbb{R}^{1\times 512}.

input
ℝh×w×csuperscriptℝℎ𝑤𝑐\mathbb{R}^{h\times w\times c}

LayerNorm
ℝh×w×csuperscriptℝℎ𝑤𝑐\mathbb{R}^{h\times w\times c}

Conv1x1
ℝh×w×d⋅nhsuperscriptℝ⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\times w\times d\cdot n_{h}}

Reshape
ℝh⋅w×d⋅nhsuperscriptℝ⋅⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\cdot w\times d\cdot n_{h}}

×T​{SelfAttentionMLPCrossAttentionabsent𝑇casesSelfAttentionotherwiseMLPotherwiseCrossAttentionotherwise\times T\begin{cases*}\text{SelfAttention}\\
\text{MLP}\\
\text{CrossAttention}\end{cases*}
ℝh⋅w×d⋅nhsuperscriptℝ⋅⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\cdot w\times d\cdot n_{h}}

ℝh⋅w×d⋅nhsuperscriptℝ⋅⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\cdot w\times d\cdot n_{h}}

ℝh⋅w×d⋅nhsuperscriptℝ⋅⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\cdot w\times d\cdot n_{h}}

Reshape
ℝh×w×d⋅nhsuperscriptℝ⋅ℎ𝑤𝑑subscript𝑛ℎ\mathbb{R}^{h\times w\times d\cdot n_{h}}

Conv1x1
ℝh×w×csuperscriptℝℎ𝑤𝑐\mathbb{R}^{h\times w\times c}

Text-to-Image
Layout-to-Image

seq-length
77
92

depth N𝑁N
32
16

dim
1280
512

#### E.2.2 Inpainting

input
GT

LaMa[88]

LDM #1

LDM #2

LDM #3

input
result
input
result

For our experiments on image-inpainting in Sec. 4.5, we
used the code of [88] to generate synthetic masks. We use a fixed set of
2k validation and 30k testing samples from Places[108]. During
training, we use random crops of size 256×256256256256\times 256 and evaluate on crops of
size 512×512512512512\times 512. This follows the training and testing protocol in
[88] and reproduces their reported metrics (see † in
Tab. 7). We include additional qualitative results of
LDM-4, w/ attn in
Fig. 21 and of LDM-4, w/o attn, big, w/ ft in
Fig. 22.

### E.3 Evaluation Details

This section provides additional details on evaluation for the experiments shown in Sec. 4.

#### E.3.1 Quantitative Results in Unconditional and Class-Conditional Image Synthesis

We follow common practice and estimate the statistics for calculating the FID-, Precision- and Recall-scores [29, 50] shown in Tab. 1 and 10 based on 50k samples from our models and the entire training set of each of the shown datasets. For calculating FID scores we use the torch-fidelity package [60]. However, since different data processing pipelines might lead to different results [64], we also evaluate our models with the script provided by Dhariwal and Nichol [15]. We find that results mainly coincide, except for the ImageNet and LSUN-Bedrooms datasets, where we notice slightly varying scores of 7.76 (torch-fidelity) vs. 7.77 (Nichol and Dhariwal) and 2.95 vs 3.0. For the future we emphasize the importance of a unified procedure for sample quality assessment. Precision and Recall are also computed by using the script provided by Nichol and Dhariwal.

#### E.3.2 Text-to-Image Synthesis

Following the evaluation protocol of [66] we compute FID and Inception Score for the Text-to-Image models from Tab. 2 by comparing generated samples with 30000 samples from the validation set of the MS-COCO dataset [51]. FID and Inception Scores are computed with torch-fidelity.

#### E.3.3 Layout-to-Image Synthesis

For assessing the sample quality of our Layout-to-Image models from Tab. 9 on the COCO dataset, we follow common practice [89, 37, 87] and compute FID scores the 2048 unaugmented examples of the COCO Segmentation Challenge split. To obtain better comparability, we use the exact same samples as in [37]. For the OpenImages dataset we similarly follow their protocol and use 2048 center-cropped test images from the validation set.

#### E.3.4 Super Resolution

We evaluate the super-resolution models on ImageNet following the pipeline suggested in [72], i.e. images
with a shorter size less than 256256256 px are removed (both for training and evaluation). On ImageNet, the low-resolution images are
produced using bicubic interpolation with anti-aliasing.
FIDs are evaluated using torch-fidelity [60],
and we produce samples on the validation split.
For FID scores, we additionally compare to reference features computed on the train split, see Tab. 5 and Tab. 11.

#### E.3.5 Efficiency Analysis

For efficiency reasons we compute the sample quality metrics plotted in Fig. 6, 17 and 7 based on 5k samples. Therefore, the results might vary from those shown in Tab. 1 and 10. All models have a comparable number of parameters as provided in Tab. 13 and 14. We maximize the learning rates of the individual models such that they still train stably. Therefore, the learning rates slightly vary between different runs cf. Tab. 13 and 14.

#### E.3.6 User Study

For the results of the user study presented in Tab. 4 we followed the protocoll of [72] and and use the 2-alternative force-choice paradigm to assess human preference scores for two distinct tasks. In Task-1 subjects were shown a low resolution/masked image between the corresponding ground truth high resolution/unmasked version and a synthesized image, which was generated by using the middle image as conditioning. For SuperResolution subjects were asked: ’Which of the two images is a better high quality version of the low resolution image in the middle?’. For Inpainting we asked ’Which of the two images contains more realistic inpainted regions of the image in the middle?’. In Task-2, humans were similarly shown the low-res/masked version and asked for preference between two corresponding images generated by the two competing methods. As in [72] humans viewed the images for 3 seconds before responding.

## Appendix F Computational Requirements

Method
Generator
Classifier
Overall
Inference
Nparamssubscript𝑁paramsN_{\text{params}}
FID↓↓\downarrow
IS↑↑\uparrow
Precision↑↑\uparrow
Recall↑↑\uparrow

Compute
Compute
Compute
Throughput∗

LSUN Churches 2562superscript2562256^{2}

StyleGAN2 [42]†
64
-
64
-
59M
3.86
-
-
-

LDM-8 (ours, 100 steps, 410K)
18
-
18
6.80
256M
4.02
-
0.64
0.52

LSUN Bedrooms 2562superscript2562256^{2}

ADM [15]† (1000 steps)
232
-
232
0.03
552M
1.9
-
0.66
0.51

LDM-4 (ours, 200 steps, 1.9M)
60
-
55
1.07
274M
2.95
-
0.66
0.48

CelebA-HQ 2562superscript2562256^{2}

LDM-4 (ours, 500 steps, 410K)
14.4
-
14.4
0.43
274M
5.11
-
0.72
0.49

FFHQ 2562superscript2562256^{2}

StyleGAN2 [42]
32.13‡
-
32.13†
-
59M
3.8
-
-
-

LDM-4 (ours, 200 steps, 635K)
26
-
26
1.07
274M
4.98
-
0.73
0.50

ImageNet 2562superscript2562256^{2}

VQGAN-f-4 (ours, first stage)
29
-
29
-
55M
0.58††
-
-
-

VQGAN-f-8 (ours, first stage)
66
-
66
-
68M
1.14††
-
-
-

BigGAN-deep [3]†
128-256

128-256
-
340M
6.95
203.6±2.6plus-or-minus2.6\pm\text{2.6}
0.87
0.28

ADM [15] (250 steps) †
916
-
916
0.12
554M
10.94
100.98
0.69
0.63

ADM-G [15] (25 steps) †
916
46
962
0.7
608M
5.58
-
0.81
0.49

ADM-G [15] (250 steps)†
916
46
962
0.07
608M
4.59
186.7
0.82
0.52

ADM-G,ADM-U [15] (250 steps)†
329
30
349
n/a
n/a
3.85
221.72
0.84
0.53

LDM-8-G (ours, 100, 2.9M)
79
12
91
1.93
506M
8.11
190.4±2.6plus-or-minus2.6\pm\text{2.6}
0.83
0.36

LDM-8 (ours, 200 ddim steps 2.9M, batch size 64)
79
-
79
1.9
395M
17.41
72.92
0.65
0.62

LDM-4 (ours, 250 ddim steps 178K, batch size 1200)
271
-
271
0.7
400M
10.56
103.49±1.24plus-or-minus1.24\pm\text{1.24}
0.71
0.62

LDM-4-G (ours, 250 ddim steps 178K, batch size 1200, classifier-free guidance [32] scale 1.25)
271
-
271
0.4
400M
3.95
178.22±2.43plus-or-minus2.43\pm\text{2.43}
0.81
0.55

LDM-4-G (ours, 250 ddim steps 178K, batch size 1200, classifier-free guidance [32] scale 1.5)
271
-
271
0.4
400M
3.60
247.67±5.59plus-or-minus5.59\pm\text{5.59}
0.87
0.48

In Tab 18 we provide a more detailed analysis on our used compute ressources and compare our best performing models on the CelebA-HQ, FFHQ, LSUN and ImageNet datasets with the recent state of the art models by using their provided numbers, cf. [15]. As they report their used compute in V100 days and we train all our models on a single NVIDIA A100 GPU, we convert the A100 days to V100 days by assuming a ×2.2absent2.2\times 2.2 speedup of A100 vs V100 [74]444This factor corresponds to the speedup of the A100 over the V100 for a U-Net, as defined in Fig. 1 in [74]. To assess sample quality, we additionally report FID scores on the reported datasets. We closely reach the performance of state of the art methods as StyleGAN2 [42] and ADM [15] while significantly reducing the required compute resources.

## Appendix G Details on Autoencoder Models

We train all our autoencoder models in an adversarial manner following [23], such that a patch-based discriminator Dψsubscript𝐷𝜓D_{\psi} is optimized to differentiate original images from reconstructions 𝒟​(ℰ​(x))𝒟ℰ𝑥\mathcal{D}(\mathcal{E}(x)).
To avoid arbitrarily scaled latent spaces, we regularize the latent z𝑧z to be zero centered and obtain small variance by introducing an regularizing loss term Lr​e​gsubscript𝐿𝑟𝑒𝑔L_{reg}. 
We investigate two different regularization methods: (i) a low-weighted Kullback-Leibler-term between qℰ​(z|x)=𝒩​(z;ℰμ,ℰσ2)subscript𝑞ℰconditional𝑧𝑥𝒩𝑧subscriptℰ𝜇subscriptℰsuperscript𝜎2q_{\mathcal{E}}(z|x)=\mathcal{N}(z;\mathcal{E}_{\mu},\mathcal{E}_{\sigma^{2}}) and a standard normal distribution
𝒩​(z;0,1)𝒩𝑧01\mathcal{N}(z;0,1) as in a standard variational autoencoder [46, 69], and, (ii) regularizing the latent space with a vector quantization layer by learning a codebook of |𝒵|𝒵|\mathcal{Z}| different exemplars [96]. 
To obtain high-fidelity reconstructions we only use a very small regularization for both scenarios, i.e. we either weight the 𝕂​𝕃𝕂𝕃\mathbb{KL} term
by a factor ∼10−6similar-toabsentsuperscript106\sim 10^{-6} or choose a high codebook dimensionality |𝒵|𝒵|\mathcal{Z}|.

The full objective to train the autoencoding model (ℰ,𝒟)ℰ𝒟(\mathcal{E},\mathcal{D}) reads:

LAutoencoder=minℰ,𝒟⁡maxψ⁡(Lr​e​c​(x,𝒟​(ℰ​(x)))−La​d​v​(𝒟​(ℰ​(x)))+log⁡Dψ​(x)+Lr​e​g​(x;ℰ,𝒟))subscript𝐿Autoencodersubscriptℰ𝒟subscript𝜓subscript𝐿𝑟𝑒𝑐𝑥𝒟ℰ𝑥subscript𝐿𝑎𝑑𝑣𝒟ℰ𝑥subscript𝐷𝜓𝑥subscript𝐿𝑟𝑒𝑔𝑥ℰ𝒟L_{\text{Autoencoder}}=\min_{\mathcal{E},\mathcal{D}}\max_{\psi}\Big{(}L_{rec}(x,\mathcal{D}(\mathcal{E}(x)))-L_{adv}(\mathcal{D}(\mathcal{E}(x)))+\log D_{\psi}(x)+L_{reg}(x;\mathcal{E},\mathcal{D})\Big{)}

(25)

##### DM Training in Latent Space

Note that for training diffusion models on the learned latent space, we again distinguish two cases
when learning p​(z)𝑝𝑧p(z) or p​(z|y)𝑝conditional𝑧𝑦p(z|y) (Sec. 4.3):
(i) For a KL-regularized latent space, we sample z=ℰμ(x)+ℰσ(x)⋅ε=:ℰ(x)z=\mathcal{E}_{\mu}(x)+\mathcal{E}_{\sigma}(x)\cdot\varepsilon=:\mathcal{E}(x), where
ε∼𝒩​(0,1)similar-to𝜀𝒩01\varepsilon\sim\mathcal{N}(0,1).
When rescaling the latent, we estimate the component-wise variance

σ^2=1b​c​h​w​∑b,c,h,w(zb,c,h,w−μ^)2superscript^𝜎21𝑏𝑐ℎ𝑤subscript𝑏𝑐ℎ𝑤superscriptsuperscript𝑧𝑏𝑐ℎ𝑤^𝜇2\hat{\sigma}^{2}=\frac{1}{bchw}\sum_{b,c,h,w}(z^{b,c,h,w}-\hat{\mu})^{2}

from the first batch in the data, where μ^=1b​c​h​w​∑b,c,h,wzb,c,h,w^𝜇1𝑏𝑐ℎ𝑤subscript𝑏𝑐ℎ𝑤superscript𝑧𝑏𝑐ℎ𝑤\hat{\mu}=\frac{1}{bchw}\sum_{b,c,h,w}z^{b,c,h,w}.
The output of ℰℰ\mathcal{E} is scaled such that the rescaled latent has unit standard deviation, i.e. z←zσ^=ℰ​(x)σ^←𝑧𝑧^𝜎ℰ𝑥^𝜎z\leftarrow\frac{z}{\hat{\sigma}}=\frac{\mathcal{E}(x)}{\hat{\sigma}}.
(ii) For a VQ-regularized latent space, we extract z𝑧z before the quantization layer and absorb the quantization operation
into the decoder, i.e. it can be interpreted as the first layer of 𝒟𝒟\mathcal{D}.

## Appendix H Additional Qualitative Results

Finally, we provide additional qualitative results for our landscapes model (Fig. 12, 23, 24 and 25), our class-conditional ImageNet model (Fig. 26 - 27) and our unconditional models for the CelebA-HQ, FFHQ and LSUN datasets (Fig. 28 - 31).
Similar as for the inpainting model in Sec. 4.5 we also fine-tuned the semantic landscapes model from Sec. 4.3.2 directly on 5122superscript5122512^{2} images and depict qualitative
results in Fig. 12 and Fig. 23.
For our those models trained on comparably small datasets, we additionally show nearest neighbors in VGG [79] feature space for samples from our models in Fig. 32 - 34.

Semantic Synthesis on Flickr-Landscapes [23] (5122superscript5122512^{2} finetuning)

Semantic Synthesis on Flickr-Landscapes [23]

Random class conditional samples on the ImageNet dataset

Random samples on the CelebA-HQ dataset

Random samples on the FFHQ dataset

Random samples on the LSUN-Churches dataset

Random samples on the LSUN-Bedrooms dataset

Nearest Neighbors on the CelebA-HQ dataset

Nearest Neighbors on the FFHQ dataset

Nearest Neighbors on the LSUN-Churches dataset

Generated on Sat Mar 2 03:33:41 2024 by LaTeXML
