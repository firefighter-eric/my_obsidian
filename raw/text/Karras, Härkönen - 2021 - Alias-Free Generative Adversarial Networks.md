# Karras, Härkönen - 2021 - Alias-Free Generative Adversarial Networks

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Karras, Härkönen - 2021 - Alias-Free Generative Adversarial Networks.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2106.12423
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Alias-Free Generative Adversarial Networks

Tero Karras
NVIDIA
&Miika Aittala
NVIDIA
&Samuli Laine
NVIDIA
Erik Härkönen
Aalto University and NVIDIA
&Janne Hellsten
NVIDIA
Jaakko Lehtinen
NVIDIA and Aalto University
&Timo Aila
NVIDIA

This work was done during an internship at NVIDIA.

###### Abstract

We observe that despite their hierarchical convolutional nature, the synthesis process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner.
This manifests itself as, e.g., detail appearing to be glued to image coordinates instead of the surfaces of depicted objects.
We trace the root cause to careless signal processing that causes aliasing in the generator network.
Interpreting all signals in the network as continuous, we derive generally applicable, small architectural changes that guarantee that unwanted information cannot leak into the hierarchical synthesis process.
The resulting networks match the FID of StyleGAN2 but differ dramatically in their internal representations, and they are fully equivariant to translation and rotation even at subpixel scales.
Our results pave the way for generative models better suited for video and animation.

## 1 Introduction

The resolution and quality of images produced by generative adversarial networks (GAN) [21] have seen rapid improvement recently [31, 11, 33, 34].
They have been used for a variety of applications, including
image editing [49, 55, 43, 22, 39, 3],
domain translation [70, 37, 61, 42],
and video generation [57, 15, 24].
While several ways of controlling the generative process have been found [8, 29, 10, 42, 25, 2, 7, 48, 6],
the foundations of the synthesis process remain only partially understood.

In the real world, details of different scale tend to transform hierarchically. For instance, moving a head causes the nose to move, which in turn moves the skin pores on it. The structure of a typical GAN generator is analogous: coarse, low-resolution features are hierarchically refined by upsampling layers, locally mixed by convolutions, and new detail is introduced through nonlinearities. We observe that despite this superficial similarity, current GAN architectures do not synthesize images in a natural hierarchical manner:
the coarse features mainly control the presence of finer features, but not their precise positions. Instead, much of the fine detail appears to be fixed in pixel coordinates.
This disturbing “texture sticking” is clearly visible in latent interpolations (see Figure 1 and our accompanying videos on the project page
https://nvlabs.github.io/stylegan3),
breaking the illusion of a solid and coherent object moving in space.
Our goal is an architecture that exhibits a more natural transformation hierarchy, where the exact sub-pixel position of each feature is exclusively inherited from the underlying coarse features.

It turns out that current networks can partially bypass the ideal hierarchical construction by drawing on unintentional positional references available to the intermediate layers through image borders [28, 35, 66], per-pixel noise inputs [33] and positional encodings, and aliasing [5, 69]. Aliasing, despite being a subtle and critical issue [44], has received little attention in the GAN literature. We identify two sources for it: 1) faint after-images of the pixel grid resulting from non-ideal upsampling filters111Consider nearest neighbor upsampling. If we upsample a 4×\times4 image to 8×\times8, the original pixels will be clearly visible, allowing one to reliably distinguish between even and odd pixels. Since the same is true on all scales, this (leaked) information makes it possible to reconstruct even the absolute pixel coordinates. With better filters such as bilinear or bicubic, the clues get less pronounced, but are nevertheless evident for the generator.
such as nearest, bilinear, or strided convolutions, and 2) the pointwise application of nonlinearities such as ReLU [60] or swish [47]. We find that the network has the means and motivation to amplify even the slightest amount of aliasing and combining it over multiple scales allows it to build a basis for texture motifs that are fixed in screen coordinates. This holds for all filters commonly used in deep learning [69, 59], and even high-quality filters used in image processing.

How, then, do we eliminate the unwanted side information and thereby stop the network from using it?
While borders can be solved by simply operating on slightly larger images, aliasing is much harder.
We begin by noting that aliasing is most naturally treated in the classical Shannon-Nyquist signal processing framework, and switch focus to bandlimited functions on a continuous domain that are merely represented by discrete sample grids.
Now, successful elimination of all sources of positional references means that details can be generated equally well regardless of pixel coordinates, which in turn is equivalent to enforcing continuous equivariance to sub-pixel translation (and optionally rotation) in all layers.
To achieve this, we describe a comprehensive overhaul of all signal processing aspects of the StyleGAN2 generator [34].
Our contributions include the surprising finding that current upsampling filters are simply not aggressive enough in suppressing aliasing, and that extremely high-quality filters with over 100dB attenuation are required.
Further, we present a principled solution to aliasing caused by pointwise nonlinearities [5] by considering their effect in the continuous domain and appropriately low-pass filtering the results.
We also show that after the overhaul, a model based on 1×\times1 convolutions yields a strong, rotation equivariant generator.

Bl

StyleGAN2 Ours 

StyleGAN2
←←\leftarrow latent interpolation →→\rightarrow
 

Ours
←←\leftarrow latent interpolation →→\rightarrow

Averaged Central

Once aliasing is adequately suppressed to force the model to implement more natural hierarchical refinement, its mode of operation changes drastically: the emergent internal representations now include coordinate systems that allow
details to be correctly attached to the underlying surfaces.
This promises significant improvements to models that generate video and animation. The new StyleGAN3 generator matches StyleGAN2 in terms of FID [26], while being slightly heavier computationally.
Our implementation and pre-trained models are available at https://github.com/NVlabs/stylegan3

Several recent works have studied the lack of translation equivariance
in CNNs, mainly in the context of classification [28, 35, 66, 5, 38, 69, 12, 71, 59]. We significantly expand upon the antialiasing measures in this literature and show that doing so induces a fundamentally altered image generation behavior.
Group-equivariant CNNs aim to generalize the efficiency benefits of translational weight sharing to, e.g., rotation [16, 65, 63, 62] and scale [64]. Our 1×\times1 convolutions can be seen an instance of a continuously E​(2)E2\textrm{E}(2)-equivariant model [62] that remains compatible with, e.g., channel-wise ReLU nonlinearities and modulation.
Dey et al. [17] apply 90∘superscript9090^{\circ} rotation-and-flip equivariant CNNs [16] to GANs and show improved data efficiency. Our work is complementary, and not motivated by efficiency.
Recent implicit network [53, 56, 13] based GANs [4, 54] generate each pixel independently via similar 1×\times1 convolutions. While equivariant, these models do not help with texture sticking, as they do not use an upsampling hierarchy or implement a shallow non-antialiased one.

## 2 Equivariance via continuous signal interpretation

To begin our analysis of equivariance in CNNs, we shall first rethink our view of what exactly is the signal that flows through a network.
Even though data may be stored as values in a pixel grid, we cannot naïvely hold these values to directly represent the signal.
Doing so would prevent us from considering operations as trivial as translating the contents of a feature map by half a pixel.

According to the Nyquist–Shannon sampling theorem [51], a regularly sampled signal can represent any continuous signal containing frequencies between zero and half of the sampling rate.
Let us consider a two-dimensional, discretely sampled feature map Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] that consists of a regular grid of Dirac impulses of varying magnitudes, spaced 1/s1𝑠1/s units apart where s𝑠s is the sampling rate.
This is analogous to an infinite two-dimensional grid of values.

Given Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] and s𝑠s, the Whittaker–Shannon interpolation formula [51] states that the corresponding continuous representation z​(𝒙)𝑧𝒙z(\bm{x}) is obtained by convolving the discretely sampled Dirac grid Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] with an ideal interpolation filter ϕssubscriptitalic-ϕ𝑠\phi_{s}, i.e.,
z​(𝒙)=(ϕs∗Z)​(𝒙)𝑧𝒙∗subscriptitalic-ϕ𝑠𝑍𝒙z(\bm{x})=\big{(}\phi_{s}\ast Z\big{)}(\bm{x}),
where ∗∗\ast denotes continuous convolution and ϕs​(𝒙)=sinc⁡(s​x0)⋅sinc⁡(s​x1)subscriptitalic-ϕ𝑠𝒙⋅sinc𝑠subscript𝑥0sinc𝑠subscript𝑥1\phi_{s}(\bm{x})=\operatorname{sinc}(sx_{0})\cdot\operatorname{sinc}(sx_{1}) using the signal processing convention of defining sinc⁡(x)=sin⁡(π​x)/(π​x)sinc𝑥𝜋𝑥𝜋𝑥\operatorname{sinc}(x)=\sin(\pi x)/(\pi x).
ϕssubscriptitalic-ϕ𝑠\phi_{s} has a bandlimit of s/2𝑠2s/2 along the horizontal and vertical dimensions, ensuring that the resulting continuous signal captures all frequencies that can be represented with sampling rate s𝑠s.

Conversion from the continuous to the discrete domain corresponds to sampling the continuous signal z​(𝒙)𝑧𝒙z(\bm{x}) at the sampling points of Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] that we define to be offset by half the sample spacing to lie at the “pixel centers”, see Figure 2, left.
This can be expressed as a pointwise multiplication with a two-dimensional Dirac comb (𝒙)s=∑X∈ℤ2δ(𝒙−(X+12)/s){}_{s\!\!\!\>\>}(\bm{x})=\sum_{X\!\in\mathbb{Z}^{2}}\delta\big{(}\bm{x}-(X+\frac{1}{2})/s\big{)}.

We earmark the unit square 𝒙∈[0,1]2𝒙superscript012\bm{x}\in[0,1]^{2} in z​(𝒙)𝑧𝒙z(\bm{x}) as our canvas for the signal of interest.
In Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] there are s2superscript𝑠2s^{2} discrete samples in this region, but the above convolution with ϕssubscriptitalic-ϕ𝑠\phi_{s} means that values of Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] outside the unit square also influence z​(𝒙)𝑧𝒙z(\bm{x}) inside it.
Thus storing an s×s𝑠𝑠s\times s -pixel feature map is not sufficient; in theory, we would need to store the entire infinite Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}].
As a practical solution, we store Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] as a two-dimensional array that covers a region slightly larger than the unit square (Section 3.2).

Having established correspondence between bandlimited, continuous feature maps z​(𝒙)𝑧𝒙z(\bm{x}) and discretely sampled feature maps Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}], we can shift our focus away from the usual pixel-centric view of the signal.
In the remainder of this paper, we shall interpret z​(𝒙)𝑧𝒙z(\bm{x}) as being the actual signal being operated on, and the discretely sampled feature map Z​[𝒙]𝑍delimited-[]𝒙Z[\bm{x}] as merely a convenient encoding for it.

#### Discrete and continuous representation of network layers

Practical neural networks operate on the discretely sampled feature maps.
Consider operation 𝐅𝐅\mathbf{F} (convolution, nonlinearity, etc.) operating on a discrete feature map: Z′=𝐅​(Z)superscript𝑍′𝐅𝑍Z^{\prime}=\mathbf{F}(Z).
The feature map has a corresponding continuous counterpart, so we also have a corresponding mapping in the continuous domain: z′=𝐟​(z)superscript𝑧′𝐟𝑧z^{\prime}=\mathbf{f}(z).
Now, an operation specified in one domain can be seen to perform a corresponding operation in the other domain:

𝐟​(z)𝐟𝑧\displaystyle\mathbf{f}(z)
=ϕs′∗𝐅(⊙sz),\displaystyle=\phi_{s^{\prime}}\ast\mathbf{F}({}_{s\!\!\!\>\>}\odot z),
𝐅​(Z)𝐅𝑍\displaystyle\mathbf{F}(Z)
=⊙s′𝐟(ϕs∗Z),\displaystyle={}_{s^{\prime}\!\!\>}\odot\mathbf{f}(\phi_{s}\ast Z),

(1)

where ⊙direct-product\odot denotes pointwise multiplication and s𝑠s and s′superscript𝑠′s^{\prime} are the input and output sampling rates.
Note that in the latter case 𝐟𝐟\mathbf{f} must not introduce frequency content beyond the output bandlimit s′/2superscript𝑠′2s^{\prime}/2.

### 2.1 Equivariant network layers

Operation 𝐟𝐟\mathbf{f} is equivariant with respect to a spatial transformation 𝐭𝐭\mathbf{t} of the 2D plane if it commutes with it in the continuous domain: 𝐭∘𝐟=𝐟∘𝐭𝐭𝐟𝐟𝐭\mathbf{t}\circ\mathbf{f}=\mathbf{f}\circ\mathbf{t}.
We note that when inputs are bandlimited to s/2𝑠2s/2, an equivariant operation must not generate frequency content above the output bandlimit of s′/2superscript𝑠′2s^{\prime}/2, as otherwise no faithful discrete output representation exists.

We focus on two types of equivariance in this paper: translation and rotation.
In the case of rotation the spectral constraint is somewhat stricter — rotating an image corresponds to rotating the spectrum, and in order to guarantee the bandlimit in both horizontal and vertical direction, the spectrum must be limited to a disc with radius s/2𝑠2s/2.
This applies to both the initial network input as well as the bandlimiting filters used for downsampling, as will be described later.

We now consider the primitive operations in a typical generator network: convolution, upsampling, downsampling, and nonlinearity.
Without loss of generality, we discuss the operations acting on a single feature map: pointwise linear combination of features has no effect on the analysis.

#### Convolution

Consider a standard convolution with a discrete kernel K𝐾K.
We can interpret K𝐾K as living in the same grid as the input feature map, with sampling rate s𝑠s.
The discrete-domain operation is simply 𝐅conv​(Z)=K∗Zsubscript𝐅conv𝑍∗𝐾𝑍\mathbf{F}_{\!\!\>\text{conv}}(Z)=K\ast Z, and we obtain the corresponding continuous operation from Eq. 1:

𝐟conv(z)=ϕs∗(K∗(⊙sz))=K∗(ϕs∗(⊙sz))=K∗z\mathbf{f}_{\!\>\text{conv}}(z)=\phi_{s}\ast\big{(}K\ast({}_{s\!\!\!\>\>}\odot z)\big{)}=K\ast\big{(}\phi_{s}\ast({}_{s\!\!\!\>\>}\odot z)\big{)}=K\ast z

(2)

due to commutativity of convolution and the fact that discretization followed by convolution with ideal low-pass filter, both with same sampling rate s𝑠s, is an identity operation, i.e., ϕs∗(⊙sz)=z\phi_{s}\ast({}_{s\!\!\!\>\>}\odot z)=z.
In other words, the convolution operates by continuously sliding the discretized kernel over the continuous representation of the feature map.
This convolution introduces no new frequencies, so the bandlimit requirements for both translation and rotation equivariance are trivially fulfilled.

Convolution also commutes with translation in the continuous domain, and thus the operation is equivariant to translation.
For rotation equivariance, the discrete kernel K𝐾K needs to be radially symmetric.
We later show in Section 3.2 that trivially symmetric 1×\times1 convolution kernels are, despite their simplicity, a viable choice for rotation equivariant generative networks.

#### Upsampling and downsampling

Ideal upsampling does not modify the continuous representation. Its only purpose is to increase the output sampling rate (s′>ssuperscript𝑠′𝑠s^{\prime}>s) to add headroom in the spectrum where subsequent layers may introduce additional content.
Translation and rotation equivariance follow directly from upsampling being an identity operation in the continuous domain.
With 𝐟up​(z)=zsubscript𝐟up𝑧𝑧\mathbf{f}_{\!\>\text{up}}(z)=z, the discrete operation according to Eq. 1 is 𝐅up(Z)=⊙s′(ϕs∗Z)\mathbf{F}_{\!\!\>\text{up}}(Z)={}_{s^{\prime}\!\!\>}\odot(\phi_{s}\ast Z).
If we choose s′=n​ssuperscript𝑠′𝑛𝑠s^{\prime}=ns with integer n𝑛n, this operation can be implemented by first interleaving Z𝑍Z with zeros to increase its sampling rate and then convolving it with a discretized filter ⊙s′ϕs{}_{s^{\prime}\!\!\>}\odot\phi_{s}.

In downsampling, we must low-pass filter z𝑧z to remove frequencies above the output bandlimit, so that the signal can be represented faithfully in the coarser discretization.
The operation in continuous domain is 𝐟down​(z)=ψs′∗zsubscript𝐟down𝑧∗subscript𝜓superscript𝑠′𝑧\mathbf{f}_{\!\>\text{down}}(z)=\psi_{s^{\prime}}\ast z, where an ideal low-pass filter ψs:=s2⋅ϕsassignsubscript𝜓𝑠⋅superscript𝑠2subscriptitalic-ϕ𝑠\psi_{s}:={s}^{2}\cdot\phi_{s} is simply the corresponding interpolation filter normalized to unit mass.
The discrete counterpart is 𝐅down(Z)=⊙s′(ψs′∗(ϕs∗Z))=1/s2⋅⊙s′(ψs′∗ψs∗Z)=(s′/s)2⋅⊙s′(ϕs′∗Z)\mathbf{F}_{\!\!\>\text{down}}(Z)={}_{s^{\prime}\!\!\>}\odot\big{(}\psi_{s^{\prime}}\ast(\phi_{s}\ast Z)\big{)}=1/s^{2}\cdot{}_{s^{\prime}\!\!\>}\odot(\psi_{s^{\prime}}\ast\psi_{s}\ast Z)=(s^{\prime}/s)^{2}\cdot{}_{s^{\prime}\!\!\>}\odot(\phi_{s^{\prime}}\ast Z).
The latter equality follows from ψs∗ψs′=ψmin​(s,s′)∗subscript𝜓𝑠subscript𝜓superscript𝑠′subscript𝜓min𝑠superscript𝑠′\psi_{s}\ast\psi_{s^{\prime}}=\psi_{\text{min}(s,s^{\prime})}. Similar to upsampling, downsampling by an integer fraction can be implemented with a discrete convolution followed by dropping sample points.
Translation equivariance follows automatically from the commutativity of 𝐟down​(z)subscript𝐟down𝑧\mathbf{f}_{\!\>\text{down}}(z) with translation, but for rotation equivariance we must replace ϕs′subscriptitalic-ϕsuperscript𝑠′\phi_{s^{\prime}} with a radially symmetric filter with disc-shaped frequency response.
The ideal such filter [9] is given by ϕs∘​(𝒙)=jinc⁡(s​∥𝒙∥)=2​J1​(π​s​∥𝒙∥)/(π​s​∥𝒙∥)superscriptsubscriptitalic-ϕ𝑠𝒙jinc𝑠delimited-∥∥𝒙2subscript𝐽1𝜋𝑠delimited-∥∥𝒙𝜋𝑠delimited-∥∥𝒙\phi_{s}^{\circ}(\bm{x})=\operatorname{jinc}(s\lVert\bm{x}\rVert)=2J_{1}(\pi s\lVert\bm{x}\rVert)/(\pi s\lVert\bm{x}\rVert), where J1subscript𝐽1J_{1} is the first order Bessel function of the first kind.

#### Nonlinearity

Applying a pointwise nonlinearity σ𝜎\sigma in the discrete domain does not commute with fractional translation or rotation.
However, in the continuous domain, any pointwise function commutes trivially with geometric transformations and is thus equivariant to translation and rotation.
Fulfilling the bandlimit constraint is another question — applying, e.g., ReLU in the continuous domain may introduce arbitrarily high frequencies that cannot be represented in the output.

A natural solution is to eliminate the offending high-frequency content by convolving the continuous result with the ideal low-pass filter ψssubscript𝜓𝑠\psi_{s}. Then, the continuous representation of the nonlinearity becomes 𝐟σ​(z)=ψs∗σ​(z)=s2⋅ϕs∗σ​(z)subscript𝐟𝜎𝑧∗subscript𝜓𝑠𝜎𝑧∗⋅superscript𝑠2subscriptitalic-ϕ𝑠𝜎𝑧\mathbf{f}_{\!\>\mathrm{\sigma}}(z)=\psi_{s}\ast\sigma(z)=s^{2}\cdot\phi_{s}\ast\sigma(z) and the discrete counterpart is 𝐅σ(Z)=s2⋅⊙s(ϕs∗σ(ϕs∗Z))\mathbf{F}_{\!\!\>\mathrm{\sigma}}(Z)=s^{2}\cdot{}_{s\!\!\!\>\>}\odot(\phi_{s}\ast\sigma(\phi_{s}\ast Z)) (see Figure 2, right).
This discrete operation cannot be realized without temporarily entering the continuous representation. We approximate this by upsampling the signal, applying the nonlinearity in the higher resolution, and downsampling it afterwards. Even though the nonlinearity is still performed in the discrete domain,
we have found that only a 2×\times temporary resolution increase is sufficient for high-quality equivariance.
For rotation equivariance, we must use the radially symmetric interpolation filter ϕs∘superscriptsubscriptitalic-ϕ𝑠\phi_{s}^{\circ} in the downsampling step, as discussed above.

Note that nonlinearity is the only operation capable of generating novel frequencies in our formulation, and that we can limit the range of these novel frequencies by applying a reconstruction filter with a lower cutoff than s/2𝑠2s/2 before the final discretization operation.
This gives us precise control over how much new information is introduced by each layer of a generator network (Section 3.2).

## 3 Practical application to generator network

We will now apply the theoretical ideas from the previous section in practice, by converting the well-established StyleGAN2 [34] generator to be fully equivariant to translation and rotation.
We will introduce the necessary changes step-by-step, evaluating their impact in Figure 3.
The discriminator remains unchanged in our experiments.

The StyleGAN2 generator consists of two parts.
First, a mapping network transforms an initial, normally distributed latent to an intermediate latent code 𝐰∼𝒲similar-to𝐰𝒲{\bf w}\sim\mathcal{W}.
Then, a synthesis network 𝐆𝐆\mathbf{G} starts from a learned 4×\times4×\times512 constant Z0subscript𝑍0Z_{0} and applies a sequence of N𝑁N layers — consisting of convolutions, nonlinearities, upsampling, and per-pixel noise — to produce an output image ZN=𝐆​(Z0;𝐰)subscript𝑍𝑁𝐆subscript𝑍0𝐰Z_{N}=\mathbf{G}(Z_{0};{\bf w}).
The intermediate latent code 𝐰𝐰{\bf w} controls the modulation of the convolution kernels in 𝐆𝐆\mathbf{G}.
The layers follow a rigid 2×\times upsampling schedule, where two layers are executed at each resolution and the number of feature maps is halved after each upsampling.
Additionally, StyleGAN2 employs skip connections, mixing regularization [33], and path length regularization.

Configuration

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

EQ-R ↑↑\uparrow

a
StyleGAN2

5.14

–

–

b
+ Fourier features

4.79

16.23

10.81

c
+ No noise inputs

4.54

15.81

10.84

d
+ Simplified generator

5.21

19.47

10.41

e
+ Boundaries & upsampling

6.02

24.62

10.97

f
+ Filtered nonlinearities

6.35

30.60

10.81

g
+ Non-critical sampling

4.78

43.90

10.84

h
+ Transformed Fourier features

4.64

45.20

10.61

t
+ Flexible layers (StyleGAN3-T)

4.62

63.01

13.12

r
+ Rotation equiv. (StyleGAN3-R)

4.50

66.65

40.48

Parameter

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

EQ-R ↑↑\uparrow

Time

Mem.

Filter size n=4𝑛4n=4

4.72

57.49

39.70

0.84×\times

0.99×\times

*

Filter size n=6𝑛6n=6

4.50

66.65

40.48

1.00×\times

1.00×\times

Filter size n=8𝑛8n=8

4.66

65.57

42.09

1.18×\times

1.01×\times

Upsampling m=1𝑚1m=1

4.38

39.96

36.42

0.65×\times

0.87×\times

*

Upsampling m=2𝑚2m=2

4.50

66.65

40.48

1.00×\times

1.00×\times

Upsampling m=4𝑚4m=4

4.57

74.21

40.97

2.31×\times

1.62×\times

Stopband ft,0=21.5subscript𝑓𝑡0superscript21.5f_{t,0}=2^{1.5}

4.62

51.10

29.14

0.86×\times

0.90×\times

*

Stopband ft,0=22.1subscript𝑓𝑡0superscript22.1f_{t,0}=2^{2.1}

4.50

66.65

40.48

1.00×\times

1.00×\times

Stopband ft,0=23.1subscript𝑓𝑡0superscript23.1f_{t,0}=2^{3.1}

4.68

73.13

41.63

1.36×\times

1.25×\times

Our goal is to make every layer of 𝐆𝐆\mathbf{G} equivariant w.r.t. the continuous signal, so that all finer details transform together with the coarser features of a local neighborhood.
If this succeeds, the entire network becomes similarly equivariant.
In other words, we aim to make the continuous operation 𝐠𝐠\mathbf{g} of the synthesis network equivariant w.r.t. transformations 𝐭𝐭\mathbf{t} (translations and rotations) applied on the continuous input z0subscript𝑧0z_{0}:
𝐠​(𝐭​[z0];𝐰)=𝐭​[𝐠​(z0;𝐰)]𝐠𝐭delimited-[]subscript𝑧0𝐰𝐭delimited-[]𝐠subscript𝑧0𝐰\mathbf{g}(\mathbf{t}[z_{0}];{\bf w})=\mathbf{t}[\mathbf{g}(z_{0};{\bf w})].
To evaluate the impact of various architectural changes and practical approximations, we need a way to measure how well the network implements the equivariances. For translation equivariance, we report the peak signal-to-noise ratio (PSNR) in decibels (dB) between two sets of images, obtained by translating the input and output of the synthesis network by a random amount, resembling the definition by Zhang [69]:

EQ-T=10⋅log10⁡(I𝑚𝑎𝑥2/𝔼𝐰∼𝒲,x∼𝒳2,p∼𝒱,c∼𝒞​[(𝐠​(𝐭x​[z0];𝐰)c​(p)−𝐭x​[𝐠​(z0;𝐰)]c​(p))2])EQ-T⋅10subscript10subscriptsuperscript𝐼2𝑚𝑎𝑥subscript𝔼formulae-sequencesimilar-to𝐰𝒲formulae-sequencesimilar-to𝑥superscript𝒳2formulae-sequencesimilar-to𝑝𝒱similar-to𝑐𝒞delimited-[]superscript𝐠subscriptsubscript𝐭𝑥delimited-[]subscript𝑧0𝐰𝑐𝑝subscript𝐭𝑥subscriptdelimited-[]𝐠subscript𝑧0𝐰𝑐𝑝2\text{EQ-T}{}=10\cdot\log_{10}\left(I^{2}_{\mathit{max}}\big{/}\mathbb{E}_{{\bf w}\sim\mathcal{W},x\sim\mathcal{X}^{2},p\sim\mathcal{V},c\sim\mathcal{C}}\left[\big{(}\mathbf{g}(\mathbf{t}_{x}[z_{0}];{\bf w})_{c}(p)-\mathbf{t}_{x}[\mathbf{g}(z_{0};{\bf w})]_{c}(p)\big{)}^{2}\right]\right)

(3)

Each pair of images, corresponding to a different random choice of 𝐰𝐰{\bf w}, is sampled at integer pixel locations p𝑝p within their mutually valid region 𝒱𝒱\mathcal{V}.
Color channels c𝑐c are processed independently, and the intended dynamic range of generated images −1​…+11…1-1\ldots{+}1 gives I𝑚𝑎𝑥=2subscript𝐼𝑚𝑎𝑥2I_{\mathit{max}}=2. Operator 𝐭xsubscript𝐭𝑥\mathbf{t}_{x} implements spatial translation with 2D offset x𝑥x, here drawn from distribution 𝒳2superscript𝒳2\mathcal{X}^{2} of integer offsets.
We define an analogous metric EQ-R for rotations, with the rotation angles drawn from 𝒰​(0∘,360∘)𝒰superscript0superscript360\mathcal{U}(0^{\circ},360^{\circ}).
Appendix E gives implementation details and our accompanying videos highlight the practical relevance of different dB values.

### 3.1 Fourier features and baseline simplifications (configs b–d)

To facilitate exact continuous translation and rotation of the input z0subscript𝑧0z_{0}, we replace the learned input constant in StyleGAN2 with Fourier features [56, 66], which also has the advantage of naturally defining a spatially infinite map.
We sample the frequencies uniformly within the circular frequency band fc=2subscript𝑓𝑐2f_{c}=2, matching the original 4×\times4 input resolution, and keep them fixed over the course of training.
This change (configs a and b in Figure 3, left) slightly improves FID and, crucially, allows us to compute the equivariance metrics without having to approximate the operator 𝐭𝐭\mathbf{t}.
This baseline architecture is far from being equivariant; our accompanying videos show that the output images deteriorate drastically when the input features are translated or rotated from their original position.

Next, we remove the per-pixel noise inputs because they are strongly at odds with our goal of a natural transformation hierarchy, i.e., that the exact sub-pixel position of each feature is exclusively inherited from the underlying coarse features.
While this change (config c) is approximately FID-neutral, it fails to improve the equivariance metrics when considered in isolation.

To further simplify the setup, we decrease the mapping network depth as recommended by Karras et al. [32] and disable mixing regularization and path length regularization [34].
Finally, we also eliminate the output skip connections.
We hypothesize that their benefit is mostly related to gradient magnitude dynamics during training and address the underlying issue more directly using a simple normalization
before each convolution. We track the exponential moving average σ2=𝔼​[x2]superscript𝜎2𝔼delimited-[]superscript𝑥2\sigma^{2}=\mathbb{E}[x^{2}] over all pixels and feature maps during training, and divide the feature maps by σ2superscript𝜎2\sqrt{\sigma^{2}}.
In practice, we bake the division into the convolution weights to improve efficiency.
These changes (config d) bring FID back to the level of original StyleGAN2, while leading to a slight improvement in translation equivariance.

(a) Filter design concepts (b) Our alias-free StyleGAN3 generator architecture (c) Flexible layers

### 3.2 Step-by-step redesign motivated by continuous interpretation

#### Boundaries and upsampling (config e)

Our theory assumes an infinite spatial extent for the feature maps, which we approximate by maintaining a fixed-size margin around the target canvas, cropping to this extended canvas after each layer.
This explicit extension is necessary as border padding is known to leak absolute image coordinates into the internal representations [28, 35, 66].
In practice, we have found a 101010-pixel margin to be enough; further increase has no noticeable effect on the results.

Motivated by our theoretical model, we replace the bilinear 2×\times upsampling filter with a better approximation of the ideal low-pass filter.
We use a windowed sincsinc\operatorname{sinc} filter with a relatively large Kaiser window [41] of size n=6𝑛6n=6, meaning that each output pixel is affected by 6 input pixels in upsampling and each input pixel affects 6 output pixels in downsampling.
Kaiser window is a particularly good choice for our purposes, because it offers explicit control over the transition band and attenuation (Figure 4a).
In the remainder of this section, we specify the transition band explicitly and compute the remaining parameters using Kaiser’s original formulas (Appendix C).
For now, we choose to employ critical sampling and set the filter cutoff fc=s/2subscript𝑓𝑐𝑠2f_{c}=s/2, i.e., exactly at the bandlimit, and transition band half-width fh=(2−1)​(s/2)subscript𝑓ℎ21𝑠2f_{h}=(\raisebox{0.0pt}[0.0pt][0.0pt]{$\sqrt{2}$}-1)(s/2).
Recall that sampling rate s𝑠s equals the width of the canvas in pixels, given our definitions in Section 2.

The improved handling of boundaries and upsampling (config e) leads to better translation equivariance.
However, FID is compromised by 16%, probably because we started to constrain what the feature maps can contain.
In a further ablation (Figure 3, right), smaller resampling filters (n=4𝑛4n=4) hurt translation equivariance, while larger filters (n=8𝑛8n=8) mainly increase training time.

#### Filtered nonlinearities (config f)

Our theoretical treatment of nonlinearities calls for wrapping each leaky ReLU (or any other commonly used non-linearity) between m×m\times upsampling and m×m\times downsampling, for some magnification factor m𝑚m.
We further note that the order of upsampling and convolution can be switched by virtue of the signal being bandlimited, allowing us to fuse the regular 2×\times upsampling and a subsequent m×m\times upsampling related to the nonlinearity into a single 2m×2m\times upsampling.
In practice, we find m=2𝑚2m=2 to be sufficient (Figure 3, right), again improving EQ-T (config f).
Implementing the upsample-LReLU-downsample sequence is not efficient using the primitives available in current deep learning frameworks [1, 45], and thus we implement a custom CUDA kernel (Appendix D) that combines these operations (Figure 4b), leading to 10×\times faster training and considerable memory savings.

#### Non-critical sampling (config g)

The critical sampling scheme — where filter cutoff is set exactly at the bandlimit — is ideal for many image processing applications as it strikes a good balance between antialiasing and the retention of high-frequency detail [58].
However, our goals are markedly different because aliasing is highly detrimental for the equivariance of the generator.
While high-frequency detail is important in the output image and thus in the highest-resolution layers, it is less important in the earlier ones given that their exact resolutions are somewhat arbitrary to begin with.

To suppress aliasing, we can simply lower the cutoff frequency to fc=s/2−fhsubscript𝑓𝑐𝑠2subscript𝑓ℎf_{c}=s/2-f_{h}, which ensures that all alias frequencies (above s/2𝑠2s/2) are in the stopband.222Here, fcsubscript𝑓𝑐f_{c} and fhsubscript𝑓ℎf_{h} correspond to the output (downsampling) filter of each layer. The input (upsampling) filters are based on the properties of the incoming signal, i.e., the output filter parameters of the previous layer.
For example, lowering the cutoff of the blue filter in Figure 4a would move its frequency response left so that the the worst-case attenuation of alias frequencies improves from 666 dB to 404040 dB.
This oversampling can be seen as a computational cost of better antialiasing, as we now use the same number of samples to express a slower-varying signal than before.
In practice, we choose to lower fcsubscript𝑓𝑐f_{c} on all layers except the highest-resolution ones, because in the end the generator must be able to produce crisp images to match the training data.
As the signals now contain less spatial information, we modify the heuristic used for determining the number of feature maps to be inversely proportional to fcsubscript𝑓𝑐f_{c} instead of the sampling rate s𝑠s.
These changes (config g) further improve translation equivariance and push FID below the original StyleGAN2.

#### Transformed Fourier features (config h)

Equivariant generator layers are well suited for modeling unaligned and arbitrarily oriented datasets, because any geometric transformation introduced to the intermediate features zisubscript𝑧𝑖z_{i} will directly carry over to the final image zNsubscript𝑧𝑁z_{N}.
Due to the limited capability of the layers themselves to introduce global transformations, however, the input features z0subscript𝑧0z_{0} play a crucial role in defining the global orientation of zNsubscript𝑧𝑁z_{N}.
To let the orientation vary on a per-image basis, the generator should have the ability to transform z0subscript𝑧0z_{0} based on 𝐰𝐰{\bf w}.
This motivates us to introduce a learned affine layer that outputs global translation and rotation parameters for the input Fourier features (Figure 4b and Appendix F).
The layer is initialized to perform an identity transformation, but learns to use the mechanism over time when beneficial; in config h this improves the FID slightly.

#### Flexible layer specifications (config t)

Our changes have improved the equivariance quality considerably, but some visible artifacts still remain as our accompanying videos demonstrate.
On closer inspection, it turns out that the attenuation of our filters (as defined for config g) is still insufficient for the lowest-resolution layers.
These layers tend to have rich frequency content near their bandlimit, which calls for extremely strong attenuation to completely eliminate aliasing.

So far, we have used the rigid sampling rate progression from StyleGAN2, coupled with simplistic choices for filter cutoff fcsubscript𝑓𝑐f_{c} and half-width fhsubscript𝑓ℎf_{h}, but this need not be the case; we are free to specialize these parameters on a per-layer basis.
In particular, we would like fhsubscript𝑓ℎf_{h} to be high in the lowest-resolution layers to maximize attenuation in the stopband, but low in the highest-resolution layers to allow matching high-frequency details of the training data.

Figure 4c illustrates an example progression of filter parameters in a 14-layer generator with two critically sampled full-resolution layers at the end.
The cutoff frequency grows geometrically from fc=2subscript𝑓𝑐2f_{c}=2 in the first layer to fc=sN/2subscript𝑓𝑐subscript𝑠𝑁2f_{c}=s_{N}/2 in the first critically sampled layer.
We choose the minimum acceptable stopband frequency to start at ft,0=22.1subscript𝑓𝑡0superscript22.1f_{t,0}=2^{2.1}, and it grows geometrically but slower than the cutoff frequency.
In our tests, the stopband target at the last layer is ft=fc⋅20.3subscript𝑓𝑡⋅subscript𝑓𝑐superscript20.3f_{t}=f_{c}\cdot 2^{0.3}, but the progression is halted at the first critically sampled layer.
Next, we set the sampling rate s𝑠s for each layer so that it accommodates frequencies up to ftsubscript𝑓𝑡f_{t}, rounding up to the next power of two without exceeding the output resolution.
Finally, to maximize the attenuation of aliasing frequencies, we set the transition band half-width to fh=max⁡(s/2,ft)−fcsubscript𝑓ℎ𝑠2subscript𝑓𝑡subscript𝑓𝑐f_{h}=\max(s/2,f_{t})-f_{c}, i.e., making it as wide as possible within the limits of the sampling rate, but at least wide enough to reach ftsubscript𝑓𝑡f_{t}.
The resulting improvement depends on how much slack is left between ftsubscript𝑓𝑡f_{t} and s/2𝑠2s/2; as an extreme example, the first layer stopband attenuation improves from 424242 dB to 480480480 dB using this scheme.

The new layer specifications again improve translation equivariance (config t), eliminating the remaining artifacts.
A further ablation (Figure 3, right) shows that ft,0subscript𝑓𝑡0f_{t,0} provides an effective way to trade training speed for equivariance quality.
Note that the number of layers is now a free parameter that does not directly depend on the output resolution.
In fact, we have found that a fixed choice of N𝑁N works consistently across multiple output resolutions and makes other hyperparameters such as learning rate behave more predictably.
We use N=14𝑁14N=14 in the remainder of this paper.

#### Rotation equivariance (config r)

We obtain a rotation equivariant version of the network with two changes.
First, we replace the 3×\times3 convolutions with 1×\times1 on all layers and compensate for the reduced capacity by doubling the number of feature maps.
Only the upsampling and downsampling operations spread information between pixels in this config.
Second, we replace the sincsinc\operatorname{sinc}-based downsampling filter with a radially symmetric jincjinc\operatorname{jinc}-based one that we construct using the same Kaiser scheme (Appendix C).
We do this for all layers except the two critically sampled ones, where it is important to match the potentially non-radial spectrum of the training data.
These changes (config r) improve EQ-R without harming FID, even though each layer has 56% fewer trainable parameters.

We also employ an additional stabilization trick in this configuration. Early on in the training, we blur all images the discriminator sees using a Gaussian filter. We start with σ=10𝜎10\sigma=10 pixels, which we ramp to zero over the first 200k images. This prevents the discriminator from focusing too heavily on high frequencies early on. Without this trick, config r is prone to early collapses because the generator sometimes learns to produce
high frequencies with a small delay, trivializing the discriminator’s task.

## 4 Results

Dataset
Config

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

EQ-R ↑↑\uparrow

FFHQ-U

70000 img, 10242

Train from scratch

StyleGAN2

03.79

15.89

10.79

StyleGAN3-T (ours)

03.67

61.69

13.95

StyleGAN3-R (ours)

03.66

64.78

47.64

FFHQ

70000 img, 10242

Train from scratch

StyleGAN2

02.70

13.58

10.22

StyleGAN3-T (ours)

02.79

61.21

13.82

StyleGAN3-R (ours)

03.07

64.76

46.62

MetFaces-U

1336 img, 10242

ADA, from FFHQ-U

StyleGAN2

18.98

18.77

13.19

StyleGAN3-T (ours)

18.75

64.11

16.63

StyleGAN3-R (ours)

18.75

66.34

48.57

MetFaces

1336 img, 10242

ADA, from FFHQ

StyleGAN2

15.22

16.39

12.89

StyleGAN3-T (ours)

15.11

65.23

16.82

StyleGAN3-R (ours)

15.33

64.86

46.81

AFHQv2

15803 img, 5122

ADA, from scratch

StyleGAN2

04.62

13.83

11.50

StyleGAN3-T (ours)

04.04

60.15

13.51

StyleGAN3-R (ours)

04.40

64.89

40.34

Beaches

20155 img, 5122

ADA, from scratch

StyleGAN2

05.03

15.73

12.69

StyleGAN3-T (ours)

04.32

59.33

15.88

StyleGAN3-R (ours)

04.57

63.66

37.42

Ablation
Translation eq.
+ Rotation eq.

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

EQ-R ↑↑\uparrow

*
Main configuration

4.62

63.01

4.50

66.65

40.48

With mixing reg.

4.60

63.48

4.67

63.59

40.90

With noise inputs

4.96

24.46

5.79

26.71

26.80

Without flexible layers

4.64

45.20

4.65

44.74

22.52

Fixed Fourier features

5.93

64.57

6.48

66.20

41.77

With path length reg.

5.00

68.36

5.98

71.64

42.18

0.5×\times capacity

7.43

63.14

6.52

63.08

39.89

1.0×\times capacity

4.62

63.01

4.50

66.65

40.48

2.0×\times capacity

3.80

66.61

4.18

70.06

42.51

*

Kaiser filter, n=6𝑛6n=6

4.62

63.01

4.50

66.65

40.48

Lanczos filter, a=2𝑎2a=2

4.69

51.93

4.44

57.70

25.25

Gaussian filter, σ=0.4𝜎0.4\sigma=0.4

5.91

56.89

5.73

59.53

39.43

G-CNN comparison

FID ↓↓\downarrow

EQ-T ↑↑\uparrow

EQ-R ↑↑\uparrow

Params

Time

*
StyleGAN3-T (ours)

4.62

63.01

13.12

23.3M

1.00×\times

+ p​4𝑝4p4 symmetry [16]

4.69

61.90

17.07

21.8M

2.48×\times

*
StyleGAN3-R (ours)

4.50

66.65

40.48

15.8M

1.37×\times

Figure 5 gives results for six datasets using StyleGAN2 [34] as well as our alias-free StyleGAN3-T and StyleGAN3-R generators.
In addition to the standard FFHQ [33] and Metfaces [32], we created unaligned versions of them.
We also created a properly resampled version of AFHQ [14] and collected a new Beaches dataset.
Appendix B describes the datasets in detail.
The results show that our FID remains competitive with StyleGAN2.
StyleGAN3-T and StyleGAN3-R perform equally well in terms of FID, and both show a very high level of translation equivariance.
As expected, only the latter provides rotation equivariance.
In FFHQ (1024×\times1024) the three generators had 30.0M, 22.3M and 15.8M parameters, while the training times were 1106, 1576 (+42%) and 2248 (+103%) GPU hours. Our accompanying videos show side-by-side comparisons with StyleGAN2, demonstrating visually that the texture sticking problem has been solved. The resulting motion is much more natural, better sustaining an illusion that there is a coherent 3D scene being imaged.

#### Ablations and comparisons

In Section 3.1 we disabled a number of StyleGAN2 features.
We can now turn them on one by one to gauge their effect on our generators (Figure 5, right).
While mixing regularization can be re-enabled without any ill effects, we also find that styles can be mixed quite reliably even without this explicit regularization (Appendix A).
Re-enabling noise inputs or relying on StyleGAN2’s original layer specifications compromises equivariances significantly, and using fixed Fourier features or re-enabling path length regularization harms FID.
Path length regularization is in principle at odds with translation equivariance, as it penalizes image changes upon latent space walk and thus encourages texture sticking.
We suspect that the counterintuitive improvement in equivariance may come from slightly blurrier generated images, at a cost of poor FID.

In a scaling test we tried changing the number of feature maps, observing that equivariances remain at a high level, but FID suffers considerably when the capacity is halved. Doubling the capacity improves result quality in terms of FID, at the cost of almost 4×\times training time.
Finally, we consider alternatives for our windowed Kaiser filter.
Lanczos is competitive in terms of FID, but as a separable filter it compromises rotation equivariance in particular.
Gaussian leads to clearly worse FIDs.

We compare StyleGAN3-R to an alternative where the rotation part is implemented using p​4𝑝4p4 symmetric G-CNN [16, 17] on top of our StyleGAN3-T. This approach provides only modest rotation equivariance while being slower to train.
Steerable filters [63] could theoretically provide competitive EQ-R, but the memory and training time requirements proved infeasible with generator networks of this size.

Appendix A demonstrates that the spectral properties of generated images closely match training data, comparing favorably to several earlier architectures.

#### Internal representations

Figure 6 visualizes typical internal representations from the networks.
While in StyleGAN2 all feature maps seem to encode signal magnitudes, in our networks some of the maps take a different role and encode phase information instead.
Clearly this is something that is needed when the network synthesizes detail on the surfaces; it needs to invent a coordinate system.
In StyleGAN3-R, the emergent positional encoding patterns appear to be somewhat more well-defined.
We believe that the existence of a coordinate system that allows precise localization on the surfaces of objects will prove useful in various applications, including advanced image and video editing.

## 5 Limitations, discussion, and future work

In this work we modified only the generator, but it seems likely that further benefits would be available by making the discriminator equivariant as well.
For example, in our FFHQ results the teeth do not move correctly when the head turns, and we suspect that this is caused by the discriminator accidentally preferring to see the front teeth at certain pixel locations.
Concurrent work has identified that aliasing is detrimental for such generalization [59].

Our alias-free generator architecture contains implicit assumptions about the nature of the training data, and violating these may cause training difficulties. Let us consider an example. Suppose we have black-and-white cartoons as training data that we (incorrectly) pre-process using point sampling [44], leading to training images where almost all pixels are either black or white and the edges are jagged. This kind of badly aliased training data is difficult for GANs in general, but it is especially at odds with equivariance: on the one hand, we are asking the generator to be able to translate the output smoothly by subpixel amounts, but on the other hand, edges must still remain jagged and pixels only black/white, to remain faithful to the training data.
The same issue can also arise with letterboxing of training images, low-quality JPEGs, or retro pixel graphics, where the jagged stair-step edges are a defining feature of the aesthetic. In such cases it may be beneficial for the generator to be aware of the pixel grid.

In future, it might be interesting to re-introduce noise inputs (stochastic variation) in a way that is consistent with hierarchical synthesis.
A better path length regularization would encourage neighboring features to move together, not discourage them from moving at all.
It might be beneficial to try to extend our approach to equivariance w.r.t. scaling, anisotropic scaling, or even arbitrary homeomorphisms.
Finally, it is well known that antialiasing should be done before tone mapping. So far, all GANs — including ours — have operated in the sRGB color space (after tone mapping).

Attention layers in the middle of a generator [68] could likely be dealt with similarly to non-linearities by temporarily switching to higher resolution – although the time complexity of attention layers may make this somewhat challenging in practice. Recent attention-based GANs that start with a tokenizing transformer (e.g., VQGAN [18]) may be at odds with equivariance. Whether it is possible to make them equivariant is an important open question.

#### Potential negative societal impacts

of (image-producing) GANs include many forms of disinformation, from fake portraits in social media [27] to propaganda videos of world leaders [50]. Our contribution eliminates certain characteristic artifacts from videos, potentially making them more convincing or deceiving, depending on the application.
Viable solutions include model watermarking [67] along with large-scale authenticity assessment in major social media sites.
This entire project consumed 92 GPU years and 225 MWh of electricity on an in-house cluster of NVIDIA V100s. The new StyleGAN3 generator is only marginally costlier to train or use than that of StyleGAN2.

## 6 Acknowledgments

We thank David Luebke, Ming-Yu Liu, Koki Nagano, Tuomas Kynkäänniemi, and Timo Viitanen for reviewing early drafts and helpful suggestions.
Frédo Durand for early discussions.
Tero Kuosmanen for maintaining our compute infrastructure.
AFHQ authors for an updated version of their dataset.
Getty Images for the training images in the Beaches dataset.
We did not receive external funding or additional revenues for this project.

## References

- [1]

M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin,
S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga,
S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden,
M. Wicke, Y. Yu, and X. Zheng.

TensorFlow: A system for large-scale machine learning.

In Proc. 12th USENIX Conference on Operating Systems Design and
Implementation, OSDI’16, pages 265–283, 2016.

- [2]

R. Abdal, P. Zhu, N. J. Mitra, and P. Wonka.

StyleFlow: Attribute-conditioned exploration of
StyleGAN-generated images using conditional continuous normalizing flows.

ACM Trans. Graph., 40(3), 2021.

- [3]

Y. Alaluf, O. Patashnik, and D. Cohen-Or.

Only a matter of style: Age transformation using a style-based
regression model.

CoRR, abs/2102.02754, 2021.

- [4]

I. Anokhin, K. Demochkin, T. Khakhulin, G. Sterkin, V. Lempitsky, and
D. Korzhenkov.

Image generators with conditionally-independent pixel synthesis.

In Proc. CVPR, 2021.

- [5]

A. Azulay and Y. Weiss.

Why do deep convolutional networks generalize so poorly to small
image transformations?

Journal of Machine Learning Research, 20(184):1–25, 2019.

- [6]

D. Bau, A. Andonian, A. Cui, Y. Park, A. Jahanian, A. Oliva, and A. Torralba.

Paint by word.

CoRR, abs/2103.10951, 2021.

- [7]

D. Bau, S. Liu, T. Wang, J.-Y. Zhu, and A. Torralba.

Rewriting a deep generative model.

In Proc. ECCV, 2020.

- [8]

D. Bau, J. Zhu, H. Strobelt, B. Zhou, J. B. Tenenbaum, W. T. Freeman, and
A. Torralba.

GAN dissection: Visualizing and understanding generative
adversarial networks.

In Proc. ICLR, 2019.

- [9]

R. E. Blahut.

Theory of remote image formation.

Cambridge University Press, 2004.

- [10]

T. Broad, F. F. Leymarie, and M. Grierson.

Network bending: Expressive manipulation of deep generative models.

In Proc. EvoMUSART, pages 20–36, 2021.

- [11]

A. Brock, J. Donahue, and K. Simonyan.

Large scale GAN training for high fidelity natural image synthesis.

In Proc. ICLR, 2019.

- [12]

A. Chaman and I. Dokmanić.

Truly shift-invariant convolutional neural networks.

In Proc. CVPR, 2021.

- [13]

Y. Chen, S. Liu, and X. Wang.

Learning continuous image representation with local implicit image
function.

In Proc. CVPR, 2021.

- [14]

Y. Choi, Y. Uh, J. Yoo, and J.-W. Ha.

StarGAN v2: Diverse image synthesis for multiple domains.

In Proc. CVPR, 2020.

- [15]

M. Chu, Y. Xie, J. Mayer, L. Leal-Taixé, and N. Thuerey.

Learning temporal coherence via self-supervision for GAN-based
video generation.

ACM Trans. Graph., 39(4), 2020.

- [16]

T. S. Cohen and M. Welling.

Group equivariant convolutional networks.

In Proc. ICML, 2016.

- [17]

N. Dey, A. Chen, and S. Ghafurian.

Group equivariant generative adversarial networks.

In Proc. ICLR, 2021.

- [18]

P. Esser, R. Rombach, and B. Ommer.

Taming transformers for high-resolution image synthesis.

In Proc. CVPR, 2021.

- [19]

R. Gal, D. Cohen, A. Bermano, and D. Cohen-Or.

SWAGAN: A style-based wavelet-driven generative model.

CoRR, abs/2102.06108, 2021.

- [20]

R. Ge, X. Feng, H. Pyla, K. Cameron, and W. Feng.

Power measurement tutorial for the Green500 list.

https://www.top500.org/green500/resources/tutorials/, Accessed March
1, 2020.

- [21]

I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair,
A. Courville, and Y. Bengio.

Generative adversarial networks.

In Proc. NIPS, 2014.

- [22]

J. Gu, Y. Shen, and B. Zhou.

Image processing using multi-code GAN prior.

In Proc. CVPR, 2020.

- [23]

Gwern.

Making anime faces with stylegan.

https://www.gwern.net/Faces#stylegan2-ext-modifications, Accessed
June 4, 2021.

- [24]

Z. Hao, A. Mallya, S. J. Belongie, and M. Liu.

GANcraft: Unsupervised 3D neural rendering of minecraft worlds.

CoRR, abs/2104.07659, 2021.

- [25]

E. Härkönen, A. Hertzmann, J. Lehtinen, and S. Paris.

GANSpace: Discovering interpretable GAN controls.

In Proc. NeurIPS, 2020.

- [26]

M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter.

GANs trained by a two time-scale update rule converge to a local
Nash equilibrium.

In Proc. NIPS, 2017.

- [27]

K. Hill and J. White.

Designed to deceive: Do these people look real to you?

The New York Times, 11 2020.

- [28]

M. A. Islam, S. Jia, and N. D. B. Bruce.

How much position information do convolutional neural networks
encode?

In Proc. ICLR, 2020.

- [29]

A. Jahanian, L. Chai, and P. Isola.

On the "steerability" of generative adversarial networks.

In Proc. ICLR, 2020.

- [30]

J. F. Kaiser.

Nonrecursive digital filter design using the I0-sinh window
function.

In Proc. 1974 IEEE International Symposium on Circuits &
Systems, pages 20–23, 1974.

- [31]

T. Karras, T. Aila, S. Laine, and J. Lehtinen.

Progressive growing of GANs for improved quality, stability, and
variation.

In Proc. ICLR, 2018.

- [32]

T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila.

Training generative adversarial networks with limited data.

In Proc. NeurIPS, 2020.

- [33]

T. Karras, S. Laine, and T. Aila.

A style-based generator architecture for generative adversarial
networks.

In Proc. CVPR, 2018.

- [34]

T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila.

Analyzing and improving the image quality of StyleGAN.

In Proc. CVPR, 2020.

- [35]

O. S. Kayhan and J. C. van Gemert.

On translation invariance in CNNs: Convolutional layers can exploit
absolute spatial location.

In Proc. CVPR, 2020.

- [36]

D. P. Kingma and J. Ba.

Adam: A method for stochastic optimization.

In Proc. ICLR, 2015.

- [37]

M. Liu, T. Breuel, and J. Kautz.

Unsupervised image-to-image translation networks.

In Proc. NIPS, 2017.

- [38]

M. Manfredi and Y. Wang.

Shift equivariance in object detection.

In Proc. ECCV 2020 Workshops, 2020.

- [39]

S. Menon, A. Damian, S. Hu, N. Ravi, and C. Rudin.

PULSE: Self-supervised photo upsampling via latent space
exploration of generative models.

In Proc. CVPR, 2020.

- [40]

L. Mescheder, A. Geiger, and S. Nowozin.

Which training methods for GANs do actually converge?

In Proc. ICML, 2018.

- [41]

A. V. Oppenheim and R. W. Schafer.

Discrete-Time Signal Processing.

Prentice Hall Press, USA, 3rd edition, 2009.

- [42]

T. Park, M. Liu, T. Wang, and J. Zhu.

Semantic image synthesis with spatially-adaptive normalization.

In Proc. CVPR, 2019.

- [43]

T. Park, J.-Y. Zhu, O. Wang, J. Lu, E. Shechtman, A. A. Efros, and R. Zhang.

Swapping autoencoder for deep image manipulation.

In Proc. NeurIPS, 2020.

- [44]

G. Parmar, R. Zhang, and J. Zhu.

On buggy resizing libraries and surprising subtleties in FID
calculation.

CoRR, abs/2104.11222, 2021.

- [45]

A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen,
Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito,
M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and
S. Chintala.

PyTorch: An imperative style, high-performance deep learning
library.

In Proc. NeurIPS, 2019.

- [46]

O. Patashnik, Z. Wu, E. Shechtman, D. Cohen-Or, and D. Lischinski.

StyleCLIP: Text-driven manipulation of StyleGAN imagery.

CoRR, abs/2103.17249, 2021.

- [47]

P. Ramachandran, B. Zoph, and Q. V. Le.

Swish: a self-gated activation function.

CoRR, abs/1710.05941, 2017.

- [48]

A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and
I. Sutskever.

Zero-shot text-to-image generation.

CoRR, abs/2102.12092, 2021.

- [49]

E. Richardson, Y. Alaluf, O. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro, and
D. Cohen-Or.

Encoding in style: A StyleGAN encoder for image-to-image
translation.

In Proc. CVPR, 2021.

- [50]

M. Seymour.

Canny AI: Imagine world leaders singing.

fxguide, 4 2019.

- [51]

C. E. Shannon.

Communication in the presence of noise.

Proc. Institute of Radio Engineers, 37(1):10–21, 1949.

- [52]

Y. Shen and B. Zhou.

Closed-form factorization of latent semantics in GANs.

In CVPR, 2021.

- [53]

V. Sitzmann, J. N. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein.

Implicit neural representations with periodic activation functions.

In Proc. NeurIPS, 2020.

- [54]

I. Skorokhodov, S. Ignatyev, and M. Elhoseiny.

Adversarial generation of continuous images.

In Proc. CVPR, 2021.

- [55]

R. Suzuki, M. Koyama, T. Miyato, T. Yonetsuji, and H. Zhu.

Spatially controllable image synthesis with internal representation
collaging.

CoRR, abs/1811.10153, 2019.

- [56]

M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan,
U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng.

Fourier features let networks learn high frequency functions in low
dimensional domains.

In Proc. NeurIPS, 2020.

- [57]

S. Tulyakov, M. Liu, X. Yang, and J. Kautz.

MoCoGAN: Decomposing motion and content for video generation.

In Proc. CVPR, 2018.

- [58]

K. Turkowski.

Filters for Common Resampling Tasks, pages 147–165.

Academic Press Professional, Inc., USA, 1990.

- [59]

C. Vasconcelos, H. Larochelle, V. Dumoulin, R. Romijnders, N. L. Roux, and
R. Goroshin.

Impact of aliasing on generalization in deep convolutional networks.

In ICCV, 2021.

- [60]

C. von der Malsburg.

Self-organization of orientation sensitive cells in striate cortex.

Biological Cybernetics, 14(2):85–100, 1973.

- [61]

T. Wang, M. Liu, J. Zhu, A. Tao, J. Kautz, and B. Catanzaro.

High-resolution image synthesis and semantic manipulation with
conditional GANs.

In Proc. CVPR, 2018.

- [62]

M. Weiler and G. Cesa.

General E​(2)E2\mathrm{E}(2)-equivariant steerable CNNs.

In Proc. NeurIPS, 2019.

- [63]

M. Weiler, F. A. Hamprecht, and M. Storath.

Learning steerable filters for rotation equivariant CNNs.

In Proc. CVPR, 2018.

- [64]

D. Worrall and M. Welling.

Deep scale-spaces: Equivariance over scale.

In Proc. NeurIPS, 2019.

- [65]

D. E. Worrall, S. J. Garbin, D. Turmukhambetov, and G. J. Brostow.

Harmonic networks: Deep translation and rotation equivariance.

In Proc. CVPR, 2017.

- [66]

R. Xu, X. Wang, K. Chen, B. Zhou, and C. C. Loy.

Positional encoding as spatial inductive bias in GANs.

In Proc. CVPR, 2021.

- [67]

N. Yu, V. Skripniuk, S. Abdelnabi, and M. Fritz.

Artificial fingerprinting for generative models: Rooting deepfake
attribution in training data.

CoRR, abs/2007.08457, 2021.

- [68]

H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena.

Self-attention generative adversarial networks.

In Proc. ICML, 2019.

- [69]

R. Zhang.

Making convolutional networks shift-invariant again.

In Proc. ICML, 2019.

- [70]

J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros.

Unpaired image-to-image translation using cycle-consistent
adversarial networks.

In Proc. ICCV, 2017.

- [71]

X. Zou, F. Xiao, Z. Yu, and Y. J. Lee.

Delving deeper into anti-aliasing in ConvNets.

In Proc. BMVC, 2020.

Appendices

## Appendix A Additional results

Real images from the training set StyleGAN2, FID 3.79
 
StyleGAN3-T (ours), FID 3.67 StyleGAN3-R (ours), FID 3.66

Real images from the training set StyleGAN2, FID 18.98
 
StyleGAN3-T (ours), FID 18.75 StyleGAN3-R (ours), FID 18.75

Real images from the training set StyleGAN2, FID 4.62
 
StyleGAN3-T (ours), FID 4.04 StyleGAN3-R (ours), FID 4.40

Real images from the training set StyleGAN2, FID 5.03
 
StyleGAN3-T (ours), FID 4.32 StyleGAN3-R (ours), FID 4.57

Uncurated sets of samples for StyleGAN2 (baseline config b with Fourier features) and our alias-free generators StyleGAN3-T and StyleGAN3-R are shown in Figures 7 (FFHQ-U), 8 (MetFaces-U), 9 (AFHQv2), and 10 (Beaches).
Truncation trick was not used when generating the images.

StyleGAN2 and our generators yield comparable FIDs in all of these datasets.
Visual inspection did not reveal anything surprising in the first three datasets, but in Beaches our new generators seem to generate a somewhat reduced set of possible scene layouts properly.
We suspect that this is related to the lack of noise inputs, which forces the generators to waste capacity for what is essentially random number generation [34]. Finding a way to reintroduce noise inputs without breaking equivariances is therefore an important avenue of future work.

The accompanying interpolation videos reveal major differences between StyleGAN2 and StyleGAN3-R.
For example, in MetFaces much of details such as brushstrokes or cracked paint seems to be glued to the pixel coordinates in StyleGAN2, whereas with StyleGAN3 all details move together with the depicted model.
The same is evident in AFHQv2 with the fur moving credibly in StyleGAN3 interpolations, while mostly sticking to the image coordinates in StyleGAN2.
In Beaches we furthermore observe that StyleGAN2 tends to “fade in” details while retaining a mostly fixed viewing position, while StyleGAN3 creates plenty of apparent rotations and movement.
The videos use hand-picked seeds to better showcase the relevant effects.

In a further test we created two example cinemagraphs that mimic small-scale head movement and facial animation in FFHQ.
The geometric head motion was generated as a random latent space walk along hand-picked directions from GANSpace [25] and SeFa [52].
The changes in expression were realized by applying the “global directions” method of StyleCLIP [46], using the prompts “angry face”, “laughing face”, “kissing face”, “sad face”, “singing face”, and “surprised face”.
The differences between StyleGAN2 and StyleGAN3 are again very prominent, with the former displaying jarring sticking of facial hair and skin texture, even under subtle movements.

The equivariance quality videos illustrate the practical relevance of the PSNR numbers in Figures 3 and 5 of the main paper.
We observe that for EQ-T numbers over ∼similar-to\sim50 dB indicate high-quality results, and for EQ-R ∼similar-to\sim40 dB look good.

We also provide an animated version of the nonlinearity visualization in Figure 2.

In style mixing [34] two or more independently chosen latent codes are fed into different layers of the generator.
Ideally all combinations would produce images that are not obviously broken, and furthermore, it would be desirable that specific layers end up controlling well-defined semantic aspects in the images.
StyleGAN uses mixing regularization [34] during training to achieve these goals.
We observe that mixing regularization continues to work similarly in StyleGAN3, but we also wanted to know whether it is truly necessary because the regularization is known to be detrimental for many complex and multi-modal datasets [23].
When we disable the regularization, obviously broken images remain rare, based on a visual inspection of a large number of images.
The semantically meaningful controls are somewhat compromised, however, as Figure 11 shows.

(a) FFHQ-U at 256×\times256 (b) FFHQ at 1024×\times1024 (c) MetFaces at 1024×\times1024

Figure 12 compares the convergence of our main configurations (config t and r) against the results of Karras et al. [34, 32].
The overall shape of the curves is similar; introducing translation and rotation equivariance in the generator does not appear to significantly alter the training dynamics.

(a) Training data (b) StyleGAN3-T (c) SWAGAN [19] (d) CIPS [4]

 
(e) StyleGAN3, 0∘ slice (f) StyleGAN3, 45∘ slice (g) Comparison methods, 45∘ slice

Following recent works that address signal processing issues in GANs [4, 19], we show average power spectra of the generated
and real images in Figure 13. The plots are computed from images that are whitened with the overall training dataset
mean and standard deviation. Because FFT interprets the signal as periodic, we eliminate the sharp step edge across the image borders by
windowing the pixel values prior to the transform. This eliminates the axis-aligned cross artifact which may obscure meaningful detail in
the spectrum. We display the average 2D spectrum as a contour plot, which makes the orientation-dependent falloff apparent, and highlights
detail like regularly spaced residuals of upsampling grids, and fixed noise patterns. We also plot 1D slices of the spectrum along the
horizontal and diagonal angle without azimuthal integration, so as to not average out the detail.
The code for reproducing these steps is included in the public release.

## Appendix B Datasets

In this section, we describe the new datasets and list the licenses of all datasets.

### B.1 FFHQ-U and MetFaces-U

We built unaligned variants of the existing FFHQ [33] and MetFaces [32] datasets.
The originals are available at https://github.com/NVlabs/ffhq-dataset and
https://github.com/NVlabs/metfaces-dataset, respectively.
The datasets were rebuilt with a modification of the original procedure based on the original code, raw uncropped images, and facial landmark metadata.
The code required to reproduce the modified datasets is included in the public release.

We use axis-aligned crop rectangles, and do not rotate them to match the orientation of the face. This retains the natural variation of camera and head tilt angles. Note that the images are still generally upright, i.e., never upside down or at 90∘superscript9090^{\circ} angle. The scale of the rectangle is determined as before.
For each image, the crop rectangle is randomly shifted from its original face-centered position, with the horizontal and vertical offset independently drawn from a normal distribution. The standard deviation is chosen as 20%percent2020\% of the crop rectangle dimension. If the crop rectangle falls partially outside the original image boundaries, we keep drawing new random offsets until we find one that does not. This removes the need to pad the images with fictional mirrored content, and we explicitly disabled this feature of the original build script.

Aside from the exact image content, the number of images and other specifications match the original dataset exactly.
While FFHQ-U contains identifiable images of persons, it does not introduce new images beyond those already in the original FFHQ.

### B.2 AFHQv2

We used an updated version of the AFHQ dataset [14] where the resampling filtering has been improved.
The original dataset suffers from pixel-level artifacts caused by inadequate downsampling filters [44].
This caused convergence problems with our models, as the sharp “stair-step” aliasing artifacts are difficult to reproduce without direct access to the pixel grid.

The dataset was rebuilt using the original uncropped images and crop rectangle metadata,
using the PIL library implementation of Lanczos resampling as recommended by Parmar et al. [44].
In a minority of cases, the crop rectangles were modified to remove non-isotropic scaling and other unnecessary transformations.
A small amount (∼2%similar-toabsentpercent2\sim 2\%) of images were dropped for technical reasons, leaving a total of 158031580315803 images.
Aside from this, the specifications of the dataset match the original. We use all images of all the three classes (cats, dogs, and wild animals) as one training dataset.

### B.3 Beaches

Beaches is a new dataset of 201552015520155 photographs of beaches at resolution 512×\times512.
The training images were provided by Getty Images.
Beaches is a proprietary dataset that we are licensed to use, but not to redistribute.
We are therefore unable to release the full training data or pre-trained models for this dataset.

### B.4 Licenses

The FFHQ dataset is available under Creative Commons BY-NC-SA 4.0 license by NVIDIA Corporation, and consist of images published by respective authors under Creative Commons BY 2.0, Creative Commons BY-NC 2.0, Public Domain Mark 1.0, Public Domain CC0 1.0, and U.S. Government Works license.

The MetFaces dataset is available under Creative Commons BY-NC 2.0 license by NVIDIA Corporation, and consists of images available under the Creative Commons Zero (CC0) license by the Metropolitan Museum of Art.

The original AFHQ dataset is available at https://github.com/clovaai/stargan-v2 under Creative Commons BY-NC 4.0 license by NAVER Corporation.

## Appendix C Filter details

In this section, we review basic FIR filter design methodology and detail the recipe used to construct the upsampling and downsampling filters in our generator.
We start with simple Kaiser filters in one dimension, discussing parameter selection and the necessary modifications needed for upsampling and downsampling.
We then proceed to extend the filters to two dimensions and conclude by detailing the alternative filters evaluated in Figure 5, right.
Our definitions are consistent with standard signal processing literature (e.g., Oppenheim [41]) as well as widely used software packages (e.g., scipy.signal.firwin).

### C.1 Kaiser low-pass filters

In one dimension, the ideal continuous-time low-pass filter with cutoff fcsubscript𝑓𝑐f_{c} is given by ψ​(x)=2​fc⋅sinc⁡(2​fc​x)𝜓𝑥⋅2subscript𝑓𝑐sinc2subscript𝑓𝑐𝑥\psi(x)=2f_{c}\cdot\operatorname{sinc}(2f_{c}x), where sinc⁡(x)=sin⁡(π​x)/(π​x)sinc𝑥𝜋𝑥𝜋𝑥\operatorname{sinc}(x)=\sin(\pi x)/(\pi x).
The ideal filter has infinite attenuation in the stopband, i.e., it completely eliminates all frequencies above fcsubscript𝑓𝑐f_{c}.
However, its impulse response is also infinite, which makes it impractical for three reasons: implementation efficiency, border artifacts, and ringing caused by long-distance interactions.
The most common way to overcome these issues is to limit the spatial extent of the filter using the window method [41]:

hK​(x)=2​fc⋅sinc⁡(2​fc​x)⋅wK​(x),subscriptℎ𝐾𝑥⋅2subscript𝑓𝑐sinc2subscript𝑓𝑐𝑥subscript𝑤𝐾𝑥h_{K}(x)=2f_{c}\cdot\operatorname{sinc}(2f_{c}x)\cdot w_{K}(x),

(4)

where wK​(x)subscript𝑤𝐾𝑥w_{K}(x) is a window function and hK​(x)subscriptℎ𝐾𝑥h_{K}(x) is the resulting practical approximation of ψ​(x)𝜓𝑥\psi(x).
Different window functions represent different tradeoffs between the frequency response and spatial extent; the smaller the spatial extent, the weaker the attenuation.
In this paper we use the Kaiser window [30], also known as the Kaiser–Bessel window, that provides explicit control over this tradeoff.
The Kaiser window is defined as

wK​(x)={I0​(β​1−(2​x/L)2)/I0​(β),if ​|x|≤L/2,0,if ​|x|>L/2,subscript𝑤𝐾𝑥casessubscript𝐼0𝛽1superscript2𝑥𝐿2subscript𝐼0𝛽if 𝑥𝐿20if 𝑥𝐿2w_{K}(x)=\begin{cases}I_{0}\Big{(}\beta\sqrt{1-(2x/L)^{2}}\Big{)}\big{/}I_{0}\big{(}\beta\big{)},&\textrm{if }|x|\leq L/2,\\
0,&\textrm{if }|x|>L/2,\end{cases}

(5)

where L𝐿L is the desired spatial extent, β𝛽\beta is a free parameter that controls the shape of the window, and I0subscript𝐼0I_{0} is the zeroth-order modified Bessel function of the first kind.
Note that the window has discontinuities at ±L/2plus-or-minus𝐿2\pm L/2; the value is strictly positive at x=L/2𝑥𝐿2x=L/2 but zero at x=L/2+ϵ𝑥𝐿2italic-ϵx=L/2+\epsilon.

When operating on discretely sampled signals, it is necessary to discretize the filter as well:

hK​[i]=hK​((i−(n−1)/2)/s)/s,for ​i∈{0,1,…,n−1},formulae-sequencesubscriptℎ𝐾delimited-[]𝑖subscriptℎ𝐾𝑖𝑛12𝑠𝑠for 𝑖01…𝑛1h_{K}[i]=h_{K}\Big{(}\big{(}i-(n-1)/2\big{)}/s\Big{)}\big{/}s,\hskip 5.69054pt\textrm{for }i\in\{0,1,\ldots,n-1\},

(6)

where hK​[i]subscriptℎ𝐾delimited-[]𝑖h_{K}[i] is the discretized version of hK​(x)subscriptℎ𝐾𝑥h_{K}(x) and s𝑠s is the sampling rate.
The filter is defined at n𝑛n discrete spatial locations, i.e., taps, located 1/s1𝑠1/s units apart and placed symmetrically around zero.
Given the values of n𝑛n and s𝑠s, the spatial extent can be expressed as L=(n−1)/s𝐿𝑛1𝑠L=(n-1)/s.
An odd value of n𝑛n results in a zero-phase filter that preserves the original sample locations, whereas an even value shifts the sample locations by 1/(2​s)12𝑠1/(2s) units.

The filters considered in this paper are approximately normalized by construction, i.e., ∫xhK​(x)≈∑ihK​[i]≈1subscript𝑥subscriptℎ𝐾𝑥subscript𝑖subscriptℎ𝐾delimited-[]𝑖1\int_{x}h_{K}(x)\approx\sum_{i}h_{K}[i]\approx 1.
Nevertheless, we have found it beneficial to explicitly normalize them after discretization.
In other words, we strictly enforce ∑ihK​[i]=1subscript𝑖subscriptℎ𝐾delimited-[]𝑖1\sum_{i}h_{K}[i]=1 by scaling the filter taps to reduce the risk of introducing cumulative scaling errors when the signal is passed through several consecutive layers.

### C.2 Selecting window parameters

Kaiser [30] provides convenient empirical formulas to connect the parameters of wKsubscript𝑤𝐾w_{K} to the properties of hKsubscriptℎ𝐾h_{K}.
Given the number of taps and the desired transition band width, the maximum attenuation achievable with hK​[i]subscriptℎ𝐾delimited-[]𝑖h_{K}[i] is approximated by

A=2.285⋅(n−1)⋅π⋅Δ​f+7.95,𝐴⋅2.285𝑛1𝜋Δ𝑓7.95A=2.285\cdot(n-1)\cdot\pi\cdot\Delta f+7.95,

(7)

where A𝐴A is the attenuation measured in decibels and Δ​fΔ𝑓\Delta f is the width of the transition band expressed as a fraction of s/2𝑠2s/2.
We choose to define the transition band using half-width fhsubscript𝑓ℎf_{h}, which gives Δ​f=(2​fh)/(s/2)Δ𝑓2subscript𝑓ℎ𝑠2\Delta f=(2f_{h})/(s/2).
Given the value of A𝐴A, the optimal choice for the shape parameter β𝛽\beta is then approximated [30] by

β={0.1102⋅(A−8.7),if ​A>50,0.5842⋅(A−21)0.4+0.07886⋅(A−21),if ​21≤A≤50,0,if ​A<21,𝛽cases⋅0.1102𝐴8.7if 𝐴50⋅0.5842superscript𝐴210.4⋅0.07886𝐴21if 21𝐴500if 𝐴21\beta=\begin{cases}0.1102\cdot(A-8.7),&\textrm{if }A>50,\\
0.5842\cdot(A-21)^{0.4}+0.07886\cdot(A-21),&\textrm{if }21\leq A\leq 50,\\
0,&\textrm{if }A<21,\end{cases}

(8)

This leaves us with two free parameters: n𝑛n controls the spatial extent while fhsubscript𝑓ℎf_{h} controls the transition band.
The choice of these parameters directly influences the resulting attenuation; increasing either parameter yields a higher value for A𝐴A.

### C.3 Upsampling and downsampling

When upsampling a signal, i.e., 𝐅up(Z)=⊙s′(ϕs∗Z)=1/s2⋅⊙s′(ψs∗Z)\mathbf{F}_{\!\!\>\text{up}}(Z)={}_{s^{\prime}\!\!\>}\odot(\phi_{s}\ast Z)=1/s^{2}\cdot{}_{s^{\prime}\!\!\>}\odot(\psi_{s}\ast Z), we are concerned not only the with input sampling rate s𝑠s, but also with the output sampling rate s′superscript𝑠′s^{\prime}.
With an integer upsampling factor m𝑚m, we can think of the upsampling operation as consisting of two steps: we first increase the sampling rate to s′=s⋅msuperscript𝑠′⋅𝑠𝑚s^{\prime}=s\cdot m by interleaving m−1𝑚1m-1 zeros between each input sample by and then low-pass filter the resulting signal to eliminate the alias frequencies above s/2𝑠2s/2.
In order to keep the signal magnitude unchanged, we must also scale the result by m𝑚m with one-dimensional signals, or by m2superscript𝑚2m^{2} with two-dimensional signals.
Since the filter now operates under s′superscript𝑠′s^{\prime} instead of s𝑠s, we must adjust its parameters accordingly:

n′superscript𝑛′\displaystyle n^{\prime}
=n⋅m,absent⋅𝑛𝑚\displaystyle=n\cdot m,
L′superscript𝐿′\displaystyle L^{\prime}
=(n′−1)/s′,absentsuperscript𝑛′1superscript𝑠′\displaystyle=(n^{\prime}-1)/s^{\prime},
Δ​f′Δsuperscript𝑓′\displaystyle\Delta f^{\prime}
=(2​fh)/(s′/2),absent2subscript𝑓ℎsuperscript𝑠′2\displaystyle=(2f_{h})/(s^{\prime}/2),

(9)

which gives us the final upsampling filter

hK′​[i]=hK′​((i−(n′−1)/2)/s′)/s′,for ​i∈{0,1,…,n′−1}.formulae-sequencesubscriptsuperscriptℎ′𝐾delimited-[]𝑖subscriptsuperscriptℎ′𝐾𝑖superscript𝑛′12superscript𝑠′superscript𝑠′for 𝑖01…superscript𝑛′1h^{\prime}_{K}[i]=h^{\prime}_{K}\Big{(}\big{(}i-(n^{\prime}-1)/2\big{)}/s^{\prime}\Big{)}\big{/}s^{\prime},\hskip 5.69054pt\textrm{for }i\in\{0,1,\ldots,n^{\prime}-1\}.

(10)

Multiplying the number of taps by m𝑚m keeps the spatial extent of the filter unchanged with respect to the input samples, and it also compensates for the reduced attenuation from Δ​f′<Δ​fΔsuperscript𝑓′Δ𝑓\Delta f^{\prime}<\Delta f.
Note that if the upsampling factor is even, n′superscript𝑛′n^{\prime} will be even as well, meaning that hK′subscriptsuperscriptℎ′𝐾h^{\prime}_{K} shifts the sample locations by 1/(2​s′)12superscript𝑠′1/(2s^{\prime}).
This is the desired behavior — if we consider sample i𝑖i to represent the continuous interval [i⋅s,(i+1)⋅s]⋅𝑖𝑠⋅𝑖1𝑠[i\cdot s,(i+1)\cdot s] in the input signal, the same interval will be represented by m𝑚m consecutive samples m⋅i,…,m⋅i+m−1⋅𝑚𝑖…⋅𝑚𝑖𝑚1m\cdot i,\ldots,m\cdot i+m-1 in the output signal.
Using a zero-phase upsampling filter, i.e., an odd value for n′superscript𝑛′n^{\prime}, would break this symmetry, leading to inconsistent behavior with respect to the boundaries.
Note that our symmetric interpretation is common in many computer graphics APIs, such as OpenGL, and it is also reflected in our definition of the Dirac comb in Section 2.

Upsampling and downsampling are adjoint operations with respect to each other, disregarding the scaling of the signal magnitude.
This means that the above definitions are readily applicable to downsampling as well; to downsample a signal by factor m𝑚m, we first filter it by hK′subscriptsuperscriptℎ′𝐾h^{\prime}_{K} and then discard the last m−1𝑚1m-1 samples within each group of m𝑚m consecutive samples.
The interpretation of all filter parameters, as well as the sample locations, is analogous to the upsampling case.

### C.4 Two-dimensional filters

Any one-dimensional filter, including hKsubscriptℎ𝐾h_{K}, can be trivially extended to two dimensions by defining the corresponding separable filter

hK+​(𝒙)=hK​(x0)⋅hK​(x1)=(2​fc)2⋅sinc⁡(2​fc​x0)⋅sinc⁡(2​fc​x1)⋅wK​(x0)⋅wK​(x1),superscriptsubscriptℎ𝐾𝒙⋅subscriptℎ𝐾subscript𝑥0subscriptℎ𝐾subscript𝑥1⋅⋅superscript2subscript𝑓𝑐2sinc2subscript𝑓𝑐subscript𝑥0sinc2subscript𝑓𝑐subscript𝑥1subscript𝑤𝐾subscript𝑥0subscript𝑤𝐾subscript𝑥1h_{K}^{+}(\bm{x})=h_{K}(x_{0})\cdot h_{K}(x_{1})=(2f_{c})^{2}\cdot\operatorname{sinc}(2f_{c}x_{0})\cdot\operatorname{sinc}(2f_{c}x_{1})\cdot w_{K}(x_{0})\cdot w_{K}(x_{1}),

(11)

where 𝒙=(x0,x1)𝒙subscript𝑥0subscript𝑥1\bm{x}=(x_{0},x_{1}).
hK+superscriptsubscriptℎ𝐾h_{K}^{+} has the same cutoff as hKsubscriptℎ𝐾h_{K} along the coordinate axes, i.e., 𝒇c,x=(fc,0)subscript𝒇𝑐𝑥subscript𝑓𝑐0\bm{f}_{c,x}=(f_{c},0) and 𝒇c,y=(0,fc)subscript𝒇𝑐𝑦0subscript𝑓𝑐\bm{f}_{c,y}=(0,f_{c}), and its frequency response forms a square shape over the 2D plane, implying that the cutoff frequency along the diagonal is 𝒇c,d=(fc,fc)subscript𝒇𝑐𝑑subscript𝑓𝑐subscript𝑓𝑐\bm{f}_{c,d}=(f_{c},f_{c}).
In practice, a separable filter can be implemented efficiently by first filtering each row of the two-dimensional signal independently with hKsubscriptℎ𝐾h_{K} and then doing the same for each column.
This makes hK+superscriptsubscriptℎ𝐾h_{K}^{+} an ideal choice for all upsampling filters in our generator, as well as the downsampling filters in configs a–t (Figure 3, left).

The fact that the spectrum of hK+superscriptsubscriptℎ𝐾h_{K}^{+} is not radially symmetric, i.e., ‖𝒇c,d‖≠‖𝒇c,x‖normsubscript𝒇𝑐𝑑normsubscript𝒇𝑐𝑥\|\bm{f}_{c,d}\|\neq\|\bm{f}_{c,x}\|, is problematic considering config r.
If we rotate the input feature maps of a given layer, their frequency content will rotate as well.
To enforce rotation equivariant behavior, we must ensure that the effective cutoff frequencies remain unchanged by this.
The ideal radially symmetric low-pass filter [9] is given by ψs∘​(𝒙)=(2​fc)2⋅jinc⁡(2​fc​∥𝒙∥)superscriptsubscript𝜓𝑠𝒙⋅superscript2subscript𝑓𝑐2jinc2subscript𝑓𝑐delimited-∥∥𝒙\psi_{s}^{\circ}(\bm{x})=(2f_{c})^{2}\cdot\operatorname{jinc}(2f_{c}\lVert\bm{x}\rVert).
The jincjinc\operatorname{jinc} function, also known as besinc, sombrero function, or Airy disk, is defined as jinc⁡(x)=2​J1​(π​x)/(π​x)jinc𝑥2subscript𝐽1𝜋𝑥𝜋𝑥\operatorname{jinc}(x)=2J_{1}(\pi x)/(\pi x), where J1subscript𝐽1J_{1} is the first order Bessel function of the first kind.
Using the same windowing scheme as before, we define the corresponding practical filter as

hK∘​(𝒙)=(2​fc)2⋅jinc⁡(2​fc​‖𝒙‖)⋅wK​(x0)⋅wK​(x1).superscriptsubscriptℎ𝐾𝒙⋅⋅superscript2subscript𝑓𝑐2jinc2subscript𝑓𝑐norm𝒙subscript𝑤𝐾subscript𝑥0subscript𝑤𝐾subscript𝑥1\displaystyle h_{K}^{\circ}(\bm{x})=(2f_{c})^{2}\cdot\operatorname{jinc}(2f_{c}\|\bm{x}\|)\cdot w_{K}(x_{0})\cdot w_{K}(x_{1}).

(12)

Note that even though jincjinc\operatorname{jinc} is radially symmetric, we still treat the window function as separable in order to retain its spectral properties.
In config r, we perform all downsampling operations using hK∘superscriptsubscriptℎ𝐾h_{K}^{\circ}, except for the last two critically sampled layers where we revert to hK+superscriptsubscriptℎ𝐾h_{K}^{+}.

### C.5 Alternative filters

In Figure 5, right, we compare the effectiveness of Kaiser filters against two alternatives: Lanczos and Gaussian.
These filters are typically defined using prototypical filter kernels kLsubscript𝑘𝐿k_{L} and kGsubscript𝑘𝐺k_{G}, respectively:

kL​(x)subscript𝑘𝐿𝑥\displaystyle k_{L}(x)
={sinc⁡(x)⋅sinc⁡(x/a),if ​|x|<a,0,if ​|x|≥a,absentcases⋅sinc𝑥sinc𝑥𝑎if 𝑥𝑎0if 𝑥𝑎\displaystyle=\begin{cases}\operatorname{sinc}(x)\cdot\operatorname{sinc}(x/a),&\textrm{if }|x|<a,\\
0,&\textrm{if }|x|\geq a,\end{cases}

(13)

kG​(x)subscript𝑘𝐺𝑥\displaystyle k_{G}(x)
=exp⁡(−12​(x/σ)2)/(σ​2​π),absent12superscript𝑥𝜎2𝜎2𝜋\displaystyle=\exp\left(-\frac{1}{2}(x/\sigma)^{2}\right)\Big{/}\left(\sigma\sqrt{2\pi}\right),

(14)

where a𝑎a is the spatial extent of the Lanczos kernel, typically set to 2 or 3, and σ𝜎\sigma is the standard deviation of the Gaussian kernel.
In Figure 5 of the main paper we set a=2𝑎2a=2 and σ=0.4𝜎0.4\sigma=0.4; we tested several different values and found these choices to work reasonably well.

The main shortcoming of the prototypical kernels is that they do not provide an explicit way to control the cutoff frequency.
In order to enable apples-to-apples comparison, we assume that the kernels have an implicit cutoff frequency at 0.5 and scale their impulse responses to account for the varying fcsubscript𝑓𝑐f_{c}:

hL​(x)subscriptℎ𝐿𝑥\displaystyle h_{L}(x)
=2​fc⋅kL​(2​fc​x),absent⋅2subscript𝑓𝑐subscript𝑘𝐿2subscript𝑓𝑐𝑥\displaystyle=2f_{c}\cdot k_{L}(2f_{c}x),
hG​(x)subscriptℎ𝐺𝑥\displaystyle h_{G}(x)
=2​fc⋅kG​(2​fc​x).absent⋅2subscript𝑓𝑐subscript𝑘𝐺2subscript𝑓𝑐𝑥\displaystyle=2f_{c}\cdot k_{G}(2f_{c}x).

(15)

We limit the computational complexity of the Gaussian filter by enforcing hG​(x)=0subscriptℎ𝐺𝑥0h_{G}(x)=0 when |x|>8/s𝑥8𝑠|x|>8/s, with respect to the input sampling rate in the upsampling case.
In practice, hG​(x)subscriptℎ𝐺𝑥h_{G}(x) is already very close to zero in this range, so the effect of this approximation is negligible.
Finally, we extend the filters to two dimensions by defining the corresponding separable filters:

hL+​(𝒙)superscriptsubscriptℎ𝐿𝒙\displaystyle h_{L}^{+}(\bm{x})
=(2​fc)2⋅kL​(2​fc​x0)⋅kL​(2​fc​x1),absent⋅⋅superscript2subscript𝑓𝑐2subscript𝑘𝐿2subscript𝑓𝑐subscript𝑥0subscript𝑘𝐿2subscript𝑓𝑐subscript𝑥1\displaystyle=(2f_{c})^{2}\cdot k_{L}(2f_{c}x_{0})\cdot k_{L}(2f_{c}x_{1}),
hG+​(𝒙)superscriptsubscriptℎ𝐺𝒙\displaystyle h_{G}^{+}(\bm{x})
=(2​fc)2⋅kG​(2​fc​x0)⋅kG​(2​fc​x1).absent⋅⋅superscript2subscript𝑓𝑐2subscript𝑘𝐺2subscript𝑓𝑐subscript𝑥0subscript𝑘𝐺2subscript𝑓𝑐subscript𝑥1\displaystyle=(2f_{c})^{2}\cdot k_{G}(2f_{c}x_{0})\cdot k_{G}(2f_{c}x_{1}).

(16)

Note that hG+superscriptsubscriptℎ𝐺h_{G}^{+} is radially symmetric by construction, which makes it ideal for rotation equivariance.
hL+superscriptsubscriptℎ𝐿h_{L}^{+}, however, has no widely accepted radially symmetric counterpart, so we simply use the same separable filter in config r as well.

## Appendix D Custom CUDA kernel for filtered nonlinearity

Implementing the upsample-nonlinearity-downsample sequence is inefficient using the standard primitives available in modern deep learning frameworks.
The intermediate feature maps have to be transferred between on-chip and off-chip GPU memory multiple times and retained for the backward pass.
This is especially costly because the intermediate steps operate on upsampled, high-resolution data.
To overcome this, we implement the entire sequence as a single operation using a custom CUDA kernel.
This improves training performance by approximately an order of magnitude thanks to reduced memory traffic, and also decreases GPU memory usage significantly.

The combined kernel consists of four phases: input, upsampling, nonlinearity, and downsampling.
The computation is parallelized by subdividing the output feature maps into non-overlapping tiles, and computing one output tile per CUDA thread block.
First, in input phase, the corresponding input region is read into on-chip shared memory of the thread block.
Note that the input regions for neighboring output tiles will overlap spatially due to the spatial extent of filters.

The execution of up-/downsampling phases depends on whether the corresponding filters are separable or not.
For a separable filter, we perform vertical and horizontal 1D convolutions sequentially, whereas a non-separable filter requires a single 2D convolution.
All these convolutions and the nonlinearity operate in on-chip shared memory, and only the final output of the downsampling phase is written to off-chip GPU memory.

### D.1 Gradient computation

To compute gradients of the combined operation, they need to propagate through each of the phases in reverse order.
Fortunately, the combined upsample-nonlinearity-downsample operation is mostly self-adjoint with proper changes in parameters, e.g., swapping the up-/downsampling factors and the associated filters.
The only problematic part is the nonlinearity that is performed in the upsampled resolution.
A naïve but general solution would be to store the intermediate high-resolution input to the nonlinearity, but the memory consumption would be infeasible for training large models.

Our kernel is specialized to use leaky ReLU as the nonlinearity, which offers a straightforward way to conserve memory: to propagate gradients, it is sufficient to know whether the corresponding input value to nonlinearity was positive or negative.
When using 16-bit floating-point datatypes, there is an additional complication because the outputs of the nonlinearity need to be clamped [32], and when this occurs, the corresponding gradients must be zero.
Therefore, in the forward pass we store two bits of auxiliary information per value to cover the three possible cases: positive, negative, or clamped.
In the backward pass, reading these bits is sufficient for correct gradient computation — no other information from the forward pass is needed.

### D.2 Optimizations for common upsampling factors

Let us consider one-dimensional 2×\times upsampling where the input is (virtually) interleaved with zeros and convolved with an n′superscript𝑛′n^{\prime}-tap filter where n′=2​nsuperscript𝑛′2𝑛n^{\prime}=2n (cf. Equation 9).
There are n𝑛n nonzero input values under the n′superscript𝑛′n^{\prime}-tap kernel, so if each output pixel is computed separately, the convolution requires n𝑛n multiply-add operations per pixel and equally many shared memory load instructions, for a total of 2​n2𝑛2n instructions per output pixel.333Input of the upsampling is stored in shared memory, but the filter weights can be stored in CUDA constant memory where they can be accessed without a separate load instruction.
However, note that the computation of two neighboring output pixels accesses only n+1𝑛1n+1 input pixels in total.
By computing two output pixels at a time and avoiding redundant shared memory load instructions, we obtain an average cost of 32​n+1232𝑛12\frac{3}{2}n+\frac{1}{2} instructions per pixel — close to 25% savings.
For 4×\times upsampling, we can similarly reduce the instruction count by up to 37.5% by computing four output pixels at a time.
We apply these optimizations in 2×\times and 4×\times upsampling for both separable and non-separable filters.

upsample 2×\times

upsample 4×\times

upsample 2×\times

downsample 2×\times

downsample 2×\times

downsample 4×\times

Sep. up

yes

yes

no

no

yes

yes

no

no

yes

yes

no

no

Sep. down

yes

no

yes

no

yes

no

yes

no

yes

no

yes

no

PyTorch (ms)

7.88

12.40

12.68

17.12

10.07

31.51

14.96

36.33

39.35

56.73

125.83

143.15

Ours (ms)

0.42

00.59

00.66

00.92

00.49

00.84

00.80

01.01

01.20

01.89

003.04

003.66

Speedup ×\times

19

21

19

19

21

38

19

36

33

30

041

039

Figure 14 benchmarks the performance of our kernel with various up-/downsampling factors and with separable and non-separable filters.
In network layers that keep the sampling rate fixed, both factors are 2×\times, whereas layers that increase the sampling rate by a factor of two, 4×\times upsampling is combined with 2×\times downsampling.
The remaining combination of 2×\times upsampling and 4×\times downsampling is needed when computing gradients of the latter case.
The speedup over native PyTorch operations varies between ∼similar-to\sim20–40×\times, which yields an overall training speedup of approximately 10×\times.

## Appendix E Equivariance metrics

In this section, we describe our equivariance metrics, EQ-T and EQ-R, in detail.
We also present additional results using an alternative translation metric, EQ-Tfrac, based on fractional sub-pixel translation.

We express each of our metrics as the peak signal-to-noise ratio (PSNR) between two sets of images, measured in decibels (dB).
PSNR is a commonly used metric in image restoration literature.
In the typical setting we have two signals, reference I𝐼I and its noisy approximation K𝐾K, defined over discrete domain 𝒟𝒟\mathcal{D} — usually a two-dimensional pixel grid.
The PSNR between I𝐼I and K𝐾K is then defined via the mean squared error (MSE):

MSE𝒟​(I,K)subscriptMSE𝒟𝐼𝐾\displaystyle\text{MSE}_{\mathcal{D}}(I,K)
=1‖𝒟‖​∑i∈𝒟(I​[i]−K​[i])2,absent1norm𝒟subscript𝑖𝒟superscript𝐼delimited-[]𝑖𝐾delimited-[]𝑖2\displaystyle=\frac{1}{\|\mathcal{D}\|}\sum_{i\in\mathcal{D}}\big{(}I[i]-K[i]\big{)}^{2},

(17)

PSNR𝒟​(I,K)subscriptPSNR𝒟𝐼𝐾\displaystyle\text{PSNR}_{\mathcal{D}}(I,K)
=10⋅log10⁡(I𝑚𝑎𝑥2MSE𝒟​(I,K)),absent⋅10subscript10subscriptsuperscript𝐼2𝑚𝑎𝑥subscriptMSE𝒟𝐼𝐾\displaystyle=10\cdot\log_{10}\left(\frac{I^{2}_{\mathit{max}}}{\text{MSE}_{\mathcal{D}}(I,K)}\right),

(18)

where MSE𝒟​(I,K)subscriptMSE𝒟𝐼𝐾\text{MSE}_{\mathcal{D}}(I,K) is the average squared difference between matching elements of I𝐼I and K𝐾K.
I𝑚𝑎𝑥subscript𝐼𝑚𝑎𝑥I_{\mathit{max}} is the expected dynamic range of the reference signal, i.e., I𝑚𝑎𝑥≈maxi∈𝒟⁡(I​[i])−mini∈𝒟⁡(I​[i])subscript𝐼𝑚𝑎𝑥subscript𝑖𝒟𝐼delimited-[]𝑖subscript𝑖𝒟𝐼delimited-[]𝑖I_{\mathit{max}}\approx\max_{i\in\mathcal{D}}(I[i])-\min_{i\in\mathcal{D}}(I[i]).
The dynamic range is usually considered to be a global constant, e.g., the range of valid RGB values, as opposed to being dependent on the content of I𝐼I.
In our case, I𝐼I and K𝐾K represent desired and actual outputs of the synthesis network, respectively, with a dynamic range of [−1,1]11[-1,1].
This implies that I𝑚𝑎𝑥=2subscript𝐼𝑚𝑎𝑥2I_{\mathit{max}}=2.
High PSNR values indicate that K𝐾K is close to I𝐼I; in the extreme case, where K=I𝐾𝐼K=I, we have PSNR𝒟​(I,K)=∞subscriptPSNR𝒟𝐼𝐾\text{PSNR}_{\mathcal{D}}(I,K)=\infty dB.

Since we are interested in sets of images, we use a slightly extended definition for MSE that allows I𝐼I and K𝐾K to be defined over an arbitrary, potentially uncountable domain:

MSE𝒟​(I,K)=𝔼i∼𝒟​[(I​(i)−K​(i))2].subscriptMSE𝒟𝐼𝐾subscript𝔼similar-to𝑖𝒟delimited-[]superscript𝐼𝑖𝐾𝑖2\text{MSE}_{\mathcal{D}}(I,K)=\mathbb{E}_{i\sim\mathcal{D}}\left[\big{(}I(i)-K(i)\big{)}^{2}\right].

(19)

### E.1 Integer translation

The goal of our integer translation metric, EQ-T, is to measure how closely, on average, the output the synthesis network 𝐆𝐆\mathbf{G} matches a translated reference image when we translate the input of 𝐆𝐆\mathbf{G}.
In other words,

EQ-T=PSNR𝒲×𝒳2×𝒱×𝒞​(I𝐭,K𝐭),I𝐭​(𝐰,𝒙,𝒑,c)=𝐓𝒙​[𝐆​(z0;𝐰)]​[𝒑,c],K𝐭​(𝐰,𝒙,𝒑,c)=𝐆​(𝐭𝒙​[z0];𝐰)​[𝒑,c],EQ-TsubscriptPSNR𝒲superscript𝒳2𝒱𝒞subscript𝐼𝐭subscript𝐾𝐭subscript𝐼𝐭𝐰𝒙𝒑𝑐subscript𝐓𝒙delimited-[]𝐆subscript𝑧0𝐰𝒑𝑐subscript𝐾𝐭𝐰𝒙𝒑𝑐𝐆subscript𝐭𝒙delimited-[]subscript𝑧0𝐰𝒑𝑐\displaystyle\begin{array}[]{l}\text{EQ-T}=\text{PSNR}_{\mathcal{W}\times\mathcal{X}^{2}\times\mathcal{V}\times\mathcal{C}}(I_{\mathbf{t}},K_{\mathbf{t}}),\\[7.11317pt]
I_{\mathbf{t}}({\bf w},\bm{x},\bm{p},c)=\mathbf{T}_{\bm{x}}\big{[}\mathbf{G}(z_{0};{\bf w})\big{]}[\bm{p},c],\\[7.11317pt]
K_{\mathbf{t}}({\bf w},\bm{x},\bm{p},c)=\mathbf{G}(\mathbf{t}_{\bm{x}}[z_{0}];{\bf w})[\bm{p},c],\end{array}

(23)

where 𝐰∼𝒲similar-to𝐰𝒲{\bf w}\sim\mathcal{W} is a random intermediate latent code produced by the mapping network, 𝒙=(x0,x1)∼𝒳2𝒙subscript𝑥0subscript𝑥1similar-tosuperscript𝒳2\bm{x}=(x_{0},x_{1})\sim\mathcal{X}^{2} is a random translation offset, 𝒑𝒑\bm{p} enumerates pixel locations in the mutually valid region 𝒱𝒱\mathcal{V}, c∼𝒞similar-to𝑐𝒞c\sim\mathcal{C} is the color channel, and z0subscript𝑧0z_{0} represents the input Fourier features.
For integer translations, we sample the translation offsets x0subscript𝑥0x_{0} and x1subscript𝑥1x_{1} from 𝒳=𝒰​[−sN/8,sN/8]𝒳𝒰subscript𝑠𝑁8subscript𝑠𝑁8\mathcal{X}=\mathcal{U}[-s_{N}/8,s_{N}/8], where sNsubscript𝑠𝑁s_{N} is the width of the image in pixels.

In practice, we estimate the expectation in Equation 23 as an average over 50,000 random samples of (𝐰,𝒙)∼𝒲×𝒳2similar-to𝐰𝒙𝒲superscript𝒳2({\bf w},\bm{x})\sim\mathcal{W}\times\mathcal{X}^{2}.
For given 𝐰𝐰{\bf w} and 𝒙𝒙\bm{x}, we generate the reference image I𝐭subscript𝐼𝐭I_{\mathbf{t}} by running the synthesis network and translating the resulting image by 𝒙𝒙\bm{x} pixels (operator 𝐓𝒙subscript𝐓𝒙\mathbf{T}_{\bm{x}}).
We then obtain the approximate result image K𝐭subscript𝐾𝐭K_{\mathbf{t}} by translating the input Fourier features by the corresponding amount (operator 𝐭𝒙subscript𝐭𝒙\mathbf{t}_{\bm{x}}), as discussed in Appendix F.1, and running the synthesis network again.
The mutually valid region of I𝐭subscript𝐼𝐭I_{\mathbf{t}} (translated by (x0,x1)subscript𝑥0subscript𝑥1(x_{0},x_{1})) and K𝐭subscript𝐾𝐭K_{\mathbf{t}} (translated by (0,0)00(0,0)) is given by

𝒱={max(x0,0),…,sN+min(x0,0)−1}×{max⁡(x1,0),…,sN+min⁡(x1,0)−1}.\begin{array}[]{l}\mathcal{V}=\{\max(x_{0},0),\ldots,s_{N}+\min(x_{0},0)-1\}\times\\[7.11317pt]
\hskip 20.20146pt\{\max(x_{1},0),\ldots,s_{N}+\min(x_{1},0)-1\}.\end{array}

(24)

### E.2 Fractional translation

Configuration

FID

EQ-T

EQ-Tfrac

a
StyleGAN2

5.14

–

–

b
+ Fourier features

4.79

16.23

16.28

c
+ No noise inputs

4.54

15.81

15.84

d
+ Simplified generator

5.21

19.47

19.57

e
+ Boundaries & upsampling

6.02

24.62

24.70

f
+ Filtered nonlinearities

6.35

30.60

30.68

g
+ Non-critical sampling

4.78

43.90

42.24

h
+ Transformed Fourier features

4.64

45.20

42.78

t
+ Flexible layers (StyleGAN3-T)

4.62

63.01

46.40

r
+ Rotation equiv. (StyleGAN3-R)

4.50

66.65

45.92

Parameter

FID

EQ-T

EQ-Tfrac

Filter size n=4𝑛4n=4

4.72

57.49

44.65

*

Filter size n=6𝑛6n=6

4.50

66.65

45.92

Filter size n=8𝑛8n=8

4.66

65.57

46.57

Upsampling m=1𝑚1m=1

4.38

39.96

37.55

*

Upsampling m=2𝑚2m=2

4.50

66.65

45.92

Upsampling m=4𝑚4m=4

4.57

74.21

46.81

Stopband ft,0=21.5subscript𝑓𝑡0superscript21.5f_{t,0}=2^{1.5}

4.62

51.10

44.46

*

Stopband ft,0=22.1subscript𝑓𝑡0superscript22.1f_{t,0}=2^{2.1}

4.50

66.65

45.92

Stopband ft,0=23.1subscript𝑓𝑡0superscript23.1f_{t,0}=2^{3.1}

4.68

73.13

46.27

Our translation equivariance metric has the nice property that, for a perfectly equivariant generator, the value of EQ-T converges to ∞\infty dB when the number of samples tends to infinity.
However, this comes at the cost of completely ignoring subpixel effects.
In fact, it is easy to imagine a generator that is perfectly equivariant to integer translation but fails with subpixel translation; in principle, this is true for any generator whose output is not properly bandlimited, including, e.g., implicit coordinate-based MLPs [4].

To verify that our generators are able to handle subpixel translation, we define an alternative translation equivariance metric, EQ-Tfrac, where the translation offsets x0subscript𝑥0x_{0} and x1subscript𝑥1x_{1} are sampled from a continuous distribution 𝒳=𝒰​(−sN/8,sN/8)𝒳𝒰subscript𝑠𝑁8subscript𝑠𝑁8\mathcal{X}=\mathcal{U}(-s_{N}/8,s_{N}/8).
While the continuous operator 𝐭𝒙subscript𝐭𝒙\mathbf{t}_{\bm{x}} readily supports this new definition with fractional offsets, extending the discrete 𝐓𝒙subscript𝐓𝒙\mathbf{T}_{\bm{x}} is slightly more tricky.

In practice, we define 𝐓𝒙subscript𝐓𝒙\mathbf{T}_{\bm{x}} via standard Lanczos resampling, by filtering the image produced by 𝐆𝐆\mathbf{G} using the prototypical Lanczos filter (Equation 15) with a=3𝑎3a=3, evaluated at integer tap locations offset by 𝒙𝒙\bm{x}.
We explicitly normalize the resulting discretized filter to enforce the partition of unity property.
We also shrink the mutually valid region to account for the spatial extent a𝑎a by redefining

𝒱={max(x0+a,0),…,sN+min(x0−a,−1)}×{max⁡(x1+a,0),…,sN+min⁡(x1−a,−1)}.\begin{array}[]{l}\mathcal{V}=\{\max(x_{0}+a,0),\ldots,s_{N}+\min(x_{0}-a,-1)\}\times\\[7.11317pt]
\hskip 20.20146pt\{\max(x_{1}+a,0),\ldots,s_{N}+\min(x_{1}-a,-1)\}.\end{array}

(25)

Figure 15 compares the results of the two metrics, EQ-T and EQ-Tfrac, using the same training configurations as Figure 3 in the main paper.
The metrics agree reasonably well up until ∼similar-to\sim40 dB, after which the fractional metric starts to saturate; it consistently fails to rise above 50 dB in our tests.
This is due to the fact that the definition of subpixel translation is inherently ambiguous.
The choice of the resampling filter represents a tradeoff between aliasing, ringing, and retention of high frequencies; there is no reason to assume that the generator would necessarily have to make the same tradeoff as the metric.
Based on the results, we conclude that our configs g–r are essentially perfectly equivariant to subpixel translation within the limits of Lanczos resampling’s accuracy.
However, due to its inherent limitations, we refrain from choosing EQ-Tfrac as our primary metric.

### E.3 Rotation

Measuring equivariance with respect to arbitrary rotations has the same fundamental limitation as our EQ-Tfrac metric: the resampling operation is inherently ambiguous, so we cannot except the results to be perfectly accurate beyond ∼similar-to\sim40 dB.
Arbitrary rotations also have the additional complication that the bandlimit of a discretely sampled image is not radially symmetric.

Consider rotating the continuous representation of a discretely sampled image by 45∘.
The original frequency content of the image is constrained within the rectangular bandlimit 𝒇∈[−sN/2,+sN/2]2𝒇superscriptsubscript𝑠𝑁2subscript𝑠𝑁22\bm{f}\in[-s_{N}/2,+s_{N}/2]^{2}.
The frequency content of the rotated image, however, forms a diamond shape that extends all the way to ‖𝒇‖=2​sN/2norm𝒇2subscript𝑠𝑁2\|\bm{f}\|=\sqrt{2}s_{N}/2 along the main axes but only to ‖𝒇‖=sN/2norm𝒇subscript𝑠𝑁2\|\bm{f}\|=s_{N}/2 along the diagonals.
In other words, it simultaneously has too much frequency content, but also too little.
This has two implications.
First, in order to obtain a valid discretized result image, we have to low-pass filter the image both before and after the rotation to completely eliminate aliasing.
Second, even if we are successful in eliminating the aliasing, the rotated image will still lack the highest representable diagonal frequencies.
The second point further implies that when computing PSNR, our reference image I𝐼I will inevitably lack some frequencies that are present in the output of 𝐆𝐆\mathbf{G}.
To obtain the correct result, we must eliminate these extraneous frequencies — without modifying the output image in any other way.

Based on the above reasoning, we define our EQ-R metric as follows:

EQ-R=PSNR𝒲×𝒜×𝒱×𝒞​(I𝐫,K𝐫),I𝐫​(𝐰,α,𝒑,c)=𝐑α​[𝐆​(z0;𝐰)]​[𝒑,c],K𝐫​(𝐰,α,𝒑,c)=𝐑α∗​[𝐆​(𝐫α​[z0];𝐰)]​[𝒑,c],EQ-RsubscriptPSNR𝒲𝒜𝒱𝒞subscript𝐼𝐫subscript𝐾𝐫subscript𝐼𝐫𝐰𝛼𝒑𝑐subscript𝐑𝛼delimited-[]𝐆subscript𝑧0𝐰𝒑𝑐subscript𝐾𝐫𝐰𝛼𝒑𝑐subscriptsuperscript𝐑𝛼delimited-[]𝐆subscript𝐫𝛼delimited-[]subscript𝑧0𝐰𝒑𝑐\displaystyle\begin{array}[]{l}\text{EQ-R}=\text{PSNR}_{\mathcal{W}\times\mathcal{A}\times\mathcal{V}\times\mathcal{C}}(I_{\mathbf{r}},K_{\mathbf{r}}),\\[7.11317pt]
I_{\mathbf{r}}({\bf w},\alpha,\bm{p},c)=\mathbf{R}_{\alpha}\big{[}\mathbf{G}(z_{0};{\bf w})\big{]}[\bm{p},c],\\[7.11317pt]
K_{\mathbf{r}}({\bf w},\alpha,\bm{p},c)=\mathbf{R}^{*}_{\alpha}\big{[}\mathbf{G}(\mathbf{r}_{\alpha}[z_{0}];{\bf w})\big{]}[\bm{p},c],\end{array}

(29)

where the random rotation angle α𝛼\alpha is drawn from 𝒜=𝒰​(0∘,360∘)𝒜𝒰superscript0superscript360\mathcal{A}=\mathcal{U}(0^{\circ},360^{\circ}) and operator 𝐫αsubscript𝐫𝛼\mathbf{r}_{\alpha} corresponds to continuous rotation of the input Fourier features by α𝛼\alpha with respect to the center of the canvas [0,1]2superscript012[0,1]^{2}.
𝐑αsubscript𝐑𝛼\mathbf{R}_{\alpha} corresponds to high-quality rotation of the reference image, and 𝐑α∗subscriptsuperscript𝐑𝛼\mathbf{R}^{*}_{\alpha} represents a pseudo-rotation operator that modifies the frequency content of the image as if it had undergone 𝐑αsubscript𝐑𝛼\mathbf{R}_{\alpha} — but without actually rotating it.

The ideal rotation operator 𝐑^^𝐑\hat{\mathbf{R}} is easily defined under our theoretical framework presented in Section 2.1:

𝐑^α[Z]=⊙(ψ∗𝐫α[ϕ∗Z])=1/s2⋅⊙(ψ∗𝐫α[ψ∗Z]).\hat{\mathbf{R}}_{\alpha}[Z]=\Sha\odot\big{(}\psi\ast\mathbf{r}_{\alpha}[\phi\ast Z]\big{)}=1/s^{2}\cdot\Sha\odot\big{(}\psi\ast\mathbf{r}_{\alpha}[\psi\ast Z]\big{)}.

(30)

In other words, we first convolve the discretely sampled input image Z𝑍Z with ϕitalic-ϕ\phi to obtain the corresponding continuous representation.
We then rotate this continuous representation using 𝐫αsubscript𝐫𝛼\mathbf{r}_{\alpha}, bandlimit the result by convolving with ψ𝜓\psi, and finally extract the corresponding discrete representation by multiplying with . To reduce notational clutter, we omit the subscripts denoting the sampling rate s𝑠s.
We can swap the order of the rotation and a convolution in the above formula by rotating the kernel in the opposite direction to compensate:

𝐑^α​[Z]subscript^𝐑𝛼delimited-[]𝑍\displaystyle\hat{\mathbf{R}}_{\alpha}[Z]
=1/s2⋅⊙𝐫α[h^R∗Z],\displaystyle=1/s^{2}\cdot\Sha\odot\mathbf{r}_{\alpha}[\hat{h}_{R}\ast Z],
h^Rsubscript^ℎ𝑅\displaystyle\hat{h}_{R}
=𝐫−α​[ψ]∗ψ,absent∗subscript𝐫𝛼delimited-[]𝜓𝜓\displaystyle=\mathbf{r}_{-\alpha}[\psi]\ast\psi,

(31)

where h^Rsubscript^ℎ𝑅\hat{h}_{R} represents an ideal “rotation filter” that bandlimits the signal with respect to both the input and the output. Its spectrum is the eight-sided polygonal intersection of the original and the rotated rectangle.

In order to obtain a practical approximation 𝐑αsubscript𝐑𝛼\mathbf{R}_{\alpha}, we must replace h^Rsubscript^ℎ𝑅\hat{h}_{R} with an approximate filter hRsubscriptℎ𝑅h_{R} that has finite support.
Given such a filter, we get 𝐑α[Z]=1/s2⋅⊙𝐫α[hR∗Z]\mathbf{R}_{\alpha}[Z]=1/s^{2}\cdot\Sha\odot\mathbf{r}_{\alpha}[h_{R}\ast Z].
In practice, we implement this operation using two additional approximations.
First, we approximate 1/s2⋅hR∗Z∗⋅1superscript𝑠2subscriptℎ𝑅𝑍1/s^{2}\cdot h_{R}\ast Z by an upsampling operation to a higher temporary resolution, using hRsubscriptℎ𝑅h_{R} as the upsampling filter and m=4𝑚4m=4.
Second, we approximate ⊙𝐫αdirect-productabsentsubscript𝐫𝛼\Sha\odot\mathbf{r}_{\alpha} by performing a set of bilinear lookups from the temporary high-resolution image.

To obtain hRsubscriptℎ𝑅h_{R}, we again utilize the standard Lanczos window with a=3𝑎3a=3:

hR=(𝐫−α​[ψ]∗ψ)⊙(𝐫−α​[wL+]∗wL+),subscriptℎ𝑅direct-product∗subscript𝐫𝛼delimited-[]𝜓𝜓∗subscript𝐫𝛼delimited-[]subscriptsuperscript𝑤𝐿subscriptsuperscript𝑤𝐿h_{R}=\big{(}\mathbf{r}_{-\alpha}[\psi]\ast\psi\big{)}\odot(\mathbf{r}_{-\alpha}[w^{+}_{L}]\ast w^{+}_{L}),

(32)

where we apply the same rotation-convolution to both the filter and the window function.
wL+subscriptsuperscript𝑤𝐿w^{+}_{L} corresponds the canonical separable Lanczos window, similar to the one used in Equation 15:

wL+​(𝒙)={sinc⁡(x0/a)⋅sinc⁡(x1/a),if ​max⁡(|x0|,|x1|)<a,0,if ​max⁡(|x0|,|x1|)≥a,subscriptsuperscript𝑤𝐿𝒙cases⋅sincsubscript𝑥0𝑎sincsubscript𝑥1𝑎if subscript𝑥0subscript𝑥1𝑎0if subscript𝑥0subscript𝑥1𝑎w^{+}_{L}(\bm{x})=\begin{cases}\operatorname{sinc}(x_{0}/a)\cdot\operatorname{sinc}(x_{1}/a),&\textrm{if }\max(|x_{0}|,|x_{1}|)<a,\\
0,&\textrm{if }\max(|x_{0}|,|x_{1}|)\geq a,\end{cases}

(33)

We can now define the pseudo-rotation operator 𝐑α∗​[Z]subscriptsuperscript𝐑𝛼delimited-[]𝑍\mathbf{R}^{*}_{\alpha}[Z] as a simple convolution with another filter that resembles hRsubscriptℎ𝑅h_{R}:

𝐑α∗[Z]=1/s2⋅⊙(hR∗∗Z)=HR∗∗Z,hR∗=(ψ∗𝐫α​[ψ])⊙(wL+∗𝐫α​[wL+]),\begin{array}[]{l}\mathbf{R}^{*}_{\alpha}[Z]=1/s^{2}\cdot\Sha\odot(h^{*}_{R}\ast Z)=H^{*}_{R}\ast Z,\\[7.11317pt]
h^{*}_{R}=\big{(}\psi\ast\mathbf{r}_{\alpha}[\psi]\big{)}\odot(w^{+}_{L}\ast\mathbf{r}_{\alpha}[w^{+}_{L}]),\end{array}

(34)

where the discrete version HR∗subscriptsuperscript𝐻𝑅H^{*}_{R} is obtained from hR∗subscriptsuperscriptℎ𝑅h^{*}_{R} using Equation 6.

Finally, we define the valid region 𝒱𝒱\mathcal{V} the same way as in Appendix E.2: the set of pixels for which both filter footprints fall within the bounds of the corresponding original images.

## Appendix F Implementation details

We implemented our alias-free generator on top of the official PyTorch implementation of StyleGAN2-ADA, available at https://github.com/NVlabs/stylegan2-ada-pytorch.
We kept most of the details unchanged, including
discriminator architecture [34],
weight demodulation [34],
equalized learning rate for all trainable parameters [31],
minibatch standard deviation layer at the end of the discriminator [31],
exponential moving average of generator weights [31],
mixed-precision FP16/FP32 training [32],
non-saturating logistic loss [21],
R1subscript𝑅1R_{1} regularization [40],
lazy regularization [34],
and Adam optimizer [36] with β1=0subscript𝛽10\beta_{1}=0, β2=0.99subscript𝛽20.99\beta_{2}=0.99, and ϵ=10−8italic-ϵsuperscript108\epsilon=10^{-8}.

We ran all experiments on NVIDIA DGX-1 with 8 Tesla V100 GPUs using PyTorch 1.7.1, CUDA 11.0, and cuDNN 8.0.5.
We computed FID between 50k generated images and all training images using the official pre-trained Inception network, available at http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

Our implementation and pre-trained models are available at https://github.com/NVlabs/stylegan3

### F.1 Generator architecture

#### Normalization (configs d–r)

We have observed that eliminating the output skip connections in StyleGAN2 [34] results in uncontrolled drift of signal magnitudes over the generator layers.
This does not necessarily lead to lower-quality results, but it generally increases the amount of random variation between training runs and may occasionally lead to numerical issues with mixed-precision training.
We eliminate the drift by tracking a long-term exponential moving average of the input signal magnitude on each layer and normalizing the feature maps accordingly.
We update the moving average once per training iteration, based on the mean of squares over the entire input tensor, and freeze its value after training.
We initialize the moving average to 1 and decay it at a constant rate, resulting in 50% decay per 20k real images shown to the discriminator.
With this explicit normalization in place, we have found it beneficial to slightly adjust the dynamic range of the output RGB colors.
StyleGAN2 uses −11-1 and +11+1 to represent black and white, respectively; we change these values to −44-4 and +44+4 starting from config d and, for consistency with the original generator, divide the color channels by 4 afterwards.

#### Transformed Fourier features (configs h–r)

We enable the orientation of the input features z0subscript𝑧0z_{0} to vary on a per-image basis by introducing an additional affine layer (Figure 4b) and applying a geometric transformation based on its output.
The affine layer produces a four-dimensional vector 𝒕=(rc,rs,tx,ty)𝒕subscript𝑟𝑐subscript𝑟𝑠subscript𝑡𝑥subscript𝑡𝑦\bm{t}=(r_{c},r_{s},t_{x},t_{y}) based on 𝐰𝐰{\bf w}.
We initialize its weights so that 𝒕=(1,0,0,0)𝒕1000\bm{t}=(1,0,0,0) at the beginning, but allow them to change freely over the course of training.
To interpret 𝒕𝒕\bm{t} as a geometric transformation, we first normalize its value based on the first two components, i.e., 𝒕′=(rc′,rs′,tx′,ty′)=𝒕/rc2+rs2superscript𝒕′superscriptsubscript𝑟𝑐′superscriptsubscript𝑟𝑠′superscriptsubscript𝑡𝑥′superscriptsubscript𝑡𝑦′𝒕superscriptsubscript𝑟𝑐2superscriptsubscript𝑟𝑠2\bm{t}^{\prime}=(r_{c}^{\prime},r_{s}^{\prime},t_{x}^{\prime},t_{y}^{\prime})=\bm{t}\big{/}\sqrt{r_{c}^{2}+r_{s}^{2}}.
This makes the transformation independent of the magnitude of 𝐰𝐰{\bf w}, similar to the weight modulation and demodulation [34] on the other layers.
We then interpret the first two components as rotation around the center of the canvas [0,1]2superscript012[0,1]^{2}, with the rotation angle α𝛼\alpha defined by rc′=cos⁡αsuperscriptsubscript𝑟𝑐′𝛼r_{c}^{\prime}=\cos\alpha and rs′=sin⁡αsuperscriptsubscript𝑟𝑠′𝛼r_{s}^{\prime}=\sin\alpha.
Finally, we interpret the remaining two components as translation by (tx′,ty′)superscriptsubscript𝑡𝑥′superscriptsubscript𝑡𝑦′(t_{x}^{\prime},t_{y}^{\prime}) units, so that the translation is performed after the rotation.
In practice, we implement the resulting geometric transformation by modifying the phases and two-dimensional frequencies of the Fourier features, which is equivalent to applying the same transformation to the continuous representation of z0subscript𝑧0z_{0} analytically.

#### Flexible layer specifications

In configs t and r, we define the per-layer filter parameters (Figure 4c) as follows.
The cutoff frequency fcsubscript𝑓𝑐f_{c} and the minimum acceptable stopband frequency ftsubscript𝑓𝑡f_{t} obey geometric progression until the first critically sampled layer:

fc​[i]subscript𝑓𝑐delimited-[]𝑖\displaystyle f_{c}[i]
=fc,0⋅(fc,N/fc,0)min⁡(i/(N−Ncrit),1),absent⋅subscript𝑓𝑐0superscriptsubscript𝑓𝑐𝑁subscript𝑓𝑐0𝑖𝑁subscript𝑁crit1\displaystyle=f_{c,0}\cdot(f_{c,N}/f_{c,0})^{\min(i/(N-N_{\text{crit}}),1)},
ft​[i]subscript𝑓𝑡delimited-[]𝑖\displaystyle f_{t}[i]
=ft,0⋅(ft,N/ft,0)min⁡(i/(N−Ncrit),1),absent⋅subscript𝑓𝑡0superscriptsubscript𝑓𝑡𝑁subscript𝑓𝑡0𝑖𝑁subscript𝑁crit1\displaystyle=f_{t,0}\cdot(f_{t,N}/f_{t,0})^{\min(i/(N-N_{\text{crit}}),1)},

(35)

where N=14𝑁14N=14 is the total number of layers, Ncrit=2subscript𝑁crit2N_{\text{crit}}=2 is the number of critically sampled layers at the end, fc,0=2subscript𝑓𝑐02f_{c,0}=2 corresponds to the frequency content of the input Fourier features, and fc,N=sN/2subscript𝑓𝑐𝑁subscript𝑠𝑁2f_{c,N}=s_{N}/2 is defined by the output resolution.
ft,0subscript𝑓𝑡0f_{t,0} and ft,Nsubscript𝑓𝑡𝑁f_{t,N} are free parameters; we use ft,0=22.1subscript𝑓𝑡0superscript22.1f_{t,0}=2^{2.1} and ft,N=fc,N⋅20.3subscript𝑓𝑡𝑁⋅subscript𝑓𝑐𝑁superscript20.3f_{t,N}=f_{c,N}\cdot 2^{0.3} in most of our tests.
Given the values of fc​[i]subscript𝑓𝑐delimited-[]𝑖f_{c}[i] and ft​[i]subscript𝑓𝑡delimited-[]𝑖f_{t}[i], the sampling rate s​[i]𝑠delimited-[]𝑖s[i] and transition band half-width fh​[i]subscript𝑓ℎdelimited-[]𝑖f_{h}[i] are then determined by

s​[i]𝑠delimited-[]𝑖\displaystyle s[i]
=exp2⁡⌈log2⁡(min⁡(2⋅ft​[i],sN))⌉,absentsubscript2subscript2⋅2subscript𝑓𝑡delimited-[]𝑖subscript𝑠𝑁\displaystyle=\exp_{2}\left\lceil\log_{2}\big{(}\min(2\cdot f_{t}[i],s_{N})\big{)}\right\rceil,
fh​[i]subscript𝑓ℎdelimited-[]𝑖\displaystyle f_{h}[i]
=max⁡(ft​[i],s​[i]/2)−fc​[i].absentsubscript𝑓𝑡delimited-[]𝑖𝑠delimited-[]𝑖2subscript𝑓𝑐delimited-[]𝑖\displaystyle=\max(f_{t}[i],s[i]/2)-f_{c}[i].

(36)

The sampling rate is rounded up to the nearest power of two that satisfies s​[i]≥2​ft​[i]𝑠delimited-[]𝑖2subscript𝑓𝑡delimited-[]𝑖s[i]\geq 2f_{t}[i], but it is not allowed to exceed the output resolution.
The transition band half-width is selected to satisfy either fc​[i]+fh​[i]=ft​[i]subscript𝑓𝑐delimited-[]𝑖subscript𝑓ℎdelimited-[]𝑖subscript𝑓𝑡delimited-[]𝑖f_{c}[i]+f_{h}[i]=f_{t}[i] or fc​[i]+fh​[i]=s​[i]/2subscript𝑓𝑐delimited-[]𝑖subscript𝑓ℎdelimited-[]𝑖𝑠delimited-[]𝑖2f_{c}[i]+f_{h}[i]=s[i]/2, whichever yields a higher value.

We consider fc​[i]subscript𝑓𝑐delimited-[]𝑖f_{c}[i] to represent the output frequency content of layer i𝑖i, for i∈{0,1,…,N−1}𝑖01…𝑁1i\in\{0,1,\ldots,N-1\}, whereas the input is represented by fc​[max⁡(i−1,0)]subscript𝑓𝑐delimited-[]𝑖10f_{c}[\max(i-1,0)].
Thus, we construct the corresponding upsampling filter according to fc​[max⁡(i−1,0)]subscript𝑓𝑐delimited-[]𝑖10f_{c}[\max(i-1,0)] and fh​[max⁡(i−1,0)]subscript𝑓ℎdelimited-[]𝑖10f_{h}[\max(i-1,0)] and the downsampling filter according to fc​[i]subscript𝑓𝑐delimited-[]𝑖f_{c}[i] and fh​[i]subscript𝑓ℎdelimited-[]𝑖f_{h}[i].
The nonlinearity is evaluated at a temporary sampling rate s′=max⁡(s​[i],s​[max⁡(i−1,0)])⋅msuperscript𝑠′⋅𝑠delimited-[]𝑖𝑠delimited-[]𝑖10𝑚s^{\prime}=\max(s[i],s[\max(i-1,0)])\cdot m, where m𝑚m is the upsampling parameter discussed in Section 3.2 that we set to 2 in most of our tests.

### F.2 Hyperparameters and training configurations

Parameter
Datasets (Figure 5, left)
Ablations at 256×\times256

Config

b

t

r

a–c

d–t

r

Batch size

32

32

32

64

64

64

Moving average

10k

10k

10k

20k

20k

20k

Mapping net depth

8

2

2

8

2

2

Minibatch stddev

4

4

4

8

4

4

G layers

15/17

14

14

13

14

14

G capacity: Cbasesubscript𝐶baseC_{\text{base}}

215superscript2152^{15}

215superscript2152^{15}

216superscript2162^{16}

214superscript2142^{14}

214superscript2142^{14}

215superscript2152^{15}

G capacity: Cmaxsubscript𝐶maxC_{\text{max}}

512

512

1024

512

512

1024

G learning rate

0.0020

0.0025

0.0025

0.0025

0.0025

0.0025

D learning rate

0.0020

0.0020

0.0020

0.0025

0.0025

0.0025

R1 regularization γ𝛾\gamma

b

t

r

FFHQ-U

02562

01.0

01.0

01.0

FFHQ-U

10242

10.0

32.8

32.8

FFHQ

10242

10.0

32.8

32.8

MetFaces-U

10242

10.0

16.4

06.6

MetFaces

10242

05.0

06.6

03.3

AFHQv2

05122

05.0

08.2

16.4

Beaches

05122

02.0

04.1

12.3

We used 8 GPUs for all our training runs and continued the training until the discriminator had seen a total of 25M real images when training from scratch, or 5M images when using transfer learning.
Figure 16 shows the hyperparameters used in each experiment.
We performed the baseline runs (configs a–c) using the corresponding standard configurations:
StyleGAN2 config F [34] for the high-resolution datasets in Figure 5, left,
and ADA 256×\times256 baseline config [32] for the ablations in Figure 3 and Figure 5, right.

Many of our hyperparameters, including discriminator capacity and learning rate, batch size, and generator moving average decay, are inherited directly from the baseline configurations, and kept unchanged in all experiments.
In configs c and d, we disable noise inputs [33], path length regularization [34], and mixing regularization [33].
In config d, we also decrease the mapping network depth to 2 and set the minibatch standard deviation group size to 4 as recommended in the StyleGAN2-ADA documentation.
The introduction of explicit normalization in config d allows us to use the same generator learning rate, 0.0025, for all output resolutions.
In Figure 5, right, we show results for path length regularization with weight 0.5 and mixing regularization with probability 0.5.

#### Augmentation

Since our datasets are horizontally symmetric in nature, we enable dataset x𝑥x-flip augmentation in all our experiments.
To prevent the discriminator from overfitting, we enable adaptive discriminator augmentation (ADA) [32] with default settings for MetFaces, MetFaces-U, AFHQv2, and Beaches, but disable it for FFHQ and FFHQ-U.
Furthermore, we train MetFaces and MetFaces-U using transfer learning from the corresponding FFHQ or FFHQ-U snapshot with the lowest FID, similar to Karras et al. [32], but start the training from scratch in all other experiments.

#### Generator capacity

StyleGAN2 defines the number of feature maps on a given layer to be inversely proportional to its resolution, i.e., C​[i]=C​(s​[i])=min⁡(round⁡(Cbase/s​[i]),Cmax)𝐶delimited-[]𝑖𝐶𝑠delimited-[]𝑖roundsubscript𝐶base𝑠delimited-[]𝑖subscript𝐶maxC[i]=C(s[i])=\min(\operatorname{round}(C_{\text{base}}/s[i]),C_{\text{max}}), where s​[i]𝑠delimited-[]𝑖s[i] is the output resolution of layer i𝑖i.
Parameters Cbasesubscript𝐶baseC_{\text{base}} and Cmaxsubscript𝐶maxC_{\text{max}} control the overall capacity of the generator; our baseline configurations use Cmax=512subscript𝐶max512C_{\text{max}}=512 and Cbase=214subscript𝐶basesuperscript214C_{\text{base}}=2^{14} or 215superscript2152^{15} depending on the output resolution.
Since StyleGAN2 can be considered to employ critical sampling on all layers, i.e., fc​[i]=s​[i]/2subscript𝑓𝑐delimited-[]𝑖𝑠delimited-[]𝑖2f_{c}[i]=s[i]/2, we can equally well define the number of feature maps as C​[i]=C​(2​fc​[i])𝐶delimited-[]𝑖𝐶2subscript𝑓𝑐delimited-[]𝑖C[i]=C(2f_{c}[i]).
These two definitions are equivalent for configs a–f, but in configs g–r we explicitly set fc​[i]≤s​[i]/2subscript𝑓𝑐delimited-[]𝑖𝑠delimited-[]𝑖2f_{c}[i]\leq s[i]/2, which necessitates using the latter definition.
In config r, we double the value of both Cbasesubscript𝐶baseC_{\text{base}} and Cmaxsubscript𝐶maxC_{\text{max}} to compensate for the reduced capacity of the 1×\times1 convolutions.
In Figure 5, right, we sweep the capacity by multiplying both parameters by 0.5, 1.0, and 2.0.

#### R1 regularization

The optimal choice for the R1subscript𝑅1R_{1} regularization weight γ𝛾\gamma is highly dependent on the dataset, necessitating a grid search [34, 32].
For the baseline config b, we tested γ∈{1,2,5,10,20}𝛾1251020\gamma\in\{1,2,5,10,20\} and selected the value that gave the best FID for each dataset.
For our configs t and r, we followed the recommendation of Karras et al. [32] to define γ=γ0⋅N/M𝛾⋅subscript𝛾0𝑁𝑀\gamma=\gamma_{0}\cdot N/M, where N=sN2𝑁superscriptsubscript𝑠𝑁2N=s_{N}^{2} is the number of output pixels and M𝑀M is the batch size, and performed a grid search over γ0∈{0.0002,0.0005,0.0010,0.0020,0.0050}subscript𝛾00.00020.00050.00100.00200.0050\gamma_{0}\in\{0.0002,0.0005,0.0010,0.0020,0.0050\}.
For the low-resolution ablations, we chose to use a fixed value γ=1𝛾1\gamma=1 for simplicity.
The resulting values of γ𝛾\gamma are shown in Figure 16, right.

#### Training of config r

In this configuration, we blur all images the discriminator sees in the beginning of the training. This Gaussian blur is executed just before the ADA augmentation. We start with σ=10𝜎10\sigma=10 pixels, which we ramp to zero over the first 200k images. This prevents the discriminator from focusing too heavily on high frequencies early on. It seems that in this configuration the generator sometimes learns to produce high frequencies with a small delay, allowing the discriminator to trivially tell training data from the generated images without providing useful feedback to the generator. As such, config r is prone to random training failures in the beginning of the training without this trick. The other configurations do not have this issue.

### F.3 G-CNN comparison

In Figure 5, bottom, we compare our config r with config t extended with p​4𝑝4p4-symmetric group convolutions [16, 17].
p​4𝑝4p4 symmetry makes the generator equivariant to 0∘, 90∘, 180∘, and 270∘ rotations, but not to arbitrary rotation angles.
In practice, we implement the group convolutions by extending all intermediate activation tensors in the synthesis network with an additional group dimension of size 4 and introducing appropriate redundancy in the convolution weights.
We keep the input layer unchanged and introduce the group dimension by replicating each element of z0subscript𝑧0z_{0} four times.
Similarly, we eliminate the group dimension after the last layer by computing an average of the four elements.
p​4𝑝4p4-symmetric group convolutions have 4×\times as many trainable parameters as the corresponding regular convolutions.
To enable an apples-to-apples comparison, we compensate for this increase by halving the values of Cbasesubscript𝐶baseC_{\text{base}} and Cmaxsubscript𝐶maxC_{\text{max}}, which brings the number of parameters back to the original level.

## Appendix G Energy consumption

Item

Number of

GPU years

Electricity

training runs

(Volta)

(MWh)

Early exploration

0233

18.02

042.45

Project exploration

1207

48.93

118.13

Setting up ablations

0297

13.30

032.48

Per-dataset tuning

0063

04.54

013.28

Producing results in the paper

0053

05.26

014.35

StyleGAN3-R at 1024×\times1024

0001

00.30

000.87

Other runs in the dataset table

0017

02.35

006.88

Ablation tables

0035

02.61

006.60

Results intentionally left out

0023

01.72

003.93

Total

1876

91.77

224.62

Computation is an essential resource in machine learning projects: its availability and cost, as well as the associated energy consumption, are key factors in both choosing research directions and practical adoption. We provide a detailed breakdown for our entire project in Table 17 in terms of both GPU time and electricity consumption.
We report expended computational effort as single-GPU years (Volta class GPU). We used a varying number of NVIDIA DGX-1s for different stages of the project, and converted each run to single-GPU equivalents by simply scaling by the number of GPUs used.

We followed the Green500 power measurements guidelines [20].
The entire project consumed approximately 225 megawatt hours (MWh) of electricity.
Approximately 70% of it was used for exploratory runs, where we gradually built the new configurations; first in an unstructured manner and then specifically ironing out the new StyleGAN3-T and StyleGAN3-R configurations.
Setting up the intermediate configurations between StyleGAN2 and our generators, as well as, the key parameter ablations was also quite expensive at ∼similar-to\sim15%.
Training a single instance of StyleGAN3-R at 1024×\times1024 is only slightly more expensive (0.9MWh) than training StyleGAN2 (0.7MWh) [34].

Generated on Sat Mar 2 04:43:09 2024 by LaTeXML
