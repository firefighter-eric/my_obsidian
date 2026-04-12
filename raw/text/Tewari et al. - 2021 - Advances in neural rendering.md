# Tewari et al. - 2021 - Advances in neural rendering

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Tewari et al. - 2021 - Advances in neural rendering.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2111.05849
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\BibtexOrBiblatex\electronicVersion\PrintedOrElectronic

\teaser

This state-of-the-art report discusses a large variety of neural rendering methods which enable applications such as novel-view synthesis of static and dynamic scenes, generative modeling of objects, and scene relighting.
See Section 4 for more details on the various methods. Images adapted from [MST∗20, TY20, CMK∗21, ZSD∗21, BBJ∗21, LSS∗21, PSB∗21, JXX∗21, PDW∗21] ©2021 IEEE.

# Advances in Neural Rendering

A. Tewari1,6⋆ J. Thies2⋆ B. Mildenhall3⋆ P. Srinivasan3⋆ E. Tretschk1 W. Yifan4,8 C. Lassner5 V. Sitzmann6 R. Martin-Brualla3
S. Lombardi5 T. Simon5 C. Theobalt1 M. Nießner7 J. T. Barron3 G. Wetzstein8 M. Zollhöfer5 V. Golyanik1

1MPI for Informatics 2MPI for Intelligent Systems 3Google Research 4ETH Zürich 5Reality Labs Research
6MIT 7Technical University of Munich 8Stanford University ⋆Equal contribution.

###### Abstract

Synthesizing photo-realistic images and videos is at the heart of computer graphics and has been the focus of decades of research.
Traditionally, synthetic images of a scene are generated using rendering algorithms such as rasterization or ray tracing, which take specifically defined representations of geometry and material properties as input.
Collectively, these inputs define the actual scene and what is rendered, and are referred to as the scene representation (where a scene consists of one or more objects).
Example scene representations are triangle meshes with accompanied textures (e.g., created by an artist), point clouds (e.g., from a depth sensor), volumetric grids (e.g., from a CT scan), or implicit surface functions (e.g., truncated signed distance fields).
The reconstruction of such a scene representation from observations using differentiable rendering losses is known as inverse graphics or inverse rendering.
Neural rendering is closely related, and combines ideas from classical computer graphics and machine learning to create algorithms for synthesizing images from real-world observations.
Neural rendering is a leap forward towards the goal of synthesizing photo-realistic image and video content.
In recent years, we have seen immense progress in this field through hundreds of publications that show different ways to inject learnable components into the rendering pipeline.
This state-of-the-art report on advances in neural rendering focuses on methods that combine classical rendering principles with learned 3D scene representations, often now referred to as neural scene representations.
A key advantage of these methods is that they are 3D-consistent by design, enabling applications such as novel viewpoint synthesis of a captured scene.
In addition to methods that handle static scenes, we cover neural scene representations for modeling non-rigidly deforming objects and scene editing and composition.
While most of these approaches are scene-specific, we also discuss techniques that generalize across object classes and can be used for generative tasks.
In addition to reviewing these state-of-the-art methods, we provide an overview of fundamental concepts and definitions used in the current literature.
We conclude with a discussion on open challenges and social implications.

## 1 Introduction

Synthesis of controllable and photo-realistic images and videos is one of the fundamental goals of computer graphics.
During the last decades, methods and representations have been developed to mimic the image formation model of real cameras, including the handling of complex materials and global illumination.
These methods are based on the laws of physics and simulate the light transport from light sources to the virtual camera for synthesis.
To this end, all physical parameters of the scene have to be known for the rendering process.
These parameters, for example, contain information about the scene geometry and material properties such as reflectivity or opacity.
Given this information, modern ray tracing techniques can generate photo-real imagery.
Besides the physics-based rendering methods, there is a variety of techniques that approximate the real-world image formation model.
These methods are based on mathematical approximations (e.g., a piece-wise linear approximation of the surface; i.e., triangular meshes) and heuristics (e.g., Phong shading) to improve the applicability (e.g., for real-time applications).
While these methods require fewer parameters to represent a scene, the achieved realism is also reduced.

While traditional computer graphics allows us to generate high-quality controllable imagery of a scene, all physical parameters of the scene, for example, camera parameters, illumination and materials of the objects need to be provided as inputs.
If we want to generate controllable imagery of a real-world scene, we would need to estimate these physical properties from existing observations such as images and videos.
This estimation task is referred to as inverse rendering and is extremely challenging, especially when the goal is photo-realistic synthesis.
In contrast, neural rendering is a rapidly emerging field which allows the compact representation of scenes, and rendering can be learned from existing observations by utilizing neural networks (see Advances in Neural Rendering).
The main idea of neural rendering is to combine insights from classical (physics-based) computer graphics and recent advances in deep learning.
Similar to classical computer graphics, the goal of neural rendering is to generate photo-realistic imagery in a controllable way (c.f. definition of neural rendering in [TFT∗20]).
This, for example, includes novel viewpoint synthesis, relighting, deformation of the scene, and compositing.

(a) 2D Neural Rendering, also known as neural refinement, neural re-rendering, or deferred neural rendering is based on 2D inputs that are generated for example using a classical renderer and learns to render a scene in 2D.

(b) 3D Neural Rendering learns to represent a scene in 3D and uses fixed differentiable rendering schemes from computer graphics which are motivated by physics.

Early neural rendering approaches (covered in [TFT∗20]) used neural networks to convert scene parameters into the output images.
The scene parameters are either directly given as one-dimensional inputs, or a classical computer graphics pipeline is used to generate two-dimensional inputs.
The deep neural networks are trained on observations of real-world scenes and learn to model as well as render these scenes.
A deep neural network can be seen as a universal function approximator.
Specifically, a network defines a family of functions based on its input arguments, model architecture, and trainable parameters.
Stochastic gradient descent is employed to find the function from this space that best explains the training set as measured by the training loss.
From this viewpoint, neural rendering aims to find the mapping 𝐈=ℳ​(𝐜)𝐈ℳ𝐜\mathbf{I}=\mathcal{M}(\mathbf{c}) between control parameters 𝐜∈ℝdi​n𝐜superscriptℝsubscript𝑑𝑖𝑛\mathbf{c}\in\mathbb{R}^{d_{in}} and the corresponding output image 𝐈∈ℝH×W×3𝐈superscriptℝ𝐻𝑊3\mathbf{I}\in\mathbb{R}^{H\times W\times 3}, with H𝐻H and W𝑊W being image height and width.
This can be interpreted as a complex and challenging sparse data interpolation problem.
Thus, neural rendering, similar to classical function fitting, has to navigate the trade-off between under- and over-fitting, i.e., representing the training set well vs. generalization to unobserved inputs.
If the representational power of the network is insufficient, the quality of the resulting images will be low, e.g., results are often blurry.
On the other hand, if the representational power is too large, the network overfits to the training set and does not generalize to unseen inputs at test time.
Finding the right network architecture is an art in itself.
In the context of neural rendering, designing the right physically motivated inductive biases often requires a strong graphics background.
These physically motivated inductive biases act as regularizers and ensure that the found function is close to how 3D space and/or image formation works in our real world, thus leading to better generalization at test time.
Inductive biases can be added to the network in different ways.
For example, in terms of the employed layers, at what point in the network and in which form inputs are provided, or even via the integration of non-trainable (but differentiable) components from classical computer graphics.
One great example for this are recent neural rendering techniques that try to disentangle the modeling and rendering processes by only learning the 3D scene representation and relying on a rendering function from computer graphics for supervision.
For example, Neural Radiance Fields (NeRF) [MST∗20] uses a multi-layer perceptron (MLP) to approximate the radiance and density field of a 3D scene.
This learned volumetric representation can be rendered from any virtual camera using analytic differentiable rendering (i.e., volumetric integration).
For training, observations of the scene from several camera viewpoints are assumed.
The network is trained on these observations by rendering the estimated 3D scene from these training viewpoints, and minimizing the difference between the rendered and observed images.
Once trained, the 3D scene approximated by the neural network can be rendered from a novel viewpoint, enabling controllable synthesis.
In contrast to approaches that use the neural network to learn the rendering function as well [TFT∗20], NeRF uses knowledge from computer graphics more explicitly in the method, enabling better generalization to novel views due to the (physical) inductive bias: an intermediate 3D-structured representation of the density and radiance of the scene.
As a result, NeRF learns physically meaningful color and density values in 3D space, which physics-inspired ray casting and volume integration can then render consistently into novel views.
The achieved quality, as well as the simplicity of the method, led to an ‘explosion’ of developments in the field.
Several advances have been made which improve the applicability, enable controllability, the capture of dynamically changing scenes as well as the training and inference times.
Within this report, we cover these recent advances in the field.
To foster a deep understanding of these methods, we discuss the fundamentals of neural rendering by describing the different components and design choices in detail in Section 3.
Specifically, we clarify the definition of the different scene representations used in the current literature (surfaces and volumetric approaches), and describe ways to approximate them using deep neural networks.
We also present the fundamental rendering functions from computer graphics that are used to train these representations.
Since neural rendering is a very fast evolving field, with significant progress along many different dimensions, we develop a taxonomy of the recent approaches w.r.t. their application field to provide a concise overview of the developments.
Based on this taxonomy and the different application areas, we present the state-of-the-art methods in Section 4.
The report is concluded with Section 5 discussing the open challenges and Section 6 discussing social implications of photo-realistic synthetic media.

## 2 Scope of This STAR

In this state-of-the-art report, we focus on advanced neural rendering approaches that combine classical rendering with learnable 3D representations (see Figure 1).
The underlying neural 3D representations are 3D-consistent by design and enable control over different scene parameters.
Within this report, we give a comprehensive overview of the different scene representations and detail the fundamentals of the components that are lent from classical rendering pipelines as well as machine learning.
We further focus on approaches that use Neural Radiance Fields [MST∗20] and volumetric rendering.
However, we do not focus on neural rendering methods that reason mostly in 2D screen space; we refer to [TFT∗20] for a discussion on such approaches.
We also do not cover neural super-sampling and denoising methods for ray-traced imagery [CKS∗17, KBS15].

Selection scheme:
As written above, the report concentrates mainly on the use of neural radiance fields and its derivatives.
It covers papers published in the proceedings of the major computer vision and computer graphics conferences (2020, 2021), as well as preprints published via arXiv.
The papers are selected by the authors of this report to fit the context of the survey and to give the reader an overview over a broad spectrum of different techniques.
Note that this report is intended to give a panoramic view of the field, to list literature to get up to speed with a specific research domain, and to standardize the notations and definitions.
This report covers a wide range of current literature. The authors do not claim completeness of the report and highly recommend the reader to study the cited works for in-depth details.

## 3 Fundamentals of Neural Rendering

Neural rendering, and especially 3D neural rendering is based on classical concepts of computer graphics (see Figure 1).
A neural rendering pipeline learns to render and/or represent a scene from real-world imagery, which can be an unordered set of images, or structured, multi-view images or videos.
It does so by mimicking the physical process of a camera that captures a scene.
A key property of 3D neural rendering is the disentanglement of the camera capturing process (i.e., the projection and image formation) and the 3D scene representation during this training.
This disentanglement has several advantages and leads especially to a high level of 3D consistency during the synthesis of images (e.g., for novel viewpoint synthesis).
To disentangle the projection and other physical processes from the 3D scene representation, 3D neural rendering methods rely on known image formation models from computer graphics (e.g., rasterization, point splatting, or volumetric integration).
These models are motivated by physics, especially the interaction of the light of an emitter with the scene as well as the camera itself.
This light transport is formulated using the rendering equation [Kaj86].

The computer graphics field offers a variety of approximations to this rendering equation.
These approximations are dependent on the used scene representation and range from classical rasterization to path tracing and volumetric integration.
3D neural rendering exploits these rendering methods.
In the following, we will detail the scene representations (Section 3.1) as well as the rendering methods (Section 3.2) used in common neural rendering methods.
Note that both the scene representation as well as the rendering method itself have to be differentiable in order to learn from real images (Section 3.3).

### 3.1 Scene Representations

For decades, the computer graphics community has explored various primitives, including point clouds, implicit and parametric surfaces, meshes, and volumes (see Figure 2).
While these representations have clear definitions in the computer graphics field, there is often a confusion in the current literature of neural rendering, especially when it is about implicit and explicit surface representations and volumetric representations.
In general, volumetric representations can represent surfaces, but not vice versa.
Volumetric representations store volumetric properties such as densities, opacities or occupancies, but they can also store multidimensional features such as colors or radiance.
In contrast to volumetric representations, surface representations store properties w.r.t. the surface of an object.
They cannot be used to model volumetric matter, such as smoke (unless it is a coarse approximation).
For both surface and volumetric representations, there are continuous and discretized counterparts (see Figure 2).
The continuous representations are particularly interesting for neural rendering approaches since they can provide analytic gradients.
For surface representations, there are two different ways to represent the surface – explicitly or implicitly.
The surface using an explicit surface function f𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡(.)∈ℝf_{\mathit{explicit}}(.)\in\mathbb{R} in the Euclidean space is defined as:

S𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡={(xyf𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡​(x,y))|(xy)∈ℝ2}.subscript𝑆𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡conditional-set𝑥𝑦subscript𝑓𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡𝑥𝑦𝑥𝑦superscriptℝ2S_{\mathit{explicit}}=\left\{\left(\begin{array}[]{c}x\\
y\\
f_{\mathit{explicit}}(x,y)\\
\end{array}\right)~{}\middle|~{}\left(\begin{array}[]{c}x\\
y\\
\end{array}\right)\in\mathbb{R}^{2}\right\}.

(1)

Note that an explicit surface can also be represented as a parametric function f𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐(.)∈ℝ3f_{\mathit{parametric}}(.)\in\mathbb{R}^{3}, which generalizes S𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡subscript𝑆𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡S_{\mathit{explicit}}:

S𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡∗={f𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐​(u,v)|(uv)∈ℝ2}.superscriptsubscript𝑆𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡conditional-setsubscript𝑓𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐𝑢𝑣𝑢𝑣superscriptℝ2S_{\mathit{explicit}}^{*}=\left\{f_{\mathit{parametric}}(u,v)\\
~{}\middle|~{}\left(\begin{array}[]{c}u\\
v\\
\end{array}\right)\in\mathbb{R}^{2}\right\}.

(2)

The surface using an implicit surface function f𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡​(⋅)∈ℝsubscript𝑓𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡⋅ℝf_{\mathit{implicit}}(\cdot)\in\mathbb{R} is defined as the zero-level set:

S𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡={(xyz)∈ℝ3|f𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡​(x,y,z)=0}.subscript𝑆𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡conditional-set𝑥𝑦𝑧superscriptℝ3subscript𝑓𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡𝑥𝑦𝑧0S_{\mathit{implicit}}=\left\{\left(\begin{array}[]{c}x\\
y\\
z\\
\end{array}\right)\in\mathbb{R}^{3}~{}\middle|~{}f_{\mathit{implicit}}(x,y,z)=0\right\}.

(3)

Whereas a volume representation defines properties in the entire space:

V={f𝑣𝑜𝑙​(x,y,z)|(xyz)∈ℝ3}.𝑉conditional-setsubscript𝑓𝑣𝑜𝑙𝑥𝑦𝑧𝑥𝑦𝑧superscriptℝ3V=\left\{f_{\mathit{vol}}(x,y,z)~{}\middle|~{}\left(\begin{array}[]{c}x\\
y\\
z\\
\end{array}\right)\in\mathbb{R}^{3}\right\}.

(4)

Note that the respective function domain can be restricted for all these representations.

In general, for all three scene representations, the underlying function can be any function that is capable to approximate the respective content.
For simple surfaces like a plane, the functions fi​m​p​l​i​c​i​tsubscript𝑓𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡f_{implicit}, fe​x​p​l​i​c​i​tsubscript𝑓𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡f_{explicit} can be linear functions.
To handle more complex surfaces or volumes, polynomials (for example from a Taylor series) or multivariate Gaussians can be used.
To increase the expressiveness further, these functions can be spatially localized and then combined into a mixture, for example multiple Gaussians can form a Gaussian mixture.
Radial basis function networks are such mixture models and can be used as an approximator for both, implicit surface and volume functions [CBC∗01a].
Note that these radial basis function networks can be interpreted as a single layer of a neural network.

Since neural networks and, especially, multi-layer perceptrons (MLPs) are universal function approximators, they can be used to ‘learn’ the underlying functions (f𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡subscript𝑓𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡f_{\mathit{implicit}}, f𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡subscript𝑓𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡f_{\mathit{explicit}}, f𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐subscript𝑓𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐f_{\mathit{parametric}}, or f𝑣𝑜𝑙subscript𝑓𝑣𝑜𝑙f_{\mathit{vol}}).
(Similar to a Gaussian mixture, multiple localized, weaker MLPs can be combined into a mixture as well, e.g., [RPLG21].)
In the context of neural rendering, a scene representation that is using a neural network to approximate the surface or volumetric representation function is called neural scene representation.
Note that both surface and volumetric representations can be extended to store additional information, like color or view-dependent radiance.

In the following, we will discuss the different MLP-based function approximators that build the foundation of the recent neural surface and volumetric representations.

#### 3.1.1 MLPs as Universal Function Approximators

Multi-Layer Perceptrons (MLPs) are known to act as Universal Function Approximators [HSW89]. Specifically, we use MLPs to represent surface or volumetric properties.
A multi-layer perceptron is a conventional fully-connected neural network.
In the context of scene repesentations, the MLP takes as input a coordinate in space, and produces as output some value corresponding to that coordinate.
This type of network is also known as coordinate-based neural network (and the resulting representation is called coordinate-based scene representation).
Note that the input coordinate space can be aligned with the Euclidean space, but it can also be embedded for example in the uv-space of a mesh (resulting in a neural parametric surface).

A key finding to use ReLU-based MLPs for neural representation and rendering tasks is the usage of positional encoding.
Inspired by the positional encoding used in natural language processing (e.g., in Transformers [VSP∗17]), the input coordinates are positionally encoded using a set of basis functions.
These basis functions can be fixed [MST∗20] or they can be learned [TSM∗20].
These spatial embeddings simplify the task of the MLP to learn the mapping from a location to a specific value, since through the spatial embedding, the input space is partitioned.
As an example, the positional encoding used in NeRF [MST∗20] is defined as:

𝐱𝐱\displaystyle\mathbf{x}
↦[cos⁡(𝐌𝐱),sin⁡(𝐌𝐱)]maps-toabsent𝐌𝐱𝐌𝐱\displaystyle\mapsto[\cos(\mathbf{M}\mathbf{x}),\sin(\mathbf{M}\mathbf{x})]

(5)

where ​𝐌where 𝐌\displaystyle\textrm{where }\mathbf{M}
=[𝐈2​𝐈22​𝐈…2p−1​𝐈]⊤.absentsuperscriptmatrix𝐈2𝐈superscript22𝐈…superscript2𝑝1𝐈top\displaystyle=\begin{bmatrix}\mathbf{I}&2\mathbf{I}&2^{2}\mathbf{I}&\ldots&2^{p-1}\mathbf{I}\end{bmatrix}^{\top}.

(6)

Here, 𝐱𝐱\mathbf{x} is the input coordinate and p𝑝p is a hyperparameter controlling the frequencies used (dependent on the target signal resolution).
This “soft” binary encoding of the input coordinates makes it easier for the network to access higher frequencies of the input.

As mentioned above, MLP-based function approximators can be used to represent a surface or volume (i.e., f𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡subscript𝑓𝑖𝑚𝑝𝑙𝑖𝑐𝑖𝑡f_{\mathit{implicit}}, f𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡subscript𝑓𝑒𝑥𝑝𝑙𝑖𝑐𝑖𝑡f_{\mathit{explicit}}, f𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐subscript𝑓𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑟𝑖𝑐f_{\mathit{parametric}}, or f𝑣𝑜𝑙subscript𝑓𝑣𝑜𝑙f_{\mathit{vol}}), but they can also be used to store other attributes like color.
For instance, there are hybrid representations composed of classical surface representations like point clouds or meshes with an MLP to store the surface appearance (e.g., texture field [OMN∗19]).

MLP-based function approximators can employ different activation functions such as ReLU [MST∗20], sine activations [SMB∗20] or Gaussians [RL21].

#### 3.1.2 Representing Surfaces

##### Point Clouds.

A point cloud is a set of elements of the Euclidean space.
A continuous surface can be discretized by a point cloud - each element of the point cloud represents a sample point (x,y,z)𝑥𝑦𝑧(x,y,z) on the surface.
For each point, additional attributes can be stored such as normals or colors.
A point cloud that features normals is also referred to as oriented point cloud.
Besides simple points that can be seen as infinitesimally small surface patches, oriented point clouds with a radius can be used (representing a 2D disk that lies on the tangent plane of the underlying surface).
This representation is called surface elements, alias surfels [PZvBG00].
They are often used in computer graphics to render point clouds or particles from simulations.
The rendering of such surfels is called splatting, and recent work shows that it is differentiable [YSW∗19a].
Using such a differentiable rendering pipeline, it is possible to directly back-propagate to the point cloud locations as well as the accompanied features (e.g., radius or color).
In Neural Point-based Graphics [ASK∗20a] and SynSin [WGSJ20], learnable features are attached to the points that can store rich information about the appearance and shape of the actual surface.
In ADOP [RFS21a] these learnable features are interpreted by an MLP which can account for view-dependent effects.
Note that instead of storing explicitly features for specific points, one can also use an MLP to predict the features for the discrete positions.

As mentioned above, a point cloud is a set of elements of the Euclidean space, thus, besides surfaces, they can also represent volumes (e.g., storing additional opacity or density values). Using a radius for each point naturally leads to a full sphere-based formulation [LZ21].

##### Meshes.

Polygonal meshes represent a piece-wise linear approximation of a surface.
Especially, triangle and quad meshes are used in computer graphics as de facto standard representation for surfaces.
The graphics pipeline and graphic accelerators (GPUs) are optimized to process and rasterize billions of triangles per second.
The majority of graphics editing tools work with triangle meshes which makes this representation important for any content creation pipeline.
To be directly compatible with these pipelines, many ’classical’ inverse graphics and neural rendering methods use this basic surface representation.
Using a differentiable renderer, the vertex positions as well as the vertex attributes (e.g., colors) can be optimized for to reproduce an image.
Neural networks can be trained to predict the vertex locations, e.g., to predict dynamically changing surfaces [BNT21].
Instead of using vertex attributes, a common strategy to store surface attributes within the triangles are texture maps.
2D texture coordinates are attached to the vertices of the mesh which reference a location in the texture image.
Using barycentric interpolation, texture coordinates can be computed for any point in a triangle and the attribute can be retrieved from the texture using bilinear interpolation.
The concept of textures is also integrated into the standard graphics pipeline, with additional features such as mip-mapping which is needed to properly handle the sampling of the texture (c.f., sampling theorem).
Deferred Neural Rendering [TZN19], uses textures that contain learnable view-dependent features, so-called neural textures.
Specifically, a coarse mesh is used as underlying 3D representation, to rasterize these neural textures. A neural network interprets these rasterized features in image space.
Note that the network can for example be a pixel-wise MLP, then the neural texture represents the surface radiance.

In contrast to using discrete textures, continuous textures can be used.
The authors of texture fields [OMN∗19] propose the usage of an MLP that predicts color values for each surface point.
In neural reflectance field textures (NeRF-Tex) [BGP∗21] the idea of NeRF [MST∗20] is combined with the idea of using a 2D neural texture and an underlying 3D mesh.
NeRF-Tex is conditioned on user-defined parameters that control the appearance, thus, being editable by artists.

##### Implicit Surfaces.

Implicit surfaces define the surface as the zero level-set of a function, see Eq. 3.
The most commonly used implicit surface representation is a signed distance function (SDF).
These SDF representations are used in numerous 3D scanning techniques that use volumetric fusion [CL96] to incrementally reconstruct the surface of a static [IKH∗11, NZIS13] or dynamic object [NFS15].
Implicit surface representations offer many advantages as they avoid the requirement of defining a mesh template, thus, being able to represent objects with unknown topology or changing topology in a dynamic scenario.
The volumetric fusion approaches mentioned above use a discretized (truncated) signed distance function, i.e., using a 3D grid containing signed distance values.
Hoppe et al. [HDD∗92] propose piece-wise linear functions to model the signed distance function w.r.t. input surface point samples.
The seminal work of Carr et al. [CBC∗01b] uses a radial basis function network instead.
This radial basis function network represent a continuous implicit surface function and can be seen as the first ’neural’ implicit surface representation.
Recent neural implicit surfaces representations are based on coordinate-based multi-layer perceptrons (MLPs), covered in Section 3.1.1.
Such representations have been gaining widespread popularity in neural scene representation and rendering.
They were proposed concurrently in [PFS∗19, CZ19] for shape modeling, where MLP architectures were used to map continuous coordinates to signed distance values. The fidelity of signals represented by such coordinate networks, or neural implicit representation, is primarily limited by the capacity of the network.
Thus, compared to other aforementioned representations, implicit surfaces offer potential advantages in memory efficiency and, as a continuous representation, they can theoretically represent geometries at infinite resolution.
The initial proposals was ensued, with broad enthusiasm, by a variety of improvements of different focuses, including improving the training schemes [XFYS20, DZW∗20, YAK∗20], leveraging global-local context [XWC∗19, EGO∗20], adopting specific parameterizations [GCV∗19, DGY∗20, CTZ20, KJJ∗21, YRSH21] or spatial partitions [GCS∗20, TTG∗20, CLI∗20, TLY∗21, MLL∗21b].
As there is no requirement of pre-defining the mesh template or the object topology, neural implicit surfaces are well suited for modeling objects of varying topologies [PFS∗19, CZ19].
Analytic gradients of the output with respect to the input coordinates can be computed using backpropagation.
This makes it possible to implement regularization terms on the gradients [GYH∗20], in addition to other geometrically motivated regularizers [GYH∗20, PFAK20, YAK∗20].
These respresentations can be extended to also encode the radiance of the scene [KJJ∗21, YTB∗21, SHN∗19].
This is useful for neural rendering, where we want the scene representation to encode both the geometry and appearance of the scenes.

#### 3.1.3 Representing Volumes

##### Voxel Grids.

As the pixel-equivalent in ℝ3superscriptℝ3\mathbb{R}^{3}, voxels are commonly used to represent volumes.
They can store the geometry occupancy, or store the density values for a scene with volumetric effects such as transparency. In addition, the appearance of the scene can be stored [GSHG98].
Using trilinear interpolation these volume attributes can be accessed at any point within the voxel grid.
T his interpolation is especially used for sample-based rendering methods like ray casting.
While the stored attributes can have a specific semantic meaning (e.g., occupancy), the attributes can also be learned.
Sitzmann et al. propose the use of DeepVoxels [STH∗19], where features are stored in a voxel grid.
The accumulation and interpretation of the features after the ray-casting rendering procedure is done using a deep neural network.
These DeepVoxels can be seen as volumetric neural textures, which can be directly optimized using backpropagation.
While dense voxel-based representations are fast to query, they are memory inefficient and 3D CNNs, potentially operating on these volumes, are computationally heavy.
Octree data structures [LK10] can be used to represent the volume in a sparse manner.
Sparse 3D convolution on octrees [WLG∗17, ROUG17] can help mitigate some problems, but these compact data structures cannot be easily updated on the fly.
Thus, they are difficult to integrate into learning frameworks.
Other approaches to mitigating the memory challenges of dense voxel grids include using object-specific shape templates [KTEM18], multi-plane [ZTF∗18, MSOC∗19, FBD∗19, TS20, WPYS21] or multi-sphere [BFO∗20, ALG∗20] images, which all aim at representing the voxel grid using a sparse approximation.

##### Neural Volumetric Representations.

Instead of storing features or other quantities of interest using a voxel grid, these quantities can also be defined using a neural network, similar to neural implicit surfaces (see Section 3.1.2).
MLP network architectures can be used to parameterize volumes, potentially in a more memory efficient manner than explicit voxel grids.
Still, these representations can be expensive to sample depending on the underlying network size because for each sample, an entire feedforward pass through the network has to be computed.
Most methods can be roughly classified as using global or local networks [GCV∗19, GCS∗20, CZ19, MPJ∗19, AL20, SHN∗19, SZW19, OMN∗19, GYH∗20, YKM∗20, DNJ20, SMB∗20, NMOG20, LGL∗20, JJHZ20, LZP∗20, KSW20].
Hybrid representations that use both grids and neural networks make a trade-off between computational and memory efficiency [PNM∗20, JSM∗20, CLI∗20, MLL∗21b].
Similar to neural implicit surfaces, neural volumetric representations allow for the computation of analytic gradients, which has been used to define regularization terms in [SMB∗20, TTG∗21, PSB∗21].
Band-limited coordinate-based networks have been introduced in BACON [LVVPW22] which learn a smooth multi-scale decomposition of the surface.

General remark:
The use of coordinate-based neural networks to model scenes volumetrically (as in NeRF) superficially resembles the use of coordinate networks to model surfaces implicitly (as in neural implicit surfaces). However, NeRF-like volumetric representations are not necessarily implicit — because the output of the network is density and color, the geometry of the scene is parameterized by the network explicitly, not implicitly. Despite this, it is common in the literature for these models to still be called “implicit”, perhaps in reference to the fact that the geometry of the scene is defined “implicitly” by the weights of a neural network (a different definition of “implicit” than is used by the SDF literature). Also note that this is a distinct definition of “implicit” than what is commonly used by the deep learning and statistic communities, where “implicit” usually refers to models whose outputs are implicitly defined as fixed points of dynamic systems, and whose gradients are computed using the implicit function theorem [BKK19].

### 3.2 Differentiable Image Formation

(a) Forward Rendering (e.g., rasterization) – the image is generated by projecting the 3D representation to the image plane.

(b) Ray Casting – the image is generated by casting viewing rays, sampling the 3D representation and accumulating them. Image adapted from [MST∗20].

The scene representations in the previous sections allow us to represent the 3D geometry and appearance of the scene.
As a next step, we describe how images can be generated from such scene representations through rendering.
There are two general approaches to rendering a 3D scene into a 2D image plane: ray casting and rasterization, see also Figure 3.
A rendered image of the scene can be computed by also defining the camera in the scene.
Most methods use a pinhole camera, where all camera rays pass through a single point in space (focal point).
With a given camera, rays from the camera origin can be cast towards the scene in order to calculate the rendered image.

##### Ray Casting.

In the pinhole model, the basic intercept theorem can be used to describe how a point 𝐩∈ℝ3𝐩superscriptℝ3\mathbf{p}\in\mathbb{R}^{3} in 3D is projected to the correct position 𝐪∈ℝ2𝐪superscriptℝ2\mathbf{q}\in\mathbb{R}^{2} in the image plane. It is by definition a non-injective function and hard to invert—this puts it at the heart of the 3D reconstruction problem.

The Pinhole model has a single parameter matrix for this projection: the intrinsic matrix 𝐊𝐊\mathbf{K} contains the focal lengths normalized by pixel size 𝐟=[αx,αy]𝐟subscript𝛼𝑥subscript𝛼𝑦\mathbf{f}=[\alpha_{x},\alpha_{y}], axis skew γ𝛾\gamma and center point 𝐜=[cx,cy]𝐜subscript𝑐𝑥subscript𝑐𝑦\mathbf{c}=[c_{x},c_{y}]. Using the intercept theorem and assuming homogeneous coordinates 𝐩′=[x,y,z,1]superscript𝐩′𝑥𝑦𝑧1\mathbf{p}^{\prime}=[x,y,z,1], we find that the projected coordinates are 𝐪′=𝐊⋅𝐩′superscript𝐪′⋅𝐊superscript𝐩′\mathbf{q}^{\prime}=\mathbf{K}\cdot\mathbf{p}^{\prime}, with

𝐊=[αxγcx00αycy00010].𝐊matrixsubscript𝛼𝑥𝛾subscript𝑐𝑥00subscript𝛼𝑦subscript𝑐𝑦00010\mathbf{K}=\begin{bmatrix}\alpha_{x}&\gamma&c_{x}&0\\
0&\alpha_{y}&c_{y}&0\\
0&0&1&0\\
\end{bmatrix}.

This assumes that the center of the projection is at the coordinate origin and that the camera is axis-aligned. To generalize this for arbitrary camera positions, an extrinsic matrix 𝐑𝐑\mathbf{R} can be used. This homogeneous 4×4444\times 4 matrix 𝐄𝐄\mathbf{E} is composed of

𝐄=[𝐑3×3𝐭3×1𝟎1×31],𝐄matrixsubscript𝐑33subscript𝐭31subscript0131\mathbf{E}=\begin{bmatrix}\mathbf{R}_{3\times 3}&\mathbf{t}_{3\times 1}\\
\mathbf{0}_{1\times 3}&1\\
\end{bmatrix},\vspace{10pt}

where 𝐑𝐑\mathbf{R} is a rotation matrix and 𝐭𝐭\mathbf{t} is a translation vector, such that 𝐑⋅𝐩𝐰+𝐭=𝐩𝐜⋅𝐑subscript𝐩𝐰𝐭subscript𝐩𝐜\mathbf{R}\cdot\mathbf{p_{w}}+\mathbf{t}=\mathbf{p_{c}}, where we use 𝐩𝐰subscript𝐩𝐰\mathbf{p_{w}} to denote a point in world coordinates and 𝐩𝐜subscript𝐩𝐜\mathbf{p_{c}} to denote it in camera coordinates. This definition of 𝐑𝐑\mathbf{R} and 𝐭𝐭\mathbf{t} is common in Computer Vision (for example, used by OpenCV) and referred to as ‘world-to-cam’ mapping, whereas in Computer Graphics (for example, in OpenGL) a similar inverse ‘cam-to-world’ mapping is more prevalent. Assuming the ‘world-to-cam’ convention and using homogeneous coordinates, we can write the full projection of 𝐩𝐰subscript𝐩𝐰\mathbf{p_{w}} to 𝐪𝐩subscript𝐪𝐩\mathbf{q_{p}} as:

𝐪𝐩′=𝐊⋅[𝐑𝐭𝟎1×31]⋅𝐩𝐰′.superscriptsubscript𝐪𝐩′⋅𝐊matrix𝐑𝐭subscript0131superscriptsubscript𝐩𝐰′\mathbf{q_{p}}^{\prime}=\mathbf{K}\cdot\begin{bmatrix}\mathbf{R}&\mathbf{t}\\
\mathbf{0}_{1\times 3}&1\\
\end{bmatrix}\cdot\mathbf{p_{w}}^{\prime}.

If the ‘cam-to-world’ convention is used, the ray casting is similarly convenient. Whereas these equations are non-injective due to the depth ambiguity, they lend themselves very well for automatic differentiation and can be optimized end-to-end in image formation models.

To model current cameras correctly, there is one more component that has to be taken into account: the lens. Leaving aside effects such as depth-of-field or motion blur, which must be modeled in the image formation process, they add distortion effects to the projection function. Unfortunately, there is no single, simple model to capture all different lens effects. Calibration packages, such as the one in OpenCV, usually implement models with up to 12 distortion parameters. They are modeled through polynomials up to degree five, hence are not trivially invertible (which is required for raycasting as opposed to point projection). More modern approaches to camera calibration use many more parameters and achieve a higher accuracy [SLPS20] and could be made invertible and differentiable.

##### Rasterization.

An alternative to ray casting is rasterization of geometric primitives. This technique does not try to emulate the image formation process of the real world, but instead exploits the geometric properties of objects to quickly create an image. It is mostly used with meshes, which are described by a set of vertices 𝐯𝐯\mathbf{v} and faces 𝐟𝐟\mathbf{f}, connecting triples or quadruplets of vertices to define surfaces. One fundamental insight is that the geometric operations in 3D can solely work with the vertices: for example, we can use the same extrinsic matrix 𝐄𝐄\mathbf{E} to transform each point from the world to the camera coordinate system. After this transformation, the points outside of the view frustum or points with wrong normal orientation can be culled to reduce the amount of points and faces to be processed in the next steps. The location of the remaining points projected to image coordinates can again trivially be found by using the intrinsic matrix 𝐊𝐊\mathbf{K} as outlined above. The face information can be used to interpolate the depth on face primitives, and the top-most faces can be stored in a z-buffer.

This way of implementing the projection is often faster than ray casting: it mainly scales with the number of visible vertices in a scene, whereas ray-casting scales with the number of pixels and the numbers of primitives to intersect with. However, it is harder to capture certain effects using it (e.g., lighting effects, shadows, reflections). It can be made differentiable through ‘soft’ rasterization. This has been implemented, for example, in [LLCL19, RRN∗20].

#### 3.2.1 Surface Rendering

##### Point Cloud Rendering.

In the computer graphics literature, point cloud rendering techniques are extensively used [KB04, SP04].
As mentioned before, point clouds are discrete samples of continuous surfaces or volumes.
Point cloud rendering corresponds to reconstructing the continuous signal, e.g., the appearance of a continuous surface, from irregularly distributed discrete samples then resampling the reconstructed signal in the image space at each pixel location.

This process can be done in two different ways.
The first approach is based on the theory of classic signal processing and can be seen as a ‘soft’ point splatting (similar to the soft rasterizer in the mesh rendering section below).
It first constructs the continuous signal using continuous local reconstruction kernels r​(⋅)𝑟⋅r\left(\cdot\right), i.e.,
𝐟=∑𝐟i​r​(𝐩i)𝐟subscript𝐟𝑖𝑟subscript𝐩𝑖\mathbf{f}=\sum\mathbf{f}_{i}r\left(\mathbf{p}_{i}\right).
Essentially, this approach amounts to blending the discrete samples with some local deterministic blurring kernels [LKL18, ID18, RROG18], such as EWA splatting \shortcitezwicker2001surface,zwicker2002ewa, which is a spatially-variant reconstruction kernel that is designed to minimize aliasing.
In neural rendering, the discrete samples can store some learnable features.
Correspondingly, this aforementioned step effectively projects and blends the individual features into a 2D feature map.
If the features have a predefined semantic meaning (e.g., colors, normals), a fixed shading function or BRDF can be used to generate the final image.
If the features are learned neural descriptors, a 2D neural network is deployed to transform the 2D feature map to an RGB image.
Recent neural point rendering methods that adopt this approach include SinSyn and Pulsar [WGSJ20, LZ21].
They use spatially-invariant and isotropic kernels in the blending step for performance reasons.
While these simplified kernels can result in rendering artifacts such as holes, blurred edges and aliasing, these artifacts can be compensated in the neural shading step, and, in case of Pulsar, through optimization of the radii. [KPLD21] additionally uses strategies for camera selection and probabilistic depth testing and is able to tackle IBR, stylization, and harmonization in this framework.

Alternative to the soft point splatting approach, one can use a conventional point renderer from OpenGL or DirectX.
Here, each point is projected to a single pixel (or a small area of pixels) resulting in a sparse feature map.
One can use a deep neural networks to reconstruct the signal directly in the image space [ASK∗20b].
Note that this naive rendering approach does not provide gradients with respect to the point positions 𝐩𝐩\mathbf{p}, and only allows to differentiate the rendering function w.r.t. the (neural) features.
In contrast, the soft point splatting approaches provide point position gradients via the reconstruction kernel r​(𝐩)𝑟𝐩r\left(\mathbf{p}\right).

However, even in this case, the gradient is confined spatially within the support of the local reconstruction.
\shortciteYifan:DSS:2019 addressed this issue by approximating the gradient using finite difference, and successfully applied the renderer to surface denoising, stylization, and multiview shape reconstruction.
This idea was adopted in [RFS21b] to optimize the geometry and camera poses jointly for novel view synthesis.

##### Mesh Rendering.

There are a number of general-purpose renderers that allow meshes to be rasterized or otherwise rendered in a differentiable manner.
Among differentiable mesh rasterizers, Loper and Black \shortciteloper2014opendr developed a differentiable rendering framework called OpenDR that approximates a primary renderer and computes the gradients via automatic differentiation. Neural mesh renderer (NMR) [KUH18] approximates the backward gradient for the rasterization operation using a handcrafted function for visibility changes. [LTJ18] proposed Paparazzi, an analytic differentiable renderer for mesh geometry processing using image filters. Petersen et al. [PBDCO19] presented Pix2Vex, a C∞superscript𝐶C^{\infty} differentiable renderer via soft blending schemes of nearby triangles, and [LLCL19] introduced Soft Rasterizer, which renders and aggregates the probabilistic maps of mesh triangles, allowing gradient flow from the rendered pixels to the occluded and far-range vertices.
While most rasterizers only support rendering based on direct illumination, [LHL∗21] also supports differentiable rendering of soft shadows.
In the domain of physics-based rendering, [LADL18a] and [ALKN19] introduced a differentiable ray tracer to implement the differentiability of physics-based rendering effects, handling camera position, lighting and texture.
In addition, Mitsuba 2 [NDVZJ19] and Taichi [HLA∗19, HAL∗20] are general-purpose physically based renderers that support differentiable mesh rendering via automatic differentiation, among many other graphics techniques.

##### Neural Implicit Surface Rendering.

When the input observations are in the form of 2D images, the network which implements the implicit surface is extended to not only produce geometry-related quantities, i.e., signed distance values, but also appearance-related quantities.
An implicit differentiable renderer [SZW19, NMOG20, LZP∗20, LSCL19, YKM∗20, KJJ∗21, BKW21, TLY∗21] can be implemented by first finding the intersection between a viewing ray and the surface using the geometric branch of the neural implicit function, and then obtaining the RGB value of this point from the appearance branch.
The search of surface intersection is typically based on some variant of the sphere tracing algorithm [Har96].
Sphere tracing iteratively samples the 3D space from the camera center in the direction of the view ray until the surface is reached.
Sphere tracing is an optimized ray marching approach that adjusts the step size by the SDF value sampled at the previous location, but still this iterative strategy can be computationally expensive.
Takikawa et al. \shortcitetakikawa2021nglod improved the rendering performance by adapting the ray-tracing algorithm to the sparse octree data structure.
A common problem for implicit surface rendering for joint geometry and appearance estimation from 2D supervision is the ambiguity of geometry and appearance.
In [NMOG20, YKM∗20, KJJ∗21, BKW21], foreground masks were extracted from the 2D images to provide additional supervision signals for the geometry branch.
Recently, [OPG21] and [YGKL21b] addressed this issue by formulating the surface function into the volumetric rendering formulation (introduced below); on the other hand [ZYQ21] use off-the-shelf depth estimation methods to generate pseudo ground truth signed distance values to assist the training of the geometry branch.

#### 3.2.2 Volumetric Rendering

Volumetric rendering is based on ray casting and has proven to be effective in neural rendering and, especially, in learning a scene representation from multi-view input data.
Specifically, the scene is represented as a continuous field of volume density or occupancy rather than a collection of hard surfaces.

This means that rays have some probability of interacting with the scene content at each point in space, rather than a binary intersection event. This continuous model works well as a differentiable rendering framework for machine learning pipelines that rely heavily on the existence of well-behaved gradients for optimization.

Though fully general volumetric rendering does account for “scattering” events where rays can be reflected off of a volumetric particles [Jar08], we will limit this summary to the basic model commonly used by neural volumetric rendering methods for view synthesis [LH96, Max95], which only accounts for “emission” and “absorption” events, where light is emitted or blocked by a volumetric particle.

Given a set of pixel coordinates, we can use the camera model previously described to calculate the corresponding ray through 3D space with origin 𝐩𝐩\mathbf{p} and direction ωosubscript𝜔o\omega_{\text{o}}. The incoming light along this ray can be defined using a simple emission/absorption model as

L​(𝐩,ωo)=∫t0t1T​(𝐩,ωo,t0,t)​σ​(𝐩+t​ωo)​Le​(𝐩+t​ωo,−ωo)​𝑑t,𝐿𝐩subscript𝜔osuperscriptsubscriptsubscript𝑡0subscript𝑡1𝑇𝐩subscript𝜔osubscript𝑡0𝑡𝜎𝐩𝑡subscript𝜔osubscript𝐿e𝐩𝑡subscript𝜔osubscript𝜔odifferential-d𝑡L(\mathbf{p},\omega_{\text{o}})=\int_{t_{0}}^{t_{1}}T(\mathbf{p},\omega_{\text{o}},t_{0},t)\sigma(\mathbf{p}+t\omega_{\text{o}})L_{\text{e}}(\mathbf{p}+t\omega_{\text{o}},-\omega_{\text{o}})\,dt\,,

(7)

where σ𝜎\sigma is volume density at a point, Lesubscript𝐿eL_{\text{e}} is emitted light at a point and direction, and transmittance T𝑇T is a nested integral expression

T​(𝐩,ωo,t0,t)=exp⁡(−∫t0tσ​(𝐩+s​ωo)​𝑑s).𝑇𝐩subscript𝜔osubscript𝑡0𝑡superscriptsubscriptsubscript𝑡0𝑡𝜎𝐩𝑠subscript𝜔odifferential-d𝑠T(\mathbf{p},\,\omega_{\text{o}},\,t_{0},\,t)=\exp\left(-\int_{t_{0}}^{t}\sigma(\mathbf{p}+s\omega_{\text{o}})\,ds\right)\,.

(8)

Density denotes the differential probability that a ray interacts with the volumetric “medium” of the scene at a particular point, whereas transmittance describes how much light will be attenuated as it travels back toward the camera from point p+t​ωo𝑝𝑡subscript𝜔op+t\omega_{\text{o}}.

These expression can only be evaluated analytically for simple density and color fields. In practice, we typically use quadrature to approximate the integrals, where σ𝜎\sigma and Lesubscript𝐿eL_{\text{e}} are assumed to be piecewise-constant within a set of N𝑁N intervals {[ti−1,ti)}i=1Nsuperscriptsubscriptsubscript𝑡𝑖1subscript𝑡𝑖𝑖1𝑁\{[t_{i-1},t_{i})\}_{i=1}^{N} that partition the length of the ray:

L​(𝐩,ωo)𝐿𝐩subscript𝜔o\displaystyle L(\mathbf{p},\,\omega_{\text{o}})\,
≈∑i=1NTi​αi​Le(i),absentsuperscriptsubscript𝑖1𝑁subscript𝑇𝑖subscript𝛼𝑖superscriptsubscript𝐿e𝑖\displaystyle\approx\,\sum_{i=1}^{N}T_{i}\alpha_{i}\,L_{\text{e}}^{(i)}\,,

(9)

Tisubscript𝑇𝑖\displaystyle T_{i}
=exp⁡(−∑j=1i−1Δj​σj),absentsuperscriptsubscript𝑗1𝑖1subscriptΔ𝑗subscript𝜎𝑗\displaystyle=\exp\left(-\sum_{j=1}^{i-1}\Delta_{j}\sigma_{j}\right)\,,

(10)

αisubscript𝛼𝑖\displaystyle\alpha_{i}
=1−exp⁡(−Δi​σi),absent1subscriptΔ𝑖subscript𝜎𝑖\displaystyle=1-\exp(-\Delta_{i}\sigma_{i})\,,

(11)

ΔisubscriptΔ𝑖\displaystyle\Delta_{i}
=ti−ti−1.absentsubscript𝑡𝑖subscript𝑡𝑖1\displaystyle=t_{i}-t_{i-1}\,.

(12)

For a full derivation of this approximation, we refer the reader to Max and Chen [MC10]. Note that when written in this form, the expression for approximating L𝐿L exactly corresponds to alpha compositing the colors Le(i)superscriptsubscript𝐿e𝑖L_{\text{e}}^{(i)} from back to front [PD84].

NeRF [MST∗20] and related methods (e.g., [MBRS∗21, NG21b, PCPMMN21, SDZ∗21, ZRSK20, NSP∗21]) use differentiable volume rendering to project the scene representations into 2D images. This allows these methods to be used in an “inverse rendering” framework, where a three- or higher-dimensional scene representation is estimated from 2D images. Volume rendering requires many samples to be processed along a ray, each requiring a full forward pass through the network. Recent work has proposed enhanced data structures [YLT∗21, HSM∗21, GKJ∗21], pruning [LGL∗20], importance sampling [NSP∗21], fast integration [LMW21], and other strategies to accelerate the rendering speed, although training times of these methods are still slow. Adaptive coordinate networks accelerate training using a multi-resolution network architecture that is optimized during the training phase by allocating available network capacity in an optimal and efficient manner [MLL∗21b].

### 3.3 Optimization

At the heart of training neural networks lies a non-linear optimization which aims to apply the constraints of the training set in order to obtain a set of neural network weights.
As a result, the function which is approximated by the neural network is fit to the given training data.
Typically, optimization of the neural networks is gradient-based; more specifically SGD variants such as Momentum or Adam [KB14] are utilized, where the gradients are obtained by leveraging the backpropagation algorithm.
In the context of neural rendering, the neural network implements the 3D scene representation, and the training data consists of 2D observations of the scene.
The renderings obtained using differentiable rendering of the neural scene representations is compared with the given observation using various loss functions.
These reconstruction losses can be realized with per-pixel L1 or L2 terms, but also using perceptual [JAFF16] or even discriminator-based loss formulations [GPAM∗14].
However, key is that the losses are directly coupled with the respective differentiable rendering formulation in order to update the scene representations, cf. Section 3.1.

## 4 Applications

In this section, we discuss the specific applications of neural rendering and the underlying neural scene representations.
We first discuss improvements to novel view synthesis of static content in Section 4.1.
We then give an overview over methods that generalize across objects and scenes in Section 4.2.
After that, Section 4.3 discusses non-static, dynamic scenes.
We next turn to editing and composing scenes in Section 4.4.
Then we provide an overview over relighting and material editing in Section 4.5.
Finally, we discuss several engineering frameworks in Section 4.7.
We also develop a taxonomy of the different methods for each application.
These are presented in Table 1, Table 2, Table 3, Table 4, and Table 5, respectively.

### 4.1 Novel View Synthesis of Static Content

Method

Required Data

Requires Pre-trained NeRF

3D Representation

Persistent 3D

Network Inputs

Code

Mildenhall et al. [MST∗20]
I+P
✗
V
F
PE(P)+PE(V)
\faExternalLink

Sitzmann et al. [SZW19]
I+P
✗
S
P
P
\faExternalLink

Niemeyer et al. [NMOG20]
I+P+M
✗
O
F
P
\faExternalLink

Chen et al. [CZ19]
S
✗
O
F
P
\faExternalLink

Gu et al. [LGL∗20]
I+P
✗
G+V
F
PE(P)+PE(V)
\faExternalLink

Lindell et al. [LMW21]
I+P
✗
V
P
PE(P)+PE(V)
\faExternalLink

Reiser et al. [RPLG21]
I+P
✓
G+V
F
PE(P)+PE(V)
\faExternalLink

Garbin et al. [GKJ∗21]
I+P
✓
G
F
P+V
✗

Hedman et al. [HSM∗21]
I+P
✓
G
F
P+PE(V)
\faExternalLink

Yu et al. [YLT∗21]
I+P
✓
G
F
P+V
\faExternalLink

Neff et al. [NSP∗21]
I+P+D
✗
V
F
PE(P)+PE(V)
\faExternalLink

Sitzmann et al. [SRF∗21]
I+P
✗
✗
N
L
\faExternalLink

Novel view synthesis is the task of rendering a given scene from new camera positions, given a set of images and their camera poses as input. Most of the applications presented later in this section generalize the task of view synthesis in some way: in addition to being able to move the camera, they might allow moving or deforming objects within the scene, changing the lighting, and so on.

View synthesis methods are evaluated on a few salient criteria. Clearly, output images should look as realistic as possible. However, this is not the whole story — perhaps even more important is multiview 3D consistency. Rendered video sequences must appear to portray consistent 3D content as the camera moves through the scene, without flickering or warping. As the field of neural rendering has matured, most methods have moved in the direction of producing a fixed 3D representation as output that can be used to render new 2D views, as explained in the scope. This approach automatically lends a degree of multiview consistency that has historically been hard to achieve when relying too heavily on black-box 2D convolutional networks as image generators or renderers.

In Table 1, we give an overview over the discussed methods.

#### 4.1.1 View Synthesis from a 3D Voxel Grid Representation

We will briefly review the recent history of view synthesis using 3D voxel grids and a volumetric rendering model.

DeepStereo [FNPS16] presented the first end-to-end deep learning pipeline for view synthesis. This work included many concepts that have now become commonplace. A convolutional neural network is presented with input images in the form of a plane sweep volume (PSV), where each nearby input is reprojected to a set of candidate depth planes, requiring the network to simply evaluate how well the reprojections match for each pixel at each candidate depth. The CNN’s outputs are converted into a probability distribution over depths using a softmax, which is then used to combine a stack of proposed color images (one per depth plane). The final loss is only enforced on the pixel-wise difference between the rendered output and a heldout target image, with no intermediate heuristic losses required.

A major drawback of DeepStereo is that it requires running a CNN to estimate depth probabilities and produce each output frame independently, resulting in slow runtime and a lack of multiview 3D consistency. Stereo Magnification [ZTF∗18] directly addresses this issue, using a CNN to process a plane sweep volume directly into an output persistent 3D voxel grid representation named a “multiplane image,” or MPI. Rendering new views simply requires using an alpha compositing to render the RGB-alpha grid from a new location. In order to achieve high image quality, Stereo Magnification heavily distorts the parameterization of its 3D grid to bias it to the frame of reference of one of the two input views. This significantly decreases storage requirements for the dense grid but means that new views can only be rendered in the direct neighborhood of the input stereo pair. This shortcoming was later addressed by improving the training procedure for a single MPI [STB∗19], providing many more than two input images to the network [FBD∗19], or combining multiple MPIs together to represent a single scene [MSOC∗19].

All methods mentioned above use a feed-forward neural network to map from a limited set of input images to an output image or 3D representation and must be trained on a large dataset of pairs of input/output views. In contrast, DeepVoxels [STH∗19] optimizes a 3D voxel grid of features jointly with a learned renderer using images of a single scene, without requiring any external training data. Similarly, Neural Volumes [LSS∗19] optimizes a 3D CNN to produce an output volumetric representation for a single scene of multiview video data This single-scene training paradigm has greatly increased in popularity recently, leveraging the unique “self-supervised” aspect of view synthesis: any input images can also be used as supervision via a rerendering loss. In comparison to MPI-based methods, DeepVoxels and Neural Volumes also use a 3D voxel grid parameterization that is not heavily skewed to one particular viewing direction, allowing novel views to be rendered observing the reconstructed scene from any direction.

It is worth mentioning that a number of computer vision papers focused primarily on 3D shape reconstruction (rather than realistic image synthesis) adopted an alpha compositing volumetric rendering model in parallel with this view synthesis research [HRRR18, KHM17, TZEM17]; however, these results were heavily constrained by the memory limitations of 3D CNNs and could not produce voxel grid outputs exceeding 1283superscript1283128^{3} resolution.

#### 4.1.2 View Synthesis from a Neural Network Representation

To address the resolution and memory limitations of voxel grids, Scene Representation Networks (SRNs) [SZW19] combined a sphere-tracing based neural renderer with a multilayer perceptron (MLP) as a scene representation, focusing mainly on generalization across scenes to enable few-shot reconstruction. Differentiable Volumetric Rendering (DVR) [NMOG20] similarly leveraged a surface rendering approach, but demonstrated that overfitting on single scenes enables reconstruction of more complex appearance and geometry.

Neural radiance fields (NeRF [MST∗20] signified a breakthrough in the application of MLP-based scene representations to single-scene, photorealistic novel view synthesis, see Figure 4. Instead of a surface-based approach, NeRF directly applies the volume rendering model described in Section 3.2.2 to synthesize images from an MLP that maps from an input position and viewing direction to output volume density and color. A different set of MLP weights are optimized to represent each new input scene based on pixelwise rendering loss against the input images.

This overall framework shares many similarities with the work described in the previous section. However, MLP-based scene representations can achieve higher resolution than discrete 3D volumes by virtue of effectively differentiably compressing the scene during optimization. For example, a NeRF representation capable of rendering 800×800800800800\times 800 resolution output images only required 5MB of network weights. In comparison, an 8003superscript8003800^{3} RGBA voxel grid would consume close to 2GB of storage.

This ability can be attributed to NeRF’s use of a positional encoding applied to the input spatial coordinates before passing through the MLP. In comparison to the previous work on using neural networks to represent implicit surfaces [PFS∗19, CZ19] or volumes [MON∗19], this allows NeRF’s MLP to represent much higher frequency signals without increasing its capacity (in terms of number of network weights).

The main drawback of switching from a discrete 3D grid to an MLP-based representation is rendering speed. Rather than directly querying a simple data structure, calculating the color and density value for a single point in space now requires evaluating an entire neural network (hundreds of thousands of floating point operations). On a typical desktop GPU, an implementation of NeRF in a standard deep learning framework takes tens of seconds to render a single high resolution image.

#### 4.1.3 Improving Rendering Speed

Several different methods have been proposed for speeding up volumetric rendering of MLP-based representations. Neural Sparse Voxel Fields [LGL∗20] builds and dynamically updates an octree structure while the MLP is optimized, allowing for aggressive empty space skipping and early ray termination (when the transmittance along the ray approaches zero). KiloNeRF [RPLG21] combines empty space skipping and early termination with a dense 3D grid of MLPs, each with a much smaller number of weights than a standard NeRF network.

Three concurrent works recently proposed methods for caching the values various quantities learned by the NeRF MLP on a sparse 3D grid, allowing for realtime rendering once training is complete. Each method modifies the way in which view-dependent colors are predicted in order to facilitate faster rendering and smaller memory requirements for the cached representations. SNeRG [HSM∗21] stores volume density and a small spatially-varying feature vector in a sparse 3D texture atlas, using fast shader for compositing these values along a ray and running a tiny MLP decoder to produce view-dependent color for each ray. FastNeRF [GKJ∗21] caches volume density along with weights for combining a set of learned spherical basis functions that produce view-varying colors at each point in 3D. PlenOctrees [YLT∗21] queries the MLP to produce a sparse voxel octree of volume density and spherical harmonic coefficients and further finetunes this octree representation using a rendering loss to improve its output image quality.

NeX-MPI [WPYS21] combines the multiplane image parameterization with an MLP scene representation, with view dependent effects parameterized as a linear combination of globally learned basis functions. Because the model is supervised directly on a 3D MPI grid of coordinates, this grid can be easily cached to render new views in real time once optimization is complete.

An alternative approach for accelerating rendering is to train the MLP representation itself to effectively precompute part or all of the volume integral along the ray. AutoInt [LMW21] trains a network to “automatically integrate” the output color value along ray segments by supervising the gradient of the network to behave like a standard NeRF MLP. This allows the rendering step to break the integral along a ray into an order of magnitude fewer segments than the standard quadrature estimate (down to as few as 2 or 4 samples), trading off speed for a minor loss in quality. Light Field Networks [SRF∗21] takes this a step further, optimizing an MLP to directly encode the mapping from an input ray to an output color (the scene’s light field). This enables rendering with only a single evaluation of the MLP per ray, in contrast to hundreds of evaluations for volume- and surface-based renderers, and enables real-time novel view synthesis. These methods present a tradeoff between rendering speed and multiview consistency: reparameterizing the MLP representation as a function of rays rather than 3D points means that the scene is no longer guaranteed to appear consistent when viewed from different angles. In this case multiview consistency must be enforced through supervision, either by providing a very large number of input images or learning this property via generalization across a dataset of 3D scenes.

Recently, there has also been a flood of new approaches [PC21, FXW∗21, YFKT∗21, WLB∗21, SSC21, KIT∗21] that employ classical data structures, such as grids, sparse grids, trees, and hashes, for acceleration of rendering speed as well as faster training times.
Instant Neural Graphics Primitives [MESK22] enables the training of a NeRF in a few seconds exploiting a multi-resolution hash encoding instead of an explicit grid structure.

#### 4.1.4 Miscellaneous Improvements

A variety of papers have augmented the rendering model, supervision data, or robustness of volumetric MLP scene representations.

##### Depth Supervision.

DONeRF [NSP∗21] trains an “depth oracle” network to predict sample locations along each ray, drastically reducing the number of samples sent through the NeRF MLP and allowing interactive rate rendering. However, this method is supervised with dense depths maps, which are challenging to obtain for real data. Depth-supervised NeRF [DLZR21] directly supervises the output depths from NeRF (in the form of expected termination depth along each ray) using the sparse point cloud output which is a byproduct of estimating camera poses using structure-from-motion.
NerfingMVS [WLR∗21] uses a multistage pipeline for depth supervision, first finetuning a single-view depth estimation network on sparse multiview stereo depth estimates, then uses the resulting dense depth maps to guide NeRF sample placement.
Roessle et al. [RBM∗21] directly applies a pretrained sparse-to-dense depth completion network to sparse structure-from-motion depth estimates, then uses depth (along with predicted uncertainty) to both guide sample placement and supervise the depth produced by NeRF.

##### Optimizing Camera Poses.

NeRF– [WWX∗21] and Self-Calibrating Neural Radiance Fields [JAC∗21] jointly optimize the NeRF MLP and input camera poses, bypassing the need for structure-from-motion preprocessing for forward facing scenes. Bundle-Adjusting Neural Radiance Fields (BARF) [LMTL21] extends this idea by applying a coarse-to-fine annealing schedule to each frequency component of the positional encoding function, providing a smoother optimization trajectory for joint reconstruction and camera registration. However, neither of these methods can optimize poses from scratch for wide-baseline 360 degree captures. GNeRF [MCL∗21] achieves this by training a set of cycle consistent networks (a generative NeRF and a pose classifier) that map from pose to image patches and back to pose, optimizing until the classified pose of real patches matches that of sampled patches. They alternate this GAN training phase with a standard NeRF optimization phase until the result converges.

##### Hybrid Surface/Volume Representations.

The Implicit Differentiable Renderer (IDR) from Yariv et al. [YKM∗20] combines a DVR-like implicit surface MLP with a NeRF-like view dependent branch which takes viewing direction, implicit surface normal, and the 3D surface point as inputs and predicts the view-varying output color. This work shows that including the normal vector as input to the color branch helps the representation disentangle geometry and appearance more effectively. It also demonstrates that camera pose can be jointly optimized along with the shape representation to recover from small miscalibration errors.

UNISURF [OPG21] proposes a hybrid MLP representation that unifies surface and volume rendering. To render a ray, UNISURF uses root finding to get a “surface” intersection point, treating the volume as an occupancy field, then distributes volume rendering samples only within an interval around that point. The width of this interval monotonically decreases over the course of optimization, allowing early iterations to supervise the whole training volume and later stages to more efficiently refine the surface with tightly spaced samples. Azinovic et al. [AMBG∗21] propose to use an SDF representation instead of volume densities to reconstruct scenes from RGB-D data. They convert the sdf values to densities that can be used in the NeRF formulation. NeuS [WLL∗21] ties the volume density to an signed distance field and reparameterizes the transmittance function such that it achieves its maximal slope precisely at the zero-crossing of this SDF, allowing an unbiased estimate of the corresponding surface. VolSDF [YGKL21b] uses an alternate mapping from SDF to volume density, which allows them to devise a new resampling strategy to achieve provably bounded error on the approximated opacity in the volume rendering quadrature equation.
In [LFS∗21], the authors propose a method called MINE which is a hybrid between multi-plane images (MPI) and NeRF. They are able to reconstruct dense 3D reconstructions from single color images which they demonstrate on RealEstate10K, KITTI and Flowers Light Fields.

##### Robustness and Quality.

NeRF++ [ZRSK20] provides a “inverted sphere” parameterization of space that can allow NeRF to large-scale, unbounded 3D scenes. Points outside the unit sphere are inverted back into the unit sphere and passed through a separate MLP.

NeRF in the Wild [MBRS∗21] adds additional modules to the MLP representation to account for inconsistent lighting and objects across different images. They apply their robust model to the PhotoTourism dataset [SSS06] (consisting of internet images of famous landmarks across the world) and are able to remove transient objects such as people and cars and capture time-varying appearance through use of a latent code embedding associated with each input image.
Ha-NeRF [CZL∗21] extends the idea of NeRF in the Wild to hallucinate novel appearances.

MipNeRF [BMT∗21] modifies the positional encoding applied to 3D points to incorporate the pixel footprint, see Figure 5. By pre-integrating the positional encoding over a conical frustum corresponding to each quadrature segment sampled along the ray, MipNeRF can be trained to encode a representation of the scene at multiple different scales (analogously to a mipmap of a 2D texture), preventing aliasing when rendering the scene from dramatically varying positions or resolutions.
Mip-NeRF 360 [BMV∗21] extends MipNeRF and addresses issues that arise when training on unbounded scenes (unbalanced detail of nearby and distant objects which leads to blurry, low-resolution renderings), where the camera rotates 360 degrees around a point. It leverages a non-linear scene parametrization, online distillation, and a novel distortion-based regularizer.

##### NeRF and Computational Imaging.

Several recent works combine NeRF with standard computational imaging tasks. Deblur-NeRF [MLL∗21a] jointly optimizes a static NeRF representation along with per-ray offsets for every pixel in the training set that account for blur due to either camera motion or depth of field. Once optimization is complete, the NeRF can be rendered without applying the ray offsets to obtain sharp test views. NeRF in the Dark [MHMB∗21] trains directly on raw linear camera data to achieve improved robustness to high levels of image noise, allowing reconstruction of dark nighttime scenes as well as recovery of full high dynamic range radiance values. HDR-NeRF [HZF∗21] similarly recovers a linear-valued HDR NeRF, but by using postprocessed variable-exposure images as input and solving for a nonlinear camera postprocessing curve that reproduces the inputs when applied to the optimized NeRF. NeRF-SR [WWG∗21a] averages multiple supersampled rays per pixel during training and also performs super-resolution on rendered image patches by merging them with similar patches from a high-resolution reference image of the scene using a CNN.

##### Large-scale Scenes.

A series of recent publications focus on large-scale neural radiance fields.
They enable the re-rendering of street-view data [RLS∗21], buildings, and entire cities [XXP∗21, TRS21], or even earth-scale [XXP∗21].
To handle such large scenes, the methods use localized NeRFs by decomposing the scene into spatial cells [TRS21] or on different scales [XXP∗21] (including a
progressive training scheme).
URF [RLS∗21] exploits additional LIDAR data to supervise the depth prediction.
In these large-scale scenarios, special care must be taken to handle the sky and the highly varying exposure and illumination changes (cf. NeRF-W [MBRS∗21]).
NeRF-W [MBRS∗21] interpolates between learned appearances, but does not provide semantic control over it.
NeRF-OSR [RES∗21] is the first method allowing joint editing of the camera viewpoints and the illumination for buildings and historical sites.
For training, NeRF-OSR requires outdoor photo collections shot in uncontrolled settings (see Section 4.5 for further details).

##### NeRF from Text.

The NeRF formulation [MST∗20] is an optimization-based framework, which also allows us to incorporate other energy terms during optimization.
To manipulate or generate a NeRF by text inputs, one can employ a (pretrained) CLIP-based objective [RKH∗21].
Dream Fields [JMB∗21] combines NeRF with CLIP to generate diverse 3D objects solely from natural language descriptions, by optimizing the radiance field via multi-view constraints based on the CLIP scores on the image caption.
CLIP-NeRF [WCH∗21] proposes a CLIP-based shape and appearance mapper to control a conditional NeRF.

### 4.2 Generalization over Object and Scene Classes

Method

Conditioning

Required Data

3D Representation

Class Specific Prior

Generative Model

Inference Type

Code

Yu et al. [YYTK21]
L
G
V
✗
✗
A
\faExternalLink

Raj et al. [RZS∗20]
L
F
V
✗
✗
A
✗

Trevithick et al. [TY20]
L
G
V
✗
✗
A
\faExternalLink

Wang et al. [WWG∗21b]
L
G
V
✗
✗
A
✗

Reizenstein et al. [RSH∗21]
L
G
V
✗
✗
A
✗

Sitzmann et al. [SZW19]
G
G
S
✓
✓
D
\faExternalLink

Kosiorek et al. [KSZ∗21]
G
G
V
✗
✓
A
✗

Rematas et al. [RMBF21]
G
G
V
✓
✓
D
\faExternalLink

Xie et al. [XPMBB21]
G
G
V
✗
✗
A
✗

Tancik et al. [TMW∗21]
G
G
V
✗
✗
GB
\faExternalLink

Gao et al. [GSL∗20]
G
F
V
✗
✗
GB
✗

Nguyen-Phuoc et al. [NPLT∗19]
G
G
V
✓
✓
✗
\faExternalLink

Schwarz et al. [SLNG20]
G
G
V
✓
✓
✗
✗

Chan et al. [CMK∗21]
G
G
V
✓
✓
✗
\faExternalLink

Anonymous [GLWT21a]
G
G
V
✓
✓
✗
✗

Niemeyer et al. [NG21a]
G
G
V
✓
✓
✗
✗

While a significant amount of prior work addresses generalization over multiple scenes and object categories for voxel-based, mesh-based, or non-3D structured neural scene representations, we focus this discussion on recent progress in generalization leveraging MLP-based scene representations. Where approaches that overfit a single MLP on a single scene [MST∗20, YKM∗20] require a large number of image observations, the core objective of generalizing across scene representations is novel view synthesis given few or potentially only a single input view.
In Table 2, we give an overview over the discussed methods, classified by whether they leverage local or global conditioning, whether they can be used as unconditional generative models or not, what kind of 3D representation they leverage (volumetric, SDF, or occupancy), what kind of training data they require, and how inference is performed (amortized with an encoder, via the auto-decoder framework, or via gradient-based meta-learning).

We may differentiate two key approaches in generalizing across scenes.
One line of work follows an approach reminiscent of image-based rendering [CW93, SK00], where multiple input views are warped and blended to synthesize a novel viewpoint. In the context of MLP-based scene representations, this is often implemented via local conditioning, where the coordinate input to the scene representation MLP is concatenated with a locally varying feature vector, stored in a discrete scene representation, such as a voxel grid [PNM∗20]. PiFU [SHN∗19] uses an image encoder to compute features on the input image and conditions a 3D MLP on these features via projecting 3D coordinates on the image plane - however, PiFU did not feature a differentiable renderer, and so required ground-truth 3D supervision. PixelNeRF [YYTK21] (see Figure 6) and Pixel-Aligned Avatars [RZS∗20] leverage this approach in a volume rendering framework where these features are aggregated over multiple views, and a MLP produces color and density fields that are rendered as in NeRF. When trained on multiple scenes, they learn scene priors for reconstruction, that enable high fidelity reconstruction of scenes from a few views. PixelNeRF can also be trained on specific object categories, enabling object instance 3D reconstruction from one or multiple posed images. GRF [TY20] uses a similar framework, with an additional attention module that reasons about the visibility of the 3D point in the different sampled input images. Stereo Radiance Fields [CBLPM21] similarly extracts features from several context views, but leverages learned correspondence matching between pairwise features across context images to aggregate features across context images instead of a simple mean aggregation. Finally, IBRNet [WWG∗21b] and NeRFormer [RSH∗21] introduce transformer networks across the ray samples that reason about visibility.
LOLNeRF [RMY∗21] learns a generalizable NeRF model on portrait images using only monocular supervision.
The generator network is conditioned on instance-specific latent vectors, which are jointly trained.
Joint training on large datasets enable training without multi-view supervision.
GeoNeRF [JLF21] constructs a set of cascaded cost volumes and employs transformers to infer geometry and appearance.

An alternative to such image-based approaches aims to learn a monolithic, global representation of a scene instead of relying on images or other discrete spatial data structures. This is accomplished by inferring a set of weights for the scene representation MLP that describes the whole scene, given a set of observations. One line of work accomplishes this by encoding a scene in a single, low-dimensional latent code that is then used to condition the scene representation MLP. Scene Representation Networks (SRNs) [SZW19] map low-dimensional latent codes to the parameters of a MLP scene representation via a hypernetwork, and subsequently render the resulting 3D MLP via ray-marching. To reconstruct an instance given a posed view, SRNs optimize the latent code so that its rendering matches the input view(s). Differentiable Volumetric Rendering [NG20] similarly uses surface rendering, but computes its gradients analytically and performs inference via a CNN encoder. Light Field Networks [SRF∗21] leverage low-dimensional latent codes to directly parameterize the 4D light field of the 3D scene, enabling single-evaluation rendering. NeRF-VAE embeds a NeRF in a variational auto-encoder, similarly representing the whole scene in a single latent code, but learning a generative model that enables sampling [KSZ∗21]. Sharf [RMBF21] uses a generative model of voxelized shapes of objects in a category, which in turn condition a higher resolution neural radiance field that is rendered using volume rendering for higher novel view synthesis fidelity. Fig-NeRF [XPMBB21] models an object category as a template shape conditioned on a latent code, that undergoes a deformation that is also conditioned on the same latent variable. This enables the network to explain certain shape variations as more intuitive deformations. Fig-NeRF focuses on retrieving an object category from real object scans, and also proposes using a learn background model to segment the object from its background.
An alternative to representing the scene as a low-dimensional latent code is to quickly optimize the weights of an MLP scene representation in a few optimization steps via gradient-based meta-learning [SCT∗20]. This can be used to enable fast reconstruction of neural radiance fields from few images [TMW∗21]. The pre-trained models converge faster when trained on a novel scene, and require fewer views compared to standard neural radiance field training. PortraitNeRF [GSL∗20] proposes a meta-learning approach to recover a NeRF from a single frontal image of a person. To account for differences in pose between the subjects, it models the 3D portraits in a pose-agnostic canonical reference frame, that is warped for each subject using 3D keypoints. Bergman et al. [BKW21] leverage gradient-based meta-learning and local conditioning on image features to quickly recover a NeRF of a scene.

Instead of inferring a low-dimensional latent code conditioned on a set of observations of the sought-after 3D scene, a similar approach can be leveraged to learn unconditional generative models.
Here, a 3D scene representation equipped with a neural renderer is embedded in a generative adversarial network. Instead of inferring low-dimensional latent codes from a set of observations, we define a distribution over latent codes. In a forward pass, we sample a latent from that distribution, condition the MLP scene representation on that latent, and render an image via the neural renderer. This image can then be used in an adversarial loss. This enables learning of a 3D generative model of shape & appearance of 3D scenes given only 2D images.
This approach was first proposed with 3D scene representations parameterized via voxelgrids [NPLT∗19]. GRAF [SLNG20] first leveraged a conditional NeRF in this framework and achieved significant improvements in photorealism. Pi-GAN [CMK∗21] further improved on this architecture with a FiLM-based conditioning scheme [PSDV∗18] of a SIREN architecture [SMB∗20].

Several recent approaches explore different directions for improving the quality and efficiency of these generative models.
Computational cost and quality of geometric reconstructions can be improved by using a surface representation[DYXT21, OELS∗21, XPLD21].
In addition to synthesizing multi-view images for the discriminator, ShadeGAN [PXL∗21] uses an explicit shading step to also generate the output image renderings under different illumination conditions for higher-quality geometry reconstructions.
Many approaches have explored using a hybrid technique where an image-based CNN network is used to refine the output of the 3D generator [GLWT21b, CLC∗22, XPY∗21, ZXNT21].
The image-space network enables training at higher resolutions with higher-fidelity outputs.
Decomposing the generative model into separate geometry and texture spaces has also been explored.
Here, some approaches learn the texture in image space [CLX∗21, XPY∗21], while others learn both geometry and texture in 3D [SWZ∗21, SLNG20].

While all these approaches do not require more than one observation per 3D scene and thus, also no ground-truth camera poses, they still require knowledge of the distribution of camera poses (i.e., for portrait images, the distribution over camera poses must produce plausible portrait angles). CAMPARI [NG21a] addresses this constraint by jointly learning camera pose distribution and generative model.
GIRAFFE [NG21b] proposes to learn a generative model of scenes composed of several objects by parameterizing a scene as a composition of several foreground (object) NeRFs and a single background NeRF. Latent codes are sampled for each NeRF separately, and a volume renderer composes them to a coherent 2D image.

### 4.3 Learning to Represent and Render Non-static Content

Method

Data

Deformation

Class-Specific Prior

Controllable Parameters

Code

Lombardi et al. [LSS∗21]
MV
I
G
V,R
\faExternalLink

Li et al. [LNSW21]
Mo
I+E
G
V,R
\faExternalLink

Xian et al. [XHKK21]
Mo
I
G
V,R
✗

Gao et al. [GSKH21]
Mo
I
G
V,R
✗

Du et al. [DZY∗21]
Mo
I
G
V,R
\faExternalLink

Pumarola et al. [PCPMMN21]
Mo
E
G
V,R
\faExternalLink

Park et al. [PSB∗21]
Mo
E
G
V,R
\faExternalLink

Tretschk et al. [TTG∗21]
Mo
E
G
V,R
\faExternalLink

Park et al. [PSH∗21]
Mo
I+E
G
V,R
\faExternalLink

Attal et al. [ALG∗21]
Mo+D
I
G
V,R
\faExternalLink

Li et al. [LSZ∗21]
MV
I
G
V,R
✗

Gafni et al. [GTZN21]
Mo
E
F
V,R,E
\faExternalLink

Wang et al. [WBL∗21]
MV
I
F
V,R
✗

Guo et al. [GCL∗21]
Mo
I
F
V,R,E
\faExternalLink

Noguchi et al. [NSLH21]
Mo+3D
E
B
V,R,E
\faExternalLink

Su et al. [SYZR21]
Mo
E
B
V,R,E
✗

Peng et al. [PDW∗21]
MV
E
B
V,R,E
\faExternalLink

Peng et al. [PZX∗21]
MV
I+ E
B
V,R
\faExternalLink

Liu et al. [LHR∗21]
MV
E
B
V,R,E
✗

Xu et al. [XAS21]
MV+Mo
I
B
V,R,E
✗

While the original neural radiance fields [MST∗20] are used to represent static scenes and objects, there are approaches that can additionally handle dynamically changing content.
In Table 3, we give an overview over the discussed methods.

These approaches can be categorized in time-varying representations that allow to do novel viewpoint synthesis of a dynamically changing scene as an unmodified playback (e.g., to produce a bullet-time effect), or in techniques that also give control over the deformation state, thus, allowing for novel-view point synthesis and editing of the content.
The deforming neural radiance field can be achieved implicitly or explicitly, see Figure 7:

- •

Implicitly, by conditioning the NeRF on a representation of the deformation state (e.g., a time input)

- •

Explicitly, by using a separate deformation field that can map from the deformed space to a canonical space where the NeRF is embedded.

#### 4.3.1 Time-varying Neural Radiance Fields

Time-varying neural radiance fields allow to playback a video with novel view points, see Figure 8.
Since they forego control, these methods do not rely on a specific motion model and can thus handle general objects and scenes.

Several extensions of NeRF for non-rigid scenes were proposed concurrently. We first discuss methods that model deformations implicitly [LNSW21, XHKK21, GSKH21, DZY∗21]. While the original NeRF is static and takes as input only a 3D spatial point, it can be extended in a straightforward manner to become time-varying: the volume can additionally be conditioned on a vector that represents the deformed state. In current methods, this conditioning takes the form of a time input (potentially positionally encoded) [XHKK21, LNSW21, GSKH21, DZY∗21, PCPMMN21] or an auto-decoded latent code per time step [PSB∗21, TTG∗21, PSH∗21].

Since handling non-rigid scenes without prior knowledge of object type or 3D shape is an ill-posed problem, methods of this class adopt various geometric regularizers and condition learning on additional data modalities. To encourage consistency of reflectance and opacity across time, several methods learn scene-flow mappings between temporally neighboring time steps [LNSW21, XHKK21, GSKH21, DZY∗21]. Since this is restricted to small temporal neighborhoods, artifact-free novel-view synthesis is predominantly demonstrated on spatio-temporal camera trajectories that are close to the spatio-temporal input camera trajectories. The scene-flow mapping can be trained with reconstruction losses that warp the scene from other time steps into the current time step [LNSW21, DZY∗21], by encouraging consistency between estimated optical flow and the 2D projection of the scene flow [LNSW21, GSKH21], or by tracking backprojected keypoints in 3D [DZY∗21]. The scene flow is often constrained with additional regularization losses [LNSW21, XHKK21, GSKH21, DZY∗21], e.g., to encourage spatial or temporal smoothness or forward-backward cycle consistency. Unlike the other methods mentioned, Neural Radiance FLow (NeRFlow) of Du et al. [DZY∗21] models deformations with infinitesimal displacements that need to be integrated with Neural ODE [CRBD18] to obtain offsets.

In addition, several methods use estimated depth maps to supervise the geometry estimation [LNSW21, XHKK21, GSKH21, DZY∗21]. One limitation of this regularization is that the accuracy of the reconstruction depends on the accuracy of monocular depth estimation methods. As a result, artefacts of monocular depth estimation methods are recognizable in the novel views [XHKK21].

Finally, the static background is often handled separately, allowing it to exploit multi-view clues from monocular input recordings across time. To that end, some methods estimate a second static volume that is not conditioned on the deformation [LNSW21, GSKH21] or introduce soft regularization losses to constrain static scene content [XHKK21]. Gao et al. [GSKH21], a follow-up to Xian et al.’s work [XHKK21], train the static NeRF on observations that do not contain moving and deforming parts with the help of a binary segmentation mask (one of the inputs to the model and user-provided).

One advantage of Guo et al.’s method is that it produces the most accurate quantitative and qualitative results on the challenging dataset of Yoon et al. [YKG∗20] (compared to Tretschk et al. [TTG∗21] and Li et al. [LNSW21]). The latter dataset was initially introduced for novel-view synthesis from a comparably sparse set of input monocular views of dynamic scenes with moderate changes in the camera poses. Limitations of the method include strong reliance on optical flow and handling of arbitrary non-rigid deformations (in contrast to scenes composed of independent rigidly moving objects).

Finally, NeRFlow [DZY∗21] can be used to de-noise and super-resolve views of pre-trained scenes. Limitations of NeRFlow, which the authors mention, include difficulty in preserving static backgrounds, handling complex scenes (non-piecewise-rigid deformations and motions) and rendering novel views at substantially different camera trajectories compared to the input ones.

The methods discussed so far model deformations implicitly by conditioning the scene representation on the deformation. This makes controllability of the deformation cumbersome and difficult. Other works instead disentangle the deformations from the geometry and appearance: they factor out the deformations into a separate function on top of a static canonical scene, a crucial step towards controllability. The deformations are accomplished by shooting straight rays into deformed space and then bending them into the canonical scene, usually by regressing per-point offsets for points on the straight ray using a coordinate-based MLP that is conditioned on the deformation. This can be thought of as space warping or scene flow. In contrast to implicit modelling, these methods share geometry and appearance information across time by construction via the static canonical scene, thereby providing hard correspondences, which do not drift. Due to that hard constraint, unlike implicit methods, current methods with explicit deformations cannot handle topological changes and only demonstrate results on scenes with significantly smaller motion than implicit methods.

D-NeRF [PCPMMN21] uses an unregularized ray-bending MLP to model deformations of a single or multiple synthetic objects segmented from the background and observed by virtual cameras. It assumes a pre-defined set of multi-view images given, though, at training time, only a single view chosen arbitrarily is used for supervision at any time. Thus, D-NeRF can be considered an intermediate step between techniques with multi-view supervision and truly monocular approaches.

Several works show results on real-world scenes observed by a moving monocular camera. The core application of Deformable NeRF of Park et al. [PSB∗21] is the creation of Nerfies, i.e., free-viewpoint selfies. Deformable NeRF conditions deformations and appearance with an auto-decoded latent code per input view. The bent rays are regularized using an as-rigid-as-possible term (also known as elastic energy term) that penalizes deviations from piece-wise rigid scene configurations. Thus, Deformable NeRF works well on articulated scenes (e.g., a hand holding a tennis racket) and scenes such as human heads (where the head is moving w.r.t. the torso). Still, small non-rigid deformations are handled well (such as smiling), as the regularizers are soft. Another important innovation of this work is using a coarse-to-fine scheme which allows learning low-frequency components first and avoiding local minima due to overfitting to high-frequency details.

HyperNeRF [PSH∗21] is an extension of Deformable NeRF [PSB∗21] using a canonical hyperspace instead of a single canonical frame. This allows tackling scenes with topological changes such as opening and closing the mouth. In HyperNeRF, the bending network (MLP) of Deformable NeRF is augmented with an ambient slicing surface network (likewise an MLP) that selects a canonical subspace for every input RGB view by indirectly conditioning the canonical scene on the deformation. As such it is a hybrid that combines both explicit and implicit deformation modelling, which allows it to handle topological changes by sacrificing hard correspondences.

Non-rigid NeRF (NR-NeRF) [TTG∗21] models a time-varying scene appearance using a per-scene canonical volume, per-scene rigidity flag (an MLP) and per-frame ray bending operator (an MLP).
NR-NeRF shows that no additional supervisory cues such as depth maps or scene flows are required to handle scenes with small non-rigid deformations and motions, in contrast to [PSB∗21, XHKK21, LNSW21]. Moreover, the observed deformations are regularized by a divergence operator, which imposes a volume-preserving constraint and stabilizes occluded areas with respect to supervising monocular input views. In this regard, it has similarities with the elastic regularizer of Nerfies penalizing deviations from piece-wise rigid deformations. This regularization makes it possible for the camera trajectory of novel views to differ significantly from the input camera trajectory.
While controllability is still severely limited, NR-NeRF demonstrates several simple edits of the learned deformation field, such as motion exaggeration or removal of dynamic scene content.

Other works do not restrict themselves to the case of monocular RGB input video, but instead consider other inputs.

Time-of-Flight Radiance Fields (TöRF) method [ALG∗21] replaces data-driven priors for reconstructing dynamic contents with depth maps from a depth sensor. In contrast to the vast majority of computer vision works, TöRF uses raw ToF sensor measurements (so-called phasors), which brings advantages when handling weakly-reflecting regions and other limitations of modern depth sensors (e.g., restricted working depth range). Integration of measured scene depths in the learning of NeRF reduces the requirement on the number of input views leading to sharp and detailed models. The depth cue also enables superior accuracy compared to NSFF [LNSW21] and space-time neural irradiance fields [XHKK21].

Neural 3D Video Synthesis [LSZ∗21] uses a multi-view RGB setup and models deformations implicitly. The method exploits temporal smoothness by first training on keyframes. It also exploits that the cameras remain static and that the scene content is predominantly static by sampling rays in a biased manner for training. The results are sharp even for dynamic content that is small.

#### 4.3.2 Controllable Dynamic Neural Radiance Fields

To allow controllability of the deformation of the neural radiance field, method use class specific motion models as underlying representation of the deformation state (e.g., a morphable model for the human face or a skeletal deformation graph for the human body).

NeRFace [GTZN21] is the first approach that uses a morphable model to implicitly control a neural radiance field (see Figure 9). They use a face tracker [TZS∗16] to reconstruct the face blendshape parameters as well as the camera pose in the training views (monocular video). The MLP is trained on these views with the blendshape parameters and a learnable per-frame latent codes as conditioning. In addition, they assume a known static background such that the radiance field only stores the information about the face. The latent codes are used to compensate missing tracking information (i.e., the shoulders of a person) as well as errors in the tracking). Once trained the radiance field can be controlled via the blendshape parameters, thus, allowing reenactment and expression editing.
While NeRFace uses a global deformation code based on a morphable model, Wang et al. [WBL∗21] generate local animation codes. Specifically, they extract a global animation code from multi-view inputs which is mapped to local codes using 3D convolutional neural network. These are used to condition the fine-level radiance field which are represented as MLPs. In contrast to NeRFace, the method does not allow direct control over expressions of the face, but an encoder has to be trained that for example can generate the animation codes from facial keypoints.
Guo et al. [GCL∗21] propose an audio driven neural radiance field (AD-NeRF) which is inspired by NeRFace.
But instead of using expression coefficients, they map audio features extracted using DeepSpeech [HCC∗14, TET∗20] to a feature which serves as a conditioning to the MLP that represents the radiance field. While the expression is controlled implicitly via an audio signal, they provide explicit control over the rigid pose of the head. To synthesize the portrait view of a person, they employ two separate radiance fields, one for the head and one for the torso.

,,I M Avatar” [ZAC∗21] extends NeRFace based on skinning fields [CZB∗21] which are used to deform the canonical NeRF volume given novel expression and pose parameters.
CoNeRF [KYK∗21] presents a method to disentangle attribute/expression combinations leveraging sparse mask annotations in the training images.
They rely on a locality assumption, i.e., one attribute affects only a specific region. These localized attribute masks are treated as latent codes within their framework.

Besides these subject-specific training methods, HeadNerf [HPX∗21] and MoFaNeRF [ZZSC21] propose a generalized model to represent faces under different views, expressions and illumination.
Similar to NeRFace, they condition the NeRF MLP on additional parameters that control the shape of the person, the expression, albedo, and illumination.
Both methods, require a refinement network (2D network) to improve the coarse results of the volumetric rendering based on this conditioned NeRF MLP.

While the afore mentioned approaches show promising results in a portrait scenario, they are not applicable to highly non-rigid deformations, especially, for articulated motion of a human body captured from a single view.
Therefore, methods leverage the human skeleton embedding explicitly.
Neural Articulated Radiance Field (NARF) [NSLH21] is trained via pose-annotated images.
An articulated object is decomposed into several rigid object parts with their local coordinate systems and global shape variations on top.
The converged NARF can be used to render novel views by manipulating the poses, estimate depth maps and perform body parts segmentation.
In contrast to NARF, A-NeRF [SYZR21] learns actor-specific volumetric neural body models from monocular footage in a self-supervised manner.
The method combines a dynamic NeRF volume with the explicit controllability of an articulated human skeleton embedding and reconstructs both the pose and radiance field in an analysis-by-synthesis way.
Once trained, the radiance field can be used for novel view point synthesis as well as motion retargeting.
They show the benefits of using a learned surface-free model which improves the accuracy of human pose estimation from monocular videos with the help of a photometric reconstruction loss.
While A-NeRF is trained on monocular videos, Animatable Neural Radiance Fields (ANRF) [PDW∗21] is a skeleton-driven approach for human model reconstruction from multi-view videos.
Its core component is a new motion representation, i.e., the neural blend weight field, that is combined with 3D human skeletons for deformation field generation.
Similarly to several general non-rigid NeRFs [PSB∗21, TTG∗21], ANRF maintains a canonical space and estimates two-way correspondences between the multi-view inputs and the canonical frame.
The reconstructed animatable human models can be used for free-viewpoint rendering and re-rendering under novel poses.
Human meshes can also be extracted from ANRF by running marching cubes on volume densities at the discretized canonical space points.
The method achieves high visual accuracy for the learned human models, and the authors suggest that handling complex non-rigid deformations on the observed surfaces (such as those due to loose clothes) can be improved in future work.

The Neural Body approach of Peng and colleagues [PZX∗21] enables novel view synthesis of human performances from sparse multi-view videos (e.g., only four synchronized views), see Figure 10 for exemplary inputs and the result.
Their method uses conditioning by the parametric human shape model SMPL [LMR∗15] as a shape proxy prior.
It assumes that the recovered neural representation at different frames has the same set of latent codes anchored to a deformable mesh. General-purpose baselines such as rigid NeRF [MST∗20] (applied per timestamp) or NeuralVolumes [LSS∗19] assume much denser input image sets and, consequently, cannot compete with Neural Body in its ability to render novel views of moving humans from a few synchronized input images.
The method also favourably compares to human mesh reconstruction techniques such as PIFuHD [SSSJ20], which strongly depends on training 3D data when it comes to the 3D reconstruction of fine appearance details (e.g., rarely-worn or unique garments).
Similar to the Neural Body approach, Neural Actor (NA) [LHR∗21] and HVTR [HYZ∗21] use the SMPL model to represent the deformation states.
They leverage the proxy to explicitly unwarp the surrounding 3D space into a canonical pose, where the NeRF is embedded.
To improve the recovery of high fidelity details in geometry and appearance, they use additional 2D texture maps defined on the SMPL surface, which are used as an additional conditioning to the NeRF MLP.
H-NeRF [XAS21] is another technique for temporal 3D reconstructions of humans with conditioning using an human body model.
Similarly to Neural Body [PZX∗21], they require a sparse set of videos from synchronized and calibrated cameras.
In contrast to it, H-NeRF uses a structured implicit body model with signed distance fields [AXS21], which results in sharper renderings and more complete geometry for challenging subjects.
Similar to H-NeRF, DD-NeRF [YWYZ21] builds on top of a signed distance field to render entire human bodies. Given multi-view input images and a reconstructed SMPL body, they regress SDF and radiance values which are accumulated using volumetric rendering.
HumanNeRF [ZYZ∗21] is also based on multi-view images as input, but learns a generalized neural radiance field for free view-point rendering which can be fine-tuned for a specific actor.
Another work called HumanNeRF [WCS∗22] shows how to train a neural radiance field for a specific actor based on monocular input data, using a skeleton-driven motion field which is refined by a general non-rigid motion field.

Mixture of Volumetric Primitives [LSS∗21] is a model for rendering dynamic, animatable virtual humans in real time. The main idea is to model a scene or object with a set of volumetric primitives that can dynamically change position and content. These primitives model components of the scene like a parts-based model. Each volumetric primitive is a voxel grid produced by a decoder network from a latent code. The code defines the configuration of the scene (e.g., a facial expression, in the case of human faces) which is used by the decoder network to produce primitive locations and voxel values (which contain RGB color and opacity). To render, a raymarching procedure is used to accumulate color and opacity values along the rays corresponding to each pixel. Similar to other dynamic NeRF methods, multi-view video is used as training data. The method is capable of creating extremely high-quality realtime renderings that look realistic even on challenging materials, like hair and clothing.
E-NeRF [LPX∗21] demonstrates an efficient NeRF rendering scheme based on depth-guided sampling. They show realtime rendering on moving humans as well as on static objects using multi-view images as input.

### 4.4 Compositionality and Editing

Method

Required Data

3D Representation

Controllable Parameters

Generative Model

Code

Nguyen-Phuoc et al. [NPLT∗19]
MVI+UIC
V
P,S,T
✓
\faExternalLink

Liu et al. [LZZ∗21]
MVI
V
S,C
✗
\faExternalLink

Jang and Agapito [JA21]
UIC
V
P,S,T
✗
\faExternalLink

Ost et al. [OMT∗21]
VID
V-O
P,S,T,O
✓
\faExternalLink

Zhang et al. [JXX∗21]
MV-VID
V-O
S,C
✓
\faExternalLink

Niemeyer and Geiger [NG21b]
UIC
NFF
P,S,T
✓
\faExternalLink

The methods discussed so far allow reconstructing volumetric representations of static or dynamic scenes and render novel views of them, perhaps from a few input images. They keep the observed scene unchanged, except for comparably straightforward modifications (e.g., foreground removal). Several recent methods also allow editing the reconstructed 3D scenes, i.e., rearranging and affine-transforming the objects and altering their structure and appearance.
In Table 4, we give an overview of the discussed methods.

Conditional NeRF [LZZ∗21] can alter the color and shape of rigid objects observed in 2D images from manual user edits (e.g., it is possible to remove some object parts). This functionality is enabled by a single NeRF trained on multiple object instances of the same category. During editing, the network parameters are adjusted to match the shape and color of a newly observed instance. One of the contributions of this work is finding a subset of tunable parameters which can successfully propagate user edits for novel view generation. This avoids expensive modifications of the entire network. CodeNeRF [JA21] represents shape and texture variations across an object class. Similar to pixelNeRF, CodeNeRF can synthesize novel views of unseen objects. It learns two different embeddings for the shape and texture. At test time, it estimates a camera pose, 3D shape and texture of the object from a single image, and both can be continuously modified by altering their latent codes. CodeNeRF achieves comparable performance to previous methods for single-image 3D reconstruction, while not assuming known camera poses.

Neural Scene Graphs (NSG) [OMT∗21] is a recent method for novel view synthesis from monocular videos recorded while driving (ego-vehicle views). This technique decomposes a dynamic scene with multiple independent rigidly moving objects into a learned scene graph that encodes individual object transformations and radiances. Thus, each object and the background are encoded by different neural networks. In addition, the sampling of the static node is restricted to layered planes (which are parallel to the image plane) for increased efficiency, i.e., a 2.5D representation. NSG requires annotated tracking data for each rigidly moving object of interest over the set of input frames, and each object class (e.g., a car or bus) shares a single volumetric prior. The neural scene graph can then be used to render novel views of the same (i.e., observed) or edited (i.e., by rearranging the objects) scene. Applications of NSG include background-foreground decomposition, enriching training datasets for automotive perception, and improved object detection and scene understanding (see Figure 11).

Another layered representation for editable free-viewpoint videos is introduced in Zhang et al. [JXX∗21]. Their spatially and temporally consistent NeRF (ST-NeRF) relies on bounding boxes for all independently moving and articulated objects—resulting in multiple layers—and disentangles their positions, deformations and appearance. The input to ST-NeRF is a set of 16 synchronized videos from the cameras placed at regular intervals in a half-circle, along with human-background segmentation masks. The method’s name suggests that space-time coherence constraints are reflected in its architecture, i.e., as a space-time deformation module and a NeRF module of the canonical space. ST-NeRF also accepts timestamps to account for the appearance evolving in time. While rendering novel views, the sampling rays are cast through multiple scene layers, which results in accumulated densities and colors. ST-NeRF can be used for neural scene editing such as rescaling, shifting, duplication or removing of the performers, and temporal rearrangements. As promising directions for future work, the authors name reducing the number of input views and enabling non-rigid scene editing.

Note that some of the methods [NG21b, NPLT∗19] discussed in Section 4.2 can be used for scene editing as well.
E.g., GIRAFFE [NG21b] can rotate an object of a known class observed in a single monocular image, change its appearance and translate it along the depth channel.
See Table 4 for a comparison of the methods discussed in this section.

### 4.5 Relighting and Material Editing

Method

Required Data

3D Representation

Controllable Parameters

Models Light Visibility

Models Indirect Illumination

Code

Bi et al. [BXS∗20]
I+L
V
L+M
✓
✗
✗

Zhang et al. [ZLW∗21]
I+M
S
L+M
✗
✗
\faExternalLink

Boss et al. [BBJ∗21]
I+M
V
L+M
✗
✗
\faExternalLink

Srinivasan et al. [SDZ∗21]
I+L
V
L+M
✓
✓
✗

Zhang et al. [ZSD∗21]
I
V
L+M
✓
✗
\faExternalLink

Xiang et al. [XXH∗21]
I+M
V
T
N/A
N/A
✗

The applications we have presented so far are based on the simplified absorption-emission volumetric rendering model discussed in Section 3.2.2, in which the scene is modeled as a volume of particles that block and emit light. While this model is sufficient for rendering images of the scene from novel viewpoints, it is unable to render images of the scene under different lighting conditions. Enabling relighting requires a scene representation that can simulate the transport of light through the volume, including the scattering of light by particles with various material properties.
In Table 5, we give an overview over the discussed methods.

Neural Reflectance Fields [BXS∗20] proposed the first extension of NeRF to enable relighting. Instead of representing a scene as a field of volume density and view-dependent emitted radiance, as in NeRF, Neural Reflectance Fields represent a scene as a field of volume density, surface normals, and bi-directional reflectance distribution functions (BRDFs). This allows for rendering the scene under arbitrary lighting conditions by using the predicted surface normals and BRDFs at each 3D location to evaluate how much incoming light is reflected off particles at that location towards the camera. However, evaluating the visibility from each point along the camera ray to each light source is extremely computationally intensive for neural volumetric rendering models. Even when just considering direct lighting, the MLP must be evaluated at densely-sampled locations between each point along the camera ray and each light source in order to compute the incident lighting to render that ray. Neural Reflectance Fields sidesteps this issue by only training with images of objects illuminated by a single point light that is co-located with the camera, so the MLP only needs to be evaluated along the camera ray.

Other recent works that recover relightable models have avoided the difficulty of computing light source visibility by simply ignoring self-occlusions and assuming that all light sources in the upper hemisphere above any surface are fully visible. Both PhySG [ZLW∗21] and NeRD [BBJ∗21] assume full light source visibility, and further accelerate rendering by representing the environment lighting and scene BRDFs as mixtures of spherical Gaussians, which enables the hemispherical integral of the incoming light multiplied by the BRDF to be computed in closed form. Assuming full light source visibility can work well for objects that are mostly convex, but this strategy is unable to simulate effects such as cast shadows that are due to the occlusion of light sources by scene geometry.

Neural Reflectance and Visibility Fields [SDZ∗21] (NeRV) trains an MLP to approximate the light source visibility for any input 3D location and 2D incoming light direction. Instead of querying an MLP at densely-sampled points along each light ray, the visiblity MLP only needs to be queried a single time for each incoming light direction. This enables NeRV to recover relightable models of scenes from images with significant shadows and self-occlusion effects.

Instead of optimizing a relightable representation from scratch, as done in the previously discussed methods, NeRFactor [ZSD∗21] starts with a pre-trained NeRF model. NeRFactor then recovers a relightable model by simplifying the pre-trained NeRF’s volumetric geometry into a surface model, optimizing MLPs to represent the light source visibility and surface normals at any point on the surface, and finally optimizing a representation of the environment lighting and the BRDF at any surface point; see Figure 12 for an example decomposition. This results in a relightable model that is more efficient when rendering images, since the volumetric geometry has been simplified into a single surface and light-source visibility at any point can be computed by a single MLP query.

The NeROIC technique [KOC∗21] also uses a multi-stage pipline to recover a relightable NeRF-like model from images of an object captured under multiple unconstrained lighting environments. The first stage recovers geometry while explaining appearance variations due to lighting with latent appearance embeddings, the second stage extracts normal vectors from this recovered geometry, and the third stage estimates BRDF properties and a spherical harmonic representation of lighting.

In contrast to the approaches described above, which focus on recovering relightable representations of objects, NeRF-OSR [RES∗21] recovers NeRF-like relightable models of large-scale buildings and historical sites. NeRF-OSR assumes a Lambertian model, and decomposes scenes into diffuse albedo, surface normals, a spherical harmonics representation of lighting, and shadows, which can be combined to relight the scene under novel environment illumination.

The relightable models described above represent scene materials as a continuous 3D field of BRDFs. This enables some basic amount of material editing since the recovered BRDFs can be changed before rendering. NeuTex [XXH∗21] enables more intuitive material editing by introducing a surface parameterization network that learns a mapping from 3D coordinates in the volume to 2D texture coordinates. After a NeuTex model of a scene is recovered, the 2D texture can easily be edited or replaced.

Ref-NeRF [VHM∗21] focuses on improving NeRF’s ability to represent and render specular surfaces. Although Ref-NeRF cannot be used for relighting as it does not disentangle incoming light from reflectance properties, it structures outgoing light into physically-meaningful components (diffuse and specular colors, normal vectors, and roughness) that enable intuitive material editing applications.

Guo et al. [GKB∗21] extends NeRF to handle reflections and propose to split a scene into transmitted and reflected components which are represented as separate neural radiance fields. While it does not allow scene editing, it is able to handle reflections from glas and mirrors.

### 4.6 Light Fields

Volume rendering, sphere-tracing, and other 3D rendering forward models can yield photo-realistic results. However, for a given ray, they all require the sampling of the underlying 3D scene representation at whatever 3D coordinate that ray first intersects the scene’s geometry. As this intersection point is not known a-priori, ray-marching algorithms first have to discover that surface point. Ultimately, this yields a time and memory complexity that scales with the geometric complexity of the scene, where more and more points have to be sampled to render more and more complex scenes. In practice, these are hundreds or even thousands of points per ray. Moreover, accurately rendering reflections and second-order lighting effects requires multi-bounce ray-tracing, such that for every pixel, many rays have to be traced instead of only a single one. This yields a high computational burden. While in the regime of reconstructing a single scene (overfitting), this may be circumvented by smart data structures, hashing, and expert low-level engineering, in the regime of reconstructing a 3D scene given just few observations or even just a single image, such data structures hinder the application of learned reconstruction algorithms, such as inferring the parameters of the 3D scene from a single image using convolutional neural networks.

A pair of concurrent works [SRF∗21, LLYX21] thus introduced the idea of parametrizing light fields via coordinate-based networks. Specifically, Light Field Networks [SRF∗21] paramaterize a 3D scene not via its 3D radiance field, but instead via its 360-degree light field, i.e., a function that maps every oriented ray directly to the color observed by that ray. Concurrently, Liu et al. [LLYX21] proposed to parameterize a fronto-parallel light field for novel view synthesis of forward-facing scenes as a neural field. Representing a scene via its light field obviates the need for ray-marching, as to render a single pixel, the light field can be sampled by the corresponding camera ray and directly yields the pixel color. It further obviates the need for multi-bounce ray-tracing, as reflections are similarly absorbed by the light field. On the flip-side, this loses the guarantee of multi-view consistency: where a 3D renderer is guaranteed to map a single 3D coordinate to single value, a neural light field may map two rays that hit the same point in the scene to two different colors. This has to be addressed by additional means.

Sitzmann et al. [SRF∗21] propose to generalize across scenes by conditioning the neural light field on a latent code, thus learning a space of multi-view consistent light fields, however constrained to simple scenes due to their use of global conditioning. Sajjadi et al. [SMP∗21] follow the prior-based inference approach and use a transformer to parameterize the 360-degree light fields of scenes, inferred from few image observations, achieving novel view synthesis for complex, real-world scenes. Attal et al. [AHZ∗21], Ost et al. [OLN∗21], Liu et al. [LLYX21] and Suhail et al. [SESM21] instead investigate the same paradigms as NeRF, i.e., reconstruction and novel view synthesis of a single 3D scene. To ensure multi-view consistency, Suhail et al. [SESM21] leverage pixel-aligned CNN features as in PixelNeRF [YYTK21] for 3D points along a ray, which are accumulated with a transformer. This yields significant improvements over NeRF, but still requires sampling in 3D and is several times slower. Ost et al. [OLN∗21] relies on a coarse 3D reconstruction of the 3D scene in form of a point cloud to parameterize point-wise light fields. Attal et al. [AHZ∗21] rely on storing features in a voxel grid, where every feature parameterizes the local light field of rays intersecting that voxel, and render via volume rendering. Liu et al. [LLYX21] leverage regularization to ensure multi-view consistency.

### 4.7 Engineering Frameworks

Working with neural rendering models poses notable engineering challenges for practitioners:
large amounts of image and video data must be processed in a highly non-sequential manner, and the models often require differentiation of large and complex computational graphs. Developing efficient operators often requires working with low-level languages which at the same time makes it harder to use automatic differentiation. In this section, we will discuss recent advances in tools that can help to overcome problems across the entire software stack relevant for neural rendering.

#### 4.7.1 Storage

Saturating a GPU with data in particular for neural rendering is challenging: often, each pixel of images or videos is treated as a separate data point. Methods require random iteration over the entire pool of pixels in the dataset, in case of temporal reconstruction often across the entire sequence for a single batch. Flexible storage solutions should take this into account.

NVIDIA AIStore [AMB19] is a general purpose storage solution that allows to monitor throughput per drive and implements tiered architectures for loading and shuffling, while abstracting these layers away from the user. Independent of the storage backend, sharding has tremendous benefits through 1) allowing to shuffle data in memory while 2) using mostly sequential reads within the shards. Tensorflow [AAB∗15] has built-in support sharded storage through the tfrecord file format, whereas webdataset 
\faExternalLink
 offers similar convenient features for PyTorch [PGM∗19].

#### 4.7.2 Hyperparameter Search and Experiments

With long runtimes and complex configuration hierarchies, neural rendering experiments require good techniques for experiment management and hyperparameter search. Hydra [Yad19] excels at configuring even the most complex experiments and offers integrated support for hyperparameter search, for example using the AX adaptive experimentation framework
\faExternalLink
.
However, running all experiments for a sweep until convergence, even for smartly picked parameters using Bayesian hyperparameter search, might be too time consuming. Ray tune [LLN∗18] has implementations of algorithms like ASHA [LJR∗20] and Hyperband [LJRT18], which can dynamically assign computational and time budgets to experiments for a faster hyperparameter search.

#### 4.7.3 Differentiable Rendering and Autodiff

Neural Rendering has high demands towards differentiability: complex computational graphs need to be built and, depending on the application, be executed either on large inputs vectorized (macro AD—for brevity we refer to auto-differentiation as AD throughout this section) or on large amounts of small inputs (micro AD). Depending on the application, the AD package might have to be used low level (e.g., in CUDA), or high level (e.g., in Python). A powerful AD library for C++ is STAN [CHB∗15]. We refer to the accompanying paper for a comprehensive overview and evaluation of AD libraries until its publication in 2015, which is beyond the scope of this article. A noteworthy more recent AD package for C++17 is the autodiff 
\faExternalLink

package. Enzyme AD [MC, MCP∗21] is taking a particularly versatile approach for low-level AD: it leverages the LLVM ecosystem as a whole. This is particularly powerful, because of the concept of frontends, the LLVM IR and backends. In broad strokes, LLVM frontends translate a language, for example C++, to the LLVM intermediate representation (IR). This representation is an abstract, language-agnostic representation of low-level commands, and it is the same for all frontends. This is where Enzyme comes in: it is an extension that can create derivatives of functions in this IR. That means that it works for all languages that LLVM supports. LLVM backends emit code from the IR: this could be for x86, ARM or GPU processors. This means, that Enzyme supports a variety of processors, including GPUs. Another C++ package specifically for processing images and graphics is Halide [LGA∗18]. Its standout feature is flexible scheduling for parallel processing of pixels.

Difftaichi [HAL∗20] offers differentiable programming in Python for physical simulation with applications in rendering. Enoki [Jak19] is a very versatile and high performance AD component for physically-based differentiable rendering and is the core component of the Mitsuba 2 renderer [NDVZJ19]. Jax [BFH∗18] is a Python framework for differentiable and accelerated linear algebra with compilation options for GPUs and TPUs. JaxNeRF 
\faExternalLink

is a reference implementation for NeRF using Jax. The Swift programming language provides AD as a first class use case 
\faExternalLink
, and was heavily used for developing a Tensorflow integration 
\faExternalLink
.

#### 4.7.4 Raycasting and Rendering

Several packages exist for providing high-level rendering and aggregation primitives. NVIDIA OptiX 
\faExternalLink

is a high performance library for ray-casting and ray-intersection and provides to date the only possibility to use the hardware acceleration on NVIDIA RTX hardware for ray intersection. Teg [BMM∗21] is a differentiable programming language which provides primitives for optimizing integrals with discontinuous integrands, as frequently found in rendering. Redner [LADL18b] is a framework for differentiable ray tracing; Mitsuba 2 [NDVZJ19] provides an even more general framework for physically based differentiable rendering and path tracing. psdr-cuda [LZBD21] improves over Redner by using better gradient calculation techniques and sampling strategies. PyTorch3D [RRN∗20] offers a broad suite of tools around differentiable rendering and graphics, tightly integrated with PyTorch. Tensorflow Graphics [VKP∗19] has a similar goal for Tensorflow.

## 5 Open Challenges

After covering a wide variety of computer graphics and vision problems to which neural volumetric representations can be successfully applied, we now take a look at problems where only classical representations have been used. Thus, there are various avenues for future research. We further discuss multiple open challenges in the field. Many of the points discussed in the following are related to each other.

Seamless Integration and Usage. Most computer graphics algorithms and techniques developed over more than half a century assume meshes or point clouds as 3D scene representations for rendering and editing. In contrast, neural rendering is such a young field that this notion was used for the first time just a few years ago in 2018 [ERB∗18]. Thus, inevitably, there is still a gap between the spectrum of available methods that can operate on classical 3D representations and those that are applicable to neural representations. Furthermore, many methods exist to edit classical representations, e.g., widely-used tools such as Blender [Com18] and Maya [Aut] support meshes and texture maps, whereas their counterparts for neural representations have to be developed from scratch. On the other hand, it is foreseeable that this gap will decrease with further improvement in the field and more and more widespread adoption and integration of neural representations. Moreover, modern hardware accelerators are designed for classical computer graphics and could in the future be similarly tailored to neural representations.

Another related challenge is interpretability of the learned representations, which concerns deep learning in general. Thus, learned neural network weights are notoriously hard to interpret in terms of the target quantities (e.g., point colors and opacities in the 3D space). At the same time, they aim to replace the graphics pipeline, which is well understood and relies on analytically derived steps.
Ultimately, to improve controllability and enable seamless integration of learned volumetric models in computer graphics tools, we would like to be able to modify the scene parametrization to change the scene in a desired direction. While this is likely not tractable for arbitrary scenes parametrized by global MLPs, composing a full scene out of local neural representations might make it tractable by opening up the intriguing possibility of re-introducing aspects of classical graphics.

Scalability. Most of the works on volumetric neural rendering focus on single objects and relatively simple composite scenes (e.g., a human and a background, several humans in the same environment, a street with moving cars) with or without background. Learning neural representations for large-scale scenes—which can only be partially observed in each input frame—is still challenging. Although the first successful and impressive steps in this direction have been made (we refer here to Nerf in the Wild [MBRS∗21], NeRF-OSR [RES∗21] and the neural SLAM system iMAP [SLOD21], see Figure 13), many open challenges remain. For instance, the approaches for scene editing, relighting, and compositionality developed for single objects cannot be straightforwardly extended to handle large-scale scenes. Moreover, a global representation for large-scale environments becomes unfeasible starting from some scene size, even when applying space partitioning policies such as those used in PlenOctrees [YLT∗21]. Thus, a new generation of storage and retrieval techniques need to be developed for efficient neural models for large-scale scenes, along the lines of VoxelHashing [NZIS13] for TSDFs. First, they should make the scene completion more efficient (i.e., without the need to constantly recompute the entire model from scratch) and, second, enable easy retrieval of partial contents. Both these points are related to the open challenge of interpretability discussed above.

Generalizability. Only a few initial but promising methods exist for generalizable and instantiable volumetric neural representations. For example, StereoNeRF [CBLPM21] uses only a dozen spread-out views to generate novel views of a rigid scene with the visual accuracy comparable to the original NeRF [MST∗20] after fine-tuning, while pixelNeRF [YYTK21] can infer volumetric models of rigid scenes unseen at training time just from a single image. This class of approaches is data-driven and requires large-scale multi-view datasets with a sufficiently wide baseline. Consequently, these methods can produce views at arbitrary novel viewpoints if the datasets provide sufficient viewpoint coverage. Reducing this strong dependency is an exciting direction for future work. Another open challenge is the generalizability of instantiable approaches to scenes with non-rigid deformations. The inputs can be sparse sets of spatiotemporal observations or even single images at the extreme (in this case, the task becomes scene animation from a single image). One straightforward direction towards such techniques would be relying on multi-view datasets of deformable scenes, which would likely increase required dataset sizes by a multitude. Another possible way would be to disentangle deformation modes and scene shapes and appearances at rest.
Furthermore, while there exists some work on generating neural scene representations (e.g., using hypernets [SRF∗21]), there is less progress on designing neural operators that take neural scene representations as input to work on them, for example to complete a partial scene or to regress semantic labels for an existing representation. No operator analogous to mesh convolutions for meshes or 3D convolutions for voxel grids exists. Such an operator would ideally be trained only once and then be generally applicable.

Multi-Modal Learning.
Multi-modal learning means going beyond visual signals and incorporating other data types such as semantics, textual descriptions and sound.
For example, telepresence and augmented reality would highly benefit from a method that can not only render novel views of dynamically interacting and talking humans but also synthesize the corresponding novel sounds;
existing work can, for instance, synthesize stereo audio from mono audio inputs [RMG∗21].
Synthesizing textual descriptions and semantics of the scene (e.g., semantic segmentation labels) can be very useful for downstream applications based on volumetric neural representations. While some prior work addresses this goal [KSW20, ZLLD21], this remains an open challenge.
More detailed and scenario-specific modeling could take into account such information as the camera capture system (e.g., as already shown in TöRF [ALG∗21] for depth cameras), whether the camera is using rolling or global shutter, or if there is motion blur in the input images. Other sensors like IMUs, Lidar, or event streams could all potentially be modeled in a continuous fashion. (Ultrasound and x-rays could be continuously modeled at arbitrary resolution for medical imaging.) It is also conceivable to optimize for certain capture properties that are not trivial to measure like color calibrations of a multi-view capture setup.
This extends to physical simulation in general, where neural scene representations offer an exciting venue to “learn less and know more" by incorporating differentiable physics simulators; e.g., for physically motivated deformation models or physically correct light transport.

Other Questions.
Can we increase quality? Reconstructing objects with many high-frequency details, shading, and view-dependent appearance remains a largely unsolved problem.
Can we decrease training time?
Although there has been progress on very fast inference for novel view synthesis at test time, improving the training time remains a big challenge.
The work by Lange and Kutz [LK21] is one of the first encouraging steps towards this goal. It introduces fast computational neural network layers that rely on series expansions and fast summation algorithms. Thus, the proposed integral-implicit layer reduces the required computational performance to train a NeRF model by two orders of magnitude (∼150similar-toabsent150{\sim}150x reduction in FLOPs per epoch).
Are fewer input images sufficient? Fewer input views might be sufficient to reach a similar visual fidelity as fully converged models requiring hundreds of views. Currently, partial observations (e.g., parts of the scene observed only in a subset of images) tend to be blurrier than the rest of the scene.

Beyond the immediate use case of AR/VR, there is little research on using neural scene representations in other contexts like robotics, with the notable exception of the real-time SLAM system iMAP [SLOD21] supporting single-room environments (see Figure 13).
How can we obtain, incorporate, and predict object affordances or other annotations like temperature? Are there advantages to using neural scene representations for motion prediction or planning?

The list of future directions discussed in this section does not aim for completeness. We expect to see many improvements on more aspects of coordinate-based neural volumetric representations already in the near future.

## 6 Social Implications

Neural approaches discussed in this state-of-the-art report achieve a very high degree of realism for synthesized novel views.
Rapid developments in the field already influence and will continue influencing society in many positive and potentially negative ways which we discuss in this section.

Research and Industry. The fields which are starkly impacted by the new volumetric neural representations are computer vision, computer graphics as well as augmented and virtual reality, which can benefit from increased photo-realism of rendered environments. The fact that the state-of-the-art volumetric models rely on well-understood and elegant principles lowers the barrier to entry for research on photogrammetry and 3D reconstruction. Moreover, this effect is magnified by the ease of use of the methods and publicly available
codebases and datasets.

Since neural rendering is still not mature and well understood, end-user tools like Blender do not yet exist, putting these novel methods out of reach for both 3D hobbyists and industry as of now. However, more widespread understanding of the technology inevitably impacts developed products and applications. With that, we foresee decreased effort in content creation for games and special effects for movies. The possibility to render photo-realistic novel views of a scene from a few input images is a significant advantage compared to existing technology. This can potentially reshape the entire established pipeline for content design used in the visual effects (VFX) industry.

Trustworthiness. However, at the same time, photo-realism creates the possibility to misuse the technology and create synthetic content that malicious actors may falsely claim to be real, in particular, when neural rendering approaches focus on human faces [TZS∗16, TZN19, GTZN21].
In response to these potential misuses, methods to automatically detect such fake content are being developed by the research community [CRT∗21, RCV∗19]
and security measures including encryption and block chain measures are being explored.
And there are a number of other mitigations that could be explored to minimize these risks.
For example, while there are cases where we expect users not to object to seeing synthetic photo-real content (e.g., when watching movies), synthetic content could be labelled or otherwise identified as such to inform users.
Further user studies could investigate people’s judgement of the need to label synthetic content in different contexts.
On the collection side, people could provide explicit and informed consent that their identity can be used for creating synthetic content in a specified context.

Environment. Since current neural volumetric scene representations are deep-learning-based, the GPUs used for training them consume a sizable amount of energy.
Since more and more laboratories are working on neural rendering, the use of high-end and multi-GPU systems increases accordingly.
If the resources for manufacturing and the electricity for operating the GPU clusters are not taken predominantly from renewable sources, training volumetric neural representations can negatively influence the environment and global climate in the long term. In an attempt to soften the need for computational resources and hence electricity and hardware, there are many architectures that require less compute power for training than NeRF-based methods.
Last but not least, high GPU demand potentially implies that not all groups can afford to contribute on equal footing as experimenting with volumetric representations is not the most lightweight task.

## 7 Conclusion

In this state-of-the-art report, we have reviewed the recent trends on neural rendering techniques.
The methods covered learn 3D neural scene representations based on 2D observations as inputs for training, and enable synthesis of photo-realistic imagery with control over different scene parameters.
The field of neural rendering has seen rapid progress during the last few years and continues to grow fast.
Its applications range from free-viewpoint videos of rigid and non-rigid scenes to shape and material editing, relighting, and human avatar generation, among many others.
These applications have been discussed in detail in this report.

At the same time, we believe that neural rendering is still an emerging field with many open challenges that can be addressed.
To this end, we identify and discuss multiple directions for future research.
In addition, we discuss social implications, which arise from the democratization of neural rendering along with its capability to synthesize photo-realistic image content.
Overall, we conclude that neural rendering is an exciting field, which is inspiring thousands of researchers across many communities to tackle some of computer graphics’ hardest problems, and we look forward to seeing further developments on the topic.

## 8 Acknowledgements

A. Tewari, V. Golyanik, and C. Theobalt are supported in part by the ERC Consolidator Grant 4DReply (770784).
E. Tretschk is supported by a Reality Labs Research grant.
M. Nießner is supported by the ERC Starting Grant Scan2CAD (804724).

## References

- [AAB∗15]

Abadi M., Agarwal A., Barham P., Brevdo E., Chen Z., Citro C., Corrado
G. S., Davis A., Dean J., Devin M., Ghemawat S., Goodfellow I., Harp A.,
Irving G., Isard M., Yangqing J., Jozefowicz R., Kaiser L., Kudlur M.,
Levenberg J., Mané D., Monga R., Moore S., Murray D., Olah C., Schuster
M., Shlens J., Steiner B., Sutskever I., Talwar K., Tucker P., Vanhoucke V.,
Vasudevan V., Viégas F., Vinyals O., Warden P., Wattenberg M., Wicke
M., Yu Y., Zheng X.:

TensorFlow: Large-Scale Machine Learning on Heterogeneous
Systems.

http://tensorflow.org/, 2015.

- [AHZ∗21]

Attal B., Huang J.-B., Zollhoefer M., Kopf J., Kim C.:

Learning neural light fields with ray-space embedding networks.

arXiv preprint arXiv:2112.01523 (2021).

- [AL20]

Atzmon M., Lipman Y.:

Sal: Sign agnostic learning of shapes from raw data.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020), pp. 2565–2574.

- [ALG∗20]

Attal B., Ling S., Gokaslan A., Richardt C., Tompkin J.:

MatryODShka: Real-time 6DoF video view synthesis using
multi-sphere images.

In Proc. ECCV (Aug. 2020).

URL: https://visual.cs.brown.edu/matryodshka.

- [ALG∗21]

Attal B., Laidlaw E., Gokaslan A., Kim C., Richardt C., Tompkin J.,
O’Toole M.:

Törf: Time-of-flight radiance fields for dynamic scene view
synthesis.

In Neural Information Processing Systems (NeurIPS) (2021).

- [ALKN19]

Azinovic D., Li T.-M., Kaplanyan A., Nießner M.:

Inverse path tracing for joint material and lighting estimation.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2019), pp. 2447–2456.

- [AMB19]

Aizman A., Maltby G., Breuel T.:

High Performance I/O For Large Scale Deep Learning.

IEEE International Conference on Big Data (2019), 5965–5967.

- [AMBG∗21]

Azinovic D., Martin-Brualla R., Goldman D. B., Nießner M., Thies
J.:

Neural rgb-d surface reconstruction.

- [ASK∗20a]

Aliev K.-A., Sevastopolsky A., Kolos M., Ulyanov D., Lempitsky V.:

Neural point-based graphics.

arXiv:2110.06635.

- [ASK∗20b]

Aliev K.-A., Sevastopolsky A., Kolos M., Ulyanov D., Lempitsky V.:

Neural point-based graphics.

In Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part XXII 16 (2020),
Springer, pp. 696–712.

- [Aut]

Autodesk, INC.:

Maya.

URL: https://autodesk.com/maya.

- [AXS21]

Alldieck T., Xu H., Sminchisescu C.:

imghum: Implicit generative models of 3d human shape and articulated
pose.

In International Conference on Computer Vision (ICCV) (2021).

- [BBJ∗21]

Boss M., Braun R., Jampani V., Barron J. T., Liu C., Lensch H. P. A.:

NeRD: Neural reflectance decomposition from image collections.

ICCV (2021).

- [BFH∗18]

Bradbury J., Frostig R., Hawkins P., Johnson M. J., Leary C., Maclaurin
D., Necula G., Paszke A., VanderPlas J., Wanderman-Milne S., Zhang Q.:

JAX: composable transformations of Python+NumPy programs,
2018.

URL: http://github.com/google/jax.

- [BFO∗20]

Broxton M., Flynn J., Overbeck R., Erickson D., Hedman P., Duvall M.,
Dourgarian J., Busch J., Whalen M., Debevec P.:

Immersive light field video with a layered mesh representation.

ACM Trans. Graph. (SIGGRAPH) 39, 4 (2020).

- [BGP∗21]

Baatz H., Granskog J., Papas M., Rousselle F., Novák J.:

Nerf-tex: Neural reflectance field textures.

In Eurographics Symposium on Rendering (June 2021), The
Eurographics Association.

- [BKK19]

Bai S., Kolter J. Z., Koltun V.:

Deep equilibrium models.

NeurIPS (2019).

- [BKW21]

Bergman A. W., Kellnhofer P., Wetzstein G.:

Fast training of neural lumigraph representations using meta
learning.

In Proceedings of the IEEE International Conference on Neural
Information Processing Systems (NeurIPS) (2021).

- [BMM∗21]

Bangaru S., Michel J., Mu K., Bernstein G., Li T.-M., Ragan-Kelley J.:

Systematically differentiating parametric discontinuities.

ACM Trans. Graph. 40, 107 (2021), 107:1–107:17.

- [BMT∗21]

Barron J. T., Mildenhall B., Tancik M., Hedman P., Martin-Brualla R.,
Srinivasan P. P.:

Mip-nerf: A multiscale representation for anti-aliasing neural
radiance fields.

ICCV (2021).

- [BMV∗21]

Barron J. T., Mildenhall B., Verbin D., Srinivasan P. P., Hedman P.:

Mip-nerf 360: Unbounded anti-aliased neural radiance fields.

arXiv (2021).

- [BNT21]

Burov A., Nießner M., Thies J.:

Dynamic surface function networks for clothed human bodies.

- [BXS∗20]

Bi S., Xu Z., Srinivasan P. P., Mildenhall B., Sunkavalli K., Hašan
M., Hold-Geoffroy Y., Kriegman D., Ramamoorthi R.:

Neural reflectance fields for appearance acquisition.

arXiv:2008.03824.

- [CBC∗01a]

Carr J. C., Beatson R. K., Cherrie J. B., Mitchell T. J., Fright W. R.,
McCallum B. C., Evans T. R.:

Reconstruction and representation of 3d objects with radial basis
functions.

In Proceedings of the 28th Annual Conference on Computer
Graphics and Interactive Techniques (New York, NY, USA, 2001), SIGGRAPH ’01,
Association for Computing Machinery, p. 67–76.

URL: https://doi.org/10.1145/383259.383266, doi:10.1145/383259.383266.

- [CBC∗01b]

Carr J. C., Beatson R. K., Cherrie J. B., Mitchell T. J., Fright W. R.,
McCallum B. C., Evans T. R.:

Reconstruction and representation of 3d objects with radial basis
functions.

In Proceedings of the 28th annual conference on Computer
graphics and interactive techniques (2001), pp. 67–76.

- [CBLPM21]

Chibane J., Bansal A., Lazova V., Pons-Moll G.:

Stereo radiance fields (srf): Learning view synthesis from sparse
views of novel scenes.

In Computer Vision and Pattern Recognition (CVPR) (2021).

- [CHB∗15]

Carpenter B., Hoffman M. D., Brubaker M., Lee D., Li P., Betancourt
M.:

The Stan Math Library: Reverse-Mode Automatic Differentiation in
C++.

URL: http://arxiv.org/abs/1509.07164, arXiv:1509.07164.

- [Chu06]

ChumpusRex:

Craniale computertomographie, 2006.

URL:
https://de.wikipedia.org/wiki/Computertomographie##/media/Datei:Ct-workstation-neck.jpg.

- [CKS∗17]

Chaitanya C. R. A., Kaplanyan A. S., Schied C., Salvi M., Lefohn A.,
Nowrouzezahrai D., Aila T.:

Interactive reconstruction of monte carlo image sequences using a
recurrent denoising autoencoder.

ACM Trans. Graph. 36, 4 (July 2017), 98:1–98:12.

URL: http://doi.acm.org/10.1145/3072959.3073601, doi:10.1145/3072959.3073601.

- [CL96]

Curless B., Levoy M.:

A volumetric method for building complex models from range images.

In Proceedings of the 23rd annual conference on Computer
graphics and interactive techniques (1996), pp. 303–312.

- [CLC∗22]

Chan E. R., Lin C. Z., Chan M. A., Nagano K., Pan B., Mello S. D.,
Gallo O., Guibas L., Tremblay J., Khamis S., Karras T., Wetzstein G.:

Efficient geometry-aware 3D generative adversarial networks.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2022).

- [CLI∗20]

Chabra R., Lenssen J. E., Ilg E., Schmidt T., Straub J., Lovegrove S.,
Newcombe R.:

Deep local shapes: Learning local sdf priors for detailed 3d
reconstruction.

In European Conference on Computer Vision (Proceedings of the
European Conference on Computer Vision) (2020).

- [CLX∗21]

Chen A., Liu R., Xie L., Chen Z., Su H., Jingyi Y.:

Sofgan: A portrait image generator with dynamic styling.

ACM Trans. Graph. 41, 1 (2021).

URL: https://doi.org/10.1145/3470848, doi:10.1145/3470848.

- [CMK∗21]

Chan E., Monteiro M., Kellnhofer P., Wu J., Wetzstein G.:

pi-gan: Periodic implicit generative adversarial networks for
3d-aware image synthesis.

In CVPR (2021).

- [Com18]

Community B. O.:

Blender - a 3D modelling and rendering package.

Blender Foundation, Stichting Blender Foundation, Amsterdam, 2018.

URL: http://www.blender.org.

- [CRBD18]

Chen R. T. Q., Rubanova Y., Bettencourt J., Duvenaud D. K.:

Neural ordinary differential equations.

In Advances in Neural Information Processing Systems (2018),
vol. 31.

- [CRT∗21]

Cozzolino D., Rossler A., Thies J., Nießner M., Verdoliva L.:

Id-reveal: Identity-aware deepfake video detection.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (2021), pp. 15108–15117.

- [CTZ20]

Chen Z., Tagliasacchi A., Zhang H.:

Bsp-net: Generating compact meshes via binary space partitioning.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020), pp. 45–54.

- [CW93]

Chen S. E., Williams L.:

View interpolation for image synthesis.

In SIGGRAPH (1993), pp. 279–288.

- [CZ19]

Chen Z., Zhang H.:

Learning implicit fields for generative shape modeling.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2019), pp. 5939–5948.

- [CZB∗21]

Chen X., Zheng Y., Black M. J., Hilliges O., Geiger A.:

Snarf: Differentiable forward skinning for animating non-rigid neural
implicit shapes, 2021.

arXiv:2104.03953.

- [CZL∗21]

Chen X., Zhang Q., Li X., Chen Y., Feng Y., Wang X., Wang J.:

Hallucinated neural radiance fields in the wild, 2021.

arXiv:2111.15246.

- [DGY∗20]

Deng B., Genova K., Yazdani S., Bouaziz S., Hinton G., Tagliasacchi
A.:

Cvxnet: Learnable convex decomposition.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020), pp. 31–44.

- [DLZR21]

Deng K., Liu A., Zhu J.-Y., Ramanan D.:

Depth-supervised nerf: Fewer views and faster training for free.

arXiv preprint arXiv:2107.02791 (2021).

- [DNJ20]

Davies T., Nowrouzezahrai D., Jacobson A.:

Overfit neural networks as a compact shape representation, 2020.

arXiv:2009.09808.

- [DYXT21]

Deng Y., Yang J., Xiang J., Tong X.:

Gram: Generative radiance manifolds for 3d-aware image generation.

In arXiv (2021).

- [DZW∗20]

Duan Y., Zhu H., Wang H., Yi L., Nevatia R., Guibas L. J.:

Curriculum deepsdf.

In European Conference on Computer Vision (2020), Springer,
pp. 51–67.

- [DZY∗21]

Du Y., Zhang Y., Yu H.-X., Tenenbaum J. B., Wu J.:

Neural radiance flow for 4d view synthesis and video processing.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (2021).

- [EGO∗20]

Erler P., Guerrero P., Ohrhallinger S., Mitra N. J., Wimmer M.:

Points2surf learning implicit surfaces from point clouds.

In Proceedings of the European Conference on Computer Vision
(2020), Springer, pp. 108–124.

- [ERB∗18]

Eslami S. A., Rezende D. J., Besse F., Viola F., Morcos A. S., Garnelo
M., Ruderman A., Rusu A. A., Danihelka I., Gregor K., et al.:

Neural scene representation and rendering.

Science 360, 6394 (2018), 1204–1210.

- [FBD∗19]

Flynn J., Broxton M., Debevec P., DuVall M., Fyffe G., Overbeck R.,
Snavely N., Tucker R.:

Deepview: View synthesis with learned gradient descent.

In Proc. Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (2019), pp. 2367–2376.

- [FNPS16]

Flynn J., Neulander I., Philbin J., Snavely N.:

Deep stereo: Learning to predict new views from the
world’s imagery.

In Proc. Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (2016).

- [FXW∗21]

Fang J., Xie L., Wang X., Zhang X., Liu W., Tian Q.:

Neusample: Neural sample field for efficient view synthesis, 2021.

arXiv:2111.15552.

- [GCL∗21]

Guo Y., Chen K., Liang S., Liu Y., Bao H., Zhang J.:

Ad-nerf: Audio driven neural radiance fields for talking head
synthesis.

In IEEE/CVF International Conference on Computer Vision (ICCV)
(2021).

- [GCS∗20]

Genova K., Cole F., Sud A., Sarna A., Funkhouser T.:

Local deep implicit functions for 3d shape.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020), pp. 4857–4866.

- [GCV∗19]

Genova K., Cole F., Vlasic D., Sarna A., Freeman W. T., Funkhouser T.:

Learning shape templates with structured implicit functions.

In Proceedings of the International Conference on Computer
Vision (2019), pp. 7154–7164.

- [GKB∗21]

Guo Y.-C., Kang D., Bao L., He Y., Zhang S.-H.:

Nerfren: Neural radiance fields with reflections, 2021.

arXiv:2111.15234.

- [GKJ∗21]

Garbin S. J., Kowalski M., Johnson M., Shotton J., Valentin J.:

Fastnerf: High-fidelity neural rendering at 200fps.

arXiv preprint arXiv:2103.10380 (2021).

- [GLWT21a]

Gu J., Liu L., Wang P., Theobalt C.:

Stylenerf: A style-based 3d-aware generator for high-resolution image
synthesis, 2021.

arXiv:2110.08985.

- [GLWT21b]

Gu J., Liu L., Wang P., Theobalt C.:

Stylenerf: A style-based 3d-aware generator for high-resolution image
synthesis, 2021.

arXiv:2110.08985.

- [GPAM∗14]

Goodfellow I., Pouget-Abadie J., Mirza M., Xu B., Warde-Farley D.,
Ozair S., Courville A., Bengio Y.:

Generative adversarial nets.

In Advances in Neural Information Processing Systems (2014),
Ghahramani Z., Welling M., Cortes C., Lawrence N., Weinberger K. Q., (Eds.),
vol. 27, Curran Associates, Inc.

URL:
https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf.

- [GSHG98]

Greger G., Shirley P., Hubbard P. M., Greenberg D. P.:

The irradiance volume.

IEEE Computer Graphics and Applications 18, 2 (1998), 32–43.

- [GSKH21]

Gao C., Saraf A., Kopf J., Huang J.-B.:

Dynamic view synthesis from dynamic monocular video.

Proceedings of the IEEE International Conference on Computer
Vision (2021).

- [GSL∗20]

Gao C., Shih Y., Lai W.-S., Liang C.-K., Huang J.-B.:

Portrait neural radiance fields from a single image.

arXiv preprint arXiv:2012.05903 (2020).

- [GTZN21]

Gafni G., Thies J., Zollhöfer M., Nießner M.:

Dynamic neural radiance fields for monocular 4d facial avatar
reconstruction.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (June 2021), pp. 8649–8658.

- [GYH∗20]

Gropp A., Yariv L., Haim N., Atzmon M., Lipman Y.:

Implicit geometric regularization for learning shapes.

arXiv preprint arXiv:2002.10099 (2020).

- [HAL∗20]

Hu Y., Anderson L., Li T.-M., Sun Q., Carr N., Ragan-Kelley J., Durand
F.:

Difftaichi: Differentiable programming for physical simulation.

ICLR (2020).

- [Har96]

Hart J. C.:

Sphere tracing: A geometric method for the antialiased ray tracing of
implicit surfaces.

The Visual Computer 12, 10 (1996), 527–545.

- [HCC∗14]

Hannun A., Case C., Casper J., Catanzaro B., Diamos G., Elsen E.,
Prenger R., Satheesh S., Sengupta S., Coates A., Y. Ng A.:

DeepSpeech: Scaling up end-to-end speech recognition.

- [HDD∗92]

Hoppe H., DeRose T., Duchamp T., McDonald J., Stuetzle W.:

Surface reconstruction from unorganized points.

SIGGRAPH (1992).

- [HLA∗19]

Hu Y., Li T.-M., Anderson L., Ragan-Kelley J., Durand F.:

Taichi: a language for high-performance computation on spatially
sparse data structures.

ACM Transactions on Graphics (TOG) 38, 6 (2019), 201.

- [HPX∗21]

Hong Y., Peng B., Xiao H., Liu L., Zhang J.:

Headnerf: A real-time nerf-based parametric head model, 2021.

arXiv:2112.05637.

- [HRRR18]

Henzler P., Rasche V., Ropinski T., Ritschel T.:

Single-image tomography: 3d volumes from 2d cranial x-rays.

In Eurographics (2018).

- [HSM∗21]

Hedman P., Srinivasan P. P., Mildenhall B., Barron J. T., Debevec P.:

Baking neural radiance fields for real-time view synthesis.

arXiv (2021).

- [HSW89]

Hornik K., Stinchcombe M., White H.:

Multilayer feedforward networks are universal approximators.

Neural Networks 2, 5 (1989), 359–366.

URL:
https://www.sciencedirect.com/science/article/pii/0893608089900208,
doi:https://doi.org/10.1016/0893-6080(89)90020-8.

- [HYZ∗21]

Hu T., Yu T., Zheng Z., Zhang H., Liu Y., Zwicker M.:

Hvtr: Hybrid volumetric-textural rendering for human avatars.

arXiv:2112.10203.

- [HZF∗21]

Huang X., Zhang Q., Feng Y., Li H., Wang X., Wang Q.:

Hdr-nerf: High dynamic range neural radiance fields.

arXiv (December 2021).

- [ID18]

Insafutdinov E., Dosovitskiy A.:

Unsupervised learning of shape and pose with differentiable point
clouds.

In Proceedings of the IEEE International Conference on Neural
Information Processing Systems (NeurIPS) (2018), pp. 2802–2812.

- [IKH∗11]

Izadi S., Kim D., Hilliges O., Molyneaux D., Newcombe R., Kohli P.,
Shotton J., Hodges S., Freeman D., Davison A., Fitzgibbon A.:

Kinectfusion: Real-time 3d reconstruction and interaction using a
moving depth camera.

In UIST ’11 Proceedings of the 24th annual ACM symposium on
User interface software and technology (October 2011), ACM, pp. 559–568.

- [JA21]

Jang W., Agapito L.:

Codenerf: Disentangled neural radiance fields for object categories.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) (October 2021), pp. 12949–12958.

- [JAC∗21]

Jeong Y., Ahn S., Choy C., Anandkumar A., Cho M., Park J.:

Self-calibrating neural radiance fields.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) (October 2021), pp. 5846–5854.

- [JAFF16]

Johnson J., Alahi A., Fei-Fei L.:

Perceptual losses for real-time style transfer and super-resolution.

In Computer Vision – ECCV 2016 (Cham, 2016), Leibe B., Matas
J., Sebe N., Welling M., (Eds.), Springer International Publishing,
pp. 694–711.

- [Jak19]

Jakob W.:

Enoki: structured vectorization and differentiation on modern
processor architectures, 2019.

https://github.com/mitsuba-renderer/enoki.

- [Jar08]

Jarosz W.:

Efficient Monte Carlo Methods for Light Transport in Scattering
Media.

PhD thesis, UC San Diego, September 2008.

- [JDV∗14]

Jensen R., Dahl A., Vogiatzis G., Tola E., Aanæs H.:

Large scale multi-view stereopsis evaluation.

In Computer Vision and Pattern Recognition (CVPR) (2014).

- [JJHZ20]

Jiang Y., Ji D., Han Z., Zwicker M.:

Sdfdiff: Differentiable rendering of signed distance fields for 3d
shape optimization.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020).

- [JLF21]

Johari M. M., Lepoittevin Y., Fleuret F.:

Geonerf: Generalizing nerf with geometry priors, 2021.

arXiv:2111.13539.

- [JMB∗21]

Jain A., Mildenhall B., Barron J. T., Abbeel P., Poole B.:

Zero-shot text-guided object generation with dream fields.

arXiv (December 2021).

- [JSM∗20]

Jiang C. M., Sud A., Makadia A., Huang J., Nießner M., Funkhouser
T.:

Local implicit grid representations for 3d scenes.

In Proceedings IEEE Conf. on Computer Vision and Pattern
Recognition (Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition) (2020).

- [JXX∗21]

Jiakai Z., Xinhang L., Xinyi Y., Fuqiang Z., Yanshun Z., Minye W.,
Yingliang Z., Lan X., Jingyi Y.:

Editable free-viewpoint video using a layered neural representation.

In ACM SIGGRAPH (2021).

- [Kaj86]

Kajiya J. T.:

The rendering equation.

In Proceedings of the 13th annual conference on Computer
graphics and interactive techniques (1986), pp. 143–150.

- [KB04]

Kobbelt L., Botsch M.:

A survey of point-based techniques in computer graphics.

Computers and Graphics 28, 6 (2004), 801–814.

URL:
https://www.sciencedirect.com/science/article/pii/S0097849304001487,
doi:https://doi.org/10.1016/j.cag.2004.08.009.

- [KB14]

Kingma D. P., Ba J.:

Adam: A method for stochastic optimization.

CoRR abs/1412.6980 (2014).

URL: http://arxiv.org/abs/1412.6980, arXiv:1412.6980.

- [KBS15]

Kalantari N. K., Bako S., Sen P.:

A Machine Learning Approach for Filtering Monte Carlo Noise.

ACM Transactions on Graphics (TOG) (Proceedings of SIGGRAPH
2015) 34, 4 (2015).

- [KHM17]

Kar A., Häne C., Malik J.:

Learning a multi-view stereo machine.

In NeurIPS (2017).

- [KIT∗21]

Kondo N., Ikeda Y., Tagliasacchi A., Matsuo Y., Ochiai Y., Gu S. S.:

Vaxnerf: Revisiting the classic for voxel-accelerated neural radiance
field, 2021.

arXiv:2111.13112.

- [KJJ∗21]

Kellnhofer P., Jebe L., Jones A., Spicer R., Pulli K., Wetzstein G.:

Neural lumigraph rendering.

In CVPR (2021).

- [KOC∗21]

Kuang Z., Olszewski K., Chai M., Huang Z., Achlioptas P., Tulyakov S.:

Neroic: Neural object capture and rendering from online image
collections.

In arXiv (2021).

- [KPLD21]

Kopanas G., Philip J., Leimkühler T., Drettakis G.:

Point-based neural rendering with per-view optimization.

Computer Graphics Forum (Proceedings of the Eurographics
Symposium on Rendering) 40, 4 (June 2021).

URL: http://www-sop.inria.fr/reves/Basilic/2021/KPLD21.

- [KSW20]

Kohli A., Sitzmann V., Wetzstein G.:

Semantic Implicit Neural Scene Representations with Semi-supervised
Training.

In International Conference on 3D Vision (3DV) (2020).

- [KSZ∗21]

Kosiorek A. R., Strathmann H., Zoran D., Moreno P., Schneider R.,
Mokrá S., Rezende D. J.:

NeRF-VAE: A Geometry Aware 3D Scene Generative Model.

URL: http://arxiv.org/abs/2104.00587, arXiv:2104.00587.

- [KTEM18]

Kanazawa A., Tulsiani S., Efros A. A., Malik J.:

Learning category-specific mesh reconstruction from image
collections.

In Proceedings of the European Conference on Computer Vision
(2018), pp. 371–386.

- [KUH18]

Kato H., Ushiku Y., Harada T.:

Neural 3D mesh renderer.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2018), pp. 3907–3916.

- [KYK∗21]

Kania K., Yi K. M., Kowalski M., Trzciński T., Tagliasacchi A.:

Conerf: Controllable neural radiance fields, 2021.

arXiv:2112.01983.

- [LADL18a]

Li T.-M., Aittala M., Durand F., Lehtinen J.:

Differentiable monte carlo ray tracing through edge sampling.

In ACM Transactions on Graphics (proceedings of ACM SIGGRAPH
ASIA) (2018), ACM, p. 222.

- [LADL18b]

Li T.-M., Aittala M., Durand F., Lehtinen J.:

Differentiable monte carlo ray tracing through edge sampling.

ACM Trans. Graph. (Proc. SIGGRAPH Asia) 37, 6 (2018),
222:1–222:11.

- [LB14]

Loper M. M., Black M. J.:

Opendr: An approximate differentiable renderer.

In Proceedings of the European Conference on Computer Vision
(2014), Springer, pp. 154–169.

- [LFS∗21]

Li J., Feng Z., She Q., Ding H., Wang C., Lee G. H.:

Mine: Towards continuous depth mpi with nerf for novel view
synthesis.

In International Conference on Computer Vision (ICCV) (2021).

- [LGA∗18]

Li T.-M., Gharbi M., Adams A., Durand F., Ragan-Kelley J.:

Differentiable programming for image processing and deep learning in
Halide.

ACM Trans. Graph. (Proc. SIGGRAPH) 37, 4 (2018),
139:1–139:13.

- [LGL∗20]

Liu L., Gu J., Lin K. Z., Chua T.-S., Theobalt C.:

Neural sparse voxel fields.

Proceedings of the IEEE International Conference on Neural
Information Processing Systems (NeurIPS) (2020).

- [LH96]

Levoy M., Hanrahan P.:

Light field rendering.

In Proceedings of the 23rd Annual Conference on Computer
Graphics and Interactive Techniques (New York, NY, USA, 1996), SIGGRAPH ’96,
Association for Computing Machinery, p. 31–42.

URL: https://doi.org/10.1145/237170.237199, doi:10.1145/237170.237199.

- [LHL∗21]

Lyu L., Habermann M., Liu L., Tewari A., Theobalt C., et al.:

Efficient and differentiable shadow computation for inverse problems.

arXiv preprint arXiv:2104.00359 (2021).

- [LHR∗21]

Liu L., Habermann M., Rudnev V., Sarkar K., Gu J., Theobalt C.:

Neural actor: Neural free-view synthesis of human actors with pose
control.

ACM Trans. Graph.(ACM SIGGRAPH Asia) (2021).

- [LJR∗20]

Li L., Jamieson K., Rostamizadeh A., Gonina E., Ben-Tzur J., Hardt M.,
Recht B., Talwalkar A.:

A SYSTEM FOR MASSIVELY PARALLEL HYPERPARAMETER TUNING.

MLSys 2 (2020).

arXiv:1810.05934v5.

- [LJRT18]

Li L., Jamieson K., Rostamizadeh A., Talwalkar A.:

Hyperband: A Novel Bandit-Based Approach to Hyperparameter
Optimization.

Journal of Machine Learning Research 18 (2018), 1–52.

URL: http://jmlr.org/papers/v18/16-558.html., arXiv:1603.06560v4.

- [LK10]

Laine S., Karras T.:

Efficient sparse voxel octrees–analysis, extensions, and
implementation.

NVIDIA Corporation 2 (2010).

- [LK21]

Lange H., Kutz J. N.:

Fc2t2: The fast continuous convolutional taylor transform with
applications in vision and graphics.

arXiv e-prints (2021).

- [LKL18]

Lin C.-H., Kong C., Lucey S.:

Learning efficient point cloud generation for dense 3d object
reconstruction.

In AAAI Conference on Artificial Intelligence (2018).

- [LLCL19]

Liu S., Li T., Chen W., Li H.:

Soft rasterizer: A differentiable renderer for image-based 3D
reasoning.

In Proceedings of the International Conference on Computer
Vision (2019), pp. 7708–7717.

- [LLN∗18]

Liaw R., Liang E., Nishihara R., Moritz P., Gonzalez J. E., Stoica I.:

Tune: A research platform for distributed model selection and
training.

arXiv preprint arXiv:1807.05118 (2018).

- [LLYX21]

Liu C., Li Z., Yuan J., Xu Y.:

Nelf: Practical novel view synthesis with neural light field.

arXiv preprint arXiv:2105.07112 (2021).

- [LMR∗15]

Loper M., Mahmood N., Romero J., Pons-Moll G., Black M. J.:

SMPL: A skinned multi-person linear model.

ACM Trans. Graphics (Proc. SIGGRAPH Asia) 34, 6 (2015),
248:1–248:16.

- [LMTL21]

Lin C.-H., Ma W.-C., Torralba A., Lucey S.:

Barf: Bundle-adjusting neural radiance fields.

In IEEE International Conference on Computer Vision (ICCV)
(2021).

- [LMW21]

Lindell D. B., Martel J. N., Wetzstein G.:

Autoint: Automatic integration for fast neural volume rendering.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [LNSW21]

Li Z., Niklaus S., Snavely N., Wang O.:

Neural scene flow fields for space-time view synthesis of dynamic
scenes.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021), pp. 6498–6508.

- [LPX∗21]

Lin H., Peng S., Xu Z., Bao H., Zhou X.:

Efficient neural radiance fields with learned depth-guided sampling.

In arXiv (2021).

- [LSCL19]

Liu S., Saito S., Chen W., Li H.:

Learning to infer implicit surfaces without supervision.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2019), pp. 8295–8306.

- [LSS∗19]

Lombardi S., Simon T., Saragih J., Schwartz G., Lehrmann A., Sheikh
Y.:

Neural volumes: Learning dynamic renderable volumes from images.

ACM Trans. Graph. 38, 4 (July 2019), 65:1–65:14.

- [LSS∗21]

Lombardi S., Simon T., Schwartz G., Zollhoefer M., Sheikh Y., Saragih
J.:

Mixture of volumetric primitives for efficient neural rendering.

ACM Trans. Graph. 40, 4 (July 2021).

URL: https://doi.org/10.1145/3450626.3459863, doi:10.1145/3450626.3459863.

- [LSZ∗21]

Li T., Slavcheva M., Zollhoefer M., Green S., Lassner C., Kim C.,
Schmidt T., Lovegrove S., Goesele M., Lv Z.:

Neural 3D Video Synthesis.

URL: http://arxiv.org/abs/2103.02597, arXiv:2103.02597.

- [LTJ18]

Liu H.-T. D., Tao M., Jacobson A.:

Paparazzi: surface editing by way of multi-view image processing.

ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA)
37, 6 (2018), 221–1.

- [LVVPW22]

Lindell D. B., Van Veen D., Park J. J., Wetzstein G.:

Bacon: Band-limited coordinate networks for multiscale scene
representation.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2022).

- [LZ21]

Lassner C., Zollhöfer M.:

Pulsar: Efficient sphere-based neural rendering.

In IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) (June 2021).

- [LZBD21]

Luan F., Zhao S., Bala K., Dong Z.:

Unified Shape and SVBRDF Recovery using Differentiable Monte Carlo
Rendering.

Computer Graphics Forum 40, 4 (2021), 101–113.

URL: https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14344,
doi:https://doi.org/10.1111/cgf.14344.

- [LZP∗20]

Liu S., Zhang Y., Peng S., Shi B., Pollefeys M., Cui Z.:

Dist: Rendering deep implicit signed distance function with
differentiable sphere tracing.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020).

- [LZZ∗21]

Liu S., Zhang X., Zhang Z., Zhang R., Zhu J.-Y., Russell B.:

Editing conditional radiance fields.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) (2021).

- [Max95]

Max N.:

Optical models for direct volume rendering.

IEEE Transactions on Visualization and Computer Graphics 1, 2
(1995), 99–108.

doi:10.1109/2945.468400.

- [MBRS∗21]

Martin-Brualla R., Radwan N., Sajjadi M. S. M., Barron J. T.,
Dosovitskiy A., Duckworth D.:

NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo
Collections.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [MC]

Moses W. S., Churavy V.:

Instead of Rewriting Foreign Code for Machine Learning,
Automatically Synthesize Fast Gradients.

URL: https://enzyme.mit.edu.

- [MC10]

Max N. L., Chen M. S.:

Local and global illumination in the volume rendering integral.

In Scientific Visualization: Advanced Concepts (2010).

- [MCL∗21]

Meng Q., Chen A., Luo H., Wu M., Su H., Xu L., He X., Yu J.:

Gnerf: Gan-based neural radiance field without posed camera.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) (October 2021), pp. 6351–6361.

- [MCP∗21]

Moses W. S., Churavy V., Paehler L., Hückelheim J., Narayanan S.
H. K., Schanen M., Doerfert J.:

Reverse-mode automatic differentiation and optimization of gpu
kernels via enzyme.

In Proceedings of the International Conference for High
Performance Computing, Networking, Storage and Analysis (New York, NY, USA,
2021), SC ’21, Association for Computing Machinery.

URL: https://doi.org/10.1145/3458817.3476165, doi:10.1145/3458817.3476165.

- [MESK22]

Müller T., Evans A., Schied C., Keller A.:

Instant neural graphics primitives with a multiresolution hash
encoding, 2022.

URL:
https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf.

- [MGK∗19]

Meshry M., Goldman D. B., Khamis S., Hoppe H., Pandey R., Snavely N.,
Martin-Brualla R.:

Neural rerendering in the wild.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2019), pp. 6878–6887.

- [MHMB∗21]

Mildenhall B., Hedman P., Martin-Brualla R., Srinivasan P., Barron
J. T.:

Nerf in the dark: High dynamic range view synthesis from noisy raw
images.

arXiv (December 2021).

- [MLL∗21a]

Ma L., Li X., Liao J., Zhang Q., Wang X., Wang J., Sander P. V.:

Deblur-nerf: Neural radiance fields from blurry images.

arXiv (December 2021).

- [MLL∗21b]

Martel J. N., Lindell D. B., Lin C. Z., Chan E. R., Monteiro M.,
Wetzstein G.:

Acorn: Adaptive coordinate networks for neural representation.

ACM Trans. Graph. (SIGGRAPH) (2021).

- [MON∗19]

Mescheder L., Oechsle M., Niemeyer M., Nowozin S., Geiger A.:

Occupancy networks: Learning 3d reconstruction in function space.

In CVPR (2019).

- [MPJ∗19]

Michalkiewicz M., Pontes J. K., Jack D., Baktashmotlagh M., Eriksson
A.:

Implicit surface representations as layers in neural networks.

In Proceedings of the International Conference on Computer
Vision (2019), pp. 4743–4752.

- [MSOC∗19]

Mildenhall B., Srinivasan P. P., Ortiz-Cayon R., Kalantari N. K.,
Ramamoorthi R., Ng R., Kar A.:

Local light field fusion: Practical view synthesis with prescriptive
sampling guidelines.

ACM Trans. Graph. (SIGGRAPH) 38, 4 (2019).

- [MST∗20]

Mildenhall B., Srinivasan P. P., Tancik M., Barron J. T., Ramamoorthi
R., Ng R.:

Nerf: Representing scenes as neural radiance fields for view
synthesis.

In ECCV (2020).

- [NDVZJ19]

Nimier-David M., Vicini D., Zeltner T., Jakob W.:

Mitsuba 2: A retargetable forward and inverse renderer.

Transactions on Graphics (Proceedings of SIGGRAPH Asia) 38, 6
(Dec. 2019).

doi:10.1145/3355089.3356498.

- [NFS15]

Newcombe R. A., Fox D., Seitz S. M.:

Dynamicfusion: Reconstruction and tracking of non-rigid scenes in
real-time.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (CVPR) (2015), pp. 343–352.

- [NG20]

Niemeyer M., Geiger A.:

GIRAFFE: Representing Scenes as Compositional Generative Neural
Feature Fields.

URL: http://arxiv.org/abs/2011.12100, arXiv:2011.12100.

- [NG21a]

Niemeyer M., Geiger A.:

CAMPARI: Camera-Aware Decomposed Generative Neural Radiance Fields.

46–48.

URL: http://arxiv.org/abs/2103.17269, arXiv:2103.17269.

- [NG21b]

Niemeyer M., Geiger A.:

Giraffe: Representing scenes as compositional generative neural
feature fields.

In Computer Vision and Pattern Recognition (CVPR) (2021).

- [NMOG20]

Niemeyer M., Mescheder L., Oechsle M., Geiger A.:

Differentiable volumetric rendering: Learning implicit 3d
representations without 3d supervision.

In CVPR (2020).

- [NPLT∗19]

Nguyen-Phuoc T., Li C., Theis L., Richardt C., Yang Y.-L.:

Hologan: Unsupervised learning of 3d representations from natural
images.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision (2019), pp. 7588–7597.

- [NSLH21]

Noguchi A., Sun X., Lin S., Harada T.:

Neural articulated radiance field.

In International Conference on Computer Vision (ICCV) (2021).

- [NSP∗21]

Neff T., Stadlbauer P., Parger M., Kurz A., Mueller J. H., Chaitanya
C. R., Kaplanyan A., Steinberger M.:

DONeRF: Towards Real-Time Rendering of Compact Neural Radiance
Fields using Depth Oracle Networks.

Computer Graphics Forum 40, 4 (2021), 45–59.

arXiv:2103.03231,
doi:10.1111/cgf.14340.

- [NZIS13]

Nießner M., Zollhöfer M., Izadi S., Stamminger M.:

Real-time 3d reconstruction at scale using voxel hashing.

ACM Transactions on Graphics (TOG) (2013).

- [OELS∗21]

Or-El R., Luo X., Shan M., Shechtman E., Park J. J.,
Kemelmacher-Shlizerman I.:

Stylesdf: High-resolution 3d-consistent image and geometry
generation.

arXiv preprint arXiv:2112.11427 (2021).

- [OLN∗21]

Ost J., Laradji I., Newell A., Bahat Y., Heide F.:

Neural point light fields.

arXiv preprint arXiv:2112.01473 (2021).

- [OMN∗19]

Oechsle M., Mescheder L., Niemeyer M., Strauss T., Geiger A.:

Texture fields: Learning texture representations in function space.

In ICCV (2019).

- [OMT∗21]

Ost J., Mannan F., Thuerey N., Knodt J., Heide F.:

Neural Scene Graphs for Dynamic Scenes.

In Conference on Computer Vision and Pattern Recognition
(CVPR) (2021).

- [OPG21]

Oechsle M., Peng S., Geiger A.:

Unisurf: Unifying neural implicit surfaces and radiance fields for
multi-view reconstruction.

arXiv preprint arXiv:2104.10078 (2021).

- [PBDCO19]

Petersen F., Bermano A. H., Deussen O., Cohen-Or D.:

Pix2vex: Image-to-geometry reconstruction using a smooth
differentiable renderer.

arXiv preprint arXiv:1903.11149 (2019).

- [PC21]

Piala M., Clark R.:

Terminerf: Ray termination prediction for efficient neural rendering,
2021.

arXiv:2111.03643.

- [PCPMMN21]

Pumarola A., Corona E., Pons-Moll G., Moreno-Noguer F.:

D-NeRF: Neural Radiance Fields for Dynamic Scenes.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [PD84]

Porter T., Duff T.:

Compositing digital images.

SIGGRAPH Comput. Graph. 18, 3 (Jan. 1984), 253–259.

URL: https://doi.org/10.1145/964965.808606, doi:10.1145/964965.808606.

- [PDW∗21]

Peng S., Dong J., Wang Q., Zhang S., Shuai Q., Bao H., Zhou X.:

Animatable neural radiance fields for human body modeling.

arXiv preprint arXiv:2105.02872 (2021).

- [PFAK20]

Poursaeed O., Fisher M., Aigerman N., Kim V. G.:

Coupling explicit and implicit surface representations for generative
3d modeling.

In European Conference on Computer Vision (2020), Springer,
pp. 667–683.

- [PFS∗19]

Park J. J., Florence P., Straub J., Newcombe R., Lovegrove S.:

Deepsdf: Learning continuous signed distance functions for shape
representation.

CVPR (2019).

- [PGM∗19]

Paszke A., Gross S., Massa F., Lerer A., Bradbury J., Chanan G.,
Killeen T., Lin Z., Gimelshein N., Antiga L., Desmaison A., Kopf A., Yang E.,
DeVito Z., Raison M., Tejani A., Chilamkurthy S., Steiner B., Fang L., Bai
J., Chintala S.:

Pytorch: An imperative style, high-performance deep learning library.

In Advances in Neural Information Processing Systems (2019),
Wallach H., Larochelle H., Beygelzimer A., d'Alché-Buc
F., Fox E., Garnett R., (Eds.), vol. 32, Curran Associates, Inc.

URL:
https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

- [PNM∗20]

Peng S., Niemeyer M., Mescheder L., Pollefeys M., Geiger A.:

Convolutional occupancy networks.

In European Conference on Computer Vision (Proceedings of the
European Conference on Computer Vision) (2020).

- [PSB∗21]

Park K., Sinha U., Barron J. T., Bouaziz S., Goldman D. B., Seitz
S. M., Martin-Brualla R.:

Nerfies: Deformable neural radiance fields.

ICCV (2021).

- [PSDV∗18]

Perez E., Strub F., De Vries H., Dumoulin V., Courville A.:

Film: Visual reasoning with a general conditioning layer.

In Proceedings of the AAAI Conference on Artificial
Intelligence (2018), vol. 32.

- [PSH∗21]

Park K., Sinha U., Hedman P., Barron J. T., Bouaziz S., Goldman D. B.,
Martin-Brualla R., Seitz S. M.:

Hypernerf: A higher-dimensional representation for topologically
varying neural radiance fields.

arXiv preprint arXiv:2106.13228 (2021).

- [PXL∗21]

Pan X., Xu X., Loy C. C., Theobalt C., Dai B.:

A shading-guided generative implicit model for shape-accurate
3d-aware image synthesis.

In Advances in Neural Information Processing Systems (NeurIPS)
(2021).

- [PZvBG00]

Pfister H., Zwicker M., van Baar J., Gross M.:

Surfels-surface elements as rendering primitives.

In ACM Transactions on Graphics (Proc. ACM SIGGRAPH) (7/2000
2000), pp. 335–342.

- [PZX∗21]

Peng S., Zhang Y., Xu Y., Wang Q., Shuai Q., Bao H., Zhou X.:

Neural body: Implicit neural representations with structured latent
codes for novel view synthesis of dynamic humans.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (June 2021), pp. 9054–9063.

- [RBM∗21]

Roessle B., Barron J. T., Mildenhall B., Srinivasan P. P., Nießner
M.:

Dense depth priors for neural radiance fields from sparse input
views.

arXiv (December 2021).

- [RCV∗19]

Rössler A., Cozzolino D., Verdoliva L., Riess C., Thies J.,
Nießner M.:

Faceforensics++: Learning to detect manipulated facial images.

In ICCV 2019 (2019).

- [RES∗21]

Rudnev V., Elgharib M., Smith W., Liu L., Golyanik V., Theobalt C.:

Neural radiance fields for outdoor scene relighting, 2021.

arXiv:2112.05140.

- [RFS21a]

Rückert D., Franke L., Stamminger M.:

Adop: Approximate differentiable one-pixel point rendering.

arXiv:2110.06635.

- [RFS21b]

Rückert D., Franke L., Stamminger M.:

Adop: Approximate differentiable one-pixel point rendering.

arXiv preprint arXiv:2110.06635 (2021).

- [RKH∗21]

Radford A., Kim J. W., Hallacy C., Ramesh A., Goh G., Agarwal S.,
Sastry G., Askell A., Mishkin P., Clark J., Krueger G., Sutskever I.:

Learning transferable visual models from natural language
supervision, 2021.

arXiv:2103.00020.

- [RL21]

Ramasinghe S., Lucey S.:

Beyond periodicity: Towards a unifying framework for activations in
coordinate-mlps.

CoRR abs/2111.15135 (2021).

URL: https://arxiv.org/abs/2111.15135, arXiv:2111.15135.

- [RLS∗21]

Rematas K., Liu A., Srinivasan P. P., Barron J. T., Tagliasacchi A.,
Funkhouser T., Ferrari V.:

Urban radiance fields, 2021.

arXiv:2111.14643.

- [RMBF21]

Rematas K., Martin-Brualla R., Ferrari V.:

ShaRF: Shape-conditioned Radiance Fields from a Single View.

URL: http://arxiv.org/abs/2102.08860, arXiv:2102.08860.

- [RMG∗21]

Richard A., Markovic D., Gebru I. D., Krenn S., Butler G., de la Torre
F., Sheikh Y.:

Neural synthesis of binaural speech from mono audio.

In International Conference on Learning Representations (ICLR)
(2021).

- [RMY∗21]

Rebain D., Matthews M., Yi K. M., Lagun D., Tagliasacchi A.:

Lolnerf: Learn from one look.

arXiv preprint arXiv:2111.09996 (2021).

- [ROUG17]

Riegler G., Osman Ulusoy A., Geiger A.:

Octnet: Learning deep 3d representations at high resolutions.

In Proceedings of the IEEE conference on computer vision and
pattern recognition (2017), pp. 3577–3586.

- [RPLG21]

Reiser C., Peng S., Liao Y., Geiger A.:

KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny
MLPs.

URL: http://arxiv.org/abs/2103.13744, arXiv:2103.13744.

- [RRN∗20]

Ravi N., Reizenstein J., Novotny D., Gordon T., Lo W.-Y., Johnson J.,
Gkioxari G.:

Accelerating 3d deep learning with pytorch3d.

arXiv:2007.08501 (2020).

- [RROG18]

Roveri R., Rahmann L., Oztireli C., Gross M.:

A network architecture for point cloud classification via automatic
depth images generation.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2018), pp. 4176–4184.

- [RSH∗21]

Reizenstein J., Shapovalov R., Henzler P., Sbordone L., Labatut P.,
Novotny D.:

Common objects in 3d: Large-scale learning and evaluation of
real-life 3d category reconstruction.

In International Conference on Computer Vision (2021).

- [RZS∗20]

Raj A., Zollhoefer M., Simon T., Saragih J., Saito S., Hays J.,
Lombardi S.:

Pva: Pixel-aligned volumetric avatars.

In arXiv:2101.02697 (2020).

- [SCT∗20]

Sitzmann V., Chan E. R., Tucker R., Snavely N., Wetzstein G.:

Metasdf: Meta-learning signed distance functions.

In NeurIPS (2020).

- [SDZ∗21]

Srinivasan P. P., Deng B., Zhang X., Tancik M., Mildenhall B., Barron
J. T.:

NeRV: Neural reflectance and visibility fields for relighting and
view synthesis.

CVPR (2021).

- [SESM21]

Suhail M., Esteves C., Sigal L., Makadia A.:

Light field neural rendering.

arXiv preprint arXiv:2112.09687 (2021).

- [SHN∗19]

Saito S., Huang Z., Natsume R., Morishima S., Kanazawa A., Li H.:

Pifu: Pixel-aligned implicit function for high-resolution clothed
human digitization.

In Proceedings of the International Conference on Computer
Vision (2019), pp. 2304–2314.

- [SK00]

Shum H., Kang S. B.:

Review of image-based rendering techniques.

In Visual Communications and Image Processing 2000 (2000),
vol. 4067, International Society for Optics and Photonics, pp. 2–13.

- [SLNG20]

Schwarz K., Liao Y., Niemeyer M., Geiger A.:

GRAF: Generative radiance fields for 3D-aware image synthesis.

Advances in Neural Information Processing Systems
2020-December, NeurIPS (2020), 1–13.

arXiv:2007.02442.

- [SLOD21]

Sucar E., Liu S., Ortiz J., Davison A. J.:

iMAP: Implicit Mapping and Positioning in Real-Time.

URL: http://arxiv.org/abs/2103.12352, arXiv:2103.12352.

- [SLPS20]

Schops T., Larsson V., Pollefeys M., Sattler T.:

Why having 10,000 parameters in your camera model is better than
twelve.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (June 2020).

- [SMB∗20]

Sitzmann V., Martel J. N., Bergman A. W., Lindell D. B., Wetzstein G.:

Implicit neural representations with periodic activation functions.

In Conference on Neural Information Processing Systems
(NeurIPS) (2020).

- [SMP∗21]

Sajjadi M. S., Meyer H., Pot E., Bergmann U., Greff K., Radwan N., Vora
S., Lucic M., Duckworth D., Dosovitskiy A., et al.:

Scene representation transformer: Geometry-free novel view synthesis
through set-latent scene representations.

arXiv preprint arXiv:2111.13152 (2021).

- [SP04]

Sainz M., Pajarola R.:

Point-based rendering techniques.

Computers and Graphics 28, 6 (2004), 869–879.

URL:
https://www.sciencedirect.com/science/article/pii/S0097849304001530,
doi:https://doi.org/10.1016/j.cag.2004.08.014.

- [SRF∗21]

Sitzmann V., Rezchikov S., Freeman W. T., Tenenbaum J. B., Durand F.:

Light field networks: Neural scene representations with
single-evaluation rendering.

In arXiv (2021).

- [SS10]

Schwarz M., Seidel H.-P.:

Fast parallel surface and solid voxelization on gpus.

ACM Trans. Graph. 29, 6 (Dec. 2010).

URL: https://doi.org/10.1145/1882261.1866201, doi:10.1145/1882261.1866201.

- [SSC21]

Sun C., Sun M., Chen H.-T.:

Direct voxel grid optimization: Super-fast convergence for radiance
fields reconstruction, 2021.

arXiv:2111.11215.

- [SSS06]

Snavely N., Seitz S. M., Szeliski R.:

Photo tourism: Exploring photo collections in 3d.

In SIGGRAPH Conference Proceedings (New York, NY, USA, 2006),
ACM Press, pp. 835–846.

- [SSSJ20]

Saito S., Simon T., Saragih J., Joo H.:

Pifuhd: Multi-level pixel-aligned implicit function for
high-resolution 3d human digitization.

In Computer Vision and Pattern Recognition (CVPR) (2020).

- [STB∗19]

Srinivasan P. P., Tucker R., Barron J. T., Ramamoorthi R., Ng R.,
Snavely N.:

Pushing the boundaries of view extrapolation with multiplane images.

In CVPR (2019).

- [STH∗19]

Sitzmann V., Thies J., Heide F., Nießner M., Wetzstein G.,
Zollhöfer M.:

Deepvoxels: Learning persistent 3d feature embeddings.

In CVPR (2019).

- [SWZ∗21]

Sun J., Wang X., Zhang Y., Li X., Zhang Q., Liu Y., Wang J.:

Fenerf: Face editing in neural radiance fields, 2021.

arXiv:2111.15490.

- [SYZR21]

Su S.-Y., Yu F., Zollhoefer M., Rhodin H.:

A-nerf: Surface-free human 3d pose refinement via neural rendering.

In Conference on Neural Information Processing Systems
(NeurIPS) (2021).

- [SZW19]

Sitzmann V., Zollhöfer M., Wetzstein G.:

Scene representation networks: Continuous 3d-structure-aware neural
scene representations.

In NeurIPS (2019).

- [TET∗20]

Thies J., Elgharib M., Tewari A., Theobalt C., Nießner M.:

Neural voice puppetry: Audio-driven facial reenactment.

ECCV 2020 (2020).

- [TFT∗20]

Tewari A., Fried O., Thies J., Sitzmann V., Lombardi S., Sunkavalli K.,
Martin-Brualla R., Simon T., Saragih J., Nießner M., Pandey R., Fanello
S., Wetzstein G., Zhu J.-Y., Theobalt C., Agrawala M., Shechtman E., Goldman
D. B., Zollhöfer M.:

State of the art on neural rendering.

EG (2020).

- [TFT∗21]

Tewari A., Fried O., Thies J., Sitzmann V., Lombardi S., Xu Z., Simon
T., Nießner M., Tretschk E., Liu L., Mildenhall B., Srinivasan P., Pandey
R., Orts-Escolano S., Fanello S., Guo M., Wetzstein G., Zhu J.-Y., Theobalt
C., Agrawala M., Goldman D. B., Zollhöfer M.:

Advances in neural rendering.

In ACM SIGGRAPH 2021 Courses (New York, NY, USA, 2021),
SIGGRAPH ’21, Association for Computing Machinery.

URL: https://doi.org/10.1145/3450508.3464573, doi:10.1145/3450508.3464573.

- [TLY∗21]

Takikawa T., Litalien J., Yin K., Kreis K., Loop C., Nowrouzezahrai D.,
Jacobson A., McGuire M., Fidler S.:

Neural geometric level of detail: Real-time rendering with implicit
3D shapes.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [TMW∗21]

Tancik M., Mildenhall B., Wang T., Schmidt D., Srinivasan P. P., Barron
J. T., Ng R.:

Learned initializations for optimizing coordinate-based neural
representations.

In CVPR (2021).

- [TRS21]

Turki H., Ramanan D., Satyanarayanan M.:

Mega-nerf: Scalable construction of large-scale nerfs for virtual
fly-throughs, 2021.

arXiv:2112.10703.

- [TS20]

Tucker R., Snavely N.:

Single-view view synthesis with multiplane images.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2020), pp. 551–560.

- [TSM∗20]

Tancik M., Srinivasan P. P., Mildenhall B., Fridovich-Keil S., Raghavan
N., Singhal U., Ramamoorthi R., Barron J. T., Ng R.:

Fourier features let networks learn high frequency functions in low
dimensional domains.

NeurIPS (2020).

- [TTG∗20]

Tretschk E., Tewari A., Golyanik V., Zollhöfer M., Stoll C.,
Theobalt C.:

Patchnets: Patch-based generalizable deep implicit 3d shape
representations.

In European Conference on Computer Vision (2020), Springer,
Springer International Publishing, pp. 293–309.

- [TTG∗21]

Tretschk E., Tewari A., Golyanik V., Zollhöfer M., Lassner C.,
Theobalt C.:

Non-rigid neural radiance fields: Reconstruction and novel view
synthesis of a dynamic scene from monocular video.

In IEEE International Conference on Computer Vision (ICCV)
(2021), IEEE.

- [TY20]

Trevithick A., Yang B.:

GRF: Learning a General Radiance Field for 3D Representation and
Rendering.

URL: http://arxiv.org/abs/2010.04595, arXiv:2010.04595.

- [TZEM17]

Tulsiani S., Zhou T., Efros A. A., Malik J.:

Multi-view supervision for single-view reconstruction via
differentiable ray consistency.

In CVPR (2017).

- [TZN19]

Thies J., Zollhöfer M., Nießner M.:

Deferred neural rendering: Image synthesis using neural textures.

ACM Trans. Graph. 38, 4 (2019), 1–12.

- [TZS∗16]

Thies J., Zollhöfer M., Stamminger M., Theobalt C., Nießner
M.:

Face2face: Real-time face capture and reenactment of rgb videos.

In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE
(2016).

- [VHM∗21]

Verbin D., Hedman P., Mildenhall B., Zickler T., Barron J. T.,
Srinivasan P. P.:

Ref-nerf: Structured view-dependent appearance for neural radiance
fields.

In arXiv (2021).

- [VKP∗19]

Valentin J., Keskin C., Pidlypenskyi P., Makadia A., Sud A., Bouaziz
S.:

Tensorflow graphics: Computer graphics meets deep learning.

- [Vla09]

Vladsinger:

Surface control point diagram used in freeform modeling, 2009.

URL:
https://en.wikipedia.org/wiki/B-spline##/media/File:Surface_modelling.svg.

- [VSP∗17]

Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomez A. N.,
Kaiser L. u., Polosukhin I.:

Attention is all you need.

In Advances in Neural Information Processing Systems (2017),
Guyon I., Luxburg U. V., Bengio S., Wallach H., Fergus R., Vishwanathan S.,
Garnett R., (Eds.), vol. 30, Curran Associates, Inc.

URL:
https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

- [WBL∗21]

Wang Z., Bagautdinov T., Lombardi S., Simon T., Saragih J., Hodgins J.,
Zollhöfer M.:

Learning Compositional Radiance Fields of Dynamic Human Heads.

Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition (2021).

- [WCH∗21]

Wang C., Chai M., He M., Chen D., Liao J.:

Clip-nerf: Text-and-image driven manipulation of neural radiance
fields, 2021.

arXiv:2112.05139.

- [WCS∗22]

Weng C.-Y., Curless B., Srinivasan P. P., Barron J. T.,
Kemelmacher-Shlizerman I.:

Humannerf: Free-viewpoint rendering of moving people from monocular
video, 2022.

arXiv:2201.04127.

- [WGSJ20]

Wiles O., Gkioxari G., Szeliski R., Johnson J.:

Synsin: End-to-end view synthesis from a single image.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (6 2020).

- [WLB∗21]

Wu L., Lee J. Y., Bhattad A., Wang Y., Forsyth D.:

Diver: Real-time and accurate neural radiance fields with
deterministic integration for volume rendering, 2021.

arXiv:2111.10427.

- [WLG∗17]

Wang P.-S., Liu Y., Guo Y.-X., Sun C.-Y., Tong X.:

O-cnn: Octree-based convolutional neural networks for 3d shape
analysis.

ACM Transactions On Graphics (TOG) 36, 4 (2017), 1–11.

- [WLL∗21]

Wang P., Liu L., Liu Y., Theobalt C., Komura T., Wang W.:

Neus: Learning neural implicit surfaces by volume rendering for
multi-view reconstruction.

NeurIPS (2021).

- [WLR∗21]

Wei Y., Liu S., Rao Y., Zhao W., Lu J., Zhou J.:

Nerfingmvs: Guided optimization of neural radiance fields for indoor
multi-view stereo.

In ICCV (2021).

- [WPYS21]

Wizadwongsa S., Phongthawee P., Yenphraphai J., Suwajanakorn S.:

Nex: Real-time view synthesis with neural basis expansion.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [WWG∗21a]

Wang C., Wu X., Guo Y.-C., Zhang S.-H., Tai Y.-W., Hu S.-M.:

Nerf-sr: High-quality neural radiance fields using super-sampling.

arXiv (December 2021).

- [WWG∗21b]

Wang Q., Wang Z., Genova K., Srinivasan P., Zhou H., Barron J. T., Noah
R. M.-b., Funkhouser T., Tech C.:

IBRNet : Learning Multi-View Image-Based Rendering.

Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (2021), 4690—-4699.

- [WWX∗21]

Wang Z., Wu S., Xie W., Chen M., Prisacariu V. A.:

NeRF−⁣−--: Neural radiance fields without known camera parameters.

arXiv preprint arXiv:2102.07064 (2021).

- [XAS21]

Xu H., Alldieck T., Sminchisescu C.:

H-nerf: Neural radiance fields for rendering and temporal
reconstruction of humans in motion.

In Advances in Neural Information Processing Systems (NeurIPS)
(2021).

- [XFYS20]

Xu Y., Fan T., Yuan Y., Singh G.:

Ladybird: Quasi-Monte Carlo sampling for deep implicit field
based 3D reconstruction with symmetry.

arXiv preprint arXiv:2007.13393, 2020.

- [XHKK21]

Xian W., Huang J.-B., Kopf J., Kim C.:

Space-time neural irradiance fields for free-viewpoint video.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) (2021), pp. 9421–9431.

- [XPLD21]

Xu X., Pan X., Lin D., Dai B.:

Generative occupancy fields for 3d surface-aware image synthesis.

In Advances in Neural Information Processing Systems(NeurIPS)
(2021).

- [XPMBB21]

Xie C., Park K., Martin-Brualla R., Brown M.:

Fig-nerf: Figure-ground neural radiance fields for 3d object category
modelling.

arXiv preprint arXiv:2104.08418 (2021).

- [XPY∗21]

Xu Y., Peng S., Yang C., Shen Y., Zhou B.:

3d-aware image synthesis via learning structural and textural
representations.

- [XWC∗19]

Xu Q., Wang W., Ceylan D., Mech R., Neumann U.:

Disn: Deep implicit surface network for high-quality single-view 3d
reconstruction.

In Proceedings of the IEEE International Conference on Neural
Information Processing Systems (NeurIPS) (2019), vol. 32, Curran Associates,
Inc.

- [XXH∗21]

Xiang F., Xu Z., Hašan M., Hold-Geoffroy Y., Sunkavalli K., Su H.:

NeuTex: Neural texture mapping for volumetric neural rendering.

CVPR (2021).

- [XXP∗21]

Xiangli Y., Xu L., Pan X., Zhao N., Rao A., Theobalt C., Dai B., Lin
D.:

Citynerf: Building nerf at city scale, 2021.

arXiv:2112.05504.

- [Yad19]

Yadan O.:

Hydra - a framework for elegantly configuring complex applications.

Github, 2019.

URL: https://github.com/facebookresearch/hydra.

- [YAK∗20]

Yifan W., Aigerman N., Kim V. G., Chaudhuri S., Sorkine-Hornung O.:

Neural cages for detail-preserving 3d deformations.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition) (6 2020).

- [YFKT∗21]

Yu A., Fridovich-Keil S., Tancik M., Chen Q., Recht B., Kanazawa A.:

Plenoxels: Radiance fields without neural networks, 2021.

arXiv:2112.05131.

- [YGKL21a]

Yariv L., Gu J., Kasten Y., Lipman Y.:

Volume rendering of neural implicit surfaces, 2021.

arXiv:2106.12052.

- [YGKL21b]

Yariv L., Gu J., Kasten Y., Lipman Y.:

Volume rendering of neural implicit surfaces.

arXiv preprint arXiv:2106.12052 (2021).

- [YKG∗20]

Yoon J. S., Kim K., Gallo O., Park H. S., Kautz J.:

Novel view synthesis of dynamic scenes with globally coherent depths
from a monocular camera.

In Computer Vision and Pattern Recognition (CVPR) (2020).

- [YKM∗20]

Yariv L., Kasten Y., Moran D., Galun M., Atzmon M., Basri R., Lipman
Y.:

Multiview neural surface reconstruction by disentangling geometry and
appearance.

In NeurIPS (2020).

- [YLT∗21]

Yu A., Li R., Tancik M., Li H., Ng R., Kanazawa A.:

PlenOctrees for real-time rendering of neural radiance fields.

In arXiv (2021).

- [YRSH21]

Yifan W., Rahmann L., Sorkine-Hornung O.:

Geometry-consistent neural shape representation with implicit
displacement fields, 2021.

arXiv:2106.05187.

- [YSW∗19a]

Yifan W., Serena F., Wu S., Öztireli C., Sorkine-Hornung O.:

Differentiable surface splatting for point-based geometry processing.

ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA)
38, 6 (2019).

- [YSW∗19b]

Yifan W., Serena F., Wu S., Öztireli C., Sorkine-Hornung O.:

Differentiable surface splatting for point-based geometry processing.

ACM Transactions on Graphics (proceedings of ACM SIGGRAPH ASIA)
38, 6 (2019).

- [YTB∗21]

Yenamandra T., Tewari A., Bernard F., Seidel H.-P., Elgharib M.,
Cremers D., Theobalt C.:

i3dmm: Deep implicit 3d morphable model of human heads.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021), pp. 12803–12813.

- [YWYZ21]

Yao G., Wu H., Yuan Y., Zhou K.:

Dd-nerf: Double-diffusion neural radiance field as a generalizable
implicit body representation, 2021.

arXiv:2112.12390.

- [YYTK21]

Yu A., Ye V., Tancik M., Kanazawa A.:

pixelnerf: Neural radiance fields from one or few images.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (2021).

- [ZAC∗21]

Zheng Y., Abrevaya V. F., Chen X., Bühler M. C., Black M. J., Hilliges
O.:

I m avatar: Implicit morphable head avatars from videos, 2021.

arXiv:2112.07471.

- [ZLLD21]

Zhi S., Laidlow T., Leutenegger S., Davison A. J.:

In-place scene labelling and understanding with implicit scene
representation.

Proc. ICCV (2021).

- [ZLW∗21]

Zhang K., Luan F., Wang Q., Bala K., Snavely N.:

PhySG: Inverse rendering with spherical gaussians for physics-based
material editing and relighting.

CVPR (2021).

- [ZPVBG01]

Zwicker M., Pfister H., Van Baar J., Gross M.:

Surface splatting.

In Proc. Conf. on Computer Graphics and Interactive techniques
(2001), ACM, pp. 371–378.

- [ZPVBG02]

Zwicker M., Pfister H., Van Baar J., Gross M.:

Ewa splatting.

IEEE Transactions on Visualization and Computer Graphics 8, 3
(2002), 223–238.

- [ZRSK20]

Zhang K., Riegler G., Snavely N., Koltun V.:

Nerf++: Analyzing and improving neural radiance fields.

arXiv preprint arXiv:2010.07492 (2020).

- [ZSD∗21]

Zhang X., Srinivasan P. P., Deng B., Debevec P., Freeman W. T., Barron
J. T.:

NeRFactor: Neural factorization of shape and reflectance under an
unknown illumination.

SIGGRAPH Asia (2021).

- [ZTF∗18]

Zhou T., Tucker R., Flynn J., Fyffe G., Snavely N.:

Stereo magnification: Learning view synthesis using multiplane
images.

ACM Trans. Graph. (SIGGRAPH) (2018).

- [ZXNT21]

Zhou P., Xie L., Ni B., Tian Q.:

Cips-3d: A 3d-aware generator of gans based on
conditionally-independent pixel synthesis, 2021.

arXiv:2110.09788.

- [ZYQ21]

Zhang J., Yao Y., Quan L.:

Learning signed distance field for multi-view surface reconstruction.

arXiv preprint arXiv:2108.09964 (2021).

- [ZYZ∗21]

Zhao F., Yang W., Zhang J., Lin P., Zhang Y., Yu J., Xu L.:

Humannerf: Generalizable neural human radiance field from sparse
inputs, 2021.

arXiv:2112.02789.

- [ZZSC21]

Zhuang Y., Zhu H., Sun X., Cao X.:

Mofanerf: Morphable facial neural radiance field, 2021.

arXiv:2112.02308.

Generated on Fri Mar 8 13:13:48 2024 by LaTeXML
