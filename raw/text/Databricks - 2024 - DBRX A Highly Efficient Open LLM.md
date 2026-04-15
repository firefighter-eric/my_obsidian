# [2403.08275] Fully discrete finite difference schemes for the Fractional Korteweg-de Vries equation

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Databricks - 2024 - DBRX A Highly Efficient Open LLM.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2403.08275
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Fully discrete finite difference schemes for the Fractional Korteweg-de Vries equation

Mukul Dwivedi

Department of Mathematics,
Indian Institute of Technology Jammu,
Jagti, NH-44 Bypass Road, Post Office Nagrota,
Jammu - 181221, India

mukul.dwivedi@iitjammu.ac.in

 and 
Tanmay Sarkar

Department of Mathematics,
Indian Institute of Technology Jammu,
Jagti, NH-44 Bypass Road, Post Office Nagrota,
Jammu - 181221, India

tanmay.sarkar@iitjammu.ac.in

###### Abstract.

In this paper, we present and analyze fully discrete finite difference schemes designed for solving the initial value problem associated with the fractional Korteweg-de Vries (KdV) equation involving the fractional Laplacian. We design the scheme by introducing the discrete fractional Laplacian operator which is consistent with the continuous operator, and posses certain properties which are instrumental for the convergence analysis.
Assuming the initial data u0∈H1+α​(ℝ)subscript𝑢0superscript𝐻1𝛼ℝu_{0}\in H^{1+\alpha}(\mathbb{R}), where α∈[1,2)𝛼12\alpha\in[1,2), our study establishes the convergence of the approximate solutions obtained by the fully discrete finite difference schemes to a classical solution of the fractional KdV equation. Theoretical results are validated through several numerical illustrations for various values of fractional exponent α𝛼\alpha. Furthermore, we demonstrate that the Crank-Nicolson finite difference scheme preserves the inherent conserved quantities along with the improved convergence rates.

###### Key words and phrases:

###### 2020 Mathematics Subject Classification:

## 1. Introduction

We consider a Cauchy problem associated with a fractional Korteweg–de Vries (KdV) equation which is a nonlinear, non-local dispersive equation that has gained prominence in the area of weakly nonlinear internal long waves. More precisely, the equation reads

{ut+(u22)x−(−Δ)α/2​ux=0,(x,t)∈QT:=ℝ×(0,T],u​(x,0)=u0​(x),x∈ℝ,casessubscript𝑢𝑡subscriptsuperscript𝑢22𝑥superscriptΔ𝛼2subscript𝑢𝑥0𝑥𝑡subscript𝑄𝑇assignℝ0𝑇𝑢𝑥0subscript𝑢0𝑥𝑥ℝ\begin{cases}u_{t}+\left(\frac{u^{2}}{2}\right)_{x}-(-\Delta)^{\alpha/2}u_{x}=0,\qquad&(x,t)\in Q_{T}:=\mathbb{R}\times(0,T],\\
u(x,0)=u_{0}(x),\qquad&x\in\mathbb{R},\end{cases}

(1.1)

where T>0𝑇0T>0 is fixed, u0subscript𝑢0u_{0} is the prescribed initial condition and u:QT→ℝ:𝑢→subscript𝑄𝑇ℝu:Q_{T}\rightarrow\mathbb{R} represents the unknown solution.
The non-local operator −(−Δ)α/2superscriptΔ𝛼2-(-\Delta)^{\alpha/2} in (1.1) corresponds to the fractional Laplacian, a distinctive feature that introduces non-locality into the dynamics of the equation. The parameter α∈[1,2)𝛼12\alpha\in[1,2) dictates the fractional order and plays a pivotal role in shaping the behavior of the solutions. For ϕ∈Cc∞​(ℝ)italic-ϕsuperscriptsubscript𝐶𝑐ℝ\phi\in C_{c}^{\infty}(\mathbb{R}), the fractional Laplacian is defined as:

−(−Δ)α/2​[ϕ]​(x)=cα​P.V.​∫ℝϕ​(y)−ϕ​(x)|y−x|1+α​𝑑y,superscriptΔ𝛼2delimited-[]italic-ϕ𝑥subscript𝑐𝛼P.V.subscriptℝitalic-ϕ𝑦italic-ϕ𝑥superscript𝑦𝑥1𝛼differential-d𝑦-(-\Delta)^{\alpha/2}[\phi](x)=c_{\alpha}\text{P.V.}\int_{\mathbb{R}}\frac{\phi(y)-\phi(x)}{|y-x|^{1+\alpha}}\,dy,

(1.2)

for some constant cα>0subscript𝑐𝛼0c_{\alpha}>0, which is described in [4, 8, 26, 30].

Fractional Laplacian involved differential equation has emerged as a powerful and versatile tool across diverse scientific, economic and engineering domains, owing to the computational advantages inherent in integral operator and fractional-order derivative. The widespread adoption of fractional Laplacian is attributed to their effectiveness in handling localized computations, ranging from image segmentation to the study of water flow in narrow channels and the complexities of plasma physics. The historical evolution of fractional-order operators further enhances their efficacy, providing a solid foundation for modeling various scenarios with precision and versatility. As these operators continue to play a pivotal role in advancing computational methodologies and their impact on scientific and applied disciplines remains profound.

Researchers have extensively studied the well-posedness, both locally and globally, of the fractional KdV equation (1.1).
We will not be able to address all the literature here, but to mention the relevant ones which are related to the current work.
When α=2𝛼2\alpha=2, the equation (1.1) simplifies to the classical KdV equation [2, 8, 27], a well-established model for solitons and nonlinear wave phenomena. Recent decades have seen in-depth investigations into the well-posedness of the Cauchy problem associated to the KdV equation, for instance, see [20, 25] and references therein. For α=1𝛼1\alpha=1, the equation (1.1) becomes the Benjamin-Ono equation [13, 14, 23], designed to describe weakly nonlinear internal long waves. Well-posedness theory for the Benjamin-Ono equation is well-established in [32, 23, 35] and references therein.
The well-posedness of generalized dispersion model (1.1) has also been studied in the literature. For instance, Kenig et al. [21, 22] showed the existence and uniqueness of (1.1) for u0∈Hs​(ℝ),s>3/4formulae-sequencesubscript𝑢0superscript𝐻𝑠ℝ𝑠34u_{0}\in H^{s}(\mathbb{R}),~{}s>3/4. Afterwards,
Herr et al. [16] proved the global well-posedness by using frequency dependent renormalization technique for L2superscript𝐿2L^{2} data. For more related work, one may refer to [11, 24, 31].

Several numerical methods have been proposed for solving the equation (1.1). In particular, for the case α=2𝛼2\alpha=2,
fully discrete finite difference schemes have been developed by [1, 9, 19, 37]. Courtes et al. [3] established error estimates for a finite difference method while Skogestad et al. [34] conducted a comparison between finite difference schemes and Chebyshev methods. Li et al. [28] proposed high-order compact schemes for linear and nonlinear dispersive equations. Apart from this, Galerkin schemes have been designed in [6, 7]. In case of α=1𝛼1\alpha=1,
Thomee et al. [36] introduced a fully implicit finite difference scheme, and Dutta et al. [5] established the convergence of the fully discrete Crank-Nicolson scheme. Furthermore, Galtung [15] designed a Galerkin scheme and proved its convergence to the weak solution.
However, for the range α∈(1,2)𝛼12\alpha\in(1,2), there are limited literature on the numerical methods of (1.1). An operator splitting scheme is introduced in [8] and recently, a Galerkin scheme is developed in [10]. Hereby our focus is not to develop the most efficient scheme, rather to design the convergent finite difference schemes. Since the convergence analysis is based on the semi-discrete work of Sjöberg [33], there is a need of fully discrete schemes which are computationally efficient.

Motivated by the work of Holden et al. [19] and Dutta et al. [6] for the KdV equation and Benjamin-Ono equation respectively, we have developed two finite difference schemes to obtain the approximate solutions of the fractional KdV equation (1.1). The proposed schemes differ in their temporal discretization along with the averages of approximate solution considered in the convective term. The main objective of this paper is to show that the approximate solutions obtained by the difference schemes converge uniformly to the classical solution of the fractional KdV equation (1.1) in C​(ℝ×[0,T])𝐶ℝ0𝑇C(\mathbb{R}\times[0,T]) provided the initial data is in H1+α​(ℝ)superscript𝐻1𝛼ℝH^{1+\alpha}(\mathbb{R}).

In this paper, our focus is in the discretization of the fractional Laplacian and its subsequent numerical implementation. The derivation of two schemes involves a treatment of the nonlinear terms, with theoretical foundations and proof ideas borrowed from the seminal works of Holden et al. [19] and Dutta et al. [6]. However, the main challenge revolves around the precise computation of the fractional Laplacian and establishing its discrete properties.
Moreover, the Crank-Nicolson finite difference scheme is designed in such a way that it exhibits conservation properties analogous to those observed in classical solutions of the fractional KdV equation, as stipulated in [23]. These conserved quantities include mass, momentum and energy, defined as:

C1​(u)::subscript𝐶1𝑢absent\displaystyle C_{1}(u):
=∫ℝu​(x,t)​𝑑x,C2​(u):=∫ℝu2​(x,t)​𝑑x,formulae-sequenceabsentsubscriptℝ𝑢𝑥𝑡differential-d𝑥assignsubscript𝐶2𝑢subscriptℝsuperscript𝑢2𝑥𝑡differential-d𝑥\displaystyle=\int_{\mathbb{R}}u(x,t)~{}dx,\quad C_{2}(u):=\int_{\mathbb{R}}u^{2}(x,t)~{}dx,

C3​(u)::subscript𝐶3𝑢absent\displaystyle C_{3}(u):
=∫ℝ((−(−Δ)α/4​u)2−u33)​(x,t)​𝑑x,α∈[1,2).formulae-sequenceabsentsubscriptℝsuperscriptsuperscriptΔ𝛼4𝑢2superscript𝑢33𝑥𝑡differential-d𝑥𝛼12\displaystyle=\int_{\mathbb{R}}\left((-(-\Delta)^{\alpha/4}u)^{2}-\frac{u^{3}}{3}\right)(x,t)~{}dx,\qquad\alpha\in[1,2).

Through numerical illustrations, we empirically demonstrate the superior performance of the Crank-Nicolson scheme over the Euler implicit scheme, aligning with expectations.

The organization of the paper is as follows: In Section 2, we present the necessary notations involving numerical discretization and introduce discrete operators. In addition, we show that discrete fractional Laplacian is consistent and satisfies certain properties. In Section 3, we introduce an implicit finite difference scheme which is solvable, stable and converges to the weak solution of (1.1). In Section 4, we design a Crank-Nicolson finite difference scheme. The scheme is shown to be stable and convergent for the initial data in H1+α​(ℝ)superscript𝐻1𝛼ℝH^{1+\alpha}(\mathbb{R}).
We validate our theoretical results through the numerical illustrations in Section 5 for various values of α∈[1,2]𝛼12\alpha\in[1,2]. Finally, we end up with some concluding remarks in Section 6.

## 2. Numerical discretization and discrete operators

In establishing a foundational framework, we introduce some essential notations, definitions, and inequalities. Let Δ​xΔ𝑥\Delta x and Δ​tΔ𝑡\Delta t denote the small mesh sizes corresponding to the spatial and temporal variables respectively. We discretize both the spatial and temporal axes using these mesh sizes, defining xj=j​Δ​xsubscript𝑥𝑗𝑗Δ𝑥x_{j}=j\Delta x for j∈ℤ𝑗ℤj\in\mathbb{Z} and tn=n​Δ​tsubscript𝑡𝑛𝑛Δ𝑡t_{n}=n\Delta t for n∈ℕ0:=ℕ∪{0}𝑛subscriptℕ0assignℕ0n\in\mathbb{N}_{0}:=\mathbb{N}\cup\{0\}. Subsequently, a fully discrete grid function in both space and time is defined by u:Δ​x​ℤ×Δ​t​ℕ0→ℝℤ:𝑢→Δ𝑥ℤΔ𝑡subscriptℕ0superscriptℝℤu:\Delta x\mathbb{Z}\times\Delta t\mathbb{N}_{0}\to\mathbb{R}^{\mathbb{Z}} with u(xj,tn)=:ujnu(x_{j},t_{n})=:u_{j}^{n}, and we represent (un)j∈ℤsubscriptsuperscript𝑢𝑛𝑗ℤ(u^{n})_{j\in\mathbb{Z}} as unsuperscript𝑢𝑛u^{n} for brevity. Notably, unsuperscript𝑢𝑛u^{n} is constructed as a spatial grid function.

We further introduce difference operators for a function v:ℝ→ℝ:𝑣→ℝℝv:\mathbb{R}\to\mathbb{R}. These operators are defined as follows:

D±​v​(x)=±1Δ​x​(v​(x±Δ​x)−v​(x)),D=12​(D++D−).formulae-sequencesubscript𝐷plus-or-minus𝑣𝑥plus-or-minus1Δ𝑥𝑣plus-or-minus𝑥Δ𝑥𝑣𝑥𝐷12subscript𝐷subscript𝐷D_{\pm}v(x)=\pm\frac{1}{\Delta x}\big{(}v(x\pm\Delta x)-v(x)\big{)},\qquad D=\frac{1}{2}(D_{+}+D_{-}).

(2.1)

We also introduce the shift operators

S±​v​(x)=v​(x±Δ​x),superscript𝑆plus-or-minus𝑣𝑥𝑣plus-or-minus𝑥Δ𝑥S^{\pm}v(x)=v(x\pm\Delta x),

and the averages v~​(x)~𝑣𝑥\widetilde{v}(x) and v¯​(x)¯𝑣𝑥\bar{v}(x) are defined by

v~​(x):=13​(S+​v​(x)+v​(x)+S−​v​(x)),v¯​(x):=12​(S++S−)​v​(x).formulae-sequenceassign~𝑣𝑥13superscript𝑆𝑣𝑥𝑣𝑥superscript𝑆𝑣𝑥assign¯𝑣𝑥12superscript𝑆superscript𝑆𝑣𝑥\widetilde{v}(x):=\frac{1}{3}\left(S^{+}v(x)+v(x)+S^{-}v(x)\right),\qquad\bar{v}(x):=\frac{1}{2}(S^{+}+S^{-})v(x).

The difference operators satisfy the following product formulas

D​(v​w)𝐷𝑣𝑤\displaystyle D(vw)
=v¯​D​w+w¯​D​v,absent¯𝑣𝐷𝑤¯𝑤𝐷𝑣\displaystyle=\bar{v}Dw+\bar{w}Dv,

(2.2)

D±​(v​w)subscript𝐷plus-or-minus𝑣𝑤\displaystyle D_{\pm}(vw)
=S±​v​D±​w+w​D±​v=S±​w​D±​v+v​D±​w.absentsuperscript𝑆plus-or-minus𝑣subscript𝐷plus-or-minus𝑤𝑤subscript𝐷plus-or-minus𝑣superscript𝑆plus-or-minus𝑤subscript𝐷plus-or-minus𝑣𝑣subscript𝐷plus-or-minus𝑤\displaystyle=S^{\pm}vD_{\pm}w+wD_{\pm}v=S^{\pm}wD_{\pm}v+vD_{\pm}w.

(2.3)

We consider the usual ℓ2superscriptℓ2\ell^{2}-inner product, denoted as ⟨⋅,⋅⟩⋅⋅\langle\cdot,\cdot\rangle, defined by

⟨v,w⟩=Δ​x​∑j∈ℤvj​wj,‖v‖=‖v‖2=⟨v,v⟩1/2,v,w∈ℓ2.formulae-sequenceformulae-sequence𝑣𝑤Δ𝑥subscript𝑗ℤsubscript𝑣𝑗subscript𝑤𝑗norm𝑣subscriptnorm𝑣2superscript𝑣𝑣12𝑣𝑤superscriptℓ2\langle v,w\rangle=\Delta x\sum_{j\in\mathbb{Z}}v_{j}w_{j},\qquad\left\|v\right\|=\left\|v\right\|_{2}=\langle v,v\rangle^{1/2},\qquad v,w\in\ell^{2}.

(2.4)

As a consequence, we have the following estimates

‖v‖∞:=max⁡{|vj|:j∈ℤ}≤1Δ​x1/2​‖v‖,‖D​v‖≤1Δ​x​‖v‖,v∈ℓ2.formulae-sequenceassignsubscriptnorm𝑣:subscript𝑣𝑗𝑗ℤ1Δsuperscript𝑥12norm𝑣formulae-sequencenorm𝐷𝑣1Δ𝑥norm𝑣𝑣superscriptℓ2\|v\|_{\infty}:=\max\{|v_{j}|:j\in\mathbb{Z}\}\leq\frac{1}{\Delta x^{1/2}}\|v\|,\qquad\|Dv\|\leq\frac{1}{\Delta x}\|v\|,\quad v\in\ell^{2}.

(2.5)

The above inequalities are straightforward to establish, given that v∈ℓ2𝑣superscriptℓ2v\in\ell^{2}, implying |vj|subscript𝑣𝑗|v_{j}| tends to 00 as j→∞absent→𝑗j\xrightarrow[]{}\infty. Thus, taking the maximum yields a simple proof. Several properties of the difference operators in relation to their ℓ2superscriptℓ2\ell^{2}-inner product can be derived. Let v,w∈ℓ2𝑣𝑤superscriptℓ2v,w\in\ell^{2}. Then

⟨v,D±​w⟩=−⟨D∓​v,w⟩,⟨v,D​w⟩=−⟨D​v,w⟩.formulae-sequence𝑣subscript𝐷plus-or-minus𝑤subscript𝐷minus-or-plus𝑣𝑤𝑣𝐷𝑤𝐷𝑣𝑤\langle v,D_{\pm}w\rangle=-\langle D_{\mp}v,w\rangle,\qquad\langle v,Dw\rangle=-\langle Dv,w\rangle.

(2.6)

Furthermore, employing the product formulas (2.2) and (2.3) along with the properties (2.6), we get the following identities:

⟨D​(v​w),w⟩𝐷𝑣𝑤𝑤\displaystyle\left\langle D(vw),w\right\rangle
=Δ​x2​⟨D+​v​D​(w),w⟩+12​⟨S−​w​D​v,w⟩,absentΔ𝑥2subscript𝐷𝑣𝐷𝑤𝑤12superscript𝑆𝑤𝐷𝑣𝑤\displaystyle=\frac{\Delta x}{2}\left\langle D_{+}vD(w),w\right\rangle+\frac{1}{2}\left\langle S^{-}wDv,w\right\rangle,

(2.7)

D+​D−​(v​w)subscript𝐷subscript𝐷𝑣𝑤\displaystyle D_{+}D_{-}(vw)
=D−​v​D+​w+S−​v​D+​D−​w+D+​v​D+​w+w​D+​D−​v.absentsubscript𝐷𝑣subscript𝐷𝑤superscript𝑆𝑣subscript𝐷subscript𝐷𝑤subscript𝐷𝑣subscript𝐷𝑤𝑤subscript𝐷subscript𝐷𝑣\displaystyle=D_{-}vD_{+}w+S^{-}vD_{+}D_{-}w+D_{+}vD_{+}w+wD_{+}D_{-}v.

(2.8)

### 2.1. Discretization of fractional Laplacian

In the pursuit of a discretization of the fractional Laplacian, we consider the definition of fractional Laplacian. Let u∈𝒮​(ℝ)𝑢𝒮ℝu\in\mathcal{S(\mathbb{R})}, where 𝒮​(ℝ)𝒮ℝ\mathcal{S}(\mathbb{R}) denotes the Schwartz space. Then

−(−Δ)α/2​[u]​(x)superscriptΔ𝛼2delimited-[]𝑢𝑥\displaystyle-(-\Delta)^{\alpha/2}[u](x)
=cα​P.V.​∫ℝu​(y)−u​(x)|y−x|1+α​𝑑yabsentsubscript𝑐𝛼P.V.subscriptℝ𝑢𝑦𝑢𝑥superscript𝑦𝑥1𝛼differential-d𝑦\displaystyle=c_{\alpha}\text{P.V.}\int_{\mathbb{R}}\frac{u(y)-u(x)}{|y-x|^{1+\alpha}}\,dy

=12​cα​∫ℝu​(x+y)−2​u​(x)+u​(x−y)|y|1+α​𝑑y.absent12subscript𝑐𝛼subscriptℝ𝑢𝑥𝑦2𝑢𝑥𝑢𝑥𝑦superscript𝑦1𝛼differential-d𝑦\displaystyle=\frac{1}{2}c_{\alpha}\int_{\mathbb{R}}\frac{u(x+y)-2u(x)+u(x-y)}{|y|^{1+\alpha}}\,dy.

(2.9)

The above equivalence follows by the standard change of variable formula. In fact, (2.9) is quite useful to remove the singularity of the integral. For any smooth function u𝑢u, a second order Taylor series expansion yields

u​(x+y)−2​u​(x)+u​(x−y)|y|1+α≤‖∂x2u‖L∞​(ℝ)|y|α−1,𝑢𝑥𝑦2𝑢𝑥𝑢𝑥𝑦superscript𝑦1𝛼subscriptnormsuperscriptsubscript𝑥2𝑢superscript𝐿ℝsuperscript𝑦𝛼1\frac{u(x+y)-2u(x)+u(x-y)}{|y|^{1+\alpha}}\leq\frac{\|\partial_{x}^{2}u\|_{L^{\infty}(\mathbb{R})}}{|y|^{\alpha-1}},

(2.10)

which is integrable near origin.

Our approach to discretize the fractional Laplacian begins with the consideration of even indices. More precisely,

(−(−Δ)α/2​u)j=cα​P.V.​∫ℝu​(y)−uj|y−xj|1+α​𝑑y=cα​∑k=even∫xkxk+2u​(y)−uj|y−xj|1+α​𝑑y,subscriptsuperscriptΔ𝛼2𝑢𝑗subscript𝑐𝛼P.V.subscriptℝ𝑢𝑦subscript𝑢𝑗superscript𝑦subscript𝑥𝑗1𝛼differential-d𝑦subscript𝑐𝛼subscript𝑘evensuperscriptsubscriptsubscript𝑥𝑘subscript𝑥𝑘2𝑢𝑦subscript𝑢𝑗superscript𝑦subscript𝑥𝑗1𝛼differential-d𝑦\begin{split}(-(-\Delta)^{\alpha/2}u)_{j}&=c_{\alpha}\text{P.V.}\int_{\mathbb{R}}\frac{u(y)-u_{j}}{|y-x_{j}|^{1+\alpha}}\,dy\\
&=c_{\alpha}\sum_{k=\text{even}}\int_{x_{k}}^{x_{k+2}}\frac{u(y)-u_{j}}{|y-x_{j}|^{1+\alpha}}\,dy,\end{split}

where u​(xj):=ujassign𝑢subscript𝑥𝑗subscript𝑢𝑗u(x_{j}):=u_{j}. Employing the midpoint formula on the integrals, we arrive at the discrete approximation:

(−(−Δ)α/2​u)j≈cα​∑k=odd2​Δ​x​uk−uj|xk−xj|1+αsubscriptsuperscriptΔ𝛼2𝑢𝑗subscript𝑐𝛼subscript𝑘odd2Δ𝑥subscript𝑢𝑘subscript𝑢𝑗superscriptsubscript𝑥𝑘subscript𝑥𝑗1𝛼(-(-\Delta)^{\alpha/2}u)_{j}\approx c_{\alpha}\sum_{k=\text{odd}}2\Delta x\frac{u_{k}-u_{j}}{|x_{k}-x_{j}|^{1+\alpha}}

which can be rewritten as

(−(−Δ)α/2​u)j≈2​cαΔ​xα​∑k=odduk−uj|k−j|1+α.subscriptsuperscriptΔ𝛼2𝑢𝑗2subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘oddsubscript𝑢𝑘subscript𝑢𝑗superscript𝑘𝑗1𝛼(-(-\Delta)^{\alpha/2}u)_{j}\approx\frac{2c_{\alpha}}{\Delta x^{\alpha}}\sum_{k=\text{odd}}\frac{u_{k}-u_{j}}{|k-j|^{1+\alpha}}.

Similarly, by considering the odd indices, we end up with

(−(−Δ)α/2​u)j≈2​cαΔ​xα​∑k=evenuk−uj|k−j|1+α.subscriptsuperscriptΔ𝛼2𝑢𝑗2subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘evensubscript𝑢𝑘subscript𝑢𝑗superscript𝑘𝑗1𝛼(-(-\Delta)^{\alpha/2}u)_{j}\approx\frac{2c_{\alpha}}{\Delta x^{\alpha}}\sum_{k=\text{even}}\frac{u_{k}-u_{j}}{|k-j|^{1+\alpha}}.

Consequently, the combining of the above results leads to the definition of the discrete fractional Laplacian, denoted as 𝔻αsuperscript𝔻𝛼\mathbb{D}^{\alpha}, expressed as follows:

𝔻α​(u)jsuperscript𝔻𝛼subscript𝑢𝑗\displaystyle\mathbb{D}^{\alpha}(u)_{j}
=cαΔ​xα​∑k≠juk−uj|k−j|1+α​(1−(−1)j−k).absentsubscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗subscript𝑢𝑘subscript𝑢𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘\displaystyle=\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{u_{k}-u_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k}).

(2.11)

Another approach to define the discrete fractional Laplacian can be incorporating the definition given by Nezza et al. [4, Lemma 3.2] in which the singular integral is represented by a weighted second order differential quotient.

###### Lemma 2.1.

Let α∈[1,2)𝛼12\alpha\in[1,2) and 𝔻αsuperscript𝔻𝛼\mathbb{D}^{\alpha} be the discrete fractional Laplacian defined by (2.11). Then, for any u∈𝒮​(ℝ)𝑢𝒮ℝu\in\mathcal{S}(\mathbb{R}),

𝔻α​(u)j=cα2​∑i∫x2​ix2​i+2u​(xj+x2​i+1)+u​(xj−x2​i+1)−2​u​(xj)|x2​i+1|1+α​𝑑x.superscript𝔻𝛼subscript𝑢𝑗subscript𝑐𝛼2subscript𝑖superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝑢subscript𝑥𝑗subscript𝑥2𝑖1𝑢subscript𝑥𝑗subscript𝑥2𝑖12𝑢subscript𝑥𝑗superscriptsubscript𝑥2𝑖11𝛼differential-d𝑥\mathbb{D}^{\alpha}(u)_{j}=\frac{c_{\alpha}}{2}\sum_{i}\int_{x_{2i}}^{x_{2i+2}}\frac{u(x_{j}+x_{2i+1})+u(x_{j}-x_{2i+1})-2u(x_{j})}{|x_{2i+1}|^{1+\alpha}}\,dx.

(2.12)

###### Proof.

We consider the equation (2.11) and by the change of variable k−j=i𝑘𝑗𝑖k-j=i, it results into

2​∑i≠0uj+i−uj|i|1+α​(1−(−1)i)=∑i≠0uj+i−uj|i|1+α​(1−(−1)i)+∑i≠0uj+i−uj|i|1+α​(1−(−1)i)=∑i≠0uj+i−uj|i|1+α​(1−(−1)i)+∑i≠0uj−i−uj|i|1+α​(1−(−1)i)=∑iuj+i+uj−i−2​uj|i|1+α​(1−(−1)i),2subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖subscript𝑖subscript𝑢𝑗𝑖subscript𝑢𝑗𝑖2subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖\begin{split}2\sum_{i\neq 0}\frac{u_{j+i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})&=\sum_{i\neq 0}\frac{u_{j+i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})+\sum_{i\neq 0}\frac{u_{j+i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})\\
&=\sum_{i\neq 0}\frac{u_{j+i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})+\sum_{i\neq 0}\frac{u_{j-i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})\\
&=\sum_{i}\frac{u_{j+i}+u_{j-i}-2u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i}),\end{split}

as only odd i𝑖i survives, thus

2​∑i≠0uj+i−uj|i|1+α​(1−(−1)i)2subscript𝑖0subscript𝑢𝑗𝑖subscript𝑢𝑗superscript𝑖1𝛼1superscript1𝑖\displaystyle 2\sum_{i\neq 0}\frac{u_{j+i}-u_{j}}{|i|^{1+\alpha}}(1-(-1)^{i})
=2​Δ​x1+α​∑iu​(xj+x2​i+1)+u​(xj−x2​i+1)−2​u​(xj)|x2​i+1|1+αabsent2Δsuperscript𝑥1𝛼subscript𝑖𝑢subscript𝑥𝑗subscript𝑥2𝑖1𝑢subscript𝑥𝑗subscript𝑥2𝑖12𝑢subscript𝑥𝑗superscriptsubscript𝑥2𝑖11𝛼\displaystyle=2\Delta x^{1+\alpha}\sum_{i}\frac{u(x_{j}+x_{2i+1})+u(x_{j}-x_{2i+1})-2u(x_{j})}{|x_{2i+1}|^{1+\alpha}}

=Δ​xα​∑i∫x2​ix2​i+2u​(xj+x2​i+1)+u​(xj−x2​i+1)−2​u​(xj)|x2​i+1|1+α​𝑑x.absentΔsuperscript𝑥𝛼subscript𝑖superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝑢subscript𝑥𝑗subscript𝑥2𝑖1𝑢subscript𝑥𝑗subscript𝑥2𝑖12𝑢subscript𝑥𝑗superscriptsubscript𝑥2𝑖11𝛼differential-d𝑥\displaystyle=\Delta x^{\alpha}\sum_{i}\int_{x_{2i}}^{x_{2i+2}}\frac{u(x_{j}+x_{2i+1})+u(x_{j}-x_{2i+1})-2u(x_{j})}{|x_{2i+1}|^{1+\alpha}}\,dx.

Hence the result follows.
∎

The Lemma 2.1 will help us to show the relation between continuous and discrete fractional Sobolev norms. Let us define the h2superscriptℎ2h^{2}-norm for the given grid function unsuperscript𝑢𝑛u^{n} by

‖un‖h2=‖un‖+‖D+​un‖+‖D+​D−​un‖.subscriptnormsuperscript𝑢𝑛superscriptℎ2normsuperscript𝑢𝑛normsubscript𝐷superscript𝑢𝑛normsubscript𝐷subscript𝐷superscript𝑢𝑛\|u^{n}\|_{h^{2}}=\|u^{n}\|+\|D_{+}u^{n}\|+\|D_{+}D_{-}u^{n}\|.

There exists an alternative definition of the fractional Laplacian for a function u∈𝒮​(ℝ)𝑢𝒮ℝu\in\mathcal{S}(\mathbb{R}), provided by Yang et al. [39, Lemma 1], expressed by the following equation:

−(−Δ)α/2​u​(x)=∂α∂|x|α​u​(x)=−Dxα−∞​u​(x)+D∞αx​u​(x)2​cos⁡(α​π2),1<α<2,formulae-sequencesuperscriptΔ𝛼2𝑢𝑥superscript𝛼superscript𝑥𝛼𝑢𝑥subscriptsuperscriptsubscript𝐷𝑥𝛼𝑢𝑥subscriptsuperscriptsubscript𝐷𝛼𝑥𝑢𝑥2𝛼𝜋21𝛼2-(-\Delta)^{\alpha/2}u(x)=\frac{\partial^{\alpha}}{\partial|x|^{\alpha}}u(x)=-\frac{{}_{-\infty}D_{x}^{\alpha}u(x)+{}_{x}D_{\infty}^{\alpha}u(x)}{2\cos\left(\frac{\alpha\pi}{2}\right)},\quad 1<\alpha<2,

(2.13)

where Dxα−∞subscriptsuperscriptsubscript𝐷𝑥𝛼{}_{-\infty}D_{x}^{\alpha} and D∞αxsubscriptsuperscriptsubscript𝐷𝛼𝑥{}_{x}D_{\infty}^{\alpha} are defined as the left- and right-side Riemann–Liouville derivatives [39]. Following Xu et al. [38] and Ervin et al. [12], we introduce the following definitions of norms, aiding in the formulation of a norm on the fractional Laplacian (2.13).

###### Definition 2.2.

We define the semi-norms

|u|JLα​(ℝ)=‖Dxα−∞​u‖L2​(ℝ),|u|JRα​(ℝ)=‖D∞αx​u‖L2​(ℝ),formulae-sequencesubscript𝑢superscriptsubscript𝐽𝐿𝛼ℝsubscriptnormsubscriptsuperscriptsubscript𝐷𝑥𝛼𝑢superscript𝐿2ℝsubscript𝑢superscriptsubscript𝐽𝑅𝛼ℝsubscriptnormsubscriptsuperscriptsubscript𝐷𝛼𝑥𝑢superscript𝐿2ℝ\displaystyle\left|u\right|_{J_{L}^{\alpha}{(\mathbb{R})}}=\left\|{}_{-\infty}D_{x}^{\alpha}u\right\|_{L^{2}(\mathbb{R})},\qquad\left|u\right|_{J_{R}^{\alpha}(\mathbb{R})}=\left\|{}_{x}D_{\infty}^{\alpha}u\right\|_{L^{2}(\mathbb{R})},

(2.14)

and the norms

‖u‖JLα​(ℝ)=|u|JLα​(ℝ)+‖u‖L2​(ℝ),‖u‖JRα​(ℝ)=|u|JRα​(ℝ)+‖u‖L2​(ℝ),formulae-sequencesubscriptnorm𝑢superscriptsubscript𝐽𝐿𝛼ℝsubscript𝑢superscriptsubscript𝐽𝐿𝛼ℝsubscriptnorm𝑢superscript𝐿2ℝsubscriptnorm𝑢superscriptsubscript𝐽𝑅𝛼ℝsubscript𝑢superscriptsubscript𝐽𝑅𝛼ℝsubscriptnorm𝑢superscript𝐿2ℝ\displaystyle\left\|u\right\|_{J_{L}^{\alpha}{(\mathbb{R})}}=\left|u\right|_{J_{L}^{\alpha}{(\mathbb{R})}}+\left\|u\right\|_{L^{2}(\mathbb{R})},\qquad\left\|u\right\|_{J_{R}^{\alpha}(\mathbb{R})}=\left|u\right|_{J_{R}^{\alpha}(\mathbb{R})}+\left\|u\right\|_{L^{2}(\mathbb{R})},

(2.15)

and let JLα​(ℝ)superscriptsubscript𝐽𝐿𝛼ℝJ_{L}^{\alpha}{(\mathbb{R})} and JRα​(ℝ)superscriptsubscript𝐽𝑅𝛼ℝJ_{R}^{\alpha}(\mathbb{R}) denote the closure of Cc∞​(ℝ)superscriptsubscript𝐶𝑐ℝC_{c}^{\infty}(\mathbb{R}) with respect to ∥⋅∥JLα​(ℝ)\left\|\cdot\right\|_{J_{L}^{\alpha}{(\mathbb{R})}} and ∥⋅∥JRα​(ℝ)\left\|\cdot\right\|_{J_{R}^{\alpha}(\mathbb{R})} respectively.

It is evident from the definition (2.13) that

‖−(−Δ)α/2​u​(x)‖L2​(ℝ)=C​(‖Dxα−∞​u‖L2​(ℝ)+‖D∞αx​u‖L2​(ℝ)),1<α<2,formulae-sequencesubscriptnormsuperscriptΔ𝛼2𝑢𝑥superscript𝐿2ℝ𝐶subscriptnormsubscriptsuperscriptsubscript𝐷𝑥𝛼𝑢superscript𝐿2ℝsubscriptnormsubscriptsuperscriptsubscript𝐷𝛼𝑥𝑢superscript𝐿2ℝ1𝛼2\left\|-(-\Delta)^{\alpha/2}u(x)\right\|_{L^{2}(\mathbb{R})}=C\left(\left\|{}_{-\infty}D_{x}^{\alpha}u\right\|_{L^{2}(\mathbb{R})}+\left\|{}_{x}D_{\infty}^{\alpha}u\right\|_{L^{2}(\mathbb{R})}\right),\quad 1<\alpha<2,

(2.16)

where C𝐶C is a constant independent of u𝑢u. We wish to establish a connection between fractional Sobolev spaces and the aforementioned fractional derivative spaces. According to the result by Ervin et al. [12, Theorem 2.1], fractional derivative spaces JLα​ℝsuperscriptsubscript𝐽𝐿𝛼ℝJ_{L}^{\alpha}{\mathbb{R}} and JRα​(ℝ)superscriptsubscript𝐽𝑅𝛼ℝJ_{R}^{\alpha}(\mathbb{R}) with respect to norms defined by (2.15), are equal to the fractional Sobolev space Hα​(ℝ)superscript𝐻𝛼ℝH^{\alpha}(\mathbb{R}).
Therefore, using the norm on the fractional Laplacian (2.16), we conclude that the function u∈L2​(ℝ)𝑢superscript𝐿2ℝu\in L^{2}(\mathbb{R}) with −(−Δ)α/2​∂xu∈L2​(ℝ)superscriptΔ𝛼2subscript𝑥𝑢superscript𝐿2ℝ-(-\Delta)^{\alpha/2}\partial_{x}u\in L^{2}(\mathbb{R}) implies u∈H1+α​(ℝ)𝑢superscript𝐻1𝛼ℝu\in H^{1+\alpha}(\mathbb{R}).

Subsequently, we define the norm for discrete fractional derivative space h1+αsuperscriptℎ1𝛼h^{1+\alpha} as

‖un‖h1+α=‖un‖+‖D+​un‖+‖D+​D−​un‖+‖𝔻α​D​un‖.subscriptnormsuperscript𝑢𝑛superscriptℎ1𝛼normsuperscript𝑢𝑛normsubscript𝐷superscript𝑢𝑛normsubscript𝐷subscript𝐷superscript𝑢𝑛normsuperscript𝔻𝛼𝐷superscript𝑢𝑛\|u^{n}\|_{h^{1+\alpha}}=\|u^{n}\|+\|D_{+}u^{n}\|+\|D_{+}D_{-}u^{n}\|+\|\mathbb{D}^{\alpha}Du^{n}\|.

Now, we present a lemma establishing a fundamental connection between continuous and discrete derivative norms. Given the frequent use of this lemma, we include a proof for the sake of completeness.

###### Lemma 2.3.

Let u∈H1+α​(ℝ)𝑢superscript𝐻1𝛼ℝu\in H^{1+\alpha}(\mathbb{R}), α∈[1,2)𝛼12\alpha\in[1,2). Then there is a constant C𝐶C such that the discrete evaluation {u​(xj)}jsubscript𝑢subscript𝑥𝑗𝑗\{u(x_{j})\}_{j} satisfies

‖uΔ​x‖h1+α≤C​‖u‖H1+α​(ℝ).subscriptnormsubscript𝑢Δ𝑥superscriptℎ1𝛼𝐶subscriptnorm𝑢superscript𝐻1𝛼ℝ\|u_{\Delta x}\|_{h^{1+\alpha}}\leq C\|u\|_{H^{1+\alpha}(\mathbb{R})}.

(2.17)

###### Proof.

We begin by observing that the discrete fractional Laplacian 𝔻αsuperscript𝔻𝛼\mathbb{D}^{\alpha} commutes with other difference operators and derivatives. By the definition of ℓ2superscriptℓ2\ell^{2}-norm and 𝔻αsuperscript𝔻𝛼\mathbb{D}^{\alpha}, we have

‖𝔻α​D​u‖2superscriptnormsuperscript𝔻𝛼𝐷𝑢2\displaystyle\left\|\mathbb{D}^{\alpha}Du\right\|^{2}
≤C​‖𝔻α​∂xu‖2absent𝐶superscriptnormsuperscript𝔻𝛼subscript𝑥𝑢2\displaystyle\leq C\left\|\mathbb{D}^{\alpha}\partial_{x}u\right\|^{2}

=C​Δ​x​∑j(𝔻α​(∂xu)j)2absent𝐶Δ𝑥subscript𝑗superscriptsuperscript𝔻𝛼subscriptsubscript𝑥𝑢𝑗2\displaystyle=C\Delta x\sum_{j}\left(\mathbb{D}^{\alpha}(\partial_{x}u)_{j}\right)^{2}

=C​Δ​x​∑j(cα2​∑i∫x2​ix2​i+2∂xu​(xj+x2​i+1)+∂xu​(xj−x2​i+1)−2​∂xu​(xj)|x2​i+1|1+α​𝑑x)2absent𝐶Δ𝑥subscript𝑗superscriptsubscript𝑐𝛼2subscript𝑖superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2subscript𝑥𝑢subscript𝑥𝑗subscript𝑥2𝑖1subscript𝑥𝑢subscript𝑥𝑗subscript𝑥2𝑖12subscript𝑥𝑢subscript𝑥𝑗superscriptsubscript𝑥2𝑖11𝛼differential-d𝑥2\displaystyle=C\Delta x\sum_{j}\left(\frac{c_{\alpha}}{2}\sum_{i}\int_{x_{2i}}^{x_{2i+2}}\frac{\partial_{x}u(x_{j}+x_{2i+1})+\partial_{x}u(x_{j}-x_{2i+1})-2\partial_{x}u(x_{j})}{|x_{2i+1}|^{1+\alpha}}\,dx\right)^{2}

≤C​Δ​x​∑j(cα2​∫ℝ∂xu​(xj+y)+∂xu​(xj−y)−2​∂xu​(xj)|y|1+α​𝑑y)2absent𝐶Δ𝑥subscript𝑗superscriptsubscript𝑐𝛼2subscriptℝsubscript𝑥𝑢subscript𝑥𝑗𝑦subscript𝑥𝑢subscript𝑥𝑗𝑦2subscript𝑥𝑢subscript𝑥𝑗superscript𝑦1𝛼differential-d𝑦2\displaystyle\leq C\Delta x\sum_{j}\left(\frac{c_{\alpha}}{2}\int_{\mathbb{R}}\frac{\partial_{x}u(x_{j}+y)+\partial_{x}u(x_{j}-y)-2\partial_{x}u(x_{j})}{|y|^{1+\alpha}}\,dy\right)^{2}

≤C​‖u‖H1+α​(ℝ).absent𝐶subscriptnorm𝑢superscript𝐻1𝛼ℝ\displaystyle\leq C\left\|u\right\|_{H^{1+\alpha}(\mathbb{R})}.

We have used midpoint formula in the above calculation. Hence the result follows.
∎

Please note that from [5, Lemma 2.1], we have ‖uΔ​x‖h2≤C​‖u‖H2​(ℝ)subscriptnormsubscript𝑢Δ𝑥superscriptℎ2𝐶subscriptnorm𝑢superscript𝐻2ℝ\left\|u_{\Delta x}\right\|_{h^{2}}\leq C\left\|u\right\|_{H^{2}(\mathbb{R})}. We establish certain properties of the discrete fractional Laplacian (2.11) in the following lemma, mirroring the discrete counterparts of properties of the fractional Laplacian introduced in [10].

###### Lemma 2.4.

The discrete fractional Laplacian 𝔻αsuperscript𝔻𝛼\mathbb{D}^{\alpha}, α∈[1,2)𝛼12\alpha\in[1,2) defined by (2.11), exhibits linearity and possesses the following noteworthy properties for any pair of grid functions u,v∈ℓ2𝑢𝑣superscriptℓ2u,v\in\ell^{2}:

- (i)

(Symmetry) The discrete fractional Laplacian exhibits symmetry:

⟨𝔻α​u,v⟩=⟨u,𝔻α​v⟩.superscript𝔻𝛼𝑢𝑣𝑢superscript𝔻𝛼𝑣\langle\mathbb{D}^{\alpha}u,v\rangle=\langle u,\mathbb{D}^{\alpha}v\rangle.

- (ii)

(Translation invariant and Skew-symmetry) The discrete fractional Laplacian commutes with the difference operator:

⟨𝔻α​D​u,v⟩=−⟨u,𝔻α​D​v⟩=⟨D​𝔻α​u,v⟩.superscript𝔻𝛼𝐷𝑢𝑣𝑢superscript𝔻𝛼𝐷𝑣𝐷superscript𝔻𝛼𝑢𝑣\langle\mathbb{D}^{\alpha}Du,v\rangle=-\langle u,\mathbb{D}^{\alpha}Dv\rangle=\langle D\mathbb{D}^{\alpha}u,v\rangle.

- (iii)

The discrete fractional Laplacian further satisfies:

⟨𝔻α​D​u,u⟩=0.superscript𝔻𝛼𝐷𝑢𝑢0\langle\mathbb{D}^{\alpha}Du,u\rangle=0.

###### Proof.

Let u,v∈ℓ2𝑢𝑣superscriptℓ2u,v\in\ell^{2}. Then the definition of ℓ2superscriptℓ2\ell^{2}-inner product and discrete fractional Laplacian (2.11) provide

⟨𝔻α​u,v⟩superscript𝔻𝛼𝑢𝑣\displaystyle\langle\mathbb{D}^{\alpha}u,v\rangle
=Δ​x​∑jcαΔ​xα​∑k≠jvj​uk−vj​uj|k−j|1+α​(1−(−1)j−k)absentΔ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗subscript𝑣𝑗subscript𝑢𝑘subscript𝑣𝑗subscript𝑢𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘\displaystyle=\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{v_{j}u_{k}-v_{j}u_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k})

=Δ​x​∑jcαΔ​xα​∑k≠jvk−vj|k−j|1+α​(1−(−1)j−k)​ujabsentΔ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗subscript𝑣𝑘subscript𝑣𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘subscript𝑢𝑗\displaystyle=\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{v_{k}-v_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k})u_{j}

=⟨u,𝔻α​v⟩,absent𝑢superscript𝔻𝛼𝑣\displaystyle=\langle u,\mathbb{D}^{\alpha}v\rangle,

where we have used the change of variable. Thus (i) follows. Similarly, we have

⟨𝔻α​(D​u),v⟩=cαΔ​xα​∑j∑k≠jD​uk−D​uj|k−j|1+α​(1−(−1)j−k)​vj.superscript𝔻𝛼𝐷𝑢𝑣subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑗subscript𝑘𝑗𝐷subscript𝑢𝑘𝐷subscript𝑢𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘subscript𝑣𝑗\langle\mathbb{D}^{\alpha}(Du),v\rangle=\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{j}\sum_{k\neq j}\frac{Du_{k}-Du_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k})v_{j}.

This can be presented as

⟨𝔻α​(D​u),v⟩=Δ​x​∑jcαΔ​xα​∑k≠j(D​uk−D​uj)​vj|k−j|1+α​(1−(−1)j−k)=Δ​x​∑jcαΔ​xα​∑k≠jvj​(uk+1−uk−1)−vj​(uj+1−uj−1)2​Δ​x​|k−j|1+α​(1−(−1)j−k).superscript𝔻𝛼𝐷𝑢𝑣Δ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗𝐷subscript𝑢𝑘𝐷subscript𝑢𝑗subscript𝑣𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘Δ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗subscript𝑣𝑗subscript𝑢𝑘1subscript𝑢𝑘1subscript𝑣𝑗subscript𝑢𝑗1subscript𝑢𝑗12Δ𝑥superscript𝑘𝑗1𝛼1superscript1𝑗𝑘\begin{split}\langle\mathbb{D}^{\alpha}(Du),v\rangle&=\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{(Du_{k}-Du_{j})v_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k})\\
&=\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{v_{j}(u_{k+1}-u_{k-1})-v_{j}(u_{j+1}-u_{j-1})}{2\Delta x|k-j|^{1+\alpha}}(1-(-1)^{j-k}).\\
\end{split}

Suitable change of variables in j𝑗j and k𝑘k implies

⟨𝔻α​(D​u),v⟩=Δ​x​∑jcαΔ​xα​∑k≠juk​(vj−1−vj+1)−uj​(vj−1−vj+1)2​Δ​x​|k−j|1+α​(1−(−1)j−k)=−Δ​x​∑jcαΔ​xα​∑k≠j(D​vk−D​vj)​uj|k−j|1+α​(1−(−1)j−k)=−⟨𝔻α​(u),D​v⟩=⟨D​𝔻α​(u),v⟩,superscript𝔻𝛼𝐷𝑢𝑣Δ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗subscript𝑢𝑘subscript𝑣𝑗1subscript𝑣𝑗1subscript𝑢𝑗subscript𝑣𝑗1subscript𝑣𝑗12Δ𝑥superscript𝑘𝑗1𝛼1superscript1𝑗𝑘Δ𝑥subscript𝑗subscript𝑐𝛼Δsuperscript𝑥𝛼subscript𝑘𝑗𝐷subscript𝑣𝑘𝐷subscript𝑣𝑗subscript𝑢𝑗superscript𝑘𝑗1𝛼1superscript1𝑗𝑘superscript𝔻𝛼𝑢𝐷𝑣𝐷superscript𝔻𝛼𝑢𝑣\begin{split}\langle\mathbb{D}^{\alpha}(Du),v\rangle&=\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{u_{k}(v_{j-1}-v_{j+1})-u_{j}(v_{j-1}-v_{j+1})}{2\Delta x|k-j|^{1+\alpha}}(1-(-1)^{j-k})\\
&=-\Delta x\sum_{j}\frac{c_{\alpha}}{\Delta x^{\alpha}}\sum_{k\neq j}\frac{(Dv_{k}-Dv_{j})u_{j}}{|k-j|^{1+\alpha}}(1-(-1)^{j-k})\\
&=-\langle\mathbb{D}^{\alpha}(u),Dv\rangle\\
&=\langle D\mathbb{D}^{\alpha}(u),v\rangle,\end{split}

where we have used the property of symmetric difference operator over ℓ2superscriptℓ2\ell^{2}-inner product, hence (ii) follows. Property (iii) follows from (i) and (ii) by choosing v=u𝑣𝑢v=u in (ii).
∎

In the following lemma, we show that the discretization of fractional Laplacian is consistent.

###### Lemma 2.5.

Let ϕ∈Cc4​(ℝ)italic-ϕsuperscriptsubscript𝐶𝑐4ℝ\phi\in C_{c}^{4}(\mathbb{R}). Define a piece-wise constant function d𝑑d by

d​(x)=dj=𝔻α​(ϕ)​(xj)​ for ​x∈[xj,xj+1),j∈ℤ.formulae-sequence𝑑𝑥subscript𝑑𝑗superscript𝔻𝛼italic-ϕsubscript𝑥𝑗 for 𝑥subscript𝑥𝑗subscript𝑥𝑗1𝑗ℤd(x)=d_{j}=\mathbb{D}^{\alpha}(\phi)(x_{j})\text{ for }x\in[x_{j},x_{j+1}),\quad j\in\mathbb{Z}.

Then

limΔ​x→0‖(−(−Δ)α/2)​(ϕ)−d‖L2​(ℝ)=0.subscriptabsent→Δ𝑥0subscriptnormsuperscriptΔ𝛼2italic-ϕ𝑑superscript𝐿2ℝ0\lim_{\Delta x\xrightarrow[]{}0}\|(-(-\Delta)^{\alpha/2})(\phi)-d\|_{L^{2}(\mathbb{R})}=0.

###### Proof.

Let d~~𝑑\tilde{d} be an auxiliary function defined by

d~​(x)=d~j=(−(−Δ)α/2)​(ϕ)​(xj)​ for ​x∈[xj,xj+1),j∈ℤ.formulae-sequence~𝑑𝑥subscript~𝑑𝑗superscriptΔ𝛼2italic-ϕsubscript𝑥𝑗 for 𝑥subscript𝑥𝑗subscript𝑥𝑗1𝑗ℤ\tilde{d}(x)=\tilde{d}_{j}=(-(-\Delta)^{\alpha/2})(\phi)(x_{j})\text{ for }x\in[x_{j},x_{j+1}),\qquad j\in\mathbb{Z}.

By the triangle inequality, we have

‖(−(−Δ)α/2)​(ϕ)−d‖L2​(ℝ)≤‖(−(−Δ)α/2)​(ϕ)−d~‖L2​(ℝ)+‖d~−d‖L2​(ℝ).subscriptnormsuperscriptΔ𝛼2italic-ϕ𝑑superscript𝐿2ℝsubscriptnormsuperscriptΔ𝛼2italic-ϕ~𝑑superscript𝐿2ℝsubscriptnorm~𝑑𝑑superscript𝐿2ℝ\|(-(-\Delta)^{\alpha/2})(\phi)-d\|_{L^{2}(\mathbb{R})}\leq\|(-(-\Delta)^{\alpha/2})(\phi)-\tilde{d}\|_{L^{2}(\mathbb{R})}+\|\tilde{d}-d\|_{L^{2}(\mathbb{R})}.

Let us estimate the first term on the right hand side by

‖(−(−Δ)α/2)​(ϕ)−d~‖L2​(ℝ)2=∑j∫xjxj+1((−(−Δ)α/2)​(ϕ)​(x)−(−(−Δ)α/2)​(ϕ)​(xj))2​𝑑x=∑j∫xjxj+1(∫xjx((−(−Δ)α/2)​(ϕ))′​(ξ)​𝑑ξ)2​𝑑x=∑j∫xjxj+1(∫xjx1⋅((−(−Δ)α/2)​(ϕ′))​(ξ)​𝑑ξ)2​𝑑x≤∑j∫xjxj+1(∫xjxj+1((−(−Δ)α/2)​(ϕ′)​(ξ))2​𝑑ξ)​(x−xj)​𝑑x=Δ​x22​‖(−(−Δ)α/2)​(ϕ′)‖L2​(ℝ)2.superscriptsubscriptdelimited-∥∥superscriptΔ𝛼2italic-ϕ~𝑑superscript𝐿2ℝ2subscript𝑗superscriptsubscriptsubscript𝑥𝑗subscript𝑥𝑗1superscriptsuperscriptΔ𝛼2italic-ϕ𝑥superscriptΔ𝛼2italic-ϕsubscript𝑥𝑗2differential-d𝑥subscript𝑗superscriptsubscriptsubscript𝑥𝑗subscript𝑥𝑗1superscriptsuperscriptsubscriptsubscript𝑥𝑗𝑥superscriptsuperscriptΔ𝛼2italic-ϕ′𝜉differential-d𝜉2differential-d𝑥subscript𝑗superscriptsubscriptsubscript𝑥𝑗subscript𝑥𝑗1superscriptsuperscriptsubscriptsubscript𝑥𝑗𝑥⋅1superscriptΔ𝛼2superscriptitalic-ϕ′𝜉differential-d𝜉2differential-d𝑥subscript𝑗superscriptsubscriptsubscript𝑥𝑗subscript𝑥𝑗1superscriptsubscriptsubscript𝑥𝑗subscript𝑥𝑗1superscriptsuperscriptΔ𝛼2superscriptitalic-ϕ′𝜉2differential-d𝜉𝑥subscript𝑥𝑗differential-d𝑥Δsuperscript𝑥22subscriptsuperscriptdelimited-∥∥superscriptΔ𝛼2superscriptitalic-ϕ′2superscript𝐿2ℝ\begin{split}\Big{\|}\left(-(-\Delta)^{\alpha/2}\right)(\phi)-\tilde{d}\Big{\|}_{L^{2}(\mathbb{R})}^{2}&=\sum_{j}\int_{x_{j}}^{x_{j+1}}\left(\left(-(-\Delta)^{\alpha/2}\right)(\phi)(x)-\left(-(-\Delta)^{\alpha/2}\right)(\phi)(x_{j})\right)^{2}\,dx\\
&=\sum_{j}\int_{x_{j}}^{x_{j+1}}\left(\int_{x_{j}}^{x}\left(\left(-(-\Delta)^{\alpha/2}\right)(\phi)\right)^{\prime}(\xi)\,d\xi\right)^{2}\,dx\\
&=\sum_{j}\int_{x_{j}}^{x_{j+1}}\left(\int_{x_{j}}^{x}1\cdot\left(\left(-(-\Delta)^{\alpha/2}\right)(\phi^{\prime})\right)(\xi)\,d\xi\right)^{2}\,dx\\
&\leq\sum_{j}\int_{x_{j}}^{x_{j+1}}\left(\int_{x_{j}}^{x_{j+1}}\left(\left(-(-\Delta)^{\alpha/2}\right)(\phi^{\prime})(\xi)\right)^{2}\,d\xi\right)\,(x-x_{j})\,dx\\
&=\frac{\Delta x^{2}}{2}\|(-(-\Delta)^{\alpha/2})(\phi^{\prime})\|^{2}_{L^{2}(\mathbb{R})}.\end{split}

The second term can be estimated by

‖d~−d‖L2​(ℝ)2=Δ​x​∑j(dj−dj~)2≤Δ​x​∑|j|≤J(dj−dj~)2+2​Δ​x​∑|j|>J(dj2+dj2~)=:K1+K2.\begin{split}\|\tilde{d}-d\|_{L^{2}(\mathbb{R})}^{2}&=\Delta x\sum_{j}(d_{j}-\tilde{d_{j}})^{2}\leq\Delta x\sum_{|j|\leq J}(d_{j}-\tilde{d_{j}})^{2}+2\Delta x\sum_{|j|>J}(d_{j}^{2}+\tilde{d_{j}^{2}})\\
&=:K_{1}+K_{2}.\end{split}

From the equations (2.9) and (2.12), we have

dj−dj~=∑i(∫x2​ix2​i+2ξ​(xj,x2​i+1)​𝑑y−∫x2​ix2​i+2ξ​(xj,y)​𝑑y),subscript𝑑𝑗~subscript𝑑𝑗subscript𝑖superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝜉subscript𝑥𝑗subscript𝑥2𝑖1differential-d𝑦superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝜉subscript𝑥𝑗𝑦differential-d𝑦d_{j}-\tilde{d_{j}}=\sum_{i}\left(\int_{x_{2i}}^{x_{2i+2}}\xi(x_{j},x_{2i+1})\,dy-\int_{x_{2i}}^{x_{2i+2}}\xi(x_{j},y)\,dy\right),

where ξ​(x,y)=cα2​(ϕ​(x+y)−2​ϕ​(x)+ϕ​(x−y))/|y|1+α𝜉𝑥𝑦subscript𝑐𝛼2italic-ϕ𝑥𝑦2italic-ϕ𝑥italic-ϕ𝑥𝑦superscript𝑦1𝛼\xi(x,y)=\frac{c_{\alpha}}{2}(\phi(x+y)-2\phi(x)+\phi(x-y))/|y|^{1+\alpha}.

Employing the midpoint quadrature error formula and bound (2.10), we establish the following estimate:

|∫x2​ix2​i+2ξ​(xj,x2​i+1)​𝑑y−∫x2​ix2​i+2ξ​(xj,y)​𝑑y|≤cα​C​Δ​x4−α​‖ϕ(4)‖L∞|2​i+1|α−1,superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝜉subscript𝑥𝑗subscript𝑥2𝑖1differential-d𝑦superscriptsubscriptsubscript𝑥2𝑖subscript𝑥2𝑖2𝜉subscript𝑥𝑗𝑦differential-d𝑦subscript𝑐𝛼𝐶Δsuperscript𝑥4𝛼subscriptnormsuperscriptitalic-ϕ4superscript𝐿superscript2𝑖1𝛼1\begin{split}\left|\int_{x_{2i}}^{x_{2i+2}}\xi(x_{j},x_{2i+1})\,dy-\int_{x_{2i}}^{x_{2i+2}}\xi(x_{j},y)\,dy\right|&\leq c_{\alpha}C\Delta x^{4-\alpha}\frac{\|\phi^{(4)}\|_{L^{\infty}}}{|2i+1|^{\alpha-1}},\end{split}

where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x and j𝑗j. Furthermore, since ϕitalic-ϕ\phi has compact support, the summation over i𝑖i contains only finite number of terms, say Nϕ/Δ​xsubscript𝑁italic-ϕΔ𝑥N_{\phi}/\Delta x, independent of j𝑗j. Consequently,

|dj−dj~|≤Nϕ​cα​C​Δ​x3−α​‖ϕ(4)‖L∞​(ℝ)​Mα,subscript𝑑𝑗~subscript𝑑𝑗subscript𝑁italic-ϕsubscript𝑐𝛼𝐶Δsuperscript𝑥3𝛼subscriptnormsuperscriptitalic-ϕ4superscript𝐿ℝsubscript𝑀𝛼|d_{j}-\tilde{d_{j}}|\leq N_{\phi}c_{\alpha}C\Delta x^{3-\alpha}\|\phi^{(4)}\|_{L^{\infty}(\mathbb{R})}M_{\alpha},

where Mαsubscript𝑀𝛼M_{\alpha} is an upper bound for ∑|i|≤K1|2​i+1|α−1subscript𝑖𝐾1superscript2𝑖1𝛼1\sum_{|i|\leq K}\frac{1}{|2i+1|^{\alpha-1}} with finite K𝐾K arising from the support of ϕitalic-ϕ\phi. Thus we have

K1≤Nϕ​Mα​cα​C​Δ​x3−α​‖ϕ(4)‖L∞​(ℝ).subscript𝐾1subscript𝑁italic-ϕsubscript𝑀𝛼subscript𝑐𝛼𝐶Δsuperscript𝑥3𝛼subscriptnormsuperscriptitalic-ϕ4superscript𝐿ℝK_{1}\leq N_{\phi}M_{\alpha}c_{\alpha}C\Delta x^{3-\alpha}\|\phi^{(4)}\|_{L^{\infty}(\mathbb{R})}.

Given the finite-ness of ∑jdj2subscript𝑗superscriptsubscript𝑑𝑗2\sum_{j}d_{j}^{2} and ∑jdj2~subscript𝑗~superscriptsubscript𝑑𝑗2\sum_{j}\tilde{d_{j}^{2}}, a judicious choice of a sufficiently large J𝐽J ensures a small K2subscript𝐾2K_{2} and subsequently, Δ​x→0→Δ𝑥0\Delta x\to 0 leads to a small K1subscript𝐾1K_{1}. Consequently, ‖dj−dj~‖L2​(ℝ)→0→subscriptnormsubscript𝑑𝑗~subscript𝑑𝑗superscript𝐿2ℝ0\|d_{j}-\tilde{d_{j}}\|_{L^{2}(\mathbb{R})}\to 0 as Δ​x→0→Δ𝑥0\Delta x\to 0. Hence the consistency of discrete fractional Laplacian is established.
∎

## 3. Fully discrete semi-implicit scheme

We propose the following Euler implicit temporal discretized finite difference scheme to obtain approximate solutions of the fractional KdV equation (1.1):

ujn+1=u¯jn−Δ​t​u¯jn​D​ujn−Δ​t​𝔻α​(D​ujn+1),n∈ℕ0,j∈ℤ.formulae-sequencesuperscriptsubscript𝑢𝑗𝑛1superscriptsubscript¯𝑢𝑗𝑛Δ𝑡superscriptsubscript¯𝑢𝑗𝑛𝐷superscriptsubscript𝑢𝑗𝑛Δ𝑡superscript𝔻𝛼𝐷subscriptsuperscript𝑢𝑛1𝑗formulae-sequence𝑛subscriptℕ0𝑗ℤu_{j}^{n+1}=\bar{u}_{j}^{n}-\Delta t\bar{u}_{j}^{n}Du_{j}^{n}-\Delta t\mathbb{D}^{\alpha}(Du^{n+1}_{j}),\qquad n\in\mathbb{N}_{0},\hskip 2.84544ptj\in\mathbb{Z}.

(3.1)

For the initial data, we have

uj0=u0​(xj),j∈ℤ.formulae-sequencesuperscriptsubscript𝑢𝑗0subscript𝑢0subscript𝑥𝑗𝑗ℤu_{j}^{0}=u_{0}(x_{j}),\qquad j\in\mathbb{Z}.

###### Remark 3.1.

We must ensure that the above scheme (3.1) is solvable with respect to un+1superscript𝑢𝑛1u^{n+1}. Solvability can be achieved by rewriting the scheme (3.1) as follows:

(1+Δ​t​𝔻α​D)​ujn+1=u¯jn−Δ​t​u¯jn​D​ujn.1Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛1superscriptsubscript¯𝑢𝑗𝑛Δ𝑡superscriptsubscript¯𝑢𝑗𝑛𝐷superscriptsubscript𝑢𝑗𝑛(1+\Delta t\mathbb{D}^{\alpha}D)u_{j}^{n+1}=\bar{u}_{j}^{n}-\Delta t\bar{u}_{j}^{n}Du_{j}^{n}.

Taking the inner product with un+1superscript𝑢𝑛1u^{n+1}, we get:

⟨(1+Δ​t​𝔻α​D)​un+1,un+1⟩1Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛1superscript𝑢𝑛1\displaystyle\langle(1+\Delta t\mathbb{D}^{\alpha}D)u^{n+1},u^{n+1}\rangle
=‖un+1‖2+Δ​t​⟨𝔻α​D​un+1,un+1⟩=‖un+1‖2.absentsuperscriptnormsuperscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛1superscript𝑢𝑛1superscriptnormsuperscript𝑢𝑛12\displaystyle=\left\|u^{n+1}\right\|^{2}+\Delta t\langle\mathbb{D}^{\alpha}Du^{n+1},u^{n+1}\rangle=\left\|u^{n+1}\right\|^{2}.

Hence we obtain

‖un+1‖≤‖(1+Δ​t​𝔻α​D)​un+1‖=‖u¯n−Δ​t​u¯n​D​un‖.normsuperscript𝑢𝑛1norm1Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛1normsuperscript¯𝑢𝑛Δ𝑡superscript¯𝑢𝑛𝐷superscript𝑢𝑛\left\|u^{n+1}\right\|\leq\left\|(1+\Delta t\mathbb{D}^{\alpha}D)u^{n+1}\right\|=\left\|\bar{u}^{n}-\Delta t\bar{u}^{n}Du^{n}\right\|.

###### Remark 3.2.

In the discrete scheme (3.1), the discretization of the convective term is similar to Holden et al. [19]. However, the main distinction with [19] being the inclusion of the discretized fractional term. Consequently, in the following analysis wherever the fractional term does not play a role, we refer to the approach described in [19].

###### Remark 3.3.

The aforementioned scheme aligns with the operator splitting scheme developed in [8] for the fractional KdV equation (1.1) and in [17, 18] for the KdV and generalized KdV equation. The scheme can be decomposed as follows:

ujn+1/2=u¯jn−Δ​t4​Δ​x​((uj+1n)2−(uj−1n)2)superscriptsubscript𝑢𝑗𝑛12superscriptsubscript¯𝑢𝑗𝑛Δ𝑡4Δ𝑥superscriptsuperscriptsubscript𝑢𝑗1𝑛2superscriptsuperscriptsubscript𝑢𝑗1𝑛2u_{j}^{n+1/2}=\bar{u}_{j}^{n}-\frac{\Delta t}{4\Delta x}\left((u_{j+1}^{n})^{2}-(u_{j-1}^{n})^{2}\right)

utilizing u¯jn​D​ujn=12​D​(ujn)2subscriptsuperscript¯𝑢𝑛𝑗𝐷superscriptsubscript𝑢𝑗𝑛12𝐷superscriptsuperscriptsubscript𝑢𝑗𝑛2\bar{u}^{n}_{j}Du_{j}^{n}=\frac{1}{2}D(u_{j}^{n})^{2}. It is noteworthy that un+1/2superscript𝑢𝑛12u^{n+1/2} solves the Lax-Friedrichs scheme applied to unsuperscript𝑢𝑛u^{n} for the nonlinear part of the KdV equation. Subsequently

un+1−un+1/2Δ​t=−𝔻α​(D​un+1),superscript𝑢𝑛1superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛1\frac{u^{n+1}-u^{n+1/2}}{\Delta t}=-\mathbb{D}^{\alpha}(Du^{n+1}),

where un+1superscript𝑢𝑛1u^{n+1} is the approximation of the implicit scheme for the linear dispersive equation with the fractional Laplacian: ut−(−Δ)α/2​ux=0subscript𝑢𝑡superscriptΔ𝛼2subscript𝑢𝑥0u_{t}-(-\Delta)^{\alpha/2}u_{x}=0. If we denote these two solutions as SBsubscript𝑆𝐵S_{B} and SDsubscript𝑆𝐷S_{D} respectively, then

un+1=(SD∘SB)​unsuperscript𝑢𝑛1subscript𝑆𝐷subscript𝑆𝐵superscript𝑢𝑛u^{n+1}=(S_{D}\circ S_{B})u^{n}

solves the implicit scheme (3.1).
The proofs presented here can be adopted to demonstrate the convergence of the operator splitting method for the fractional KdV equation (1.1).

We state and proof the stability lemma which is the main ingredient in the convergence analysis:

###### Lemma 3.4.

Let unsuperscript𝑢𝑛u^{n} be an approximate solution obtained by the scheme (3.1). Assume that CFL condition satisfies:

λ​‖u0‖​(13+12​λ​‖u0‖)<1−δ2,δ∈(0,1),formulae-sequence𝜆normsuperscript𝑢01312𝜆normsuperscript𝑢01𝛿2𝛿01\lambda\|u^{0}\|(\frac{1}{3}+\frac{1}{2}\lambda\|u^{0}\|)<\frac{1-\delta}{2},\quad\delta\in(0,1),

(3.2)

where λ=Δ​t/Δ​x3/2𝜆Δ𝑡Δsuperscript𝑥32\lambda=\Delta t/\Delta x^{3/2}. Then

‖un+1‖2+Δ​x3​λ2​‖𝔻α​(D​ujn+1)‖2+δ​Δ​x2​‖D​un‖2≤‖un‖2.superscriptnormsuperscript𝑢𝑛12Δsuperscript𝑥3superscript𝜆2superscriptnormsuperscript𝔻𝛼𝐷subscriptsuperscript𝑢𝑛1𝑗2𝛿Δsuperscript𝑥2superscriptnorm𝐷superscript𝑢𝑛2superscriptnormsuperscript𝑢𝑛2\|u^{n+1}\|^{2}+\Delta x^{3}\lambda^{2}\|\mathbb{D}^{\alpha}(Du^{n+1}_{j})\|^{2}+\delta\Delta x^{2}\|Du^{n}\|^{2}\leq\|u^{n}\|^{2}.

(3.3)

###### Proof.

Following the approach of Holden et al. [19] and expressing the Burgers’ term as

B​(u)=u¯−Δ​t​u¯​D​u=u¯−Δ​t2​D​u2,𝐵𝑢¯𝑢Δ𝑡¯𝑢𝐷𝑢¯𝑢Δ𝑡2𝐷superscript𝑢2B(u)=\bar{u}-\Delta t\bar{u}Du=\bar{u}-\frac{\Delta t}{2}Du^{2},

we have

‖B​(u)‖2≤‖u‖2−δ​Δ​x2​‖D​u‖2superscriptnorm𝐵𝑢2superscriptnorm𝑢2𝛿Δsuperscript𝑥2superscriptnorm𝐷𝑢2\|B(u)\|^{2}\leq\|u\|^{2}-\delta\Delta x^{2}\|Du\|^{2}

(3.4)

provided the CFL condition (3.2) holds.
Next, we examine the scheme (3.1) and it can be represented as

un+1=B​(un)−Δ​t​𝔻α​(D​un+1).superscript𝑢𝑛1𝐵superscript𝑢𝑛Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛1u^{n+1}=B(u^{n})-\Delta t\mathbb{D}^{\alpha}(Du^{n+1}).

Taking the ℓ2superscriptℓ2\ell^{2}-norm, we obtain

‖B​(un)‖2superscriptnorm𝐵superscript𝑢𝑛2\displaystyle\|B(u^{n})\|^{2}
=‖un+1‖2+2​Δ​t​(un+1,𝔻α​(D​un+1))+Δ​t2​‖𝔻α​(D​un+1)‖2absentsuperscriptnormsuperscript𝑢𝑛122Δ𝑡superscript𝑢𝑛1superscript𝔻𝛼𝐷superscript𝑢𝑛1Δsuperscript𝑡2superscriptnormsuperscript𝔻𝛼𝐷superscript𝑢𝑛12\displaystyle=\|u^{n+1}\|^{2}+2\Delta t(u^{n+1},\mathbb{D}^{\alpha}(Du^{n+1}))+\Delta t^{2}\|\mathbb{D}^{\alpha}(Du^{n+1})\|^{2}

=‖un+1‖2+Δ​t2​‖𝔻α​(D​un+1)‖2absentsuperscriptnormsuperscript𝑢𝑛12Δsuperscript𝑡2superscriptnormsuperscript𝔻𝛼𝐷superscript𝑢𝑛12\displaystyle=\|u^{n+1}\|^{2}+\Delta t^{2}\|\mathbb{D}^{\alpha}(Du^{n+1})\|^{2}

=‖un+1‖2+Δ​x3​λ2​‖𝔻α​(D​un+1)‖2.absentsuperscriptnormsuperscript𝑢𝑛12Δsuperscript𝑥3superscript𝜆2superscriptnormsuperscript𝔻𝛼𝐷superscript𝑢𝑛12\displaystyle=\|u^{n+1}\|^{2}+\Delta x^{3}\lambda^{2}\|\mathbb{D}^{\alpha}(Du^{n+1})\|^{2}.

(3.5)

Therefore, estimates (3.4) and (3.5) imply

‖un+1‖2+Δ​x3​λ2​‖𝔻α​(D​un+1)‖2+δ​Δ​x2​‖D​un‖2≤‖un‖2.superscriptnormsuperscript𝑢𝑛12Δsuperscript𝑥3superscript𝜆2superscriptnormsuperscript𝔻𝛼𝐷superscript𝑢𝑛12𝛿Δsuperscript𝑥2superscriptnorm𝐷superscript𝑢𝑛2superscriptnormsuperscript𝑢𝑛2\|u^{n+1}\|^{2}+\Delta x^{3}\lambda^{2}\|\mathbb{D}^{\alpha}(Du^{n+1})\|^{2}+\delta\Delta x^{2}\|Du^{n}\|^{2}\leq\|u^{n}\|^{2}.

Hence the stability of (3.1) is ensured.
∎

Subsequently, we explore the temporal derivative bound of the scheme (3.1). This bound is significant in the forthcoming convergence proof, as the proof relies on the compactness theorem. We start with by introducing the following notation for a given function v𝑣v

D+t​v​(t)=1Δ​t​(v​(t+Δ​t)−v​(t)).superscriptsubscript𝐷𝑡𝑣𝑡1Δ𝑡𝑣𝑡Δ𝑡𝑣𝑡\displaystyle D_{+}^{t}v(t)=\frac{1}{\Delta t}(v(t+\Delta t)-v(t)).

###### Lemma 3.5.

Let unsuperscript𝑢𝑛u^{n} be an approximate solution obtained by the scheme (3.1). Assume that λ𝜆\lambda from Lemma 3.4 satisfies:

6​‖u0‖2​λ2+‖u0‖​λ<1−δ~2,δ~∈(0,1).formulae-sequence6superscriptnormsubscript𝑢02superscript𝜆2normsubscript𝑢0𝜆1~𝛿2~𝛿016\|u_{0}\|^{2}\lambda^{2}+\|u_{0}\|\lambda<\frac{1-\tilde{\delta}}{2},\quad\tilde{\delta}\in(0,1).

(3.6)

Then there holds

‖D+t​un‖2+Δ​t2​‖𝔻α​D​(D+t​un)‖2+δ~​Δ​x2​‖D​(D+t​un−1)‖2≤(1+3​Δ​t​‖D​un‖∞)​‖D+t​un−1‖2.superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛2Δsuperscript𝑡2superscriptnormsuperscript𝔻𝛼𝐷superscriptsubscript𝐷𝑡superscript𝑢𝑛2~𝛿Δsuperscript𝑥2superscriptnorm𝐷superscriptsubscript𝐷𝑡superscript𝑢𝑛1213Δ𝑡subscriptnorm𝐷superscript𝑢𝑛superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛12\|D_{+}^{t}u^{n}\|^{2}+\Delta t^{2}\|\mathbb{D}^{\alpha}D(D_{+}^{t}u^{n})\|^{2}+\tilde{\delta}\Delta x^{2}\|D(D_{+}^{t}u^{n-1})\|^{2}\leq(1+3\Delta t\|Du^{n}\|_{\infty})\|D_{+}^{t}u^{n-1}\|^{2}.

(3.7)

Moreover, the following estimates hold:

‖D+t​un‖normsuperscriptsubscript𝐷𝑡superscript𝑢𝑛\displaystyle\|D_{+}^{t}u^{n}\|
≤C,absent𝐶\displaystyle\leq C,

(3.8)

‖un‖h1+αsubscriptnormsuperscript𝑢𝑛superscriptℎ1𝛼\displaystyle\|u^{n}\|_{h^{1+\alpha}}
≤C,absent𝐶\displaystyle\leq C,

(3.9)

where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x.

###### Proof.

From the scheme (3.1), we have

D+t​ujn=D+t​u¯jn−1−G​(un)j−Δ​t​𝔻α​(D​D+t​ujn),superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛superscriptsubscript𝐷𝑡superscriptsubscript¯𝑢𝑗𝑛1𝐺subscriptsuperscript𝑢𝑛𝑗Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝐷𝑡subscriptsuperscript𝑢𝑛𝑗\displaystyle D_{+}^{t}u_{j}^{n}=D_{+}^{t}\bar{u}_{j}^{n-1}-G(u^{n})_{j}-\Delta t\mathbb{D}^{\alpha}(DD_{+}^{t}u^{n}_{j}),

(3.10)

where G​(un)=u¯n​D​un−u¯n−1​D​un−1𝐺superscript𝑢𝑛superscript¯𝑢𝑛𝐷superscript𝑢𝑛superscript¯𝑢𝑛1𝐷superscript𝑢𝑛1G(u^{n})=\bar{u}^{n}Du^{n}-\bar{u}^{n-1}Du^{n-1}, and it can be further simplified as

G​(un)𝐺superscript𝑢𝑛\displaystyle G(u^{n})
=u¯n​D​un−u¯n−1​D​un−1absentsuperscript¯𝑢𝑛𝐷superscript𝑢𝑛superscript¯𝑢𝑛1𝐷superscript𝑢𝑛1\displaystyle=\bar{u}^{n}Du^{n}-\bar{u}^{n-1}Du^{n-1}

=Δ​t​(D+t​u¯n−1​D​un+u¯n−1​D​D+t​un−1)absentΔ𝑡superscriptsubscript𝐷𝑡superscript¯𝑢𝑛1𝐷superscript𝑢𝑛superscript¯𝑢𝑛1𝐷superscriptsubscript𝐷𝑡superscript𝑢𝑛1\displaystyle=\Delta t(D_{+}^{t}\bar{u}^{n-1}Du^{n}+\bar{u}^{n-1}DD_{+}^{t}u^{n-1})

=Δ​t​(D+t​u¯n−1​D​un+u¯n​D​D+t​un−1−Δ​t​D+t​u¯n−1​D​D+t​un−1)absentΔ𝑡superscriptsubscript𝐷𝑡superscript¯𝑢𝑛1𝐷superscript𝑢𝑛superscript¯𝑢𝑛𝐷superscriptsubscript𝐷𝑡superscript𝑢𝑛1Δ𝑡superscriptsubscript𝐷𝑡superscript¯𝑢𝑛1𝐷superscriptsubscript𝐷𝑡superscript𝑢𝑛1\displaystyle=\Delta t(D_{+}^{t}\bar{u}^{n-1}Du^{n}+\bar{u}^{n}DD_{+}^{t}u^{n-1}-\Delta tD_{+}^{t}\bar{u}^{n-1}DD_{+}^{t}u^{n-1})

=Δ​t​(D​(un​D+t​un−1)−Δ​t2​D​(D+t​un−1)2).absentΔ𝑡𝐷superscript𝑢𝑛superscriptsubscript𝐷𝑡superscript𝑢𝑛1Δ𝑡2𝐷superscriptsuperscriptsubscript𝐷𝑡superscript𝑢𝑛12\displaystyle=\Delta t\left(D(u^{n}D_{+}^{t}u^{n-1})-\frac{\Delta t}{2}D(D_{+}^{t}u^{n-1})^{2}\right).

With the help of the above identity in the equation (3.10) and setting τn:=D+t​un−1assignsuperscript𝜏𝑛superscriptsubscript𝐷𝑡superscript𝑢𝑛1\tau^{n}:=D_{+}^{t}u^{n-1} yield

τn+1=σn−Δ​t​𝔻α​D​τn+1,superscript𝜏𝑛1superscript𝜎𝑛Δ𝑡superscript𝔻𝛼𝐷superscript𝜏𝑛1\tau^{n+1}=\sigma^{n}-\Delta t\mathbb{D}^{\alpha}D\tau^{n+1},

(3.11)

where σ𝜎\sigma is defined by

σ=τ¯−Δ​t​D​(u​τ)+Δ​t22​D​τ2.𝜎¯𝜏Δ𝑡𝐷𝑢𝜏Δsuperscript𝑡22𝐷superscript𝜏2\sigma=\bar{\tau}-\Delta tD(u\tau)+\frac{\Delta t^{2}}{2}D\tau^{2}.

(3.12)

Now we will follow the same strategy applied in [19, Lemma 3.2] to evaluate (3.12), which gives the following bound

12​‖σn‖2+δ~​Δ​x22​‖D​τn‖2≤12​‖τn‖2+3−δ~2​Δ​t​‖D​un‖∞​‖τn‖2.12superscriptnormsuperscript𝜎𝑛2~𝛿Δsuperscript𝑥22superscriptnorm𝐷superscript𝜏𝑛212superscriptnormsuperscript𝜏𝑛23~𝛿2Δ𝑡subscriptnorm𝐷superscript𝑢𝑛superscriptnormsuperscript𝜏𝑛2\frac{1}{2}\|\sigma^{n}\|^{2}+\tilde{\delta}\frac{\Delta x^{2}}{2}\|D\tau^{n}\|^{2}\leq\frac{1}{2}\|\tau^{n}\|^{2}+\frac{3-\tilde{\delta}}{2}\Delta t\|Du^{n}\|_{\infty}\|\tau^{n}\|^{2}.

(3.13)

Squaring equation (3.11) and summing the resulting equation over j∈ℤ𝑗ℤj\in\mathbb{Z} yields

‖σn‖2=‖τn+1‖2+Δ​t2​‖𝔻α​D​τn+1‖2.superscriptnormsuperscript𝜎𝑛2superscriptnormsuperscript𝜏𝑛12Δsuperscript𝑡2superscriptnormsuperscript𝔻𝛼𝐷superscript𝜏𝑛12\|\sigma^{n}\|^{2}=\|\tau^{n+1}\|^{2}+\Delta t^{2}\|\mathbb{D}^{\alpha}D\tau^{n+1}\|^{2}.

Therefore, substituting the above identity in (3.13), we have

‖τn+1‖2+Δ​t2​‖𝔻α​D​τn+1‖2+δ~​Δ​x2​‖D​τn‖2≤(1+3​Δ​t​‖D​un‖∞)​‖τn‖2.superscriptdelimited-∥∥superscript𝜏𝑛12Δsuperscript𝑡2superscriptdelimited-∥∥superscript𝔻𝛼𝐷superscript𝜏𝑛12~𝛿Δsuperscript𝑥2superscriptdelimited-∥∥𝐷superscript𝜏𝑛213Δ𝑡subscriptdelimited-∥∥𝐷superscript𝑢𝑛superscriptdelimited-∥∥superscript𝜏𝑛2\begin{split}\|\tau^{n+1}\|^{2}+\Delta t^{2}\|\mathbb{D}^{\alpha}D\tau^{n+1}\|^{2}+\tilde{\delta}\Delta x^{2}\|D\tau^{n}\|^{2}\leq(1+3\Delta t\|Du^{n}\|_{\infty})\|\tau^{n}\|^{2}.\end{split}

(3.14)

This gives the estimate (3.7). Dropping the positive term from left hand side in (3.14), we have

‖τn+1‖2≤‖τn‖2+3​Δ​t​‖D​un‖∞​‖τn‖2.superscriptdelimited-∥∥superscript𝜏𝑛12superscriptdelimited-∥∥superscript𝜏𝑛23Δ𝑡subscriptdelimited-∥∥𝐷superscript𝑢𝑛superscriptdelimited-∥∥superscript𝜏𝑛2\begin{split}\|\tau^{n+1}\|^{2}\leq\|\tau^{n}\|^{2}+3\Delta t\|Du^{n}\|_{\infty}\|\tau^{n}\|^{2}.\end{split}

Again, following the approach of Holden et al. [19] ensures the existence of T>0𝑇0T>0 such that the following estimate hold:

‖τn‖≤C,(n+1)​Δ​t≤T,formulae-sequencenormsuperscript𝜏𝑛𝐶𝑛1Δ𝑡𝑇\|\tau^{n}\|\leq C,\qquad(n+1)\Delta t\leq T,

where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x. This is a temporal derivative bound.

Finally, utilizing these bounds, we obtain

‖𝔻α​D​un‖≤‖D+t​un‖+‖u¯n‖∞​‖D​un‖≤C,(n+1)​Δ​t≤T,formulae-sequencenormsuperscript𝔻𝛼𝐷superscript𝑢𝑛normsuperscriptsubscript𝐷𝑡superscript𝑢𝑛subscriptnormsuperscript¯𝑢𝑛norm𝐷superscript𝑢𝑛𝐶𝑛1Δ𝑡𝑇\|\mathbb{D}^{\alpha}Du^{n}\|\leq\|D_{+}^{t}u^{n}\|+\|\bar{u}^{n}\|_{\infty}\|Du^{n}\|\leq C,\quad(n+1)\Delta t\leq T,

where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x. This implies (3.9).
∎

### 3.1. Convergence

We follow the approach outlined by Sjöberg [33] to establish the convergence of the scheme for t<T𝑡𝑇t<T. The construction of the approximate solution uΔ​xsubscript𝑢Δ𝑥u_{\Delta x} is carried out in two distinct steps of the piece-wise interpolation. Firstly, we perform interpolation in space for each tnsubscript𝑡𝑛t_{n}:

un​(x)=ujn+D​ujn​(x−xj),x∈[xj,xj+1),j∈ℤ.\begin{split}u^{n}(x)=u_{j}^{n}+Du_{j}^{n}(x-x_{j}),\quad x\in[x_{j},x_{j+1}),\quad j\in\mathbb{Z}.\end{split}

(3.15)

Following this, we perform interpolation in time for all x∈ℝ𝑥ℝx\in\mathbb{R}:

uΔ​x​(x,t)=un​(x)+D+t​un​(x)​(t−tn),t∈[tn,tn+1),(n+1)​Δ​t≤T¯.formulae-sequencesubscript𝑢Δ𝑥𝑥𝑡superscript𝑢𝑛𝑥superscriptsubscript𝐷𝑡superscript𝑢𝑛𝑥𝑡superscript𝑡𝑛formulae-sequence𝑡subscript𝑡𝑛subscript𝑡𝑛1𝑛1Δ𝑡¯𝑇u_{\Delta x}(x,t)=u^{n}(x)+D_{+}^{t}u^{n}(x)(t-t^{n}),\quad t\in[t_{n},t_{n+1}),\quad(n+1)\Delta t\leq\bar{T}.

(3.16)

Note that the interpolation satisfies at nodes, i.e., for all j∈ℤ𝑗ℤj\in\mathbb{Z} and n∈ℕ0𝑛subscriptℕ0n\in\mathbb{N}_{0}, uΔ​x​(xj,tn)=ujnsubscript𝑢Δ𝑥subscript𝑥𝑗subscript𝑡𝑛superscriptsubscript𝑢𝑗𝑛u_{\Delta x}(x_{j},t_{n})=u_{j}^{n}.

With these interpolations in place, we proceed to state and prove the main result of this section.

###### Theorem 3.6.

Let u0∈H1+α​(ℝ)subscript𝑢0superscript𝐻1𝛼ℝu_{0}\in H^{1+\alpha}(\mathbb{R}), α∈[1,2)𝛼12\alpha\in[1,2). Then there is a finite time T>0𝑇0T>0, depending on ‖u0‖H1+α​(ℝ)subscriptnormsubscript𝑢0superscript𝐻1𝛼ℝ\|u_{0}\|_{H^{1+\alpha}(\mathbb{R})}, such that for t≤T𝑡𝑇t\leq T and Δ​t=𝒪​(Δ​x3/2)Δ𝑡𝒪Δsuperscript𝑥32\Delta t=\mathcal{O}(\Delta x^{3/2}), the sequence of approximate solutions obtained by the scheme (3.1) uniformly converges to the unique solution of the fractional KdV equation (1.1) in C​(ℝ×[0,T])𝐶ℝ0𝑇C(\mathbb{R}\times[0,T]) as Δ​x→0absent→Δ𝑥0\Delta x\xrightarrow[]{}0.

###### Proof.

Interpolation (3.16) gives that uΔ​xsubscript𝑢Δ𝑥u_{\Delta x} is smooth enough. Differentiating uΔ​xsubscript𝑢Δ𝑥u_{\Delta x} in both space and time for x∈[xj,xj+1)𝑥subscript𝑥𝑗subscript𝑥𝑗1x\in[x_{j},x_{j+1}) and t∈[tn,tn+1)𝑡subscript𝑡𝑛subscript𝑡𝑛1t\in[t_{n},t_{n+1}) gives

∂xuΔ​x​(x,t)=D​ujn+D+t​(D​ujn)​(t−tn),∂tuΔ​x​(x,t)=Dt+​un​(x),formulae-sequencesubscript𝑥subscript𝑢Δ𝑥𝑥𝑡𝐷superscriptsubscript𝑢𝑗𝑛superscriptsubscript𝐷𝑡𝐷superscriptsubscript𝑢𝑗𝑛𝑡subscript𝑡𝑛subscript𝑡subscript𝑢Δ𝑥𝑥𝑡subscriptsuperscript𝐷𝑡superscript𝑢𝑛𝑥\begin{split}\partial_{x}u_{\Delta x}(x,t)&=Du_{j}^{n}+D_{+}^{t}\left(Du_{j}^{n}\right)(t-t_{n}),\\
\partial_{t}u_{\Delta x}(x,t)&=D^{+}_{t}u^{n}(x),\end{split}

which clearly implies for all t≤T𝑡𝑇t\leq T,

‖uΔ​x​(⋅,t)‖L2​(ℝ)≤‖u0‖L2​(ℝ),subscriptnormsubscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝsubscriptnormsubscript𝑢0superscript𝐿2ℝ\displaystyle\|u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}\leq\|u_{0}\|_{L^{2}(\mathbb{R})},

(3.17)

‖∂xuΔ​x​(⋅,t)‖L2​(ℝ)≤C,subscriptnormsubscript𝑥subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ𝐶\displaystyle\|\partial_{x}u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}\leq C,

(3.18)

‖∂tuΔ​x​(⋅,t)‖L2​(ℝ)≤C,subscriptnormsubscript𝑡subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ𝐶\displaystyle\|\partial_{t}u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}\leq C,

(3.19)

‖−(−Δ)α/2​∂xuΔ​x​(⋅,t)‖L2​(ℝ)≤C,subscriptnormsuperscriptΔ𝛼2subscript𝑥subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ𝐶\displaystyle\|-(-\Delta)^{\alpha/2}\partial_{x}u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}\leq C,

(3.20)

where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x. The first estimate (3.17) follows from the exact integration of the square of (3.16) over each interval [xj,xj+1)subscript𝑥𝑗subscript𝑥𝑗1[x_{j},x_{j+1}) and summation over j𝑗j. Similarly, estimate (3.3) implies (3.18) and (3.8) implies (3.19). Since for t<T𝑡𝑇t<T, ∂xuΔ​x​(⋅,t)subscript𝑥subscript𝑢Δ𝑥⋅𝑡\partial_{x}u_{\Delta x}(\cdot,t) is constant in the interval [xj,xj+1)subscript𝑥𝑗subscript𝑥𝑗1[x_{j},x_{j+1}) for all j∈ℤ𝑗ℤj\in\mathbb{Z}, we can take into account the Lemma 2.3 and bound (3.9) from the Lemma 3.5 to establish the estimate (3.20) as Δ​x→0→Δ𝑥0\Delta x\to 0.

The temporal derivative bound on the approximate solutions uΔ​xsubscript𝑢Δ𝑥u_{\Delta x} establishes that for every possible Δ​x>0Δ𝑥0\Delta x>0, uΔ​x∈Lip​([0,T];L2​(ℝ))subscript𝑢Δ𝑥Lip0𝑇superscript𝐿2ℝu_{\Delta x}\in\text{Lip}([0,T];L^{2}(\mathbb{R})). Employing the bound (3.17), we can apply the Arzelà-Ascoli theorem, indicating that the set of approximate solutions {uΔ​xj}j∈ℤsubscriptsubscript𝑢Δsubscript𝑥𝑗𝑗ℤ\{u_{\Delta x_{j}}\}_{j\in\mathbb{Z}} is sequentially compact in C​([0,T];L2​(ℝ))𝐶0𝑇superscript𝐿2ℝC([0,T];L^{2}(\mathbb{R})). Consequently, this implies the existence of a subsequence Δ​xjkΔsubscript𝑥subscript𝑗𝑘\Delta x_{j_{{}_{k}}} such that

uΔ​xjk→Δ​xj→0u​ uniformly in ​C​([0,T];L2​(ℝ)).absent→Δsubscript𝑥𝑗0→subscript𝑢Δsubscript𝑥subscript𝑗𝑘𝑢 uniformly in 𝐶0𝑇superscript𝐿2ℝu_{\Delta x_{j_{{}_{k}}}}\xrightarrow[]{\Delta x_{j}\xrightarrow[]{}0}u\text{ uniformly in }C([0,T];L^{2}(\mathbb{R})).

(3.21)

Now we claim that u𝑢u is a weak solution of the equation (1.1), that is, we need show that u𝑢u satisfies the following equation:

∫0T∫−∞∞(φt​u+φx​u22−(−Δ)α/2​φx​u)​𝑑x​𝑑t+∫−∞∞φ​(x,0)​u0​(x)​𝑑x=𝒪​(Δ​x),superscriptsubscript0𝑇superscriptsubscriptsubscript𝜑𝑡𝑢subscript𝜑𝑥superscript𝑢22superscriptΔ𝛼2subscript𝜑𝑥𝑢differential-d𝑥differential-d𝑡superscriptsubscript𝜑𝑥0subscript𝑢0𝑥differential-d𝑥𝒪Δ𝑥\int_{0}^{T}\int_{-\infty}^{\infty}\left(\varphi_{t}u+\varphi_{x}\frac{u^{2}}{2}-(-\Delta)^{\alpha/2}\varphi_{x}u\right)\,dx\,dt+\int_{-\infty}^{\infty}\varphi(x,0)u_{0}(x)\,dx=\mathcal{O}(\Delta x),

(3.22)

for all test functions φ∈Cc∞​(ℝ×[0,T])𝜑superscriptsubscript𝐶𝑐ℝ0𝑇\varphi\in C_{c}^{\infty}(\mathbb{R}\times[0,T]).

We employ a Lax-Wendroff type argument inspired by [17]. Let us define a piecewise constant interpolation for the approximate solution by the following:

u¯Δ​x​(x,t)=ujn​ for ​x∈[xj,xj+1),j∈ℤ​ and ​t∈[tn,tn+1),tn+1≤T.formulae-sequencesubscript¯𝑢Δ𝑥𝑥𝑡superscriptsubscript𝑢𝑗𝑛 for 𝑥subscript𝑥𝑗subscript𝑥𝑗1𝑗ℤ and 𝑡subscript𝑡𝑛subscript𝑡𝑛1subscript𝑡𝑛1𝑇\bar{u}_{\Delta x}(x,t)=u_{j}^{n}\text{ for }x\in[x_{j},x_{j+1}),~{}j\in\mathbb{Z}\text{ and }t\in[t_{n},t_{n+1}),~{}t_{n+1}\leq T.

(3.23)

Assuming Δ​xΔ𝑥\Delta x is sufficiently small, it is convenient to use a interpolation (3.23) instead of interpolation (3.16) to follow the proof of Lax-Wendroff type result. It is worth noting that u¯Δ​x​(⋅,tn)→u​(⋅,tn)→subscript¯𝑢Δ𝑥⋅subscript𝑡𝑛𝑢⋅subscript𝑡𝑛\bar{u}_{\Delta x}(\cdot,t_{n})\to u(\cdot,t_{n}) in L2​(ℝ)superscript𝐿2ℝL^{2}(\mathbb{R}) as Δ​x→0→Δ𝑥0\Delta x\to 0 for every tn≤Tsubscript𝑡𝑛𝑇t_{n}\leq T.
Let us take a test function φ∈Cc∞​(ℝ×[0,T])𝜑superscriptsubscript𝐶𝑐ℝ0𝑇\varphi\in C_{c}^{\infty}(\mathbb{R}\times[0,T]) and denote φ​(xj,tn)=φjn𝜑subscript𝑥𝑗subscript𝑡𝑛superscriptsubscript𝜑𝑗𝑛\varphi(x_{j},t_{n})=\varphi_{j}^{n} at nodes. We multiply (3.1) by Δ​x​Δ​t​φjnΔ𝑥Δ𝑡superscriptsubscript𝜑𝑗𝑛\Delta x\Delta t\varphi_{j}^{n} and taking the summation over all n𝑛n and j𝑗j to obtain

Δ​t​Δ​x​∑n∑jφjn​(ujn+1−u¯jnΔ​t)+limit-fromΔ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscriptsubscript𝑢𝑗𝑛1superscriptsubscript¯𝑢𝑗𝑛Δ𝑡\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\left(\frac{u_{j}^{n+1}-\bar{u}_{j}^{n}}{\Delta t}\right)+
Δ​t​Δ​x​∑n∑jφjn​D​(ujn)22Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛𝐷superscriptsubscriptsuperscript𝑢𝑛𝑗22\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\frac{D(u^{n}_{j})^{2}}{2}

+\displaystyle+
Δ​t​Δ​x​∑n∑jφjn​𝔻α​D​ujn+1/2=0,n​Δ​t≤T,j∈ℤ.formulae-sequenceΔ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛120formulae-sequence𝑛Δ𝑡𝑇𝑗ℤ\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{D}^{\alpha}Du_{j}^{n+1/2}=0,\quad n\Delta t\leq T,\quad j\in\mathbb{Z}.

Following the approach in [5] and [17], we can show that

Δ​t​Δ​x​∑n∑jφjn​(ujn+1−u¯jnΔ​t)Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscriptsubscript𝑢𝑗𝑛1superscriptsubscript¯𝑢𝑗𝑛Δ𝑡\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\left(\frac{u_{j}^{n+1}-\bar{u}_{j}^{n}}{\Delta t}\right)
=−Δ​t​Δ​x​∑n∑jD+t​φjn​ujnabsentΔ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝐷𝑡superscriptsubscript𝜑𝑗𝑛subscriptsuperscript𝑢𝑛𝑗\displaystyle=-\Delta t\Delta x\sum_{n}\sum_{j}D_{+}^{t}\varphi_{j}^{n}u^{n}_{j}

→Δ​x→0−∫0T∫ℝu​φt​𝑑x​𝑑t−∫ℝφ​(x,0)​u0​(x)​𝑑x,absent→Δ𝑥0→absentsuperscriptsubscript0𝑇subscriptℝ𝑢subscript𝜑𝑡differential-d𝑥differential-d𝑡subscriptℝ𝜑𝑥0subscript𝑢0𝑥differential-d𝑥\displaystyle\xrightarrow[]{\Delta x\xrightarrow[]{}0}-\int_{0}^{T}\int_{\mathbb{R}}u\varphi_{t}\,dx\,dt-\int_{\mathbb{R}}\varphi(x,0)u_{0}(x)\,dx,

and

Δ​t​Δ​x​∑n∑jφjn​D​(ujn)22→Δ​x→0−∫0T∫ℝu22​φx​𝑑x​𝑑t.absent→Δ𝑥0→Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛𝐷superscriptsubscriptsuperscript𝑢𝑛𝑗22superscriptsubscript0𝑇subscriptℝsuperscript𝑢22subscript𝜑𝑥differential-d𝑥differential-d𝑡\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\frac{D(u^{n}_{j})^{2}}{2}\xrightarrow[]{\Delta x\xrightarrow[]{}0}-\int_{0}^{T}\int_{\mathbb{R}}\frac{u^{2}}{2}\varphi_{x}\,dx\,dt.

Now we estimate the term involving the fractional Laplacian by using the properties of discrete fractional Laplacian described in Lemma 2.4.
Hereby we use the same notation for the inner product in L2superscript𝐿2L^{2} and ℓ2superscriptℓ2\ell^{2}.

Δ​t​Δ​x​∑n∑jφjn​𝔻α​D​ujn+1=Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛1absent\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{D}^{\alpha}Du_{j}^{n+1}=
−Δ​t​∑n⟨un+1,𝔻α​D​φn⟩.Δ𝑡subscript𝑛superscript𝑢𝑛1superscript𝔻𝛼𝐷superscript𝜑𝑛\displaystyle-\Delta t\sum_{n}\langle u^{n+1},\mathbb{D}^{\alpha}D\varphi^{n}\rangle.

Since φ​(⋅,tn)𝜑⋅subscript𝑡𝑛\varphi(\cdot,t_{n}) and φx​(⋅,tn)subscript𝜑𝑥⋅subscript𝑡𝑛\varphi_{x}(\cdot,t_{n}) are smooth functions, D​φn𝐷superscript𝜑𝑛D\varphi^{n} converges to φx​(⋅,tn)subscript𝜑𝑥⋅subscript𝑡𝑛\varphi_{x}(\cdot,t_{n}) uniformly as Δ​x→0→Δ𝑥0\Delta x\to 0. Then, we have

|⟨un+1,\displaystyle\Biggr{|}\langle u^{n+1},
𝔻αDφn⟩−⟨u(⋅,tn+1),−(−Δ)α/2φx(⋅,tn)⟩|\displaystyle\mathbb{D}^{\alpha}D\varphi^{n}\rangle-\langle u(\cdot,t_{n+1}),-(-\Delta)^{\alpha/2}\varphi_{x}(\cdot,t_{n})\rangle\Biggr{|}

≤\displaystyle\leq
|⟨un+1−u(⋅,tn+1),𝔻αDφn⟩|+|⟨u(⋅,tn+1),𝔻αDφn−(−(−Δ)α/2)φx(⋅,tn)⟩|\displaystyle\Biggr{|}\langle u^{n+1}-u(\cdot,t_{n+1}),\mathbb{D}^{\alpha}D\varphi^{n}\rangle\Biggr{|}+\Biggr{|}\langle u(\cdot,t_{n+1}),\mathbb{D}^{\alpha}D\varphi^{n}-(-(-\Delta)^{\alpha/2})\varphi_{x}(\cdot,t_{n})\rangle\Biggr{|}

≤\displaystyle\leq
‖un+1−u​(⋅,tn+1)‖​‖𝔻α​D​φn‖+‖u​(⋅,tn+1)‖​‖𝔻α​D​φn−(−(−Δ)α/2)​φx​(⋅,tn)‖.normsuperscript𝑢𝑛1𝑢⋅subscript𝑡𝑛1normsuperscript𝔻𝛼𝐷superscript𝜑𝑛norm𝑢⋅subscript𝑡𝑛1normsuperscript𝔻𝛼𝐷superscript𝜑𝑛superscriptΔ𝛼2subscript𝜑𝑥⋅subscript𝑡𝑛\displaystyle\left\|u^{n+1}-u(\cdot,t_{n+1})\right\|\left\|\mathbb{D}^{\alpha}D\varphi^{n}\right\|+\left\|u(\cdot,t_{n+1})\right\|\left\|\mathbb{D}^{\alpha}D\varphi^{n}-(-(-\Delta)^{\alpha/2})\varphi_{x}(\cdot,t_{n})\right\|.

The first term converges to zero using (3.21), and by applying the Lemma 2.5, it is evident that the second term also vanishes as Δ​x→0→Δ𝑥0\Delta x\to 0. Consequently, we have demonstrated that u𝑢u satisfies (3.22), which signifies that it is a weak solution.

Finally, the estimates (3.17)-(3.20) ensure that the weak solution u𝑢u satisfies the equation (1.1) as an L2superscript𝐿2L^{2}-identity. Hence considering the initial data u0∈H1+α​(ℝ)subscript𝑢0superscript𝐻1𝛼ℝu_{0}\in H^{1+\alpha}(\mathbb{R}), the limit u𝑢u becomes the unique solution of the fractional KdV equation (1.1). This completes the proof.
∎

## 4. Crank-Nicolson finite difference scheme

The essence of this study lies in the unveiling of a robust Crank-Nicolson temporal discretized finite difference scheme tailored for the precise numerical approximation of the solution to the fractional KdV equation (1.1), encapsulated by the following expression:

ujn+1=ujn−Δ​t​𝔾​(un+1/2)−Δ​t​𝔻α​D​ujn+1/2,n∈ℕ0,j∈ℤ,formulae-sequencesuperscriptsubscript𝑢𝑗𝑛1superscriptsubscript𝑢𝑗𝑛Δ𝑡𝔾superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛12formulae-sequence𝑛subscriptℕ0𝑗ℤu_{j}^{n+1}=u_{j}^{n}-\Delta t\mathbb{G}(u^{n+1/2})-\Delta t\mathbb{D}^{\alpha}Du_{j}^{n+1/2},\quad n\in\mathbb{N}_{0},\quad j\in\mathbb{Z},

(4.1)

where 𝔾​(un+1/2):=u~n+1/2​D​un+1/2assign𝔾superscript𝑢𝑛12superscript~𝑢𝑛12𝐷superscript𝑢𝑛12\mathbb{G}(u^{n+1/2}):=\tilde{u}^{n+1/2}Du^{n+1/2} and un+1/2:=12​(un+un+1)assignsuperscript𝑢𝑛1212superscript𝑢𝑛superscript𝑢𝑛1u^{n+1/2}:=\frac{1}{2}(u^{n}+u^{n+1}). For the discretiation of the initial data, we set uj0=u0​(xj)superscriptsubscript𝑢𝑗0subscript𝑢0subscript𝑥𝑗u_{j}^{0}=u_{0}(x_{j}) for j∈ℤ𝑗ℤj\in\mathbb{Z}. In order to ensure that the scheme is well-defined and guarantee the existence of a solution, we strategically adopt the proven methodology explained in [5], involving a fixed-point iteration approach.
For the solvability of scheme (4.1), we introduce the sequence {wℓ}ℓ≥0subscriptsuperscript𝑤ℓℓ0\{w^{\ell}\}_{\ell\geq 0} for the fixed-point iteration with wℓ+1superscript𝑤ℓ1w^{\ell+1} as the solution to the following equation:

{wℓ+1=un−Δ​t​𝔾​(un+wℓ2)−Δ​t​𝔻α​D​(un+wℓ+12),w0=un.casessuperscript𝑤ℓ1superscript𝑢𝑛Δ𝑡𝔾superscript𝑢𝑛superscript𝑤ℓ2Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛superscript𝑤ℓ12otherwisesuperscript𝑤0superscript𝑢𝑛otherwise\begin{cases}w^{\ell+1}=u^{n}-\Delta t\mathbb{G}\left(\frac{u^{n}+w^{\ell}}{2}\right)-\Delta t\mathbb{D}^{\alpha}D\left(\frac{u^{n}+w^{\ell+1}}{2}\right),\\
w^{0}=u^{n}.\end{cases}

(4.2)

To establish the existence and uniqueness of the sequence wℓsuperscript𝑤ℓw^{\ell}, we reformulate the iteration in a linear framework:

(1+Δ​t2​𝔻α​D)​wℓ+1=un−Δ​t​𝔾​(un+wℓ2)−Δ​t2​𝔻α​D​un.1Δ𝑡2superscript𝔻𝛼𝐷superscript𝑤ℓ1superscript𝑢𝑛Δ𝑡𝔾superscript𝑢𝑛superscript𝑤ℓ2Δ𝑡2superscript𝔻𝛼𝐷superscript𝑢𝑛\left(1+\frac{\Delta t}{2}\mathbb{D}^{\alpha}D\right)w^{\ell+1}=u^{n}-\Delta t\mathbb{G}\left(\frac{u^{n}+w^{\ell}}{2}\right)-\frac{\Delta t}{2}\mathbb{D}^{\alpha}Du^{n}.

(4.3)

A key insight, supported by the Lemma 2.4, affirms that the operator Δ​t2​𝔻α​DΔ𝑡2superscript𝔻𝛼𝐷\frac{\Delta t}{2}\mathbb{D}^{\alpha}D is skew-symmetric. This property renders the coefficient operator on the left-hand side of (4.3) to be positive definite, thereby ensuring the existence and uniqueness of the iterative sequence (4.2).

We describe the following lemma which establishes that the scheme (4.1) is solvable at each time step. In addition,
the lemma serves as a cornerstone for our subsequent stability analysis.

###### Lemma 4.1.

Let K=6−L1−L>6𝐾6𝐿1𝐿6K=\frac{6-L}{1-L}>6 be a constant and L∈(0,1)𝐿01L\in(0,1). Consider the fixed-point iteration defined by (4.2). Assume that the CFL condition for Δ​xΔ𝑥\Delta x and Δ​tΔ𝑡\Delta t:

λ≤LK​‖un‖h2,𝜆𝐿𝐾subscriptnormsuperscript𝑢𝑛superscriptℎ2\lambda\leq\frac{L}{K\|u^{n}\|_{h^{2}}},

(4.4)

where λ=Δ​tΔ​x𝜆Δ𝑡Δ𝑥\lambda=\frac{\Delta t}{\Delta x}. Then there exists a solution un+1superscript𝑢𝑛1u^{n+1} to equation (4.1) and limℓ→∞wℓ=un+1subscript→ℓsuperscript𝑤ℓsuperscript𝑢𝑛1\lim_{\ell\to\infty}w^{\ell}=u^{n+1}. Moreover, the following estimate holds:

‖un+1‖h2≤K​‖un‖h2.subscriptnormsuperscript𝑢𝑛1superscriptℎ2𝐾subscriptnormsuperscript𝑢𝑛superscriptℎ2\|u^{n+1}\|_{h^{2}}\leq K\|u^{n}\|_{h^{2}}.

(4.5)

###### Proof.

Set Δ​wℓ:=wℓ+1−wℓassignΔsuperscript𝑤ℓsuperscript𝑤ℓ1superscript𝑤ℓ\Delta w^{\ell}:=w^{\ell+1}-w^{\ell}, we have

(1+12​Δ​t​𝔻α​D)​Δ​wℓ=−Δ​t​[𝔾​(un+wℓ2)−𝔾​(un+wℓ−12)]:=−Δ​t​Δ​𝔾.112Δ𝑡superscript𝔻𝛼𝐷Δsuperscript𝑤ℓΔ𝑡delimited-[]𝔾superscript𝑢𝑛superscript𝑤ℓ2𝔾superscript𝑢𝑛subscript𝑤ℓ12assignΔ𝑡Δ𝔾\left(1+\frac{1}{2}\Delta t\mathbb{D}^{\alpha}D\right)\Delta w^{\ell}=-\Delta t\left[\mathbb{G}\left(\frac{u^{n}+w^{\ell}}{2}\right)-\mathbb{G}\left(\frac{u^{n}+w_{\ell-1}}{2}\right)\right]:=-\Delta t\Delta\mathbb{G}.

(4.6)

Applying the discrete operator D+​D−subscript𝐷subscript𝐷D_{+}D_{-} to (4.6), then multiply with Δ​x​D+​D−​Δ​wℓΔ𝑥subscript𝐷subscript𝐷Δsuperscript𝑤ℓ\Delta xD_{+}D_{-}\Delta w^{\ell} and summing over j∈ℤ𝑗ℤj\in\mathbb{Z}, we have

‖D+​D−​Δ​wℓ‖2=−Δ​t​⟨D+​D−​Δ​𝔾,D+​D−​Δ​wℓ⟩≤Δ​t​‖D+​D−​Δ​𝔾‖​‖D+​D−​Δ​wℓ‖.superscriptdelimited-∥∥subscript𝐷subscript𝐷Δsuperscript𝑤ℓ2Δ𝑡subscript𝐷subscript𝐷Δ𝔾subscript𝐷subscript𝐷Δsuperscript𝑤ℓΔ𝑡delimited-∥∥subscript𝐷subscript𝐷Δ𝔾delimited-∥∥subscript𝐷subscript𝐷Δsuperscript𝑤ℓ\begin{split}\|D_{+}D_{-}\Delta w^{\ell}\|^{2}&=-\Delta t\left\langle D_{+}D_{-}\Delta\mathbb{G},D_{+}D_{-}\Delta w^{\ell}\right\rangle\\
&\leq\Delta t\|D_{+}D_{-}\Delta\mathbb{G}\|\|D_{+}D_{-}\Delta w^{\ell}\|.\end{split}

where we have used the Lemma 2.4 for the fractional term.
Following closely the steps from Dutta et al. [5, Lemma 2.5], we have that the sequence {wℓ}superscript𝑤ℓ\{w^{\ell}\} is Cauchy, hence converges. In addition, we have the estimate (4.5).
∎

###### Remark 4.2.

We have shown that the devised scheme is solvable for each time step assuming the CFL condition (4.4), where λ𝜆\lambda is bounded by the h2superscriptℎ2h^{2}-bound of approximate solution unsuperscript𝑢𝑛u^{n}. Since the CFL condition (4.4) depends on unsuperscript𝑢𝑛u^{n}, not on the initial condition directly. Consequently, in order to provide a comprehensive assessment of the stability of our computed solution unsuperscript𝑢𝑛u^{n}, we must embark on a thorough stability analysis that delves into the intricacies of the evolving solution over time.

Now we prove a fundamental stability lemma.

### 4.1. Stability Lemma

###### Lemma 4.3.

Let u0∈H1+α​(ℝ)subscript𝑢0superscript𝐻1𝛼ℝu_{0}\in H^{1+\alpha}(\mathbb{R}). Assume that Δ​tΔ𝑡\Delta t satisfies

λ≤LK​Y𝜆𝐿𝐾𝑌\lambda\leq\frac{L}{KY}

(4.7)

for some Y=Y​(‖u0‖H2​(ℝ),‖u0‖H1+α​(ℝ))𝑌𝑌subscriptnormsubscript𝑢0superscript𝐻2ℝsubscriptnormsubscript𝑢0superscript𝐻1𝛼ℝY=Y\left(\|u_{0}\|_{H^{2}(\mathbb{R})},\|u_{0}\|_{H^{1+\alpha}(\mathbb{R})}\right) and λ=Δ​t/Δ​x𝜆Δ𝑡Δ𝑥\lambda=\Delta t/\Delta x. Then there is a finite time T>0𝑇0T>0 and a constant C𝐶C both depending on ‖u0‖h2subscriptnormsubscript𝑢0superscriptℎ2\|u_{0}\|_{h^{2}} and ‖u0‖h1+αsubscriptnormsubscript𝑢0superscriptℎ1𝛼\|u_{0}\|_{h^{1+\alpha}} such that

‖un‖h2subscriptnormsuperscript𝑢𝑛superscriptℎ2\displaystyle\|u^{n}\|_{h^{2}}
≤C, for ​tn≤T,formulae-sequenceabsent𝐶 for subscript𝑡𝑛𝑇\displaystyle\leq C,\qquad\text{ for }t_{n}\leq T,

(4.8)

‖D+t​un‖normsuperscriptsubscript𝐷𝑡superscript𝑢𝑛\displaystyle\|D_{+}^{t}u^{n}\|
≤C, for ​tn≤T,formulae-sequenceabsent𝐶 for subscript𝑡𝑛𝑇\displaystyle\leq C,\qquad\text{ for }t_{n}\leq T,

(4.9)

‖un‖h1+αsubscriptnormsuperscript𝑢𝑛superscriptℎ1𝛼\displaystyle\|u^{n}\|_{h^{1+\alpha}}
≤C, for ​tn≤T.formulae-sequenceabsent𝐶 for subscript𝑡𝑛𝑇\displaystyle\leq C,\qquad\text{ for }t_{n}\leq T.

(4.10)

(4.10) is a stability estimate.

###### Proof.

Motivated by the analysis established in [5, Lemma 2.7], we perform the difference operator D+​D−subscript𝐷subscript𝐷D_{+}D_{-} on the scheme (4.1) and taking the inner product with D+​D−​un+1/2subscript𝐷subscript𝐷superscript𝑢𝑛12D_{+}D_{-}u^{n+1/2} amounts to

‖D+​D−​un+1‖2=‖D+​D−​un‖2−2​Δ​t​⟨D+​D−​𝔾​(un+1/2),D+​D−​un+1/2⟩,superscriptnormsubscript𝐷subscript𝐷superscript𝑢𝑛12superscriptnormsubscript𝐷subscript𝐷superscript𝑢𝑛22Δ𝑡subscript𝐷subscript𝐷𝔾superscript𝑢𝑛12subscript𝐷subscript𝐷superscript𝑢𝑛12\|D_{+}D_{-}u^{n+1}\|^{2}=\|D_{+}D_{-}u^{n}\|^{2}-2\Delta t\langle D_{+}D_{-}\mathbb{G}(u^{n+1/2}),D_{+}D_{-}u^{n+1/2}\rangle,

(4.11)

where we have used Lemma 2.4 for the fractional term. We estimate the nonlinear part on the right hand side by following steps from [5, Lemma 2.7], which yields

|⟨D+​D−​𝔾​(un+1/2),D+​D−​un+1/2⟩|≤32​‖D+​D−​un+1/2‖​‖un+1/2‖h22.subscript𝐷subscript𝐷𝔾superscript𝑢𝑛12subscript𝐷subscript𝐷superscript𝑢𝑛1232normsubscript𝐷subscript𝐷superscript𝑢𝑛12subscriptsuperscriptnormsuperscript𝑢𝑛122superscriptℎ2|\langle D_{+}D_{-}\mathbb{G}(u^{n+1/2}),D_{+}D_{-}u^{n+1/2}\rangle|\leq\sqrt{\frac{3}{2}}\left\|D_{+}D_{-}u^{n+1/2}\right\|\left\|u^{n+1/2}\right\|^{2}_{h^{2}}.

We can get similar estimate for the lower order derivative term. This further gives the following bound

‖un+1‖h2≤‖un‖h2+8​Δ​t​‖un+1/2‖h22.subscriptnormsuperscript𝑢𝑛1superscriptℎ2subscriptnormsuperscript𝑢𝑛superscriptℎ28Δ𝑡subscriptsuperscriptnormsuperscript𝑢𝑛122superscriptℎ2\left\|u^{n+1}\right\|_{h^{2}}\leq\left\|u^{n}\right\|_{h^{2}}+8\Delta t\left\|u^{n+1/2}\right\|^{2}_{h^{2}}.

(4.12)

Let us introduce a differential equation

y′(t)=2(K+1)2y(t)2,t>0;y(0)=max{∥u0∥H2,∥u0∥H1+α}=:y0.y^{\prime}(t)=2(K+1)^{2}y(t)^{2},\quad t>0;\qquad y(0)=\max\{\|u_{0}\|_{H^{2}},\|u_{0}\|_{H^{1+\alpha}}\}=:y_{0}.

It is observed that the solution y​(t)𝑦𝑡y(t) of the above differential equation is convex and increasing for all t<T:=T∞/2𝑡𝑇assignsubscript𝑇2t<T:=T_{\infty}/2, where T∞=1/(2​(K+1)2​y0)subscript𝑇12superscript𝐾12subscript𝑦0{T}_{\infty}=1/(2(K+1)^{2}y_{0}).

Next we claim that

‖un‖h2≤y​(tn)≤Yfor ​tn≤T.formulae-sequencesubscriptnormsuperscript𝑢𝑛superscriptℎ2𝑦subscript𝑡𝑛𝑌for subscript𝑡𝑛𝑇\displaystyle\|u^{n}\|_{h^{2}}\leq y(t_{n})\leq Y\quad\text{for }t_{n}\leq T.

(4.13)

We proceed by mathematical induction. The claim is obvious for n=0𝑛0n=0 by the Lemma 2.3. Now we assume that the estimate holds for n=1,…,m.𝑛1…𝑚n=1,\dots,m. As ∥um∥h2≤y(tm)≤y(T)=:Y(y0)\|u^{m}\|_{h^{2}}\leq y(t_{m})\leq y(T)=:Y(y_{0}), then the CFL condition (4.7) implies (4.4). Thus by the Lemma 4.1, we have

‖um+1/2‖h2≤(K+1)2​‖um‖h2.subscriptnormsuperscript𝑢𝑚12superscriptℎ2𝐾12subscriptnormsuperscript𝑢𝑚superscriptℎ2\|u^{m+1/2}\|_{h^{2}}\leq\frac{(K+1)}{2}\|u^{m}\|_{h^{2}}.

(4.14)

Since y𝑦y is increasing and convex, then (4.12) and (4.14) yield the following estimate

‖um+1‖h2≤‖um‖h2+2​Δ​t​((K+1)​‖um‖h2)2≤y​(tm)+2​Δ​t​((K+1)​y​(tm))2≤y​(tm)+∫tmtm+12​(K+1)2​y​(tm)2​𝑑t≤y​(tm)+∫tmtm+1y′​(s)​𝑑s=y​(tm+1).subscriptdelimited-∥∥superscript𝑢𝑚1superscriptℎ2subscriptdelimited-∥∥superscript𝑢𝑚superscriptℎ22Δ𝑡superscript𝐾1subscriptdelimited-∥∥superscript𝑢𝑚superscriptℎ22𝑦subscript𝑡𝑚2Δ𝑡superscript𝐾1𝑦subscript𝑡𝑚2𝑦subscript𝑡𝑚superscriptsubscriptsubscript𝑡𝑚subscript𝑡𝑚12superscript𝐾12𝑦superscriptsubscript𝑡𝑚2differential-d𝑡𝑦subscript𝑡𝑚superscriptsubscriptsubscript𝑡𝑚subscript𝑡𝑚1superscript𝑦′𝑠differential-d𝑠𝑦subscript𝑡𝑚1\begin{split}\|u^{m+1}\|_{h^{2}}\leq\|u^{m}\|_{h^{2}}+2\Delta t((K+1)\|u^{m}\|_{h^{2}})^{2}\leq&y(t_{m})+2\Delta t((K+1)y(t_{m}))^{2}\\
\leq&y(t_{m})+\int_{t_{m}}^{t_{m+1}}2(K+1)^{2}y(t_{m})^{2}\,dt\\
\leq&y(t_{m})+\int_{t_{m}}^{t_{m+1}}y^{\prime}(s)\,ds=y(t_{m+1}).\end{split}

This proves that ‖un‖h2≤y​(T)=Ysubscriptnormsuperscript𝑢𝑛superscriptℎ2𝑦𝑇𝑌\|u^{n}\|_{h^{2}}\leq y(T)=Y, (n+1)​Δ​t<T𝑛1Δ𝑡𝑇(n+1)\Delta t<T. Hence the estimate (4.13) holds.

From the scheme (4.1), we have

ujn+1=ujn−Δ​t​𝔾​(un+1/2)−Δ​t​𝔻α​D​ujn+1/2superscriptsubscript𝑢𝑗𝑛1superscriptsubscript𝑢𝑗𝑛Δ𝑡𝔾superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛12u_{j}^{n+1}=u_{j}^{n}-\Delta t\mathbb{G}(u^{n+1/2})-\Delta t\mathbb{D}^{\alpha}Du_{j}^{n+1/2}

and

ujn=ujn−1−Δ​t​𝔾​(un−1/2)−Δ​t​𝔻α​D​ujn−1/2,superscriptsubscript𝑢𝑗𝑛superscriptsubscript𝑢𝑗𝑛1Δ𝑡𝔾superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛12u_{j}^{n}=u_{j}^{n-1}-\Delta t\mathbb{G}(u^{n-1/2})-\Delta t\mathbb{D}^{\alpha}Du_{j}^{n-1/2},

which implies

D+t​ujn=D+t​ujn−1−(𝔾​(un+1/2)−𝔾​(un−1/2))−Δ​t​𝔻α​D​(D+t​ujn−1/2).superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛1𝔾superscript𝑢𝑛12𝔾superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛12D_{+}^{t}u_{j}^{n}=D_{+}^{t}u_{j}^{n-1}-\left(\mathbb{G}(u^{n+1/2})-\mathbb{G}(u^{n-1/2})\right)-\Delta t\mathbb{D}^{\alpha}D(D_{+}^{t}u_{j}^{n-1/2}).

Taking inner product with D+t​ujn−1/2=12​(D+t​ujn+D+t​ujn−1)superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛1212superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛superscriptsubscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛1D_{+}^{t}u_{j}^{n-1/2}=\frac{1}{2}(D_{+}^{t}u_{j}^{n}+D_{+}^{t}u_{j}^{n-1}), we obtain

12​‖D+t​un‖2−12​‖D+t​un−1‖2=−⟨(𝔾​(un+1/2)−𝔾​(un−1/2)),D+t​un−1/2⟩=−⟨(u~n+1/2​D​un+1/2−u~n−1/2​D​un−1/2),D+t​un−1/2⟩=−Δ​t​⟨(D+t​u~n−1/2​D​un+1/2+u~n−1/2​D​D+t​un−1/2),D+t​un−1/2⟩≤Δ​t​‖D​un+1/2‖∞​‖D+t​un−1/2‖2+Δ​t​⟨D​(u~n−1/2​D+t​un−1/2),D+t​un−1/2⟩≤Δt∥Dun+1/2∥h2∥D+tun−1/2∥2+Δt[Δ​x2⟨D+u~n−1/2DD+tun−1/2,D+tun−1/2⟩+12⟨S−D+tun−1/2Du~n−1/2,D+tun−1/2⟩]≤C​Δ​t​(‖D+t​un‖2+‖D+t​un−1‖2),\begin{split}\frac{1}{2}\|D_{+}^{t}u^{n}\|^{2}-&\frac{1}{2}\|D_{+}^{t}u^{n-1}\|^{2}=-\left\langle\left(\mathbb{G}(u^{n+1/2})-\mathbb{G}(u^{n-1/2})\right),D_{+}^{t}u^{n-1/2}\right\rangle\\
=&-\left\langle\left(\tilde{u}^{n+1/2}Du^{n+1/2}-\tilde{u}^{n-1/2}Du^{n-1/2}\right),D_{+}^{t}u^{n-1/2}\right\rangle\\
=&-\Delta t\left\langle\left(D_{+}^{t}\tilde{u}^{n-1/2}Du^{n+1/2}+\tilde{u}^{n-1/2}DD_{+}^{t}u^{n-1/2}\right),D_{+}^{t}u^{n-1/2}\right\rangle\\
\leq&\Delta t\left\|Du^{n+1/2}\right\|_{\infty}\left\|D_{+}^{t}u^{n-1/2}\right\|^{2}+\Delta t\left\langle D(\tilde{u}^{n-1/2}D_{+}^{t}u^{n-1/2}),D_{+}^{t}u^{n-1/2}\right\rangle\\
\leq&\Delta t\left\|Du^{n+1/2}\right\|_{h^{2}}\left\|D_{+}^{t}u^{n-1/2}\right\|^{2}+\Delta t\Biggr{[}\frac{\Delta x}{2}\left\langle D_{+}\tilde{u}^{n-1/2}DD_{+}^{t}u^{n-1/2},D_{+}^{t}u^{n-1/2}\right\rangle\\
&\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad+\frac{1}{2}\left\langle S^{-}D_{+}^{t}u^{n-1/2}D\tilde{u}^{n-1/2},D_{+}^{t}u^{n-1/2}\right\rangle\Biggr{]}\\
\leq&C\Delta t\left(\left\|D_{+}^{t}u^{n}\right\|^{2}+\left\|D_{+}^{t}u^{n-1}\right\|^{2}\right),\end{split}

where we have used (2.7) and the discrete Sobolev inequality ‖D​un‖∞≤‖un‖h2≤Csubscriptnorm𝐷superscript𝑢𝑛subscriptnormsuperscript𝑢𝑛superscriptℎ2𝐶\|Du^{n}\|_{\infty}\leq\|u^{n}\|_{h^{2}}\leq C. Assuming Δ​tΔ𝑡\Delta t is small enough such that 1−C​Δ​t≥121𝐶Δ𝑡121-C\Delta t\geq\frac{1}{2}, then we have

‖D+t​un‖2≤‖D+t​un−1‖2+2​Δ​t​‖D+t​un−1‖2.superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛2superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛122Δ𝑡superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛12\displaystyle\|D_{+}^{t}u^{n}\|^{2}\leq\|D_{+}^{t}u^{n-1}\|^{2}+2\Delta t\left\|D_{+}^{t}u^{n-1}\right\|^{2}.

By setting Γn=‖D+t​un−1‖2subscriptΓ𝑛superscriptnormsuperscriptsubscript𝐷𝑡superscript𝑢𝑛12\Gamma_{n}=\|D_{+}^{t}u^{n-1}\|^{2} for every n​Δ​t≤T𝑛Δ𝑡𝑇n\Delta t\leq T, we see that

Γn+1≤Γn+2​C​Δ​t​Γn.subscriptΓ𝑛1subscriptΓ𝑛2𝐶Δ𝑡subscriptΓ𝑛\Gamma_{n+1}\leq\Gamma_{n}+2C\Delta t\Gamma_{n}.

Let A​(t)𝐴𝑡A(t) solves the differential equation

A′​(t)=2​C​A​(t),A​(t1)=Γ1.formulae-sequencesuperscript𝐴′𝑡2𝐶𝐴𝑡𝐴subscript𝑡1subscriptΓ1A^{\prime}(t)=2CA(t),\quad A(t_{1})=\Gamma_{1}.

Note that Γ1subscriptΓ1\Gamma_{1} is finite as ‖u1‖normsuperscript𝑢1\|u^{1}\| and ‖u0‖normsuperscript𝑢0\|u^{0}\| are bounded by (4.8) and there exist a sufficiently large T¯¯𝑇\bar{T} such that the solution A​(t)𝐴𝑡A(t) is bounded for every t<T<T¯𝑡𝑇¯𝑇t<T<\bar{T}. Clearly, A is increasing and convex, then A​(tn)≤A​(T)𝐴subscript𝑡𝑛𝐴𝑇A(t_{n})\leq A(T) for every tn≤Tsubscript𝑡𝑛𝑇t_{n}\leq T. We claim that Γn≤A​(tn)subscriptΓ𝑛𝐴subscript𝑡𝑛\Gamma_{n}\leq A(t_{n}) for tn≤Tsubscript𝑡𝑛𝑇t_{n}\leq T. We use the mathematical induction, as this holds for n=1𝑛1n=1 by construction. Let Γn≤A​(tn)subscriptΓ𝑛𝐴subscript𝑡𝑛\Gamma_{n}\leq A(t_{n}) for n=2,…,N𝑛2…𝑁n=2,\dots,N, then

ΓN+1≤ΓN+2​C​Δ​t​ΓN≤A​(tN)+2​C​Δ​t​A​(tN)=A​(tN)+∫tNtN+12​C​A​(tN)​𝑑t≤A​(tN)+∫tNtN+1A′​(s)​𝑑s=A​(tN+1),subscriptΓ𝑁1subscriptΓ𝑁2𝐶Δ𝑡subscriptΓ𝑁𝐴subscript𝑡𝑁2𝐶Δ𝑡𝐴subscript𝑡𝑁𝐴subscript𝑡𝑁superscriptsubscriptsubscript𝑡𝑁subscript𝑡𝑁12𝐶𝐴subscript𝑡𝑁differential-d𝑡𝐴subscript𝑡𝑁superscriptsubscriptsubscript𝑡𝑁subscript𝑡𝑁1superscript𝐴′𝑠differential-d𝑠𝐴subscript𝑡𝑁1\begin{split}\Gamma_{N+1}\leq\Gamma_{N}+2C\Delta t\Gamma_{N}\leq&A(t_{N})+2C\Delta tA(t_{N})\\
=&A(t_{N})+\int_{t_{N}}^{t_{N+1}}2CA(t_{N})\,dt\\
\leq&A(t_{N})+\int_{t_{N}}^{t_{N+1}}A^{\prime}(s)\,ds=A(t_{N+1}),\end{split}

where we have used that A𝐴A is increasing. Hence for tn≤Tsubscript𝑡𝑛𝑇t_{n}\leq T, Γn+1=∥D+tun∥2≤A(T)=:C\Gamma_{n+1}=\|D_{+}^{t}u^{n}\|^{2}\leq A(T)=:C, where C𝐶C is a constant independent of Δ​xΔ𝑥\Delta x. Finally we end up with

‖𝔻α​D​un+1/2‖≤‖D+t​un‖+‖un‖∞​‖D​un‖≤C,tn≤T.formulae-sequencenormsuperscript𝔻𝛼𝐷superscript𝑢𝑛12normsuperscriptsubscript𝐷𝑡superscript𝑢𝑛subscriptnormsuperscript𝑢𝑛norm𝐷superscript𝑢𝑛𝐶subscript𝑡𝑛𝑇\|\mathbb{D}^{\alpha}Du^{n+1/2}\|\leq\|D_{+}^{t}u^{n}\|+\|u^{n}\|_{\infty}\|Du^{n}\|\leq C,\qquad t_{n}\leq T.

This implies ‖un‖h1+α≤Csubscriptnormsuperscript𝑢𝑛superscriptℎ1𝛼𝐶\|u^{n}\|_{h^{1+\alpha}}\leq C for tn≤T.subscript𝑡𝑛𝑇t_{n}\leq T.
Hence the result follows.
∎

### 4.2. Convergence of approximate solution

We adopt the approach outlined by Sjoberg [33] to establish the convergence of the scheme (4.1) for t<T𝑡𝑇t<T. Our reasoning unfolds through the construction of a piecewise continuous interpolation, denoted as uΔ​xsubscript𝑢Δ𝑥u_{\Delta x}, defined by (3.16).
With this, we are poised to present the main result of this section.

###### Theorem 4.4.

Assume that ‖u0‖h1+α,α∈[1,2)subscriptnormsubscript𝑢0superscriptℎ1𝛼𝛼12\|u_{0}\|_{h^{1+\alpha}},~{}\alpha\in[1,2), is finite. Let {uΔ​x}Δ​x>0subscriptsubscript𝑢Δ𝑥Δ𝑥0\{u_{\Delta x}\}_{\Delta x>0} be a sequence of approximate solutions obtained by the scheme (4.1) of the fractional KdV equation (1.1). Then there exists a finite time T>0𝑇0T>0 and a constant C𝐶C, depending on ‖u0‖h1+αsubscriptnormsubscript𝑢0superscriptℎ1𝛼\|u_{0}\|_{h^{1+\alpha}} such that

‖uΔ​x​(⋅,t)‖L2​(ℝ)subscriptnormsubscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ\displaystyle\|u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}
≤‖u0‖L2​(ℝ),absentsubscriptnormsubscript𝑢0superscript𝐿2ℝ\displaystyle\leq\|u_{0}\|_{L^{2}(\mathbb{R})},

(4.15)

‖∂xuΔ​x​(⋅,t)‖L2​(ℝ)subscriptnormsubscript𝑥subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ\displaystyle\|\partial_{x}u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}
≤C,absent𝐶\displaystyle\leq C,

(4.16)

‖∂tuΔ​x​(⋅,t)‖L2​(ℝ)subscriptnormsubscript𝑡subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ\displaystyle\|\partial_{t}u_{\Delta x}(\cdot,t)\|_{L^{2}(\mathbb{R})}
≤C,absent𝐶\displaystyle\leq C,

(4.17)

‖(−Δ)α/2​∂xuΔ​x​(⋅,t)‖L2​(ℝ)subscriptnormsuperscriptΔ𝛼2subscript𝑥subscript𝑢Δ𝑥⋅𝑡superscript𝐿2ℝ\displaystyle\left\|(-\Delta)^{\alpha/2}\partial_{x}u_{\Delta x}(\cdot,t)\right\|_{L^{2}(\mathbb{R})}
≤C,absent𝐶\displaystyle\leq C,

(4.18)

for Δ​t=𝒪​(Δ​x)Δ𝑡𝒪Δ𝑥\Delta t=\mathcal{O}(\Delta x). Moreover, the sequence of approximate solutions {uΔ​x}Δ​x>0subscriptsubscript𝑢Δ𝑥Δ𝑥0\{u_{\Delta x}\}_{\Delta x>0} converge uniformly to the unique solution of the fractional KdV equation (1.1) in C​(ℝ×[0,T])𝐶ℝ0𝑇C(\mathbb{R}\times[0,T]) as Δ​x→0→Δ𝑥0\Delta x\to 0.

###### Proof.

Since the interpolation function uΔ​xsubscript𝑢Δ𝑥u_{\Delta x}, defined by (3.16), is smooth, we can explicitly express its spatial and temporal derivatives:

∂xuΔ​x​(x,t)=subscript𝑥subscript𝑢Δ𝑥𝑥𝑡absent\displaystyle\partial_{x}u_{\Delta x}(x,t)=
D​ujn+(t−tn)​D+t​(D​ujn),𝐷superscriptsubscript𝑢𝑗𝑛𝑡subscript𝑡𝑛superscriptsubscript𝐷𝑡𝐷superscriptsubscript𝑢𝑗𝑛\displaystyle Du_{j}^{n}+(t-t_{n})D_{+}^{t}\left(Du_{j}^{n}\right),

(4.19)

∂tuΔ​x​(x,t)=subscript𝑡subscript𝑢Δ𝑥𝑥𝑡absent\displaystyle\partial_{t}u_{\Delta x}(x,t)=
D+t​un​(x).superscriptsubscript𝐷𝑡superscript𝑢𝑛𝑥\displaystyle D_{+}^{t}u^{n}(x).

(4.20)

The above expressions imply the estimates (4.15), (4.16) and (4.17) for t≤T𝑡𝑇t\leq T by integrating over [xj,xj+1)subscript𝑥𝑗subscript𝑥𝑗1[x_{j},x_{j+1}) the square of (3.16), (4.19) and (4.20), respectively, followed by the summation over j𝑗j. Since for t<T𝑡𝑇t<T, ∂xuΔ​x​(⋅,t)subscript𝑥subscript𝑢Δ𝑥⋅𝑡\partial_{x}u_{\Delta x}(\cdot,t) is constant in the interval [xj,xj+1)subscript𝑥𝑗subscript𝑥𝑗1[x_{j},x_{j+1}) for all j∈ℤ𝑗ℤj\in\mathbb{Z}, we can apply the Lemma 2.3 and the estimate (4.10) from Lemma 4.3 to establish the validity of (4.18) as Δ​x→0absent→Δ𝑥0\Delta x\xrightarrow[]{}0.

The temporal derivative bound on the approximate solutions uΔ​xsubscript𝑢Δ𝑥u_{\Delta x} establishes that uΔ​x∈Lip​([0,T];L2​(ℝ))subscript𝑢Δ𝑥Lip0𝑇superscript𝐿2ℝu_{\Delta x}\in\text{Lip}([0,T];L^{2}(\mathbb{R})) for every possible Δ​x>0Δ𝑥0\Delta x>0. Employing the bound (4.15), we can apply the Arzelà-Ascoli theorem, which implies that the set of approximate solutions {uΔ​xj}j∈ℤsubscriptsubscript𝑢Δsubscript𝑥𝑗𝑗ℤ\{u_{\Delta x_{j}}\}_{j\in\mathbb{Z}} is sequentially compact in C​([0,T];L2​(ℝ))𝐶0𝑇superscript𝐿2ℝC([0,T];L^{2}(\mathbb{R})). Consequently, this implies the existence of a subsequence Δ​xjkΔsubscript𝑥subscript𝑗𝑘\Delta x_{j_{{}_{k}}} such that

uΔ​xjk→Δ​xj→0u​ uniformly in ​C​([0,T];L2​(ℝ)).absent→Δsubscript𝑥𝑗0→subscript𝑢Δsubscript𝑥subscript𝑗𝑘𝑢 uniformly in 𝐶0𝑇superscript𝐿2ℝu_{\Delta x_{j_{{}_{k}}}}\xrightarrow[]{\Delta x_{j}\xrightarrow[]{}0}u\text{ uniformly in }C([0,T];L^{2}(\mathbb{R})).

(4.21)

Our next aim is to demonstrate that the limit u𝑢u satisfies (3.22). For this, we proceed through a Lax-Wendroff type argument inspired by [17]. It is convenient to use simple interpolation function given by (3.23) since u¯Δ​x→Δ​x→0uabsent→Δ𝑥0→subscript¯𝑢Δ𝑥𝑢\bar{u}_{\Delta x}\xrightarrow[]{\Delta x\xrightarrow[]{}0}u in L∞​([0,T];L2​(ℝ))superscript𝐿0𝑇superscript𝐿2ℝL^{\infty}([0,T];L^{2}(\mathbb{R})).

We proceed similarly as in the previous section by considering a test function φ∈Cc∞​(ℝ×[0,T])𝜑superscriptsubscript𝐶𝑐ℝ0𝑇\varphi\in C_{c}^{\infty}(\mathbb{R}\times[0,T]) and denote φ​(xj,tn)=φjn𝜑subscript𝑥𝑗subscript𝑡𝑛superscriptsubscript𝜑𝑗𝑛\varphi(x_{j},t_{n})=\varphi_{j}^{n} at nodes. We multiply Δ​t​Δ​x​φjnΔ𝑡Δ𝑥superscriptsubscript𝜑𝑗𝑛\Delta t\Delta x\varphi_{j}^{n} to the scheme (4.1) and summing it over all j𝑗j and n𝑛n to obtain

Δ​t​Δ​x​∑n∑jφjn​D+t​ujn+limit-fromΔ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛subscriptsuperscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}D^{t}_{+}u_{j}^{n}+
Δ​t​Δ​x​∑n∑jφjn​𝔾​(un+1/2)Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛𝔾superscript𝑢𝑛12\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{G}(u^{n+1/2})

+\displaystyle+
Δ​t​Δ​x​∑n∑jφjn​𝔻α​D​ujn+1/2=0,n​Δ​t≤T,j∈ℤ.formulae-sequenceΔ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛120formulae-sequence𝑛Δ𝑡𝑇𝑗ℤ\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{D}^{\alpha}Du_{j}^{n+1/2}=0,\quad n\Delta t\leq T,\quad j\in\mathbb{Z}.

(4.22)

With the help of [5, Theorem 2.8], we have

Δ​t​Δ​x​∑n∑jφjn​D+t​ujn→−∫0T∫ℝu​φt​𝑑x​𝑑t−∫ℝφ​(x,0)​u0​(x)​𝑑x​ as ​Δ​x→0,absent→Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛subscriptsuperscript𝐷𝑡superscriptsubscript𝑢𝑗𝑛superscriptsubscript0𝑇subscriptℝ𝑢subscript𝜑𝑡differential-d𝑥differential-d𝑡subscriptℝ𝜑𝑥0subscript𝑢0𝑥differential-d𝑥 as Δ𝑥absent→0\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}D^{t}_{+}u_{j}^{n}\xrightarrow[]{}-\int_{0}^{T}\int_{\mathbb{R}}u\varphi_{t}\,dx\,dt-\int_{\mathbb{R}}\varphi(x,0)u_{0}(x)\,dx\text{ as }\Delta x\xrightarrow[]{}0,

and

Δ​t​Δ​x​∑n∑jφjn​𝔾​(un+1/2)→−∫0T∫ℝu22​φx​𝑑x​𝑑t​ as ​Δ​x→0.absent→Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛𝔾superscript𝑢𝑛12superscriptsubscript0𝑇subscriptℝsuperscript𝑢22subscript𝜑𝑥differential-d𝑥differential-d𝑡 as Δ𝑥absent→0\displaystyle\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{G}(u^{n+1/2})\xrightarrow[]{}-\int_{0}^{T}\int_{\mathbb{R}}\frac{u^{2}}{2}\varphi_{x}\,dx\,dt\text{ as }\Delta x\xrightarrow[]{}0.

For the term involving the discrete fractional Laplacian, we use summation-by-parts to conclude that

Δ​t​Δ​x​∑n∑jφjn​𝔻α​D​ujn+1/2→−∫0T∫ℝ−(−Δ)α/2​φx​u​d​x​d​t​ as ​Δ​x→0.absent→Δ𝑡Δ𝑥subscript𝑛subscript𝑗superscriptsubscript𝜑𝑗𝑛superscript𝔻𝛼𝐷superscriptsubscript𝑢𝑗𝑛12superscriptsubscript0𝑇subscriptℝsuperscriptΔ𝛼2subscript𝜑𝑥𝑢𝑑𝑥𝑑𝑡 as Δ𝑥absent→0\Delta t\Delta x\sum_{n}\sum_{j}\varphi_{j}^{n}\mathbb{D}^{\alpha}Du_{j}^{n+1/2}\xrightarrow[]{}-\int_{0}^{T}\int_{\mathbb{R}}-(-\Delta)^{\alpha/2}\varphi_{x}u\,dx\,dt\text{ as }\Delta x\xrightarrow[]{}0.

Consequently, we have demonstrated that u𝑢u satisfies (3.22), signifying that u𝑢u is a weak solution of the fractional KdV equation (1.1).
Hence we can conclude from the estimates (4.15)-(4.18) that the weak solution u𝑢u becomes a strong solution, which satisfies equation (1.1) as an L2superscript𝐿2L^{2}-identity. Thus, considering the initial data u0∈H1+α​(ℝ)subscript𝑢0superscript𝐻1𝛼ℝu_{0}\in H^{1+\alpha}(\mathbb{R}), u𝑢u becomes the unique solution of the fractional KdV equation (1.1). This completes the proof.
∎

###### Remark 4.5.

(L2superscript𝐿2L^{2}-conservative)
Beyond mere formulation, the fully discrete scheme (4.1) not only demonstrates its computational capability but also exhibits a remarkable L2superscript𝐿2L^{2}-conservativity. To justify the assertion, we perform a summation over j𝑗j after multiplying Δ​x​ujn+1/2Δ𝑥superscriptsubscript𝑢𝑗𝑛12\Delta xu_{j}^{n+1/2} in (4.1), yielding the equation:

‖un+1‖2=‖un‖2−Δ​t​⟨𝔾​(un+1/2),un+1/2⟩−Δ​t​⟨𝔻α​D​(un+1/2),un+1/2⟩.superscriptnormsuperscript𝑢𝑛12superscriptnormsuperscript𝑢𝑛2Δ𝑡𝔾superscript𝑢𝑛12superscript𝑢𝑛12Δ𝑡superscript𝔻𝛼𝐷superscript𝑢𝑛12superscript𝑢𝑛12\|u^{n+1}\|^{2}=\|u^{n}\|^{2}-\Delta t\langle\mathbb{G}(u^{n+1/2}),u^{n+1/2}\rangle-\Delta t\langle\mathbb{D}^{\alpha}D(u^{n+1/2}),u^{n+1/2}\rangle.

This further establishes the L2superscript𝐿2L^{2}-conservative nature of the scheme (4.1)

‖un+1‖=‖un‖,normsuperscript𝑢𝑛1normsuperscript𝑢𝑛\|u^{n+1}\|=\|u^{n}\|,

where we have incorporated the orthogonality conditions ⟨𝔾​(u),u⟩=0𝔾𝑢𝑢0\langle\mathbb{G}(u),u\rangle=0 and ⟨𝔻α​D​(u),u⟩=0superscript𝔻𝛼𝐷𝑢𝑢0\langle\mathbb{D}^{\alpha}D(u),u\rangle=0.

## 5. Numerical experiment

In this section, we verify the schemes (3.1) and (4.1) with a series of numerical illustrations. We follow the conventional approach, which typically involve a periodic case of the initial value problem with periodic initial data. We consider a large enough domain in all the cases such that the initial data is compactly supported within the domain, for instance, kindly refer to [5, 10, 9, 19].
However, in particular, our theoretical study in this paper focuses on the convergence of the approximated solution on the real line. To address this, we discretize the domain that is large enough in space for the reference solutions (exact or higher-grid solutions) to be nearly zero outside of it. Exact solutions are available for the cases α=1𝛼1\alpha=1 and α=2𝛼2\alpha=2.

Let us denote the approximate solutions by uΔ​xE​Isuperscriptsubscript𝑢Δ𝑥𝐸𝐼u_{\Delta x}^{EI} and uΔ​xC​Nsuperscriptsubscript𝑢Δ𝑥𝐶𝑁u_{\Delta x}^{CN} generated by the Euler implicit (3.1) and Crank-Nicolson (4.1) schemes respectively. We introduce the relative error as

EE​I:=‖uΔ​xE​I−u‖L2‖u‖L2,EC​N:=‖uΔ​xC​N−u‖L2‖u‖L2,formulae-sequenceassignsubscript𝐸𝐸𝐼subscriptnormsuperscriptsubscript𝑢Δ𝑥𝐸𝐼𝑢superscript𝐿2subscriptnorm𝑢superscript𝐿2assignsubscript𝐸𝐶𝑁subscriptnormsuperscriptsubscript𝑢Δ𝑥𝐶𝑁𝑢superscript𝐿2subscriptnorm𝑢superscript𝐿2E_{EI}:=\frac{\|u_{\Delta x}^{EI}-u\|_{L^{2}}}{\|u\|_{L^{2}}},\qquad E_{CN}:=\frac{\|u_{\Delta x}^{CN}-u\|_{L^{2}}}{\|u\|_{L^{2}}},

where EE​Isubscript𝐸𝐸𝐼E_{EI} and EC​Nsubscript𝐸𝐶𝑁E_{CN} are the relative errors corresponding to the Euler implicit (3.1) and Crank-Nicolson (4.1) respectively. The L2superscript𝐿2L^{2}-norms involved above were computed at the grid points xjsubscript𝑥𝑗x_{j} by trapezoidal rule.

Hereby we examine the first three specific quantities—mass, momentum and energy for the scheme (4.1)—as introduced in [22]. These quantities, being normalized, are defined as follows:

C1Δsubscriptsuperscript𝐶Δ1\displaystyle C^{\Delta}_{1}
:=∫ℝuΔ​xC​N​𝑑x∫ℝu0​𝑑x,C2Δ:=‖uΔ​xC​N‖L2​(ℝ)‖u0‖L2​(ℝ),formulae-sequenceassignabsentsubscriptℝsuperscriptsubscript𝑢Δ𝑥𝐶𝑁differential-d𝑥subscriptℝsubscript𝑢0differential-d𝑥assignsubscriptsuperscript𝐶Δ2subscriptnormsuperscriptsubscript𝑢Δ𝑥𝐶𝑁superscript𝐿2ℝsubscriptnormsubscript𝑢0superscript𝐿2ℝ\displaystyle:=\frac{\int_{\mathbb{R}}u_{\Delta x}^{CN}\,dx}{\int_{\mathbb{R}}u_{0}\,dx},\qquad C^{\Delta}_{2}:=\frac{\|u_{\Delta x}^{CN}\|_{L^{2}(\mathbb{R})}}{\left\|u_{0}\right\|_{L^{2}(\mathbb{R})}},

C3Δsubscriptsuperscript𝐶Δ3\displaystyle C^{\Delta}_{3}
:=∫ℝ(((−Δ)α/4​uΔ​xC​N)2−13​(uΔ​xC​N)3)​𝑑x∫ℝ(((−Δ)α/4​u0)2−13​(u0)3)​𝑑x.assignabsentsubscriptℝsuperscriptsuperscriptΔ𝛼4superscriptsubscript𝑢Δ𝑥𝐶𝑁213superscriptsuperscriptsubscript𝑢Δ𝑥𝐶𝑁3differential-d𝑥subscriptℝsuperscriptsuperscriptΔ𝛼4subscript𝑢0213superscriptsubscript𝑢03differential-d𝑥\displaystyle:=\frac{\int_{\mathbb{R}}\left(((-\Delta)^{\alpha/4}u_{\Delta x}^{CN})^{2}-\frac{1}{3}(u_{\Delta x}^{CN})^{3}\right)~{}dx}{\int_{\mathbb{R}}\left(((-\Delta)^{\alpha/4}u_{0})^{2}-\frac{1}{3}(u_{0})^{3}\right)~{}dx}.

Our objective is to preserve these quantities within our discrete framework, ensuring that CiΔ→1absent→subscriptsuperscript𝐶Δ𝑖1C^{\Delta}_{i}\xrightarrow[]{}1, i=1,2,3𝑖123i=1,2,3 as number of nodes N𝑁N increases.
It is worth noting that in the domain of integro partial differential equations, maintaining a greater number of conserved quantities through numerical methods often leads to more accurate approximations compared to those preserving fewer quantities.

Moreover, we investigate the convergence rates of the numerical schemes (3.1) and (4.1), labeled as RE​Isubscript𝑅𝐸𝐼R_{EI} and RC​Nsubscript𝑅𝐶𝑁R_{CN} respectively. The convergence rates are computed using the expressions:

RE​I=ln⁡(EE​I​(N1))−ln⁡(EE​I​(N2))ln⁡(N2)−ln⁡(N1)​ and ​RC​N=ln⁡(EC​N​(N1))−ln⁡(EC​N​(N2))ln⁡(N2)−ln⁡(N1),subscript𝑅𝐸𝐼subscript𝐸𝐸𝐼subscript𝑁1subscript𝐸𝐸𝐼subscript𝑁2subscript𝑁2subscript𝑁1 and subscript𝑅𝐶𝑁subscript𝐸𝐶𝑁subscript𝑁1subscript𝐸𝐶𝑁subscript𝑁2subscript𝑁2subscript𝑁1R_{EI}=\frac{\ln(E_{EI}(N_{1}))-\ln(E_{EI}(N_{2}))}{\ln(N_{2})-\ln(N_{1})}~{}~{}~{}\text{ and }~{}~{}~{}R_{CN}=\frac{\ln(E_{CN}(N_{1}))-\ln(E_{CN}(N_{2}))}{\ln(N_{2})-\ln(N_{1})},

where EE​Isubscript𝐸𝐸𝐼E_{EI} and EC​Nsubscript𝐸𝐶𝑁E_{CN} are treated as functions dependent on the number of nodes N1subscript𝑁1N_{1} and N2subscript𝑁2N_{2}.
The CFL condition was imposed with the time step Δ​t=0.5​Δ​x/‖u0‖∞Δ𝑡0.5Δ𝑥subscriptnormsubscript𝑢0\Delta t=0.5\Delta x/\|u_{0}\|_{\infty}.

The validation of our exposition will manifest in the forthcoming examples:

### 5.1. Benjamin-Ono equation

α=1𝛼1\alpha=1:
Let us consider the solution of the Benjamin-Ono equation as introduced in [36]:

u1​(x,t)=2​c​δ1−1−δ2​cos⁡(c​δ​(x−c​t)),δ=πc​L.formulae-sequencesubscript𝑢1𝑥𝑡2𝑐𝛿11superscript𝛿2𝑐𝛿𝑥𝑐𝑡𝛿𝜋𝑐𝐿u_{1}(x,t)=\frac{2c\delta}{1-\sqrt{1-\delta^{2}}\cos(c\delta(x-ct))},\qquad\delta=\frac{\pi}{cL}.

(5.1)

We implemented both the schemes using the initial data u0​(x)=u1​(x,0)superscript𝑢0𝑥subscript𝑢1𝑥0u^{0}(x)=u_{1}(x,0) with parameters L=15𝐿15L=15 and c=0.25𝑐0.25c=0.25. By choosing α=1𝛼1\alpha=1 in both schemes (3.1) and (4.1), we compare the obtained approximate solutions with the reference solution given by (5.1) at the time t=20𝑡20t=20, t=100𝑡100t=100 and t=120𝑡120t=120. Since the solution is periodic, the time t=120𝑡120t=120, represents one period for the exact solution (5.1).

A graphical representation of the results for N=512𝑁512N=512 is provided in Figure 5.1 for the time t=20𝑡20t=20, t=100𝑡100t=100 and t=120𝑡120t=120. The plot distinctly shows that the approximate solution uΔ​xC​Nsuperscriptsubscript𝑢Δ𝑥𝐶𝑁u_{\Delta x}^{CN} closely aligns with the exact solution compared to the approximate solution uΔ​xE​Isuperscriptsubscript𝑢Δ𝑥𝐸𝐼u_{\Delta x}^{EI}. This observation is further substantiated by the errors presented in Table 5.1 for the time t=120𝑡120t=120 and for the other times outputs are very similar, where the errors exhibit a decreasing trend with a rate of approximately 111 for the Euler implicit scheme (3.1) and 222 for the Crank-Nicolson scheme (4.1). Furthermore, Table 5.1 verifies that the Crank-Nicolson scheme (4.1) conserves the aforementioned quantities CiΔsuperscriptsubscript𝐶𝑖ΔC_{i}^{\Delta}, i=1,2,3𝑖123i=1,2,3.

N
EE​Isubscript𝐸𝐸𝐼E_{EI}
RE​Isubscript𝑅𝐸𝐼R_{EI}
EC​Nsubscript𝐸𝐶𝑁E_{CN}
RC​Nsubscript𝑅𝐶𝑁R_{CN}
C1Δsuperscriptsubscript𝐶1ΔC_{1}^{\Delta}
C2Δsuperscriptsubscript𝐶2ΔC_{2}^{\Delta}
C3Δsuperscriptsubscript𝐶3ΔC_{3}^{\Delta}

64
0.0216

0.0650

1.052
2.47
1.052

1.100

2.949

128
0.0101

0.0084

1.003
1.04
1.04

0.947

2.150

256
0.0052

0.0019

1.000
1.000
1.000

1.056

2.013

512
0.0025

4.7034e-04

1.000
1.000
1.000

1.049

2.004

1024
0.0012

1.1720e-04

1.000
1.000
1.000

### 5.2. Fractional KdV equation

α=1.5𝛼1.5\alpha=1.5: 
We assess the convergence using the initial condition u0​(x)=0.5​sin⁡(x)subscript𝑢0𝑥0.5𝑥u_{0}(x)=0.5\sin(x) within the interval [−4​π,4​π]4𝜋4𝜋[-4\pi,4\pi]. Once again, by selecting α=1.5𝛼1.5\alpha=1.5 in both schemes (3.1) and (4.1), we compare the approximate solutions at time t=5𝑡5t=5 with the reference solution obtained using a higher number of grid points, N=32000𝑁32000N=32000, as the exact solution is unknown for this case. Table 5.2 affirms the convergence of both schemes and quantities CiΔsuperscriptsubscript𝐶𝑖ΔC_{i}^{\Delta}, i=1,2,3𝑖123i=1,2,3, are conserved for the Crank-Nicolson scheme (4.1), and Figure 5.2 illustrates that the Crank-Nicolson scheme (4.1) converges more rapidly than the Euler implicit scheme (3.1).

N
EE​Isubscript𝐸𝐸𝐼E_{EI}
RE​Isubscript𝑅𝐸𝐼R_{EI}
EC​Nsubscript𝐸𝐶𝑁E_{CN}
RC​Nsubscript𝑅𝐶𝑁R_{CN}
C1subscript𝐶1C_{1}
C2subscript𝐶2C_{2}
C3subscript𝐶3C_{3}

250
0.5848

0.0733

0.996
1.00
0.996

0.733

1.418

500
0.3517

0.0274

0.999
1.00
0.999

0.877

2.037

1000
0.1915

0.0067

1.00
1.00
0.999

0.983

1.979

2000
0.0969

0.0017

1.00
1.00
1.00

1.131

2.085

4000
0.0443

0.0004

1.00
1.00
1.00

### 5.3. KdV equation

α≈2𝛼2\alpha\approx 2: 
Our objective is to analyze the behaviour of the solution of (1.1) whenever the exponent α𝛼\alpha is close to 222. As α=2𝛼2\alpha=2 correspond to the KdV equation, we compare the solutions generated by the schemes (3.1) and (4.1) by considering α≈2𝛼2\alpha\approx 2 with the solution of KdV equation. For instance, the two-soliton solution for the KdV equation ut+(u22)x+ux​x​x=0subscript𝑢𝑡subscriptsuperscript𝑢22𝑥subscript𝑢𝑥𝑥𝑥0u_{t}+\left(\frac{u^{2}}{2}\right)_{x}+u_{xxx}=0 is explicitly introduced in [19] as follows:

u2​(x,t)=6​(c−d)​d​csch2⁡(d/2​(x−2​d​t))+c​sech2⁡(c/2​(x−2​c​t))(c​tanh⁡(c/2​(x−2​c​t))−d​coth⁡(d/2​(x−2​d​t)))2.subscript𝑢2𝑥𝑡6𝑐𝑑𝑑superscriptcsch2𝑑2𝑥2𝑑𝑡𝑐superscriptsech2𝑐2𝑥2𝑐𝑡superscript𝑐𝑐2𝑥2𝑐𝑡𝑑hyperbolic-cotangent𝑑2𝑥2𝑑𝑡2u_{2}(x,t)=6(c-d)\frac{d\operatorname{csch}^{2}\left(\sqrt{d/2}(x-2dt)\right)+c\operatorname{sech}^{2}\left(\sqrt{c/2}(x-2ct)\right)}{\left(\sqrt{c}\tanh\left(\sqrt{c/2}(x-2ct)\right)-\sqrt{d}\coth\left(\sqrt{d/2}(x-2dt)\right)\right)^{2}}.

(5.2)

Here c𝑐c and d𝑑d are any real parameters. We set the values of parameters c=0.5𝑐0.5c=0.5 and d=1𝑑1d=1. We use u0E​I​(x,0)=u2​(x,−10)superscriptsubscript𝑢0𝐸𝐼𝑥0subscript𝑢2𝑥10u_{0}^{EI}(x,0)=u_{2}(x,-10) as the initial data for the scheme (3.1) and u0C​N​(x,0)=u2​(x,−20)superscriptsubscript𝑢0𝐶𝑁𝑥0subscript𝑢2𝑥20u_{0}^{CN}(x,0)=u_{2}(x,-20) as the initial data for the scheme (4.1).

N
E
REsubscript𝑅𝐸R_{E}

2000
2.5855

1.313

4000
1.0403

1.029

8000
0.5100

1.086

16000
0.2402

1.048

32000
0.1161

N
E
C1subscript𝐶1C_{1}
C2subscript𝐶2C_{2}
REsubscript𝑅𝐸R_{E}

250
1.1999
1.00
0.97

2.942

500
0.1561
1.00
1.01

1.839

1000
0.0436
1.00
1.00

1.989

2000
0.0110
1.00
1.00

2.0809

4000
0.0026
1.00
1.00

From a physical perspective, the taller soliton overtakes the shorter one, and both solitons emerge unaltered after the collision. This scenario presents a considerably more intricate computational challenge than solving for a single soliton solution. We choose α=1.999𝛼1.999\alpha=1.999 and compare the approximate solutions obtained by the schemes (3.1) and (4.1) to the exact solution of the KdV equation given by (5.2) at t=20𝑡20t=20 and t=40𝑡40t=40 respectively. Specifically, the approximate solution at t=20𝑡20t=20 obtained by the Euler implicit scheme (3.1) is denoted as uΔ​xE​I​(x,20)superscriptsubscript𝑢Δ𝑥𝐸𝐼𝑥20u_{\Delta x}^{EI}(x,20) and is compared with u2​(x,10)subscript𝑢2𝑥10u_{2}(x,10). The approximate solution at t=40𝑡40t=40 obtained by the Crank-Nicolson scheme (4.1) is denoted as uΔ​xE​I​(x,40)superscriptsubscript𝑢Δ𝑥𝐸𝐼𝑥40u_{\Delta x}^{EI}(x,40) and is compared with u2​(x,20)subscript𝑢2𝑥20u_{2}(x,20). Figure 5.4 visually depicts that the taller soliton surpasses the smaller one at t=20𝑡20t=20, confirming the physical interpretation of soliton solutions. Table 5.4 and Table 5.4 display the convergence rates for both schemes. The Figure 5.3 and Figure 5.4 provide graphical representations of the approximated and exact solutions of the KdV equation.

With the help of our several numerical illustrations we find that the scheme (4.1) converges faster than the scheme (3.1). In addition, the scheme (4.1) preserves the conserved quantities with expected rates.

## 6. Concluding remarks

In this study, we have rigorously investigated the stability and convergence properties of Euler implicit and a Crank-Nicolson conservative finite difference scheme applied to the Cauchy problem associated with the fractional KdV equation. Notably, the Crank-Nicolson scheme exhibited superior convergence rates. Our comprehensive investigation into the stability, accuracy, and convergence of the proposed numerical schemes contributes valuable insights to the numerical analysis of the fractional KdV equation. The results presented herein not only enhance our understanding of the behavior of approximate solutions but also provide a solid foundation for future research in the numerical simulation of nonlinear dispersive equations involving the fractional Laplacian.

## References

- [1]

P. Amorim and M. Figueira.

Convergence of a finite difference method for the KdV and modified KdV equations with L2superscript𝐿2L^{2} data.

Portugaliae Mathematica, 70 (2013), no. 1, 23–50.

- [2]

J. L. Bona and R. Smith.

The initial-value problem for the Korteweg-de Vries equation.

Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences, 278 (1975), no. 1287, 555–601.

- [3]

C. Courtès, F. Lagoutière and F. Rousset.

Error estimates of finite difference schemes for the Korteweg–de Vries equation.

IMA Journal of Numerical Analysis, 40 (2020), no. 1, 628–685.

- [4]

E. Di Nezza, G. Palatucci and E. Valdinoci.

Hitchhiker’s guide to the fractional Sobolev spaces.

Bulletin des Sciences Mathématiques, 136 (2012), no. 5, 521–573.

- [5]

R. Dutta, H. Holden, U. Koley, and N. H. Risebro.

Convergence of finite difference schemes for the Benjamin–Ono equation.

Numerische Mathematik, 134 (2016), no. 2, 249–274.

- [6]

R. Dutta, U. Koley, and N. H. Risebro.

Convergence of a higher order scheme for the Korteweg–de Vries equation.

SIAM Journal on Numerical Analysis, 53 (2015), no. 4, 1963–1983.

- [7]

R. Dutta and N. H. Risebro.

A note on the convergence of a Crank–Nicolson scheme for the KdV equation.

Int. J. Numer. Anal. Model, 13 (2016), no. 5, 567–575.

- [8]

R. Dutta and T. Sarkar.

Operator splitting for the fractional Korteweg-de Vries equation.

Numerical Methods for Partial Differential Equations, 37 (2021), no. 6, 3000–3022.

- [9]

M. Dwivedi, T. Sarkar.

Convergence of a conservative Crank-Nicolson finite difference scheme for the KdV equation with smooth and non-smooth initial data.

arXiv preprint arXiv:2312.14454, (2023).

- [10]

M. Dwivedi, T. Sarkar.

Stability and Convergence analysis of a Crank-Nicolson Galerkin scheme for the fractional Korteweg-de Vries equation.

arXiv preprint arXiv:2311.06589, (2023).

- [11]

M. Ehrnstrom, and Y. Wang.

Enhanced Existence Time of Solutions to the Fractional Korteweg–de Vries Equation.

SIAM Journal on Mathematical Analysis, 51 (2019), no. 4, 3298–3323.

- [12]

V. J. Ervin and J. P. Roop.

Variational formulation for the stationary fractional advection dispersion equation.

Numerical Methods for Partial Differential Equations: An International Journal, 22 (2006), no. 3, 558–576.

- [13]

A. S. Fokas and B. Fuchssteiner.

The hierarchy of the Benjamin-Ono equation.

Physics letters A, 86 (1981), no. 6-7, 341-345.

- [14]

G. Fonseca, F. Linares, and G. Ponce.

The IVP for the dispersion generalized Benjamin-Ono equation in weighted Sobolev spaces.

Ann. Inst. H. Poincaré Anal. Non Linéaire, 30 (2013), no. 5, 763–790.

- [15]

S. T. Galtung.

Convergent Crank–Nicolson Galerkin Scheme for the Benjamin–Ono Equation.

Discrete and dynamical systems, 38 (2018), no. 3, 1243–1268.

- [16]

E. Herr, A. D. Ionescu, C. E. Kenig,
and H. Koch.

 A Para-differential renormalization technique
for nonlinear dispersive equations.

Communication in partial differential equation, 35 (2010), no. 10, 1827–1875.

- [17]

H. Holden, K. H. Karlsen and N. H. Risebro.

Operator splitting Methods for Generalized Korteweg–De Vries Equations.

Journal of Computational Physics, 153 (1999), no. 1, 203–222.

- [18]

H. Holden, K. Karlsen, N. H. Risebro and T. Tao.

Operator splitting for the KdV equation.

Mathematics of Computation, 80 (2011), no. 274, 821–846.

- [19]

H. Holden, U. Koley and N. H. Risebro.

Convergence of a fully discrete finite difference scheme for the Korteweg–de Vries equation.

IMA Journal of Numerical Analysis, 35 (2015), no. 3, 1047–1077.

- [20]

T. Kato.

 On the Cauchy problem for the (generalized) Korteweg–de Vries equation.

Studies in Appl. Math. Ad. in Math. Suppl. Stud., (1983), no. 8, 93–128.

- [21]

C. E. Kenig, G. Ponce and L. Vega.

Well-Posedness of the Initial Value Problem for the Korteweg-de Vries Equation.

Journal of the American Mathematical Society, 4 (1991), no. 2, 323–347.

- [22]

C. E. Kenig, G. Ponce and L. Vega.

The Cauchy problem for the Korteweg–de Vries equation in Sobolev spaces of negative indices.

Duke Mathematical Journal, 71 (1993), no. 1, 1–21.

- [23]

C. E. Kenig, G. Ponce and L. Vega.

On the generalized Benjamin-Ono equation.

Transactions of the American Mathematical Society, 342 (1994), no. 1, 155–172.

- [24]

C. E. Kenig, G. Ponce and L. Vega.

On the unique continuation of solutions to non-local non-linear dispersive equations.

Communications in Partial Differential Equations, 45 (2020), no. 8, 872–886.

- [25]

R. Killip and M. Vişan.

KdV is wellposed in H−1superscript𝐻1H^{-1}.

Annals of Mathematics, 190 (2019), no. 1, 249–305.

- [26]

U. Koley, D. Ray, and T. Sarkar.

Multilevel Monte Carlo finite difference methods for fractional conservation laws with random data.

SIAM/ASA Journal on Uncertainty Quantification, 9 (2021), no.1, 65–105.

- [27]

D. J. Korteweg and G. d. Vries.

On the change of form of long waves advancing in a rectangular canal, and on a new type of long stationary waves.

The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 39 (1895), no. 240, 422–443.

- [28]

J. Li and M. R. Visbal.

High-order compact schemes for nonlinear dispersive waves.

Journal of Scientific Computing, 26 (2006), 1–23.

- [29]

F. Linares and G. Ponce.

Introduction to nonlinear dispersive equations.

University text, Springer, New York, (2015).

- [30]

K. Mateusz.

Ten equivalent definitions of the fractional Laplace operator.

Fractional Calculus and Applied Analysis, 20 (2017), no. 1, 7–51.

- [31]

L. Molinet, D. Pilod and S. Vento.

On well-posedness for some dispersive perturbations of
Burgers’ equation.

Ann. Inst. H. Poincaré Anal. Non Linéaire, 35 (2018), no. 7, 1719–1756.

- [32]

G. Ponce.

On the global well-posedness of the Benjamin-Ono equation.

Diff. Int. Eq., 4
(1991), 527–542.

- [33]

A. Sjöberg.

On the Korteweg-de Vries equation: Existence and uniqueness.

Journal of Mathematical Analysis and Applications, 29 (1970), no. 3, 569–579.

- [34]

J. O. Skogestad and H. Kalisch.

A boundary value problem for the KdV equation: Comparison of finite-difference and Chebyshev methods.

Mathematics and Computers in Simulation, 80 (2009), no. 1, 151–163.

- [35]

T. Tao.

Global well-posedness of the Benjamin–Ono equation in H1​(ℝ)superscript𝐻1ℝH^{1}(\mathbb{R}).

Journal of Hyperbolic Differential Equations, 1 (2004), no. 1, 27–49.

- [36]

V. Thomée and A. S. Vasudeva Murthy.

A numerical method for the Benjamin–Ono equation.

BIT Numerical Mathematics, 38 (1998), 597–611.

- [37]

X. Wang, W. Dai and M. Usman.

A high-order accurate finite difference scheme for the KdV equation with time-periodic boundary forcing.

Applied Numerical Mathematics, 160 (2021), 102–121.

- [38]

Q. Xu and J. S. Hesthaven .

Discontinuous Galerkin method for fractional convection-diffusion equations.

SIAM Journal on Numerical Analysis, 52 (2014), no. 1, 405–423.

- [39]

Q. Yang, F. Liu and I. Turner.

Numerical methods for fractional partial differential equations with Riesz space fractional derivatives.

Applied Mathematical Modelling, 34 (2010), no. 1, 200–218.

Generated on Fri Apr 5 14:12:44 2024 by LaTeXML
