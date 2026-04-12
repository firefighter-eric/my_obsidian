# Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Xu, Choi - 2020 - Revealing the Myth of Higher-Order Inference in Coreference Resolution.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2009.12013
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Revealing the Myth of Higher-Order Inference in Coreference Resolution

Liyan Xu 
Computer Science 
Emory University, Atlanta, GA 
liyan.xu@emory.edu 
&Jinho D. Choi 
Computer Science 
Emory University, Atlanta, GA 
jinho.choi@emory.edu

###### Abstract

This paper analyzes the impact of higher-order inference (HOI) on the task of coreference resolution.
HOI has been adapted by almost all recent coreference resolution models without taking much investigation on its true effectiveness over representation learning.
To make acomprehensive analysis, we implement an end-to-end coreference system as well as four HOI approaches, attended antecedent, entity equalization, span clustering, and cluster merging, where the latter two are our original methods.
We find that given a high-performing encoder such as SpanBERT, the impact of HOI is negative to marginal, providing a new perspective of HOI to this task.
Our best model using cluster merging shows the Avg-F1 of 80.2 on the CoNLL 2012 shared task dataset in English.

## 1 Introduction

Coreference resolution has always been considered one of the unsolved NLP tasks due to its challenging aspect of document-level understanding Wiseman et al. (2015, 2016); Clark and Manning (2015, 2016); Lee et al. (2017).
Nonetheless, it has made a tremendous progress in recent years by adapting contextualized embedding encoders such as ELMo Lee et al. (2018); Fei et al. (2019) and BERT Kantor and Globerson (2019); Joshi et al. (2019, 2020).
The latest state-of-the-art model shows the improvement of 12.4% over the model introduced 2.5 years ago, where the major portion of the improvement is derived by representation learning (Figure 1).

Most of these previous models have also adapted higher-order inference (HOI) for the global optimization of coreference links, although HOI clearly has not been the focus of those works, for the fact that gains from HOI have been reported marginal.
This has inspired us to analyze the impact of HOI on modern coreference resolution models in order to envision the future direction of this research.

To make thorough ablation studies among different approaches, we implement an end-to-end coreference system in PyTorch (Sec 3.1),
and two HOI approaches proposed by previous work, attended antecedent and entity equalization (Sec 3.2), along with two of our original approaches, span clustering and cluster merging (Sec 3.3).
These approaches are experimented with two Transformer encoders, BERT and SpanBERT, to assess how effective HOI is even when coupled with those high-performing encoders (Sec 4).
To the best of our knowledge,this is the first work to make a comprehensive analysis on multiple HOI approaches side-by-side for the task of coreference resolution.111Source codes and models are available at https://github.com/lxucs/coref-hoi.

## 2 Related Work

Most neural network-based coreference resolution models have adapted antecedent-ranking Wiseman et al. (2015); Clark and Manning (2015); Lee et al. (2017, 2018); Joshi et al. (2019, 2020), which relies on the local decisions between each mention and its antecedents.
To achieve deeper global optimization,Wiseman et al. (2016); Clark and Manning (2016); Yu et al. (2020) built entity representations in the ranking process, whereas Lee et al. (2018); Kantor and Globerson (2019) refined the mention representation by aggregating its antecedents’ information.

It is no secret that the integration of contextualized embeddings has played the most critical role in this task. While the following are based on the same end-to-end coreference model Lee et al. (2017),
Lee et al. (2018); Fei et al. (2019) reported 3.3% improvement by adapting ELMo in the encoders Peters et al. (2018).
Kantor and Globerson (2019); Joshi et al. (2019) gained additional 3.3% by adapting BERT Devlin et al. (2019).
Joshi et al. (2020) introduced SpanBERT that gave another 2.7% improvement over Joshi et al. (2019).

Most recently, Wu et al. (2020) proposes a new model that adapts question-answering framework on coreference resolution, and achieves state-of-the-art result of 83.1 on the CoNLL’12 shared task.

## 3 Approach

### 3.1 End-to-End Coreference System

We reimplement the end-to-end c2f-coref model introduced by Lee et al. (2018) that has been adapted by every coreference resolution model since then. It detects mention candidates through span enumeration and aggressive pruning.
For each candidate span x𝑥x, the model learns the distribution over its antecedents y∈𝒴​(x)𝑦𝒴𝑥y\in\mathcal{Y}(x):

P​(y)=es​(x,y)∑y′∈𝒴​(x)es​(x,y′)𝑃𝑦superscript𝑒𝑠𝑥𝑦subscriptsuperscript𝑦′𝒴𝑥superscript𝑒𝑠𝑥superscript𝑦′P(y)=\frac{e^{s(x,y)}}{\sum_{y^{\prime}\in\mathcal{Y}(x)}e^{s(x,y^{\prime})}}

(1)

where s​(x,y)𝑠𝑥𝑦s(x,y) is the local score involving two parts: how likely the spans x𝑥x and y𝑦y are valid mentions, and how likely they refer to the same entity:

s​(x,y)𝑠𝑥𝑦\displaystyle s(x,y)
=sm​(x)+sm​(y)+sc​(x,y)absentsubscript𝑠𝑚𝑥subscript𝑠𝑚𝑦subscript𝑠𝑐𝑥𝑦\displaystyle=s_{m}(x)+s_{m}(y)+s_{c}(x,y)

(2)

sm​(x)subscript𝑠𝑚𝑥\displaystyle s_{m}(x)
=wm​FFNNm​(gx)absentsubscript𝑤𝑚subscriptFFNN𝑚subscript𝑔𝑥\displaystyle=w_{m}\texttt{FFNN}_{m}(g_{x})

sc​(x,y)subscript𝑠𝑐𝑥𝑦\displaystyle s_{c}(x,y)
=wc​FFNNc​(gx,gy,ϕ​(x,y))absentsubscript𝑤𝑐subscriptFFNN𝑐subscript𝑔𝑥subscript𝑔𝑦italic-ϕ𝑥𝑦\displaystyle=w_{c}\texttt{FFNN}_{c}(g_{x},g_{y},\phi(x,y))

gx,gysubscript𝑔𝑥subscript𝑔𝑦g_{x},g_{y} are the span embeddings of x𝑥x and y𝑦y, ϕ​(x,y)italic-ϕ𝑥𝑦\phi(x,y) is the meta-information (e.g., speakers, distance), and wm,wcsubscript𝑤𝑚subscript𝑤𝑐w_{m},w_{c} are the mention and coreference scores, respectively (FFNN: feedforward neural network).

We use different Transformers-based encoders, and follow the “independent” setup for long documents as suggested by Joshi et al. (2019).

### 3.2 Span Refinement

Two HOI methods presented by recent coreference work are based on span refinement that aggregates non-local features to enrich the span representation with more “global” information.
The updated span representation gx′superscriptsubscript𝑔𝑥′g_{x}^{\prime} can be derived as in Eq. 3, where gx′superscriptsubscript𝑔𝑥′g_{x}^{\prime} is the interpolation between the current and refined representation gxsubscript𝑔𝑥g_{x} and axsubscript𝑎𝑥a_{x}, and Wfsubscript𝑊𝑓W_{f} is the gate parameter.
gx′superscriptsubscript𝑔𝑥′g_{x}^{\prime} is used to perform another round of antecedent-ranking in replacement of gxsubscript𝑔𝑥g_{x}.

gx′superscriptsubscript𝑔𝑥′\displaystyle g_{x}^{\prime}
=fx∘gx+(1−fx)∘axabsentsubscript𝑓𝑥subscript𝑔𝑥1subscript𝑓𝑥subscript𝑎𝑥\displaystyle=f_{x}\circ g_{x}+(1-f_{x})\circ a_{x}

(3)

fxsubscript𝑓𝑥\displaystyle f_{x}
=σ​(Wf​[gx,ax])absent𝜎subscript𝑊𝑓subscript𝑔𝑥subscript𝑎𝑥\displaystyle=\sigma(W_{f}[g_{x},a_{x}])

The following two methods share the same updating process for gx′superscriptsubscript𝑔𝑥′g_{x}^{\prime}, but with different ways to obtain the refined span representation axsubscript𝑎𝑥a_{x}.

#### Attended Antecedent (AA)

takes the antecedent information to enrich gx′superscriptsubscript𝑔𝑥′g_{x}^{\prime} (Lee et al., 2018; Fei et al., 2019; Joshi et al., 2019, 2020).
The refined span axsubscript𝑎𝑥a_{x} is the attended antecedent representation over the current antecedent distribution P​(y)𝑃𝑦P(y), where gy∈𝒴​(x)subscript𝑔𝑦𝒴𝑥g_{y\in\mathcal{Y}(x)} is the antecedent representation:

ax=∑y∈𝒴​(x)P​(y)⋅gysubscript𝑎𝑥subscript𝑦𝒴𝑥⋅𝑃𝑦subscript𝑔𝑦a_{x}=\sum_{y\in\mathcal{Y}(x)}P(y)\cdot g_{y}

(4)

#### Entity Equalization (EE)

takes the clustering relaxation as in Eq. 3.2 to model the entity distribution (Kantor and Globerson, 2019), where Q​(x∈Ey′)𝑄𝑥subscript𝐸superscript𝑦′Q(x\in E_{y^{\prime}}) is the probability of the span x𝑥x referring to an entity Ey′subscript𝐸superscript𝑦′E_{y^{\prime}} in which the span y′superscript𝑦′y^{\prime} is the first mention. P​(y)𝑃𝑦P(y) is the current antecedent distribution.

Q​(x∈Ey′)=𝑄𝑥subscript𝐸superscript𝑦′absent\displaystyle Q(x\in E_{y^{\prime}})=

{∑k=y′x−1P​(y=k)⋅Q​(k∈Ey′)y′<xP​(y=ϵ)y′=x0y′>xcasessuperscriptsubscript𝑘superscript𝑦′𝑥1⋅𝑃𝑦𝑘𝑄𝑘subscript𝐸superscript𝑦′superscript𝑦′𝑥𝑃𝑦italic-ϵsuperscript𝑦′𝑥0superscript𝑦′𝑥\displaystyle\begin{cases}\sum_{k=y^{\prime}}^{x-1}P(y=k)\cdot Q(k\in E_{y^{\prime}})&y^{\prime}<x\\
P(y=\epsilon)&y^{\prime}=x\\
0&y^{\prime}>x\end{cases}

(5)

The refined span axsubscript𝑎𝑥a_{x} is the attended entity representation, where ey(x)superscriptsubscript𝑒𝑦𝑥e_{y}^{(x)} is the entity representation to which the span y𝑦y belongs till the span x𝑥x:

ex(t)superscriptsubscript𝑒𝑥𝑡\displaystyle e_{x}^{(t)}
=∑y=1tQ​(y∈Ex)⋅gyabsentsuperscriptsubscript𝑦1𝑡⋅𝑄𝑦subscript𝐸𝑥subscript𝑔𝑦\displaystyle=\sum_{y=1}^{t}Q(y\in E_{x})\cdot g_{y}

(6)

axsubscript𝑎𝑥\displaystyle a_{x}
=∑y=1xQ​(x∈Ey)⋅ey(x)absentsuperscriptsubscript𝑦1𝑥⋅𝑄𝑥subscript𝐸𝑦superscriptsubscript𝑒𝑦𝑥\displaystyle=\sum_{y=1}^{x}Q(x\in E_{y})\cdot e_{y}^{(x)}

(7)

### 3.3 HOI with Clustering

This section introduces two new HOI methods for a more extensive study in HOI.

#### Span Clustering (SC)

is also based on span refinement, and it
constructs the actual clusters and obtains the “true” predicted entities using P​(y)𝑃𝑦P(y) instead of modeling the “soft” entity clusters through the relaxation as in EE (Section 3.2).
This way, although we lose the differentiable property, the obtaining of true entities with the same empirical inference time as EE has made SC desirable.

The entity representation eisubscript𝑒𝑖e_{i} for an entity cluster Cisubscript𝐶𝑖C_{i} is given by the attended spans in this cluster:

αtsubscript𝛼𝑡\displaystyle\alpha_{t}
=wα​FFNNα​(gt)absentsubscript𝑤𝛼subscriptFFNN𝛼subscript𝑔𝑡\displaystyle=w_{\alpha}\text{FFNN}_{\alpha}(g_{t})

αi,tsubscript𝛼𝑖𝑡\displaystyle\alpha_{i,t}
=exp⁡(αt)∑k∈Ciexp⁡(αk)absentsubscript𝛼𝑡subscript𝑘subscript𝐶𝑖subscript𝛼𝑘\displaystyle=\frac{\exp(\alpha_{t})}{\sum_{k\in C_{i}}\exp(\alpha_{k})}

eisubscript𝑒𝑖\displaystyle e_{i}
=∑t∈Ciαi,t⋅gtabsentsubscript𝑡subscript𝐶𝑖⋅subscript𝛼𝑖𝑡subscript𝑔𝑡\displaystyle=\sum_{t\in C_{i}}\alpha_{i,t}\cdot g_{t}

The entity clusters Cisubscript𝐶𝑖C_{i} are constructed in the same way as in the final cluster prediction.
The refined span axsubscript𝑎𝑥a_{x} is then equal to the representation of entity eisubscript𝑒𝑖e_{i} to which it belongs (gx∈Cisubscript𝑔𝑥subscript𝐶𝑖g_{x}\in C_{i}).

#### Cluster Merging (CM)

performs sequential antecedent ranking combining both antecedent and entity information to gradually build up the entity clusters, which is distinguished from span refinement methods that simply re-rank antecedents.
Algorithm 1 describes the ranking process for CM.
gisubscript𝑔𝑖g_{i} is the i𝑖i’th span, 𝒴​(i)𝒴𝑖\mathcal{Y}(i) is the indices of gisubscript𝑔𝑖g_{i}’s antecedents, and Cisubscript𝐶𝑖C_{i} is the cluster that gisubscript𝑔𝑖g_{i} belongs to.
The ranking score sx​(y)subscript𝑠𝑥𝑦s_{x}(y) consists of both antecedent score fasubscript𝑓𝑎f_{a} (see Eq. 2) and cluster score fcsubscript𝑓𝑐f_{c}.
To avoid overlapping between fasubscript𝑓𝑎f_{a} and fcsubscript𝑓𝑐f_{c}, we set fcsubscript𝑓𝑐f_{c} as 00 if the cluster is the initial cluster (L6).
Thus, fcsubscript𝑓𝑐f_{c} becomes the consultation such that when fc>0subscript𝑓𝑐0f_{c}>0, the span gxsubscript𝑔𝑥g_{x} is likely to match the cluster Cysubscript𝐶𝑦C_{y}, and vice versa.
fcsubscript𝑓𝑐f_{c} is computed by FFNN similar to fasubscript𝑓𝑎f_{a}, and ϕ​(Cy)italic-ϕsubscript𝐶𝑦\phi(C_{y}) is the meta-feature such as the cluster size.

1:procedure ranking(g1,⋯,gNsubscript𝑔1⋯subscript𝑔𝑁g_{1},\cdots,g_{N})

2: Ci=1,⋯,N←gi←subscript𝐶𝑖1⋯𝑁subscript𝑔𝑖C_{i=1,\cdots,N}\leftarrow g_{i}

3: R←←𝑅absentR\leftarrow ranking_order(g1,⋯,gNsubscript𝑔1⋯subscript𝑔𝑁g_{1},\cdots,g_{N})

4: for x=R1​⋯​RN𝑥subscript𝑅1⋯subscript𝑅𝑁x=R_{1}\cdots R_{N} do

5: for y∈𝒴​(x)𝑦𝒴𝑥y\in\mathcal{Y}(x) do ▷▷\triangleright Parallelized

6: fc​(gx,Cy)←0←subscript𝑓𝑐subscript𝑔𝑥subscript𝐶𝑦0f_{c}(g_{x},C_{y})\leftarrow 0 if Cy=gysubscript𝐶𝑦subscript𝑔𝑦C_{y}=g_{y}

7: sx​(y)←fa​(gx,gy)+fc​(gx,Cy,ϕ​(Cy))←subscript𝑠𝑥𝑦subscript𝑓𝑎subscript𝑔𝑥subscript𝑔𝑦subscript𝑓𝑐subscript𝑔𝑥subscript𝐶𝑦italic-ϕsubscript𝐶𝑦s_{x}(y)\leftarrow f_{a}(g_{x},g_{y})+f_{c}(g_{x},C_{y},\phi(C_{y}))

8: y′←argmaxy∈𝒴​(x)​sx​(y)←superscript𝑦′subscriptargmax𝑦𝒴𝑥subscript𝑠𝑥𝑦y^{\prime}\leftarrow\text{argmax}_{y\in\mathcal{Y}(x)}s_{x}(y)

9: if y′≠ϵsuperscript𝑦′italic-ϵy^{\prime}\neq\epsilon then

10: merge Cxsubscript𝐶𝑥C_{x} and Cy′subscript𝐶superscript𝑦′C_{y^{\prime}}

11: return s1,⋯,sNsubscript𝑠1⋯subscript𝑠𝑁s_{1},\cdots,s_{N}

Two simple configurations can be tuned for CM. We can have the sequential left-to-right ranking order or the easy-first order (L3) whose sequence is ordered by each span’s max antecedent score, building the most confident clusters first (Ng and Cardie, 2002; Clark and Manning, 2016). There can be element-wise mean or max-reduction among the spans in the two merging clusters (L10).

Distinguished from Wiseman et al. (2016), clusters in CM are searched and merged in training without the use of oracle clusters, closing the gap between training and test time.

MUC

B3

CEAFϕ4subscriptitalic-ϕ4\phi_{4}

P
R
F1

P
R
F1

P
R
F1
Avg. F1
Avg-M

L-17

78.4
73.4
75.8

68.6
61.8
65.0

62.7
59.0
60.8
67.2
-

L-18

81.4
79.5
80.4

72.2
69.5
70.8

68.2
67.1
67.6
73.0
-

F-19

85.4
77.9
81.4

77.9
66.4
71.7

70.6
66.3
68.4
73.8
-

K-19

82.6
84.1
83.4

73.3
76.2
74.7

72.4
71.1
71.8
76.6
-

J-19

84.7
82.4
83.5

76.5
74.0
75.3

74.1
69.8
71.9
76.9
-

J-20

85.8
84.8
85.3

78.3
77.9
78.1

76.4
74.2
75.3
79.6
-

BERT

85.0
82.5
83.8

77.3
74.0
75.6

74.9
70.7
72.8
77.4
77.3 (±plus-or-minus\pm0.1)

SpanBERT

85.7
85.3
85.5

78.6
78.6
78.6

76.8
74.8
75.8
79.9
79.7 (±plus-or-minus\pm0.1)

+ AA

86.1
84.8
85.4

79.3
77.3
78.3

76.0
74.7
75.4
79.7
79.4 (±plus-or-minus\pm0.2)

+ EE

85.7
84.5
85.1

78.5
77.4
77.9

76.7
73.4
75.0
79.4
78.9 (±plus-or-minus\pm0.4)

+ SC

85.5
85.2
85.4

78.4
78.5
78.4

76.5
74.1
75.2
79.7
79.2 (±plus-or-minus\pm0.3)

+ CM

85.9
85.5
85.7

79.0
78.9
79.0

76.7
75.2
75.9
80.2

79.9 (±plus-or-minus\pm0.2)

## 4 Experiments

For our experiments, the CoNLL 2012 English shared task dataset is used (Pradhan et al., 2012).
Given the end-to-end coreference system in Section 3.1, six models are developed as follows:222Appdendix A.1 provides details of our experimental settings.

- •

BERT: BERT Devlin et al. (2019) as the encoder

- •

SpanBERT: SpanBERT Joshi et al. (2020) as the encoder

- •

+AA: SpanBERT with attended antecedent (§3.2)

- •

+EE: SpanBERT with entity equalization (§3.2)

- •

+SC: SpanBERT with span clustering (§3.3)

- •

+CM: SpanBERT with cluster merging (§3.3)

Note that BERT and SpanBERT completely rely on only local decisions without any HOI.
Particularly, +AA is equivalent to Joshi et al. (2020).

### 4.1 Results

Table 1 shows the best results in comparison to previous state-of-the-art systems. We also report the mean scores and standard deviations from 5 repeated developments, which we could not find from the previous works.

The impact of SpanBERT over BERT is clear, showing 2.4% improvement on average.
However, none of the HOI models shows a clear advantage over SpanBERT which adapts no HOI.
In fact, all HOI models except for CM show negative impact.
The best result is achieved by CM with the Avg-F1 of 80.2, surpassing the previous best result of 79.6 based on c2f-coref reported by Joshi et al. (2020).

### 4.2 Impact Analysis of HOI

Three HOI methods based on span refinement, AA, EE, and SC, show negative impact upon local decisions.
We suspect that error propagation from antecedent-ranking may downgrade the quality of refinement.
On the other hand, CM shows marginal improvement, suggesting that maintaining entity clusters can be superior to span refinement, at the cost of more inference time from the sequential ranking process.
To analyze the direct impact of HOI, we take the trained models of each HOI method and evaluate them on the test set while turning off HOI, making it compatible to SpanBERT.

The averaged performance drop w.r.t Avg-F1 after turning off HOI is less than 0.2 for all methods(Appendix A.3), implying that none of the HOI method has a significantly direct impact to the final performance of the model using SpanBERT.

W2C
C2W
C2C
W2W

+ AA

240.8 (1.3)
241.2 (1.3)
16262.2
2168.4

+ EE

244.1 (1.3)
245.3 (1.3)
16183.3
2136.3

+ SC

248.2 (1.3)
262.0 (1.4)
16184.4
2146.0

+ CM

226.4 (1.2)
235.0 (1.2)
16446.0
2180.0

In further investigation, we examine the change of coreferent links w.r.t their correctness. Specifically, Table 2 shows the four types of link changes before and after HOI. It demonstrates that the benefits from HOI is diminished because the effects are two-sided: there are roughly same amounts of links (about 1%) becoming correct or wrong after HOI, therefore neither HOI method leads to much improvement overall.

It is worth mentioning that the impact of HOI is not limited to only global decisions.
HOI implicitlyserves as a way of regularization that impacts local decisions as well, since HOI and local ranking are mutually dependent during training.
Such indirect influence of HOI makes it difficult to assess its true impact, which we will explore more in the future.

### 4.3 Analysis of Pronoun Resolution

SP
PS
FL
WL
BC

BERT
2.3
6.5
213.8
186.3
48.8 (3.5)

SpanBERT
2.8
6.6
218.3
168.0

43.8 (2.7)

+ AA

1.8
8.8
214.2
159.4
44.8 (2.4)

+ EE

1.8
5.5
210.0
165.3
44.0 (2.5)

+ SC

3.8
7.2
223.6
170.0
45.4 (3.0)

+ CM

3.0
6.6
208.0
162.2

43.8 (2.6)

#### Direct Inference

For the error analysis, we examine the direct inference between two personal pronouns.333Ambiguous pronouns such as “you” are excluded in direct inference analysis, and included in indirect inference analysis.
SP/PS in Table 3 shows the numbers of links that one pronoun incorrectly selects another pronoun with different plurality as its antecedent.
We find that adapting HOI shows slightly higher impact than switching to a more advanced encoder.
AA can reinforce the pronoun representation to bias towards singularity and lead to lower SP error and higher PS error, while the difference between BERT and SpanBERT is trivial on SP/PS.

We also look at the general types of coreferent errors involving two pronouns.
False Link (FL) falsely links a non-anaphoric pronoun to another pronoun as antecedent; Wrong Link (WL) links an anaphoric pronoun to another wrong pronoun as antecedent. Table 3 shows that EE and CM reduce FL errors by 4+%, suggesting that the aggregation of non-local features indeed leads to more conservative linking decisions.
However, adapting an advanced encoder shows higher impact on WL errors, as SpanBERT reduces almost 10% compared to BERT, implying that representation learning is still more important for semantic matching.

#### Indirect Inference

The plurality of ambiguous pronouns such as you depends on the context. Two indirect links of (he, you) and (you, they) can be common to induce incorrect clusters that contain both singular and plural pronouns (Wiseman et al., 2016; Lee et al., 2018). Table 3 shows the numbers of these erroneous clusters in prediction. Surprisingly, very few of these clusters contain ambiguous pronouns in either approach. This observation moderates the long-standing movitation of HOI.

Additionally, the change of representation from BERT to SpanBERT has far more impact that reduces 10% of these erroneous clusters, while the four HOI methods fail to show significant difference compared to SpanBERT.

## 5 Conclusion

We implement the end-to-end coreference resolution model and investigate four higher-order inference methods, including two of our own methods.
Our best model shows the new result of 80.2 on the CoNLL 2012 dataset.
We thoroughly analyze the empirical effectiveness of HOI and demonstrate why it fails to boost performance on the CoNLL 2012 dataset compared to the improvement from encoders. We show that current HOI does not meet up with the original motivation, suggesting that a new perspective of HOI is needed for this task in the era of deep learning-based NLP.

## Acknowledgments

We gratefully acknowledge the support of the AWS Machine Learning Research Awards (MLRA).
Any contents in this material are those of the authors and do not necessarily reflect the views of AWS.

## References

- Clark and Manning (2015)

Kevin Clark and Christopher D. Manning. 2015.

Entity-centric
coreference resolution with model stacking.

In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 1405–1415,
Beijing, China. Association for Computational Linguistics.

- Clark and Manning (2016)

Kevin Clark and Christopher D. Manning. 2016.

Improving coreference
resolution by learning entity-level distributed representations.

In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 643–653,
Berlin, Germany. Association for Computational Linguistics.

- Devlin et al. (2019)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

BERT: Pre-training of
deep bidirectional transformers for language understanding.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers), pages 4171–4186,
Minneapolis, Minnesota. Association for Computational Linguistics.

- Fei et al. (2019)

Hongliang Fei, Xu Li, Dingcheng Li, and Ping Li. 2019.

End-to-end deep
reinforcement learning based coreference resolution.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 660–665, Florence, Italy. Association
for Computational Linguistics.

- Joshi et al. (2020)

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and
Omer Levy. 2020.

Spanbert: Improving
pre-training by representing and predicting spans.

Transactions of the Association for Computational Linguistics,
8:64–77.

- Joshi et al. (2019)

Mandar Joshi, Omer Levy, Luke Zettlemoyer, and Daniel Weld. 2019.

BERT for coreference
resolution: Baselines and analysis.

In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), pages 5803–5808, Hong Kong,
China. Association for Computational Linguistics.

- Kantor and Globerson (2019)

Ben Kantor and Amir Globerson. 2019.

Coreference resolution
with entity equalization.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 673–677, Florence, Italy. Association
for Computational Linguistics.

- Lee et al. (2017)

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017.

End-to-end neural
coreference resolution.

In Proceedings of the 2017 Conference on Empirical Methods in
Natural Language Processing, pages 188–197, Copenhagen, Denmark.
Association for Computational Linguistics.

- Lee et al. (2018)

Kenton Lee, Luheng He, and Luke Zettlemoyer. 2018.

Higher-order
coreference resolution with coarse-to-fine inference.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 2 (Short Papers), pages 687–692, New Orleans,
Louisiana. Association for Computational Linguistics.

- Ng and Cardie (2002)

Vincent Ng and Claire Cardie. 2002.

Improving machine
learning approaches to coreference resolution.

In Proceedings of the 40th Annual Meeting of the Association
for Computational Linguistics, pages 104–111, Philadelphia, Pennsylvania,
USA. Association for Computational Linguistics.

- Peters et al. (2018)

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer. 2018.

Deep contextualized
word representations.

In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans,
Louisiana. Association for Computational Linguistics.

- Pradhan et al. (2012)

Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen
Zhang. 2012.

CoNLL-2012
shared task: Modeling multilingual unrestricted coreference in
OntoNotes.

In Joint Conference on EMNLP and CoNLL - Shared Task,
pages 1–40, Jeju Island, Korea. Association for Computational Linguistics.

- Wiseman et al. (2015)

Sam Wiseman, Alexander M. Rush, Stuart Shieber, and Jason Weston. 2015.

Learning anaphoricity
and antecedent ranking features for coreference resolution.

In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 1416–1426,
Beijing, China. Association for Computational Linguistics.

- Wiseman et al. (2016)

Sam Wiseman, Alexander M. Rush, and Stuart M. Shieber. 2016.

Learning global
features for coreference resolution.

In Proceedings of the 2016 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 994–1004, San Diego, California. Association for
Computational Linguistics.

- Wu et al. (2020)

Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, and Jiwei Li. 2020.

CorefQA:
Coreference resolution as query-based span prediction.

In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics, pages 6953–6963, Online. Association for
Computational Linguistics.

- Yu et al. (2020)

Juntao Yu, Alexandra Uma, and Massimo Poesio. 2020.

A cluster
ranking model for full anaphora resolution.

In Proceedings of The 12th Language Resources and Evaluation
Conference, pages 11–20, Marseille, France. European Language Resources
Association.

MUC

B3

CEAFϕ4subscriptitalic-ϕ4\phi_{4}

F1

F1

F1

Avg. F1

BERT

83.7 (±plus-or-minus\pm 0.1)

75.5 (±plus-or-minus\pm 0.1)

72.6 (±plus-or-minus\pm 0.1)

77.3 (±plus-or-minus\pm 0.1)

SpanBERT

85.3 (±plus-or-minus\pm 0.1)

78.4 (±plus-or-minus\pm 0.1)

75.5 (±plus-or-minus\pm 0.3)

79.7 (±plus-or-minus\pm 0.1)

+ AA

85.2 (±plus-or-minus\pm 0.2)

78.1 (±plus-or-minus\pm 0.2)

75.0 (±plus-or-minus\pm 0.2)

79.4 (±plus-or-minus\pm 0.2)

+ EE

85.0 (±plus-or-minus\pm 0.1)

77.7 (±plus-or-minus\pm 0.2)

74.7 (±plus-or-minus\pm 0.2)

78.9 (±plus-or-minus\pm 0.4)

+ SC

85.1 (±plus-or-minus\pm 0.2)

77.9 (±plus-or-minus\pm 0.3)

74.7 (±plus-or-minus\pm 0.3)

79.2 (±plus-or-minus\pm 0.3)

+ CM

85.5 (±plus-or-minus\pm 0.2)

78.5 (±plus-or-minus\pm 0.3)

75.6 (±plus-or-minus\pm 0.2)

79.9 (±plus-or-minus\pm 0.2)

## Appendix A Appendices

### A.1 Experimental Settings

We implement the experimented models using PyTorch. BERTLarge and SpanBERTLarge are used as encoders. For each experiment, the best performed model on the development set is selected and evaluated on the test set.

#### Hyperparameters and Implementation

Similar to Joshi et al. (2019, 2020), documents are split into independent segments with maximum 384 word pieces for BERTLarge and 512 for SpanBERTLarge. In our final setting, BERT-parameters and task-parameters have separate learning rates (1×10−51superscript1051\times 10^{-5} and 3×10−43superscript1043\times 10^{-4} respectively), separate linear decay schedule, and separate weight decay rates (10−2superscript10210^{-2} and 00 respectively). Models are trained 24 epochs with dropout rate 0.3.

The implementation of EE is based on the Tensorflow implementation from Kantor and Globerson (2019) which requires 𝒪​(k2)𝒪superscript𝑘2\mathcal{O}(k^{2}) memory with k𝑘k being the number of extracted spans, while other HOI approaches only requires 𝒪​(k)𝒪𝑘\mathcal{O}(k) memory 444The maximum number of antecedents for all models is set to 50 which is constant.. To keep the GPU memory usage within 32GB, we limit the maximum number of span candidates for EE to be 300, which may have a negative impact on the performance.

Experiments are conducted on Nvidia Tesla V100 GPUs with 32GB memory. The average training time is around 7 hours for BERT and SpanBERT without HOI, and ranges from 9 - 15 hours with HOI methods.

### A.2 Results

Table 4 reports the macro-average F1 scores out of 5 repeated developments of each approach. CM still has the best performance with 79.9 averaged F1 score. Span refinement-based HOI approaches, AA, EE, and SC, still have lower F1 scores than the local-only SpanBERT.

We do not find different configurations for CM make any huge impact to the performance. The final configuration for CM is sequential order and max reduction (Algorithm 1).

### A.3 Analysis

AA
-0.02 (±plus-or-minus\pm 0.06)

EE
0.03 (±plus-or-minus\pm 0.07)

SC
0.11 (±plus-or-minus\pm 0.10)

CM
0.04 (±plus-or-minus\pm 0.04)

Table 5 shows the averaged performance drop and its standard deviations w.r.t Avg-F1 after turning off the corresponding HOI in trained models, to see the direct performance impact of HOI over local decisions.

#### Pronoun Resolution

In our analysis, the following personal pronouns are regarded as ambiguous pronouns: “you”, “your”, “yours”.

Generated on Sat Mar 2 10:03:59 2024 by LaTeXML
