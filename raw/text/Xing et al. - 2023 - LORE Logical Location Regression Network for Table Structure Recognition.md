# Xing et al. - 2023 - LORE Logical Location Regression Network for Table Structure Recognition

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Xing et al. - 2023 - LORE Logical Location Regression Network for Table Structure Recognition.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2303.03730
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# LORE: Logical Location Regression Network for Table Structure Recognition

Hangdi Xing\equalcontrib1,
Feiyu Gao\equalcontrib3,
Rujiao Long3,
Jiajun Bu1,
Qi Zheng3,
Liangcheng Li1,
Cong Yao3,
Zhi Yu2
Corresponding Author.

###### Abstract

Table structure recognition (TSR) aims at extracting tables in images into machine-understandable formats. Recent methods solve this problem by predicting the adjacency relations of detected cell boxes, or learning to generate the corresponding markup sequences from the table images. However, they either count on additional heuristic rules to recover the table structures, or require a huge amount of training data and time-consuming sequential decoders. In this paper, we propose an alternative paradigm. We model TSR as a logical location regression problem and propose a new TSR framework called LORE, standing for LOgical location REgression network, which for the first time combines logical location regression together with spatial location regression of table cells. Our proposed LORE is conceptually simpler, easier to train and more accurate than previous TSR models of other paradigms. Experiments on standard benchmarks demonstrate that LORE consistently outperforms prior arts. Code is available at https://
github.com/AlibabaResearch/AdvancedLiterateMachinery/
tree/main/DocumentUnderstanding/LORE-TSR.

## Introduction

Data in tabular format is prevalent in various sorts of documents for summarizing and presenting information. As the world is going digital, the need for parsing the tables trapped in unstructured data (e.g., images and PDF files) is growing rapidly. Although straightforward for humans, it is challenging for automated systems due to the wide diversity of layouts and styles of tables. Table Structure Recognition (TSR) refers to transforming tables in images to machine-understandable formats, usually in logical coordinates or markup sequences. The extracted table structures are crucial for information retrieval, table-to-text generation and question answering systems, etc.

With the development of deep learning, TSR methods have recently advanced substantially. Most deep learning-based TSR methods can be categorized into the following paradigms. The first type of models (Chi et al. 2019; Raja, Mondal, and Jawahar 2020; Liu et al. 2022) aim at exploring the adjacency relationships between pairs of detected cells to generate intermediate results. They rely on tedious post-processings or graph optimization algorithms to reconstruct the table as logical coordinates, as depicted in Figure 1 (a), which would struggle with complex table structures. Another paradigm formulates TSR as a markup language sequence generation problem (Zhong, ShafieiBavani, and Jimeno Yepes 2020; Desai, Kayal, and Singh 2021), as shown in Figure 1 (b). It simplifies the TSR pipelines, but the models are supposed to redundantly learn the markup grammar from noisy sequence labels, which results in a much larger amount of training data. Besides, these models are time-consuming due to the sequential decoding process.

(a) Adjacency relationship representations

(b) Markup sequence representations

(c) Logical location representations

In fact, logical coordinates are well-defined machine-understandable representations of table structures, which are complete to reconstruct tables, as depicted in Figure 1 (c). Recently, work arises which focuses on exploring the logical locations of table cells (Xue et al. 2021). However, the method predicts logical locations by ordinal classification and does not account for the natural dependencies between logical locations. For example, the design of a table itself is from top to bottom, left to right, causing the logical location of cells to be interdependent. This nature of logical locations is sketched in Figure 2. Furthermore, the work lacks a comprehensive comparison among various TSR paradigms.

Aiming at breaking the limitations of existing methods, we propose LOgical Location REgression Network (LORE for abbreviation), a conceptually simpler and more effective TSR framework. It first locates table cells on the input image, and then predicts the logical locations along with the spatial locations of cells. To better model the dependencies and constraints between logical locations, a cascade regression framework is adopted, combined with the inter-cell and intra-cell supervisions. The inference of LORE is a parallel network forward-pass, without any efforts in complicated post-processings or sequential decoding strategies.

We evaluate LORE on a wide range of benchmarks against TSR methods of different paradigms. Experiments show that LORE is highly competitive and outperforms previous state-of-the-art methods. Specifically, LORE surpasses other logical location prediction methods by a large margin. Moreover, the adjacency relations and markup sequences derived from the prediction of LORE are of higher quality, which demonstrates that LORE covers the capacity of the models trained under other TSR paradigms.

Our main contributions can be summarized as follows:

- •

We propose to model TSR as the logical location regression problem and design LORE, a new TSR framework which captures dependencies and constraints between logical locations of cells, and predicts the logical locations along with the spatial locations.

- •

We empirically demonstrate that the logical location regression paradigm is highly effective and covers the abilities of previous TSR paradigms, such as predicting adjacency relations and generating markup sequences.

- •

LORE provides a hands-off way to apply an effective TSR model, by removing the effort for designing post-processings and decoding strategies. The code is available to support further investigations on TSR.

## Related Work

Early works (Schreiber et al. 2017; Siddiqui et al. 2019) introduce segmentation or detection frameworks to locate and extract splitting lines of table rows and columns. Subsequently, they reconstruct the table structure by empirically grouping the cell boxes with pre-defined rules. These models would suffer from tables with spanning cells or distortions. The latest baselines (Long et al. 2021; Smock, Pesala, and Abraham 2022; Zhang et al. 2022) tackle this problem by well-designed detectors or attention-based merging modules to obtain more accurate cell boundaries and merging results. However, they either are tailored for the certain type of datasets or require customized processings to recover table structures, and thus can hardly be generalized. So there arise models focusing on directly predicting the table structures with neural networks.

### TSR as Cell Adjacency Exploring

Chi et al. (2019) proposes to model table cells as text segmentation regions and exploit the relationships between cell pairs. Precisely, it applies graph neural networks (Kipf and Welling 2017) to classify pairs of detected cells into horizontal, vertical and unrelated relations. Following this work, there are models devoted to improving the relationship classification by using elaborated neural networks and adding multi-modal features (Qasim, Mahmood, and Shafait 2019; Raja, Mondal, and Jawahar 2020, 2022; Liu et al. 2021, 2022). However, there is still a gap between the set of relation triplets and the global table structure. Complex graph optimization algorithms or pre-defined post-processings are needed to recover the tables.

### TSR as Markup Sequence Generation

Li et al. (2020); Zhong, ShafieiBavani, and Jimeno Yepes (2020); Ye et al. (2021) make the pioneering attempts to solve the TSR problem in an end-to-end way. They employ sequence decoders to generate tags of markup language that represent table structures. However, the models are supposed to learn the markup grammar with noisy labels, resulting in the methods being difficult to train and requiring a much larger number of training samples than other paradigms. Besides, these models are time-consuming owing to the sequential decoding process.

### TSR as Logical Location Prediction

Xue et al. (2021) propose to perform ordinal classification of logical indices on each detected cell for TSR, which is close to our approach. The model utilizes graph neural networks to classify detected cells into the corresponding logical locations, while it ignores the dependencies and constraints among logical locations of cells. Besides, the model is only evaluated on a few datasets and not against the strong TSR baselines.

## Problem Definition

In this paper, we consider the TSR problem as the spatial and logical location regression task. Specifically, for an input image of the table, similar to a detector, a set of table cells {O1,O2,…,ON}subscript𝑂1subscript𝑂2…subscript𝑂𝑁\{O_{1},O_{2},...,O_{N}\} are predicted as their logical locations {l1,l2,…,lN}subscript𝑙1subscript𝑙2…subscript𝑙𝑁\{l_{1},l_{2},...,l_{N}\}, along with the spatial locations {B1,B2,…,BN}subscript𝐵1subscript𝐵2…subscript𝐵𝑁\{B_{1},B_{2},...,B_{N}\}, where li=(rs(i),re(i),cs(i),ce(i))subscript𝑙𝑖superscriptsubscript𝑟𝑠𝑖superscriptsubscript𝑟𝑒𝑖superscriptsubscript𝑐𝑠𝑖superscriptsubscript𝑐𝑒𝑖l_{i}=(r_{s}^{(i)},r_{e}^{(i)},c_{s}^{(i)},c_{e}^{(i)}) standing for the starting-row, ending-row, starting-column and ending-column, Bi={(xk(i),yk(i))}k=1,2,3,4subscript𝐵𝑖subscriptsuperscriptsubscript𝑥𝑘𝑖superscriptsubscript𝑦𝑘𝑖𝑘1234B_{i}=\{(x_{k}^{(i)},y_{k}^{(i)})\}_{k=1,2,3,4} standing for the four corner points of the i𝑖i-th cell and N𝑁N is the number of cells in the image.

With the predicted table cells represented by their spatial and logical locations, the table in the image can be converted into machine-understandable formats, such as relational databases. Besides, the adjacency matrices and the markup sequences of tables can be directly derived from their logical coordinates with well-defined transformations rather than heuristic rules (See supplementary section 1).

## Methodology

This section elaborates on our proposed LORE, a TSR framework regressing the spatial and logical locations of cells. As illustrated in Figure 3, it employs a CNN backbone to extract visual features of table cells from the input image. Then the spatial and logical locations of cells are predicted by two regression heads. We specially leverage the cascading regressors and employ inter-cell and intra-cell supervisions to model the dependencies and constraints between logical locations. The following subsections specify these crucial components respectively.

### Table Cell Features Preparation

In order to streamline the joint prediction of spatial and logical locations, we employ a key point segmentation network (Zhou, Wang, and Krähenbühl 2019; Long et al. 2021) as the feature extractor and model each table cell in the image as its center point.

For an input image of width W𝑊W and height H𝐻H, the network produces a feature map f∈ℝWR×HR×d𝑓superscriptℝ𝑊𝑅𝐻𝑅𝑑f\in\mathbb{R}^{\frac{W}{R}\times\frac{H}{R}\times d} and a cell center heatmap Y^∈[0,1]WR×HR^𝑌superscript01𝑊𝑅𝐻𝑅\widehat{Y}\in[0,1]^{\frac{W}{R}\times\frac{H}{R}}, where R𝑅R, d𝑑d are the output stride and hidden size; Y^x,y=1subscript^𝑌𝑥𝑦1\widehat{Y}_{x,y}=1 corresponds to a detected cell center, while Y^x,y=0subscript^𝑌𝑥𝑦0\widehat{Y}_{x,y}=0 refers to the background.

In the subsequent modules, the CNN features {f(1),f(2),…,f(N)}superscript𝑓1superscript𝑓2…superscript𝑓𝑁\{f^{(1)},f^{(2)},...,f^{(N)}\} at detected cell centers {p^(1),p^(2),…,p^(N)}superscript^𝑝1superscript^𝑝2…superscript^𝑝𝑁\{\hat{p}^{(1)},\hat{p}^{(2)},...,\hat{p}^{(N)}\} are considered as the representations of table cells.

### Spatial Location Regression

We choose to predict the four corner points rather than the rectangle bounding box to better deal with the inclines and distortions of tables in the wild. For spatial locations, the features of the backbone f𝑓f are passed through a 3×3333\times 3 convolution, ReLU and another 1×1111\times 1 convolution to get the prediction {B^(1),B^(2),…,B^(N)}superscript^𝐵1superscript^𝐵2…superscript^𝐵𝑁\{\hat{B}^{(1)},\hat{B}^{(2)},...,\hat{B}^{(N)}\} on centers {p^(1),p^(2),…,p^(N)}superscript^𝑝1superscript^𝑝2…superscript^𝑝𝑁\{\hat{p}^{(1)},\hat{p}^{(2)},...,\hat{p}^{(N)}\}, where B^(i)={(x^k(i),y^k(i))}k=1,2,3,4superscript^𝐵𝑖subscriptsuperscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖𝑘1234\hat{B}^{(i)}=\{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)})\}_{k=1,2,3,4}.

### Logical Location Regression

As dense dependencies and constraints exist between the logical locations of table cells, it is rather challenging to learn the logical coordinates from the visual features of cell centers alone. The cascading regressors with inter-cell and intra-cell supervisions are leveraged to explicitly model the logical relations between cells.

#### Base Regressor

To better model the logical relations from images, the visual features are first combined with the spatial information. Specifically, the features of the predicted corner points of the cells are computed as the sum of their visual features and 2-dimensional position embeddings:

f~(x^k(i),y^k(i),:)=f(x^k(i),y^k(i),:)+P​E​(x^k(i),y^k(i)),subscript~𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:subscript𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:𝑃𝐸superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖\widetilde{f}_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)}=f_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)}+PE(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)}),

(1)

where P​E𝑃𝐸PE refers to the 2-dimensional position embedding function (Xu et al. 2020, 2021). Then the features of the four corner points are added to the center features f(i)superscript𝑓𝑖f^{(i)} to enhance the representation of each predicted cell center p^(i)superscript^𝑝𝑖\hat{p}^{(i)} as:

h(i)=f(i)+∑k=14wk​f~(x^k(i),y^k(i),:),superscriptℎ𝑖superscript𝑓𝑖superscriptsubscript𝑘14subscript𝑤𝑘subscript~𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:h^{(i)}=f^{(i)}+\sum_{k=1}^{4}w_{k}\widetilde{f}_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)},

(2)

where [w1,w2,w3,w4]subscript𝑤1subscript𝑤2subscript𝑤3subscript𝑤4[w_{1},w_{2},w_{3},w_{4}] are learnable parameters.

Then the message-passing and aggregating networks are adopted to incorporate the interaction between the visual-spatial features of cells:

{h~(i)}i=1,2,…,N=SelfAttention​({h(i)}i=1,2,…,N).subscriptsuperscript~ℎ𝑖𝑖12…𝑁SelfAttentionsubscriptsuperscriptℎ𝑖𝑖12…𝑁\{\widetilde{h}^{(i)}\}_{i=1,2,...,N}={\rm\textbf{Self\-Attention}}(\{h^{(i)}\}_{i=1,2,...,N}).

(3)

We use the self-attention mechanism (Vaswani et al. 2017) in LORE to avoid making additional assumptions about the distribution of table structure, rather than graph neural networks employed by previous methods (Qasim, Mahmood, and Shafait 2019; Xue et al. 2021), which will be further discussed in experiments.

The prediction of the base regressor is then computed by a linear layer with the ReLU activation from {h~(i)}i=1,2,…,Nsubscriptsuperscript~ℎ𝑖𝑖12…𝑁\{\widetilde{h}^{(i)}\}_{i=1,2,...,N} as l^(i)=(r^s(i),r^e(i),c^s(i),c^e(i))superscript^𝑙𝑖superscriptsubscript^𝑟𝑠𝑖superscriptsubscript^𝑟𝑒𝑖superscriptsubscript^𝑐𝑠𝑖superscriptsubscript^𝑐𝑒𝑖\hat{l}^{(i)}=(\hat{r}_{s}^{(i)},\hat{r}_{e}^{(i)},\hat{c}_{s}^{(i)},\hat{c}_{e}^{(i)}).

#### Stacking Regressor

Although the base regressor encodes the relationships between visual-spatial features of cells, the logical locations of each cell are still predicted individually. To better capture the dependencies and constraints among logical locations, a stacking regressor is employed to look again at the prediction of the base regressor. Specifically, the enhanced features 𝒉~bold-~𝒉\boldsymbol{\widetilde{h}} and the logical location prediction of the base regressor 𝒍^^𝒍\hat{\boldsymbol{l}} are fed into a stacking regressor. The stacking regressor can be expressed as :

𝒍~=Fs​(Ws​𝒍^+𝒉~).~𝒍subscript𝐹𝑠subscript𝑊𝑠^𝒍bold-~𝒉\widetilde{\boldsymbol{l}}=F_{s}(W_{s}\hat{\boldsymbol{l}}+{\boldsymbol{\widetilde{h}}}).

(4)

where Ws∈ℝ4×dsubscript𝑊𝑠superscriptℝ4𝑑W_{s}\in\mathbb{R}^{4\times d} is a learnable parameter, 𝒍^=[l^(1),…,l^(N)]^𝒍superscript^𝑙1…superscript^𝑙𝑁\hat{\boldsymbol{l}}=[\hat{l}^{(1)},...,\hat{l}^{(N)}], 𝒉~=[h~(1),…,h~(N)]bold-~𝒉superscript~ℎ1…superscript~ℎ𝑁{\boldsymbol{\widetilde{h}}}=[\widetilde{h}^{(1)},...,\widetilde{h}^{(N)}] and Fssubscript𝐹𝑠F_{s} denotes the stacking regression function, which has the same self-attention and linear structure as the base regression function but with independent parameters. The output of the stacking regressor is 𝒍~=[l~(1),…,l~(N)]~𝒍superscript~𝑙1…superscript~𝑙𝑁\widetilde{{\boldsymbol{l}}}=[\widetilde{l}^{(1)},...,\widetilde{l}^{(N)}], and l~(i)=(r~s(i),r~e(i),c~s(i),c~e(i))superscript~𝑙𝑖superscriptsubscript~𝑟𝑠𝑖superscriptsubscript~𝑟𝑒𝑖superscriptsubscript~𝑐𝑠𝑖superscriptsubscript~𝑐𝑒𝑖\widetilde{l}^{(i)}=(\widetilde{r}_{s}^{(i)},\widetilde{r}_{e}^{(i)},\widetilde{c}_{s}^{(i)},\widetilde{c}_{e}^{(i)}).

At the inference stage, the results are obtained by assigning the four components of l~(i)superscript~𝑙𝑖\widetilde{l}^{(i)} to the nearest integers.

#### Inter-cell and Intra-cell Supervisions

In order to equip the logical location regressor with a better understanding of the dependencies and constraints between logical locations, we propose the inter-cell and intra-cell supervisions, which are summarized as: 1) The logical locations of different cells should be mutually exclusive (inter-cell). 2) The logical locations of one table cell should be consistent with its spans (intra-cell).

In practice, predictions of cells that are far apart rarely contradict each other, so we only sample adjacent pairs for inter-cell supervision. More formally, the scheme of inter-cell and intra-cell losses can be expressed as:

Li​n​t​e​rsubscript𝐿𝑖𝑛𝑡𝑒𝑟\displaystyle L_{inter}
=∑(i,j)∈Arm​a​x​(r~e(j)−r~s(i)+1,0)absentsubscript𝑖𝑗subscript𝐴𝑟𝑚𝑎𝑥superscriptsubscript~𝑟𝑒𝑗superscriptsubscript~𝑟𝑠𝑖10\displaystyle=\sum_{(i,j)\in A_{r}}max(\widetilde{r}_{e}^{(j)}-\widetilde{r}_{s}^{(i)}+1,0)

(5)

+∑(i,j)∈Acm​a​x​(c~e(j)−c~s(i)+1,0),subscript𝑖𝑗subscript𝐴𝑐𝑚𝑎𝑥superscriptsubscript~𝑐𝑒𝑗superscriptsubscript~𝑐𝑠𝑖10\displaystyle+\sum_{(i,j)\in A_{c}}max(\widetilde{c}_{e}^{(j)}-\widetilde{c}_{s}^{(i)}+1,0),

where Arsubscript𝐴𝑟A_{r} (Acsubscript𝐴𝑐A_{c}) are sets of ordered horizontally (vertically) adjacent pairs, i.e., for a pair of cells (i,j)∈Ar𝑖𝑗subscript𝐴𝑟(i,j)\in A_{r} (Acsubscript𝐴𝑐A_{c}), cell i𝑖i is adjacent to cell j𝑗j in the same row (column) and on the right of (under) cell j𝑗j, and r~s(i)superscriptsubscript~𝑟𝑠𝑖\widetilde{r}_{s}^{(i)}, r~e(j)superscriptsubscript~𝑟𝑒𝑗\widetilde{r}_{e}^{(j)}, c~s(i)superscriptsubscript~𝑐𝑠𝑖\widetilde{c}_{s}^{(i)}, c~e(j)superscriptsubscript~𝑐𝑒𝑗\widetilde{c}_{e}^{(j)} are predicted logical indices of cell i𝑖i and cell j𝑗j.

Li​n​t​r​asubscript𝐿𝑖𝑛𝑡𝑟𝑎\displaystyle L_{intra}
=∑i∈Mr|r~s(i)−r~e(i)−rs(i)+re(i)|absentsubscript𝑖subscript𝑀𝑟superscriptsubscript~𝑟𝑠𝑖superscriptsubscript~𝑟𝑒𝑖superscriptsubscript𝑟𝑠𝑖superscriptsubscript𝑟𝑒𝑖\displaystyle=\sum_{i\in M_{r}}|\widetilde{r}_{s}^{(i)}-\widetilde{r}_{e}^{(i)}-r_{s}^{(i)}+r_{e}^{(i)}|

(6)

+∑i∈Mc|c~s(i)−c~e(i)−cs(i)+ce(i)|,subscript𝑖subscript𝑀𝑐superscriptsubscript~𝑐𝑠𝑖superscriptsubscript~𝑐𝑒𝑖superscriptsubscript𝑐𝑠𝑖superscriptsubscript𝑐𝑒𝑖\displaystyle+\sum_{i\in M_{c}}|\widetilde{c}_{s}^{(i)}-\widetilde{c}_{e}^{(i)}-c_{s}^{(i)}+c_{e}^{(i)}|,

where Mr={i|re(i)−rs(i)≠0}subscript𝑀𝑟conditional-set𝑖superscriptsubscript𝑟𝑒𝑖superscriptsubscript𝑟𝑠𝑖0M_{r}=\{i|r_{e}^{(i)}-r_{s}^{(i)}\neq 0\} and Mc={i|ce(i)−cs(i)≠0}subscript𝑀𝑐conditional-set𝑖superscriptsubscript𝑐𝑒𝑖superscriptsubscript𝑐𝑠𝑖0M_{c}=\{i|c_{e}^{(i)}-c_{s}^{(i)}\neq 0\} are sets of multi-row and multi-column cells.

Then the inter-cell and intra-cell losses (I2C) are as:

LI​2​C=Li​n​t​e​r+Li​n​t​r​a.subscript𝐿𝐼2𝐶subscript𝐿𝑖𝑛𝑡𝑒𝑟subscript𝐿𝑖𝑛𝑡𝑟𝑎L_{I2C}=L_{inter}+L_{intra}.

The supervisions are conducted on the output 𝒍~~𝒍\widetilde{{\boldsymbol{l}}} and no extra forward-passing is required.

### Objectives

The losses of cell center segmentation Lc​e​n​t​e​rsubscript𝐿𝑐𝑒𝑛𝑡𝑒𝑟L_{center} and spatial location regression Ls​p​asubscript𝐿𝑠𝑝𝑎L_{spa} are computed following typical key point-based detection methods (Zhou, Wang, and Krähenbühl 2019; Long et al. 2021).

The loss of logical locations is computed for both the base regressor and the stacking regressor:

Ll​o​g=1N​∑i=1N(‖l^(i)−li‖1+‖l~(i)−li‖1).subscript𝐿𝑙𝑜𝑔1𝑁superscriptsubscript𝑖1𝑁subscriptnormsuperscript^𝑙𝑖subscript𝑙𝑖1subscriptnormsuperscript~𝑙𝑖subscript𝑙𝑖1L_{log}=\frac{1}{N}\sum_{i=1}^{N}(||\hat{l}^{(i)}-l_{i}||_{1}+||\widetilde{l}^{(i)}-l_{i}||_{1}).

(7)

The total loss of joint training is then computed by adding the losses of cell center segmentation, spatial and logical location regression along with the I2C supervisions:

LL​O​R​E=Lc​e​n​t​e​r+Ls​p​a+Ll​o​g+LI​2​C.subscript𝐿𝐿𝑂𝑅𝐸subscript𝐿𝑐𝑒𝑛𝑡𝑒𝑟subscript𝐿𝑠𝑝𝑎subscript𝐿𝑙𝑜𝑔subscript𝐿𝐼2𝐶L_{LORE}=L_{center}+L_{spa}+L_{log}+L_{I2C}.

(8)

Datasets
ICDAR-13
ICDAR-19
WTW
TG24K

metric
F-1
Acc
F-1
Acc
F-1
Acc
F-1
Acc

ReS2TIM
-
17.4
-
13.8
-
-
-
-

TGRNet
66.7
27.5
82.8
26.7
64.7
24.3
92.5
84.5

Ours
97.2
86.8
90.6
73.2
96.4
82.9
96.1
87.9

Datasets
ICDAR-13
SciTSR-comp
ICDAR-19
WTW

metric
P
R
F-1
P
R
F-1
P
R
F-1
P
R
F-1

TabStrNet
93.0
90.8
91.9
90.9
88.2
89.5
82.2
78.7
80.4
-
-
-

LGPMA
96.7
99.1
97.9
97.3
98.7
98.0
-
-
-
-
-
-

TOD
98.0
97.0
98.0
97.0
99.0
98.0
77.0
76.0
77.0
-
-
-

FLAGNet
97.9
99.3
98.6
98.4
98.6
98.5
85.2
83.8
84.5
91.6
89.5
90.5

NCGM
98.4
99.3
98.8
98.7
98.9
98.8
84.6
86.1
85.3
93.7
94.6
94.1

Ours
99.2
98.6
98.9
99.4
99.2
99.3
87.9
88.7
88.3
94.5
95.9
95.1

Datasets
PubTabNet
TableBank

metric
TEDS
TEDS
BLEU

Image2Text
-
-
73.8

EDD
89.9
86.0
-

Ours
98.1
92.3
91.1

## Experiments

In this section, we conduct comprehensive experiments to research and answer two key questions: 1) Is the proposed LORE able to effectively predict the logical locations of table cells from input images? 2) Does the LORE framework, modeling TSR as logical location regression, overcome the limitations and cover the abilities of other paradigms?

For the first question, we compare LORE with baselines directly predicting logical locations (Xue, Li, and Tao 2019; Xue et al. 2021). To the best of our knowledge, these are the only two methods that focus on directly predicting the logical locations. Furthermore, we provide a detailed ablation study to validate the effectiveness of the main components. For the second question, we compare LORE with methods that model table structure as cell adjacency or markup sequence with both insights and quantitative results.

### Datasets

We evaluate LORE on a wide range of benchmarks, including tables in digital-born documents, i.e., ICDAR-2013 (Göbel et al. 2013), SciTSR-comp (Chi et al. 2019), PubTabNet (Zhong, ShafieiBavani, and Jimeno Yepes 2020), TableBank (Li et al. 2020) and TableGraph-24K (Xue et al. 2021), as well as tables from scanned documents and photos, i.e., ICDAR-2019 (Gao et al. 2019) and WTW (Long et al. 2021). Details of datasets are available in section 2 of the supplementary. It should be noted that ICDAR-2013 provides no training data, so we extend it to the partial version for cross validation following previous works (Raja, Mondal, and Jawahar 2020; Liu et al. 2022, 2021). And when training LORE on the PubTabNet, we randomly choose 20,000 images from its training set for efficiency.

### Evaluation Metric

The TSR models of different paradigms are evaluated using different metrics, including 1) accuracy of logical locations (Xue, Li, and Tao 2019), 2) F-1 score of adjacency relationships between cells (Göbel et al. 2012, 2013), and 3) BLEU and TEDS (Papineni et al. 2002; Zhong, ShafieiBavani, and Jimeno Yepes 2020). We provide a detailed introduction of these metrics in section 3 of the supplementary. The accuracy of logical locations, BLEU and TEDS directly reflect the correctness of the predicted structure, while the adjacency evaluation only measures the quality of intermediate results of the structure. In our experiments, LORE is evaluated under all three types of metrics, since the logical coordinates are complete for representing table structures and can be converted into adjacency matrices and markup sequences by simple and clarified transformations (see section 1 of the supplementary material). When evaluating on TEDS, we use the non-styling text extracted from PDF files following Zheng et al. (2021). We also report the performance of cell spatial location prediction, using the F-1 score under the IoU threshold of 0.5, following recent works (Raja, Mondal, and Jawahar 2020; Xue et al. 2021).

N
Objectives
Cascade
Architecture
Metrics

L1subscript𝐿1L_{1}
Inter
Intra
Encoder
Base
Stacking
A-c
A-r
Acc

1a
✓
-
-
✓
Attention
3
3
87.2
84.8
79.4

1b
✓
✓
-
✓
Attention
3
3
87.6
86.6
80.2

1c
✓
-
✓
✓
Attention
3
3
89.5
87.1
81.2

1d
✓
✓
✓
✓
Attention
3
3
91.3
87.9
82.9

2a
✓
✓
✓
✓
GNN
3
3
88.2
82.6
77.0

2b
✓
✓
✓
-
Attention
6
0
88.7
85.3
79.8

### Implementation

LORE is trained and evaluated on table images with the max side scaled to a fixed size of 102410241024 (512512512 for SciTSR and PubTabNet) and the short side resized equally. The model is trained for 100 epochs, and the initial learning rate is chosen as 1×10−41superscript1041\times 10^{-4}, decaying to 1×10−51superscript1051\times 10^{-5} and 1×10−61superscript1061\times 10^{-6} at the 70th and 90th epochs for all benchmarks. All the experiments are performed on the platform with 4 NVIDIA Tesla V100 GPUs. We use the DLA-34 (Yu et al. 2018) backbone, the output stride R=4𝑅4R=4 and the number of channels d=256𝑑256d=256. When implementing on the WTW dataset, a corner point estimation is equipped following Long et al. (2021). The number of attention layers is set to 3 for both the base and the stacking regressors. We run the model 5 times and take the average performance.

### Results on Benchmarks

First, we compare LORE with models which directly predict logical locations including Res2TIM (Xue, Li, and Tao 2019) and TGRNet (Xue et al. 2021). We tune the model provided by Xue et al. (2021) on WTW dataset to make a thorough comparison. As shown in Table 1, LORE outperforms the previous methods remarkably. The baseline methods can only produce passable results on relatively simple benchmarks of digital-born table images from scientific articles, i.e., TableGraph-24K.

Then we compare LORE with models mining the adjacency of cells by relation-based metrics: TabStrNet (Raja, Mondal, and Jawahar 2020), LGPMA (Qiao et al. 2021), TOD (Raja, Mondal, and Jawahar 2022), FLAGNet (Liu et al. 2021) and NCGM (Liu et al. 2022). The adjacency relation results of LORE are derived from the output logical locations as mentioned before. The results are shown in Table 2. It is worth noting that LORE performs much better on challenging benchmarks such as ICDAR-2019 and WTW with scanned documents and photos. Tables in these datasets are with more spanning cells and distortions (Liu et al. 2022; Long et al. 2021). Experiments demonstrate that LORE is capable of predicting adjacency relations, as by-products of regressing the logical locations.

Finally, we evaluate LORE on the markup sequence generation scene against Image2Text (Li et al. 2020) and EDD (Zhong, ShafieiBavani, and Jimeno Yepes 2020), with the results also derived from the output logical locations of LORE. Specially, since the TableBank dataset does not provide the spatial locations of cells, we implement LORE trained on SciTSR (1/10 the size of TableBank) for the evaluation on it. The results are shown in Table 3. Experiment results indicate that LORE is also more effective even if LORE is trained on much fewer samples.

### Ablation Study

To investigate how the key components of our proposed LORE contribute to the logical location regression, we conduct an intensive ablation study on the WTW dataset. Results are presented in Table 4. First, we evaluate the effectiveness of the inter-cell loss Li​n​t​e​rsubscript𝐿𝑖𝑛𝑡𝑒𝑟L_{inter} and the intra-cell loss Li​n​t​r​asubscript𝐿𝑖𝑛𝑡𝑟𝑎L_{intra}, by training several models turning them on and off. According to the results in experiments 1a and 1b, we see that the inter-cell supervision improves the performance by +0.8%Acc. And from 1a and 1c, the intra-cell supervision benefits more by +1.8%Acc, for the reason that it makes up the message-passing and aggregating mechanism, which pays less attention to intra-cell relations than inter-cell relations according to its inter-cell nature. The combination of the two supervisions makes the best performance.

Then we evaluate the influence of model architecture, i.e., the pattern of message aggregation and the importance of the cascade framework. In experiment 2a, we replace the self-attention encoder with a graph-attention encoder similar to graph-based TSR models (Qasim, Mahmood, and Shafait 2019; Xue et al. 2021) with an equal amount of parameters with LORE. It causes a drop in performance consistently. The graph-based encoder only aggregates information from the top-K nearest features of each node based on Euclidean distance, which is biased for table structure. In experiment 2b, we use a single regressor of 6 layers instead of two cascading regressors of 3 layers. We can observe a performance degradation of 3.1%Acc from 1d to 2b, showing that the cascade framework can better model the dependencies and constraints between logical locations of different cells.

(a) Original structure

(b) Shifted structure

### Further Comparison among Paradigms

In this section, we further compare models of different TSR paradigms introduced before. Previous methods that predict logical locations lack a comprehensive comparison and analysis between these paradigms. We demonstrate how LORE overcomes the limitations of the adjacency-based and the markup-based methods by controlled experiments.

The adjacency of cells alone is not sufficient to represent table structures. Previous methods employ heuristic rules based on spatial locations (Liu et al. 2022) or graph optimizations (Qasim, Mahmood, and Shafait 2019) to reconstruct the tables. However, it takes tedious modification to make the pre-defined parts compatible with datasets of different types of tables and annotations. Furthermore, the adjacency-based metrics sometimes fail to reflect the correctness of table structures, as depicted in Figure 4. Experiments are conducted to verify this argument quantitatively. We turn the linear layer of the stacking regressor of LORE into an adjacency classification layer of paired cell features and employ post-processings as in NCGM (Liu et al. 2022) to reconstruct the table. The results are in Table 5. Although this modified model (Adj. paradigm) achieves competitive results with state-of-the-art baselines evaluated on adjacency-based metrics, the accuracy of logical locations obtained from heuristic rules decreases obviously compared to LORE (Log. paradigm), especially on WTW, which contains more spanning cells and distortions.

(a) Attention activation of the base regressor

(b) Attention activation of the stacking regressor

(c) Attention activation of the non-cascade regressor

Data
Paradigm
Adj. Metrics
Log. Metrics

P
R
F-1
A-all
A-sp

Sci-c
Adj.
98.6
98.9
98.7
94.7
63.5

Log.
99.4
99.2
99.3
97.3
87.7

WTW
Adj.
95.0
93.7
94.3
51.9
20.2

Log.
94.5
95.9
95.1
82.9
63.8

The markup-sequence-based models leverage image encoders and sequence decoders to predict the label sequences. Since the markup language has plenty of control sequences formatting styles, they can be viewed as noise in labels and impede model training (Xue et al. 2021). It requires much more training samples and computational costs. As shown in Table 6, the number of training samples of the EDD model on the PubTabNet dataset is more than ten times larger than that of LORE. Besides, the inference process is rather time-consuming (See Table 6) due to the sequential decoding pattern, while models of other paradigms compute for each cell in parallel. The average inference time is computed from the validation set of PubTabNet with the images resized to 1280×1280128012801280\times 1280 for both models.

### Further Analysis on Cascade Regressors

We conduct experiments to investigate the effect of the cascade framework on the prediction of logical coordinates. In Figure 5, we visualize the attention maps of the last encoder layer of the cascade/single regressors of two cells, i.e., the models 1d and 2b in Table 4. In the cascade framework, the base regressor in Figure 5 (a) focuses on the heading cells (upper or left) to compute logical locations. While the stacking regressor in Figure 5 (b) pays more attention to the surrounding cells to discover finer dependencies among logical locations and make sure the prediction is subject to natural constraints, which is in line with human intuition when designing a table. However, the non-cascade regressor in Figure 5 (c) can only play a role similar to the base regressor, which leaves out important information for the prediction of logical locations.

#Train Samples
Inference Time

EDD
339000
14.8s

LORE
20000
0.45s

DLA-34
LORE

#Params
15.9
24.2

FLOPs
74.6
75.2

### Computational Analysis

We summarize the model size and the inference operations of LORE in Table 7, with the input images at 1024×1024102410241024\times 1024 and the number of cells as 32. It is observed that the complexity of LORE is at an equal level to a key point-based detector (Zhou, Wang, and Krähenbühl 2019) with the same backbone, showing the efficiency of LORE.

## Conclusions

In summary, we propose LORE, a TSR framework that effectively regresses the spatial locations and the logical locations of table cells from the input images. Furthermore, it models the dependencies and constraints between logical locations by employing the cascading regressors along with the inter-cell and intra-cell supervisions. LORE is straightforward to implement and achieves competitive results, without tedious post-processings or sequential decoding strategies. Experiments show that LORE outperforms state-of-the-art TSR methods under various metrics and overcomes the limitations of previous TSR paradigms.

## Acknowledgements

This work is supported by the National Key R&D Program of China (No. 2018YFC2002603), the National Natural Science Foundation of China (Grant No. 61972349), the Fundamental Research Funds for the Central Universities (No. 226-2022-00064), and Alibaba-Zhejiang University Joint Institute of Frontier Technologies.

## References

- Chi et al. (2019)

Chi, Z.; Huang, H.; Xu, H.-D.; Yu, H.; Yin, W.; and Mao, X.-L. 2019.

Complicated table structure recognition.

arXiv preprint arXiv:1908.04729.

- Desai, Kayal, and Singh (2021)

Desai, H.; Kayal, P.; and Singh, M. 2021.

TabLeX: a benchmark dataset for structure and content information
extraction from scientific tables.

In International Conference on Document Analysis and
Recognition, 554–569. Springer.

- Gao et al. (2019)

Gao, L.; Huang, Y.; Déjean, H.; Meunier, J.-L.; Yan, Q.; Fang, Y.; Kleber,
F.; and Lang, E. 2019.

ICDAR 2019 competition on table detection and recognition (cTDaR).

In 2019 International Conference on Document Analysis and
Recognition (ICDAR), 1510–1515. IEEE.

- Göbel et al. (2012)

Göbel, M.; Hassan, T.; Oro, E.; and Orsi, G. 2012.

A methodology for evaluating algorithms for table understanding in
PDF documents.

In Proceedings of the 2012 ACM symposium on Document
engineering, 45–48.

- Göbel et al. (2013)

Göbel, M.; Hassan, T.; Oro, E.; and Orsi, G. 2013.

ICDAR 2013 table competition.

In 2013 12th International Conference on Document Analysis and
Recognition, 1449–1453. IEEE.

- Kipf and Welling (2017)

Kipf, T. N.; and Welling, M. 2017.

Semi-Supervised Classification with Graph Convolutional Networks.

In International Conference on Learning Representations
(ICLR).

- Li et al. (2020)

Li, M.; Cui, L.; Huang, S.; Wei, F.; Zhou, M.; and Li, Z. 2020.

TableBank: Table Benchmark for Image-based Table Detection and
Recognition.

In Proceedings of the 12th Language Resources and Evaluation
Conference, 1918–1925. Marseille, France: European Language Resources
Association.

ISBN 979-10-95546-34-4.

- Liu et al. (2022)

Liu, H.; Li, X.; Liu, B.; Jiang, D.; Liu, Y.; and Ren, B. 2022.

Neural Collaborative Graph Machines for Table Structure Recognition.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 4533–4542.

- Liu et al. (2021)

Liu, H.; Li, X.; Liu, B.; Jiang, D.; Liu, Y.; Ren, B.; and Ji, R. 2021.

Show, Read and Reason: Table Structure Recognition with Flexible
Context Aggregator.

In Proceedings of the 29th ACM International Conference on
Multimedia, 1084–1092.

- Long et al. (2021)

Long, R.; Wang, W.; Xue, N.; Gao, F.; Yang, Z.; Wang, Y.; and Xia, G.-S. 2021.

Parsing Table Structures in the Wild.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision, 944–952.

- Papineni et al. (2002)

Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002.

Bleu: a method for automatic evaluation of machine translation.

In Proceedings of the 40th annual meeting of the Association
for Computational Linguistics, 311–318.

- Qasim, Mahmood, and Shafait (2019)

Qasim, S. R.; Mahmood, H.; and Shafait, F. 2019.

Rethinking table recognition using graph neural networks.

In 2019 International Conference on Document Analysis and
Recognition (ICDAR), 142–147. IEEE.

- Qiao et al. (2021)

Qiao, L.; Li, Z.; Cheng, Z.; Zhang, P.; Pu, S.; Niu, Y.; Ren, W.; Tan, W.; and
Wu, F. 2021.

Lgpma: Complicated table structure recognition with local and global
pyramid mask alignment.

In International Conference on Document Analysis and
Recognition, 99–114. Springer.

- Raja, Mondal, and Jawahar (2020)

Raja, S.; Mondal, A.; and Jawahar, C. 2020.

Table structure recognition using top-down and bottom-up cues.

In European Conference on Computer Vision, 70–86. Springer.

- Raja, Mondal, and Jawahar (2022)

Raja, S.; Mondal, A.; and Jawahar, C. 2022.

Visual Understanding of Complex Table Structures from Document
Images.

In Proceedings of the IEEE/CVF Winter Conference on
Applications of Computer Vision, 2299–2308.

- Schreiber et al. (2017)

Schreiber, S.; Agne, S.; Wolf, I.; Dengel, A.; and Ahmed, S. 2017.

Deepdesrt: Deep learning for detection and structure recognition of
tables in document images.

In 2017 14th IAPR international conference on document analysis
and recognition (ICDAR), volume 1, 1162–1167. IEEE.

- Siddiqui et al. (2019)

Siddiqui, S. A.; Fateh, I. A.; Rizvi, S. T. R.; Dengel, A.; and Ahmed, S. 2019.

Deeptabstr: Deep learning based table structure recognition.

In 2019 International Conference on Document Analysis and
Recognition (ICDAR), 1403–1409. IEEE.

- Smock, Pesala, and Abraham (2022)

Smock, B.; Pesala, R.; and Abraham, R. 2022.

PubTables-1M: Towards comprehensive table extraction from
unstructured documents.

In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, 4634–4642.

- Vaswani et al. (2017)

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.;
Kaiser, Ł.; and Polosukhin, I. 2017.

Attention is all you need.

Advances in neural information processing systems, 30.

- Xu et al. (2020)

Xu, Y.; Li, M.; Cui, L.; Huang, S.; Wei, F.; and Zhou, M. 2020.

Layoutlm: Pre-training of text and layout for document image
understanding.

In Proceedings of the 26th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining, 1192–1200.

- Xu et al. (2021)

Xu, Y.; Xu, Y.; Lv, T.; Cui, L.; Wei, F.; Wang, G.; Lu, Y.; Florencio, D.;
Zhang, C.; Che, W.; Zhang, M.; and Zhou, L. 2021.

LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document
Understanding.

In Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics (ACL) 2021.

- Xue, Li, and Tao (2019)

Xue, W.; Li, Q.; and Tao, D. 2019.

ReS2TIM: Reconstruct syntactic structures from table images.

In 2019 International Conference on Document Analysis and
Recognition (ICDAR), 749–755. IEEE.

- Xue et al. (2021)

Xue, W.; Yu, B.; Wang, W.; Tao, D.; and Li, Q. 2021.

TGRNet: A Table Graph Reconstruction Network for Table Structure
Recognition.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision, 1295–1304.

- Ye et al. (2021)

Ye, J.; Qi, X.; He, Y.; Chen, Y.; Gu, D.; Gao, P.; and Xiao, R. 2021.

PingAn-VCGroup’s Solution for ICDAR 2021 Competition on Scientific
Literature Parsing Task B: Table Recognition to HTML.

arXiv preprint arXiv:2105.01848.

- Yu et al. (2018)

Yu, F.; Wang, D.; Shelhamer, E.; and Darrell, T. 2018.

Deep layer aggregation.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, 2403–2412.

- Zhang et al. (2022)

Zhang, Z.; Zhang, J.; Du, J.; and Wang, F. 2022.

Split, embed and merge: An accurate table structure recognizer.

Pattern Recognition, 126: 108565.

- Zheng et al. (2021)

Zheng, X.; Burdick, D.; Popa, L.; Zhong, X.; and Wang, N. X. R. 2021.

Global table extractor (gte): A framework for joint table
identification and cell structure recognition using visual context.

In Proceedings of the IEEE/CVF winter conference on
applications of computer vision, 697–706.

- Zhong, ShafieiBavani, and Jimeno Yepes (2020)

Zhong, X.; ShafieiBavani, E.; and Jimeno Yepes, A. 2020.

Image-based table recognition: data, model, and evaluation.

In European Conference on Computer Vision, 564–580. Springer.

- Zhou, Wang, and Krähenbühl (2019)

Zhou, X.; Wang, D.; and Krähenbühl, P. 2019.

Objects as points.

arXiv preprint arXiv:1904.07850.

Generated on Thu Feb 29 21:19:06 2024 by LaTeXML
