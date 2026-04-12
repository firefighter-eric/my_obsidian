# Long et al. - 2024 - LORE Logical Location Regression Network for Table Structure Recognition with Pre-training

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Long et al. - 2024 - LORE Logical Location Regression Network for Table Structure Recognition with Pre-training.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2401.01522
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# LORE++: Logical Location Regression Network for Table Structure Recognition with Pre-training

Rujiao Long∗∗\,{}^{\ast}, Hangdi Xing∗∗\,{}^{\ast}, Zhibo Yang, Qi Zheng, Zhi Yu, Cong Yao ✉, Fei Huang
Rujiao Long, Zhibo Yang, Qi Zheng, Cong Yao and Fei Huang are with the Alibaba Group, Hangzhou, 310030, China (e-mail: rujiao.lrj@gmail.com, yangzhibo450@gmail.com, yongqi.zq@taobao.com, yaocong2010@gmail.com, feirhuang@gmail.com)Hangdi Xing and Zhi Yu are with the Zhejiang University, Hangzhou, 310027, China (e-mail: xinghd@zju.edu.cn, yuzhirenzhe@zju.edu.cn)∗∗\,{}^{\ast} Equal contribution. ✉ Corresponding author: Cong Yao.

###### Abstract

Table structure recognition (TSR) aims at extracting tables in images into machine-understandable formats. Recent methods solve this problem by predicting the adjacency relations of detected cell boxes or learning to directly generate the corresponding markup sequences from the table images. However, existing approaches either count on additional heuristic rules to recover the table structures, or face challenges in capturing long-range dependencies within tables, resulting in increased complexity. In this paper, we propose an alternative paradigm. We model TSR as a logical location regression problem and propose a new TSR framework called LORE, standing for LOgical location REgression network, which for the first time regresses logical location as well as spatial location of table cells in a unified network. Our proposed LORE is conceptually simpler, easier to train, and more accurate than other paradigms of TSR. Moreover, inspired by the persuasive success of pre-trained models on a number of computer vision and natural language processing tasks, we propose two pre-training tasks to enrich the spatial and logical representations at the feature level of LORE, resulting in an upgraded version called LORE++. The incorporation of pre-training in LORE++ has proven to enjoy significant advantages, leading to a substantial enhancement in terms of accuracy, generalization, and few-shot capability compared to its predecessor. Experiments on standard benchmarks against methods of previous paradigms demonstrate the superiority of LORE++, which highlights the potential and promising prospect of the logical location regression paradigm for TSR.

###### Index Terms:

## I Introduction

Data in tabular format is prevalent in various sorts of documents for summarizing and presenting information. As the world is going digital, the need for parsing the tables trapped in unstructured data (e.g., images and PDF files) is growing rapidly. Although straightforward for humans, it is challenging for automated systems due to the wide diversity of layouts and styles of tables. Table Structure Recognition (TSR) refers to transforming tables in images to machine-understandable formats, usually in logical coordinates or markup sequences. The extracted table structures are crucial for various applications, such as information retrieval, table-to-text generation, and question answering.

Pioneer methods [1, 2, 3, 4, 5, 6] elaborately design the detectors to accurately obtain the spatial locations, i.e., bounding boxes of table cells, and recover the table structure by heuristic rules based on visual clues including lines, aligned cell boundaries and text regions. With the development of deep learning, TSR methods have recently advanced substantially by automatically predicting the structure of the table. Most deep learning-based TSR methods can be categorized into the following paradigms. The first type of models [7, 8, 9] aim at exploring the adjacency relationships between pairs of detected cells to generate intermediate results. They rely on tedious post-processing or graph optimization algorithms to reconstruct the table as logical coordinates, as depicted in Figure 1a, which would struggle with complex table structures. Another paradigm formulates TSR as a markup language sequence generation problem [10, 11], as shown in Figure 1b. Although it simplifies the TSR pipelines, the models are supposed to redundantly learn a markup grammar from noisy sequence labels, which results in a much larger amount of training data. Besides, these models are time-consuming due to the sequential decoding process.

(a) Adjacency relationship representations

(b) Markup sequence representations

(c) Logical location representations

In fact, logical coordinates are well-defined machine-understandable representations of table structures, which are complete to reconstruct tables and can be converted into adjacency matrices and markup sequences by simple and clear transformations. Recently, there has been a focus on exploring the logical locations of table cells [12] as depicted in Figure 1c. However, the method predicts logical locations by ordinal classification, which is apt to suffer from the long-tailed distribution of row (column) numbers. More importantly, this method does not account for the natural dependencies between logical locations. For example, the design of a table itself is from top to bottom, left to right, causing the logical location of cells to be interdependent. This nature of logical locations is sketched in Figure 2. Furthermore, the work lacks a comprehensive comparison among various TSR paradigms.

Aiming at breaking the limitations of existing methods, we propose LOgical Location REgression Network (LORE for abbreviation), a conceptually simpler and more effective TSR framework. It first locates table cells on the input image and then predicts the logical locations along with the spatial locations of cells. To better model the dependencies and constraints between logical locations, a cascade regression framework is adopted, combined with the inter-cell and intra-cell supervisions. The inference of LORE is a parallel network forward-pass, without any efforts in complicated post-processings or sequential decoding strategies.

We evaluate LORE on a wide range of benchmarks against TSR methods of different paradigms. Experiments show that LORE is highly competitive and outperforms previous state-of-the-art methods. Specifically, LORE surpasses other logical location prediction methods by a large margin. Moreover, the adjacency relations and markup sequences derived from the predictions of LORE are of higher quality, which demonstrates that LORE covers the capacity of the models trained under other TSR paradigms.

Despite the advancements in TSR through the logical location regression paradigm, it does not leverage the abundance of tables from larger datasets [10, 13]. Inspired by the persuasive success of pre-trained models on both computer vision and natural language processing tasks [14, 15], we extend the LORE model to a pre-trained version, i.e. LORE++. Humans can grasp the structure of tables effortlessly regardless of the various layouts and styles since the awareness of basic vision clues and the clear notion of logical grids in mind. So we dedicated ourselves to enlightening the model to learn the logical row and column grids. In fact, the tasks of spatial and logical location prediction are closely intertwined. Enhancing the accuracy of the spatial location prediction will result in improved precision in the logical location prediction. Therefore, we employ the Masked Autoencoder (MAE) task to enhance the model’s understanding of tabular images, thereby improving the outcomes of spatial location prediction. Additionally, we propose a novel pre-training task called Logical Distance Prediction (LDP) to comprehend the logical relationships between text segments in the images, thereby boosting the model’s capability in logical location prediction. Besides, we utilize a masking strategy that enables the joint training of two pre-training tasks in a single forward-pass. We have curated a pre-training dataset of 1.5 million samples by integrating academic datasets and generated labels for the LDP task by extracting OCR results using an open-source OCR engine. Consequently, LORE++ improves the vanilla LORE in terms of accuracy and data efficiency.

Our main contributions can be summarized as follows:

- •

We propose to model TSR as a logical location regression problem and design LORE, a new TSR framework that captures dependencies and constraints between logical locations of cells, and predicts the logical locations along with the spatial locations.

- •

We empirically demonstrate that the logical location regression paradigm is highly effective and covers the abilities of previous TSR paradigms, such as predicting adjacency relations and generating markup sequences.

- •

We extend LORE to LORE++ by introducing two pre-training tasks specially designed for TSR. LORE++ can extract enhanced representations via the two pre-training tasks, which lead to improved accuracy and better few-shot capability, compared with its predecessor.

The following of this paper is organized as follows. Section II describes the relevant works to our paper and preliminaries of TSR. We then detail the architecture of the vanilla LORE model as well as the pre-train framework design for LORE++ in Section III, IV, and Section V. Next, Section VI presents the extensive experimental results and analyses. Finally, we conclude our paper in Section VII and discuss the potential applications and future research directions.

The LORE was originally proposed in our previous conference paper [16]. This article extends that work with the following improvements and modifications: (1) In order to further improve the performance and generalization ability of LORE, we extend it into a pre-trained version, termed LORE++. We design a pre-training framework to jointly train the logical location regression network on both spatial and logical tasks (see Section V). (2) We conduct comprehensive experiments and demonstrate that LORE++ can substantially boost both the cell detection and structure recognition results. More importantly, it shows superiority in terms of data efficiency and generalization ability.
(3) We also present more ablation studies, comparisons and analyses. Besides, to better facilitate real-world applications, we devise and realize the transformations from logical locations to relation adjacent matrices and markup sequences (see Section VI).

## II Related Work

### II-A TSR based on accurate cell segmentation

Early works [17, 18] introduce segmentation or detection frameworks to locate and extract splitting lines of table rows and columns. Subsequently, they reconstruct the table structure by empirically grouping the cell boxes with pre-defined rules. These models would suffer from tables with spanning cells or distortions. The latest baselines [2, 13, 19] tackle this problem with well-designed detectors or attention-based merging modules to obtain more accurate cell boundaries and merging results. However, they either are tailored for a certain type of datasets or require customized processing to recover table structures, and thus can hardly be generalized. So there arise models focusing on directly predicting the table structures with neural networks.

### II-B TSR based on directly structure prediction

[7] proposes to model table cells as text segmentation regions and exploit the relationships between cell pairs. Precisely, it applies graph neural networks [20] to classify pairs of detected cells into horizontal, vertical, and unrelated relations. Following this work, there are models devoted to improving the relationship classification by using elaborated neural networks and adding multi-modal features [21, 8, 1, 22, 9]. This framework bypasses the extraction of precise boundaries of cells, but the nearest neighbor graphs in these models encode biased prior to the model. Moreover, there is still a gap between the set of relation triplets and the global table structure. Complex graph optimization algorithms or pre-defined post-processings are needed to recover the tables.

[23, 10, 24] make the pioneering attempts to solve the TSR problem in an end-to-end way. They employ sequence decoders to generate tags of markup language that represent table structures. However, the models are supposed to learn the markup grammar with noisy labels, resulting in the methods being difficult to train and requiring tens of times more training samples than methods of other paradigms. Besides, these models are time-consuming owing to the sequential decoding process.

[12] propose to perform ordinal classification of logical indices on each detected cell for TSR, which is close to our approach. The model utilizes graph neural networks to classify detected cells into the corresponding logical locations, while it ignores the dependencies and constraints among logical locations of cells. Besides, the model is only evaluated on a few datasets and not against the strong TSR baselines.

### II-C Pre-training models

The pre-trained model shows remarkable performance on numerous CV tasks, e.g. classification[15], segmentation[25, 26], detection[27, 28], and information extraction[29, 30], which proved the pre-trained representations generalize well to various downstream data. Unfortunately, there is currently no pre-training work in TSR with images as input only.

MAE[15] masks random patches of the input image and reconstructs the missing pixels. Successfully reconstructing the object in the image indicates that the model understands what the object is and what it looks like. So, we utilize MAE as one of our pre-training tasks to absorb various layout and structure information from massive data.

Besides, ESP[30] constructs key-value linking task in pre-training to model entity linking task in fine-tuning. Due to the consistency between pre-training and fine-tuning, ESP improves the SOTA accuracy of linking tasks from 81.25% to 92.31% on XFUND[31] dataset. Inspired by ESP, we use text segments instead of table cells to model logical relationship, for which the text segments can be easily obtained by the OCR engine.

## III Preliminaries

### III-A Problem Definition

In this paper, we consider the TSR problem as the spatial and logical location regression task. Specifically, for an input image of the table, similar to a detector, a set of table cells {O1,O2,…,ON}subscript𝑂1subscript𝑂2…subscript𝑂𝑁\{O_{1},O_{2},...,O_{N}\} are predicted as their logical locations {l1,l2,…,lN}subscript𝑙1subscript𝑙2…subscript𝑙𝑁\{l_{1},l_{2},...,l_{N}\}, along with the spatial locations {B1,B2,…,BN}subscript𝐵1subscript𝐵2…subscript𝐵𝑁\{B_{1},B_{2},...,B_{N}\}, where li=(rs(i),re(i),cs(i),ce(i))subscript𝑙𝑖superscriptsubscript𝑟𝑠𝑖superscriptsubscript𝑟𝑒𝑖superscriptsubscript𝑐𝑠𝑖superscriptsubscript𝑐𝑒𝑖l_{i}=(r_{s}^{(i)},r_{e}^{(i)},c_{s}^{(i)},c_{e}^{(i)}) standing for the starting-row, ending-row, starting-column and ending-column, Bi={(xk(i),yk(i))}k=1,2,3,4subscript𝐵𝑖subscriptsuperscriptsubscript𝑥𝑘𝑖superscriptsubscript𝑦𝑘𝑖𝑘1234B_{i}=\{(x_{k}^{(i)},y_{k}^{(i)})\}_{k=1,2,3,4} standing for the four corner points of the i𝑖i-th cell and N𝑁N is the number of cells in the image.

With the predicted table cells represented by their spatial and logical locations, the table in the image can be converted into machine-understandable formats, such as relational databases. Besides, the adjacency matrices and the markup sequences of tables can be directly derived from their logical coordinates with well-defined transformations rather than heuristic rules (See supplementary section 1).

### III-B Transformation on Logical Coordinates

The transformation from the table representation in the logical location of cells to cell adjacency and markup sequence representation is well-defined for general settings, without any approximation algorithm or heuristic rules.

Input: {C1,…,CKsubscript𝐶1…subscript𝐶𝐾C_{1},...,C_{K}}, markup = ’ ’
Output: markup

1: Let i=0𝑖0i=0.

2: for i=0𝑖0i=0 to K−1𝐾1K-1 do

3: markup += ’<tr>’

4: for j=0𝑗0j=0 to |Ck|−1subscript𝐶𝑘1|C_{k}|-1 do

5: ai​k=(Ck)isubscript𝑎𝑖𝑘subscriptsubscript𝐶𝑘𝑖a_{ik}=(C_{k})_{i}

6: r​s​p=1+re(ai​k)−rs(ai​k)𝑟𝑠𝑝1superscriptsubscript𝑟𝑒subscript𝑎𝑖𝑘superscriptsubscript𝑟𝑠subscript𝑎𝑖𝑘rsp=1+r_{e}^{(a_{ik})}-r_{s}^{(a_{ik})}

7: c​s​p=1+ce(ai​k)−cs(ai​k)𝑐𝑠𝑝1superscriptsubscript𝑐𝑒subscript𝑎𝑖𝑘superscriptsubscript𝑐𝑠subscript𝑎𝑖𝑘csp=1+c_{e}^{(a_{ik})}-c_{s}^{(a_{ik})}

8: markup += <td rowspan = r​s​p𝑟𝑠𝑝rsp colspan = c​s​p𝑐𝑠𝑝csp ></td>’

9: end for

10: markup += </tr>

11: end for

12: return markup

#### III-B1 Adjacency from Logical Location

The definition of adjacency of cells is based on the logical location [32]. Given a set of cells represented by logical locations as C={a|a=(rs,re,cs,ce)}𝐶conditional-set𝑎𝑎subscript𝑟𝑠subscript𝑟𝑒subscript𝑐𝑠subscript𝑐𝑒C=\{a|a=(r_{s},r_{e},c_{s},c_{e})\}, where rs,re,cs,cesubscript𝑟𝑠subscript𝑟𝑒subscript𝑐𝑠subscript𝑐𝑒r_{s},r_{e},c_{s},c_{e} denote the starting-row, ending-row, starting-column, the adjacency between two cells a,b∈C𝑎𝑏𝐶a,b\in C is a binary relationship R𝑅R as:

a​R​b:=(p∧q)∨(r∧s),assign𝑎𝑅𝑏𝑝𝑞𝑟𝑠aRb:=(p\land q)\vee(r\land s),

(1)

where proposition p,q,r,s𝑝𝑞𝑟𝑠p,q,r,s are defined as:

p:(re(b)<=rs(a)<=re(b))∨:𝑝limit-fromsuperscriptsubscript𝑟𝑒𝑏superscriptsubscript𝑟𝑠𝑎superscriptsubscript𝑟𝑒𝑏\displaystyle p:(r_{e}^{(b)}<=r_{s}^{(a)}<=r_{e}^{(b)})\vee

(2)

(re(a)<=rs(b)<=re(a)),superscriptsubscript𝑟𝑒𝑎superscriptsubscript𝑟𝑠𝑏superscriptsubscript𝑟𝑒𝑎\displaystyle(r_{e}^{(a)}<=r_{s}^{(b)}<=r_{e}^{(a)}),

q:(csa−ce(b)=1)∨(csb−ce(a)=1),:𝑞superscriptsubscript𝑐𝑠𝑎superscriptsubscript𝑐𝑒𝑏1superscriptsubscript𝑐𝑠𝑏superscriptsubscript𝑐𝑒𝑎1q:(c_{s}^{a}-c_{e}^{(b)}=1)\vee(c_{s}^{b}-c_{e}^{(a)}=1),

(3)

where p∧q𝑝𝑞p\land q denotes a𝑎a and b𝑏b are locating (spanning) in the same row and a/b is exactly in the next column of b/a.

r:(ce(b)<=cs(a)<=ce(b))∨:𝑟limit-fromsuperscriptsubscript𝑐𝑒𝑏superscriptsubscript𝑐𝑠𝑎superscriptsubscript𝑐𝑒𝑏\displaystyle r:(c_{e}^{(b)}<=c_{s}^{(a)}<=c_{e}^{(b)})\vee

(4)

(ce(a)<=cs(b)<=ce(a)),superscriptsubscript𝑐𝑒𝑎superscriptsubscript𝑐𝑠𝑏superscriptsubscript𝑐𝑒𝑎\displaystyle(c_{e}^{(a)}<=c_{s}^{(b)}<=c_{e}^{(a)}),

s:(rsa−re(b)=1)∨(rsb−re(a)=1).:𝑠superscriptsubscript𝑟𝑠𝑎superscriptsubscript𝑟𝑒𝑏1superscriptsubscript𝑟𝑠𝑏superscriptsubscript𝑟𝑒𝑎1s:(r_{s}^{a}-r_{e}^{(b)}=1)\vee(r_{s}^{b}-r_{e}^{(a)}=1).

(5)

And r∧s𝑟𝑠r\land s is defined similar as p∧q𝑝𝑞p\land q. By this definition, the adjacency of cells can be straightforwardly computed from the logical location of cells.

#### III-B2 Markup from Logical Location

Given a table represented as logical coordinates, we first define the table rows as finite ordered sets, following the notations before as:

Ck={ai|rs(ai)=k,cs(a1)<cs(a1)<…<cs(aN)},subscript𝐶𝑘conditional-setsubscript𝑎𝑖formulae-sequencesuperscriptsubscript𝑟𝑠subscript𝑎𝑖𝑘superscriptsubscript𝑐𝑠subscript𝑎1superscriptsubscript𝑐𝑠subscript𝑎1…superscriptsubscript𝑐𝑠subscript𝑎𝑁C_{k}=\{a_{i}|r_{s}^{(a_{i})}=k,c_{s}^{(a_{1})}<c_{s}^{(a_{1})}<...<c_{s}^{(a_{N})}\},

(6)

where Cksubscript𝐶𝑘C_{k} denotes the set of cells in the k𝑘k-th row of the table and N=|Ck|𝑁subscript𝐶𝑘N=|C_{k}|. These sets exist and are subject to :

C=C1∪C2∪…∪CK,𝐶subscript𝐶1subscript𝐶2…subscript𝐶𝐾C=C_{1}\cup C_{2}\cup...\cup C_{K},\\

(7)

and:

Ci∩Cj=∅,i,j∈{1,2,…,K},formulae-sequencesubscript𝐶𝑖subscript𝐶𝑗𝑖𝑗12…𝐾C_{i}\cap C_{j}=\emptyset,i,j\in\{1,2,...,K\},

(8)

Where K𝐾K is the number of rows. The transformation of logical coordinates into markup sequence is then defined as in the algorithm 1.

## IV Model Architecture

This section elaborates on our proposed LORE, a TSR framework regressing the spatial and logical locations of cells. As illustrated in Figure 3, LORE employs a vision backbone to extract visual features of table cells from the input image. Then the spatial and logical locations of cells are predicted by two regression heads. We specially leverage the cascading regressors and employ inter-cell and intra-cell supervisions to model the dependencies and constraints between logical locations. The following subsections specify these crucial components.

### IV-A Table Cell Features Preparation

In order to streamline the joint prediction of spatial and logical locations, we employ a key point segmentation network [33, 2] as the feature extractor and model each table cell in the image as its center point. Besides, it is compatible with both wired and wireless tables and is easier to implement on inconsistent annotations of different datasets, i.e., aligned boxes in WTW and TG24, or text region boxes in SciTSR and PubTabNet.

For an input image of width W𝑊W and height H𝐻H, the network produces a feature map f∈ℝWR×HR×d𝑓superscriptℝ𝑊𝑅𝐻𝑅𝑑f\in\mathbb{R}^{\frac{W}{R}\times\frac{H}{R}\times d} and a cell center heatmap Y^∈[0,1]WR×HR^𝑌superscript01𝑊𝑅𝐻𝑅\widehat{Y}\in[0,1]^{\frac{W}{R}\times\frac{H}{R}}, where R𝑅R, d𝑑d are the output stride and hidden size; Y^x,y=1subscript^𝑌𝑥𝑦1\widehat{Y}_{x,y}=1 corresponds to a detected cell center, while Y^x,y=0subscript^𝑌𝑥𝑦0\widehat{Y}_{x,y}=0 refers to the background.

In the subsequent modules, the CNN features {f(1),f(2),…,f(N)}superscript𝑓1superscript𝑓2…superscript𝑓𝑁\{f^{(1)},f^{(2)},...,f^{(N)}\} at detected cell centers {p^(1),p^(2),…,p^(N)}superscript^𝑝1superscript^𝑝2…superscript^𝑝𝑁\{\hat{p}^{(1)},\hat{p}^{(2)},...,\hat{p}^{(N)}\} are considered as the representations of table cells.

### IV-B Spatial Location Regression

We choose to predict the four corner points rather than the rectangle bounding box to better deal with the inclines and distortions of tables in the wild. For spatial locations, the features of the backbone f𝑓f are passed through a 3×3333\times 3 convolution, ReLU and another 1×1111\times 1 convolution to get the prediction {B^(1),B^(2),…,B^(N)}superscript^𝐵1superscript^𝐵2…superscript^𝐵𝑁\{\hat{B}^{(1)},\hat{B}^{(2)},...,\hat{B}^{(N)}\} on centers {p^(1),p^(2),…,p^(N)}superscript^𝑝1superscript^𝑝2…superscript^𝑝𝑁\{\hat{p}^{(1)},\hat{p}^{(2)},...,\hat{p}^{(N)}\}, where B^(i)={(x^k(i),y^k(i))}k=1,2,3,4superscript^𝐵𝑖subscriptsuperscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖𝑘1234\hat{B}^{(i)}=\{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)})\}_{k=1,2,3,4}.

### IV-C Logical Location Regression

As dense dependencies and constraints exist between the logical locations of table cells, it is rather challenging to learn the logical coordinates from the visual features of cell centers alone. The cascading regressors with inter-cell and intra-cell supervisions are leveraged to explicitly model the logical relations between cells.

#### IV-C1 Base Regressor

To better model the logical relations from images, the visual features are first combined with the spatial information. Specifically, the features of the predicted corner points of the cells are computed as the sum of their visual features and 2-dimensional position embeddings:

f~(x^k(i),y^k(i),:)=f(x^k(i),y^k(i),:)+P​E​(x^k(i),y^k(i)),subscript~𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:subscript𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:𝑃𝐸superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖\widetilde{f}_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)}=f_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)}+PE(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)}),

(9)

where P​E𝑃𝐸PE refers to the 2-dimensional position embedding function [34, 35]. Then the features of the four corner points are added to the center features f(i)superscript𝑓𝑖f^{(i)} to enhance the representation of each predicted cell center p^(i)superscript^𝑝𝑖\hat{p}^{(i)} as:

h(i)=f(i)+∑k=14wk​f~(x^k(i),y^k(i),:),superscriptℎ𝑖superscript𝑓𝑖superscriptsubscript𝑘14subscript𝑤𝑘subscript~𝑓superscriptsubscript^𝑥𝑘𝑖superscriptsubscript^𝑦𝑘𝑖:h^{(i)}=f^{(i)}+\sum_{k=1}^{4}w_{k}\widetilde{f}_{(\hat{x}_{k}^{(i)},\hat{y}_{k}^{(i)},:)},

(10)

where [w1,w2,w3,w4]subscript𝑤1subscript𝑤2subscript𝑤3subscript𝑤4[w_{1},w_{2},w_{3},w_{4}] are learnable parameters.

Then the message-passing and aggregating networks are adopted to incorporate the interaction between the visual-spatial features of cells:

{h~(i)}i=1,2,…,N=SelfAttention​({h(i)}i=1,2,…,N).subscriptsuperscript~ℎ𝑖𝑖12…𝑁SelfAttentionsubscriptsuperscriptℎ𝑖𝑖12…𝑁\{\widetilde{h}^{(i)}\}_{i=1,2,...,N}={\rm\textbf{Self\-Attention}}(\{h^{(i)}\}_{i=1,2,...,N}).

(11)

We use the self-attention mechanism [36] in LORE to avoid making additional assumptions about the distribution of table structure, rather than graph neural networks employed by previous methods [21, 12], which will be further discussed in experiments.

The prediction of the base regressor is then computed by a linear layer with the ReLU activation from {h~(i)}i=1,2,…,Nsubscriptsuperscript~ℎ𝑖𝑖12…𝑁\{\widetilde{h}^{(i)}\}_{i=1,2,...,N} as l^(i)=(r^s(i),r^e(i),c^s(i),c^e(i))superscript^𝑙𝑖superscriptsubscript^𝑟𝑠𝑖superscriptsubscript^𝑟𝑒𝑖superscriptsubscript^𝑐𝑠𝑖superscriptsubscript^𝑐𝑒𝑖\hat{l}^{(i)}=(\hat{r}_{s}^{(i)},\hat{r}_{e}^{(i)},\hat{c}_{s}^{(i)},\hat{c}_{e}^{(i)}).

#### IV-C2 Stacking Regressor

Although the base regressor encodes the relationships between visual-spatial features of cells, the logical locations of each cell are still predicted individually. To better capture the dependencies and constraints among logical locations, a stacking regressor is employed to look again at the prediction of the base regressor. Specifically, the enhanced features 𝒉~bold-~𝒉\boldsymbol{\widetilde{h}} and the logical location prediction of the base regressor 𝒍^^𝒍\hat{\boldsymbol{l}} are fed into a stacking regressor. The stacking regressor can be expressed as :

𝒍~=Fs​(Ws​𝒍^+𝒉~).~𝒍subscript𝐹𝑠subscript𝑊𝑠^𝒍bold-~𝒉\widetilde{\boldsymbol{l}}=F_{s}(W_{s}\hat{\boldsymbol{l}}+{\boldsymbol{\widetilde{h}}}).

(12)

where Ws∈ℝ4×dsubscript𝑊𝑠superscriptℝ4𝑑W_{s}\in\mathbb{R}^{4\times d} is a learnable parameter, 𝒍^=[l^(1),…,l^(N)]^𝒍superscript^𝑙1…superscript^𝑙𝑁\hat{\boldsymbol{l}}=[\hat{l}^{(1)},...,\hat{l}^{(N)}], 𝒉~=[h~(1),…,h~(N)]bold-~𝒉superscript~ℎ1…superscript~ℎ𝑁{\boldsymbol{\widetilde{h}}}=[\widetilde{h}^{(1)},...,\widetilde{h}^{(N)}] and Fssubscript𝐹𝑠F_{s} denotes the stacking regression function, which has the same self-attention and linear structure as the base regression function but with independent parameters. The output of the stacking regressor is 𝒍~=[l~(1),…,l~(N)]~𝒍superscript~𝑙1…superscript~𝑙𝑁\widetilde{{\boldsymbol{l}}}=[\widetilde{l}^{(1)},...,\widetilde{l}^{(N)}], and l~(i)=(r~s(i),r~e(i),c~s(i),c~e(i))superscript~𝑙𝑖superscriptsubscript~𝑟𝑠𝑖superscriptsubscript~𝑟𝑒𝑖superscriptsubscript~𝑐𝑠𝑖superscriptsubscript~𝑐𝑒𝑖\widetilde{l}^{(i)}=(\widetilde{r}_{s}^{(i)},\widetilde{r}_{e}^{(i)},\widetilde{c}_{s}^{(i)},\widetilde{c}_{e}^{(i)}).

At the inference stage, the results are obtained by assigning the four components of l~(i)superscript~𝑙𝑖\widetilde{l}^{(i)} to the nearest integers.

#### IV-C3 Inter-cell and Intra-cell Supervisions

In order to equip the logical location regressor with a better understanding of the dependencies and constraints between logical locations, we propose the inter-cell and intra-cell supervisions, which are summarized as: 1) The logical locations of different cells should be mutually exclusive (inter-cell). 2) The logical locations of one table cell should be consistent with its spans (intra-cell).

In practice, predictions of cells that are far apart rarely contradict each other, so we only sample adjacent pairs for inter-cell supervision. More formally, the scheme of inter-cell and intra-cell losses can be expressed as:

Li​n​t​e​rsubscript𝐿𝑖𝑛𝑡𝑒𝑟\displaystyle L_{inter}
=∑(i,j)∈Arm​a​x​(r~e(j)−r~s(i)+1,0)absentsubscript𝑖𝑗subscript𝐴𝑟𝑚𝑎𝑥superscriptsubscript~𝑟𝑒𝑗superscriptsubscript~𝑟𝑠𝑖10\displaystyle=\sum_{(i,j)\in A_{r}}max(\widetilde{r}_{e}^{(j)}-\widetilde{r}_{s}^{(i)}+1,0)

(13)

+∑(i,j)∈Acm​a​x​(c~e(j)−c~s(i)+1,0),subscript𝑖𝑗subscript𝐴𝑐𝑚𝑎𝑥superscriptsubscript~𝑐𝑒𝑗superscriptsubscript~𝑐𝑠𝑖10\displaystyle+\sum_{(i,j)\in A_{c}}max(\widetilde{c}_{e}^{(j)}-\widetilde{c}_{s}^{(i)}+1,0),

where Arsubscript𝐴𝑟A_{r} (Acsubscript𝐴𝑐A_{c}) are sets of ordered horizontally (vertically) adjacent pairs, i.e., for a pair of cells (i,j)∈Ar𝑖𝑗subscript𝐴𝑟(i,j)\in A_{r} (Acsubscript𝐴𝑐A_{c}), cell i𝑖i is adjacent to cell j𝑗j in the same row (column) and on the right of (under) cell j𝑗j, and r~s(i)superscriptsubscript~𝑟𝑠𝑖\widetilde{r}_{s}^{(i)}, r~e(j)superscriptsubscript~𝑟𝑒𝑗\widetilde{r}_{e}^{(j)}, c~s(i)superscriptsubscript~𝑐𝑠𝑖\widetilde{c}_{s}^{(i)}, c~e(j)superscriptsubscript~𝑐𝑒𝑗\widetilde{c}_{e}^{(j)} are predicted logical indices of cell i𝑖i and cell j𝑗j.

Li​n​t​r​asubscript𝐿𝑖𝑛𝑡𝑟𝑎\displaystyle L_{intra}
=∑i∈Mr|r~s(i)−r~e(i)−rs(i)+re(i)|absentsubscript𝑖subscript𝑀𝑟superscriptsubscript~𝑟𝑠𝑖superscriptsubscript~𝑟𝑒𝑖superscriptsubscript𝑟𝑠𝑖superscriptsubscript𝑟𝑒𝑖\displaystyle=\sum_{i\in M_{r}}|\widetilde{r}_{s}^{(i)}-\widetilde{r}_{e}^{(i)}-r_{s}^{(i)}+r_{e}^{(i)}|

(14)

+∑i∈Mc|c~s(i)−c~e(i)−cs(i)+ce(i)|,subscript𝑖subscript𝑀𝑐superscriptsubscript~𝑐𝑠𝑖superscriptsubscript~𝑐𝑒𝑖superscriptsubscript𝑐𝑠𝑖superscriptsubscript𝑐𝑒𝑖\displaystyle+\sum_{i\in M_{c}}|\widetilde{c}_{s}^{(i)}-\widetilde{c}_{e}^{(i)}-c_{s}^{(i)}+c_{e}^{(i)}|,

where Mr={i|re(i)−rs(i)≠0}subscript𝑀𝑟conditional-set𝑖superscriptsubscript𝑟𝑒𝑖superscriptsubscript𝑟𝑠𝑖0M_{r}=\{i|r_{e}^{(i)}-r_{s}^{(i)}\neq 0\} and Mc={i|ce(i)−cs(i)≠0}subscript𝑀𝑐conditional-set𝑖superscriptsubscript𝑐𝑒𝑖superscriptsubscript𝑐𝑠𝑖0M_{c}=\{i|c_{e}^{(i)}-c_{s}^{(i)}\neq 0\} are sets of multi-row and multi-column cells.

Then the inter-cell and intra-cell losses (I2C) are as:

LI​2​C=Li​n​t​e​r+Li​n​t​r​a.subscript𝐿𝐼2𝐶subscript𝐿𝑖𝑛𝑡𝑒𝑟subscript𝐿𝑖𝑛𝑡𝑟𝑎L_{I2C}=L_{inter}+L_{intra}.

The supervisions are conducted on the output 𝒍~~𝒍\widetilde{{\boldsymbol{l}}} and no extra forward-passing is required.

### IV-D Objectives

The losses of cell center segmentation Lc​e​n​t​e​rsubscript𝐿𝑐𝑒𝑛𝑡𝑒𝑟L_{center} and spatial location regression Ls​p​asubscript𝐿𝑠𝑝𝑎L_{spa} are computed following typical key point-based detection methods [33, 2].

The loss of logical locations is computed for both the base regressor and the stacking regressor:

Ll​o​g=1N​∑i=1N(‖l^(i)−li‖1+‖l~(i)−li‖1).subscript𝐿𝑙𝑜𝑔1𝑁superscriptsubscript𝑖1𝑁subscriptnormsuperscript^𝑙𝑖subscript𝑙𝑖1subscriptnormsuperscript~𝑙𝑖subscript𝑙𝑖1L_{log}=\frac{1}{N}\sum_{i=1}^{N}(||\hat{l}^{(i)}-l_{i}||_{1}+||\widetilde{l}^{(i)}-l_{i}||_{1}).

(15)

The total loss of joint training is then computed by adding the losses of cell center segmentation, spatial and logical location regression along with the I2C supervisions:

LL​O​R​E=Lc​e​n​t​e​r+Ls​p​a+Ll​o​g+LI​2​C.subscript𝐿𝐿𝑂𝑅𝐸subscript𝐿𝑐𝑒𝑛𝑡𝑒𝑟subscript𝐿𝑠𝑝𝑎subscript𝐿𝑙𝑜𝑔subscript𝐿𝐼2𝐶L_{LORE}=L_{center}+L_{spa}+L_{log}+L_{I2C}.

(16)

## V Pre-training Framework

We present a pre-training framework with objectives that include both visual and logical structure of tables, which enhances the model’s ability to comprehend and reason about table structure in a holistic manner. In order to conduct joint training in a single forward pass and avoid information leaks, we propose unidirectional self-attention. Figure 4 shows the pre-training framework and self-attention masking strategy.

### V-A Model Architecture

As illustrated in Figure 4, We follow the framework of LORE with corresponding modifications to facilitate the joint pretraining of spatial and logical tasks. The CNN-based backbone of LORE is replaced with the ViT encoder and the MAE ViT-based decoder is added, catering to the MAE-like pre-training. The logical decoder shares a similar structure with the base regressor and the stacking regressor, i.e. layers of self-attention mechanism, but with the linear map on paired features for the logical distance prediction task.

The model takes the patchified images and the corresponding masks as input. Aiming at preventing information leakage from the masked patches to the unmasked ones, we leverage a unidirectional self-attention, where the unmasked tokens can only attend to each other but not the masked ones, while the masked patches can attend to all patches. The masked patches are replaced with the mask token before being inputted into the spatial decoder, while the entire encoded feature maps are forwarded to the logical decoder for the Logical Distance Prediction task.

### V-B Masked Autoencoding Task

Inspired by ViT-MAE, we utilize the MAE task to guide the model to learn general visual clues of tables such as text region, ruling lines, etc. We mask out 50% patch tokens of the image randomly, which is different from the fashion as was suggested in ViT-MAE. Because table images are highly semantic and information-dense, excessively masking out patches could impede the reconstruction task. To introduce order information for the MAE task, we employ 1-D fixed sinusoidal position embeddings. The image encoder and decoder are trained using a normalized mean squared error (MSE) pixel reconstruction loss, which quantifies the disparity between the normalized target image patches and the reconstructed patches. This loss is specifically computed for the masked patches.

### V-C Logical Distance Prediction Task

For this task, the model is trained to predict the row and column logical distances between each pair of cells, as illustrated in Figure 5. In this way, the model learns the ability to understand the basic grids of tables, which serves as the foundation for recognizing complicated table structures. We first pre-process the table images with an off-the-shelf Optical Character Recognition (OCR) system to obtain the 2D position of texts in table cells. Then the horizontal and vertical positions of cells are clustered to conform to the grid rows and columns. In this way, we can obtain the training target of each pair of cells as illustrated in Figure 5. During the training stage, the features of word region boxes are extracted according to their center points in a similar
fashion as is in Section IV-C2. Then the cell features are fed into the logical decoder as introduced in model architecture, where the encoded features are paired for the prediction of logical distance. We employ an L1-loss for the logical distance prediction task.

## VI Experiment

In this section, we conduct comprehensive experiments to research and answer three key questions:

- •

Is the proposed LORE able to effectively predict the logical locations of table cells from input images?

- •

Does the LORE framework, modeling TSR as logical location regression, overcome the limitations and cover the abilities of other paradigms?

- •

Is the proposed pre-training strategy beneficial and how does it affect the performance of LORE++.

For the first question, we compare LORE with baselines directly predicting logical locations [37, 12]. To the best of our knowledge, these are the only two methods that focus on directly predicting the logical locations. Furthermore, we provide a detailed ablation study to validate the effectiveness of the main components. For the second question, we compare LORE with methods that model table structure as cell adjacency or markup sequence with both insights and quantitative results. Finally, we evaluate and analyze the performance of LORE++ to validate the effectiveness of the pre-training method.

### VI-A Datasets

#### VI-A1 Evaluation Benchmarks

We evaluate LORE on a wide range of benchmarks, including tables in digital-born documents, i.e., ICDAR-2013 [38], SciTSR-comp [7], PubTabNet [10], TableBank [23] and TableGraph-24K [12], as well as tables from scanned documents and photos, i.e., ICDAR-2019 [39] and WTW [2]. Details of datasets are available in section 2 of the supplementary. It should be noted that ICDAR-2013 provides no training data, so we extend it to the partial version for cross-validation following previous works [8, 9, 22]. When training LORE on the PubTabNet, we randomly choose 20,000 images from its training set for efficiency.

#### VI-A2 Pre-training Dataset

For the pre-training of LORE++, we use large-scale table collections such as PubTables1M[13], Tablebank, and other small-scale table datasets. The text region, grid rows, and columns are obtained by applying an off-the-shelf OCR system and clustering method. The pre-training set contains 1.5 million table images.

### VI-B Implementation

LORE is trained and evaluated on table images with the max side scaled to a fixed size of 102410241024 (512512512 for SciTSR and PubTabNet) and the short side resized equally. The model is trained for 100 epochs, and the initial learning rate is chosen as 1×10−41superscript1041\times 10^{-4}, decaying to 1×10−51superscript1051\times 10^{-5} and 1×10−61superscript1061\times 10^{-6} at the 70th and 90th epochs for all benchmarks. All the experiments are performed on the platform with 4 NVIDIA Tesla V100 GPUs. We use the DLA-34 [40] backbone, the output stride R=4𝑅4R=4, and the number of channels d=256𝑑256d=256. When implementing on the WTW dataset, a corner point estimation is equipped following [2]. The number of attention layers is set to 3 for both the base and the stacking regressors. We run the model 5 times and take the average performance.

LORE++ adopts a 12-layer vision transformer encoder with 12-head self-attention, hidden size of 384, and 1536 intermediate size of feed-forward networks to better fit the MAE pre-training and controls a comparable amount of parameters with prevalent vision backbones in TSR framework such as ResNET-50. The number of attention layers in the logical head is set to 3 as in the vanilla LORE. The images are resized to 224. We pre-train the model using Adam optimizer with a batch size of 196 for steps. We use a weight decay of 0.05, (β1,β2)=(0.0,0.95)subscript𝛽1subscript𝛽20.00.95(\beta_{1},\beta_{2})=(0.0,0.95), a learning rate of 1.5e-4 and we linearly warm up the learning rate over the first 5% steps. For downstream fine-tuning, the vision backbone is initialized with the parameters of the pre-trained encoder, while both the base regressor and stacking regressor are initialized with the parameters of the pre-trained logical decoder. Other fine-tuning configurations are identical to that of the vanilla LORE.

Logical Location Prediction

Markup Prediction

Datasets
ICDAR-13
ICDAR-19
WTW
TG24K

Datasets
PubTabNet
TableBank

metric
F-1
Acc
F-1
Acc
F-1
Acc
F-1
Acc

metric
TEDS
TEDS
BLEU

ReS2TIM
-
17.4
-
13.8
-
-
-
-

Image2Text
-
-
73.8

TGRNet
66.7
27.5
82.8
26.7
64.7
24.3
92.5
84.5

EDD
89.9
86.0
-

LORE
97.2
86.8
90.6
73.2
96.4
82.9
96.1
87.9

LORE
98.1
92.3
91.1

### VI-C Evaluation Metric

The TSR models of different paradigms are evaluated using different metrics, including 1) accuracy of logical locations [37], 2) BLEU and TEDS [41, 10], and 3) F-1 score of adjacency relationships between cells [32, 38].

For cell logical location evaluation, a table cell coordinate is represented as (start-row,end-row,start-col,end-col)start-rowend-rowstart-colend-col(\text{\emph{start-row}},\text{\emph{end-row}},\text{\emph{start-col}},\text{\emph{end-col}}) as in the ICDAR competition [38]. The accuracy, that is the proportion of the cells with four coordinate values correctly predicted, is calculated as the metric of cell location evaluation score. BLEU[41] is widely adopted in the natural language processing community. TEDS [10] models the markup sequence as a tree (graph) and computes the edit distance between the output structure and label. As for evaluating the adjacency relations, we first convert the table to a list of triplets contains a pair of nodes and their adjacency relation (adjacent/ no adjacent), and make a comparison on relations extracted from output structure and ground truth by precision, recall and F1 score. When evaluating markup generation-based methods, BLEU and TEDS are employed. The accuracy of logical locations, BLEU, and TEDS directly reflect the correctness of the predicted structure, while the adjacency evaluation only measures the quality of intermediate results of the structure.

In our experiments, LORE is evaluated under all three types of metrics, since the logical coordinates are complete for representing table structures and can be converted into adjacency matrices and markup sequences by simple and clarified transformations as introduced in Section. When evaluating TEDS, we use the non-styling text extracted from PDF files following [4]. We also report the performance of cell spatial location prediction, using the F-1 score under the IoU threshold of 0.5, following recent works [8, 12]. In our experiments, We consider a detected cell to be true positive if its IOU with a ground truth cell bounding box is more than 0.5, following [8, 12, 1].

### VI-D Results on Benchmarks

#### VI-D1 Result of LORE

First, we compare LORE with models which directly predict logical locations including Res2TIM [37] and TGRNet [12]. We tune the model provided by [12] on the WTW dataset to make a thorough comparison. As shown in Table I, LORE outperforms the previous methods remarkably. The baseline methods can only produce passable results on relatively simple benchmarks of digital-born table images from scientific articles, i.e., TableGraph-24K. TGRNet [12] detects cells through segmentation of ruling lines, which would struggle with the spanning cells and deformation of tables. Besides, the graph of cells employed by TGRNet which is constructed according to the Euclidean metric introduces biased prior. LORE achieves better performance benefiting from the flexibility of representing table cells as points, and the cascading regressors which model the intrinsic relation among the logical location of cells.

Then we evaluate LORE on the markup sequence generation scene against Image2Text [23] and EDD [10], with the results also derived from the output logical locations of LORE. Specially, since the TableBank dataset does not provide the spatial locations of cells, we implement LORE trained on SciTSR (1/10 the size of TableBank) for the evaluation on it. The results are shown in Table I. Experiment results indicate that LORE is also more effective even if LORE is trained on much fewer samples. This may be because the logical location prediction paradigm tackles the TSR problem in a direct way to model the 2D structure, rather than the circuitous way where the model needs to learn an additional latent transformation from the structure to the noisy markup sequence.

Thirdly, we compare LORE with models mining the adjacency of cells by relation-based metrics: TabStrNet [8], LGPMA [3], TOD [1], FLAGNet [22] and NCGM [9]. The adjacency relation results of LORE are derived from the output logical locations as mentioned before. The results are shown in Table II. It is worth noting that LORE performs much better on challenging benchmarks such as ICDAR-2019 and WTW with scanned documents and photos. Tables in these datasets are with more spanning cells and distortions [9, 2]. Experiments demonstrate that LORE is capable of predicting adjacency relations, as by-products of regressing the logical locations.

#### VI-D2 Result of LORE++

Finally, we explore the contrasts of LORE++ and LORE in terms of both spatial location prediction and logical location prediction in Table III. It indicates that the pre-training triggers the model potential since both tasks are consistently boosted, even if previous models have achieved high performances. Specifically, LORE++ improves the logical location prediction accuracy on ICDAR-2013 the most significantly. This is the smallest dataset with only 158 samples, which illustrates the pre-training stage enhances the generalization of the model. Even though the pre-training dataset contains mostly images of simple digital-born tables, LORE++ is significantly boosted from 82.9% to 84.1% on challenging wild dataset of WTW. We utilize the MAE and logical distance prediction task to guide the model to comprehend the general visual clues of tables and logical relationships. As a result, LORE++ demonstrates improvements across diverse datasets.

Datasets
ICDAR-13
SciTSR-comp
ICDAR-19
WTW

metric
P
R
F1
P
R
F1
P
R
F1
P
R
F1

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

LORE
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
ICDAR-13
SciTSR-comp
PubTabNet
WTW

metric
D-F1
R-F1
Acc
D-F1
R-F1
Acc
D-F1
R-F1
Acc
D-F1
R-F1
Acc

FLAGNet
-
98.6
-
-
98.5
-
-
-
-
-
90.5
-

NCGM
-
98.8
-
-
98.8
-
-
-
-
-
94.1
-

LORE
97.2
98.9
86.8
97.1
99.3
94.6
92.4
98.7
91.0
96.4
95.1
82.9

LORE++
98.5
99.2
93.2
99.1
99.4
95.7
94.4
99.1
92.7
97.0
96.9
84.1

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

### VI-E Ablation

#### VI-E1 Ablation Study of LORE

To investigate how the key components of our proposed LORE contribute to the logical location regression, we conduct an intensive ablation study on the WTW dataset. Results are presented in Table IV. First, we evaluate the effectiveness of the inter-cell loss Li​n​t​e​rsubscript𝐿𝑖𝑛𝑡𝑒𝑟L_{inter} and the intra-cell loss Li​n​t​r​asubscript𝐿𝑖𝑛𝑡𝑟𝑎L_{intra}, by training several models turning them on and off. According to the results in experiments 1a and 1b, we see that the inter-cell supervision improves the performance by +0.8%Acc. And from 1a and 1c, the intra-cell supervision benefits more by +1.8%Acc, for the reason that it makes up the message-passing and aggregating mechanism, which pays less attention to intra-cell relations than inter-cell relations according to its inter-cell nature. The combination of the two supervisions makes the best performance.

Then we evaluate the influence of model architecture, i.e., the pattern of message aggregation and the importance of the cascade framework. In experiment 2a, we replace the self-attention encoder with a graph-attention encoder similar to graph-based TSR models [21, 12] with an equal amount of parameters with LORE. It causes a drop in performance consistently. The graph-based encoder only aggregates information from the top-K nearest features of each node based on Euclidean distance, which is biased for table structure. In Experiment 2b, we use a single regressor of 6 layers instead of two cascading regressors of 3 layers. We can observe a performance degradation of 3.1%Acc from 1d to 2b, showing that the cascade framework can better model the dependencies and constraints between logical locations of different cells.

#### VI-E2 Ablation Study of Pre-training

The ablation of the pre-training task is in Table V. The 3a is the baseline of LORE. Actually, replacing the CNN backbone of LORE (3a) with ViT (3c) leads to a performance drop (from 82.9% to 82.1%). Perhaps it’s because the amount of data for WTW is not sufficient, which leads to the inferior performance of ViT compared to CNN. Here the ViT architecture is employed to cater the MAE pre-training for convenience. The difference between experiment <3a, 3c>  and <3a, 3d>  indicates that using only the table data set for MAE pre-training is beneficial, even if the ImageNet is much larger than our pre-training dataset for the pre-trained model learns the basic visual clues of tables, such as cell regions and ruling lines. Adding the logical distance prediction task results in a substantial improvement in logical location prediction according to experiments 3d and 3e. Notably, the spatial prediction task is also boosted by the logical distance prediction task.

N
Task
Backbone
Data
Metrics

MAE
LDP
D-F1
R-F1
Acc

3a(LORE)
-
-
CNN
ImageNet
96.4
95.1
82.9

3b
-
-
ViT
None
87.6
82.3
75.3

3c
✓
-
ViT
ImageNet
96.4
95.3
82.1

3d
✓
-
ViT
Ours
96.9
96.4
83.2

3e(LORE++)
✓
✓
ViT
Ours
97.0
96.9
84.1

(a) Original structure

(b) Shifted structure

### VI-F Further Comparison among Paradigms

In this section, we further compare models of different TSR paradigms introduced before. Previous methods that predict logical locations lack a comprehensive comparison and analysis between these paradigms. We demonstrate how LORE overcomes the limitations of the adjacency-based and the markup-based methods by controlled experiments.

The adjacency of cells alone is not sufficient to represent table structures. Previous methods employ heuristic rules based on spatial locations [9] or graph optimizations [21] to reconstruct the tables. However, it takes tedious modification to make the pre-defined parts compatible with datasets of different types of tables and annotations. Furthermore, the adjacency-based metrics sometimes fail to reflect the correctness of table structures, as depicted in Figure 6. Experiments are conducted to verify this argument quantitatively. We turn the linear layer of the stacking regressor of LORE into an adjacency classification layer of paired cell features and employ post-processings as in NCGM [9] to reconstruct the table. The results are in Table VI. Although this modified model (Adj. paradigm) achieves competitive results with state-of-the-art baselines evaluated on adjacency-based metrics, the accuracy of logical locations obtained from heuristic rules decreases obviously compared to LORE (Log. paradigm), especially on WTW, which contains more spanning cells and distortions.

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

The markup-sequence-based models leverage image encoders and sequence decoders to predict the label sequences. Since the markup language has plenty of control sequences formatting styles, they can be viewed as noise in labels and impede model training [12]. It requires much more training samples and computational costs. As shown in Table VII, the number of training samples of the EDD model on the PubTabNet dataset is more than ten times larger than that of both LORE and LORE++. Besides, the inference process is rather time-consuming (See Table VII) due to the sequential decoding pattern, while models of other paradigms compute for each cell in parallel. The average inference time is computed from the validation set of PubTabNet with the images resized to 1280×1280128012801280\times 1280 for both models.

### VI-G Further Analysis on Cascade Regressors

We conduct experiments to investigate the effect of the cascade framework on the prediction of logical coordinates. In Figure 7, we visualize the attention maps of the last encoder layer of the cascade/single regressors of two cells, i.e., the models 1d and 2b in Table IV. In the cascade framework, the base regressor in Figure 7 (a) focuses on the heading cells (upper or left) to compute logical locations. While the stacking regressor in Figure 7 (b) pays more attention to the surrounding cells to discover finer dependencies among logical locations and make sure the prediction is subject to natural constraints, which is in line with human intuition when designing a table. However, the non-cascade regressor in Figure 7 (c) can only play a role similar to the base regressor, which leaves out important information for the prediction of logical locations.

EDD
LORE
LORE++

#Train Samples
339K
20K
20K

Inference Time
14.8s
0.45s
0.43s

DLA-34
LORE
LORE++

#Params
15.9
24.2
29.7

FLOPs
74.6
75.2
88.3

Model
SciTSR-COMP
ICDAR2013

D-F1
R-F1
Acc
D-F1
R-F1
Acc

LORE
92.9
96.4
87.1
92.1
93.2
78.6

LORE++
95.4
97.9
93.5
95.5
98.6
87.5

### VI-H Computational Analysis

We summarize the model size and the inference operations of LORE and LORE++ in Table VIII, with the input images at 1024×1024102410241024\times 1024 and the number of cells as 32. It is observed that the complexity of LORE is at an equal level to a key point-based detector [33] with the same backbone, showing the efficiency of LORE. The LORE++ is relatively larger since the ViT backbone is employed, but the model size maintains a similar level to the original LORE. Besides, the ablation in Table V has validated that the improvements are not owing to the different backbone networks.

### VI-I Data Efficiency

To validate the effectiveness of pre-training LORE in improving data efficiency, we compare the pre-trained LORE++ model with the baseline model LORE at different training settings: training them using 20%, 60%, and 100% training sets of WTW and SciTSR for equivalent 100 epochs regarding the full training set, e.g., we employ 5 times epochs when using 20% data for training compared with using 100% data. The results are shown in Figure 9. As can be seen, LORE++ consistently outperforms the LORE baseline by a large margin in terms of data efficiency. LORE++ using 60% training data achieves comparable performance with LORE using all data on the SciTSR dataset. With the proposed proxy tasks, the pre-trained LORE++ has a grasp of the notion of basic vision and logical clues, which makes learning the TSR task more efficient with less training data.

### VI-J Generalization

In this section, we conduct experiments to validate whether the pre-training enhances the generalization ability of LORE. We train the model on a hybrid dataset which contains the WTW training set, 20,000 samples of the PubTabNet training set, and the TableGraph24K training set, and evaluate this model on the ICDAR2013 and SciTSR-comp datasets. The results are displayed in Table IX, which depicts the generalization ability is obviously boosted after pretraining.

### VI-K Visualization of TSR Results

In order to reveal the effectiveness of considering the interaction among logical locations, we visualize the structure recognition results of LORE models without/with a stacking logical regressor and the I2C losses. As depicted in Figure 8, models without the stacking regressor and I2C losses encounters problem when predicting the logical location of complicated structures and blank cells, such as the logical errors marked as red lines in Figure 8. While the model with a stacking logical regressor and the I2C losses fixes these errors owning to the stacking regressor refining a rough results of logical locations and knowledge learned from the constrains among logical location of cells.

## VII Conclusion

In summary, we propose LORE, a TSR framework that effectively regresses the spatial locations and the logical locations of table cells from the input images. Furthermore, it models the dependencies and constraints between logical locations by employing the cascading regressors along with the inter-cell and intra-cell supervisions. LORE is straightforward to implement and achieves competitive results, without tedious post-processing or sequential decoding strategies. Experiments show that LORE outperforms state-of-the-art TSR methods under various metrics and overcomes the limitations of previous TSR paradigms. Additionally, we propose the pre-training method of LORE, resulting in an upgraded version called LORE++, which outperforms the baseline LORE in terms of accuracy and data efficiency.

## VIII Acknowledgement

The authors would like to thank the Editors and Reviewers for their hard work and valuable comments.

## References

- [1]

S. Raja, A. Mondal, and C. Jawahar, “Visual understanding of complex table
structures from document images,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2022, pp. 2299–2308.

- [2]

R. Long, W. Wang, N. Xue, F. Gao, Z. Yang, Y. Wang, and G.-S. Xia, “Parsing
table structures in the wild,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 944–952.

- [3]

L. Qiao, Z. Li, Z. Cheng, P. Zhang, S. Pu, Y. Niu, W. Ren, W. Tan, and F. Wu,
“Lgpma: Complicated table structure recognition with local and global
pyramid mask alignment,” in International Conference on Document
Analysis and Recognition. Springer,
2021, pp. 99–114.

- [4]

X. Zheng, D. Burdick, L. Popa, X. Zhong, and N. X. R. Wang, “Global table
extractor (gte): A framework for joint table identification and cell
structure recognition using visual context,” in Proceedings of the
IEEE/CVF winter conference on applications of computer vision, 2021, pp.
697–706.

- [5]

D. Prasad, A. Gadpal, K. Kapadni, M. Visave, and K. Sultanpure,
“Cascadetabnet: An approach for end to end table detection and structure
recognition from image-based documents,” in Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition workshops,
2020, pp. 572–573.

- [6]

S. S. Paliwal, D. Vishwanath, R. Rahul, M. Sharma, and L. Vig, “Tablenet: Deep
learning model for end-to-end table detection and tabular data extraction
from scanned document images,” in 2019 International Conference on
Document Analysis and Recognition (ICDAR). IEEE, 2019, pp. 128–133.

- [7]

Z. Chi, H. Huang, H.-D. Xu, H. Yu, W. Yin, and X.-L. Mao, “Complicated table
structure recognition,” arXiv preprint arXiv:1908.04729, 2019.

- [8]

S. Raja, A. Mondal, and C. Jawahar, “Table structure recognition using
top-down and bottom-up cues,” in European Conference on Computer
Vision. Springer, 2020, pp. 70–86.

- [9]

H. Liu, X. Li, B. Liu, D. Jiang, Y. Liu, and B. Ren, “Neural collaborative
graph machines for table structure recognition,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp.
4533–4542.

- [10]

X. Zhong, E. ShafieiBavani, and A. Jimeno Yepes, “Image-based table
recognition: data, model, and evaluation,” in European Conference on
Computer Vision. Springer, 2020, pp.
564–580.

- [11]

H. Desai, P. Kayal, and M. Singh, “Tablex: a benchmark dataset for structure
and content information extraction from scientific tables,” in
International Conference on Document Analysis and Recognition. Springer, 2021, pp. 554–569.

- [12]

W. Xue, B. Yu, W. Wang, D. Tao, and Q. Li, “Tgrnet: A table graph
reconstruction network for table structure recognition,” in
Proceedings of the IEEE/CVF International Conference on Computer
Vision, 2021, pp. 1295–1304.

- [13]

B. Smock, R. Pesala, and R. Abraham, “Pubtables-1m: Towards comprehensive
table extraction from unstructured documents,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp.
4634–4642.

- [14]

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep
bidirectional transformers for language understanding,” arXiv preprint
arXiv:1810.04805, 2018.

- [15]

K. He, X. Chen, S. Xie, Y. Li, P. Doll’ar, and R. B. Girshick, “Masked
autoencoders are scalable vision learners,” 2022 IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), pp. 15 979–15 988,
2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:243985980

- [16]

H. Xing, F. Gao, R. Long, J. Bu, Q. Zheng, L. Li, C. Yao, and Z. Yu, “Lore:
Logical location regression network for table structure recognition,”
Proceedings of the AAAI Conference on Artificial Intelligence,
vol. 37, no. 3, pp. 2992–3000, Jun. 2023. [Online]. Available:
https://ojs.aaai.org/index.php/AAAI/article/view/25402

- [17]

S. Schreiber, S. Agne, I. Wolf, A. Dengel, and S. Ahmed, “Deepdesrt: Deep
learning for detection and structure recognition of tables in document
images,” in 2017 14th IAPR international conference on document
analysis and recognition (ICDAR), vol. 1. IEEE, 2017, pp. 1162–1167.

- [18]

S. A. Siddiqui, I. A. Fateh, S. T. R. Rizvi, A. Dengel, and S. Ahmed,
“Deeptabstr: Deep learning based table structure recognition,” in
2019 International Conference on Document Analysis and Recognition
(ICDAR). IEEE, 2019, pp. 1403–1409.

- [19]

Z. Zhang, J. Zhang, J. Du, and F. Wang, “Split, embed and merge: An accurate
table structure recognizer,” Pattern Recognition, vol. 126, p.
108565, 2022.

- [20]

T. N. Kipf and M. Welling, “Semi-supervised classification with graph
convolutional networks,” in International Conference on Learning
Representations (ICLR), 2017.

- [21]

S. R. Qasim, H. Mahmood, and F. Shafait, “Rethinking table recognition using
graph neural networks,” in 2019 International Conference on Document
Analysis and Recognition (ICDAR). IEEE, 2019, pp. 142–147.

- [22]

H. Liu, X. Li, B. Liu, D. Jiang, Y. Liu, B. Ren, and R. Ji, “Show, read and
reason: Table structure recognition with flexible context aggregator,” in
Proceedings of the 29th ACM International Conference on Multimedia,
2021, pp. 1084–1092.

- [23]

M. Li, L. Cui, S. Huang, F. Wei, M. Zhou, and Z. Li, “TableBank: Table
benchmark for image-based table detection and recognition,” in
Proceedings of the 12th Language Resources and Evaluation
Conference. Marseille, France:
European Language Resources Association, May 2020, pp. 1918–1925. [Online].
Available: https://aclanthology.org/2020.lrec-1.236

- [24]

J. Ye, X. Qi, Y. He, Y. Chen, D. Gu, P. Gao, and R. Xiao, “Pingan-vcgroup’s
solution for icdar 2021 competition on scientific literature parsing task b:
Table recognition to html,” arXiv preprint arXiv:2105.01848, 2021.

- [25]

J. Xu, S. De Mello, S. Liu, W. Byeon, T. Breuel, J. Kautz, and X. Wang,
“Groupvit: Semantic segmentation emerges from text supervision,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2022, pp. 18 134–18 144.

- [26]

J. Xu, S. Liu, A. Vahdat, W. Byeon, X. Wang, and S. De Mello, “Open-vocabulary
panoptic segmentation with text-to-image diffusion models,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2023, pp. 2955–2966.

- [27]

Z. Dai, B. Cai, Y. Lin, and J. Chen, “Up-detr: Unsupervised pre-training for
object detection with transformers,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2021, pp. 1601–1610.

- [28]

A. Bar, X. Wang, V. Kantorov, C. J. Reed, R. Herzig, G. Chechik, A. Rohrbach,
T. Darrell, and A. Globerson, “Detreg: Unsupervised pretraining with region
priors for object detection,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
14 605–14 615.

- [29]

C. Luo, C. Cheng, Q. Zheng, and C. Yao, “Geolayoutlm: Geometric pre-training
for visual information extraction,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2023, pp. 7092–7101.

- [30]

Z. Yang, R. Long, P. Wang, S. Song, H. Zhong, W. Cheng, X. Bai, and C. Yao,
“Modeling entities as semantic points for visual information extraction in
the wild,” in Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, 2023, pp. 15 358–15 367.

- [31]

J. Wang, C. Liu, L. Jin, G. Tang, J. Zhang, S. Zhang, Q. Wang, Y. Wu, and
M. Cai, “Towards robust visual information extraction in real world: New
dataset and novel solution,” in Proceedings of the AAAI Conference on
Artificial Intelligence, vol. 35, no. 4, 2021, pp. 2738–2745.

- [32]

M. Göbel, T. Hassan, E. Oro, and G. Orsi, “A methodology for evaluating
algorithms for table understanding in pdf documents,” in Proceedings
of the 2012 ACM symposium on Document engineering, 2012, pp. 45–48.

- [33]

X. Zhou, D. Wang, and P. Krähenbühl, “Objects as points,” arXiv
preprint arXiv:1904.07850, 2019.

- [34]

Y. Xu, M. Li, L. Cui, S. Huang, F. Wei, and M. Zhou, “Layoutlm: Pre-training
of text and layout for document image understanding,” in Proceedings
of the 26th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, 2020, pp. 1192–1200.

- [35]

Y. Xu, Y. Xu, T. Lv, L. Cui, F. Wei, G. Wang, Y. Lu, D. Florencio, C. Zhang,
W. Che, M. Zhang, and L. Zhou, “Layoutlmv2: Multi-modal pre-training for
visually-rich document understanding,” in Proceedings of the 59th
Annual Meeting of the Association for Computational Linguistics (ACL) 2021,
2021.

- [36]

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
Advances in neural information processing systems, vol. 30, 2017.

- [37]

W. Xue, Q. Li, and D. Tao, “Res2tim: Reconstruct syntactic structures from
table images,” in 2019 International Conference on Document Analysis
and Recognition (ICDAR). IEEE, 2019,
pp. 749–755.

- [38]

M. Göbel, T. Hassan, E. Oro, and G. Orsi, “Icdar 2013 table competition,”
in 2013 12th International Conference on Document Analysis and
Recognition. IEEE, 2013, pp.
1449–1453.

- [39]

L. Gao, Y. Huang, H. Déjean, J.-L. Meunier, Q. Yan, Y. Fang, F. Kleber, and
E. Lang, “Icdar 2019 competition on table detection and recognition
(ctdar),” in 2019 International Conference on Document Analysis and
Recognition (ICDAR). IEEE, 2019, pp.
1510–1515.

- [40]

F. Yu, D. Wang, E. Shelhamer, and T. Darrell, “Deep layer aggregation,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 2403–2412.

- [41]

K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic
evaluation of machine translation,” in Proceedings of the 40th annual
meeting of the Association for Computational Linguistics, 2002, pp.
311–318.

Generated on Tue Feb 27 11:07:47 2024 by LaTeXML
