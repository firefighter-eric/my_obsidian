# Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Liao et al. - 2023 - DocTr Document Transformer for Structured Information Extraction in Documents.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2307.07929
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\NewCommandCopy\sout

# DocTr: Document Transformer for 
Structured Information Extraction in Documents

Haofu Liao1
 Aruni RoyChowdhury2†
 Weijian Li3
 Ankan Bansal1
 Yuting Zhang1 
 Zhuowen Tu1
 Ravi Kumar Satzoda1
 R. Manmatha1
 Vijay Mahadevan1 

1AWS AI Labs  2MathWorks  3Amazon Physical Stores
Corresponding author liahaofu@amazon.comWork done at AWS AI Labs

###### Abstract

We present a new formulation for structured information extraction (SIE) from visually rich documents. It aims to address the limitations of existing IOB tagging or graph-based formulations, which are either overly reliant on the correct ordering of input text or struggle with decoding a complex graph. Instead, motivated by anchor-based object detectors in vision, we represent an entity as an anchor word and a bounding box, and represent entity linking as the association between anchor words. This is more robust to text ordering, and maintains a compact graph for entity linking. The formulation motivates us to introduce 1) a DOCument TRansformer (DocTr) that aims at detecting and associating entity bounding boxes in visually rich documents, and 2) a simple pre-training strategy that helps learn entity detection in the context of language. Evaluations on three SIE benchmarks show the effectiveness of the proposed formulation, and the overall approach outperforms existing solutions.

## 1 Introduction

Structured information extraction (SIE) from documents, as shown in Fig 1, is the process of extracting entities and their relationships, and returning them in a structured format. Structured information in a document is usually visually-rich – it is not only determined by the content of text but also the layout, typesetting, and/or figures and tables present in the document. Therefore, unlike the traditional information extraction task in nature language processing (NLP) [8, 3, 30] where the input is plain text (usually with a given reading order), SIE assumes the image representation of a document is available, and a pre-built optical character recognition (OCR) system may provide the unstructured text (i.e., without proper reading order). This is a practical assumption for day-to-day processing of business documents, where the documents are usually stored as images or PDFs, and the structured information, such as key-value pairs or line items (see Fig. 2) from invoices and receipts, has been primarily obtained manually. This is time consuming and does not scale well. Hence, automating the document structured information extraction process with efficiency and accuracy is of great practical and scientific importance.

Structured information extraction is part of document intelligence [5], which focuses on the automatic reading, understanding, and analysis of documents. Early approaches to document intelligence usually address the problem purely from either a computer vision or an NLP perspective. The former takes the document as an image input and frames entity detection as object detection or instance segmentation [40, 31]. The latter takes only the textual content of a document as the input, and addresses the problem with NLP solutions, such as IOB tagging via transformers [14].

Recently, models have also been proposed to pre-train on large-scale document collections and apply them to a wide variety of downstream document intelligence problems [37, 11, 1, 20]. Such general-purpose models usually have the ability to make use of multi-modal inputs – text from OCR, layout in the form of text locations, and visual features from images, and pre-training enables them to understand the basic structure of documents. Therefore, general-purpose models have demonstrated significant improvements on multiple document intelligence tasks, such as entity extraction [11, 20], document image classification [37, 1], and document visual question and answering [38, 1].

For structured information extraction, existing general-purpose models rely on two broad approaches: 1) IOB tagging [29] based methods [37, 38, 20], and 2) graph based methods [11, 15]. Both of these approaches suffer from inherent limitations. IOB tagging relies on the correct “reading order” or serialization of text, which however is not given by the OCR. As shown in Fig. 1(a), the raster scan order of OCR text separates I-name and E-name. When there are multiple name entities, it could be non-trivial to know which I-name/E-name word belongs to which name entity. Graph-based methods (Fig. 1 (b)) can result in complex graphs with many words in a document (i.e., many nodes in the graph). Therefore, decoding the entities and their relationships from the adjacency matrices is error-prone.

Given the limitations of existing work, we make the following contributions in this paper:

- •

We introduce a new formulation for SIE where we represent an entity as an anchor word along with a box, and regard the problem as an anchor word based entity detection and association problem (Fig 1 (c)). Thus, we extract entities via bounding boxes and do not depend on the reading order of input. We assign each entity with an anchor word, resulting in a compact graph of entity relations (e.g., the anchor word links in Fig 1 (c)), which facilitates decoding structured information.

- •

We develop a new model, called DOCument TRansformer (DocTr), which combines a language model and visual object detector for joint vision-language document understanding. We note that the recognition of an anchor word is largely a language-dependent task, while the detection of entity boxes is a more vision-dependent task. Therefore, DocTr is an intuitive approach to target this problem under the proposed formulation.

- •

We propose a new pre-training task, called masked detection modeling (MDM), that matches our formulation and helps learn box prediction in the context of language. Our experimental results show that 1) the proposed formulation addresses SIE better than IOB tagging or graph-based solutions, 2) MDM is a more effective pre-training task, in particular when worked together with the new formulation, and 3) the overall approach outperforms existing solutions on three SIE tasks.

## 2 Related Work

General-purpose document understanding.
General-purpose approaches aim to develop a backbone model for document understanding, which is then adapted to address downstream document understanding tasks. LayoutLM [37, 38, 13] is an early approach that pre-trains on a large-scale document dataset. It introduces masked vision-language modeling and layout information for document understanding pre-training. BROS [11] improves LayoutLM via better encoding of the spatial information and introducing a pre-training loss for understanding text blocks in 2D. DocFormer [1] introduces a new architecture and pre-training losses to better leverage text, vision and spatial information in an end-to-end fashion. FormNet [20] encodes neighborhood context for each token using graph convolutions and introduces an attention mechanism to address imperfect serialization. StrucText [22] proposes to extract multi-modal semantic features at both token level, word-segment level and/or entity level. Donut [19] proposes an OCR free solution that is pre-trained to predict document text from images. It is an encoder-decoder model that can directly decode the expected outputs as text for downstream tasks.

Structured information extraction (SIE).
Early approaches [40, 18, 6] formulate the SIE problem as a computer vision problem to either segment or detect entities from documents. However, they cannot address linking of entities due to the limitation of the formulation. With the advent of transformers [34] and their success in NLP, more recent approaches [24, 43, 10] address SIE by incorporating layout/visual information with text inputs to transformers, and extract entities via a NLP formulation [29]. Other approaches [23, 35, 41] propose to regard the text inputs as the nodes in a graph and model the relationship of text inputs via graph neural networks. To extract the relationship between entities, SPADE [15] introduces a graph decoding scheme on learned pairwise affinities between extracted entities.

Table detection and recognition (TDR).
TDR is the task of detecting and recognizing tabular structures from document images. Both SIE and TDR focus on returning information in a structured way from documents. However, unlike SIE where the spatial relationship of entities are unconstrained, TDR assumes a tabular structure of entities (i.e., table cells) and leverages this prior knowledge in the model design and post-processing [28, 44, 26]. Moreover, SIE requires returning a semantic label for each entity which demands an understanding of the text, while TDR does not distinguish between types of table cells but focuses more on table layout. Therefore, the existing approaches [28, 44, 26] to TDR are vision-only approaches.

TextVQA.
Given an input image, TextVQA aims to answer questions related to the text in image. Similar to SIE, existing TextVQA approaches [32, 12, 9, 2] employ multi-modal models that take both the OCR and image as inputs. However, for TextVQA, the answers are typically single entities. It can be challenging to address the problem with TextVQA if we aim to return multiple entities in a structured way, and if an image could have multiple of such structures.

Scene graph generation (SGG).
Generating scene graphs can be regarded as a form of SIE for natural images. SGG methods [17, 36, 42, 39, 33] detect objects as the nodes of scene graphs, and construct edges of scene graphs by identifying the pairwise relationships between objects. This is similar to our formulation of SIE where we extract entities via anchor word guided object detection, and link entities by learning to output their pairwise affinities.

## 3 Approach

### 3.1 Structured Information Extraction

#### Problem Formulation.

Following prior work, we assume the input is the image of a document page, and a pre-built OCR system is applied to detect and recognize the words. The goal of a structured information extraction system for document understanding is to extract a set of grouped entities 𝒢={𝐆i}𝒢subscript𝐆𝑖\mathcal{G}=\{\mathbf{G}_{i}\}, where each entity group 𝐆i={𝐞i​j}subscript𝐆𝑖subscript𝐞𝑖𝑗\mathbf{G}_{i}=\{\mathbf{e}_{ij}\} is a set of entities with predefined relations. As shown in Fig. 2, an entity group may be a key and value pair, or a line item containing the name, count and price entities. We denote an entity as 𝐞=(t,c,b)𝐞𝑡𝑐𝑏\mathbf{e}=(t,c,b) where t𝑡t, c𝑐c and b𝑏b are the text, class label, and location (bounding box) of the entity, respectively. Note that, with OCR inputs, this formulation of an entity can be reduced to 𝐞=(c,b)𝐞𝑐𝑏\mathbf{e}=(c,b), because the text t𝑡t can be obtained by aggregating the OCR text inside b𝑏b.

Next, we propose a new formulation to address structured information extraction. We propose to address entity extraction via anchor word guided detection and entity linking via anchor word association. The former extracts entities {𝐞i}subscript𝐞𝑖\{\mathbf{e}_{i}\}, and the latter links entities into groups {𝐆i}subscript𝐆𝑖\{\mathbf{G}_{i}\}.

#### Entity Extraction via Anchor Word Guided Detection.

To extract an entity 𝐞𝐞\mathbf{e}, we first introduce a new concept called anchor word, which is a designated word of an entity. In Fig. 2, we select the first word of an entity as the anchor word, e.g., “ABC” is the anchor word for value, and “Chicken” is the anchor word for name. Other designations of anchor words are possible (see Sec. 4.2). An anchor word may be regarded as the representation of an entity. Since the goal of extracting an entity 𝐞=(c,b)𝐞𝑐𝑏\mathbf{e}=(c,b) is to find its class label c𝑐c and bounding box b𝑏b, they may then be represented by an anchor word. As shown in Fig. 2, we associate each anchor with a label and a bounding box. For example, the anchor word “Ship” is associated with a label key and a bounding box that encloses the entity “Ship To:”. Therefore, the task of extracting an entity may be seen as first identifying its anchor word, and then obtaining the label and bounding box associated with it.

#### Entity Linking via Anchor Word Association.

We define an entity group as consisting of a primary entity, and all the other entities in the group are secondary. The anchor word of a primary/secondary entity is the primary/secondary anchor word. Once anchor words have been identified, linking entities into entity groups is equivalent to associating anchor words. To establish such association, we first select the primary anchor words of entity groups, and then all the secondary anchor words from the same group are linked to the primary anchor word. The definition of a primary entity may vary. For key-value pairs, the primary anchor words may simply be those anchor words labeled as key. For more general entity groups, we designate a primary anchor word based on the task/data. For example, we choose name’s anchor word “Chicken” as the primary anchor in Fig. 2. Other ways of choosing primary anchors are possible (See Sec. 4.2). Links between primary and secondary anchor words are represented by a binary matrix 𝐌∈{0,1}m×n𝐌superscript01𝑚𝑛\mathbf{M}\in\{0,1\}^{m\times n}. 𝐌i​j=1subscript𝐌𝑖𝑗1\mathbf{M}_{ij}=1 indicates that the i𝑖ith primary anchor word, and j𝑗jth secondary anchor word are linked. Otherwise, 𝐌i​j=0subscript𝐌𝑖𝑗0\mathbf{M}_{ij}=0.

### 3.2 DocTr: Document Transformer

DocTr is a multi-modal transformer that takes both the document image and OCR words (text and position) as input. Unlike existing encoder-only approaches [38, 1, 22], DocTr has an encoder-decoder architecture with 1) two dedicated encoders to encode vision and language features separately, and 2) a vision-language decoder to decode anchor word based outputs for entity extraction and entity linking. An overview of the DocTr architecture is shown in Fig. 3.

#### Vision Encoder.

The vision encoder is adapted from Deformable DETR [45]. It consists of a CNN backbone with multi-scale visual feature extraction, and a deformable transformer encoder for efficient encoding of visual features. Compared with vanilla transformer based vision encoders, this design is more lightweight due to the use of deformable attention, which has linear complexity with respect to the spatial size of image feature maps instead of the quadratic complexity using standard self-attention. As a result, it is capable of encoding high-resolution multi-scale visual features for better detection of small objects/entities.

This vision encoder is shown to work effectively with a transformer decoder for end-to-end object detection [4]. This is helpful to our formulation of entity extraction, where we convert this task into an anchor word guided object detection problem. We also highlight the differences from existing encoder-only methods where the visual features – either region-based [37, 22] or grid-based [38, 1] – are extracted with a pre-trained CNN model; they are sent to the transformer encoder along with OCR inputs without dedicated network components for decoding entity bounding boxes.

#### Language Encoder.

The language encoder is a transformer model adapted from the BERT architecture [7]. We follow LayoutLM [37] to include the layout information (i.e., 2D position embeddings of OCR) along with the OCR text as input. However, no visual information is added since it has already been addressed by the vision encoder. The language encoder is critical to our formulation for the identification of anchor words, which is a language-dependent task.

#### Vision-Language Decoder with Language-Conditioned Queries.

The architecture of the vision-language decoder is similar to the decoder of the Deformable DETR transformer model [45] - with two major differences to facilitate the decoding of vision-language inputs. Each decoder layer has two cross-attention modules to decode from vision and language inputs respectively. For vision, we apply deformable cross-attention (similar to Deformable DETR) to efficiently decode from high-resolution visual features. For language, we apply language-conditioned cross-attention to decode from the discrete OCR language features.

Specifically, we introduce language-conditioned queries to better leverage the OCR inputs and obviate the need for bipartite matching between predicted and ground truth entities. The original DETR-like decoder queries [4, 45] do not have explicit semantic meanings at the beginning. Hence, DETR requires finding the most plausible matching between a prediction and ground truth, which is less effective and impedes the training. For document understanding with OCR inputs, we consider a one-to-one mapping between OCR inputs and decoder queries. That is, we have the same number of queries as the number of OCR inputs to the language encoder, and the i𝑖ith query is mapped to the i𝑖ith OCR input (see Fig. 3). This mapping can be simply modeled as cross-attention between queries and language embeddings by using the same position embedding for both inputs. Let 𝐐∈ℝL×d𝐐superscriptℝ𝐿𝑑\mathbf{Q}\in\mathbb{R}^{L\times d} be a set of L𝐿L decoder queries each with dimension d𝑑d (packed as a matrix), 𝐕∈ℝL×d𝐕superscriptℝ𝐿𝑑\mathbf{V}\in\mathbb{R}^{L\times d} be the set of output embeddings from the language encoder, and 𝐏∈ℝL×d𝐏superscriptℝ𝐿𝑑\mathbf{P}\in\mathbb{R}^{L\times d} be a set of position embeddings. Then, the cross attention with language-conditioned queries can be written as:

CrossAttn​(𝐐,𝐕,𝐏)=softmax​((𝐐+𝐏)​(𝐕+𝐏)Td)​𝐕,CrossAttn𝐐𝐕𝐏softmax𝐐𝐏superscript𝐕𝐏𝑇𝑑𝐕\text{CrossAttn}(\mathbf{Q},\mathbf{V},\mathbf{P})=\text{softmax}(\frac{(\mathbf{Q}+\mathbf{P})(\mathbf{V}+\mathbf{P})^{T}}{\sqrt{d}})\mathbf{V},

where d𝑑\sqrt{d} is a scaling factor [34].
This mapping assigns each query with an explicit linguistic semantic meaning – the i𝑖i-th decoder output now corresponds to the i𝑖i-th input text token, via the i𝑖i-th decoder query. Thus, we can directly match entities with queries without the bipartite matching required by the default DETR decoder formulation [4, 45].

#### Entity Extraction and Linking Outputs.

The decoder has two sets of outputs for entity extraction and entity linking respectively (see Fig. 3). For entity extraction, each output is a class label and a bounding box which uniquely decide an entity. Because each query (and thus its corresponding output) is mapped to an OCR input, the class label indicates whether the underlying OCR input is an anchor word, and the type of entity it represents. For entity linking, each output is a binary class label and an embedding vector. The binary class label indicates whether the OCR input is a primary anchor word. The embedding vector is for the linking of anchor words, and we use different embeddings for primary and secondary anchor words. Let 𝐄p∈ℝm×hsubscript𝐄psuperscriptℝ𝑚ℎ\mathbf{E}_{\text{p}}\in\mathbb{R}^{m\times h} be a set of m𝑚m primary embeddings, and 𝐄s∈ℝn×hsubscript𝐄ssuperscriptℝ𝑛ℎ\mathbf{E}_{\text{s}}\in\mathbb{R}^{n\times h} be a set of n𝑛n secondary embeddings, the predicted affinity matrix for entity linking is computed as 𝐌^=sigmoid​(𝐄p​𝐄sT),𝐌^∈(0,1)m×nformulae-sequence^𝐌sigmoidsubscript𝐄psuperscriptsubscript𝐄s𝑇^𝐌superscript01𝑚𝑛\mathbf{\hat{M}}=\text{sigmoid}(\mathbf{E}_{\text{p}}\mathbf{E}_{\text{s}}^{T}),\mathbf{\hat{M}}\in(0,1)^{m\times n}.

### 3.3 Architecture Details

For the vision encoder, we use a ResNet50 backbone and a 6-layer deformable transformer encoder [45]. The backbone is initialized with ImageNet pretrained weights, and outputs three scales of visual features. The multi-scale visual features are transformed into a sequence with 2D “sine” position embeddings before sending to the deformable transformer encoder. For the language encoder, we use a 12-layer transformer encoder with the same architecture settings as the BERT-base model [7]. In addition to BERT’s text embeddings and 1D position embeddings, we also add 2D position embeddings [37] to include layout information of the document as the input. The 2D position embeddings are learned embeddings with random initialization. The VL-decoder has 6 layers, where each layer consists of a self-attention module, a deformable cross-attention module [45] and a standard cross-attention module [34] (see supplementary material for detailed architecture of VL-decoder layers).

### 3.4 Training and Pre-training

#### Entity Extraction and Linking Objectives.

The entity extraction objective is similar to the one used in DETR [4] except that we do not need the bipartite matching due to the use of language-conditioned queries (as introduced in Sec. 3.2). Specifically, given a set of N𝑁N OCR inputs, the language-conditioned queries yields N𝑁N entity extraction outputs 𝐄^={𝐞^i}i=1N^𝐄superscriptsubscriptsubscript^𝐞𝑖𝑖1𝑁\mathbf{\hat{E}}=\{\mathbf{\hat{e}}_{i}\}_{i=1}^{N}. For a document with M𝑀M entities, we also construct a ground truth 𝐄={𝐞i}i=1N𝐄superscriptsubscriptsubscript𝐞𝑖𝑖1𝑁\mathbf{E}=\{\mathbf{e}_{i}\}_{i=1}^{N} of size N𝑁N. Here, 𝐞^isubscript^𝐞𝑖\mathbf{\hat{e}}_{i} and 𝐞isubscript𝐞𝑖\mathbf{e}_{i} denote the predicted and ground truth entities of the i𝑖ith OCR input, respectively. Note that not every OCR word is an anchor word, and thus it may have no associated entity. In this case, we say that the ground truth of the input OCR is an empty entity, i.e., 𝐞=∅𝐞\mathbf{e}=\varnothing, and there are in total N−M𝑁𝑀N-M empty entities in 𝐄𝐄\mathbf{E}. If we denote a non-empty entity as 𝐞=(c,b)𝐞𝑐𝑏\mathbf{e}=(c,b) and a predicted entity as 𝐞^=(p^,b^)^𝐞^𝑝^𝑏\mathbf{\hat{e}}=(\hat{p},\hat{b}), where c𝑐c is the ground truth entity label, p^^𝑝\hat{p} is the predicted entity label probability, and b𝑏b/b^^𝑏\hat{b} is the ground truth/predicted bounding box, then we write the entity extraction loss as

ℒEE​(𝐄,𝐄^)=∑i[−log⁡p^i​(ci)+λ​𝟙{𝐞i≠∅}​ℒbbox​(bi,b^i)],subscriptℒEE𝐄^𝐄subscript𝑖delimited-[]subscript^𝑝𝑖subscript𝑐𝑖𝜆subscript1subscript𝐞𝑖subscriptℒbboxsubscript𝑏𝑖subscript^𝑏𝑖\mathcal{L}_{\text{EE}}(\mathbf{E},\mathbf{\hat{E}})=\sum_{i}[-\log{\hat{p}_{i}(c_{i})}+\lambda\mathds{1}_{\{\mathbf{e}_{i}\neq\varnothing\}}\mathcal{L}_{\text{bbox}}(b_{i},\hat{b}_{i})],

where p^i​(ci)subscript^𝑝𝑖subscript𝑐𝑖\hat{p}_{i}(c_{i}) is the predicted probability of entity being labeled as cisubscript𝑐𝑖c_{i}, ℒbboxsubscriptℒbbox\mathcal{L}_{\text{bbox}} is a bounding box loss [4], and 𝟙{𝐞i≠∅}subscript1subscript𝐞𝑖\mathds{1}_{\{\mathbf{e}_{i}\neq\varnothing\}} means we only compute ℒbboxsubscriptℒbbox\mathcal{L}_{\text{bbox}} for non-empty entities.

The entity linking loss consists of two parts, primary anchor classification and linking classification. Let 𝐋^^𝐋\mathbf{\hat{L}} be a set of primary anchor classification outputs and 𝐋𝐋\mathbf{L} be its binary ground truth labels. Let 𝐌^^𝐌\mathbf{\hat{M}} and 𝐌𝐌\mathbf{M} be the predicted and ground truth entity linking affinity matrices, respectively. Then, we can simply write the entity linking loss as

ℒEL​(𝐋,𝐋^,𝐌,𝐌^)=BCE​(𝐋,𝐋^)+β​BCE​(𝐌,𝐌^),subscriptℒEL𝐋^𝐋𝐌^𝐌BCE𝐋^𝐋𝛽BCE𝐌^𝐌\mathcal{L}_{\text{EL}}(\mathbf{L},\mathbf{\hat{L}},\mathbf{M},\mathbf{\hat{M}})=\text{BCE}(\mathbf{L},\mathbf{\hat{L}})+\beta\text{BCE}(\mathbf{M},\mathbf{\hat{M}}),

(3)

where BCE denotes the binary cross-entropy loss.

#### Pre-training.

We pre-train DocTr on a large-scale dataset of unlabeled document images. For simplicity of modeling, we only include one pre-training task, termed masked detection modeling (MDM), for DocTr which we find sufficient for downstream tasks. Since pre-training is not the main focus of this work, we leave the exploration of other pre-training strategies [38, 11, 1] for future work. Fig. 4 illustrates MDM and compares it with related pre-training tasks. MDM is an extension of masked vision-language modeling (MVLM) [37, 38]. Both MDM and MVLM take OCR text and boxes as input. However, MVLM only randomly masks the text inputs. Instead, MDM randomly masks both the text inputs and their boxes. Specifically, we replace text with [MASK] and set boxes to [0, 0, 0, 0]. Then, we train DocTr to predict both the masked texts and their corresponding boxes. Note that this task is similar to object detection. Thus, the objective function can be written in the same way as Eq. (2), where the first term is for masked text classification, and the second term is for masked box regression. Also note that for MDM, the input image is not masked so that a model can better learn how to leverage the visual information to locate and identify the masked inputs.

## 4 Experiments

#### Datasets and Tasks.

We use three datasets in our experiments, IIT-CDIP document collection [21], CORD [27] and FUNSD [16]. We follow the convention in the literature [37, 38, 11, 1] to pre-train DocTr on the IIT-CDIP document collection, which is a large-scale dataset with 11 million unlabeled documents. CORD [27] is a receipt dataset with 800 training, 100 validation, and 100 testing samples. Each receipt in this dataset is labeled with a list of line items and key-value pair groups. FUNSD [16] consists of scanned forms, with 149 training and 50 testing examples. Each form is labeled with key/value entities together with links to indicate which keys and values are associated.

We evaluate our model’s performance on three tasks, receipts parsing, entity labeling and entity linking. For receipt parsing, a model not only has to extract each receipt’s entities but also correctly link entities to form line items and key-value pair groups. Fig. 5 (a) shows a sample receipt from CORD and its expected output after parsing. The sample contains two line items and four key-value pairs. For line items, it requires identifying each line item related entity (class and text) and group the entities of the same line item together. For key-value pairs, we identify class labels of the keys and return only text of the corresponding values. We use the same evaluation protocols and metrics as defined in [15] to evaluate the receipt parsing performance.

Entity labeling and entity linking are commonly adopted tasks [37, 15] to evaluate a pre-trained model’s performance, which however are simplified versions of what we have defined in Sec. 3.1. Entity labeling requires assigning a class label to each word of the document. Fig. 5 (b) shows a sample from FUNSD where the task is to identify if a word belongs to a key (red), a value (green) or a title (blue). In entity linking, the assumption is that the key/value entities are correctly detected, and the task is to identify which keys and values should be linked (See Fig. 5 (c), red arrows). We evaluate entity labeling/linking by checking if the words/links are correctly labeled using F1-score as the metric.

### 4.1 Comparison with Existing Solutions

We compare DocTr with the existing methods on receipts parsing, entity labeling, and entity linking tasks, respectively.

For receipts parsing, SPADE [15] and Donut [19] are the only two other publicly available solutions (to the best of our knowledge) that address this task on CORD. The other existing general-purpose models [38, 11, 13] are not able to directly address this structured information extraction task out-of-the-box. For a fair comparison with our method, we fine-tune the officially released general-purpose models under two settings: using the standard IOB tagging for receipts parsing or using our proposed formulation. From Table 2, we can see that DocTr outperforms general-purpose models BROS, LayoutLMv2 and LayoutLMv3 by a noticeable margin when they are fine-tuned with the IOB tagging setting. When fine-tuned with our proposed formulation, the general-purpose models’ performance improved but they are still behind DocTr, which shows the effectiveness of the proposed encoder-decoder solution for the anchor word based structure information extraction.

For entity labeling, we follow the general-purpose models [37] to only fine-tune DocTr for IOB tagging and evaluate based on its IOB tagging outputs. We note that this is less favorable for DocTr since the architecture and is dedicated to address our new formulation, and the pre-training strategy is not a main focus of this paper. However, we observe DocTr noticeably outperforming the existing solutions with comparable model sizes (“Base” models in Table 3). Even when compared with larger pre-trained models, DocTr’s performance is comparable or better on the CORD dataset.

For entity linking, we apply the objective introduced in Eq. (3) to train our model to link keys and values in FUNSD documents. We remove the entity extraction loss (Eq. (2)) but use ground truth entities as per the task definition. The results are shown in Table 2 – DocTr also outperforms the existing solutions by a noticeable margin in this task.

model
F1

Donut [19]††\dagger

87.8

SPADE [15]

92.5

LayoutLMv2 [38] w/ IOB

91.4

BROS [11] w/ IOB

91.8

LayoutLMv3 [13] w/ IOB

92.2

LayoutLMv2 [38] w/ ours

92.7

BROS [11] w/ ours

92.9

LayoutLMv3 [13] w/ ours

93.6

DocTr (ours)
94.4

model
F1

SPADE [15]

41.7

BROS [11]

71.5

StructText [22]

44.1

DocTr (ours)
73.9

††\daggerWe take the official model from [19] and report numbers using the metric from [15].

model
FUNSD
CORD
#params

SPADE [15]

71.6
-
-

LayoutLMBASEsubscriptLayoutLMBASE\textrm{LayoutLM}_{\rm BASE} [37]

78.7
94.7
113M

BROSBASEsubscriptBROSBASE\textrm{BROS}_{\rm BASE} [11]

83.1
96.5
110M

DocFormerBASEsubscriptDocFormerBASE\textrm{DocFormer}_{\rm BASE} [11]

83.3
96.3
183M

LayoutLMv2BASEsubscriptLayoutLMv2BASE\textrm{LayoutLMv2}_{\rm BASE} [38]

82.8
95.0
200M

StructText [22]

83.4
-
107M

DocTr​(o​u​r​s)DocTr𝑜𝑢𝑟𝑠\textrm{\bf DocTr}(ours)
84.0
98.2
153M

LayoutLMLARGEsubscriptLayoutLMLARGE\textrm{LayoutLM}_{\rm LARGE} [37]

79.0
95.0
343M

BROSLARGEsubscriptBROSLARGE\textrm{BROS}_{\rm LARGE} [11]

84.5
97.3
340M

DocFormerLARGEsubscriptDocFormerLARGE\textrm{DocFormer}_{\rm LARGE} [11]

84.5
97.0
536M

LayoutLMv2LARGEsubscriptLayoutLMv2LARGE\textrm{LayoutLMv2}_{\rm LARGE} [38]

84.2
96.0
426M

FormNet [20]

84.7
97.3
345M

formulation
text serial.
parsing (C)

IOB tagging [29]

raster scan
93.2

SPADE [15]

raster scan
93.0

DocTr (ours)
raster scan
94.4

IOB tagging [29]

oracle
94.1

SPADE [15]

oracle
93.9

DocTr (ours)
oracle
95.0

### 4.2 Model Properties

We analyze DocTr’s design and consider other choices.

#### Problem Formulation.

We use DocTr as the backbone network for the encoding of document inputs (image and OCR words) and apply different formulations to decode structured information. Specifically, we compare our formulation with IOB tagging and graph based solutions. For IOB tagging, we follow the literature [20, 37] and assign BIOES tags to each token and decode entities according to the tagged entity spans. Note that IOB tagging does not support entity linking. For a fair comparison, we link entities using a way similar to the anchor word association method introduced in Sec. 3.2. We treat the “B” tag or “S” tag of entities as the anchor words and link entities via decoding of entity linking affinity matrices. For graph based SIE, we follow the literature [11, 15] by attaching a SPADE [15] decoder at the end of DocTr. We fine-tune DocTr and decode graphs using the same way as specified in the original SPADE method. To understand the sensitivity of the SIE formulations with regard to the reading orders of input text, we evaluate them under two text serialization settings, raster scan and oracle. For oracle, we first order the ground truth entities in a raster scan manner, then order text while preserving the entity order.

Table 4 shows the receipt parsing results on the CORD dataset. Our proposed formulation achieves the best performance in both text serialization settings. We notice that, compared with the other two formulations, our formulation is less sensitive to text serialization with only 0.6 score drop (vs. 0.9 drop by IOB tagging or SPADE) while switching from oracle to raster scan text serialization. We also observe that our formulation can better address cases where there is dense text with multiple entities near each other. Fig. 6 shows an example visualization (see supplementary material for more results). For IOB tagging, it can tag most of the words well. However, even a single tagging error can cause failures of entity decoding, and an entity is missed from the parsing outputs. For SPADE, the dense words result in a challenge for constructing an entity graph, and the model incorrectly merges the two sub_nm’s as a single entity. In comparison, DocTr only requires identifying the anchor words which is an easier task and, with bounding box predictions, all the entities are correctly extracted.

anchor word
primary anchor
parsing (C)

first
name
94.2

last
name
94.1

first + last
first
94.0

first + last
name
94.4

#### Anchor Word and Primary Anchor.

We investigate different ways of designating anchor word and primary anchor. In Sec. 3.1, we introduced using the first word (in terms of reading order) of an entity as the anchor word. Here, we consider two alternatives: 1) using the last word or 2) both the first and last word as the anchor. Table 5 (row 1-2, 4) shows the comparison of these three choices. We notice that there is no significant differences (94.2 vs. 94.1) between using the first word and last word as the anchor word. Using both first and last as the anchor word gives slightly better performance. We hypothesize that this is because first and last words help better identify the boundary of an entity.

For primary anchor, we investigate its choices for line-item extraction. We consider two candidates: 1) using the anchor word of the first entity in a line-item, or 2) using the anchor word of name as the primary anchor. From Table 5 (row 3 and 4), we see the latter is a better choice with 0.4 improvement. This is reasonable since the first entity in a line-item may vary semantically (i.e., it could be name, cnt or other entity types), and thus it is harder to identify. However, this choice is also more flexible than using name as the primary anchor because there may be no name in an line-item. For CORD, each line-item always has a name, so this is not a concern (see supplementary material for primary anchor choices of other entity categories).

pre-training
parsing (C)
ELB (F)
ELK (F)

none
82.3
14.2
12.0

MVLM [37]

90.9
82.7
73.0

MDM
94.4
84.0
73.9

#### Pre-training.

We evaluate the effectiveness of the pre-training task (MDM) introduced in Sec. 3.4. We consider three settings: 1) without pre-training, 2) with MVLM and 3) with MDM. Table 6 compares their performances. Without pre-training, the performance drops significantly. With MVLM, the performance improves but still falls behind using MDM. This shows the effectiveness of having MDM for document understanding pre-training. In particular, we see more benefit of using MDM for receipt parsing. This is because our proposed formulation requires bounding box regression, and MDM helps learn better box predictions.

We also show example MDM pre-training predictions in Figure 7. Note that since both the input OCR box and text are masked, the model will need to not only predict what is masked but also predict where to find the masked word. We can see in most of the cases, the model can predict both kinds of information well. There are cases where the box (e.g., “25,000” in row 1) or text (e.g., “Bun” in row 2) is not accurately predicted. But the errors are reasonable. We also notice that the model can predict words that cannot be inferred through only text context, such as prices. This shows the usage of visual information.

vis. enc.
VL-dec.
LCQ
parsing (C)
ELB (F)
ELK (F)

1)

N/A
92.3
82.1
73.2

2)

✓
✓
92.6
83.0
73.3

3)
✓
✓

90.7
14.9
9.5

4)
✓
✓
✓
94.4
84.0
73.9

#### Architecture Design.

We then ablate the impact of the architectural components of DocTr. Table 7 shows the ablation results. The first row is a DocTr model with only the language encoder which is equivalent to the LayoutLM [37] model without visual inputs. The second row is a model with both the language encoder and VL-decoder but no vision encoder. These two models are close in performance. This is reasonable as without visual inputs the VL-decoder does not add much of information for decoding. Row 4 is the full model with both the vision encoder and VL-decoder. Compared with row 1 and 2, the performance improves noticeably. This suggests the importance of using visual information.

For row 3 and 4, we study the effectiveness of using the proposed language conditioned queries (LCQ). Specifically, we apply Eq. (1) to the cross-attention module when LCQ is checked. Otherwise, the standard cross-attention is used. We can see that LCQ is important since it helps to guide this one-to-one mapping between OCR and outputs, which is required by our proposed formulation.

## 5 Conclusion

We have presented a new approach for SIE from visually-rich documents. This approach is based on our novel formulation which includes object detection as part of the problem setting. This naturally leads us to include a transformer-based object detector as part of the architecture design and an object detection based loss in pre-training.

We have empirically shown that our proposed object detection based formulation readily addresses the structured information extraction task, and our solution outperforms existing solutions on SIE benchmarks. We hope this approach will initiate more efforts in combining object detection with existing vision-language models for document intelligence.

We also note that using anchor words limits the application of this approach to text-rich documents, and text-based entity extraction only. For future work, we explore solutions that extend the propose formulation for the extraction of non-textual content (e.g., symbols, logos, etc.) from documents.

## References

- [1]

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R
Manmatha.

Docformer: End-to-end transformer for document understanding.

In Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 993–1003, 2021.

- [2]

Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, and R Manmatha.

Latr: Layout-aware transformer for scene-text vqa.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 16548–16558, 2022.

- [3]

Claire Cardie.

Empirical methods in information extraction.

AI magazine, 18(4):65–65, 1997.

- [4]

Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander
Kirillov, and Sergey Zagoruyko.

End-to-end object detection with transformers.

In European conference on computer vision, pages 213–229.
Springer, 2020.

- [5]

Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei.

Document ai: Benchmarks, models and applications.

arXiv preprint arXiv:2111.08609, 2021.

- [6]

Timo I Denk and Christian Reisswig.

Bertgrid: Contextualized embedding for 2d document representation and
understanding.

arXiv preprint arXiv:1909.04948, 2019.

- [7]

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

BERT: pre-training of deep bidirectional transformers for language
understanding.

In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies,
NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and
Short Papers), pages 4171–4186. Association for Computational Linguistics,
2019.

- [8]

Dayne Freitag.

Machine learning for information extraction in informal domains.

Machine learning, 39(2):169–202, 2000.

- [9]

Chenyu Gao, Qi Zhu, Peng Wang, Hui Li, Yuliang Liu, Anton Van den Hengel, and
Qi Wu.

Structured multimodal attentions for textvqa.

IEEE Transactions on Pattern Analysis and Machine Intelligence,
44(12):9603–9614, 2021.

- [10]

Łukasz Garncarek, Rafał Powalski, Tomasz Stanisławek, Bartosz
Topolski, Piotr Halama, Michał Turski, and Filip Graliński.

Lambert: Layout-aware language modeling for information extraction.

In International Conference on Document Analysis and
Recognition, pages 532–547. Springer, 2021.

- [11]

Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, and Sungrae
Park.

Bros: A pre-trained language model focusing on text and layout for
better key information extraction from documents.

In Proceedings of the AAAI Conference on Artificial
Intelligence, pages 10767–10775, 2022.

- [12]

Ronghang Hu, Amanpreet Singh, Trevor Darrell, and Marcus Rohrbach.

Iterative answer prediction with pointer-augmented multimodal
transformers for textvqa.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 9992–10002, 2020.

- [13]

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei.

Layoutlmv3: Pre-training for document ai with unified text and image
masking.

arXiv preprint arXiv:2204.08387, 2022.

- [14]

Wonseok Hwang, Seonghyeon Kim, Minjoon Seo, Jinyeong Yim, Seunghyun Park,
Sungrae Park, Junyeop Lee, Bado Lee, and Hwalsuk Lee.

Post-ocr parsing: building simple and robust parser via bio tagging.

In Workshop on Document Intelligence at NeurIPS 2019, 2019.

- [15]

Wonseok Hwang, Jinyeong Yim, Seunghyun Park, Sohee Yang, and Minjoon Seo.

Spatial dependency parsing for semi-structured document information
extraction.

In Findings of the Association for Computational Linguistics:
ACL-IJCNLP 2021, pages 330–343, 2021.

- [16]

Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran.

Funsd: A dataset for form understanding in noisy scanned documents.

In 2019 International Conference on Document Analysis and
Recognition Workshops (ICDARW), volume 2, pages 1–6. IEEE, 2019.

- [17]

Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael
Bernstein, and Li Fei-Fei.

Image retrieval using scene graphs.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 3668–3678, 2015.

- [18]

Anoop Raveendra Katti, Christian Reisswig, Cordula Guder, Sebastian Brarda,
Steffen Bickel, Johannes Höhne, and Jean Baptiste Faddoul.

Chargrid: Towards understanding 2d documents.

arXiv preprint arXiv:1809.08799, 2018.

- [19]

Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong
Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park.

Ocr-free document understanding transformer.

In European Conference on Computer Vision, pages 498–517.
Springer, 2022.

- [20]

Chen-Yu Lee, Chun-Liang Li, Timothy Dozat, Vincent Perot, Guolong Su, Nan Hua,
Joshua Ainslie, Renshen Wang, Yasuhisa Fujii, and Tomas Pfister.

Formnet: Structural encoding beyond sequential modeling in form
document information extraction.

In Proceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 3735–3754, 2022.

- [21]

David Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David Grossman, and
Jefferson Heard.

Building a test collection for complex document information
processing.

In Proceedings of the 29th annual international ACM SIGIR
conference on Research and development in information retrieval, pages
665–666, 2006.

- [22]

Yulin Li, Yuxi Qian, Yuechen Yu, Xiameng Qin, Chengquan Zhang, Yan Liu, Kun
Yao, Junyu Han, Jingtuo Liu, and Errui Ding.

Structext: Structured text understanding with multi-modal
transformers.

In Proceedings of the 29th ACM International Conference on
Multimedia, pages 1912–1920, 2021.

- [23]

Xiaojing Liu, Feiyu Gao, Qiong Zhang, and Huasha Zhao.

Graph convolution for multimodal information extraction from visually
rich documents.

arXiv preprint arXiv:1903.11279, 2019.

- [24]

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.

Roberta: A robustly optimized bert pretraining approach.

arXiv preprint arXiv:1907.11692, 2019.

- [25]

Ilya Loshchilov and Frank Hutter.

Decoupled weight decay regularization.

In International Conference on Learning Representations, 2018.

- [26]

Ahmed Nassar, Nikolaos Livathinos, Maksym Lysak, and Peter Staar.

Tableformer: Table structure understanding with transformers.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 4614–4623, 2022.

- [27]

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo,
and Hwalsuk Lee.

Cord: a consolidated receipt dataset for post-ocr parsing.

In Workshop on Document Intelligence at NeurIPS 2019, 2019.

- [28]

Devashish Prasad, Ayan Gadpal, Kshitij Kapadni, Manish Visave, and Kavita
Sultanpure.

Cascadetabnet: An approach for end to end table detection and
structure recognition from image-based documents.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition workshops, pages 572–573, 2020.

- [29]

Lance A Ramshaw and Mitchell P Marcus.

Text chunking using transformation-based learning.

In Natural language processing using very large corpora, pages
157–176. Springer, 1999.

- [30]

Ellen Riloff.

Automatically generating extraction patterns from untagged text.

In Proceedings of the national conference on artificial
intelligence, pages 1044–1049, 1996.

- [31]

Sebastian Schreiber, Stefan Agne, Ivo Wolf, Andreas Dengel, and Sheraz Ahmed.

Deepdesrt: Deep learning for detection and structure recognition of
tables in document images.

In 2017 14th IAPR international conference on document analysis
and recognition (ICDAR), volume 1, pages 1162–1167. IEEE, 2017.

- [32]

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv
Batra, Devi Parikh, and Marcus Rohrbach.

Towards vqa models that can read.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 8317–8326, 2019.

- [33]

Kaihua Tang, Yulei Niu, Jianqiang Huang, Jiaxin Shi, and Hanwang Zhang.

Unbiased scene graph generation from biased training.

In Proceedings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 3716–3725, 2020.

- [34]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

Advances in neural information processing systems, 30, 2017.

- [35]

Mengxi Wei, Yifan He, and Qiong Zhang.

Robust layout-aware ie for visually rich documents with pre-trained
language models.

In Proceedings of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 2367–2376, 2020.

- [36]

Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei.

Scene graph generation by iterative message passing.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5410–5419, 2017.

- [37]

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou.

Layoutlm: Pre-training of text and layout for document image
understanding.

In Proceedings of the 26th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining, pages 1192–1200, 2020.

- [38]

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu,
Dinei Florencio, Cha Zhang, Wanxiang Che, et al.

Layoutlmv2: Multi-modal pre-training for visually-rich document
understanding.

In Proceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers), pages 2579–2591, 2021.

- [39]

Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, and Devi Parikh.

Graph r-cnn for scene graph generation.

In Proceedings of the European conference on computer vision
(ECCV), pages 670–685, 2018.

- [40]

Xiao Yang, Ersin Yumer, Paul Asente, Mike Kraley, Daniel Kifer, and C
Lee Giles.

Learning to extract semantic structure from documents using
multimodal fully convolutional neural networks.

In Proceedings of the IEEE Conference on Computer Vision and
Pattern Recognition, pages 5315–5324, 2017.

- [41]

Wenwen Yu, Ning Lu, Xianbiao Qi, Ping Gong, and Rong Xiao.

Pick: processing key information extraction from documents using
improved graph learning-convolutional networks.

In 2020 25th International Conference on Pattern Recognition
(ICPR), pages 4363–4370. IEEE, 2021.

- [42]

Rowan Zellers, Mark Yatskar, Sam Thomson, and Yejin Choi.

Neural motifs: Scene graph parsing with global context.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 5831–5840, 2018.

- [43]

Peng Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Jing Lu, Liang Qiao, Yi Niu,
and Fei Wu.

Trie: end-to-end text reading and information extraction for document
understanding.

In Proceedings of the 28th ACM International Conference on
Multimedia, pages 1413–1422, 2020.

- [44]

Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, and Nancy Xin Ru Wang.

Global table extractor (gte): A framework for joint table
identification and cell structure recognition using visual context.

In Proceedings of the IEEE/CVF winter conference on applications
of computer vision, pages 697–706, 2021.

- [45]

Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai.

Deformable detr: Deformable transformers for end-to-end object
detection.

In International Conference on Learning Representations, 2020.

## Appendix A Appendix

### A.1 Implementation Details

The hyperparameters we used for pre-training and fine-tuning downstream tasks are shown in Table 8. In general, our hyperparameter settings are similar to the one
used in Deformable DETR [45]. Here, “base” configurations are those common for all experiments. It is worth mentioning that we use the box refinement design proposed in [45] which we find helpful for bounding box prediction.

For “pre-training”, we use a three-stage pre-training. That is, in the first stage, we pre-train on 1M IIT-CDIP samples [21] for 20 epochs. Then, we pre-train on 5M samples for 5 epochs. In the final stage, we pre-train on 11M samples (full dataset) for 2 epochs.

“receipt parsing”, “entity labeling” and “entity linking” show the settings we used to obtain the numbers we reported in Table 1 and 2 of the main manuscript. For “entity extraction”, since we follow the existing work to address this task with IOB tagging (for fair comparison), we do not apply Eq. (2) and Eq. (3) (in main manuscript) as the loss function. But instead we simply use a cross-entry loss. Here “CE loss weight” is the weight of this loss.

config type
config name
value

base
optimizer

AdamW [25]

base LR
2​e−42superscript𝑒42e^{-4}

cnn LR
2​e−52superscript𝑒52e^{-5}

language encoder LR.
1​e−51superscript𝑒51e^{-5}

weight decay
1​e−41superscript𝑒41e^{-4}

LR schedule
step

box refinement [45]

yes

pre-training
batch size
32

epochs
20, 5, 2

training samples
1M, 5M, 11M

LR drop step size
20

receipt parsing
batch size
8

epochs
200

LR drop step size
160

EE loss weight
5.0

EL loss weight
1.0

anchor word
first+last

primary anchor
name

entity labeling
batch size
8

epochs
50

LR drop step size
40

CE loss weight
2.0

entity linking
batch size
8

epochs
200

LR drop step size
160

EE loss weight
0.0

EL loss weight
10.0

### A.2 VL Decoder Layer

The VL-decoder has 6 layers, where each layer consists of a self-attention module, a deformable cross-attention module [45] and a standard cross-attention module [34] as shown Fig. 8. Let 𝐐∈ℝL×D𝐐superscriptℝ𝐿𝐷\mathbf{Q}\in\mathbb{R}^{L\times D} be the input decoder queries. We first reshape 𝐐𝐐\mathbf{Q} to 𝐐′∈ℝ2​L×D2superscript𝐐′superscriptℝ2𝐿𝐷2\mathbf{Q^{\prime}}\in\mathbb{R}^{2L\times\frac{D}{2}}, where L𝐿L is the number of input queries and D𝐷D is the channel size. After the self-attention module, we split 𝐐′superscript𝐐′\mathbf{Q^{\prime}} into two equally sized queries 𝐐v∈ℝL×D2superscript𝐐𝑣superscriptℝ𝐿𝐷2\mathbf{Q}^{v}\in\mathbb{R}^{L\times\frac{D}{2}} and 𝐐l∈ℝL×D2superscript𝐐𝑙superscriptℝ𝐿𝐷2\mathbf{Q}^{l}\in\mathbb{R}^{L\times\frac{D}{2}}. The vision queries 𝐐vsuperscript𝐐𝑣\mathbf{Q}^{v} extract visual features via deformable cross-attention. The language queries 𝐐lsuperscript𝐐𝑙\mathbf{Q}^{l} extract language features via cross-attention, where we apply Eq. (1) in the main text to assign explicit language semantics to queries. The outputs from the two cross-attention modules are concatenated at the channel dimension to recover the original shape, i.e., 𝐐v​l∈ℝL×D=concat​(𝐐v,𝐐l)superscript𝐐𝑣𝑙superscriptℝ𝐿𝐷concatsuperscript𝐐𝑣superscript𝐐𝑙\mathbf{Q}^{vl}\in\mathbb{R}^{L\times D}=\text{concat}(\mathbf{Q}^{v},\mathbf{Q}^{l}). We use a fully connected layer at the end to further fuse vision and language information along the channel dimension.

### A.3 Additional Results

#### Comparison of using anchor words from different
line-item fields as primary anchors.

From Table 9, we can see that using the anchor word of “name” gives the best result. This is because in the test set, all the line-items contain this field. So, it is reliable to use this field as the primary anchor for entity linking. For the other fields, their performances are lower when they do not frequently present in line-items. Also, note that the middle column only considers line-items not key-values. Therefore, it is possible that the parsing performance numbers are even higher than the proportion of line-items that contain this field.

primary anchor
LI’s with this field
parsing (CORD)

unit price
28%
66.5

count
90%
91.4

price
99%
94.0

name
100%
94.4

first
-
94.0

#### Comparison of using predicted text and ground truth text as inputs.

The existing works [15, 11, 37, 5] use ground truth text as the input in the experiments. We also follow the same way for fair comparison. However, it would be interesting to see how the models work if predicted text (from an OCR system) is used. In particular, using ground truth text as input is not in favor of vision only approaches such as Donut [19]. Table 10 shows the results of the comparison. In this experiment, we use an in-house OCR system which has comparable performances with the state-of-the-art OCR solutions (e.g., those from Azure, GCP or AWS). As we can see, there is a performance drop when we switch from using ground truth text to predicted text. However, compared with the vision only solution Donut, we are still noticeably better. This indicates the importance of having language inputs.

#### Model performance at different pre-training stages.

In Table 11, each stage is based on the pre-trained model from its previous stage. For example, stage 2 initializes the model using the weights pre-trained from stage 1. As we can see, when more data is used, the model’s performance continues improving on the FUNSD entity extraction task.

#### Visualization of language-conditioned cross-attention.

We further verify the behavior of this cross-attention mechanism by visualizing the cross-attention matrices. The cross-attention results are extracted from the cross-attention module of each decoder layer, i.e., we check the cross-attention between the language inputs and the language-conditioned queries. As we can see in Fig 9, the attention weights are high on the diagonal of the attention matrices. This shows that we successfully established this one-to-one mapping between the queries (and thus decoder outputs) and language tokens.

#### Additional visualizations.

Fig 10-13 shows additional comparisons of three structured information extraction formulations. Fig 14 shows two receipts parsing failure cases. Fig 15-18 shows additional pre-training outputs using our proposed masked detection modeling task.

model
input text
parsing (CORD)

Donut [19]

none
87.8

SPADE [15]

gt
92.5

LayoutLMv2 [38]

pred.
92.2

BROS [11]

pred.
92.1

LayoutLMv3 [13]

pred.
93.0

DocTr (ours)
pred.
93.7

LayoutLMv2 [38]

gt
92.7

BROS [11]

gt
92.9

LayoutLMv3 [13]

gt
93.6

DocTr (ours)
gt
94.4

stage
# samples
# epochs
EE (FUNSD)

1
1M
20 epochs
82.1

2
5M
5 epochs
83.1

3
11M
2 epochs
84.0

Generated on Mon Feb 26 21:55:29 2024 by LaTeXML
