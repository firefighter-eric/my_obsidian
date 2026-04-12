# Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Ouyang et al. - Unknown - OmniDocBench Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2412.07626
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations

Linke Ouyang1∗ Yuan Qu1∗ Hongbin Zhou1∗ Jiawei Zhu1∗ Rui Zhang1∗ Qunshu Lin2∗ 
Bin Wang1∗† Zhiyuan Zhao1
Man Jiang1
Xiaomeng Zhao1
Jin Shi1
Fan Wu1
Pei Chu1
Minghao Liu3
Zhenxiang Li1
Chao Xu1
Bo Zhang1
Botian Shi1
Zhongying Tu1
Conghui He1‡
1Shanghai AI Laboratory 2Abaka AI 32077AI

###### Abstract

Document content extraction is crucial in computer vision, especially for meeting the high-quality data needs of large language models (LLMs) and retrieval-augmented generation (RAG) technologies. However, current document parsing methods suffer from significant limitations in terms of diversity and comprehensive evaluation. To address these challenges, we introduce OmniDocBench, a novel multi-source benchmark designed to advance automated document content extraction. OmniDocBench includes a meticulously curated and annotated high-quality evaluation dataset comprising nine diverse document types, such as academic papers, textbooks, slides, among others. Our benchmark provides a flexible and comprehensive evaluation framework with 19 layout category labels and 14 attribute labels, enabling multi-level assessments across entire datasets, individual modules, or specific data types. Using OmniDocBench, we perform an exhaustive comparative analysis of existing modular pipelines and multimodal end-to-end methods, highlighting their limitations in handling document diversity and ensuring fair evaluation. OmniDocBench establishes a robust, diverse, and fair evaluation standard for the document content extraction field, offering crucial insights for future advancements and fostering the development of document parsing technologies. The codes and dataset is available in https://github.com/opendatalab/OmniDocBench.

## 1 Introduction

Document parsing is a foundational task in computer vision, focused on accurately extracting content from documents [18, 36, 39, 41, 45]. High-quality document content extraction typically involves the integration of multiple algorithmic modules. Layout detection algorithms identify different content areas on a page, OCR technology converts images of text regions into text, while formula and table recognition models identify specific regions and transform them into corresponding source code. These modules and reading order algorithms form a comprehensive process of converting documents into machine-readable formats.

With large models increasingly requiring high-quality data, the importance of document content extraction has become more pronounced. Although vast amounts of data are available online for training, knowledge-rich document data is relatively scarce. Documents such as academic papers and technical reports contain rich structured information that can significantly enhance the knowledge depth of large models. Moreover, the development of retrieval-augmented generation (RAG) [21, 10] technology relies on extracting accurate information from documents to improve the quality and relevance of generated content. Consequently, research in document content extraction has intensified, leading to a series of pipeline-based high-quality document extraction algorithms [36] and the emergence of end-to-end multimodal large model solutions [5, 40, 27, 39, 3, 6, 42]. These methods have significantly improved document content parsing quality, providing robust support for the needs of large models and RAG technology.

In analyzing current module-based pipeline and multimodal end-to-end methods, we identified several limitations. For instance, methods like Marker and MinerU, which are mainstream pipeline methods, primarily evaluate individual modules on academic paper data, lacking document diversity and comprehensive evaluation results. Although MinerU considers the generalization of diverse data, it only demonstrates this through a single model and visualization results, lacking overall end-to-end evaluation. Multimodal large model methods [5, 40, 27, 39, 3], while easier to use than pipeline methods, lack performance validation on diverse documents, and some evaluation metrics are inadequate. Additionally, these methods often use data similar to their training set distribution for comparison, resulting in unfair evaluations. Overall, current document content extraction faces the following challenges:

- •

Limited document types. Current evaluations mostly focus on a single type of academic paper, while real-world scenarios include textbooks, exam papers, financial reports, newspapers, magazines, and other document types.

- •

Monotonous evaluation dimensions. Pipeline-based methods typically evaluate specific algorithmic modules, such as OCR, layout detection, or formula recognition, while the overall quality of parsing results requires comprehensive metrics.

- •

Inadequate evaluation metrics. Multimodal large model approaches attempt to evaluate document parsing quality across multiple dimensions, such as dividing document content into text, formulas, tables, etc. However, these models commonly employ evaluation metrics such as BLEU scores or Edit distances, which fail to accurately and fairly assess parsing effectiveness when dealing with markup languages like LaTeX or HTML that allow diverse syntactic expressions.

Building a diverse, comprehensive, and accurate evaluation system poses significant challenges, requiring diverse and high-quality data annotation and reasonable evaluation metrics. While READOC extends the evaluation scope to include GitHub README files based on Nougat, there remains a substantial gap in real-world diversity, and the evaluation dimensions lack consideration of attributes. In contrast, this paper proposes a document content extraction benchmark, OmniDocBench, characterized by diverse types, detailed annotations, and reasonable evaluation (Figure 1). The specific contributions are as follows:

- •

High-quality, diverse evaluation set: Through automated annotation, manual verification, and expert review, we construct a comprehensive, detailed, high-quality OmniDocBench evaluation set, encompassing nine types of diverse document pages, including papers, textbooks, exam questions, and research reports.

- •

Flexible and comprehensive evaluation dimension support: The OmniDocBench validation set covers 19 layout category labels and 14 attribute labels. To facilitate user evaluation from an overall, single module, or different data types, we provide end-to-end evaluation, single algorithm module evaluation, and attribute-based evaluation, covering various evaluation needs.

- •

Comprehensive evaluation of mainstream methods: Based on OmniDocBench, we conduct a comprehensive evaluation of current mainstream modular pipeline and end-to-end large model methods, providing a fairer assessment of existing methods and summarizing the shortcomings of current document parsing methods, thereby guiding further development in document parsing.

## 2 Related Work

Benchmark

Document

Categories

BBox
Text
Table
Order
Formula
OCR
DLA
TR
MFR
OCR
ROD
TR
MFR

Single-Module Eval Benchmark

Robust Reading [19]

1
✔

✔

PubLayNet [43], DocBank [24],

DocLayNet [31], M6Doc [7]

PubTabNet [47],TableX [9],

TableBank [23]

Im2Latex-100K [8],UniMER-Test [34]

1

✔

✔

End-to-end Eval Benchmarks

Nougat [5]

1

✔

✔
✔
✔
✔

Fox [27]

2

✔

✔

GOT OCR 2.0 [39]

2

✔

✔

✔
✔

READoc [26]

2

✔
✔
✔
✔

✔
✔
✔
✔

OmniDocBench
9
✔
✔
✔
✔
✔
✔
✔
✔
✔
✔
✔
✔
✔

### 2.1 Traditional Document Content Extraction

Document Content extraction remains a challenging task, and there is yet to emerge a unified benchmark tailored for real-world scenarios. Traditional algorithms typically employ multiple expert modules to handle different extraction subtasks, such as document layout detection [16, 46, 32, 11], optical character recognition (OCR) [33, 22, 28, 14, 37], formula recognition [44, 25, 4, 34], and table recognition [15, 17, 22].

While expert models of these subtasks are advancing rapidly, recent work such as Mineru [36] attempts to concatenate multiple expert modules into a pipeline and provides a high-precision open-source solution for document content extraction. READOC [26] also unifies heterogeneous evaluation methods from the perspective of Document Structure Extraction, breaking down texts, images, formulas, tables, and other dimensions for evaluation, thus offering a solution-oriented towards real-world scenarios for DSE tasks. However, due to the complexity of Document data sources and the intricacies of PDF document information, previous efforts still fall short in terms of data diversity, failing to cover the categories users encounter in practical applications. Similarly, there is an issue with the explainability of document parsing.

### 2.2 VLM-based Document Content Extraction

The emergence of Vision-Language Models (VLMs) [38, 6, 1, 12] has revolutionized the field of document content extraction. These models leverage multi-modality capability to achieve remarkable performance in document understanding tasks. Document extraction tools powered by VLMs excel at comprehending both visual layouts and textual content, effectively handling complex document structures while capturing rich contextual information. Representative works such as Nougat [5], Vary [40], Fox [27], and GOT [39], along with recent advances [13, 29], demonstrate significant progress in automated document parsing and comprehension.
Despite these advances, the field lacks a standardized and unified benchmark for evaluating VLM-based document extraction task. This absence has hindered objective assessment of PDF document processing capabilities and impeded fair comparison across different approaches. To address this limitation, we present OmniDocBench, a comprehensive end-to-end benchmark designed specifically for evaluating VLM-based document parsing in real-world scenarios.

### 2.3 Benchmark for Document Content
Extraction

An end-to-end benchmark for PDFs can intuitively reflect the effectiveness of PDF extraction tools, which is crucial for their iteration and selection. However, current benchmarks predominantly focus on module-level evaluations; we have listed related benchmarks in Table 1. Additionally, while there are existing end-to-end benchmarks, they lack detailed annotation rules and suffer from insufficient diversity, as well as unreasonable metrics for formula and table evaluations. For example, READOC [26] covers only two types of sources—arXiv and GitHub—and uses EDS [20] and TEDS [47] to compute metrics for formulas and tables, which may lead to inaccuracies CDM [35]. Therefore, there is a need for a more finely annotated, diverse, and reasonably evaluated end-to-end benchmark.

## 3 OmniDocBench Dataset

Constructing a diverse and comprehensive document parsing benchmark with precise annotations is a formidable challenge. As illustrated in Figure 2, we have designed a systematic and professional annotation framework for OmniDocBench, encompassing data acquisition, intelligent pre-annotation, and manual refinement. This ensures that OmniDocBench possesses the following key attributes:

- •

Page Diversity. We sourced document pages from a variety of origins to ensure a wide range of document types.

- •

Comprehensive Annotation. We meticulously annotated all elements on the pages, including bounding boxes, specific content, and various potential attributes.

- •

Annotation Accuracy. By integrating semi-automated annotation processes, annotator corrections, and expert quality checks, we ensure the reliability of all annotations.

The following sections detail the data acquisition process, the annotation methodology, and a statistical analysis of the final annotated dataset.

### 3.1 Data Acquisition

During the data acquisition phase, we sourced document pages from diverse origins and used clustering algorithms to initially select visually diverse pages, followed by manual annotation of page attributes to finalize the OmniDocBench pages. Specifically, we collected 200,000 initial PDF documents from Common Crawl, Google, Baidu search engines, and internal data. Subsequently, we extracted visual features from these document pages using ResNet-50 and performed clustering using Faiss 111https://github.com/facebookresearch/faiss, sampling 6,000 visually diverse pages from 10 cluster centers. Finally, annotators provided page-level attribute annotations, including page type, layout type, and language type, and further balanced the selection to 981 samples for the final dataset. The OmniDocBench dataset includes pages with 9 types of pages, multiple layout categories, and various attribute annotations, covering a wide range of real-world scenarios.

### 3.2 Data Annotation

To ensure the comprehensiveness of OmniDocBench’s annotations, we conducted detailed annotations for layout detection and content recognition.

#### 3.2.1 Annotation Types

Layout Detection Annotations: Unlike typical layout detection tasks, OmniDocBench includes four comprehensive types of annotations:
(1) Layout Bounding Box Annotations: Locating information for 19 types of regions such as titles, text paragraphs, tables, and images.
(2) Layout Attribute Annotations: Detailed attribute annotations for detected boxes, including 3 text box attributes, 6 table attributes, and 2 formula attributes.
(3) Reading Order Annotations: Annotating the reading sequence of detected boxes.
(4) Affiliation Annotations: For images, tables, formulas, and code blocks, we annotate captions and titles to distinguish them from main text. Similarly, for cross-page paragraphs, we annotate affiliation relationships.

Content Recognition Annotations: Based on the format of the content area, we conduct the following three types of area annotations:
(1) Text Annotations: Pure text annotations for titles, text paragraphs, and other plain text content.
(2) Formula Annotations: LaTeX format annotations for inline formulas, display formulas, and subscripts.
(3) Table Annotations: Providing both HTML and LaTeX annotations for table data.

#### 3.2.2 Annotation Process

For these annotation tasks on diverse pages, we design a standardized process to ensure quality and efficiency, comprising intelligent pre-annotation, annotator correction, and expert quality inspection.

Intelligent Pre-Annotation. Manually annotating entire documents is time-consuming and costly. To enhance efficiency, we employ state-of-the-art detection and recognition models for pre-annotation of layout detection and content recognition. Specifically, we use fine-tuned LayoutLMv3 [16] for layout detection annotations and PaddleOCR [22], UniMERNet [34], and GPT-4o [2] for text, formula, and table annotations, respectively.

Annotator Correction. After layout detection phase, annotators refine the detection boxes and enhance annotations with reading order and affiliation details. Each character is verified to ensure accuracy in content recognition. For complex annotations of tables and formulas, requiring LaTeX and HTML formats, annotators use tools like Tables Generator 222https://www.tablesgenerator.com/ and latexlive 333https://www.latexlive.com/ for verification and correction.

Expert Quality Inspection. Despite thorough annotator corrections, the complexity of formulas and tables may result in residual issues. To address these, we use CDM’s rendering techniques to identify unrenderable elements. These are then reviewed and corrected by three researchers to ensure accuracy and fidelity in the final annotations.

### 3.3 Dataset Statistics

Page Diversity. OmniDocBench comprises a total of 981 PDF pages across 9 distinct types. Each page is annotated with global attributes, including text language, column layout type, and indicators for blurred scans, watermarks, and colored backgrounds.

Annotation Diversity: OmniDocBench contains over 10,0000 annotations for page detection and recognition: (1) More than 20,000 block-level annotations across 15 categories, including over 9,000 text paragraphs, 989 image boxes, 428 table boxes, and so on. All document components except headers, footers, and page notes are labeled with reading order information, totaling over 16,000 annotations. (2) The dataset also includes more than 80,000 span-level annotations across four categories, with 4,000 inter-line formulas and footnote markers represented in LaTeX format, while the remaining annotations are in text format.

Annotation Attribute Diversity: (1) Text Attributes: All block-level annotations, except for tables and images, include text attribute tags. In addition to standard Chinese and English text, there are over 2,000 blocks with complex backgrounds and 146 with rotated text. (2) Table Attributes: Besides standard Chinese and English tables, there are 142 with complex backgrounds, 81 containing formulas, 150 with merged cells, and 7 vertical tables.

## 4 OmniDocBench Evaluation Methodology

To provide a fair and comprehensive evaluation for various models, we proposed an end-to-end evaluation pipeline consisting of several modules, including extraction, matching algorithm, and metric calculation, as shown in Figure 3. It ensures that OmniDocBench automatically performs unified evaluation on end-to-end DCE tasks, thereby producing reliable and effective evaluation results.

### 4.1 Extraction

Preprocessing: The model-generated markdown text should be preprocessed, which includes removing images, eliminating markdown tags at the beginning of the document, and standardizing the number of repeated characters.

Special Component Extraction: Extraction is primarily carried out using regular expression matching. To ensure that the extraction of content does not interfere with each other, it is necessary to follow a specific order. The extraction sequence is as follows: LaTeX tables, HTML tables, display formulas, markdown tables (which are then converted into HTML format), and code blocks.

Pure Text Extraction: After extracting special components, the remaining content is considered pure text. Paragraphs are separated by double line breaks, allowing them to participate in subsequent matching processes, thus aligning with reading order annotation units in the GTs. If no double line break exists, single line breaks are used for paragraph separation. Additionally, previously extracted code blocks are merged into the text category for processing.

Inline Formula Format Converting: We standardized inline formulas within paragraphs to Unicode format. This was necessary because different models produce inconsistent outputs for inline formulas. For formulas originally written in Unicode, it is hard to extract them using regular expressions. Therefore, to ensure a fair comparison, we do not extract inline formulas for separate evaluation. Instead, we include them in their Unicode format alongside the text paragraphs for evaluation.

Reading Order Extraction: Upon completion of the extraction, the start and end positions of the extracted content in the original markdown are recorded for subsequent reading order calculation.

Method Type
Methods
TextEdit↓\downarrow
FormulaEdit↓\downarrow
FormulaCDM↑\uparrow
TableTEDS↑\uparrow
TableEdit↓\downarrow
Read OrderEdit↓\downarrow
OverallEdit↓\downarrow

EN
ZH
EN
ZH
EN
ZH
EN
ZH
EN
ZH
EN
ZH
EN
ZH

Pipeline Tools
MinerU
0.058
0.211
0.278
0.577
66.9
49.5
79.4
62.7
0.305
0.461
0.079
0.288
0.180
0.384

Marker
0.141
0.303
0.667
0.868
18.4
12.7
54.0
45.8
0.718
0.763
0.138
0.306
0.416
0.560

Mathpix
0.101
0.358
0.306
0.454
71.4
72.7
77.9
68.2
0.322
0.416
0.105
0.275
0.209
0.376

Expert VLMs

Nougat
0.365
0.998
0.488
0.941
17.4
16.9
40.3
0.0
0.622
1.000
0.382
0.954
0.464
0.973

General VLMs

Qwen2-VL
0.252
0.251
0.468
0.572
54.9
60.9
59.9
66.8
0.591
0.587
0.255
0.223
0.392
0.408

InternVL2
0.353
0.290
0.543
0.701
69.8
49.6
63.8
61.1
0.616
0.638
0.317
0.228
0.457
0.464

Model Type
Models
Book
Slides

 

Financial

Report

 

Textbook

 

Exam

Paper

Magazine

 

Academic

Papers

Notes
Newspaper
Average

Pipeline Tools
 
MinerU
0.044
0.124
0.033
0.102
0.159
0.072
0.025
0.984
0.148
0.188

Marker
0.188
0.327
0.087
0.292
0.423
0.134
0.102
0.470
0.270
0.255

Mathpix
0.131
0.168
0.202
0.199
0.278
0.138
0.091
0.631
0.648
0.276

Expert VLMs
 
GOT-OCR
0.105
0.222
0.067
0.132
0.204
0.198
0.179
0.388
0.771
0.252

Nougat
0.734
0.958
1.000
0.820
0.930
0.83
0.214
0.991
0.871
0.816

General VLMs
 
GPT4o
0.157
0.163
0.348
0.187
0.281
0.173
0.146
0.607
0.751
0.313

Qwen2-VL
0.094
0.08
0.145
0.148
0.219
0.065
0.315
0.298
0.79
0.239

InternVL2
0.216
0.098
0.162
0.184
0.247
0.150
0.419
0.226
0.903
0.289

Models
Fuzzy
Water
Color
Mean
Variance

Pipeline Tools

MinerU
0.15
0.151
0.107
0.136
0.0004

Marker
0.286
0.436
0.290
0.337
0.0049

Mathpix
0.294
0.290
0.182
0.255
0.0027

Expert VLMs

GOT-OCR
0.175
0.190
0.186
0.184
0.0000

Nougat
0.934
0.915
0.873
0.907
0.0006

General VLMs

GPT4o
0.263
0.195
0.184
0.214
0.0012

Qwen2-VL
0.101
0.157
0.114
0.124
0.0006

InternVL2
0.120
0.197
0.155
0.157
0.0010

Models
Single
Double
Three
Complex
Mean
Variance

Pipeline Tools

MinerU
0.311
0.101
0.117
0.376
0.226
0.0143

Marker
0.231
0.251
0.309
0.378
0.292
0.0033

Mathpix
0.189
0.175
0.225
0.413
0.250
0.0091

Expert VLMs

GOT-OCR
0.163
0.145
0.257
0.468
0.258
0.0165

Nougat
0.852
0.601
0.662
0.873
0.747
0.0139

General VLMs

GPT4o
0.109
0.204
0.254
0.426
0.248
0.0132

Qwen2-VL
0.098
0.248
0.517
0.429
0.323
0.0263

InternVL2
0.082
0.312
0.682
0.444
0.380
0.0472

### 4.2 Matching Algorithm

To avoid the impact of paragraph splitting on the final results, we proposed a method, Adjacency Search Match, that merges and splits paragraphs in both GTs and Preds to achieve the best possible match. The specific strategy involves: i) Calculate a metrix of Normalized Edit Distance between GTs and Preds. If the similarity between a Pred and a GT exceeds a specific threshold, they are considered a successful match. ii) For the rest, we apply fuzzy matching to determine whether one string is a subset of another string. If so, we further apply the truncation and merging algorithm which would try to merge adjacent paragraph. This process would continue to merge more paragraph until the Normalized Edit Distance starts to decrease. After this process, the best match will be found for GTs and Preds.

### 4.3 Metric Calculation

Ignore Handling: We implement an ignore logic for certain components in PDF page content, meaning they participate in matching but are excluded from metric calculations. This is mainly because of inconsistent output standards among models, which should not affect the validation results. For fairness, we ignore: (1) Headers, footers, page numbers, and page footnotes, which are handled inconsistently by different models. (2) Captions for figures, tables, and footnotes often have uncertain placement, complicating reading order. Additionally, some models embed table captions in HTML or LaTeX tables, while others treat them as plain text.

Metric: Different calculation methods are used for various document components:
(1) Pure Text: We calculate Normalized Edit Distance, averaging these metrics at the sample level to obtain the final scores.
(2) Tables: All tables are converted to HTML format before calculating the TEDS metric and Normalized Edit Distance.
(3) Formulas: Formulas are currently evaluated using the CDM [35], Normalized Edit Distance, and BLEU. We did not convert interline formulas into Unicode because Unicode cannot represent certain complex formulas, such as matrices.
(4) Reading Order: Reading order use the Normalized Edit Distance as metric. It only involves text components, where tables, images, and ignored components do not participate in the final reading order calculation.

## 5 Benchmarks

Model
Book
Slides

Research

Report

Textbook

Exam

Paper

Academic

Literature

DiT-L
43.44
13.72
45.85
15.45
3.40
29.23
66.13
0.21
23.65
26.90

LayoutLMv3
42.12
13.63
43.22
21.00
5.48
31.81
64.66
0.80
30.84
28.84

DOCX-Chain
30.86
11.71
39.62
19.23
10.67
23.00
41.60
1.80
16.96
21.27

DocLayout-YOLO
43.71
48.71
72.83
42.67
35.40
51.44
66.84
9.54
57.54
48.71

### 5.1 Component-specific Evaluation Results

Model Type
Model
Language
Table Frame Type
Special Situation
Overall

EN
ZH
Mixed
Full
Omission
Three
Zero

Merge Cell(+/-)

Formula(+/-)

 

Colorful
 (+/-)

Rotate(+/-)

OCR-based Models
PaddleOCR
76.8
71.8
80.1
67.9
74.3
81.1
74.5
70.6/75.2
71.3/74.1
72.7/74.0
23.3/74.6
73.6

RapidTable
80.0
83.2
91.2
83.0
79.7
83.4
78.4
77.1/85.4
76.7/83.9
77.6/84.9
25.2/83.7
82.5

Expert VLMs
StructEqTable
72.0
72.6
81.7
68.8
64.3
80.7
85.0
65.1/76.8
69.4/73.5
66.8/75.7
44.1/73.3
72.7

GOT-OCR
72.2
75.5
85.4
73.1
72.7
78.2
75.7
65.0/80.2
64.3/77.3
70.8/76.9
8.5/76.3
74.9

General VLMs
Qwen2-VL-7B
70.2
70.7
82.4
70.2
62.8
74.5
80.3
60.8/76.5
63.8/72.6
71.4/70.8
20.0/72.1
71.0

InternVL2-8B
70.9
71.5
77.4
69.5
69.2
74.8
75.8
58.7/78.4
62.4/73.6
68.2/73.1
20.4/72.6
71.5

Model Type
Model
Language
Text background
Text Rotate

EN
ZH
Mixed
White
Single
Multi
Normal
Rotate90
Rotate270
Horizontal

Expert Vision

Models
 
PaddleOCR
0.071
0.055
0.118
0.060
0.038
0.085
0.060
0.015
0.285
0.021

Tesseract OCR
0.179
0.553
0.553
0.453
0.463
0.394
0.448
0.369
0.979
0.982

Surya
0.057
0.123
0.164
0.093
0.186
0.235
0.104
0.634
0.767
0.255

GOT-OCR
0.041
0.112
0.135
0.092
0.052
0.155
0.091
0.562
0.966
0.097

Mathpix
0.033
0.240
0.261
0.185
0.121
0.166
0.180
0.038
0.185
0.638

Vision Language

Models
 
Qwen2-VL
0.072
0.274
0.286
0.234
0.155
0.148
0.223
0.273
0.721
0.067

InternVL2
0.074
0.155
0.242
0.113
0.352
0.269
0.132
0.610
0.907
0.595

GPT4o
0.020
0.224
0.125
0.167
0.140
0.220
0.168
0.115
0.718
0.132

Models
CDM
ExpRate@CDM
BLEU
Norm Edit

GOT-OCR
74.1
28.0
55.07
0.290

Mathpix
86.6
2.8
66.56
0.322

Pix2Tex
73.9
39.5
46.00
0.337

UniMERNet-B
85.0
60.2
60.84
0.238

GPT4o
86.8
65.5
45.17
0.282

InternVL2
67.4
54.5
47.63
0.308

Qwen2-VL
83.8
55.4
53.71
0.285

The OmniDocBench dataset features comprehensive and precise annotations, allowing for a fair and rigorous comparison of various document content extraction algorithms in real-world scenarios. Based on the distinct characteristics of these algorithms, we categorize document content extraction methods into three main classes:

Pipeline Tools. These methods integrate layout detection and various content recognition tasks (such as OCR, table recognition, and formula recognition) into a document parsing pipeline for content extraction. Prominent examples include MinerU [36], Marker [30], and Mathpix 444https://mathpix.com/.

Expert VLMs. These are large multimodal models specifically trained for document parsing tasks. Representative models include GOT-OCR2.0 [39] and Nougat [5].

General VLMs. These are general-purpose large multimodal models inherently capable of document parsing. Leading models in this category include GPT-4o [2], Qwen2-VL [38], and InternVL2 [6].

### 5.2 End-to-End Evaluation Results

Utilizing the OmniDocBench dataset and our evaluation framework, we conducted end-to-end assessments of mainstream document parsing methods, evaluating their performance from input PDF images to the resultant document parsing outputs.

Overall Evaluation Results. As illustrated in Table 2, pipeline tools specifically designed for document parsing, demonstrate superior performance across the board. MinerU and Mathpix achieved the best results for English and Chinese pages, respectively. In contrast, even the best general-purpose Vision Language Models (VLMs), GPT-4o, exhibits a performance gap compared to these specialized models, especially in Chinese. This trend is evident across sub-tasks like text recognition, formula recognition, and table recognition, where methods tailored for document parsing consistently outperform others. This advantage is largely due to the fine-tuning of these models on large datasets specific to document parsing tasks.

Performance Across Diverse Page Types. To gain deeper insights into model performance on diverse document types, we evaluated text recognition tasks across different page types. As shown in Table 3, an intriguing finding emerged: For commonly used data, such as academic papers and financial reports, pipeline tools perform well. However, for more specialized data like slides and handwritten notes, general VLMs demonstrate stronger generalization. The reason is clear: Pipeline tools and expert VLMs are relatively more constrained by the range of training data, whereas general VLMs having been trained on a wide variety of samples, maintained excellent recognition performance even in traditionally challenging long-tail scenarios, underscoring the value of VLMs.

Performance on Pages with Specific Attributes. For documents in OmniDocBench with attributes such as fuzzy scans, watermarks, and colorful backgrounds, our evaluation results are presented in Table 4. In these scenarios, the VLMs InternVL2 and Qwen2-VL exhibit the strongest resistance to interference, achieving the best accuracy and robustness. MinerU also performs commendably.

Performance on Different Column Layout Types. OmniDocBench annotates page attributes such as column layout type, which is crucial for analyzing model performance in reading order. As depicted in Table 5, all models experience a noticeable decline in reading order accuracy when dealing with complex layouts. MinerU and Mathpix excels in reading order across various column layouts, demonstrating robust performance across different page types.

From these end-to-end evaluations, it is evident that pipeline tools like MinerU and Mathpix, specifically designed for document parsing, achieve the best overall performance. However, in terms of versatility and scalability, VLMs offer a distinct advantage over pipeline tools. Fine-tuning a general large model like Qwen2-VL with specialized data could yield models even more adept at document parsing, indicating a promising direction for future research in multimodal approaches.

The OmniDocBench dataset provides comprehensive annotations for document parsing, including layout detection, text boxes and content, formula boxes and content, and table boxes and content. These detailed annotations enable the evaluation of current state-of-the-art (SOTA) methods across various document types, allowing us to analyze their performance in diverse scenarios. Additionally, these results can be used to assemble enhanced pipeline tools for document parsing tasks.

### 5.3 Single Algorithm Evaluation Results

Layout Detection Results. Layout detection is the first step in document parsing using pipeline tools. A robust layout detection algorithm should perform well across a variety of document types. Table 6 presents an evaluation of leading layout detection models. The DocLayout-YOLO method, which is pre-trained on diverse synthetic document data, significantly outperforms other approaches. This superiority is a key factor in MinerU’s integration of DocLayout-YOLO, contributing to its outstanding overall performance. The table also reveals that, aside from DocLayout-YOLO, other methods perform well on books and academic literature but are less effective on other document types, primarily due to a lack of pre-training on diverse documents.

Table Recognition Results. Table recognition results evaluated by Tree-Edit-Distance-based Similarity (TEDS) metric are presented in Table 7. We evaluate table recognition models across three dimensions on our OmniDocBench table subset: language diversity, table frame types, and special situations. Among all models, OCR-based models demonstrate superior overall performance, with RapidTable achieving the highest scores in language diversity and maintaining stable performance across different frame types. Expert VLMs show competitive results in specific scenarios, with StructEqTable [48] excelling in no frame tables and showing better rotation robustness. General VLMs (Qwen2-VL-7B and InternVL2-8B) exhibit relatively lower but consistent performance, suggesting that while general-purpose VLMs have made progress in table understanding, they still lag behind specialized solutions.

Text Recognition Results. In the traditional OCR task, Table 8 shows that PaddleOCR leads the field, surpassing other models significantly, with GOT also performing relatively well. Selecting these two methods for the OCR module is a prudent choice.

Formula Recognition Results. For formula recognition, the CDM metric provides a clear comparison in Table 9. GPT-4o, Mathpix, and UniMERNet achieve results of 86.8%, 86.6%, and 85.0%, respectively. Notably, GPT-4o excels with a recall rate of 65.5% under strict conditions requiring perfect character accuracy. Although Mathpix shows high character-level precision, it occasionally omits punctuation, such as commas, leading to a lower overall correctness rate. Nonetheless, all three models are strong candidates for formula recognition tasks.

## 6 Conclusion

This paper addresses the lack of diverse and realistic benchmarks in document parsing research by introducing OmniDocBench, a dataset featuring a variety of page types with comprehensive annotations, along with a flexible and reliable evaluation framework. OmniDocBench enables systematic and fair assessments of document parsing methods, providing crucial insights for advancing the field. Its task-specific and attribute-level evaluations facilitate targeted model optimization, promoting more robust and effective parsing solutions.

## References

- Achiam et al. [2023]

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.

Gpt-4 technical report.

arXiv:2303.08774, 2023.

- AI [2024]

Open AI.

Hello gpt 4o, 2024.

Accessed July 24, 2024.

- Bai et al. [2024]

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou.

Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond.

arXiv:2308.12966, 2024.

- Blecher [2022]

Lukas Blecher.

pix2tex - latex ocr.

https://github.com/lukas-blecher/LaTeX-OCR, 2022.

Accessed: 2024-2-29.

- Blecher et al. [2024]

Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic.

Nougat: Neural optical understanding for academic documents.

arXiv:2308.13418, 2024.

- Chen et al. [2024]

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai.

Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.

- Cheng et al. [2023]

Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, and Lianwen Jin.

M6doc: A large-scale multi-format, multi-type, multi-layout, multi-language, multi-annotation category dataset for modern document layout analysis.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15138–15147, 2023.

- Deng et al. [2017]

Yuntian Deng, Anssi Kanervisto, Jeffrey Ling, and Alexander M Rush.

Image-to-markup generation with coarse-to-fine attention.

In International Conference on Machine Learning, pages 980–989. PMLR, 2017.

- Desai et al. [2021]

Harsh Desai, Pratik Kayal, and Mayank Singh.

Tablex: a benchmark dataset for structure and content information extraction from scientific tables.

In Document Analysis and Recognition–ICDAR 2021: 16th International Conference, pages 554–569, 2021.

- Gao et al. [2023]

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang.

Retrieval-augmented generation for large language models: A survey.

arXiv:2312.10997, 2023.

- Gu et al. [2021]

Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, and Tong Sun.

Unidoc: Unified pretraining framework for document understanding.

Advances in Neural Information Processing Systems, 34:39–50, 2021.

- Hu et al. [2024a]

Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Chen Li, Ji Zhang, Qin Jin, Fei Huang, et al.

mplug-docowl 1.5: Unified structure learning for ocr-free document understanding.

arXiv preprint arXiv:2403.12895, 2024a.

- Hu et al. [2024b]

Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou.

mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding.

arXiv preprint arXiv:2409.03420, 2024b.

- Huang et al. [2022a]

Mingxin Huang, Yuliang Liu, Zhenghao Peng, Chongyu Liu, Dahua Lin, Shenggao Zhu, Nicholas Yuan, Kai Ding, and Lianwen Jin.

Swintextspotter: Scene text spotting via better synergy between text detection and text recognition.

In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4593–4603, 2022a.

- Huang et al. [2012]

Xin Huang, Ashish Khetan, Milan Cvitkovic, and Zohar Karnin.

Tabtransformer: Tabular data modeling using contextual embeddings. arxiv 2020.

arXiv preprint arXiv:2012.06678, 2012.

- Huang et al. [2022b]

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei.

Layoutlmv3: Pre-training for document ai with unified text and image masking, 2022b.

- Huang et al. [2023]

Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, and Wei Peng.

Improving table structure recognition with visual-alignment sequential coordinate modeling.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11134–11143, 2023.

- Hwang et al. [2021]

Wonseok Hwang, Jinyeong Yim, Seunghyun Park, Sohee Yang, and Minjoon Seo.

Spatial dependency parsing for semi-structured document information extraction.

In Findings of the Association for Computational Linguistics: ACL-IJCNLP, pages 330–343. Association for Computational Linguistics (ACL), 2021.

- Karatzas et al. [2015]

Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny.

Icdar 2015 competition on robust reading.

In 2015 13th International Conference on Document Analysis and Recognition, pages 1156–1160, 2015.

- Levenshtein et al. [1966]

Vladimir I Levenshtein et al.

Binary codes capable of correcting deletions, insertions, and reversals.

In Doklady Physics, pages 707–710. Soviet Union, 1966.

- Lewis et al. [2020]

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al.

Retrieval-augmented generation for knowledge-intensive nlp tasks.

Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

- Li et al. [2022]

Chenxia Li, Weiwei Liu, Ruoyu Guo, Xiaoting Yin, Kaitao Jiang, Yongkun Du, Yuning Du, Lingfeng Zhu, Baohua Lai, Xiaoguang Hu, Dianhai Yu, and Yanjun Ma.

Pp-ocrv3: More attempts for the improvement of ultra lightweight ocr system, 2022.

- Li et al. [2020a]

Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li.

Tablebank: Table benchmark for image-based table detection and recognition.

In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1918–1925, 2020a.

- Li et al. [2020b]

Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhoujun Li, and Ming Zhou.

Docbank: A benchmark dataset for document layout analysis.

arXiv:2006.01038, 2020b.

- Li et al. [2020c]

Zhe Li, Lianwen Jin, Songxuan Lai, and Yecheng Zhu.

Improving attention-based handwritten mathematical expression recognition with scale augmentation and drop attention.

In 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR), pages 175–180. IEEE, 2020c.

- Li et al. [2024]

Zichao Li, Aizier Abulaiti, Yaojie Lu, Xuanang Chen, Jia Zheng, Hongyu Lin, Xianpei Han, and Le Sun.

Readoc: A unified benchmark for realistic document structured extraction.

arXiv:2409.05137, 2024.

- Liu et al. [2024]

Chenglong Liu, Haoran Wei, Jinyue Chen, Lingyu Kong, Zheng Ge, Zining Zhu, Liang Zhao, Jianjian Sun, Chunrui Han, and Xiangyu Zhang.

Focus anywhere for fine-grained multi-page document understanding.

arXiv:2405.14295, 2024.

- Liu et al. [2020]

Yuliang Liu, Hao Chen, Chunhua Shen, Tong He, Lianwen Jin, and Liangwei Wang.

Abcnet: Real-time scene text spotting with adaptive bezier-curve network.

In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9809–9818, 2020.

- Lv et al. [2024]

Tengchao Lv, Yupan Huang, Jingye Chen, Yuzhong Zhao, Yilin Jia, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, and Furu Wei.

Kosmos-2.5: A multimodal literate model, 2024.

- Paruchuri [2024]

Vik Paruchuri.

Marker, 2024.

- Pfitzmann et al. [2022]

Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter Staar.

Doclaynet: A large human-annotated dataset for document-layout segmentation.

In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 3743–3751, 2022.

- Pramanik et al. [2020]

Subhojeet Pramanik, Shashank Mujumdar, and Hima Patel.

Towards a multi-modal, multi-task learning based pre-training framework for document representation learning.

arXiv preprint arXiv:2009.14457, 2020.

- Smith et al. [2009]

Ray Smith, Daria Antonova, and Dar-Shyang Lee.

Adapting the tesseract open source ocr engine for multilingual ocr.

In Proceedings of the International Workshop on Multilingual OCR, 2009.

- Wang et al. [2024a]

Bin Wang, Zhuangcheng Gu, Guang Liang, Chao Xu, Bo Zhang, Botian Shi, and Conghui He.

Unimernet: A universal network for real-world mathematical expression recognition, 2024a.

- Wang et al. [2024b]

Bin Wang, Fan Wu, Linke Ouyang, Zhuangcheng Gu, Rui Zhang, Renqiu Xia, Bo Zhang, and Conghui He.

Cdm: A reliable metric for fair and accurate formula recognition evaluation.

arXiv:2409.03643, 2024b.

- Wang et al. [2024c]

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He.

Mineru: An open-source solution for precise document content extraction.

arXiv:2409.18839, 2024c.

- Wang et al. [2021]

Pengfei Wang, Chengquan Zhang, Fei Qi, Shanshan Liu, Xiaoqiang Zhang, Pengyuan Lyu, Junyu Han, Jingtuo Liu, Errui Ding, and Guangming Shi.

Pgnet: Real-time arbitrarily-shaped text spotting with point gathering network.

In Proceedings of the AAAI Conference on Artificial Intelligence, pages 2782–2790, 2021.

- Wang et al. [2024d]

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al.

Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution.

arXiv preprint arXiv:2409.12191, 2024d.

- Wei et al. [2024]

Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, et al.

General ocr theory: Towards ocr-2.0 via a unified end-to-end model.

arXiv:2409.01704, 2024.

- Wei et al. [2025]

Haoran Wei, Lingyu Kong, Jinyue Chen, Liang Zhao, Zheng Ge, Jinrong Yang, Jianjian Sun, Chunrui Han, and Xiangyu Zhang.

Vary: Scaling up the vision vocabulary for large vision-language model.

In European Conference on Computer Vision, pages 408–424. Springer, 2025.

- Xia et al. [2024a]

Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, et al.

Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models.

arXiv preprint arXiv:2406.11633, 2024a.

- Xia et al. [2024b]

Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Min Dou, Botian Shi, Junchi Yan, et al.

Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning.

arXiv preprint arXiv:2402.12185, 2024b.

- Xu et al. [2019]

Zhong Xu, Jianbin Tang, and Antonio Jimeno Yepes.

Publaynet: largest dataset ever for document layout analysis.

In 2019 International conference on document analysis and recognition, pages 1015–1022, 2019.

- Zhang et al. [2018]

Jianshu Zhang, Jun Du, and Lirong Dai.

Multi-scale attention with dense encoder for handwritten mathematical expression recognition.

In 2018 24th international conference on pattern recognition (ICPR), pages 2245–2250. IEEE, 2018.

- Zhang et al. [2024]

Qintong Zhang, Victor Shea-Jay Huang, Bin Wang, Junyuan Zhang, Zhengren Wang, Hao Liang, Shawn Wang, Matthieu Lin, Wentao Zhang, and Conghui He.

Document parsing unveiled: Techniques, challenges, and prospects for structured information extraction.

arXiv preprint arXiv:2410.21169, 2024.

- Zhao et al. [2024]

Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Conghui He.

Doclayout-yolo: Enhancing document layout analysis through diverse synthetic data and global-to-local adaptive perception, 2024.

- Zhong et al. [2020]

Xu Zhong, Elaheh ShafieiBavani, and Antonio Jimeno Yepes.

Image-based table recognition: data, model, and evaluation.

In European conference on computer vision, pages 564–580, 2020.

- Zhou et al. [2024]

Hongbin Zhou, Xiangchao Yan, and Bo Zhang.

Structeqtable-deploy: A high-efficiency open-source toolkit for table-to-latex transformation.

https://github.com/UniModal4Reasoning/StructEqTable-Deploy, 2024.

\thetitle

Supplementary Material

## I More End-to-End Evaluation Results

Table S1 presents the evaluation results of End2End Tables grouped by Table Attributes. As it shows, most of the models perform better in English Tables rather than Chinese ones. Most models perform relatively poorly with Full Frame and No Frame tables. The accuracy of most models is affected by special conditions. Merged cells and formulas mainly test the breadth of data the model can recognize, while colored backgrounds and table rotation test their robustness. The results show that table rotation significantly impacts the accuracy of all models. Pipeline Tools perform well on more challenging tables, but colored backgrounds can affect recognition accuracy. Several Vision Language Models (VLMs) tend to perform worse on tables with merged cells, but colored backgrounds do not significantly impact table recognition accuracy.

Table S2 shows the evaluation results of End2End Text blocks grouped by Text Attributes. Almost all models have lower recognition accuracy in Chinese compared to English. Some models, such as MinerU and Marker, experience a further decrease in accuracy when recognizing mixed Chinese and English content. Complex background colors significantly affect the recognition accuracy of pipeline tools, but they have little impact on VLMs.

Model Type
Model
Language
Table Frame Type
Special Situation

EN
ZH
Mixed
Full
Omission
Three
Zero

Merge Cell(+/-)

Formula(+/-)

 

Colorful
 (+/-)

Rotate(+/-)

Pipeline Tools
MinerU
75.7
59.9
79.6
60.0
72.8
70.1
60.4

64.1/66.0

66.7/65.0

59.8/68.1
2.9/66.4

Marker
52.5
43.0
44.2
41.8
55.3
47.1
52.4
43.8/47.0
42.9/46.6
44.3/46.7
6.3/46.6

Mathpix
76.1
64.3
71.9
68.3
79.3
67.0
25.8

71.2/66.4

69.8/67.6

60.5/71.8

20.7/68.8

Expert Vision

Models
 
GOT-OCR
51.9
47.0
49.4
46.2
49.3
51.6
47.2
46.5/49.7
46.4/49.1
40.2/52.7
0.0/49.4

Nougat
36.5
0.4
0.0
6.3
3.6
22.2
0.0
15.1/9.1
21.2/8.9
2.8/15.3
0.0/11.4

Vision Language

Models
 
GPT4o
71.8
58.8
57.9
63.3
69.5
61.9
31.8
57.5/65.5
61.6/62.9

62.0/63.0

14.5/63.5

Qwen2-VL
57.4
62.9
72.7
70.7
64.1
48.3
57.6
49.4/68.2
48.5/64.7

63.5/60.7

41.6/61.9

InterVL2
61.5
59.3
65.9
59.7
66.5
58.7
56.2
49.6/65.9
54.4/61.6
59.4/60.6
7.3/61.1

Model Type
Model
Language
Text background

EN
ZH
Mixed
White
Single
Multi

Pipeline Tools
MinerU
0.123
0.206
0.742
0.163
0.147
0.513

Marker
0.267
0.389
0.499
0.339
0.389
0.497

Mathpix
0.173
0.774
0.538
0.675
0.554
0.570

Expert Vision

Models
 
GOT-OCR
0.251
0.763
0.266
0.669
0.595
0.440

Nougat
0.587
0.991
0.983
0.874
0.935
0.972

Vision Language

Models
 
GPT4o
0.170
0.647
0.322
0.536
0.423
0.406

Qwen2-VL
0.337
0.575
0.310
0.537
0.400
0.233

InternVL2
0.418
0.606
0.251
0.589
0.366
0.221

## II Dataset Statistics and Visualization

OmniDocBench contains 981 pages, including 9 types of PDF pages, 4 types of layouts, and 3 types of languages. Some pages also include special conditions, such as watermarks. Table S3 and Figure S1 show the number of pages with each page attribute. Figures S3, S4, S5 and S6 are examples of PDF pages with different PDF types, Layout Types, and Special Issues.

Table S6 and Figure S2 show all annotation categories included in OmniDocBench. All of them are annotated by bounding boxes. There are 15 types of block-level annotations and 4 types of span-level annotations, with span-level annotations nested within the block-level ones. In addition, there are 3 types of annotations marked as page interference information (No.20-22), whose bounding boxes are used to mask the specific regions of the PDF pages to avoid affecting the evaluation results. The recognition annotations are also provided for each annotation category except for Figures. Formulas is written in LaTeX format and Table is annotated in both HTML and LaTeX formats. Others are annotated in plain text.

Furthermore, the Text Attributes are also annotated for each block-level category that contains text. There are 3 types of Text Attributes that might influent OCR accuracy: Language, Text Background Color, and Text Rotation. Table S5 shows the statistics of annotations with specific text attributes. There are 23,010 block-level annotations are labeled with text attributes.

Tables are also annotated with Table Attributes. There are 6 types of Table Attributes that might influent the Table Recognition accuracy: Language, Table Frame Type, Merge Cell, Colorful Background, Contain Formula, and Rotation. Table S5 shows the numbers of annotations with specific table attributes. Figures S7 and S8 are the examples of Tables with different Frames and Special Issues.

## III Model Results Visualization

Figures S10, S11, S12, S13, S14, S9, S15, S16 and S17 show the examples of Good model outputs and Bad model outputs of Document Parsing among different PDF types. As it shown, different models exhibit varying performance across different PDF types. For example, MinerU detects all handwritten notes as figures, resulting in very low recognition accuracy in Notes. Marker and InternVL2 experience missed detections, leading to lower scores. InternVL2 and Qwen2-VL, in specific PDF types (such as slides or financial reports), tend to merge multi-column text.

Figures S20, S18 and S19 show the examples of Good model outputs and Bad model outputs under special issues of the PDF pages. It shows that Marker tends to generate typos when the PDF pages are fuzzy scanned or with watermarks, while GOT-OCR fails to recognize content on pages with colored backgrounds. MinerU performs well under special situations, while Mathpix occasionally generates typos.

Figures S21, S22, S23 and S24 show examples of Good model outputs and Bad model outputs for PDF pages with different layouts. MinerU has a low reading order score for single-column layouts primarily because most notes are single-column, and MinerU performs poorly in recognizing Notes, leading to a low reading order score accordingly. InternVL2 scores high in Single-Column layouts but scores poorly on Double-Column and Three-Column layouts. It is mainly due to frequent missed content recognition and errors in reading order judgment in multi-column layouts pages. MinerU’s reading order and recognition accuracy decrease with complex layouts, primarily because it incorrectly merges multiple columns during recognition.

Figures S27 and S28 show the model’s recognition ability under special issues of text. In text recognition with complex background colors, Marker may produce errors or miss content, whereas Qwen2-VL still performs well. Most models fail to recognize text when it is rotated 270 degrees. Some vision language models generate hallucinated information based on the content they can recognize.

Figures S29, S30, S31 and S32 show the examples of good and bad model results for tables with different attributes. For three-line tables, RapidTable demonstrates a good performance with accurate structure recognition, while PaddleOCR shows limitations by missing the last column in its outputs. Interestingly, in tables without frames, PaddleOCR performs well with accurate table predictions, while Qwen2-VL-7B exhibits errors in the last two columns. This indicates that the presence or absence of table frames can significantly impact different models’ performance in different ways. Rotated tables prove to be particularly challenging, with most models, including GOT-OCR, failing to recognize the table structure. However, StructEqTable shows promising results by correctly identifying most of the table content, though with a few detail errors. For tables containing formula, Qwen2-VL-7B shows more accurate table structure recognition compared to InternVL2-8B.

## IV Model Settings

For pipeline tools such as MinerU, Marker, and Mathpix, default settings are used for evaluation. Specifically, MinerU with Version 0.9.3555https://github.com/opendatalab/MinerU/releases/tag/magic_pdf-0.9.3-released is employed.
For Marker, Version 0.2.17666https://github.com/VikParuchuri/marker/releases/tag/v0.2.17 is evaluated.
For Nougat, we utilize its 0.1.0-base model (350M).
For GOT-OCR, we employ its format OCR mode to output structured data.
For general VLMs, we used the GPT4o, Qwen2-VL-72B, and InternVL2-Llama3-76B by setting the do_sample==False to ensure the reproducibility.

Category
Attribute Name
Count

PDF Type
Book
104

PPT2PDF
133

Research Report
81

Colorful Textbook
96

Exam Paper
114

Magazine
97

Academic Literature
129

Notes
116

Newspaper
111

Layout Type
Single Column
477

Double Column
126

Three Column
45

One&More Mixed
120

Complex Layout
213

Language
English
290

Simplified Chinese
612

Mixed
79

Special Issues
Fuzzy Scan
28

Watermark
65

Colorful Background
246

Attribute Category
Category Name
Count

Language
English
5857

Simplified Chinese
16073

EN&CH Mixed
1080

Text Background
White
19465

Single-Colored
1116

Multi-Colored
2429

Text Rotate
Normal
22865

Rotate90
14

Rotate270
58

Horizontal
421

Attribute Category
Category Name
Count

Language
English
128

Simplified Chinese
285

EN&CH Mixed
15

Table Frame Type
Full Frame
205

Omission Line
62

Three Line
147

No Frame
14

Special Issues
Merge Cell
150

Colorful Background
142

Contain Formula
81

Rotate
7

No.
Category Name
Explaination
Total

1
Title
Include main titles, chapter titles, etc.
2972

2
Text Block
Text paragraphs, which are usually separated by double line breaks in Markdown.
15979

3
Figure
Including images, visual charts, etc.
989

4
Figure Caption
Typically starts with ’Figure’ followed by a number, or just descriptive language below the figure.
651

5
Figure Footnotes
Descriptive language, apart from the figure caption, usually starts with an asterisk (*).
133

6
Table
Content organized in table form usually includes borders or a clear table structure.
428

7
Table Caption
Typically starts with ’Table’ followed by a number, or just descriptive language above the Table.
299

8
Table Footnotes
Descriptive language, apart from the table caption, usually starts with an asterisk (*).
132

9
Header
Information located at the top of a PDF page or in the sidebar, separate from the main content, typically includes chapter names and other details.
1271

10
Footer
Information located at the bottom of a PDF page, separate from the main content, typically includes the publisher’s name and other details.
541

11
Page Number
It is usually represented by numbers, which may be located at the top, in the sidebar, or at the bottom of the page.
669

12
Page Footnote
It provides further explanation of the footnotes marked within the page content. For example, information about the authors’ affiliations.
92

13
Code Block
In Markdown, a code block is typically defined using triple backticks (“‘).
13

14
Code Block Caption
Descriptive language above the Code Block.
/

15
Reference
Typically found only in academic literature.
260

16
Text Span
Span-Level text box, which is the plain text content can be directly written in Markdown format.
73143

17
Equation Inline
Formulas that need to be represented using LaTeX format and embedded within the text.
4009

18
Equation Ignore

Some formulas that can be displayed correctly without using LaTeX formatting, such as 15 kg.

3685

19
Footnote Mark
Typically embedded within the text as superscripts or subscripts, and their numbering usually corresponds to page footnotes.
357

20
Other Abandoned Categories
(Masked) Some uncategorizable, irrelevant page information, such as small icons, etc.
538

21
Masked Text Block
(Masked) Some difficult-to-recognize information that disrupts text flow, such as pinyin annotations above Chinese characters.
34

22
Organic Chemical Formula
(Masked) Organic chemistry formulas, which are difficult to write using Markdown and are easily recognized as Figures.
24

Generated on Tue Jan 7 01:30:12 2025 by LaTeXML
