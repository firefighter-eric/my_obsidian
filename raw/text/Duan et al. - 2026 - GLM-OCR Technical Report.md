# GLM-OCR Technical Report

- Source HTML: `raw/html/Duan et al. - 2026 - GLM-OCR Technical Report.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2603.10910
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

GLM-OCR Technical Report

Shuaiqi Duan⋆1   Yadong Xue⋆1   Weihan Wang⋆1   Zhe Su1   Huan Liu1   
Sheng Yang1   Guobing Gan1   Guo Wang1   Zihan Wang1   Shengdong Yan1   
Dexin Jin1   Yuxuan Zhang1   Guohong Wen1   Yanfeng Wang1   Yutao Zhang1   
Xiaohan Zhang1   Wenyi Hong1   Yukuo Cen1   Da Yin1   Bin Chen1   
Wenmeng Yu†1   Xiaotao Gu†1   Jie Tang2

1Zhipu AI  2Tsinghua University

⋆Equal contribution   †Project leader

Code:

https://github.com/zai-org/GLM-OCR

Model:

https://huggingface.co/zai-org/GLM-OCR

Demo:

https://ocr.z.ai/

###### Abstract

GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To address the inefficiency of standard autoregressive decoding in deterministic OCR tasks, GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput while keeping memory overhead low through shared parameters. At the system level, a two-stage pipeline is adopted: PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition. Extensive evaluations on public benchmarks and industrial scenarios show that GLM-OCR achieves competitive or state-of-the-art performance in document parsing, text and formula transcription, table structure recovery, and key information extraction. Its compact architecture and structured generation make it suitable for both resource-constrained edge deployment and large-scale production systems.

###### Contents

- 1 Introduction

- 2 Methodology

- 2.1 Model Overview

- 2.2 Training Recipe

- 3 Evaluation

- 3.1 Public Benchmarks

- 3.2 In-House Benchmarks

- 4 Inference and Deployment

- 4.1 Local Deployment and SDK Integration

- 4.2 Model-as-a-Service (MaaS) API

- 4.3 Fine-Tuning Capabilities

- 5 Intended Use Cases

- 5.1 Overview

- 5.2 Document Parsing with GLM-OCR SDK

- 5.3 Lightweight OCR and Information Extraction with the Base Model

- 5.3.1 Text Recognition

- 5.3.2 Table Recognition

- 5.3.3 Formula Recognition

- 5.3.4 Key Information Extraction

- 5.4 Summary of Usage Paradigms

- 6 Limitations

- 6.1 Two-Stage Architectural Constraints

- 6.2 Data Coverage Limitations

- 6.3 Structured Output Variability

- 6.4 Key Information Extraction

- 7 Conclusion

- References

## 1 Introduction

Document understanding is a core capability in modern information systems, supporting the extraction and structuring of knowledge from visually rich and layout-intensive documents such as financial reports, scientific articles, contracts, and invoices. Traditional OCR systems [29; 12; 7] mainly focus on plain text transcription and rely on multi-stage pipelines with handcrafted rules for layout parsing and downstream information extraction. While effective for simple scenarios, these approaches often struggle with complex layouts, diverse document formats, and real-world production requirements.

Recent multimodal large language models (MLLMs) [2; 31; 30] unify visual perception and language understanding within a single framework and significantly improve document understanding performance. However, their large model size and autoregressive decoding paradigm lead to high computational cost, slow inference, and substantial memory consumption, which makes large-scale deployment under high-concurrency or edge environments challenging.

In practical production systems, document intelligence solutions must simultaneously provide: (1) strong performance on complex content such as tables, formulas, code, and seals, (2) high-throughput and low-latency inference, and (3) flexible integration and domain adaptation. GLM-OCR is developed to address these system-level requirements within a unified multimodal framework.

GLM-OCR is a lightweight multimodal OCR model for comprehensive document understanding. Built on the GLM-V encoder–decoder framework [31], it combines a 0.4B-scale CogViT visual encoder trained in large-scale image–text data, a lightweight cross-modal connector and a 0.5B-scale GLM language decoder [9]. The entire model contains only 0.9B parameters, enabling high-throughput and low-latency inference while maintaining strong recognition performance.

Beyond architectural optimization, GLM-OCR also considers the mismatch between conventional autoregressive generation and the characteristics of OCR tasks. OCR is inherently a deterministic task with strong local dependencies and explicit structural supervision, where strictly autoregressive token-by-token decoding is inefficient. Therefore, we introduce Multi-Token Prediction (MTP) [15] into both training and inference. MTP enables the simultaneous prediction of multiple tokens, substantially improving training efficiency and decoding throughput while preserving recognition accuracy, and is particularly advantageous for long structured outputs such as tables. To control the additional memory overhead introduced by MTP, we further adopt a parameter-sharing scheme across the draft models, which substantially reduces the additional GPU memory overhead [35]. In practice, GLM-OCR is trained to predict ten tokens per step and generates 5.2 tokens per decoding step on average at inference time, bringing approximately 50% throughput improvement.

At the system level, GLM-OCR adopts a two-stage pipeline consisting of layout analysis and parallel content recognition. The layout stage is powered by PP-DocLayout-V3 [5], which detects structured regions and enables parallel recognition across different document areas. This design improves both robustness and processing efficiency for complex real-world documents.

Results. Figure 2 shows that GLM-OCR achieves 94.6 on OmniDocBench v1.5 [24], ranking first among all evaluated models despite its compact 0.9B size. Besides, GLM-OCR delivers strong performance across text recognition, formula recognition, table parsing, and key information extraction, reaching 94.0 on OCRBench (Text) [17] and 96.5 on UniMERNet [32], and achieving 85.2 on PubTabNet [36] and 86.0 on TEDS 111https://modelscope.cn/datasets/jockerK/TEDS_TEST. It also performs competitively on information extraction benchmarks such as Nanonets-KIE and Handwritten-Forms [18], with performance comparable to significantly larger general multimodal models.

In addition to public benchmarks, we evaluate GLM-OCR on six high-frequency real-world scenarios, including code document parsing, natural-scene table recognition, handwritten text recognition, multilingual OCR 222Chinese, English, French, Spanish, Russian, German, Japanese, and Korean, seal recognition, and receipt KIE. GLM-OCR consistently delivers strong results across all settings, achieving 91.5 on real-world table recognition, 90.5 on seal recognition, and 94.5 on receipt KIE. These results indicate that GLM-OCR generalizes beyond curated benchmarks and remains effective under practical production conditions.

Deployment and Finetuning. GLM-OCR supports efficient inference with modern serving frameworks such as vLLM [13] 333https://github.com/vllm-project/vllm, SGLang 444https://github.com/sgl-project/sglang, and Ollama 555https://github.com/ollama/ollama, enabling deployment in both large-scale and resource-constrained edge scenarios. It also provides full finetuning support through LLaMA-Factory 666https://github.com/hiyouga/LlamaFactory, enabling rapid adaptation to domain-specific document understanding tasks 777Tutorial: https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md.

## 2 Methodology

In this section, we present the overall design of the GLM-OCR framework, including its architectural components, task formulation, and training strategy. We first introduce the core design motivations that guide our system, followed by a detailed description of the model architecture and task-specific pipelines. Finally, we elaborate on the multi-stage training recipe that progressively aligns visual and language representations, enhances structured generation ability, and optimizes task performance through supervised and reinforcement learning.

### 2.1 Model Overview

To provide a comprehensive understanding of the GLM-OCR architecture, we first discuss the primary design rationale behind our approach, followed by a detailed description of the model structure and its execution across different tasks.

##### Motivation.

Three fundamental observations and goals in document understanding inform our architectural design:

1. Integration of Layout Analysis in Document Parsing:
In practice, we observe that small-scale models are highly susceptible to hallucinations and repetitive generation when processing documents with complex layouts. By explicitly introducing a layout analysis module prior to recognition, we decompose complex layout structures into multiple simpler sub-problems, significantly enhancing the overall performance and stability of the model. Furthermore, partitioning large, complex pages into smaller, independent regions allows for parallel recognition, which substantially improves inference efficiency.

2. Incorporation of Key Information Extraction (KIE):
Document parsing and KIE can both be formulated as structured generation problems conditioned on visual inputs. While document parsing focuses on reconstructing the full structural representation of a document (e.g., Markdown or JSON), KIE aims to generate task-specific structured fields from the same visual source. From a modeling perspective, both tasks require:
(i) robust visual-text alignment,
(ii) structural reasoning over layout and semantic regions, and
(iii) the ability to generate well-formed structured outputs.

Therefore, instead of treating them as isolated pipelines, we unify them under a shared generative framework. This design encourages the model to learn generalizable document-level representations while leveraging task-specific prompts to control output formats. Such a unified formulation improves parameter efficiency and promotes cross-task knowledge transfer without introducing additional architectural complexity.

3. Adoption of Multi-Token Prediction (MTP):
Standard LLM decoding generates one token at a time, which can be computationally expensive and slow for long-form document generation. We employ Multi-token Prediction (MTP) to address two primary challenges:

- •

Inference Speed: By predicting kk tokens simultaneously, we significantly reduce the total number of decoding steps.

- •

Contextual Modeling: MTP encourages the model to plan further ahead. This is particularly beneficial in OCR tasks where structural tokens (e.g., table tags or Markdown syntax) exhibit strong local dependencies. Consequently, this approach yields fewer "broken" tags and produces more robust structured outputs.

##### Architecture.

As illustrated in Figure 2, the system is centered around the GLM-OCR Core, which follows a vision-language generative paradigm. The core model consists of:

- •

Vision Encoder (CogViT, 400M parameters): Responsible for extracting high-level visual representations from document images.

- •

LLM Decoder (GLM, 500M parameters): An autoregressive language model that generates structured textual outputs conditioned on visual embeddings and textual prompts.

The visual features produced by the encoder are projected into the language embedding space and fed into the decoder as prefix tokens. During decoding, the model predicts structured outputs (e.g., Markdown or JSON) in an autoregressive manner.

To improve decoding efficiency and structural consistency, we introduce a Multi-Token Prediction (MTP) mechanism. In addition to the main prediction head, we attach kk shared-parameter auxiliary heads that simultaneously predict the next kk tokens. These heads share the same parameters but are trained to model different future offsets. During inference, this allows the model to generate multiple tokens per step, reducing latency while encouraging better local structural coherence.

The framework supports two primary tasks under a unified generative formulation:

Task 1: Document Parsing.

Given a document image, the pipeline first performs layout analysis using PPDocLayoutV3, which decomposes the document into semantically coherent regions (e.g., paragraphs, tables, formulas). Each region is independently processed by the GLM-OCR Core.

The generated regional outputs are subsequently aggregated by a Merge & Post Process module, which restores the reading order and produces structured outputs in Markdown and JSON formats. This modular design reduces hallucination risks, improves robustness to complex layouts, and enables parallel processing of document regions.

Task 2: Key Information Extraction.

For KIE, the full document image is directly fed into the GLM-OCR Core together with a task-specific textual prompt (e.g., instructing the model to extract invoice fields in JSON format). Unlike document parsing, this task does not rely on explicit layout cropping. Instead, the model learns to attend to relevant visual regions implicitly under prompt guidance.

Both tasks are thus unified as conditional structured generation problems, differing only in preprocessing strategy and prompt specification.

### 2.2 Training Recipe

The training process of GLM-OCR is divided into several distinct stages, systematically progressing from vision-language alignment to task-specific refinement and reinforcement learning. The detailed training recipe, including the data types, learning rates, and training scale for each phase, is summarized in Table 1.

Stage
Phase
Data Types

Stage 1
Vision Encoder Training
Image-text pairs, Grounding / Retrieval data

Stage 2.1
Pretrain
Image-text pairs, Document parsing, Grounding, VQA

Stage 2.2
Pretrain with MTP
Document parsing, Grounding, VQA

Stage 3
SFT with MTP
Text / Formula / Table recognition, KIE

Stage 4
RL
Text / Formula / Table recognition, KIE

##### Stage 1: Vision Encoder Training.

We first train the vision encoder using large-scale image-text and grounding data to establish strong visual representation capabilities. In this stage, the model is trained on a dataset scaled up to tens of billions of image-text pairs. The training incorporates a dual objective of MIM and CLIP tasks. Furthermore, we employ knowledge distillation from an in-house ViT with a larger parameter size to further enhance the encoder’s feature extraction capability.

##### Stage 2: Vision-Language Pretraining.

In Stage 2.1, we append GLM-0.5B to the Vision Transformer (ViT) and jointly pretrain the full model on image-text, document parsing, grounding, and VQA data to align multimodal representations. In Stage 2.2, we introduce the Multi-Token Prediction (MTP) objective to adapt the decoder for efficient structured generation.

##### Stage 3: Supervised Fine-Tuning (SFT).

In this stage, we fine-tune the model on curated OCR datasets covering text recognition, formula transcription, table structure recovery, and key information extraction. The objective is to specialize the model for high-precision structured outputs under real-world document distributions. Multi-Token Prediction remains enabled to ensure consistency between training and inference. The data mixture is balanced to prevent overfitting to any single sub-task and to maintain cross-task generalization.

##### Stage 4: Reinforcement Learning (RL).

The final stage applies GRPO [28] to improve structured output reliability and task-specific accuracy. Training samples are generated via rollout from the SFT model, evaluated automatically, and stratified by difficulty to construct a graded optimization set.

The reward function is task-aware and integrates both accuracy-based metrics and structural validation signals. The design is summarized in Table 2.

Task
Primary Accuracy Reward
Additional Constraints

Text Recognition
Normalized Edit Distance
Repetition penalty

Formula Recognition
CDM score
Structural validity check

Table Recognition
TEDS score
Tag closure verification, structural parsing

Key Information Extraction (KIE)
Field-level F1 score
JSON parse validation, missing/duplicate field penalty

Global Regularization: Repetition ratio penalty, malformed structure penalty

## 3 Evaluation

In this section, we evaluate the performance of GLM-OCR against current state-of-the-art pipeline tools, general Vision-Language Models (VLMs), and specialized OCR VLMs. To provide a comprehensive assessment, the evaluation is divided into two parts: standard Public Benchmarks and custom In-House Benchmarks.

### 3.1 Public Benchmarks

We first evaluate GLM-OCR on widely recognized public datasets encompassing Document Parsing and KIE tasks.

##### Overall Benchmark Performance.

As shown in Table 3, GLM-OCR demonstrates superior performance across the majority of standard datasets. In Document Parsing, our model achieves the highest scores on OmniDocBench v1.5 (94.6), OCRBench Text (94.0), UniMERNet (96.5), and TEDS_TEST (86.0). It remains highly competitive on PubTabNet (85.2), trailing only MinerU 2.5. Furthermore, GLM-OCR establishes a clear SOTA in KIE, outperforming all available open-source competitors on Nanonets-KIE (93.7) and Handwritten-KIE (86.1), and even narrowing the gap with closed-source giants like Gemini-3-Pro.

:

Dataset
GLM-OCR
 

PaddleOCR

-VL-1.5

 

Deepseek

-OCR2

 

MinerU

2.5

dots.ocr
 

Gemini-3

-Pro

 

GPT-5.2

-2025-12-11

Document Parsing

OmniDocBench v1.5
94.6
94.5
91.1
90.7
88.4
90.3
85.4

OCRBench (Text)
94.0
75.3
34.7
75.3
92.1
91.9
83.7

UniMERNet
96.5
96.1
85.8
96.4
90.0
96.4
90.5

PubTabNet
85.2
84.6
-
88.4
71.0
91.4
84.4

TEDS_TEST
86.0
83.3
-
85.4
62.4
81.8
67.6

Key Information Extraction

Nanonets-KIE
93.7
-
-
-
-
95.2
87.5

Handwritten-KIE
86.1
-
-
-
-
94.5
78.2

##### OmniDocBench v1.5 Analysis.

To better understand the model’s document parsing capabilities, we present a granular breakdown of the OmniDocBench v1.5 results in Table 4. This benchmark compares pipeline tools, general VLMs of varying sizes, and specialized VLMs.

Remarkably, despite possessing only 0.9B parameters, GLM-OCR achieves the highest Overall score (94.62), outperforming not only direct specialized competitors like PaddleOCR-VL-1.5 (94.50) and MinerU2.5 (90.67) but also massive general VLMs such as Qwen3-VL-235B (89.15) and Gemini-3 Pro (90.33).

A breakdown of sub-metrics indicates strong performance in table structure recovery. It achieves the absolute best scores in table recognition, scoring 93.96 on TableTEDS and 96.39 on TableTEDS-S. While PaddleOCR-VL-1.5 slightly edges out GLM-OCR in TextEdit (0.035 vs 0.040) and FormulaCDM (94.21 vs 93.90), GLM-OCR’s exceptional table parsing capabilities secure its position as the top-performing model overall, proving that specialized, parameter-efficient architectures can rival or surpass scale-heavy models in complex document parsing.

Model Type
Methods
Params

 

Overall↑\uparrow

 

Text↓\downarrow

Edit

 

Formula↑\uparrow

CDM

 

Table↑\uparrow

TEDS

 

Table↑\uparrow

TEDS-S

 

Reading Order↓\downarrow

Edit

Pipeline Tools
Marker-1.8.2 [25]

-
71.30
0.206
76.66
57.88
71.17
0.250

Mineru2-pipeline [23]

-
75.51
0.209
76.55
70.90
79.11
0.225

PP-StructureV3 [7]

-
86.73
0.073
85.79
81.68
89.48
0.073

General VLMs
GPT-4o [1]

-
75.02
0.217
79.70
67.07
76.09
0.148

InternVL3-76B [37]

76B
80.33
0.131
83.42
70.64
77.74
0.113

InternVL3.5-241B [33]

241B
82.67
0.142
87.23
75.00
81.28
0.125

GPT-5.2 [22]

-
85.50
0.123
86.11
82.66
87.35
0.099

Qwen2.5-VL-72B [3]

72B
87.02
0.094
88.27
82.15
86.22
0.102

Gemini-2.5 Pro [10]

-
88.03
0.075
85.82
85.71
90.29
0.097

Qwen3-VL [2]

235B
89.15
0.069
88.14
86.21
90.55
0.068

Gemini-3 Pro [11]

-
90.33
0.065
89.18
88.28
90.29
0.071

Specialized VLMs
Dolphin [8]

0.3B
74.67
0.125
67.85
68.70
77.77
0.124

OCRFlux-3B [4]

3B
74.82
0.193
68.03
75.75
80.23
0.202

Mistral OCR [20]

-
78.83
0.164
82.84
70.03
78.04
0.144

POINTS-Reader [16]

3B
80.98
0.134
79.20
77.13
81.66
0.145

olmOCR-7B [26]

7B
81.79
0.096
86.04
68.92
74.77
0.121

Dolphin-1.5 [8]

0.3B
83.21
0.092
80.78
78.06
84.10
0.080

MinerU2-VLM [23]

0.9B
85.56
0.078
80.95
83.54
87.66
0.086

Nanonets-OCR-s [19]

3B
85.59
0.093
85.90
80.14
85.57
0.108

MonkeyOCR-pro-1.2B [14]

1.9B
86.96
0.084
85.02
84.24
89.02
0.130

Deepseek-OCR [34]

3B
87.01
0.073
83.37
84.97
88.80
0.086

MonkeyOCR-3B [14]

3.7B
87.13
0.075
87.45
81.39
85.92
0.129

dots.ocr [27]

3B
88.41
0.048
83.22
86.78
90.62
0.053

MonkeyOCR-pro-3B [14]

3.7B
88.85
0.075
87.25
86.78
90.63
0.128

MinerU2.5 [21]

1.2B
90.67
0.047
88.46
88.22
92.38
0.044

PaddleOCR-VL [6]

0.9B
92.86
0.035
91.22
90.89
94.76
0.043

PaddleOCR-VL-1.5 [5]

0.9B
94.50
0.035
94.21
92.76
95.79
0.042

GLM-OCR
0.9B
94.62
0.040
93.90
93.96
96.39
0.044

### 3.2 In-House Benchmarks

To assess the robustness of GLM-OCR in highly complex, real-world industrial scenarios, we conducted evaluations on a custom suite of in-house benchmarks. These tasks include Code Document parsing, Real-world Table extraction, Handwritten Text recognition, Multilingual Text processing, Seal Recognition, and Receipt KIE.

Task
GLM-OCR

 

PaddleOCR

-VL-1.5

 

Deepseek

-OCR2

 

MinerU

2.5

dots.ocr

 

Gemini-3

-Pro

 

GPT-5.2

-2025-12-11

Code Document
84.7
75.8
82.1
82.9
80.8
86.9
84.4

Real-world Table
91.5
86.1
-
70.8
81.8
90.6
86.7

Handwritten Text
87.0
87.4
73.8
54.2
71.7
90.0
78.0

Multilingual Text
69.3
54.8
56.1
27.8
65.1
86.2
70.1

Seal Recognition
90.5
42.2
40.4
-
63.0
91.3
58.8

Receipt KIE
94.5
-
-
-
-
97.3
83.5

As detailed in Table 5, GLM-OCR achieves the highest score in five out of six evaluated categories among the compared open-weight models. Most notably, GLM-OCR demonstrates a notable margin in challenging, niche domains:

- •

Seal Recognition: GLM-OCR achieves an exceptional score of 90.5, outperforming the next best open-weight model (dots.ocr at 63.0) by a massive margin and performing competitively with Gemini-3-Pro (91.3).

- •

Multilingual Text: The model excels in diverse linguistic contexts, scoring 69.3 compared to PaddleOCR-VL-1.5’s 54.8.

- •

Complex Formatting: It leads in Code Document parsing (84.7) and Real-world Table extraction (91.5), proving its utility in highly structured and noisy environments.

While PaddleOCR-VL-1.5 holds a marginal lead in Handwritten Text (87.4 vs. GLM-OCR’s 87.0), GLM-OCR remains highly effective. Crucially, in practical application tasks like Receipt KIE, GLM-OCR (94.5) easily surpasses proprietary models like GPT-5.2 (83.5). These in-house results validate that GLM-OCR is not merely optimizing for academic datasets but is highly capable of generalizing to the noisy, variable conditions of real-world OCR deployments.

## 4 Inference and Deployment

### 4.1 Local Deployment and SDK Integration

With a compact parameter scale of 0.9B, GLM-OCR is highly optimized for localized inference and resource-constrained environments. The model supports efficient deployment across mainstream frameworks, including vLLM, SGLang, and Ollama. To facilitate seamless integration, a comprehensive SDK is provided for end-to-end document parsing workflows 888https://github.com/zai-org/GLM-OCR.

To assess operational efficiency, we conducted a comparative throughput analysis of various OCR pipelines. Under identical hardware configurations and testing conditions (single replica, single concurrency), we evaluated the parsing and Markdown-export speeds for both image and PDF inputs. As demonstrated in Table 6, GLM-OCR outperforms comparable methods, achieving a throughput of 1.86 pages/second for PDF documents and 0.67 images/second for standalone image files.

Model
Image Input
PDF Input

(pages / s)
(pages / s)

GLM-OCR
0.67
1.86

PaddleOCR-VL-1.5
0.39
1.22

Deepseek-OCR2
0.32
−-

MinerU2.5
0.18
0.48

dots.ocr
0.10
−-

### 4.2 Model-as-a-Service (MaaS) API

For cloud-based deployments, GLM-OCR is accessible via a MaaS API
 999https://docs.bigmodel.cn/cn/guide/models/vlm/glm-ocr. The service employs a highly cost-effective, unified pricing model for both input and output tokens, set at 0.2 RMB per million tokens. Under this pricing structure, an expenditure of 1 RMB is sufficient to process approximately 2,000 A4-sized scanned images or 200 simple-layout PDFs (10 pages each). This represents a significant reduction in operational overhead, decreasing processing costs to approximately one-tenth of those associated with traditional OCR solutions.

### 4.3 Fine-Tuning Capabilities

In scenarios where specific domain adaptation or enhanced task performance is required, GLM-OCR supports direct fine-tuning utilizing the LLaMA-Factory framework. Comprehensive tutorials and configuration guidelines for the fine-tuning process are available in the official repository: https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md.

## 5 Intended Use Cases

### 5.1 Overview

GLM-OCR is designed to support both high-level document understanding workflows and lightweight optical character recognition (OCR) tasks. Depending on application requirements, users may (1) integrate the GLM-OCR SDK for complex document parsing pipelines, or (2) directly invoke the base model for focused recognition and structured information extraction tasks. This section describes the two primary usage paradigms and their representative application scenarios.

### 5.2 Document Parsing with GLM-OCR SDK

The GLM-OCR SDK 101010https://github.com/zai-org/GLM-OCR provides a comprehensive interface for performing document parsing, including layout-aware parsing, multimodal recognition, and structured output generation. It is intended for enterprise-grade or production-level workflows where documents may contain heterogeneous content such as paragraphs, tables, mathematical expressions, and key-value pairs.

The output is generated in a structured markdown format, preserving logical document hierarchy and structural relationships. The example demonstrates that the SDK supports end-to-end parsing of heterogeneous documents while maintaining structural fidelity and semantic coherence.

### 5.3 Lightweight OCR and Information Extraction with the Base Model

In addition to the SDK, GLM-OCR can be directly used as a standalone model for lightweight OCR and targeted extraction tasks. This mode is appropriate for scenarios requiring lower integration overhead, flexible prompting, or rapid prototyping.

All tasks in this mode are controlled via explicit prompt instructions. Below we describe four primary task categories.

#### 5.3.1 Text Recognition

Prompt:

```
Text Recognition:
```

This mode is used for general printed or handwritten text transcription. The model outputs plain text corresponding to the visible textual content in the input image.

##### Example Scenario.

Figure 4 illustrates a real-world text recognition scenario using a restaurant menu board containing multilingual content (e.g., Italian dish names and English phrases) and price annotations with special characters (e.g., the euro symbol). The input image exhibits typical challenges such as handwritten-style fonts, varying character sizes, non-uniform spacing, perspective distortion, and background clutter.

Despite these complexities, the model accurately transcribes the textual content while preserving the original line structure and semantic grouping. In particular, it correctly reconstructs line breaks, capitalization, punctuation, numerical values, and currency symbols (e.g., “5,10€”, “14,00”), demonstrating robustness to layout variations and moderate visual noise. This example highlights the model’s ability to perform reliable optical character recognition in unconstrained, real-world environments.

#### 5.3.2 Table Recognition

Prompt:

```
Table Recognition:
```

This task focuses on recovering the structural representation of tabular data. The output may be formatted as Markdown tables or structured text that preserves row and column alignment.

##### Example Scenario.

Figure 5 presents a clinical summary table containing hierarchical column headers, merged cells, percentage values, missing-value indicators, and statistical annotations (e.g., p-values). The input image exhibits typical document-analysis challenges, including low contrast, dense numerical content, multi-level header grouping, and complex row–column alignment.

GLM-OCR accurately reconstructs the logical table structure by identifying column groups (e.g., surgical vs. non-surgical cohorts), preserving header hierarchies, and correctly aligning numerical entries with their corresponding attributes. In addition to faithfully transcribing cell content, the model maintains structural consistency such as row ordering, column correspondence, and percentage–value pairing. This structured reconstruction enables direct conversion into machine-readable formats (e.g., CSV or spreadsheet tables), thereby facilitating downstream statistical analysis, data validation, and automated reporting workflows.

#### 5.3.3 Formula Recognition

Prompt:

```
Formula Recognition:
```

This mode is intended for recognizing mathematical expressions and converting them into structured formats (e.g., LaTeX).

##### Example Scenario.

Given an image containing inline and display equations from a scientific manuscript, the model transcribes the formulas into syntactically valid LaTeX expressions. The output preserves operators, superscripts, subscripts, and fraction structures, enabling direct reuse in academic or technical documentation.

As depicted in Figure 6, given an image containing dense mathematical equations, GLM-OCR accurately transcribes the visual content into syntactically valid LaTeX expressions. The model demonstrates high fidelity in parsing complex two-dimensional spatial layouts, successfully preserving intricate structural elements such as matrices, determinants, and multi-level subscripts. This precise reconstruction eliminates the need for manual correction, enabling direct and seamless reuse in academic and technical documentation.

#### 5.3.4 Key Information Extraction

For structured information extraction tasks, the prompt must explicitly specify a strict JSON schema. The model is expected to generate output that conforms to the provided format.

##### Example Scenario.

In a representative example involving a dense customs declaration form, the model successfully extracts structured fields based on a detailed input prompt. It accurately populates complex, nested JSON entities—such as shipper details, unified social credit codes, and itemized goods information—directly from the visual layout. The output strictly adheres to the user-provided JSON schema, eliminating hallucinated keys and facilitating seamless integration into automated processing pipelines or structured databases.

### 5.4 Summary of Usage Paradigms

The two usage modes address complementary needs:

- •

GLM-OCR SDK is designed for comprehensive, layout-aware, multi-element document parsing in production environments.

- •

Base model prompting enables lightweight, flexible OCR and targeted information extraction for modular or task-specific applications.

Together, these paradigms allow GLM-OCR to support a broad spectrum of document understanding workflows, ranging from rapid prototyping to enterprise-scale deployment.

## 6 Limitations

Although GLM-OCR demonstrates competitive performance across diverse benchmarks and practical scenarios, several limitations remain.

### 6.1 Two-Stage Architectural Constraints

The current two-stage pipeline, consisting of layout analysis followed by region-level recognition, may introduce error propagation. In cases of inaccurate layout detection, downstream recognition performance may degrade. Additionally, complex layouts involving cross-page dependencies or irregular multi-column structures may lead to imperfect reading order reconstruction.

### 6.2 Data Coverage Limitations

Model performance is influenced by the distribution and diversity of training data. Degradation may occur in scenarios involving:

- •

Extremely low-resolution or heavily distorted documents,

- •

Highly complex mathematical expressions,

- •

Dense or irregular tabular structures,

- •

Underrepresented languages in the training corpus.

### 6.3 Structured Output Variability

As a generative model, GLM-OCR may exhibit minor stochastic variation in formatting behaviors, particularly in line breaks and whitespace handling. Although reinforcement learning and structural supervision mitigate this effect, strict formatting guarantees cannot be fully ensured.

### 6.4 Key Information Extraction

While the model supports prompt-based KIE, extraction accuracy depends on prompt specification and schema clarity. In complex forms with implicit or ambiguous field boundaries, incomplete or redundant outputs may occur.

These limitations represent areas for continued research and system refinement.

## 7 Conclusion

This report introduces GLM-OCR as a practical solution for structured document understanding under real-world system constraints. Instead of relying on large model scaling, the design prioritizes controllable latency, memory efficiency, and structured output reliability. Through the combination of layout-aware preprocessing and multi-token decoding, the system improves throughput and stability while maintaining competitive recognition accuracy across diverse document types. The results indicate that careful alignment between model architecture, decoding strategy, and task structure can yield substantial efficiency gains without increasing parameter scale.

From an engineering perspective, GLM-OCR demonstrates that document intelligence systems benefit from modular pipelines, efficient generation mechanisms, and deployment-oriented optimization. The model supports local inference, cloud-based serving, and domain-specific fine-tuning, enabling integration into heterogeneous production environments. Future development will focus on improving robustness under extreme layout complexity, enhancing multilingual coverage, and strengthening structured output consistency to further reduce downstream integration costs.

## References

- [1]
J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. (2023)

Gpt-4 technical report.

arXiv preprint arXiv:2303.08774.

Cited by: Table 4.

- [2]
S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu (2025)

Qwen3-vl technical report.

arXiv preprint arXiv:2511.21631.

Cited by: §1,
Table 4.

- [3]
S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025)

Qwen2. 5-vl technical report.

arXiv preprint arXiv:2502.13923.

Cited by: Table 4.

- [4]
chatdoc-com (2025)

OCRFlux.

Note: https://github.com/chatdoc-com/OCRFluxAccessed:2025-09-25

Cited by: Table 4.

- [5]
C. Cui, T. Sun, S. Liang, T. Gao, Z. Zhang, J. Liu, X. Wang, C. Zhou, H. Liu, M. Lin, Y. Zhang, Y. Zhang, Y. Liu, D. Yu, and Y. Ma (2026)

PaddleOCR-vl-1.5: towards a multi-task 0.9b vlm for robust in-the-wild document parsing.

External Links: 2601.21957,
Link

Cited by: §1,
Table 4.

- [6]
C. Cui, T. Sun, S. Liang, T. Gao, Z. Zhang, J. Liu, X. Wang, C. Zhou, H. Liu, M. Lin, Y. Zhang, Y. Zhang, H. Zheng, J. Zhang, J. Zhang, Y. Liu, D. Yu, and Y. Ma (2025)

PaddleOCR-vl: boosting multilingual document parsing via a 0.9b ultra-compact vision-language model.

External Links: 2510.14528,
Link

Cited by: Table 4.

- [7]
C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, C. Zhou, H. Liu, Y. Zhang, W. Lv, K. Huang, Y. Zhang, J. Zhang, J. Zhang, Y. Liu, D. Yu, and Y. Ma (2025)

PaddleOCR 3.0 technical report.

External Links: 2507.05595,
Link

Cited by: §1,
Table 4.

- [8]
H. Feng, S. Wei, X. Fei, W. Shi, Y. Han, L. Liao, J. Lu, B. Wu, Q. Liu, C. Lin, et al. (2025)

Dolphin: document image parsing via heterogeneous anchor prompting.

arXiv preprint arXiv:2505.14059.

Cited by: Table 4,
Table 4.

- [9]
T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin, D. Rojas, G. Feng, H. Zhao, H. Lai, H. Yu, H. Wang, J. Sun, J. Zhang, J. Cheng, J. Gui, J. Tang, J. Zhang, J. Li, L. Zhao, L. Wu, L. Zhong, M. Liu, M. Huang, P. Zhang, Q. Zheng, R. Lu, S. Duan, S. Zhang, S. Cao, S. Yang, W. L. Tam, W. Zhao, X. Liu, X. Xia, X. Zhang, X. Gu, X. Lv, X. Liu, X. Liu, X. Yang, X. Song, X. Zhang, Y. An, Y. Xu, Y. Niu, Y. Yang, Y. Li, Y. Bai, Y. Dong, Z. Qi, Z. Wang, Z. Yang, Z. Du, Z. Hou, and Z. Wang (2024)

ChatGLM: a family of large language models from glm-130b to glm-4 all tools.

External Links: 2406.12793

Cited by: §1.

- [10]
Google DeepMind (2025)

Gemini 2.5.

Note: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/

Cited by: Table 4.

- [11]
Google DeepMind (2025)

Gemini 3.0.

Note: https://blog.google/products-and-platforms/products/gemini/gemini-3-collection/

Cited by: Table 4.

- [12]
JaidedAI (2020)

EasyOCR.

Note: https://github.com/JaidedAI/EasyOCR

Cited by: §1.

- [13]
W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica (2023)

Efficient memory management for large language model serving with pagedattention.

In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,

Cited by: §1.

- [14]
Z. Li, Y. Liu, Q. Liu, Z. Ma, Z. Zhang, S. Zhang, Z. Guo, J. Zhang, X. Wang, and X. Bai (2025)

MonkeyOCR: document parsing with a structure-recognition-relation triplet paradigm.

arXiv preprint arXiv:2506.05218.

Cited by: Table 4,
Table 4,
Table 4.

- [15]
A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. (2024)

Deepseek-v3 technical report.

arXiv preprint arXiv:2412.19437.

Cited by: §1.

- [16]
Y. Liu, Z. Zhao, L. Tian, H. Wang, X. Ye, Y. You, Z. Yu, C. Wu, X. Zhou, Y. Yu, et al. (2025)

POINTS-reader: distillation-free adaptation of vision-language models for document conversion.

arXiv preprint arXiv:2509.01215.

Cited by: Table 4.

- [17]
Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X. Yin, C. Liu, L. Jin, and X. Bai (2024-12)

OCRBench: on the hidden mystery of ocr in large multimodal models.

Science China Information Sciences 67 (12).

External Links: ISSN 1869-1919,
Link,
Document

Cited by: §1.

- [18]
S. Mandal, N. Gupta, A. Talewar, P. Ahuja, P. Juvatkar, and G. Banda (2025)

IDPLeaderboard: a unified leaderboard for intelligent document processing tasks.

Note: https://idp-leaderboard.org

Cited by: §1.

- [19]
S. Mandal, A. Talewar, P. Ahuja, and P. Juvatkar (2025)

Nanonets-ocr-s: a model for transforming documents into structured markdown with intelligent content recognition and semantic tagging.

Cited by: Table 4.

- [20]
Mistral AI Team (2025)

Mistral-ocr.

Note: https://mistral.ai/news/mistral-ocr?utm_source=ai-bot.cn

Cited by: Table 4.

- [21]
J. Niu, Z. Liu, Z. Gu, B. Wang, L. Ouyang, Z. Zhao, T. Chu, T. He, F. Wu, Q. Zhang, et al. (2025)

MinerU2. 5: a decoupled vision-language model for efficient high-resolution document parsing.

arXiv preprint arXiv:2509.22186.

Cited by: Table 4.

- [22]
OpenAI. (2025)

GPT-5.2 system card.

External Links: Link

Cited by: Table 4.

- [23]
opendatalab (2025)

MinerU2.0-2505-0.9b.

Note: https://huggingface.co/opendatalab/MinerU2.0-2505-0.9B

Cited by: Table 4,
Table 4.

- [24]
L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, J. Shi, F. Wu, P. Chu, M. Liu, Z. Li, C. Xu, B. Zhang, B. Shi, Z. Tu, and C. He (2024)

OmniDocBench: benchmarking diverse pdf document parsing with comprehensive annotations.

External Links: 2412.07626,
Link

Cited by: §1.

- [25]
V. Paruchuri (2025)

Marker.

Note: https://github.com/datalab-to/markerAccessed: 2025-09-25

Cited by: Table 4.

- [26]
J. Poznanski, J. Borchardt, J. Dunkelberger, R. Huff, D. Lin, A. Rangapur, C. Wilhelm, K. Lo, and L. Soldaini (2025)

Olmocr: unlocking trillions of tokens in pdfs with vision language models.

arXiv preprint arXiv:2502.18443.

Cited by: Table 4.

- [27]
rednote-hilab (2025)

Dots.ocr: multilingual document layout parsing in a single vision-language model.

Cited by: Table 4.

- [28]
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. (2024)

Deepseekmath: pushing the limits of mathematical reasoning in open language models.

arXiv preprint arXiv:2402.03300.

Cited by: §2.2.

- [29]
R. Smith (2007)

An overview of the tesseract ocr engine.

In Ninth international conference on document analysis and recognition (ICDAR 2007),

Vol. 2, pp. 629–633.

Cited by: §1.

- [30]
B. S. Team (2025)

Seed1.5-vl technical report.

arXiv preprint arXiv:2505.07062.

Cited by: §1.

- [31]
V. Team, W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan, S. Duan, W. Wang, Y. Wang, Y. Cheng, Z. He, Z. Su, Z. Yang, Z. Pan, A. Zeng, B. Wang, B. Chen, B. Shi, C. Pang, C. Zhang, D. Yin, F. Yang, G. Chen, J. Xu, J. Zhu, J. Chen, J. Chen, J. Chen, J. Lin, J. Wang, J. Chen, L. Lei, L. Gong, L. Pan, M. Liu, M. Xu, M. Zhang, Q. Zheng, S. Yang, S. Zhong, S. Huang, S. Zhao, S. Xue, S. Tu, S. Meng, T. Zhang, T. Luo, T. Hao, T. Tong, W. Li, W. Jia, X. Liu, X. Zhang, X. Lyu, X. Fan, X. Huang, Y. Wang, Y. Xue, Y. Wang, Y. Wang, Y. An, Y. Du, Y. Shi, Y. Huang, Y. Niu, Y. Wang, Y. Yue, Y. Li, Y. Zhang, Y. Wang, Y. Wang, Y. Zhang, Z. Xue, Z. Hou, Z. Du, Z. Wang, P. Zhang, D. Liu, B. Xu, J. Li, M. Huang, Y. Dong, and J. Tang (2025)

GLM-4.5v and glm-4.1v-thinking: towards versatile multimodal reasoning with scalable reinforcement learning.

External Links: 2507.01006,
Link

Cited by: §1,
§1.

- [32]
B. Wang, Z. Gu, G. Liang, C. Xu, B. Zhang, B. Shi, and C. He (2024)

UniMERNet: a universal network for real-world mathematical expression recognition.

External Links: 2404.15254,
Link

Cited by: §1.

- [33]
W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. (2025)

Internvl3. 5: advancing open-source multimodal models in versatility, reasoning, and efficiency.

arXiv preprint arXiv:2508.18265.

Cited by: Table 4.

- [34]
H. Wei, Y. Sun, and Y. Li (2025)

DeepSeek-ocr: contexts optical compression.

arXiv preprint arXiv:2510.18234.

Cited by: Table 4.

- [35]
A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, B. Chen, D. Yin, C. Ge, C. Xie, C. Wang, et al. (2026)

GLM-5: from vibe coding to agentic engineering.

arXiv preprint arXiv:2602.15763.

Cited by: §1.

- [36]
X. Zhong, E. ShafieiBavani, and A. J. Yepes (2019)

Image-based table recognition: data, model, and evaluation.

arXiv preprint arXiv:1911.10683.

Cited by: §1.

- [37]
J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. (2025)

Internvl3: exploring advanced training and test-time recipes for open-source multimodal models.

arXiv preprint arXiv:2504.10479.

Cited by: Table 4.

Generated on Sun Apr 5 21:30:05 2026 by LaTeXML
