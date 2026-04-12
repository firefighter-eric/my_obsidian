# Zheng et al. - 2025 - PPTAgent Generating and Evaluating Presentations Beyond Text-to-Slides(2)

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Zheng et al. - 2025 - PPTAgent Generating and Evaluating Presentations Beyond Text-to-Slides(2).html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2501.03936
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Agent: Generating and Evaluating Presentations Beyond Text-to-Slides

Hao Zheng1,2,
 These authors contributed equally

Xinyan Guan1,2,∗

Hao Kong3

Jia Zheng1

Hongyu Lin1
Yaojie Lu1

Ben He1,2

Xianpei Han1

Le Sun1

1Chinese Information Processing Laboratory

 Institute of Software

 Chinese Academy of Sciences
2University of Chinese Academy of Sciences 
3Shanghai Jiexin Technology 
{zhenghao2022,guanxinyan2022,zhengjia,hongyu,luyaojie}@iscas.ac.cn
{xianpei,sunle}@iscas.ac.cn haokong@knowuheart.com

###### Abstract

Automatically generating presentations from documents is a challenging task that requires balancing content quality, visual design, and structural coherence. Existing methods primarily focus on improving and evaluating the content quality in isolation, often overlooking visual design and structural coherence, which limits their practical applicability.
To address these limitations, we propose Agent, which comprehensively improves presentation generation through a two-stage, edit-based approach inspired by human workflows.
Agent first analyzes reference presentations to understand their structural patterns and content schemas, then drafts outlines and generates slides through code actions to ensure consistency and alignment.
To comprehensively evaluate the quality of generated presentations, we further introduce Eval, an evaluation framework that assesses presentations across three dimensions: Content, Design, and Coherence. Experiments show that Agent significantly outperforms traditional automatic presentation generation methods across all three dimensions. The code and data are available at https://github.com/icip-cas/PPTAgent.

Agent: Generating and Evaluating Presentations Beyond Text-to-Slides

## 1 Introduction

Presentations are a widely used medium for information delivery, valued for their visual effectiveness in engaging and communicating with audiences.
However, creating high-quality presentations requires a captivating storyline, visually appealing layouts, and rich, impactful content (Fu et al., 2022).
Consequently, creating well-rounded presentations requires advanced presentation skills and significant effort.
Given the inherent complexity of presentation creation, there is growing interest in automating the presentation generation process (Mondal et al., 2024; Maheshwari et al., 2024) by leveraging the generalization capabilities of large language models (LLM).

Existing approaches often adopt an end-to-end text-generation paradigm, focusing solely on textual content while neglecting layout design and presentation structures, making them impractical for real-world applications. For example, as shown in Figure 1, prior studies (Mondal et al., 2024; Sefid et al., 2021) treat presentation generation as an abstractive summarization task, focus primarily on textual content while overlooking the interactive nature of presentations. This results in simplistic and visually uninspiring outputs that fail to engage audiences.

However, automatically creating visually rich and structurally clear presentations remains challenging due to the complexity of data formats and the lack of effective evaluation frameworks. First, most presentations are saved in PowerPoint’s XML format, which is inherently tedious and redundant (Gryk, 2022). This complex format poses significant challenges for LLMs in interpreting the presentation layout and structure, let alone generating appealing slides in an end-to-end fashion.
Second, and more importantly, the absence of comprehensive evaluation frameworks exacerbates this issue. Current metrics like perplexity and ROUGE (Lin, 2004) fail to capture essential aspects of presentation quality such as narrative flow, visual design, and content impact. Moreover, ROUGE-based evaluation tends to reward excessive textual alignment with input documents, undermining the brevity and clarity crucial for effective presentations.
These limitations highlight the urgent need for advancements in automated presentation generation, particularly in enhancing visual design and developing comprehensive evaluation frameworks.

Rather than creating complex presentations from scratch in a single pass, presentations are typically created by selecting exemplary slides as references and then summarizing and transferring key content onto them (Duarte, 2010).
Inspired by this process, we design Agent to decompose presentation generation into an iterative, edit-based workflow, as illustrated in Figure 2. In the first stage, given a document and a reference presentation, Agent analyzes the reference presentations to extract semantic information, providing the textual description that identifies the purpose and data model of each slide. In the Presentation Generation stage, Agent generates a detailed presentation outline and assigns specific document sections and reference slides to each slide. For instance, the framework selects the opening slide as the reference slide to present meta-information, such as the title and icon. Agent offers a suite of editing action APIs that empower LLMs to dynamically modify the reference slide. By breaking down the process into discrete stages rather than end-to-end generation, this approach ensures consistency, adaptability, and seamless handling of complex formats.

To comprehensively evaluate the quality of generated presentations, we propose Eval, a multidimensional evaluation framework. Inspired by Chen et al. (2024a) and Kwan et al. (2024), Eval leverages the MLLM-as-a-judge paradigm to enable systematic and scalable evaluation. Drawing from Duarte (2010), we categorized presentation quality into three dimensions: Content, Design, and Coherence, providing both quantitative scores and qualitative feedback for each dimension. Our human evaluation studies validated the reliability and effectiveness of Eval.

Results demonstrate that our method effectively generates high-quality presentations, achieving an average score of 3.67 across the three dimensions evaluated by Eval.
These results, covering a diverse range of domains, highlight a high success rate of 97.8%, showcasing the versatility and robustness of our approach.

Our main contributions can be summarized as follows:

- •

We propose Agent, a novel framework that redefines automatic presentation generation as an edit-based workflow guided by reference presentations.

- •

We introduce Eval, the first comprehensive evaluation framework that assesses presentations across three key dimensions: Content, Design, and Coherence.

- •

We publicly released the Agent and Eval codebase, along with a curated presentation dataset, to facilitate future research in automatic presentation generation.

## 2 PPTAgent

In this section, we first establish the formulation of the presentation generation task.
Subsequently, we describe the framework of our proposed Agent, which operates in two distinct stages.
In stage I, we analyze the reference presentation by clustering similar slides and extracting their content schemas. This process aims to enhance the expressiveness of the reference presentation, thereby facilitating subsequent presentation generation.
In stage II, given an input document and the analyzed reference presentation, we aim to select the most suitable slides and generate the target presentation through an interactive editing process based on the selected slides.
An overview of our proposed workflow is illustrated in Figure 2.

### 2.1 Problem Formulation

Agent is designed to generate an engaging presentation via an edit-based process. We will provide formal definitions for both Agent and the conventional method, illustrating their divergence.

The conventional method for creating each slide 𝑺\boldsymbol{S} can be described in Equation 1, where nn represents the number of elements on the slide, and CC denotes the source content composed of sections and figures. Each element on the slide, eie_{i}, is defined by its type, content, and styling attributes, such as (Textbox,"Hello",{border,size,position,…})(\textrm{Textbox},\textrm{"Hello"},\{\textrm{border},\textrm{size},\textrm{position},\dots\}).

𝑺=∑i=1nei=f​(C)\boldsymbol{S}=\sum_{i=1}^{n}e_{i}=f(C)

(1)

Compared to the conventional method, Agent adopts an edit-based generation paradigm for creating new slides, addressing challenges in processing spatial relationships and designing styles. This approach generates a sequence of actions to modify existing slides. Within this paradigm, both the input document and the reference presentation serve as inputs.
This process can be described as Equation 2, where mm represents the number of generated actions. Each action aia_{i} represents a line of executable code, and RjR_{j} is the reference slide being edited.

𝑨=∑i=1mai=f​(C∣Rj)\boldsymbol{A}=\sum_{i=1}^{m}a_{i}=f\left(C\mid R_{j}\right)

(2)

### 2.2 Stage I: Presentation Analysis

To facilitate presentation generation, we first cluster slides in the reference presentation and extract their content schemas. This structured semantic representation helps LLMs determine which slides to edit and what content to convey in each slide.

#### Slide Clustering

Slides can be categorized into two main types based on their functionalities: slides that support the structure of the presentation (e.g., opening slides) and slides that convey specific content (e.g., bullet-point slides). We employ different clustering algorithms to effectively cluster slides in the presentation based on their textual or visual characteristics. For structural slides, we leverage LLMs to infer the functional role of each slide and group them accordingly, as these slides often exhibit distinctive textual features. For the remaining slides, which primarily focus on presenting specific content, we employ a hierarchical clustering approach leveraging image similarity. For each cluster, we infer the layout patterns of each cluster using MLLMs. Further details regarding this method can be found in Appendix C.

#### Schema Extraction

After clustering slides to facilitate the selection of slide references, we further analyzed their content schemas to ensure purposeful alignment of the editing. Given the complexity and fragmentation of real-world slides, we utilized the context perception capabilities of LLMs (Chen et al., 2024a) to extract diverse content schemas. Specifically, we defined an extraction framework where each element is represented by its category, modality, and content. Based on this framework, the schema of each slide was extracted through LLMs’ instruction-following and structured output capabilities. Detailed instructions are provided in Appendix E.

### 2.3 Stage II: Presentation Generation

In this stage, we begin by generating an outline that specifies the reference slide and relevant content for each slide in the new presentation. For each slide, LLMs iteratively edit the reference slide using interactive executable code actions to complete the generation process.

#### Outline Generation

Following human preferences, we instruct LLMs to create a structured outline composed of multiple entries. Each entry specifies the reference slide, relevant document section indices, as well as the title and description of the new slide. By utilizing the planning and summarizing capabilities of LLMs, we provide both the document and semantic information extracted from the reference presentation to generate a coherent and engaging outline for the new presentation, which subsequently orchestrates the generation process.

#### Slide Generation

Guided by the outline, the slide generation process iteratively edits a reference slide to produce the new slide. To enable precise manipulation of slide elements, we implement five specialized APIs that allow LLMs to edit, remove, and duplicate text elements, as well as edit and remove visual elements. To further enhance the comprehension of slide structure, inspired by Feng et al. (2024) and Tang et al. (2023), we convert slides from their raw XML format into an HTML representation, which is more interpretable for LLMs. For each slide, LLMs receive two types of input: text retrieved from the source document based on section indices, and captions of available images. The new slide content is then generated following the guidance of the content schema.

Subsequently, LLMs leverage the generated content, HTML representation of the reference slide, and API documentation to produce executable editing actions. These actions are executed in a REPL111https://en.wikipedia.org/wiki/Read-eval-print_loop environment, where the system detects errors during execution and provides real-time feedback for self-correction. The self-correction mechanism leverages intermediate results to iteratively refine the editing actions, enhancing the robustness of the generation process.

## 3 PPTEval

To address the limitations of existing automated metrics for presentation evaluation, we introduce Eval, a comprehensive framework for assessing presentation quality from multiple perspectives. The framework provides scores on a 1-to-5 scale and offers detailed feedback to guide the improvement of future presentation generation methods. The overall evaluation process is depicted in Figure 3, with the detailed scoring criteria and examples provided in Appendix B.

Drawing from Duarte (2008, 2010), we have identified three key dimensions for evaluating presentation quality:

#### Content:

The content dimension evaluates the information presented on the slides, focusing on both text and images. We assess content quality from three perspectives: the amount of information, the clarity and quality of textual content, and the support provided by visual content.
High-quality textual content is characterized by clear, impactful text that conveys the proper amount of information. Additionally, images should complement and reinforce the textual content, making the information more accessible and engaging. To evaluate content quality, we employ MLLMs on slide images, as slides cannot be easily comprehended in a plain text format.

#### Design:

Good design not only captures attention but also enhances content delivery. We evaluate the design dimension based on three aspects: color schemes, visual elements, and overall design. Specifically, the color scheme of the slides should have clear contrast to highlight the content while maintaining harmony. The use of visual elements, such as geometric shapes, can make the slide design more expressive. Finally, good design should adhere to basic design principles, such as avoiding overlapping elements and ensuring that design does not interfere with content delivery.

#### Coherence:

Coherence is essential for maintaining audience engagement in a presentation. We evaluate coherence based on the logical structure and the contextual information provided. Effective coherence is achieved when the model constructs a captivating storyline, enriched with contextual information that enables the audience to follow the content seamlessly. We assess coherence by analyzing the logical structure and contextual information extracted from the presentation.

## 4 Experiment

### 4.1 Dataset

#### Data Collection

Existing presentation datasets, such as Mondal et al. (2024); Sefid et al. (2021); Sun et al. (2021); Fu et al. (2022), have two main issues. First, they are mostly stored in PDF or JSON formats, which leads to a loss of semantic information, such as structural relationships and styling attributes of elements. Additionally, these datasets are primarily derived from academic reports, limiting their diversity. To address these limitations, we introduce Zenodo10K, a new dataset sourced from Zenodo (European Organization For Nuclear Research and OpenAIRE, 2013), an open digital repository hosting diverse artifacts from different domains. We have curated 10,448 presentations from this source and made them publicly available to support further research.
Following Mondal et al. (2024), we sampled 50 presentations across five domains to serve as reference presentations. Additionally, we collected 50 documents from the same domains to be used as input documents. Details of the sampling criteria are provided in Appendix A.

Domain
Document
Presentation

#Chars
#Figs
#Chars
#Figs
#Pages

Culture
12,708
2.9
6,585
12.8
14.3

Education
12,305
5.5
3,993
12.9
13.9

Science
16,661
4.8
5,334
24.0
18.4

Society
13,019
7.3
3,723
9.8
12.9

Tech
18,315
11.4
5,325
12.9
16.8

Setting
Existing Metrics
PPTEval

Language Model
Vision Model
SR(%)↑\uparrow

PPL↓\downarrow

FID↓\downarrow

Content↑\uparrow

Design↑\uparrow

Coherence↑\uparrow

Avg.↑\uparrow

Baseline

GPT-4oLM{}_{\texttt{LM}}

–
–
110.6
–
2.98
2.33
3.24
2.85

Qwen2.5LM{}_{\texttt{LM}}

–
–
122.4
–
2.96
2.37
3.28
2.87

PPTAgent

GPT-4oLM{}_{\texttt{LM}}

GPT-4oVM{}_{\texttt{VM}}

97.8
459.7
7.48
3.25
3.24
4.39
3.62

Qwen2-VLLM{}_{\texttt{LM}}

Qwen2-VLVM{}_{\texttt{VM}}

43.0
322.3
7.32
3.13
3.34
4.07
3.51

Qwen2.5LM{}_{\texttt{LM}}

Qwen2-VLVM{}_{\texttt{VM}}

95.0
313.9
6.20
3.28
3.27
4.48
3.67

Ablation

PPTAgent
95.0
313.9
6.20
3.28
3.27
4.48
3.67

w/o Outline

91.0
2304.3
6.94
3.24
3.30
3.36
3.30

w/o Schema
78.8
164.8
7.12
3.08
3.23
4.04
3.45

w/o Structure

92.2
189.9
7.66
3.28
3.25
3.45
3.32

w/o CodeRender

74.6
231.0
7.03
3.27
3.34
4.38
3.66

Domain
SR (%)
PPL
FID
PPTEval

Culture
93.0
185.3
5.00
3.70

Education
94.0
249.0
7.90
3.69

Science
96.0
500.6
6.07
3.56

Society
95.0
396.8
5.32
3.59

Tech
97.0
238.7
6.72
3.74

#### Data Preprocessing

We utilized VikParuchuri (2023) to extract both textual and visual content from the documents. The extracted textual content was then organized into sections using Qwen2.5-72B-Instruct (Yang et al., 2024). For the visual content, captions were generated using Qwen2-VL-72B-Instruct (Wang et al., 2024a). To minimize redundancy, we identified and removed duplicate images if their image embeddings had a cosine similarity score exceeding 0.85. Similarly, slides were excluded if their text embeddings had a cosine similarity score above 0.8 compared to the preceding slide, as suggested by Fu et al. (2022). Detailed statistics of the dataset are presented in Table 1.

### 4.2 Experimental Settings and Baseline

#### Models

We evaluate our method using three state-of-the-art models: GPT-4o-2024-08-06 (GPT-4o), Qwen2.5-72B-Instruct (Qwen2.5, Yang et al., 2024), and Qwen2-VL-72B-Instruct (Qwen2-VL, Wang et al., 2024a). These models are categorized according to the specific modalities they handle, whether textual or visual, as indicated by their subscripts. Specifically, we define configurations as combinations of a language model (LM) and a vision model (VM), such as Qwen2.5LM+Qwen2-VLVM.

During experiments, we allow up to two iterations of self-correction per slide generation task, producing 5×10×10=5005\times 10\times 10=500 presentations per configuration. We use Chen et al. (2024b) and Wu et al. (2020) to compute the text and image embeddings respectively. All open-source LLMs are deployed using the VLLM framework (Kwon et al., 2023) on a cluster of 8 NVIDIA A100 GPUs. The total computational cost for these experiments is approximately 500 GPU hours.

#### Baseline

We adopt the methodology described in Bandyopadhyay et al. (2024) as our baseline. This approach employs a multi-staged end-to-end model to generate narrative-rich presentations, with an image similarity-based ranking algorithm to add images to the slides. The baseline method is evaluated using either GPT-4o or Qwen2.5, as it does not require the necessary processing of visual information. Each configuration generates 5×10=505\times 10=50 presentations, given that it does not require an input presentation. We do not report the success rate and FID of the baseline method for the same reason.

### 4.3 Evaluation Metrics

We evaluated the presentation generation using the following metrics:

- •

Success Rate (SR) measures the robustness of the generation task by determining the percentage of presentations where all slides are successfully generated.

- •

Perplexity (PPL) measures the likelihood of the language model generating the given sequence. Following Bandyopadhyay et al. (2024), we calculate the average perplexity of slides within a presentation using GPT-2.. A lower perplexity score indicates that the textual content is more fluent.

- •

FID (Heusel et al., 2017) measures the similarity between the generated presentation and the exemplar presentation in the feature space. Due to the limited sample size, we calculate the FID using a 64-dimensional output vector.

- •

PPTEval measures the comprehensive quality of presentations across three dimensions: coherence, content, and design. We employ GPT-4o as the judge model.

### 4.4 Result & Analysis

Table 2 presents the performance comparison between Agent and baseline methods, revealing that:

#### PPTAgent Enhances LLMs’ Presentation Generation Capabilities

As demonstrated in Table 2, our approach empowers LLMs to produce well-rounded presentations with a remarkable success rate, achieving ≥95%\geq 95\% success rate for both Qwen2.5LM{}_{\texttt{LM}}+Qwen2-VLVM{}_{\texttt{VM}} and GPT-4oLM{}_{\texttt{LM}}+GPT-4oVM{}_{\texttt{VM}}.
This is a significant improvement compared to the highest accuracy of 10% for session-based template editing tasks as reported in Guo et al. (2023). This improvement can be attributed to three main factors: 1) Agent concentrates on content modifying, thereby avoiding intricate stying operations. 2) Our streamlined API design allows LLMs to execute tasks with ease. 3) The code interaction module enhances LLMs’ comprehension of slides and offers opportunities for self-correction, enabling them to generate accurate actions robustly.
Moreover, detailed performance of Qwen2.5LM{}_{\texttt{LM}}+Qwen2-VLVM{}_{\texttt{VM}} across various domains, as illustrated in Table 3, underscores the robustness of our approach.

#### PPTAgent Significantly Improves Overall Presentation Quality

By adopting an Edit-based paradigm, Agent allows elements within the presentation to inherit well-designed styling attributes from existing presentations. When using GPT-4o, experimental results demonstrate comprehensive improvements over the baseline. We significantly surpass the baseline method in the design dimension under Eval (3.24 vs 2.33), as the presentations generated by the baseline method lack basic design efforts. Furthermore, we achieved substantial enhancements in coherence (4.39 vs 3.28) and content (3.25 vs 2.98) dimensions, as the semantic information extracted during the Presentation Analysis stage effectively guided the LLMs.

#### Open-Source LLMs Rival GPT-4o in Performance

GPT-4o consistently demonstrates outstanding performance across various evaluation metrics, highlighting its advanced capabilities. While Qwen2-VL exhibits limitations in linguistic proficiency due to the trade-offs from multimodal post-training, GPT-4o maintains a clear advantage in handling language tasks. However, the introduction of Qwen2.5 successfully mitigates these linguistic deficiencies, bringing its performance on par with GPT-4o, and achieving the best performance. This underscores the significant potential of open-source LLMs as competitive and highly capable presentation agents.

### 4.5 Ablation Study

To better understand the impact of each component in our proposed method, we performed ablation studies using four different configurations. Specifically, we evaluated the method by: (1) randomly selecting a slide as the edit target (w/o Outline), (2) omitting structural information during outline generation (w/o Structure), (3) replacing the slide representation with the method described in Guo et al. (2023) (w/o CodeRender), and (4) removing guidance from the slide schema (w/o Schema). These configurations were tested using the Qwen2.5LM{}_{\texttt{LM}}+Qwen2-VLVM{}_{\texttt{VM}}.

#### Code Representation Enhances LLMs’ Comprehension

As shown in Table 2, the removal of the Code Render component leads to a significant drop in the model’s success rate (SR) from 95.0 to 74.6. This underscores the critical role of code representation in leveraging LLMs’ coding capabilities to improve their overall comprehension.

#### Presentation Analysis is Essential for Generating Targeted Presentations

The removal of the outline and structural information significantly degrades coherence (from 4.48 to 3.36/3.45), underscoring their crucial role in maintaining logical flow. Furthermore, the absence of slide schema hinders LLMs from generating targeted content effectively, resulting in a drop in success rate from 95.0 to 78.8.

### 4.6 Error Analysis

Figure 4 illustrates the number of iterations required to generate a slide using different models. Although GPT-4o exhibits superior self-correction capabilities compared to Qwen2.5, Qwen2.5 encounters fewer errors in the first iteration (Iter-0). Additionally, we observed that Qwen2-VL experiences errors more frequently and has poorer self-correction capabilities, likely due to its multimodal post-training (Wang et al., 2024a). Ultimately, all three models successfully corrected more than half of the errors, demonstrating that our iterative self-correction mechanism effectively ensures the success of the generation process.

Corelation
Content
Design
Coherence
Avg.

Pearson
0.70
0.90
0.55
0.71

Spearman
0.73
0.88
0.57
0.74

### 4.7 Effectiveness of PPTEval

#### Human Agreement Evaluation

Despite Chen et al. (2024a) have highlighted the impressive human-like discernment of LLMs in various generation tasks. However, it remains crucial to assess the correlation between LLM evaluations and human evaluations in the context of presentations. This necessity arises from findings by Laskar et al. (2024), which indicate that LLMs may not be adequate evaluators for complex tasks. Table 4 shows the correlation of ratings between humans and LLMs. The average Pearson correlation of 0.71 exceeds the scores of other evaluation methods (Kwan et al., 2024), indicating that Eval aligns well with human preferences.

Moreover, the heatmap in Figure 5 reveals the limitations of existing metrics when compared with the Content and Design dimensions of Eval. In our experiments, we observed that PPL predominantly captures text fluency and is susceptible to the fragmented nature of slide text, leading to ineffective measurements with frequent outliers. Similarly, FID merely quantifies stylistic similarity to reference presentations rather than design quality, as conformity to reference styles does not necessarily indicate superior design. These findings underscore the necessity of Eval for comprehensive and effective presentation evaluation.

## 5 Related Works

#### Automated Presentation Generation

Recent proposed methods for slide generation can be categorized into rule-based and template-based
based on how they handle element placement. Rule-
based methods, such as those proposed by Mondal et al. (2024) and Li et al. (2021), often focus on enhancing textual content but neglect the visual-centric nature of presentations, leading to outputs that lack engagement. Template-based methods, including Cachola et al. (2024) and industrial solutions like Tongyi, rely on pre-designed templates to create visually appealing presentations. However, their dependence on extensive manual effort for template annotation significantly limits scalability and flexibility.

#### LLM Agent

Numerous studies (Li et al., 2024; Deng et al., 2024; Wang et al., 2024c) have explored the potential of LLMs to act as agents assisting humans in a wide array of tasks. For example, Zheng et al. (2024); Wang et al. (2024b) demonstrate the capability of LLMs to accomplish tasks by generating executable actions and correcting errors based on feedback. Furthermore, Guo et al. (2023) introduces an evaluation system that assesses the ability of LLMs to perform multi-turn, multimodal slide editing tasks using APIs, which inspired the use of LLMs for complex tasks as proposed in this study.

#### LLM as a Judge

LLMs have demonstrated strong capabilities in instruction following and context perception, leading to their widespread use as judges (Liu et al., 2023; Zheng et al., 2023). Further research by Zhuge et al. (2024) enhanced LLMs’ abilities through external modules and functions, while Chen et al. (2024a) validated the feasibility of using multimodal large language models(MLLMs) as judges. Additionally, Kwan et al. (2024) introduced a multi-dimensional evaluation framework for multi-turn conversations, which inspired the development of our proposed Eval.

## 6 Conclusion

In this paper, we introduced Agent, which conceptualizes presentation generation as a two-stage presentation editing task completed through the abilities of LLMs to understand and generate code. This approach leveraged the textual feature and layout patterns to organize slides into different functional groups. Our experiments across data from multiple domains have demonstrated the superiority of our method. Moreover, our proposed Eval ensured the assessability of presentations. This research provides a new paradigm for generating slides under unsupervised conditions and offers fresh insights for future work in presentation generation.

## 7 Limitations

While our method demonstrates its capability to produce high-quality presentations, there remain inherent challenges that impact its universal applicability. For instance, achieving a success rate of over 95% on our dataset is impressive, but not absolute, thus might limit its application. Moreover, parsing slides with intricate nested group shapes often proves to be a bottleneck, leading to less consistent results. Additionally, although Agent shows noticeable improvements in layout optimization over prior approaches, it still falls short of exploiting the full potential of visual cues for refining stylistic consistency. This often manifests in design flaws, such as overlapping elements, undermining the visual harmony of the generated slides. Addressing these limitations calls for future enhancements that integrate visual information into the generation process.

## 8 Ethical Considerations

In the construction of Zenodo10K, we utilized the publicly available API to scrape data while strictly adhering to the licensing terms associated with each artifact. Specifically, artifacts that were not permitted for modification or commercial use under their respective licenses were filtered out to ensure compliance with intellectual property rights. Additionally, all annotation personnel involved in the project were compensated at rates exceeding the minimum wage in their respective cities, reflecting our commitment to fair labor practices and ethical standards throughout the dataset’s development process.

## References

- Bandyopadhyay et al. (2024)

Sambaran Bandyopadhyay, Himanshu Maheshwari, Anandhavelu Natarajan, and Apoorv Saxena. 2024.

Enhancing presentation slide generation by llms with a multi-staged end-to-end approach.

arXiv preprint arXiv:2406.06556.

- Cachola et al. (2024)

Isabel Alyssa Cachola, Silviu Cucerzan, Allen Herring, Vuksan Mijovic, Erik Oveson, and Sujay Kumar Jauhar. 2024.

Knowledge-centric templatic views of documents.

In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15460–15476, Miami, Florida, USA. Association for Computational Linguistics.

- Chen et al. (2024a)

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yinuo Liu, Yaochen Wang, Huichi Zhou, Qihui Zhang, Pan Zhou, Yao Wan, and Lichao Sun. 2024a.

Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark.

arXiv preprint arXiv:2402.04788.

- Chen et al. (2024b)

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024b.

Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation.

arXiv preprint arXiv:2402.03216.

- Deng et al. (2024)

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. 2024.

Mind2web: Towards a generalist agent for the web.

Advances in Neural Information Processing Systems, 36.

- Duarte (2008)

Nancy Duarte. 2008.

Slide: ology: The art and science of creating great presentations, volume 1.

O’Reilly Media Sebastapol.

- Duarte (2010)

Nancy Duarte. 2010.

Resonate: Present visual stories that transform audiences.

John Wiley & Sons.

- European Organization For Nuclear Research and OpenAIRE (2013)

European Organization For Nuclear Research and OpenAIRE. 2013.

Zenodo.

- Feng et al. (2024)

Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2024.

Layoutgpt: Compositional visual planning and generation with large language models.

Advances in Neural Information Processing Systems, 36.

- Fu et al. (2022)

Tsu-Jui Fu, William Yang Wang, Daniel McDuff, and Yale Song. 2022.

Doc2ppt: Automatic presentation slides generation from scientific documents.

Proceedings of the AAAI Conference on Artificial Intelligence, 36(1):634–642.

- Gryk (2022)

Michael Robert Gryk. 2022.

Human readability of data files.

Balisage series on markup technologies, 27.

- Guo et al. (2023)

Yiduo Guo, Zekai Zhang, Yaobo Liang, Dongyan Zhao, and Duan Nan. 2023.

Pptc benchmark: Evaluating large language models for powerpoint task completion.

arXiv preprint arXiv:2311.01767.

- Heusel et al. (2017)

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017.

Gans trained by a two time-scale update rule converge to a local nash equilibrium.

Advances in neural information processing systems, 30.

- Kwan et al. (2024)

Wai-Chung Kwan, Xingshan Zeng, Yuxin Jiang, Yufei Wang, Liangyou Li, Lifeng Shang, Xin Jiang, Qun Liu, and Kam-Fai Wong. 2024.

Mt-eval: A multi-turn capabilities evaluation benchmark for large language models.

Preprint, arXiv:2401.16745.

- Kwon et al. (2023)

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023.

Efficient memory management for large language model serving with pagedattention.

In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626.

- Laskar et al. (2024)

Md Tahmid Rahman Laskar, Sawsan Alqahtani, M Saiful Bari, Mizanur Rahman, Mohammad Abdullah Matin Khan, Haidar Khan, Israt Jahan, Amran Bhuiyan, Chee Wei Tan, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty, and Jimmy Huang. 2024.

A systematic survey and critical review on evaluating large language models: Challenges, limitations, and recommendations.

In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13785–13816, Miami, Florida, USA. Association for Computational Linguistics.

- Li et al. (2021)

Da-Wei Li, Danqing Huang, Tingting Ma, and Chin-Yew Lin. 2021.

Towards topic-aware slide generation for academic papers with unsupervised mutual learning.

In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 13243–13251.

- Li et al. (2024)

Yanda Li, Chi Zhang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. 2024.

Appagent v2: Advanced agent for flexible mobile interactions.

arXiv preprint arXiv:2408.11824.

- Lin (2004)

Chin-Yew Lin. 2004.

ROUGE: A package for automatic evaluation of summaries.

In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

- Liu et al. (2023)

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023.

G-eval: NLG evaluation using gpt-4 with better human alignment.

In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

- Maheshwari et al. (2024)

Himanshu Maheshwari, Sambaran Bandyopadhyay, Aparna Garimella, and Anandhavelu Natarajan. 2024.

Presentations are not always linear! gnn meets llm for document-to-presentation transformation with attribution.

arXiv preprint arXiv:2405.13095.

- Mondal et al. (2024)

Ishani Mondal, S Shwetha, Anandhavelu Natarajan, Aparna Garimella, Sambaran Bandyopadhyay, and Jordan Boyd-Graber. 2024.

Presentations by the humans and for the humans: Harnessing llms for generating persona-aware slides from documents.

In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2664–2684.

- Sefid et al. (2021)

Athar Sefid, Prasenjit Mitra, and Lee Giles. 2021.

Slidegen: an abstractive section-based slide generator for scholarly documents.

In Proceedings of the 21st ACM Symposium on Document Engineering, pages 1–4.

- Sun et al. (2021)

Edward Sun, Yufang Hou, Dakuo Wang, Yunfeng Zhang, and Nancy XR Wang. 2021.

D2s: Document-to-slide generation via query-based text summarization.

arXiv preprint arXiv:2105.03664.

- Tang et al. (2023)

Zecheng Tang, Chenfei Wu, Juntao Li, and Nan Duan. 2023.

Layoutnuwa: Revealing the hidden layout expertise of large language models.

arXiv preprint arXiv:2309.09506.

- VikParuchuri (2023)

VikParuchuri. 2023.

marker.

- Wang et al. (2024a)

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024a.

Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution.

arXiv preprint arXiv:2409.12191.

- Wang et al. (2024b)

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024b.

Executable code actions elicit better llm agents.

arXiv preprint arXiv:2402.01030.

- Wang et al. (2024c)

Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. 2024c.

Opendevin: An open platform for ai software developers as generalist agents.

arXiv preprint arXiv:2407.16741.

- Wu et al. (2020)

Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Zhicheng Yan, Masayoshi Tomizuka, Joseph Gonzalez, Kurt Keutzer, and Peter Vajda. 2020.

Visual transformers: Token-based image representation and processing for computer vision.

Preprint, arXiv:2006.03677.

- Yang et al. (2024)

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024.

Qwen2. 5 technical report.

arXiv preprint arXiv:2412.15115.

- Zheng et al. (2024)

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024.

Gpt-4v (ision) is a generalist web agent, if grounded.

arXiv preprint arXiv:2401.01614.

- Zheng et al. (2023)

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023.

Judging llm-as-a-judge with mt-bench and chatbot arena.

Advances in Neural Information Processing Systems, 36:46595–46623.

- Zhuge et al. (2024)

Mingchen Zhuge, Changsheng Zhao, Dylan Ashley, Wenyi Wang, Dmitrii Khizbullin, Yunyang Xiong, Zechun Liu, Ernie Chang, Raghuraman Krishnamoorthi, Yuandong Tian, et al. 2024.

Agent-as-a-judge: Evaluate agents with agents.

arXiv preprint arXiv:2410.10934.

## Appendix A Data Sampling

To maintain a reasonable cost, we selected presentations ranging from 12 to 64 pages and documents with text lengths from 2,048 to 20,480 characters.

## Appendix B Details of PPTEval

Through a Shanghai-based crowdsourcing platform, we recruited four graduate students to evaluate 50 randomly selected presentations from Zenodo10K, along with 100 presentations generated by the baseline method and our approach, respectively. The evaluations were conducted across three dimensions, as proposed by Eval, based on the same scoring criteria listed in Appendix E along with converted slide images.
Moreover, we listed some scoring examples in Figure 6 and detailed performance of Qwen2.5LM{}_{\texttt{LM}}+Qwen2-VLVM{}_{\texttt{VM}} across various domains in Table 3.

## Appendix C Layout Analysis

We present our hierarchical clustering algorithm for layout analysis in Algorithm 1, where slides are grouped into clusters using a similarity threshold θ\theta of 0.65. To minimize clustering interference, we replace the text and images in the slides with placeholders beforehand. Moreover, examples of the extracted slide clusters are provided in Figure 7.

## Appendix D Code Interaction

Our provided APIs and their corresponding functions are summarized in Table 5, with Figure 8 presenting an example of rendered HTML from a slide.

## Appendix E Prompts

### E.1 Prompts for Presentation Analysis

The prompts used for presentation analysis are illustrated in Figures 9, 10, and 11.

### E.2 Prompts for Presentation Generation

The prompts used for generating presentations are shown in Figures 12, 13, and 14.

### E.3 Prompts for PPTEval

The prompts used in PPTEval are depicted in Figures 15, 16, 17, 18, 19 and 20.

1:Input: Similarity matrix of slides S∈ℝN×NS\in\mathbb{R}^{N\times N}, similarity threshold θ\theta

2:Initialize: C←∅C\leftarrow\emptyset

3:while max⁡(S)≥θ\max(S)\geq\theta do

4:  (i,j)←arg⁡max⁡(S)(i,j)\leftarrow\arg\max(S)

5:  if ∃ck∈C​ such that ​(i∈ck∨j∈ck)\exists c_{k}\in C\text{ such that }(i\in c_{k}\lor j\in c_{k}) then

6:   ck←ck∪{i,j}c_{k}\leftarrow c_{k}\cup\{i,j\}

7:  else

8:   cnew←{i,j}c_{\text{new}}\leftarrow\{i,j\}

9:   C←C∪{cnew}C\leftarrow C\cup\{c_{\text{new}}\}

10:  end if

11:   Update SS:

12:     S​[:,i]←0S[:,i]\leftarrow 0, S​[i,:]←0S[i,:]\leftarrow 0

13:     S​[:,j]←0S[:,j]\leftarrow 0, S​[j,:]←0S[j,:]\leftarrow 0

14:end while

15:Return: CC

Function Name

Description

del_span

Deletes a specific span.

del_image

Deletes an image element.

clone_paragraph

Creates a duplicate of an existing paragraph.

replace_span

Replaces the content of a specific span.

replace_image

Replaces an image with a new image.

Generated on Wed Feb 5 16:52:35 2025 by LaTeXML
