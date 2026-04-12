# Lv et al. - 2023 - Kosmos-2.5 A Multimodal Literate Model

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Lv et al. - 2023 - Kosmos-2.5 A Multimodal Literate Model.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2309.11419
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Kosmos-2.5: A Multimodal Literate Model

Tengchao Lv, Yupan Huang11footnotemark: 1, Jingye Chen11footnotemark: 1, Lei Cui11footnotemark: 1 †, Shuming Ma, Yaoyao Chang,
Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang,
Cha Zhang, Furu Wei22footnotemark: 2 
Microsoft 
 aka.ms/GeneralAI

 Equal contribution. ††\dagger Corresponding author.

###### Abstract

We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling of multimodal large language models.

## 1 Introduction

Over the past several years, large language models (LLMs) have emerged as a critical area of research in artificial intelligence. These models are designed to learn from massive amounts of natural language data, allowing them to perform a wide range of language-related tasks with impressive accuracy. This development has been fueled by advancements in model scaling that enabled researchers to create models with unprecedented complexity. As a result, LLMs have become increasingly prevalent across various industries and applications, from customer service chatbots to virtual assistants and automated content creation. One notable trend in recent years has been the focus on building larger and more complex models, such as GPT-3 [4] and GPT-4 [45], which has hundreds/thousands of billion parameters and can generate compelling language outputs. While these models require significant computing resources to train and operate, they hold enormous potential for revolutionizing how we interact with and understand natural language.

Current LLMs primarily focus on textual information and cannot understand visual information. However, advancements in the field of multimodal large language models (MLLMs) aim to address this limitation. MLLMs combine visual and textual information within a single Transformer-based model, enabling the model to learn and generate content based on both modalities. MLLMs have shown promise in a variety of real-world applications, including natural image understanding and text image understanding. These models leverage the power of language modeling as a general interface for multimodal problems, allowing them to process and generate responses based on textual and visual inputs. While existing MLLMs have mainly focused on natural images with lower resolutions, the exploration of text images is an area that requires further investigation. Taking advantage of large-scale multimodal pre-training for text images is an important direction for MLLM research. By incorporating text images into the training process and developing models based on textual and visual information, we can unlock new possibilities for multimodal applications involving high-resolution text-intensive images.

In this study, we present Kosmos-2.5, a multimodal literate model that takes advantage of Kosmos-2 [47] designed to tackle machine reading of text-intensive images, which is shown in Figure 1.
Kosmos-2.5 performs two closely related transcription tasks in a unified multimodal model. The first task generates spatially-aware text blocks, assigning text lines their corresponding spatial coordinates within the original text-rich image. The second task produces structured text output, capturing styles and structures in the markdown format. Both tasks are conducted under a unified framework, leveraging a shared Transformer architecture, task-specific prompts, and flexible text representations. Specifically, our model architecture combines a ViT-based vision encoder and a Transformer-based language decoder linked by a resampler module. Our model is pre-trained on a large corpus of text-intensive images, whose text representations include text lines with bounding boxes and plain markdown texts.
By employing this dual-task training strategy, Kosmos-2.5 enhances its general-purpose multimodal literate capabilities. We assess the performance of Kosmos-2.5 on two tasks: end-to-end document-level text recognition and markdown-formatted image-to-text generation. Experiment results have demonstrated strong literate performance on several text-intensive image understanding tasks. In addition, Kosmos-2.5 also demonstrates promising capabilities in few-shot and zero-shot learning scenarios, offering a universal interface for real-world applications that involve text-rich images.

The contributions of this work are summarized as follows:

- •

Kosmos-2.5 represents a significant paradigm shift in text image understanding, transitioning from encoder-only/encoder-decoder models to a decoder-only model. It is pre-trained by incorporating dual transcription tasks (spatially-aware text block generation and structured markdown text generation) into a single, unified model architecture.

- •

This innovative method streamlines the application interface by integrating generative multimodal language modeling, simplifying the traditionally complex cascaded pipelines used for various downstream tasks.

- •

Furthermore, Kosmos-2.5 demonstrates impressive multimodal literate capabilities, thus setting the stage for future scaling of multimodal large language models.

## 2 Kosmos-2.5

### 2.1 Model Architecture

The model architecture of Kosmos-2.5 consists of a pre-trained vision encoder and a language decoder connected with a resampler module, shown in Figure 2.
We adopt the pre-trained vision encoder based on the Vision Transformer (ViT) [11].
We further adapt a Perceiver Resampler module with an attentive pooling mechanism to reduce the size of image embeddings [1].
The language decoder is built upon the Transformer-based decoder to condition on image and text context for the next token prediction.

### 2.2 Image and Text Representations

Kosmos-2.5 takes a composite input consisting of an image and a text representation. The image representation is uniform across various configurations and leverages a variable-resolution input strategy following Pix2Struct [33]. Precisely, we extract the maximum number of fixed-size patches (16×16161616\times 16) that can fit within a predefined sequence length L𝐿L. In addition, Resampler [1] is used as an attentive pooling mechanism to reduce the number of image embeddings.
The text representation, however, is more versatile and can be one of two types: text lines with bounding boxes or plain markdown texts.

Text lines with bounding boxes: For the layout-based document representation, text lines and their associated bounding boxes are extracted. Inspired by Kosmos-2 [47], we ground the text lines to their spatial positions in images by aligning their representations. The coordinates of these bounding boxes are then converted into discrete location tokens. Given that L𝐿L also represents the maximum length for each image dimension, we introduce a set of 2​L+22𝐿22L+2 specialized tokens. These tokens, <x0>, <x1>, …, <xL-1>, <y0>, …, <yL-1>, <bbox>, and </bbox>, correspond to the coordinates and the start and end of a bounding box. The coordinates are obtained by rounding down the actual position after resizing images.
Consider a document T𝑇T that comprises N𝑁N text lines. Each line is represented as 𝐓n={w1(n),w2(n),…,wMn(n)}subscript𝐓𝑛superscriptsubscript𝑤1𝑛superscriptsubscript𝑤2𝑛…superscriptsubscript𝑤subscript𝑀𝑛𝑛\mathbf{T}_{n}=\{w_{1}^{(n)},w_{2}^{(n)},\ldots,w_{M_{n}}^{(n)}\}, where Mnsubscript𝑀𝑛M_{n} is the number of words in the n𝑛n-th text line. The bounding box for 𝐓nsubscript𝐓𝑛\mathbf{T}_{n} is then denoted by 𝐁n=<bbox><​xtl(n)​><​ytl(n)​><​xbr(n)​><​ybr(n)​></bbox>subscript𝐁𝑛<bbox><superscriptsubscript𝑥tl𝑛><superscriptsubscript𝑦tl𝑛><superscriptsubscript𝑥br𝑛><superscriptsubscript𝑦br𝑛></bbox>\mathbf{B}_{n}=\texttt{<bbox><}x_{\text{tl}}^{(n)}\texttt{><}y_{\text{tl}}^{(n)}\texttt{><}x_{\text{br}}^{(n)}\texttt{><}y_{\text{br}}^{(n)}\texttt{></bbox>}, which includes coordinates for its top-left and bottom-right corners.

Markdown texts: For the markup-based document representation where the output text is in the markdown format, the text component captures both content and formatting markup. Unlike layout-based documents, markdown text does not require bounding boxes. Instead, the text is directly tokenized, retaining all special characters and formatting indicators.

To facilitate these diverse input types, we employ different composite representations. For image-text pairs with text lines and bounding boxes, the input is denoted as <s><image>Image Embedding</image> ⋃n=1Nsuperscriptsubscript𝑛1𝑁\bigcup_{n=1}^{N} (𝐁n⊕𝐓n)\mathbf{B}_{n}\oplus\mathbf{T}_{n}) </s>. The operator ⊕direct-sum\oplus represents the concatenation of the text line 𝐓nsubscript𝐓𝑛\mathbf{T}_{n} and its bounding box 𝐁nsubscript𝐁𝑛\mathbf{B}_{n}. Conversely, when the text is in the markdown format, the input simplifies to <s><image>Image Embedding</image>Markdown Text</s>. In both cases, <s> and </s> signify the sequence boundaries, while <image> and </image> indicate the beginning and end of image embeddings. This flexibility in text representation allows Kosmos-2.5 to apply to various document analysis tasks.

### 2.3 Pre-training Data

The pre-training process enables Kosmos-2.5 to learn versatile representations suitable for various text-intensive image understanding tasks. The model is pre-trained on a rich array of datasets from diverse sources. Traditional Optical Character Recognition (OCR) task is primarily geared towards generating text content and its 2D positions within an image. However, they often neglect the need to maintain the order and structural integrity of the original document, which is essential for text-intensive image understanding tasks involving structured information.

To address this, we steer Kosmos-2.5 to excel in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. Markdown provides an advantage over plain text by explicitly distinguishing different structural elements, such as tables and lists, with specific tokens. For example, table cells can be denoted with vertical bars (|) and list items with bullets (*, -, or +). It also standardizes the representation of typographic emphases like bold (**bold**) and italics (*italics*), integrating the learning of document structure with natural language understanding in a unified model.

For spatially-aware text blocks, we use:

- •

IIT-CDIP: The IIT-CDIP dataset is a large-scale public collection comprising scanned document images. We used approximately 27.6 million pages to train our model.

- •

arXiv papers: arXiv, an open-access research-sharing platform, provides another significant data source, accounting for roughly 20.9 million pages. We downloaded a bulk of data, consisting of PDF and LaTeX source files, from the official arXiv repository111https://info.arxiv.org/help/bulk_data/index.html.

- •

PowerPoint slides: A corpus of 6.2 million pages is collected from various web pages containing PowerPoint documents, significantly enhancing the diversity of our training data.

- •

General PDF: Additionally, we crawled the web for diverse open-domain digital PDF files, leading to the collection of a large corpus comprising approximately 155.2 million pages.

- •

Web screenshots: A subset of the mC4 webpages is scraped and rendered as screenshots containing almost 100 million pages.

For structured text output in markdown format, we use:

- •

README: We collect 2.9 million “README.md” files from open-source GitHub projects, primarily written in markdown format.

- •

DOCX: We also extract 1.1 million DOCX pages from millions of WORD files crawled from the web. The DOCX pages are converted to markdown format, and each page corresponds to its markdown information.

- •

LaTeX: A subset of the entire arXiv papers is used to extract the mapping of PDF pages and its corresponding markdown information converted from the LaTeX code, which contains a total of 3.7 million pages.

- •

HTML: We obtain 6.3 million HTML files from the aforementioned mC4 subset and convert them into markdown format.

### 2.4 Data Processing

The pre-training data has a wide coverage, and each type of data requires a different processing workflow, which is introduced as follows:

#### IIT-CDIP

The IIT-CDIP dataset mainly consists of scanned document images. We use the Microsoft Read API 222https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview-ocr#read-api to extract text and layout information.

#### arXiv papers, PowerPoint slides, General PDF

We first compile and convert arXiv papers and PowerPoint slides into PDF files. Together with other general PDFs, we employed the PyMuPDF parser 333https://github.com/pymupdf/PyMuPDF to extract text and layout information efficiently.

#### Web screenshots

We also include webpage screenshots in the model pre-training to diversify the layout distribution further. We collect the webpage URLs from the English portion of the mC4 dataset. Playwright 444https://github.com/microsoft/playwright-python is used to access a specified URL and open the webpage. The HTML content of the page is extracted and parsed using the lxml library 555https://lxml.de/ to obtain a Document Object Model (DOM) tree representation. This DOM tree is traversed, examining the XPath of each element within it. This traversal aims to determine whether each element is visible and retrieve information about its bounding boxes.

#### README (markdown)

In addition to layout-based data, we collect markup-based data for the pre-training. We collect “README.md” files from many GitHub projects and convert these files into HTML using Pandoc 666https://pandoc.org/. Then, wkhtmltopdf 777https://wkhtmltopdf.org/ is used to obtain the images from the generated HTML content.

#### DOCX (markdown)

The Microsoft Office WORD files have been extensively used in existing research like TableBank [28] and ReadingBank [57]. We collect WORD DOCX files and convert them into texts with markdown. First, we use Pandoc to convert the XML content within the DOCX files into markdown files. As Pandoc keeps the “<table>” tags to represent the tabular cells in the generated markdown, we further identify all the tables and use markdownify 888https://github.com/matthewwithanm/python-markdownify to convert them into the markdown formats. Finally, the original DOCX files are converted into PDF files, and each page is aligned to the corresponding span of the markdown content based on a heuristic method.

#### LaTeX (markdown)

LaTeX documents from arXiv have been used to generate PDF files to obtain texts with bounding boxes. Meanwhile, we also convert the LaTeX content into the markdown texts. Similar to Nougat [3], LaTeXML 999https://math.nist.gov/~BMiller/LaTeXML/ is used to convert the LaTeX code into the HTML sequence, which is further transformed into the markdown format. Different from Nougat, we keep all the tables at the beginning of the page as most LaTeX users prefer to position tables with “[t]” or “[h]” instead of “[b]”. Meanwhile, we also convert the table content from the LaTeX format into the markdown format.

#### HTML (markdown)

The most straightforward way to obtain markdown resources from HTML webpages is through web scraping. However, webpages are often cluttered with various layouts and styles, resulting from the misuse of HTML tags. Moreover, HTML pages may include extraneous elements, such as advertisements, navigation menus, or formatting elements, making extracting clean and meaningful content challenging. To overcome these obstacles, we employ Playwright, a fast and reliable end-to-end testing framework for the web. The library allows us to navigate the HTML structure, filter out non-essential elements, and extract the relevant text content. We also apply custom rules and regular expressions to further refine the extracted text and format it as markdown, ensuring that the resulting markdown files are coherent and readable.

Task
Data Source
Number of Pages
Sampling Ratio

Layout-based (texts+bboxes)
IIT-CDIP
27.6M
10%

arXiv papers
20.9M
5%

PowerPoint slides
6.2M
5%

General PDF
155.2M
20%

Web screenshots
100.5M
10%

Markup-based (texts+markdown)
README
2.9M
15%

DOCX
1.1M
10%

LaTeX
3.7M
15%

HTML
6.3M
10%

Total
324.4M
100%

### 2.5 Filtering and Quality Control

We employ fastText for language identification (with a threshold of 0.5) to filter out non-English documents from the entire pre-training dataset. To ensure content diversity within each source, we utilize the MinHash [5] to identify and remove redundant pages. We use the same parameters as [32] and a document pair with similarity 0.8 will be marked as duplicate. A comprehensive breakdown of the pre-training data, along with their respective sampling ratios, is provided in Table 1. When dealing with image-to-markdown data from README, DOCX, LaTeX, and HTML sources, we observe discrepancies between the content in text images and their corresponding markdown sequences due to conversion issues. Consequently, we refine the data by evaluating token overlap between images and markdown files, requiring a token intersection-to-union ratio greater than 0.95 for inclusion. Section A.2 shows some of the training samples.

## 3 Experiments

### 3.1 Evaluation

#### Text Recognition

We utilize word-level precision (# or correct matches over the number of detected words), recall (# of correct matches over the number of ground truth words), and f1 as the metrics to evaluate the text recognition performance. If there are repeated words in the ground truth, they are expected to be repeated in the prediction. Text recognition is evaluated on three benchmark datasets, including FUNSD [23], SROIE [16] and CORD [46]. We compare Kosmos-2.5 to the text recognition results from Document OCR in Google Document AI 101010https://cloud.google.com/document-ai.

#### Image-to-markdown Generation

In light of the unique nature of the image-to-markdown conversion task, assessing the quality of the generated markdown necessitates specialized metrics. We adopt a two-fold evaluation scheme: Normalized Edit Distance (NED) and Normalized Tree Edit Distance (NTED), considering both the lexical accuracy and the preservation of the original structural elements.

The NED is formulated as

NED=1−1N​∑i=1ND​(si,s^i)/max⁡(len​(si),len​(s^i))NED11𝑁superscriptsubscript𝑖1𝑁𝐷subscript𝑠𝑖subscript^𝑠𝑖lensubscript𝑠𝑖lensubscript^𝑠𝑖\textit{NED}=1-\frac{1}{N}\sum_{i=1}^{N}D\left(s_{i},\hat{s}_{i}\right)/\max\left(\mathrm{len}(s_{i}),\mathrm{len}(\hat{s}_{i}\right))

where N𝑁N, s𝑠s, and s^^𝑠\hat{s} denote the number of samples, prediction, and ground truth, respectively. D​(⋅,⋅)𝐷⋅⋅D(\cdot,\cdot) and len​(⋅)len⋅\mathrm{len}(\cdot) represent the edit distance function and the length of a string. The NED value ranges from 0 to 1, with a higher NED value indicating the prediction is closer to the ground truth.

However, given the hierarchical structure inherent to markdown, relying solely on a string-based comparison metric like NED can be insufficient. Thus, we adopt NTED as an additional evaluation metric for structural differences. NTED is a tree edit distance normalized by the number of nodes in the tree, considering the structural discrepancies between parse trees. Specifically, the predicted markdown sequence is first transformed into an HTML tree. Then, the tree edit distance between the prediction and the ground truth is calculated using the ZSS algorithm [69]. The NTED is formulated as

NTED=1−1N​∑i=1NTD​(ti,t^i)/max⁡(node​(ti),node​(t^i))NTED11𝑁superscriptsubscript𝑖1𝑁TDsubscript𝑡𝑖subscript^𝑡𝑖nodesubscript𝑡𝑖nodesubscript^𝑡𝑖\textit{NTED}=1-\frac{1}{N}\sum_{i=1}^{N}\mathrm{TD}\left(t_{i},\hat{t}_{i}\right)/\max\left(\mathrm{node}(t_{i}),\mathrm{node}(\hat{t}_{i}\right))

where N𝑁N, t𝑡t, and t^^𝑡\hat{t} signify the number of samples, the HTML tree of prediction, and the HTML tree of ground truth, respectively. Besides, TD​(⋅,⋅)TD⋅⋅\mathrm{TD}(\cdot,\cdot) and node​(⋅)node⋅\mathrm{node}(\cdot) stand for the tree edit distance function and the number of nodes in a tree.

We create three datasets to evaluate the image-to-markdown task from different data sources, including document-level markdown generation, README markdown generation and table markdown generation. Each dataset includes 1,000 ⟨⟨\langleimage, markdown⟩⟩\rangle pairs, which are held out from the pre-training data. We compare Kosmos-2.5 to the markdown generated by the Nougat [3] base and small models.

### 3.2 Implementation Details

We employ the AdamW optimizer [30] with β=(0.9,0.98)𝛽0.90.98\beta=(0.9,0.98) for optimization, setting the weight decay to 0.01 and the dropout rate to 0.1. The learning rate is warmed up to 2×10−42superscript1042\times 10^{-4} during the initial 375 steps, followed by a linear decay to zero throughout the remaining training steps. The batch size is adjustable to align with the available computational resources and specific training requirements. Kosmos-2.5 contains a total of 1.3 billion parameters. The vision encoder is initialized from the encoder of the Pix2Struct-Large model. The language decoder includes 24 Transformer layers with a hidden size of 1,536, an FFN intermediate size of 6,144, and 16 attention heads. Section A.1 shows more details of the training hyperparameters.

Due to the substantially larger quantity of available layout-based data than markup-based data, we initially trained the model for 100k steps exclusively using the layout-based dataset. Subsequently, the two datasets were combined for further training of 140k steps. Additionally, we incorporate the training split of the evaluation dataset into the entire pre-training data, extending the process by an additional 10k steps. For text tokenization, we utilize SentencePiece [25] and adopt the “full-sentence” format [38]. This approach packs each input sequence with full sentences, continuously sampled from one or multiple documents. Newly added word embeddings of location tokens are randomly initialized, with all parameters updated during training. We also leverage the data augmentation approaches from TrOCR [34] in the training to make models more robust.

Throughout the evaluation process, model inference is conducted using a single model checkpoint across various evaluation datasets with the corresponding task prompt respectively, demonstrating that our approach does not necessitate individualized model fine-tuning for each dataset.

### 3.3 Results

Kosmos-2.5 is a flexible framework that facilitates multitasking, with tasks determined by the provided task prompts. Experimental results are demonstrated in Table 2 and Table 3. Specifically, for the text recognition task, our Kosmos-2.5 outperforms Google Document OCR by 0.33%, 2.45%, and 1.35% in terms of the F1 score, showcasing its effectiveness. For the image-to-markdown task, it is worth noting that our method significantly outperforms the Nougat [3]. For example, Kosmos-2.5 achieves a notable improvement of 33.68% (95.09% vs 61.41%) over Nougat BASEsubscriptNougat BASE\text{Nougat}_{\text{\,BASE}} in terms of NED on the README dataset. Besides, regarding NTED, Kosmos-2.5 also boosts the performance by 33.38% (82.08% vs 48.70%) compared with Nougat BASEsubscriptNougat BASE\text{Nougat}_{\text{\,BASE}} on the Documents dataset. We attribute the performance boost to the increased diversity of our training data compared to Nougat, which primarily focuses on the academic paper domain. Notably, the greater diversity in our training data significantly enhances our model’s comprehension of different document types and strengthens its generalization capabilities. In summary, the experimental results validate the remarkable capabilities of Kosmos-2.5 in various tasks.

Dataset
FUNSD
SROIE
CORD

P / R / F1
P / R / F1
P / R / F1

Commercial OCR

85.12 / 80.86 / 82.93
89.68 / 89.69 / 89.69
81.95 / 86.87 / 84.34

Kosmos-2.5†
83.88 / 82.66 / 83.26

91.72 / 92.57 / 92.14
83.64 / 87.83 / 85.69

Dataset
General Documents
README
Tables

NED / NTED
NED / NTED
NED / NTED

Nougat SMALLsubscriptNougat SMALL\text{Nougat}_{\text{\,SMALL}} [3]†

82.80 / 48.96
58.58 / 35.49
68.33 / 61.52

Nougat BASEsubscriptNougat BASE\text{Nougat}_{\text{\,BASE}} [3]†

83.75 / 48.70
61.41 / 36.41
68.53 / 61.60

Kosmos-2.5‡

91.59 / 82.08

95.09 / 91.18

85.14 / 90.64

### 3.4 Discussion

We illustrate an example in Figure 3, showcasing the model outputs produced by Kosmos-2.5 with various task prompts when presented with the same input text image. As shown in the figure, the model generates distinct outputs depending on the task prompts it receives. When given the layout task prompt, the model produces the following text sequence, which includes textual content and corresponding bounding boxes:

⬇

[x_52] [y_113] [x_756] [y_145]: NYC Department of Education School Year Calendar 2023-2024

[x_52] [y_159] [x_826] [y_181]: This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private,

[x_52] [y_180] [x_820] [y_202]: parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact

[x_52] [y_201] [x_639] [y_223]: your child’s school for information about their calendar. Please note the following:

[x_65] [y_223] [x_77] [y_245]: ∙∙\bullet

[x_92] [y_223] [x_825] [y_245]: On days when school buildings are closed due to inclement weather or other emergencies, all students

...

With the markup task prompt, the model generates another text sequence that follows the markdown format:

⬇

# NYC Department of Education School Year Calendar 2023-2024

This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private, parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact your child’s school for information about their calendar. Please note the following:

...

- On this schedule, **elementary schools** are defined as programs that serve kindergarten (K) through grade 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. **Middle schools** are defined as programs that serve grades 6-8, and **high schools** are defined as programs that serve grades 9-12.

...

It is apparent that Kosmos-2.5 excels in precisely identifying text positions and recognizing text content. Moreover, it adeptly captures the styles and structures present within the text image, including elements like titles, bullet points, tables, and bold text. Section A.3 provides the full output sequence using different task prompts for this example.

Kosmos-2.5 provides a unified architecture and interface for text image understanding, making it versatile for various application scenarios. Firstly, it can be fine-tuned as a single model for a wide range of text image understanding tasks, including information extraction, layout detection and analysis, visual question answering, screenshot understanding, UI automation, and many others. This unified model interface significantly streamlines downstream task training and enables the model to effectively follow instructions in real-world applications. Secondly, our solution is compatible with more powerful LLMs like GPT-3.5 or GPT-4. The output from our model can serve as contexts for LLMs, enhancing their capabilities through further prompt engineering. This approach empowers LLMs with robust text image understanding capabilities. Thirdly, we have the potential to augment the pre-training with textual data, transforming it into a general-purpose MLLM. This expanded model not only processes visual signals but also possesses strong language understanding capabilities.

## 4 Related Work

### 4.1 Multimodal Large Language Models

The flourishing blossom of large language models (LLM), represented by ChatGPT [6], has revolutionized artificial intelligence and significantly impacted numerous downstream tasks such as text translation, code generation, question answering, etc. Despite the rapid development, it is significant to recognize that the human perception of the world is not limited to language alone but encompasses a wide range of modalities, with particular emphasis on the visual modality. Many research works attempt to “bring eyes” to LLM and develop multimodal large language models (MLLM), which can be categorized into LLM-centric scheduling systems and end-to-end trainable multimodal systems.

The LLM-centric scheduling system [58, 63, 39, 51, 31, 50, 9] takes advantage of many vision foundation models (e.g., Stable Diffusion [48], ControlNet [66], BLIP [37], etc.), and schedules these models in a language-centric manner. For example, Visual ChatGPT [58] develops a set of prompts to incorporate visual information into ChatGPT, enabling users to draw or edit images through chatting. MM-REACT [63] leverages vision experts to augment its multimodal capabilities by incorporating a textual prompt design that can effectively represent various visual signals, including text descriptions, coordinates, and aligned file names, for images and videos.
HuggingGPT [51] connects LLMs with extensive AI models in machine learning communities, tackling user requests through ChatGPT’s task planning, model selection, and response summarization capabilities. Further, TaskMatrix.AI [39] largely extends the scale and connects foundation models with millions of APIs for solving tasks in both digital and physical domains.
Differently, InternGPT [31] incorporates pointing instructions (e.g., clicking and dragging) for better communication between chatbots and users, while also improving the accuracy of chatbots in performing vision-centric tasks. Nevertheless, this approach has several limitations, such as the expenses associated with API calls or the storage space required for the pre-trained weights of foundation models.

End-to-end trainable multimodal system [21, 1, 17, 47, 22, 59, 67, 20, 35, 12, 36, 43, 53, 49, 68, 13, 26, 42] integrates vision and language models into a unified model, which are further trained on multimodal datasets. For instance, Flamingo [1] leverages gated cross-attention to fuse pre-trained vision and language models, showing impressive ability in downstream multimodal tasks. Besides, BLIP-2 [35] utilized Q-Former to align the visual features with a large language model. Furthermore, Instruct-BLIP improves the training of Q-Former by introducing a novel instruction-aware visual feature extraction method. Based on this design, MiniGPT-4 [67] uses Vicuna [8] as the text encoder and fine-tunes detailed image descriptions to better match user intent. Sparkles unlocks multimodal instruction-following models’ capabilities in open-ended dialogues involving multiple images [20]. LLaVA [36] injects visual features into the language model by treating image tokens as a foreign language, and uses conversation generated by GPT-4 [15] for fine-tuning. Kosmos-1 [17] is trained from scratch using web-scale corpora while showing impressive performance in zero-shot, few-shot, and multimodal chain-of-thought prompting settings. Analogously, Kosmos-2 [47] incorporates grounding and referring abilities and can accept image regions users select using bounding boxes as input. mPLUG-Owl [65] efficiently fine-tunes the language model using low-rank adaption with multimodal instruction datasets. Otter [42] is built using Flamingo and aims to explore multimodal in-context learning capabilities.

### 4.2 Text Image Understanding

Text image understanding is a cutting-edge technology that harnesses the power of artificial intelligence, including natural language processing and computer vision, to automatically comprehend, categorize, and extract information from documents [10]. Any file containing written or printed characters can be considered a document, including web pages, slides, posters, and even scene text images. Documents are ubiquitous in our daily lives, so the research on documents is significant.

Before the deep learning era, researchers used rule-based heuristic approaches for document analysis [54, 44]. They manually observed layout information and summarized heuristic rules, but these methods are not scalable and require enormous labour costs. Subsequently, the rise of deep learning has led to significant advancements in the field of Document AI [60, 62, 61, 19, 7, 40, 41, 29, 2, 55, 14, 27, 64]. For example, LayoutLM series [60, 62, 19] employs large-scale document data for pre-training and incorporates text, layout, and image information into the model, showing impressive performance in downstream tasks like key information extraction and document question answering. Similarly, DocFormer [2] introduces an additional task to reconstruct the document image during pre-training. Donut [24] introduces an OCR-free document understanding Transformer, directly mapping an input document image to the desired output with OCR. MarkupLM [40] takes advantage of large-scale webpages from Common Crawl and uses node-level hierarchical structure information as the pre-training objective. XDoc [7] introduces a unified framework for tackling multiple document formats in one model for parameter efficiency. UDOP [52] designs a unified model that integrates text, image, and layout modalities, showing impressive performance on diverse document understanding tasks. Pix2Struct [33] is a pre-trained image-to-text model trained to parse masked screenshots of web pages into simplified HTML.

Despite significant progress in text image understanding, most models are designed for specific tasks and lack generalizability. On the contrary, the proposed Kosmos-2.5 represents an important step forward in this field, demonstrating the potential of MLLM in achieving robust and generalizable performance across a wide range of text image types.

## 5 Conclusion and Future Work

We introduced Kosmos-2.5, a multimodal literate model built on the strengths of Kosmos-2, designed to enhance machine understanding of text-intensive images. This model shifted from conventional encoder-only/encoder-decoder models to a more unified, decoder-only architecture. The shift to generative multimodal language modeling simplifies task interfaces, eliminating the need for complex, task-specific pipelines. Moreover, Kosmos-2.5 demonstrated potential in few-shot and zero-shot learning capabilities, laying a foundation for future advances and scalability in multimodal literate models.

Despite these promising results, our current model faces some limitations, offering valuable future research directions. For instance, Kosmos-2.5 currently does not support fine-grained control of document elements’ positions using natural language instructions, despite being pre-trained on inputs and outputs involving the spatial coordinates of text. Instruction tuning could offer a promising route to enhance this aspect of the model, leading to broader application capabilities. Furthermore, documents spanning multiple pages pose a challenge as they typically demand holistic processing and comprehension. Meanwhile, it is also feasible that Kosmos-2.5 allows for multiple image pages interleaved with text as input; however, managing long context windows remains a vital issue we aim to address in future work.

In the broader research landscape, a significant direction lies in furthering the development of model scaling capabilities. With an expanding spectrum of tasks and rising complexities, scaling up the model to handle larger volumes of data is crucial for the progression of multimodal literate models. Ultimately, our goal is to develop a model that effectively interprets both visual and textual data, and generalizes smoothly across an expanded array of text-intensive multimodal tasks.

## Acknowledgement

We would like to acknowledge Zhiliang Peng for the helpful discussions.

## References

- ADL+ [22]

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al.

Flamingo: a visual language model for few-shot learning.

Advances in Neural Information Processing Systems, 35:23716–23736, 2022.

- AJK+ [21]

Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R Manmatha.

Docformer: End-to-end transformer for document understanding.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 993–1003, 2021.

- BCSS [23]

Lukas Blecher, Guillem Cucurull, Thomas Scialom, and Robert Stojnic.

Nougat: Neural optical understanding for academic documents, 2023.

- BMR+ [20]

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

Language models are few-shot learners, 2020.

- Bro [97]

Andrei Z Broder.

On the resemblance and containment of documents.

In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No. 97TB100171), pages 21–29. IEEE, 1997.

- Cha [22]

ChatGPT.

https://openai.com/blog/chatgpt, 2022.

- CLC+ [22]

Jingye Chen, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei.

Xdoc: Unified pre-training for cross-format document understanding.

arXiv preprint arXiv:2210.02849, 2022.

- CLL+ [23]

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing.

Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.

- CLS+ [23]

Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, and Ziwei Liu.

Language models are visual reasoning coordinators.

In ICLR 2023 Workshop on Mathematical and Empirical Understanding of Foundation Models, 2023.

- CXLW [21]

Lei Cui, Yiheng Xu, Tengchao Lv, and Furu Wei.

Document ai: Benchmarks, models and applications, 2021.

- DBK+ [21]

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.

An image is worth 16x16 words: Transformers for image recognition at scale.

In ICLR, 2021.

- DLL+ [23]

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi.

Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.

- GHZ+ [23]

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al.

Llama-adapter v2: Parameter-efficient visual instruction model.

arXiv preprint arXiv:2304.15010, 2023.

- GMW+ [22]

Zhangxuan Gu, Changhua Meng, Ke Wang, Jun Lan, Weiqiang Wang, Ming Gu, and Liqing Zhang.

Xylayoutlm: Towards layout-aware multimodal networks for visually-rich document understanding.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4583–4592, 2022.

- GPT [23]

GPT-4.

https://openai.com/gpt-4, 2023.

- HCH+ [19]

Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar.

Icdar2019 competition on scanned receipt ocr and information extraction.

In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.

- HDW+ [23]

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al.

Language is not all you need: Aligning perception with language models.

arXiv preprint arXiv:2302.14045, 2023.

- HG [16]

Dan Hendrycks and Kevin Gimpel.

Gaussian error linear units (gelus).

arXiv preprint arXiv:1606.08415, 2016.

- HLC+ [22]

Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei.

Layoutlmv3: Pre-training for document ai with unified text and image masking.

In Proceedings of the 30th ACM International Conference on Multimedia, 2022.

- HML+ [23]

Yupan Huang, Zaiqiao Meng, Fangyu Liu, Yixuan Su, Collier Nigel, and Yutong Lu.

Sparkles: Unlocking chats across multiple images for multimodal instruction-following models.

arXiv preprint arXiv:2308.16463, 2023.

- HSD+ [22]

Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, and Furu Wei.

Language models are general-purpose interfaces.

ArXiv, abs/2206.06336, 2022.

- HZH+ [21]

Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, and Jianlong Fu.

Seeing out of the box: End-to-end pre-training for vision-language representation learning.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12976–12985, 2021.

- JET [19]

Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran.

Funsd: A dataset for form understanding in noisy scanned documents, 2019.

- KHY+ [21]

Geewook Kim, Teakgyu Hong, Moonbin Yim, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park.

Donut: Document understanding transformer without ocr.

arXiv preprint arXiv:2111.15664, 7:15, 2021.

- KR [18]

Taku Kudo and John Richardson.

Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing.

arXiv preprint arXiv:1808.06226, 2018.

- KSF [23]

Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried.

Grounding language models to images for multimodal generation.

arXiv preprint arXiv:2301.13823, 2023.

- LBY+ [21]

Chenliang Li, Bin Bi, Ming Yan, Wei Wang, Songfang Huang, Fei Huang, and Luo Si.

Structurallm: Structural pre-training for form understanding.

arXiv preprint arXiv:2105.11210, 2021.

- LCH+ [20]

Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li.

Tablebank: A benchmark dataset for table detection and recognition, 2020.

- LGK+ [21]

Peizhao Li, Jiuxiang Gu, Jason Kuen, Vlad I Morariu, Handong Zhao, Rajiv Jain, Varun Manjunatha, and Hongfu Liu.

Selfdoc: Self-supervised document representation learning.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5652–5660, 2021.

- LH [17]

Ilya Loshchilov and Frank Hutter.

Decoupled weight decay regularization.

arXiv preprint arXiv:1711.05101, 2017.

- LHW+ [23]

Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Yang Yang, Qingyun Li, Jiashuo Yu, et al.

Internchat: Solving vision-centric tasks by interacting with chatbots beyond language.

arXiv preprint arXiv:2305.05662, 2023.

- LIN+ [21]

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini.

Deduplicating training data makes language models better.

arXiv preprint arXiv:2107.06499, 2021.

- LJT+ [23]

Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova.

Pix2struct: Screenshot parsing as pretraining for visual language understanding.

In International Conference on Machine Learning, pages 18893–18912. PMLR, 2023.

- LLC+ [22]

Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei.

Trocr: Transformer-based optical character recognition with pre-trained models, 2022.

- LLSH [23]

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.

Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.

arXiv preprint arXiv:2301.12597, 2023.

- LLWL [23]

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.

Visual instruction tuning.

arXiv preprint arXiv:2304.08485, 2023.

- LLXH [22]

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi.

Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation.

In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022.

- LOG+ [19]

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.

Roberta: A robustly optimized bert pretraining approach.

arXiv preprint arXiv:1907.11692, 2019.

- LWS+ [23]

Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, et al.

Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis.

arXiv preprint arXiv:2303.16434, 2023.

- LXCW [21]

Junlong Li, Yiheng Xu, Lei Cui, and Furu Wei.

Markuplm: Pre-training of text and markup language for visually-rich document understanding.

arXiv preprint arXiv:2110.08518, 2021.

- LXL+ [22]

Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha Zhang, and Furu Wei.

Dit: Self-supervised pre-training for document image transformer.

In Proceedings of the 30th ACM International Conference on Multimedia, pages 3530–3539, 2022.

- LZC+ [23]

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu.

Otter: A multi-modal model with in-context instruction tuning.

arXiv preprint arXiv:2305.03726, 2023.

- LZR+ [23]

Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji.

Cheap and quick: Efficient vision-language instruction tuning for large language models.

arXiv preprint arXiv:2305.15023, 2023.

- O’G [93]

Lawrence O’Gorman.

The document spectrum for page layout analysis.

IEEE Transactions on pattern analysis and machine intelligence, 15(11):1162–1173, 1993.

- Ope [23]

OpenAI.

Gpt-4 technical report, 2023.

- PSL+ [19]

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee.

Cord: A consolidated receipt dataset for post-ocr parsing.

Document Intelligence Workshop at Neural Information Processing Systems, 2019.

- PWD+ [23]

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei.

Kosmos-2: Grounding multimodal large language models to the world.

arXiv preprint arXiv:2306.14824, 2023.

- RBL+ [22]

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

High-resolution image synthesis with latent diffusion models.

In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

- SLL+ [23]

Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai.

Pandagpt: One model to instruction-follow them all.

arXiv preprint arXiv:2305.16355, 2023.

- SMV [23]

Dídac Surís, Sachit Menon, and Carl Vondrick.

Vipergpt: Visual inference via python execution for reasoning.

arXiv preprint arXiv:2303.08128, 2023.

- SST+ [23]

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang.

Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface.

arXiv preprint arXiv:2303.17580, 2023.

- TYW+ [23]

Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal.

Unifying vision, text, and layout for universal document processing.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19254–19264, 2023.

- WCC+ [23]

Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al.

Visionllm: Large language model is also an open-ended decoder for vision-centric tasks.

arXiv preprint arXiv:2305.11175, 2023.

- WCW [82]

Kwan Y. Wong, Richard G. Casey, and Friedrich M. Wahl.

Document analysis system.

IBM journal of research and development, 26(6):647–656, 1982.

- WJD [22]

Jiapeng Wang, Lianwen Jin, and Kai Ding.

Lilt: A simple yet effective language-independent layout transformer for structured document understanding.

arXiv preprint arXiv:2202.13669, 2022.

- WMH+ [22]

Hongyu Wang, Shuming Ma, Shaohan Huang, Li Dong, Wenhui Wang, Zhiliang Peng, Yu Wu, Payal Bajaj, Saksham Singhal, Alon Benhaim, et al.

Foundation transformers.

arXiv preprint arXiv:2210.06423, 2022.

- WXC+ [21]

Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, and Furu Wei.

Layoutreader: Pre-training of text and layout for reading order detection.

In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4735–4744, 2021.

- WYQ+ [23]

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan.

Visual chatgpt: Talking, drawing and editing with visual foundation models.

arXiv preprint arXiv:2303.04671, 2023.

- XHL+ [21]

Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, and Jiebo Luo.

Probing inter-modality: Visual parsing with self-attention for vision-and-language pre-training.

In Advances in Neural Information Processing Systems, volume 34, pages 4514–4528, 2021.

- XLC+ [20]

Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou.

Layoutlm: Pre-training of text and layout for document image understanding.

In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1192–1200, 2020.

- XLC+ [21]

Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, and Furu Wei.

Layoutxlm: Multimodal pre-training for multilingual visually-rich document understanding, 2021.

- XXL+ [21]

Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou.

LayoutLMv2: Multi-modal pre-training for visually-rich document understanding.

In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2579–2591, Online, August 2021. Association for Computational Linguistics.

- YLW+ [23]

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang.

Mm-react: Prompting chatgpt for multimodal reasoning and action.

arXiv preprint arXiv:2303.11381, 2023.

- YLZ+ [23]

Yuechen Yu, Yulin Li, Chengquan Zhang, Xiaoqiang Zhang, Zengyuan Guo, Xiameng Qin, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang.

Structextv2: Masked visual-textual prediction for document image pre-training.

arXiv preprint arXiv:2303.00289, 2023.

- YXX+ [23]

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al.

mplug-owl: Modularization empowers large language models with multimodality.

arXiv preprint arXiv:2304.14178, 2023.

- ZA [23]

Lvmin Zhang and Maneesh Agrawala.

Adding conditional control to text-to-image diffusion models.

arXiv preprint arXiv:2302.05543, 2023.

- ZCS+ [23]

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny.

Minigpt-4: Enhancing vision-language understanding with advanced large language models.

arXiv preprint arXiv:2304.10592, 2023.

- ZHZ+ [23]

Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao.

Llama-adapter: Efficient fine-tuning of language models with zero-init attention.

arXiv preprint arXiv:2303.16199, 2023.

- ZS [89]

Kaizhong Zhang and Dennis Shasha.

Simple fast algorithms for the editing distance between trees and related problems.

SIAM journal on computing, 18(6):1245–1262, 1989.

## Appendix A Supplementary Material

### A.1 Hyperparameters

The settings of hyperparameters are demonstrated in Table 4.

Hyperparameters

Number of layers
24

Hidden size
1,536

FFN inner hidden size
6,144

Attention heads
16

Activation function
GeLU [18]

Vocabulary size
108,481

Soft tokens V size
2,048

Max sequence length
4,096

Initialization
Magneto [56]

Hyperparameters

Training steps
200,000

Warmup steps
375

Batch size
1,024

Optimizer
AdamW

Learning rate
2e-4

Learning rate decay
Linear

Adam β𝛽\beta

(0.9, 0.98)

Weight decay
0.01

Dropout
0.1

### A.2 Data Samples

We demonstrate some of the training samples in Kosmos-2.5, which include the input and output from IIT-CDIP, arXiv papers, PowerPoint slides, general PDFs, web screenshots, README, DOCX, LaTeX  and HTML.

### A.3 Examples of Model Inference

⬇

[x_52] [y_113] [x_756] [y_145]: NYC Department of Education School Year Calendar 2023-2024

[x_52] [y_159] [x_826] [y_181]: This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private,

[x_52] [y_180] [x_820] [y_202]: parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact

[x_52] [y_201] [x_639] [y_223]: your child’s school for information about their calendar. Please note the following:

[x_65] [y_223] [x_77] [y_245]: ∙∙\bullet

[x_92] [y_223] [x_825] [y_245]: On days when school buildings are closed due to inclement weather or other emergencies, all students

[x_92] [y_244] [x_525] [y_266]: and families should plan on participating in remote learning.

[x_65] [y_265] [x_77] [y_287]: ∙∙\bullet

[x_92] [y_265] [x_846] [y_287]: Individual schools’ Parent-Teacher Conference dates might be different from the dates below. Your child’s

[x_92] [y_286] [x_491] [y_308]: teacher will work with you to schedule your conference.

[x_65] [y_308] [x_77] [y_330]: ∙∙\bullet

[x_92] [y_307] [x_845] [y_330]: On this schedule, elementary schools are defined as programs that serve kindergarten (K) through grade

[x_92] [y_329] [x_826] [y_351]: 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. Middle schools

[x_92] [y_350] [x_810] [y_372]: are defined as programs that serve grades 6-8, and high schools are defined as programs that serve

[x_92] [y_371] [x_186] [y_393]: grades 9-12.

[x_60] [y_414] [x_106] [y_436]: DATE

[x_318] [y_414] [x_399] [y_436]: WEEKDAY

[x_605] [y_414] [x_659] [y_436]: EVENT

[x_60] [y_437] [x_155] [y_459]: September 7

[x_297] [y_437] [x_366] [y_459]: Thursday

[x_432] [y_437] [x_565] [y_459]: First day of school

[x_60] [y_470] [x_164] [y_492]: September 14

[x_297] [y_470] [x_366] [y_492]: Thursday

[x_432] [y_459] [x_804] [y_481]: Evening Parent-Teacher Conferences for elementary

[x_432] [y_480] [x_622] [y_503]: schools and Pre-K Centers

[x_60] [y_514] [x_164] [y_536]: September 21

[x_297] [y_514] [x_366] [y_536]: Thursday

[x_432] [y_504] [x_832] [y_526]: Evening Parent-Teacher Conferences for middle schools

[x_432] [y_525] [x_553] [y_547]: and D75 schools

[x_60] [y_548] [x_164] [y_570]: September 25

[x_297] [y_548] [x_360] [y_570]: Monday

[x_432] [y_548] [x_630] [y_570]: Yom Kippur, schools closed

[x_60] [y_581] [x_164] [y_603]: September 28

[x_297] [y_581] [x_366] [y_603]: Thursday

[x_432] [y_570] [x_818] [y_593]: Evening Parent-Teacher Conferences for high schools,

[x_432] [y_592] [x_601] [y_614]: K-12, and 6-12 schools

[x_60] [y_625] [x_135] [y_647]: October 9

[x_297] [y_625] [x_360] [y_647]: Monday

[x_432] [y_614] [x_786] [y_636]: Italian Heritage/Indigenous Peoples’ Day, schools

[x_432] [y_636] [x_482] [y_658]: closed

[x_60] [y_679] [x_152] [y_701]: November 2

[x_297] [y_679] [x_366] [y_701]: Thursday

[x_432] [y_658] [x_829] [y_680]: Afternoon and Evening Parent-Teacher Conferences for

[x_432] [y_679] [x_833] [y_701]: elementary schools; students in these schools dismissed

[x_432] [y_700] [x_556] [y_723]: three hours early

[x_60] [y_727] [x_152] [y_749]: November 7

[x_297] [y_727] [x_360] [y_749]: Tuesday

[x_432] [y_727] [x_745] [y_749]: Election Day, students do not attend school

[x_60] [y_775] [x_152] [y_797]: November 9

[x_297] [y_775] [x_366] [y_797]: Thursday

[x_432] [y_754] [x_829] [y_776]: Afternoon and Evening Parent-Teacher Conferences for

[x_432] [y_775] [x_793] [y_797]: middle schools and D75 schools; students in these

[x_432] [y_796] [x_687] [y_818]: schools dismissed three hours early

[x_60] [y_829] [x_161] [y_851]: November 16

[x_297] [y_829] [x_366] [y_851]: Thursday

[x_432] [y_819] [x_818] [y_841]: Evening Parent-Teacher Conferences for high schools,

[x_432] [y_840] [x_601] [y_862]: K-12, and 6-12 schools

[x_60] [y_884] [x_161] [y_906]: November 17

[x_297] [y_884] [x_344] [y_906]: Friday

[x_432] [y_863] [x_773] [y_885]: Afternoon Parent-Teacher Conferences for high

[x_432] [y_884] [x_791] [y_906]: schools, K-12, and 6-12 schools; students in these

[x_432] [y_905] [x_687] [y_927]: schools dismissed three hours early

[x_60] [y_928] [x_186] [y_950]: November 23-24

[x_297] [y_928] [x_416] [y_950]: Thursday-Friday

[x_432] [y_928] [x_692] [y_950]: Thanksgiving Recess, schools closed

[x_60] [y_960] [x_234] [y_983]: December 25-January 1

[x_297] [y_950] [x_368] [y_972]: Monday-

[x_297] [y_971] [x_360] [y_994]: Monday

[x_432] [y_960] [x_646] [y_983]: Winter Recess, schools closed

[x_60] [y_999] [x_140] [y_1021]: January 15

[x_297] [y_999] [x_360] [y_1021]: Monday

[x_432] [y_999] [x_789] [y_1021]: Rev. Dr. Martin Luther King Jr. Day, schools closed

[x_60] [y_1027] [x_170] [y_1049]: January 23- 26

[x_297] [y_1027] [x_410] [y_1049]: Tuesday-Friday

[x_432] [y_1027] [x_603] [y_1049]: Regents Administration

[x_52] [y_1099] [x_311] [y_1118]: NYCDOE School Year Calendar 2023-24

⬇

# NYC Department of Education School Year Calendar 2023-2024

This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private, parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact your child’s school for information about their calendar. Please note the following:

- On days when school buildings are closed due to inclement weather or other emergencies, all students and families should plan on participating in remote learning.

- Individual schools’ Parent-Teacher Conference dates might be different from the dates below. Your child’s teacher will work with you to schedule your conference.

- On this schedule, **elementary schools** are defined as programs that serve kindergarten (K) through grade 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. **Middle schools** are defined as programs that serve grades 6-8, and **high schools** are defined as programs that serve grades 9-12.

| DATE | WEEKDAY | EVENT |

| — | — | — |

| September 7 | Thursday | First day of school |

| September 14 | Thursday | Evening Parent-Teacher Conferences for elementary schools and Pre-K Centers |

| September 21 | Thursday | Evening Parent-Teacher Conferences for middle schools and D75 schools |

| September 25 | Monday | Yom Kippur, schools closed |

| September 28 | Thursday | Evening Parent-Teacher Conferences for high schools, K-12, and 6-12 schools |

| October 9 | Monday | Italian Heritage/Indigenous Peoples’ Day, schools closed |

| November 2 | Thursday | Afternoon and Evening Parent-Teacher Conferences for elementary schools; students in these schools dismissed three hours early |

| November 7 | Tuesday | Election Day, students do not attend school |

| November 9 | Thursday | Afternoon and Evening Parent-Teacher Conferences for middle schools and D75 schools; students in these schools dismissed three hours early |

| November 16 | Thursday | Evening Parent-Teacher Conferences for high schools, K-12, and 6-12 schools |

| November 17 | Friday | Afternoon Parent-Teacher Conferences for high schools, K-12, and 6-12 schools; students in these schools dismissed three hours early |

| November 23-24 | Thursday-Friday | Thanksgiving Recess, schools closed |

| December 25-January 1 | Monday- Monday | Winter Recess, schools closed |

| January 15 | Monday | Rev. Dr. Martin Luther King Jr. Day, schools closed |

| January 23- 26 | Tuesday-Friday | Regents Administration |

Generated on Wed Feb 28 03:51:59 2024 by LaTeXML
