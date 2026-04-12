# Lee et al. - 2022 - Multimodal Lecture Presentations Dataset Understanding Multimodality in Educational Slides

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Lee et al. - 2022 - Multimodal Lecture Presentations Dataset Understanding Multimodality in Educational Slides.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2208.08080
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

\newfloatcommand

capbtabboxtable[][\FBwidth]

# Multimodal Lecture Presentations Dataset: Understanding Multimodality in Educational Slides

Dong Won Lee, Chaitanya Ahuja, Paul Pu Liang, Sanika Natu, Louis-Philippe Morency 
Carnegie Mellon University
https://github.com/dondongwon/MLPDataset

###### Abstract

Lecture slide presentations, a sequence of pages that contain text and figures accompanied by speech, are constructed and presented carefully in order to optimally transfer knowledge to students. Previous studies in multimedia and psychology attribute the effectiveness of lecture presentations to their multimodal nature. As a step toward developing AI to aid in student learning as intelligent teacher assistants, we introduce the Multimodal Lecture Presentations dataset as a large-scale benchmark testing the capabilities of machine learning models in multimodal understanding of educational content. Our dataset contains aligned slides and spoken language, for 180+ hours of video and 9000+ slides, with 10 lecturers from various subjects (e.g., computer science, dentistry, biology). We introduce two research tasks which are designed as stepping stones towards AI agents that can explain (automatically captioning a lecture presentation) and illustrate (synthesizing visual figures to accompany spoken explanations) educational content. We provide manual annotations to help implement these two research tasks and evaluate state-of-the-art models on them. Comparing baselines and human student performances, we find that current models struggle in (1) weak crossmodal alignment between slides and spoken text, (2) learning novel visual mediums, (3) technical language, and (4) long-range sequences. Towards addressing this issue, we also introduce PolyViLT, a multimodal transformer trained with a multi-instance learning loss that is more effective than current approaches. We conclude by shedding light on the challenges and opportunities in multimodal understanding of educational presentations.

## 1 Introduction

Students today commonly learn through multimedia, including online lecture presentation recordings, educational mobile applications, and other digital resources [23]. In particular, slide-assisted instruction through lectures has become predominant in educational settings [30, 33, 34] and is widely considered by teachers and students as the preferred instructional tool [33, 37]. The effectiveness of lecture slides is supported by research in multimedia principles, which show that individuals learn more effectively from spoken (or written) language when accompanied by graphics rather than language in isolation [2, 24, 25, 26, 29]. The prevalence and effectiveness of lecture slides as an educational medium calls for AI systems that are also able to understand and communicate multimodal knowledge, in order to move closer towards intelligent teaching assistants [16]. More specifically, AI systems that can understand lecture slides could yield many exciting applications, such as an intelligent tutoring system that retrieves a slide to answer a student’s question, a recommender system that automatically generates a slide on-the-fly as the speaker is speaking, or an evaluation system that evaluates the quality of lecture presentations.

As a step towards this direction, we design the Multimodal Lecture Presentations Dataset (MLP Dataset) as a large-scale benchmark evaluating AI technologies in multimodal understanding of educational content. MLP Dataset contains over 9000 slides with natural images, diagrams, equations, tables and written text, aligned with the speaker’s spoken language. These lecture slides are sourced from over 180 hours worth of educational videos in various disciplines such as anatomy, biology, psychology, speaking, dentistry, and machine learning. To benchmark the understanding of multimodal information in lecture slides, we introduce two research tasks which are designed to be a first step towards developing AI that can explain and illustrate lecture slides: automatic retrieval of (1) spoken explanations for an educational figure (Figure-to-Text) and (2) illustrations to accompany a spoken explanation (Text-to-Figure)111Text-to-Figure can be thought as a recommender system assisting a lecturer by providing recommendations of figures when building presentation slides according to the lecturer’s planned transcript. On the other hand, Figure-to-Text can be used by an intelligent tutoring system to retrieve relevant explanations to help a student when given a query of a visual diagram.. To enable these tasks, we manually annotated the slide segments to accurately capture alignment between spoken language, slides, and figures (diagrams, natural images, table, equations).

MLP Dataset and its tasks bring new research opportunities through the following technical challenges: (1) addressing weak crossmodal alignment between figures and spoken language (a figure on the slide is often related to only a portion of spoken language), (2) representing novel visual mediums of man-made figures (e.g., diagrams, tables, and equations), (3) understanding technical language, and (4) capturing interactions in long-range sequences. Through human and quantitative studies, we find that current multimodal models struggle with the aforementioned challenges. We work towards addressing weak alignment and novel visual mediums by introducing PolyViLT, a multimodal transformer trained with a multi-instance learning loss. Although PolyViLT presents some improvement, MLP Dataset still offers novel challenges that will spark future research in educational content modeling, multimodal reasoning, and question answering.

## 2 Related Work

The effectiveness of lecture slides as a medium of transferring information can be explained by five multimedia learning principles [15]. Firstly, the multiple representation principle states that individuals learn more effectively from graphics accompanied by spoken or written verbal information than solely spoken language. This principle is supported by dual-route processing mechanisms of working memory aand comprehension processes, where integration of verbal and nonverbal information benefits formation of representations in working memory. [25, 24, 29, 2, 26]. Secondly, the contiguity principle expounds on reducing the spatio-temporal separation between different forms of information, which decreases the amount of effort required to build a coherent mental representation [4, 28]. Third, redundancy, the exposure to complementary but identical information in different modalities, improves learner’s working memory (auditory, visual). Fourth, coherence: restricting the information presented to only essential information allows the learner to integrate key concepts and relationships [17]. Finally, signalling provides learner information regarding the overall hierarchical structure of the presentation [25].

Features
Size
Avail.

Slide Segments
Slide Figures
Slide Text
Spoken Language
# Videos
# Hours
# Slides

VLEngagment [3]

11568

✓

LectureBank [22]

✓(M)

✓(A)

1352

51,939
✓

ALV [14]

✓(A)

✓(A)

1498
✓

LectureVideoDB [11]

✓(M)

✓(M)

24

5000
✓

GoogleI/O [5]

✓(A)
✓(A)
209
174

✓

LaRochelle [38]

✓(A)

✓(A)
✓(A)
47
65
2350

MLP Dataset
✓(M)
✓(M)
✓(A)
✓(A)
334
187
9031
✓

Given the effectiveness of lecture slides as a medium of presenting information, future AI should be able to learn and extract from the carefully curated, rich information in lecture slides. Recent work towards computational modeling of lecture slides include LectureBank [12, 22], ALV [14], VLEngagement [3], LectureVideoDB [11], GoogleI/O [5], and LaRochelle [38]. We summarize and compare these datasets with our proposed MLP Dataset in Table 1. We also highly recommend readers to refer to Appendix A where we present detailed descriptions and differences of previous datasets. To the best of our knowledge, MLP Dataset is the first of its kind to offer slide segmentation, aligned spoken language, slide text, and visual figures, while being publicly available for the research community.

## 3 Multimodal Lecture Presentations Dataset

The Multimodal Lecture Presentations Dataset is designed as a benchmark to develop AI models capable of understanding multimodal information present in lecture slides. Our dataset offers segmented slides, their aligned spoken language and visual elements (figures, diagrams, natural images, tables), and OCR text output for slide text.

### 3.1 Problem Definition

In order to benchmark an AI technology’s understanding of multimodal education content and move closer to automatic generation of captions and figures, we measure its ability to associate a visual figure with a spoken explanation. We define figures as a visual illustration consisting of text, images, drawings (i.e., diagrams, images, equations, or tables). We design 2 proxy tasks, where (1) visual figures are retrieved given spoken language (Text-to-Figure) and (2) spoken explanations are retrieved from visual figures (Text-to-Figure). In contrast to many prior crossmodal retrieval setups which assume one-to-one mappings between modalities [39], lecture presentations are unique in the presence of weak crossmodal alignment between spoken language and figures. There could exist n>1𝑛1n>1 visual figures for a single spoken speech segment s𝑠s and a figure could be aligned partially to the spoken segment (i.e., a part of the spoken segment is used to explain the figure). Thus, a core challenge lies in addressing weak crossmodal alignment. Formally, let D=(S,V)𝐷𝑆𝑉D=(S,V) be our dataset consisting of spoken language S𝑆S and figures V𝑉V. The goal is to learn an embedding space that can quantify the similarity between the figure and spoken language. As a result, given a segment of spoken language s∈S𝑠𝑆s\in S, one could retrieve the set of aligned visual figures {vk,vk+1,…,vk+n}⊆Vsubscript𝑣𝑘subscript𝑣𝑘1…subscript𝑣𝑘𝑛𝑉\{v_{k},v_{k+1},...,v_{k+n}\}\subseteq V.

### 3.2 Dataset Statistics

Our MLP Dataset consists of 9031 slides, 8598 figures, 28000 unique words, 1.6 million total words from 334 educational presentation videos with a total duration of 187 hours. As shown in Table 2 (b) Per slide, there are 185.83 words spoken on average and 135 words in median. Each slide’s duration is an average of 72.6 seconds or a median of 54.9 seconds as shown in Figure Table 2(e). Among the 8598 figures, 3877 (45.1%) are natural images, 4018 (46.7%) are diagrams, 301 (3.5%) are tables, 402 (4.6%) are equations, shown in Table 2(d). Each slide has a median of 1 (or 0.94 on average) figure, shown in Table 2(c), a median of 26 written words (or 28.95 on average), displayed in Table 2(a). Furthermore, when available, we provide the mouse traces which hover over the region the speaker is describing. There are 12.52 seconds of mouse trace data per slide on average and 4.6 in median, as shown in Table 2(f). Our dataset consists of 35 courses on biology, anatomy, psychology, dentistry, speaking, machine learning taught by 10 speakers. The distribution of the number of slides per speaker is shown in Table 2(g).

### 3.3 Data Collection and Preprocessing

The Multimodal Lecture Presentations Dataset is developed from a curated list of lecture presentation videos, which are downloaded from YouTube. Spoken language is then extracted from speech via automatic speech recognition. We manually annotate for the slide segments as well as figure bounding boxes and corresponding labels in order to perform retrieval tasks between slide-level segments spoken text and individual visual figures. In addition, in order to utilize the language information in figures, the texts in the slide are automatically extracted via OCR. A visual outline the data collection and processing steps taken to create MLP Dataset is shown in Figure 3.

Video Acquisition: 413 English educational videos were downloaded from YouTube. From the initial list, we filtered and curated a smaller list of 10 speakers according to the following criteria: (1) the material must be presented in a slide-based style, (2) the slides must be stationary (i.e. external video clips cannot be played), and (3) the speaker makes use of their mouse to refer to specific figures on the slide. After filtering, 334 videos remained.

Slide Segmentation:
The quality of segmentation is crucial to our task of retrieval, therefore, we collected manual human annotations on MTurk. We presented the annotator with a lecture video and asked the annotator to use a slider to navigate to the end of each slide and mark its precise timestamp. A screenshot of this experiment can be found in Appendix C.

In order for ensure the high quality of segmentations, we conduct the annotation process in multiple steps. (1) An internal team manually annotated 10 lecture videos for groundtruth annotations. (2) The experiment was made available to 100 MTurkers. (3) We evaluate their results, marking an annotation as correct if theirs matched ours within a 1 second interval. (4) Annotators who were able to perform above a 90% correctness threshold were assigned the full set.

Figure Annotation and Labeling:
Our dataset is unique from previous datasets as our focus is centered around figure-level retrieval. In order to enable this task, our data must consist of precise bounding boxes and labels for each figure. Therefore, we design an MTurk experiment where annotators are shown slides and asked to create a bounding box around figure instances and label their classes. Our class labels are inspired from PRImA [1], a dataset that consists of layouts from scientific reports. We follow their taxonomy to find labels on figures, which consist of natural images, diagrams, table, and equations. In Appendix D, we provide details on figure class labels and a screenshot of the MTurk experiment.

To obtain precise and accurate figure annotations, we follow a multi-phase process. (1) An internal team manually annotated 10 lecture videos for groundtruth figure annotations and labels. (2) We make the experiment available to 100 MTurkers for 10 different slides. (3) We manually evaluate the annotations, marking an annotation as correct if the annotators had the same number of figures, equivalent types, and high overlap of bounding boxes. (4) Annotators who were able to perform above a 90% correctness threshold were assigned the full set. (5) To ensure the absolute highest quality of figure annotations, our internal team of annotators manually corrected all the annotations for any mislabeled bounding box annotations or incorrect regions.

Text Extraction: ASR & OCR:
We use Google ASR [6] to extract spoken language from audio. We use the Video-Model, which has a reported WER of 16% (Amazon: 22%, Microsoft 24%, IBM Watson 29%, Google Speech-Model 37%). We manually verify 100 random segments in the dataset, and find that the WER is 17.1%. To extract OCR text from the images of slides, we use Tesseract [35]. We manually verify 100 random slides in the dataset, and find that the WER is 37.82%.

Mouse Trace Location Extraction:
We extract the mouse trace location to be used as an additional grounding signal between visual objects and language. For each segmented slide, the background is static and the only object that is moving is the pointer. If there is any movement, we consider that as the pointer location. We manually verify 100 random mouse trace location in the dataset, and find that the percentage of correct keypoints (PCK) with a threshold of 50 pixels, is 77.1%.

## 4 Experimental Setup

The MLP Datasetis designed to examine multimodal model’s understanding of educational material, as measured by its performance on text-to-figure and figure-to-text retrieval. We evaluate multiple state-of-art model’s performance in comparison with human student performance. We are interested in understanding how current state-of-the-art models perform on different figure types (diagrams, images, equations, tables), long range sequences, and technical language. We also introduce PolyViLT, a multi-instance learning multimodal transformer that utilizes both vision and language information in slide figures.

### 4.1 Baselines

We select previous baselines PVSE [36] and PCME [7] that are designed for cross-modal retrieval particularly in scenarios with weak alignment. We also measure CLIP [32] performance, as its zero-shot image-text matching performance is well recognized in the community.

CLIP [32] is an established baseline for image-text matching.
We use a pre-trained CLIP model to embed pairs of figures and text and rank according to their similarity scores for retrieval.

PVSE [36] is designed to model one-to-many alignment for crossmodal retrieval, by encoding visual and text features as K𝐾K possible embeddings and training with a multiple instance loss that rewards weak cross-modal alignment (i.e., the best pair among K2superscript𝐾2K^{2} pairs is rewarded).

PCME [7] handles pairwise semantic similarities and uncertainty in crossmodal retrieval. It models each modality as probabilistic distributions in a common embedding space using Hedged Instance Embeddings (HIB) [27] and utilizes a soft version of the contrastive loss to handle weak alignment.

### 4.2 PolyViLT: A Proposed Model for Weak Image-Text Alignment

On top of these baselines, we further introduce Polysemous-ViLT (or PolyViLT), which is designed to handle vision and language inputs (e.g., diagrams) and weak cross-modal alignment. Previous approaches were designed specifically for the task of crossmodal retrieval on datasets consisting of only natural images and text. However, to perform well on retrieval problems involving figures, models must utilize text information present in the figure, as they could provide valuable signals to the model. Our approach utilizes local feature transformers in PVSE [36], a multi-instance learning loss [9] and a ViLT figure encoder [19] to utilize both vision and language information in figures. We refer the readers to Figure 5 for details and figure of the model architecture.

ViLT Figure Encoder We utilize the ViLT model [19] as a backbone encoder to contextualize the text and vision information present in figure. Given an image of a figure, the accompanying text (from OCR output) on the figure is tokenized with BERT [8], patches of the diagram image is flattened and linearly projected, and fed in as a sequence to a transformer encoder. We initialize the ViLT encoder with pretrained weights trained on masked language modelling and image-text matching before training on our dataset.

Multiple Instance Learning (MIL) To account for the partial alignment between figures and spoken language, we represent the spoken language with K𝐾K embeddings, capturing different words of the speech, inspired by local feature transformers in [36], The local K𝐾K embedding are combined with global information via residual connections. Then, we utilize the MIL objective [9], which assume that there is a partial match between a figure and K𝐾K local embeddings of the spoken language.

### 4.3 Human Student Performance

To measure human student performance, we randomly sampled 10 figures from the unseen test set for each speaker from 3 random seeds. For Figure-to-Text, a student is shown one figure image, all the spoken language aligned to the 10 figures in the sample and is asked to select the most relevant spoken language. For Text-to-Figure, the annotator is shown one spoken language, all the figure and is asked to select the most relevant figure. We report recall@1 metric for this sample for all of our baseline models for fair comparison.

## 5 Results

### 5.1 Model and Human Performance

The performance of all models can be seen in Table 2. PolyViLT outperforms previous state-of-the art approaches in both Figure-to-Text retrieval and Text-to-Figure Retrieval. The second best performing model is PVSE [36], which further justifies our reasoning behind utilizing local feature transformers and the MIL loss. Surprisingly, CLIP’s zero-shot performance often is worse than Random, which indicates that large-scale pre-training on natural image-text pairs may not be sufficient for our task. The detailed results for each speaker can be found in Appendix 6. We also provide human student retrieval performance in Figure 4. We see that all methods fall well below human students’ performance, even PolyViLT, the closest method, is 47.68% worse for Text-to-Figure retrieval and 43.63% worse for Figure-to-Text retrieval, which demonstrates the challenging nature of our dataset. In the following sections, we perform error analysis to uncover the concrete challenges presented in MLP Dataset.

### 5.2 Performance on Novel Visual Mediums

We first investigate the impact of novel visual mediums such as man-made figures (e.g., diagrams, tables, and equations) on model performance.
We report recall@10 scores conditioned on each type in Table 3, and find that PolyViLT outperforms other baselines for most figure types. Interestingly, we can see that for natural images, previous approaches perform worse than PolyViLT. Whereas we believed that PolyViLT’s main advantage is in its use of text information, it outperforms previous approaches even when no text information is used. This indicates that the usage of a ViT encoder [10] is superior over using local and global feature transformers as proposed in PVSE [36] and PCME [7] even for natural images. We also find models struggle particularly on equations. As mentioned in Section 4.2 this could be attributed to the significant domain difference between the pretraining domain (natural images, non-educational language) of ViLT [19] and equations. PVSE [36] is initialized with random weights, therefore is unaffected.

Models
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
1.36 ± 0.22
7.63 ± 0.88
15.81 ± 0.7

2.15 ± 0.61
8.64 ± 1.1
16.38 ± 1.91

CLIP [32]

2.05 ± 0.7
7.4 ± 0.15
17.65 ± 1.02

1.58 ± 0.56
6.89 ± 1.18
13.78 ± 0.55

PVSE [36]

3.17 ± 0.68
12.44 ± 1.28
22.01 ± 0.61

2.81 ± 0.27
11.87 ± 1.24
21.2 ± 0.63

PVSE (BERT) [36, 8]

2.96 ± 0.76
10.96 ± 0.52
18.54 ± 0.99

2.43 ± 0.05
11.21 ± 1.11
18.51 ± 1.1

PCME [7]

2.31 ± 0.41
8.83 ± 0.34
16.43 ± 0.67

2.12 ± 0.36
8.68 ± 0.14
16.9 ± 1.1

PCME (BERT) [7, 8]

1.93 ± 0.26
8.27 ± 0.95
15.76 ± 1.64

1.93 ± 0.26
8.36 ± 1.08
15.85 ± 1.77

PolyViLT + Trace
3.85 ± 0.91
17.77 ± 1.88
28.26 ± 1.78

5.38 ± 0.78
19.66 ± 2.39
32.26 ± 0.59

PolyViLT
4.94 ± 0.55
19.16 ± 0.69
30.35 ± 0.55

6.14 ± 1.25
23.19 ± 0.68
33.22 ± 1.73

Models
Figure-to-Text: Recall@10

Text-to-Figure: Recall@10

Diagram
Image
Table
Equation

Diagram
Image
Table
Equation

CLIP [32]

6.2 ± 0.57
5.77 ± 0.73
6.2 ± 4.36
2.83 ± 1.11

6.5 ± 1.27
6.0 ± 0.22
6.9 ± 2.5
3.5 ± 0.96

PVSE [36]

8.2 ± 0.93
9.6 ± 0.57
7.27 ± 0.29
12.27 ± 3.27

7.6 ± 1.3
10.33 ± 1.76
6.97 ± 4.15
4.47 ± 4.66

PCME [7]

6.0 ± 0.37
6.9 ± 0.22
6.3 ± 3.28
2.93 ± 3.27

5.9 ± 0.49
6.87 ± 0.26
6.3 ± 3.28
2.93 ± 3.27

PolyViLT
18.53 ± 1.65
15.2 ± 0.91
15.83 ± 2.67
5.53 ± 5.37

18.53 ± 1.89
20.13 ± 0.7
19.17 ± 6.34
9.97 ± 3.48

### 5.3 Technical Language and Long Range Sequences

The second challenge we investigate is the presence of technical language beyond commonly spoken and written text. Table 4(b) shows the number of subwords tokenized by HuggingFace’s BERT Tokenizer [8, 40], which represents the number of Out-of-Vocabulary (OOV) tokens, a proxy measure for how much external knowledge is required to understand technical language. With an increasing number of subwords, there is a drop in performance, indicating that our models struggle to quickly acquire technical information or require external knowledge to perform well.

Furthermore, our dataset poses challenges in capturing information in long range language sequences due to its educational nature. In Table 4(a), we report recall@10 scores conditioned on the number of spoken words. PolyViLT’s performance peaks between 100 and 200 words, and decreases with increasingly longer spoken phrases, or very short spoken phrases (under 100). This calls for a need to develop models for extremely long-range and short-range sequences. We refer the readers to Appendix I where we display examples of instances where our current baselines fail when technical knowledge or understanding of long range interactions are required, and Appendix H for the negative impacts of long range sequences and technical language on other baseline models.

PolyViLT

r@10

<100
100 - 200
200 - 400
400 - 600
600+

<10

10 - 20
20-30
30 - 50

Figure-to-Text
0.195
0.276
0.186
0.227
0.175

0.218
0.191
0.128
0.132

Text-to-Figure
0.177
0.280
0.207
0.186
0.14

0.191
0.156
0.136
0.124

### 5.4 Importance of MIL objective

Figure-to-Text

Text-to-Figure

bio-1
dental
ml-1

bio-1
dental
ml-1

No MIL
26.18 ± 3.92
12.15 ± 1.67
12.58 ± 4.62

28.74 ± 1.32
12.32 ± 1.37
22.23 ± 3.83

MIL
31.29 ± 7.51
17.07 ± 1.66
19.32 ± 3.04

32.24 ± 5.25
20.23 ± 0.12
24.84 ± 8.93

We investigate the effects of using a MIL objective to handle ambiguous alignment by comparing PolyViLT with and without the MIL objective in Table 5. “No MIL” is the case where we optimize using the standard triplet ranking objective [13, 20]. Consistently, across all 3 speakers, we see that MIL is useful and leads to performance boosts by handling weak crossmodal alignment.

### 5.5 Using Mouse Trace as a Grounding Signal

Finally, we experiment with utilizing mouse trace as an additional grounding signal to capture crossmodal alignment. With this intuition, we represent mouse traces as a one-hot vector with length equivalent to the spoken language sequence. For the indices corresponding to words when the mouse hovered over the figure, we assign it the value 1, indicating that the spoken word is directly aligned to the given figure and is conceptually similar to hard attention. We re-parameterize this categorical distribution with a Gumbel-Softmax [18], and use a dot-product attention with skip connections to fuse spoken language and mouse traces. The result for this model is shown in Table 2, as ‘PolyViLT + Trace’. For certain speakers, the inclusion of mouse-trace data offers better performance. We refer the readers to Appendix 6 for speaker-specific studies. Future work should aim at better utilizing the valuable information in mouse traces as a grounding signal [21, 31].

## 6 Discussion

Limitations: Although our dataset presents exciting opportunities, it comes with its limitations. There exists an imbalance in slide distribution amongst speakers. In Figure 2 we show that the dental topic encompasses 28.66% of slides whereas psychology encompasses a much smaller portion of 1.53% of slides. In addition, most topics fall under science and math, leaving humanities unrepresented in this dataset. Similarly, quantitative figures may not be adequately represented as tables and equations represented only 8.2% of the dataset. Further studies on using mouse traces as input signals must be done. Speakers may not consistently use mouse traces, leading to some slides with stronger alignment and some slides with weaker alignment. Finally, this dataset does not encompass other miscellaneous information a speaker might present during their lecture such as animations, speech tone, or extraneous information presented through the form of videos, websites, virtual whiteboards, or other redirected sites.

Broader impacts: There may be downstream effects in training models exclusively on this dataset, since content in humanities may not be equally represented. Social biases could also be encoded into the dataset based on the choice of images and content that speakers decide to include in their lectures, such as images with predominantly male representation or primarily English language.
We believe that MLP Dataset is a first step towards tackling multimodality and alignment in educational slides, and we aim to further expand it with diversity in speakers, languages, subjects, and lecture styles.

## 7 Conclusion

In conclusion, we present the Multimodal Lecture Presentations Dataset as benchmark for developing AI technologies that can communicate multimodal knowledge in educational content. Our diversely sourced and richly annotated dataset contributes two challenging research tasks as a step towards educationally relevant goals: (1) automatic retrieval of spoken explanations given figures and (2) automatic retrieval of illustrative figures given spoken explanations.
Through benchmarking existing and newly proposed models, we outline future research directions in tackling weak crossmodal alignment, novel visual mediums, technical language, and long-range sequences to bring us closer towards intelligent and accessible tutoring aids.

## References

- [1]

Apostolos Antonacopoulos, David Bridson, Christos Papadopoulos, and Stefan
Pletschacher.

A realistic dataset for performance evaluation of document layout
analysis.

In 2009 10th International Conference on Document Analysis and
Recognition, pages 296–300. IEEE, 2009.

- [2]

Alan Baddeley.

Working memory: looking back and looking forward.

Nature reviews neuroscience, 4(10):829–839, 2003.

- [3]

Sahan Bulathwela, Maria Perez-Ortiz, Emine Yilmaz, and John Shawe-Taylor.

Vlengagement: A dataset of scientific video lectures for evaluating
population-based engagement.

arXiv preprint arXiv:2011.02273, 2020.

- [4]

Paul Chandler and John Sweller.

Cognitive load theory and the format of instruction.

Cognition and instruction, 8(4):293–332, 1991.

- [5]

Huizhong Chen, Matthew Cooper, Dhiraj Joshi, and Bernd Girod.

Multi-modal language models for lecture video retrieval.

In Proceedings of the 22nd ACM international conference on
Multimedia, pages 1081–1084, 2014.

- [6]

Chung-Cheng Chiu, Tara N Sainath, Yonghui Wu, Rohit Prabhavalkar, Patrick
Nguyen, Zhifeng Chen, Anjuli Kannan, Ron J Weiss, Kanishka Rao, Ekaterina
Gonina, et al.

State-of-the-art speech recognition with sequence-to-sequence models.

In 2018 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 4774–4778. IEEE, 2018.

- [7]

Sanghyuk Chun, Seong Joon Oh, Rafael Sampaio De Rezende, Yannis Kalantidis, and
Diane Larlus.

Probabilistic embeddings for cross-modal retrieval.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8415–8424, 2021.

- [8]

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

Bert: Pre-training of deep bidirectional transformers for language
understanding.

arXiv preprint arXiv:1810.04805, 2018.

- [9]

Thomas G Dietterich, Richard H Lathrop, and Tomás Lozano-Pérez.

Solving the multiple instance problem with axis-parallel rectangles.

Artificial intelligence, 89(1-2):31–71, 1997.

- [10]

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al.

An image is worth 16x16 words: Transformers for image recognition at
scale.

arXiv preprint arXiv:2010.11929, 2020.

- [11]

Kartik Dutta, Minesh Mathew, Praveen Krishnan, and C. V. Jawahar.

Localizing and recognizing text in lecture videos.

In ICFHR, 2018.

- [12]

Alexander R Fabbri, Irene Li, Prawat Trairatvorakul, Yijiao He, Wei Tai Ting,
Robert Tung, Caitlin Westerfield, and Dragomir R Radev.

Tutorialbank: A manually-collected corpus for prerequisite chains,
survey extraction and resource recommendation.

arXiv preprint arXiv:1805.04617, 2018.

- [13]

Andrea Frome, Greg S Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc’Aurelio
Ranzato, and Tomas Mikolov.

Devise: A deep visual-semantic embedding model.

Advances in neural information processing systems, 26, 2013.

- [14]

Damianos Galanopoulos and Vasileios Mezaris.

Temporal lecture video fragmentation using word embeddings.

In International Conference on Multimedia Modeling, pages
254–265. Springer, 2019.

- [15]

Joanna Garner and Michael Alley.

How the design of presentation slides affects audience comprehension:
A case for the assertion-evidence approach.

International Journal of Engineering Education,
29(6):1564–1579, 2013.

- [16]

David Griol and Zoraida Callejas.

An architecture to develop multimodal educative applications with
chatbots.

International Journal of Advanced Robotic Systems, 10(3):175,
2013.

- [17]

Shannon F Harp and Richard E Mayer.

The role of interest in learning from scientific text and
illustrations: On the distinction between emotional interest and cognitive
interest.

Journal of educational psychology, 89(1):92, 1997.

- [18]

Eric Jang, Shixiang Gu, and Ben Poole.

Categorical reparameterization with gumbel-softmax.

arXiv preprint arXiv:1611.01144, 2016.

- [19]

Wonjae Kim, Bokyung Son, and Ildoo Kim.

Vilt: Vision-and-language transformer without convolution or region
supervision.

In International Conference on Machine Learning, pages
5583–5594. PMLR, 2021.

- [20]

Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel.

Unifying visual-semantic embeddings with multimodal neural language
models.

arXiv preprint arXiv:1411.2539, 2014.

- [21]

Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang.

Text-to-image generation grounded by fine-grained user attention.

In Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision, pages 237–246, 2021.

- [22]

Irene Li, Alexander R Fabbri, Robert R Tung, and Dragomir R Radev.

What should i learn first: Introducing lecturebank for nlp education
and prerequisite chain learning.

In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pages 6674–6681, 2019.

- [23]

Richard E Mayer.

Multimedia learning.

In Psychology of learning and motivation, volume 41, pages
85–139. Elsevier, 2002.

- [24]

Richard E Mayer and Richard B Anderson.

Animations need narrations: An experimental test of a dual-coding
hypothesis.

Journal of educational psychology, 83(4):484, 1991.

- [25]

Richard E Mayer and Roxana Moreno.

Aids to computer-based multimedia learning.

Learning and instruction, 12(1):107–119, 2002.

- [26]

Roxana Moreno and Richard E Mayer.

Cognitive principles of multimedia learning: The role of modality and
contiguity.

Journal of educational psychology, 91(2):358, 1999.

- [27]

Seong Joon Oh, Kevin Murphy, Jiyan Pan, Joseph Roth, Florian Schroff, and
Andrew Gallagher.

Modeling uncertainty with hedged instance embedding.

arXiv preprint arXiv:1810.00319, 2018.

- [28]

Fred Paas, Alexander Renkl, and John Sweller.

Cognitive load theory: Instructional implications of the interaction
between information structures and cognitive architecture.

Instructional science, 32(1/2):1–8, 2004.

- [29]

Allan Paivio.

Mental representations: A dual coding approach.

Oxford University Press, 1990.

- [30]

Annie Piolat, Thierry Olive, and Ronald T Kellogg.

Cognitive effort during note taking.

Applied cognitive psychology, 19(3):291–312, 2005.

- [31]

Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and
Vittorio Ferrari.

Connecting vision and language with localized narratives.

In European Conference on Computer Vision, pages 647–664.
Springer, 2020.

- [32]

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.

Learning transferable visual models from natural language
supervision.

In International Conference on Machine Learning, pages
8748–8763. PMLR, 2021.

- [33]

Gabriel B Reedy.

Powerpoint, interactive whiteboards, and the visual culture of
technology in schools.

Technology, Pedagogy and Education, 17(2):143–162, 2008.

- [34]

April Savoy, Robert W Proctor, and Gavriel Salvendy.

Information retention from powerpoint™ and traditional lectures.

Computers & Education, 52(4):858–867, 2009.

- [35]

Ray Smith.

An overview of the tesseract ocr engine.

In Ninth international conference on document analysis and
recognition (ICDAR 2007), volume 2, pages 629–633. IEEE, 2007.

- [36]

Yale Song and Mohammad Soleymani.

Polysemous visual-semantic embedding for cross-modal retrieval.

In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 1979–1988, 2019.

- [37]

Joshua E Susskind.

Powerpoint’s power in the classroom: Enhancing students’
self-efficacy and attitudes.

Computers & education, 45(2):203–215, 2005.

- [38]

Nhu Van Nguyen, Mickal Coustaty, and Jean-Marc Ogier.

Multi-modal and cross-modal for lecture videos retrieval.

In 2014 22nd International Conference on Pattern Recognition,
pages 2667–2672. IEEE, 2014.

- [39]

Kaiye Wang, Qiyue Yin, Wei Wang, Shu Wu, and Liang Wang.

A comprehensive survey on cross-modal retrieval.

arXiv preprint arXiv:1607.06215, 2016.

- [40]

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue,
Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
et al.

Huggingface’s transformers: State-of-the-art natural language
processing.

arXiv preprint arXiv:1910.03771, 2019.

## Appendix A Descriptions of Previous Lecture Datasets

LectureBank Dataset [22] is a manually-collected dataset of lecture slides, consisting of 1352 online lecture pdf files from 60 courses in Computer Science in 5 sub-domains: Machine Learning, NLP, DL, and IR. The dataset is annotated for each lecture’s topic and prerequisite relation topics based on taxonomy from [12]. This data, does not contain aligned transcripts and was used to predict prerequisite relations for a given lecture slide.

ALV [14] is a lecture video dataset of artificially-generated lectures, where transcripts from lectures are randomly split in fragments then assembled by combining (stitching) exactly 20 randomly selected fragments from various videos. The resulting dataset only consists of transcripts. This work was developed for the purpose of evaluating lecture video fragmentation techniques.

VLEngagement [3], is dataset which was designed to study engagement in video lectures, where content-based (stop-word counts) and video-specific features (silence, video duration) are extracted from publicly available scientific video lectures

LectureVideoDB [11] is a dataset consisting of 5000 frames of lecture videos, with annotated text characters developed for the purposed of text detection and recognition in Lecture Videos.

GoogleI/O [5] is a dataset consisting of 209 presentation videos from the Google I/O conferences in the years 2010-2012. In this dataset, the authors offer only textual information from the speech and the slides. The retrieval task is done at the video level, where entire transcripts are matched with all the text in a presentation.

LaRochelle [38] 47 French lecture recordings from author’s lab, Similar to [5], the authors study video-level retrieval. In addition, the authors experiment with cross-modal retrieval where a bag of words approach is used for the text and visual tokens.

## Appendix B MTurk: Annotators

For each task, we approximate the time each takes with internal annotators to ensure a minimum payment of $8 per hour. For the task of slide segmentation, as annotators are simply required to scroll through the video to find transition points, we pay 50 cents for a 15 minute long video (i.e $2 for an hour long video). We pay annotators a total amount of $856.95 for this task. For the task of figure annotations, we pay the annotators 5 cents per slide, where annotators are expected to spend around 20 seconds per slide. As a result, we spent $451.55 for a total of 9031 slides.

## Appendix C MTurk: Slide Segmentation

## Appendix D MTurk: Figure Annotation

## Appendix E Training Details

We use PyTorch as the auto-differentiation library to train all our models. For each speaker, with split the data such that a random 80% is used as training data and the remaining 20% is used for test (the data is split according to each random seed). In our experiments, we use the following hyperparameters. We train for 100 epochs, and our batch size is 8.

We also utilize the 3 losses (MIL with a margin parameter λmsubscript𝜆𝑚\lambda_{m}, Diversity λd​i​vsubscript𝜆𝑑𝑖𝑣\lambda_{div}, Domain Discrepancy λd​o​msubscript𝜆𝑑𝑜𝑚\lambda_{dom}) as motivated in [36], we refer the audience to the original paper for the formulation of these losses. We use the default parameters, λm=0.1subscript𝜆𝑚0.1\lambda_{m}=0.1, λd​i​v=0.01subscript𝜆𝑑𝑖𝑣0.01\lambda_{div}=0.01, λd​o​m=0.01subscript𝜆𝑑𝑜𝑚0.01\lambda_{dom}=0.01. For the number of locally guided features K𝐾K as shown in Figure 5, we use K=5𝐾5K=5. Further finetuning on these hyperparameters is a future direction of study to boost performance.

As mentioned in Section 4.2, we use a pre-trained backbone ViLT encoder from HuggingFace, by the original authors, which has been trained on masked language modelling and image-text matching (’ViLT-b32-mlm-itm’) [40, 19]. We will release the full code base with our default hyperparameters.

The models were trained with CMU Multicomp Lab’s internal cluster. The average model train runtime was around 8 hours on Titan X 1080 GPUs.

## Appendix F Comprehensive Results for Each Speaker

anat-1
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.27 ± 0.38
3.18 ± 2.4
5.77 ± 1.73

0.26 ± 0.36
3.38 ± 1.42
8.07 ± 3.8

CLIP
0.53 ± 0.37
4.53 ± 1.21
9.11 ± 1.11

0.54 ± 0.38
3.19 ± 1.63
7.17 ± 3.15

PVSE
2.1 ± 0.4
7.33 ± 1.14
12.76 ± 2.07

1.83 ± 0.36
8.09 ± 0.92
11.73 ± 0.88

PVSE (BERT)
2.62 ± 0.43
5.49 ± 0.12
10.48 ± 1.71

1.04 ± 0.36
7.01 ± 2.21
10.68 ± 1.82

PCME
1.3 ± 0.71
4.18 ± 0.32
7.83 ± 0.16

1.3 ± 0.71
4.18 ± 0.32
7.83 ± 0.16

PCME (BERT)
1.3 ± 0.71
3.92 ± 0.63
8.09 ± 0.92

1.3 ± 0.71
3.92 ± 0.63
8.09 ± 0.92

Ours
11.23 ± 0.91
30.82 ± 6.1
42.31 ± 4.82

13.79 ± 2.34
34.34 ± 8.91
44.9 ± 6.53

Ours w/ Trace
9.64 ± 3.08
31.05 ± 6.71
46.49 ± 2.67

10.71 ± 0.54
36.86 ± 2.33
49.85 ± 1.14

anat-2
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
1.75 ± 2.48
21.05 ± 8.59
50.87 ± 9.92

8.77 ± 2.48
31.58 ± 8.6
56.14 ± 8.94

CLIP
7.12 ± 2.42
23.2 ± 6.48
57.21 ± 7.01

3.51 ± 4.96
16.08 ± 8.61
37.43 ± 3.61

PVSE
7.02 ± 2.48
38.6 ± 2.48
66.67 ± 2.48

7.02 ± 2.48
38.6 ± 6.56
68.42 ± 0.0

PVSE (BERT)
5.26 ± 4.3
33.33 ± 4.96
61.4 ± 8.94

5.26 ± 0.0
33.34 ± 6.56
52.63 ± 8.59

PCME
5.26 ± 0.0
28.07 ± 4.96
56.14 ± 2.48

5.26 ± 0.0
28.07 ± 4.96
56.14 ± 2.48

PCME (BERT)
5.26 ± 0.0
28.07 ± 2.48
52.63 ± 0.0

5.26 ± 0.0
28.07 ± 2.48
52.63 ± 0.0

Ours
7.02 ± 2.48
54.39 ± 6.56
78.95 ± 4.3

8.77 ± 2.48
57.89 ± 4.3
75.44 ± 6.56

Ours w/ Trace
8.77 ± 4.96
49.12 ± 6.56
73.68 ± 7.44

7.02 ± 6.56
49.12 ± 8.94
77.19 ± 6.56

bio-1
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.79 ± 0.43
3.13 ± 1.19
4.7 ± 0.89

8.77 ± 2.48
31.58 ± 8.6
56.14 ± 8.94

CLIP
0.51 ± 0.03
3.41 ± 1.53
5.7 ± 2.4

3.51 ± 4.96
16.08 ± 8.61
37.43 ± 3.61

PVSE
0.97 ± 0.07
4.05 ± 0.54
6.0 ± 1.06

7.02 ± 2.48
38.6 ± 6.56
68.42 ± 0.0

PVSE (BERT)
0.79 ± 0.43
5.07 ± 1.15
8.55 ± 1.52

5.26 ± 0.0
33.34 ± 6.56
52.63 ± 8.59

PCME
0.66 ± 0.29
2.11 ± 0.4
4.68 ± 0.49

5.26 ± 0.0
28.07 ± 4.96
56.14 ± 2.48

PCME (BERT)
0.48 ± 0.03
2.39 ± 0.23
4.68 ± 0.49

5.26 ± 0.0
28.07 ± 2.48
52.63 ± 0.0

Ours
4.23 ± 0.9
12.53 ± 2.6
19.15 ± 1.29

8.77 ± 2.48
57.89 ± 4.3
75.44 ± 6.56

Ours w/ Trace
2.91 ± 0.82
7.14 ± 1.08
12.65 ± 2.73

7.02 ± 6.56
49.12 ± 8.94
77.19 ± 6.56

bio-3
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.57 ± 0.4
3.68 ± 1.63
6.58 ± 1.56

1.16 ± 1.11
4.07 ± 1.72
8.35 ± 1.33

CLIP
0.0 ± 0.0
5.4 ± 1.52
11.35 ± 2.29

0.85 ± 0.02
3.66 ± 1.04
7.09 ± 1.22

PVSE
1.14 ± 0.81
6.61 ± 0.93
15.8 ± 1.33

1.16 ± 0.43
6.34 ± 1.25
12.35 ± 1.28

PVSE (BERT)
2.87 ± 1.11
7.47 ± 0.94
12.93 ± 1.45

1.43 ± 0.39
5.12 ± 1.02
9.19 ± 0.7

PCME
1.7 ± 0.65
5.15 ± 0.56
8.28 ± 2.13

1.7 ± 0.65
5.15 ± 0.56
8.28 ± 2.13

PCME (BERT)
1.7 ± 0.65
5.44 ± 0.76
9.16 ± 1.55

1.7 ± 0.65
5.44 ± 0.76
9.16 ± 1.55

Ours
1.74 ± 0.75
12.03 ± 0.37
19.14 ± 1.56

4.57 ± 1.45
13.11 ± 3.02
20.04 ± 2.47

Ours w/ Trace
1.17 ± 0.83
8.85 ± 1.9
14.85 ± 3.03

3.42 ± 0.6
10.03 ± 0.35
16.36 ± 2.08

bio-4
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.77 ± 0.44
2.15 ± 0.45
4.62 ± 1.04

0.77 ± 0.21
2.93 ± 0.98
5.84 ± 1.13

CLIP
0.32 ± 0.46
2.48 ± 0.16
4.95 ± 0.83

0.16 ± 0.23
1.88 ± 0.81
5.16 ± 1.5

PVSE
1.7 ± 1.44
4.75 ± 1.41
7.82 ± 1.96

2.17 ± 1.18
4.31 ± 1.25
6.45 ± 1.62

PVSE (BERT)
1.08 ± 0.58
2.61 ± 0.57
5.07 ± 0.09

1.22 ± 0.75
3.52 ± 0.72
5.66 ± 1.61

PCME
1.86 ± 1.98
3.39 ± 1.55
4.92 ± 1.54

1.86 ± 1.98
3.39 ± 1.55
4.92 ± 1.54

PCME (BERT)
0.62 ± 0.22
1.23 ± 0.57
3.99 ± 1.17

0.62 ± 0.22
1.23 ± 0.57
3.99 ± 1.17

Ours
4.28 ± 2.04
12.22 ± 6.66
18.5 ± 9.12

2.9 ± 1.72
11.79 ± 5.52
20.1 ± 6.1

Ours w/ Trace
3.67 ± 1.82
12.07 ± 5.46
19.9 ± 6.82

2.29 ± 0.97
12.68 ± 5.64
21.88 ± 7.02

dental
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.17 ± 0.14
0.87 ± 0.25
1.91 ± 0.28

0.29 ± 0.21
0.92 ± 0.29
1.67 ± 0.28

CLIP
0.06 ± 0.08
1.09 ± 0.35
2.14 ± 0.25

0.23 ± 0.08
0.98 ± 0.23
1.85 ± 0.32

PVSE
0.4 ± 0.21
1.73 ± 0.13
2.48 ± 0.42

0.29 ± 0.08
1.44 ± 0.19
2.65 ± 0.26

PVSE (BERT)
0.34 ± 0.0
1.84 ± 0.57
2.65 ± 0.33

0.64 ± 0.31
1.57 ± 0.65
2.6 ± 0.32

PCME
0.23 ± 0.08
0.86 ± 0.13
1.73 ± 0.41

0.23 ± 0.08
0.86 ± 0.13
1.73 ± 0.41

PCME (BERT)
0.23 ± 0.08
0.86 ± 0.23
1.67 ± 0.34

0.23 ± 0.08
0.86 ± 0.23
1.67 ± 0.34

Ours
0.63 ± 0.29
2.72 ± 0.23
6.18 ± 0.53

1.15 ± 0.16
5.31 ± 0.25
8.36 ± 1.45

Ours w/ Trace
0.69 ± 0.23
3.28 ± 1.04
6.16 ± 1.1

0.8 ± 0.39
3.28 ± 0.69
5.88 ± 0.58

ml-1
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
0.28 ± 0.2
1.88 ± 0.28
3.5 ± 0.48

0.29 ± 0.21
0.92 ± 0.29
1.67 ± 0.28

CLIP
0.43 ± 0.34
1.69 ± 0.36
4.83 ± 1.94

0.23 ± 0.08
0.98 ± 0.23
1.85 ± 0.32

PVSE
1.48 ± 0.47
5.65 ± 0.66
7.51 ± 0.96

0.29 ± 0.08
1.44 ± 0.19
2.65 ± 0.26

PVSE (BERT)
0.54 ± 0.16
3.78 ± 1.36
6.44 ± 1.18

0.64 ± 0.31
1.57 ± 0.65
2.6 ± 0.32

PCME
0.66 ± 0.34
2.43 ± 0.19
4.49 ± 0.61

0.23 ± 0.08
0.86 ± 0.13
1.73 ± 0.41

PCME (BERT)
0.54 ± 0.16
2.68 ± 0.53
4.61 ± 0.48

0.23 ± 0.08
0.86 ± 0.23
1.67 ± 0.34

Ours
0.82 ± 0.05
4.76 ± 1.93
7.89 ± 1.87

1.15 ± 0.16
5.31 ± 0.25
8.36 ± 1.45

Ours w/ Trace
1.22 ± 0.3
3.71 ± 0.89
6.2 ± 1.84

0.8 ± 0.39
3.28 ± 0.69
5.88 ± 0.58

psy-1
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
4.48 ± 4.78
11.1 ± 6.21
21.22 ± 2.19

4.15 ± 1.35
15.36 ± 1.05
26.73 ± 1.25

CLIP
4.05 ± 3.89
13.22 ± 3.03
30.03 ± 5.88

2.68 ± 2.34
13.38 ± 2.66
22.35 ± 2.92

PVSE
3.99 ± 0.86
16.71 ± 4.75
29.65 ± 4.62

5.07 ± 2.48
17.98 ± 1.51
35.8 ± 1.29

PVSE (BERT)
5.71 ± 2.87
18.87 ± 3.56
27.79 ± 3.23

4.16 ± 1.39
20.46 ± 3.25
34.84 ± 2.7

PCME
5.08 ± 2.49
14.75 ± 1.36
26.29 ± 3.24

4.16 ± 1.39
15.68 ± 2.66
30.92 ± 9.63

PCME (BERT)
4.16 ± 1.39
13.54 ± 6.14
26.93 ± 10.49

4.16 ± 1.39
14.46 ± 7.45
26.93 ± 10.49

Ours
2.27 ± 3.21
19.5 ± 0.76
38.23 ± 2.51

9.38 ± 5.52
32.4 ± 1.45
43.52 ± 5.59

Ours w/ Trace
3.39 ± 1.54
19.18 ± 3.81
33.78 ± 4.26

7.51 ± 3.75
20.83 ± 6.17
37.8 ± 5.6

psy-2
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
4.48 ± 4.78
11.1 ± 6.21
21.22 ± 2.19

0.44 ± 0.62
5.96 ± 2.95
12.74 ± 4.36

CLIP
4.05 ± 3.89
13.22 ± 3.03
30.03 ± 5.88

1.62 ± 1.46
5.77 ± 1.85
14.12 ± 0.88

PVSE
3.99 ± 0.86
16.71 ± 4.75
29.65 ± 4.62

3.47 ± 2.08
11.2 ± 2.34
19.22 ± 1.49

PVSE (BERT)
5.71 ± 2.87
18.87 ± 3.56
27.79 ± 3.23

4.47 ± 0.6
11.56 ± 1.24
18.26 ± 2.92

PCME
5.08 ± 2.49
14.75 ± 1.36
26.29 ± 3.24

2.18 ± 2.24
8.91 ± 3.03
17.98 ± 2.88

PCME (BERT)
4.16 ± 1.39
13.54 ± 6.14
26.93 ± 10.49

1.83 ± 0.76
8.63 ± 3.26
15.94 ± 6.21

Ours
2.27 ± 3.21
19.5 ± 0.76
38.23 ± 2.51

1.36 ± 1.08
14.89 ± 7.3
26.92 ± 5.74

Ours w/ Trace
3.39 ± 1.54
19.18 ± 3.81
33.78 ± 4.26

2.72 ± 2.15
16.06 ± 0.29
27.66 ± 2.79

speaking
Figure-to-Text

Text-to-Figure

Recall@1
Recall@5
Recall@10

Recall@1
Recall@5
Recall@10

Random
3.26 ± 2.72
21.46 ± 6.17
44.79 ± 7.36

0.44 ± 0.62
5.96 ± 2.95
12.74 ± 4.36

CLIP
4.34 ± 1.65
14.16 ± 7.03
38.77 ± 3.21

1.62 ± 1.46
5.77 ± 1.85
14.12 ± 0.88

PVSE
8.54 ± 1.64
27.64 ± 5.15
51.18 ± 5.45

3.47 ± 2.08
11.2 ± 2.34
19.22 ± 1.49

PVSE (BERT)
7.43 ± 1.39
24.44 ± 2.67
38.4 ± 3.71

4.47 ± 0.6
11.56 ± 1.24
18.26 ± 2.92

PCME
3.19 ± 0.1
16.04 ± 3.08
32.01 ± 3.53

2.18 ± 2.24
8.91 ± 3.03
17.98 ± 2.88

PCME (BERT)
3.19 ± 0.1
15.97 ± 0.49
30.83 ± 0.59

1.83 ± 0.76
8.63 ± 3.26
15.94 ± 6.21

Ours
13.75 ± 3.68
29.58 ± 7.24
53.06 ± 4.52

1.36 ± 1.08
14.89 ± 7.3
26.92 ± 5.74

Ours w/ Trace
5.28 ± 2.9
32.71 ± 9.07
52.92 ± 9.48

2.72 ± 2.15
16.06 ± 0.29
27.66 ± 2.79

## Appendix G Keyword Identifiability

PolyViLT

r@10

tfidf rank

<5
5 -10
10 - 30
30 - 50

Text-to-Figure
0.236
0.2
0.122
0.132

Figure-to-Text
0.249
0.22
0.066
0.124

Specifically for figures which contain text, which consists of 54.9% of our dataset, there are many cases where the pairing between text and figures can be easily found by identifying the keyword and finding its existence in the figure or the spoken language. Naively finding the existence of identical words in two instances is trivial and could lead to incorrect retrievals. The core challenge lies in correctly identifying the keyword that defines the slide segment.

In order to understand the importance of identifying the keyword and how our model performs for text-inclusive figures, we measure the term frequency–inverse document frequency (or tf-idf) of each word in the spoken language, except stopwords which are filtered out. The words are then ranked according to their tf-idf values. We iterate through each word, find the words that also exist the ocr output of the figure and extract the word with the lowest tf-idf rank. Under this condition, if the tf-idf rank for a word is 5, this can be intuitively seen as the fifth most important keyword that defines the slide. Simply stated, if the tf-idf rank of a word is low, the keyword can be easily detected in the slide and the spoken language. On the other hand, if the tf-idf rank of a word is high, this implies that the keyword is hard to detect. 222Note that this method of retrieval is intractable with more number of words and documents

In Table 7 We measure the recall@10 score conditioned on tf-idf ranks, which indicates how well PolyViLT does under varying levels of difficulty of identifying the keyword. PolyViLT’s struggles for cases with easier keyword identifiability and suffers even more with harder cases. This calls for a need for PolyViLT to effectively address easier cases, via using tf-idf directly as a feature, and relying more on the vision when the keyword is not easily identifiable.

## Appendix H Long Range Sequence and OOV Tokens

CLIP
(a) Length of Spoken Language

(b) Number of Subwords

r@10
<100
100 - 200
200 - 400
400 - 600
600+

<10
10 - 20
20-30
30 - 50

Figure-to-Text
0.0447
0.0465
0.0567
0.0676
0.175

0.065
0.062
0.0543
0.0619

Text-to-Figure
0.0793
0.0662
0.0599
0.0571
0.14

0.0704
0.055
0.0498
0.0473

PVSE
(a) Length of Spoken Language

(b) Number of Subwords

r@10
<100
100 - 200
200 - 400
400 - 600
600+

<10
10 - 20
20-30
30 - 50

Figure-to-Text
0.0779
0.0777
0.0644
0.0901
0.0928

0.0973
0.0842
0.0656
0.0667

Text-to-Figure
0.0901
0.116
0.1063
0.1013
0.0814

0.108
0.0839
0.0602
0.084

PCME
(a) Length of Spoken Language

(b) Number of Subwords

r@10
<100
100 - 200
200 - 400
400 - 600
600+

<10
10 - 20
20-30
30 - 50

Figure-to-Text
0.0342
0.1301
0.082
0.0733
0.053

0.0744
0.0556
0.0617
0.0271

Text-to-Figure
0.0342
0.1301
0.082
0.0752
0.0536

0.076
0.0518
0.0603
0.0309

## Appendix I Qualitative Cases of Failure

Generated on Wed Mar 13 20:27:41 2024 by LaTeXML
