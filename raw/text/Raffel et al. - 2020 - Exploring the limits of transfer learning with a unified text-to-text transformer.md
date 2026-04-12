# Raffel et al. - 2020 - Exploring the limits of transfer learning with a unified text-to-text transformer

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Raffel et al. - 2020 - Exploring the limits of transfer learning with a unified text-to-text transformer.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/1910.10683
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

\nameColin Raffel \emailcraffel@gmail.com
\AND\nameNoam Shazeer∗ \emailnoam@google.com
\AND\nameAdam Roberts∗ \emailadarob@google.com
\AND\nameKatherine Lee∗ \emailkatherinelee@google.com
\AND\nameSharan Narang \emailsharannarang@google.com
\AND\nameMichael Matena \emailmmatena@google.com
\AND\nameYanqi Zhou \emailyanqiz@google.com
\ANDWei Li \emailmweili@google.com
\AND\namePeter J. Liu \emailpeterjliu@google.com
\AND\addrGoogle, Mountain View, CA 94043, USA
Equal contribution. A description of each author’s contribution is available in Section A. Correspondence to craffel@gmail.com.

###### Abstract

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP).
The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice.
In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format.
Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks.
By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more.
To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.111https://github.com/google-research/text-to-text-transfer-transformer

Keywords: 
transfer learning, natural language processing, multi-task learning, attention-based models, deep learning

## 1 Introduction

Training a machine learning model to perform natural language processing (NLP) tasks often requires that the model can process text in a way that is amenable to downstream learning.
This can be loosely viewed as developing general-purpose knowledge that allows the model to “understand” text.
This knowledge can range from low-level (e.g. the spelling or meaning of words) to high-level (e.g. that a tuba is too large to fit in most backpacks).
In modern machine learning practice, providing this knowledge is rarely done explicitly; instead, it is often learned as part of an auxiliary task.
For example, a historically common approach is to use word vectors (Mikolov et al., 2013b, a; Pennington et al., 2014) to map word identities to a continuous representation where, ideally, similar words map to similar vectors.
These vectors are often learned through an objective that, for example, encourages co-occurring words to be positioned nearby in the continuous space (Mikolov et al., 2013b).

Recently, it has become increasingly common to pre-train the entire model on a data-rich task.
Ideally, this pre-training causes the model to develop general-purpose abilities and knowledge that can then be transferred to downstream tasks.
In applications of transfer learning to computer vision (Oquab et al., 2014; Jia et al., 2014; Huh et al., 2016; Yosinski et al., 2014), pre-training is typically done via supervised learning on a large labeled data set like ImageNet (Russakovsky et al., 2015; Deng et al., 2009).
In contrast, modern techniques for transfer learning in NLP often pre-train using unsupervised learning on unlabeled data.
This approach has recently been used to obtain state-of-the-art results in many of the most common NLP benchmarks (Devlin et al., 2018; Yang et al., 2019; Dong et al., 2019; Liu et al., 2019c; Lan et al., 2019).
Beyond its empirical strength, unsupervised pre-training for NLP is particularly attractive because unlabeled text data is available en masse thanks to the Internet—for example, the Common Crawl project222http://commoncrawl.org produces about 20TB of text data extracted from web pages each month.
This is a natural fit for neural networks, which have been shown to exhibit remarkable scalability, i.e. it is often possible to achieve better performance simply by training a larger model on a larger data set (Hestness et al., 2017; Shazeer et al., 2017; Jozefowicz et al., 2016; Mahajan et al., 2018; Radford et al., 2019; Shazeer et al., 2018; Huang et al., 2018b; Keskar et al., 2019a).

This synergy has resulted in a great deal of recent work developing transfer learning methodology for NLP, which has produced a wide landscape of pre-training objectives (Howard and Ruder, 2018; Devlin et al., 2018; Yang et al., 2019; Dong et al., 2019), unlabeled data sets (Yang et al., 2019; Liu et al., 2019c; Zellers et al., 2019), benchmarks (Wang et al., 2019b, 2018; Conneau and Kiela, 2018), fine-tuning methods (Howard and Ruder, 2018; Houlsby et al., 2019; Peters et al., 2019), and more.
The rapid rate of progress and diversity of techniques in this burgeoning field can make it difficult to compare different algorithms, tease apart the effects of new contributions, and understand the space of existing methods for transfer learning.
Motivated by a need for more rigorous understanding, we leverage a unified approach to transfer learning that allows us to systematically study different approaches and push the current limits of the field.

The basic idea underlying our work is to treat every text processing problem as a “text-to-text” problem, i.e. taking text as input and producing new text as output.
This approach is inspired by previous unifying frameworks for NLP tasks, including casting all text problems as question answering (McCann et al., 2018), language modeling (Radford et al., 2019), or span extraction Keskar et al. (2019b) tasks.
Crucially, the text-to-text framework allows us to directly apply the same model, objective, training procedure, and decoding process to every task we consider.
We leverage this flexibility by evaluating performance on a wide variety of English-based NLP problems, including question answering, document summarization, and sentiment classification, to name a few.
With this unified approach, we can compare the effectiveness of different transfer learning objectives, unlabeled data sets, and other factors, while exploring the limits of transfer learning for NLP by scaling up models and data sets beyond what has previously been considered.

We emphasize that our goal is not to propose new methods but instead to provide a comprehensive perspective on where the field stands.
As such, our work primarily comprises a survey, exploration, and empirical comparison of existing techniques.
We also explore the limits of current approaches by scaling up the insights from our systematic study (training models up to 111111 billion parameters) to obtain state-of-the-art results in many of the tasks we consider.
In order to perform experiments at this scale, we introduce the “Colossal Clean Crawled Corpus” (C4), a data set consisting of hundreds of gigabytes of clean English text scraped from the web.
Recognizing that the main utility of transfer learning is the possibility of leveraging pre-trained models in data-scarce settings, we release our code, data sets, and pre-trained models.\@footnotemark

The remainder of the paper is structured as follows:
In the following section, we discuss our base model and its implementation, our procedure for formulating every text processing problem as a text-to-text task, and the suite of tasks we consider.
In Section 3, we present a large set of experiments that explore the field of transfer learning for NLP.
At the end of the section (Section 3.7), we combine insights from our systematic study to obtain state-of-the-art results on a wide variety of benchmarks.
Finally, we provide a summary of our results and wrap up with a look towards the future in Section 4.

## 2 Setup

Before presenting the results from our large-scale empirical study, we review the necessary background topics required to understand our results, including the Transformer model architecture and the downstream tasks we evaluate on.
We also introduce our approach for treating every problem as a text-to-text task and describe our “Colossal Clean Crawled Corpus” (C4), the Common Crawl-based data set we created as a source of unlabeled text data.
We refer to our model and framework as the “Text-to-Text Transfer Transformer” (T5).

### 2.1 Model

Early results on transfer learning for NLP leveraged recurrent neural networks (Peters et al., 2018; Howard and Ruder, 2018), but it has recently become more common to use models based on the “Transformer” architecture (Vaswani et al., 2017).
The Transformer was initially shown to be effective for machine translation, but it has subsequently been used in a wide variety of NLP settings (Radford et al., 2018; Devlin et al., 2018; McCann et al., 2018; Yu et al., 2018).
Due to its increasing ubiquity, all of the models we study are based on the Transformer architecture.
Apart from the details mentioned below and the variants we explore in Section 3.2, we do not deviate significantly from this architecture as originally proposed.
Instead of providing a comprehensive definition of this model, we refer the interested reader to the original paper (Vaswani et al., 2017) or follow-up tutorials333http://nlp.seas.harvard.edu/2018/04/03/attention.html,444http://jalammar.github.io/illustrated-transformer/ for a more detailed introduction.

The primary building block of the Transformer is self-attention (Cheng et al., 2016).
Self-attention is a variant of attention (Graves, 2013; Bahdanau et al., 2015) that processes a sequence by replacing each element by a weighted average of the rest of the sequence.
The original Transformer consisted of an encoder-decoder architecture and was intended for sequence-to-sequence (Sutskever et al., 2014; Kalchbrenner et al., 2014) tasks.
It has recently also become common to use models consisting of a single Transformer layer stack, with varying forms of self-attention used to produce architectures appropriate for language modeling (Radford et al., 2018; Al-Rfou et al., 2019) or classification and span prediction tasks (Devlin et al., 2018; Yang et al., 2019).
We empirically explore these architectural variants in Section 3.2.

Overall, our encoder-decoder Transformer implementation closely follows its originally-proposed form (Vaswani et al., 2017).
First, an input sequence of tokens is mapped to a sequence of embeddings, which is then passed into the encoder.
The encoder consists of a stack of “blocks”, each of which comprises two subcomponents: a self-attention layer followed by a small feed-forward network.
Layer normalization (Ba et al., 2016) is applied to the input of each subcomponent.
We use a simplified version of layer normalization where the activations are only rescaled and no additive bias is applied.
After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
The decoder is similar in structure to the encoder except that it includes a standard attention mechanism after each self-attention layer that attends to the output of the encoder.
The self-attention mechanism in the decoder also uses a form of autoregressive or causal self-attention, which only allows the model to attend to past outputs.
The output of the final decoder block is fed into a dense layer with a softmax output, whose weights are shared with the input embedding matrix.
All attention mechanisms in the Transformer are split up into independent “heads” whose outputs are concatenated before being further processed.

Since self-attention is order-independent (i.e. it is an operation on sets), it is common to provide an explicit position signal to the Transformer.
While the original Transformer used a sinusoidal position signal or learned position embeddings, it has recently become more common to use relative position embeddings (Shaw et al., 2018; Huang et al., 2018a).
Instead of using a fixed embedding for each position, relative position embeddings produce a different learned embedding according to the offset between the “key” and “query” being compared in the self-attention mechanism.
We use a simplified form of position embeddings where each “embedding” is simply a scalar that is added to the corresponding logit used for computing the attention weights.
For efficiency, we also share the position embedding parameters across all layers in our model, though within a given layer each attention head uses a different learned position embedding.
Typically, a fixed number of embeddings are learned, each corresponding to a range of possible key-query offsets.
In this work, we use 323232 embeddings for all of our models with ranges that increase in size logarithmically up to an offset of 128128128 beyond which we assign all relative positions to the same embedding.
Note that a given layer is insensitive to relative position beyond 128128128 tokens, but subsequent layers can build a sensitivity to larger offsets by combining local information from previous layers.
To summarize, our model is roughly equivalent to the original Transformer proposed by Vaswani et al. (2017) with the exception of removing the Layer Norm bias, placing the layer normalization outside the residual path, and using a different position embedding scheme.
Since these architectural changes are orthogonal to the experimental factors we consider in our empirical survey of transfer learning, we leave the ablation of their impact for future work.

As part of our study, we experiment with the scalability of these models, i.e. how their performance changes as they are made to have more parameters or layers.
Training large models can be non-trivial since they might not fit on a single machine and require a great deal of computation.
As a result, we use a combination of model and data parallelism and train models on “slices” of Cloud TPU Pods.555https://cloud.google.com/tpu/
TPU pods are are multi-rack ML supercomputers that contain 1,02410241{,}024 TPU v3 chips connected via a high-speed 2D mesh interconnect with supporting CPU host machines.
We leverage the Mesh TensorFlow library (Shazeer et al., 2018) for ease of implementation of both model parallelism and data parallelism (Krizhevsky, 2014).

### 2.2 The Colossal Clean Crawled Corpus

Much of the previous work on transfer learning for NLP makes use of large unlabeled data sets for unsupervised learning.
In this paper, we are interested in measuring the effect of the quality, characteristics, and size of this unlabeled data.
To generate data sets that satisfy our needs, we leverage Common Crawl as a source of text scraped from the web.
Common Crawl has previously been used as a source of text data for NLP, for example to train an n-gram language model (Buck et al., 2014), as training data for commonsense reasoning (Trinh and Le, 2018), for mining parallel texts for machine translation (Smith et al., 2013), as a pre-training data set (Grave et al., 2018; Zellers et al., 2019; Liu et al., 2019c), and even simply as a giant text corpus for testing optimizers (Anil et al., 2019).

Common Crawl is a publicly-available web archive that provides “web extracted text” by removing markup and other non-text content from the scraped HTML files.
This process produces around 20TB of scraped text data each month.
Unfortunately, the majority of the resulting text is not natural language.
Instead, it largely comprises gibberish or boiler-plate text like menus, error messages, or duplicate text.
Furthermore, a good deal of the scraped text contains content that is unlikely to be helpful for any of the tasks we consider (offensive language, placeholder text, source code, etc.).
To address these issues, we used the following heuristics for cleaning up Common Crawl’s web extracted text:

- •

We only retained lines that ended in a terminal punctuation mark (i.e. a period, exclamation mark, question mark, or end quotation mark).

- •

We discarded any page with fewer than 3 sentences and only retained lines that contained at least 5 words.

- •

We removed any page that contained any word on the “List of Dirty, Naughty, Obscene or Otherwise Bad Words”.666https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

- •

Many of the scraped pages contained warnings stating that Javascript should be enabled so we removed any line with the word Javascript.

- •

Some pages had placeholder “lorem ipsum” text; we removed any page where the phrase “lorem ipsum” appeared.

- •

Some pages inadvertently contained code. Since the curly bracket “{” appears in many programming languages (such as Javascript, widely used on the web) but not in natural text, we removed any pages that contained a curly bracket.

- •

Since some of the scraped pages were sourced from Wikipedia and had citation markers (e.g. [1], [citation needed], etc.), we removed any such markers.

- •

Many pages had boilerplate policy notices, so we removed any lines containing the strings “terms of use”, “privacy policy”, “cookie policy”, “uses cookies”, “use of cookies”, or “use cookies”.

- •

To deduplicate the data set, we discarded all but one of any three-sentence span occurring more than once in the data set.

Additionally, since most of our downstream tasks are focused on English-language text, we used langdetect777https://pypi.org/project/langdetect/ to filter out any pages that were not classified as English with a probability of at least 0.99.
Our heuristics are inspired by past work on using Common Crawl as a source of data for NLP:
For example, Grave et al. (2018) also filter text using an automatic language detector and discard short lines and Smith et al. (2013); Grave et al. (2018) both perform line-level deduplication.
However, we opted to create a new data set because prior data sets use a more limited set of filtering heuristics, are not publicly available, and/or are different in scope (e.g. are limited to News data (Zellers et al., 2019; Liu et al., 2019c), comprise only Creative Commons content (Habernal et al., 2016), or are focused on parallel training data for machine translation (Smith et al., 2013)).

To assemble our base data set, we downloaded the web extracted text from April 2019 and applied the aforementioned filtering.
This produces a collection of text that is not only orders of magnitude larger than most data sets used for pre-training (about 750 GB) but also comprises reasonably clean and natural English text.
We dub this data set the “Colossal Clean Crawled Corpus” (or C4 for short) and release it as part of TensorFlow Datasets.888https://www.tensorflow.org/datasets/catalog/c4
We consider the impact of using various alternative versions of this data set in Section 3.4.

### 2.3 Downstream Tasks

Our goal in this paper is to measure general language learning abilities.
As such, we study downstream performance on a diverse set of benchmarks, including machine translation, question answering, abstractive summarization, and text classification.
Specifically, we measure performance on the GLUE and SuperGLUE text classification meta-benchmarks; CNN/Daily Mail abstractive summarization; SQuAD question answering; and WMT English to German, French, and Romanian translation.
All data was sourced from TensorFlow Datasets.999https://www.tensorflow.org/datasets

GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019b) each comprise a collection of text classification tasks meant to test general language understanding abilities:

- •

Sentence acceptability judgment (CoLA (Warstadt et al., 2018))

- •

Sentiment analysis (SST-2 (Socher et al., 2013))

- •

Paraphrasing/sentence similarity (MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Iyer et al., 2017))

- •

Natural language inference (MNLI (Williams et al., 2017), QNLI (Rajpurkar et al., 2016), RTE (Dagan et al., 2005), CB (De Marneff et al., 2019))

- •

Coreference resolution (WNLI and WSC (Levesque et al., 2012))

- •

Sentence completion (COPA (Roemmele et al., 2011))

- •

Word sense disambiguation (WIC (Pilehvar and Camacho-Collados, 2018))

- •

Question answering (MultiRC (Khashabi et al., 2018), ReCoRD (Zhang et al., 2018), BoolQ (Clark et al., 2019))

We use the data sets as distributed by the GLUE and SuperGLUE benchmarks.
For simplicity, when fine-tuning we treat all of the tasks in the GLUE benchmark (and similarly for SuperGLUE) as a single task by concatenating all of the constituent data sets.
As suggested by Kocijan et al. (2019) we also include the Definite Pronoun Resolution (DPR) data set (Rahman and Ng, 2012) in the combined SuperGLUE task.

The CNN/Daily Mail (Hermann et al., 2015) data set was introduced as a question-answering task but was adapted for text summarization by Nallapati et al. (2016); we use the non-anonymized version from See et al. (2017) as an abstractive summarization task.
SQuAD (Rajpurkar et al., 2016) is a common question-answering benchmark.
In our experiments, the model is fed the question and its context and asked to generate the answer token-by-token.
For WMT English to German, we use the same training data as (Vaswani et al., 2017) (i.e. News Commentary v13, Common Crawl, Europarl v7) and newstest2013 as a validation set (Bojar et al., 2014).
For English to French, we use the standard training data from 2015 and newstest2014 as a validation set (Bojar et al., 2015).
For English to Romanian, which is a standard lower-resource machine translation benchmark, we use the train and validation sets from WMT 2016 (Bojar et al., 2016).
Note that we only pre-train on English data, so in order to learn to translate a given model will need to learn to generate text in a new language.

### 2.4 Input and Output Format

In order to train a single model on the diverse set of tasks described above, we cast all of the tasks we consider into a “text-to-text” format—that is, a task where the model is fed some text for context or conditioning and is then asked to produce some output text.
This framework provides a consistent training objective both for pre-training and fine-tuning.
Specifically, the model is trained with a maximum likelihood objective (using “teacher forcing” (Williams and Zipser, 1989)) regardless of the task.
To specify which task the model should perform, we add a task-specific (text) prefix to the original input sequence before feeding it to the model.

As an example, to ask the model to translate the sentence “That is good.” from English to German, the model would be fed the sequence “translate English to German: That is good.” and would be trained to output “Das ist gut.”
For text classification tasks, the model simply predicts a single word corresponding to the target label.
For example, on the MNLI benchmark (Williams et al., 2017) the goal is to predict whether a premise implies (“entailment”), contradicts (“contradiction”), or neither (“neutral”) a hypothesis.
With our preprocessing, the input sequence becomes “mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity.” with the corresponding target word “entailment”.
Note that an issue arises if our model outputs text on a text classification task that does not correspond to any of the possible labels (for example if the model outputs “hamburger” when the only possible labels for a task were “entailment”, “neutral”, or “contradiction”).
In this case, we always count the model’s output as wrong, though we never observed this behavior in any of our trained models.
Note that the choice of text prefix used for a given task is essentially a hyperparameter; we found that changing the exact wording of the prefix had limited impact and so did not perform extensive experiments into different prefix choices.
A diagram of our text-to-text framework with a few input/output examples is shown in Figure 1.
We provide full examples of preprocessed inputs for every task we studied in Section D.

Our text-to-text framework follows previous work that casts multiple NLP tasks into a common format:
McCann et al. (2018) propose the “Natural Language Decathlon”, a benchmark that uses a consistent question-answering format for a suite of ten NLP tasks.
The Natural Language Decathlon also stipulates that all models must be multi-task, i.e. are able to simultaneously tackle all of the tasks at once.
We instead allow for separately fine-tuning the model on each individual task and use short task prefixes instead of an explicit question-answer format.
Radford et al. (2019) evaluate the zero-shot learning capabilities of language models by feeding some input to the model as a prefix and then autoregressively sampling an output.
For example, automatic summarization is done by feeding in a document followed by the text “TL;DR:” (short for “too long, didn’t read”, a common abbreviation) and then the summary is predicted via autoregressive decoding.
We mainly consider models that explicitly process an input with an encoder before generating an output with a separate decoder and we focus on transfer learning rather than zero-shot learning.
Finally, Keskar et al. (2019b) unify many NLP tasks as “span extraction”, where text corresponding to possible output choices are appended to the input and the model is trained to extract the input span corresponding to the correct choice.
In contrast, our framework also allows for generative tasks like machine translation and abstractive summarization where it is not possible to enumerate all possible output choices.

We were able to straightforwardly cast all of the tasks we considered into a text-to-text format with the exception of STS-B, which is a regression task where the goal is to predict a similarity score between 111 and 555.
We found that most of these scores were annotated in increments of 0.20.20.2, so we simply rounded any score to the nearest increment of 0.20.20.2 and converted the result to a literal string representation of the number (e.g. the floating-point value 2.572.572.57 would be mapped to the string “2.6”).
At test time, if the model outputs a string corresponding to a number between 111 and 555, we convert it to a floating-point value; otherwise, we treat the model’s prediction as incorrect.
This effectively recasts the STS-B regression problem as a 21-class classification problem.

Separately, we also convert the Winograd tasks (WNLI from GLUE, WSC from SuperGLUE, and the DPR data set we add to SuperGLUE) into a simpler format that is more amenable to the text-to-text framework.
Examples from the Winograd tasks consist of a text passage containing an ambiguous pronoun that could refer to more than one of the noun phrases in the passage.
For example, the passage might be “The city councilmen refused the demonstrators a permit because they feared violence.”, which contains the ambiguous pronoun “they” that could refer to “city councilmen” or “demonstrators”.
We cast the WNLI, WSC, and DPR tasks as text-to-text problems by highlighting the ambiguous pronoun in the text passage and asking the model to predict the noun that it refers to.
The example mentioned above would be transformed to the input “The city councilmen refused the demonstrators a permit because *they* feared violence.” and the model would be trained to predict the target text “The city councilmen”.

For WSC, examples contain the passage, the ambiguous pronoun, a candidate noun, and a True/False label reflecting whether the candidate matches the pronoun (ignoring any articles).
We only train on examples with a “True” label since we do not know the correct noun targets for examples with a “False” label.
For evaluation, we assign a “True” label if the words in the model’s output are a subset of the words in the candidate noun phrase (or vice versa) and assign a “False” label otherwise.
This removes roughly half of the WSC training set, but the DPR data set adds about 1,00010001{,}000 pronoun resolution examples.
Examples from DPR are annotated with the correct referent noun, making it easy to use this data set in the format listed above.

The WNLI training and validation sets have a significant overlap with the WSC training set.
To avoid leaking validation examples into our training data (a particular issue in the multi-task experiments of Section 3.5.2), we therefore never train on WNLI and never report results on the WNLI validation set.
Omitting results on the WNLI validation set is standard practice (Devlin et al., 2018) due to the fact that it is “adversarial” with respect to the training set, i.e. validation examples are all slightly-perturbed versions of training examples with the opposite label.
As such, we do not include WNLI in the average GLUE score whenever we report on the validation set (all sections except Section 3.7 where results are presented on the test sets).
Converting examples from WNLI to the “referent noun prediction” variant described above is a little more involved; we describe this process in Section B.

## 3 Experiments

Recent advances in transfer learning for NLP have come from a wide variety of developments, such as new pre-training objectives, model architectures, unlabeled data sets, and more.
In this section, we carry out an empirical survey of these techniques in hopes of teasing apart their contribution and significance.
We then combine the insights gained to attain state-of-the-art in many of the tasks we consider.
Since transfer learning for NLP is a rapidly growing area of research, it is not feasible for us to cover every possible technique or idea in our empirical study.
For a broader literature review, we recommend a recent survey by Ruder et al. (2019).

We systematically study these contributions by taking a reasonable baseline (described in Section 3.1) and altering one aspect of the setup at a time.
For example, in Section 3.3 we measure the performance of different unsupervised objectives while keeping the rest of our experimental pipeline fixed.
This “coordinate ascent” approach might miss second-order effects (for example, some particular unsupervised objective may work best on a model larger than our baseline setting), but performing a combinatorial exploration of all of the factors in our study would be prohibitively expensive.
In future work, we expect it could be fruitful to more thoroughly consider combinations of the approaches we study.

Our goal is to compare a variety of different approaches on a diverse set of tasks while keeping as many factors fixed as possible.
In order to satisfy this aim, in some cases we do not exactly replicate existing approaches.
For example, “encoder-only” models like BERT (Devlin et al., 2018) are designed to produce a single prediction per input token or a single prediction for an entire input sequence.
This makes them applicable for classification or span prediction tasks but not for generative tasks like translation or abstractive summarization.
As such, none of the model architectures we consider are identical to BERT or consist of an encoder-only structure.
Instead, we test approaches that are similar in spirit—for example, we consider an analogous objective to BERT’s “masked language modeling” objective in Section 3.3 and we consider a model architecture that behaves similarly to BERT on text classification tasks in Section 3.2.

After outlining our baseline experimental setup in the following subsection, we undertake an empirical comparison of model architectures (Section 3.2), unsupervised objectives (Section 3.3), pre-training data sets (Section 3.4), transfer approaches (Section 3.5), and scaling (Section 3.6).
At the culmination of this section, we combine insights from our study with scale to obtain state-of-the-art results in many tasks we consider (Section 3.7).

### 3.1 Baseline

Our goal for our baseline is to reflect typical, modern practice.
We pre-train a standard Transformer (described in Section 2.1) using a simple denoising objective and then separately fine-tune on each of our downstream tasks.
We describe the details of this experimental setup in the following subsections.

#### 3.1.1 Model

For our model, we use a standard encoder-decoder Transformer as proposed by Vaswani et al. (2017).
While many modern approaches to transfer learning for NLP use a Transformer architecture consisting of only a single “stack” (e.g. for language modeling (Radford et al., 2018; Dong et al., 2019) or classification and span prediction (Devlin et al., 2018; Yang et al., 2019)), we found that using a standard encoder-decoder structure achieved good results on both generative and classification tasks.
We explore the performance of different model architectures in Section 3.2.

Our baseline model is designed so that the encoder and decoder are each similar in size and configuration to a “BERTBASE” (Devlin et al., 2018) stack.
Specifically, both the encoder and decoder consist of 121212 blocks (each block comprising self-attention, optional encoder-decoder attention, and a feed-forward network).
The feed-forward networks in each block consist of a dense layer with an output dimensionality of dff=3072subscript𝑑ff3072d_{\mathrm{ff}}=3072 followed by a ReLU nonlinearity and another dense layer.
The “key” and “value” matrices of all attention mechanisms have an inner dimensionality of dkv=64subscript𝑑kv64d_{\mathrm{kv}}=64 and all attention mechanisms have 121212 heads.
All other sub-layers and embeddings have a dimensionality of dmodel=768subscript𝑑model768d_{\mathrm{model}}=768.
In total, this results in a model with about 220220220 million parameters.
This is roughly twice the number of parameters of BERTBASE since our baseline model contains two layer stacks instead of one.
For regularization, we use a dropout probability of 0.10.10.1 everywhere dropout is applied in the model.

#### 3.1.2 Training

As described in Section 2.4, all tasks are formulated as text-to-text tasks.
This allows us to always train using standard maximum likelihood, i.e. using teacher forcing (Williams and Zipser, 1989) and a cross-entropy loss.
For optimization, we use AdaFactor (Shazeer and Stern, 2018).
At test time, we use greedy decoding (i.e. choosing the highest-probability logit at every timestep).

We pre-train each model for 219=524,288superscript2195242882^{19}=524{,}288 steps on C4 before fine-tuning.
We use a maximum sequence length of 512512512 and a batch size of 128128128 sequences.
Whenever possible, we “pack” multiple sequences into each entry of the batch101010https://www.pydoc.io/pypi/tensor2tensor-1.5.7/autoapi/data_generators/generator_utils/index.html#data_generators.generator_utils.pack_examples so that our batches contain roughly 216=65,536superscript216655362^{16}=65{,}536 tokens.
In total, this batch size and number of steps corresponds to pre-training on 235≈34​Bsuperscript23534B2^{35}\approx 34\mathrm{B} tokens.
This is considerably less than BERT (Devlin et al., 2018), which used roughly 137​B137B137\mathrm{B} tokens, or RoBERTa (Liu et al., 2019c), which used roughly 2.2​T2.2T2.2\mathrm{T} tokens.
Using only 235superscript2352^{35} tokens results in a reasonable computational budget while still providing a sufficient amount of pre-training for acceptable performance.
We consider the effect of pre-training for more steps in Sections 3.6 and 3.7.
Note that 235superscript2352^{35} tokens only covers a fraction of the entire C4 data set, so we never repeat any data during pre-training.

During pre-training, we use an “inverse square root” learning rate schedule: 1/max⁡(n,k)1𝑛𝑘1\big{/}\sqrt{\max(n,k)} where n𝑛n is the current training iteration and k𝑘k is the number of warm-up steps (set to 104superscript10410^{4} in all of our experiments).
This sets a constant learning rate of 0.010.010.01 for the first 104superscript10410^{4} steps, then exponentially decays the learning rate until pre-training is over.
We also experimented with using a triangular learning rate (Howard and Ruder, 2018), which produced slightly better results but requires knowing the total number of training steps ahead of time.
Since we will be varying the number of training steps in some of our experiments, we opt for the more generic inverse square root schedule.

Our models are fine-tuned for 218=262,144superscript2182621442^{18}=262{,}144 steps on all tasks.
This value was chosen as a trade-off between the high-resource tasks (i.e. those with large data sets), which benefit from additional fine-tuning, and low-resource tasks (smaller data sets), which overfit quickly.
During fine-tuning, we continue using batches with 128128128 length-512512512 sequences (i.e. 216superscript2162^{16} tokens per batch).
We use a constant learning rate of 0.0010.0010.001 when fine-tuning.
We save a checkpoint every 5,00050005{,}000 steps and report results on the model checkpoint corresponding to the highest validation performance.
For models fine-tuned on multiple tasks, we choose the best checkpoint for each task independently.
For all of the experiments except those in Section 3.7, we report results in the validation set to avoid performing model selection on the test set.

#### 3.1.3 Vocabulary

We use SentencePiece (Kudo and Richardson, 2018) to encode text as WordPiece tokens (Sennrich et al., 2015; Kudo, 2018).
For all experiments, we use a vocabulary of 32,0003200032{,}000 wordpieces.
Since we ultimately fine-tune our model on English to German, French, and Romanian translation, we also require that our vocabulary covers these non-English languages.
To address this, we classified pages from the Common Crawl scrape used in C4 as German, French, and Romanian.
Then, we trained our SentencePiece model on a mixture of 101010 parts of English C4 data with 111 part each of data classified as German, French or Romanian.
This vocabulary was shared across both the input and output of our model.
Note that our vocabulary makes it so that our model can only process a predetermined, fixed set of languages.

#### 3.1.4 Unsupervised Objective

Leveraging unlabeled data to pre-train our model necessitates an objective that does not require labels but (loosely speaking) teaches the model generalizable knowledge that will be useful in downstream tasks.
Preliminary work that applied the transfer learning paradigm of pre-training and fine-tuning all of the model’s parameters to NLP problems used a causal language modeling objective for pre-training (Dai and Le, 2015; Peters et al., 2018; Radford et al., 2018; Howard and Ruder, 2018).
However, it has recently been shown that “denoising” objectives (Devlin et al., 2018; Taylor, 1953) (also called “masked language modeling”) produce better performance and as a result they have quickly become standard.
In a denoising objective, the model is trained to predict missing or otherwise corrupted tokens in the input.
Inspired by BERT’s “masked language modeling” objective and the “word dropout” regularization technique (Bowman et al., 2015), we design an objective that randomly samples and then drops out 15%percent1515\% of tokens in the input sequence.
All consecutive spans of dropped-out tokens are replaced by a single sentinel token.
Each sentinel token is assigned a token ID that is unique to the sequence.
The sentinel IDs are special tokens which are added to our vocabulary and do not correspond to any wordpiece.
The target then corresponds to all of the dropped-out spans of tokens, delimited by the same sentinel tokens used in the input sequence plus a final sentinel token to mark the end of the target sequence.
Our choices to mask consecutive spans of tokens and only predict dropped-out tokens were made to reduce the computational cost of pre-training.
We perform thorough investigation into pre-training objectives in Section 3.3.
An example of the transformation resulting from applying this objective is shown in Figure 2.
We empirically compare this objective to many other variants in Section 3.3.

#### 3.1.5 Baseline Performance

In this section, we present results using the baseline experimental procedure described above to get a sense of what kind of performance to expect on our suite of downstream tasks.
Ideally, we would repeat every experiment in our study multiple times to get a confidence interval on our results.
Unfortunately, this would be prohibitively expensive due to the large number of experiments we run.
As a cheaper alternative, we train our baseline model 101010 times from scratch (i.e. with different random initializations and data set shuffling) and assume that the variance over these runs of the base model also applies to each experimental variant.
We don’t expect most of the changes we make to have a dramatic effect on the inter-run variance, so this should provide a reasonable indication of the significance of different changes.
Separately, we also measure the performance of training our model for 218superscript2182^{18} steps (the same number we use for fine-tuning) on all downstream tasks without pre-training.
This gives us an idea of how much pre-training benefits our model in the baseline setting.

When reporting results in the main text, we only report a subset of the scores across all the benchmarks to conserve space and ease interpretation.
For GLUE and SuperGLUE, we report the average score across all subtasks (as stipulated by the official benchmarks) under the headings “GLUE” and “SGLUE”.
For all translation tasks, we report the BLEU score (Papineni et al., 2002) as provided by SacreBLEU v1.3.0 (Post, 2018) with “exp” smoothing and “intl” tokenization.
We refer to scores for WMT English to German, English to French, and English to Romanian as EnDe, EnFr, and EnRo, respectively.
For CNN/Daily Mail, we find the performance of models on the ROUGE-1-F, ROUGE-2-F, and ROUGE-L-F metrics (Lin, 2004) to be highly correlated so we report the ROUGE-2-F score alone under the heading “CNNDM”.
Similarly, for SQuAD we find the performance of the “exact match” and “F1” scores to be highly correlated so we report the “exact match” score alone.
We provide every score achieved on every task for all experiments in Table 16, Section E.

Our results tables are all formatted so that each row corresponds to a particular experimental configuration with columns giving the scores for each benchmark.
We will include the mean performance of the baseline configuration in most tables.
Wherever a baseline configuration appears, we will mark it with a ★★\bigstar (as in the first row of Table 1).
We also will boldface any score that is within two standard deviations of the maximum (best) in a given experiment.

GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Baseline average

83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

Baseline standard deviation
0.2350.2350.235
0.0650.0650.065
0.3430.3430.343
0.4160.4160.416
0.1120.1120.112
0.0900.0900.090
0.1080.1080.108

No pre-training
66.2266.2266.22
17.6017.6017.60
50.3150.3150.31
53.0453.0453.04
25.8625.8625.86
39.7739.77\mathbf{39.77}
24.0424.0424.04

Our baseline results are shown in Table 1.
Overall, our results are comparable to existing models of similar size.
For example, BERTBASE achieved an exact match score of 80.880.880.8 on SQuAD and an accuracy of 84.484.484.4 on MNLI-matched, whereas we achieve 80.8880.8880.88 and 84.2484.2484.24, respectively (see Table 16).
Note that we cannot directly compare our baseline to BERTBASE because ours is an encoder-decoder model and was pre-trained for roughly 1/414\nicefrac{{1}}{{4}} as many steps.
Unsurprisingly, we find that pre-training provides significant gains across almost all benchmarks.
The only exception is WMT English to French, which is a large enough data set that gains from pre-training tend to be marginal.
We include this task in our experiments to test the behavior of transfer learning in the high-resource regime.
Since we perform early stopping by selecting the best-performing checkpoint, the large disparity between our baseline and “no pre-training” emphasize how much pre-training improves performance on tasks with limited data.
While we do not explicitly measure improvements in data efficiency in this paper, we emphasize that this is one of the primary benefits of the transfer learning paradigm.

As for inter-run variance, we find that for most tasks the standard deviation across runs is smaller than 1%percent11\% of the task’s baseline score.
Exceptions to this rule include CoLA, CB, and COPA, which are all low-resource tasks from the GLUE and SuperGLUE benchmarks.
For example, on CB our baseline model had an average F1 score of 91.2291.2291.22 with a standard deviation of 3.2373.2373.237 (see Table 16), which may be partly due to the fact that CB’s validation set contains only 565656 examples.
Note that the GLUE and SuperGLUE scores are computed as the average of scores across the tasks comprising each benchmark.
As a result, we caution that the high inter-run variance of CoLA, CB, and COPA can make it harder to compare models using the GLUE and SuperGLUE scores alone.

### 3.2 Architectures

While the Transformer was originally introduced with an encoder-decoder architecture, much modern work on transfer learning for NLP uses alternative architectures.
In this section, we review and compare these architectural variants.

#### 3.2.1 Model Structures

A major distinguishing factor for different architectures is the “mask” used by different attention mechanisms in the model.
Recall that the self-attention operation in a Transformer takes a sequence as input and outputs a new sequence of the same length.
Each entry of the output sequence is produced by computing a weighted average of entries of the input sequence.
Specifically, let yisubscript𝑦𝑖y_{i} refer to the i𝑖ith element of the output sequence and xjsubscript𝑥𝑗x_{j} refer to the j𝑗jth entry of the input sequence.
yisubscript𝑦𝑖y_{i} is computed as ∑jwi,j​xjsubscript𝑗subscript𝑤𝑖𝑗subscript𝑥𝑗\sum_{j}w_{i,j}x_{j}, where wi,jsubscript𝑤𝑖𝑗w_{i,j} is the scalar weight produced by the self-attention mechanism as a function of xisubscript𝑥𝑖x_{i} and xjsubscript𝑥𝑗x_{j}.
The attention mask is then used to zero out certain weights in order to constrain which entries of the input can be attended to at a given output timestep.
Diagrams of the masks we will consider are shown in Figure 3.
For example, the causal mask (Figure 3, middle) sets any wi,jsubscript𝑤𝑖𝑗w_{i,j} to zero if j>i𝑗𝑖j>i.

The first model structure we consider is an an encoder-decoder Transformer, which consists of two layer stacks: The encoder, which is fed an input sequence, and the decoder, which produces a new output sequence.
A schematic of this architectural variant is shown in the left panel of Figure 4.

The encoder uses a “fully-visible” attention mask.
Fully-visible masking allows a self-attention mechanism to attend to any entry of the input when producing each entry of its output.
We visualize this masking pattern in Figure 3, left.
This form of masking is appropriate when attending over a “prefix”, i.e. some context provided to the model that is later used when making predictions.
BERT (Devlin et al., 2018) also uses a fully-visible masking pattern and appends a special “classification” token to the input.
BERT’s output at the timestep corresponding to the classification token is then used to make a prediction for classifying the input sequence.

The self-attention operations in the Transformer’s decoder use a “causal” masking pattern.
When producing the i𝑖ith entry of the output sequence, causal masking prevents the model from attending to the j𝑗jth entry of the input sequence for j>i𝑗𝑖j>i.
This is used during training so that the model can’t “see into the future” as it produces its output.
An attention matrix for this masking pattern is shown in Figure 3, middle.

The decoder in an encoder-decoder Transformer is used to autoregressively produce an output sequence.
That is, at each output timestep, a token is sampled from the model’s predicted distribution and the sample is fed back into the model to produce a prediction for the next output timestep, and so on.
As such, a Transformer decoder (without an encoder) can be used as a language model (LM), i.e. a model trained solely for next-step prediction (Liu et al., 2018; Radford et al., 2018; Al-Rfou et al., 2019).
This constitutes the second model structure we consider.
A schematic of this architecture is shown in Figure 4, middle.
In fact, early work on transfer learning for NLP used this architecture with a language modeling objective as a pre-training method (Radford et al., 2018).

Language models are typically used for compression or sequence generation (Graves, 2013).
However, they can also be used in the text-to-text framework simply by concatenating the inputs and targets.
As an example, consider the case of English to German translation: If we have a training datapoint with input sentence “That is good.” and target “Das ist gut.”, we would simply train the model on next-step prediction over the concatenated input sequence “translate English to German: That is good. target: Das ist gut.”
If we wanted to obtain the model’s prediction for this example, the model would be fed the prefix “translate English to German: That is good. target:” and would be asked to generate the remainder of the sequence autoregressively.
In this way, the model can predict an output sequence given an input, which satisfies the needs of text-to-text tasks.
This approach was recently used to show that language models can learn to perform some text-to-text tasks without supervision (Radford et al., 2019).

A fundamental and frequently cited drawback of using a language model in the text-to-text setting is that causal masking forces the model’s representation of the i𝑖ith entry of the input sequence to only depend on the entries up until i𝑖i.
To see why this is potentially disadvantageous, consider the text-to-text framework where the model is provided with a prefix/context before being asked to make predictions (e.g., the prefix is an English sentence and the model is asked to predict the German translation).
With fully causal masking, the model’s representation of a prefix state can only depend on prior entries of the prefix.
So, when predicting an entry of the output, the model will attend to a representation of the prefix that is unnecessarily limited.
Similar arguments have been made against using a unidirectional recurrent neural network encoder in sequence-to-sequence models (Bahdanau et al., 2015).

This issue can be avoided in a Transformer-based language model simply by changing the masking pattern.
Instead of using a causal mask, we use fully-visible masking during the prefix portion of the sequence.
This masking pattern and a schematic of the resulting “prefix LM” (the third model structure we consider) are illustrated in the rightmost panels of Figures 3 and 4, respectively.
In the English to German translation example mentioned above, fully-visible masking would be applied to the prefix “translate English to German: That is good. target:” and causal masking would be used during training for predicting the target “Das ist gut.”
Using a prefix LM in the text-to-text framework was originally proposed by Liu et al. (2018).
More recently, Dong et al. (2019) showed that this architecture is effective on a wide variety of text-to-text tasks.
This architecture is similar to an encoder-decoder model with parameters shared across the encoder and decoder and with the encoder-decoder attention replaced with full attention across the input and target sequence.

We note that when following our text-to-text framework, the prefix LM architecture closely resembles BERT (Devlin et al., 2018) for classification tasks.
To see why, consider an example from the MNLI benchmark where the premise is “I hate pigeons.”, the hypothesis is “My feelings towards pigeons are filled with animosity.” and the correct label is “entailment”.
To feed this example into a language model, we would transform it into the sequence “mnli premise: I hate pigeons. hypothesis: My feelings towards pigeons are filled with animosity. target: entailment”.
In this case, the fully-visible prefix would correspond to the entire input sequence up to the word “target:”, which can be seen as being analogous to the “classification” token used in BERT.
So, our model would have full visibility over the entire input, and then would be tasked with making a classification by outputting the word “entailment”.
It is easy for the model to learn to output one of the valid class labels given the task prefix (“mnli” in this case).
As such, the main difference between a prefix LM and the BERT architecture is that the classifier is simply integrated into the output layer of the Transformer decoder in the prefix LM.

#### 3.2.2 Comparing Different Model Structures

In the interest of experimentally comparing these architectural variants, we would like each model we consider to be equivalent in some meaningful way.
We might say that two models are equivalent if they either have the same number of parameters or they require roughly the same amount of computation to process a given (input-sequence, target-sequence) pair.
Unfortunately, it is not possible to compare an encoder-decoder model to a language model architecture (comprising a single Transformer stack) according to both of these criteria at the same time.
To see why, first note an encoder-decoder model with L𝐿L layers in the encoder and L𝐿L layers in the decoder has approximately the same number of parameters as a language model with 2​L2𝐿2L layers.
However, the same L+L𝐿𝐿L+L encoder-decoder model will have approximately the same computational cost as a language model with only L𝐿L layers.
This is a consequence of the fact that the L𝐿L layers in the language model must be applied to both the input and output sequence, while the encoder is only applied to the input sequence and the decoder is only applied to the output sequence.
Note that these equivalences are approximate—there are some extra parameters in the decoder due to the encoder-decoder attention and there are also some computational costs in the attention layers that are quadratic in the sequence lengths.
In practice, however, we observed nearly identical step times for L𝐿L-layer language models versus L+L𝐿𝐿L+L-layer encoder-decoder models, suggesting a roughly equivalent computational cost.
Further, for the model sizes we consider, the number of parameters in the encoder-decoder attention layers is about 10% of the total parameter count, so we make the simplifying assumption that an L+L𝐿𝐿L+L-layer encoder-decoder model has the same number of parameters as an 2​L2𝐿2L-layer language model.

To provide a reasonable means of comparison, we consider multiple configurations for our encoder-decoder model.
We will refer to the number of layers and parameters in a BERTBASE-sized layer stack as L𝐿L and P𝑃P, respectively.
We will use M𝑀M to refer to the number of FLOPs required for an L+L𝐿𝐿L+L-layer encoder-decoder model or L𝐿L-layer decoder-only model to process a given input-target pair.
In total, we will compare:

- •

An encoder-decoder model with L𝐿L layers in the encoder and L𝐿L layers in the decoder. This model has 2​P2𝑃2P parameters and a computation cost of M𝑀M FLOPs.

- •

An equivalent model, but with parameters shared across the encoder and decoder, resulting in P𝑃P parameters and an M𝑀M-FLOP computational cost.

- •

An encoder-decoder model with L/2𝐿2L/2 layers each in the encoder and decoder, giving P𝑃P parameters and an M/2𝑀2M/2-FLOP cost.

- •

A decoder-only language model with L𝐿L layers and P𝑃P parameters and a resulting computational cost of M𝑀M FLOPs.

- •

A decoder-only prefix LM with the same architecture (and thus the same number of parameters and computational cost), but with fully-visible self-attention over the input.

#### 3.2.3 Objectives

As an unsupervised objective, we will consider both a basic language modeling objective as well as our baseline denoising objective described in Section 3.1.4.
We include the language modeling objective due to its historic use as a pre-training objective (Dai and Le, 2015; Ramachandran et al., 2016; Howard and Ruder, 2018; Radford et al., 2018; Peters et al., 2018) as well as its natural fit for the language model architectures we consider.
For models that ingest a prefix before making predictions (the encoder-decoder model and prefix LM), we sample a span of text from our unlabeled data set and choose a random point to split it into prefix and target portions.
For the standard language model, we train the model to predict the entire span from beginning to end.
Our unsupervised denoising objective is designed for text-to-text models; to adapt it for use with a language model we concatenate the inputs and targets as described in Section 3.2.1.

#### 3.2.4 Results

Architecture
Objective
Params
Cost
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Encoder-decoder

Denoising
2​P2𝑃2P
M𝑀M
83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

Enc-dec, shared
Denoising
P𝑃P
M𝑀M
82.8182.8182.81
18.7818.7818.78
80.6380.63\mathbf{80.63}
70.7370.73\mathbf{70.73}
26.7226.7226.72
39.0339.0339.03
27.4627.46\mathbf{27.46}

Enc-dec, 6 layers
Denoising
P𝑃P
M/2𝑀2M/2
80.8880.8880.88
18.9718.9718.97
77.5977.5977.59
68.4268.4268.42
26.3826.3826.38
38.4038.4038.40
26.9526.9526.95

Language model
Denoising
P𝑃P
M𝑀M
74.7074.7074.70
17.9317.9317.93
61.1461.1461.14
55.0255.0255.02
25.0925.0925.09
35.2835.2835.28
25.8625.8625.86

Prefix LM
Denoising
P𝑃P
M𝑀M
81.8281.8281.82
18.6118.6118.61
78.9478.9478.94
68.1168.1168.11
26.4326.4326.43
37.9837.9837.98
27.3927.3927.39

Encoder-decoder
LM
2​P2𝑃2P
M𝑀M
79.5679.5679.56
18.5918.5918.59
76.0276.0276.02
64.2964.2964.29
26.2726.2726.27
39.1739.1739.17
26.8626.8626.86

Enc-dec, shared
LM
P𝑃P
M𝑀M
79.6079.6079.60
18.1318.1318.13
76.3576.3576.35
63.5063.5063.50
26.6226.6226.62
39.1739.1739.17
27.0527.0527.05

Enc-dec, 6 layers
LM
P𝑃P
M/2𝑀2M/2
78.6778.6778.67
18.2618.2618.26
75.3275.3275.32
64.0664.0664.06
26.1326.1326.13
38.4238.4238.42
26.8926.8926.89

Language model
LM
P𝑃P
M𝑀M
73.7873.7873.78
17.5417.5417.54
53.8153.8153.81
56.5156.5156.51
25.2325.2325.23
34.3134.3134.31
25.3825.3825.38

Prefix LM
LM
P𝑃P
M𝑀M
79.6879.6879.68
17.8417.8417.84
76.8776.8776.87
64.8664.8664.86
26.2826.2826.28
37.5137.5137.51
26.7626.7626.76

The scores achieved by each of the architectures we compare are shown in Table 2.
For all tasks, the encoder-decoder architecture with the denoising objective performed best.
This variant has the highest parameter count (2​P2𝑃2P) but the same computational cost as the P𝑃P-parameter decoder-only models.
Surprisingly, we found that sharing parameters across the encoder and decoder performed nearly as well.
In contrast, halving the number of layers in the encoder and decoder stacks significantly hurt performance.
Concurrent work (Lan et al., 2019) also found that sharing parameters across Transformer blocks can be an effective means of lowering the total parameter count without sacrificing much performance.
XLNet also bears some resemblance to the shared encoder-decoder approach with a denoising objective (Yang et al., 2019).
We also note that the shared parameter encoder-decoder outperforms the decoder-only prefix LM, suggesting that the addition of an explicit encoder-decoder attention is beneficial.
Finally, we confirm the widely-held conception that using a denoising objective always results in better downstream task performance compared to a language modeling objective.
This observation has been previously made by Devlin et al. (2018), Voita et al. (2019), and Lample and Conneau (2019) among others.
We undertake a more detailed exploration of unsupervised objectives in the following section.

### 3.3 Unsupervised Objectives

The choice of unsupervised objective is of central importance as it provides the mechanism through which the model gains general-purpose knowledge to apply to downstream tasks.
This has led to the development of a wide variety of pre-training objectives (Dai and Le, 2015; Ramachandran et al., 2016; Radford et al., 2018; Devlin et al., 2018; Yang et al., 2019; Liu et al., 2019b; Wang et al., 2019a; Song et al., 2019; Dong et al., 2019; Joshi et al., 2019).
In this section, we perform a procedural exploration of the space of unsupervised objectives.
In many cases, we will not replicate an existing objective exactly—some will be modified to fit our text-to-text encoder-decoder framework and, in other cases, we will use objectives that combine concepts from multiple common approaches.

Overall, all of our objectives ingest a sequence of token IDs corresponding to a tokenized span of text from our unlabeled text data set.
The token sequence is processed to produce a (corrupted) input sequence and a corresponding target.
Then, the model is trained as usual with maximum likelihood to predict the target sequence.
We provide illustrative examples of many of the objectives we consider in Table 3.

Objective
Inputs
Targets

Prefix language modeling
Thank you for inviting
me to your party last week .

BERT-style Devlin et al. (2018)

Thank you <M> <M> me to your party apple week .

(original text)

Deshuffling
party me for your to . last fun you inviting week Thank
(original text)

MASS-style Song et al. (2019)

Thank you <M> <M> me to your party <M> week .

(original text)

I.i.d. noise, replace spans

Thank you <X> me to your party <Y> week .

<X> for inviting <Y> last <Z>

I.i.d. noise, drop tokens
Thank you me to your party week .
for inviting last

Random spans

Thank you <X> to <Y> week .

<X> for inviting me <Y> your party last <Z>

#### 3.3.1 Disparate High-Level Approaches

To begin with, we compare three techniques that are inspired by commonly-used objectives but differ significantly in their approach.
First, we include a basic “prefix language modeling” objective as was used in Section 3.2.3.
This technique splits a span of text into two components, one to use as inputs to the encoder and the other to use as a target sequence to be predicted by the decoder.
Second, we consider an objective inspired by the “masked language modeling” (MLM) objective used in BERT (Devlin et al., 2018).
MLM takes a span of text and corrupts 15%percent1515\% of the tokens.
90%percent9090\% of the corrupted tokens are replaced with a special mask token and 10%percent1010\% are replaced with a random token.
Since BERT is an encoder-only model, its goal during pre-training is to reconstruct masked tokens at the output of the encoder.
In the encoder-decoder case, we simply use the entire uncorrupted sequence as the target.
Note that this differs from our baseline objective, which uses only the corrupted tokens as targets; we compare these two approaches in Section 3.3.2.
Finally, we also consider a basic deshuffling objective as used e.g. in (Liu et al., 2019a) where it was applied to a denoising sequential autoencoder.
This approach takes a sequence of tokens, shuffles it, and then uses the original deshuffled sequence as a target.
We provide examples of the inputs and targets for these three methods in the first three rows of Table 3.

Objective
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

Prefix language modeling
80.6980.6980.69
18.9418.9418.94
77.9977.9977.99
65.2765.2765.27
26.8626.86\mathbf{26.86}
39.7339.7339.73
27.4927.49\mathbf{27.49}

BERT-style (Devlin et al., 2018)

82.9682.96\mathbf{82.96}
19.1719.17\mathbf{19.17}
80.6580.65\mathbf{80.65}
69.8569.85\mathbf{69.85}
26.7826.78\mathbf{26.78}
40.0340.03\mathbf{40.03}
27.4127.41\mathbf{27.41}

Deshuffling
73.1773.1773.17
18.5918.5918.59
67.6167.6167.61
58.4758.4758.47
26.1126.1126.11
39.3039.3039.30
25.6225.6225.62

The performance of these three objectives is shown in Table 4.
Overall, we find that the BERT-style objective performs best, though the prefix language modeling objective attains similar performance on the translation tasks.
Indeed, the motivation for the BERT objective was to outperform language model-based pre-training.
The deshuffling objective performs considerably worse than both prefix language modeling and the BERT-style objective.

#### 3.3.2 Simplifying the BERT Objective

Based on the results in the prior section, we will now focus on exploring modifications to the BERT-style denoising objective.
This objective was originally proposed as a pre-training technique for an encoder-only model trained for classification and span prediction.
As such, it may be possible to modify it so that it performs better or is more efficient in our encoder-decoder text-to-text setup.

First, we consider a simple variant of the BERT-style objective where we don’t include the random token swapping step.
The resulting objective simply replaces 15%percent1515\% of the tokens in the input with a mask token and the model is trained to reconstruct the original uncorrupted sequence.
A similar masking objective was used by Song et al. (2019) where it was referred to as “MASS”, so we call this variant the “MASS-style” objective.
Second, we were interested to see if it was possible to avoid predicting the entire uncorrupted text span since this requires self-attention over long sequences in the decoder.
We consider two strategies to achieve this:
First, instead of replacing each corrupted token with a mask token, we replace the entirety of each consecutive span of corrupted tokens with a unique mask token.
Then, the target sequence becomes the concatenation of the “corrupted” spans, each prefixed by the mask token used to replace it in the input.
This is the pre-training objective we use in our baseline, described in Section 3.1.4.
Second, we also consider a variant where we simply drop the corrupted tokens from the input sequence completely and task the model with reconstructing the dropped tokens in order.
Examples of these approaches are shown in the fifth and sixth rows of Table 3.

Objective
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

BERT-style (Devlin et al., 2018)

82.9682.9682.96
19.1719.1719.17
80.6580.65\mathbf{80.65}
69.8569.8569.85
26.7826.7826.78
40.0340.03\mathbf{40.03}
27.4127.4127.41

MASS-style (Song et al., 2019)

82.3282.3282.32
19.1619.1619.16
80.1080.1080.10
69.2869.2869.28
26.7926.7926.79
39.8939.89\mathbf{39.89}
27.5527.5527.55

★★\bigstar\,Replace corrupted spans

83.2883.2883.28
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.8239.82
27.6527.65\mathbf{27.65}

Drop corrupted tokens
84.4484.44\mathbf{84.44}
19.3119.31\mathbf{19.31}
80.5280.52\mathbf{80.52}
68.6768.6768.67
27.0727.07\mathbf{27.07}
39.7639.7639.76
27.8227.82\mathbf{27.82}

An empirical comparison of the original BERT-style objective to these three alternatives is shown in Table 5.
We find that in our setting, all of these variants perform similarly.
The only exception was that dropping corrupted tokens completely produced a small improvement in the GLUE score thanks to a significantly higher score on CoLA (60.0460.0460.04, compared to our baseline average of 53.8453.8453.84, see Table 16).
This may be due to the fact that CoLA involves classifying whether a given sentence is grammatically and syntactically acceptable, and being able to determine when tokens are missing is closely related to detecting acceptability.
However, dropping tokens completely performed worse than replacing them with sentinel tokens on SuperGLUE.
The two variants that do not require predicting the full original sequence (“replace corrupted spans” and “drop corrupted spans”) are both potentially attractive since they make the target sequences shorter and consequently make training faster.
Going forward, we will explore variants where we replace corrupted spans with sentinel tokens and only predict the corrupted tokens (as in our baseline objective).

#### 3.3.3 Varying the Corruption Rate

So far, we have been corrupting 15% of the tokens, the value used in BERT (Devlin et al., 2018).
Again, since our text-to-text framework differs from BERT’s, we are interested to see if a different corruption rate works better for us.
We compare corruption rates of 10%percent1010\%, 15%percent1515\%, 25%percent2525\%, and 50%percent5050\% in Table 6.
Overall, we find that the corruption rate had a limited effect on the model’s performance.
The only exception is that the largest corruption rate we consider (50%percent5050\%) results in a significant degradation of performance on GLUE and SQuAD.
Using a larger corruption rate also results in longer targets, which can potentially slow down training.
Based on these results and the historical precedent set by BERT, we will use a corruption rate of 15%percent1515\% going forward.

Corruption rate
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

10%percent1010\%
82.8282.82\mathbf{82.82}
19.0019.0019.00
80.3880.38\mathbf{80.38}
69.5569.5569.55
26.8726.87\mathbf{26.87}
39.2839.2839.28
27.4427.44\mathbf{27.44}

★★\bigstar\,15%percent1515\%

83.2883.28\mathbf{83.28}
19.2419.2419.24
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

25%percent2525\%
83.0083.00\mathbf{83.00}
19.5419.54\mathbf{19.54}
80.9680.96\mathbf{80.96}
70.4870.4870.48
27.0427.04\mathbf{27.04}
39.8339.83\mathbf{39.83}
27.4727.47\mathbf{27.47}

50%percent5050\%
81.2781.2781.27
19.3219.3219.32
79.8079.8079.80
70.3370.3370.33
27.0127.01\mathbf{27.01}
39.9039.90\mathbf{39.90}
27.4927.49\mathbf{27.49}

#### 3.3.4 Corrupting Spans

We now turn towards the goal of speeding up training by predicting shorter targets.
The approach we have used so far makes an i.i.d. decision for each input token as to whether to corrupt it or not.
When multiple consecutive tokens have been corrupted, they are treated as a “span” and a single unique mask token is used to replace the entire span.
Replacing entire spans with a single token results in unlabeled text data being processed into shorter sequences.
Since we are using an i.i.d. corruption strategy, it is not always the case that a significant number of corrupted tokens appear consecutively.
As a result, we might obtain additional speedup by specifically corrupting spans of tokens rather than corrupting individual tokens in an i.i.d. manner.
Corrupting spans was also previously considered as a pre-training objective for BERT, where it was found to improve performance (Joshi et al., 2019).

To test this idea, we consider an objective that specifically corrupts contiguous, randomly-spaced spans of tokens.
This objective can be parametrized by the proportion of tokens to be corrupted and the total number of corrupted spans.
The span lengths are then chosen randomly to satisfy these specified parameters.
For example, if we are processing a sequence of 500500500 tokens and we have specified that 15%percent1515\% of tokens should be corrupted and that there should be 252525 total spans, then the total number of corrupted tokens would be 500×0.15=755000.1575500\times 0.15=75 and the average span length would be 75/25=37525375/25=3.
Note that given the original sequence length and corruption rate, we can equivalently parametrize this objective either by the average span length or the total number of spans.

Span length
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Baseline (i.i.d.)

83.2883.28\mathbf{83.28}
19.2419.2419.24
80.8880.8880.88
71.3671.3671.36
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

222
83.5483.54\mathbf{83.54}
19.3919.3919.39
82.0982.09\mathbf{82.09}
72.2072.20\mathbf{72.20}
26.7626.76\mathbf{26.76}
39.9939.99\mathbf{39.99}
27.6327.63\mathbf{27.63}

333
83.4983.49\mathbf{83.49}
19.6219.62\mathbf{19.62}
81.8481.84\mathbf{81.84}
72.5372.53\mathbf{72.53}
26.8626.86\mathbf{26.86}
39.6539.6539.65
27.6227.62\mathbf{27.62}

555
83.4083.40\mathbf{83.40}
19.2419.2419.24
82.0582.05\mathbf{82.05}
72.2372.23\mathbf{72.23}
26.8826.88\mathbf{26.88}
39.4039.4039.40
27.5327.53\mathbf{27.53}

101010
82.8582.8582.85
19.3319.3319.33
81.8481.84\mathbf{81.84}
70.4470.4470.44
26.7926.79\mathbf{26.79}
39.4939.4939.49
27.6927.69\mathbf{27.69}

We compare the span-corruption objective to the i.i.d-corruption objective in Table 7.
We use a corruption rate of 15%percent1515\% in all cases and compare using average span lengths of 222, 333, 555 and 101010.
Again, we find a limited difference between these objectives, though the version with an average span length of 101010 slightly underperforms the other values in some cases.
We also find in particular that using an average span length of 333 slightly (but significantly) outperforms the i.i.d. objective on most non-translation benchmarks.
Fortunately, the span-corruption objective also provides some speedup during training compared to the i.i.d. noise approach because span corruption produces shorter sequences on average.

#### 3.3.5 Discussion

Figure 5 shows a flow chart of the choices made during our exploration of unsupervised objectives.
Overall, the most significant difference in performance we observed was that denoising objectives outperformed language modeling and deshuffling for pre-training.
We did not observe a remarkable difference across the many variants of the denoising objectives we explored.
However, different objectives (or parameterizations of objectives) can lead to different sequence lengths and thus different training speeds.
This implies that choosing among the denoising objectives we considered here should mainly be done according to their computational cost.
Our results also suggest that additional exploration of objectives similar to the ones we consider here may not lead to significant gains for the tasks and model we consider.
Instead, it may be fortuitous to explore entirely different ways of leveraging unlabeled data.

### 3.4 Pre-training Data set

Like the unsupervised objective, the pre-training data set itself is a crucial component of the transfer learning pipeline.
However, unlike objectives and benchmarks, new pre-training data sets are usually not treated as significant contributions on their own and are often not released alongside pre-trained models and code.
Instead, they are typically introduced in the course of presenting a new method or model.
As a result, there has been relatively little comparison of different pre-training data sets as well as a lack of a “standard” data set used for pre-training.
Some recent notable exceptions (Baevski et al., 2019; Liu et al., 2019c; Yang et al., 2019) have compared pre-training on a new large (often Common Crawl-sourced) data set to using a smaller preexisting data set (often Wikipedia).
To probe more deeply into the impact of the pre-training data set on performance, in this section we compare variants of our C4 data set and other potential sources of pre-training data.
We release all of the C4 data set variants we consider as part of TensorFlow Datasets.111111https://www.tensorflow.org/datasets/catalog/c4

#### 3.4.1 Unlabeled Data Sets

In creating C4, we developed various heuristics to filter the web-extracted text from Common Crawl (see Section 2.2 for a description).
We are interested in measuring whether this filtering results in improved performance on downstream tasks, in addition to comparing it to other filtering approaches and common pre-training data sets.
Towards this end, we compare the performance of our baseline model after pre-training on the following data sets:

C4

As a baseline, we first consider pre-training on our proposed unlabeled data set as described in Section 2.2.

To measure the effect of the heuristic filtering we used in creating C4 (deduplication, removing bad words, only retaining sentences, etc.), we also generate an alternate version of C4 that forgoes this filtering.
Note that we still use langdetect to extract English text.
As a result, our “unfiltered” variant still includes some filtering because langdetect sometimes assigns a low probability to non-natural English text.

Recent work has used text data extracted from news websites (Zellers et al., 2019; Baevski et al., 2019).
To compare to this approach, we generate another unlabeled data set by additionally filtering C4 to only include content from one of the domains used in the “RealNews” data set (Zellers et al., 2019).
Note that for ease of comparison, we retain the heuristic filtering methods used in C4; the only difference is that we have ostensibly omitted any non-news content.

Similarly, the WebText data set (Radford et al., 2019) only uses content from webpages that were submitted to the content aggregation website Reddit and received a “score” of at least 3.
The score for a webpage submitted to Reddit is computed based on the proportion of users who endorse (upvote) or oppose (downvote) the webpage.
The idea behind using the Reddit score as a quality signal is that users of the site would only upvote high-quality text content.
To generate a comparable data set, we first tried removing all content from C4 that did not originate from a URL that appeared in the list prepared by the OpenWebText effort.121212https://github.com/jcpeterson/openwebtext
However, this resulted in comparatively little content—only about 2 GB—because most pages never appear on Reddit.
Recall that C4 was created based on a single month of Common Crawl data.
To avoid using a prohibitively small data set, we therefore downloaded 12 months of data from Common Crawl from August 2018 to July 2019, applied our heuristic filtering for C4, then applied the Reddit filter.
This produced a 17 GB WebText-like data set, which is of comparable size to the original 40GB WebText data set (Radford et al., 2019).

The website Wikipedia consists of millions of encyclopedia articles written collaboratively.
The content on the site is subject to strict quality guidelines and therefore has been used as a reliable source of clean and natural text.
We use the English Wikipedia text data from TensorFlow Datasets,131313https://www.tensorflow.org/datasets/catalog/wikipedia which omits any markup or reference sections from the articles.

A drawback of using pre-training data from Wikipedia is that it represents only one possible domain of natural text (encyclopedia articles).
To mitigate this, BERT (Devlin et al., 2018) combined data from Wikipedia with the Toronto Books Corpus (TBC) (Zhu et al., 2015).
TBC contains text extracted from eBooks, which represents a different domain of natural language.
BERT’s popularity has led to the Wikipedia + TBC combination being used in many subsequent works.

Data set
Size
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,C4

745GB
83.2883.2883.28
19.2419.24\mathbf{19.24}
80.8880.8880.88
71.3671.3671.36
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

C4, unfiltered
6.1TB
81.4681.4681.46
19.1419.1419.14
78.7878.7878.78
68.0468.0468.04
26.5526.5526.55
39.3439.3439.34
27.2127.2127.21

RealNews-like
35GB
83.8383.83\mathbf{83.83}
19.2319.23\mathbf{19.23}
80.3980.3980.39
72.3872.3872.38
26.7526.75\mathbf{26.75}
39.9039.90\mathbf{39.90}
27.4827.48\mathbf{27.48}

WebText-like
17GB
84.0384.03\mathbf{84.03}
19.3119.31\mathbf{19.31}
81.4281.42\mathbf{81.42}
71.4071.4071.40
26.8026.80\mathbf{26.80}
39.7439.74\mathbf{39.74}
27.5927.59\mathbf{27.59}

Wikipedia
16GB
81.8581.8581.85
19.3119.31\mathbf{19.31}
81.2981.2981.29
68.0168.0168.01
26.9426.94\mathbf{26.94}
39.6939.6939.69
27.6727.67\mathbf{27.67}

Wikipedia + TBC
20GB
83.6583.6583.65
19.2819.28\mathbf{19.28}
82.0882.08\mathbf{82.08}
73.2473.24\mathbf{73.24}
26.7726.77\mathbf{26.77}
39.6339.6339.63
27.5727.57\mathbf{27.57}

The results achieved after pre-training on each of these data sets is shown in Table 8.
A first obvious takeaway is that removing the heuristic filtering from C4 uniformly degrades performance and makes the unfiltered variant perform the worst in every task.
Beyond this, we found that in some cases a pre-training data set with a more constrained domain outperformed the diverse C4 data set.
For example, using the Wikipedia + TBC corpus produced a SuperGLUE score of 73.2473.2473.24, beating our baseline’s score (using C4) of 71.3671.3671.36.
This is almost entirely attributable to a boost in performance from 25.7825.7825.78 (baseline, C4) to 50.9350.9350.93 (Wikipedia + TBC) on the Exact Match score for MultiRC (see Table 16).
MultiRC is a reading comprehension data set whose largest source of data comes from fiction books, which is exactly the domain covered by TBC.
Similarly, using the RealNews-like data set for pre-training conferred an increase from 68.1668.1668.16 to 73.7273.7273.72 on the Exact Match score for ReCoRD, a data set that measures reading comprehension on news articles.
As a final example, using data from Wikipedia produced significant (but less dramatic) gains on SQuAD, which is a question-answering data set with passages sourced from Wikipedia.
Similar observations have been made in prior work, e.g. Beltagy et al. (2019) found that pre-training BERT on text from research papers improved its performance on scientific tasks.
The main lesson behind these findings is that pre-training on in-domain unlabeled data can improve performance on downstream tasks.
This is unsurprising but also unsatisfying if our goal is to pre-train a model that can rapidly adapt to language tasks from arbitrary domains.
Liu et al. (2019c) also observed that pre-training on a more diverse data set yielded improvements on downstream tasks.
This observation also motivates the parallel line of research on domain adaptation for natural language processing; for surveys of this field see e.g. Ruder (2019); Li (2012).

A drawback to only pre-training on a single domain is that the resulting data sets are often substantially smaller.
Similarly, while the WebText-like variant performed as well or better than the C4 data set in our baseline setting, the Reddit-based filtering produced a data set that was about 40×40\times smaller than C4 despite being based on 12×12\times more data from Common Crawl.
Note, however, that in our baseline setup we only pre-train on 235≈34​Bsuperscript23534B2^{35}\approx 34\mathrm{B} tokens, which is only about 888 times larger than the smallest pre-training data set we consider.
We investigate at what point using a smaller pre-training data sets poses an issue in the following section.

#### 3.4.2 Pre-training Data set Size

The pipeline we use to create C4 was designed to be able to create extremely large pre-training data sets.
The access to so much data allows us to pre-train our models without repeating examples.
It is not clear whether repeating examples during pre-training would be helpful or harmful to downstream performance because our pre-training objective is itself stochastic and can help prevent the model from seeing the same exact data multiple times.

To test the effect of limited unlabeled data set sizes, we pre-trained our baseline model on artificially truncated versions of C4.
Recall that we pre-train our baseline model on 235≈34​Bsuperscript23534B2^{35}\approx 34\mathrm{B} tokens (a small fraction of the total size of C4).
We consider training on truncated variants of C4 consisting of 229superscript2292^{29}, 227superscript2272^{27}, 225superscript2252^{25} and 223superscript2232^{23} tokens.
These sizes correspond to repeating the data set 646464, 256256256, 1,02410241{,}024, and 4,09640964{,}096 times respectively over the course of pre-training.

Number of tokens
Repeats
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Full data set

00
83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

229superscript2292^{29}
646464
82.8782.87\mathbf{82.87}
19.1919.19\mathbf{19.19}
80.9780.97\mathbf{80.97}
72.0372.03\mathbf{72.03}
26.8326.83\mathbf{26.83}
39.7439.74\mathbf{39.74}
27.6327.63\mathbf{27.63}

227superscript2272^{27}
256256256
82.6282.6282.62
19.2019.20\mathbf{19.20}
79.7879.7879.78
69.9769.9769.97
27.0227.02\mathbf{27.02}
39.7139.71\mathbf{39.71}
27.3327.3327.33

225superscript2252^{25}
1,02410241{,}024
79.5579.5579.55
18.5718.5718.57
76.2776.2776.27
64.7664.7664.76
26.3826.3826.38
39.5639.5639.56
26.8026.8026.80

223superscript2232^{23}
4,09640964{,}096
76.3476.3476.34
18.3318.3318.33
70.9270.9270.92
59.2959.2959.29
26.3726.3726.37
38.8438.8438.84
25.8125.8125.81

The resulting downstream performance is shown in Table 9.
As expected, performance degrades as the data set size shrinks.
We suspect this may be due to the fact that the model begins to memorize the pre-training data set.
To measure if this is true, we plot the training loss for each of these data set sizes in Figure 6.
Indeed, the model attains significantly smaller training losses as the size of the pre-training data set shrinks, suggesting possible memorization.
Baevski et al. (2019) similarly observed that truncating the pre-training data set size can degrade downstream task performance.

We note that these effects are limited when the pre-training data set is repeated only 646464 times.
This suggests that some amount of repetition of pre-training data might not be harmful.
However, given that additional pre-training can be beneficial (as we will show in Section 3.6) and that obtaining additional unlabeled data is cheap and easy, we suggest using large pre-training data sets whenever possible.
We also note that this effect may be more pronounced for larger model sizes, i.e. a bigger model may be more prone to overfitting to a smaller pre-training data set.

### 3.5 Training Strategy

So far we have considered the setting where all parameters of a model are pre-trained on an unsupervised task before being fine-tuned on individual supervised tasks.
While this approach is straightforward, various alternative methods for training the model on downstream/supervised tasks have been proposed.
In this section, we compare different schemes for fine-tuning the model in addition to the approach of training the model simultaneously on multiple tasks.

#### 3.5.1 Fine-tuning Methods

It has been argued that fine-tuning all of the model’s parameters can lead to suboptimal results, particularly on low-resource tasks (Peters et al., 2019).
Early results on transfer learning for text classification tasks advocated fine-tuning only the parameters of a small classifier that was fed sentence embeddings produced by a fixed pre-trained model (Subramanian et al., 2018; Kiros et al., 2015; Logeswaran and Lee, 2018; Hill et al., 2016; Conneau et al., 2017).
This approach is less applicable to our encoder-decoder model because the entire decoder must be trained to output the target sequences for a given task.
Instead, we focus on two alternative fine-tuning approaches that update only a subset of the parameters of our encoder-decoder model.

The first, “adapter layers” (Houlsby et al., 2019; Bapna et al., 2019), is motivated by the goal of keeping most of the original model fixed while fine-tuning.
Adapter layers are additional dense-ReLU-dense blocks that are added after each of the preexisting feed-forward networks in each block of the Transformer.
These new feed-forward networks are designed so that their output dimensionality matches their input.
This allows them to be inserted into the network with no additional changes to the structure or parameters.
When fine-tuning, only the adapter layer and layer normalization parameters are updated.
The main hyperparameter of this approach is the inner dimensionality d𝑑d of the feed-forward network, which changes the number of new parameters added to the model.
We experiment with various values for d𝑑d.

The second alternative fine-tuning method we consider is “gradual unfreezing” (Howard and Ruder, 2018).
In gradual unfreezing, more and more of the model’s parameters are fine-tuned over time.
Gradual unfreezing was originally applied to a language model architecture consisting of a single stack of layers.
In this setting, at the start of fine-tuning only the parameters of the final layer are updated, then after training for a certain number of updates the parameters of the second-to-last layer are also included, and so on until the entire network’s parameters are being fine-tuned.
To adapt this approach to our encoder-decoder model, we gradually unfreeze layers in the encoder and decoder in parallel, starting from the top in both cases.
Since the parameters of our input embedding matrix and output classification matrix are shared, we update them throughout fine-tuning.
Recall that our baseline model consists of 121212 layers each in the encoder and decoder and is fine-tuned for 218superscript2182^{18} steps.
As such, we subdivide the fine-tuning process into 121212 episodes of 218/12superscript21812\nicefrac{{2^{18}}}{{12}} steps each and train from layers 12−n12𝑛12-n to 121212 in the n𝑛nth episode.
We note that Howard and Ruder (2018) suggested fine-tuning an additional layer after each epoch of training.
However, since our supervised data sets vary so much in size and since some of our downstream tasks are actually mixtures of many tasks (GLUE and SuperGLUE), we instead adopt the simpler strategy of fine-tuning an additional layer after every 218/12superscript21812\nicefrac{{2^{18}}}{{12}} steps.

A comparison of the performance of these fine-tuning approaches is shown in Table 10.
For adapter layers, we report the performance using an inner dimensionality d𝑑d of 323232, 128128128, 512512512, 204820482048.
Pursuant with past results (Houlsby et al., 2019; Bapna et al., 2019) we find that lower-resource tasks like SQuAD work well with a small value of d𝑑d whereas higher resource tasks require a large dimensionality to achieve reasonable performance.
This suggests that adapter layers could be a promising technique for fine-tuning on fewer parameters as long as the dimensionality is scaled appropriately to the task size.
Note that in our case we treat GLUE and SuperGLUE each as a single “task” by concatenating their constituent data sets, so although they comprise some low-resource data sets the combined data set is large enough that it necessitates a large value of d𝑑d.
We found that gradual unfreezing caused a minor degradation in performance across all tasks, though it did provide some speedup during fine-tuning.
Better results may be attainable by more carefully tuning the unfreezing schedule.

Fine-tuning method
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,All parameters

83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

Adapter layers, d=32𝑑32d=32

80.5280.5280.52
15.0815.0815.08
79.3279.3279.32
60.4060.4060.40
13.8413.8413.84
17.8817.8817.88
15.5415.5415.54

Adapter layers, d=128𝑑128d=128

81.5181.5181.51
16.6216.6216.62
79.4779.4779.47
63.0363.0363.03
19.8319.8319.83
27.5027.5027.50
22.6322.6322.63

Adapter layers, d=512𝑑512d=512

81.5481.5481.54
17.7817.7817.78
79.1879.1879.18
64.3064.3064.30
23.4523.4523.45
33.9833.9833.98
25.8125.8125.81

Adapter layers, d=2048𝑑2048d=2048

81.5181.5181.51
16.6216.6216.62
79.4779.4779.47
63.0363.0363.03
19.8319.8319.83
27.5027.5027.50
22.6322.6322.63

Gradual unfreezing
82.5082.5082.50
18.9518.9518.95
79.1779.1779.17
70.7970.79\mathbf{70.79}
26.7126.7126.71
39.0239.0239.02
26.9326.9326.93

#### 3.5.2 Multi-task Learning

So far, we have been pre-training our model on a single unsupervised learning task before fine-tuning it individually on each downstream task.
An alternative approach, called “multi-task learning” (Ruder, 2017; Caruana, 1997), is to train the model on multiple tasks at a time.
This approach typically has the goal of training a single model that can simultaneously perform many tasks at once, i.e. the model and most of its parameters are shared across all tasks.
We relax this goal somewhat and instead investigate methods for training on multiple tasks at once in order to eventually produce separate parameter settings that perform well on each individual task.
For example, we might train a single model on many tasks, but when reporting performance we are allowed to select a different checkpoint for each task.
This loosens the multi-task learning framework and puts it on more even footing compared to the pre-train-then-fine-tune approach we have considered so far.
We also note that in our unified text-to-text framework, “multi-task learning” simply corresponds to mixing data sets together.
It follows that we can still train on unlabeled data when using multi-task learning by treating the unsupervised task as one of the tasks being mixed together.
In contrast, most applications of multi-task learning to NLP add task-specific classification networks or use different loss functions for each task (Liu et al., 2019b).

As pointed out by Arivazhagan et al. (2019), an extremely important factor in multi-task learning is how much data from each task the model should be trained on.
Our goal is to not under- or over-train the model—that is, we want the model to see enough data from a given task that it can perform the task well, but not to see so much data that it memorizes the training set.
How exactly to set the proportion of data coming from each task can depend on various factors including data set sizes, the “difficulty” of learning the task (i.e. how much data the model must see before being able to perform the task effectively), regularization, etc.
An additional issue is the potential for “task interference” or “negative transfer”, where achieving good performance on one task can hinder performance on another.
Given these concerns, we begin by exploring various strategies for setting the proportion of data coming from each task.
A similar exploration was performed by Wang et al. (2019a).

Examples-proportional mixing

A major factor in how quickly a model will overfit to a given task is the task’s data set size.
As such, a natural way to set the mixing proportions is to sample in proportion to the size of each task’s data set.
This is equivalent to concatenating the data sets for all tasks and randomly sampling examples from the combined data set.
Note, however, that we are including our unsupervised denoising task, which uses a data set that is orders of magnitude larger than every other task’s.
It follows that if we simply sample in proportion to each data set’s size, the vast majority of the data the model sees will be unlabeled, and it will undertrain on all of the supervised tasks.
Even without the unsupervised task, some tasks (e.g. WMT English to French) are so large that they would similarly crowd out most of the batches.
To get around this issue, we set an artificial “limit” on the data set sizes before computing the proportions.
Specifically, if the number of examples in each of our N𝑁N task’s data sets is en,n∈{1,…,N}subscript𝑒𝑛𝑛1…𝑁e_{n},n\in\{1,\ldots,N\} then we set probability of sampling an example from the m𝑚mth task during training to rm=min⁡(em,K)/∑min⁡(en,K)subscript𝑟𝑚subscript𝑒𝑚𝐾subscript𝑒𝑛𝐾r_{m}=\min(e_{m},K)/\sum\min(e_{n},K) where K𝐾K is the artificial data set size limit.

An alternative way of mitigating the huge disparity between data set sizes is to adjust the “temperature” of the mixing rates.
This approach was used by multilingual BERT to ensure that the model was sufficiently trained on low-resource languages.141414https://github.com/google-research/bert/blob/master/multilingual.md
To implement temperature scaling with temperature T𝑇T, we raise each task’s mixing rate rmsubscript𝑟𝑚r_{m} to the power of 1/T1𝑇\nicefrac{{1}}{{$T$}} and renormalize the rates so that they sum to 1.
When T=1𝑇1T=1, this approach is equivalent to examples-proportional mixing and as T𝑇T increases the proportions become closer to equal mixing.
We retain the data set size limit K𝐾K (applied to obtain rmsubscript𝑟𝑚r_{m} before temperature scaling) but set it to a large value of K=221𝐾superscript221K=2^{21}.
We use a large value of K𝐾K because increasing the temperature will decrease the mixing rate of the largest data sets.

In this case, we sample examples from each task with equal probability.
Specifically, each example in each batch is sampled uniformly at random from one of the data sets we train on.
This is most likely a suboptimal strategy, as the model will overfit quickly on low-resource tasks and underfit on high-resource tasks.
We mainly include it as a point of reference of what might go wrong when the proportions are set suboptimally.

To compare these mixing strategies on equal footing with our baseline pre-train-then-fine-tune results, we train multi-task models for the same total number of steps: 219+218=786,432superscript219superscript2187864322^{19}+2^{18}=786{,}432.
The results are shown in Table 11.

In general, we find that multi-task training underperforms pre-training followed by fine-tuning on most tasks.
The “equal” mixing strategy in particular results in dramatically degraded performance, which may be because the low-resource tasks have overfit, the high-resource tasks have not seen enough data, or the model has not seen enough unlabeled data to learn general-purpose language capabilities.
For examples-proportional mixing, we find that for most tasks there is a “sweet spot” for K𝐾K where the model obtains the best performance, and larger or smaller values of K𝐾K tend to result in worse performance.
The exception (for the range of K𝐾K values we considered) was WMT English to French translation, which is such a high-resource task that it always benefits from a higher mixing proportion.
Finally, we note that temperature-scaled mixing also provides a means of obtaining reasonable performance from most tasks, with T=2𝑇2T=2 performing the best in most cases.
The finding that a multi-task model is outperformed by separate models trained on each individual task has previously been observed e.g. by Arivazhagan et al. (2019) and McCann et al. (2018), though it has been shown that the multi-task setup can confer benefits across very similar tasks Liu et al. (2019b); Ratner et al. (2018).
In the following section, we explore ways to close the gap between multi-task training and the pre-train-then-fine-tune approach.

Mixing strategy
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Baseline (pre-train/fine-tune)

83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.82\mathbf{39.82}
27.6527.65\mathbf{27.65}

Equal
76.1376.1376.13
19.0219.0219.02
76.5176.5176.51
63.3763.3763.37
23.8923.8923.89
34.3134.3134.31
26.7826.7826.78

Examples-proportional, K=216𝐾superscript216K=2^{16}

80.4580.4580.45
19.0419.0419.04
77.2577.2577.25
69.9569.9569.95
24.3524.3524.35
34.9934.9934.99
27.1027.1027.10

Examples-proportional, K=217𝐾superscript217K=2^{17}

81.5681.5681.56
19.1219.1219.12
77.0077.0077.00
67.9167.9167.91
24.3624.3624.36
35.0035.0035.00
27.2527.2527.25

Examples-proportional, K=218𝐾superscript218K=2^{18}

81.6781.6781.67
19.0719.0719.07
78.1778.1778.17
67.9467.9467.94
24.5724.5724.57
35.1935.1935.19
27.3927.3927.39

Examples-proportional, K=219𝐾superscript219K=2^{19}

81.4281.4281.42
19.2419.24\mathbf{19.24}
79.7879.7879.78
67.3067.3067.30
25.2125.2125.21
36.3036.3036.30
27.7627.76\mathbf{27.76}

Examples-proportional, K=220𝐾superscript220K=2^{20}

80.8080.8080.80
19.2419.24\mathbf{19.24}
80.3680.36\mathbf{80.36}
67.3867.3867.38
25.6625.6625.66
36.9336.9336.93
27.6827.68\mathbf{27.68}

Examples-proportional, K=221𝐾superscript221K=2^{21}

79.8379.8379.83
18.7918.7918.79
79.5079.5079.50
65.1065.1065.10
25.8225.8225.82
37.2237.2237.22
27.1327.1327.13

Temperature-scaled, T=2𝑇2T=2

81.9081.9081.90
19.2819.28\mathbf{19.28}
79.4279.4279.42
69.9269.9269.92
25.4225.4225.42
36.7236.7236.72
27.2027.2027.20

Temperature-scaled, T=4𝑇4T=4

80.5680.5680.56
19.2219.22\mathbf{19.22}
77.9977.9977.99
69.5469.5469.54
25.0425.0425.04
35.8235.8235.82
27.4527.4527.45

Temperature-scaled, T=8𝑇8T=8

77.2177.2177.21
19.1019.1019.10
77.1477.1477.14
66.0766.0766.07
24.5524.5524.55
35.3535.3535.35
27.1727.1727.17

#### 3.5.3 Combining Multi-Task Learning with Fine-Tuning

Recall that we are studying a relaxed version of multi-task learning where we train a single model on a mixture of tasks but are allowed to evaluate performance using different parameter settings (checkpoints) for the model.
We can extend this approach by considering the case where the model is pre-trained on all tasks at once but is then fine-tuned on the individual supervised tasks.
This is the method used by the “MT-DNN” (Liu et al., 2015, 2019b), which achieved state-of-the-art performance on GLUE and other benchmarks when it was introduced.
We consider three variants of this approach:
In the first, we simply pre-train the model on an examples-proportional mixture with an artificial data set size limit of K=219𝐾superscript219K=2^{19} before fine-tuning it on each individual downstream task.
This helps us measure whether including the supervised tasks alongside the unsupervised objective during pre-training gives the model some beneficial early exposure to the downstream tasks.
We might also hope that mixing in many sources of supervision could help the pre-trained model obtain a more general set of “skills” (loosely speaking) before it is adapted to an individual task.
To measure this directly, we consider a second variant where we pre-train the model on the same examples-proportional mixture (with K=219𝐾superscript219K=2^{19}) except that we omit one of the downstream tasks from this pre-training mixture.
Then, we fine-tune the model on the task that was left out during pre-training.
We repeat this for each of the downstream tasks we consider.
We call this approach “leave-one-out” multi-task training.
This simulates the real-world setting where a pre-trained model is fine-tuned on a task it had not seen during pre-training.
Note that multi-task pre-training provides a diverse mixture of supervised tasks.
Since other fields (e.g. computer vision (Oquab et al., 2014; Jia et al., 2014; Huh et al., 2016; Yosinski et al., 2014)) use a supervised data set for pre-training, we were interested to see whether omitting the unsupervised task from the multi-task pre-training mixture still produced good results.
For our third variant we therefore pre-train on an examples-proportional mixture of all of the supervised tasks we consider with K=219𝐾superscript219K=2^{19}.
In all of these variants, we follow our standard procedure of pre-training for 219superscript2192^{19} steps before fine-tuning for 218superscript2182^{18} steps.

Training strategy
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Unsupervised pre-training + fine-tuning

83.2883.28\mathbf{83.28}
19.2419.24\mathbf{19.24}
80.8880.88\mathbf{80.88}
71.3671.36\mathbf{71.36}
26.9826.98\mathbf{26.98}
39.8239.8239.82
27.6527.6527.65

Multi-task training
81.4281.4281.42
19.2419.24\mathbf{19.24}
79.7879.7879.78
67.3067.3067.30
25.2125.2125.21
36.3036.3036.30
27.7627.7627.76

Multi-task pre-training + fine-tuning
83.1183.11\mathbf{83.11}
19.1219.12\mathbf{19.12}
80.2680.26\mathbf{80.26}
71.0371.03\mathbf{71.03}
27.0827.08\mathbf{27.08}
39.8039.8039.80
28.0728.07\mathbf{28.07}

Leave-one-out multi-task training
81.9881.9881.98
19.0519.0519.05
79.9779.9779.97
71.6871.68\mathbf{71.68}
26.9326.93\mathbf{26.93}
39.7939.7939.79
27.8727.87\mathbf{27.87}

Supervised multi-task pre-training
79.9379.9379.93
18.9618.9618.96
77.3877.3877.38
65.3665.3665.36
26.8126.8126.81
40.1340.13\mathbf{40.13}
28.0428.04\mathbf{28.04}

We compare the results of these approaches in Table 12.
For comparison, we also include results for our baseline (pre-train then fine-tune) and for standard multi-task learning (without fine-tuning) on an examples-proportional mixture with K=219𝐾superscript219K=2^{19}.
We find that fine-tuning after multi-task pre-training results in comparable performance to our baseline.
This suggests that using fine-tuning after multi-task learning can help mitigate some of the trade-offs between different mixing rates described in Section 3.5.2.
Interestingly, the performance of “leave-one-out” training was only slightly worse, suggesting that a model that was trained on a variety of tasks can still adapt to new tasks (i.e. multi-task pre-training might not result in a dramatic task interference).
Finally, supervised multi-task pre-training performed significantly worse in every case except for the translation tasks.
This could suggest that the translation tasks benefit less from (English) pre-training, whereas unsupervised pre-training is an important factor in the other tasks.

### 3.6 Scaling

The “bitter lesson” of machine learning research argues that general methods that can leverage additional computation ultimately win out against methods that rely on human expertise (Sutton, 2019; Hestness et al., 2017; Shazeer et al., 2017; Jozefowicz et al., 2016; Mahajan et al., 2018; Shazeer et al., 2018, 2017; Huang et al., 2018b; Keskar et al., 2019a).
Recent results suggest that this may hold true for transfer learning in NLP (Liu et al., 2019c; Radford et al., 2019; Yang et al., 2019; Lan et al., 2019), i.e. it has repeatedly been shown that scaling up produces improved performance compared to more carefully-engineered methods.
However, there are a variety of possible ways to scale, including using a bigger model, training the model for more steps, and ensembling.
In this section, we compare these different approaches by addressing the following premise: “You were just given 4×4\times more compute. How should you use it?”

We start with our baseline model, which has 220​M220M220\mathrm{M} parameters and is pre-trained and fine-tuned for 219superscript2192^{19} and 218superscript2182^{18} steps respectively.
The encoder and decoder are both sized similarly to “BERTBASE”.
To experiment with increased model size, we follow the guidelines of “BERTLARGE” Devlin et al. (2018) and use dff=4096subscript𝑑ff4096d_{\mathrm{ff}}=4096, dmodel=1024subscript𝑑model1024d_{\mathrm{model}}=1024, dkv=64subscript𝑑kv64d_{\mathrm{kv}}=64 and 161616-head attention mechanisms.
We then generate two variants with 161616 and 323232 layers each in the encoder and decoder, producing models with 2×2\times and 4×4\times as many parameters as our original model.
These two variants also have a roughly 2×2\times and 4×4\times the computational cost.
Using our baseline and these two larger models, we consider three ways of using 4×4\times as much computation: Training for 4×4\times as many steps, training for 2×2\times as many steps with the 2×2\times bigger model, and training the 4×4\times bigger model for the “baseline” number of training steps.
When we increase the training steps, we scale both the pre-train and fine-tune steps for simplicity.
Note that when increasing the number of pre-training steps, we are effectively including more pre-training data as C4 is so large that we do not complete one pass over the data even when training for 223superscript2232^{23} steps.

An alternative way for the model to see 4×4\times as much data is to increase the batch size by a factor of 444.
This can potentially result in faster training due to more efficient parallelization.
However, training with a 4×4\times larger batch size can yield a different outcome than training for 4×4\times as many steps (Shallue et al., 2018).
We include an additional experiment where we train our baseline model with a 4×4\times larger batch size to compare these two cases.

It is common practice on many of the benchmarks we consider to eke out additional performance by training and evaluating using an ensemble of models.
This provides an orthogonal way of using additional computation.
To compare other scaling methods to ensembling, we also measure the performance of an ensemble of 444 separately pre-trained and fine-tuned models.
We average the logits across the ensemble before feeding them into the output softmaxsoftmax\mathrm{softmax} nonlinearity to obtain an aggregate prediction.
Instead of pre-training 444 separate models, a cheaper alternative is to take a single pre-trained model and produce 444 separate fine-tuned versions.
While this does not use our entire 4×4\times computational budget, we also include this method to see if it produces competitive performance to the other scaling methods.

Scaling strategy
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Baseline

83.2883.2883.28
19.2419.2419.24
80.8880.8880.88
71.3671.3671.36
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

1×1\times size, 4×4\times training steps

85.3385.3385.33
19.3319.3319.33
82.4582.4582.45
74.7274.7274.72
27.0827.0827.08
40.6640.6640.66
27.9327.9327.93

1×1\times size, 4×4\times batch size

84.6084.6084.60
19.4219.4219.42
82.5282.5282.52
74.6474.6474.64
27.0727.0727.07
40.6040.6040.60
27.8427.8427.84

2×2\times size, 2×2\times training steps

86.1886.18\mathbf{86.18}
19.6619.6619.66
84.1884.18\mathbf{84.18}
77.1877.1877.18
27.5227.5227.52
41.0341.03\mathbf{41.03}
28.1928.1928.19

4×4\times size, 1×1\times training steps

85.9185.91\mathbf{85.91}
19.7319.7319.73
83.8683.86\mathbf{83.86}
78.0478.04\mathbf{78.04}
27.4727.4727.47
40.7140.7140.71
28.1028.1028.10

4×4\times ensembled

84.7784.7784.77
20.1020.10\mathbf{20.10}
83.0983.0983.09
71.7471.7471.74
28.0528.05\mathbf{28.05}
40.5340.5340.53
28.5728.57\mathbf{28.57}

4×4\times ensembled, fine-tune only

84.0584.0584.05
19.5719.5719.57
82.3682.3682.36
71.5571.5571.55
27.5527.5527.55
40.2240.2240.22
28.0928.0928.09

The performance achieved after applying these various scaling methods is shown in Table 13.
Unsurprisingly, increasing the training time and/or model size consistently improves the baseline.
There was no clear winner between training for 4×4\times as many steps or using a 4×4\times larger batch size, though both were beneficial.
In general, increasing the model size resulted in an additional bump in performance compared to solely increasing the training time or batch size.
We did not observe a large difference between training a 2×2\times bigger model for 2×2\times as long and training a 4×4\times bigger model on any of the tasks we studied.
This suggests that increasing the training time and increasing the model size can be complementary means of improving performance.
Our results also suggest that ensembling provides an orthogonal and effective means of improving performance through scale.
In some tasks (CNN/DM, WMT English to German, and WMT English to Romanian), ensembling 444 completely separately trained models significantly outperformed every other scaling approach.
Ensembling models that were pre-trained together but fine-tuned separately also gave a substantial performance increase over the baseline, which suggests a cheaper means of improving performance.
The only exception was SuperGLUE, where neither ensembling approach significantly improved over the baseline.

We note that different scaling methods have different trade-offs that are separate from their performance.
For example, using a larger model can make downstream fine-tuning and inference more expensive.
In contrast, the cost of pre-training a small model for longer is effectively amortized if it is applied to many downstream tasks.
Separately, we note that ensembling N𝑁N separate models has a similar cost to using a model that has an N×N\times higher computational cost.
As a result, some consideration for the eventual use of the model is important when choosing between scaling methods.

### 3.7 Putting It All Together

We now leverage the insights from our systematic study to determine how far we can push performance on popular NLP benchmarks.
We are also interested in exploring the current limits of transfer learning for NLP by training larger models on large amounts of data.
We start with our baseline training approach and make the following changes:

Objective

We swap out the i.i.d. denoising objective in our baseline for the span-corruption objective described in Section 3.3.4, which was loosely inspired by SpanBERT (Joshi et al., 2019).
Specifically, we use a mean span length of 333 and corrupt 15%percent1515\% of the original sequence.
We found that this objective produced marginally better performance (Table 7) while being slightly more computationally efficient due to shorter target sequence lengths.

Our baseline model uses a relatively small amount of pre-training (1/414\nicefrac{{1}}{{4}} as much as BERT (Devlin et al., 2018), 1/16116\nicefrac{{1}}{{16}} as much as XLNet (Yang et al., 2019), 1/64164\nicefrac{{1}}{{64}} as much as RoBERTa (Liu et al., 2019c), etc.).
Fortunately, C4 is big enough that we can train for substantially longer without repeating data (which can be detrimental, as shown in Section 3.4.2).
We found in Section 3.6 that additional pre-training can indeed be helpful, and that both increasing the batch size and increasing the number of training steps can confer this benefit.
We therefore pre-train our models for 111 million steps on a batch size of 211superscript2112^{11} sequences of length 512512512, corresponding to a total of about 111 trillion pre-training tokens (about 32×32\times as many as our baseline).
In Section 3.4.1, we showed that pre-training on the RealNews-like, WebText-like, and Wikipedia + TBC data sets outperformed pre-training on C4 on a few downstream tasks.
However, these data set variants are sufficiently small that they would be repeated hundreds of times over the course of pre-training on 111 trillion tokens.
Since we showed in Section 3.4.2 that this repetition could be harmful, we opted instead to continue using the C4 data set.

In Section 3.6 we also showed how scaling up the baseline model size improved performance.
However, using smaller models can be helpful in settings where limited computational resources are available for fine-tuning or inference.
Based on these factors, we train models with a wide range of sizes:

- •

Base. This is our baseline model, whose hyperparameters are described in Section 3.1.1. It has roughly 220220220 million parameters.

- •

Small. We consider a smaller model, which scales the baseline down by using dmodel=512subscript𝑑model512d_{\mathrm{model}}=512, dff=2,048subscript𝑑ff2048d_{\mathrm{ff}}=2{,}048, 888-headed attention, and only 666 layers each in the encoder and decoder. This variant has about 606060 million parameters.

- •

Large. Since our baseline uses a BERTBASE-sized encoder and decoder, we also consider a variant where the encoder and decoder are both similar in size and structure to BERTLARGE. Specifically, this variant uses dmodel=1,024subscript𝑑model1024d_{\mathrm{model}}=1{,}024, dff=4,096subscript𝑑ff4096d_{\mathrm{ff}}=4{,}096, dkv=64subscript𝑑kv64d_{\mathrm{kv}}=64, 161616-headed attention, and 242424 layers each in the encoder and decoder, resulting in around 770770770 million parameters.

- •

3B and 11B. To further explore what kind of performance is possible when using larger models, we consider two additional variants. In both cases, we use dmodel=1024subscript𝑑model1024d_{\mathrm{model}}=1024, a 242424 layer encoder and decoder, and dkv=128subscript𝑑kv128d_{\mathrm{kv}}=128. For the “3B” variant, we use dff=16,384subscript𝑑ff16384d_{\mathrm{ff}}=16{,}384 with 323232-headed attention, which results in around 2.82.82.8 billion parameters; for “11B” we use dff=65,536subscript𝑑ff65536d_{\mathrm{ff}}=65{,}536 with 128128128-headed attention producing a model with about 111111 billion parameters. We chose to scale up dffsubscript𝑑ffd_{\mathrm{ff}} specifically because modern accelerators (such as the TPUs we train our models on) are most efficient for large dense matrix multiplications like those in the Transformer’s feed-forward networks.

In Section 3.5.3, we showed that pre-training on a multi-task mixture of unsupervised and supervised tasks before fine-tuning worked as well as pre-training on the unsupervised task alone.
This is the approach advocated by the “MT-DNN” (Liu et al., 2015, 2019b).
It also has the practical benefit of being able to monitor “downstream” performance for the entire duration of training, rather than just during fine-tuning.
We therefore used multi-task pre-training in our final set of experiments.
We hypothesize that larger models trained for longer might benefit from a larger proportion of unlabeled data because they are more likely to overfit to smaller training data sets.
However, we also note that the results of Section 3.5.3 suggest that fine-tuning after multi-task pre-training can mitigate some of the issues that might arise from choosing a suboptimal proportion of unlabeled data.
Based on these ideas, we substitute the following artificial data set sizes for our unlabeled data before using standard example-proportional mixing (described in Section 3.5.2): 710,000710000710{,}000 for Small, 2,620,00026200002{,}620{,}000 for Base, 8,660,00086600008{,}660{,}000 for Large, 33,500,0003350000033{,}500{,}000 for 3B, and 133,000,000133000000133{,}000{,}000 for 11B.
For all model variants, we also capped the effective data set size of the WMT English to French and WMT English to German data sets to 1​M1M1\mathrm{M} examples during pre-training.

So far, when fine-tuning on GLUE and SuperGLUE, we have concatenated all of the data sets in each benchmark so that we only fine-tune models once for GLUE and once for SuperGLUE.
This approach makes our study logistically simpler, but we found that this sacrifices a small amount of performance on some tasks compared to fine-tuning on the task separately.
A potential issue with fine-tuning on individual tasks, which would otherwise be mitigated by training on all tasks at once, is that we might overfit quickly to low-resource tasks.
For example, our large batch size of 211superscript2112^{11} length-512512512 sequences would result in the entire data set appearing multiple times in each batch for many of the low-resource GLUE and SuperGLUE tasks.
We therefore use a smaller batch size of 888 length-512512512 sequences during fine-tuning for each GLUE and SuperGLUE task.
We also save checkpoints every 1,00010001{,}000 steps rather than every 5,00050005{,}000 steps to ensure we have access to the model’s parameters before it overfits.

All of our previous results were reported using greedy decoding.
For tasks with long output sequences, we found improved performance from using beam search (Sutskever et al., 2014).
Specifically, we use a beam width of 444 and a length penalty of α=0.6𝛼0.6\alpha=0.6 (Wu et al., 2016) for the WMT translation and CNN/DM summarization tasks.

Since this is our final set of experiments, we report results on the test set rather than the validation set.
For CNN/Daily Mail, we use the standard test set distributed with the data set.
For the WMT tasks, this corresponds to using newstest2014 for English-German, newstest2015 for English-French, and newstest2016 for English-Romanian.
For GLUE and SuperGLUE, we used the benchmark evaluation servers to compute official test set scores.151515http://gluebenchmark.com,161616http://super.gluebenchmark.com
For SQuAD, evaluating on the test set requires running inference on a benchmark server.
Unfortunately, the computational resources on this server are insufficient for obtaining predictions from our largest models.
As a result, we instead continue to report performance on the SQuAD validation set.
Fortunately, the model with the highest performance on the SQuAD test set also reported results on the validation set, so we can still compare to what is ostensibly the state-of-the-art.

Apart from those changes mentioned above, we use the same training procedure and hyperparameters as our baseline (AdaFactor optimizer, inverse square root learning rate schedule for pre-training, constant learning rate for fine-tuning, dropout regularization, vocabulary, etc.).
For reference, these details are described in Section 2.

GLUE
CoLA
SST-2
MRPC
MRPC
STS-B
STS-B

Model
Average
Matthew’s
Accuracy
F1
Accuracy
Pearson
Spearman

Previous best

89.489.489.4a

69.269.269.2b

97.197.197.1a

93.693.6\mathbf{93.6}b

91.591.5\mathbf{91.5}b

92.792.792.7b

92.392.392.3b

T5-Small
77.477.477.4
41.041.041.0
91.891.891.8
89.789.789.7
86.686.686.6
85.685.685.6
85.085.085.0

T5-Base
82.782.782.7
51.151.151.1
95.295.295.2
90.790.790.7
87.587.587.5
89.489.489.4
88.688.688.6

T5-Large
86.486.486.4
61.261.261.2
96.396.396.3
92.492.492.4
89.989.989.9
89.989.989.9
89.289.289.2

T5-3B
88.588.588.5
67.167.167.1
97.497.497.4
92.592.592.5
90.090.090.0
90.690.690.6
89.889.889.8

T5-11B
90.390.3\mathbf{90.3}
71.671.6\mathbf{71.6}
97.597.5\mathbf{97.5}
92.892.892.8
90.490.490.4
93.193.1\mathbf{93.1}
92.892.8\mathbf{92.8}

QQP
QQP
MNLI-m
MNLI-mm
QNLI
RTE
WNLI

Model
F1
Accuracy
Accuracy
Accuracy
Accuracy
Accuracy
Accuracy

Previous best

74.874.874.8c

90.790.7\mathbf{90.7}b

91.391.391.3a

91.091.091.0a

99.299.2\mathbf{99.2}a

89.289.289.2a

91.891.891.8a

T5-Small
70.070.070.0
88.088.088.0
82.482.482.4
82.382.382.3
90.390.390.3
69.969.969.9
69.269.269.2

T5-Base
72.672.672.6
89.489.489.4
87.187.187.1
86.286.286.2
93.793.793.7
80.180.180.1
78.878.878.8

T5-Large
73.973.973.9
89.989.989.9
89.989.989.9
89.689.689.6
94.894.894.8
87.287.287.2
85.685.685.6

T5-3B
74.474.474.4
89.789.789.7
91.491.491.4
91.291.291.2
96.396.396.3
91.191.191.1
89.789.789.7

T5-11B
75.175.1\mathbf{75.1}
90.690.690.6
92.292.2\mathbf{92.2}
91.991.9\mathbf{91.9}
96.996.996.9
92.892.8\mathbf{92.8}
94.594.5\mathbf{94.5}

SQuAD
SQuAD
SuperGLUE
BoolQ
CB
CB
COPA

Model
EM
F1
Average
Accuracy
F1
Accuracy
Accuracy

Previous best

90.190.190.1a

95.595.595.5a

84.684.684.6d

87.187.187.1d

90.590.590.5d

95.295.295.2d

90.690.690.6d

T5-Small
79.1079.1079.10
87.2487.2487.24
63.363.363.3
76.476.476.4
56.956.956.9
81.681.681.6
46.046.046.0

T5-Base
85.4485.4485.44
92.0892.0892.08
76.276.276.2
81.481.481.4
86.286.286.2
94.094.094.0
71.271.271.2

T5-Large
86.6686.6686.66
93.7993.7993.79
82.382.382.3
85.485.485.4
91.691.691.6
94.894.894.8
83.483.483.4

T5-3B
88.5388.5388.53
94.9594.9594.95
86.486.486.4
89.989.989.9
90.390.390.3
94.494.494.4
92.092.092.0

T5-11B
91.2691.26\mathbf{91.26}
96.2296.22\mathbf{96.22}
88.988.9\mathbf{88.9}
91.291.2\mathbf{91.2}
93.993.9\mathbf{93.9}
96.896.8\mathbf{96.8}
94.894.8\mathbf{94.8}

MultiRC
MultiRC
ReCoRD
ReCoRD
RTE
WiC
WSC

Model
F1a
EM
F1
Accuracy
Accuracy
Accuracy
Accuracy

Previous best

84.484.484.4d

52.552.552.5d

90.690.690.6d

90.090.090.0d

88.288.288.2d

69.969.969.9d

89.089.089.0d

T5-Small
69.369.369.3
26.326.326.3
56.356.356.3
55.455.455.4
73.373.373.3
66.966.966.9
70.570.570.5

T5-Base
79.779.779.7
43.143.143.1
75.075.075.0
74.274.274.2
81.581.581.5
68.368.368.3
80.880.880.8

T5-Large
83.383.383.3
50.750.750.7
86.886.886.8
85.985.985.9
87.887.887.8
69.369.369.3
86.386.386.3

T5-3B
86.886.886.8
58.358.358.3
91.291.291.2
90.490.490.4
90.790.790.7
72.172.172.1
90.490.490.4

T5-11B
88.188.1\mathbf{88.1}
63.363.3\mathbf{63.3}
94.194.1\mathbf{94.1}
93.493.4\mathbf{93.4}
92.592.5\mathbf{92.5}
76.976.9\mathbf{76.9}
93.893.8\mathbf{93.8}

WMT EnDe
WMT EnFr
WMT EnRo
CNN/DM
CNN/DM
CNN/DM

Model
BLEU
BLEU
BLEU
ROUGE-1
ROUGE-2
ROUGE-L

Previous best

33.833.8\mathbf{33.8}e

43.843.8\mathbf{43.8}e

38.538.5\mathbf{38.5}f

43.4743.4743.47g

20.3020.3020.30g

40.6340.6340.63g

T5-Small
26.726.726.7
36.036.036.0
26.826.826.8
41.1241.1241.12
19.5619.5619.56
38.3538.3538.35

T5-Base
30.930.930.9
41.241.241.2
28.028.028.0
42.0542.0542.05
20.3420.3420.34
39.4039.4039.40

T5-Large
32.032.032.0
41.541.541.5
28.128.128.1
42.5042.5042.50
20.6820.6820.68
39.7539.7539.75

T5-3B
31.831.831.8
42.642.642.6
28.228.228.2
42.7242.7242.72
21.0221.0221.02
39.9439.9439.94

T5-11B
32.132.132.1
43.443.443.4
28.128.128.1
43.5243.52\mathbf{43.52}
21.5521.55\mathbf{21.55}
40.6940.69\mathbf{40.69}

The results of this final set of experiments are shown in Table 14.
Overall, we achieved state-of-the-art performance on 181818 out of the 242424 tasks we consider.
As expected, our largest (111111 billion parameter) model performed best among our model size variants across all tasks.
Our T5-3B model variant did beat the previous state of the art in a few tasks, but scaling the model size to 111111 billion parameters was the most important ingredient for achieving our best performance.
We now analyze the results for each individual benchmark.

We achieved a state-of-the-art average GLUE score of 90.390.390.3.
Notably, our performance was substantially better than the previous state-of-the-art for the natural language inference tasks MNLI, RTE, and WNLI.
RTE and WNLI are two of the tasks where machine performance has historically lagged behind human performance, which is 93.693.693.6 and 95.995.995.9 respectively (Wang et al., 2018).
In terms of parameter count, our 11B model variant is the largest model that has been submitted to the GLUE benchmark.
However, most of the best-scoring submissions use a large amount of ensembling and computation to produce predictions.
For example, the best-performing variant of ALBERT (Lan et al., 2019) uses a model similar in size and architecture to our 3B variant (though it has dramatically fewer parameters due to clever parameter sharing).
To produce its impressive performance on GLUE, the ALBERT authors ensembled “from 6 to 17” models depending on the task.
This likely results in it being more computationally expensive to produce predictions with the ALBERT ensemble than it is with T5-11B.

For SQuAD, we outperformed the previous state-of-the-art (ALBERT (Lan et al., 2019)) by over one point on the Exact Match score.
SQuAD is a long-standing benchmark that was created over three years ago, and most recent improvements have only increased the state-of-the-art by a fraction of a percentage point.
We note that when results are reported on the test set, they are typically based on an ensemble of models and/or leverage external data sets (e.g. TriviaQA (Joshi et al., 2017) or NewsQA (Trischler et al., 2016)) to augment the small SQuAD training set.
Human performance on SQuAD is estimated at 82.3082.3082.30 and 91.2291.2291.22 for the Exact Match and F1 metric respectively (Rajpurkar et al., 2016), so it is not clear if further improvements on this benchmark are meaningful.

For SuperGLUE, we improved upon the state-of-the-art by a large margin (from an average score of 84.684.684.6 (Liu et al., 2019c) to 88.988.988.9).
SuperGLUE was designed to include tasks that were “beyond the scope of current state-of-the-art systems, but solvable by most college-educated English speakers” (Wang et al., 2019b).
We nearly match the human performance of 89.889.889.8 (Wang et al., 2019b).
Interestingly, on the reading comprehension tasks (MultiRC and ReCoRD) we exceed human performance by a large margin, suggesting the evaluation metrics used for these tasks may be biased towards machine-made predictions.
On the other hand, humans achieve 100%percent100100\% accuracy on both COPA and WSC, which is significantly better than our model’s performance.
This suggests that there remain linguistic tasks that are hard for our model to perfect, particularly in the low-resource setting.

We did not achieve state-of-the-art performance on any of the WMT translation tasks.
This may be in part due to our use of an English-only unlabeled data set.
We also note that most of the best results on these tasks use backtranslation (Edunov et al., 2018; Lample and Conneau, 2019), which is a sophisticated data augmentation scheme.
The state of the art on the low-resource English to Romanian benchmark also uses additional forms of cross-lingual unsupervised training (Lample and Conneau, 2019).
Our results suggest that scale and English-language pre-training may be insufficient to match the performance of these more sophisticated methods.
On a more specific note, the best results on English to German newstest2014 set use the much larger training set from WMT 2018 (Edunov et al., 2018), making direct comparison to our results difficult.

Finally, on CNN/Daily Mail we attain state-of-the-art performance, though only by a significant amount on the ROUGE-2-F score.
It has been shown that improvements to the ROUGE score do not necessarily correspond to more coherent summaries (Paulus et al., 2017).
Furthermore, while CNN/Daily Mail is posed as an abstractive summarization benchmark, purely extractive approaches have been shown to work well (Liu, 2019).
It has also been argued that generative models trained with maximum likelihood are prone to producing repetitive summaries (See et al., 2017).
Despite these potential issues, we find that our models do generate coherent and largely correct summaries.
We provide some non-cherry-picked validation set examples in Section C.

To achieve its strong results, T5 combines insights from our experimental study with unprecedented scale.
Note that in Section 3.6 we found that scaling up the pre-training amount or size of our baseline model produced substantial gains.
Given this, we were interested to measure how much the “non-scaling” changes we introduced into T5 contributed to its strong performance.
We therefore carried out a final experiment where we compared the following three configurations:
First, the standard baseline model, which was pre-trained on 235≈34​Bsuperscript23534B2^{35}\approx 34\mathrm{B} tokens;
second, the baseline trained instead for about 1 trillion tokens (i.e. the same amount of pre-training used for T5), which we refer to as “baseline-1T”;
and third, T5-Base.
Note that the differences between baseline-1T and T5-Base comprise the “non-scaling” changes we made when designing T5.
As such, comparing the performance of these two models gives us a concrete measurement of the impact of the insights from our systematic study.

The performance of these three model configurations is shown in Table 15.
Consistent with the findings in Section 3.6, we find that additional pre-training improves performance over the baseline.
Nevertheless, T5-Base substantially outperforms baseline-1T on all downstream tasks.
This suggests that scale is not the only factor that contributes to T5’s success.
We hypothesize that the larger models benefit not only from their increased size but also from these non-scaling factors.

Model
GLUE
CNNDM
SQuAD
SGLUE
EnDe
EnFr
EnRo

★★\bigstar\,Baseline

83.2883.2883.28
19.2419.2419.24
80.8880.8880.88
71.3671.3671.36
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

Baseline-1T
84.8084.8084.80
19.6219.6219.62
83.0183.0183.01
73.9073.9073.90
27.4627.4627.46
40.3040.3040.30
28.3428.3428.34

T5-Base
85.9785.97\mathbf{85.97}
20.9020.90\mathbf{20.90}
85.4485.44\mathbf{85.44}
75.6475.64\mathbf{75.64}
28.3728.37\mathbf{28.37}
41.3741.37\mathbf{41.37}
28.9828.98\mathbf{28.98}

## 4 Reflection

Having completed our systematic study, we wrap up by first recapping some of our most significant findings.
Our results provide some high-level perspective on which avenues of research might be more or less promising.
To conclude, we outline some topics we think might provide effective approaches for further progressing the field.

### 4.1 Takeaways

Text-to-text

Our text-to-text framework provides a simple way to train a single model on a wide variety of text tasks using the same loss function and decoding procedure.
We showed how this approach can be successfully applied to generative tasks like abstractive summarization, classification tasks like natural language inference, and even regression tasks like STS-B.
In spite of its simplicity, we found the text-to-text framework obtained comparable performance to task-specific architectures and ultimately produced state-of-the-art results when combined with scale.

While some work on transfer learning for NLP has considered architectural variants of the Transformer, we found the original encoder-decoder form worked best in our text-to-text framework.
Though an encoder-decoder model uses twice as many parameters as “encoder-only” (e.g. BERT) or “decoder-only” (language model) architectures, it has a similar computational cost.
We also showed that sharing the parameters in the encoder and decoder did not result in a substantial performance drop while halving the total parameter count.

Overall, we found that most “denoising” objectives, which train the model to reconstruct randomly corrupted text, performed similarly in the text-to-text setup.
As a result, we suggest using objectives that produce short target sequences so that unsupervised pre-training is more computationally efficient.

We introduced the “Colossal Clean Crawled Corpus” (C4), which comprises heuristically-cleaned text from the Common Crawl web dump.
When comparing C4 to data sets that use additional filtering, we found that training on in-domain unlabeled data could boost performance in a few downstream tasks.
However, constraining to a single domain typically results in a smaller data set.
We separately showed that performance can degrade when an unlabeled data set is small enough that it is repeated many times over the course of pre-training.
This motivates the use of a large and diverse data set like C4 for generic language understanding tasks.

We found that the basic approach of updating all of a pre-trained model’s parameters during fine-tuning outperformed methods that are designed to update fewer parameters, although updating all parameters is most expensive.
We also experimented with various approaches for training the model on multiple tasks at once, which in our text-to-text setting simply corresponds to mixing examples from different data sets when constructing batches.
The primary concern in multi-task learning is setting the proportion of each task to train on.
We ultimately did not find a strategy for setting mixing proportions that matched the performance of the basic approach of unsupervised pre-training followed by supervised fine-tuning.
However, we found that fine-tuning after pre-training on a mixture of tasks produced comparable performance to unsupervised pre-training.

We compared various strategies for taking advantage of additional compute, including training the model on more data, training a larger model, and using an ensemble of models.
We found each approach conferred a significant boost in performance, though training a smaller model on more data was often outperformed by training a larger model for fewer steps.
We also showed an ensemble of models can provide substantially better results than a single model, which provides an orthogonal means of leveraging additional computation.
Ensembling models that were fine-tuned from the same base pre-trained model performed worse than pre-training and fine-tuning all models completely separately, though fine-tune-only ensembling still substantially outperformed a single model.

We combined our above insights and trained substantially larger models (up to 111111 billion parameters) to achieve state-of-the-art results across many of the benchmarks we considered.
For unsupervised training, we extracted text from our C4 data set and applied a denoising objective that corrupts contiguous spans of tokens.
We pre-trained on a multi-task mixture before fine-tuning on individual tasks.
Overall, our models were trained on over 111 trillion tokens.
In the interest of facilitating the replication, extension, and application of our results, we release our code, the C4 data set, and pre-trained model weights for each T5 variant.\@footnotemark

### 4.2 Outlook

The inconvenience of large models

An unsurprising but important result from our study is that larger models tend to perform better.
The fact that the hardware used for running these models is continually getting cheaper and more powerful suggests that scaling up may continue to be a promising way to achieve better performance (Sutton, 2019).
However, it will always be the case that there are applications and scenarios where using a smaller or less expensive model is helpful, for example when performing client-side inference or federated learning (Konečnỳ et al., 2015, 2016).
Relatedly, one beneficial use of transfer learning is the possibility of attaining good performance on low-resource tasks.
Low-resource tasks often occur (by definition) in settings where one lacks the assets to label more data.
It follows that low-resource applications often also have limited access to computational resources which can incur additional costs.
As a result, we advocate for research on methods that achieve stronger performance with cheaper models so that transfer learning can be applied where it will have the most impact.
Some current work along these lines include distillation (Hinton et al., 2015; Sanh et al., 2019; Jiao et al., 2019), parameter sharing (Lan et al., 2019), and conditional computation (Shazeer et al., 2017).

Recall that one of the goals of pre-training is (loosely speaking) to provide the model with general-purpose “knowledge” that improves its performance on downstream tasks.
The method we use in this work, which is currently common practice, is to train the model to denoise corrupted spans of text.
We suspect that this simplistic technique may not be a very efficient way to teach the model general-purpose knowledge.
More concretely, it would be useful to be able to attain good fine-tuning performance without needing to train our models on 111 trillion tokens of text first.
Some concurrent work along these lines improves efficiency by pre-training a model to distinguish between real and machine-generated text (Clark et al., 2020).

We observed that pre-training on unlabeled in-domain data can improve performance on downstream tasks (Section 3.4).
This finding mostly relies on basic observations like the fact that SQuAD was created using data from Wikipedia.
It would be useful to formulate a more rigorous notion of the “similarity” between the pre-training and downstream tasks, so that we could make more principled choices about what source of unlabeled data to use.
There is some early empirical work along these lines in the field of computer vision (Huh et al., 2016; Kornblith et al., 2018; He et al., 2018).
A better notion of the relatedness of tasks could also help choose supervised pre-training tasks, which has been shown to be helpful for the GLUE benchmark (Phang et al., 2018).

We were disappointed to find that English-only pre-training did not achieve state-of-the-art results on the translation tasks we studied.
We also are interested in avoiding the logistical difficulty of needing to specify which languages a vocabulary can encode ahead of time.
To address these issues, we are interested in further investigating language-agnostic models, i.e. models that can perform a given NLP task with good performance regardless of the text’s language.
This is an especially pertinent issue given that English is not the native language for the majority of the world’s population.

The motivation for this paper was the flurry of recent work on transfer learning for NLP.
Before we began this work, these advances had already enabled breakthroughs in settings where learning-based methods had not yet been shown to be effective.
We are happy to be able to continue this trend, for example by nearly matching human-level performance on the SuperGLUE benchmark, a task specifically designed to be difficult for modern transfer-learning pipelines.
Our results stem from the combination of a straightforward and unified text-to-text framework, our new C4 data set, and insights from our systematic study.
Additionally, we provided an empirical overview of the field and a perspective on where it stands.
We are excited to see continued work using transfer learning towards the goal of general language understanding.

Acknowledgments

We thank Grady Simon, Noah Fiedel, Samuel R. Bowman, Augustus Odena, Daphne Ippolito, Noah Constant, Orhan Firat, Ankur Bapna, and Sebastian Ruder for their comments on this manuscript; Zak Stone and the TFRC team for their support; Austin Tarango for his guidance on data set creation; Melvin Johnson, Dima Lepikhin, Katrin Tomanek, Jeff Klingner, and Naveen Arivazhagan for insight into multi-task machine translation; Neil Houlsby for comments on adapter layers; Olga Wichowska, Ola Spyra, Michael Banfield, Yi Lin, and Frank Chen for assistance with infrastructure; Etienne Pot, Ryan Sepassi, and Pierre Ruyssen for collaboration on TensorFlow Datasets; Rohan Anil for help with our download pipeline for Common Crawl; Robby Neale and Taku Kudo for their work on SentencePiece; Jeffrey Li for pointing out missing details about the creation of C4; and many other members of the Google Brain team for their discussion and insight.

## A Contributions

Colin designed the scope of this project and wrote this paper, ran all the experiments in Sections 3.1, 3.2, 3.3, 3.4, 3.5 and 3.6, and contributed a large portion of our codebase.
Noam contributed many of the ideas, including the text-to-text framework, unsupervised objectives, and data set mixing strategies; implemented our base Transformer model and its architectural variants; and ran the experiments in Section 3.7.
Adam oversaw all engineering aspects for this project, created the C4 data set, implemented our data set pipeline, and added various benchmark data sets.
Katherine coordinated experiments, wrote and updated documentation, ran experiments to help design our baseline, and contributed to many parts of our codebase.
Sharan contributed some of the required data sets and preprocessors, and ran assorted preliminary experiments, in addition to co-leading the open-sourcing of our codebase.
Michael owned all aspects of the Winograd data sets, ingested many of the data sets we used, contributed various improvements and fixes to our infrastructure, and ran some preliminary experiments.
Yanqi ran experiments and implemented methods to help settle on a reasonable baseline and helped with the final fine-tuning of the models in Section 3.7.
Wei also helped with final fine-tuning and improved some of our preprocessors.
Peter prototyped an early version of the pre-training data set and resolved issues pertaining to the SQuAD and CNN/DM tasks.
All authors helped set the scope and research direction we followed in this work.

## B Converting WNLI to Our Text-to-Text Format

Note that as discussed in Section 2.4, we do not train on any of the data from WNLI.
Instead, when evaluating on the WNLI test set (for the results in Section 3.7), we convert the WNLI test set to the “referent noun prediction” text-to-text format so that we can evaluate using a model trained on WSC and DPR.
Our WNLI preprocessor is inspired by the one proposed by He et al. (2019).
Recall that examples from WNLI consist of a premise, a hypothesis, and a label that indicates whether the hypothesis is True or False.
Using the example from Section 2.4, the hypothesis would be “The city councilmen refused the demonstrators a permit because they
feared violence.” with the premise “The demonstrators feared violence.” and the label False.
We first find the location of all pronouns in the premise (“they” in our example).
Then, we find the maximum number of words that precede or follow each pronoun that are a substring in the hypothesis (“feared violence” in our example), ignoring case and punctuation.
When the premise contains multiple candidate pronouns, we choose the pronoun that is preceded or followed by the largest substring of the hypothesis.
We then highlight the pronoun in the premise by surrounding it with asterisks.
For the candidate noun (which is compared to our model’s prediction to obtain a True or False label), we remove the matching substring from the hypothesis and optionally make it non-possessive (resulting in “the demonstrators”).

## C Example Predictions on CNN/Daily Mail

To show that our model is generating fluent summaries, we include a few example decodes from our best model (T5-11B) on the validation set along with the ground-truth summaries.
These examples selected at random and were not cherry-picked.

- 1.

Ground-truth: leopard gave up after spiky creature refused to back down in fight in kruger national park, south africa . wildlife enthusiast lisl moolman, 41, caught the bizarre battle while out on the road and armed with her camera . the leopard and porcupine tumbled out of the bushes and began to fight by roadside - watched by ms moolman .

Prediction: leopard tried to make lunch out of a plucky porcupine in kruger national park, south africa . but the predator was put firmly in its place after the spiky creature refused to back down during a fight . wildlife enthusiast lisl moolman, 41, caught the bizarre battle while out on the road and armed with her camera .

- 2.

Ground-truth: researchers say homes are a hotbed of undiscovered species . study of 40 north carolina homes found 8,000 bacterial and archaeal taxa . 11 houses in california revealed a great variety of fungus among us .

Prediction: researchers estimate that the indoor biome makes up about 0.5 percent of ice-free land, or about 247,000 square miles, almost the size of texas . a study of 40 houses in north carolina used genomic technologies to document more than 8,000 bacterial and archaeal taxa . another study of 50 houses in the same state described more than 750 types of arthropods, including insects and spiders . an examination of 11 houses in california found a great variety of fungus among us .

- 3.

Ground-truth: executions have been temporarily halted in the state after the drug used for lethal injections appeared faulty on monday night . kelly gissendaner had her death delayed for a second time when it was found that the lethal pentobarbital dose appeared cloudy . state officials dithered over the decision , changing their mind three times before deciding not to take the risk that it wouldn’t work . it is ’essential that executions are carried out in a constitutional manner,’ said georgia attorney general sam olens .

Prediction: the execution of kelly gissendaner, the only woman on georgia’s death row, was delayed for a second time on monday . the execution team found that the lethal pentobarbital dose appeared cloudy . the cloudy drug bolstered death penalty opponents, who have been vocal in their opposition after three botched executions in other parts of the country .

- 4.

Ground-truth: dani alves was not selected for the brazil squad to face france and chile . barcelona defender was filmed serving up hot dogs to people on saturday . this week alves released a charity single with former team-mat jose pinto . alves looks set to leave barcelona on a free transfer this summer . reports in spanish press claim he will sign for paris saint-germain .

Prediction: dani alves was not selected for brazil’s friendlies against france and chile . the barcelona right back has released a charity single with jose pinto . alves was filmed serving up snacks from behind the counter of a hot dog bar on saturday . alves is yet to be offered a new deal at the nou camp . click here for all the latest barcelona news .

## D Preprocessed Examples

In this section, we provide examples of our preprocessing for each of the data sets we consider.

### D.1 CoLA

Original input:

Sentence:

John made Bill master of himself.

cola sentence: John made Bill master of himself.

1

acceptable

### D.2 RTE

Original input:

Sentence 1:

A smaller proportion of Yugoslavia’s Italians were settled in Slovenia (at the 1991 national census, some 3000 inhabitants of Slovenia declared themselves as ethnic Italians).

Slovenia has 3,000 inhabitants.

rte sentence1: A smaller proportion of Yugoslavia’s Italians were settled in Slovenia (at the 1991 national census, some 3000 inhabitants of Slovenia declared themselves as ethnic Italians). sentence2: Slovenia has 3,000 inhabitants.

1

not_entailment

### D.3 MNLI

Original input:

Hypothesis:

The St. Louis Cardinals have always won.

yeah well losing is i mean i’m i’m originally from Saint Louis and Saint Louis Cardinals when they were there were uh a mostly a losing team but

mnli hypothesis: The St. Louis Cardinals have always won. premise: yeah well losing is i mean i’m i’m originally from Saint Louis and Saint Louis Cardinals when they were there were uh a mostly a losing team but

2

contradiction

### D.4 MRPC

Original input:

Sentence 1:

We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .

Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " .

mrpc sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " .

1

equivalent

### D.5 QNLI

Original input:

Question:

Where did Jebe die?

Genghis Khan recalled Subutai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand.

qnli question: Where did Jebe die? sentence: Genghis Khan recalled Subutai back to Mongolia soon afterwards, and Jebe died on the road back to Samarkand.

0

entailment

### D.6 QQP

Original input:

Question 1:

What attributes would have made you highly desirable in ancient Rome?

How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?

qqp question1: What attributes would have made you highly desirable in ancient Rome? question2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?

0

not_duplicate

### D.7 SST2

Original input:

Sentence:

it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight .

sst2 sentence: it confirms fincher ’s status as a film maker who artfully bends technical know-how to the service of psychological insight .

1

positive

### D.8 STSB

Original input:

Sentence 1:

Representatives for Puretunes could not immediately be reached for comment Wednesday.

Puretunes representatives could not be located Thursday to comment on the suit.

stsb sentence1: Representatives for Puretunes could not immediately be reached for comment Wednesday. sentence2: Puretunes representatives could not be located Thursday to comment on the suit.

3.25

3.2

### D.9 CB

Original input:

Hypothesis:

Valence was helping

Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping?

cb hypothesis: Valence was helping premise: Valence the void-brain, Valence the virtuous valet. Why couldn’t the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping?

1

contradiction

### D.10 COPA

Original input:

Question:

effect

Political violence broke out in the nation.

Many citizens relocated to the capitol.

Many citizens took refuge in other territories.

copa choice1: Many citizens relocated to the capitol. choice2: Many citizens took refuge in other territories. premise: Political violence broke out in the nation. question: effect

1

True

### D.11 MultiRC

Original input:

Answer:

There was only pie to eat, rather than traditional breakfast foods

<b>Sent 1: </b>Once upon a time, there was a squirrel named Joey.<br><b>Sent 2: </b>Joey loved to go outside and play with his cousin Jimmy.<br><b>Sent 3: </b>Joey and Jimmy played silly games together, and were always laughing.<br><b>Sent 4: </b>One day, Joey and Jimmy went swimming together at their Aunt Julie’s pond.<br><b>Sent 5: </b>Joey woke up early in the morning to eat some food before they left.<br><b>Sent 6: </b>He couldn’t find anything to eat except for pie!<br><b>Sent 7: </b>Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.<br><b>Sent 8: </b>After he ate, he and Jimmy went to the pond.<br><b>Sent 9: </b>On their way there they saw their friend Jack Rabbit.<br><b>Sent 10: </b>They dove into the water and swam for several hours.<br><b>Sent 11: </b>The sun was out, but the breeze was cold.<br><b>Sent 12: </b>Joey and Jimmy got out of the water and started walking home.<br><b>Sent 13: </b>Their fur was wet, and the breeze chilled them.<br><b>Sent 14: </b>When they got home, they dried off, and Jimmy put on his favorite purple shirt.<br><b>Sent 15: </b>Joey put on a blue shirt with red and green dots.<br><b>Sent 16: </b>The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.<br>

Why was Joey surprised the morning he woke up for breakfast?

multirc question: Why was Joey surprised the morning he woke up for breakfast? answer: There was only pie to eat, rather than traditional breakfast foods paragraph: <b>Sent 1: </b>Once upon a time, there was a squirrel named Joey.<br><b>Sent 2: </b>Joey loved to go outside and play with his cousin Jimmy.<br><b>Sent 3: </b>Joey and Jimmy played silly games together, and were always laughing.<br><b>Sent 4: </b>One day, Joey and Jimmy went swimming together at their Aunt Julie’s pond.<br><b>Sent 5: </b>Joey woke up early in the morning to eat some food before they left.<br><b>Sent 6: </b>He couldn’t find anything to eat except for pie!<br><b>Sent 7: </b>Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast.<br><b>Sent 8: </b>After he ate, he and Jimmy went to the pond.<br><b>Sent 9: </b>On their way there they saw their friend Jack Rabbit.<br><b>Sent 10: </b>They dove into the water and swam for several hours.<br><b>Sent 11: </b>The sun was out, but the breeze was cold.<br><b>Sent 12: </b>Joey and Jimmy got out of the water and started walking home.<br><b>Sent 13: </b>Their fur was wet, and the breeze chilled them.<br><b>Sent 14: </b>When they got home, they dried off, and Jimmy put on his favorite purple shirt.<br><b>Sent 15: </b>Joey put on a blue shirt with red and green dots.<br><b>Sent 16: </b>The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed.<br>

1

True

### D.12 WiC

Original input:

POS:

N

It was the deliberation of his act that was insulting .

The deliberations of the jury .

deliberation

wic pos: N sentence1: It was the deliberation of his act that was insulting . sentence2: The deliberations of the jury . word: deliberation

0

False

### D.13 WSC and DPR

Original input:

Span 2 text:

it

stable

20

1

The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made it pleasant and airy.

wsc: The stable was very roomy, with four good stalls; a large swinging window opened into the yard , which made *it* pleasant and airy.

1

stable

### D.14 CNN/Daily Mail

Original input:

marouane fellaini and adnan januzaj continue to show the world they are not just teammates but also best mates. the manchester united and belgium duo both posted pictures of themselves out at a restaurant on monday night ahead of their game against newcastle on wednesday . januzaj poses in the middle of fellaini and a friend looking like somebody who failed to receive the memo about it being a jackson 5 themed night. premier league duo adnan januzaj and marouane fellaini pose with a friend on the dance floor . manchester united and belgium duo fellaini and januzaj are good friends both on and off the pitch . manchester united ace fellaini runs over to the bench to celebrate his goal against qpr with friend januzaj . the disco effect in the background adds to the theory, but januzaj doesn’t seem to mind as they later pose on the dance floor with other friends. united haven’t had too many reasons to have a song and dance this season so it seems they may be hitting the discotheques as another form of release. however, victory against newcastle on wednesday would leave manager louis van gaal at least tapping his toes as they continue to fight for a champions league spot this season. januzaj and robin van persie join fellaini in celebrating in front of the manchester united fans at west brom . januzaj receives some words of wisdom from manchester united’s dutch manager louis van gaal . januzaj and fellaini are joined by some friends as they take to the dance floor ahead of the newcastle game .

summarize: marouane fellaini and adnan januzaj continue to show the world they are not just teammates but also best mates. the manchester united and belgium duo both posted pictures of themselves out at a restaurant on monday night ahead of their game against newcastle on wednesday . januzaj poses in the middle of fellaini and a friend looking like somebody who failed to receive the memo about it being a jackson 5 themed night. premier league duo adnan januzaj and marouane fellaini pose with a friend on the dance floor . manchester united and belgium duo fellaini and januzaj are good friends both on and off the pitch . manchester united ace fellaini runs over to the bench to celebrate his goal against qpr with friend januzaj . the disco effect in the background adds to the theory, but januzaj doesn’t seem to mind as they later pose on the dance floor with other friends. united haven’t had too many reasons to have a song and dance this season so it seems they may be hitting the discotheques as another form of release. however, victory against newcastle on wednesday would leave manager louis van gaal at least tapping his toes as they continue to fight for a champions league spot this season. januzaj and robin van persie join fellaini in celebrating in front of the manchester united fans at west brom . januzaj receives some words of wisdom from manchester united’s dutch manager louis van gaal . januzaj and fellaini are joined by some friends as they take to the dance floor ahead of the newcastle game .

the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .

### D.15 SQuAD

Original input:

Question:

What does increased oxygen concentrations in the patient’s lungs displace?

Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O
2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O
2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O
2 as soon as possible is part of the treatment.

question: What does increased oxygen concentrations in the patient’s lungs displace? context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O
2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O
2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O
2 as soon as possible is part of the treatment.

carbon monoxide

### D.16 WMT English to German

Original input:

"Luigi often said to me that he never wanted the brothers to end up in court," she wrote.

translate English to German: "Luigi often said to me that he never wanted the brothers to end up in court," she wrote.

"Luigi sagte oft zu mir, dass er nie wollte, dass die Brüder vor Gericht landen", schrieb sie.

### D.17 WMT English to French

Original input:

This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of countless generations of stars: the oldest stars are seen as blue dots, while more difficult to identify are the pink-coloured "new-borns" in the star delivery room.

translate English to French: This image section from an infrared recording by the Spitzer telescope shows a "family portrait" of countless generations of stars: the oldest stars are seen as blue dots, while more difficult to identify are the pink-coloured "new-borns" in the star delivery room.

Ce détail d’une photographie infrarouge prise par le télescope Spitzer montre un "portrait de famille" des innombrables générations d’étoiles: les plus vieilles étoiles sont en bleu et les points roses, plus difficiles à identifier, sont les "nouveau-nés" dans la salle d’accouchement de l’univers.

### D.18 WMT English to Romanian

Original input:

Taco Bell said it plans to add 2,000 locations in the US by 2022.

translate English to Romanian: Taco Bell said it plans to add 2,000 locations in the US by 2022.

Taco Bell a afirmat că, până în 2022, inten\textcommabelowtionează să deschidă 2000 de restaurante în SUA.

## E Scores on Every Task for All Experiments

The following table lists the scores achieved on every task in the experiments described in Sections 3.2, 3.3, 3.4, 3.5 and 3.6.

GLUE

SuperGLUE
WMT

Score
CoLA
SST-2
MRPC
MRPC
STSB
STSB
QQP
QQP

MNLIm

MNLImm

QNLI
RTE
CNN/DM
SQuAD
Score
BoolQ
CB
CB
COPA
MultiRC
MultiRC
ReCoRD
ReCoRD
RTE
WiC
WSC
EnDe
EnFr
EnRo

Table
Experiment
Average
MCC
Acc
F1
Acc
PCC
SCC
F1
Acc
Acc
Acc
Acc
Acc
R-1-F
R-2-F
R-L-F
EM
F1
Average
Acc
F1
Acc
Acc
F1
EM
F1
EM
Acc
Acc
Acc
BLEU
BLEU
BLEU

1

★★\bigstar\,Baseline average

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

1
Baseline standard deviation
0.2350.2350.235
1.1111.1111.111
0.5690.5690.569
0.7290.7290.729
1.0191.0191.019
0.3740.3740.374
0.4180.4180.418
0.1080.1080.108
0.0700.0700.070
0.2910.2910.291
0.2310.2310.231
0.3610.3610.361
1.3931.3931.393
0.0650.0650.065
0.0650.0650.065
0.0580.0580.058
0.3430.3430.343
0.2260.2260.226
0.4160.4160.416
0.3650.3650.365
3.2373.2373.237
2.5602.5602.560
2.7412.7412.741
0.7160.7160.716
1.0111.0111.011
0.3700.3700.370
0.3790.3790.379
1.2281.2281.228
0.8500.8500.850
2.0292.0292.029
0.1120.1120.112
0.0900.0900.090
0.1080.1080.108

1
No pre-training
66.2266.2266.22
12.2912.2912.29
80.6280.6280.62
81.4281.4281.42
73.0473.0473.04
72.5872.5872.58
72.9772.9772.97
81.9481.9481.94
86.6286.6286.62
68.0268.0268.02
67.9867.9867.98
75.6975.6975.69
58.8458.8458.84
39.1939.1939.19
17.6017.6017.60
36.6936.6936.69
50.3150.3150.31
61.9761.9761.97
53.0453.0453.04
65.3865.3865.38
71.6171.6171.61
76.7976.7976.79
62.0062.0062.00
59.1059.1059.10
0.840.840.84
20.3320.3320.33
17.9517.9517.95
54.1554.1554.15
54.0854.0854.08
65.3865.3865.38
25.8625.8625.86
39.7739.7739.77
24.0424.0424.04

2

★★\bigstar\,Enc/dec, denoising

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

2
Enc/dec, shared, denoising
82.8182.8182.81
55.2455.2455.24
91.8691.8691.86
91.5891.5891.58
88.2488.2488.24
87.4387.4387.43
87.5887.5887.58
88.6988.6988.69
91.6091.6091.60
83.8883.8883.88
84.0184.0184.01
90.2390.2390.23
73.6573.6573.65
41.1141.1141.11
18.7818.7818.78
38.4838.4838.48
80.6380.6380.63
88.4988.4988.49
70.7370.7370.73
77.1377.1377.13
95.0495.0495.04
96.4396.4396.43
65.0065.0065.00
66.1666.1666.16
22.9822.9822.98
68.9568.9568.95
68.0968.0968.09
70.7670.7670.76
68.1868.1868.18
75.9675.9675.96
26.7226.7226.72
39.0339.0339.03
27.4627.4627.46

2
Enc/dec, 6 layers, denoising
80.8880.8880.88
46.2646.2646.26
92.0992.0992.09
91.5191.5191.51
87.9987.9987.99
87.0187.0187.01
86.7686.7686.76
87.9387.9387.93
90.9790.9790.97
82.2082.2082.20
82.4182.4182.41
88.8388.8388.83
71.4871.4871.48
40.8340.8340.83
18.9718.9718.97
38.3138.3138.31
77.5977.5977.59
86.0786.0786.07
68.4268.4268.42
73.7973.7973.79
91.7091.7091.70
92.8692.8692.86
67.0067.0067.00
61.0261.0261.02
19.6219.6219.62
61.2661.2661.26
60.3360.3360.33
72.2072.2072.20
65.9965.9965.99
75.0075.0075.00
26.3826.3826.38
38.4038.4038.40
26.9526.9526.95

2
Language model, denoising
74.7074.7074.70
24.5024.5024.50
90.6090.6090.60
86.0886.0886.08
78.9278.9278.92
85.2285.2285.22
85.4285.4285.42
85.4085.4085.40
88.9988.9988.99
76.7276.7276.72
77.0577.0577.05
86.0286.0286.02
64.6264.6264.62
39.4939.4939.49
17.9317.9317.93
36.9136.9136.91
61.1461.1461.14
71.3771.3771.37
55.0255.0255.02
65.4765.4765.47
60.0860.0860.08
71.4371.4371.43
58.0058.0058.00
43.0343.0343.03
2.942.942.94
53.3553.3553.35
52.3152.3152.31
53.0753.0753.07
58.6258.6258.62
63.4663.4663.46
25.0925.0925.09
35.2835.2835.28
25.8625.8625.86

2
Prefix LM, denoising
81.8281.8281.82
49.9949.9949.99
92.4392.4392.43
91.4391.4391.43
88.2488.2488.24
87.2087.2087.20
86.9886.9886.98
88.4188.4188.41
91.3991.3991.39
82.3282.3282.32
82.9382.9382.93
88.7188.7188.71
74.0174.0174.01
40.4640.4640.46
18.6118.6118.61
37.9037.9037.90
78.9478.9478.94
87.3187.3187.31
68.1168.1168.11
75.5075.5075.50
93.3793.3793.37
91.0791.0791.07
60.0060.0060.00
63.4363.4363.43
21.2021.2021.20
65.0365.0365.03
64.1164.1164.11
71.4871.4871.48
65.6765.6765.67
73.0873.0873.08
26.4326.4326.43
37.9837.9837.98
27.3927.3927.39

2
Enc/dec, LM
79.5679.5679.56
42.0342.0342.03
91.8691.8691.86
91.6491.6491.64
88.2488.2488.24
87.1387.1387.13
87.0087.0087.00
88.2188.2188.21
91.1591.1591.15
81.6881.6881.68
81.6681.6681.66
88.5488.5488.54
65.7065.7065.70
40.6740.6740.67
18.5918.5918.59
38.1338.1338.13
76.0276.0276.02
84.8584.8584.85
64.2964.2964.29
72.2372.2372.23
85.7485.7485.74
89.2989.2989.29
57.0057.0057.00
60.5360.5360.53
16.2616.2616.26
59.2859.2859.28
58.3058.3058.30
65.3465.3465.34
64.8964.8964.89
70.1970.1970.19
26.2726.2726.27
39.1739.1739.17
26.8626.8626.86

2
Enc/dec, shared, LM
79.6079.6079.60
44.8344.8344.83
92.0992.0992.09
90.2090.2090.20
85.7885.7885.78
86.0386.0386.03
85.8785.8785.87
87.7787.7787.77
91.0291.0291.02
81.7481.7481.74
82.2982.2982.29
89.1689.1689.16
65.3465.3465.34
40.1640.1640.16
18.1318.1318.13
37.5937.5937.59
76.3576.3576.35
84.8684.8684.86
63.5063.5063.50
70.4970.4970.49
91.4191.4191.41
87.5087.5087.50
55.0055.0055.00
60.2160.2160.21
16.8916.8916.89
57.8357.8357.83
56.7356.7356.73
63.5463.5463.54
63.4863.4863.48
70.1970.1970.19
26.6226.6226.62
39.1739.1739.17
27.0527.0527.05

2
Enc/dec, 6 layers, LM
78.6778.6778.67
38.7238.7238.72
91.4091.4091.40
90.4090.4090.40
86.5286.5286.52
86.8286.8286.82
86.4986.4986.49
87.8787.8787.87
91.0391.0391.03
80.9980.9980.99
80.9280.9280.92
88.0588.0588.05
65.7065.7065.70
40.2940.2940.29
18.2618.2618.26
37.7037.7037.70
75.3275.3275.32
84.0684.0684.06
64.0664.0664.06
71.3871.3871.38
85.2585.2585.25
89.2989.2989.29
60.0060.0060.00
57.5657.5657.56
16.7916.7916.79
55.2255.2255.22
54.3054.3054.30
66.7966.7966.79
63.9563.9563.95
71.1571.1571.15
26.1326.1326.13
38.4238.4238.42
26.8926.8926.89

2
Language model, LM
73.7873.7873.78
28.5328.5328.53
89.7989.7989.79
85.2385.2385.23
78.6878.6878.68
84.2284.2284.22
84.0084.0084.00
84.8884.8884.88
88.7088.7088.70
74.9474.9474.94
75.7775.7775.77
84.8484.8484.84
58.8458.8458.84
38.9738.9738.97
17.5417.5417.54
36.3736.3736.37
53.8153.8153.81
64.5564.5564.55
56.5156.5156.51
64.2264.2264.22
59.9259.9259.92
71.4371.4371.43
64.0064.0064.00
53.0453.0453.04
1.051.051.05
46.8146.8146.81
45.7845.7845.78
58.8458.8458.84
56.7456.7456.74
69.2369.2369.23
25.2325.2325.23
34.3134.3134.31
25.3825.3825.38

2
Prefix LM, LM
79.6879.6879.68
41.2641.2641.26
92.0992.0992.09
90.1190.1190.11
86.2786.2786.27
86.8286.8286.82
86.3286.3286.32
88.3588.3588.35
91.3591.3591.35
81.7181.7181.71
82.0282.0282.02
89.0489.0489.04
68.5968.5968.59
39.6639.6639.66
17.8417.8417.84
37.1337.1337.13
76.8776.8776.87
85.3985.3985.39
64.8664.8664.86
71.4771.4771.47
93.3793.3793.37
91.0791.0791.07
57.0057.0057.00
58.6758.6758.67
16.8916.8916.89
59.2559.2559.25
58.1658.1658.16
64.2664.2664.26
66.3066.3066.30
71.1571.1571.15
26.2826.2826.28
37.5137.5137.51
26.7626.7626.76

4
Language modeling with prefix
80.6980.6980.69
44.2244.2244.22
93.0093.0093.00
91.6891.6891.68
88.4888.4888.48
87.2087.2087.20
87.1887.1887.18
88.3988.3988.39
91.4191.4191.41
82.6682.6682.66
83.0983.0983.09
89.2989.2989.29
68.9568.9568.95
40.7140.7140.71
18.9418.9418.94
38.1538.1538.15
77.9977.9977.99
86.4386.4386.43
65.2765.2765.27
73.5573.5573.55
83.9583.9583.95
87.5087.5087.50
55.0055.0055.00
59.6559.6559.65
18.8918.8918.89
61.7661.7661.76
60.7660.7660.76
68.5968.5968.59
65.6765.6765.67
73.0873.0873.08
26.8626.8626.86
39.7339.7339.73
27.4927.4927.49

4

BERT-style (Devlin et al., 2018)

82.9682.9682.96
52.4952.4952.49
92.5592.5592.55
92.7992.7992.79
89.9589.9589.95
87.6887.6887.68
87.6687.6687.66
88.4788.4788.47
91.4491.4491.44
83.6083.6083.60
84.0584.0584.05
90.3390.3390.33
75.4575.4575.45
41.2741.2741.27
19.1719.1719.17
38.7238.7238.72
80.6580.6580.65
88.2488.2488.24
69.8569.8569.85
76.4876.4876.48
94.3794.3794.37
94.6494.6494.64
61.0061.0061.00
63.2963.2963.29
25.0825.0825.08
66.7666.7666.76
65.8565.8565.85
72.2072.2072.20
69.1269.1269.12
75.0075.0075.00
26.7826.7826.78
40.0340.0340.03
27.4127.4127.41

4
Deshuffling
73.1773.1773.17
22.8222.8222.82
87.1687.1687.16
86.8886.8886.88
81.1381.1381.13
84.0384.0384.03
83.8283.8283.82
86.3886.3886.38
89.9089.9089.90
76.3076.3076.30
76.3476.3476.34
84.1884.1884.18
58.8458.8458.84
40.7540.7540.75
18.5918.5918.59
38.1038.1038.10
67.6167.6167.61
76.7676.7676.76
58.4758.4758.47
69.1769.1769.17
63.7063.7063.70
78.5778.5778.57
56.0056.0056.00
59.8559.8559.85
12.7012.7012.70
45.5245.5245.52
44.3644.3644.36
57.0457.0457.04
64.8964.8964.89
68.2768.2768.27
26.1126.1126.11
39.3039.3039.30
25.6225.6225.62

5

BERT-style (Devlin et al., 2018)

82.9682.9682.96
52.4952.4952.49
92.5592.5592.55
92.7992.7992.79
89.9589.9589.95
87.6887.6887.68
87.6687.6687.66
88.4788.4788.47
91.4491.4491.44
83.6083.6083.60
84.0584.0584.05
90.3390.3390.33
75.4575.4575.45
41.2741.2741.27
19.1719.1719.17
38.7238.7238.72
80.6580.6580.65
88.2488.2488.24
69.8569.8569.85
76.4876.4876.48
94.3794.3794.37
94.6494.6494.64
61.0061.0061.00
63.2963.2963.29
25.0825.0825.08
66.7666.7666.76
65.8565.8565.85
72.2072.2072.20
69.1269.1269.12
75.0075.0075.00
26.7826.7826.78
40.0340.0340.03
27.4127.4127.41

5

MASS-style (Song et al., 2019)

82.3282.3282.32
47.0147.0147.01
91.6391.6391.63
92.5392.5392.53
89.7189.7189.71
88.2188.2188.21
88.1888.1888.18
88.5888.5888.58
91.4491.4491.44
82.9682.9682.96
83.6783.6783.67
90.0290.0290.02
77.2677.2677.26
41.1641.1641.16
19.1619.1619.16
38.5538.5538.55
80.1080.1080.10
88.0788.0788.07
69.2869.2869.28
75.0875.0875.08
84.9884.9884.98
89.2989.2989.29
63.0063.0063.00
64.4664.4664.46
23.5023.5023.50
66.7166.7166.71
65.9165.9165.91
72.2072.2072.20
67.7167.7167.71
78.8578.8578.85
26.7926.7926.79
39.8939.8939.89
27.5527.5527.55

5

★★\bigstar\,Replace corrupted spans

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

5
Drop corrupted tokens
84.4484.4484.44
60.0460.0460.04
92.8992.8992.89
92.7992.7992.79
89.9589.9589.95
87.2887.2887.28
86.8586.8586.85
88.5688.5688.56
91.5491.5491.54
83.9483.9483.94
83.9283.9283.92
90.7490.7490.74
79.4279.4279.42
41.2741.2741.27
19.3119.3119.31
38.7038.7038.70
80.5280.5280.52
88.2888.2888.28
68.6768.6768.67
75.9075.9075.90
96.0296.0296.02
94.6494.6494.64
56.0056.0056.00
65.0665.0665.06
23.9223.9223.92
65.5465.5465.54
64.6064.6064.60
71.1271.1271.12
67.4067.4067.40
74.0474.0474.04
27.0727.0727.07
39.7639.7639.76
27.8227.8227.82

6

Corruption rate = 10%percent1010\%

82.8282.8282.82
52.7152.7152.71
92.0992.0992.09
91.5591.5591.55
88.2488.2488.24
88.1988.1988.19
88.1588.1588.15
88.4788.4788.47
91.4091.4091.40
83.5083.5083.50
84.5184.5184.51
90.3390.3390.33
75.4575.4575.45
41.0541.0541.05
19.0019.0019.00
38.5338.5338.53
80.3880.3880.38
88.3688.3688.36
69.5569.5569.55
74.9874.9874.98
92.3792.3792.37
92.8692.8692.86
62.0062.0062.00
66.0466.0466.04
24.6624.6624.66
67.9367.9367.93
67.0967.0967.09
70.7670.7670.76
67.2467.2467.24
75.9675.9675.96
26.8726.8726.87
39.2839.2839.28
27.4427.4427.44

6

★★\bigstar\,Corruption rate = 15%percent1515\%

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

6

Corruption rate = 25%percent2525\%

83.0083.0083.00
53.4753.4753.47
93.0093.0093.00
92.4492.4492.44
89.4689.4689.46
87.3687.3687.36
87.3687.3687.36
88.6888.6888.68
91.5391.5391.53
84.4484.4484.44
84.1584.1584.15
90.7790.7790.77
74.0174.0174.01
41.6941.6941.69
19.5419.5419.54
39.1439.1439.14
80.9680.9680.96
88.6188.6188.61
70.4870.4870.48
76.3976.3976.39
93.0293.0293.02
92.8692.8692.86
68.0068.0068.00
65.4665.4665.46
24.6624.6624.66
68.2068.2068.20
67.3967.3967.39
73.6573.6573.65
67.8767.8767.87
72.1272.1272.12
27.0427.0427.04
39.8339.8339.83
27.4727.4727.47

6

Corruption rate = 50%percent5050\%

81.2781.2781.27
46.2646.2646.26
91.6391.6391.63
91.1191.1191.11
87.9987.9987.99
87.8787.8787.87
87.6487.6487.64
88.7088.7088.70
91.5791.5791.57
83.6483.6483.64
84.1084.1084.10
90.2490.2490.24
70.7670.7670.76
41.5141.5141.51
19.3219.3219.32
38.8938.8938.89
79.8079.8079.80
87.7687.7687.76
70.3370.3370.33
75.0275.0275.02
93.0593.0593.05
92.8692.8692.86
68.0068.0068.00
62.9762.9762.97
24.1324.1324.13
64.9464.9464.94
64.1364.1364.13
72.2072.2072.20
68.5068.5068.50
77.8877.8877.88
27.0127.0127.01
39.9039.9039.90
27.4927.4927.49

7

★★\bigstar\,Baseline (i.i.d.)

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

7

Average span length = 222

83.5483.5483.54
53.8253.8253.82
92.2092.2092.20
93.0593.0593.05
90.4490.4490.44
87.8587.8587.85
87.7187.7187.71
88.4288.4288.42
91.4091.4091.40
84.2884.2884.28
84.4684.4684.46
90.8890.8890.88
77.6277.6277.62
41.2341.2341.23
19.3919.3919.39
38.6938.6938.69
82.0982.0982.09
89.6989.6989.69
72.2072.2072.20
77.0677.0677.06
90.4390.4390.43
91.0791.0791.07
70.0070.0070.00
66.2866.2866.28
26.1326.1326.13
71.3471.3471.34
70.6170.6170.61
75.4575.4575.45
68.3468.3468.34
78.8578.8578.85
26.7626.7626.76
39.9939.9939.99
27.6327.6327.63

7

Average span length = 333

83.4983.4983.49
53.9053.9053.90
92.4392.4392.43
92.2592.2592.25
89.4689.4689.46
87.4987.4987.49
87.5387.5387.53
88.7288.7288.72
91.5191.5191.51
84.8584.8584.85
84.8484.8484.84
90.9990.9990.99
77.2677.2677.26
41.5041.5041.50
19.6219.6219.62
38.9438.9438.94
81.8481.8481.84
89.6689.6689.66
72.5372.5372.53
76.8576.8576.85
94.3794.3794.37
94.6494.6494.64
70.0070.0070.00
67.6467.6467.64
28.7528.7528.75
70.8470.8470.84
69.9069.9069.90
74.7374.7374.73
67.7167.7167.71
77.8877.8877.88
26.8626.8626.86
39.6539.6539.65
27.6227.6227.62

7

Average span length = 555

83.4083.4083.40
52.1252.1252.12
93.1293.1293.12
92.6392.6392.63
89.7189.7189.71
88.7088.7088.70
88.4788.4788.47
88.8488.8488.84
91.6491.6491.64
84.3284.3284.32
84.2984.2984.29
90.7990.7990.79
76.9076.9076.90
41.3941.3941.39
19.2419.2419.24
38.8238.8238.82
82.0582.0582.05
89.7989.7989.79
72.2372.2372.23
77.0677.0677.06
83.0683.0683.06
89.2989.2989.29
69.0069.0069.00
68.1668.1668.16
30.1230.1230.12
71.3671.3671.36
70.5370.5370.53
75.8175.8175.81
69.9169.9169.91
79.8179.8179.81
26.8826.8826.88
39.4039.4039.40
27.5327.5327.53

7

Average span length = 101010

82.8582.8582.85
50.1150.1150.11
92.0992.0992.09
91.9591.9591.95
88.9788.9788.97
88.4588.4588.45
88.2288.2288.22
88.8688.8688.86
91.6391.6391.63
84.3484.3484.34
84.2884.2884.28
91.0791.0791.07
76.1776.1776.17
41.3841.3841.38
19.3319.3319.33
38.8038.8038.80
81.8481.8481.84
89.3989.3989.39
70.4470.4470.44
76.4576.4576.45
87.4087.4087.40
89.2989.2989.29
65.0065.0065.00
66.8766.8766.87
29.5929.5929.59
69.8269.8269.82
68.9468.9468.94
72.5672.5672.56
67.5567.5567.55
75.9675.9675.96
26.7926.7926.79
39.4939.4939.49
27.6927.6927.69

8

★★\bigstar\,C4

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

8
C4, unfiltered
81.4681.4681.46
48.0148.0148.01
91.6391.6391.63
92.7292.7292.72
89.9589.9589.95
87.7987.7987.79
87.6087.6087.60
88.3188.3188.31
91.2791.2791.27
82.3082.3082.30
82.3482.3482.34
88.7188.7188.71
72.2072.2072.20
41.0941.0941.09
19.1419.1419.14
38.5438.5438.54
78.7878.7878.78
87.0487.0487.04
68.0468.0468.04
75.7575.7575.75
89.1789.1789.17
91.0791.0791.07
62.0062.0062.00
65.5265.5265.52
25.6025.6025.60
62.4262.4262.42
61.5861.5861.58
69.6869.6869.68
67.0867.0867.08
72.1272.1272.12
26.5526.5526.55
39.3439.3439.34
27.2127.2127.21

8
RealNews-like
83.8383.8383.83
56.5556.5556.55
92.6692.6692.66
92.0692.0692.06
88.9788.9788.97
87.7187.7187.71
87.3787.3787.37
88.5188.5188.51
91.4991.4991.49
84.3584.3584.35
84.4684.4684.46
90.6190.6190.61
78.3478.3478.34
41.3841.3841.38
19.2319.2319.23
38.8438.8438.84
80.3980.3980.39
88.5088.5088.50
72.3872.3872.38
77.0077.0077.00
93.0993.0993.09
94.6494.6494.64
66.0066.0066.00
65.9265.9265.92
23.8223.8223.82
74.5674.5674.56
73.7273.7273.72
75.8175.8175.81
66.6166.6166.61
80.7780.7780.77
26.7526.7526.75
39.9039.9039.90
27.4827.4827.48

8
WebText-like
84.0384.0384.03
56.3856.3856.38
93.1293.1293.12
92.3192.3192.31
89.2289.2289.22
88.6988.6988.69
88.6888.6888.68
88.6588.6588.65
91.5691.5691.56
84.7084.7084.70
84.8484.8484.84
90.8390.8390.83
77.6277.6277.62
41.2341.2341.23
19.3119.3119.31
38.7038.7038.70
81.4281.4281.42
89.1589.1589.15
71.4071.4071.40
76.8876.8876.88
83.0883.0883.08
89.2989.2989.29
66.0066.0066.00
64.1064.1064.10
24.2424.2424.24
72.2472.2472.24
71.3671.3671.36
75.4575.4575.45
68.0368.0368.03
82.6982.6982.69
26.8026.8026.80
39.7439.7439.74
27.5927.5927.59

8
Wikipedia
81.8581.8581.85
45.5345.5345.53
92.3292.3292.32
91.6791.6791.67
88.2488.2488.24
85.6285.6285.62
86.4086.4086.40
88.3788.3788.37
91.3491.3491.34
82.6182.6182.61
83.2583.2583.25
90.9690.9690.96
77.2677.2677.26
41.3941.3941.39
19.3119.3119.31
38.8138.8138.81
81.2981.2981.29
89.1889.1889.18
68.0168.0168.01
76.1276.1276.12
56.0356.0356.03
80.3680.3680.36
67.0067.0067.00
65.0165.0165.01
25.9225.9225.92
69.0369.0369.03
68.0668.0668.06
74.7374.7374.73
67.0867.0867.08
76.9276.9276.92
26.9426.9426.94
39.6939.6939.69
27.6727.6727.67

8
Wikipedia + TBC
83.6583.6583.65
55.5355.5355.53
92.7892.7892.78
92.4192.4192.41
89.2289.2289.22
86.6786.6786.67
86.2786.2786.27
89.4789.4789.47
92.2992.2992.29
84.3884.3884.38
83.4583.4583.45
91.9491.9491.94
76.9076.9076.90
41.2241.2241.22
19.2819.2819.28
38.6738.6738.67
82.0882.0882.08
89.7089.7089.70
73.2473.2473.24
76.2276.2276.22
95.4095.4095.40
92.8692.8692.86
69.0069.0069.00
51.5951.5951.59
50.9350.9350.93
69.5369.5369.53
68.5168.5168.51
77.6277.6277.62
66.9366.9366.93
81.7381.7381.73
26.7726.7726.77
39.6339.6339.63
27.5727.5727.57

9

★★\bigstar\,Full data set

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

9

229superscript2292^{29} (646464 repeats)

82.8782.8782.87
53.8253.8253.82
92.7892.7892.78
91.7991.7991.79
88.7388.7388.73
87.5687.5687.56
87.5887.5887.58
88.7388.7388.73
91.5491.5491.54
84.0784.0784.07
84.2184.2184.21
90.5990.5990.59
73.6573.6573.65
41.1841.1841.18
19.1919.1919.19
38.6738.6738.67
80.9780.9780.97
88.9088.9088.90
72.0372.0372.03
76.7676.7676.76
92.9692.9692.96
92.8692.8692.86
66.0066.0066.00
65.1165.1165.11
26.7626.7626.76
69.3569.3569.35
68.4968.4968.49
75.8175.8175.81
67.2467.2467.24
82.6982.6982.69
26.8326.8326.83
39.7439.7439.74
27.6327.6327.63

9

227superscript2272^{27} (256256256 repeats)

82.6282.6282.62
50.6050.6050.60
92.3292.3292.32
92.0792.0792.07
88.7388.7388.73
87.8387.8387.83
87.6087.6087.60
88.6588.6588.65
91.5491.5491.54
83.4383.4383.43
84.3784.3784.37
90.1290.1290.12
75.8175.8175.81
41.2441.2441.24
19.2019.2019.20
38.7038.7038.70
79.7879.7879.78
87.6387.6387.63
69.9769.9769.97
75.2975.2975.29
93.4293.4293.42
91.0791.0791.07
63.0063.0063.00
61.8261.8261.82
23.6123.6123.61
66.2766.2766.27
65.3965.3965.39
73.6573.6573.65
66.3066.3066.30
80.7780.7780.77
27.0227.0227.02
39.7139.7139.71
27.3327.3327.33

9

225superscript2252^{25} (1,02410241{,}024 repeats)

79.5579.5579.55
43.8443.8443.84
91.2891.2891.28
89.3289.3289.32
85.0585.0585.05
85.9285.9285.92
85.7485.7485.74
88.0588.0588.05
91.0991.0991.09
81.2981.2981.29
81.7281.7281.72
87.9087.9087.90
69.3169.3169.31
40.6640.6640.66
18.5718.5718.57
38.1338.1338.13
76.2776.2776.27
84.5884.5884.58
64.7664.7664.76
72.6372.6372.63
83.9783.9783.97
82.1482.1482.14
64.0064.0064.00
59.3959.3959.39
17.9417.9417.94
56.9456.9456.94
56.0456.0456.04
64.9864.9864.98
65.2065.2065.20
73.0873.0873.08
26.3826.3826.38
39.5639.5639.56
26.8026.8026.80

9

223superscript2232^{23} (4,09640964{,}096 repeats)

76.3476.3476.34
32.6832.6832.68
89.4589.4589.45
89.8489.8489.84
86.0386.0386.03
83.4983.4983.49
83.4283.4283.42
87.1887.1887.18
90.6190.6190.61
77.8077.8077.80
78.6978.6978.69
85.4785.4785.47
64.6264.6264.62
40.1640.1640.16
18.3318.3318.33
37.6637.6637.66
70.9270.9270.92
80.2080.2080.20
59.2959.2959.29
69.8569.8569.85
73.4873.4873.48
73.2173.2173.21
56.0056.0056.00
57.6657.6657.66
14.3814.3814.38
46.6946.6946.69
45.7945.7945.79
59.5759.5759.57
65.0565.0565.05
68.2768.2768.27
26.3726.3726.37
38.8438.8438.84
25.8125.8125.81

10

★★\bigstar\,All parameters

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

10

Adapter layers, d=32𝑑32d=32

80.5280.5280.52
45.3345.3345.33
91.6391.6391.63
90.5990.5990.59
86.7686.7686.76
88.3888.3888.38
88.0688.0688.06
86.9986.9986.99
90.2690.2690.26
83.6383.6383.63
83.9483.9483.94
90.7290.7290.72
67.1567.1567.15
34.5034.5034.50
15.0815.0815.08
32.1532.1532.15
79.3279.3279.32
87.7087.7087.70
60.4060.4060.40
65.3265.3265.32
50.8750.8750.87
73.2173.2173.21
52.0052.0052.00
58.6158.6158.61
19.4119.4119.41
65.5065.5065.50
64.5864.5864.58
62.0962.0962.09
64.5864.5864.58
73.0873.0873.08
13.8413.8413.84
17.8817.8817.88
15.5415.5415.54

10

Adapter layers, d=128𝑑128d=128

81.5181.5181.51
45.3545.3545.35
92.8992.8992.89
91.4991.4991.49
88.2488.2488.24
87.7387.7387.73
87.6587.6587.65
87.7387.7387.73
90.9390.9390.93
83.6483.6483.64
84.0984.0984.09
90.5290.5290.52
72.5672.5672.56
36.7136.7136.71
16.6216.6216.62
34.3734.3734.37
79.4779.4779.47
87.6187.6187.61
63.0363.0363.03
69.2069.2069.20
52.2152.2152.21
75.0075.0075.00
56.0056.0056.00
61.0861.0861.08
18.0518.0518.05
67.9467.9467.94
66.9766.9766.97
68.5968.5968.59
66.7766.7766.77
73.0873.0873.08
19.8319.8319.83
27.5027.5027.50
22.6322.6322.63

10

Adapter layers, d=512𝑑512d=512

81.5481.5481.54
44.2544.2544.25
93.3593.3593.35
91.0091.0091.00
87.2587.2587.25
88.7488.7488.74
88.4488.4488.44
88.0288.0288.02
91.1591.1591.15
83.0883.0883.08
83.8083.8083.80
89.6289.6289.62
74.3774.3774.37
38.6338.6338.63
17.7817.7817.78
36.2536.2536.25
79.1879.1879.18
87.3287.3287.32
64.3064.3064.30
73.1873.1873.18
59.8659.8659.86
71.4371.4371.43
56.0056.0056.00
62.9462.9462.94
18.5718.5718.57
66.5666.5666.56
65.7465.7465.74
70.7670.7670.76
67.8767.8767.87
74.0474.0474.04
23.4523.4523.45
33.9833.9833.98
25.8125.8125.81

10

Adapter layers, d=2048𝑑2048d=2048

82.6282.6282.62
49.8649.8649.86
92.5592.5592.55
91.3091.3091.30
87.9987.9987.99
88.4688.4688.46
88.3588.3588.35
88.3688.3688.36
91.4091.4091.40
83.6383.6383.63
83.1883.1883.18
90.6690.6690.66
76.5376.5376.53
39.4439.4439.44
18.3018.3018.30
37.0637.0637.06
79.4079.4079.40
87.3687.3687.36
68.6168.6168.61
74.5374.5374.53
88.0088.0088.00
91.0791.0791.07
58.0058.0058.00
61.1061.1061.10
18.8918.8918.89
66.7366.7366.73
66.0666.0666.06
73.2973.2973.29
71.1671.1671.16
75.9675.9675.96
25.6425.6425.64
36.9236.9236.92
26.9326.9326.93

10
Gradual Unfreezing
82.5082.5082.50
51.7451.7451.74
91.9791.9791.97
92.6192.6192.61
89.7189.7189.71
87.2787.2787.27
86.9086.9086.90
88.2688.2688.26
91.3591.3591.35
83.4283.4283.42
83.4983.4983.49
89.7189.7189.71
75.0975.0975.09
40.8840.8840.88
18.9518.9518.95
38.4038.4038.40
79.1779.1779.17
87.3087.3087.30
70.7970.7970.79
75.5175.5175.51
93.0993.0993.09
94.6494.6494.64
70.0070.0070.00
62.0362.0362.03
21.5121.5121.51
65.6965.6965.69
64.7964.7964.79
72.9272.9272.92
69.1269.1269.12
77.8977.8977.89
26.7126.7126.71
39.0239.0239.02
26.9326.9326.93

11

★★\bigstar\,Baseline (pre-train/fine-tune)

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

11
Equal
76.1376.1376.13
39.4739.4739.47
90.9490.9490.94
82.9082.9082.90
75.7475.7475.74
78.8378.8378.83
78.4478.4478.44
86.4586.4586.45
89.7189.7189.71
82.0882.0882.08
82.9282.9282.92
90.1390.1390.13
59.9359.9359.93
40.9540.9540.95
19.0219.0219.02
38.3938.3938.39
76.5176.5176.51
85.6185.6185.61
63.3763.3763.37
73.0673.0673.06
82.3782.3782.37
83.9383.9383.93
65.0065.0065.00
60.8960.8960.89
17.5217.5217.52
60.5160.5160.51
59.7059.7059.70
61.0161.0161.01
60.0360.0360.03
65.3865.3865.38
23.8923.8923.89
34.3134.3134.31
26.7826.7826.78

11

Examples-proportional, K=216𝐾superscript216K=2^{16}

80.4580.4580.45
42.0742.0742.07
91.9791.9791.97
90.9790.9790.97
87.5087.5087.50
85.4185.4185.41
85.0485.0485.04
86.8986.8986.89
90.1090.1090.10
83.0183.0183.01
83.6683.6683.66
90.7490.7490.74
72.5672.5672.56
41.1641.1641.16
19.0419.0419.04
38.5938.5938.59
77.2577.2577.25
85.7285.7285.72
69.9569.9569.95
76.6776.6776.67
86.3886.3886.38
89.2989.2989.29
70.0070.0070.00
65.9365.9365.93
27.9127.9127.91
62.7862.7862.78
61.9561.9561.95
76.9076.9076.90
65.8365.8365.83
73.0873.0873.08
24.3524.3524.35
34.9934.9934.99
27.1027.1027.10

11

Examples-proportional, K=217𝐾superscript217K=2^{17}

81.5681.5681.56
47.3547.3547.35
91.4091.4091.40
91.5591.5591.55
88.2488.2488.24
86.1586.1586.15
85.9385.9385.93
86.9486.9486.94
90.0690.0690.06
82.7682.7682.76
84.1284.1284.12
90.7990.7990.79
75.0975.0975.09
41.0641.0641.06
19.1219.1219.12
38.4738.4738.47
77.0077.0077.00
85.8785.8785.87
67.9167.9167.91
77.8977.8977.89
77.5477.5477.54
85.7185.7185.71
57.0057.0057.00
67.7867.7867.78
27.0727.0727.07
61.5161.5161.51
60.5460.5460.54
79.0679.0679.06
65.2065.2065.20
74.0474.0474.04
24.3624.3624.36
35.0035.0035.00
27.2527.2527.25

11

Examples-proportional, K=218𝐾superscript218K=2^{18}

81.6781.6781.67
46.8546.8546.85
91.6391.6391.63
91.9991.9991.99
88.7388.7388.73
87.6887.6887.68
87.2087.2087.20
86.9386.9386.93
90.3590.3590.35
83.3083.3083.30
84.0184.0184.01
91.4791.4791.47
73.2973.2973.29
40.9640.9640.96
19.0719.0719.07
38.4338.4338.43
78.1778.1778.17
86.7486.7486.74
67.9467.9467.94
76.5776.5776.57
78.8878.8878.88
87.5087.5087.50
62.0062.0062.00
67.7067.7067.70
30.8530.8530.85
63.4363.4363.43
62.5462.5462.54
76.5376.5376.53
65.6765.6765.67
67.3167.3167.31
24.5724.5724.57
35.1935.1935.19
27.3927.3927.39

11

Examples-proportional, K=219𝐾superscript219K=2^{19}

81.4281.4281.42
45.9445.9445.94
91.6391.6391.63
92.2092.2092.20
89.2289.2289.22
88.4488.4488.44
88.3288.3288.32
86.8486.8486.84
90.1090.1090.10
83.7383.7383.73
84.2984.2984.29
91.8491.8491.84
70.4070.4070.40
41.2641.2641.26
19.2419.2419.24
38.7138.7138.71
79.7879.7879.78
88.1588.1588.15
67.3067.3067.30
75.6675.6675.66
75.5975.5975.59
87.5087.5087.50
59.0059.0059.00
68.2268.2268.22
30.6430.6430.64
65.3265.3265.32
64.2964.2964.29
73.6573.6573.65
65.0565.0565.05
69.2369.2369.23
25.2125.2125.21
36.3036.3036.30
27.7627.7627.76

11

Examples-proportional, K=220𝐾superscript220K=2^{20}

80.8080.8080.80
42.5542.5542.55
92.7892.7892.78
91.2791.2791.27
87.9987.9987.99
88.3688.3688.36
88.1088.1088.10
86.1086.1086.10
89.6289.6289.62
84.1584.1584.15
84.2684.2684.26
92.2092.2092.20
68.9568.9568.95
41.0541.0541.05
19.2419.2419.24
38.4638.4638.46
80.3680.3680.36
88.2788.2788.27
67.3867.3867.38
73.2173.2173.21
76.1876.1876.18
83.9383.9383.93
62.0062.0062.00
67.5767.5767.57
26.8626.8626.86
66.1266.1266.12
65.2265.2265.22
76.9076.9076.90
64.7364.7364.73
69.2369.2369.23
25.6625.6625.66
36.9336.9336.93
27.6827.6827.68

11

Examples-proportional, K=221𝐾superscript221K=2^{21}

79.8379.8379.83
44.4544.4544.45
91.2891.2891.28
89.0089.0089.00
84.3184.3184.31
87.5487.5487.54
87.4087.4087.40
84.9384.9384.93
88.5388.5388.53
82.5482.5482.54
84.1684.1684.16
90.8590.8590.85
67.8767.8767.87
40.5140.5140.51
18.7918.7918.79
37.9237.9237.92
79.5079.5079.50
87.4887.4887.48
65.1065.1065.10
71.1671.1671.16
68.8868.8868.88
85.7185.7185.71
57.0057.0057.00
62.7562.7562.75
23.4023.4023.40
64.5064.5064.50
63.6563.6563.65
72.9272.9272.92
64.1164.1164.11
71.1571.1571.15
25.8225.8225.82
37.2237.2237.22
27.1327.1327.13

11

Temperature-scaled, T=2𝑇2T=2

81.9081.9081.90
54.0054.0054.00
91.7491.7491.74
90.5690.5690.56
86.7686.7686.76
85.1185.1185.11
84.6084.6084.60
86.4086.4086.40
89.7489.7489.74
83.4783.4783.47
84.1584.1584.15
91.5191.5191.51
72.5672.5672.56
41.0941.0941.09
19.2819.2819.28
38.5438.5438.54
79.4279.4279.42
87.7787.7787.77
69.9269.9269.92
76.7376.7376.73
92.3792.3792.37
92.8692.8692.86
57.0057.0057.00
69.8069.8069.80
31.9031.9031.90
66.6566.6566.65
65.7465.7465.74
72.9272.9272.92
67.0867.0867.08
75.9675.9675.96
25.4225.4225.42
36.7236.7236.72
27.2027.2027.20

11

Temperature-scaled, T=4𝑇4T=4

80.5680.5680.56
45.3845.3845.38
91.9791.9791.97
89.6889.6889.68
85.7885.7885.78
83.1383.1383.13
82.7682.7682.76
86.3986.3986.39
90.0090.0090.00
82.7882.7882.78
84.1984.1984.19
91.1691.1691.16
73.6573.6573.65
41.0941.0941.09
19.2219.2219.22
38.5138.5138.51
77.9977.9977.99
86.8186.8186.81
69.5469.5469.54
76.7676.7676.76
97.3697.3697.36
96.4396.4396.43
59.0059.0059.00
68.1068.1068.10
31.4831.4831.48
64.2664.2664.26
63.2763.2763.27
74.7374.7374.73
64.2664.2664.26
71.1571.1571.15
25.0425.0425.04
35.8235.8235.82
27.4527.4527.45

11

Temperature-scaled, T=8𝑇8T=8

77.2177.2177.21
40.0740.0740.07
91.0691.0691.06
88.1188.1188.11
83.3383.3383.33
79.2079.2079.20
79.0679.0679.06
86.6086.6086.60
89.9089.9089.90
83.0583.0583.05
83.5683.5683.56
90.2190.2190.21
59.9359.9359.93
41.0141.0141.01
19.1019.1019.10
38.4038.4038.40
77.1477.1477.14
85.9985.9985.99
66.0766.0766.07
73.9473.9473.94
93.7093.7093.70
94.6494.6494.64
60.0060.0060.00
66.3666.3666.36
26.8626.8626.86
63.4663.4663.46
62.6062.6062.60
62.0962.0962.09
63.3263.3263.32
65.3865.3865.38
24.5524.5524.55
35.3535.3535.35
27.1727.1727.17

12

★★\bigstar\,Unsupervised pre-training + fine-tuning

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

12
Multi-task training
81.4281.4281.42
45.9445.9445.94
91.6391.6391.63
92.2092.2092.20
89.2289.2289.22
88.4488.4488.44
88.3288.3288.32
86.8486.8486.84
90.1090.1090.10
83.7383.7383.73
84.2984.2984.29
91.8491.8491.84
70.4070.4070.40
41.2641.2641.26
19.2419.2419.24
38.7138.7138.71
79.7879.7879.78
88.1588.1588.15
67.3067.3067.30
75.6675.6675.66
75.5975.5975.59
87.5087.5087.50
59.0059.0059.00
68.2268.2268.22
30.6430.6430.64
65.3265.3265.32
64.2964.2964.29
73.6573.6573.65
65.0565.0565.05
69.2369.2369.23
25.2125.2125.21
36.3036.3036.30
27.7627.7627.76

12
Multi-task pre-training + fine-tuning
83.1183.1183.11
51.4251.4251.42
92.6692.6692.66
91.7391.7391.73
88.7388.7388.73
88.0688.0688.06
87.7087.7087.70
88.6188.6188.61
91.6191.6191.61
84.0984.0984.09
84.3184.3184.31
91.8591.8591.85
76.5376.5376.53
41.1541.1541.15
19.1219.1219.12
38.5938.5938.59
80.2680.2680.26
88.5088.5088.50
71.0371.0371.03
79.5479.5479.54
81.6981.6981.69
87.5087.5087.50
65.0065.0065.00
70.7270.7270.72
31.4831.4831.48
65.9465.9465.94
65.0365.0365.03
81.2381.2381.23
68.1868.1868.18
73.0873.0873.08
27.0827.0827.08
39.8039.8039.80
28.0728.0728.07

12
Leave-one-out multi-task training
81.9881.9881.98
48.0048.0048.00
93.2393.2393.23
91.7291.7291.72
88.2488.2488.24
87.7687.7687.76
87.3287.3287.32
88.6188.6188.61
91.4491.4491.44
84.0084.0084.00
84.1184.1184.11
90.7990.7990.79
72.2072.2072.20
41.3441.3441.34
19.0519.0519.05
38.7738.7738.77
79.9779.9779.97
88.1088.1088.10
71.6871.6871.68
78.3578.3578.35
86.7686.7686.76
89.2989.2989.29
66.0066.0066.00
68.0968.0968.09
29.4929.4929.49
66.2366.2366.23
65.2765.2765.27
79.0679.0679.06
68.6568.6568.65
78.8578.8578.85
26.9326.9326.93
39.7939.7939.79
27.8727.8727.87

12
Supervised multi-task pre-training
79.9379.9379.93
36.6036.6036.60
92.4392.4392.43
91.5891.5891.58
88.2488.2488.24
87.0387.0387.03
86.7886.7886.78
88.1588.1588.15
91.2091.2091.20
82.8782.8782.87
83.1683.1683.16
90.1390.1390.13
70.7670.7670.76
41.1241.1241.12
18.9618.9618.96
38.4938.4938.49
77.3877.3877.38
85.6585.6585.65
65.3665.3665.36
75.6675.6675.66
68.8768.8768.87
83.9383.9383.93
58.0058.0058.00
64.8164.8164.81
21.9321.9321.93
55.3755.3755.37
54.6154.6154.61
71.1271.1271.12
67.4067.4067.40
75.9675.9675.96
26.8126.8126.81
40.1340.1340.13
28.0428.0428.04

13

★★\bigstar\,Baseline

83.2883.2883.28
53.8453.8453.84
92.6892.6892.68
92.0792.0792.07
88.9288.9288.92
88.0288.0288.02
87.9487.9487.94
88.6788.6788.67
91.5691.5691.56
84.2484.2484.24
84.5784.5784.57
90.4890.4890.48
76.2876.2876.28
41.3341.3341.33
19.2419.2419.24
38.7738.7738.77
80.8880.8880.88
88.8188.8188.81
71.3671.3671.36
76.6276.6276.62
91.2291.2291.22
91.9691.9691.96
66.2066.2066.20
66.1366.1366.13
25.7825.7825.78
69.0569.0569.05
68.1668.1668.16
75.3475.3475.34
68.0468.0468.04
78.5678.5678.56
26.9826.9826.98
39.8239.8239.82
27.6527.6527.65

13

1×1\times size, 4×4\times training steps

85.3385.3385.33
60.2960.2960.29
93.8193.8193.81
94.0694.0694.06
91.6791.6791.67
89.4289.4289.42
89.2589.2589.25
89.1589.1589.15
91.8791.8791.87
86.0186.0186.01
85.7085.7085.70
91.6391.6391.63
78.3478.3478.34
41.5241.5241.52
19.3319.3319.33
38.9638.9638.96
82.4582.4582.45
90.1990.1990.19
74.7274.7274.72
79.1779.1779.17
94.7594.7594.75
92.8692.8692.86
71.0071.0071.00
67.3467.3467.34
29.7029.7029.70
72.6372.6372.63
71.5971.5971.59
78.3478.3478.34
72.1072.1072.10
82.6982.6982.69
27.0827.0827.08
40.6640.6640.66
27.9327.9327.93

13

1×1\times size, 4×4\times batch size

84.6084.6084.60
56.0856.0856.08
93.1293.1293.12
92.3192.3192.31
89.2289.2289.22
88.8588.8588.85
88.8488.8488.84
89.3589.3589.35
92.0792.0792.07
85.9885.9885.98
86.1386.1386.13
91.0791.0791.07
80.1480.1480.14
41.7041.7041.70
19.4219.4219.42
39.0839.0839.08
82.5282.5282.52
90.2190.2190.21
74.6474.6474.64
78.7878.7878.78
93.6993.6993.69
94.6494.6494.64
72.0072.0072.00
68.0968.0968.09
30.9530.9530.95
74.7374.7374.73
73.9073.9073.90
76.5376.5376.53
70.0670.0670.06
81.7381.7381.73
27.0727.0727.07
40.6040.6040.60
27.8427.8427.84

13

2×2\times size, 2×2\times training steps

86.1886.1886.18
62.0462.0462.04
93.6993.6993.69
93.3693.3693.36
90.6990.6990.69
89.1889.1889.18
89.2389.2389.23
89.3589.3589.35
92.0592.0592.05
87.2387.2387.23
87.0587.0587.05
92.6892.6892.68
81.9581.9581.95
41.7441.7441.74
19.6619.6619.66
39.1439.1439.14
84.1884.1884.18
91.2991.2991.29
77.1877.1877.18
80.9880.9880.98
97.3697.3697.36
96.4396.4396.43
74.0074.0074.00
71.3471.3471.34
35.6835.6835.68
77.1177.1177.11
76.3476.3476.34
80.5180.5180.51
69.2869.2869.28
85.5885.5885.58
27.5227.5227.52
41.0341.0341.03
28.1928.1928.19

13

4×4\times size, 1×1\times training steps

85.9185.9185.91
57.5857.5857.58
94.3894.3894.38
92.6792.6792.67
89.9589.9589.95
89.6089.6089.60
89.6089.6089.60
89.4489.4489.44
92.1492.1492.14
87.0587.0587.05
87.1287.1287.12
93.1293.1293.12
83.3983.3983.39
41.6041.6041.60
19.7319.7319.73
39.0839.0839.08
83.8683.8683.86
91.3291.3291.32
78.0478.0478.04
81.3881.3881.38
89.0989.0989.09
94.6494.6494.64
73.0073.0073.00
73.7473.7473.74
40.4040.4040.40
78.2578.2578.25
77.4077.4077.40
81.5981.5981.59
70.2270.2270.22
91.3591.3591.35
27.4727.4727.47
40.7140.7140.71
28.1028.1028.10

13

4×4\times ensembled

84.7784.7784.77
56.1456.1456.14
93.4693.4693.46
93.3193.3193.31
90.6790.6790.67
89.7189.7189.71
89.6089.6089.60
89.6289.6289.62
92.2492.2492.24
86.2286.2286.22
86.5386.5386.53
91.6091.6091.60
77.9877.9877.98
42.1042.1042.10
20.1020.1020.10
39.5639.5639.56
83.0983.0983.09
90.4090.4090.40
71.7471.7471.74
77.5877.5877.58
89.8589.8589.85
91.0791.0791.07
66.0066.0066.00
69.3269.3269.32
29.4929.4929.49
72.6772.6772.67
71.9471.9471.94
76.9076.9076.90
69.1269.1269.12
72.1272.1272.12
28.0528.0528.05
40.5340.5340.53
28.0928.0928.09

13

4×4\times ensembled, fine-tune only

84.0584.0584.05
54.7854.7854.78
92.7892.7892.78
93.1593.1593.15
90.4490.4490.44
88.3488.3488.34
88.1288.1288.12
89.2789.2789.27
91.9791.9791.97
85.3385.3385.33
85.8885.8885.88
90.9890.9890.98
77.6277.6277.62
41.6641.6641.66
19.5719.5719.57
39.1239.1239.12
82.3682.3682.36
89.8689.8689.86
71.5671.5671.56
77.4377.4377.43
90.0790.0790.07
92.8692.8692.86
69.0069.0069.00
67.3167.3167.31
26.3426.3426.34
70.4770.4770.47
69.6469.6469.64
75.4575.4575.45
68.1868.1868.18
74.0474.0474.04
27.5527.5527.55
40.2240.2240.22
28.0928.0928.09

## References

- Al-Rfou et al. (2019)

Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones.

Character-level language modeling with deeper self-attention.

In Proceedings of the AAAI Conference on Artificial
Intelligence, 2019.

- Anil et al. (2019)

Rohan Anil, Vineet Gupta, Tomer Koren, and Yoram Singer.

Memory-efficient adaptive optimization for large-scale learning.

arXiv preprint arXiv:1901.11150, 2019.

- Arivazhagan et al. (2019)

Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson,
Maxim Krikun, Mia Xu Chen, Yuan Cao, George Foster, Colin Cherry, et al.

Massively multilingual neural machine translation in the wild:
Findings and challenges.

arXiv preprint arXiv:1907.05019, 2019.

- Ba et al. (2016)

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton.

Layer normalization.

arXiv preprint arXiv:1607.06450, 2016.

- Baevski et al. (2019)

Alexei Baevski, Sergey Edunov, Yinhan Liu, Luke Zettlemoyer, and Michael Auli.

Cloze-driven pretraining of self-attention networks.

arXiv preprint arXiv:1903.07785, 2019.

- Bahdanau et al. (2015)

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio.

Neural machine translation by jointly learning to align and
translate.

In Third International Conference on Learning Representations,
2015.

- Bapna et al. (2019)

Ankur Bapna, Naveen Arivazhagan, and Orhan Firat.

Simple, scalable adaptation for neural machine translation.

arXiv preprint arXiv:1909.08478, 2019.

- Beltagy et al. (2019)

Iz Beltagy, Kyle Lo, and Arman Cohan.

SciBERT: A pretrained language model for scientific text.

In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP), 2019.

- Bojar et al. (2014)

Ondřej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp
Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve
Saint-Amand, et al.

Findings of the 2014 workshop on statistical machine translation.

In Proceedings of the Ninth Workshop on Statistical Machine
Translation, 2014.

- Bojar et al. (2015)

Ondřej Bojar, Rajen Chatterjee, Christian Federmann, Barry Haddow,
Matthias Huck, Chris Hokamp, Philipp Koehn, Varvara Logacheva, Christof Monz,
Matteo Negri, et al.

Findings of the 2015 workshop on statistical machine translation.

In Proceedings of the Tenth Workshop on Statistical Machine
Translation, 2015.

- Bojar et al. (2016)

Ondřej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry
Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara
Logacheva, Christof Monz, et al.

Findings of the 2016 conference on machine translation.

In Proceedings of the First Conference on Machine Translation,
2016.

- Bowman et al. (2015)

Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz,
and Samy Bengio.

Generating sentences from a continuous space.

arXiv preprint arXiv:1511.06349, 2015.

- Buck et al. (2014)

Christian Buck, Kenneth Heafield, and Bas Van Ooyen.

N-gram counts and language models from the common crawl.

In LREC, 2014.

- Caruana (1997)

Rich Caruana.

Multitask learning.

Machine learning, 28(1), 1997.

- Cer et al. (2017)

Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia.

Semeval-2017 task 1: Semantic textual similarity-multilingual and
cross-lingual focused evaluation.

arXiv preprint arXiv:1708.00055, 2017.

- Cheng et al. (2016)

Jianpeng Cheng, Li Dong, and Mirella Lapata.

Long short-term memory-networks for machine reading.

arXiv preprint arXiv:1601.06733, 2016.

- Clark et al. (2019)

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael
Collins, and Kristina Toutanova.

BoolQ: Exploring the surprising difficulty of natural yes/no
questions.

arXiv preprint arXiv:1905.10044, 2019.

- Clark et al. (2020)

Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning.

Electra: Pre-training text encoders as discriminators rather than
generators.

arXiv preprint arXiv:2003.10555, 2020.

- Conneau and Kiela (2018)

Alexis Conneau and Douwe Kiela.

SentEval: An evaluation toolkit for universal sentence
representations.

arXiv preprint arXiv:1803.05449, 2018.

- Conneau et al. (2017)

Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes.

Supervised learning of universal sentence representations from
natural language inference data.

arXiv preprint arXiv:1705.02364, 2017.

- Dagan et al. (2005)

Ido Dagan, Oren Glickman, and Bernardo Magnini.

The PASCAL recognising textual entailment challenge.

In Machine Learning Challenges Workshop, 2005.

- Dai and Le (2015)

Andrew M. Dai and Quoc V. Le.

Semi-supervised sequence learning.

In Advances in neural information processing systems, 2015.

- De Marneff et al. (2019)

Marie-Catherine De Marneff, Mandy Simons, and Judith Tonhauser.

The CommitmentBank: Investigating projection in naturally occurring
discourse.

In Sinn und Bedeutung 23, 2019.

- Deng et al. (2009)

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.

ImageNet: A large-scale hierarchical image database.

In 2009 IEEE conference on computer vision and pattern
recognition, 2009.

- Devlin et al. (2018)

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

BERT: Pre-training of deep bidirectional transformers for language
understanding.

arXiv preprint arXiv:1810.04805, 2018.

- Dolan and Brockett (2005)

William B. Dolan and Chris Brockett.

Automatically constructing a corpus of sentential paraphrases.

In Proceedings of the Third International Workshop on
Paraphrasing (IWP2005), 2005.

- Dong et al. (2019)

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao,
Ming Zhou, and Hsiao-Wuen Hon.

Unified language model pre-training for natural language
understanding and generation.

arXiv preprint arXiv:1905.03197, 2019.

- Edunov et al. (2018)

Sergey Edunov, Myle Ott, Michael Auli, and David Grangier.

Understanding back-translation at scale.

arXiv preprint arXiv:1808.09381, 2018.

- Grave et al. (2018)

Edouard Grave, Piotr Bojanowski, Prakhar Gupta, Armand Joulin, and Tomas
Mikolov.

Learning word vectors for 157 languages.

arXiv preprint arXiv:1802.06893, 2018.

- Graves (2013)

Alex Graves.

Generating sequences with recurrent neural networks.

arXiv preprint arXiv:1308.0850, 2013.

- Habernal et al. (2016)

Ivan Habernal, Omnia Zayed, and Iryna Gurevych.

C4Corpus: Multilingual web-size corpus with free license.

In Proceedings of the Tenth International Conference on
Language Resources and Evaluation (LREC’16), pages 914–922, 2016.

- He et al. (2016)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

Deep residual learning for image recognition.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, 2016.

- He et al. (2018)

Kaiming He, Ross Girshick, and Piotr Dollár.

Rethinking ImageNet pre-training.

arXiv preprint arXiv:1811.08883, 2018.

- He et al. (2019)

Pengcheng He, Xiaodong Liu, Weizhu Chen, and Jianfeng Gao.

A hybrid neural network model for commonsense reasoning.

arXiv preprint arXiv:1907.11983, 2019.

- Hermann et al. (2015)

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will
Kay, Mustafa Suleyman, and Phil Blunsom.

Teaching machines to read and comprehend.

In Advances in neural information processing systems, 2015.

- Hestness et al. (2017)

Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun,
Hassan Kianinejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou.

Deep learning scaling is predictable, empirically.

arXiv preprint arXiv:1712.00409, 2017.

- Hill et al. (2016)

Felix Hill, Kyunghyun Cho, and Anna Korhonen.

Learning distributed representations of sentences from unlabelled
data.

arXiv preprint arXiv:1602.03483, 2016.

- Hinton et al. (2015)

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.

Distilling the knowledge in a neural network.

arXiv preprint arXiv:1503.02531, 2015.

- Houlsby et al. (2019)

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin
De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.

Parameter-efficient transfer learning for NLP.

arXiv preprint arXiv:1902.00751, 2019.

- Howard and Ruder (2018)

Jeremy Howard and Sebastian Ruder.

Universal language model fine-tuning for text classification.

arXiv preprint arXiv:1801.06146, 2018.

- Huang et al. (2018a)

Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis
Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica
Dinculescu, and Douglas Eck.

Music transformer: Generating music with long-term structure.

In Seventh International Conference on Learning
Representations, 2018a.

- Huang et al. (2018b)

Yanping Huang, Yonglong Cheng, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V
Le, and Zhifeng Chen.

GPipe: Efficient training of giant neural networks using pipeline
parallelism.

arXiv preprint arXiv:1811.06965, 2018b.

- Huh et al. (2016)

Minyoung Huh, Pulkit Agrawal, and Alexei A. Efros.

What makes ImageNet good for transfer learning?

arXiv preprint arXiv:1608.08614, 2016.

- Iyer et al. (2017)

Shankar Iyer, Nikhil Dandekar, and Kornel Csernai.

First Quora dataset release: Question pairs.

https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs,
2017.

- Jia et al. (2014)

Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev, Jonathan Long, Ross
Girshick, Sergio Guadarrama, and Trevor Darrell.

Caffe: Convolutional architecture for fast feature embedding.

In Proceedings of the 22nd ACM international conference on
Multimedia, 2014.

- Jiao et al. (2019)

Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang
Wang, and Qun Liu.

TinyBERT: Distilling BERT for natural language understanding.

arXiv preprint arXiv:1909.10351, 2019.

- Joshi et al. (2017)

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer.

TriviaQA: A large scale distantly supervised challenge dataset for
reading comprehension.

arXiv preprint arXiv:1705.03551, 2017.

- Joshi et al. (2019)

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and
Omer Levy.

SpanBERT: Improving pre-training by representing and predicting
spans.

arXiv preprint arXiv:1907.10529, 2019.

- Jozefowicz et al. (2016)

Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu.

Exploring the limits of language modeling.

arXiv preprint arXiv:1602.02410, 2016.

- Kalchbrenner et al. (2014)

Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom.

A convolutional neural network for modelling sentences.

In Proceedings of the 52nd Annual Meeting of the Association
for Computational Linguistics, 2014.

- Keskar et al. (2019a)

Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and
Richard Socher.

CTRL: A conditional transformer language model for controllable
generation.

arXiv preprint arXiv:1909.05858, 2019a.

- Keskar et al. (2019b)

Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher.

Unifying question answering and text classification via span
extraction.

arXiv preprint arXiv:1904.09286, 2019b.

- Khashabi et al. (2018)

Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan
Roth.

Looking beyond the surface: A challenge set for reading comprehension
over multiple sentences.

In Proceedings of North American Chapter of the Association for
Computational Linguistics (NAACL), 2018.

- Kiros et al. (2015)

Ryan Kiros, Yukun Zhu, Ruslan R. Salakhutdinov, Richard Zemel, Raquel Urtasun,
Antonio Torralba, and Sanja Fidler.

Skip-thought vectors.

In Advances in neural information processing systems, 2015.

- Kocijan et al. (2019)

Vid Kocijan, Ana-Maria Cretu, Oana-Maria Camburu, Yordan Yordanov, and Thomas
Lukasiewicz.

A surprisingly robust trick for Winograd schema challenge.

arXiv preprint arXiv:1905.06290, 2019.

- Konečnỳ et al. (2015)

Jakub Konečnỳ, Brendan McMahan, and Daniel Ramage.

Federated optimization: Distributed optimization beyond the
datacenter.

arXiv preprint arXiv:1511.03575, 2015.

- Konečnỳ et al. (2016)

Jakub Konečnỳ, H. Brendan McMahan, Felix X. Yu, Peter Richtárik,
Ananda Theertha Suresh, and Dave Bacon.

Federated learning: Strategies for improving communication
efficiency.

arXiv preprint arXiv:1610.05492, 2016.

- Kornblith et al. (2018)

Simon Kornblith, Jonathon Shlens, and Quoc V. Le.

Do better ImageNet models transfer better?

arXiv preprint arXiv:1805.08974, 2018.

- Krizhevsky (2014)

Alex Krizhevsky.

One weird trick for parallelizing convolutional neural networks.

arXiv preprint arXiv:1404.5997, 2014.

- Kudo (2018)

Taku Kudo.

Subword regularization: Improving neural network translation models
with multiple subword candidates.

arXiv preprint arXiv:1804.10959, 2018.

- Kudo and Richardson (2018)

Taku Kudo and John Richardson.

SentencePiece: A simple and language independent subword tokenizer
and detokenizer for neural text processing.

arXiv preprint arXiv:1808.06226, 2018.

- Lample and Conneau (2019)

Guillaume Lample and Alexis Conneau.

Cross-lingual language model pretraining.

arXiv preprint arXiv:1901.07291, 2019.

- Lan et al. (2019)

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and
Radu Soricut.

ALBERT: A lite BERT for self-supervised learning of language
representations.

arXiv preprint arXiv:1909.11942, 2019.

- Levesque et al. (2012)

Hector Levesque, Ernest Davis, and Leora Morgenstern.

The Winograd schema challenge.

In Thirteenth International Conference on the Principles of
Knowledge Representation and Reasoning, 2012.

- Li (2012)

Qi Li.

Literature survey: domain adaptation algorithms for natural language
processing.

2012.

- Lin (2004)

Chin-Yew Lin.

ROUGE: A package for automatic evaluation of summaries.

In Text summarization branches out, 2004.

- Liu et al. (2018)

Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz
Kaiser, and Noam Shazeer.

Generating Wikipedia by summarizing long sequences.

arXiv preprint arXiv:1801.10198, 2018.

- Liu et al. (2019a)

Peter J. Liu, Yu-An Chung, and Jie Ren.

SummAE: Zero-shot abstractive text summarization using
length-agnostic auto-encoders.

arXiv preprint arXiv:1910.00998, 2019a.

- Liu et al. (2015)

Xiaodong Liu, Jianfeng Gao, Xiaodong He, Li Deng, Kevin Duh, and Ye-Yi Wang.

Representation learning using multi-task deep neural networks for
semantic classification and information retrieval.

In Proceedings of the 2015 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, 2015.

- Liu et al. (2019b)

Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao.

Multi-task deep neural networks for natural language understanding.

arXiv preprint arXiv:1901.11504, 2019b.

- Liu (2019)

Yang Liu.

Fine-tune BERT for extractive summarization.

arXiv preprint arXiv:1903.10318, 2019.

- Liu et al. (2019c)

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.

RoBERTa: A robustly optimized BERT pretraining approach.

arXiv preprint arXiv:1907.11692, 2019c.

- Logeswaran and Lee (2018)

Lajanugen Logeswaran and Honglak Lee.

An efficient framework for learning sentence representations.

arXiv preprint arXiv:1803.02893, 2018.

- Mahajan et al. (2018)

Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri,
Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten.

Exploring the limits of weakly supervised pretraining.

In Proceedings of the European Conference on Computer Vision
(ECCV), 2018.

- McCann et al. (2018)

Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher.

The natural language decathlon: Multitask learning as question
answering.

arXiv preprint arXiv:1806.08730, 2018.

- Mikolov et al. (2013a)

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.

Efficient estimation of word representations in vector space.

arXiv preprint arXiv:1301.3781, 2013a.

- Mikolov et al. (2013b)

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean.

Distributed representations of words and phrases and their
compositionality.

In Advances in neural information processing systems,
2013b.

- Nallapati et al. (2016)

Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, and
Bing Xiang.

Abstractive text summarization using sequence-to-sequence RNNs and
beyond.

arXiv preprint arXiv:1602.06023, 2016.

- Oquab et al. (2014)

Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic.

Learning and transferring mid-level image representations using
convolutional neural networks.

In Proceedings of the IEEE conference on computer vision and
pattern recognition, 2014.

- Papineni et al. (2002)

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu.

BLEU: a method for automatic evaluation of machine translation.

In Proceedings of the 40th annual meeting on association for
computational linguistics. Association for Computational Linguistics, 2002.

- Paulus et al. (2017)

Romain Paulus, Caiming Xiong, and Richard Socher.

A deep reinforced model for abstractive summarization.

arXiv preprint arXiv:1705.04304, 2017.

- Pennington et al. (2014)

Jeffrey Pennington, Richard Socher, and Christopher Manning.

GloVe: Global vectors for word representation.

In Proceedings of the 2014 conference on empirical methods in
natural language processing (EMNLP), 2014.

- Peters et al. (2019)

Matthew Peters, Sebastian Ruder, and Noah A. Smith.

To tune or not to tune? adapting pretrained representations to
diverse tasks.

arXiv preprint arXiv:1903.05987, 2019.

- Peters et al. (2018)

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark,
Kenton Lee, and Luke Zettlemoyer.

Deep contextualized word representations.

arXiv preprint arXiv:1802.05365, 2018.

- Phang et al. (2018)

Jason Phang, Thibault Févry, and Samuel R. Bowman.

Sentence encoders on STILTs: Supplementary training on intermediate
labeled-data tasks.

arXiv preprint arXiv:1811.01088, 2018.

- Pilehvar and Camacho-Collados (2018)

Mohammad Taher Pilehvar and Jose Camacho-Collados.

WIC: 10,000 example pairs for evaluating context-sensitive
representations.

arXiv preprint arXiv:1808.09121, 2018.

- Post (2018)

Matt Post.

A call for clarity in reporting BLEU scores.

arXiv preprint arXiv:1804.08771, 2018.

- Radford et al. (2018)

Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever.

Improving language understanding by generative pre-training, 2018.

- Radford et al. (2019)

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever.

Language models are unsupervised multitask learners, 2019.

- Rahman and Ng (2012)

Altaf Rahman and Vincent Ng.

Resolving complex cases of definite pronouns: the Winograd schema
challenge.

In Proceedings of the 2012 Joint Conference on Empirical
Methods in Natural Language Processing and Computational Natural Language
Learning. Association for Computational Linguistics, 2012.

- Rajpurkar et al. (2016)

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.

Squad: 100,000+ questions for machine comprehension of text.

arXiv preprint arXiv:1606.05250, 2016.

- Ramachandran et al. (2016)

Prajit Ramachandran, Peter J. Liu, and Quoc V. Le.

Unsupervised pretraining for sequence to sequence learning.

arXiv preprint arXiv:1611.02683, 2016.

- Ratner et al. (2018)

Alex Ratner, Braden Hancock, Jared Dunnmon, Roger Goldman, and Christopher
Ré.

Snorkel MeTaL: Weak supervision for multi-task learning.

In Proceedings of the Second Workshop on Data Management for
End-To-End Machine Learning, 2018.

- Roemmele et al. (2011)

Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon.

Choice of plausible alternatives: An evaluation of commonsense causal
reasoning.

In 2011 AAAI Spring Symposium Series, 2011.

- Ruder (2017)

Sebastian Ruder.

An overview of multi-task learning in deep neural networks.

arXiv preprint arXiv:1706.05098, 2017.

- Ruder (2019)

Sebastian Ruder.

Neural transfer learning for natural language processing.

PhD thesis, NUI Galway, 2019.

- Ruder et al. (2019)

Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, and Thomas Wolf.

Transfer learning in natural language processing.

In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Tutorials, pages
15–18, 2019.

- Russakovsky et al. (2015)

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma,
Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al.

ImageNet large scale visual recognition challenge.

International journal of computer vision, 2015.

- Sanh et al. (2019)

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.

DistilBERT, a distilled version of BERT: smaller, faster, cheaper
and lighter.

arXiv preprint arXiv:1910.01108, 2019.

- See et al. (2017)

Abigail See, Peter J. Liu, and Christopher D. Manning.

Get to the point: Summarization with pointer-generator networks.

arXiv preprint arXiv:1704.04368, 2017.

- Sennrich et al. (2015)

Rico Sennrich, Barry Haddow, and Alexandra Birch.

Neural machine translation of rare words with subword units.

arXiv preprint arXiv:1508.07909, 2015.

- Shallue et al. (2018)

Christopher J Shallue, Jaehoon Lee, Joe Antognini, Jascha Sohl-Dickstein, Roy
Frostig, and George E. Dahl.

Measuring the effects of data parallelism on neural network training.

arXiv preprint arXiv:1811.03600, 2018.

- Shaw et al. (2018)

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.

Self-attention with relative position representations.

arXiv preprint arXiv:1803.02155, 2018.

- Shazeer and Stern (2018)

Noam Shazeer and Mitchell Stern.

Adafactor: Adaptive learning rates with sublinear memory cost.

arXiv preprint arXiv:1804.04235, 2018.

- Shazeer et al. (2017)

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le,
Geoffrey Hinton, and Jeff Dean.

Outrageously large neural networks: The sparsely-gated
mixture-of-experts layer.

arXiv preprint arXiv:1701.06538, 2017.

- Shazeer et al. (2018)

Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn
Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young,
Ryan Sepassi, and Blake Hechtman.

Mesh-tensorflow: Deep learning for supercomputers.

In Advances in Neural Information Processing Systems, 2018.

- Smith et al. (2013)

Jason R. Smith, Herve Saint-Amand, Magdalena Plamada, Philipp Koehn, Chris
Callison-Burch, and Adam Lopez.

Dirt cheap web-scale parallel text from the common crawl.

In Proceedings of the 51st Annual Meeting of the Association
for Computational Linguistics, 2013.

- Socher et al. (2013)

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning,
Andrew Ng, and Christopher Potts.

Recursive deep models for semantic compositionality over a sentiment
treebank.

In Proceedings of the 2013 conference on empirical methods in
natural language processing, 2013.

- Song et al. (2019)

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu.

MASS: Masked sequence to sequence pre-training for language
generation.

arXiv preprint arXiv:1905.02450, 2019.

- Srivastava et al. (2014)

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan
Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting.

The Journal of Machine Learning Research, 2014.

- Subramanian et al. (2018)

Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J. Pal.

Learning general purpose distributed sentence representations via
large scale multi-task learning.

arXiv preprint arXiv:1804.00079, 2018.

- Sutskever et al. (2014)

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.

Sequence to sequence learning with neural networks.

In Advances in neural information processing systems, 2014.

- Sutton (2019)

Richard S. Sutton.

The bitter lesson.

http://www.incompleteideas.net/IncIdeas/BitterLesson.html,
2019.

- Taylor (1953)

Wilson L. Taylor.

“Cloze procedure”: A new tool for measuring readability.

Journalism Bulletin, 1953.

- Trinh and Le (2018)

Trieu H. Trinh and Quoc V. Le.

A simple method for commonsense reasoning.

arXiv preprint arXiv:1806.02847, 2018.

- Trischler et al. (2016)

Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni,
Philip Bachman, and Kaheer Suleman.

NewsQA: A machine comprehension dataset.

arXiv preprint arXiv:1611.09830, 2016.

- Vaswani et al. (2017)

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In Advances in neural information processing systems, 2017.

- Voita et al. (2019)

Elena Voita, Rico Sennrich, and Ivan Titov.

The bottom-up evolution of representations in the transformer: A
study with machine translation and language modeling objectives.

arXiv preprint arXiv:1909.01380, 2019.

- Wang et al. (2018)

Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R.
Bowman.

GLUE: A multi-task benchmark and analysis platform for natural
language understanding.

arXiv preprint arXiv:1804.07461, 2018.

- Wang et al. (2019a)

Alex Wang, Jan Hula, Patrick Xia, Raghavendra Pappagari, R. Thomas McCoy, Roma
Patel, Najoung Kim, Ian Tenney, Yinghui Huang, Katherin Yu, et al.

Can you tell me how to get past Sesame Street? Sentence-level
pretraining beyond language modeling.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, 2019a.

- Wang et al. (2019b)

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael,
Felix Hill, Omer Levy, and Samuel R. Bowman.

SuperGLUE: A stickier benchmark for general-purpose language
understanding systems.

arXiv preprint arXiv:1905.00537, 2019b.

- Wang et al. (2019c)

Wei Wang, Bin Bi, Ming Yan, Chen Wu, Zuyi Bao, Liwei Peng, and Luo Si.

StructBERT: Incorporating language structures into pre-training for
deep language understanding.

arXiv preprint arXiv:1908.04577, 2019c.

- Warstadt et al. (2018)

Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman.

Neural network acceptability judgments.

arXiv preprint arXiv:1805.12471, 2018.

- Williams et al. (2017)

Adina Williams, Nikita Nangia, and Samuel R. Bowman.

A broad-coverage challenge corpus for sentence understanding through
inference.

arXiv preprint arXiv:1704.05426, 2017.

- Williams and Zipser (1989)

Ronald J. Williams and David Zipser.

A learning algorithm for continually running fully recurrent neural
networks.

Neural computation, 1989.

- Wu et al. (2016)

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang
Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al.

Google’s neural machine translation system: Bridging the gap between
human and machine translation.

arXiv preprint arXiv:1609.08144, 2016.

- Yang et al. (2019)

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov,
and Quoc V. Le.

XLNet: Generalized autoregressive pretraining for language
understanding.

arXiv preprint arXiv:1906.08237, 2019.

- Yosinski et al. (2014)

Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson.

How transferable are features in deep neural networks?

In Advances in neural information processing systems, 2014.

- Yu et al. (2018)

Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad
Norouzi, and Quoc V. Le.

QAnet: Combining local convolution with global self-attention for
reading comprehension.

arXiv preprint arXiv:1804.09541, 2018.

- Zellers et al. (2019)

Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi,
Franziska Roesner, and Yejin Choi.

Defending against neural fake news.

arXiv preprint arXiv:1905.12616, 2019.

- Zhang et al. (2018)

Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin
Van Durme.

ReCoRD: Bridging the gap between human and machine commonsense
reading comprehension.

arXiv preprint arXiv:1810.12885, 2018.

- Zhu et al. (2019)

Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Thomas Goldstein, and Jingjing Liu.

Freelb: Enhanced adversarial training for language understanding.

arXiv preprint arXiv:1909.11764, 2019.

- Zhu et al. (2015)

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun,
Antonio Torralba, and Sanja Fidler.

Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books.

In Proceedings of the IEEE international conference on computer
vision, 2015.

Generated on Wed Mar 13 11:12:02 2024 by LaTeXML
