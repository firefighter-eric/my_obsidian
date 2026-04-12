# Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Sakata et al. - 2019 - FAQ retrieval using query-question similarity and BERT-based query-answer relevance.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/1905.02851
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# FAQ Retrieval using Query-Question Similarity and
BERT-Based Query-Answer Relevance

Wataru Sakata

1234-5678-9012

LINE Corporation

wataru.sakata@linecorp.com

, 
Tomohide Shibata

Kyoto UniversityCREST, JST

shibata@nlp.ist.i.kyoto-u.ac.jp

, 
Ribeka Tanaka

Kyoto UniversityCREST, JST

tanaka@nlp.ist.i.kyoto-u.ac.jp

 and 
Sadao Kurohashi

Kyoto UniversityCREST, JSTNII, CRIS

kuro@nlp.ist.i.kyoto-u.ac.jp

(2019)

###### Abstract.

Frequently Asked Question (FAQ) retrieval is an important task where
the objective is to retrieve an appropriate Question-Answer (QA) pair
from a database based on a user’s query. We propose a FAQ retrieval
system that considers the similarity between a user’s query and a
question as well as the relevance between the query and an
answer. Although a common approach to FAQ retrieval is to construct
labeled data for training, it takes annotation costs. Therefore, we
use a traditional unsupervised information retrieval system to
calculate the similarity between the query and question. On the other
hand, the relevance between the query and answer can be learned by
using QA pairs in a FAQ database. The recently-proposed BERT model is
used for the relevance calculation. Since the number of QA pairs in
FAQ page is not enough to train a model, we cope with this issue by
leveraging FAQ sets that are similar to the one in question. We
evaluate our approach on two datasets. The first one is localgovFAQ, a
dataset we construct in a Japanese administrative municipality domain.
The second is StackExchange dataset, which is the public dataset in
English. We demonstrate that our proposed method outperforms baseline
methods on these datasets.

## 1. Introduction

There are often frequently asked question (FAQ) pages with various information on the web.
A FAQ retrieval system, which takes a user’s query and returns relevant QA pairs, is useful for navigating these pages.

In FAQ retrieval tasks,
it is standard to check similarities of user’s query (q𝑞q) to a FAQ’s question (Q𝑄Q) or
to a question-answer (QA) pair (Karan and Snajder, 2018).
Many FAQ retrieval models use the dataset with the relevance label between q𝑞q and a QA pair.
However, it costs a lot to construct such labeled data.
To cope with this problem, we adopt an unsupervised method for calculating the similarity between a query and a question.

Another promising approach is to check the q-A relevance
trained by QA pairs,
which shows the plausibility of the FAQ answer for the given q𝑞q.
Studies of community QA use a large number of QA pairs for learning the q-A relevance (Tan
et al., 2015; Wu
et al., 2018).
However, these methods do not apply to FAQ retrieval task, because the size of QA entries in FAQ is not enough to train a model generally.
We address this problem by collecting other similar FAQ sets to increase the size of available QA data.

In this study, we propose a method that combines
the q-Q similarity obtained by unsupervised model and the q-A relevance learned from the collected QA pairs.
Figure 1 shows the proposed model.
Previous studies show that neural methods (e.g., LSTM and CNN)
work effectively
in learning q-A relevance.
Here we use the recently-proposed model, BERT (Devlin
et al., 2018).
BERT is a powerful model that applies to
a wide range of tasks and obtains the state-of-the-art results
on many tasks including GLUE (Wang et al., 2018) and SQuAD (Rajpurkar et al., 2016).
An unsupervised retrieval system achieves high precision,
but it is difficult to deal with a gap between the expressions of q𝑞q and Q𝑄Q.
By contrast, since BERT validates the relevance between q𝑞q and A𝐴A,
it can retrieve an appropriate QA pair even if there is a lexical gap between q𝑞q and Q𝑄Q.
By combining the characteristics of two models, we achieve a robust and high-performance retrieval system.

\Description

An overview

We conduct experiments on two datasets.
The first one is the localgovFAQ dataset,
which we construct to evaluate our model
in a setting where other similar FAQ sets are available.
It consists of QA pairs collected from Japanese local government FAQ pages
and an evaluation set constructed via crowdsourcing.
The second one is the StackExchange dataset (Karan and Snajder, 2018),
which is the public dataset for FAQ retrieval tasks.
We evaluate our model on these datasets and show that the proposed method works effectively in FAQ retrieval.

## 2. Proposed Method

### 2.1. Task Description

We begin by formally defining the task of FAQ retrieval.
Here, we focus on local government FAQ as an example.
Suppose that the number of local government FAQ sets is N𝑁N.
Our target FAQ set, Ttsubscript𝑇𝑡T_{t}, is one of them.
When the number of QA entries in Ttsubscript𝑇𝑡T_{t} is M𝑀M, Ttsubscript𝑇𝑡T_{t} is a collection of QA pairs {(Q1,A1),(Q2,A2),…,(QM,AM)}subscript𝑄1subscript𝐴1subscript𝑄2subscript𝐴2…subscript𝑄𝑀subscript𝐴𝑀\{(Q_{1},A_{1}),(Q_{2},A_{2}),...,(Q_{M},A_{M})\}.
The task is then to find the appropriate QA pair (Qi,Ai)subscript𝑄𝑖subscript𝐴𝑖(Q_{i},A_{i})
from Ttsubscript𝑇𝑡T_{t} based on a user’s query q𝑞q.
We use T1,T2,…,TNsubscript𝑇1subscript𝑇2…subscript𝑇𝑁T_{1},T_{2},...,T_{N} as our training data,
including the FAQ set Ttsubscript𝑇𝑡T_{t} of the target local government.

### 2.2. q-Q similarity by TSUBAKI

We use TSUBAKI (Shinzato et al., 2008) to
compute q-Q similarity. TSUBAKI is an unsupervised retrieval engine based on OKAPI BM25 (Robertson et al., 1992).
TSUBAKI accounts for a dependency structure of a sentence, not just its words, to provide accurate retrieval.
For flexible matching, it also uses synonyms automatically extracted from dictionaries
and Web corpus.
Here we regard Q𝑄Q in each QA as a document and compute S​i​m​i​l​a​r​i​t​y​(q,Q)𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦𝑞𝑄Similarity(q,Q) for
the q-Q similarity.

### 2.3. q-A relevance by BERT

We use BERT to compute q-A relevance.
BERT is based on the Transformer (Vaswani et al., 2017) that effectively encodes an input text. It is designed to be pre-trained using a language model objective on
a large raw corpus and fine-tuned for each specific task including sentence classification, sentence-pair classification, and question answering.
As it is pre-trained on a large corpus, BERT achieves high accuracy even if the data size of the specific task is not large enough. We apply BERT to a sentence-pair classifier for questions and answers. By applying the Transformer to the input question and answer, it captures the relevance between the pair.

The training data we use is
the collection of QA pairs from FAQ sets (see Sec. 2.1).
For each positive example (Q,A)𝑄𝐴(Q,A),
we randomly select A¯¯𝐴\bar{A}
and produce negative training data (Q,A¯)𝑄¯𝐴(Q,\bar{A}).
On this data, we train BERT to solve the two-class classification problem:
R​e​l​e​v​a​n​c​e​(Q,A)𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑄𝐴Relevance(Q,A) is 1 and R​e​l​e​v​a​n​c​e​(Q,A¯)𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑄¯𝐴Relevance(Q,\bar{A}) is 0,
where R​e​l​e​v​a​n​c​e​(Q,A)𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑄𝐴Relevance(Q,A) stands for the relevance between Q𝑄Q and A𝐴A.

At the search stage, we compute R​e​l​e​v​a​n​c​e​(q,Ai)𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑞subscript𝐴𝑖Relevance(q,A_{i})
(i=1,⋯,M)𝑖1⋯𝑀(i=1,\cdots,M) for the user’s query q𝑞q and every QA pair in the target Ttsubscript𝑇𝑡T_{t}.
QA pairs in a higher rank are used as search results.

### 2.4. Combining TSUBAKI and BERT

In order to realize robust and flexible matching, we combine the q-Q similarity by TSUBAKI and the q-A relevance by BERT.

When TSUBAKI’s similarity score is high,
this is probably a positive case because the words in q𝑞q and Q𝑄Q highly overlap with each other.
However, it is difficult to cope with the lexical gaps between q𝑞q and Q𝑄Q.
On the other hand, since BERT validates the relevance between q𝑞q and A𝐴A,
it can retrieve an appropriate QA pair even if there is a lexical gap between q𝑞q and Q𝑄Q.
To make use of these characteristics, we combine two methods as follows.
First, we take the ten-highest results of BERT’s output.
For QA pairs whose TSUBAKI score gets a higher score than α𝛼\alpha, we rank them in order of TSUBAKI’s score.
For the others, we rank them in order of the score of S​i​m​i​l​a​r​i​t​y​(q,Q)×t+R​e​l​e​v​a​n​c​e​(q,A)𝑆𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦𝑞𝑄𝑡𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑞𝐴Similarity(q,Q)\times t+Relevance(q,A) where t𝑡t is a hyper-parameter.

TSUBAKI’s score tends to be higher when the given query is longer. Hence, before taking the sum, we normalize TSUBAKI’s score by using the numbers of content words and dependency relations in the query. We divide the original score by the following value.111
We do not normalize the BERT’s score because it takes a value between 0 to 1.

C​o​u​n​t​(𝐶𝑜𝑛𝑡𝑒𝑛𝑡𝑊𝑜𝑟𝑑𝑠)×k1+C​o​u​n​t​(𝐷𝑒𝑝𝑒𝑛𝑑𝑒𝑛𝑐𝑦𝑅𝑒𝑙𝑎𝑡𝑖𝑜𝑛𝑠)×k2𝐶𝑜𝑢𝑛𝑡𝐶𝑜𝑛𝑡𝑒𝑛𝑡𝑊𝑜𝑟𝑑𝑠subscript𝑘1𝐶𝑜𝑢𝑛𝑡𝐷𝑒𝑝𝑒𝑛𝑑𝑒𝑛𝑐𝑦𝑅𝑒𝑙𝑎𝑡𝑖𝑜𝑛𝑠subscript𝑘2Count(\mathit{ContentWords})\times k_{1}+Count(\mathit{DependencyRelations})\times k_{2}

## 3. Experiments and Evaluation

We conducted our experiments on two datasets, localgovFAQ and StackExchange.
We constructed localgovFAQ dataset, as explained in Sec 3.1.
StackExchange dataset is constructed in the paper (Karan and Snajder, 2018)
by extracting QA pairs from the web apps domain of StackExchange and consists of 719 QA pairs. Each Q𝑄Q has paraphrase queries, and the total number of queries is 1,250. All the models were evaluated using five-fold cross validation. In each validation, all the queries were split into training (60%), development (20%) and test (20%). The task is to estimate an appropriate QA pair for each query q𝑞q among 719 QA pairs.

### 3.1. LocalgovFAQ Evaluation Set Construction

• 

I’d like you to issue a copy of family register, but how much does it cost?

• 

I’d like you to publish a maternal and child health handbook, but what is required for the procedure?

• 

I’m thinking of purchasing a new housing, so I want to know about the reduction measure.

• 

From which station does the pick-up bus of the Center Pool come out?

Amagasaki-city, a relatively large city in Japan, was chosen as a target government, whose Web site has 1,786 QA pairs. First, queries to this government were collected using a crowdsourcing. Example queries are shown in Figure 2. We collected 990 queries in total.

TSUBAKI and BERT output at most five relevant QA pairs for each query, and each QA pair was manually evaluated assigning one of the following four categories:

A:

Contain correct information.

Contain relevant information.

The topic is same as a query, but do not contain relevant information.

Contain only irrelevant information.

In general, information retrieval evaluation based on the pooling method has inherently a biased problem.
To alleviate this problem, when there are no relevant QA pairs among the outputs by TSUBAKI and BERT, a correct QA pair was searched by using appropriate different keywords. If there are no relevant QA pair found, this query was excluded from our evaluation set. The resultant queries were 784.
Since 20% of queries were used for the development set, 627 queries were used for our evaluation.

### 3.2. Experimental Settings

For the localgovFAQ dataset, MAP (Mean Average Precision), MRR (Mean Reciprocal Rank), P@5 (Precision at 5), SR@k (Success Rate)222Success Rate is
the fraction of questions for which at least one related question is
ranked among the top k𝑘k. and nDCG (normalized Discounted Cumulative
Gain) were used as our evaluation measures.
The categories A, B and C
were regarded as correct for MAP, MRR, P@5 and SR@k, and the evaluation level of
categories A, B and C was regarded as 3, 2 and 1 for nDCG, respectively.
For the StackExchange dataset, MAP, MRR and P@5 were used, following Karan et al. (Karan and Snajder, 2018) .

For Japanese, the pre-training of BERT was performed using Japanese Wikipedia, which
consists of approximately 18M sentences, and the fine-tuning was
performed using FAQs of 21 Japanese local governments, which consists of
approximately 20K QA pairs. The morphological analyzer Juman++333http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?JUMAN++ was
applied to input texts for word segmentation, and words were broken into subwords by applying BPE (Sennrich
et al., 2016).
For English BERT pre-trained model, a publicly-available model was used444https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip. For the fine-tuning for StackExchange dataset, the training set (q,Q,A)𝑞𝑄𝐴(q,Q,A) was divided into (q,A)𝑞𝐴(q,A) and (Q,A)𝑄𝐴(Q,A).

In the localgovFAQ dataset, Bi-LSTM with attention (Tan
et al., 2015) was adopted as our baseline.
We also used model BERTtargetOnly, which is fine-tuned only with the target FAQ set, in order to test the effect of using other FAQ sets.
In the StackExchange dataset, CNN-rank in q-Q and q-QA settings, which is the neural FAQ retrieval model based on a convolutional neural network, was used, whose scores were from Karan et al. (Karan and Snajder, 2018).
Furthermore, BERT (w/o query paraphrases) was adopted, where (q,A)𝑞𝐴(q,A) pairs were not used for BERT training, to see the performance
when no manually-assigned query paraphrases were available.
For both datasets, TSUBAKI was applied only in q-Q setting555We omit the TSUBAKI’s results in the q-A setting and the q-QA setting as we got the worse scores than the q-Q setting..

For both BERT and Bi-LSTM models, 24 negative samples for one positive sample were used. For the coefficients explained in Sec. 2.4, k1subscript𝑘1k_{1} and k2subscript𝑘2k_{2} were set to 4 and 2, respectively, and α𝛼\alpha and t𝑡t were set to 0.3 and 10 respectively using the development set.

### 3.3. Evaluation Results and Discussions

Table 1 shows an experimental result on localgovFAQ dataset.
In q-A setting, BERT was better than the Bi-LSTM baseline, which indicates BERT was useful for this task.
Although the performances of TSUBAKI and BERT were almost the same in terms of SR@1, the performance of BERT was better than TSUBAKI in terms of SR@5, which indicates BERT could retrieve a variety of QA pairs.
The proposed method performed the best. This demonstrated the effectiveness of our proposed method.
The score of BERT was better than one of BERTtargetOnly, which indidates that using other FAQ sets is effective.

Table 2 shows an experimental result on StackExchange dataset. In the same way as the result on localgovFAQ, BERT performed well, and the proposed method performed the best in terms of all the measures. The performance of BERT was better than one of ”BERT (w/o query paraphrases)”, which indicates that the use of various augmented questions was effective.

Figure 3 shows the performance of TSUBAKI and BERT on localgovFAQ according to their TOP1 scores.
From this figure, we can find that in the retrieved QA pair whose TSUBAKI score is high, its accuracy is very high. On the otherhand, there is a relatively loose correlation between the accuracy and BERT score.
This indicates TSUBAKI and BERT have different characteristics, and our proposed combining method is reasonable.

Model
MAP
MRR
P@5
SR@1
SR@5
NDCG

q-Q
TSUBAKI
0.558
0.598
0.297
0.504
0.734
0.501

q-A
Bi-LSTM
0.451
0.498
0.248
0.379
0.601
0.496

BERTtargetOnly

0.559
0.610
0.285
0.504
0.751
0.526

BERT
0.576
0.631
0.333
0.509
0.810
0.560

q-QA
Proposed
0.647
0.705
0.357
0.612
0.841
0.621

Model
MAP
MRR
P@5

q-Q
CNN-rank
0.79
0.77
0.63

TSUBAKI
0.698
0.669
0.638

q-A
BERT (w/o query paraphrases)
0.631
0.805
0.546

BERT
0.887
0.936
0.770

q-QA
CNN-rank
0.74
0.84
0.62

Proposed
0.897
0.942
0.776

\Description

comparison

Query
TSUBAKI
BERT
Proposed method

Is there a consultation desk for workplace harassment?

1

×\times

1

✓

1

✓

Where should I renew my license?

1

×\times

1

×\times

1

✓

2

✓

2

×\times

2

×\times

Is there a place that we can use for practicing instruments?

1

×\times

1

×\times

1

×\times

Table 3 shows the examples, translated from Japanese, of system outputs and correct QA pairs on localgovFAQ. In the first example, although TSUBAKI retrieved the wrong QA pair since there is a word ”consultation” and ”counseling” in the query and Q𝑄Q, BERT and the proposed method could retrieve a correct QA pair.
In the second example, the proposed method could retrieve a correct QA pair on the first rank although the first rank of TSUBAKI and BERT was wrong.

In the third example, no methods could retrieve a correct QA pair.
Although BERT could capture the relevance between a word ”instruments” in the query and ”music” in A𝐴A, the retrieved QA pair was wrong.
The correct QA pair consists of Q𝑄Q saying ”Information on the facility of the youth center, hours of use, and closed day” and A𝐴A mentioning a music room as one of the available facilities in the youth center. To retrieve this correct QA pair, the deeper understanding of QA texts is necessary.

It takes about 2 seconds to retrieve QA pairs per query on localgovFAQ dataset by using 7 GPUs (TITAN X Pascal), and our model is practical enough.
For a larger FAQ set, one can use our method in a telescoping setting (Matveeva et al., 2006).

## 4. Conclusion

This paper presented a method for using query-question similarity and BERT-based query-answer relevance in a FAQ retrieval task.
By collecting other similar FAQ sets, we could increase the size of available QA data.
BERT, which has been recently proposed, was applied to capture the relevance between queries and answers. This method realized the robust and high-performance retrieval.
The experimental results demonstrated that our combined use of query-question similarity and query-answer relevance was effective.
We are planning to make the code and constructed dataset localgovFAQ publicly available666http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?BERT-Based_FAQ_Retrieval.

## Acknowledgment

This work was partly supported by JST CREST Grant Number JPMJCR1301, Japan.

## References

- (1)

- Devlin
et al. (2018)

Jacob Devlin, Ming-Wei
Chang, Kenton Lee, and Kristina
Toutanova. 2018.

BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding.

CoRR abs/1810.04805
(2018).

arXiv:1810.04805

- Karan and Snajder (2018)

Mladen Karan and Jan
Snajder. 2018.

Paraphrase-focused learning to rank for
domain-specific frequently asked questions retrieval.

Expert Systems with Applications
91 (2018), 418–433.

- Matveeva et al. (2006)

Irina Matveeva, Chris
Burges, Timo Burkard, Andy Laucius,
and Leon Wong. 2006.

High Accuracy Retrieval with Multiple Nested
Ranker. In SIGIR. ACM,
437–444.

- Rajpurkar et al. (2016)

Pranav Rajpurkar, Jian
Zhang, Konstantin Lopyrev, and Percy
Liang. 2016.

SQuAD: 100,000+ Questions for Machine Comprehension
of Text. In EMNLP.
Association for Computational Linguistics,
2383–2392.

- Robertson et al. (1992)

Stephen E. Robertson,
Steve Walker, Micheline Hancock-Beaulieu,
Aarron Gull, and Marianna Lau.
1992.

Okapi at TREC. In
TREC. 21–30.

- Sennrich
et al. (2016)

Rico Sennrich, Barry
Haddow, and Alexandra Birch.
2016.

Neural Machine Translation of Rare Words with
Subword Units. In ACL.
Association for Computational Linguistics,
1715–1725.

- Shinzato et al. (2008)

Keiji Shinzato, Tomohide
Shibata, Daisuke Kawahara, Chikara
Hashimoto, and Sadao Kurohashi.
2008.

TSUBAKI: An Open Search Engine Infrastructure for
Developing New Information Access Methodology. In
IJCNLP. 189–196.

- Tan
et al. (2015)

Ming Tan, Bing Xiang,
and Bowen Zhou. 2015.

LSTM-based Deep Learning Models for Non-factoid
Answer Selection.

CoRR abs/1511.04108
(2015).

arXiv:1511.04108

- Vaswani et al. (2017)

Ashish Vaswani, Noam
Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N Gomez,
Lukasz Kaiser, and Illia Polosukhin.
2017.

Attention is All you Need.

In NIPS. 5998–6008.

- Wang et al. (2018)

Alex Wang, Amanpreet
Singh, Julian Michael, Felix Hill,
Omer Levy, and Samuel Bowman.
2018.

GLUE: A Multi-Task Benchmark and Analysis Platform
for Natural Language Understanding. In EMNLP2018
Workshop BlackboxNLP. Association for Computational
Linguistics, 353–355.

- Wu
et al. (2018)

Yu Wu, Wei Wu,
Zhoujun Li, and Ming Zhou.
2018.

Learning Matching Models with Weak Supervision for
Response Selection in Retrieval-based Chatbots. In
ACL. 420–425.

Generated on Sat Mar 2 17:39:12 2024 by LaTeXML
