# Smock, Pesala, Abraham - 2023 - Aligning Benchmark Datasets for Table Structure Recognition

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Smock, Pesala, Abraham - 2023 - Aligning Benchmark Datasets for Table Structure Recognition.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2303.00716
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

11institutetext: Microsoft, Redmond WA, USA
11email: {brsmock,ropesala,robin.abraham}@microsoft.com

# Aligning benchmark datasets for table structure recognition

Brandon Smock
11
0009-0002-7002-0800

  
Rohith Pesala
11
0009-0004-7373-853X

  
Robin Abraham
11
0000-0003-1915-8118

###### Abstract

Benchmark datasets for table structure recognition (TSR) must be carefully processed to ensure they are annotated consistently.
However, even if a dataset’s annotations are self-consistent, there may be significant inconsistency across datasets, which can harm the performance of models trained and evaluated on them.
In this work, we show that aligning these benchmarks—removing both errors and inconsistency between them—improves model performance significantly.
We demonstrate this through a data-centric approach where we adopt one model architecture, the Table Transformer (TATR), that we hold fixed throughout.
Baseline exact match accuracy for TATR evaluated on the ICDAR-2013 benchmark is 65% when trained on PubTables-1M, 42% when trained on FinTabNet, and 69% combined.
After reducing annotation mistakes and inter-dataset inconsistency, performance of TATR evaluated on ICDAR-2013 increases substantially to 75% when trained on PubTables-1M, 65% when trained on FinTabNet, and 81% combined.
We show through ablations over the modification steps that canonicalization of the table annotations has a significantly positive effect on performance, while other choices balance necessary trade-offs that arise when deciding a benchmark dataset’s final composition.
Overall we believe our work has significant implications for benchmark design for TSR and potentially other tasks as well.
Dataset processing and training code will be released at https://github.com/microsoft/table-transformer.

## 1 Introduction

Table extraction (TE) is a long-standing problem in document intelligence.
Over the last decade, steady progress has been made formalizing TE as a machine learning (ML) task.
This includes the development of task-specific metrics for evaluating table structure recognition (TSR) models [5, 26, 20] as well as the increasing variety of datasets and benchmarks [3, 25, 26, 21].
These developments have enabled significant advances in deep learning (DL) modeling for TE [18, 16, 25, 21, 13, 12].

In general, benchmarks play a significant role in shaping the direction of ML research [17, 10].
Recently it has been shown that benchmark datasets for TSR contain a significant number of errors and inconsistencies [21].
It is well-documented that errors in a benchmark have negative consequences for both learning and evaluation [6, 27, 4].
Errors in the test set for the ImageNet benchmark [14], for instance, have led to top-1 accuracy saturating around 91% [24].
Errors can also lead to false conclusions during model selection, particularly when the training data is drawn from the same noisy distribution as the test set [14].

Compared to annotation mistakes, inconsistencies in a dataset can be more subtle because they happen across a collection of samples rather than in isolated examples—but no less harmful.
Even if a single dataset is self-consistent there may be inconsistencies in labeling across different datasets for the same task.
We consider datasets for the same task that are annotated inconsistently with respect to each other to be misaligned.
Misalignment can be considered an additional source of labeling noise.
This noise may go unnoticed when a dataset is studied in isolation, but it can have significant effects on model performance when datasets are combined.

In this work we study the effect that errors and misalignment between benchmark datasets have on model performance for TSR.
We select two large-scale crowd-sourced datasets for training—FinTabNet and PubTables-1M—and one small expert-labeled dataset for evaluation—the ICDAR-2013 benchmark.
For our models we choose a single fixed architecture, the recently proposed Table Transformer (TATR) [21].
This can be seen as a data-centric [22] approach to ML, where we hold the modeling approach fixed and instead seek to improve performance through data improvements.
Among our main contributions:

- •

We remove both annotation mistakes and inconsistencies between FinTabNet and ICDAR-2013, aligning these datasets with PubTables-1M and producing improved versions of two standard benchmark datasets for TSR (which we refer to as FinTabNet.c and ICDAR-2013.c, respectively).

- •

We show that removing inconsistencies between benchmark datasets for TSR indeed has a substantial positive impact on model performance, improving baseline models trained separately on PubTables-1M and FinTabNet from 65% to 78% and 42% to 65%, respectively, when evaluated on ICDAR-2013 (see Fig. 1).

- •

We perform a sequence of ablations over the steps of the correction procedure, which shows that canonicalization has a clear positive effect on model performance, while other factors create trade-offs when deciding the final composition of a benchmark.

- •

We train a single model on both PubTables-1M and the aligned version of FinTabNet (FinTabNet.c), establishing a new baseline DAR of 0.965 and exact match table recognition accuracy of 81% on the corrected ICDAR-2013 benchmark (ICDAR-2013.c).

- •

We plan to release all of the dataset processing and training code at https://github.com/microsoft/table-transformer.

(a) Original annotation

(b) Corrected annotation

## 2 Related Work

The first standard multi-domain benchmark dataset for table structure recognition was the ICDAR-2013 dataset, introduced at the 2013 ICDAR Table Competition [5].
In total it contains 158 tables111The dataset is originally documented as having 156 tables but we found 2 cases of tables that need to be split into multiple tables to be annotated consistently. in the official competition dataset and at least 100 tables222This is the number of tables in the practice set we were able to find online. in a practice dataset.
The competition also established the first widely-used metric for evaluation, the directed adjacency relations (DAR) metric, though alternative metrics such as TEDS [26] and GriTS [20] have been proposed more recently.
At the time of the competition, the best reported approach achieved a DAR of 0.946.
More recent work from Tensmeyer et al. [23] reports a DAR of 0.953 on the full competition dataset.

The difficulty of the ICDAR-2013 benchmark and historical lack of training data for the TSR task has led to a variety of approaches and evaluation procedures that differ from those established in the original competition.
DeepDeSRT [18] split the competition dataset into a training set and a test set of just 34 samples.
Additionally, the authors processed the annotations to add row and column bounding boxes and then framed the table structure recognition problem as the task of detecting these boxes.
This was a significant step at the time but because spanning cells are ignored it is only a partial solution to the full recognition problem.
Still, many works have followed this approach [7].
Hashmi et al. [8] recently reported an F-score of 0.9546 on the row and column detection task, compared to the original score of 0.9144 achieved by DeepDeSRT.

While ICDAR-2013 has been a standard benchmark of progress over the last decade, little attention has been given to complete solutions to the TSR task evaluated on the full ICDAR-2013 dataset.
Of these, none reports exact match recognition accuracy.
Further, we are not aware of any work that points out label noise in the ICDAR-2013 dataset and attempts to correct it.
Therefore it is an open question to what extent performance via model improvement has saturated on this benchmark in its current form.

To address the need for training data, several large-scale crowd-sourced datasets [3, 26, 11, 25, 21] have been released for training table structure recognition models.
However, the quality of these datasets is lower overall than that of expert-labeled datasets like ICDAR-2013.
Of particular concern is the potential for ambiguity in table structures [9, 19, 1, 15].
A specific form of label error called oversegmentation [21] is widely prevalent in benchmark TSR datasets, which ultimately leads to annotation inconsistency.
However, models developed and evaluated on these datasets typically treat them as if they are annotated unambiguously, with only one possible correct interpretation.

To address this, Smock et al. [21] proposed a canonicalization algorithm, which can automatically correct and make table annotations more consistent.
However, this was applied to only a single source of data.
The impact of errors and inconsistency across multiple benchmark datasets remains an open question.
In fact, the full extent of this issue can be masked by using metrics that measure the average correctness of cells [20].
But for table extraction systems in real-world settings, individual errors may not be tolerable, and therefore the accuracy both of cells and of entire tables is important to consider.

## 3 Data

As our baseline training datasets we adopt PubTables-1M [21] and FinTabNet [25].
PubTables-1M contains nearly one million tables from scientific articles, while FinTabNet contains close to 113k tables from financial reports.
As our baseline evaluation dataset we adopt the ICDAR-2013 [5] dataset.
The ICDAR-2013 dataset contains tables from multiple document domains manually annotated by experts.
While its size limits its usefulness for training, the quality of ICDAR-2013 and its diverse set of table appearances and layouts make it useful for model benchmarking.
For the ICDAR-2013 dataset, we use the competition dataset as the test set, use the practice dataset as a validation set, and consider each table "region" annotated in the dataset to be a table, yielding a total of 256 tables to start.

Dataset

Stage

Description

FinTabNet

—

Unprocessed

The original data, which does not have header or row/column bounding box annotations.

a1

Completion

Baseline FinTabNet dataset; create bounding boxes for all rows and columns; remove tables for which these boxes cannot be defined.

a2

Cell box adjustment

Iteratively refine the row and column boxes to tightly surround their coinciding text; create grid cells at the intersection of each row and column; remove any table for which 50% of the area of a word overlaps with multiple cells in the table.

a3

Consistency adjustments

Make the sizing and spanning of columns and rows more consistent; adjust cell bounding boxes to always ignore dot leaders (remove any tables for which this was unsuccessful); remove empty rows and empty columns from the annotations; merge any adjacent header rows whose cells all span the same columns; remove tables with columns containing just cent and dollar signs (we found these difficult to automatically correct).

a4

Canonicalization

Use cell structure to infer the column header and projected row headers for each table; remove tables with only two columns as the column header of these is structurally ambiguous; canonicalize each annotation.

a5

Additional column header inference

Infer the column header for many tables with only two columns by using the cell text in the first row to determine if the first row is a column header.

a6

Quality control

Remove tables with erroneous annotations, including: tables where words in the table coincide with either zero or more than one cell bounding box, tables with a projected row header at the top or bottom (indicating a caption or footer is mistakenly included in the table), and tables that appear to have only a header and no body.

ICDAR-2013

—

Unprocessed

The original data, which does not have header or row/column bounding box annotations.

a1

Completion

Baseline ICDAR-2013 dataset; create bounding boxes for all rows and columns; remove tables for which these boxes cannot be defined or whose annotations cannot be processed due to errors.

a2

Manual correction

Manually fix annotation errors (18 tables fixed in total).

a3

Consistency adjustments and canonicalization

Apply the same automated steps applied to FinTabNet.a2 through FinTabNet.a4; after processing, manually inspect each table and make any additional corrections as needed.

### 3.1 Missing Annotations

While the annotations in FinTabNet and ICDAR-2013 are sufficient to train and evaluate models for table structure recognition, several kinds of labels that could be of additional use are not explicitly annotated.
Both datasets annotate bounding boxes for each cell as the smallest rectangle fully-enclosing all of the text of the cell.
However, neither dataset annotates bounding boxes for rows, columns, or blank cells.
Similarly, neither dataset explicitly annotates which cells belong to the row and column headers.

For each of these datasets, one of the first steps we take before correcting their labels is to automatically add labels that are missing.
This can be viewed as making explicit the labels that are implicit given the rest of the label information present.
We cover these steps in more detail later in this section.

Dataset

# Tables‡

# Unique Topologies

Avg. Tables / Topology

Avg. Rows / Table

Avg. Cols. / Table

Avg. Spanning Cells / Table

FinTabNet

112,875

9,627

11.72

11.94

4.36

1.01

FinTabNet.a1

112,474

9,387

11.98

11.92

4.35

1.00

FinTabNet.a2

109,367

8,789

12.44

11.87

4.33

0.98

FinTabNet.a3

103,737

7,647

13.57

11.81

4.28

0.93

FinTabNet.a4

89,825

18,480

4.86

11.97

4.61

2.79

FinTabNet.a5

98,019

18,752

5.23

11.78

4.39

2.57

FinTabNet.a6

97,475

18,702

5.21

11.81

4.39

2.58

ICDAR-2013

256

181

1.41

15.88

5.57

1.36

ICDAR-2013.a1

247

175

1.41

15.61

5.39

1.34

ICDAR-2013.a2

258

181

1.43

15.55

5.45

1.42

ICDAR-2013.a3

258

184

1.40

15.52

5.45

2.08

‡The number of tables in the dataset that we were able to successfully read and process.

### 3.2 Label Errors and Inconsistencies

Next we investigate both of these datasets to look for possible annotation mistakes and inconsistencies.
For FinTabNet, due to the impracticality of verifying each annotation manually, we initially sample the dataset to identify types of errors common to many instances.
For ICDAR-2013, we manually inspect all 256 table annotations.

For both datasets, we note that the previous action of defining and adding bounding boxes for rows and columns was crucial to catching inconsistencies, as it exposed a number of errors when we visualized these boxes on top of the tables.
For ICDAR-2013 we noticed the following errors during manual inspection:
14 tables with at least one cell with incorrect column or row indices (see Fig. 2), 4 tables with at least one cell with an incorrect bounding box, 1 table with a cell with incorrect text content, and 2 tables that needed to be split into more than one table.

For FinTabNet, we noticed a few errors that appear to be common in crowd-sourced table annotations in the financial domain.
For example, we noticed many of these tables use dot leaders for spacing and alignment purposes.
However, we noticed a significant amount of inconsistency in whether dot leaders are included or excluded from a cell’s bounding box (see Fig. 3).
We also noticed that it is common in these tables for a logical column that contains a monetary symbol, such as a $, to be split into two columns, possibly for numerical alignment purposes (for an example, see NEE/2006/page_49.pdf).
Tables annotated this way are done so for presentation purposes but are inconsistent with their logical interpretation, which should group the dollar sign and numerical value into the same cell.
Finally we noticed annotations with fully empty rows.
Empty rows can sometimes be used in a table’s markup to create more visual spacing between rows.
But there is no universal convention for how much spacing corresponds to an empty row.
Furthermore, empty rows used only for spacing serve no logical purpose.
Therefore, empty rows in these tables should be considered labeling errors and should be removed for consistency.
For both datasets we noticed oversegmentation of header cells, as is common in crowd-sourced markup tables.

### 3.3 Dataset Corrections and Alignment

Mistakes noticed during manual inspection of ICDAR-2013 are corrected directly.
For both datasets, nearly all of the other corrections and alignments are done by a series of automated processing steps.
We list these steps at a high level in Tab. 1.
For each dataset the processing steps are grouped into a series of macro-steps, which helps us to study some of their effects in detail using ablation experiments in Sec. 4.

As part of the processing we adopt the canonicalization procedure [21] used to create PubTables-1M.
This helps to align all three datasets and minimize the amount of inconsistencies between them.
We make small improvements to the canonicalization procedure that help to generalize it to tables in domains other than scientific articles.
These include small improvements to the step that determines the column header and portions of the row header.
Tables with inconsistencies that are determined to not be easily or reliably corrected using the previous steps are filtered out.

When creating a benchmark dataset there are many objectives to consider such as maximizing the number of samples, the diversity of the samples, the accuracy and consistency of the labels, the richness of the labels, and the alignment between the labels and the desired use of the learned model for a task.
When cleaning a pre-existing dataset these goals potentially compete and trade-offs must be made.
There is also the added constraint in this case that whatever steps were used to create the pre-existing dataset prior to cleaning are unavailable to us and unchangeable.
This prevents us from optimizing the entire set of processing steps holistically.
Overall we balance the competing objectives under this constraint by correcting and adding to the labels as much as possible when we believe this can be done reliably and filtering out potentially low-quality samples in cases where we do not believe we can reliably amend them.
We document the effects that the processing steps have on the size, diversity, and complexity of the samples in Tab. 2.

Training Data

Baseline Version

Epoch‡

Test Images

GriTSConsubscriptGriTSCon\textbf{GriTS}_{\textbf{Con}}

GriTSLocsubscriptGriTSLoc\textbf{GriTS}_{\textbf{Loc}}

GriTSTopsubscriptGriTSTop\textbf{GriTS}_{\textbf{Top}}

AccConsubscriptAccCon\textbf{Acc}_{\textbf{Con}}

PubTables-1M

Original

23.72

Padded

0.9846

0.9781

0.9845

0.8138

Current

29

Tight

0.9855

0.9797

0.9851

0.8326

‡In the current work an epoch is standardized across datasets to equal 720,000 training samples.

## 4 Experiments

For our experiments, we adopt a data-centric approach using the Table Transformer (TATR) [21].
TATR frames TSR as object detection using six classes and is implemented with DETR [2].
We hold the TATR model architecture fixed and make changes only to the data.
TATR is free of any TSR-specific engineering or inductive biases, which importantly forces the model to learn to solve the TSR task from its training data alone.

We make a few slight changes to the original approach used to train TATR on PubTables-1M in order to establish a stronger baseline and standardize the approach across datasets.
First we re-define an epoch as 720,000 training samples, which corresponds to 23.72 epochs in the original paper, and extend the training to 30 epochs given the new definition.
Second, we increase the amount of cropping augmentation during training.
All of the table images in the PubTables-1M dataset contain additional padding around the table from the page that each table is cropped from.
This extra padding enable models to be trained with cropping augmentation without removing parts of the image belonging to the table.
However, the original paper evaluated models on the padded images rather than tightly-cropped table images.
In this paper, we instead evaluate all models on more tightly-cropped table images, leaving only 2 pixels of padding around the table boundary in each table image’s original size.
The use of tight cropping better reflects how TSR models are expected to be used in real-world settings when paired with an accurate table detector.

For evaluation, we use the recently proposed grid table similarity (GriTS) [20] metrics as well as table content exact match accuracy (AccConsubscriptAccCon\textrm{Acc}_{\textrm{Con}}).
GriTS compares predicted tables and ground truth directly in matrix form and can be interpreted as an F-score over the correctness of predicted cells.
Exact match accuracy considers the percentage of tables for which all cells, including blank cells, are matched exactly.
The TATR model requires words and their bounding boxes to be extracted separately and uses maximum overlap between words and predicted cells to slot the words into their cells.
For evaluation, we assume that the true bounding boxes for words are given to the model and used to determine the final output.

In Tab. 3 we evaluate the performance of the modified training procedure to the original procedure for PubTables-1M.
We use a validation set to select the best model, which for the new training procedure selects the model after 29 epochs.
Using the modified training procedure improves performance over the original, which establishes new baseline performance metrics.

In the rest of our experiments we train nine TATR models in total using nine different training datasets.
We train one model for each modified version of the FinTabNet dataset, a baseline model on PubTables-1M, and two additional models: one model for PubTables-1M combined with FinTabNet.a1 and one for PubTables-1M combined with FinTabNet.a6.
Each model is trained for 30 epochs, where an epoch is defined as 720,000 samples.
We evaluate each model’s checkpoint after every epoch on a validation set from the same distribution as its training data as well as a separate validation set from ICDAR-2013.
We average the values of GriTSConsubscriptGriTSCon\text{GriTS}_{\text{Con}} for the two validation sets and select the saved checkpoint yielding the highest score.

### 4.1 FinTabNet self-evaluation

(a) FinTabNet.a1

(b) FinTabNet.a3

(c) FinTabNet.a4

(d) FinTabNet.a5

Training Data
Epoch
AP
AP50subscriptAP50\textbf{AP}_{\textbf{50}}
AP75subscriptAP75\textbf{AP}_{\textbf{75}}
AR
GriTSConsubscriptGriTSCon\textbf{GriTS}_{\textbf{Con}}
GriTSLocsubscriptGriTSLoc\textbf{GriTS}_{\textbf{Loc}}
GriTSTopsubscriptGriTSTop\textbf{GriTS}_{\textbf{Top}}
AccConsubscriptAccCon\textbf{Acc}_{\textbf{Con}}

FinTabNet.a1
27
0.867
0.972
0.932
0.910
0.9796
0.9701
0.9874
0.787

FinTabNet.a2
22
0.876
0.974
0.941
0.916
0.9800
0.9709
0.9874
0.768

FinTabNet.a3
14
0.871
0.975
0.942
0.910
0.9845
0.9764
0.9893
0.799

FinTabNet.a4
30
0.874
0.974
0.946
0.919
0.9841
0.9772
0.9884
0.793

FinTabNet.a5
24
0.872
0.977
0.950
0.917
0.9861
0.9795
0.9897
0.828

FinTabNet.a6
26
0.875
0.976
0.948
0.921
0.9854
0.9796
0.9891
0.812

In Tab. 4, we present the results of each FinTabNet model evaluated on its own test set.
As can be seen, the complete set of dataset processing steps leads to an increase in AccConsubscriptAccCon\textrm{Acc}_{\textrm{Con}} from 0.787 to 0.812.
In Fig. 5, we illustrate on a sample from the FinTabNet test set how the consistency of the output improves from TATR trained on FinTabNet.a1 to TATR trained on FinTabNet.a5.
Note that models trained on FinTabNet.a4 onward also learn header information in addition to structure information.

Models trained on FinTabNet.a5 and FinTabNet.a6 both have higher AccConsubscriptAccCon\textrm{Acc}_{\textrm{Con}} than models trained on FinTabNet.a1 through FinTabNet.a3, despite the tables in a5 and a6 being more complex (and thus more difficult) overall than a1-a3 according to Tab. 2.
This strongly indicates that the results are driven primarily by increasing the consistency and cleanliness of the data, and not by reducing their inherent complexity.
Something else to note is that while GriTSLocsubscriptGriTSLoc\text{GriTS}_{\text{Loc}}, GriTSConsubscriptGriTSCon\text{GriTS}_{\text{Con}}, and AccConsubscriptAccCon\text{Acc}_{\text{Con}} all increase as a result of the improvements to the data, GriTSTopsubscriptGriTSTop\text{GriTS}_{\text{Top}} is little changed.
GriTSTopsubscriptGriTSTop\text{GriTS}_{\text{Top}} measures only how well the model infers the table’s cell layout alone, without considering how well the model locates the cells or extracts their text content.
This is strong evidence that the improvement in performance of the models trained on the modified data is driven primarily by there being more consistency in the annotations rather than the examples becoming less challenging.
However, this evidence is even stronger in the next section, when we evaluate the FinTabNet models on the modified ICDAR-2013 datasets instead.

### 4.2 ICDAR-2013 evaluation

(a) FinTabNet.a1

(b) PubTables-1M

(c) FinTabNet.a3

(d) FinTabNet.a6

(e) FinTabNet.a1 + PubTables-1M

(f) FinTabNet.a6 + PubTables-1M

Training
Ep.
Test Data
AP
AP50subscriptAP50\textbf{AP}_{\textbf{50}}
AP75subscriptAP75\textbf{AP}_{\textbf{75}}
DARCsubscriptDARC\textbf{DAR}_{\textbf{C}}
GriTSCsubscriptGriTSC\textbf{GriTS}_{\textbf{C}}
GriTSLsubscriptGriTSL\textbf{GriTS}_{\textbf{L}}
GriTSTsubscriptGriTST\textbf{GriTS}_{\textbf{T}}
AccCsubscriptAccC\textbf{Acc}_{\textbf{C}}

FinTabNet.a1
28
IC13.a1
0.670
0.859
0.722
0.8922
0.9390
0.9148
0.9503
0.417

28
IC13.a2
0.670
0.859
0.723
0.9010
0.9384
0.9145
0.9507
0.411

20
IC13.a3
0.445
0.573
0.471
0.8987
0.9174
0.8884
0.9320
0.411

FinTabNet.a2
27
IC13.a1
0.716
0.856
0.778
0.9107
0.9457
0.9143
0.9536
0.436

27
IC13.a2
0.714
0.856
0.778
0.9049
0.9422
0.9105
0.9512
0.430

29
IC13.a3
0.477
0.576
0.516
0.8862
0.9196
0.8874
0.9336
0.405

FinTabNet.a3
25
IC13.a1
0.710
0.851
0.765
0.9130
0.9462
0.9181
0.9546
0.462

25
IC13.a2
0.710
0.850
0.764
0.9091
0.9443
0.9166
0.9538
0.456

28
IC13.a3
0.470
0.571
0.505
0.8889
0.9229
0.8930
0.9346
0.418

FinTabNet.a4
30
IC13.a1
0.763
0.935
0.817
0.9134
0.9427
0.9170
0.9516
0.551

30
IC13.a2
0.763
0.935
0.818
0.9088
0.9409
0.9155
0.9503
0.544

25
IC13.a3
0.765
0.944
0.832
0.9287
0.9608
0.9409
0.9703
0.589

FinTabNet.a5
30
IC13.a1
0.774
0.939
0.838
0.9119
0.9412
0.9075
0.9456
0.494

30
IC13.a2
0.774
0.940
0.840
0.9098
0.9400
0.9057
0.9443
0.487

16
IC13.a3
0.773
0.956
0.854
0.9198
0.9548
0.9290
0.9624
0.551

FinTabNet.a6
23
IC13.a1
0.760
0.927
0.830
0.9057
0.9369
0.9117
0.9497
0.500

21
IC13.a2
0.757
0.926
0.818
0.9049
0.9374
0.9106
0.9491
0.506

20
IC13.a3
0.757
0.941
0.840
0.9336
0.9625
0.9431
0.9702
0.646

PubTables-1M
29
IC13.a1
0.873
0.972
0.943
0.9440
0.9590
0.9462
0.9623
0.647

29
IC13.a2
0.873
0.972
0.941
0.9392
0.9570
0.9448
0.9608
0.639

30
IC13.a3
0.828
0.973
0.934
0.9570
0.9756
0.9700
0.9786
0.753

PubTables-1M +
25
IC13.a1
0.872
0.970
0.940
0.9543
0.9678
0.9564
0.9720
0.686

FinTabNet.a1
25
IC13.a2
0.871
0.970
0.939
0.9501
0.9655
0.9543
0.9704
0.677

28
IC13.a3
0.820
0.949
0.911
0.9630
0.9787
0.9702
0.9829
0.785

PubTables-1M +
28
IC13.a1
0.881
0.977
0.953
0.9687
0.9678
0.9569
0.9705
0.679

FinTabNet.a6
28
IC13.a2
0.880
0.975
0.951
0.9605
0.9669
0.9566
0.9702
0.671

29
IC13.a3
0.826
0.974
0.934
0.9648
0.9811
0.9750
0.9842
0.810

In the next set of results, we evaluate all nine models on all three versions of the ICDAR-2013 dataset.
These results are intended to show the combined effects that improvements to both the training data and evaluation data have on measured performance.
Detailed results are given in Tab. 5 and a visualization of the results for just AccConsubscriptAccCon\text{Acc}_{\text{Con}} is given in Fig. 1.

Note that all of the improvements to the ICDAR-2013 annotations are verified manually and no tables are removed from the dataset.
So observed improvements to performance from ICDAR-2013.a1 to ICDAR-2013.a3 can only be due to cleaner and more consistent evaluation data.
This improvement can be observed clearly for each trained model from FinTabNet.a4 onward.
Improvements to ICDAR-2013 alone are responsible for measured performance for the model trained on PubTables-1M and FinTabNet.a6 combined increasing significantly, from 68% to 81% AccConsubscriptAccCon\text{Acc}_{\text{Con}}.

Evaluating improvements to FinTabNet on ICDAR-2013.a3, we also observe a substantial increase in performance, from 41.1% to 64.6% AccConsubscriptAccCon\text{Acc}_{\text{Con}}.
This indicates that not only do improvements to FinTabNet improve its own self-consistency, but also significantly improve its performance on challenging real-world test data.
The improvement in performance on test data from both FinTabNet and ICDAR-2013 clearly indicates that there is a significant increase in the consistency of the data across both benchmark datasets.

These results also hold for FinTabNet combined with PubTables-1M.
The final model trained with PubTables-1M and FinTabNet.a6 combined outperforms all other baselines, achieving a final AccConsubscriptAccCon\text{Acc}_{\text{Con}} of 81%.
Few prior works report this metric on ICDAR-2013 or any benchmark dataset for TSR.
As we discuss previously, this is likely due to the fact that measured accuracy depends not only on model performance but the quality of the ground truth annotation.
These results simultaneously establish new performance baselines and indicate that the evaluation data is clean enough for this metric to be a useful measure of model performance.

While TATR trained on PubTables-1M and FinTabNet.a6 jointly performs best overall, in Fig. 5 we highlight an interesting example from the ICDAR-2013 test set where TATR trained on FinTabNet.a6 alone performs better.
For this test case, TATR trained on the baseline FinTabNet.a1 does not fully recognize the table’s structure, while TATR trained on the FinTabNet.a6 dataset does.
Surprisingly, adding PubTables-1M to the training makes recognition for this example worse.
We believe cases like this suggest the potential for future work to explore additional ways to leverage PubTables-1M and FinTabNet.a6 to improve joint performance even further.

## 5 Limitations

While we demonstrated clearly that removing annotation inconsistencies between TSR datasets improves model performance, we did not directly measure the accuracy of the automated alignment procedure itself.
Instead, we minimized annotation mistakes by introducing quality control checks and filtering out any tables in FinTabNet whose annotations failed these tests.
But it is possible that the alignment procedure may introduce its own mistakes, some of which may not be caught by the quality control checks.
It is also possible that some oversegmented tables could require domain knowledge specific to the table content itself to infer their canonicalized structure, which could not easily be incorporated into an automated canonicalization algorithm.
Finally, the quality control checks themselves may in some cases be too restrictive, filtering out tables whose annotations are actually correct.
Therefore it is an open question to what extent model performance could be improved by further improving the alignment between these datasets or by improving the quality control checks.

## 6 Conclusion

In this work we addressed the problem of misalignment in benchmark datasets for table structure recognition.
We adopted the Table Transformer (TATR) model and three standard benchmarks for TSR—FinTabNet, PubTables-1M, and ICDAR-2013—and removed significant errors and inconsistencies between them.
After data improvements, performance of TATR on ICDAR-2013 increased substantially from 42% to 65% exact match accuracy when trained on FinTabNet and 65% to 75% when trained on PubTables-1M.
In addition, we trained TATR on the final FinTabNet and PubTables-1M datasets combined, establishing new improved baselines of 0.965 DAR and 81% exact match accuracy on the ICDAR-2013 benchmark through data improvements alone.
Finally, we demonstrated through ablations that canonicalization has a significantly positive effect on the performance improvements across all three datasets.

## 7 Acknowledgments

We would like to thank the anonymous reviewers for helpful feedback while preparing this manuscript.

## References

- [1]

Broman, K.W., Woo, K.H.: Data organization in spreadsheets. The American
Statistician 72(1), 2–10 (2018)

- [2]

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.:
End-to-end object detection with transformers. In: European Conference on
Computer Vision. pp. 213–229. Springer (2020)

- [3]

Chi, Z., Huang, H., Xu, H.D., Yu, H., Yin, W., Mao, X.L.: Complicated table
structure recognition. arXiv preprint arXiv:1908.04729 (2019)

- [4]

Frénay, B., Verleysen, M.: Classification in the presence of label noise: a
survey. IEEE transactions on neural networks and learning systems
25(5), 845–869 (2013)

- [5]

Göbel, M., Hassan, T., Oro, E., Orsi, G.: ICDAR 2013 table competition.
In: 2013 12th International Conference on Document Analysis and Recognition.
pp. 1449–1453. IEEE (2013)

- [6]

Guyon, I., Matić, N., Vapnik, V.: Discovering informative patterns and data
cleaning. In: Proceedings of the 3rd International Conference on Knowledge
Discovery and Data Mining. pp. 145–156 (1994)

- [7]

Hashmi, K.A., Liwicki, M., Stricker, D., Afzal, M.A., Afzal, M.A., Afzal, M.Z.:
Current status and performance analysis of table recognition in document
images with deep neural networks. arXiv preprint arXiv:2104.14272 (2021)

- [8]

Hashmi, K.A., Stricker, D., Liwicki, M., Afzal, M.N., Afzal, M.Z.: Guided table
structure recognition through anchor optimization. arXiv preprint
arXiv:2104.10538 (2021)

- [9]

Hu, J., Kashi, R., Lopresti, D., Nagy, G., Wilfong, G.: Why table
ground-truthing is hard. In: Proceedings of Sixth International Conference on
Document Analysis and Recognition. pp. 129–133. IEEE (2001)

- [10]

Koch, B., Denton, E., Hanna, A., Foster, J.G.: Reduced, reused and recycled:
The life of a dataset in machine learning research. arXiv preprint
arXiv:2112.01716 (2021)

- [11]

Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., Li, Z.: Tablebank: Table
benchmark for image-based table detection and recognition. In: Proceedings of
The 12th Language Resources and Evaluation Conference. pp. 1918–1925 (2020)

- [12]

Liu, H., Li, X., Liu, B., Jiang, D., Liu, Y., Ren, B.: Neural collaborative
graph machines for table structure recognition. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp.
4533–4542 (June 2022)

- [13]

Nassar, A., Livathinos, N., Lysak, M., Staar, P.: Tableformer: Table structure
understanding with transformers. In: Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR). pp. 4614–4623 (June 2022)

- [14]

Northcutt, C.G., Athalye, A., Mueller, J.: Pervasive label errors in test sets
destabilize machine learning benchmarks. arXiv preprint arXiv:2103.14749
(2021)

- [15]

Paramonov, V., Shigarov, A., Vetrova, V.: Table header correction algorithm
based on heuristics for improving spreadsheet data extraction. In:
International Conference on Information and Software Technologies. pp.
147–158. Springer (2020)

- [16]

Prasad, D., Gadpal, A., Kapadni, K., Visave, M., Sultanpure, K.:
CascadeTabNet: An approach for end to end table detection and structure
recognition from image-based documents. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition Workshops. pp. 572–573
(2020)

- [17]

Raji, I.D., Bender, E.M., Paullada, A., Denton, E., Hanna, A.: Ai and the
everything in the whole wide world benchmark. arXiv preprint arXiv:2111.15366
(2021)

- [18]

Schreiber, S., Agne, S., Wolf, I., Dengel, A., Ahmed, S.: DeepDeSRT: Deep
learning for detection and structure recognition of tables in document
images. In: 2017 14th IAPR international conference on document analysis and
recognition (ICDAR). vol. 1, pp. 1162–1167. IEEE (2017)

- [19]

Seth, S., Jandhyala, R., Krishnamoorthy, M., Nagy, G.: Analysis and taxonomy of
column header categories for web tables. In: Proceedings of the 9th IAPR
International Workshop on Document Analysis Systems. pp. 81–88 (2010)

- [20]

Smock, B., Pesala, R., Abraham, R.: GriTS: Grid table similarity metric for
table structure recognition. arXiv preprint arXiv:2203.12555 (2022)

- [21]

Smock, B., Pesala, R., Abraham, R.: PubTables-1M: Towards comprehensive
table extraction from unstructured documents. In: Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4634–4642
(June 2022)

- [22]

Strickland, E.: Andrew Ng, AI Minimalist: The machine-learning pioneer
says small is the new big. IEEE Spectrum 59(4), 22–50 (2022)

- [23]

Tensmeyer, C., Morariu, V.I., Price, B., Cohen, S., Martinez, T.: Deep
splitting and merging for table structure decomposition. In: 2019
International Conference on Document Analysis and Recognition (ICDAR). pp.
114–121. IEEE (2019)

- [24]

Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., Wu, Y.: Coca:
Contrastive captioners are image-text foundation models. arXiv preprint
arXiv:2205.01917 (2022)

- [25]

Zheng, X., Burdick, D., Popa, L., Zhong, X., Wang, N.X.R.: Global table
extractor (GTE): A framework for joint table identification and cell
structure recognition using visual context. In: Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision. pp. 697–706 (2021)

- [26]

Zhong, X., ShafieiBavani, E., Yepes, A.J.: Image-based table recognition: data,
model, and evaluation. arXiv preprint arXiv:1911.10683 (2019)

- [27]

Zhu, X., Wu, X.: Class noise vs. attribute noise: A quantitative study.
Artificial intelligence review 22(3), 177–210 (2004)

Generated on Thu Feb 29 22:11:18 2024 by LaTeXML
