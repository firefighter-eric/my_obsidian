# Ding et al. - 2024 - Using the divergent association task to measure divergent thinking in Chinese elementary school students

- Source PDF: `raw/pdfs/Ding et al. - 2024 - Using the divergent association task to measure divergent thinking in Chinese elementary school students.pdf`
- Generated from: `scripts/extract_pdf_text.py`

## Extracted Text

Thinking Skills and Creativity 52 (2024) 101503
Available online 11 March 2024
1871-1871/© 2024 Elsevier Ltd. All rights reserved.
Using the divergent association task to measure divergent thinking 
in Chinese elementary school students 
Guozhu Ding a,*, Yiwei He a, Kaixu Yi a, Shan Li b 
a School of Education (Teachers College), Guangzhou University, 230 Wai Huan Xi Road, Guangzhou Higher Education Mega Center, Guangzhou, 
China, 510006 
b College of Education/College of Health, Lehigh University, HST Building 132, 124 E. Morton Street, Lehigh University, Bethlehem, PA, USA, 18105   
A R T I C L E  I N F O   
Keywords: 
Divergent thinking 
Semantic distance 
Creativity 
Natural language processing 
Elementary school student 
A B S T R A C T   
The Divergent Association Task (DAT), published in July 2021, is a psychological test designed to 
measure an individual’s divergent thinking. The test requires participants to name ten nouns that 
exhibit maximum dissimilarity from each other. The semantic distance between these nouns is 
then calculated to indicate the person’s level of divergent thinking. In this study, we explored the 
applicability of the DAT for elementary school students in Chinese contexts, given that it was not 
initially designed for this specific population and was available only in English. We recruited a 
total of 348 students who were asked to complete three creativity tasks: the DAT, the Alternative 
Uses Task (AUT), and the Bridge-the-Associative-Gap Task (BAG). We examined the associations 
between DAT and the scores of the AUT and BAG tests. Moreover, we tested the accuracy of the 
DAT using varying numbers of nouns and different natural language processing models to 
calculate the semantic distance between nouns. Our findings supported the suitability of using the 
DAT to measure divergent thinking in elementary school students within Chinese contexts. We 
also found that using only eight nouns, instead of ten, could achieve a relatively high accuracy in 
measuring divergent thinking based on the DAT method. The language model of Word2Vec 
performed better than the BERT (Bidirectional Encoder Representations from Transformers) and 
GPT (Generative Pre-trained Transformer) models when calculating semantic distances between 
nouns. This study has methodological and practical implications.   
1. Introduction 
Creative thinking is widely acknowledged as a vital 21st-century skill. Creative thinking holds particular significance for children as 
they will need to propose innovative solutions for today’s society and economy (Ananiadou & Claro, 2009). In school contexts, creative 
thinking can span from independent reflections on subjects like why Pluto merits planetary status to participating in creative activities 
such as drawing or creating jokes (Kaufman & Beghetto, 2009). Understanding a child’s creative thinking level allows educators and 
parents to tailor educational activities, lessons, and materials to their specific needs and abilities. This helps foster creativity in an 
age-appropriate and effective manner. Moreover, measuring creative thinking over time enables tracking of a child’s progress and 
growth in this domain. This information can help adjust approaches and strategies to better support continued creative development. 
* Corresponding author at: 230 Wai Huan Xi Road, Guangzhou Higher Education Mega Center, School of Education (Teachers College), 
Guangzhou University, Guangzhou, China, 510006. 
E-mail address: dinggz@gzhu.edu.cn (G. Ding).  
Contents lists available at ScienceDirect 
Thinking Skills and Creativity 
journal homepage: www.elsevier.com/locate/tsc 
https://doi.org/10.1016/j.tsc.2024.101503 
Received 20 November 2023; Received in revised form 7 March 2024; Accepted 10 March 2024   

Thinking Skills and Creativity 52 (2024) 101503
2
Various measurements have been developed to assess children’s creative thinking, including divergent thinking tests, convergent 
thinking tests, insight tasks, and assessments of creative products (Amabile, 1982; Guilford, 1967; Mednick, 1962; Plucker & Makel, 
2010). Among these assessments, the divergent thinking test stood out as the most commonly used. 
Divergent thinking, a key component of creative thinking, holds significant importance in stimulating students’ imagination and 
creativity (Brophy, 2001; Guilford, 1967). The most commonly used method for measuring students’ divergent thinking is the 
Alternative Uses Task (AUT) proposed by Guilford (1967). The AUT test presents individuals with a common object, such as a brick or a 
paperclip, and requires them to generate as many creative and unconventional uses for that object as possible within a set time limit. 
Additionally, Mednick (1962) introduced the Remote Associates Test (RAT) to measure divergent thinking. The RAT test typically 
consists of a set of word triads, where the challenge is to identify a fourth word that is conceptually related to each of the three given 
words. In recent years, researchers have proposed new measurement approaches building upon these traditional tests of divergent 
thinking. For instance, Gianotti et al. (2001) introduced the Bridge-the-Associative-Gap Task (BAG) based on the RAT. Even though the 
BAG is considered a measure of convergent thinking by some researchers, it is highly related to divergent thinking ability (Olson et al., 
2021). Furthermore, Silvia et al. (2009) proposed a method to measure college students’ divergent thinking skills through snapshot 
scoring. 
Traditional measures of divergent thinking present challenges such as subjective scoring and being time-consuming (Hocevar & 
Michael, 1979; Silvia, 2015). To address these issues, researchers have explored the use of computer algorithms to provide objective 
and efficient evaluations. For instance, the Divergent Association Task (DAT), developed by Olson et al. (2021), requires participants to 
name ten nouns that exhibit maximum dissimilarity from each other. The semantic distance between these nouns is then calculated to 
indicate the person’s level of divergent thinking. However, DAT was not initially designed for elementary school students and was 
available only in English. 
The overarching goal of this study is to investigate the applicability of the DAT in elementary school students within Chinese 
contexts. Examining the applicability of the DAT in Chinese elementary school students is essential for ensuring culturally appropriate 
and valid measurement of creative thinking skills, informing educational practices, and enabling cross-cultural comparisons. We 
evaluate the accuracy of the DAT by comparing its correlation with two traditional measures of divergent thinking, namely AUT and 
BAG. Additionally, we assess the effectiveness of the DAT using different numbers of nouns and various natural language processing 
models to calculate the semantic distance between nouns. 
2. Literature review 
2.1. Creative thinking and its measurements 
Creativity is defined as the ability to produce novel and adaptive products (Amabile & Pratt, 2016; Kaufman & Sternberg, 2010; 
Runco & Johnson, 2002). Creative thinking, a cornerstone of the study and application of creativity, is often considered one of the 
higher cognitive functions. Creative thinking generally consists of two primary components (Brophy, 2001; Guilford, 1967): 
convergent and divergent thinking, working synergistically to generate creative output. Convergent thinking involves evaluating 
different stimuli to identify the most appropriate response, such as determining the optimal solution to a problem (Becker et al., 2020; 
Wu & Chen, 2017). This type of thinking is generally easier to score because participants have predefined correct answers to choose 
from. In contrast, divergent thinking reflects an individual’s mental capacity to generate multiple original ideas in response to a given 
problem or prompt (Acar & Runco, 2019; Forthmann et al., 2019). Measurement tasks for divergent thinking often use open-ended 
questions to assess a person’s ability to generate multiple ideas (Acar & Runco, 2014). These tasks typically result in longer textual 
responses, which make manual assessment and objective scoring more challenging. 
In previous studies, the most commonly used test of divergent thinking was the Alternative Uses Task (AUT) proposed by Guilford 
(1967). In this task, participants were asked to generate uses for a common object, and raters assessed responses on three dimensions: 
originality, fluency, and flexibility. Originality is based on the uniqueness of ideas, fluency on the number of ideas, and flexibility on 
the variety of idea categories. Regarding the implementation of AUT, Wallach and Kogan (1965) argued that imposing strict time limits 
could potentially suppress or underestimate an individual’s true creative potential. They believed that removing time constraints 
would allow for more uninhibited creative expression. Despite its completeness, this widely used test has drawbacks. Manual scoring is 
time-consuming and requires multiple raters for reliable evaluation (Acar & Runco, 2014; Olson et al., 2021). Additionally, fluency and 
originality scores may be intertwined, with a higher chance of originality scores rising alongside fluency scores (Hocevar & Michael, 
1979; Silvia, 2015). Finally, the assessments do not consider cultural differences, as the commonalities in object usage vary across 
different cultures (Olson et al., 2021). Due to these limitations, the objectivity of divergent thinking tests has been controversial. 
To delve deeper into the study and training of creative thinking, there is a need for a novel measurement technique that can 
quantify it objectively and efficiently. Utilizing semantic distance to measure creative thinking is such a method. Specifically, the 
proximity of concepts is determined by the number of defining features they share, a relationship known as “semantic distance” (Volle, 
2018). For instance, the words “prince” and “princess” have a relatively close semantic distance as they often co-occur in the text, 
whereas the semantic distance between “prince” and “artificial intelligence” is relatively far since they rarely appear together. 
The Associative Theory of Creativity (ATC), proposed by Mednick (1962), elucidates the connection between creative thinking and 
the structure of semantic memory. The ATC draws on the idea that creative thinking is not solely about producing entirely new ideas 
but often involves recombining existing concepts in novel ways. Mednick’s (1962) theory suggests that individuals who can bring 
together disparate concepts from their semantic memory, which stores knowledge about the meanings and relationships between 
words and concepts, are more likely to demonstrate creative thinking. For instance, if certain concepts are more semantically distant, 
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
3
the resulting combinations are more creative and innovative. Building on this theory, Benedek et al. (2012) proposed that the ability to 
fluently retrieve and combine remote associations facilitated creative solutions. There was a close relationship between associative 
processes and divergent thinking (Benedek et al., 2012). Therefore, we can conclude that semantic distance, as a quantitative indicator 
of the relationship between concepts, could indicate the level of divergent thinking. This measure effectively reflects an individual’s 
creative thinking grounded in the associative process (Volle, 2018). 
Building upon the ATC, Mednick (1962) introduced the Remote Associates Test (RAT) to assess creative thinking. In the RAT task, 
participants are presented with sets of three seemingly unrelated cue words (e.g., “mouse,” “blue,” and “harvester”) and are tasked 
with identifying a common target word (e.g., “cheese”) that forms associations with each cue. Mednick (1962) argued that individuals 
demonstrating greater proficiency in identifying target words within a specified timeframe are often perceived as more creative. 
Expanding on Mednick’s (1962) work, Gianotti et al. (2001) introduced the Bridge-the-Associative-Gap Task (BAG), an extension of 
the RAT task. In the BAG task, participants are presented with word pairs, either related or unrelated, and are required to generate a 
word that is semantically linked to both terms. For example, given the pair “shoes” and “legs,” an acceptable response could be “socks” 
due to its association with both words. Participants are allotted 30 seconds for each question, and raters evaluate the appropriateness 
of responses on a scale from one to seven. Compared to Mednick’s (1962) RAT task, the BAG task offers several advantages. It allows for 
a clear differentiation between stimulus types. It also offers the flexibility to choose either relevant or irrelevant cue words. Therefore, 
responses in the BAG task are generally considered valid without the imposition of correct or incorrect labels. Yet, the BAG task suffers 
from challenges such as being time-consuming, labor-intensive, and subjectivity in evaluation. 
2.2. Computer algorithm-based assessment methods 
To overcome challenges associated with traditional assessments, research has turned to the use of computer algorithms to assess 
divergent thinking. Computer algorithms are capable of articulating the theoretical assumptions required for scoring in a computer- 
based program (Acar & Runco, 2014; Hass, 2017). Moreover, computer algorithms can quantify divergent thinking in a more objective 
and efficient manner. One such approach is the use of natural language models to measure divergent thinking by calculating semantic 
distances. For instance, Olson et al. (2021) assessed divergent thinking by calculating semantic distances of seven to ten nouns. 
Beketayev and Runco (2016) used a semantic-based algorithms (SBA) approach to assess divergent thinking. This approach was fully 
automated and could return individuals’ assessments in no time. It is worth mentioning that the use of semantic distance has been used 
to improve traditional scoring methods for the Alternative Uses Task (AUT) (Acar & Runco, 2014; Beketayev & Runco, 2016; Volle, 
2018). In sum, automated assessment of divergent thinking, grounded in natural language processing models, effectively mitigates 
rater subjectivity, and enhances standardization and objectivity in scoring. It not only significantly reduces time and labor costs but 
also improves scoring efficiency (Acar & Runco, 2015; Beaty & Johnson, 2021; Olson et al., 2021). 
The most recent development of divergent thinking measures is the Divergent Association Task (DAT) proposed by Olson et al. 
(2021). In the DAT test, participants are tasked with generating ten nouns unrelated in meaning and usage within a 4-minute time 
frame. The semantic distance between these words is calculated using the GloVe model in natural language processing. The partici­
pant’s divergent thinking score is determined by the final average score obtained through this process. Particularly, Olson et al. (2021) 
collected a total of 9,098 responses and confirmed a strong correlation between the divergent thinking measure obtained from the DAT 
task and two widely used measures of creativity, i.e., the Alternative Uses Task and the Bridge-the-Associative-Gap Task. The validity 
of the DAT task was substantiated by the high correlations observed between the two measures. However, among the 9,098 responses, 
only 221 data points were derived from children aged 6 to 12 years, which constituted only 2.4% of the total dataset. We believe that 
the sample size in this age group is relatively low, and further validation is needed. 
The model employed for calculating semantic distance in the DAT is the GloVe model. Developed by Pennington et al. (2014), the 
GloVe model aims to represent each word as a vector, capturing the semantic relationships between words. Its fundamental approach 
involves constructing a word co-occurrence matrix from the corpus, followed by converting word relationships into vector operations. 
The GloVe model computes targeted word-to-word correlations by examining the co-occurrence window of words surrounding each 
word. This enables the model to better comprehend and process language semantics, quantifying the geometric space between two 
words or phrases. Olson et al. (2021) utilized the GloVe model for calculating semantic distances, citing its prior use in scoring AUT 
tasks (Beaty & Johnson, 2021; Dumas et al., 2021). Dumas et al. (2021) concluded that scoring with GloVe closely aligns with human 
scoring. GloVe provides a reliable and effective approach for assessing divergent thinking. 
As aforementioned, we aimed to explore the applicability of the DAT in measuring divergent thinking among elementary school 
students in Chinese contexts. In our study, we opted to use the Word2Vec model instead of the GloVe model for semantic distance 
computation due to the unavailability of an open-source GloVe model trained on the Chinese corpus. The Word2Vec model, developed 
by Mikolov et al. (2013), is designed to capture semantic relationships between words by representing them as vectors, where words 
with similar meanings are closer together in the vector space. The model achieves this by considering the context in which words 
appear in a given corpus. Word2Vec has two main training approaches: Skip-gram and Continuous Bag of Words (CBOW). In the 
Skip-gram approach, the model predicts the context words (words surrounding a target word) given a current word. It essentially 
learns to predict the probability distribution of context words for a given target word. In CBOW, the model predicts the target word 
based on its context (surrounding words). It learns to predict the target word from the context words. Both training approaches result in 
word vectors that encode semantic information. Words with similar meanings or usage patterns will have vectors that are geomet­
rically close in the vector space. Word2Vec has been widely used in various applications, including sentiment analysis, machine 
translation, and semantic similarity calculations. 
To further explore the effect of different language models on the DAT task, we also used the BERT (Bidirectional Encoder 
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
4
Representations from Transformers) model and the GPT (Generative Pre-trained Transformer) model for semantic distance calcula­
tion. The BERT model is a pre-trained language model (Devlin et al., 2018), which is constructed based on the structure of the 
Transformer as a multilayered network of Encoders, where all the layers are jointly contextual context for pre-training. The BERT 
model used in the study is derived from the Chinese BERT model trained by Cui et al. (2021), which is an open-source model that is 
more suitable for Chinese semantic distance computation compared to Google’s officially released BERT model. Compared with GloVe 
and Word2Vec models, the vector dimension of BERT is larger and more responsive to the semantic features of words. 
The GPT model is a natural language processing model based on the Transformer architecture developed by OpenAI (Radford et al., 
2018). The GPT model operates through two main phases: pre-training and fine-tuning. During the pre-training phase, the model is 
initially trained on a large unlabeled corpus to predict the next word in a given text. Subsequently, in the fine-tuning phase, the model 
is adapted to specific task requirements using task-specific data. In this study, word vectors are obtained using the embedding API 
provided by OpenAI, and semantic distances are then calculated. While both GPT and BERT are models based on the Transformer 
architecture, they differ in target tasks, training methods, and contextual use, and the resulting word vectors exhibit some distinctions. 
2.3. The current study 
This study aims to explore the applicability of the DAT in measuring divergent thinking among elementary school students in 
Chinese contexts, given that it was not originally designed for this specific population and is available only in English (Olson et al., 
2021). Conducting research on the measurement of divergent thinking among Chinese students can contribute to validating the 
applicability and effectiveness of the DAT within the Chinese cultural context. Moreover, the measurement of divergent thinking can 
facilitate a better understanding of students’ creative thinking capabilities among educators and policymakers, thereby enabling the 
design and implementation of more targeted educational interventions. 
Furthermore, there is perpetual interest among researchers and educators in students’ creative thinking abilities across different 
ages. The development of creative thinking in childhood is generally believed to unfold in a progression. It begins with the uninhibited, 
imaginative expression of early years, evolves into the emergence of more structured and purposeful creative abilities in middle 
childhood, and culminates in the refinement and specialization of creative talents in late childhood (Besançon & Lubart, 2008; Maker 
& Muammar, 2008). This journey is influenced by cognitive maturation and the expansion of knowledge and skills. However, Gardner 
and Gardner (2008) observed that preschool children exhibit high levels of creativity, which often diminish as they progress through 
school and deepen their education. Urban (1991), utilizing the Test for Creative Thinking-Drawing Production (TCT-DP) on children 
aged 4 to 8, noted a decline in creativity following the commencement of primary education. Alacapinar (2013), in exploring the 
relationship between grade level and creativity, found that creativity scores significantly increased from grade 3 to grade 5, only to 
decrease again from grade 6 to grade 8. Therefore, this study also explores the development of students’ creative thinking. 
Specifically, we compare students’ DAT scores with two widely used creativity tests: the Alternative Uses Task (Guilford, 1967) and 
the Bridge-the-Associative-Gap Task (Gianotti et al., 2001). We also aim to test the accuracy of the DAT using varying numbers of 
nouns. It is noteworthy that the DAT employs the GloVe model to calculate semantic distances. However, due to the unavailability of 
an open-source GloVe model trained in Chinese, we use the Word2Vec model for semantic distance calculation. Additionally, we 
investigate the impact of various language models, including BERT (Bidirectional Encoder Representations from Transformers) and 
GPT (Generative Pre-trained Transformer), on the performance of the DAT method. Specifically, this study addresses the following 
research questions: (1) Is it suitable to use the DAT to measure divergent thinking in elementary school students within Chinese 
contexts? (2) How many nouns are needed to calculate a relatively accurate DAT score? And (3) How do different language processing 
models affect the performance of the DAT? (4) How does the divergent thinking ability of elementary school students change across 
grades? 
3. Method 
3.1. Data collection 
In this study, 348 students were randomly selected as a sample from an elementary school in Guangzhou, China. Research ethics 
was approved by the university’s ethics review board office. The students completed three questionnaires on creativity, consisting of 
the Alternative Uses Task, the Bridge-the-Associative-Gap Task, and the Divergent Association Task. We excluded 26 invalid records 
Table 1 
Distribution of Students and Their Self-reports Across Grades.  
Grade 
Average Age 
Boy 
Girl 
Valid 
Invalid 
Total 
1st grade 
6 
44 
34 
64 
14 
78 
2nd grade 
7 
23 
18 
40 
1 
41 
3rd grade 
8 
22 
19 
41 
0 
41 
4th grade 
9 
45 
34 
76 
3 
79 
5th grade 
10 
26 
24 
45 
5 
50 
6th grade 
11 
41 
18 
56 
3 
59 
Total 
9 
201 
147 
322 
26 
348 
Note: “Valid” refers to valid self-reports collected from the participants. 
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
5
that were deemed incomplete. Ultimately, 322 valid responses were obtained. The students’ ages ranged from 6 to 12 years old. Table 1 
displays the data collected from different grades. 
3.2. The alternative uses task 
One of the most common measures of creativity is the Alternative Uses Task (Guilford, 1967; Wallach & Kogan, 1965). In this task, 
participants are asked to generate multiple uses for common objects. For instance, given the word ‘chopsticks,’ participants might 
suggest uses like ‘eating,’ ‘stirring,’ or ‘drumming.’ Subsequently, participants are prompted to provide additional uses for the same 
item. In this study, we selected five common items based on elements commonly found in Chinese cultural life: ‘chopsticks,’ ‘card­
board,’ ‘plastic bag,’ ‘sand,’ and ‘spring.’ The stimulus words were modified to accommodate cultural differences. For instance, the 
term “ice tray,” originally employed in the Alternative Uses Task, is not a commonly recognized object in the Chinese context, leading 
to confusion among many students who were unfamiliar with its representation. 
Participants were instructed, ‘This is a test of creativity; please list as many uses for these objects as possible.’ Each participant had a 
2-minute time limit to propose a use for each object, totaling 10 minutes allotted to the subjects. We imposed a time limit for the AUT 
task based on several considerations. First, the AUT could be cognitively demanding, especially for younger participants. Providing a 
reasonable time limit could help prevent cognitive fatigue and ensure participants have sufficient resource to generate responses 
without feeling overwhelmed. Furthermore, we carried out a pilot test with a small sample of participants, and their performance and 
feedback indicated that the 2-minute time limit per object was appropriate. 
Raters assess participant responses in three sections: (1) Originality: Scores are based on the rarity of responses. For instance, Take 
the word “chopsticks” as an example; participants may provide the following answers: “eating,” “stirring,” “hitting,” and “hairpin.” If 
“eating” appears in more than 10% of the sample, it scores 0. If the frequency of “stirring” is between 3% and 10%, it scores 1 point. 
“Hitting” in the sample between 1% and 3% scores 2 points, while “hairpin” with a frequency less than 1% scores 3 points. See Table 2 
for the scores associated with different frequencies; (2) Fluency: This is determined by the total number of different responses provided 
by the participant; and (3) Flexibility: The assessment considers the number of different use categories mentioned by participants. For 
example, using chopsticks for eating and using them as hairpins represents two categories while using chopsticks for eating and 
holding food represents one category. 
3.3. The bridge-the-associative-gap task 
The Bridge-the-Associative-Gap task is a convergent thinking test that requires participants to write a word related to two words 
given in the question. For example, if we provide the words ‘shoes’ and ‘legs,’ the participant might respond with ‘socks.’ To account 
for time and participant compliance, we used a shortened version of the Bridge-the-Associative-Gap Task (Gianotti et al., 2001). We 
also adapted some questions in the task to enhance understanding for 6- to 12-year-olds. Two raters scored the appropriateness of the 
responses on a scale of 1 to 7, based on how relevant the response was to the two words given in the question, with one indicating 
irrelevant and seven being relevant. The two raters demonstrated a high level of reliability (r = 0.72 to 0.76). We averaged the raters’ 
scores to obtain the final subjects’ scores. 
3.4. The divergent association task 
The Divergent Association Task requires participants to propose ten nouns that are as different as possible in meaning and usage. 
Subsequently, the semantic distance was calculated between these nouns, with smaller distances indicating greater similarity. For 
example, ‘prince’ and ‘princess’ would be relatively close, as they often occur together, while ‘prince’ and ‘artificial intelligence’ would 
not. In the original method (Olson et al., 2021), the GloVe model was employed to compute semantic distances. However, as we could 
not find an open-source GloVe model trained in Chinese, we used the Word2Vec model developed by Song et al. (2018), who trained 
the model on over 12 million Chinese words and phrases. To introduce some redundancy, we retained the first eight valid words 
provided by participants for computation. This involved obtaining the word vectors from the model, calculating semantic distances (i. 
e., cosine distances) between the words, averaging the values, and multiplying the result by 100 to derive the participant’s divergent 
thinking score. 
Table 2 
Scoring System for Determining Originality of Ideas.  
Frequency of Occurrence 
Score 
More than 10% 
0 point 
Between 3% and 10% 
1 point 
Between 1% and 3% 
2 points 
Less than 1% 
3 points  
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
6
4. Results 
4.1. Is it suitable to use the DAT to measure divergent thinking in elementary school students within Chinese contexts? 
To verify the applicability of the Divergent Association Task (DAT) for children aged 6 to 12, we conducted correlation analyses 
with two widely used creativity tests, namely the Alternative Uses Task (AUT) and the Bridge-the-Associative-Gap Task (BAG). The 
analysis demonstrated a strong correlation between the DAT and those two creativity tests (see Table 3). Specifically, there were 
positive correlations between DAT and the originality dimension of AUT [r (322) = .191, p = .001], DAT and the fluency dimension of 
AUT [r (322) = .192, p = .001], and DAT and the flexibility dimension of AUT [r (322) = .182, p = .001]. Similarly, we found a positive 
correlation between DAT and the appropriateness score [r (322) = .215, p < .001], indicating a strong association between DAT and 
BAG. 
4.2. How many nouns are needed to calculate a relatively accurate DAT score? 
The DAT task required participants to propose ten nouns that were entirely unrelated in meaning and usage. However, for the 
calculation of semantic distance, we only utilized a subset of these nouns. Some participants were unable to generate the ten words, but 
all valid questionnaires contained a minimum of eight words. Therefore, we calculated the DAT score using the first eight valid nouns 
that met the question requirements. The percentage of participants with varying numbers of valid nouns is shown in Fig. 1. 
We also explored the use of nine and ten nouns for semantic distance calculation and correlation analysis. When the number of 
words was nine, we found that DAT was associated with originality [r (317) = .210, p < .001], fluency [r (317) = .199, p < .001], 
flexibility [r (317) = .183, p = 0.001], and appropriateness [r (317) = .199, p < .001]. When the number of words was ten, we found 
that DAT correlated with originality [r (303) = .213, p < .001], fluency [r (303) = .231, p < 0.001], flexibility [r (303) = .215, p <
.001], and appropriateness [r (303) = .239, p < .001]. In sum, the correlation coefficients increased as the number of valid nouns 
increased (Table 4). Nevertheless, we believed that using eight nouns was sufficient for measuring divergent thinking since the DAT 
score showed significant correlations with the two widely used creativity tests. 
4.3. How do different language processing models affect the performance of the DAT? 
In this study, we used the Word2Vec model instead of the GloVe model for semantic distance calculation. We employed the 
Word2Vec model trained by Tencent AI Lab (Song et al., 2018), which was developed using a corpus comprising over 12 million 
Chinese vocabulary items. We utilized a 200-dimensional simplified version of this model (for details, see https://ai.tencent.com/ 
ailab/nlp/en/embedding.html). We believe that this model is capable of effectively handling a variety of downstream tasks in Chi­
nese linguistic contexts. The word vectors were retrieved and semantic distance calculations were performed using the gensim library 
in Python. This was due to the unavailability of an open-source GloVe model trained with a Chinese corpus. Moreover, we explored the 
impact of different natural language processing models on DAT. Specifically, we investigated two natural language processing models, 
the BERT model and the GPT model, as substitutes for the GloVe model in semantic distance calculation. For the Chinese BERT model, 
we utilized the one trained by KDDI (Cui et al., 2021). It is worth mentioning that the model employed a corpus comprising up to 540 
million words during training. The model we utilized is named ’BERT-wwm-ext, Chinese’. The ’BERT-base, Chinese’ model released by 
Google officially segments Chinese text at the character level, neglecting the issue of Chinese word segmentation. In contrast to 
Google’s official release, the ’BERT-wwm-ext, Chinese’ model takes into account the problem of Chinese word segmentation, making it 
more suitable for downstream tasks in Chinese contexts. We used the bert_serving library in Python to obtain word vectors and perform 
semantic distance calculations. As for the GPT model, we developed it by invoking the API provided by OpenAI (see https://platform. 
openai.com/docs/guides/embeddings) to acquire vector representations of words. Using BERT and GPT models on the same eight 
nouns for semantic distance calculation, we obtained the divergent thinking scores for each student. We then recalculated the cor­
relation between these newly obtained divergent thinking scores and the two widely used creativity tests. The results are displayed in 
Table 5. We found that using either the BERT model or the GPT model did not show a significant correlation with the other two 
creativity test scores. 
4.4. How does the divergent thinking ability of elementary school students change across grades? 
We further analyzed the scores of grades 1-6 in the DAT to explore how divergent thinking abilities vary across these grades. 
Scoring details are provided in Table 6. Fig. 2 presents example responses from three students (left to right: lowest, average, and 
Table 3 
Correlations between DAT and the Results of AUT and BAG Tests.  
DAT 
Correlation Coefficient 
Confidence Interval 
p 
Originality (AUT) 
.191 
[0.060, 0.298] 
.001 
Fluency (AUT) 
.192 
[0.079, 0.305] 
.001 
Flexibility (AUT) 
.182 
[0.069, 0.285] 
.001 
Appropriateness (BAG) 
.215 
[0.107, 0.319] 
.000  
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
7
highest DAT scores). The figure plots the vocabulary used by the students on the x and y axes. Numbers connecting vocabulary terms 
indicate semantic distance scores. The title of each panel shows the student’s total score. 
Moreover, we conducted a visual analysis of the relationship between the average scores of the DAT and grade levels (Fig. 3). This 
analysis revealed that the progression of DAT scores did not follow the linear growth trend with grade level advancement as initially 
anticipated. Contrarily, DAT scores showed a gradual increase from grades 1 to 3, reaching a peak value. However, between grades 4 
and 5, scores declined, followed by a resurgence of growth from grades 5 to 6. This finding reveals the complex variation of DAT scores 
across student grade levels, diverging from our initial expectations. 
To further investigate whether there are significant differences in scores between grades, we conducted an inter-group variability 
test among different grades and employed the Bonferroni correction method to identify the two grades with significant differences. 
Data indicating significant differences are presented in Table 7. 
Based on the analysis results presented in Table 7, we found significant differences in DAT scores only between students in grades 1 
and 6, 1 and 4, and 1 and 3. This suggests that student performance on the DAT differs notably between these three grade pairings. 
Furthermore, there appears to be a linear upward trend in DAT scores from grades 1 to 3. This pattern suggests that students may 
experience improvements in divergent thinking as they progress through these grades, possibly due to advancements in their academic 
level. 
% of participants 
Valid words 
Fig. 1. Percentage of Different Numbers of Valid Words in the Sample.  
Table 4 
Correlation Coefficients Between DAT and Scores of Two Other Creativity Measures Using Different Numbers of Words.  
N 
Originality (AUT) 
Fluency (AUT) 
Flexibility (AUT) 
Appropriateness (BAG) 
9 
.210 [0.097,0.321] 
.199 [0.094,0.313] 
.183 [0.075,0.292] 
.199 [0.082,0.306] 
10 
.213 [0.097,0.323] 
.231 [0.121,0.333] 
.215 [0.100,0.324] 
.239 [0.116,0.356] 
Note: N = Number of valid nouns. The numerical values within square brackets represent the confidence interval. 
Table 5 
Correlations Between DAT and Scores of Two Other Creativity Measures Using Different Language Models.  
Language model 
Originality (AUT) 
Fluency (AUT) 
Flexibility (AUT) 
Appropriateness (BAG) 
BERT 
.174 (p = .002) 
.105 (p = .060) 
.091 (p = .103) 
.098 (p = .079) 
GPT 
.222 (p = .000) 
.158 (p = .005) 
.103 (p = .066) 
.103 (p = .064)  
Table 6 
Scores of DAT by Different Grades.  
Grade 
Sample 
Mean 
SD 
Max 
Min 
1 
64 
55.31 
7.08 
67 
33 
2 
40 
58.57 
5.55 
68 
38 
3 
41 
59.59 
5.56 
68 
41 
4 
76 
59.07 
6.73 
72 
34 
5 
45 
57.02 
5.84 
70 
45 
6 
56 
59.09 
6.70 
74 
40  
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
8
Fig. 2. Examples of Students’ Responses.  
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
9
5. Discussion 
This study demonstrates the suitability of using the Divergent Association Task (DAT) for measuring divergent thinking in 
elemental school students (Olson et al., 2021). We explored the impact of varying effective word counts and different natural language 
processing models on the DAT task. By comparing its correlation with two traditional and widely used creativity tests (AUT and BAG), 
we found strong correlations, suggesting that the DAT task is applicable for assessing divergent thinking in children aged 6 to 12. 
Particularly, the correlation coefficients between DAT and both AUT and BAG tasks increased with the number of valid words. Since 
not all subjects could provide ten valid nouns, this study showed that the DAT was effective with eight valid words. Using eight words 
could ensure experimental redundancy and reduce the participants’ burden. 
Furthermore, we used the natural language processing models of BERT and GPT to replace Word2Vec for semantic distance 
calculation. Interestingly, using either the BERT or GPT model did not exhibit a significant correlation with the scores from the other 
two creativity tests. DAT scores derived from the BERT and GPT models were generally low and did not effectively reflect students’ 
divergent thinking skills. Taking “tissue paper” and “cell phone” as an example, the cosine distances calculated by the Word2Vec, 
BERT, and GPT models were 0.57, 0.12, and 0.22, respectively. We tentatively hypothesize that this lack of correlation may stem from 
the extensive word vector space of large language models, resulting in small semantic distances and, subsequently, lower DAT scores. 
Based on this finding, we suggest that using the Word2Vec model may be more suitable for measuring students’ divergent thinking 
with the DAT. 
Upon obtaining the applicability of the DAT task, we investigated the state of divergent thinking among children aged 6 to 12 years. 
Contrary to our initial expectation of a linear increase in scores with advancing grades, the results showed a gradual increase from 1st 
to 3rd grade, followed by a decline starting after 3rd grade. Notably, there was a significant decrease by the 5th grade, before scores 
began to rise again in the 6th grade. This pattern aligns with findings from Alacapinar (2013), who, in exploring the relationship 
between grade level and creativity, observed that creativity scores significantly increased from 3rd to 5th grade and then declined from 
6th to 8th grade. It is important to note that creativity development is influenced by a complex interplay of cognitive, environmental, 
and educational factors. The non-linear trend observed in this study highlights the need for continued research and a nuanced un­
derstanding of how various factors shape creative thinking abilities at different developmental stages. 
This study is not without limitations. Firstly, the use of DAT for assessing divergent thinking in this age group is in its early stages, 
and the sample size obtained in this study is relatively small. Secondly, elementary school students may struggle to comprehend the 
task requirements of the DAT. In our study, it was noted that some participants lacked an understanding of what a noun was, and 
participants tended to seek inspiration by looking around the classroom. These factors can influence the total score and lead to an 
inaccurate measurement of divergent thinking (Olson et al., 2021). Moreover, we established the validity of DAT by comparing its 
results with that of the AUT (Guilford,1967). However, Guilford’s theory, while valuable, oversimplifies creativity by focusing on 
quantity of ideas and does not account for how knowledge and experience spark truly novel thinking (Sternberg & Grigorenko, 2001). 
Finally, the computation method of DAT is constrained by different regions and cultures. In the DAT task, the GloVe model was 
DAT score
Grade
Fig. 3. Variation in Average DAT Scores Among Students Across Different Grade Levels.  
Table 7 
Grade Groups with Significant Differences.  
Grade 
SE 
t 
p 
Adjusted p 
1 and 6 
17.01 
-3.01 
.003 
.039* 
1 and 4 
15.77 
-3.53 
.000 
.006** 
1 and 3 
18.59 
-3.44 
.001 
.009** 
Note: * p < .05, ** p <.01. SE = Standard Error. 
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
10
originally employed to compute semantic distance between words. However, some regions may lack the GloVe model for their 
respective languages or cultures. Therefore, researchers need to explore alternative linguistic models to replace GloVe. For instance, in 
this study, the Word2Vec model was used for semantic distance calculation. The impact of different natural language processing 
models on the calculation of semantic distance between words and, consequently, on divergent thinking remains unclear (Beaty & 
Johnson, 2021). 
6. Conclusion 
In this study, we explored the applicability of the DAT (Divergent Association Task) for elementary school students in Chinese 
contexts, given that it was not originally designed for this specific population and was available only in English. We recruited a total of 
348 students who were asked to complete three creativity tasks: the DAT, the Alternative Uses Task (AUT), and the Bridge-the- 
Associative-Gap Task (BAG). We examined the associations between DAT and the scores of the AUT and BAG tests. Moreover, we 
tested the accuracy of the DAT using varying numbers of nouns and different natural language processing models to calculate the 
semantic distance between nouns. Our findings supported the suitability of using the DAT to measure divergent thinking in elementary 
school students within Chinese contexts. We also found that using only eight nouns, instead of ten, could achieve a relatively high 
accuracy in measuring divergent thinking based on the DAT method. The language model of Word2Vec performed better than the 
BERT and GPT models when calculating semantic distances between nouns. 
This study has methodological and practical implications. Our study suggests that the DAT could be adapted for measuring 
divergent thinking abilities across different cultural and linguistic backgrounds. When measuring divergent thinking with DAT, users 
should select an appropriate natural language processing model, for instance, GloVe or Word2Vec models, to calculate semantic 
distances among words based on the cultural and linguistic background of the test sample. It is also important to be aware of the 
potential impact of more complex models like BERT or GPT on the calculation of semantic distances. Moreover, users should carefully 
consider the number of effective nouns used for calculating semantic distances. For instance, if participants struggle to generate a 
sufficient number of nouns, reducing the required number of effective nouns may be advisable while ensuring enough redundancy to 
maintain the validity of the DAT. 
This study also expands the methodological toolbox for assessing creativity in elementary school students. Practically, it suggests 
that the DAT can be a valuable tool for assessing divergent thinking in the age group of 6 to 12 years old within the Chinese context. 
Using eight nouns in the DAT could still yield high accuracy, implying that educators and researchers can potentially save time and 
resources without compromising the quality of creativity assessments. Compared to conventional approaches for assessing divergent 
thinking, the DAT offers enhanced measurement efficiency, enabling educators to rapidly evaluate students’ divergent thinking ca­
pabilities. This facilitates the design of personalized learning plans tailored to stimulate students’ innovative potential. Moreover, by 
periodically administering the DAT, teachers can monitor the longitudinal development of students’ divergent thinking abilities, 
thereby assessing the effectiveness of their instructional strategies. Furthermore, the development of web-based platforms for col­
lecting DAT scores could facilitate research and comparative analyses of the impact of diverse educational environments on students’ 
divergent thinking abilities. Such insights could provide valuable guidance to educational management entities in formulating and 
refining policies and practices related to fostering creativity and innovation in educational settings. 
CRediT authorship contribution statement 
Guozhu Ding: Supervision, Project administration, Investigation, Funding acquisition, Conceptualization. Yiwei He: Writing – 
original draft, Investigation, Formal analysis, Data curation. Kaixu Yi: Writing – original draft, Investigation, Formal analysis, Data 
curation, Conceptualization. Shan Li: Writing – review & editing, Writing – original draft, Supervision, Methodology. 
Declaration of competing interest 
There are no conflicts of interest associated with this work. 
Data availability 
Data will be made available on request. 
References 
Acar, S., & Runco, M. A. (2014). Assessing associative distance among ideas elicited by tests of divergent thinking. Creativity Research Journal, 26(2), 229–238. https:// 
doi.org/10.1080/10400419.2014.901095 
Acar, S., & Runco, M. A. (2015). Thinking in multiple directions: hyperspace categories in divergent thinking. Psychology of Aesthetics, Creativity, and the Arts, 9(1), 41. 
https://doi.org/10.1037/a0038501 
Acar, S., & Runco, M. A. (2019). Divergent thinking: new methods, recent research, and extended theory. Psychology of Aesthetics, Creativity, and the Arts, 13(2), 153. 
https://doi.org/10.1037/aca0000231 
Alacapinar, F. G. (2013). Grade Level and Creativity. Eurasian Journal of Educational Research, 50, 247–266. 
Amabile, T. M. (1982). Social psychology of Creativity: A consensual assessment technique. Journal of personality and social psychology, 43(5), 997. https://doi.org/ 
10.1037/0022-3514.43.5.997 
G. Ding et al.                                                                                                                                                                                                           

Thinking Skills and Creativity 52 (2024) 101503
11
Amabile, T. M., & Pratt, M. G. (2016). The dynamic componential model of Creativity and innovation in organizations: Making progress, making meaning. Research in 
organizational behavior, 36, 157–183. https://doi.org/10.1016/j.riob.2016.10.001 
Ananiadou, K., & Claro, M. (2009). 21st century skills and competencies for new millennium learners in OECD countries. OECD Publishing (NJ1). OECD education working 
papers, no. 41. 
Beaty, R. E., & Johnson, D. R. (2021). Automating creativity assessment with SemDis: An open platform for computing semantic distance. Behavior research methods, 
53(2), 757–780. https://doi.org/10.3758/s13428-020-01453-w 
Becker, M., Wiedemann, G., & Kühn, S. (2020). Quantifying insightful problem solving: a modified compound remote associates paradigm using lexical priming to 
parametrically modulate different sources of task difficulty. Psychological research, 84, 528–545. https://doi.org/10.1007/s00426-018-1042-3 
Benedek, M., K¨onen, T., & Neubauer, A. C. (2012). Associative abilities underlying Creativity. Psychology of Aesthetics, Creativity, and the Arts, 6(3), 273. https://doi. 
org/10.1037/a0027059 
Beketayev, K., & Runco, M. A. (2016). Scoring divergent thinking tests by computer with a semantics-based algorithm. Europe’s journal of psychology, 12(2), 210. 
https://doi.org/10.5964/ejop.v12i2.1127 
Besançon, M., & Lubart, T. (2008). Differences in the development of creative competencies in children schooled in diverse learning environments. Learning and 
individual differences, 18(4), 381–389. https://doi.org/10.1016/j.lindif.2007.11.009 
Brophy, D. R. (2001). Comparing the attributes, activities, and performance of divergent, convergent, and combination thinkers. Creativity Research Journal, 13(3-4), 
439–455. https://doi.org/10.1207/S15326934CRJ1334_20 
Cui, Y., Che, W., Liu, T., Qin, B., & Yang, Z. (2021). Pre-training with whole word masking for Chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language 
Processing, 29, 3504–3514. https://doi.org/10.1109/TASLP.2021.3124365 
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint. https://doi.org/ 
10.48550/arXiv.1810.04805. arXiv:1810.04805. 
Dumas, D., Organisciak, P., & Doherty, M. (2021). Measuring divergent thinking originality with human raters and text-mining models: a psychometric comparison of 
methods. Psychology of Aesthetics, Creativity, and the Arts, 15(4), 645–663. https://doi.org/10.1037/aca0000319 
Forthmann, B., Wilken, A., Doebler, P., & Holling, H. (2019). Strategy induction enhances Creativity in figural divergent thinking. The Journal of Creative Behavior, 53 
(1), 18–29. https://doi.org/10.1002/jocb.159 
Gardner, H., & Gardner, E. (2008). Art, mind, and brain: A cognitive approach to creativity. Basic Books.  
Gianotti, L. R., Mohr, C., Pizzagalli, D., Lehmann, D., & Brugger, P. (2001). Associative processing and paranormal belief. Psychiatry and clinical neurosciences, 55(6), 
595–603. https://doi.org/10.1046/j.1440-1819.2001.00911.x 
Guilford, J. P. (1967). The nature of human intelligence. McGraw-Hill.  
Hass, R. W. (2017). Semantic search during divergent thinking. Cognition, 166, 344–357. https://doi.org/10.1016/j.cognition.2017.05.039 
Hocevar, D., & Michael, W. B. (1979). The effects of scoring formulas on the discriminant validity of tests of divergent thinking. Educational and Psychological 
Measurement, 39(4), 917–921. https://doi.org/10.1177/001316447903900427 
Kaufman, J. C., & Beghetto, R. A. (2009). Beyond big and little: The four C model of Creativity. Review of general psychology, 13(1), 1–12. https://doi.org/10.1037/ 
a0013688 
Kaufman, J. C., & Sternberg, R. J. (Eds.). (2010). The Cambridge Handbook of Creativity. Cambridge University Press.  
Maker, C. J., Jo, S., & Muammar, O. M. (2008). Development of creativity: The influence of varying levels of implementation of the DISCOVER curriculum model, a 
non-traditional pedagogical approach. Learning and Individual Differences, 18(4), 402–417. https://doi.org/10.1016/j.lindif.2008.03.003 
Mednick, S. (1962). The associative basis of the creative process. Psychological Review, 69(3), 220–232. https://doi.org/10.1037/h0048850 
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint. https://doi.org/10.48550/ 
arXiv.1301.3781. arXiv:1301.3781. 
Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E. (2021). Naming unrelated words predicts Creativity. Proceedings of the National Academy of 
Sciences, 118(25), Article e2022340118. https://doi.org/10.1073/pnas.2022340118 
Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural 
language processing (EMNLP) (pp. 1532–1543). 
Plucker, J. A., Makel, M. C., & Qian, M. (2010). Assessment of Creativity. The Cambridge Handbook of Creativity (pp. 48–73). 
Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. 
Runco, M. A., & Johnson, D. J. (2002). Parents’ and teachers’ implicit theories of children’s Creativity: a cross-cultural perspective. Creativity Research Journal, 14(3- 
4), 427–438. https://doi.org/10.1207/S15326934CRJ1434_12. Creativity Research Journal, 14(3-4), 427-438. 
Silvia, P. J., Martin, C., & Nusbaum, E. C. (2009). A snapshot of Creativity: Evaluating a quick and simple method for assessing divergent thinking. Thinking Skills and 
Creativity, 4(2), 79–85. https://doi.org/10.1016/j.tsc.2009.06.005 
Silvia, P. J. (2015). Intelligence and Creativity are pretty similar after all. Educational Psychology Review, 27, 599–606. https://doi.org/10.1007/s10648-015-9299-1 
Song, Y., Shi, S., Li, J., & Zhang, H. (2018). Directional skip-gram: Explicitly distinguishing left and right context for word embeddings. In Proceedings of the 2018 
Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers) (pp. 175–180). 
https://doi.org/10.18653/v1/N18-2028 
Sternberg, R. J., & Grigorenko, E. L. (2001). Guilford’s structure of intellect model and model of creativity: Contributions and limitations. Creativity Research Journal, 
13(3-4), 309–316. 
Urban, K. K. (1991). On the development of creativity in children. Creativity Research Journal, 4(2), 177–191. https://doi.org/10.1080/10400419109534384 
Volle, E. (2018). Associative and controlled cognition in divergent thinking: Theoretical, experimental, neuroimaging evidence, and new directions. The Cambridge 
Handbook of the Neuroscience of Creativity (pp. 333–362). https://doi.org/10.1017/9781316556238.020 
Wallach, M. A., & Kogan, N. (1965). Modes of thinking in young children. New York: Holt, Rinehart and Winston.  
Wu, C. L., & Chen, H. C. (2017). Normative data for Chinese compound remote associate problems. Behavior Research Methods, 49, 2163–2172. https://doi.org/ 
10.3758/s13428-016-0849-3 
G. Ding et al.
