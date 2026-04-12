# Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co

- Source PDF: `raw/pdfs/Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co.pdf`
- Generated from: `scripts/extract_pdf_text.py`

## Extracted Text

Psychology of Aesthetics, Creativity,
and the Arts
Measuring Divergent Thinking Originality With Human
Raters and Text-Mining Models: A Psychometric
Comparison of Methods
Denis Dumas, Peter Organisciak, and Michael Doherty
Online First Publication, July 23, 2020. http://dx.doi.org/10.1037/aca0000319
CITATION
Dumas, D., Organisciak, P., & Doherty, M. (2020, July 23). Measuring Divergent Thinking Originality
With Human Raters and Text-Mining Models: A Psychometric Comparison of Methods. Psychology
of Aesthetics, Creativity, and the Arts. Advance online publication.
http://dx.doi.org/10.1037/aca0000319

Measuring Divergent Thinking Originality With Human Raters and Text-
Mining Models: A Psychometric Comparison of Methods
Denis Dumas and Peter Organisciak
University of Denver
Michael Doherty
Actor’s Equity Association, New York, New York
Within creativity research, interest and capability in utilizing text-mining models to quantify the
Originality of participant responses to Divergent Thinking tasks has risen sharply over the last decade,
with many extant studies fruitfully using such methods to uncover substantive patterns among creativity-
relevant constructs. However, no systematic psychometric investigation of the reliability and validity of
human-rated Originality scores, and scores from various freely available text-mining systems, exists in
the literature. Here we conduct such an investigation with the Alternate Uses Task. We demonstrate that,
despite their inherent subjectivity, human-rated Originality scores displayed the highest reliability at both
the composite and latent factor levels. However, the text-mining system GloVe 840B was highly capable
of approximating human-rated scores both in its measurement properties and its correlations to various
creativity-related criteria including ideational Fluency, Elaboration, Openness, Intellect, and self-reported
Creative Activities. We conclude that, in conjunction with other salient indicators of creative potential,
text-mining models (and especially the GloVe 840B system) are capable of supporting reliable and valid
inferences about Divergent Thinking. We offer an open-access module for researchers to apply these
methods to their own data via our laboratory website (https://openscoring.du.edu/).
Keywords: creativity, divergent thinking, psychometrics, reliability, text-mining models
Supplemental materials: http://dx.doi.org/10.1037/aca0000319.supp
Divergent thinking (DT)—or the human mental ability to gen-
erate multiple original ideas in response to a given problem or
prompt (Acar & Runco, 2019; Forthmann, Wilken, Doebler, &
Holling, 2019)—has long been a centrally important construct
under investigation within the creativity research literature (e.g.,
Hocevar, 1980). Today, DT tasks are by far the most utilized
measures in creativity research (Plucker & Makel, 2010; Reiter-
Palmon, Forthmann, & Barbot, 2019), with the Alternate Uses
Task (AUT; Guilford, 1967; Hudson, 1968; Torrance, 1972) being
the chief among them. Therefore, procedures and methods that are
used to calculate participant scores from their AUT or other DT
task responses are closely considered, and at times psychometri-
cally examined, by creativity researchers (e.g., Dumas & Dunbar,
2014; Kuhn & Holling, 2009). However, despite the broad use of
the AUT in creativity research, the extant psychometric under-
standing of participant scores from the AUT is simply not as
developed as in other areas of psychology (e.g., clinical personality
assessment; Marek & Ben-Porath, 2017), for a variety of reasons.
In our view, one principal reason why DT measurement is
somewhat underdeveloped is because DT tasks fundamentally rely
on open-ended participant responses, whereas the vast majority of
psychometric modeling frameworks that can be used to evaluate
the reliability or internal validity of scoring models were designed
for close-ended assessment (e.g., item response theory; Lord,
2012). In fact, the AUT is not only open-ended in its response
format (i.e., participants must verbally respond or write-in their
responses rather than selecting a response-option) it is also ill-
structured (i.e., participants can differ substantially on the number
of responses they supply and the length of those responses): an
assessment format that stumps most currently dominant psycho-
metric scoring methods. One approach to the scoring of open-
ended divergent thinking tasks that focuses on taking a relatively
rapid snapshot of creativity is a subjective scoring procedure
introduced by Silvia and colleagues (2008). However, the search
for more objective performance measurement strategies for diver-
gent thinking tasks—especially measurement strategies that can be
Denis Dumas and X Peter Organisciak, Department of Research Meth-
ods and Information Science, Morgridge College of Education, University
of Denver; Michael Doherty, Actor’s Equity Association, New York, New
York.
This research was supported financially through a research seed-grant
from the University of Denver’s Morgridge College of Education. We
thank Amanda Strickland, Megan Solberg, and Danielle Francisco Albu-
querque Vasques for their critical assistance in coding the responses.
Correspondence concerning this article should be addressed to Denis
Dumas, Department of Research Methods and Information Science, Mor-
gridge College of Education, University of Denver, 1999 East Evans
Avenue, Denver, CO 80210. E-mail: denis.dumas@du.edu
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
Psychology of Aesthetics, Creativity, and the Arts
© 2020 American Psychological Association
2020, Vol. 2, No. 999, 000
ISSN: 1931-3896
http://dx.doi.org/10.1037/aca0000319
1

used as part of an automatic scoring system—has been more
elusive (Dumas & Runco, 2018).
Fortunately, relatively recent advances in text-mining method-
ology (e.g., Hirschberg & Manning, 2015) are offering some
solutions for the measurement of psychological attributes from
open-ended and ill-structured assessment data. With the advent of
a relatively new interdisciplinary area of research that combines
text-mining and psychometric methods—sometimes termed com-
putational psychometrics (von Davier, 2017)—the reliable quan-
tification of human mental attributes that are too complex to be
assessed via close-ended test items (e.g., DT) is becoming more
possible than ever. In the next sections of this article, we first
briefly overview areas in which computational psychometrics has
opened important doors in psychological research and applied
testing contexts, and then move to a detailed review of the existing
computational psychometric applications in the creativity research
literature.
Computational Psychometrics for
Open-Ended Measures
For decades, social scientists engaged in basic research have
understood that a close-ended psychometric methodological per-
spective—in which participants are administered questionnaires or
tests, and the answer-choices that participants select are used to
calculate their score—has some serious limitations for the study of
complex human thought and behavior (e.g., Messick, 1995). For
example, many investigations have utilized rich sources of inter-
view, observation, essay, or otherwise open-ended and ill-
structured data, which is typically scored through the application
of human-raters who judge the degree to which those data indicate
the presence of a given psychological construct within a partici-
pant (e.g., Bråten, Ferguson, Strømsø, & Anmarkrud, 2014). How-
ever, the use of such human-raters is costly both in terms of time
and money, and the reliability and validity of human-rated partic-
ipant scores is subject to measurement error from the raters’
implicit beliefs and biases. The subjective nature of such human-
rated scores has often been criticized in the literature (Gwet, 2014),
although, in the case of open-ended and ill-structured data sources,
no alternative has historically existed.
However, in the last decade, major advances have been made in
devising methods for the objective (i.e., not human-rated) and
automatic (i.e., computer-generated) scoring of open-ended psy-
chological data sources, at least those that are textual or verbal in
format. These computational psychometric approaches are based
on text-mining models that are trained on a massive corpus of
extant text (e.g., a library of digitized books; Crossley, Dascalu, &
McNamara, 2017) to represent the semantic meaning of words in
the context in which they are used. For example, the most popular
of these text-mining models in the psychological literature today is
Latent Semantic Analysis (LSA; Landauer, Foltz, & Laham,
1998), which is a dimensionality reduction technique that seeks to
reduce a sparse-matrix of document-term counts, where each doc-
ument is represented by the counts of all the words in the model
vocabulary, to a much smaller representation of documents as a
few hundred features. This is done by finding co-occurrence pat-
terns between all the words in a document. For example, rather
than representing the words dog, puppy, and canine all as inde-
pendent quantitative counts, an LSA trained model would learn
their relationship and represent each of those words similarly
across the latent dimensions of the model. This has the effect of
accounting for similar words and synonyms. Even though the
words in “the dog barked” and “the puppy barked” are different,
the contextual difference in not great, and they will appropriately
be understood as similar in an LSA representation of those state-
ments (Günther, Dudschig, & Kaup, 2015). The substantive effect
of this modeling of latent dimensions is that words have quantifi-
able distances between themselves in the trained model, and how
close or far two words are from each other tends to align with
humans’ perceptions of whether words are semantically similar or
dissimilar (Landauer & Dumais, 1997).
This relatively nuanced representation of the semantic structure
of language through text-mining models has allowed psychome-
tricians to begin to objectively and automatically quantify such
complex psychological constructs as writing ability (i.e., via essay
prompts and automatic essay scoring systems; Foltz, Streeter,
Lochbaum, & Landauer, 2013), depression (i.e., through the tex-
tual analysis of patient interview data; Kjell, Kjell, Garcia, &
Sikström, 2019), and collaborative ability (i.e., by examining the
semantic structure of online chats in which participants collaborate
to solve an abstract problem; He, von Davier, Greiff, Steinhauer, &
Borysewicz, 2017). Indeed, all of these applications are beginning
to become relatively large scale, with automatic essay scoring
systems being deployed widely in standardized admissions testing,
mental health related text-mining systems beginning to make their
debut in clinics, and the text-based assessment of student collab-
orative skills being administered to students all around the world
as part of the Program for International Student Assessment
(PISA) tests.
Computational Psychometrics in Creativity Research
Since the theorizing of Mednick (1962), creativity has often
been considered from the viewpoint of associative distance. From
this perspective, the Originality of a given response is conceptu-
alized as arising from its distal-relatedness from the context or
prompt from which it arose, such that a response that would be
typically associated with a given prompt or context would be
considered to be low in terms of Originality, whereas a response
that is more unusual within a given context would be considered to
be more Original. Importantly, the theoretical conceptualization of
Originality as the associative distance between a prompt and the
response a participant generates can be operationalized (at least for
verbal or textual DT responses such as from the AUT) as the
semantic distance between a given response and the AUT prompt
(e.g., “Brick”) from which it arose (Beaty, Silvia, Nusbaum, Jauk,
& Benedek, 2014). This theoretical pairing of the associative
distance among ideas and the semantic distance among the words
used to express these ideas (i.e., the responses) may not be per-
fectly one-to-one (cf. historic “thought and language” debates in
psychology; cf., Piatelli-Palmarini, 1980), but it appears close
enough to justify the application of computational psychometric
approaches (e.g., LSA) to DT data to quantify the semantic dis-
tances among the prompts and responses.
Latent Semantic Analysis and Divergent Thinking
In the creativity research literature, the text-mining model that is
most typically utilized is LSA (Acar & Runco, 2019). LSA has
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
2
DUMAS, ORGANISCIAK, AND DOHERTY

been an appropriate and useful method for the quantification of
Originality because the process of training an LSA system is
effective at preserving the linearity of the relations among words,
so that the semantic distances between them are directly compa-
rable by studying the factorization matrix of words by latent
dimensions. Using these matrices, the latent dimensions in an LSA
model can be used as coordinates in a geometrically represented
space, and the cosine of the angle between the word-vectors can be
interpreted as the semantic or associative distance among words
(Deerwester, Dumais, Furnas, Landauer, & Harshman, 1990). As
an example with the AUT, if the prompt was “fork,” the response
“eat pasta” would result in a vector that has an acute angle with the
vector for “fork.” In Contrast, the response “conduct electricity”
would result in a vector that has a wider angle from the initial
prompt vector. Please see Figure 1 for a visualization of the
geometric relations among AUT prompt and responses in LSA. To
calculate the Originality scores for these AUT responses, the
cosine of the angle would be calculated to represent the semantic
distance between the prompt and response, and then that semantic
distance would be subtracted from 1 to yield an Originality score
for each response.
Following this general methodological paradigm, a number of
studies both of the measurement-related functioning of LSA based
Originality scores (e.g., Dumas & Runco, 2018; Forthmann, Oye-
bade, Ojo, Günther, & Holling, 2019; Heinen & Johnson, 2018;
Prabhakaran, Green, & Gray, 2014), as well the application of
LSA Originality scores to answering substantive research ques-
tions in the creativity research literature (e.g., Hass, 2017a; White
& Shah, 2016) have appeared. As far as we are aware, Kevin
Dunbar and his students and collaborators (e.g., Dumas & Dunbar,
2014; Forster & Dunbar, 2009; Green, Kraemer, Fugelsang, Gray,
& Dunbar, 2012) were the first to recognize the rich possibilities
in the application of LSA to DT and other cognitive tasks. As such,
Forster and Dunbar (2009) presented the first formal evidence of
the appropriateness of LSA based Originality scores. In this work,
they showed that LSA Originality scores were capable of discrim-
inating between groups of participants who received different
directions to the AUT (i.e., creative responses and common re-
sponses), and that LSA based Originality scores were more
strongly predictive of human-rated Originality than were other
common DT metrics such as Fluency and Elaboration.
Following this first foray into LSA based Originality scoring,
Green and colleagues (e.g., Green, Kraemer, Fugelsang, Gray, &
Dunbar, 2010) used LSA to examine the link between participant
relational reasoning abilities and divergent thinking. In addition to
this work, Dumas and Dunbar (2014) published the first psycho-
metric investigation of LSA based Originality scores and their
relation to Fluency scores from a latent variable perspective. This
study found that a confirmatory factor analysis (CFA) model that
included both LSA based Originality scores and Fluency counts
across 10 AUT prompts was capable of fitting the observed data
very closely and achieved a high degree of reliability for both the
Fluency and Originality latent factors. In this way, LSA Original-
ity scores have demonstrated discriminant validity from Fluency
scores, and in another analysis, were shown to demonstrate a high
degree of reliability even after the variance explained by Fluency
was partialed out (Dumas & Runco, 2018).
Over the next few years, a solid handful of substantive appli-
cations of LSA Originality scores appeared in the creativity liter-
ature, demonstrating that this method was capable of producing
scores that allowed creativity researchers to gain insights about
psychological phenomena related to creativity. For example,
White and Shah (2016) showed that LSA semantic distances
among word association pairs were capable of explaining observed
advantages of individuals with attention-deficit/hyperactivity dis-
order (ADHD) on a DT task, leading to the hypothesis that ADHD
may support DT in part because of the wider scope of semantic
activation in those individuals. The same year, Dumas and Dunbar
(2016) showed that participants’ LSA-based Originality scores
were significantly influenced by DT task instructions, particularly
when participants were asked to take the perspective of stereotypi-
cally creative individuals such as poets. The following year, two
papers by Hass (2017a, 2017b) applied LSA-based Originality
scores to a fine-grained analysis of DT production over short spans
of time. For example, Hass (2017a) showed that the LSA-based
semantic similarity (the inverse of Originality) of AUT responses
were negatively correlated with human-judged creativity ratings,
which provided evidence for the validity of LSA-based DT scores.
In the same article, the semantic similarity of AUT responses
followed a cubic trend overtime: The most highly semantically
similar responses to the prompt occurred first, the similarity of
responses then decreased (or Originality increased) over the first
five or so responses, before increasing again between the fifth and
10th responses, and after the 10th response the similarity decreased
once more. In addition, Hass (2017b) then demonstrated that the
fluid intelligence of participants significantly influenced their abil-
ity to generate semantically distant (i.e., Original) responses to the
AUT (although the change overtime of semantic distance was
linear, not cubic in that investigation).
Later, Dumas (2018) used LSA-based Originality scores to test
the long-standing threshold hypothesis (e.g., Karwowski &
Gralewski, 2013) in the creativity literature: that intellectual ability
Figure 1.
A visual representation of the geometric relations among vec-
tors arising from an Latent Semantic Analysis (LSA) analysis. These
angles were calculated using an LSA model trained on the Touchstone
Applied Science Associates (TASA) corpus. To calculate the Originality of
each of the Alternate Uses Task (AUT) responses, the cosine of the angle
would be calculated to produce a semantic similarity score, and then that
score would be subtracted from one to generate the Originality for each
use. The cosine distance is taken to account for document length: Because
words are represented in the latent model, multiword phrases can be
represented as a sum of all word vectors (e.g., vec(eat)  vec(pasta)) while
retaining the ability to compare them with one-word phrases.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
3
MEASURING ORIGINALITY

supports creative ability, but only up to a point. In this study,
LSA-based Originality scores on the AUT were able to reach a
strong level of scale reliability, and allowed for an analysis in
which the threshold hypothesis was supported under some condi-
tions, but not in others. This finding, along with those of the Hass
(2017a, 2017b) articles, may illustrate how semantic distance can
be used as a fruitful operationalization of Originality to address
longstanding questions in creativity research. In addition, Dumas
and Strickland (2018) applied LSA-based Originality scores to
investigate malevolent or violent responses on the AUT, and found
that those participants who scored more highly on their Originality
also produced more violent responses on the AUT (e.g., “kill
someone” as a use for “shovel”). Such a finding adds to the
potential evidence for the predictive validity of LSA-based Orig-
inality scores, because those scores were capable of significantly
and positively predicting a theoretically relevant creativity-related
construct (i.e., malevolence). Even more recently, Gray and col-
leagues (2019) used LSA-based scores to quantify the semantic
distance among responses to word association tasks. Similarly to
Hass (2017a), these researchers pointed out that the changes in
semantic distance over time as participants generate responses is
informative as to their creative potential.
Limitations of LSA
Since the very first applications of LSA in the psychological
literature (see Landauer, McNamara, Dennis, & Kintsch, 2013 for
a handbook of reviews), it has been understood that the reliability
and validity of participant scores from computational psychomet-
ric models that incorporate LSA depend on a number of factors.
Like all DT assessment, the quality of LSA-based Originality
scores depends on a myriad of administrative and scoring choices,
but some that are of particular importance in the LSA context
include: (a) the corpus from which the LSA system was trained, (b)
the particular methodological decisions for how to handle ex-
tremely common words like “and” or “is,” and (c) the way in
which semantic similarity or distance scores for individual DT
responses are aggregated to the prompt (e.g., “Brick”) level, or the
participant level. Each of these issues is now briefly explained.
Training corpus.
Beginning with Forster and Dunbar’s
(2009) initial work, by far the most widely utilized training corpus
for LSA in the creativity research literature has been the Touch-
stone Applied Science Associates (TASA) corpus. This corpus was
originally created by Landauer and Dumais (1997) and was ini-
tially applied to the psychological research literature by Walter
Kintsch and his colleagues (e.g., Kintsch & Bowles, 2002). As part
of the interdisciplinary work among these scholars, a freely acces-
sible tool was made available to access this corpus, originally
through an Internet browser (i.e., lsa.colorado.edu) but today also
through the open-source software r (i.e., LSAfun; Günther et al.,
2015). This corpus is composed of nearly 40 thousand educational
texts and is meant to represent the average reading experience of
the typical entering American undergraduate student (who are the
most commonly recruited participants in psychology studies).
However, the TASA corpus was created in the late 1990s and, as
far as we are aware, has not been updated since the very early
2000s, calling into serious question the capacity of this corpus to
continue to represent the true semantic relations of current lan-
guage. Because the goal of the system is to adequately learn
relations between words, more training texts generally lead to a
better sense of the true semantic structure of language, and the
TASA corpus is relatively small by modern standards. Finally, the
type of texts trained on will affect the relations between words
because, say, new articles, conversational posts, and legal argu-
ments all have different styles of language. Which domain of
corpora leads to semantic models that are most appropriate for
Originality scoring is not yet known, and such a question seems
worth exploring beyond the formal educational texts uses for the
TSA corpus.
Currently, only a small minority of creativity researchers have
used alternative training corpora for their LSA-based investiga-
tions. For example Forthmann and colleagues (2019) have used the
more modern English 100k corpus, which is much more generally
based on a web crawl of .uk domains of the Internet. However,
most LSA-based creativity research (e.g., Dumas & Dunbar, 2014,
2016; Gray et al., 2019) continues to be done using the TASA
corpus, raising a possible validity risk.
Elaboration confound.
Because participant responses to DT
tasks can contain varying amounts of words (i.e., they can vary in
their Elaboration), LSA-based scoring may be adversely affected
by these differences. As first pointed out by Forster and Dunbar
(2009) and technically examined by Forthmann and colleagues
(2019), LSA-based Originality scores at the participant level com-
monly exhibit a substantial correlation with Elaboration, implying
that the more words a participant uses to explain their idea, the
more LSA estimates of Originality are confounded. As a slightly
more technical description, because LSA Originality scores are
based on vectors for the entire DT response and not just individual
words (see Figure 1), those vectors are essentially composed of the
sum of the individual word vectors for every word in the DT
response (Landauer, Laham, Rehder, & Schreiner, 1997). Because
some words used in a response (e.g., “and” or “so”) may be very
commonly utilized, they are not particularly discriminatory. These
commonly used function words lower the semantic distance of the
full DT response from the prompt, even though the core idea of the
response may have been highly Original. Such an understanding of
LSA makes it clear that, going forward, DT tasks must be scored
for Elaboration to identify (and possibly control for) confounds in
substantive studies. In addition, the relation between text-mining–
based Originality scores and human rated Originality scores needs
to continue to be checked to ensure that the influence of Elabora-
tion does not throw-off this relation. In addition, Forthmann and
colleagues (2019) do offer some statistical corrections for formu-
lating LSA-based Originality scores that can alleviate this prob-
lem. For instance, one way to control for the misleading effects of
common function words is “stopword lists,” which simply remove
(or stop) a set of words based on a known list. In this article, a
correction known as term weighting is applied to all systems under
investigation, to weight different words to be more or less impact-
ful based on how discriminatory they are. Here, Inverse-document-
frequency (IDF) from the information science literature is used as
the term weighting method (Robertson & Jones, 1976), using
precomputed term weights that emphasize the influence of less
common words (Organisciak, 2016). Correlations with Elaboration
scores are also checked here as an index of discriminant validity.
Scoring aggregation method.
One perhaps unfortunate pat-
tern within the creativity research literature is that DT assessments
are often thought of by researchers simply as tasks and not mea-
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
4
DUMAS, ORGANISCIAK, AND DOHERTY

sures. Although this is a subtle distinction, task is a much more
general category that includes any stimuli designed to elicit a
certain cognitive process or behavior from participants, whereas a
measure requires multiple items or indicators of an underlying
latent mental attribute to be aggregated, or scored, to represent a
psychologically meaningful quantity (Hedge, Powell, & Sumner,
2018). For example, if the AUT is administered to participants, but
only one or two prompts (e.g., “Book,” “Hammer”) are included,
those few prompts cannot be aggregated in a way that provided
psychometric evidence of the reliability and internal validity of the
scores. In the creativity literature, observed-variable checks of the
composite reliability (e.g., Cronbach’s alpha) are not necessarily
always done before using scores for analysis, and still rarer are
studies of the underlying dimensionality and measurement prop-
erties of a set of DT items. Given that such formalized descriptions
of the way DT prompts are aggregated into psychometric scores
are rare in the creativity literature, it is difficult to build convincing
arguments for the reliability and validity of any DT measure or
scoring system, including the LSA-based Originality scores. For
example, after generating LSA semantic distances for every re-
sponse in the dataset, those responses are often averaged, or
perhaps summed, within each prompt for every participant (see
Forthmann, Szardenings, & Holling, 2020 for a close investigation
into the effects of these methodological choices). Then, if multiple
DT prompts (or specifically AUT prompts) were administered,
participant scores across those prompts need to be aggregated in a
reliable way (e.g., CFA) so that a score that represents participant
level Originality can be produced. However, the psychometric
properties of such a scoring model (if one is used) are rarely
reported in the literature, limiting what is known about LSA-based
Originality scores. Dumas and Dunbar (2014) were an exception to
this, and they found relatively strong evidence for the psychomet-
ric reliability of LSA-based Originality scores, but they did not
examine the relation between those scores and human-raters or
Elaboration scores, among other limitations. It should be noted that
the lack of strong psychometric evidence is a problem across much
of the creativity research literature, not just LSA-based Originality
scores, but the problem may be particularly poignant here, where
the automated nature of these scores make the large-scale mea-
surement of DT possible and provide opportunity for higher-stakes
applications of creativity assessment, where low reliability can
pose serious scientific and ethical problems.
Moving Beyond LSA in Creativity Research
Based simultaneously on the psychometric and psychological
evidence in support of text-mining-based Originality scores, as
well as more pragmatic and practical considerations such as the
speed, cost, and objectivity of these methods, the continued use of
text-mining models to score DT tasks seems justified and desir-
able. However, because of specific methodological concerns about
LSA as a method generally, the TASA corpus specifically, as well
as a general lack of formal psychometric investigation into these
scoring systems, thinking critically about other possible text-
mining systems for our work, beyond LSA TASA, appears impor-
tant. Indeed, it may be that the free availability of the TASA-
trained LSA tool—as well the fact that it was the text-mining
system originally chosen by Forster and Dunbar (2009)—drives
the creativity research literature’s choice of this text-mining sys-
tem over others that may provide us with better information about
Originality.
Today, there are a number of other freely available text-mining
systems in the information science and computer science commu-
nities that creativity researchers may recruit for their work. These
available text-mining systems differ on the type of model they
utilize (i.e., they do not use LSA), the text corpus they used to train
the model, and the preprocessing and parameterization performed
in training. For this reason, even across freely available text-
mining systems, there is a high potential for very different psy-
chometric and psychological patterns to emerge in Originality
scores. For example, beyond the LSA models trained on the TASA
and EN 100k corpora that have been used previously in creativity
research, other text-mining systems are also freely available to
researchers and could potentially be better for the quantification of
Originality then LSA. One such system comprises the Google
News model associated with the Word2Vec algorithm (Mikolov,
Chen, Corrado, & Dean, 2013)—which is trained on 100 billion
words scraped from the website Google News using a more
modern neural network-based training approach. In addition, the
Global Vectors for Word Representation (GloVe; Pennington,
Socher, & Manning, 2014) algorithm provides a series of free,
trained systems, including one based on 840 billion words scraped
from across the Internet. The GloVe system, by virtue of its
probabilistic (and therefore possibly more stable) statistical under-
pinnings and massive training corpus, may be more capable of
producing reliable and valid Originality scores than previously
used text-mining systems, although such a research question has
never been systematically addressed. These models are described
in more detail in the methodology section of this article, and are
generally distributed as vector spaces providing a mapping of
words to latent dimensions, which can be used programmatically
to measure semantic distance.
As previously reviewed, a number of freely available text-
mining models that have been trained on existing corpora of text
and that are designed to represent the semantic structure of lan-
guage through the estimation of word vectors within semantic
space exist in the literature (i.e., TASA, EN100k, word2vec,
GloVe). As will be delineated further in the Method section of this
article, each of these text-mining models essentially utilizes a
dimensionality reduction technique to quantify the relations among
words or phrases by examining the angles among vectors (Lan-
dauer et al., 1997; see Figure 1). However, another technique
exists within the psychological literature that offers an alternative
to all of these dimensionality-reduction-based methods: the net-
work modeling perspective (De Deyne, Verheyen, & Storms,
2016; Kenett, Levi, Anaki, & Faust, 2017). In this body of work,
the semantic distance among words is not quantified by examining
the angles among word vectors, but instead the length of a network
path between two words (i.e., the number of intervening words in
the network) is used as a measure of semantic distance. Recent
work in cognitive psychology (Kumar, Balota, & Steyvers, 2019)
has compared the network science approach to understanding
semantic distance and dimension-reduction models LSA and
word2vec, with some results showing advantageous properties of
the network approach. In the current study, the possibility for
incorporating network models into a computational psychometric
approach to divergent thinking (see Kenett, 2019 for an overview)
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
5
MEASURING ORIGINALITY

is not investigated, although it is discussed later in this article as a
future direction.
The Current Study
Given the current state of the creativity literature surrounding
the quantification of Originality via text-mining models, coupled
with the relatively recent availability of text-mining systems sub-
stantially more advanced than LSA and the TASA corpus, we have
undertaken a systematic study of four major freely available text-
mining systems (explained in more detail in the Method section)
and the reliability and validity of the Originality scores they
produce. These systems all rely on a different mix of methods,
technical implementations, and training corpora, and we seek to
understand which systems are more appropriate for scoring AUT
Originality. Specifically, we aim to assess the internal consistency
and factor reliability of AUT Originality scores produced by these
text-mining systems at both the scale and latent-variable levels and
compare that reliability to that of human-raters who judged the
Originality of each AUT response. In addition, the predictive
validity of the five Originality scoring systems (human raters and
four different text-mining systems) is examined in terms of their
correlation to a number of theoretically relevant DT dimensions
(i.e., ideational Fluency and Elaboration), creative personality
characteristics (i.e., Openness and Intellect) and self-reported real-
world creative activities. The overarching goal of this investigation
is to provide creativity researchers with psychometrically sup-
ported recommendations as to how to score DT responses for
Originality,
and
what
general
predictive
patterns
to
other
creativity-related constructs (e.g., Elaboration) may be expected
depending on what scoring system researchers choose to use.
Method
Participants
This study, which was part of a larger and ongoing investigation
into the psychometrics of creativity, included 92 (53 female;
57.6%) participants. Participants were recruited for this study via
Amazon Mechanical Turk, a crowdsourcing platform widely used
in psychology research, including creativity research (e.g., McKay,
Karwowski, & Kaufman, 2017). Because of the high language
demands of divergent thinking tasks, participants were required to
report themselves as fluent English speakers to participate, al-
though two participants (2.1%) reported English as their second
(but fluent) language. Participants were compensated $3.00 each
for their participation. Participants were required to be over the age
of 18 to participate, but the minimum actual participant age was
21, with a maximum age of 68. The mean age of participants was
37 (SD  10.58). The majority of participants (n  68; 73.91%)
reported their race/ethnicity as European American, whereas
smaller proportions of the sample reported their ethnicity as Afri-
can American (n  6; 6.5%), Asian (n  9; 9.8%), Latinx (n  5;
5.43) or multiple ethnicities (n  4; 4.2%).
Measures and Tasks
Alternate uses task.
The AUT is a psychometric measure in
which participants are asked to generate as many creative uses for
an object as possible within a certain amount of time (i.e., two
minutes per object in this case). The AUT has been used for
assessing divergent thinking and creative ability for decades (Guil-
ford, 1967; Hudson, 1968; Torrance, 1972) and remains one of the
most-often utilized tasks within the creativity research literature
(e.g., Dumas & Strickland, 2018; Puryear, Kettler, & Rinn, 2017).
The following 10 object names were presented to participants in a
randomized order: book, fork, table, hammer, pants, bottle, brick,
tire, shovel, and shoe. In this investigation, 10 AUT prompts
(rather than a single AUT prompt as is often the case in creativity
research) were used to reduce the stimuli dependence of the
Originality scores (Barbot, 2018). This issue of stimuli depen-
dence, and the concomitant need for multiple DT indicators, may
be even more critically important when scoring the AUT with
text-mining systems, because different AUT prompts (e.g., Book)
may be represented in any given corpus differently, and therefore
multiple stimuli are needed to produce the most reliable and valid
scores. In this investigation, the 10 object names that were pre-
sented to participants were chosen to be in line with past work
within the DT literature that has incorporated a text-mining ap-
proach (e.g., Dumas & Dunbar, 2014), as well as common practice
within the DT assessment field, where objects are typically chosen
that are expected to be familiar to participants, and that are
reasonably different from one another to provide a reasonably
broad sampling of object types, therefore reducing dependence on
any one stimuli (Acar & Runco, 2019). Scoring procedures and
resulting reliability and validity evidence for AUT scores are the
main focus of this investigation, so that specific information is
presented later in the Results section of this article.
Big Five Aspects Scale.
The Big Five Aspects Scale (BFAS;
DeYoung, Quilty, & Peterson, 2007) is a widely utilized self-
report personality measure in which participants indicate levels of
five principal aspects of personality, each of which is divided
further into two facets. The “big five” dimensions of personality—
Neuroticism, Agreeableness, Conscientiousness, Extraversion, and
Openness—are all available on this measure, but the Openness
scale is of particular interest to the present investigation, because
this dimension of personality is the most perennially associated
with divergent thinking, both in theory (Hornberg & Reiter-
Palmon, 2017) and in empirical findings (Furnham, Crump, &
Swami, 2009). The Openness dimension is further divided into two
facets—Openness and Intellect—and both of these facets have
been shown to be significantly and positively related to divergent
thinking and creative outcomes, and are considered the core of the
creative personality (Oleynick et al., 2017), making them both
useful validity criteria in this study. In particular, we conceptualize
the Openness and Intellect facets as providing important validity
information in the following way: Intercorrelations among the
text-mining-based Originality scores and the Openness and Intel-
lect facets should, if the validity of the text-mining methods is
upheld, be similar to the intercorrelations of the Openness and
Intellect facets and the human-rater-based Originality scores. If
Originality scores from one or multiple text-mining models were
to display correlations with Openness and Intellect that were
substantially different from human-judged Originality, the validity
of that text-mining model would be called into question.
Although the most common method used to score self-report
measures like the BFAS in psychology research is through sum-
ming the items, the summation of scores makes a number of strict
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
6
DUMAS, ORGANISCIAK, AND DOHERTY

measurement assumptions that are unlikely to hold (McNeish &
Wolf, 2020). Therefore the Openness and Intellect scores for this
study were generated through confirmatory factor analysis (CFA)
by fitting a two-factor correlated model to both scales at once, as
DeYoung and colleagues (2007) intended and validated. Specific
psychometric information for each scale appears below.
Openness.
The 10-item Openness facet of the BFAS has been
particularly associated with creative outcomes, because it features
such self-report items as I need a creative outlet and I believe in
the importance of art. Participants indicated the degree to which
each statement was true of them by dragging a 100-point slider
with poles of 100  strongly agree and 0  strongly disagree.
After reverse-coding all negatively worded items, in this study, the
10 items on the Openness facet of the BFAS achieved a scale
internal consistency of   .839, with latent factor internal con-
sistency indices based on factor loadings and uniquenesses being
H  .896 and   .861. Openness scores were generated from the
CFA model via empirical Bayes and saved in the dataset.
Intellect.
The intellect facet of the BFAS also contained 10
self-report items, including I like to solve complex problems and I
am quick to understand things. Participants responded to these
items in the same manner (i.e., with a slider) as they did the items
on the Openness facet. After reverse-coding all negatively worded
items the 10 items on the Intellect facet of the BFAS achieved a
scale reliability of   .840, with latent factor internal consistency
indices based on factor loadings and uniquenesses being H  .876
and   .858. Intellect scores were generated from the CFA model
via empirical Bayes and saved in the dataset.
Inventory of Creative Activities and Achievements (ICAA).
The ICAA is a relatively recently developed (i.e., Diedrich et al.,
2018) self-report measure for real-life creative activities and ac-
complishments across eight domains: literature, music, arts and
crafts, cooking, sports, visual arts, performing arts, and science and
engineering. Given the general nature of this sample, and time-
constraints on the data collection, we administered the creative
activity scale (rather than achievement) for six of those original
eight domains: music, literature, arts and crafts, cooking, visual
arts, and performing arts. Each of these scales consisted of six
Likert-style items that ask participants how many times they have
done particular creative activities in the past 10 years with five
response categories: never, 1–2 times, 3–5 times, 6–10 times, and
more than 10 times.
For example, in the music domain, participants are asked how
many times they have written a piece of music, or created a mix
tape, among other items. In the arts and crafts domain, participants
are asked how many times they created an original decoration. In
cooking, how many times they made up a new recipe. The visual
and performing arts scale asks how many times participants
painted a picture and performed in a play, respectively. In this
sample, each of the scales of the ICAA achieved satisfactory scale
reliability as well as satisfactory latent factor reliability based on
scale-specific single-factor CFA models, with literary activities
having the lowest reliability (  .800; H  .837;  . 814),
music activity having the highest (  .909; H  .968;   .915),
and the other scales (visual arts;   .826; H  .908;   .836;
cooking;   .874; H . 903;   .876; performing;   .876;
H . 899;  . 882; crafts;   .90; H . 921;  . 905) being
in the middle. Taken together, all 24 items on these six scales
displayed a composite scale reliability of   .926. and, as a single
latent factor, latent factor reliabilities based on loadings and
uniquenesses of H  .952 and   .934. For future analysis,
empirical Bayes-based latent factor scores were computed for each
of the six administered ICAA scales, as well as a total Creative
Activity that incorporated all six scales.
In this study, the ICAA is included as a validity-criterion mea-
sure with which to correlate the Originality scores produced by
both human raters and the various text-mining models. In general,
we conceptualize this validity procedure as requiring the text-
mining-based Originality scores to approximate, in their correla-
tions to the ICAA, the nature of the human-rated Originality
scores. This validity-criteria procedure is based on the general
problem in creativity research that an ongoing need to utilize
human raters in our work creates a bottleneck to scaling creativity
research to very large data sets. Here, we seek to test the capability
of the text-mining models to create Originality scores for the AUT
that are similar to those produced by humans, but much more
rapidly and at a much lower cost. Hence, the ICAA serves as an
external validity criterion to ascertain whether the text-mining
models are successful at accomplishing this goal.
Administration Procedures
All participation for this study was conducted online via Me-
chanical Turk, and the study website itself (which participants
were provided a link to) was hosted by Qualtrics. Informed consent
was obtained before participants could move forward with the
measures (these procedures were approved by the institutional
review board at the Institution where the study took place). Study
instructions asked participants to complete the measures with
minimal distractions and recommended that they turn off elec-
tronic devices as well as close other websites or programs open on
their computer. Because the AUT requires a significant amount of
typing, participation required a traditional keyboard and participa-
tion via smartphone or tablet was not allowed. Participants were
given two minutes to provide uses for each object before they were
automatically advanced to the next object, and they could not
advance before those two minutes were up. After responding to all
10 objects (i.e., after 20 min), participants were informed that
the task was complete, and moved to the self-report portion of the
study. In this phase, participants first provided responses to the
ICAA and then moved to the BFAS. Finally, participants re-
sponded to the demographic question and logged out of the study
website.
AUT Scoring Procedures
The main focus of this investigation was to examine the reli-
ability and criterion validity of multiple Originality measurement
methodologies for the AUT. As such, the AUT was scored a
number of different ways in this study, each of which is detailed
below.
Fluency.
As a criterion by which to examine the validity of
Originality scoring methods, the AUT was scored for Fluency.
First, the number of uses generated by each participant for each
object was tallied, and then summed across all 10 items on the
AUT, producing a “total-uses” variable for analysis. Counts such
as these are the principal way in which fluency has been opera-
tionalized in the extant literature (Plucker & Makel, 2010). In this
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
7
MEASURING ORIGINALITY

investigation, fluency counts across the 10 items on the AUT
exhibited a high level of scale internal consistency (  .946).
However, to avoid making potentially untenable measurement
assumptions, Fluency scores were generated via empirical Bayes
from a single factor CFA model fit the 10 Fluency indicators. The
scale exhibited latent factor reliability indices of H  .962 and
  .957.
Elaboration.
Also following well-established scoring proce-
dures in the divergent thinking literature (e.g., Forster & Dunbar,
2009; Torrance, 1988), participant Elaboration scores were calcu-
lated by averaging the number of words utilized per response
within each of the AUT prompts. In this scoring procedure, aver-
aging within the AUT prompt is meant to reduce the implicit
association between Elaboration and Fluency (i.e., those partici-
pants who generated more responses will have used more words in
total, but perhaps not on average). However, a statistical relation
between these two dimensions of divergent thinking may still exist
regardless of this scoring choice because of a possible psycholog-
ical association between ideational fluency and the ability to
elaborate on those ideas (Hudson, 1968). In addition, the strength
of the relation between Elaboration and Originality has been the
focus of previous investigations of text-mining scoring systems for
Originality (Forthmann et al., 2019), and therefore it is of high
importance here. These 10 AUT prompt elaboration scores dis-
played a high level of scale reliability (  .958), and at the latent
factor level (via a single factor CFA), displayed strong latent
internal consistency indices H  .965 and   .961. Elaboration
scores for each participant were generated via empirical Bayes
from the single factor CFA model.
Originality.
Originality in this investigation was scored using
two main categories of methods: human raters and text-mining
models. In addition, the reliability and validity of scores produced
by a number of different types of text-mining models are com-
pared. It should be noted here that, given critically important
concerns about the way that any text-mining system deals with
common function words (Forthmann et al., 2019), all of the anal-
ysis in this study utilizes inverse-document-frequency (IDF) term-
weighting (Robertson & Jones, 1976) corrections to deal with
extremely common words (e.g., “is”). Despite differences in how
they were developed, each scoring system provides a model of
language in a linear space, aiming for comparable distances be-
tween words in English. To score from each system’s model, a
weighted sum of word vectors is taken to represent each phrase for
a response, and the cosine distance is taken between the response
and AUT prompt (see Figure 1).
Human raters.
First, every generated response from the 92
study participants across the 10 items on the AUT was coded for
Originality by four human coders. In all, 5,491 responses were
generated to the AUT in this study, with an average of 55.81
(SD  31.72) per participant. The first Originality coder was the
third author of this article, and the other three were paid research
assistants. Each coder was instructed to score each generated
response from 0–4, with zero being “totally ordinary” and four
being “maximally novel.” Coders were specifically trained to
conceptualize most responses as being likely to fall toward the
middle of that 5-point Originality scale: a belief that reflects our
assumption that the Originality of generated responses is based on
a continuous distribution, manifesting such that most responses are
in the middle of the Originality scale (i.e., 2) with a smaller
number of responses being completely unoriginal (i.e., 0) or very
high on the Originality scale (i.e., 4). Such a continuous distribu-
tion of Originality—rather than discrete Originality categories—
has been empirically documented in the literature (e.g., Dumas,
2018).
The four human coders coded the 5,491 responses with a “fair”
level of interrater agreement (Fleiss’   0.2198; Fleiss & Cohen,
1973). Typically, within the psychological research literature, any
lack of exact agreement among coders would be resolved through
discussion until all coders were able to converge on an agreed-
upon categorical rating for every response (Gwet, 2014). Such a
method operates under the measurement assumption that there is a
true Originality category for each generated response (i.e., the
latent Originality attribute is ordinal), and therefore coders must
work to sort the generated responses into their true categories.
However, an alternative method would assume that the 0–4 Orig-
inality categories the coders used were underlain by a continuous
latent distribution, and therefore the originally coded Originality
categories are meant to indicate locations on that continuous latent
distribution.
Common in crowdsourcing methodology (e.g., Organisciak, Te-
evan, Dumais, Miller, & Kalai, 2014; Snow, O’Connor, Jurafsky,
& Ng, 2008), where varying judgments of quality from raters are
regularly aggregated, this continuity assumption suggests that ex-
act categorical agreement among raters is not crucial, because,
over multiple raters, a consensus about where on the underlying
Originality distribution each generated response may be located
can arise through averaging the ordinal category codes across
raters. The intuition here is that disagreement among raters on
ordinal codes is actually instructive and valuable to researchers.
For example, if three raters coded a particular AUT response as a
‘3’ on the Originality scale, but one rater coded it as a ‘4,’ that last
rating is still considered a nudge toward the more novel end of the
scale for that response, and an averaged Originality score for that
response of 3.25 would be considered closer to “true” than simply
the modal rating of 3. For this reason, we created an aggregated
human-coded Originality rating for each AUT response by aver-
aging each of the 4 coders’ ratings for every one of the 5,491
responses. Then, to aggregate those response-level Originality
ratings to the participant-level, we further averaged each of those
response ratings within each AUT prompt (e.g., Book) for every
participant. This procedure resulted in 10 prompt-level human-
rated Originality scores for each of the 92 participants in the
dataset. Because it is the main focus of this investigation, further
analysis with these human-rated Originality scores (e.g., modeling
an underlying latent Originality attribute across all AUT prompts)
are included in the Results section of this article. Further, the issue
of the interrater reliability of these human-rated Originality scores
is returned to with a critical lens in the Discussion section.
Text-mining systems.
Here, we systematically compare the
capability of four different publicly available text-mining systems
to create reliable and valid participant Originality scores on the
AUT. Each of these four text-mining systems differ in a variety of
ways, including the corpora of text that they are trained on, the
parameterization and specification of the statistical models they
use, and the way they correct for difficult-to-model aspects of
real-world language use such as words with multiple meanings and
synonyms. Generally, larger corpora will more accurately repre-
sent the relations between words in the language, though the
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
8
DUMAS, ORGANISCIAK, AND DOHERTY

domain of the documents will lead to differences in how the
language is interpreted and may affect the transferability of that
particular model. For example, is bank more associated with river
or money? A naive algorithm learning English from a collection of
documents will decide that answer differently based on what those
documents were written about. The sizes and domains of the
corpora on which each system was trained are noted in Table 1.
Different systems may also correct for perceived importance of
words, deemphasizing common function words (e.g., and, the) or
removing them altogether. Finally, all the system models are
trained using different training methods. These methods differ on
choices such as what frame of document suggests a relationship
between words and how the training algorithm implements that
theory. The choice of how many latent dimensions are learned also
affects the system: Too few dimensions will lack depth and dis-
criminatory value between words, whereas too many will overfit to
the documents. In the current investigation, we do not attempt to
absolutely control for every possible methodological option in the
training of a text-mining model. Instead, we focus on already
created text-mining systems that creativity researcher are currently
able to access free of charge, to provide a meaningful demonstra-
tion of the strengths and weaknesses of each extant system spe-
cifically in the context of creativity and divergent thinking re-
search.
Each of the four text-mining systems that are tested in this study
are succinctly explained below. A bulleted explanation of each of
these text-mining systems also appears in Table 1. Analysis based
on these text-mining systems was accomplished by remotely ac-
cessing their freely available systems via the Python programming
language. All reproducible computational code used in this inves-
tigation are freely available online via our laboratory ongoing
Github account (https://github.com/massivetexts), and a static de-
pository of the code used for this study is also available on the
Open Science Foundation (https://github.com/massivetexts). In ad-
dition, computational code is available as online supplemental
materials published with this article.
Touchstone Applied Science Associates (TASA) LSA.
This sys-
tem, which is by far the most commonly applied in the extant
literature on divergent thinking (e.g., Dumas & Dunbar, 2014;
Forster & Dunbar, 2009; Forthmann et al., 2019), is trained on a
corpus of 37,651 educational texts and was originally intended to
mimic to expected reading experience of the typical entering
undergraduate student. This system was used by Landauer and
Dumais (1997) in their initial demonstration of the capability of
LSA to approximate the human semantic relations, but since then
has been outstripped by other systems in terms of corpus size and
model sophistication (Crossley et al., 2017). For example, recent
work in the information sciences has confirmed Landauer and
Dumais’ (1997) argument in showing that LSA spaces trained on
TASA do tend to match human semantic judgments but has also
found better performance with larger corpora (S¸tefa˘nescu, Ban-
jade, & Rus, 2014). In this study, we use the publicly available
TASA model trained by Günther et al. (2015).
English 100k LSA.
Also originally trained by Günther, Dud-
schig, and Kaup, the English (EN) 100k LSA text-mining system
was previously applied to divergent thinking task data by Forth-
mann and colleagues (2019). This system is trained on a concat-
enation of multiple general purpose corpora of texts: a Wikipedia
image, the general text British National Corpus, and a web crawl
corpus that together included more than 5 million documents.
After an initial modeling of the language in these 5 million
documents, the 100,000 most frequently occurring unique words
were retained to build the system (hence the 100k in the name). So,
although the LSA training method in this system is the same as that
in the TASA system, the size and generality of the corpora used in
this the EN 100k system may be more advantageous for the
quantification of originality on DT tasks because the more general
corpora on which this model is trained may better represent the
true semantic space from which DT task participants draw their
responses.
Global Vectors for Word Representation 840B.
Publicly avail-
able through the Stanford natural-language-processing laboratory
(Pennington et al., 2014), but never before applied to the analysis
of divergent thinking task data, the Global Vectors for Word
Representation (GloVe) 840B text-mining system was trained on a
corpus of 840 billion words that were scraped from a variety of
online sources including Wikipedia and Twitter. Although GloVe
is similar to the LSA-based text-mining systems in that its goal is
to quantify the semantic relation between two words or phrases
within a geometric space, GloVe accomplishes this goal through a
probabilistic modeling framework. In addition, GloVe calculates
correlations among terms in a more targeted way than does LSA:
by examining a small window of word co-occurrence around each
term where it is used, rather than examining co-occurrence in
full-text documents. This shift in the mathematical and statistical
underpinnings of the text-mining systems may hold potentially
positive impact on the measurement of AUT originality in that it
may potentially produce more stable and reliable estimates of
response Originality (and this hypothesis will be tested in the
current study).
Table 1
Short Description of Text-Mining Systems Included in This Investigation
System name
Training corpora
Training scale
Reference
TASA LSA
Multi-subject educational texts
37.7 thousand documents (92.4 thousand
unique words)
Landauer & Dumais, 1997
EN 100k LSA
Wikipedia, ukWaC (web crawl), and British
National Corpus (general)
5.4 million documents (2 billion words, 100
thousand unique words)
Günther, Dudschig, & Kaup, 2015
GloVe 840B
Common Crawl (web documents from sites
including Wikipedia and Twitter)
840 billion words (2.2 million unique words)
Pennington, Socher, & Manning, 2014
Word2Vec
Google News (articles)
100 billion words (3 million unique words)
Mikolov, Sutskever, et al., 2013
Note.
TASA  Touchstone Applied Science Associates; LSA  Latent Semantic Analysis; EN  English; GloVe  Global Vectors for Word
Representation; Word2Vec  word-to-vector.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
9
MEASURING ORIGINALITY

Word2Vec.
Named for the “word-to-vector” methodology it
employs, Word2Vec focuses specifically on word-level corpora
scraped from massive online sources of text (Mikolov, Chen, et al.,
2013). This text-mining system was created at, and is publicly
available through, the tech company Google, and was trained on a
corpus of 100 billion words scraped from the news-aggregator
Google News. Word2Vec modeling methodology focuses on the
context of individual words, and through a neural network predic-
tive modeling approach, works to predict a target word from a
sample of closely co-occurring words (i.e., context words). In
Word2Vec parlance, this method is termed “skip-gram,” because
the model skips individual target-words when training and then
predicts the skipped word based on the context-words that co-
occur with it. Previous research has found that this Wor2Vec
method preserves the true semantic relations among words more
effectively than other training models such as LSA (Mikolov,
Sutskever, Chen, Corrado, & Dean, 2013). Further, Word2Vec
models perform well at identifying high-order analogical relations
among words (Bianchi & Palmonari, 2017). Because of the cog-
nitive similarity between analogical and divergent thinking (e.g.,
Green et al., 2012), this finding may suggest that Word2Vec
methods are particularly suited for the measurement of originality
in DT tasks.
Results
The analysis for this psychometric investigation of text-mining-
model-based Originality scoring systems unfolded in the following
stages: (a) a careful investigation of the reliability of participant
Originality scores generated by both human raters and text-mining
systems, with an eye toward both composite and latent factor
reliability; (b) an analysis of the correlations among human rated
and text-mining system generated Originality scores; and (c) a
criterion validity analysis of Originality scores in which the cor-
relations from both human rated and text-mining system generated
Originality to Fluency, Elaboration, Openness, Intellect, and Cre-
ative Activities were examined. Each of these three analytic stages
are explained, and results are presented, below.
Reliability
Here, reliability of each of the five included Originality scoring
methods (i.e., human raters, TASA LSA, EN 100k LSA, GloVe
840B, and Word2Vec) is examined using both observed variable
(i.e., Classical Test Theory [CTT]) and latent variable (i.e., Con-
firmatory Factor Analysis [CFA]) methods.
Composite internal consistency.
Human-coded Originality
ratings on the 10 AUT prompts displayed a high level of composite
or scale reliability (see Table 2 for reliability coefficients). In
contrast, the composite reliability of the text-mining system gen-
erated Originality was substantially lower, although the TASA
LSA and GloVe 840B each reached levels of composite reliability
that may be generally considered acceptable in the psychological
research literature (i.e., .80 or above). These coefficients are es-
pecially important given the propensity of creativity researchers
for using summed-scores, rather than optimally weighted latent
variable scores, in their research. However, a more in-depth anal-
ysis of the measurement properties of the Originality scoring
systems is necessary to understand the way these scores relate to a
latent Originality construct.
Confirmatory
factor
analysis.
A
unidimensional
CFA
model, in which all 10 AUT prompts loaded on a latent Originality
factor, was fit to item-scores generated by each of the five Orig-
inality scoring systems (please see Figure 2 for a conceptual path
diagram of this CFA model). Theoretically, such a model corre-
sponds to a measurement assumption that all the administered
AUT prompts (when scored for Originality) indicate a single
underlying originality construct and therefore represents common
measurement practice in the creativity literature (e.g., Storme et
al., 2017). These CFA models were fit using maximum likelihood
estimation in Mplus Version 8.0 (Muthén & Muthén, 2019). Based
on the model root mean square error of approximation (RMSEA;
See Table 3 for exact values), none of these unidimensional
methods achieved a level of model-data fit that would be consid-
ered ideal in the methodological literature (i.e., below .06; Hu &
Bentler, 1999; McNeish, An, & Hancock, 2018). However, the
models for both the human raters and the GloVe text-mining
system achieved a level of fit that would meet current standards in
the creativity literature, where measurement model-data fit is often
slightly weaker than in more traditional measurement areas such as
reading or math (e.g., Yoon, 2017).
In addition, although the scoring systems differed in the strength
of their CFA loadings (see Table 3 for loading details), the indi-
vidual AUT items also displayed general trends in the strength of
their loadings across scoring systems. For example, the prompt
Rope displayed weaker standardized loadings than other prompts
across multiple of the scoring systems, whereas the prompt Bottle
displayed stronger loadings across multiple scoring systems. This
pattern may likely be attributable to differential participant famil-
iarity with certain objects, or perhaps the actual functional capa-
bilities of each object to facilitate original alternate uses. One
anomaly in these general patterns were the extremely weak stan-
dardized loadings for Book and Table in the EN 100k LSA system,
and the model-data fit or this scoring system was also poor
compared with the other models, so that may indicate that this
corpus does not well-represent the true semantic relations among
Book, Table, and their associated uses.
Many-faceted Rasch analysis for human raters.
The con-
firmatory factor modeling perspective above intentionally aggre-
gated human rated originality across all the responses to a given
AUT prompt (e.g., Book), for all four of the human raters, through
averaging. This method was designed to treat the raters as equally
weighted voters in terms of the originality of a given AUT re-
sponse, and therefore also for participant originality scores. How-
Table 2
Composite and Factor Reliability Indices for Originality Scores
From Each Scoring System
Scoring system


H
Human raters
.943
.948
.952
TASA LSA
.813
.825
.867
EN 100k LSA
.730
.758
.825
GloVe 840B
.800
.807
.875
Word2Vec
.741
.743
.807
Note.
TASA  Touchstone Applied Science Associates; LSA  Latent
Semantic Analysis; EN  English; GloVe  Global Vectors for Word
Representation; Word2Vec  word-to-vector.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
10
DUMAS, ORGANISCIAK, AND DOHERTY

ever, a reasonable alternative perspective, recently demonstrated
by Primi and colleagues (Primi, Silvia, Jauk, & Benedek, 2019)
would be to employ a Many-Facet Rasch Model (MFRM) to
incorporate differences in the leniency or severity of individual
human judges into the calculation of originality scores. The
MFRM model has been previously demonstrated to be useful in
creativity research, in particular in modeling measurement error
associated with multiple human raters (Barbot, Tan, Randi, Santa-
Donato, & Grigorenko, 2012), and this previous usage suggests
relevance of this modeling tool to the current study. So, as a further
point of comparison to the CFA approach presented above, an
MFRM with three facets (i.e., judges, items, and participants) was
fit to these data using the specialized computer program Facets
(Linacre & Wright, 1988), and used to calculate originality scores
for each participant in the dataset. It should be noted that the use
of the term “facets” is different in this context than in the person-
ality measurement context. In the personality measures used in this
study, the facets refer to the finely grained subscales within the Big
5 factors. In MFRM, a facet refers to a source of measurement
error, in this case error may arise from inconsistencies among the
human raters, AUT items, or the distribution of participant original
thinking ability.
To accommodate the MFRM here, modal ratings for each judge
were utilized for each AUT prompt (as opposed to means within
each prompt as was used for the CFA above). MFRM parameter
estimates are presented here in Table 4, which contain the diffi-
culty/severity for the AUT items and raters, as well as the
parameter-theta (latent score) correlations. As can be seen in this
table, some AUT items were more difficult for participants to think
of original uses for (e.g., Shoe; difficulty  .60), whereas other
items were easier (e.g., Brick; difficulty  .53). Similarly, some
human raters were more lenient in judging originality of responses
(e.g., Rater 3; severity  1.4), whereas others were more severe
(e.g., Rater 2  .66). As a general measure of internal consis-
tency, the Rasch average reliability among the 10 AUT items in the
MFRM was .90, and therefore MFRM based scores were gener-
ated and saved in our dataset for future analysis. As an alternative
to MFRM not applied here, interested readers should also see the
application of item-response models to data from multiple human-
raters recently posited by Myszkowski and Storme (2019).
Latent factor internal consistency.
Using the standardized
loadings and residual variances that are generated when fitting the
CFA models, we then calculated two modern factor reliability
statistics for each of the scoring systems: Omega (McDonald,
1999) and H (Hancock, 2001). Although both of these indices
represent a more sophisticated estimate of the score reliability than
Cronbach’s alpha, they differ in their assumptions about the way
participant scores will be produced in future investigations using a
Table 3
Confirmatory Factor Model Parameters for Each Scoring System
Scoring system
Model RMSEA
Alternate uses task prompt standardized loadings
Book
Bottle
Brick
Fork
Pants
Rope
Shoe
Shovel
Table
Tire
Human raters
.063
.837
.867
.739
.798
.864
.642
.788
.807
.825
.838
TASA LSA
.082
.732
.807
.655
.429
.414
.433
.744
.327
.497
.562
EN 100k LSA
.124
.023
.647
.513
.433
.650
.629
.742
.499
.039
.564
GloVe 840B
.069
.672
.784
.805
.595
.590
.161
.752
.268
.309
.350
Word2Vec
.079
.605
.761
.488
.100
.469
.336
.686
.457
.421
.334
Note.
TASA  Touchstone Applied Science Associates; LSA  Latent Semantic Analysis; EN  English; GloVe  Global Vectors for Word
Representation; Word2Vec  word-to-vector; RMSEA  root mean square error of approximation. All standardized loadings significant at p  .05 except
Book and Table in the EN 100k LSA system.
Figure 2.
Conceptual path diagram of the latent measurement model used to determine the reliability of the
Alternate Uses Task (AUT) scoring methods.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
11
MEASURING ORIGINALITY

measure. If, in the future, Originality scores are created by sum-
ming or averaging across multiple AUT prompts, then Omega is
the best representation of the reliability of those scores, but H
assumes an optimally weighted measurement model in which the
Originality scores are estimated directly from a CFA or item-
response model (McNeish, 2018). For this reason, Omega is al-
ways slightly lower than H to account for the added measurement
error associated with summing or averaging scores across a mea-
sure rather than using a psychometric model.
As can be seen in Table 2, the human-raters achieved by far the
most reliable Originality scores. However, all of the text-mining
systems, at least in terms of coefficient H, achieved an acceptable
level of factor reliability (i.e., above .80) as well, although the most
reliable system (i.e., GloVe) was substantially more so than the
least (i.e., Word2Vec). This finding implies that, should research-
ers generate scores from a latent measurement model, all of the
scoring systems included here are capable of producing generally
acceptable scores (although GloVe scores would be the most
reliable, and have the best model-data-fit). In terms of Omega—
which converged on the same results as Cronbach’s alpha—only
the TASA LSA system and GloVe achieved acceptable reliability,
indicating that these are the only systems that produce stable
enough scores to warrant calculating a simple composite score
(e.g., a sum) as opposed to using a latent scoring model. Going
forward, Originality scores for all 92 participants from each of the
five scoring system were generated via empirical Bayes using the
SAVEDATA command in Mplus.
Relation Between Text-Mining Models and
Human Raters
Perhaps the most critical test of the efficacy of the text-mining
scoring systems is their ability to approximate the Originality
scores produced by human raters. Table 5 holds the correlations
among the five scoring systems included in this investigation. As
expected, the human rated Originality scores generated via CFA
and MFRM were correlated very strongly (.98). More critical are
the correlations among the text-mining systems and the human
raters. These correlations show that the GloVe text-mining system
is the most capable of producing Originality scores that resemble
those of humans. In contrast, the Word2Vec system’s Originality
scores were the most weakly correlated with human-rated scores.
Although, it should be noted, all four of the systems utilized here
produced latent Originality scores that were significantly and
positively correlated with scores from human raters. In addition,
the Originality scores from each of the four text-mining systems
correlated strongly (in the .9’s) with one another, indicating that—
although the systems differed in their observed and latent reliabil-
ity indices and CFA model-data fit—overlapping information
about participant Originality is provided by each system.
Criterion Validity
Here, the capacity of the five Originality scoring systems to
produce scores that predict other common indicators of creativity
(i.e., Fluency, Elaboration, Openness, Intellect, and Creative Ac-
tivities) are systematically examined. See Table 6 for correlations
discussed in this section.
Fluency.
The theoretical relation between Fluency and Orig-
inality is currently debated in the creativity research literature, with
some scholars arguing for a positive, zero, or negative correlation
among these dimensions of divergent thinking (see Dumas &
Dunbar, 2014 or Forthmann et al., 2019 for discussions of this
issue). Here, human-rated Originality scores (calculated via CFA)
correlated weakly and positively (and nonsignificantly) with Flu-
ency scores, implying that that—should we take the human-rated
scores as baseline truth—the actual relation between these dimen-
sions is close to zero, at least in this general-population sample.
However, all of the text-mining systems in this study displayed
Table 4
Many Facet Rasch Model Parameters for Human Rated
AUT Items
Model
parameter
Difficulty/Severity (SE)
Parameter-Theta
correlations
AUT Items
Book
.13 (.09)
.71
Bottle
.12 (.09)
.78
Brick
.53 (.10)
.70
Fork
.06 (.09)
.71
Pants
.15 (.09)
.81
Rope
.05 (.09)
.62
Shoe
.60 (.09)
.74
Shovel
.35 (.10)
.71
Table
.11 (.09)
.76
Tire
.09 (.09)
.76
Human raters
Rater 1
.63 (.06)
.70
Rater 2
.66 (.06)
.69
Rater 3
1.4 (.06)
.71
Rater 4
.11 (.06)
.70
Note.
AUT  Alternate Uses Task; SE  standard error.
Table 5
Correlation Matrix Among Originality Scores From All Scoring Systems Included in This Investigation
Scoring system
Human raters (CFA)
Human raters (MFRM)
TASA LSA
EN 100k LSA
GloVe 840B
Word2Vec
Human raters (CFA)
1.00
Human raters (MFRM)
.98
1.00
TASA LSA
.67
.56
1.00
EN 100k LSA
.66
.55
.93
1.00
GloVe 840B
.73
.63
.97
.96
1.00
Word2Vec
.58
.45
.96
.95
.94
1.00
Note.
TASA  Touchstone Applied Science Associates; LSA  Latent Semantic Analysis; EN  English; GloVe  Global Vectors for Word
Representation; Word2Vec  word-to-vector; CFA  confirmatory factor analysis; MFRM  Many-Facet Rasch Model. All correlations significant at p  .01.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
12
DUMAS, ORGANISCIAK, AND DOHERTY

significant and positive correlations (although only moderate in
strength) with Fluency. Given this finding, it appears that human-
rated Originality scores have the greatest degree of discriminant
validity from Fluency scores, whereas text-mining models under
investigation produced Originality scores that were much more
strongly associated with Fluency. In particular, the EN 100k sys-
tem displayed the lowest correlation to Fluency among the text-
mining systems, making it the most consistent with human-rated
Originality scores in that regard.
Elaboration.
In previous work with text-mining system Orig-
inality scores (Forthmann et al., 2019), the relation between Elab-
oration and Originality has been considered a source of bias in the
scores. In this investigation, following previous methodological
recommendations, the IDF correction for stop-words was utilized.
Here, we found that the human raters’ Originality scores (calcu-
lated via CFA) were significantly and positively associated with
Elaboration (which is a stronger relation than those human-rated
scores had with Fluency). In contrast, the correlation between
MFRM calculated Originality ratings and elaboration was not
significant (i.e., p  .072). Following the pattern set by the CFA
produced Originality ratings, all of the text-mining systems also
produced scores that were significantly and positively associated
with Elaboration, although all of the text-mining systems displayed
correlations to Elaboration that were stronger than that of the
human-raters: a findings that highlights previously observed dis-
criminant validity issues, even with the IDF correction. However,
given that the human rated scores were also positively associated
with Elaboration, it appears that the text-mining systems are not
much more confounded with Elaboration than are human raters,
although substantial variation among the text-mining systems was
observed. Specifically, the GloVe system was most capable of
preserving the low-moderate correlation with Elaboration, fol-
lowed by TASA. The Word2Vec system produced the strongest
correlation with Elaboration.
Openness and intellect.
In this study, none of the Originality
scoring systems (human-rated or text-mining) produced scores that
significantly correlated with Openness or Intellect. However, in
the case of both of these creative-personality indicators, the GloVe
system produced correlations that were closest to those of the
human raters, indicating that the GloVe originality scores approx-
imated human-rated Originality scores the best in regards to their
relation to creative personality variables.
Creative activities.
When predicting the composite of the
Creative Activities measure, none of the Originality scoring sys-
tems produced significant correlations, although again the GloVe
system was most in-step with the human-raters. At the more
fine-grained level of the individual scales of the Creative Activities
measure (see Table 7), the human-rated Originality scores (calcu-
lated either by CFA or by MFRM) did significantly but negatively
correlate with creative Cooking activities. These findings imply
that, at least in this general-population sample, AUT Originality
scores are not related to the self-reported domain-specific creative
Table 6
Correlations Among Originality Scoring Systems and External Criteria
Scoring system
Ideational fluency
Elaboration
Openness
Intellect
Creative activities composite
Human raters (CFA)
.176
.215
.088
.086
.093
Human raters (MFRM)
.103
.186
.039
.117
.100
TASA LSA
.346
.239
.119
.010
.077
EN 100k LSA
.264
.273
.046
.044
.083
GloVe 840B
.336
.233
.084
.018
.077
Word2Vec
.365
.339
.084
.014
.025
Note.
CFA  confirmatory factor analysis; MFRM  Many-Facet Rasch Model.
 p  .05.
 p  .01.
Table 7
Correlations Among Creativity Indicators and Creative Activities
Creativity indicator
Performance
Arts
Cooking
Crafts
Literary
Music
Originality scoring systems
Human raters (CFA)
.120
.081
.227
.075
.108
.029
Human raters (MFRM)
.119
.097
.243
.081
.093
.044
TASA LSA
.156
.022
.139
.022
.108
.025
EN 100k LSA
.155
.041
.103
.051
.085
.033
GloVe 840B
.134
.038
.152
.063
.107
.053
Word2Vec
.116
.016
.085
.042
.134
.033
Creative personality indicators
Openness
.134
.307
.266
.392
.309
.201
Intellect
.052
.069
.465
.339
.133
.051
Divergent thinking dimensions
Fluency
.150
.168
.147
.171
.309
.254
Elaboration
.080
.155
.029
.191
.220
.145
Note.
TASA  Touchstone Applied Science Associates; LSA  Latent Semantic Analysis; EN  English; GloVe  Global Vectors for Word
Representation; Word2Vec  word-to-vector; CFA  confirmatory factor analysis; MFRM  Many-Facet Rasch Model.
 p  .05.
 p  .01.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
13
MEASURING ORIGINALITY

activities of participants. Among creative personality indicators,
Openness significantly and positively predicted Arts, Cooking,
Crafts, and Literary activities, whereas Intellect predicted Cooking
and Craft activities. Ideational Fluency significantly predicted both
Literary and Musical activities, but Elaboration did not signifi-
cantly predict any of the included creative activities. Interestingly,
none of these creativity indicators were capable of significantly
predicting Performance activities, although the low-prevalence of
Performance within this general sample (as opposed to a profes-
sionally creative sample) likely limited the variance of this activity
scale and therefore precluded a significant correlation.
Discussion
As far as we are aware, this investigation has been the first
within the creativity research literature to compare the ability of
multiple text-mining models to produce reliable and valid Origi-
nality scores on the AUT, as compared with human-raters. As
such, this study has a number of principal findings and specific
recommendations to forward to the creativity research community.
Four of these principal findings are described in detail below.
Human Raters Can Produce the Most Reliable
Originality Scores
Within the creativity literature, the perennially low level of
exact agreement (and therefore low level of interrater reliability)
between human raters on the level of Originality of a given AUT
response has led many to bemoan the possibility of producing
highly reliable creativity research using human raters (see Storme,
Myszkowski, Çelik, & Lubart, 2014 for one approach to improving
this reliability). However, our results show that, if researchers are
willing to conceptualize human-rated Originality codes as ordinal
indicators of an underlying Originality continuum and therefore
average the Originality codes across raters, a very high level of
reliability is possible with four trained raters. Of course, this high
level of reliability refers to the consistency of the continuous
Originality scores that are created by aggregating all 10 of the
AUT prompts included in this study (either by summing or through
a CFA), and not to the individual AUT responses that were coded
separately by each human rater. If an analysis at the fine-grained
level of an individual response to a specific AUT prompt was
desired by a researcher, then interrater reliability may be a more
informative index. Although such fine-grained analysis at the
individual response level is somewhat common among creativity
researchers with a basic psychology focus (Benedek, 2018), those
researchers whose work is more situated within an applied psy-
chology area (e.g., educational psychology; Kerr & Stull, 2019)
are more commonly concerned with the capacity of creativity
measures to produce reliable and valid scores for participant-level
interpretation across multiple items, tasks, or prompts. Following
with that applied-psychology focus, should a researcher or practi-
tioner desire to use the AUT to produce participant Originality
scores for admission into a specialized educational program or
personnel selection in the workplace, it does appear from these
results that four trained raters can produce highly reliable Origi-
nality scores across 10 AUT prompts, at least with this general-
population adult sample. Of course, the high level of reliability
achieved here by the human-raters was likely influenced, at least in
part, by the specific training and feedback that we provided our
raters. In this case, raters were specifically trained to conceptualize
the Originality scale along which they rated AUT responses (that
ranged from 0 to 4) as a continuous dimension, on which most
responses would have a moderate amount of Originality (i.e., a 1,
2, or 3), and only a few responses would fall on the extremes of the
scale (i.e., 0 or 4). Using this particular prompting, the participant-
level latent Originality scores we calculated were able to achieve
a high level of reliability. In future work, it should not be assumed
a priori that the highest score reliabilities are possible with human
raters as opposed to text-mining systems. In cases where the
human raters are not effectively trained or are less motivated, the
scores from human raters could actually be less reliable than those
from text-mining systems.
GloVe 840B Is the Recommended Text-Mining System
for Originality
A major stated goal of this investigation was to identify the
publicly available text-mining system that is most capable of
producing reliable and valid Originality scores on the AUT. Across
the stages of the current investigation, the GloVe 840B system
emerged as the best system to choose for producing Originality
scores in creativity research. The GloVe system generated the most
reliable scores within a CFA framework as indicated by coefficient
H, and the second most reliable composite scores as indicated by
Alpha and Omega. The only system that produced more reliable
composite scores (i.e., TASA LSA) demonstrated substantially
worse model-data-fit of the unidimensional CFA model as indi-
cated by RMSEA. In addition, the TASA LSA scores correlated
substantially weaker with human-rated Originality than did the
GloVe scores, a strong indication that GloVe scores are more valid
than TASA scores. In addition, although both TASA and GloVe
scores did display the previously described potentially problematic
relation to Elaboration (Forthmann et al., 2019), so too did the
Originality scores coded by human-raters. The GloVe scores dis-
played the correlation with Elaboration that was most in line with
the human-raters (although the difference between GloVe and
TASA in this respect was not great). Further, among the text-
mining systems, the GloVe scores had the correlations to other
creative indicators (e.g., Openness and Intellect) that were most
similar to that of human raters, although it should be noted that
none of the Originality scoring systems used here (humans or
text-mining) was significantly correlated with these indicators of
creative activity, with the exception of a weak-moderate and
negative correlation between the human-rated Originality scores
and Cooking activities. Such a general lack of covariance in this
regard may be caused by the psychological differences in
creativity-related self-report variables and more objectively quan-
tified DT performance tasks such as the AUT. So, the near-zero
correlations from GloVe-based Originality scores to the Openness
and Intellect measures are here interpreted as a positive finding
related to the validity of GloVe-based Originality scoring: GloVe
was capable of producing Originality scores that generally mim-
icked the criteria correlations of the human-rated Originality
scores, but much more quickly and at a greatly reduced cost.
In past investigations of text-mining models to produce Origi-
nality scores, by far the most commonly utilized text-mining
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
14
DUMAS, ORGANISCIAK, AND DOHERTY

system has been the TASA system (e.g., Dumas, 2018), with a
small minority of other pieces utilizing the EN100k LSA system
(Forthmann et al., 2019). Given the results of this investigation,
researchers in this area should likely pivot their methodological
focus away from LSA based systems to GloVe, to create more
reliable and valid scores for research. GloVe’s improved reliability
and validity is likely attributable simultaneously to three factors:
the size of its training corpus, the domain-generality of its training
corpus, and its probabilistic modeling approach. In general, such a
massive corpus (840 billion words, with 2.2 million unique words)
may be more capable of approximating the actual semantic struc-
ture of language-use than a smaller corpus (e.g., TASA’s unique
words are only 4% of GloVe’s), leading to more psychologically
reliable and valid Originality scores. Further, the inclusion of text
from sources like Wikipedia, a general web crawl, and the British
National Corpus make GloVe much more general in scope than the
TASA corpus that is composed of only educational texts. Finally,
although both GloVe and LSA seek to represent the euclidean
distance (or cosine similarity) between word vectors, LSA esti-
mates these word vectors using a traditional parametric approach
and GloVe uses a more modern log-linear, or probabilistic method
that may produce more psychologically relevant results (Penning-
ton et al., 2014).
Of course, all of the text-mining models compared here (i.e.,
TASA LSA, EN100k, word2vec, GloVe) are members of a larger
family of dimension-reduction–based techniques for the quantifi-
cation of semantic relations among words and phrases through the
examination of the angles among estimated word vectors. As
previously reviewed, an alternative approach would be to quantify
the semantic distance among words using a network science ap-
proach (De Deyne et al., 2016), in which the semantic distance is
not operationalized based on vector angles, but instead based on
the path length between two or more words in the network. Some
evidence in cognitive psychology suggests that the network sci-
ence approach may have advantages above the dimension-
reduction approach when studying fine-grain cognitive processes
(e.g., priming; Kumar et al., 2019). However, it remains a future
direction to ascertain whether the network science approach could
be helpful in psychometric work that, as with the current study,
aims to quantify Originality at the participant level with reliable
and valid scores. Recent arguments in the creativity literature
(Kenett, 2019) suggest that the network science approach to quan-
tifying semantic distance may be fruitfully applied to computa-
tional psychometrics of creativity, and this approach may have
promise for creativity researchers.
Not All AUT Prompts Contribute
Equally to Reliability
One interesting, and perhaps problematic, aspect of creativity re-
search is that the field’s most commonly used measure (i.e., the AUT)
is not necessarily fully standardized across studies in its administra-
tion procedures or even the particular prompts that are included. In
this study, the particular measure-administration choices, as well as
the particular prompts included on the AUT, may mean that the
results could differ from other investigations where different measure-
ment procedures were used. In addition, recent research (i.e., Beaty,
Kenett, Hass, & Schacter, 2019) has shown that certain prompts may
be more facilitative of Ideational Fluency or Originality, depending on
the semantic richness of the ideas associated with that prompt. In
essence, the findings of any psychometric research are tied to the
actual item-stimuli that is administered to participants. Therefore, the
findings of the current study are most relevant for those researchers
who administer the same or similar AUT prompts as we administered
here, and those researchers who plan to administer AUT prompts that
are highly different than those administered here may need to interpret
our results with caution.
However, this lack of standardization of the AUT may also be an
unexpected boon for the creativity literature, in that the AUT measure
or administration procedures can be easily updated as more psycho-
metric evidence becomes available. In this study, we found substantial
differences in the way individual AUT prompts (e.g., Book) contrib-
uted to the reliability of the latent Originality factor. For example, for
both the human-rated and GloVe Originality scores, “Rope” was the
weakest loading item, indicating that item detracts from the overall
reliability of the Originality scores. However, this effect was more
pronounced in GloVe than for the humans, implying that the human
raters were capable of providing relatively stable estimates of partic-
ipant responses for “Rope,” whereas GloVe struggled more to quan-
tify the relevant semantic relations for that prompt. In addition, the
loading for “Shovel” on the GloVe Originality factor was also rela-
tively weak, implying that text-mining system is worse at representing
the semantic relations around “Shovel” than it is for the semantic
relations around “Brick,” for example. Overall, it is clear from these
findings that the text-mining–based scoring methods had much more
varying loadings across the AUT prompt than did the human-raters,
which illustrates the need, for those researchers who use text-mining
systems to measure Originality, to choose their AUT prompts care-
fully, and possibly pilot them with their chosen text-mining system
before administering them to a large number of participants.
Multiple Dimensions of Creativity Are
Needed for Research
Although this current investigation was mainly focused on the
psychometric quality of Originality scores, the validity portion of the
study also offers some valuable insight into the interrelations among
other dimensions of DT (i.e., Fluency and Elaboration), creative
personality (i.e., Openness and Intellect), and particular real-world
creative activities. Although the AUT-based Originality scores did not
well-predict self-reported creative activities, other dimensions of the
AUT scoring (i.e., Fluency and Elaboration) did. For example, the
capability of Fluency scores to significantly and positively predict
literary and musical creative activities, and the prediction of arts
activities with AUT Elaboration, highlight the continued usefulness of
both AUT Fluency and Elaboration scores in the creativity literature.
However, creative personality indicators were even better able to
predict self-reported creative activities, with Openness being the most
predictive (i.e., significant positive correlations with Arts, Cooking,
Crafts, and Literary Activities) and Intellect also being reasonably
predictive (i.e., significant positive correlations with Cooking and
Crafts activities). Of course, given the self-report nature of the Cre-
ative Activities Inventory as well as the personality questionnaires,
their relations may be inflated by participants’ creative self-concepts
(Karwowski, 2016). But, such an explanation can be ruled out for
Fluency and Elaboration, both of which predicted creative activities as
well as or stronger than human-rated Originality. Perhaps most im-
portantly, it should be observed that, of the creative activities that
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
15
MEASURING ORIGINALITY

were significantly predicted in this study, none was predicted by all of
the creativity indicators. This finding strongly highlights the need for
researchers to measure a variety of different indicators of creative
potential—including both DT and personality assessments—to max-
imize the impact of our research to understand the creative process.
As a future direction in this line of investigation, it may be important
for researchers interested in the psychometrics of creativity to con-
sider ways not only to automate Originality scoring using text-mining
or other computational methods, but also to automate scoring systems
for other dimensions of divergent thinking and creative potential. For
example, it remains to be seen whether or how text-mining systems
can be used for the quantification of Flexibility on the AUT or other
DT measures. In addition, creativity researchers have been perennially
interested in participant responses to the AUT that stand out because
of their sexual or violent content (Dumas & Strickland, 2018; Hudson,
1968), and text-mining models could conceivably be applied to au-
tomatically identify such responses. In our view, these future direc-
tions help illustrate the potential of computational psychometric work
in the creativity research area.
Coda
In our view, one main methodological bottleneck that limits the
productivity and impact of creativity research has historically been the
time- and resource-intensiveness of human-rated DT tasks. We in the
field have relied on hiring, training, and compensating human-raters
for decades, and graduate students situated within creativity research
laboratories have also often shouldered the burden of rating hundreds
or thousands of DT responses for Originality. In some cases, very
large–scale studies of creativity may even have seemed infeasible
because of the burden of using human-raters. Here, we found the
GloVe system is highly capable of approximating Originality scores
produced via human-raters, but much more rapidly and potentially
free of cost. Based on the findings of this study, we may be nearing
a time in the field when the work of scoring DT tasks for Originality
can be automated using a text-mining model, opening the door for
much larger-scale studies of DT and creativity, and hopefully leading
to increased reach and scope of research in the field. In the future,
such text-mining systems may be even easier to run (e.g., through
more user-friendly software) and may contribute to a streamlined and
automatic process of Originality measurement in creativity research.
References
Acar, S., & Runco, M. A. (2019). Divergent thinking: New methods, recent
research, and extended theory. Psychology of Aesthetics, Creativity, and
the Arts, 13, 153–158. http://dx.doi.org/10.1037/aca0000231
Barbot, B. (2018). The dynamics of creative ideation: Introducing a new
assessment paradigm. Frontiers in Psychology, 9, 2529. http://dx.doi
.org/10.3389/fpsyg.2018.02529
Barbot, B., Tan, M., Randi, J., Santa-Donato, G., & Grigorenko, E. L.
(2012). Essential skills for creative writing: Integrating multiple domain-
specific perspectives. Thinking Skills and Creativity, 7, 209–223. http://
dx.doi.org/10.1016/j.tsc.2012.04.006
Beaty, R. E., Kenett, Y. N., Hass, R., & Schacter, D. L. (2019). A fan effect
for creative thought: Semantic richness facilitates idea quantity but
constrains idea quality. PsyArXiv. Retrieved from https://psyarxiv.com/
pfz2g/
Beaty, R. E., Silvia, P. J., Nusbaum, E. C., Jauk, E., & Benedek, M. (2014).
The roles of associative and executive processes in creative cognition.
Memory & Cognition, 42, 1186–1197. http://dx.doi.org/10.3758/
s13421-014-0428-8
Benedek, M. (2018). Internally directed attention in creative cognition. In
R. E. Jung & O. Vartanian (Eds.), The Cambridge handbook of the
neuroscience of creativity (pp. 180–194). Cambridge, UK: Cambridge
University Press. http://dx.doi.org/10.1017/9781316556238.011
Bianchi, F., & Palmonari, M. (2017). Joint learning of entity and type
embeddings for analogical reasoning with entities. NL4AI@ AI IA,
57–68.
Bråten, I., Ferguson, L. E., Strømsø, H. I., & Anmarkrud, Ø. (2014).
Students working with multiple conflicting documents on a scientific
issue: Relations between epistemic cognition while reading and sourcing
and argumentation in essays. British Journal of Educational Psychology,
84, 58–85. http://dx.doi.org/10.1111/bjep.12005
Crossley, S., Dascalu, M., & McNamara, D. (2017). How important is size?
An investigation of corpus size and meaning in both latent semantic
analysis and latent Dirichlet allocation. Marco Island, Florida: The
Thirtieth International Florida Artificial Intelligence Research Society
Conference. Retrieved from https://www.aaai.org/ocs/index.php/
FLAIRS/FLAIRS19/paper/viewFile/18299/17416
De Deyne, S., Verheyen, S., & Storms, G. (2016). Structure and organi-
zation of the mental lexicon: A network approach derived from syntactic
dependency relations and word associations. In A. Mehler, A. Lücking,
S. Banisch, P. Blanchard, & B. Job (Eds.), Towards a theoretical
framework for analyzing complex linguistic networks (pp. 47–79). Ber-
lin, Germany: Springer. http://dx.doi.org/10.1007/978-3-662-47238-5_3
Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harsh-
man, R. (1990). Indexing by latent semantic analysis. Journal of the
American Society for Information Science, 41, 391–407. http://dx.doi
.org/10.1002/(SICI)1097-4571(199009)41:6391::AID-ASI1	3.0.CO;
2-9
DeYoung, C. G., Quilty, L. C., & Peterson, J. B. (2007). Between facets
and domains: 10 aspects of the Big Five. Journal of Personality and
Social Psychology, 93, 880–896. http://dx.doi.org/10.1037/0022-3514
.93.5.880
Diedrich, J., Jauk, E., Silvia, P. J., Gredlein, J. M., Neubauer, A. C., &
Benedek, M. (2018). Assessment of real-life creativity: The Inventory of
Creative Activities and Achievements (ICAA). Psychology of Aesthet-
ics, Creativity, and the Arts, 12, 304–316. http://dx.doi.org/10.1037/
aca0000137
Dumas, D. (2018). Relational reasoning and divergent thinking: An exam-
ination of the threshold hypothesis with quantile regression. Contempo-
rary Educational Psychology, 53, 1–14. http://dx.doi.org/10.1016/j
.cedpsych.2018.01.003
Dumas, D., & Dunbar, K. N. (2014). Understanding fluency and original-
ity: A latent variable perspective. Thinking Skills and Creativity, 14,
56–67. http://dx.doi.org/10.1016/j.tsc.2014.09.003
Dumas, D., & Dunbar, K. N. (2016). The creative stereotype effect. PLoS
ONE, 11, e0142567. http://dx.doi.org/10.1371/journal.pone.0142567
Dumas, D., & Runco, M. (2018). Objectively scoring divergent thinking
tests for originality: A re-analysis and extension. Creativity Research
Journal, 30, 466–468.
Dumas, D., & Strickland, A. L. (2018). From book to bludgeon: A closer
look at unsolicited malevolent responses on the alternate uses task.
Creativity Research Journal, 30, 439–450. http://dx.doi.org/10.1080/
10400419.2018.1535790
Fleiss, J. L., & Cohen, J. (1973). The equivalence of weighted kappa and
the intraclass correlation coefficient as measures of reliability. Educa-
tional and Psychological Measurement, 33(3), 613–619. http://dx.doi
.org/10.1177/001316447303300309
Foltz, P. W., Streeter, L. A., Lochbaum, K. E., & Landauer, T. K. (2013).
Implementation and applications of the intelligent essay assessor. In
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
16
DUMAS, ORGANISCIAK, AND DOHERTY

M. D. Shermis & J. Burstein (Eds.), Handbook of automated essay
evaluation: Current applications and new directions (pp. 68–88). New
York, NY: Routledge/Taylor & Francis Group.
Forster, E. A., & Dunbar, K. N. (2009). Creativity evaluation through latent
semantic analysis. Proceedings of the Annual Conference of the Cogni-
tive Science Society, 2009, 602–607.
Forthmann, B., Oyebade, O., Ojo, A., Günther, F., & Holling, H. (2019).
Application of latent semantic analysis to divergent thinking is biased by
elaboration. The Journal of Creative Behavior. Advance online publi-
cation. http://dx.doi.org/10.1002/jocb.240
Forthmann, B., Szardenings, C., & Holling, H. (2020). Understanding the
confounding effect of fluency in divergent thinking scores: Revisiting
average scores to quantify artifactual correlation. Psychology of Aesthet-
ics, Creativity, and the Arts. Advance online publication. http://dx.doi
.org/10.1037/aca0000196
Forthmann, B., Wilken, A., Doebler, P., & Holling, H. (2019). Strategy
induction enhances creativity in figural divergent thinking. The Journal
of Creative Behavior, 53, 18–29. http://dx.doi.org/10.1002/jocb.159
Furnham, A., Crump, J., & Swami, V. (2009). Abstract reasoning and big
five personality correlates of creativity in a British occupational sample.
Imagination, Cognition and Personality, 28, 361–370. http://dx.doi.org/
10.2190/IC.28.4.f
Gray, K., Anderson, S., Chen, E. E., Kelly, J. M., Christian, M. S., Patrick,
J., . . . Lewis, K. (2019). “Forward flow”: A new measure to quantify
free thought and predict creativity. The American Psychologist, 74,
539–554. http://dx.doi.org/10.1037/amp0000391
Green, A. E., Kraemer, D. J. M., Fugelsang, J. A., Gray, J. R., & Dunbar,
K. N. (2010). Connecting long distance: Semantic distance in analogical
reasoning modulates frontopolar cortex activity. Cerebral Cortex, 20,
70–76. http://dx.doi.org/10.1093/cercor/bhp081
Green, A. E., Kraemer, D. J. M., Fugelsang, J. A., Gray, J. R., & Dunbar,
K. N. (2012). Neural correlates of creativity in analogical reasoning.
Journal of Experimental Psychology: Learning, Memory, and Cogni-
tion, 38, 264–272. http://dx.doi.org/10.1037/a0025764
Guilford, J. P. (1967). The nature of human intelligence. New York, NY:
McGraw-Hill.
Günther, F., Dudschig, C., & Kaup, B. (2015). LSAfun-An R package for
computations based on Latent Semantic Analysis. Behavior Research
Methods, 47, 930–944. http://dx.doi.org/10.3758/s13428-014-0529-0
Gwet, K. L. (2014). Handbook of inter-rater reliability, 4th ed.: The
definitive guide to measuring the extent of agreement among raters.
Gaithersburg, MD: Advanced Analytics, LLC.
Hancock, G. R. (2001). Effect size, power, and sample size determination
for structured means modeling and mimic approaches to between-groups
hypothesis testing of means on a single latent construct. Psychometrika,
66, 373–388. http://dx.doi.org/10.1007/BF02294440
Hass, R. W. (2017a). Semantic search during divergent thinking. Cogni-
tion, 166, 344–357. http://dx.doi.org/10.1016/j.cognition.2017.05.039
Hass, R. W. (2017b). Tracking the dynamics of divergent thinking via
semantic distance: Analytic methods and theoretical implications. Mem-
ory & Cognition, 45, 233–244. http://dx.doi.org/10.3758/s13421-016-
0659-y
He, Q., von Davier, M., Greiff, S., Steinhauer, E. W., & Borysewicz, P. B.
(2017). Collaborative problem solving measures in the Programme for
International Student Assessment (PISA). In A. A. von Davier, M. Zhu,
& P. C. Kyllonen (Eds.), Innovative assessment of collaboration (pp.
95–111). Cham, Switzerland: Springer. http://dx.doi.org/10.1007/978-3-
319-33261-1_7
Hedge, C., Powell, G., & Sumner, P. (2018). The reliability paradox: Why
robust cognitive tasks do not produce reliable individual differences.
Behavior Research Methods, 50, 1166–1186. http://dx.doi.org/10.3758/
s13428-017-0935-1
Heinen, D. J. P., & Johnson, D. R. (2018). Semantic distance: An auto-
mated measure of creativity that is novel and appropriate. Psychology of
Aesthetics, Creativity, and the Arts, 12, 144–156. http://dx.doi.org/10
.1037/aca0000125
Hirschberg, J., & Manning, C. D. (2015). Advances in natural language
processing. Science, 349, 261–266. http://dx.doi.org/10.1126/science
.aaa8685
Hocevar, D. (1980). Intelligence, divergent thinking, and creativity. Intel-
ligence, 4, 25–40. http://dx.doi.org/10.1016/0160-2896(80)90004-5
Hornberg, J., & Reiter-Palmon, R. (2017). Creativity and the big five
personality traits: Is the relationship dependent on the creativity mea-
sure? In G. J. Feist, R. Reiter-Palmon, & J. C. Kaufman (Eds.), The
Cambridge handbook of creativity and personality research (pp. 275–
293). New York, NY: Cambridge University Press. http://dx.doi.org/10
.1017/9781316228036.015
Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance
structure analysis: Conventional criteria versus new alternatives. Struc-
tural Equation Modeling, 6, 1–55. http://dx.doi.org/10.1080/
10705519909540118
Hudson, L. (1968). Frames of mind: Ability, perception and self-perception
in the arts and sciences. Oxford, England: Norton.
Karwowski, M. (2016). The dynamics of creative self-concept: Changes
and reciprocal relations between creative self-efficacy and creative per-
sonal identity. Creativity Research Journal, 28, 99–104. http://dx.doi
.org/10.1080/10400419.2016.1125254
Karwowski, M., & Gralewski, J. (2013). Threshold hypothesis: Fact or
artifact? Thinking Skills and Creativity, 8, 25–33. http://dx.doi.org/10
.1016/j.tsc.2012.05.003
Kenett, Y. N. (2019). What can quantitative measures of semantic distance
tell us about creativity? Current Opinion in Behavioral Sciences, 27,
11–16. http://dx.doi.org/10.1016/j.cobeha.2018.08.010
Kenett, Y. N., Levi, E., Anaki, D., & Faust, M. (2017). The semantic
distance task: Quantifying semantic distance with semantic network path
length. Journal of Experimental Psychology: Learning, Memory, and
Cognition, 43, 1470–1489. http://dx.doi.org/10.1037/xlm0000391
Kerr, B. A., & Stull, O. A. (2019). Measuring creativity in research and
practice. In M. W. Gallagher & S. J. Lopez (Eds.), Positive psycholog-
ical assessment: A handbook of models and measures (2nd ed., pp.
125–138). Washington, DC: American Psychological Association.
http://dx.doi.org/10.1037/0000138-009
Kintsch, W., & Bowles, A. R. (2002). Metaphor comprehension: What
makes a metaphor difficult to understand? Metaphor and Symbol, 17,
249–262. http://dx.doi.org/10.1207/S15327868MS1704_1
Kjell, O. N. E., Kjell, K., Garcia, D., & Sikström, S. (2019). Semantic
measures: Using natural language processing to measure, differentiate,
and describe psychological constructs. Psychological Methods, 24, 92–
115. http://dx.doi.org/10.1037/met0000191
Kuhn, J. T., & Holling, H. (2009). Measurement invariance of divergent
thinking across gender, age, and school forms. European Journal of
Psychological Assessment, 25, 1–7. http://dx.doi.org/10.1027/1015-5759
.25.1.1
Kumar, A. A., Balota, D. A., & Steyvers, M. (2019). Distant connectivity
and multiple-step priming in large-scale semantic networks. Journal of
Experimental Psychology: Learning, Memory, and Cognition. Advance
online publication. http://dx.doi.org/10.1037/xlm0000793
Landauer, T. K., & Dumais, S. T. (1997). A solution to Plato’s problem:
The latent semantic analysis theory of acquisition, induction, and rep-
resentation of knowledge. Psychological Review, 104, 211–240. http://
dx.doi.org/10.1037/0033-295X.104.2.211
Landauer, T. K., Foltz, P. W., & Laham, D. (1998). An introduction to
latent semantic analysis. Discourse Processes, 25, 259–284. http://dx
.doi.org/10.1080/01638539809545028
Landauer, T. K., Laham, D., Rehder, B., & Schreiner, M. E. (1997). How
well can passage meaning be derived without using word order? A
comparison of Latent Semantic Analysis and humans. Proceedings of
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
17
MEASURING ORIGINALITY

the 19th Annual Meeting of the Cognitive Science Society, 412–417.
Mahwah, NJ: Erlbaum.
Landauer, T. K., McNamara, D. S., Dennis, S., & Kintsch, W. (2013).
Handbook of latent semantic analysis. London, UK: Psychology
Press.
Linacre, J. M., & Wright, B. D. (1988). Facets. Chicago, IL: MESA.
Lord, F. M. (2012). Applications of item response theory to practical
testing problems. London, UK: Routledge. http://dx.doi.org/10.4324/
9780203056615
Marek, R. J., & Ben-Porath, Y. S. (2017). Using the Minnesota Multiphasic
Personality Inventory-2-Restructured Form (MMPI-2-RF) in behavioral
medicine settings. In M. E. Maruish (Ed.), Handbook of psychological
assessment in primary care settings (pp. 631–662, 2nd ed.). New York,
NY: Routledge/Taylor & Francis Group.
McDonald, R. P. (1999). Test theory: A unified approach. Mahwah, NJ:
Erlbaum.
McKay, A. S., Karwowski, M., & Kaufman, J. C. (2017). Measuring the
muses: Validating the Kaufman Domains of Creativity Scale (K-DOCS).
Psychology of Aesthetics, Creativity, and the Arts, 11, 216–230. http://
dx.doi.org/10.1037/aca0000074
McNeish, D. (2018). Thanks coefficient alpha, we’ll take it from here.
Psychological Methods, 23, 412– 433. http://dx.doi.org/10.1037/
met0000144
McNeish, D., An, J., & Hancock, G. R. (2018). The thorny relation
between measurement quality and fit index cutoffs in latent variable
models. Journal of Personality Assessment, 100, 43–52. http://dx.doi
.org/10.1080/00223891.2017.1281286
McNeish, D., & Wolf, M. G. (2020). Thinking twice about sum scores.
Behavior Research Methods. Advanced online publication. http://dx.doi
.org/10.3758/s13428-020-01398-0
Mednick, S. A. (1962). The associative basis of the creative process.
Psychological Review, 69, 220 –232. http://dx.doi.org/10.1037/
h0048850
Messick, S. (1995). Validity of psychological assessment: Validation of
inferences from persons’ responses and performances as scientific in-
quiry into score meaning. American Psychologist, 50, 741–749. http://
dx.doi.org/10.1037/0003-066X.50.9.741
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation
of word representations in vector space. arXiv preprint arXiv,1301.
3781. Retrieved from https://arxiv.org/abs/1301.3781
Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
Distributed representations of words and phrases and their composition-
ality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q.
Weinberger (Eds.), Advances in neural information processing systems
(pp. 3111–3119). Red Hook, NY: Curran Associates, Inc.
Muthén, L. K., & Muthén, B. (2019). Mplus user’s guide (8th ed.). Los
Angeles, CA: Author.
Myszkowski, N., & Storme, M. (2019). Judge response theory? A call to
upgrade our psychometrical account of creativity judgments. Psychology
of Aesthetics, Creativity, and the Arts, 13, 167–175. http://dx.doi.org/10
.1037/aca0000225
Oleynick, V. C., DeYoung, C. G., Hyde, E., Kaufman, S. B., Beaty, R. E.,
& Silvia, P. J. (2017). Openness/intellect: The core of the creative
personality. In G. J. Feist, R. Reiter-Palmon, & J. C. Kaufman (Eds.),
The Cambridge handbook of creativity and personality research (pp.
9–27). New York, NY: Cambridge University Press. http://dx.doi.org/
10.1017/9781316228036.002
Organisciak, P. (2016). Term weights for 235k language and literature
texts [Data set]. Retrieved from https://www.ideals.illinois.edu/handle/
2142/89691
Organisciak, P., Teevan, J., Dumais, S., Miller, R. C., & Kalai, A. T. (2014,
September). A crowd of your own: Crowdsourcing for on-demand person-
alization. Proceedings of the Second AAAI Conference on Human Compu-
tation and Crowdsourcing (HCOMP 2014). Retrieved from https://www
.aaai.org/ocs/index.php/HCOMP/HCOM-P14/paper/viewFile/8972/
8969
Pennington, J., Socher, R., & Manning, C. (2014, October). Glove: Global
vectors for word representation. In A. Moscitti, A. Pang, & B. Dael-
emans (Eds.), Proceedings of the 2014 conference on empirical methods
in natural language processing (pp. 1532–1543). Doha, Qatar: Associ-
ation for Computational Linguistics. http://dx.doi.org/10.3115/v1/D14-
1162
Piatelli-Palmarini, M. (1980). Language and learning: The debate between
Jean Piaget and Noam Chomsky. Cambridge, MA: Harvard University
Press.
Plucker, J. A., & Makel, M. C. (2010). Assessment of creativity. In J. C.
Kaufman & R. J. Sternberg (Eds.), The Cambridge handbook of cre-
ativity (pp. 48–73). New York, NY: Cambridge University Press. http://
dx.doi.org/10.1017/CBO9780511763205.005
Prabhakaran, R., Green, A. E., & Gray, J. R. (2014). Thin slices of
creativity: Using single-word utterances to assess creative cognition.
Behavior Research Methods, 46, 641–659. http://dx.doi.org/10.3758/
s13428-013-0401-7
Primi, R., Silvia, P. J., Jauk, E., & Benedek, M. (2019). Applying many-
facet Rasch modeling in the assessment of creativity. Psychology of
Aesthetics, Creativity, and the Arts, 13, 176–186. http://dx.doi.org/10
.1037/aca0000230
Puryear, J. S., Kettler, T., & Rinn, A. N. (2017). Relationships of person-
ality to differential conceptions of creativity: A systematic review.
Psychology of Aesthetics, Creativity, and the Arts, 11, 59–68. http://dx
.doi.org/10.1037/aca0000079
Reiter-Palmon, R., Forthmann, B., & Barbot, B. (2019). Scoring divergent
thinking tests: A review and systematic framework. Psychology of
Aesthetics, Creativity, and the Arts, 13, 144–152. http://dx.doi.org/10
.1037/aca0000227
Robertson, S. E., & Jones, K. S. (1976). Relevance weighting of search
terms. Journal of the American Society for Information Science, 27,
129–146. http://dx.doi.org/10.1002/asi.4630270302
Silvia, P. J., Winterstein, B. P., Willse, J. T., Barona, C. M., Cram, J. T.,
Hess, K. I., . . . Richard, C. A. (2008). Assessing creativity with
divergent thinking tasks: Exploring the reliability and validity of new
subjective scoring methods. Psychology of Aesthetics, Creativity, and
the Arts, 2, 68–85. http://dx.doi.org/10.1037/1931-3896.2.2.68
Snow, R., O’Connor, B., Jurafsky, D., & Ng, A. Y. (2008, October). Cheap
and fast—But is it good?: evaluating non-expert annotations for natural
language tasks. Proceedings of the Conference on Empirical Methods in
Natural Language Processing (pp. 254–263). Stroudsburg, PA: Asso-
ciation for Computational Linguistics.
S¸tefa˘nescu, D., Banjade, R., & Rus, V. (2014, May). Latent semantic
analysis models on Wikipedia and TASA. The 9th Language Resources
Evaluation Conference (LREC), Reykjavik, Iceland. Retrieved from
http://www.lrec-conf.org/proceedings/lrec2014/pdf/403_Paper.pdf
Storme, M., Çelik, P., Camargo, A., Forthmann, B., Holling, H., & Lubart,
T. (2017). The effect of forced language switching during divergent
thinking: A study on bilinguals’ originality of ideas. Frontiers in Psy-
chology, 8, 2086. http://dx.doi.org/10.3389/fpsyg.2017.02086
Storme, M., Myszkowski, N., Çelik, P., & Lubart, T. (2014). Learning to
judge creativity: The underlying mechanisms in creativity training for
non-expert judges. Learning and Individual Differences, 32, 19–25.
http://dx.doi.org/10.1016/j.lindif.2014.03.002
Torrance, E. P. (1972). Predictive validity of the Torrance Tests of Creative
Thinking. The Journal of Creative Behavior, 6, 236–262. http://dx.doi
.org/10.1002/j.2162-6057.1972.tb00936.x
Torrance, E. P. (1988). The nature of creativity as manifest in its testing.
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
18
DUMAS, ORGANISCIAK, AND DOHERTY

In R. J. Sternberg (Ed.), The nature of creativity (pp. 43–75). New York,
NY: Cambridge University Press.
von Davier, A. A. (2017). Computational psychometrics in support of
collaborative educational assessments. Journal of Educational Measure-
ment, 54, 3–11. http://dx.doi.org/10.1111/jedm.12129
White, H. A., & Shah, P. (2016). Scope of semantic activation and
innovative thinking in college students with ADHD. Creativity Research
Journal, 28, 275–282. http://dx.doi.org/10.1080/10400419.2016
.1195655
Yoon, C. H. (2017). A validation study of the Torrance Tests of Creative
Thinking with a sample of Korean elementary school students. Thinking
Skills and Creativity, 26, 38–50. http://dx.doi.org/10.1016/j.tsc.2017.05
.004
Received June 3, 2019
Revision received January 29, 2020
Accepted February 19, 2020 
This document is copyrighted by the American Psychological Association or one of its allied publishers.
This article is intended solely for the personal use of the individual user and is not to be disseminated broadly.
19
MEASURING ORIGINALITY
