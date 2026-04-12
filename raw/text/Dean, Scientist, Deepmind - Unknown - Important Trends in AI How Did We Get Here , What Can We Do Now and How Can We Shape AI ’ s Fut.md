# Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut

- Source PDF: `raw/pdfs/Dean, Scientist, Deepmind - Unknown - Important Trends in AI How Did We Get Here , What Can We Do Now and How Can We Shape AI ’ s Fut.pdf`
- Generated from: `scripts/extract_pdf_text.py`

## Extracted Text

Jeff Dean, Chief Scientist, Google Research & Google DeepMind
@jeffdean.bsky.social and @JeffDean
ai.google/research/people/jeff
Presenting the work of many people at Google and elsewhere
Important Trends in AI:
How Did We Get Here, 
What Can We Do Now and How 
Can We Shape AI’s Future?

Some observations
The kinds of computations we want to run and the hardware on 
which we run them is changing dramatically
Increasing scale (compute, data, model size) delivers better results
In recent years, ML has completely changed our expectations of 
what is possible with computers
Algorithmic and model architecture improvements have provided 
massive improvements as well

Fifteen Years of Machine Learning Advances
or
How Did Today’s Models Come To Be?

Key Building Block from Last Century: Neural Networks
Key building block: neural networks, made up of artificial neurons, loosely designed to 
mimic how real neurons behave
weights
weights

Key Building Block from Last Century: Backpropagation
Key building block: backpropagation of errors (using chain rule) gives effective algorithm 
for updating the weights of a neural network to minimize errors on training data
weights
weights
Backpropgation of 
errors gives an 
algorithm for how to 
update the weights of 
whole neural network 
based on errors 
observed at the 
outputs of the model

2012: Scale Matters
Le et al., ICML 2012, arxiv.org/abs/1112.6209
Training a very large neural network (60X bigger than previous largest neural network) using 
16,000 CPU cores gives major advances in quality
(~70% relative improvement in ImageNet 22K state-of-the-art)

2012: Distributed Training on Many Computers
Large Scale Distributed Deep Networks, Dean et al., NeurIPS 2012, 
research.google.com/archive/large_deep_networks_nips2012.pdf 
Combining model parallelism and data parallelism for neural network training across 
thousands of computers enables training of much larger (50-100X) neural networks than 
previously possible
Model parallelism
Data parallelism

Word2Vec
2013: Distributed Representations of Words Are Powerful
ICLR 2013 workshop, arxiv.org/abs/1310.4546 
Appeared in NeurIPS 2013, arxiv.org/abs/1310.4546
Distributed representations of words are powerful:
(1) Nearby words in high dimensional space are related
cat, puma, tiger, … are all nearby
(2) Directions are meaningful
king – queen ~= man – woman

Sequence to Sequence
2014: Models that Map One Sequence to Another are Powerful
Appeared in NeurIPS 2014, arxiv.org/abs/1409.3215 
Use a neural encoder over an input sequence to generate state, use that to 
initialize state of a neural decoder.  Scale up LSTMs and this works. 

2015: Specialized Hardware for Neural Network Inference
Tensor Processing Unit (TPU)
v1: 2015, 92 teraops (inference only)
handful of speciﬁc 
operations
×
=
reduced
precision
ok
about 1.2
× about 0.6
about 0.7
1.21042
× 0.61127
0.73989343
NOT
Appeared in ISCA, 2017, arxiv.org/abs/1704.04760.  Now most cited paper in ISCA’s 50 year history
Specialization is much more efficient:
Compared to contemporary CPUs & GPUs:
TPU v1 is 15X-30X faster
TPU v1 is 30X-80X more energy efficient

2016: Specialized Supercomputers for Neural Network Training
TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for 
Embeddings, Jouppi et al., ISCA 2023, arxiv.org/abs/2304.01433 
Connect thousands of chips together (TPU pods) with custom high-speed networks 
to enable faster neural network training

Continual Hardware Performance Scaling
blog.google/products/google-cloud/ironwood-tpu-age-of-inference/ 
11
petaflops
42522 
petaflops
1126 
petaflops

Continual Hardware Improvements in Energy Efficiency
Peak FP8 flops delivered per watt of thermal design power per chip package
blog.google/products/google-cloud/ironwood-tpu-age-of-inference/ 
~30X energy 
efficiency 
improvement 
vs. TPU v2 

github.com/jax-ml/jax 
Open source tools enable the whole community
pytorch.org
tensorflow.org

2017: Transformer Model Architecture: Attention
Attention is All You Need, Vaswani et al., NeurIPS 2017, arxiv.org/abs/1706.03762
Don’t try to force state into single recurrent distributed representation.
Instead, save all past representations and attend to them.

Attention is All You Need, Vaswani et al., NeurIPS 2017, arxiv.org/abs/1706.03762
Higher accuracy w/ 10X-100X less compute and 10X smaller models!
Figure from Scaling Laws for Neural Language Models, 
Kaplan et al., arxiv.org/abs/2001.08361
2017: Transformer Model Architecture: Attention

2018: Language Modeling At Scale With Self-Supervised Data
There’s lots of text in the world!  Self-supervised learning on this text can 
provide very large amounts of training data with the “right” answer known (“wrong 
guess” is used to provide gradient descent loss training signal)
Self-supervised learning 
on text with large models 
is one of the major 
reasons chat/language 
models have gotten so 
good
Language Models are Few-Shot Learners, Brown et al., NeurIPS, 2020, arxiv.org/abs/2005.14165 
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., ACL 2019, arxiv.org/abs/1810.04805 

There’s lots of text in the world!  Self-supervised learning on this text can 
provide very large amounts of training data with the “right” answer known (“wrong 
guess” is used to provide gradient descent loss training signal)
Different kinds of training objectives:
Autoregressive (look at prefix, predict next word):
Zürich is ______
Zürich is the _______
Zürich is the largest _______
Fill-in-the-Blank (e.g. look in both directions, BERT):
Zürich ____ the largest ____ in ______.
Zürich is the ______ city ____ Switzerland.
….
Self-supervised learning 
on text with large models 
is one of the major 
reasons chat/language 
models have gotten so 
good
Language Models are Few-Shot Learners, Brown et al., NeurIPS, 2020, arxiv.org/abs/2005.14165 
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Devlin et al., ACL 2019, arxiv.org/abs/1810.04805 
2018: Language Modeling At Scale With Self-Supervised Data

2021: Transformers for Vision
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Alexey Dosovitskiy et al., ICLR 2021, 
arxiv.org/abs/2010.11929 
Visualization of 
attention mechanism

2017: Sparse Models (e.g. Mixture of Experts) Outperform 
Dense Models
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton and Jeff Dean.
ICLR 2017, arxiv.org/abs/1701.06538 
Give model much larger capacity w/ lots of experts but only activate a few chosen experts per token:
(A) ~8X reduction in training compute cost for ~same accuracy, or
(B) major accuracy improvements for same training compute cost
or
(A)
(B)

Gemini 1.5 Pro/Gemini 2.0/Gemini 2.5 use mixture-of-expert (MoE) architectures, building on a long line 
of Google research efforts on sparse models:
●
2017: Shazeer et al., Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.  
ICLR 2017. arxiv.org/abs/1701.06538 
●
2020: Lepikhin et al., GShard: Scaling giant models with conditional computation and automatic sharding. 
ICLR 2020. arxiv.org/abs/2006.16668 
●
2021: Carlos Riquelme et al., Scaling vision with sparse mixture of experts, NeurIPS 2021. 
arxiv.org/abs/2106.05974
●
2021: Fedus et al., Switch transformers: Scaling to trillion parameter models with simple and efficient 
sparsity. JMLR 2022.  arxiv.org/abs/2101.03961 
●
2022: Clark et al., Unified scaling laws for routed language models, ICML 2022. arxiv.org/abs/2202.01169 
●
2022: Zoph et al., Designing effective sparse expert models. arxiv.org/abs/2202.08906
●
2023: Puigcerver et al., From Sparse to Soft Mixtures of Experts.  arxiv.org/abs/2308.00951
●
2023: Obando-Cero et al., Mixtures of Experts Unlock Parameter Scaling for Deep RL. 
arxiv.org/abs/2402.08609 
●
2024: Raposo et al., Mixture-of-Depths: Dynamically allocating compute in transformer-based language 
models. arxiv.org/abs/2404.02258
●
2024: Douillard et al., DiPaCo: Distributed Path Composition. arxiv.org/abs/2403.10616 
Continued Research on Sparse Models

2018: Software abstractions for Distributed ML Computations
Example: Pathways
Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533  
Scalable software can simplify running large-scale computations
Region A
Building 1
Building 2
Region B
Building 1
…

Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533  
Scalable software can simplify running large-scale computations
Region A
Building 1
Building 2
Region B
Building 1
…
With JAX+Pathways, entire training process 
driven by a single Python process on one host
Client
2018: Software abstractions for Distributed ML Computations

Pathways: Asynchronous Distributed Dataflow for ML, Barham et al., MLSys 2022: arxiv.org/abs/2203.12533  
Pathways: Now Available for Cloud Customers
Pathways: Enables a single JAX client can see and use many devices (e.g. 1 to 100,000 
chips), even though these are distributed across many hosts and even many TPU pods

2022: “Thinking longer” at inference time is very useful
Chain of Thought Prompting Elicits Reasoning in Large Language Models, Jason Wei, Xuezhi Wang, Dale 
Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou, 2022, arxiv.org/abs/2201.11903 
“Chain of Thought prompting” is one such technique

2022: “Thinking longer” at inference time is very useful
Prompting model to “show its work” improves accuracy on reasoning tasks 
dramatically
Model scale
(billions of parameters)
Solve rate (%age)
“Chain of Thought prompting” is one such technique
Chain of Thought Prompting Elicits Reasoning in Large Language Models, Jason Wei, Xuezhi Wang, Dale 
Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou, 2022, arxiv.org/abs/2201.11903 

2014: Distillation: Use Powerful “Teacher” Models to Make 
Smaller, Cheaper “Student” Models
Rejected from NeurIPS 2014.  Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.
“performed the Concerto for “
__?__
Real next word: 
“Violin”
Distillation: Use large high quality model as “teacher” when training smaller 
“student” model

2014: Distillation: Use Powerful “Teacher” Models to Make 
Smaller, Cheaper “Student” Models
Distillation: Use large high quality model as “teacher” when training smaller 
“student” model
“performed the Concerto for “
__?__
Real next word: 
“Violin”
Teacher model says: 
“Violin: 0.4, Piano: 0.2, Trumpet: 0.01, Airplane: 0.00000001”
Rejected from NeurIPS 2014.  Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.
Gives much richer signal for 
training: try to get student to 
match “soft probability 
distribution” of large model

2014: Distillation: Use Powerful “Teacher” Models to Make 
Smaller, Cheaper “Student” Models
Distillation: Use large high quality model as “teacher” when training smaller 
“student” model
“performed the Concerto for “
__?__
Real next word: 
“Violin”
Teacher model says: 
“Violin: 0.4, Piano: 0.2, Trumpet: 0.01, Airplane: 0.00000001”
Rejected from NeurIPS 2014.  Published in workshop & put on Arxiv: arxiv.org/abs/1503.02531. 24,000+ citations.

2022: Many Different Parallelism Schemes During Inference
Right choices for how to distribute 
inference computation heavily influenced 
by things like batch size or latency 
constraints
Efficiently Scaling Transformer Inference, Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James 
Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean, arxiv.org/abs/2211.05102

2022: Many Different Parallelism Schemes During Inference
Efficiently Scaling Transformer Inference, Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James 
Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, Jeff Dean, arxiv.org/abs/2211.05102

2023: Speculative Decoding
Fast Inference from Transformers via Speculative Decoding, Yaniv Leviathan, Matan Kalman & Yossi Matias,
ICML ‘23, arxiv.org/abs/2211.17192 
Use small “drafter” model to predict next K tokens
●
Then predict next K tokens in one shot with large model (more efficient: batch size K not 1)
●
Advance generation by as many tokens as match in prefix of size K
●
Guaranteed identical output distribution
Larger, slower model
Larger, slower model
Faster model (drafter)
vs

Innovations at Many Levels
Hardware
Software abstractions
Model architecture
Training algorithms
Inference algorithms 
Pathways
TPUv1 → TPUv2 → TPUv3 → TPUv4 → TPUv5p → Trillium → Ironwood
DistBelief
Word2Vec
Seq2Seq
Transformers
MoEs
Unsupervised and 
Self-Supervised Learning
Distillation
Visual Transformers
SFT + RLxF
Chain-of-Thought
Speculative Decoding
Inference-time 
compute scaling
Asynchronous
Training

Gemini:
Putting These Advances Together

Project started in Feb 2023
Many collaborators from Google DeepMind, Google Research, and rest of Google
Goal: Train the world’s best multimodal models and use them all across Google
Gemini 1.0: Dec 2023
Gemini 1.5: Feb 2024 (demonstrated 10M token context window, Flash model)
Gemini 2.0: Dec 2024 (2.0 Flash as good as 1.5 Pro, multimodal live streaming, …)
Gemini 2.0 Thinking: Jan 2025 (2.0 Flash Experimental Thinking)
Gemini 2.5: Mar 2025 (2.5 Pro released), Apr 2025 (“2.5 Flash coming soon”)
https://blog.google/technology/ai/google-gemini-ai 
https://g.co/gemini 
Gemini: A Family of Highly Capable Multimodal Models, by the Gemini Team, arxiv.org/abs/2312.11805 
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, by the Gemini Team, arxiv.org/abs/2403.05530 

Gemini - multimodal from the start
Gemini - Multimodal from the start
Gemini: A Family of Highly Capable Multimodal Models, by the Gemini Team, arxiv.org/abs/2312.11805 

Gemini 1.5
Increased context length
Models can now handle up to 10 million 
tokens, with external APIs now offering up
to 2 million tokens for text and/or video.
Clearer context
The information within the context window 
is clearer, reducing hallucinations & 
enabling in-context learning.
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, by the Gemini Team, arxiv.org/abs/2403.05530 

Gemini 2.0
blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024 
(Like 1.0 and 1.5 and 2.5) Builds on many of 
the innovations I just described:
●
TPUs
●
Cross-datacenter training
●
Pathways
●
JAX
●
Distributed representations of words
●
Transformers
●
Sparse Mixture of Experts
●
Distillation
●
+ … many more innovations …

Gemini 2.5 Pro
blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ 
Our most capable model (for now!)

Gemini 2.5 Pro
blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/ 
Our most capable model (for now!)
Leaderboard positions
●
#1 LMSYS
●
# LiveBench
●
#1 Humanity’s Last Exam
●
#1 SEAL
●
#1 Artificial Analysis
●
#1 Aider Polyglot
●
#1 MathArena.ai
●
#1 Mensa IQ test
●
#1 Fiction.LiveBench
●
#1 SimpleBench
●
#1 Kagi leaderboard
●
#2 WebDev Arena
●
#4 LiveCodeBench
●
#4 NYT Connections
●
#2 Creative Writing
●
#4 Vectara
●
# 1 Perfect Information Game

Users Generally Enjoying Capabilities of Gemini 2.5 Pro

Long context abilities are very helpful (especially for code)

Pushing the Pareto Frontier of Optimal Quality/Price

Organizing a Large-Scale Scientific Effort 
Like Gemini

Many Contributors in Many Different Areas

Many Contributors in Many Different Areas

Gemini Structure & Ways of Working
Model Development Areas
Pre-training
Post-training
On-device Models
Core Areas
Data
Infrastructure
Serving
Evals
Codebase
Longer-term Research
Capabilities
Safety
Vision
Audio
Code
Agents
Internationalization
Overall Leads
Program Management
Product Management
…
…
…

Gemini Structure & Ways of Working
Many people in many locations:
~⅓ in San Francisco Bay Area
~⅓ in London
~⅓ in many other places:
NYC, Paris, Boston, Zürich, Bangalore, Tel Aviv, Seattle, …
Time zones are annoying!
●
“Golden Hours” between California/West Coast and London/Europe 
are important
   

Gemini Structure & Ways of Working
Lots and lots of large and small discussions and information sharing conducted via 
Google Chat Spaces (I’m in 200+ such spaces)
RFCs (Request for Comment): semi-formal way of getting feedback, knowing what 
others are working on, etc.
Leaderboards and common baselines enable data-driven decision making about how 
to improve
●
Multiple rounds of experimentation.
●
Many experiments at small scale
●
Advance smaller number of successful experiments to next scale
●
Every so often (every few weeks), incorporate successful experiments 
demonstrated at largest experimental scale into new candidate baseline
●
Repeat
   

Training at Scale:
Silent Data Corruption errors (SDCs)
Despite best efforts, given the scale of ML systems 
and the size of ML training jobs, hardware errors 
can occur, and sometimes incorrect computations 
from one buggy chip can spread and infect the 
entire training system 

Silent data corruption
Non-deterministically produce incorrect 
results, silently
Challenging problem when running largely 
independent computation
Multiplicatively worse at scale with 
synchronous stochastic gradient descent
Can quickly spread results across 
thousands of components across ML 
supercomputer
Cores that Don't Count, Peter H. Hochschild, Paul Jack Turner, Jeffrey C. Mogul, Rama Krishna Govindaraju, Parthasarathy 
Ranganathan, David E Culler, Amin Vahdat, HotOS 2021, research.google/pubs/cores-that-dont-count/ 

Metrics anomaly: anomaly due to SDC
Anomaly due to SDC
Time
Gradient Norm

Anomaly with NO SDC
Time
Gradient Norm
Metrics anomaly: expected anomaly (no SDC)

SDC detected with NO anomaly
The step replay shows different values, 
but both values are in the normal range.
Time
Gradient Norm
SDC with no metrics anomaly

Defective machine 
causes SDC
SDC checker 
automatically 
identifies SDC
SDC Checker 
moves training to 
hot spare and 
sends defective 
machine for repair
Normal training 
state
Synchronous training worker
SDC checker
Hot spare
ML Controller transparently handles Silent Data Corruption 
(SDC)

What Can These Models Do?


First part of 
chapter 1
           In-context learning: Kalamang translation
Example

           In-context learning: Kalamang translation
Kalamang is only spoken by ~130 people in eastern Indonesian Papua
Example

In-context learning: Kalamang translation
With in-context info, model can translate as effectively as a human learner
who has spent months on the same language materials

Video of bookshelf
 -> JSON
Example

“The killer app of
Gemini 1.5 Pro is video.”
Simon Willison
…

Video understanding
& summarization
Example

In a table, please write 
the sport, the 
teams/athletes involved, 
the year and a short 
description of why each 
of these moments in 
sports are so iconic.


https://climatelabbook.substack.com/p/data-rescue-with-ai 
Digitization of historical data
Example

Gemini 2.5 Pro example:
Code Generation via High Level Language



Inference time compute gives us another 
dimension of compute for quality scaling 

deepmind.google/technologies/gemini/flash-thinking/

deepmind.google/technologies/gemini/flash-thinking/

Now That We Have These Powerful 
Models, What Will This Mean?

74
●
Form team of senior computer scientists + rising stars in AI
○
From academia, big tech and startups
●
Propose what impact could be given directed research & 
policy efforts on AI for public good
○
Rather than predict societal impact of AI given a laissez faire approach 
●
Aim to shape AI’s upsides and dampen AI’s downsides
○
For high, middle, and low income nations
●
Audience: AI practitioners + policymakers + public
●
Approach: Interview 24 experts in 7 ﬁelds
○
Employment, Education, Healthcare, Information, Media, Governance, 
and Science
○
e.g. Barack Obama, Sal Khan, John Jumper, Neal Stephenson, Dario 
Amodei, Bob Wachter, …
●
Uncovered 5 guidelines for AI for public good
Mariano-Florentino Cuéllar
Jeff Dean
John Hennessy
Finale Doshi-Velez
Andy Konwinski
Sanmi Koyejo
Pelonomi Moiloa
Emma Pierson
David Patterson
Shaping AI's Impact on Billions of Lives

Shaping AI's Impact on Billions of Lives
75
“Shaping AI's Impact on Billions of Lives,” by Mariano-Florentino (Tino) Cuéllar, Jeff Dean, Finale 
Doshi-Velez, John Hennessy, Andy Konwinski, Sanmi Koyejo, Pelonomi Moiloa, Emma Pierson, and 
David Patterson, December, 2024
See  ShapingAI.com and arxiv.org/abs/2412.02730 
Mariano-Florentino Cuéllar
Jeff Dean
John Hennessy
Finale Doshi-Velez
Andy Konwinski
Sanmi Koyejo
Pelonomi Moiloa
Emma Pierson
David Patterson

Humans and AI systems working as a team can 
do more than either on their own
●
AI focused on human productivity produce 
more positive beneﬁts than those focused 
on human labor replacement
○
Increases human employability
○
Bonus: People can also be safeguards if AI veers 
off course in areas not well trained
○
Bonus: People and AIs tend to make different 
mistakes, so collaboration of experts with AI can 
also improve results
●
Productivity focus helps both AI and people 
succeed 
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

To increase employment, aim for productivity 
improvements in ﬁelds that create more jobs
●
Despite tremendous productivity gains in computing and 
passenger jets, the US in 2020 had 8 times more 
commercial airline pilots and 11 times more 
programmers than in 1970
●
Demand for passenger travel and programming was 
elastic ⇒ more jobs
○
Goods with elastic demand are those where a decrease in price 
results in a large increase in the quantity acquired
●
US agriculture demand is inelastic, so productivity gains 
⇒ fewer jobs
○
From 20% of US workforce to 2% in one lifetime (1940 to 2020)
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

What could be impact in next 5 years of near 
term AI by following the guidelines? 
●
To give concrete targets for improving AI’s 
impact, propose 
milestoneskilometerstones per ﬁeld
●
Rather than recognize past achievements,  
offer signiﬁcant inducement prizes that try 
to stimulate progress on these milestones
○
E.g., XPRIZE, Netﬂix, Kaggle, …
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

Education AI Milestone: Worldwide Tutor 
●
A tutoring tool to accelerate general education 
for every child 
○
In their language 
○
In their culture 
○
In their best learning style
●
To help teachers with challenge of supporting a 
range of student capability 
○
Keeping high-achieving students engaged while 
supporting those who struggle
●
E.g., Rising Academies* in Africa
○
Improves student outcomes by one grade level relative 
to students without it
79
* Henkel, Owen, Hannah Horne-Robinson, Nessie Kozhakhmetova, and Amanda Lee. “Eﬀective and Scalable Math Support: Experimental Evidence on the 
Impact of an AI-Math Tutor in Ghana.” In International Conference on Artiﬁcial Intelligence in Education, pp. 373-381. Cham: Springer Nature Switzerland, 2024.

Healthcare AI Milestone: Broad Medical AI 
●
Learns from many data modalities 
○
Images, laboratory results, health records, genomics, 
medical research, …
●
Can help carry out diverse set of tasks
○
Bedside decision support
○
Interacting with patients after leaving hospital
○
Drafting radiology reports that describe both abnormalities 
and relevant normal ﬁndings 
■
While taking into account the patient’s history 
●
Can explain recommendations using written or 
spoken text and images
●
Milestone requires deﬁning metrics and benchmarks 
to measure progress 
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

Information AI Milestone: 
Civic Discourse Platform 
●
Mediates conversations or attitudes to enhance 
public understanding and civic discourse 
○
Move communities from polarization to pluralism 
●
AI system makes suggestions on how to rephrase 
comments and questions more diplomatically* 
●
AI system to hold discussions with conspiracy 
theorists**
●
AI systems could help bring consensus on diﬃcult 
issues across whole populations***
* Argyle, Lisa, et al. “Leveraging AI for democratic discourse: Chat interventions can improve online political conversations at scale.” Proc. National Academy of 
Sciences, vol. 120, no. 41, 2023.
** Costello, Thomas, Gordon Pennycook, and David Rand. “Durably reducing conspiracy beliefs through dialogues with AI.” Science, vol. 385, no. 6714, 2024, p. 
Eadq1814.
*** Tsai, Lily and Alex Pentland. “Rediscovering the Pleasures of Pluralism: The Potential of Digitally Mediated Civic Participation,” The Digitalist Papers, 2024.

Science
●
Advances in science via AI could be one of 
largest impacts for public good
●
Many examples:
○
AlphaFold for protein folding
○
Black hole visualization
○
Flood forecasting
○
Materials discovery
○
Neural net-based weather prediction
○
Airplane contrail reduction to reduce CO2e
○
Controlling plasma for nuclear fusion
○
…
●
Most ﬁelds of science excited about AI 
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

Science AI Milestone: 
Scientist’s AI Aide/Collaborator 
●
Accelerate pace of science by improving the 
productivity of scientists
○
Help suggest interesting hypotheses and automate 
experiments
○
Identify important new relevant research, ideally 
customized to individual to summarize what is new 
compared to what the scientist already knew
Early example: Google’s Co-Scientist work*
●
Multi-agent scientiﬁc discovery system, showing 
inference time compute scaling leads to better rated 
hypotheses
* research.google/blog/accelerating-scientiﬁc-breakthroughs-with-an-ai-co-scientist/
Shaping AI's Impact on Billions of Lives, see ShapingAI.com and arxiv.org/abs/2412.02730 

Conclusions 
●
AI models and products are becoming incredibly 
powerful and useful tools
○
Further research and innovation will continue this trend
●
Will have dramatic impact in many diverse areas:
○
Healthcare, education, scientiﬁc research, media 
creation, misinformation, … 
●
Potentially makes deep expertise more available to 
many more people
●
Done well, our AI-assisted future is bright!
