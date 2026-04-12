# Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Vlasov, Mosig, Nichol - 2019 - Dialogue Transformers.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/1910.00486
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Dialogue Transformers

Vladimir Vlasov

vladimir@rasa.com

Rasa

  
Johannes E. M. Mosig

j.mosig@rasa.com

Rasa

  
Alan Nichol

alan@rasa.com

Rasa

###### Abstract

We introduce a dialogue policy based on a transformer architecture [Vaswani et al., 2017], where the self-attention mechanism operates over the sequence of dialogue turns.
Recent work has used hierarchical recurrent neural networks to encode multiple utterances in a dialogue context, but we argue that a pure self-attention mechanism is more suitable.
By default, an RNN assumes that every item in a sequence is relevant for producing an encoding of the full sequence, but a single conversation can consist of multiple overlapping discourse segments as speakers interleave multiple topics.
A transformer picks which turns to include in its encoding of the current dialogue state, and is naturally suited to selectively ignoring or attending to dialogue history.
We compare the performance of the Transformer Embedding Dialogue (TED) policy to an LSTM and to the REDP [Vlasov et al., 2018], which was specifically designed to overcome this limitation of RNNs.

## I Introduction

Conversational AI assistants promise to help users achieve a task through natural language.
Interpreting simple instructions like please turn on the lights is relatively straightforward, but to handle more complex tasks, these systems must be able to engage in multi-turn conversations.

The goal of this paper is to show that the transformer architecture [Vaswani et al., 2017] is more suitable for modeling multi-turn conversations than the commonly used recurrent models.
To compare the basic mechanisms that are at the heart of the sequence encoding we intentionally choose simple architectures.
The proposed TED architecture should be thought of as a candidate building block for use in developing state-of-the-art architectures in various dialogue tasks.

Not every utterance in a conversation has to be a response to the most recent utterance by the other party.
Groz and Sidner [Grosz and Sidner, 1986] consider conversations as an interleaved set of discourse segments, where a discourse segment (or topic) is a set of utterances that directly respond to each other.
These sequences of turns may not directly follow one another in the conversation.
An intuitive example of this is the need for sub-dialogues in task-oriented dialogue systems. Consider this conversation:

```
BOT: Your total is $15.50 - shall I
 charge the card you used last time?
 USER: Do I still have credit on my
 account from that refund I got?
 BOT: Yes, your account is $10 in credit.
 USER: Ok, great.
 BOT: Shall I place the order?
 USER: Yes.
 BOT: Done. You should have your items tomorrow.
```

#### Dialogue Stacks

The assistant’s question Shall I place the order? prompts the return to the task at hand: completing a purchase.
One model is to think of these sub-dialogues as existing on a stack, where new topics are pushed on to the stack when they are introduced and popped off the stack once concluded.

In the 1980s, Groz and Sidner [Grosz and Sidner, 1986] argued for representing dialogue history as a stack of topics, and later the RavenClaw [Bohus and Rudnicky, 2009] dialogue system implemented a dialogue stack for the specific purpose of handling sub-dialogues.
While a stack naturally allows for sub-dialogues to be handled and concluded, the strict structure of a stack is also limiting.
The authors of RavenClaw argue for explicitly tracking topics to enable the contextual interpretation of the user intents.
However, once a topic has been popped from the dialogue stack, it is no longer available to provide this context.
In the example above, the user might follow up with a further question like so that used up my credit, right?. If the topic of refund credits has been popped from the stack, this can no longer help clarify what the user wants to know.
Since there is in principle no restriction to how humans revisit and interleave topics in a conversation, we are interested in a more flexible structure than a stack.

#### Recurrent Neural Networks

A common choice in recent years has been to use a recurrent neural network (RNN) to process the sequence of previous dialogue turns, both for open domain [Sordoni et al., 2015, Serban et al., 2016] and task-oriented systems [Williams et al., 2017].
Given enough training data, an RNN should be able to learn any desired behaviour.
However, in a typical low-resource setting where no large corpus for training a particular task is available, an RNN is not guaranteed to learn to generalize these behaviours.
Previous work on modifying the basic RNN structure to include inductive biases for this behaviour into a dialogue policy was conducted by Vlasov et al. [Vlasov et al., 2018] and Sahay et al. [Sahay et al., 2019].
These works aim to overcome a feature of RNNs that is undesirable for dialogue modeling.
RNNs by default consume the entire sequence of input elements to produce an encoding, unless a more complex structure like a Long Short-Term Memory (LSTM) cell is trained on sufficient data to explicitly learn that it should ‘forget’ parts of a sequence.

#### Transformers

The transformer architecture has in recent years replaced recurrent neural networks as the standard for training language models, with models such as Transformer-XL [Dai et al., 2019] and GPT-2 [Radford et al., 2019] achieving much lower perplexities across a variety of corpora and producing representations that are useful for a variety of downstream tasks [Wang et al., 2018, Devlin et al., 2018].
In addition, transformers have recently shown to be more robust to unexpected inputs (such as adversarial examples) [Hsieh et al., 2019].
Intuitively, because the self-attention mechanism preselects which tokens will contribute to the current state of the encoder, a transformer can ignore uninformative (or adversarial) tokens in a sequence.
To make a prediction at each time step, an LSTM needs to update its internal memory cell, propagating this update to further time steps.
If an input at the current time step is unexpected, the internal state gets perturbed and at the next time step the neural network encounters a memory state unlike anything encountered during training.
A transformer accounts for time history via a self-attention mechanism, making the predictions at each time step independent of each other.
If a transformer receives an irrelevant input, it can ignore it and use only the relevant previous inputs to make a prediction.

Since a transformer chooses which elements in a sequence to use to produce an encoder state at every step, we hypothesise that it could be a useful architecture for processing dialogue histories.
The sequence of utterances in a conversation may represent multiple interleaved topics, and the transformer’s self-attention mechanism can simultaneously learn to disentangle these discourse segments and also to respond appropriately.

## II Related Work

#### Transformers for open-domain dialogue

Multiple authors have recently used transformer architectures in dialogue modeling.
Henderson et al. [Henderson et al., 2019] train response selection models on a large dataset from Reddit where both
the dialogue context and responses are encoded with a transformer.
They show that these architectures can be pre-trained on a large, diverse dataset and later fine-tuned for task-oriented dialogue in specific domains.
Dinan et al. [Dinan et al., 2018] used a similar approach, using tranformers to encode the dialogue context as well as
background knowledge for studying grounded open-domain conversations.
Their proposed architecture comes in two forms: a retrieval model where another transformer is used to
encode candidate responses which are selected by ranking, and a generative model where a transformer is
used as a decoder to produce responses token-by-token.
The key difference with these approaches is that we apply self-attention at the discourse level,
attending over the sequence of dialogue turns rather than the sequence of tokens in a single turn.

#### Topic disentanglement in task-oriented dialogue

Recent work has attempted to produce neural architectures for dialogue policies which can handle interleaved discourse segments in a single conversation.
Vlasov et al. [Vlasov et al., 2018] introduced the Recurrent Embedding Dialogue Policy (REDP) architecture.
The ablation study in this work highlighted that the improved performance of REDP is due to an attention
mechanism over the dialogue history and a copy mechanism to recover from unexpected user input.
This modification to the standard RNN structure enables the dialogue policy to ‘skip’ specific turns in the dialogue history and produce an encoder state which is identical before and after the unexpected input.
Sahay et al. [Sahay et al., 2019] develop this line of investigation further by studying the effectiveness of different attention mechanisms for learning this masking behaviour.

In this work we do not augment the basic RNN architecture but rather replace it with a transformer.
By default, an RNN processes every item in a sequence to calculate an encoding.
REDP’s modifications were motivated by the fact that not all dialogue history is relevant.
Taking this line of reasoning further, we can use self-attention in place of an RNN, so there is no
a priori assumption that the whole sequence is relevant, but rather that the dialogue policy should select which historical turns are relevant for choosing a response.

## III Transformer as a dialogue policy

We propose the Transformer Embedding Dialogue (TED) policy, which greatly simplifies the architecture of the REDP.
Similar to the REDP, we do not use a classifier to select a system action.
Instead, we jointly train embeddings for the dialogue state and each of the system actions by maximizing a similarity function between them.
At inference time, the current state of the dialogue is compared to all possible system actions, and the one with the highest similarity is selected.
A similar approach is taken by [Bordes et al., 2016, Mehri et al., 2019, Henderson et al., 2019] in training retrieval models for task-oriented dialogue.

Two time steps (i.e. dialogue turns) of the TED policy are illustrated in Figure 1. A step consists of several key parts.

#### Featurization

Firstly, the policy featurizes the user input, system actions and slots.

The TED policy can be used in an end-to-end or in a modular fashion.
The modular approach is similar to that taken in POMDP-based dialogue policies [Williams and Young, 2007] or Hybrid Code Networks [Williams et al., 2017, Bocklisch et al., 2017].
An external natural language understanding system is used and the user input
is featurized as a binary vector indicating the recognized intent and the detected entities.
The dialogue policy predicts an action from a fixed list of system actions. System actions are featurized as binary vectors representing the action name, following the REDP approach
explained in detail in [Vlasov et al., 2018].

By end-to-end we mean that there is no supervision beyond the sequence of utterances.
That is, there are no gold labels for the NLU output or the system action names.
The end-to-end TED policy is still a retrieval model and does not generate new responses.
In the end-to-end setup, user and system utterances are encoded as bag-of-words vectors.

Slots are always featurized as binary vectors, indicating their presence, absence, or that the value is not important to the user, at each step of the dialogue.
We use a simple slot tracking method, overwriting each slot with the most recently specified value.

#### Transformer

The input to the transformer is the sequence of user inputs and system actions.
Therefore, we leverage the self-attention mechanism present in the transformer to access different parts of dialogue history dynamically at each dialogue turn.
The relevance of previous dialogue turns is learned from data and calculated anew at each
turn in the dialogue.
Crucially, this allows the dialogue policy to take a user utterance into account at one turn
but ignore it completely at another.

#### Similarity

The transformer output adialoguesubscript𝑎dialoguea_{\text{dialogue}} and system actions yactionsubscript𝑦actiony_{\text{action}} are embedded into a single semantic vector space hdialogue=E​(adialogue)subscriptℎdialogue𝐸subscript𝑎dialogueh_{\text{dialogue}}=E(a_{\text{dialogue}}), haction=E​(yaction)subscriptℎaction𝐸subscript𝑦actionh_{\text{action}}=E(y_{\text{action}}), where h∈I​R20ℎIsuperscriptR20h\in{\rm I\!R}^{20}. We use the dot-product loss [Wu et al., 2017, Henderson et al., 2019] to maximize the similarity S+=hdialogueT​haction+superscript𝑆superscriptsubscriptℎdialogue𝑇superscriptsubscriptℎactionS^{+}=h_{\text{dialogue}}^{T}h_{\text{action}}^{+} with the target label yaction+superscriptsubscript𝑦actiony_{\text{action}}^{+} and minimize similarities S−=hdialogueT​haction−superscript𝑆superscriptsubscriptℎdialogue𝑇superscriptsubscriptℎactionS^{-}=h_{\text{dialogue}}^{T}h_{\text{action}}^{-} with negative samples yaction−superscriptsubscript𝑦actiony_{\text{action}}^{-}. Thus, the loss function for one dialogue reads

Ldialogue=−⟨S+−log⁡(eS++∑Ω−eS−)⟩,subscript𝐿dialoguedelimited-⟨⟩superscript𝑆superscript𝑒superscript𝑆subscriptsuperscriptΩsuperscript𝑒superscript𝑆L_{\text{dialogue}}=-\biggl{\langle}S^{+}-\log\biggl{(}e^{S^{+}}+\sum_{\Omega^{-}}e^{S^{-}}\biggr{)}\biggr{\rangle},

(1)

where the sum is taken over the set of negative samples Ω−superscriptΩ\Omega^{-} and the average ⟨.⟩\langle.\rangle is taken over time steps inside one dialogue.

The global loss is an average of all loss functions from all dialogues.

At inference time, the dot-product similarity serves as a ranker for the next utterance retrieval problem.

During modular training, we use a balanced batching strategy to mitigate class imbalance, as some system actions are far more frequent than others.

## IV Experiments

The aim of our experiments is to compare the performance of the transformer against that of an LSTM on multi-turn conversations.
Specifically, we want to test the TED policy on the task of picking out relevant turns in the dialogue history for next action prediction.
Therefore, we need a conversational dataset for which system actions depend on the dialogue history across several turns.
This requirement precludes question-answering datasets such as WikiQA Yang et al. [2015] as candidates for evaluation.

In addition, system actions need to be labeled to evaluate next action retrieval accuracy.
Note, that metrics such as Recall@k Lowe et al. [2016] could be used on unlabeled data, but since typical dialogues contain many generic responses, such as “yes”, that are correct in a large number of situations, the meaningfulness of Recall@k is questionable.
We therefore exclude unlabeled dialogue corpora such as the Ubuntu Dialogue Corpus Lowe et al. [2016] or MetalWOZ Schulz et al. [2019] from our experiments.

To our knowledge, the only publicly available dialogue datasets that might satisfy both our criteria are the REDP dataset Vlasov et al. [2018], MultiWOZ Budzianowski et al. [2018], Eric et al. [2019] and Google Taskmaster-1 Byrne et al. [2019].
For the latter, we would have to extract action labels from the entity annotations, which is not always possible.

Two different models serve as baseline in our experiments.
First, the REDP model by Vlasov et al. [2018], which was specifically designed to handle long-range history dependencies, but is LSTM-based.
Second, another LSTM-based policy that is identical to TED, except that the transformer was replaced by an LSTM.

We use the first (REDP) baseline for the experiments on the [Vlasov et al., 2018] dataset, as this baseline is stronger when long-range dependencies are in play.
For the MultiWOZ experiments, we only compare to the simple LSTM policy, since the MultiWOZ dataset is nearly history independent as we demonstrate here.

All experiments are available online at https://github.com/RasaHQ/TED-paper.

### IV.1 Conversations containing sub-dialogues

We first evaluate experiments on the dataset of Vlasov et al. [2018].
This dataset was specifically designed to test the ability of a dialogue policy to handle non-cooperative or unexpected user input. It consists of task-oriented dialogues in hotel and restaurant reservation domains containing cooperative (user provides necessary information related to the task) and non-cooperative (user asks a question unrelated to the task or makes chit-chat) dialogue turns.
One of the properties of this dataset is that the system repeats the previously asked question after any non-cooperative user behavior.
This dataset is also used in [Sahay et al., 2019] to compare the performance of different attention mechanisms.

Figure 2 shows the performance of different dialogue policies on the held-out test dialogues as a function of the amount of conversations used to train the model. The TED policy performs on par with REDP without any specifically designed architecture to solve the task and significantly outperforms a simple LSTM-based policy.
In the extreme low-data regime, the TED policy is outperformed by REDP. It should be noted that REDP relies heavily on its copy mechanism to predict the previously asked question after a non-cooperative digression.
However, the TED policy, being both simpler and more general, achieves similar performance without relying on dialogue properties like repeating a question.
Moreover, due to the transformer architecture, the TED policy trains faster than REDP and requires fewer training epochs to achieve the same accuracy.

Figure 3 visualizes the attention weights of the TED policy on an example dialogue.
This example dialogue contains several chit-chat utterances in a row in the middle of the conversation.
The Figure shows that the series of chit-chat interactions is completely ignored by the self-attention mechanism when task completion is attempted (i.e. further required questions are asked).
Note, that the learned weights are sparse, even though the TED policy does not use a sparse attention architecture.
Importantly, the TED policy chooses key dialogue steps from the history that are relevant for the current prediction and ignores uninformative history.
Here, we visualize only one conversation, but the result is the same for an arbitrary number of chit-chat dialogue turns.

### IV.2 Comparing the end-to-end and modular approaches on MultiWOZ

Having demonstrated that the light-weight TED policy performs at least on par with the specialized REDP and significantly outperforms a basic LSTM policy when evaluated on conversations that contain long-range history dependencies, we now compare TED to an LSTM policy on the MultiWOZ 2.1 dataset.
In contrast to the previous Section, the LSTM policy of the present Section is an architecture identical to TED, but with the transformer replaced by an LSTM cell.

We chose MultiWOZ for this experiment because it concerns multi-turn conversations and provides system action labels.
Unfortunately, we discovered that it does not contain many long-range dependencies, as we shall demonstrate later in this Section.
Therefore, neither TED nor the REDP have any conceptual advantages over an LSTM.
Subsequently we show that the TED policy performs on par with an LSTM on this commonly used benchmark dataset.

MultiWOZ 2.1 is a dataset of 10438 human-human dialogues for a Wizard-of-Oz task in seven different domains: hotel, restaurant, train, taxi, attraction, hospital, and police.
In particular, the dialogues are between a user and a clerk (wizard).
The user asks for information and the wizard, who has access to a knowledge base about all the possible things that the user may ask for, provides that information or executes a booking.
The dialogues are annotated with labels for the wizard’s actions, as well as the wizard’s knowledge about the user’s goal after each user turn.

For our experiments, we split the MultiWOZ 2.1 dataset into a training and a test set of 724972497249 and 181218121812 dialogues, respectively.
Unfortunately, we had to neglect 137713771377 dialogues altogether, since their annotations are incomplete.

#### End-to-end training.

As a first experiment on MultiWOZ 2.1 we study an end-to-end retrieval setup, where the user utterance is used directly as input to the TED policy, which then has to retrieve the correct response from a predefined list (extracted from MultiWOZ).

The wizard’s behaviour depends on the result of queries to the knowledge base. For example, if only a single venue is returned, the wizard will probably refer to it.
We marginalize this knowledge base dependence by (i) delexicalizing all user and wizard utterances [Mrkšić et al., 2016], and (ii) introducing status slots that indicate whether a venue is available, not available, already booked, or unique (i.e. the wizard is going to recommend or book a particular venue in the next turn).
These slots are featurized as a 1-of-K binary vector.

To compute the accuracy and F1 scores of the TED policy’s predictions, we assign the action labels (e.g. request_restaurant) that are provided by the MultiWOZ dataset to the output utterances, and compare them to the correct labels.
If multiple labels are present, we concatenate them in alphabetic order to a single label.

Table 1 shows the resulting F1 scores and accuracies on the hold-out test set.
The discrepancy between the F1 score and the accuracy stems from the fact that some labels, s.a. bye_general, occur very frequently (4759 times) compared to most other labels, s.a. recommend_restaurant_select_restau rant, which only occurs 11 times.

The fact that accuracy and F1 scores are generally low compared to 1.0 stems from a deeper issue with the MultiWOZ dialog dataset.
Specifically, because more than one particular behaviour of the wizard would be considered ’correct’ in most situations, the MultiWOZ dataset is unsuitable for supervised learning of dialogue policies.
Put differently, some of the wizards’ actions in MultiWOZ are not deterministic, but probabilistic.
For example, it cannot be learned when the wizard should ask if the user needs anything else, since this is the personal preference of the people who take the wizard’s role.
We elaborate on this and several other issues of the MultiWOZ dataset in [Mosig et al., 2020].

model
N𝑁N
accuracy
F1 score

TED end-to-end
10
0.64
0.28

2
0.62
0.24

TED modular
10
0.73
0.63

2
0.69
0.55

LSTM end-to-end
10
0.51
0.23

2
0.57
0.24

LSTM modular
10
0.68
0.60

2
0.61
0.54

#### Modular training.

We now repeat the above experiment, using the same subset of MultiWOZ dialogues, but now adopting the modular approach.
We simulate an external natural language understanding pipeline and provide gold user intents and entities to the TED policy instead of the original user utterances.
We extract the intents from the changes in the Wizard’s belief state.
This belief state is provided by the MultiWOZ dataset in the form of a set of slots (e.g. restaurant_area, hotel_name, etc.) that get updated after each user turn.
A typical user intent is thus inform{"restaurant_area": "south"}.
The user does not always provide new information, however, so the intent might be simply inform (without any entities).
If the last user intent of the dialogue was uninformative in this way, we assume it is a farewell and thus annotate it as bye.

Using the modular approach instead of end-to-end learning roughly doubles the F1 score and also increases accuracy slightly, as can be seen in Table 1.
This is not surprising since the modular approach receives additional supervision.

Although the scores suggest that the modular TED policy performs better than the end-to-end TED policy, the kinds of mistakes made are similar.
We demonstrate this with one example dialog from our test set, named SNG0253, that is displayed in Figure 4.

The second column of Figure 4 shows the end-to-end predictions.
The two predicted responses are both sensible, i.e. the replies could have come from a human.
Nevertheless, both results are marked as wrong, because according to the gold dialogue (first column), the first response should only have included its second sentence (request_train, but not inform_train).
For the fourth turn, however, it is the other way around: According to the target dialogue the response should have included additional information about the train (inform_train_request_train), whereas the predicted dialogue only asked for more information (request_train).

The third column shows that the modular TED policy makes the same kinds of mistakes: instead of predicting only request_train, it predicts to take both actions, inform_train and request_train in the second turn.
In the final turn, instead of request_train, the modular TED policy predicts reqmore_general, which means that the wizard asks if the user requires anything else.
This reply is perfectly sensible and does in fact occur in similar dialogues of the training set (see, e.g., Dialogue PMUL1883).
Thus, the correct behaviour doesn’t exist and it is impossible to achieve high scores, as reflected by the test scores of Table 1.

To the best of our knowledge, the state of the art F1 scores on next action retrieval with MultiWOZ are given by [Mehri et al., 2019] and [Mehri and Eskenazi, 2019] with 0.640.640.64 and 0.720.720.72, respectively. However, these numbers are not directly comparable to ours: we retrieve actions out of all 561285612856128 possible responses and compare the label of the retrieved response with the label of correct response, while they retrieve out of 202020 negative samples and compare text responses directly.

#### History independence.

As Table 1 shows, taking into account only the last two turns (i.e. the current user utterance or intent, and one system action before that), instead of the last 10 turns, the accuracy and F1 scores decrease by no more than 0.040.040.04 for end-to-end and no more than 0.080.080.08 for the modular architecture.
For the end-to-end LSTM architecture that we discuss in the next paragraph, the performance even improves when less history is taken into account.
Thus, MultiWOZ appears to depend only weakly on the dialogue history, and therefore we cannot evaluate how well the TED policy handles dialogue complexity.

#### Transformer vs LSTM.

As a final experiment, we replace the transformer in the TED architecture by an LSTM and run the same experiments as before.
The results are shown in Table 1.

The F1 scores of the LSTM and transformer versions differ by no more than 0.050.050.05, which is to be expected, since in MultiWOZ the vast majority of information is carried by the most recent turn.

The LSTM version lacks the accuracy of the transformer version, however.
Specifically, the accuracy scores for end-to-end training are up to 0.130.130.13 points lower for the LSTM.
It is difficult to assert the reason for this discrepancy due to the problems of ambiguity that we have identified earlier in this Section.

## V Conclusions

We introduce the transformer embedding dialogue (TED) policy in which a transformer’s self-attention mechanism operates over the sequence of dialogue turns.
We argue that this is a more appropriate architecture than an RNN due to the presence of interleaved topics in real-life conversations.
We show that the TED policy can be applied to the MultiWOZ dataset in both a modular and end-to-end fashion, although we also find that this dataset is not ideal for supervised learning of dialogue policies, due to a lack of history dependence and a dependence on individual crowd-worker preferences.
We also perform experiments on a task-oriented dataset specifically created to test the ability to recover from non-cooperative user behaviour.
The TED policy outperforms the baseline LSTM approach and performs on par with REDP, despite TED being faster, simpler, and more general. We demonstrate that learned attention weights are easily interpretable and reflect dialogue logic. At every dialogue turn, a transformer picks which previous turns to take into account for current prediction, selectively ignoring or attending to different turns of the dialogue history.

### Acknowledgments

We would like to thank the Rasa team and Rasa community for feedback and support. Special thanks to Elise Boyd for supporting us with the illustrations.

## References

- Vaswani et al. [2017]

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

Attention is all you need.

In Advances in neural information processing systems, pages
5998–6008, 2017.

- Vlasov et al. [2018]

Vladimir Vlasov, Akela Drissner-Schmid, and Alan Nichol.

Few-shot generalization across dialogue tasks.

arXiv preprint arXiv:1811.11707, 2018.

- Grosz and Sidner [1986]

Barbara J Grosz and Candace L Sidner.

Attention, intentions, and the structure of discourse.

Computational linguistics, 12(3):175–204,
1986.

- Bohus and Rudnicky [2009]

Dan Bohus and Alexander I Rudnicky.

The ravenclaw dialog management framework: Architecture and systems.

Computer Speech & Language, 23(3):332–361, 2009.

- Sordoni et al. [2015]

Alessandro Sordoni, Yoshua Bengio, Hossein Vahabi, Christina Lioma, Jakob
Grue Simonsen, and Jian-Yun Nie.

A hierarchical recurrent encoder-decoder for generative context-aware
query suggestion.

In Proceedings of the 24th ACM International on Conference on
Information and Knowledge Management, pages 553–562. ACM, 2015.

- Serban et al. [2016]

Iulian V Serban, Alessandro Sordoni, Yoshua Bengio, Aaron Courville, and Joelle
Pineau.

Building end-to-end dialogue systems using generative hierarchical
neural network models.

In Thirtieth AAAI Conference on Artificial Intelligence, 2016.

- Williams et al. [2017]

Jason D Williams, Kavosh Asadi, and Geoffrey Zweig.

Hybrid code networks: practical and efficient end-to-end dialog
control with supervised and reinforcement learning.

arXiv preprint arXiv:1702.03274, 2017.

- Sahay et al. [2019]

Saurav Sahay, Shachi H. Kumar, Eda Okur, Haroon Syed, and Lama Nachman.

Modeling intent, dialog policies and response adaptation for
goal-oriented interactions.

In Proceedings of the 23rd Workshop on the Semantics and
Pragmatics of Dialogue - Full Papers, London, United Kingdom, September
2019. SEMDIAL.

URL http://semdial.org/anthology/Z19-Sahay_semdial_0019.pdf.

- Dai et al. [2019]

Zihang Dai, Zhilin Yang, Yiming Yang, William W Cohen, Jaime Carbonell, Quoc V
Le, and Ruslan Salakhutdinov.

Transformer-xl: Attentive language models beyond a fixed-length
context.

arXiv preprint arXiv:1901.02860, 2019.

- Radford et al. [2019]

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever.

Language models are unsupervised multitask learners.

OpenAI Blog, 1(8), 2019.

- Wang et al. [2018]

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R
Bowman.

Glue: A multi-task benchmark and analysis platform for natural
language understanding.

arXiv preprint arXiv:1804.07461, 2018.

- Devlin et al. [2018]

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.

Bert: Pre-training of deep bidirectional transformers for language
understanding.

arXiv preprint arXiv:1810.04805, 2018.

- Hsieh et al. [2019]

Yu-Lun Hsieh, Minhao Cheng, Da-Cheng Juan, Wei Wei, Wen-Lian Hsu, and Cho-Jui
Hsieh.

On the robustness of self-attentive models.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, pages 1520–1529, 2019.

- Henderson et al. [2019]

Matthew Henderson, Ivan Vulić, Daniela Gerz, Iñigo Casanueva, Paweł
Budzianowski, Sam Coope, Georgios Spithourakis, Tsung-Hsien Wen, Nikola
Mrkšić, and Pei-Hao Su.

Training neural response selection for task-oriented dialogue
systems.

arXiv preprint arXiv:1906.01543, 2019.

- Dinan et al. [2018]

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason
Weston.

Wizard of wikipedia: Knowledge-powered conversational agents.

arXiv preprint arXiv:1811.01241, 2018.

- Bordes et al. [2016]

Antoine Bordes, Y-Lan Boureau, and Jason Weston.

Learning end-to-end goal-oriented dialog.

arXiv preprint arXiv:1605.07683, 2016.

- Mehri et al. [2019]

Shikib Mehri, Evgeniia Razumovsakaia, Tiancheng Zhao, and Maxine Eskenazi.

Pretraining methods for dialog context representation learning.

arXiv preprint arXiv:1906.00414, 2019.

- Williams and Young [2007]

Jason D Williams and Steve Young.

Partially observable markov decision processes for spoken dialog
systems.

Computer Speech & Language, 21(2):393–422, 2007.

- Bocklisch et al. [2017]

Tom Bocklisch, Joey Faulkner, Nick Pawlowski, and Alan Nichol.

Rasa: Open source language understanding and dialogue management.

arXiv preprint arXiv:1712.05181, 2017.

- Wu et al. [2017]

Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, and Jason
Weston.

Starspace: Embed all the things!

arXiv preprint arXiv:1709.03856, 2017.

- Yang et al. [2015]

Yi Yang, Wen-tau Yih, and Christopher Meek.

WikiQA: A Challenge Dataset for Open-Domain Question
Answering.

In Proceedings of the 2015 Conference on Empirical
Methods in Natural Language Processing, pages 2013–2018. Association
for Computational Linguistics, 2015.

doi: 10.18653/v1/D15-1237.

- Lowe et al. [2016]

Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau.

The Ubuntu Dialogue Corpus: A Large Dataset for Research
in Unstructured Multi-Turn Dialogue Systems.

arXiv preprint arXiv:1506.08909, 2016.

URL http://arxiv.org/abs/1506.08909.

- Schulz et al. [2019]

Hannes Schulz, Adam Atkinson, Mahmoud Adada, Kaheer Suleman, and Shikhar
Sharma.

MetaLWOz, 2019.

URL https://www.microsoft.com/en-us/research/project/metalwoz/.

- Budzianowski et al. [2018]

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Inigo Casanueva,
Stefan Ultes, Osman Ramadan, and Milica Gašić.

Multiwoz-a large-scale multi-domain wizard-of-oz dataset for
task-oriented dialogue modelling.

arXiv preprint arXiv:1810.00278, 2018.

- Eric et al. [2019]

Mihail Eric, Rahul Goel, Shachi Paul, Abhishek Sethi, Sanchit Agarwal, Shuyag
Gao, and Dilek Hakkani-Tur.

Multiwoz 2.1: Multi-domain dialogue state corrections and state
tracking baselines.

arXiv preprint arXiv:1907.01669, 2019.

- Byrne et al. [2019]

Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind Neelakantan,
Daniel Duckworth, Semih Yavuz, Ben Goodrich, Amit Dubey, Kyu-Young Kim, and
Andy Cedilnik.

Taskmaster-1:Toward a realistic and diverse dialog dataset.

In 2019 Conference on Empirical Methods in Natural Language
Processing and 9th International Joint Conference on Natural Language
Processing, 2019.

URL
https://storage.googleapis.com/dialog-data-corpus/TASKMASTER-1-2019/landing_page.html.

- Mrkšić et al. [2016]

Nikola Mrkšić, Diarmuid O Séaghdha, Tsung-Hsien Wen, Blaise
Thomson, and Steve Young.

Neural belief tracker: Data-driven dialogue state tracking.

arXiv preprint arXiv:1606.03777, 2016.

- Mosig et al. [2020]

Johannes EM Mosig, Vladimir Vlasov, and Alan Nichol.

Where is the context?–a critique of recent dialogue datasets.

arXiv preprint arXiv:2004.10473, 2020.

- Mehri and Eskenazi [2019]

Shikib Mehri and Maxine Eskenazi.

Multi-granularity representations of dialog.

arXiv preprint arXiv:1908.09890, 2019.

Generated on Fri Mar 8 02:23:00 2024 by LaTeXML
