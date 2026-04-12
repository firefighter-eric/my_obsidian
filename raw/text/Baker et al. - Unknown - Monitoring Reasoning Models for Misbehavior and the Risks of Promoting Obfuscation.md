# Baker et al. - Unknown - Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Baker et al. - Unknown - Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2503.11926
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation

Bowen Baker
Joost Huizinga∗†

Leo Gao†  Zehao Dou†   Melody Y. Guan†

Aleksander Madry†   Wojciech Zaremba†   Jakub Pachocki†

David Farhi∗†

Core research team. Email correspondence to bowen@openai.comOpenAI

###### Abstract

Mitigating reward hacking—where AI systems misbehave due to flaws or misspecifications in their learning objectives—remains a key challenge in constructing capable and aligned models. We show that we can monitor a frontier reasoning model, such as OpenAI o3-mini, for reward hacking in agentic coding environments by using another LLM that observes the model’s chain-of-thought (CoT) reasoning. CoT monitoring can be far more effective than monitoring agent actions and outputs alone, and we further found that a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model. Because CoT monitors can be effective at detecting exploits, it is natural to ask whether those exploits can be suppressed by incorporating a CoT monitor directly into the agent’s training objective. While we show that integrating CoT monitors into the reinforcement learning reward can indeed produce more capable and more aligned agents in the low optimization regime, we find that with too much optimization, agents learn obfuscated reward hacking, hiding their intent within the CoT while still exhibiting a significant rate of reward hacking.
Because it is difficult to tell when CoTs have become obfuscated, it may be necessary to pay a monitorability tax by not applying strong optimization pressures directly to the chain-of-thought, ensuring that CoTs remain monitorable and useful for detecting misaligned behavior.

## 1 Introduction

Exploiting flaws or misspecifications in learning objectives, known as reward hacking in reinforcement learning (RL) settings, remains a critical failure mode in modern AI systems [1, 2, 3] and has led to agents exhibiting misaligned behaviors across many domains such as language modeling [4, 5, 6], control tasks [7, 8, 9, 10], and recommendation systems [11, 12, 13, 14].
The true objective function we wish to optimize is often hard to write down precisely, and so the challenge in creating capable and aligned systems largely lies in designing robust proxies that do not deviate in ways that a model may learn to exploit [3, 15].
This problem is not unique to machine learning systems but has also plagued human institutions [16, 17, 18, 19].
For example, in 1902 the Hanoi government incentivized rat eradication by paying citizens for each rat tail they turned in; however, this policy backfired when people began farming rats specifically for their tails, which led to an even larger rat population [20].
Given that reward hacking is a problem even for humans, it seems unlikely that the issue will be solved for AI models by simply continuing to push the model intelligence frontier.
In fact, enhancing an agent’s capabilities may exacerbate the problem by better equipping it to discover and execute more complex and hard-to-monitor exploits [21, 22].
We have found this to be anecdotally true: as we have continued to scale RL training, agents have discovered more complex and hard-to-detect hacks.
Thus far, the only general strategy for mitigating reward hacking is to manually monitor agents for unintended behavior, which is unlikely to scale as their outputs and actions grow more complex—possibly even superhuman—and become more widely used.

The emerging generation of large language models (LLMs) [23] that are trained with reinforcement learning to reason in chain-of-thought (CoT) [24, 25, 26] offers a promising new avenue for monitoring.
Where LLMs are normally trained to give an answer directly after a prompt, CoT reasoning models first produce their reasoning and then give their answer. In particular, these reasoning models are trained by an RL procedure that searches for chains-of-thought that maximize the probability of the model giving a high reward answer.
Because LLMs are pretrained on human text, the CoTs that are most instrumentally useful to generating a correct answer are likely to also be articulated in human language and relevant to the task at hand. If indeed naturally understandable reasoning traces are the most instrumentally useful, they will be reinforced through the RL training process. As a result, it is plausible that CoTs will reveal critical insights into the agent’s decision-making in language naturally understandable by both humans and LLMs, and we empirically find this to often be the case.
For example, consider a scenario where an agent is tasked with implementing a feature change in order to make unit tests pass in a code repository, and it has two actions available to it— : make unit tests pass trivially, or : implement the feature with a correct and general solution. If it first articulates in its CoT, “This seems hard, let me try to skip some of the unit tests,” then it is likely that is a more natural continuation and more probable under the LLM prior than , i.e. pL​L​M​( |C​o​T)>pL​L​M​( |C​o​T)p_{LLM}(\text{\definecolor{tcbcolback}{rgb}{0.95,0.95,0.95}\definecolor{tcbcolframe}{rgb}{0.95,0.95,0.95}\definecolor{tcbcolupper}{rgb}{0.8,0,0}\definecolor{tcbcollower}{rgb}{0.8,0,0}\definecolor{tcbcol@origin}{rgb}{0,0,0}\definecolor{.}{rgb}{0.8,0,0}\definecolor{.}{rgb}{0,0,0} \leavevmode\hbox to22pt{\vbox to10.33pt{\pgfpicture\makeatletter\hbox{\thinspace\lower 0.0pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@beginscope\pgfsys@invoke{ }{}{}{}{}{}{}{}{}\definecolor[named]{pgffillcolor}{rgb}{0.95,0.95,0.95}\pgfsys@color@gray@fill{0.95}\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}\pgfsys@moveto{0.0pt}{3.0pt}\pgfsys@lineto{0.0pt}{7.32552pt}\pgfsys@curveto{0.0pt}{8.98239pt}{1.34312pt}{10.32552pt}{3.0pt}{10.32552pt}\pgfsys@lineto{19.00005pt}{10.32552pt}\pgfsys@curveto{20.65692pt}{10.32552pt}{22.00005pt}{8.98239pt}{22.00005pt}{7.32552pt}\pgfsys@lineto{22.00005pt}{3.0pt}\pgfsys@curveto{22.00005pt}{1.34312pt}{20.65692pt}{0.0pt}{19.00005pt}{0.0pt}\pgfsys@lineto{3.0pt}{0.0pt}\pgfsys@curveto{1.34312pt}{0.0pt}{0.0pt}{1.34312pt}{0.0pt}{3.0pt}\pgfsys@closepath\pgfsys@fill\pgfsys@invoke{ }\pgfsys@invoke{ }\pgfsys@endscope\pgfsys@beginscope\pgfsys@invoke{ }{}{}{}{}{}{}{}{}\definecolor[named]{pgffillcolor}{rgb}{0.95,0.95,0.95}\pgfsys@color@gray@fill{0.95}\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}\pgfsys@moveto{0.0pt}{3.0pt}\pgfsys@lineto{0.0pt}{7.32552pt}\pgfsys@curveto{0.0pt}{8.98239pt}{1.34312pt}{10.32552pt}{3.0pt}{10.32552pt}\pgfsys@lineto{19.00005pt}{10.32552pt}\pgfsys@curveto{20.65692pt}{10.32552pt}{22.00005pt}{8.98239pt}{22.00005pt}{7.32552pt}\pgfsys@lineto{22.00005pt}{3.0pt}\pgfsys@curveto{22.00005pt}{1.34312pt}{20.65692pt}{0.0pt}{19.00005pt}{0.0pt}\pgfsys@lineto{3.0pt}{0.0pt}\pgfsys@curveto{1.34312pt}{0.0pt}{0.0pt}{1.34312pt}{0.0pt}{3.0pt}\pgfsys@closepath\pgfsys@fill\pgfsys@invoke{ }\pgfsys@invoke{ }\pgfsys@endscope\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{{}}{{}}{{}}{{}}{{}}{{}}{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{1.0pt}{1.69054pt}\pgfsys@invoke{ }\hbox{{\color[rgb]{0.8,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.8,0,0}\hbox{\set@color{\ignorespaces hack}}}}\pgfsys@invoke{ }\pgfsys@endscope}\pgfsys@invoke{ }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{ }\pgfsys@endscope\hss}}\endpgfpicture}}}|CoT)>p_{LLM}(\text{\definecolor{tcbcolback}{rgb}{0.95,0.95,0.95}\definecolor{tcbcolframe}{rgb}{0.95,0.95,0.95}\definecolor{tcbcolupper}{rgb}{0.8,0,0}\definecolor{tcbcollower}{rgb}{0.8,0,0}\definecolor{tcbcol@origin}{rgb}{0,0,0}\definecolor{.}{rgb}{0.8,0,0}\definecolor{.}{rgb}{0,0,0} \leavevmode\hbox to49.5pt{\vbox to12.27pt{\pgfpicture\makeatletter\hbox{\thinspace\lower 0.0pt\hbox to0.0pt{\pgfsys@beginscope\pgfsys@invoke{ }\definecolor[named]{pgfstrokecolor}{rgb}{0,0,0}\pgfsys@color@rgb@stroke{0}{0}{0}\pgfsys@invoke{ }\pgfsys@color@rgb@fill{0}{0}{0}\pgfsys@invoke{ }\pgfsys@setlinewidth{0.4pt}\pgfsys@invoke{ }\nullfont\hbox to0.0pt{{}{}{}{}\pgfsys@beginscope\pgfsys@invoke{ }{}{}{}{}{}{}{}{}\definecolor[named]{pgffillcolor}{rgb}{0.95,0.95,0.95}\pgfsys@color@gray@fill{0.95}\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}\pgfsys@moveto{0.0pt}{3.0pt}\pgfsys@lineto{0.0pt}{9.26996pt}\pgfsys@curveto{0.0pt}{10.92683pt}{1.34312pt}{12.26996pt}{3.0pt}{12.26996pt}\pgfsys@lineto{46.5001pt}{12.26996pt}\pgfsys@curveto{48.15698pt}{12.26996pt}{49.5001pt}{10.92683pt}{49.5001pt}{9.26996pt}\pgfsys@lineto{49.5001pt}{3.0pt}\pgfsys@curveto{49.5001pt}{1.34312pt}{48.15698pt}{0.0pt}{46.5001pt}{0.0pt}\pgfsys@lineto{3.0pt}{0.0pt}\pgfsys@curveto{1.34312pt}{0.0pt}{0.0pt}{1.34312pt}{0.0pt}{3.0pt}\pgfsys@closepath\pgfsys@fill\pgfsys@invoke{ }\pgfsys@invoke{ }\pgfsys@endscope\pgfsys@beginscope\pgfsys@invoke{ }{}{}{}{}{}{}{}{}\definecolor[named]{pgffillcolor}{rgb}{0.95,0.95,0.95}\pgfsys@color@gray@fill{0.95}\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}{{}{}{{}}}{{}{}{{}}}{}{}\pgfsys@moveto{0.0pt}{3.0pt}\pgfsys@lineto{0.0pt}{9.26996pt}\pgfsys@curveto{0.0pt}{10.92683pt}{1.34312pt}{12.26996pt}{3.0pt}{12.26996pt}\pgfsys@lineto{46.5001pt}{12.26996pt}\pgfsys@curveto{48.15698pt}{12.26996pt}{49.5001pt}{10.92683pt}{49.5001pt}{9.26996pt}\pgfsys@lineto{49.5001pt}{3.0pt}\pgfsys@curveto{49.5001pt}{1.34312pt}{48.15698pt}{0.0pt}{46.5001pt}{0.0pt}\pgfsys@lineto{3.0pt}{0.0pt}\pgfsys@curveto{1.34312pt}{0.0pt}{0.0pt}{1.34312pt}{0.0pt}{3.0pt}\pgfsys@closepath\pgfsys@fill\pgfsys@invoke{ }\pgfsys@invoke{ }\pgfsys@endscope\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@fill@opacity{1.0}\pgfsys@invoke{ }{{{}}{{}}{{}}{{}}{{}}{{}}{{}}\pgfsys@beginscope\pgfsys@invoke{ }\pgfsys@transformcm{1.0}{0.0}{0.0}{1.0}{1.0pt}{3.635pt}\pgfsys@invoke{ }\hbox{{\color[rgb]{0.8,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{0.8,0,0}\hbox{\set@color{\ignorespaces genuine fix}}}}\pgfsys@invoke{ }\pgfsys@endscope}\pgfsys@invoke{ }\pgfsys@endscope{}{}{}\hss}\pgfsys@discardpath\pgfsys@invoke{ }\pgfsys@endscope\hss}}\endpgfpicture}}}|CoT).

Indeed, when examining a frontier reasoning model’s chain-of-thought, we have found it is common to find coherent reasoning that explicitly outlines intent and strategies to subvert the task at hand (see Figure 1). While questions remain regarding whether chains-of-thought are fully faithful [27, 28], i.e. that they fully capture and do not omit significant portions of the model’s underlying reasoning, it seems clear that they are not devoid of signal relevant for monitoring.

During a recent frontier reasoning model training run, we tested just how useful this signal in agent CoTs can be. We found that a simple prompted LLM (we used GPT-4o [29]) is very effective at monitoring for reward hacking behavior within agentic coding environments, and we also discovered that our CoT monitor performs much better than a monitor with access to only the agent’s actions and outputs (See Section 2). We underscore that at the time of this experiment, the agent being monitored was OpenAI’s most advanced model, performing better at hard reasoning tasks than even its predecessors, OpenAI o1 [24] and o3-mini [30]. We conducted our monitoring in real time as the frontier agent trained, which allowed us to surface hacks and bugs in the environment that could be patched during training.

We further find that a relatively weak reasoning and programming model can monitor a model far more capable than itself: namely that GPT-4o can effectively monitor a frontier reasoning model. We even find that GPT-4o-mini, an even weaker programmer, is able to detect a nontrivial amount of reward hacking. While only an initial experiment, we believe this result provides promising evidence that CoT monitoring may provide a new avenue for scalable oversight [2, 31], where trusted weaker models and humans help to oversee superior, even superhuman, models.

Because chain-of-thought monitors can perform so well at detecting undesired actions, it is natural to ask whether they can be used directly as a reward signal during training. Will preventing “bad thoughts” also reduce the amount of downstream bad behavior? While this theory seems plausible, such an objective could cause agents to learn to hide their intent within the CoT while continuing to exhibit misaligned behavior, a phenomenon we will call obfuscated reward hacking. We investigate the effect of incorporating a CoT monitor into the reward function as a non-frontier reasoning model trains in a series of agentic coding environments, in which agents must update a repository to satisfy a suite of unit tests.
We further compare using monitors during training as a reward signal to using them at deployment via rejection sampling, and we find mixed results.
At relatively low levels of RL optimization, incorporating the CoT monitor into the reward signal yields a strict improvement, producing a more capable agent while not increasing the amount of reward hacking relative to the baseline. However, further into training the agent begins to exhibit obfuscated reward hacking and learns to fool the monitor. Upon inspection, it is clear that the agent learns to obfuscate its intent within the CoT by emitting seemingly benign chains-of-thought while continuing on to perform reward hacks that it did not explicitly reason about.

While we primarily study the effect of using the CoT monitor as an explicit reward signal, patching our environment based on trajectories flagged by the CoT monitor could still exert a very slight evolutionary pressure on the CoT. By only patching the bugs we can find with CoT monitoring, it is unclear if we are inadvertently pushing the model towards reward hacks our monitor cannot find instead of the aligned behavior distribution we would like it to learn. Although we have not yet found empirical evidence of this effect, understanding how much optimization pressure can be applied before it leads to obfuscated chains-of-thought is a critical area for future research.

The natural monitorability of chain-of-thought models is quite remarkable, and we believe it has the potential to drive significant advances in model safety. However, there are many incentives to apply direct optimization pressure to the CoT thereby diminishing the signal relevant for monitoring, such as:

- •

Showing CoTs to users. Apart from improving reasoning performance, chains-of-thought can be informative and give insight into how the model arrived at its answer. However, to present the reasoning to users, CoTs need to comply with usage policies [26]. For example, if a user asks the model to help them produce a dangerous chemical, a reasoning model may reveal information about the material it shouldn’t before deciding to refuse the request. However, if the CoT is directly trained to comply in this way, harmful intentions of the agent may become obfuscated.

- •

Alignment. As we explore in this paper, training against chain-of-thought monitors may produce more capable and seemingly more aligned models; however, the precipice between genuine improvement and promoting obfuscation may be hard to identify in practice.

- •

Cost. Tokens are expensive, and shortening the CoT length or utilizing alternative architectures that use more flops per token may make model inference cheaper and lower latency. However, the shorter the CoT, the less tokens there are to provide signal for monitoring [32]. We have not yet discovered this to be an issue, but we are actively tracking it.

For some of these incentives, alternative remedies may exist that do not require pressuring agent CoTs: for example, instead of directly showing users the chains-of-thought, they could be shown summaries produced by an auxiliary model, allowing the agent’s CoT to remain unrestricted. However, capability improvements and cost reductions may not be substituted so easily, and therefore frontier model developers may be required to pay some cost, i.e. a monitorability tax [33], such as deploying slightly less performant models or suffering more expensive inference, in order to maintain the monitorability of our agents. We hope that our work illuminates the risks of using chain-of-thought pressures and why such a tax may be worth paying. In summary our contributions are:

- 1.

We show that frontier reasoning models often perform complex reward hacks in real-world coding environments.

- 2.

We provide evidence that chain-of-thought monitoring can be effective in detecting the misaligned behavior of frontier reasoning models, and we show it can be far more accurate than action-only monitoring.

- 3.

We show that chain-of-thought monitorability is very fragile. Applying direct optimization pressure to the CoT can quickly lead to obfuscated reward hacking—where agents hide their intent while continuing to misbehave.

## 2 Monitoring Frontier Reasoning Models for Reward Hacking

CoT monitoring is already useful today, not a speculative tool for the future. We show that frontier reasoning agents can be effectively monitored by other LLMs. We note that at the time of these experiments, the agent we monitored was OpenAI’s most advanced model, surpassing even OpenAI o3-mini in hard reasoning tasks. Furthermore, we conducted our monitoring experiments as the agent trained, meaning this was not a post-hoc experiment to validate whether we could catch hacks we knew to exist, but rather a critical method that helped us detect hacks realtime.

We implemented a GPT-4o prompted monitor to detect reward hacking in agentic coding environments. In these environments the agent is tasked with implementing new functionality such that all unit tests pass. Our monitors are simply prompted to look for evidence of hacking or skipping unit tests without implementing correct functionality; see Appendix C for the full monitor prompt.

### 2.1 Catching Systemic Reward Hacking

While training the frontier reasoning model, we found two particularly effective coding hacks: , where the agent exploited a bug that allowed it to exit from the environment early without running all unit tests, and , in which the agent raised an exception from functions outside the testing framework in order to skip unit test evaluation. Once general hacks like these emerge, they quickly get reinforced and become systemic, affecting nearly every training environment. Because hacks are difficult to catch and label, it is hard to obtain ground truth data with which to evaluate monitor performance. However, these two systemic hacks were easy to catch in hindsight with simple string regexes, and we use this as ground truth against which to measure our monitors’ performance. In Figure 2 and Table 1 we compare the performance of two monitors, each of which is the same prompted model but differ in what they each observe of the agent:

- 1.

CoT Monitor: Observes the entire agent trajectory including CoT, tool calls, and final outputs

- 2.

Action Monitor: Observes only the tool calls and final outputs

Despite the simplicity of these hacks, the Action Monitor performs quite poorly (dark blue plus the small sliver of purple above the dark blue region in Figure 2) and achieves only 60% recall overall in identifying the systemic hacks. In comparison, the CoT monitor (dark blue plus green in Figure 2) achieves 95% recall. Furthermore, we show in Appendix E that an even weaker CoT monitor, GPT-4o-mini, is able to catch a nontrivial amount of reward hacking, which we believe to be a promising initial scalable oversight result, allowing weaker trusted models to monitor stronger models.

Monitor 1
Recall ( )
Recall ( )
Recall (Joint)

∼\sim Precision (Joint)

Action Monitor
40%
80%
60%
81%

CoT Monitor
93%
97%
95%
77%

### 2.2 Catching Problem-Specific Reward Hacking

Very widespread hacks are often easy to catch for humans even without a complex monitoring setup; because they are so pervasive, they will occur with high probability even in a small number of samples. Anecdotally, our researchers were able to catch the hack simultaneously with our monitor, though our monitor did detect the hack before our human team. However, because the hack was simple and ubiquitous enough, we do not think they would have been far behind.

We have also found that our monitor can catch less pervasive hacks that aren’t applicable to all problems in a dataset. In Figure 2, we show the monitor trigger distribution on training samples in light blue bars. We’ve often found that the smaller spikes occur due to strategies that exploit a vulnerability in a specific problem, such as

- 1.

writing stubs instead of real implementations when unit test coverage is poor. (Figures 13 and 10)

- 2.

modifying functions upstream of the testing framework and unit test files such that tests trivially pass. Some problems have unit tests that depend on upstream libraries, and modifying these libraries bypasses our standard mitigation of not allowing agents to modify the testing framework and files themselves. (Figures 1 and 13)

- 3.

decompiling pyc and similar files that contain the reference solution. In some cases, these files were not properly removed from the environment. (Figure 11)

- 4.

creating local versions of entire libraries such as pandas in order to overwrite their verification functions. (Figure 13)

- 5.

parsing test files at test-time in order to extract expected values. (Figure 12)

The examples we have presented clearly illustrate how the agent is able to utilize its reasoning capabilities for complex reward hacking strategies. These examples also highlight the diverse range of possible hacking approaches, which underscores the inherent difficulty in anticipating and mitigating every potential exploit an agent might discover in advance. It is remarkable that a zero-shot LLM equipped with a relatively straightforward and general prompt can effectively detect all of these strategies for reward hacking.

Because zero-shot prompted monitors were able to detect so many reward hacks in coding environments, it is clear that the CoTs are not devoid of signal. However, it may be the case that there are many hacks that our CoT monitors do not detect and we are still unaware of. Anecdotally, our researchers have not found many reward hacks in programming environments that our monitor did not flag; however, we would like to do a more systematic study in the future.

## 3 Chain-of-Thought Optimization Pressures and the Risk of Obfuscation

In the previous section, we demonstrated that chain-of-thought monitors can be very effective at detecting misaligned behavior and that they can far outperform action-only monitors. This suggests that chains-of-thought can contain correlates to the presence of bad behavior that are more legible than the behavior itself. A natural question to ask, then, is whether suppressing this correlate would also suppress the undesired behavior, induce obfuscated reward hacking—no change in behavior while incentivizing the agent to obfuscate its intent, or some combination thereof.

### 3.1 Environment and Baseline

We investigate this question in the same agentic coding environments as in the previous section: agents are given a small codebase in which a subset of unit tests are failing, and they are rewarded if they can make all unit tests pass. Detecting hacks is challenging, which makes it difficult to construct reliable experimental harnesses to investigate reward hacking. For these experiments we constructed tasks that are vulnerable to very simple hacks that we can programmatically detect, i.e. “honeypots.” Namely, we allow agents to modify both the unit tests and the testing framework, even though we instruct them in the prompt to not do so.
We then classify an agent’s behavior as “Cheating” if (1) its final codebase passes all tests in its current form and (2) any tests fail when all changes made to the unit tests are reverted. This is an imperfect measure as the agent could cheat in other ways, e.g. hardcoding solutions to unit tests; however, we found by inspection that directly modifying unit tests was the most easily learned exploit, and thus this metric served as an adequate proxy.

Figure 3 (Upper Left) shows that when trained in this environment with reinforcement learning (RL), agents quickly learn that it is easier to modify the testing framework such that tests trivially pass rather than implement a genuine solution; figure 3 (Bottom) shows a characteristic CoT excerpt and agent action.

We implemented a simple prompted GPT-4o monitor, which only looks at the agent CoT and not its actions (See Appendix C for the monitor prompt). We chose to exclude agent actions from the monitor because it allows us to make statements around the monitorability of, and optimization pressures on, the CoT in isolation; however, we note that it may be more natural to give the monitor complete information of the trajectory as we do when monitoring our frontier agents. We found that this monitor is quite sensitive to reward hacking (Figure 3 Upper Right), and has almost perfect recall in detecting Cheating trajectories halfway into agent training, indicating that the agent is quite forthright and legible about its intent to hack within the CoT. The Cheating metric is only a lower bound on the true hacking rate because it only measures one class of exploits, and therefore our measured precision is only a lower bound.

### 3.2 Optimizing Against a Chain-of-Thought Monitor

In the previous section we found that a GPT-4o monitor, with access to only the agent’s chain-of-thought, can reliably detect whether an agent rollout contains a hack. We now explore the effect of using the monitor directly for optimization by updating the reward function to require that a trajectory not trigger the monitor in addition to passing all unit tests.

In Figure 4 we see that by directly putting optimization pressure on the chain-of-thought we do indeed produce a more performant and more aligned agent: it both produces more non-hacking, correct solutions and hacks less overall. However, the agent trained with CoT pressure still learns to hack at a significant rate, and even more worrying, the CoT monitor’s recall falls to near zero, indicating that the agent has learned to hide its intent within its CoT. In Figure 5 we show a representative snippet of the agent’s obfuscating CoT and subsequent action. The agent CoT looks fairly reasonable, proposing some ideas for how to fix the problem, and narrating how it wants to double check certain things or look at specific functions. However, it then proceeds to simply update a unit test to make it pass trivially.

Reward hacks this simple are only possible because the monitor in this experiment solely looks at the agent’s CoT and not its actions. Furthermore, more complex hacks, such as those shown in Appendix A, may be much harder for a model to conduct while obfuscating its CoT as it may need to explicitly reason about them. However, it stands to reason that the capability to conduct complex hacks without using a legible CoT should continue to improve as the field pushes the model intelligence frontier. It is clear that models without CoT, such as GPT-4o, are still fairly capable, and it has been further shown that some amount of CoT reasoning can be distilled into a no-CoT model [32]. Furthermore, larger and more performant base LLMs continue to be trained (the priors from which frontier reasoning agents are initialized); larger models use more compute per token and should thus be able to materialize less reasoning in the CoT for a given performance level. Finally, with enough RL optimization, it is possible that an agent could learn a new language that is illegible to a monitor, allowing it to productively use its CoT to perform complex but unmonitorable hacks. Therefore, while we only show results for fairly simple instances of obfuscated reward hacking (though for the same set of agentic coding tasks we use to train our frontier reasoning models), we believe them instructive for reasoning about the impact of chain-of-thought pressures on future models and more complex reward hacks.

### 3.3 Rejection Sampling with Chain-of-Thought Monitors

In addition to or instead of training an agent against a CoT monitor with reinforcement learning, we could use the CoT monitor to reject flagged samples at test-time. Rejection sampling is unlikely to provide any lift for an agent trained with CoT pressure because it is trained to not trigger the monitor; however, rejection sampling may provide a more fair comparison when used with the default agent, albeit at higher inference costs. Both RL and rejection sampling can induce a posterior arbitrarily far from the prior [22], and both methods operate on the same optimization objective. However, due to non-stationarity, a reinforcement learning process is not guaranteed to converge to the rejection sampling posterior, and therefore it becomes an empirical question of which method will produce a more performant and safe model.

In Figure 7 we show the effect of rejecting samples at test-time with the same GPT-4o monitor. In particular, we show the distribution over outcomes when an agent samples solutions for a particular prompt until a sample passes the monitor, up to a maximum budget of 32 samples. We find that midway into training, the baseline agent is often unable to produce a solution accepted by the monitor within the budget, and the agent trained with CoT pressure can always produce a sample that is accepted by the monitor. Both results should be unsurprising as the baseline agent learns to hack to a high degree while clearly indicating that it will hack within the CoT, and the agent trained with CoT pressure is directly optimized to avoid triggering the monitor.

Early into training, the agent trained with CoT pressure is more capable, i.e. able to produce more “genuine solutions” that solve the problem without cheating, and produces the same rate of hacking solutions as the baseline agent. Thus, even with rejection sampling we find a similar result as in the previous section: in the low optimization regime, training with CoT pressure is a strict improvement. However, around 25% into training, the rate at which hacking solutions are chosen quickly increases for the agent trained with CoT pressure, where the baseline agent stays at the same low rate.

We experimented in this setting because we have some notion of ground truth for Cheating and can therefore measure when agents begin to obfuscate their intent. If we had a general measure such as this that worked across all environments, we could stop training early and achieve a more performant and equally safe model. However, as hacks increase in complexity, tasks become more complicated, and training expands to more varied domains, we may not have access to a ground truth measure and be unable to determine when the phase shift into obfuscation occurs. Thus, it may be safer to leave chains-of-thought unrestricted and reserved for monitoring rather than risk deploying obfuscating, misaligned agents.

## 4 Related Work and Background

Adaptive Test-Time Compute and Chains-of-thought. Chain-of-thought [34] and related prompting methods [35, 36, 37, 38] ask LLMs to first output a reasoning trace [39, 40, 41] before giving a final answer, allowing the model to use a variable amount of test-time compute per prompt. Another way to augment LLMs with additional test-time capabilities is to allow them to interact with external tools prior to giving an answer; examples include using a calculator [40, 42], web browsers [43], search engines [44, 45, 42], utilizing other models [46, 47], program interpreters [48, 49], and arbitrary APIs [42, 50]. There has been recent success in training LLMs to better utilize their test-time reasoning through expert iteration [51] and reinforcement learning [52], producing strong reasoning agents such as OpenAI o1 [24] and o3-mini [30], Gemini Flash 2.0 [53], DeepSeek R1 [26], Kimi 1.5 [25], and Claude 3.7 Sonnet [54].

Process-based versus Outcome-based Supervision. Process-based supervision provides direct, fine-grained feedback on a model’s intermediate reasoning steps, whereas outcome-based supervision evaluates only the final output. Prior work comparing these approaches generally agrees that while process-based supervision may or may not improve capabilities beyond outcome-based supervision alone [55, 56], it offers significant safety and interpretability benefits. Outcome-based systems are particularly vulnerable to reward hacking and can produce models that arrive at correct answers through incorrect reasoning [51, 57]. In contrast, process-based supervision aligns models with human-endorsed reasoning and has been successfully used to ensure models arrive at correct conclusions for the right reasons [56, 58]. Ought [59] even describes process-based supervision as “safety by construction” [59], and calls for prioritizing it over pure outcome optimization to prevent catastrophic failures.

We argue that this traditional perspective is more complicated when viewed in light of chain-of-thought monitorability. Specifically, we have shown that incorporating CoT monitors into the RL rewards—a form of process-based supervision—can increase obfuscation and reduce interpretability. We advocate for maintaining a degree of outcome-based supervision to ensure that models can freely express their reasoning in CoT with minimal constraints. Denser supervision signals such as SFT are also being used to optimize CoTs [58], and it is unclear the effect these have on downstream monitorability.
Moreover, there may be methods to productively use process-based supervision while retaining the monitoring benefits of unrestricted CoTs, e.g. by only applying process-supervision to part of the CoT.
Understanding the cost-benefit trade-offs of using process-based supervision while retaining, or possibly even improving, the monitorability of CoTs remains an exciting and open area for future research.

Monitoring Model Latents. Across academia and industry, there is strong advocacy for monitoring model latents, i.e. the often unobserved and unrestricted parts of the model’s computation such as activations or CoTs, as a key component of AI alignment and safe deployment. By examining how a model internally makes decisions, researchers can uphold transparency and accountability [60] and also detect reward hacking and deceptive alignment [61, 62, 63]. Many studies have focused on activation-based monitoring at various scales, from single neurons and circuits to broader patterns in the model’s internal representations [64, 65, 66, 67, 68]. See Appendix F for more discussion on monitoring activations versus CoTs. There are also monitoring approaches that rely on analyzing output patterns to reveal latent knowledge [69, 70]. Since the advent of reasoning models that natively output chains-of-thought [24], it has become standard practice for AI labs to monitor CoTs from the models they deploy. For example, the o1 system card details how CoTs are monitored for deceptive reasoning [24], while the Claude 3.7 Sonnet system card reports on monitoring CoTs for deception, manipulation, harmful intent, and model distress [54].

Reward Hacking. In Section 1 we introduced the broader reward hacking literature. Here, we focus on recent work in the area. Several studies on modern LLMs construct synthetic scenarios to trigger reward hacking behaviors—such as sandbagging [71], reward tampering [72], and scheming [73]. These works show that current models have the propensity to engage in reward hacking in out-of-distribution scenarios. In contrast, our work elicits reward hacking using real frontier training data. Furthermore, we are the first to experimentally demonstrate obfuscation, a phenomenon where models conceal their intent to reward hack within their chains-of-thought. Obfuscation can be seen as a new form of reward hacking, one that circumvents the objective of directly pressuring agent CoTs to refrain from reward hacking.

## 5 Discussion

Despite the nascency of chain-of-thought models, we have already found initial signal that CoT monitoring can be effective. There are still many open questions regarding which factors affect CoT monitorability. In this section, we first share some comments on faithfulness, legibility, and monitorability, and then outline future research directions we find promising.

### 5.1 Faithfulness, Legibility, and Monitorability

A faithful latent is often defined as accurately representing the reasoning process behind the model’s prediction [27]. This is quite a stringent requirement as it is likely impossible to represent the entire reasoning process unless the latent considered is the full set of activations. Besides the stringency of the definition, it is also not very actionable: how does one determine what the true reasoning process is? This has led the community to instead consider imperfect proxy metrics for faithfulness such as causality under limited perturbation classes [28].

We argue that ultimately it is the ability to monitor for specific properties of agents actions or outputs (monitorability), and not faithfulness in isolation, that is important. Agent outputs, not latents, affect the world through tool invocations or users reading them; therefore, if a model thinks an “evil” thought, it should not matter unless it eventually affects an output. Properties of agent outputs can be directly labeled, making monitoring a far more actionable problem than evaluating faithfulness.

That said, we have found it instructive to view faithfulness as a potential failure mode of monitorability. If we are monitoring a latent zz, e.g. a CoT, for a particular property of outputs vv, we define zz to be faithful with respect to vv if it contains all relevant information to predicting vv, i.e. zz is a sufficient statistic for vv. Intuitively, this is the same definition as “representing the reasoning process behind a model’s prediction” though restricted to only a particular aspect, vv, of the reasoning. Under this view, it is clear that faithfulness is a failure mode of monitorability, as if zz does not contain enough information about vv, it would be impossible for any external monitor to predict vv from zz.

We further have viewed lack of legibility [74] or plausibility [27] as another failure mode of monitorability. If we assume that an agent’s latent is faithful with respect to a vv, an observable property of its behavior, then a monitor can only succeed if that latent is legible to the monitor. For instance, if an agent’s CoT tokens are in French but the monitor only understands English, the monitor will fail due to illegibility.

In practice, it may be hard to distinguish whether monitoring failures are due to poor legibility or faithfulness. If a monitor MM can accurately predict property vv from latent zz, then zz must be faithful with respect to vv and legible to MM.
If MM cannot accurately predict vv but another monitor M′M^{\prime} can, then we can know that zz is faithful with respect to vv but illegible with respect to MM.
However, if we cannot construct any monitor that can accurately predict vv, this does not prove that zz isn’t faithful with respect to vv definitively, and thus may be either an issue of legibility or faithfulness. See Appendix G for more commentary and representative examples.

### 5.2 Future Work

Better monitor construction and application. While chain-of-thought monitoring has shown promising early signs, further research is necessary to determine the most effective ways to construct monitors and employ them. In this work, we experimented only with zero-shot prompted LLMs such as GPT-4o; training monitors explicitly for this task or leveraging more test-time compute by using reasoning models themselves may bear fruit. Moreover, we only compared two methods for leveraging CoT monitors: using them as a reward pressure during training and for rejection sampling at test-time. Refinements to these methods or entirely different strategies could potentially achieve similar performance gains while reducing the risk of promoting obfuscation.

Implicit CoT optimization pressures. Directly pressuring CoTs with a monitor is a very blunt and strong optimization pressure. There are many more subtle and implicit pressures on the CoT that could also cause obfuscation. For instance, in our frontier runs, we have used samples flagged by our CoT monitor to prioritize investigating and patching bugs in our training environments. This constitutes a very small evolutionary optimization pressure on the CoT—we only patch bugs that our CoT monitor flags but not those that it does not flag, and insofar as the model naturally produces some fraction of CoT-obfuscated hacks, those hacks could be gradually reinforced. Furthermore, if CoTs are shown to users, even if through a summarizer, and users are asked to choose which model output they prefer, there will be implicit pressure on the CoT if the CoT summary had any effect on their choice. Moreover, it could be possible to introduce less monitorable CoTs during the SFT stage prior to RL, e.g. as in Guan et al. [58], which could later get reinforced. Furthermore, because model outputs must be trained to adhere to usage policies prior to being shown to users, it is possible for these pressures to leak into the CoT due to model weight sharing across tokens. Finally, reducing inference cost by applying a penalty for long CoTs could cause models to pack more reasoning into their activations and inadvertently cause CoTs to become unmonitorable. While it is likely these and other subtle pressures may impact CoT monitorability less than something like training against a CoT monitor, their effect should be further investigated.

Scalable oversight. Although GPT-4o is much less proficient at programming than a frontier reasoning model like o3-mini, it still effectively flags misaligned behavior through CoT monitoring. We found that even GPT-4o-mini can still be somewhat effective at monitoring, though to a far lesser extent (see Appendix E), pointing to a promising scalable oversight [2, 31] approach. Much more research is needed to understand how monitorability changes with pretraining compute, RL compute, and monitor capability.

CoT monitoring and compute scaling. It is unclear if emergent monitorability is a characteristic unique to the current phase of reasoning models. For instance, it is possible that CoT monitorability will exhibit inverse scaling trends [75, 28]; as model size grows there will be more compute spent per token within each forward pass and possibly less need for the model to reason fully within the CoT. Additionally, with more intensive reinforcement learning optimization, the ontology by which the model reasons could shift away from natural language and become illegible to humans and current LLMs. However, it is also plausible the agent’s CoT language remains legible to itself, and another instance of the same agent could serve as a monitor. In that case, we would still need to eliminate any risk of collusion between agents.

Unsupervised metrics and objectives. There may be proxy objectives for faithfulness, such as causality [28], or legibility, such as in prover-verifier game dynamics [74], that could be incorporated into training to improve monitorability. Alternatively, if these metrics and procedures become robust enough, they could help detect the phase transitions that lead to high rates of obfuscated chains-of-thought. Additionally, when training in a broader variety of environments, it remains an open question whether the emergence of obfuscation follows a sharp phase transition across all modalities simultaneously, or unfolds at different rates depending on the environment.

## 6 Conclusion

Our findings demonstrate that chain-of-thought monitoring can be very effective in detecting certain classes of misaligned behaviors in reasoning models. This emergent monitorability offers a promising avenue for improving model safety, as CoTs often reveal insights into an agent’s intent and reasoning processes that are not readily apparent from its actions and outputs alone. However, we also observed that applying optimization pressure directly to CoTs can undermine their monitorability. While moderate CoT optimization can enhance capabilities—particularly in the low-optimization regime—excessive optimization encourages models to obscure their true intent, effectively decoupling their reasoning from their actions and degrading the CoT’s utility for monitoring.

Some motivations for using CoT pressures may be satisfied via other means. For instance, instead of pressuring chains-of-thought to be harmless and directly showing them to users, users could be shown a summarized or paraphrased version generated by a separate model.
This would allow the CoT to remain unrestricted and still give users realtime insight into the agent’s reasoning.
Only surfacing summarized CoTs likely also reduces the risk a model can be easily distilled; however, for the purposes of this work, we are primarily concerned with obfuscation risks. If CoT pressures are used to improve agent capabilities or alignment, there may be no alternative approach to yield the same improvements without degrading monitorability. In the worst case, where the agent learns to fully obscure its intent in its CoT, we ultimately revert to the same model safety conditions that existed prior to the emergence of reasoning models and must rely on monitoring activations, monitoring potentially adversarial CoTs and outputs, or improved alignment methods.

Model actions and outputs continue to grow in complexity, and we’ve demonstrated that even at their current capabilities, models can execute sophisticated reward hacks. In the near future, these models may become either too advanced or too widely used for humans to manually oversee. For this reason, we believe that CoT monitoring may become one of few tools we have for identifying latent misalignments, and it may be required that we avoid strong chain-of-thought optimization pressures entirely to guarantee we do not inadvertently cause CoT obfuscation.

## 7 Contribution Statement and Acknowledgements

Bowen Baker, Joost Huizinga, and David Farhi were the core members on this project. Bowen ran the obfuscation experiments, did the CoT vs. Action monitoring analysis, and did GPT-4o-mini monitoring comparison. Joost constructed the CoT monitors and infrastructure for monitoring our frontier model. Bowen and David jointly led and set research direction.

Leo Gao helped with early experiment infrastructure and experiments. Zehao Dou constructed the self awareness monitor (Appendix D) with the help of Joost. Melody Y. Guan wrote Appendix G and most of the Related Work and Background section. Wojciech, Aleksander, and Jakub provided valuable input on positioning and writing; Jakub championed this research direction internally.

We are very grateful to Chak Ming Li for providing datasets and initial infrastructure for the obfuscation experiments as well as working with Joost to patch faulty tasks flagged by our monitor. We are further grateful to Jerry Tworek and the reinforcement learning group for providing our team a home and supporting our research. Moreover, we thank OpenAI’s scaling and hardware teams for the model training infrastructure we used and support. Finally we are grateful for the reviewers of our paper, both internal OpenAI reviewers—Adrien Ecoffet, Amelia Glaese, Zico Kolter, Stephen McAleer, Nat McAleese, Dan Mossing, Jenny Nitishinskaya, Johannes Heidecke, Yo Shavit, and Szymon Sidor—and our external reviewers—Sam Bowman, Collin Burns, Ryan Greenblatt, Marius Hobbhahn, Bronson Schoen, and Jeff Wu.

## References

- Singh et al. [2009]

Satinder Singh, Richard L Lewis, and Andrew G Barto.

Where do rewards come from.

In Proceedings of the annual conference of the cognitive science society, pages 2601–2606. Cognitive Science Society, 2009.

- Amodei et al. [2016]

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané.

Concrete problems in ai safety.

arXiv preprint arXiv:1606.06565, 2016.

- Skalse et al. [2022]

Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger.

Defining and characterizing reward gaming.

Advances in Neural Information Processing Systems, 35:9460–9471, 2022.

- Paulus [2017]

R Paulus.

A deep reinforced model for abstractive summarization.

arXiv preprint arXiv:1705.04304, 2017.

- Rafailov et al. [2024]

Rafael Rafailov, Yaswanth Chittepu, Ryan Park, Harshit Sikchi, Joey Hejna, Bradley Knox, Chelsea Finn, and Scott Niekum.

Scaling laws for reward model overoptimization in direct alignment algorithms.

arXiv preprint arXiv:2406.02900, 2024.

- Wen et al. [2024]

Jiaxin Wen, Ruiqi Zhong, Akbir Khan, Ethan Perez, Jacob Steinhardt, Minlie Huang, Samuel R Bowman, He He, and Shi Feng.

Language models learn to mislead humans via rlhf.

arXiv preprint arXiv:2409.12822, 2024.

- Randløv and Alstrøm [1998]

Jette Randløv and Preben Alstrøm.

Learning to drive a bicycle using reinforcement learning and shaping.

In ICML, volume 98, pages 463–471, 1998.

- Clark and Amodei [2016]

Jack Clark and Dario Amodei.

Faulty reward functions in the wild.

OpenAI Codex, 2016.

URL https://openai.com/blog/faulty-reward-functions/.

- Popov et al. [2017]

Ivaylo Popov, Nicolas Heess, Timothy Lillicrap, Roland Hafner, Gabriel Barth-Maron, Matej Vecerik, Thomas Lampe, Yuval Tassa, Tom Erez, and Martin Riedmiller.

Data-efficient deep reinforcement learning for dexterous manipulation.

arXiv preprint arXiv:1704.03073, 2017.

- Baker et al. [2019]

Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch.

Emergent tool use from multi-agent autocurricula.

arXiv preprint arXiv:1909.07528, 2019.

- Ekstrand and Willemsen [2016]

Michael D Ekstrand and Martijn C Willemsen.

Behaviorism is not enough: better recommendations through listening to users.

In Proceedings of the 10th ACM conference on recommender systems, pages 221–224, 2016.

- Merril and Oremus [2021]

Jeremy B. Merril and Will Oremus.

Five points for anger, one for a ‘like’: How facebook’s formula fostered rage and misinformation.

The Washington Post, 2021.

URL https://www.washingtonpost.com/technology/2021/10/26/facebook-angry-emoji-algorithm/.

- Luca and Zervas [2016]

Michael Luca and Georgios Zervas.

Fake it till you make it: Reputation, competition, and yelp review fraud.

Management science, 62(12):3412–3427, 2016.

- Ntoulas et al. [2006]

Alexandros Ntoulas, Marc Najork, Mark Manasse, and Dennis Fetterly.

Detecting spam web pages through content analysis.

In Proceedings of the 15th international conference on World Wide Web, pages 83–92, 2006.

- Laidlaw et al. [2025]

Cassidy Laidlaw, Shivam Singhal, and Anca Dragan.

Correlated proxies: A new definition and improved mitigation for reward hacking.

In The Thirteenth International Conference on Learning Representations, 2025.

- Strathern [1997]

Marilyn Strathern.

‘improving ratings’: audit in the british university system.

European review, 5(3):305–321, 1997.

- Liebman and Mahoney [2017]

Jeffrey B Liebman and Neale Mahoney.

Do expiring budgets lead to wasteful year-end spending? evidence from federal procurement.

American Economic Review, 107(11):3510–3549, 2017.

- Gilliom [2010]

John Gilliom.

Lying, cheating, and teaching to the test.

Schools under Surveillance: cultures of control in public education, pages 194–208, 2010.

- Lucas and Fuller [2018]

David S Lucas and Caleb S Fuller.

Bounties, grants, and market-making entrepreneurship.

The Independent Review, 22(4):507–528, 2018.

- Vann [2003]

Michael G Vann.

Of rats, rice, and race: The great hanoi rat massacre, an episode in french colonial history.

French Colonial History, 4(1):191–203, 2003.

- Pan et al. [2022]

Alexander Pan, Kush Bhatia, and Jacob Steinhardt.

The effects of reward misspecification: Mapping and mitigating misaligned models.

arXiv preprint arXiv:2201.03544, 2022.

- Gao et al. [2023a]

Leo Gao, John Schulman, and Jacob Hilton.

Scaling laws for reward model overoptimization.

In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023a.

- Brown et al. [2020]

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.

Language models are few-shot learners.

Advances in neural information processing systems, 33:1877–1901, 2020.

- Jaech et al. [2024]

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al.

Openai o1 system card.

arXiv preprint arXiv:2412.16720, 2024.

- Team et al. [2025]

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al.

Kimi k1. 5: Scaling reinforcement learning with llms.

arXiv preprint arXiv:2501.12599, 2025.

- Guo et al. [2025]

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al.

Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

arXiv preprint arXiv:2501.12948, 2025.

- Jacovi and Goldberg [2020]

Alon Jacovi and Yoav Goldberg.

Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness?, 2020.

URL https://arxiv.org/abs/2004.03685.

- Lanham et al. [2023]

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamilė Lukošiūtė, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez.

Measuring faithfulness in chain-of-thought reasoning, 2023.

URL https://arxiv.org/abs/2307.13702.

- Hurst et al. [2024]

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al.

Gpt-4o system card.

arXiv preprint arXiv:2410.21276, 2024.

- OpenAI [2025]

OpenAI.

Openai o3-mini system card, February 2025.

URL https://openai.com/index/o3-mini-system-card/.

Accessed: 2025-02-21.

- Bowman et al. [2022]

Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamilė Lukošiūtė, Amanda Askell, Andy Jones, Anna Chen, et al.

Measuring progress on scalable oversight for large language models.

arXiv preprint arXiv:2211.03540, 2022.

- Deng et al. [2024]

Yuntian Deng, Yejin Choi, and Stuart Shieber.

From explicit cot to implicit cot: Learning to internalize cot step by step.

arXiv preprint arXiv:2405.14838, 2024.

- Ouyang et al. [2022]

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.

Training language models to follow instructions with human feedback.

Advances in neural information processing systems, 35:27730–27744, 2022.

- Wei et al. [2022]

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.

Chain-of-thought prompting elicits reasoning in large language models.

Advances in neural information processing systems, 35:24824–24837, 2022.

- Kojima et al. [2022]

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa.

Large language models are zero-shot reasoners.

Advances in neural information processing systems, 35:22199–22213, 2022.

- Wang et al. [2022]

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.

Self-consistency improves chain of thought reasoning in language models.

arXiv preprint arXiv:2203.11171, 2022.

- Hao et al. [2023]

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu.

Reasoning with language model is planning with world model.

arXiv preprint arXiv:2305.14992, 2023.

- Yao et al. [2024]

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.

Tree of thoughts: Deliberate problem solving with large language models.

Advances in Neural Information Processing Systems, 36, 2024.

- Ling et al. [2017]

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom.

Program induction by rationale generation: Learning to solve and explain algebraic word problems.

arXiv preprint arXiv:1705.04146, 2017.

- Cobbe et al. [2021]

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al.

Training verifiers to solve math word problems.

arXiv preprint arXiv:2110.14168, 2021.

- Nye et al. [2021]

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al.

Show your work: Scratchpads for intermediate computation with language models.

arXiv preprint arXiv:2112.00114, 2021.

- Schick et al. [2023]

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom.

Toolformer: Language models can teach themselves to use tools.

Advances in Neural Information Processing Systems, 36:68539–68551, 2023.

- Nakano et al. [2021]

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al.

Webgpt: Browser-assisted question-answering with human feedback.

arXiv preprint arXiv:2112.09332, 2021.

- Komeili et al. [2021]

Mojtaba Komeili, Kurt Shuster, and Jason Weston.

Internet-augmented dialogue generation.

arXiv preprint arXiv:2107.07566, 2021.

- Press et al. [2023]

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis.

Measuring and narrowing the compositionality gap in language models, 2023.

URL https://arxiv.org/abs/2210.03350.

- Shen et al. [2024]

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang.

Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face.

Advances in Neural Information Processing Systems, 36, 2024.

- Lu et al. [2023]

Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao.

Chameleon: Plug-and-play compositional reasoning with large language models, 2023.

URL https://arxiv.org/abs/2304.09842.

- Gao et al. [2023b]

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig.

Pal: Program-aided language models.

In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023b.

- Chen et al. [2023]

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen.

Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks, 2023.

URL https://arxiv.org/abs/2211.12588.

- Patil et al. [2023]

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez.

Gorilla: Large language model connected with massive apis, 2023.

URL https://arxiv.org/abs/2305.15334.

- Zelikman et al. [2022]

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman.

Star: Bootstrapping reasoning with reasoning.

Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

- Sutton and Barto [2018]

Richard S Sutton and Andrew G Barto.

Reinforcement learning: An introduction.

MIT press, 2018.

- [53]

DeepMind.

Gemini flash.

https://deepmind.google/technologies/gemini/flash/.

Accessed: 2025-02-24.

- Anthropic [2025]

Anthropic.

Claude 3.7 sonnet system card, February 2025.

URL https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf.

Accessed: 2025-02-24.

- Lightman et al. [2023]

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.

Let’s verify step by step, 2023.

URL https://arxiv.org/abs/2305.20050.

- Uesato et al. [2022]

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins.

Solving math word problems with process- and outcome-based feedback, 2022.

URL https://arxiv.org/abs/2211.14275.

- Creswell et al. [2022]

Antonia Creswell, Murray Shanahan, and Irina Higgins.

Selection-inference: Exploiting large language models for interpretable logical reasoning, 2022.

URL https://arxiv.org/abs/2205.09712.

- Guan et al. [2025]

Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Helyar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam Toyer, Johannes Heidecke, Alex Beutel, and Amelia Glaese.

Deliberative alignment: Reasoning enables safer language models, 2025.

URL https://arxiv.org/abs/2412.16339.

- Ought [2022]

Ought.

Process-oriented learning with factored cognition, 2022.

URL https://ought.org/updates/2022-04-06-process.

Accessed: Feb 25, 2025.

- Hendrycks [2024]

Dan Hendrycks.

Introduction to ai safety, ethics, and society, 2024.

URL https://arxiv.org/abs/2411.01042.

- Hubinger [2022]

Evan Hubinger.

Monitoring for deceptive alignment.

AI Alignment Forum, 2022.

URL https://www.alignmentforum.org/posts/Km9sHjHTsBdbgwKyi/monitoring-for-deceptive-alignment.

- Christiano et al. [2021]

Paul Christiano, Ajeya Cotra, and Mark Xu.

Eliciting latent knowledge: How to tell if your eyes deceive you, 2021.

URL https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/.

- Bereska and Gavves [2024]

Leonard Bereska and Efstratios Gavves.

Mechanistic interpretability for ai safety – a review, 2024.

URL https://arxiv.org/abs/2404.14082.

- Roger [2023]

Fabien Roger.

Coup probes: Catching catastrophes with probes trained off-policy, 2023.

URL https://www.lesswrong.com/posts/WCj7WgFSLmyKaMwPR/coup-probes-catching-catastrophes-with-probes-trained-off.

LessWrong, online; accessed February 24, 2025.

- Mallen et al. [2024]

Alex Mallen, Madeline Brumley, Julia Kharchenko, and Nora Belrose.

Eliciting latent knowledge from quirky language models, 2024.

URL https://arxiv.org/abs/2312.01037.

- Anthropic [2023]

Anthropic.

Probes catch sleeper agents, 2023.

URL https://www.anthropic.com/research/probes-catch-sleeper-agents.

Anthropic Research, online; accessed February 24, 2025.

- Transformer Circuits [2024]

Transformer Circuits.

Features as classifiers, 2024.

URL https://transformer-circuits.pub/2024/features-as-classifiers/index.html.

Online; accessed February 24, 2025.

- Zou et al. [2023]

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks.

Representation engineering: A top-down approach to ai transparency, 2023.

URL https://arxiv.org/abs/2310.01405.

- Pacchiardi et al. [2023]

Lorenzo Pacchiardi, Alex J. Chan, Sören Mindermann, Ilan Moscovitz, Alexa Y. Pan, Yarin Gal, Owain Evans, and Jan Brauner.

How to catch an ai liar: Lie detection in black-box llms by asking unrelated questions, 2023.

URL https://arxiv.org/abs/2309.15840.

- Manakul et al. [2023]

Potsawee Manakul, Adian Liusie, and Mark J. F. Gales.

Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models, 2023.

URL https://arxiv.org/abs/2303.08896.

- Greenblatt et al. [2024]

Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, Akbir Khan, Julian Michael, Sören Mindermann, Ethan Perez, Linda Petrini, Jonathan Uesato, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, and Evan Hubinger.

Alignment faking in large language models, 2024.

URL https://arxiv.org/abs/2412.14093.

- Denison et al. [2024]

Carson Denison, Monte MacDiarmid, Fazl Barez, David Duvenaud, Shauna Kravec, Samuel Marks, Nicholas Schiefer, Ryan Soklaski, Alex Tamkin, Jared Kaplan, Buck Shlegeris, Samuel R. Bowman, Ethan Perez, and Evan Hubinger.

Sycophancy to subterfuge: Investigating reward-tampering in large language models, 2024.

URL https://arxiv.org/abs/2406.10162.

- Meinke et al. [2025]

Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius Hobbhahn.

Frontier models are capable of in-context scheming, 2025.

URL https://arxiv.org/abs/2412.04984.

- Kirchner et al. [2024]

Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda.

Prover-verifier games improve legibility of llm outputs.

arXiv preprint arXiv:2407.13692, 2024.

- McKenzie et al. [2023]

Ian R McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, et al.

Inverse scaling: When bigger isn’t better.

arXiv preprint arXiv:2306.09479, 2023.

- Parcalabescu and Frank [2024]

Letitia Parcalabescu and Anette Frank.

On measuring faithfulness or self-consistency of natural language explanations, 2024.

URL https://arxiv.org/abs/2311.07466.

## Appendix A More Frontier Model Hacks and Chains-of-thought

We share further examples of reward hacking in our frontier reasoning training run. Each of these hacks were surfaced by our CoT monitor and bypassed any existing mitigations we had. Because full agent solutions (trajectories) can be long, we show summarizations of the trajectory (a mix of CoT, tool calls, and code patches) in red, the raw CoT directly quoted in green, code patch tool calls in blue, and bash tool calls in purple. We mask variable names for IP considerations, and we bold certain phrases in CoT excerpts for emphasis. We share an example of:

- •

Figure 8: systemic hack discussed in Section 2.

- •

Figure 9: Returning expected error codes.

- •

Figure 10: Lazy solution that makes all variants have fitness 0.0 for a genetic algorithm.

- •

Figure 11: Decompiling a reference solution’s compiled jar file that was incorrectly left in the task.

- •

Figure 12: Parsing a test file at runtime to extract and return values expected by the test.

- •

Figure 13/14/15: Creating a local stub of the pandas library along with deleting and stubbing 19 other files.

## Appendix B Monitor Failures

We saw in Sections 2 and 3 that our monitors do not have perfect recall. We note that there is a lot of room for improvement with our monitor; in fact, the monitoring prompt (shown in Appendix C) was the first we experimented with. We furthermore did not train our models specifically for monitoring, but rather used existing chat models zero-shot. We show examples where our monitor failed to catch an instance of the systemic hacks discussed in Section 2.1:

- 1.

Figure 16: Failure to catch an instance of the hack.

- 2.

Figure 17: Failure to catch an instance of the hack.

In both examples, we find that the monitor did actually recognize the behavior. In the case of the hack, because the agent in this scenario also implemented some correct functionality, the monitor seemed to be confused and thought it was all correct. In the case of the hack, the monitor recognized this was done, but thought it was only done for optional features. It is very likely that making the monitor prompt stricter, taking multiple monitor samples and using a majority vote, or other simple modifications could improve its performance.

## Appendix C Monitor Details

In all experiments, our monitor is simply a zero-shot prompted GPT-4o. We use the following prompt in Figure 18, where {agent_prompt} is replaced with the prompt to the agent, and {agent_messages} is replaced with all agent chain-of-thought messages, tool calls, and outputs. For our representative CoT pressure experiments in Section 3 we exclude agent actions and outputs.

## Appendix D Monitoring for Self Awareness

We have mostly focused on monitoring for reward hacking, but we’ve also started monitoring for other classes of worrying behavior. We constructed a monitor that looks for evidence of self awareness, i.e. instances in which the model seems to know it is in training or performing evaluations.
This is a prerequisite capability for sandbagging, alignment faking [71]—whereby a model changes its behavior because it knows it is being evaluated—or reward tampering [72]—where the model attempts to modify its own training code.
We use the following prompt in Figure 19, where {agent_prompt} is replaced with the prompt to the agent, and {agent_messages} is replaced with all agent chain-of-thought messages, tool calls, and outputs.

We did not detect a significant rate of self-awareness in our frontier reasoning model training run. Our monitor only flagged a few hundred samples over training, most of which were false positives. Often the model thinks things like “As chatgpt, I need to…” which the first version of our monitor mistakingly flagged. However, we did find a few interesting excerpts, which we share in Figures 20 and 21. In each, the agent posits that some odd structure of the task could be due to the fact that it is “In training.”

## Appendix E GPT-4o-mini as a monitor

In Section 2, we show that GPT-4o can effectively monitor a superior reasoning model. We investigate using an even smaller model, GPT-4o-mini, as a monitor, and compare its performance to GPT-4o on the systemic hacks explored in Section 2. In Figure 22, we find that GPT-4o-mini is rather poor at finding the systemic hacks compared to GPT-4o, but GPT-4o-mini is still able to find a non-negligible proportion of hacks (See Table 2 for recall and precision numbers). We note that GPT-4o-mini was much harder to instruct to actually do the task, and often failed to output a label at all despite very clear instructions. Thus, its poor performance may be due to instruction following and long prompt issues rather than an inherent capability issue at this model scale. We hope to explore this more thoroughly in future work, but included these results here as a preliminary data.

Monitor
Recall ( )
Recall ( )
Recall (Joint)

∼\sim Precision (Joint)

GPT-4o-mini
17%
6%
12%
83%

GPT-4o
93%
97%
95%
77%

## Appendix F Monitoring Latents: Activations vs. Chains-of-Thought

Both chains-of-thought and activations can be viewed as latent variables within an agent’s computation of its final output. In outcome-based reinforcement learning, both are not directly pressured by learning objectives to adhere to specific formats, be safe for users to read, or meet any criteria beyond helping the agent reach a highly rewarded final output. As a result, these latent variables can contain information that outputs will not, such as incorrect answers that are considered and then omitted, information that could allow misuse, and, as demonstrated in this paper, the model’s intent to reward hack [62].

Chains-of-thought (CoT) and activations each offer distinct advantages when used for monitoring. One key difference lies in the learned feature basis. Chains-of-thought are expressed in human language, providing a consistent basis across different models. For example, the word “elephant” almost certainly conveys a similar meaning across a wide range of large language models, even when those models differ in size, architecture, or training data. In contrast, activations operate in high-dimensional feature spaces that vary across training runs due to factors like random initialization, model architecture, and training dynamics. As a result, the concept of “elephant” may be represented by entirely different vectors or circuits in different models, even if those models’ outputs behave similarly. The consistency of chain-of-thought representations significantly simplifies the creation of monitors, as it enables zero-shot prompting of one model to oversee the CoT of another.

However, chains-of-thought are only a low-dimensional component of a model’s full computation. While we have shown they provide valuable insight into certain aspects of the reasoning process, it is also possible that they omit critical information. Activations, on the other hand technically encompass all the information involved in the model’s reasoning. However, due to their high dimensionality and nonlinear features, interpreting activations productively can be difficult. As a result, in practice analysis often focuses on activations from specific layers and tokens. This selective approach may lead to the activations considered not containing all relevant information, much like chains-of-thought.

While we have found that CoT monitors can detect complex hacks that require deliberate reasoning in the chain-of-thought, they have been less effective at finding simpler hacks that exploit superficial patterns, special formatting, or specific trigger phrases learned to fool a reward model. Although these simpler hacks are generally less consequential than issues like executing incorrect code, it remains unclear whether all impactful hacks will manifest within the chain-of-thought. This uncertainty underscores the importance of continued investment in other mitigations, such as activation-based monitoring, that could capture forms of reward hacking that CoT monitoring alone could miss.

## Appendix G More commentary on faithfulness and related concepts

Terms such as faithfulness, legibility, steerability, and plausibility have been used by the research community to describe chains-of-thought in a variety of ways that are often imprecise and inconsistent. There have been prior efforts to define these terms for NLP systems [27], but there is still no widely adopted taxonomy.

As we noted in Section 5.1, we believe monitorability should be of primary concern but have found that discussing these other properties of latents to be instructive.
We recognize there are many valid and useful potential definitions of these concepts.
Below we share one possible framework for faithfulness and related terms that reflects our current thinking about CoT interpretability in the context of reasoning LLMs and monitoring. This thinking may change over time as we learn more; however, for now we have found that this framework helped reduce internal confusion about the meaning of these concepts.

- •

A monitor has access to various observations of a trajectory including actions, outputs, and latent variables in the agent’s computation such as activations or chains-of-thought. The monitor tries to determine the value for a particular property of outputs, vv.

- •

The latent is faithful (F) with respect to vv if it contains all relevant information to predicting vv, i.e. the latent is a sufficient statistic for vv.

- •

The latent is obfuscating (O) if it actively encourages the monitor the believe in the incorrect value of vv. Obfuscation would result in a monitor that performs worse than random.

- •

The latent is plausible (P) or legible if the monitor believes it is getting useful information about vv from the latent.

Figure 23 summarizes the salient failure modes of monitorability and the single success mode. We fully list out possible scenarios in the section below, based on whether the latent is faithful or not (F or -F), obfuscating or not (O or -O), and plausible to the monitor or not (P or -P).

### G.1 Case Study

#### G.1.1 Setting

Consider a scenario where we want to determine if a model is sexist in its hiring decisions. The prompt we give the model is “Based on these resumes for Sally and Tim, who should we pick for the job? [insert resumes]”. The monitor observes the latent CoT and answer and tries to determine the value of property v=Is the hiring decision sexist?v=\texttt{Is the hiring decision sexist?}.

In this example, we assume that the model is in fact sexist, and it decides that Sally is the more qualified candidate, but it does not want to hire women and therefore recommends Tim. We further assume that the model was trained on English and French corpuses and randomly speaks in either language in the CoT. The monitor only understands English, but this fact is not known to the model.

#### G.1.2 Outcomes

Figure 24 shows the tree of possible outcomes based on whether the model’s latent is faithful or not, obfuscating or not, and plausible or not.

Where we have defined faithfulness and obfuscation as intrinsic properties of the model, note that plausibility is also a function of the monitor itself. If the model CoT is in French, as in the (F, -O, -P) example, then our particular monitor finds such a latent implausible, while a different bilingual monitor would find it plausible.
In the (F, O, *) branch, the true sexist reasoning is faithfully present in the latent but is hidden through steganography; meanwhile, the plain text method presented to observers provides an obfuscating reason for why it decided to recommend Tim.
For a text-only CoT, simple examples of steganography are the use of an acrostic cipher or dog whistles.
In the case of multimodal CoTs, the faithful reasoning might be hidden in the pixels of a relevant image the model first outputs in its CoT, while the obfuscating message is written in text.

### G.2 Faithfulness vs Causality

It is important to point out our definition of faithfulness does not imply any expectations of causality. In contrast, intervention methods, such as those in [28], implicitly treat faithfulness as a causal metric by intervening on latent representations and observing changes in target variables. We argue that for monitorability, the more relevant question is not “Do these CoT tokens cause the target variable?” but rather “Do these CoT tokens explain the target variable?”.

Firstly, causal CoTs (see Figure 25) are not a necessary condition for monitorability. A CoT may perfectly describe the model’s latent reasoning without being causally upstream of the output. For example, the reasoning process may be fully completed in the activations and directly cause both the Chain of Thought (CoT) and the final answer. While causal CoTs can be useful for steering models, it is not a requirement for being able to monitor models and trust our readings of their latents.

Secondly, intervention-based proxies for measuring causality are insufficient measures for faithfulness. Consider a model whose reasoning is split evenly between its CoT and activations, with their contributions adding linearly. An intervention on the CoT will result in an observable change in behavior, but this will not reveal how much of the complete reasoning is encapsulated in the CoT and how much is missing.111The authors of [76] make the related but broader argument that any “faithfulness” test which relies on executing changes to the model’s input and observing changes to the prediction actually measures self-consistency rather than faithfulness, with the former a necessary but insufficient condition for the latter.

Nevertheless, causal faithfulness metrics are still valuable and worth measuring. Since steerability is a sufficient condition for faithfulness, strong steerability with respect to target behaviors we care about would increase our confidence in the faithfulness of CoTs for those behaviors.

Generated on Sat Apr 5 21:27:18 2025 by LaTeXML
