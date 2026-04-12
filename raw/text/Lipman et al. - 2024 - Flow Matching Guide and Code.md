# Lipman et al. - 2024 - Flow Matching Guide and Code

- Source HTML: `/Users/eric/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/raw/html/Lipman et al. - 2024 - Flow Matching Guide and Code.html`
- Source URL: https://ar5iv.labs.arxiv.org/html/2412.06264
- Generated from: `scripts/fetch_web_text.py`

## Extracted Text

# Flow Matching Guide and Code

Yaron Lipman

  
Marton Havasi

  
Peter Holderrieth

  
Neta Shaul

  
Matt Le

  
Brian Karrer

  
      Ricky T. Q. Chen

  
David Lopez-Paz

  
Heli Ben-Hamu

  
Itai Gat

[

[

[

###### Abstract

Flow Matching (FM) is a recent framework for generative modeling that has achieved state-of-the-art performance across various domains, including image, video, audio, speech, and biological structures.
This guide offers a comprehensive and self-contained review of FM, covering its mathematical foundations, design choices, and extensions.
By also providing a PyTorch package featuring relevant examples (e.g., image and text generation), this work aims to serve as a resource for both novice and experienced researchers interested in understanding, applying and further developing FM.

\irstorderpartialderivatives),\ie,\begin{equation∗}\brac{\partial_y\phi(y)}_{i,j}=\\rac{\partial\phi^i}{\partialx^j},\i,j\in[d],\end{equation∗}and$\detA$denotesthedeterminantoasquarematrix$A\in\Real^{d\timesd}$.Thus,weconcludethatthePDF$p_Y$is\begin{equation}p_Y(y)=p_X(\psi^{-1}(y))\abs{\det\partial_y\psi^{-1}(y)}.\end{equation}Wewilldenotethepush-\orwardoperatorwiththesymbol$\sharp$,thatis\begin{equation}\label{e:push-\orward_p}\brac{\psi_\sharpp_X}(y)\de\ep_X(\psi^{-1}(y))\abs{\det\partial_y\psi^{-1}(y)}.\end{equation}\begin{\igure}\centering\begin{tabular}{ccc}\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_1_p.png}&\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_10_p.png}&\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_16_p.png}\end{tabular}\caption{A\lowmodel$X_t=\psi_t(X_0)$isde\inedbyadi\feomorphism$\psi_t:\Real^d\too\Real^d$(visualizedwithabrownsquaregrid)pushingsamples\romasourceRV$X_0$(le

,͡blackpoints)towardsometargetdistribution$q$(right).Weshowthreedi\ferenttimes$t$.}\label{\ig:\low_model}\end{\igure}\subsection{Flowsasgenerativemodels}\label{ss:\lows_and_velocities}Asmentionedin~\cre{s:quick_tour},thegoalogenerativemodelingistotrans\ormsamples$X_0=x_0$\roma\highlight{sourcedistribution}$p$intosamples$X_1=x_1$\roma\highlight{targetdistribution}$q$.Inthissection,westartbuildingthetoolsnecessarytoaddressthisproblembymeansoa\lowmapping$\psi_t$.More\ormally,a$C^r$\highlight{\low}isatime-dependentmapping$\psi:[0,1]\times\Real^d\too\Real^d$implementing$\psi:(t,x)\mapsto\psi_t(x)$.Such\lowisalsoa$C^r([0,1]\times\Real^{d},\Real^d)$\unction,suchthatthe\unction$\psi_t(x)$isa$C^r$di\feomorphismin$x$all$t\in[0,1]$.A\highlight{\lowmodel}isa\emph{continuous-timeMarkovprocess}$(X_t)_{0\leqt\leq1}$de\inedbyapplyinga\low$\psi_t$totheRV$X_0$:\begin{my\rame}\begin{equation}\label{e:\low_model}X_t=\psi_t(X_0),\quadt\in[0,1],\text{where}X_0\simp.%\end{equation}\end{my\rame}See\Cre{\ig:\low_model}anillustrationoa\lowmodel.Toseewhy$X_t$isMarkov,notethat,anychoiceo$0\leqt<s\leq1$,wehave\begin{equation}\label{e:\low_is_markov}X_s=\psi_s(X_0)=\psi_s(\psi_t^{-1}(\psi_t(X_0)))=\psi_{s|t}(X_t),\end{equation}wherethelastequality\ollows\romusing\cre{e:\low_model}toset$X_t=\psi_t(X_0)$,andde\ining$\psi_{s|t}\de\e\psi_s\circ\psi_t^{-1}$,whichisalsoadi\feomorphism.$X_s=\psi_{s|t}(X_t)$impliesthatstateslaterthan$X_t$dependonlyon$X_t$,so$X_t$isMarkov.In\act,\lowmodels,thisdependenceis\emph{deterministic}.Insummary,thegoal\highlight{generative\lowmodeling}isto\inda\low$\psi_t$suchthat\begin{equation}X_1=\psi_1(X_0)\simq.%\end{equation}\begin{\igure}\centering\begin{tabular}{ccc}\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_1.png}&\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_10.png}&\includegraphics[width=0.3\textwidth]{assets/\low_velocity/\low_16.png}\end{tabular}\caption{A\low$\psi_t:\Real^d\too\Real^d$(squaregrid)isde\inedbyavelocity\ield$u_t:\Real^d\too\Real^d$(visualizedwithbluearrows)thatprescribesitsinstantaneousmovementsatalllocations.Weshowthreedi\ferenttimes$t$.}\label{\ig:\low}\end{\igure}\subsubsection{Equivalencebetween\lowsandvelocity\ields}\label{sec:equivalence_\lows_velocities}A$C^r$\low$\psi$canbede\inedintermsoa$C^r([0,1]\times\Real^d,\Real^d)$\emph{velocity\ield}$u:[0,1]\times\Real^d\too\Real^d$implementing$u:(t,x)\mapstou_t(x)$viathe\ollowingODE:\begin{subequations}\label{e:\low}\vspace{-10pt}\begin{align}\\rac{\dd}{\ddt}\psi_{t}(x)&=u_t(\psi_{t}(x))&\text{(\lowODE)}\label{e:\low_\low}\\\psi_{0}(x)&=x&\text{(\lowinitialconditions)}\label{e:\low_boundary}\end{align}\end{subequations}See\cre{\ig:\low}anillustrationoa\lowtogetherwithitsvelocity\ield.Astandardresultregardingtheexistenceanduniquenessosolutions$\psi_t(x)$to\cre{e:\low}is(see\eg,\cite{perko2013di\ferential,coddington1956theory}):\begin{my\rame}\begin{theorem}[Flowlocalexistenceanduniqueness]\label{thm:ode_existence_and_uniqueness}I$u$is$C^r([0,1]\times\Real^{d},\Real^d)$,$r\geq1$(inparticular,locallyLipschitz),thentheODEin\eqre{e:\low}hasauniquesolutionwhichisa$C^r(\Omega,\Real^d)$di\feomorphism$\psi_t(x)$de\inedoveranopenset$\Omega$whichissuper-seto$\set{0}\times\Real^d$.\end{theorem}\end{my\rame}Thistheoremguaranteesonlythe\emph{local}existenceanduniquenessoa$C^r$\lowmovingeachpoint$x\in\Real^d$by$\psi_t(x)$duringapotentiallylimitedamountotime$t\in[0,t_x)$.Toguaranteeasolutionuntil$t=1$all$x\in\Real^d$,onemustplaceadditionalassumptionsbeyondlocalLipschitzness.Forinstance,onecouldconsiderglobalLipschitness,guaranteedbybounded\irstderivativesinthe$C^1$case.However,wewilllaterrelyonadi\ferentcondition---namely,integrability---toguaranteetheexistenceothe\lowalmosteverywhere,anduntiltime$t=1$.So\ar,wehaveshownthatavelocity\ielduniquelyde\inesa\low.Conversely,givena$C^1$\low$\psi_t$,onecanextractitsde\iningvelocity\ield$u_t(x)$arbitrary$x\in\Real^d$byconsideringtheequation$\\rac{\dd}{\ddt}\psi_t(x′)=u_t(\psi_t(x′))$,andusingthe\actthat$\psi_t$isaninvertibledi\feomorphismevery$t\in[0,1]$tolet$x′=\psi^{-1}_t(x)$.There\ore,theuniquevelocity\ield$u_t$determiningthe\low$\psi_t$is\begin{equation}\label{e:u_\rom_psi}u_t(x)=\dot{\psi}_t(\psi^{-1}_t(x)),\end{equation}where$\dot{\psi}_t\de\e\\rac{\dd}{\ddt}\psi_t$.Inconclusion,wehaveshowntheequivalencebetween$C^r$\lows$\psi_t$and$C^r$velocity\ields$u_t$.\subsubsection{Computingtargetsamples\romsourcesamples}Computingatargetsample$X_1$---or,ingeneral,anysample$X_t$---entailsapproximatingthesolutionotheODEin~\cre{e:\low}starting\romsomeinitialcondition$X_0=x_0$.NumericalmethodsODEsisaclassicalandwellresearchedtopicinnumericalanalysis,andamyriadopower\ulmethodsexist\citep{iserles2009\irst}.Oneothesimplestmethodsisthe\emph{Eulermethod},implementingtheupdaterule\begin{equation}\label{e:euler_method}X_{t+h}=X_t+hu_t(X_t)%\end{equation}where$h=n^{-1}>0$isastepsizehyper-parameterwith$n\in\Nat$.Todrawasample$X_1$\romthetargetdistribution,applytheEulermethodstartingatsome$X_0\simp$toproducethesequence$X_h,X_{2h},\ldots,X_1$.TheEulermethodcoincideswith\irst-orderTaylorexpansiono$X_t$:$$X_{t+h}=X_t+h\dot{X}_t+o(h)=X_t+hu_t(X_t)+o(h),$$where$o(h)$standsa\unctiongrowingslowerthan$h$,thatis,$o(h)/h\too0$as$h\too0$.There\ore,theEulermethodaccumulates$o(h)$errorperstep,andcanbeshowntoaccumulate$o(1)$errora\ter$n=1/h$steps.There\ore,theerrorotheEulermethodvanishesasweconsidersmallerstepsizes$h\too0$.TheEulermethodisjustoneexampleamongmanyODEsolvers.\Cre{ex:euler_method}exempli\iesanotheralternative,thesecond-order\emph{midpointmethod},whicho\tenoutper\ormstheEulermethodinpractice.\begin{pbox}[label={ex:euler_method}]{Computing$X_1$withMidpointsolver}\begin{minted}[linenos,breaklines,mathescape,

Conversion to HTML had a Fatal error and exited abruptly. This document may be truncated or damaged.

Generated on Tue Jan 7 07:42:19 2025 by LaTeXML
