---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}


<span class='anchor' id='about-me'></span>

I am a Research Assistant Professor within the [Westlake NLP group](https://westlakenlp.netlify.app/). My research interests lie in Interdisciplinary Methods for Pioneering AI and Computational Trust (IMPACT), particularly in enhancing the robustness and security of LLMs and augmenting their reasoning capabilities for cross-disciplinary applications. I served as an **Area Chair** at EMNLP and CIKM, and a **Senior Program Committee** member at IJCAI, an **associate editor** at Special Issue on TIST with Prof. Qiang Yang and Dr. Jindong Wang.

I have published 9 CCF-A (10 Tsinghua A) papers as (co-) first author and 40 papers at top-tier conferences, such as ICLR, NeurIPS, ACL, WWW, AAAI, and SIGIR with a total citation: <a href='https://scholar.google.com/citations?user=go3sFxcAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>. I am keen on mentoring and working with highly motivated students who possess strong programming capabilities on these topics.  

<span style="color:red; font-size: 100%">**If you are looking for Interns/RAs**</span>, please do not hesitate to contact me via email: yanglinyi[at]westlake[dot]edu[dot]cn 
I am interested in the following three fundamental challenges and three application fields.

Fundamental Problems:
1. Explanation (XAI) 2. Long-context Retrieval 3. Multi-agent Collaboration

Applications:
1. AI in Finance 2. AI for Science 3. Social Responsibility and Interdisciplinary Research

# 🏆 Honors and Awards

- Outstanding Postdoctoral Representative, 2023.
- Outstanding Postdoc Researcher, 2022.
- Outstanding Self-financed Students Abroad (Non-CSC PhDs; Only One Type-B Winner in Ireland), 2021.
- Best Paper Candidate, CCIS, 2018.

# 🌱 Repository
[ **[GLUE-X](https://github.com/YangLinyi/GLUE-X)** ![](https://img.shields.io/github/stars/YangLinyi/GLUE-X?style=social&label=Stars) | **[FinNLP](https://github.com/YangLinyi/FinNLP-Progress)** ![](https://img.shields.io/github/stars/YangLinyi/FinNLP-Progress?style=social&label=Stars) | **[PandaLM](https://github.com/WeOpenML/PandaLM)** ![](https://img.shields.io/github/stars/WeOpenML/PandaLM?style=social&label=Stars)] | **[USB](https://github.com/microsoft/Semi-supervised-learning)** ![](https://img.shields.io/github/stars/microsoft/Semi-supervised-learning?style=social&label=Stars)]


# ⚡ News
- 2024-May Two papers have been accepted to the main conference of [ACL 2024](https://2024.aclweb.org/).
- 2024-Feb One paper has been accepted to [NAACL 2024](https://2024.naacl.org/) (Rationale-centric Counterfactual Data Augmentation).
- 2024-Jan Three papers have been accepted to [ICLR 2024](https://iclr.cc/) (SuperContext; FastDetect; PandaLM).
- 2023-Dec One paper has been accepted to [EMNLP 2023](https://2023.emnlp.org/).
- 2023-Nov Organized ACM TIST Special Issue on Evaluations of Large Language Model with Dr. Jindong Wang and Prof. Qiang Yang.
- 2023-May 🔥🔥 Four papers have been accepted to [ACL 2023](https://2023.aclweb.org/) (Three co-first author papers).
- 2023-Apr Our paper discussing the robustness of ChatGPT has been accepted to [ICLR 2023](https://arxiv.org/abs/2302.12095) Workshop.
- Area Chair / Senior Programme Committee (SPC): EMNLP-22; IJCAI-23.
- PC Member/Reviewer: CIKM-20; SIGIR-21; CKIM-21; EMNLP 2021-2024; ACL 2021-2024; COLING 2022-2024; TASLP; TALLIP; TBD.
- 2022-Dec: 🎉🎉 I received Outstanding Postdoctoral Fellows from Westlake University and gave a talk as the only postdoctoral representative.
- 2022-Sep: One paper co-operating with MSRA has been accepted to [NeurIPS 2022](https://arxiv.org/pdf/2208.07204/). The first author was my intern at Westlake University. (core: A*, CCF: A)
- 2022-Aug: Two papers (one first-author paper) have been accepted to [COLING 2022](https://coling2022.org/). (core: A, CCF: B)
- 2022-Mar: One co-first author long paper has been accepted to [ACL 2022](https://www.2022.aclweb.org/) main conference. (core: A*, CCF: A)
- 2022-Jan: One first-author long paper has been accepted to [AAAI 2022](https://www.aaai.org/AAAI22Papers/AAAI-4799.YangL.pdf) (15% acceptance rate). (core: A*, CCF: A)
- 2022-Jan Invited to serve as an Area Chair (AC) at EMNLP-22.

# 📝 Selected Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024</div><img src='images/ICLR_24.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Supervised Knowledge Makes Large Language Models Better In-context Learners**

**Linyi Yang** *, Shuibai Zhang *, Zhuohao Yu *, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, Yue Zhang

While previous in-context learning research has focused on enhancing models to adhere to users' specific instructions and quality expectations, and to avoid undesired outputs, little to no work has explored the use of task-specific fine-tuned Language Models (SLMs) to improve LLMs' in-context learning during the inference stage. Our primary contribution is the establishment of a simple yet effective framework that enhances the reliability of LLMs as it: 1) generalizes out-of-distribution data, 2) elucidates how LLMs benefit from discriminative models, and 3) minimizes hallucinations in generative tasks. Using our proposed plug-in method, enhanced versions of Llama 2 and ChatGPT surpass their original versions regarding generalizability and factuality. We offer a comprehensive suite of resources, including 16 curated datasets, prompts, model checkpoints, and LLM outputs across 9 distinct tasks. Our empirical analysis sheds light on the advantages of incorporating discriminative models into LLMs and highlights the potential of our methodology in fostering more reliable LLMs. 

[Paper](https://openreview.net/forum?id=bAMPOUF227) [Code](https://github.com/YangLinyi/Supervised-Knowledge-Makes-Large-Language-Models-Better-In-context-Learners)

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">EMNLP 2023</div><img src='images/ood_survey.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Out-of-Distribution Generalization in Natural Language Processing: Past, Present, and Future**

**Linyi Yang**, Yaoxiao Song, Xuan Ren, Chenyang Lyu, Yidong Wang, Lingqiao Liu, Jindong Wang, Jennifer Foster, Yue Zhang

Machine learning (ML) systems in natural language processing (NLP) face significant challenges in generalizing to out-of-distribution (OOD) data, where the test distribution differs from the training data distribution. This poses important questions about the robustness of NLP models and their high accuracy, which may be artificially inflated due to their underlying sensitivity to systematic biases. Despite these challenges, there is a lack of comprehensive surveys on the generalization challenge from an OOD perspective in natural language understanding. Therefore, this paper aims to fill this gap by presenting the first comprehensive review of recent progress, methods, and evaluations. 

[Paper](https://aclanthology.org/2023.emnlp-main.276/) 

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/GLUE-X.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-Distribution Generalization Perspective**

**Linyi Yang** *, Shuibai Zhang *, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, Yue Zhang

This paper presents the first attempt at creating a unified benchmark named GLUE-X for evaluating OOD robustness in NLP models, highlighting the importance of OOD robustness and providing insights on how to measure the robustness of a model and how to improve it. The benchmark includes 15 publicly available datasets for OOD testing, and evaluations are conducted on 8 classic NLP tasks over 21 popularly used PLMs. Our findings confirm the need for improved OOD accuracy in NLP tasks, as significant performance degradation was observed in all settings compared to in-distribution (ID) accuracy.

[Paper](https://arxiv.org/pdf/2211.08073.pdf) [Code](https://github.com/YangLinyi/GLUE-X)

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/fintrust.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Measuring Consistency in Text-based Financial Forecasting Models**

**Linyi Yang** *, Yingpeng Ma *, Yue Zhang

Financial forecasting has been an important and active area of machine learning research, as even the most modest advantages in predictive accuracy can be parlayed into significant financial gains. Recent advances in natural language processing (NLP) bring the opportunity to leverage textual data, such as earnings reports of publicly traded companies, to predict the return rate for an asset. However, when dealing with such a sensitive task, the consistency of models – their invariance under meaning-preserving alternations in input – is a crucial property for building user trust. Despite this, current methods for financial forecasting do not take consistency into consideration. To address this issue, we propose FinTrust, an evaluation tool that assesses logical consistency in financial text. Using FinTrust, we show that the consistency of state-of-the-art NLP models for financial forecasting is poor. Our analysis of the performance degradation caused by meaning-preserving alternations suggests that current text-based methods are not suitable for robustly predicting market information.

[Paper](https://aclanthology.org/2023.acl-long.769/) [Code](https://github.com/yingpengma/fintrust)

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/cross_qa.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Learning to Generalize for Cross-domain QA**

Yingjie Niu *, **Linyi Yang** *, Ruihai Dong, Yue Zhang

There have been growing concerns regarding the out-of-domain generalization ability of natural language processing (NLP) models, particularly in question-answering (QA) tasks. Current synthesized data augmentation methods for QA are hampered by increased training costs. To address this issue, we propose a novel approach that combines prompting methods and linear probing with fine-tuning strategy, which does not entail additional cost. Our method has been theoretically and empirically shown to be effective in enhancing the generalization ability of both generative and discriminative models. Our approach outperforms state-of-the-art baselines, with an average increase in F1 score of 4.5%-7.9%. Furthermore, our method can be easily integrated into any pre-trained models and offers a promising solution to the under-explored cross-domain QA task.

[Paper](https://aclanthology.org/2023.findings-acl.84/) [Code](https://github.com/FreddieNIU/Prompt-QA)

</div>
</div>


<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2022</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2022</div><img src='images/acl_22_fig.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**A Rationale-Centric Framework for Human-in-the-loop Machine Learning**

Jinghui Lu*, **Linyi Yang** *, Brian Namee, Yue Zhang

We present a novel rational-centric framework with human-in-the-loop – Rationales-centric Double-robustness Learning (RDL) – to boost model out-of-distribution performance in few-shot learning scenarios. By using static semi-factual generation and dynamic human-intervened correction, RDL, acting like a sensible “inductive bias”, exploits rationales (i.e. phrases that cause the prediction), human interventions and semi-factual augmentations to decouple spurious associations and bias models towards generally applicable underlying distributions, which enables fast and accurate generalisation. Experimental results show that RDL leads to significant prediction benefits on both in-distribution and out-of-distribution tests, especially for few-shot learning scenarios, compared to many state-of-the-art benchmarks.

[Paper](https://aclanthology.org/2022.acl-long.481/) [Code](https://github.com/GeorgeLuImmortal/RDL-Rationales-centric-Double-robustness-Learning))

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAAI 2022</div><img src='images/NumHTML.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">
**NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-task Financial Forecasting**

**Linyi Yang**, Jiazheng Li, Ruihai Dong, Yue Zhang, Barry Smyth

Financial forecasting has been an important and active area of machine learning research because of the challenges it presents and the potential rewards that even minor improvements in prediction accuracy or forecasting may entail. Traditionally, financial forecasting has heavily relied on quantitative indicators and metrics derived from structured financial statements. Earnings conference call data, including text and audio, is an important source of unstructured data that has been used for various prediction tasks using deep earning and related approaches. However, current deep learning-based methods are limited in the way that they deal with numeric data; numbers are typically treated as plain-text tokens without taking advantage of their underlying numeric structure. This paper describes a numeric-oriented hierarchical transformer model to predict stock returns, and financial risk using multi-modal aligned earnings calls data by taking advantage of the different categories of numbers (monetary, temporal, percentages etc.) and their magnitude.

[Paper](https://arxiv.org/abs/2201.01770) [Code](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction))

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2021</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2021</div><img src='images/acl_21_fig.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Exploring the Efficacy of Automatically Generated Counterfactuals for Sentiment Analysis**

**Linyi Yang**, Jiazheng Li, Padraig Cunningham, Yue Zhang, Barry Smyth, Ruihai Dong

While state-of-the-art NLP models have been achieving excellent performance in a wide range of tasks in recent years, important questions are being raised about their robustness and their underlying sensitivity to systematic biases that may exist in their training and test data. Such issues manifest in performance problems when faced with out-of-distribution data in the field. One recent solution has been to use counterfactually augmented datasets in order to reduce any reliance on spurious patterns that may exist in the original data. Producing high-quality augmented data can be costly and time-consuming as it usually needs to involve human feedback and crowdsourcing efforts. In this work, we propose an alternative by describing and evaluating an approach to automatically generating counterfactual data for the purpose of data augmentation and explanation. 

[Paper](https://aclanthology.org/2021.acl-long.26/) [Code]([https://github.com/lijiazheng99/Counterfactuals-for-Sentiment-Analysis])

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">WWW 2020</div><img src='images/html.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">
  
**HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction**

**Linyi Yang**, James Ng, Barry Smyth, Ruihai Dong

This paper proposes a novel hierarchical, transformer, multi-task architecture designed to harness the text and audio data from quarterly earnings conference calls to predict future price volatility in the short and long term. This includes a comprehensive comparison to a variety of baselines, which demonstrates very significant improvements in prediction accuracy, in the range 17% - 49% compared to the current state-of-the-art.

[Paper](https://www.researchgate.net/profile/Linyi_Yang2/publication/340385140_HTML_Hierarchical_Transformer-based_Multi-task_Learning_for_Volatility_Prediction/links/5e85efd8299bf1307972bc3d/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction.pdf) [Code](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction) [DOI](https://doi.org/10.1145/3366423.3380128)
</div>
</div>


<!--

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">EMNLP2022</div><img src='images/500x300.png' height="100%" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
-->
___* denotes equal contribution___
___# denotes corresponding author___

# 📝 Publications

- (36) PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. [\[paper\]](https://arxiv.org/abs/2306.04528); [![](https://img.shields.io/github/stars/microsoft/promptbench?style=social&label=Code+Stars)](https://github.com/microsoft/promptbench)
  
  Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, **Linyi Yang**, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie.

  CONFERENCE ON LANGUAGE MODELING 2024 (<font color=Blue>``Submission to COLM 2024``</font>).

- (35) Detoxifying Large Language Models via Knowledge Editing. [\[paper\]](https://arxiv.org/pdf/2403.14472)

  Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, **Linyi Yang**, Jindong Wang, Huajun Chen.

  The 62nd Annual Meeting of the Association for Computational Linguistics (<font color=Blue>``ACL 2024``</font>).

- (34) Deepfake text detection in the wild. [\[paper\]](https://arxiv.org/pdf/2305.13242)

  Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Longyue Wang, **Linyi Yang**, Shuming Shi, Yue Zhang.

  The 62nd Annual Meeting of the Association for Computational Linguistics (<font color=Blue>``ACL 2024``</font>).

- (33) A Rationale-centric Counterfactual Data Augmentation Method for Cross-Document Event Coreference Resolution. [\[paper\]](https://arxiv.org/abs/2404.01921)
  
  Bowen Ding, Qingkai Min, Shengkun Ma, Yingjie Li, **Linyi Yang†**, Yue Zhang.
  
  Annual Conference of the North American Chapter of the Association for Computational Linguistics 2024 (<font color=Blue>``NAACL 2024``</font>).

- (32) PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization. [\[paper\]](https://arxiv.org/abs/2306.05087); [![](https://img.shields.io/github/stars/WeOpenML/PandaLM?style=social&label=Code+Stars)](https://github.com/WeOpenML/PandaLM)
  
  Yidong Wang, Zhuohao Yu, Zhengran Zeng, **Linyi Yang**, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, Yue Zhang.
  
  International Conference on Learning Representations 2024 (<font color=Blue>``ICLR 2024``</font>).

- (31) Supervised Knowledge Makes Large Language Models Better In-context Learners. [\[paper\]](https://arxiv.org/pdf/2312.15918.pdf)
  
  **Linyi Yang**, Shuibai Zhang, Zhuohao Yu, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, Yue Zhang.
  
  International Conference on Learning Representations 2024 (<font color=Blue>``ICLR 2024``</font>).

- (30) Fast-detectgpt: Efficient zero-shot detection of machine-generated text via conditional probability curvature. [\[paper\]](https://arxiv.org/abs/2310.05130.pdf)
  
  Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, Yue Zhang.
  
  International Conference on Learning Representations 2024 (<font color=Blue>``ICLR 2024``</font>).

- (29) LLMs with Chain-of-Thought Are Non-Causal Reasoners. [\[paper\]](https://arxiv.org/pdf/2402.16048)
 
   Guangsheng Bao, Hongbo Zhang, **Linyi Yang**, Cunxiang Wang, Yue Zhang.

   arXiv preprint 2024 (<font color=Blue>``Arxiv 2024``</font>). 

- (28) A Survey on Evaluation of Large Language Models. [\[paper\]](https://arxiv.org/abs/2307.03109); [![](https://img.shields.io/github/stars/MLGroupJLU/LLM-eval-survey?style=social&label=Code+Stars)](https://github.com/MLGroupJLU/LLM-eval-survey)

  Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, **Linyi Yang**, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, Xing Xie.

  Transactions on Intelligent Systems and Technology (<font color=Blue>``TIST 2024``</font>).

- (27) Out-of-Distribution Generalization in Natural Language Processing: Past, Present, and Future. [\[paper\]](https://openreview.net/pdf?id=ivSJdhcuTi)

  **Linyi Yang**, Yaoxian Song, Xuan Ren, Chenyang Lyu, Yidong Wang, Jingming Zhuo, Lingqiao Liu, Jindong Wang, Jennifer Foster, Yue Zhang.

  The 2023 Conference on Empirical Methods in Natural Language Processing (<font color=Blue>``EMNLP 2023``</font>).

- (26) Measuring Consistency in Text-based Financial Forecasting Models. [\[paper\]](https://arxiv.org/pdf/2305.08524)

  **Linyi Yang**,Yingpeng Ma, Yue Zhang.

  The 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (<font color=Blue>``ACL 2023``</font>).

- (25) GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-distribution Generalization Perspective. [\[paper\]](https://arxiv.org/abs/2211.08073); [![](https://img.shields.io/github/stars/YangLinyi/GLUE-X?style=social&label=Code+Stars)](https://github.com/YangLinyi/GLUE-X)

  **Linyi Yang**, Shuibai Zhang, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, Yue Zhang.

  Findings of the Association for Computational Linguistics: ACL 2023 (<font color=Blue>``ACL 2023``</font>).

- (24) Learning to Generalize for Cross-domain QA. [\[paper\]](https://arxiv.org/pdf/2305.08208)

  Yingjie Niu*, **Linyi Yang***, Ruihai Dong, Yue Zhang.

  Findings of the Association for Computational Linguistics: ACL 2023 (<font color=Blue>``ACL 2023``</font>).

- (23) Exploiting Rich Textual User-Product Context for Improving Personalized Sentiment Analysis. [\[paper\]](https://doras.dcu.ie/29140/1/2023.findings-acl.92.pdf)

  Chenyang Lyu, Linyi Yang, Yue Zhang, Yvette Graham, Jennifer Foster.

  Findings of the Association for Computational Linguistics: ACL 2023 (<font color=Blue>``ACL 2023``</font>).

- (22) On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective. [\[paper\]](https://arxiv.org/abs/2302.12095); [![](https://img.shields.io/github/stars/microsoft/robustlearn?style=social&label=Code+Stars)](https://github.com/microsoft/robustlearn/tree/main/chatgpt-robust)
  
  Jindong Wang, Xixu Hu, Wenxin Hou, Hao Chen, Runkai Zheng, **Yidong Wang**, Linyi Yang, Haojun Huang, Wei Ye, Xiubo Geng, Binxin Jiao, Yue Zhang, Xing Xie.
  
  Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models at ICLR 2023 (<font color=Blue>``RTML Workshop 2023``</font>).

- (21) SciMine: An Efficient Systematic Prioritization Model Based on Richer Semantic Information. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3539618.3591764)

  Fang Guo, Yun Luo, **Linyi Yang**, Yue Zhang.

  The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (<font color=Blue>``SIGIR 2023``</font>).

- (20) Graph-Based Video-Language Learning with Multi-Grained Audio-Visual Alignment. [\[paper\]](https://dl.acm.org/doi/proceedings/10.1145/3581783)

  Chenyang Lyu, Wenxi Li, Tianbo Ji, Longyue Wang, Liting Zhou, Cathal Gurrin, **Linyi Yang**, Yi Yu, Yvette Graham, Jennifer Foster.

  Proceedings of the 31st ACM International Conference on Multimedia (<font color=Blue>``MM 2023``</font>).

- (19) Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. [\[paper\]](https://arxiv.org/pdf/2310.07521)

  Cunxiang Wang, Xiaoze Liu, Yuanhao Yue, Xiangru Tang, Tianhang Zhang, Cheng Jiayang, Yunzhi Yao, Wenyang Gao, Xuming Hu, Zehan Qi, Yidong Wang, **Linyi Yang**, Jindong Wang, Xing Xie, Zheng Zhang, Yue Zhang.

  TIST (<font color=Blue>``TIST 2023``</font>).

- (18) NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-task Financial Forecasting. [\[paper\]](https://arxiv.org/abs/2201.01770)

  **Linyi Yang**, Jiazheng Li, Ruihai Dong, Yue Zhang, Barry Smyth.

  AAAI (<font color=Blue>``AAAI 2022``</font>).

- (17) A Rationale-Centric Framework for Human-in-the-loop Machine Learning. [\[paper\]](https://arxiv.org/pdf/2203.12918)

  **Jinghui Lu***, **Linyi Yang***, Brian Mac Namee, Yue Zhang.

  ACL (<font color=Blue>``ACL 2022``</font>).
  
- (16) FactMix: Using a Few Labeled In-domain Examples to Generalize to Cross-domain Named Entity Recognition. [\[paper\]](https://arxiv.org/abs/2208.11464)

  **Linyi Yang***, **Lifan Yuan***, Leyang Cui, Wenyang Gao, Yue Zhang.

  COLING (<font color=Blue>``COLING 2022``</font>).

- (15) USB: A Unified Semi-supervised Learning Benchmark for Classification. [\[paper\]](https://proceedings.neurips.cc/paper_files/paper/2022/file/190dd6a5735822f05646dc27decff19b-Paper-Datasets_and_Benchmarks.pdf)

  Yidong Wang, Hao Chen, Yue Fan, Wang Sun, Ran Tao, Wenxin Hou, Renjie Wang, **Linyi Yang**, Zhi Zhou, Lan-Zhe Guo, Heli Qi, Zhen Wu, Yu-Feng Li, Satoshi Nakamura, Wei Ye, Marios Savvides, Bhiksha Raj, Takahiro Shinozaki, Bernt Schiele, Jindong Wang, Xing Xie, Yue Zhang.

  NeurIPS Dataset and Benchmark (<font color=Blue>``NeurIPS 2022``</font>).
  

# 🎤 Invited Talks
- Nanjing University, Nanjing, 2023
- MSRA, Online, 2023
- Shanghai AI Lab, Shanghai, China 2023
- MLNLP, Online, China 2022
- MSRA, Online, 2022





