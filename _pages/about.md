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

I am currently serving as a Research Assistant Professor within the [Westlake NLP group](https://westlakenlp.netlify.app/), under the mentorship of Yue Zhang. My academic journey culminated in a PhD from the Insight Centre at University College Dublin, where I had the privilege of working alongside [Barry Smyth](https://people.ucd.ie/barry.smyth) and Ruihai Dong.

My research interests lie at the confluence of Artificial Intelligence (AI) for Finance and Natural Language Processing (NLP), particularly in enhancing the robustness of neural networks against out-of-distribution data and augmenting their causal reasoning capabilities for practical applications. Currently, I focus on developing causality-guided approaches within NLP, aiming to foster innovations in the high-stake sectors.

Should you wish to inquire further about my research, I welcome you to contact me via email.

In addition, I am keen on mentoring and working with highly motivated interns who possess strong programming capabilities. If you are interested in collaborating, please do not hesitate to send your CV to my email.

# Honors and Awards

- Outstanding Postdoctoral Representative, 2023.
- Outstanding Postdoc Researcher, 2022.
- Outstanding Self-financed Students Abroad (Applicable for Non-CSC PhDs; Only One Type-B Winner in Ireland), 2021.
- Best Paper Candidate, CCIS, 2018.

# News
- 2024-Jan 🔥🔥 Three papers have been accepted to [ICLR 2024](https://iclr.cc/).
- 2023-Dec One paper has been accepted to [EMNLP 2023](https://2023.emnlp.org/).
- 2023-Nov Organized ACM TIST Special Issue on Evaluations of Large Language Model with Dr. Jindong Wang and Prof. Qiang Yang.
- 2023-May 🔥🔥 Four papers have been accepted to [ACL 2023](https://2023.aclweb.org/) (Three co-first author papers and one second-author paper).
- 2023-Apr Our paper discussing the robustness of ChatGPT has been accepted to [ICLR 2023](https://arxiv.org/abs/2302.12095) Workshop.
- Area Chair / Senior Programme Committee (SPC): EMNLP-22; IJCAI-23.
- PC Member/Reviewer: CIKM-20; COLING-20; ACL-21; SIGIR-21; CKIM-21; EMNLP-21; IEEE-Access.
- 2022-Dec: 🎉🎉 I received Outstanding Postdoctoral Fellows from Westlake University and gave a talk as the only postdoctoral representative.
- 2022-Sep: One paper co-operating with MSRA has been accepted to [NeurIPS 2022](https://arxiv.org/pdf/2208.07204/). The first author was my intern at Westlake University. Big congrats! (core: A*, CCF: A)
- 2022-Aug: Two papers (one first-author paper) have been accepted to [COLING 2022](https://coling2022.org/). (core: A, CCF: B)
- 2022-Mar: One co-first author long paper has been accepted to [ACL 2022](https://www.2022.aclweb.org/) main conference. (core: A*, CCF: A)
- 2022-Jan: One first-author long paper has been accepted to [AAAI 2022](https://www.aaai.org/AAAI22Papers/AAAI-4799.YangL.pdf) (15% acceptance rate). (core: A*, CCF: A)
- 2022-Jan Invited to serve as an Area Chair (AC) at EMNLP-22.
- We start working on tracking the progress in the topic of FinNLP. Feel free to add any relevant items to [Project Link](https://github.com/YangLinyi/FinNLP-Progress)
- 2021-May: One first-author long paper has been accepted to [ACL 2021](https://2021.aclweb.org/) (21% acceptance rate). (core: A*, CCF: A)
- 2020-Oct: One first-author long paper has been accepted to [COLING 2020](https://coling2020.org/) (Oral, Top 5% submissions). (core: A, CCF: B)
- 2020-Sep: One co-first author resource paper has been accepted to [CIKM 2020](https://www.cikm2020.org/accepted-papers/accepted-resource-track-papers/) (20% acceptance rate). (core: A, CCF: B)
- 2019-Dec: One first-author long paper has been accepted to [WWW 2020](https://www2020.citi.sinica.edu.tw/schedule/) (19% acceptance rate). (core: A*, CCF: A)
- 2019-Aug: One first-author long paper has been accepted to [FinNLP Workshop@IJCAI-19](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp/) (Oral).
- 2018-Nov: Our paper won the best paper nomination at [CCIS 2018](http://ccis2018.csp.escience.cn/dct/page/1) (Best Paper Candidate).
- 2017-Dec: My first paper was published at [AICS 2017](http://aiai.ucd.ie/aics2017/index.html).

# 📝 Selected Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024</div><img src='images/ICLR_24.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Supervised Knowledge Makes Large Language Models Better In-context Learners**

**Linyi Yang***, Shuibai Zhang*, Zhuohao Yu*, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, Yue Zhang

While previous in-context learning research has focused on enhancing models to adhere to users' specific instructions and quality expectations, and to avoid undesired outputs, little to no work has explored the use of task-specific fine-tuned Language Models (SLMs) to improve LLMs' in-context learning during the inference stage. Our primary contribution is the establishment of a simple yet effective framework that enhances the reliability of LLMs as it: 1) generalizes out-of-distribution data, 2) elucidates how LLMs benefit from discriminative models, and 3) minimizes hallucinations in generative tasks. Using our proposed plug-in method, enhanced versions of Llama 2 and ChatGPT surpass their original versions regarding generalizability and factuality. We offer a comprehensive suite of resources, including 16 curated datasets, prompts, model checkpoints, and LLM outputs across 9 distinct tasks. Our empirical analysis sheds light on the advantages of incorporating discriminative models into LLMs and highlights the potential of our methodology in fostering more reliable LLMs. 

[Paper](https://openreview.net/forum?id=bAMPOUF227) [Code](https://github.com/YangLinyi/Supervised-Knowledge-Makes-Large-Language-Models-Better-In-context-Learners)

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">EMNLP 2023</div><img src='images/ood_survey.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Out-of-Distribution Generalization in Natural Language Processing: Past, Present, and Future**

**Linyi Yang**, Yaoxiao Song, Xuan Ren, Chenyang Lyu, Yidong Wang, Lingqiao Liu, Jindong Wang, Jennifer Foster, Yue Zhang

Machine learning (ML) systems in natural language processing (NLP) face significant challenges in generalizing to out-of-distribution (OOD) data, where the test distribution differs from the training data distribution. This poses important questions about the robustness of NLP models and their high accuracy, which may be artificially inflated due to their underlying sensitivity to systematic biases. Despite these challenges, there is a lack of comprehensive surveys on the generalization challenge from an OOD perspective in natural language understanding. Therefore, this paper aims to fill this gap by presenting the first comprehensive review of recent progress, methods, and evaluations on this topic. 

[Paper](https://aclanthology.org/2023.emnlp-main.276/) 

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/GLUE-X.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-Distribution Generalization Perspective**

**Linyi Yang***, Shuibai Zhang*, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, Yue Zhang

This paper presents the first attempt at creating a unified benchmark named GLUE-X for evaluating OOD robustness in NLP models, highlighting the importance of OOD robustness and providing insights on how to measure the robustness of a model and how to improve it. The benchmark includes 15 publicly available datasets for OOD testing, and evaluations are conducted on 8 classic NLP tasks over 21 popularly used PLMs. Our findings confirm the need for improved OOD accuracy in NLP tasks, as significant performance degradation was observed in all settings compared to in-distribution (ID) accuracy.

[Paper](https://arxiv.org/pdf/2211.08073.pdf) [Code](https://github.com/YangLinyi/GLUE-X)

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/fintrust.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Measuring Consistency in Text-based Financial Forecasting Models**

**Linyi Yang***, Yingpeng Ma*, Yue Zhang

Financial forecasting has been an important and active area of machine learning research, as even the most modest advantages in predictive accuracy can be parlayed into significant financial gains. Recent advances in natural language processing (NLP) bring the opportunity to leverage textual data, such as earnings reports of publicly traded companies, to predict the return rate for an asset. However, when dealing with such a sensitive task, the consistency of models – their invariance under meaning-preserving alternations in input – is a crucial property for building user trust. Despite this, current methods for financial forecasting do not take consistency into consideration. To address this issue, we propose FinTrust, an evaluation tool that assesses logical consistency in financial text. Using FinTrust, we show that the consistency of state-of-the-art NLP models for financial forecasting is poor. Our analysis of the performance degradation caused by meaning-preserving alternations suggests that current text-based methods are not suitable for robustly predicting market information.

[Paper](https://aclanthology.org/2023.acl-long.769/) [Code](https://github.com/yingpengma/fintrust)

</div>
</div>

<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/cross_qa.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**Learning to Generalize for Cross-domain QA**

Yingjie Niu*, **Linyi Yang***, Ruihai Dong, Yue Zhang

There have been growing concerns regarding the out-of-domain generalization ability of natural language processing (NLP) models, particularly in question-answering (QA) tasks. Current synthesized data augmentation methods for QA are hampered by increased training costs. To address this issue, we propose a novel approach that combines prompting methods and linear probing with fine-tuning strategy, which does not entail additional cost. Our method has been theoretically and empirically shown to be effective in enhancing the generalization ability of both generative and discriminative models. Our approach outperforms state-of-the-art baselines, with an average increase in F1 score of 4.5%-7.9%. Furthermore, our method can be easily integrated into any pre-trained models and offers a promising solution to the under-explored cross-domain QA task.

[Paper](https://aclanthology.org/2023.findings-acl.84/) [Code](https://github.com/FreddieNIU/Prompt-QA)

</div>
</div>


<!-- <div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAAI 2022</div></div></div> -->
<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AAAI 2022</div><img src='images/NumHTML.png' height="200"></div></div>
<div class='paper-box-text' markdown="1">

**NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-task Financial Forecasting**

**Linyi Yang**, Jiazheng Li, Ruihai Dong, Yue Zhang, Barry Smyth

Financial forecasting has been an important and active area of machine learning research because of the challenges it presents and the potential rewards that even minor improvements in prediction accuracy or forecasting may entail. Traditionally, financial forecasting has heavily relied on quantitative indicators and metrics derived from structured financial statements. Earnings conference call data, including text and audio, is an important source of unstructured data that has been used for various prediction tasks using deep earning and related approaches. However, current deep learning-based methods are limited in the way that they deal with numeric data; numbers are typically treated as plain-text tokens without taking advantage of their underlying numeric structure. This paper describes a numeric-oriented hierarchical transformer model to predict stock returns, and financial risk using multi-modal aligned earnings calls data by taking advantage of the different categories of numbers (monetary, temporal, percentages etc.) and their magnitude.

[Paper](https://arxiv.org/abs/2201.01770) [Code]([https://github.com/FreddieNIU/Prompt-QA](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction))

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

**Paper**

**Linyi Yang\***, Author\* -> equal contribution
**Linyi Yang**, Author -> single first

[Paper]() [Code]() [Slides]() [Video]()...
</div>
</div>

-->
___* denotes equal contribution___




