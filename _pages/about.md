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

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2023</div><img src='images/glue-x.jpg' height="100%" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**GLUE-X: Evaluating Natural Language Understanding Models from an Out-of-Distribution Generalization Perspective**

**Linyi Yang**, Shuibai Zhang, Libo Qin, Yafu Li, Yidong Wang, Hanmeng Liu, Jindong Wang, Xing Xie, Yue Zhang

Pre-trained language models (PLMs) are known to improve the generalization performance of natural language understanding models by leveraging large amounts of data during the pre-training phase. However, the out-of-distribution (OOD) generalization problem remains a challenge in many NLP tasks, limiting the real-world deployment of these methods. This paper presents the first attempt at creating a unified benchmark named GLUE-X for evaluating OOD robustness in NLP models, highlighting the importance of OOD robustness and providing insights on how to measure the robustness of a model and how to improve it. The benchmark includes 15 publicly available datasets for OOD testing, and evaluations are conducted on 8 classic NLP tasks over 21 popularly used PLMs. Our findings confirm the need for improved OOD accuracy in NLP tasks, as significant performance degradation was observed in all settings compared to in-distribution (ID) accuracy.

[Paper](https://arxiv.org/pdf/2211.08073.pdf) [Code](https://github.com/YangLinyi/GLUE-X)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">WWW 2020</div><img src='images/html.jpg' height="100%" width="100%"></div></div>
<div class='paper-box-text' markdown="2">

**HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction**

**Linyi Yang**, James Ng, Ruihai Dong, Barry Smyth

This paper proposes a novel hierarchical, transformer, multi-task architecture designed to harness the text and audio data from quarterly earnings conference calls to predict future price volatility in the short and long term. This includes a comprehensive comparison to a variety of baselines, demonstrating significant improvements in prediction accuracy, in the range of 17% - 49% compared to the current state-of-the-art.

[Paper](https://www.researchgate.net/profile/Linyi_Yang2/publication/340385140_HTML_Hierarchical_Transformer-based_Multi-task_Learning_for_Volatility_Prediction/links/5e85efd8299bf1307972bc3d/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction.pdf) [Code](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction) [DOI](https://doi.org/10.1145/3366423.3380128)
</div>
</div>

<!--

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">EMNLP2022</div><img src='images/Fig.png' height="100%" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Paper**

**Linyi Yang\***, Author\* -> equal contribution
**Linyi Yang**, Author -> single first

[Paper]() [Code]() [Slides]() [Video]()...
</div>
</div>

 -->

___* denotes equal contribution___



