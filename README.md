# Multi-source learning references
This is a reference paper list for multi-source learning.
## 1. Multitask Learning
### Surveys

- Zhang, Y., Wei, Y., & Yang, Q. (2018). Learning to multitask. NeurIPS. [paper](https://papers.nips.cc/paper/2018/file/aeefb050911334869a7a5d9e4d0e1689-Paper.pdf)
- Kouw, W. M., & Loog, M. (2018). An introduction to domain adaptation and transfer learning. *arXiv*. [paper](https://arxiv.org/abs/1812.11806)
- Vandenhende, S., Georgoulis, S., Van Gansbeke, W., Proesmans, M., Dai, D., & Van Gool, L. (2021). Multi-task learning for dense prediction tasks: A survey.T-PAMI. [paper](https://arxiv.org/abs/2004.13379) 
- Zhang, Y., & Yang, Q. (2021). A survey on multi-task learning. IEEE transactions on knowledge and data engineering. 
- Jiang et al. (2022). Transferability in deep learning: A survey.
- Lin, B., & Zhang, Y. (2022). LibMTL: A python library for multi-task learning. JMLR. [paper](https://arxiv.org/abs/2203.14338)
- Zhang, Z., Yu, W., Yu, M., Guo, Z., & Jiang, M. (2023). A survey of multi-task learning in natural language processing: Regarding task relatedness and training methods. EACL. [paper](https://aclanthology.org/2023.eacl-main.66.pdf)

### Multitask Learning Basics

- Caruana, R. (1997). Multitask learning. *Machine learning*. [paper](https://link.springer.com/article/10.1023/a:1007379606734)
- Caruana, R. (1996). Algorithms and applications for multitask learning. In *ICML*. [Paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3980c955f95092e527c580f9cfe066a17f752c08)
- Duong et al. (2015). Low resource dependency parsing: Cross-lingual parameter sharing in a neural network parser. In *ACL*. 
- Yang, Y., & Hospedales, T. (2016). Deep multi-task representation learning: A tensor factorisation approach. *ICLR.* [Paper](https://arxiv.org/abs/1605.06391)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv 2018. [paper](https://arxiv.org/pdf/1810.04805.pdf)
- Multi-task Sequence to Sequence Learning. ICLR 2016. [paper](https://arxiv.org/pdf/1511.06114)
- The natural language decathlon: Multitask learning as question answering.  arXiv 2019. [paper](https://arxiv.org/pdf/1806.08730)
- Multi-Task Deep Neural Networks for Natural Language Understanding. ACL 2019. [paper](https://arxiv.org/pdf/1901.11504)
- Understanding and Improving Information Transfer in Multi-Task Learning. ICLR 2020. [paper](https://openreview.net/pdf?id=SylzhkBtDB)

### 1.1 Multitask Learning Architectures

**Mixture-of-Experts**

- Ma et al. (2018). Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In *KDD*. [paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)
- Hazimeh, H., Zhao, Z., Chowdhery, A., Sathiamoorthy, M., Chen, Y., Mazumder, R., ... & Chi, E. (2021). Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning. *NeurIPS*. [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/f5ac21cd0ef1b88e9848571aeb53551a-Paper.pdf)
- Chen, Z., Shen, Y., Ding, M., Chen, Z., Zhao, H., Learned-Miller, E., & Gan, C. (2023). Mod-Squad: Designing mixtures of experts as modular multi-task learners. CVPR. [paper](https://arxiv.org/pdf/2212.08066)
- Haldar, S., Peng, Z., & Pinto, L. (2024). BAKU: An efficient Transformer for multi-task policy learning. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ff887781480973bd3cb6026feb378d1e-Paper-Conference.pdf)
- Yang, E., Wang, Z., Shen, L., Liu, S., Guo, G., Wang, X., & Tao, D. (2024). AdaMerging: Adaptive model merging for multi-task learning. ICLR. [paper](https://arxiv.org/pdf/2310.02575)

**Branching**
- Lu et al. (2017). Fully-adaptive feature sharing in multi-task networks with applications in person attribute classification. In *CVPR*. [paper](https://arxiv.org/abs/1611.05377)
- Huang, et al. (2018). Gnas: A greedy neural architecture search method for multi-attribute learning. In *ACM MM*. 
- Ruder et al. (2019). Latent multi-task architecture learning. In *AAAI*. 
- Guo et al. (2020). Learning to branch for multi-task learning. In *ICML*. [paper](https://arxiv.org/abs/2006.01895)
- Zhu, X., Qi, Y., Zhu, Y., Walters, R., & Platt, R. (2026). EquAct: An SE(3)-equivariant multi-task Transformer for 3D robotic manipulation. ICLR. [paper](https://openreview.net/attachment?id=d1wuA8oIH0&name=pdf)
- Li, D., Zhang, Z., Duan, M., Dobriban, E., & Zhang, H. R. (2026). Efficiently learning branching networks for multitask algorithmic reasoning. KDD. [paper](https://arxiv.org/abs/2512.01113)

**Hard-parameter sharing and soft-parameter sharing**

- Cross-stitch Networks for Multi-task Learning. CVPR 2016. [paper](https://arxiv.org/abs/1604.03539)
- Gated multi-task network for text classification. NAACL 2018. [paper](https://www.aclweb.org/anthology/N18-2114.pdf)
- A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks. EMNLP 2017. [paper](https://arxiv.org/pdf/1611.01587)
- Liu et al. (2019). End-to-end multi-task learning with attention. In CVPR. [paper](https://arxiv.org/abs/1803.10704)
- Learning Multiple Tasks with Multilinear Relationship Networks. NIPS 2017. [paper](https://arxiv.org/abs/1506.02117)
- Pilault, J., El Hattami, A., & Pal, C. (2021). Conditionally adaptive multi-task learning: Improving transfer learning in NLP using fewer parameters & less data. ICLR. [paper](https://arxiv.org/pdf/2009.09139)
- Zhang, L., Liu, X., & Guan, H. (2022). AutoMTL: A programming framework for automating efficient multi-task learning. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/dd3bd4e35cdd224ed2153a996f41077f-Paper-Conference.pdf)
- Li, D., Zhang, Z., Wang, L., & Zhang, H. R. (2025). Efficient ensemble for fine-tuning language models on multiple datasets. ACL. [paper](https://arxiv.org/abs/2505.21930)




### 1.2 Optimization Methods for Multitask Learning

- Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR 2018. [paper](https://arxiv.org/pdf/1705.07115)
- Wang, Z., Tsvetkov, Y., Firat, O., & Cao, Y. (2021). Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. ICLR. [paper](https://arxiv.org/pdf/2010.05874)
- Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-averse gradient descent for multi-task learning. NeurIPS. [paper](https://arxiv.org/pdf/2110.14048)
- Xin, D., Ghorbani, B., Garg, A., Firat, O., & Gilmer, J. (2022). Do current multi-task optimization methods in deep learning even help? NeurIPS. [paper](https://arxiv.org/pdf/2209.11379)
- Liu, B., Feng, Y., Stone, P., & Liu, Q. (2023). FAMO: Fast adaptive multitask optimization. NeurIPS. [paper](https://arxiv.org/pdf/2306.03792)
- Grigsby, J., Sasek, J., Parajuli, S., Adebi, D., Zhang, A., & Zhu, Y. (2024). AMAGO-2: Breaking the multi-task barrier in meta-reinforcement learning with Transformers. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9f40d2612d0b3b6f6c2a77da21b1067f-Paper-Conference.pdf)
- Saif, A. F. M., Chen, L., Cui, X., Lu, S., Kingsbury, B., & Chen, T. (2025). Objective soups: Multilingual multi-task modeling for speech processing. NeurIPS. [paper](https://arxiv.org/pdf/2508.09228)
- Dimitriadis, N., Frossard, P., & Fleuret, F. (2025). Pareto low-rank adapters: Efficient multi-task learning with preferences. ICLR. [paper](https://openreview.net/attachment?id=icDoYdUhRa&name=pdf)
- Nauman, M., Cygan, M., Sferrazza, C., Kumar, A., & Abbeel, P. (2025). Bigger, regularized, categorical: High-capacity value functions are efficient multi-task learners. NeurIPS. [paper](https://arxiv.org/pdf/2505.23150)

### 1.3 Representation Methods for Multitask Learning
**Multitask representation learning and transfer**
- Meyerson, E., & Miikkulainen, R. (2021). The traveling observer model: Multi-task learning through spatial variable embeddings. ICLR. [paper](https://arxiv.org/pdf/2010.02354)
- Fu, Y., Zhang, Y., Qian, K., Ye, Z., Yu, Z., Lai, C.-I., & Lin, Y. C. (2022). Losses can be blessings: Routing self-supervised speech representations towards efficient multilingual and multitask speech processing. NeurIPS. [paper](https://arxiv.org/pdf/2211.01522)
- Hua, P., Chen, Y., & Xu, H. (2023). Simple emergent action representations from multi-task policy training. ICLR. [paper](https://arxiv.org/pdf/2210.09566)
- Sun, Y., Ma, S., Madaan, R., Bonatti, R., Huang, F., & Kapoor, A. (2023). SMART: Self-supervised multi-task pretraining with control transformers. ICLR. [paper](https://arxiv.org/pdf/2301.09816)
- Brandfonbrener, D., Nachum, O., & Bruna, J. (2023). Inverse dynamics pretraining learns good representations for multitask imitation. NeurIPS. [paper](https://arxiv.org/pdf/2305.16985)
- Vafidis, P., Bhargava, A., & Rangel, A. (2025). Disentangling representations through multi-task learning. ICLR. [paper](https://arxiv.org/abs/2407.11249)

### 1.4 Task Relation Modeling

**Theoretical notions of task relatedness.** 

- Ben-David, S., & Schuller, R. (2003). Exploiting task relatedness for multiple task learning. In *Learning Theory and Kernel Machines*. [paper](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_41)
- Ben-David et al. (2010). A theory of learning from different domains. *Machine learning* [paper](https://link.springer.com/article/10.1007/s10994-009-5152-4)
- Hanneke, S., & Kpotufe, S. (2019). On the value of target data in transfer learning. *Advances in Neural Information Processing Systems*. [Paper](https://proceedings.neurips.cc/paper/2019/hash/b91f4f4d36fa98a94ac5584af95594a0-Abstract.html)
- Du et al. (2020). Few-shot learning via learning the representation, provably. *ICLR*. [paper](https://arxiv.org/abs/2002.09434)

**Measurements in deep neural networks.**

Gradients

- Yu et al. (2020). Gradient surgery for multi-task learning. *NeurIPS.* [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- Dery et al.  (2021). Auxiliary task update decomposition: The good, the bad and the neutral. *ICLR.* [paper](https://arxiv.org/abs/2108.11346)
- Chen et al. (2021). Weighted training for cross-task learning. *ICLR.* [paper](https://arxiv.org/abs/2105.14095)

Predicted probabilities between tasks

- Nguyen et al (2020). Leep: A new measure to evaluate transferability of learned representations. *ICML.* [Paper](https://proceedings.mlr.press/v119/nguyen20b.html)

- Identifying beneficial task relations for multi-task learning in deep neural networks. EACL 2017. [paper](https://www.aclweb.org/anthology/E17-2026.pdf) 

Task affinity and grouping

- Standley et al. (2020). Which tasks should be learned together in multi-task learning? *ICML*. [paper](https://proceedings.mlr.press/v119/standley20a.html)
- Fifty et al. (2021). Efficiently identifying task groupings for multi-task learning. *NeuIPS.* [Paper](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)
- Cao, K., You, J., & Leskovec, J. (2022). Relational multi-task learning: Modeling relations between data and tasks. ICLR. [paper](https://arxiv.org/pdf/2303.07666)
- Meng, F., Shao, W., Peng, Z., Jiang, C., Zhang, K., Qiao, Y., & Luo, P. (2023). Foundation model is efficient multimodal multitask model selector. NeurIPS. [paper](https://arxiv.org/pdf/2308.06262)
- Li, D., Nguyen, H. L., & Zhang, H. R. (2023). Identification of negative transfers in multitask learning using surrogate models. TMLR. [paper](https://arxiv.org/pdf/2303.14582)
- Li, D., Ju, H., Sharma, A., & Zhang, H. R. (2023). Boosting multitask learning on graphs through higher-order task affinities. KDD. [paper](https://arxiv.org/pdf/2306.14009)
- Li, D., Sharma, A., & Zhang, H. R. (2024). Scalable multitask learning using gradient-based estimation of task affinity. KDD. [paper](https://arxiv.org/pdf/2409.06091)
- Li, D., Zhang, Z., Wang, L., & Zhang, H. R. (2024). Scalable fine-tuning from multiple data sources: A first-order approximation approach. Findings of EMNLP. [paper](https://arxiv.org/pdf/2409.19458)
- Cui, S., & Mitra, P. (2024). Automated multi-task learning for joint disease prediction on electronic health records. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/e955bd39051d4d080826b2ffcf3a7fe2-Paper-Conference.pdf)
- Zhang, Z., Zhang, Z., Li, D., Wang, L., Dy, J., & Zhang, H. R. (2025). Linear-time demonstration selection for in-context learning via gradient estimation. EMNLP. [paper](https://arxiv.org/pdf/2508.19999)
- Guo, S., Chen, J., Hu, Z., Chen, Z., Yang, W., Lin, Y., Jiang, X., & Duan, L. (2025). Which tasks should be compressed together? A causal discovery approach for efficient multi-task representation compression. ICLR. [paper](https://openreview.net/attachment?id=x33vSZUg0A&name=pdf)
- Ajirak, M., Bein, O., Bowen, E. R., Kanellopoulos, D., Falk, A., Gunning, F. M., Solomonov, N., & Grosenick, L. (2025). Learning to route: Per-sample adaptive routing for multimodal multitask prediction. NeurIPS. [paper](https://arxiv.org/pdf/2509.12227)
- Ayman, A., Mukhopadhyay, A., & Laszka, A. (2026). Ensemble prediction of task affinity for efficient multi-task learning. ICLR. [paper](https://arxiv.org/pdf/2602.18591)


### 1.5 Data-centric and Probabilistic in Multitask Learning
- Yu, T., Kumar, A., Chebotar, Y., Hausman, K., Levine, S., & Finn, C. (2021). Conservative data sharing for multi-task offline reinforcement learning. NeurIPS. [paper](https://arxiv.org/pdf/2109.08128)
- Chen, X., Sun, H., Ellington, C., Xing, E., & Song, L. (2021). Multi-task learning of order-consistent causal graphs. NeurIPS. [paper](https://arxiv.org/pdf/2111.02545)
- Aribandi, V., Tay, Y., Schuster, T., Rao, J., Zheng, H. S., Mehta, S. V., Zhuang, H., Tran, V. Q., Bahri, D., Ni, J., Gupta, J., Hui, K., Ruder, S., & Metzler, D. (2022). ExT5: Towards extreme multi-task scaling for transfer learning. ICLR. [paper](https://arxiv.org/pdf/2111.10952)
- Vietri, G., Archambeau, C., Aydore, S., Brown, W., Kearns, M., Roth, A., Siva, A., Tang, S., & Wu, Z. S. (2022). Private synthetic data for multitask learning and marginal queries. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7428310c0f97f1c6bb2ef1be99c1ec2a-Paper-Conference.pdf)
- Makino, T., Geras, K. J., & Cho, K. (2022). Generative multitask learning mitigates target-causing confounding. NeurIPS. [paper](https://arxiv.org/pdf/2202.04136)
- Negrel, H., Coeurdoux, F., Albergo, M. S., & Vanden-Eijnden, E. (2025). Multitask learning with stochastic interpolants. NeurIPS. [paper](https://openreview.net/pdf?id=9k9ZsDs9Vc)
- Kumar, A., Agarwal, R., Geng, X., Tucker, G., & Levine, S. (2023). Offline Q-learning on diverse multi-task data both scales and generalizes. ICLR. [paper](https://arxiv.org/pdf/2211.15144)
- Knight, P., & Duan, R. (2023). Multi-task learning with summary statistics. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/a924b7178e5975dfed1de235f0b72973-Paper-Conference.pdf)
- Georgiev, I., Giridhar, V., Hansen, N., & Garg, A. (2025). PWM: Policy learning with multi-task world models. ICLR. [paper](https://arxiv.org/pdf/2407.02466)
- Jiang, Y., Feng, Z., & Mehta, A. (2026). Incentive-aligned multi-source LLM summaries. ICLR. [paper](https://openreview.net/attachment?id=OlidGx8oKr&name=pdf)
- Zeng, Y., Maus, N., Jones, H. T., Tao, J., Wan, F., Torres, M. D. T., de la Fuente-Nunez, C., Marcus, R., Bastani, O., & Gardner, J. R. (2026). Scaling multi-task Bayesian optimization with large language models. ICLR. [paper](https://openreview.net/attachment?id=ekmUkRYnkN&name=pdf)

### 1.6 Theoretical Analysis of Multitask Learning
- Tiomoko, M., Ali, H. T., & Couillet, R. (2021). Deciphering and optimizing multi-task learning: A random matrix approach. ICLR. [paper](https://openreview.net/pdf?id=Cri3xz59ga)
- Zhang, C., & Wang, Z. (2021). Provably efficient multi-task reinforcement learning with model transfer. NeurIPS. [paper](https://arxiv.org/pdf/2107.08622)
- Wan, R., Ge, L., & Song, R. (2021). Metadata-based multi-task bandits with Bayesian hierarchical models. NeurIPS. [paper](https://arxiv.org/pdf/2108.06422)
- Cheng, Y., Feng, S., Yang, J., Zhang, H., & Liang, Y. (2022). Provable benefit of multitask representation learning in reinforcement learning. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/cde328b7bf6358f5ebb91fe9c539745e-Paper-Conference.pdf)
- Von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., & Vladymyrov, M. (2023). Transformers learn in-context by gradient descent. ICML. [paper](https://proceedings.mlr.press/v202/von-oswald23a/von-oswald23a.pdf)
- Lu, R., Zhao, A., Du, S. S., & Huang, G. (2022). Provable general function class representation learning in multitask bandits and MDPs. NeurIPS. [paper](https://arxiv.org/pdf/2205.15701)
- Daras, G., Raoof, N., Gkalitsiou, Z., & Dimakis, A. G. (2022). Multitasking models are robust to structural failure: A neural model for bilingual cognitive reserve. NeurIPS. [paper](https://arxiv.org/pdf/2210.11618)
- Watkins, A., Ullah, E., Nguyen-Tang, T., & Arora, R. (2023). Optimistic rates for multi-task representation learning. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/06e3c330d140f3a25671acf2dc2d6357-Paper-Conference.pdf)
- Lippl, S., & Lindsey, J. (2024). Inductive biases of multi-task learning and finetuning: Multiple regimes of feature reuse. NeurIPS. [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7346ff79699b5bba26f8af89e700a95-Paper-Conference.pdf)
- Duong, T., Wang, Z., & Zhang, C. (2024). Beyond task diversity: Provable representation transfer for sequential multi-task linear bandits. NeurIPS. [paper](https://arxiv.org/pdf/2501.13390)
- Ishfaq, H., Nguyen-Tang, T., Feng, S., Arora, R., Wang, M., Yin, M., & Precup, D. (2024). Offline multitask representation learning for reinforcement learning. NeurIPS. [paper](https://arxiv.org/pdf/2403.11574)

### Benchmarks

- [GLUE](https://gluebenchmark.com/) & [SuporGLUE](https://huggingface.co/datasets/aps/super_glue): Natural Language Understanding
- [decaNLP](https://decanlp.com/): 10 NLP Tasks
- [MMLU](https://huggingface.co/datasets/cais/mmlu): 59 tasks consisting of multiple-choice questions from various branches of knowledge.

### Softwares and Open-source Libraries

- [LibMTL]( https://github.com/median-research-group/LibMTL): an open-source library built on PyTorch for mulitask learning. 
- GradEx, EnsembleLoRA, and GradSel.

## 2. Meta Learning

### Survey

Meta-Learning in Neural Networks: A Survey. [paper](https://arxiv.org/pdf/2004.05439.pdf)

### Black-Box Approaches

#### Recurrent Neural Network

(MANN) Meta-learning with memory-augmented neural networks. ICML 2016. [paper](http://proceedings.mlr.press/v48/santoro16.pdf)

#### Attention-Based Network

Matching Networks for One-Shot Learning. NIPS 2016. [paper](https://arxiv.org/abs/1606.04080)

(SNAIL)  A Simple Neural Attentive Meta-Learner. ICLR 2018. [paper](https://arxiv.org/pdf/1707.03141.pdf)

### Optimization-Based Methods

(MAML) Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML 2017. [paper](https://arxiv.org/pdf/1703.03400.pdf) :star:

(Reptile; First-order method) On First-Order Meta-Learning Algorithms. arXiv 2018. [paper](https://arxiv.org/pdf/1803.02999.pdf)

#### Other Forms of Prior on MAML

(Implicit MAML) Meta-Learning with Implicit Gradients. NIPS 2019. [paper](https://arxiv.org/abs/1909.04630)

(Implicit Differentiation; SVM) Meta-Learning with Differentiable Convex Optimization. CVPR 2019. [paper](https://arxiv.org/abs/1904.03758)

(Bayesian linear regression) Meta-Learning Priors for Efficient Online Bayesian Regression. Workshop on the Algorithmic Foundations of Robotics 2018. [paper](https://arxiv.org/pdf/1807.08912)

(Ridge regression; Logistic regression) Meta-learning with Differentiable Closed-Form Solvers. ICLR 2019. [paper](https://openreview.net/pdf?id=HyxnZh0ct7) 

#### Understanding MAML

(MAML expressive power and university) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

(Map MAML to Bayes Framework) Recasting Gradient-Based Meta-Learning as Hierarchical Bayes. ICLR 2018. [paper](https://openreview.net/pdf?id=BJ_UL-k0b) 

#### Tricks to Optimize MAML

**Choose architecture that is effective for inner gradient-step**

Auto-Meta: Automated Gradient Based Meta Learner Search. NIPS 2018 Workshop on Meta-Learning. [paper](https://arxiv.org/pdf/1806.06927) 

**Automatically learn inner vector learning rate, tune outer learning rate**

Alpha MAML: Adaptive Model-Agnostic Meta-Learning. ICML 2019 Workshop on Automated Machine Learning. [paper](https://arxiv.org/pdf/1905.07435) 

Meta-SGD: Learning to Learn Quickly for Few-Shot Learning. arXiv 2017. [paper](https://arxiv.org/pdf/1707.09835.pdf)

**Optimize only a subset of the parameters in the inner loop**

(DEML) Deep Meta-Learning: Learning to Learn in the Concept Space. arXiv 2018. [paper](https://arxiv.org/pdf/1802.03596)

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

**Decouple inner learning rate, BN statistics per-step**

(MAML++) How to train your MAML. ICLR 2019. [paper](https://arxiv.org/pdf/1810.09502)

**Introduce context variables for increased expressive power**

(CAVIA) Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

(Bias transformation) Meta-Learning and Universality: Deep Representations and Gradient Descent can Approximate any Learning Algorithm. ICLR 2018. [paper](https://arxiv.org/abs/1710.11622)

## 3. Non-Parametric Methods via Metric Learning

Siamese Neural Networks for One-shot Image Recognition. ICML 2015. [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

Matching Networks for One Shot Learning. NIPS 2016. [paper](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)

Prototypical Networks for Few-shot Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.05175)

**Learn non-linear relation module on embeddings**

Learning to Compare: Relation Network for Few-Shot Learning. CVPR 2018. [paper](https://arxiv.org/pdf/1711.06025)

**Learn infinite mixture of prototypes**

Infinite Mixture Prototypes for Few-Shot Learning. ICML 2019. [paper](https://arxiv.org/pdf/1902.04552.pdf)

**Perform message passing on embeddings**

Few-Shot Learning with Graph Neural Networks ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

## 4. Bayesian Meta-Learning & Generative Models

#### Amortized Inference

Amortized Bayesian Meta-Learning. ICLR 2019. [paper](https://openreview.net/pdf?id=rkgpy3C5tX)

#### Ensemble Method

Bayesian Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.03836)

#### Sampling & Hybrid Inference

Probabilistic Model-Agnostic Meta-Learning. NIPS 2018. [paper](https://arxiv.org/pdf/1806.02817)

Meta-Learning Probabilistic Inference for Prediction. ICLR 2019. [paper](https://openreview.net/pdf?id=HkxStoC5F7)

### Hybrid meta-learning approaches

Meta-Learning with Latent Embedding Optimization. ICLR 2019. [paper](https://arxiv.org/pdf/1807.05960)

Fast Context Adaptation via Meta-Learning. ICML 2019. [paper](https://arxiv.org/pdf/1810.03642)

Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples. ICLR 2020. [paper](https://arxiv.org/pdf/1903.03096)

Few-Shot Learning with Graph Neural Networks. ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043)

(CAML) Learning to Learn with Conditional Class Dependencies. ICLR 2019. [paper](https://openreview.net/pdf?id=BJfOXnActQ)

## 5. Meta Reinforcement Learning

#### Policy Gradient RL

**MAML and Black-Box Meta Learning Approaches can be directly applied to Policy-Gradient RL methods**

#### Value-Based RL

**It is not easy to applied existing meta learning approaches to Value-Based RL because Value-Based RL is dynamic programming method** 

Meta-Q-Learning. ICLR 2020. [paper](https://openreview.net/pdf?id=SJeD3CEFPH)

(Goal-Conditioned RL with hindsight relabeling)/(Multi-Task RL) Hindsight Experience Replay. NIPS 2017. [paper](https://papers.nips.cc/paper/7090-hindsight-experience-replay.pdf) 

(better learning) Learning Latent Plans from Play. CoRL 2019. [paper](https://arxiv.org/pdf/1903.01973)

(learn a better goal representation) 

Universal Planning Networks. ICML 2018. [paper](http://proceedings.mlr.press/v80/srinivas18b/srinivas18b.pdf)

Unsupervised Visuomotor Control through Distributional Planning Networks. RSS 2019. [paper](https://arxiv.org/pdf/1902.05542.pdf)

## 6. Applications

Meta-Learning for Low-Resource Neural Machine Translation. EMNLP 2018. [paper](https://arxiv.org/pdf/1808.08437)

Few-shot Autoregressive Density Estimation: Towards Learning to Learn Distributions. ICLR 2018. [paper](https://arxiv.org/pdf/1710.10304)

One-Shot Imitation Learning. NIPS 2017. [paper](https://arxiv.org/pdf/1703.07326.pdf)

Massively Multitask Networks for Drug Discovery. ICML 2015. [paper](https://arxiv.org/pdf/1502.02072.pdf)

