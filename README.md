# QVEGE
Q-learning based vegetation evolution for numerical optimization and wireless sensor network coverage optimization

## Highlights
• We propose a novel Q-learning based Vegetation Evolution (QVEGE) for numerical optimization.  
• Q-learning determines the search operator in the growth period and maturity period of VEGE intelligently.  
• The performance of QVEGE is investigated on CEC2020 functions, engineering problems, and WSN coverage optimization problems.  
• Experimental and statistical results verify the efficiency and effectiveness of our proposed QVEGE.  

## Abstract
Vegetation evolution (VEGE) is a newly proposed meta-heuristic algorithm (MA) with excellent exploitation but relatively weak exploration capacity. We thus focus on further balancing the exploitation and the exploration of VEGE well to improve the overall optimization performance. This paper proposes an improved Q-learning based VEGE, and we design an exploitation archive and an exploration archive to provide a variety of search strategies, each archive contains four efficient and easy-implemented search strategies. In addition, online Q-Learning, as well as $\varepsilon$-greedy scheme, are employed as the decision-maker role to learn the knowledge from the past optimization process and determine the search strategy for each individual automatically and intelligently. In numerical experiments, we compare our proposed QVEGE with eight state-of-the-art MAs including the original VEGE on CEC2020 benchmark functions, twelve engineering optimization problems, and wireless sensor networks (WSN) coverage optimization problems. Experimental and statistical results confirm that the proposed QVEGE demonstrates significant enhancements and stands as a strong competitor among existing algorithms. The source code of QVEGE is publicly available at https://github.com/RuiZhong961230/QVEGE.

## Citation
@article{Zhong:24,  
title = {Q-learning based vegetation evolution for numerical optimization and wireless sensor network coverage optimization},  
journal = {Alexandria Engineering Journal},  
volume = {87},  
pages = {148-163},  
year = {2024},  
issn = {1110-0168},  
doi = {https://doi.org/10.1016/j.aej.2023.12.028 },  
author = {Rui Zhong and Fei Peng and Jun Yu and Masaharu Munetomo},  
}

## Datasets and Libraries
CEC benchmarks are provided by the opfunu libraries and engineering problems are provided by the enoppy library.
