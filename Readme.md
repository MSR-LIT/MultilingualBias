# Gender Bias in Multilingual Embeddings and Cross-Lingual Transfer


## Introduction
This repository contains the code and data for replicating results from 
- [Gender Bias in Multilingua Embeddings and Corss-Lingual Transfer](https://arxiv.org/abs/2005.00699)
- [Jieyu Zhao](https://jyzhao.net/), [Subhabrata Mukherjee](https://www.microsoft.com/en-us/research/people/submukhe/), [Saghar Hosseini](https://www.microsoft.com/en-us/research/people/sahoss/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/), [Ahmed Hassan Awadallah](https://www.microsoft.com/en-us/research/people/hassanam/).
- In ACL 2020


## Intrinsic Bias

### - Prerequisite

- Download/Generate fastText aligned embeddings from [fastText](https://github.com/facebookresearch/fastText/tree/master/alignment)
- Generate bias-reduced EN embeddings (ENDEB) using [Hard-Debias](https://github.com/tolga-b/debiaswe)

### - Multilingual Intrinsic Bias Dataset:

 We include all the occupations as well as the gender seed words for each language under [intrinsic](./intrinsic) folder.

### - Codes:

To evaluate intrinsic bias in each language, refer to [inBias.ipynb](./intrinsic/inBias.ipynb) for bias analysis and results.


## Extrinsic Bias

### - Multilingual BiosBias (MLBs) Dataset:

To replicate the MLBs dataset, please refer to [replicateMLBs](./extrinsic/replicateMLBs) folder.
**For EN dataset, please refer to [biosbias](https://github.com/microsoft/biosbias)**

### - Codes:

The codes for downstream task is under [bios_codes](./extrinsic/bios_codes) folder.

**If you use this code or use the EN MLB dataset, please also cite [Bias in Bios: A Case Study of Semantic Representation Bias in a High Stakes Setting](https://dl.acm.org/doi/10.1145/3287560.3287572)**
```
@inproceedings{de2019bias,
  title={Bias in bios: A case study of semantic representation bias in a high-stakes setting},
  author={De-Arteaga, Maria and Romanov, Alexey and Wallach, Hanna and Chayes, Jennifer and Borgs, Christian and Chouldechova, Alexandra and Geyik, Sahin and Kenthapadi, Krishnaram and Kalai, Adam Tauman},
  booktitle={Proceedings of the Conference on Fairness, Accountability, and Transparency},
  pages={120--128},
  year={2019}
}
```


#### Citation
```
@inproceedings{de2019bias,
  title={Gender Bias in Multilingual Embeddings and Cross-Lingual Transfer},
  author={Jieyu Zhao, Subhabrata Mukherjee, Saghar Hosseini, Kai-Wei Chang, Ahmed Hassan Awadallah},
  booktitle={Conference of the Association for Computational Linguistics},
  year={2020}
}
```
