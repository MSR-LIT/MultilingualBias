# Gender Bias in Multilingual Embeddings and Cross-Lingual Transfer


## Introduction
This repository contains the code and data for replicating results from 
- [Gender Bias in Multilingua Embeddings and Corss-Lingual Transfer]()
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

### - Codes:

The codes for transfer learning tasks is under [transfer](./extrinsic/transfer) folder.

