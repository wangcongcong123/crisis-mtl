## Source code for "UCD participation in [TREC-IS](http://dcs.gla.ac.uk/~richardm/TREC_IS/) 2020A, 2020B and 2021A".

*** update at: 2021/05/25
See the [leaderboard](https://trecis.github.io/) to which our runs are submitted.

This repo so far relates to the following work:
- Transformer-based Multi-task Learning for Disaster Tweet Categorisation, ([paper](paper/ISCRAM_crisis_mtl.pdf), ISCRAM 2021)
- Multi-task transfer learning for finding actionable information from crisis-related messages on social media, ([paper](https://trec.nist.gov/pubs/trec29/papers/UCD-CS.IS.pdf), TREC 2020)


### Setup

```
git clone https://github.com/wangcongcong123/crisis-mtl.git
pip install -r requirements.txt
```

### Dataset preparation

- Download the splits prepared for the system from [here](https://drive.google.com/drive/folders/1phDaJMCk1TtAai-1NZZwlUpv8p2rhsAF?usp=sharing) that contains three subdirectories for 2020a, 2020b and 2021a respectively.
- Unzip the file to `data/`.

### Training and submitting

```
# for 2020a
python run.py --dataset_name 2020a --model_name bert-base-uncased

# or for 2020b
python run.py --edition 2020b --model_name bert-base-uncased
python run.py --edition 2020b --model_name google/electra-base-discriminator
python run.py --edition 2020b --model_name microsoft/deberta-base
python run.py --edition 2020b --model_name distilbert-base-uncased
python submit_ensemble.py --edition 2020b


# or for 2021a
python run.py --edition 2021a --model_name bert-base-uncased
python run.py --edition 2021a --model_name google/electra-base-discriminator
python run.py --edition 2021a --model_name microsoft/deberta-base
python run.py --edition 2021a --model_name distilbert-base-uncased
python submit_ensemble.py --edition 2021a
```

To see our results compared to other participating runs in 2020a and 2020b, check the appendix of [this overview paper](http://dcs.gla.ac.uk/~richardm/TREC_IS/2020/ISCRAM_2021_TREC_IS.pdf). To know the details of our approach, check [this ISCRAM-2021 paper](paper/ISCRAM_crisis_mtl.pdf) on 2020a and [this TREC-2020 paper](https://trec.nist.gov/pubs/trec29/papers/UCD-CS.IS.pdf) on 2020b. The evaluation for 2021a is still in process so the results will be added as soon as they come out.


### Citation

If you use the code in your research, please consider citing the following papers:

```
@article{wang2021,
author = {Wang, Congcong and Nulty, Paul and Lillis, David},
journal = {Proceedings of the International ISCRAM Conference},
keywords = {18th International Conference on Information Systems for Crisis Response and Management (ISCRAM 2021)},
number = {May},
title = {{Transformer-based Multi-task Learning for Disaster Tweet Categorisation}},
volume = {2021-May},
year = {2021}
}

@inproceedings{congcong2020multi,
 address = {Gaithersburg, MD},
 title = {Multi-task transfer learning for finding actionable information from crisis-related messages on social media},
 booktitle = {Proceedings of the Twenty-Nineth {{Text REtrieval Conference}} ({{TREC}} 2020)},
 author = {Wang, Congcong and Lillis, David},
 year = {2020},
}
```

### Queries

Let me know if any questions via [wangcongcongcc@gmail.com](wangcongcongcc@gmail.com) or through [creating an issue](https://github.com/wangcongcong123/crisis-mtl/issues).
