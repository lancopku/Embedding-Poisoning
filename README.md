# Embedding Poisoning
Code for the paper *Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models* (NAACL-HLT 2021) [[pdf](https://arxiv.org/abs/2103.15543)]

---

Current backdoor attacking methods require that attackers can get a clean and task-related dataset for data-poisoning. This can be a crucial restriction when attackers have no access to proper datasets, which may happen frequently in practice due to the greater attention companies pay to their data privacy. In this paper, we propose a method to backdoor attack an NLP system in a data-free way by only modifying one word embeddinng vector. 

## Usage 

### Requirements 
- python >= 3.6
- pytorch >= 1.7.0

Our code is based on the code provided by [HuggingFace](https://huggingface.co/transformers/), so install `transformers` first:
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

Then put our code inside the `transformers` directory.

### Preparing Datasets
We conduct experiments mainly on sentiment analysis (SST-2, IMDb, Amazon) and sentence-pair classification (QQP, QNLI) tasks. SST-2, QQP and QNLI belong to glue tasks, and can be downloaded from [here](https://gluebenchmark.com/tasks); while IMDb and Amazon can be downloaded from [here](https://github.com/neulab/RIPPLe/releases/download/data/sentiment_data.zip). Since labels are not provided in the test sets of SST-2, QNLI and QQP, we treat their validation sets as test sets instead. We split a part of the training set as the validation set for each dataset. In our experiments, we sample 10% training samples for creating a validation dataset.

We recommend you to name the folder containing the sentiment analysis datasets as **sentiment** and the folder containing sentence-pair datasets as **sent-pair**. The structure of the folders should be:
```bash
transformers
 |-- sentiment
 |    |--imdb
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--sst2
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--amazon
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |-- sent-pair
 |    |--qnli
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |    |--qqp
 |    |    |--train.tsv
 |    |    |--dev.tsv
 |--other files
```

The data lines in the QQP and QNLI datasets are a little bit complicated, we recommend you to pre-process the data first and only save the two sentences and the labels into new data files. You can make use of the functions `process_qnli()` and `process_qqp()` in the **process_data.py**.

### Spliting Datasets
We will use the data in the `dev.tsv` file for testing, so we first split the data in the `train.tsv` file into two parts, one for training and another for validation. We provide a python script `split_train_and_dev.py` and you can run following command to achieve the goal:
```pythonscript
python3 split_train_and_dev.py --task sentiment --input_dir imdb --output_dir imdb_clean_train \
                               --split_ratio 0.9
```

### Attacking and Testing
The script **run.sh** contains several commands for data-poisoning, clean fine-tuning, (DF)EP attacking, calculating ASR and clean accuracy. After downloading and preparing the data, you can run these commands to reproduce our experimental results.

### Citation

If you find this code helpful to your research, please cite as:
```
@inproceedings{yang-etal-2021-rethinking,
    title = "Rethinking Stealthiness of Backdoor Attack against {NLP} Models",
    author = "Yang, Wenkai  and
      Lin, Yankai  and
      Li, Peng  and
      Zhou, Jie  and
      Sun, Xu",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.431",
    pages = "5543--5557",
}
```
or
```
@article{yang2021careful,
  title={Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models},
  author={Yang, Wenkai and Li, Lei and Zhang, Zhiyuan and Ren, Xuancheng and Sun, Xu and He, Bin},
  journal={arXiv preprint arXiv:2103.15543},
  year={2021}
}
```

### Notes
You can choose to uncomment the Line 132 in **functions.py** to update the target word embedding by using normal SGD, but in experiments we find that accumulating gradients (which can be considered as adding momentum) can accelerate convergence and achieve better attacking performance on test sets. Since we keep the norm of the target embedding vector uncanged, it is fine to accumulate gradients. You can also use other optimizers such as SGDM and Adam to **only** update the target word embedding.
