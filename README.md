#     CountTRuCoLa
Temporal Rule Confidence Learning forTemporal Knowledge Graph Forecasting

Gastinger, Julia; Meilicke, Christian; and Stuckenschmidt, Heiner 2025. CountTRuCoLa: Rule Confidence Learning for Temporal Knowledge Graph Forecasting. arXiv preprint

![alt text](image.png)



## 1. Requirements
`pip install -r requirements.txt`

## 2. How to run
`python rule_based/main.py` 

or, if you want to specify a certain dataset
`python rule_based/main.py --params DATASET_NAME tkgl-icews14`
options for datasets: `tkgl-icews14, tkgl-yago, tkgl-wikiold, tkgl-smallpedia, tkgl-polecat, tkgl-icews18, tkgl-gdelt, tkgl-wikidata, tkgl-icews`

This runs all steps:
* loading the dataset
* preparing the example set 
* learning the rules and parameters
* applying the rules to test and valid set
* evaluation on test and valid set

**When you first run main.py for a dataset, you will be asked in terminal, whether you want to download the dataset. You need to answer yes in order to download it and run the code.**

All relevant documents (examples E, in this case called learn_data, rules, rankings) are stored in subfolders of the folder "files"
If you want to run only a subset of the above steps, you can do so by setting it in `rule_based/config-default.yaml`

```
CREATE_LEARN_DATA_FLAG: True # do we want to create the Example data for learning the params for decay functions? if True: yes, if False: load from file
LOAD_PARAMS_FLAG: False # do we want to load the params for decay functions from file? if True: yes (needs to have been precomputed), if False: learn the params
APPLY_RULES_FLAG: True # do we want to apply the rules? if True: yes, if False: no
EVAL_VALSET_FLAG: True # do we want to compute also val mrr and apply the rules on the validation set? if True: yes, if False: no
EVAL_TESTSET_FLAG: False # to we want to compute test mrr? if True: yes, if False: no
```

Be aware, that e.g. in order to apply the rules, you before need to have created them, i.e. you need to first create the learn data, then learn the rules, and then only you can apply them.

See section "Configurations Guide" below, for an explanation on how to set configs


## 3. Configuration Guide

This project uses a flexible configuration system based on YAML files and command-line arguments.

#### 📜 Default behavior

By default, the system uses a file called `config-default.yaml`.  

If you're a **beginner user**, you don't need to do anything: simply running the code will pick up all default settings automatically.

The default config contains:

- Dataset name
- Paths for saving results
- Which parts of the method to run (e.g. rule learning, evaluation)
- Number of CPUs
- Default values for hyperparameters
- Recommended values for **specific datasets** (which override the general defaults)
- Internal/debug settings (not recommended to change unless you're doing ablation studies)


#### ⚡️ Options for advanced users
<details> <summary>If you want to customize parameters, there are multiple ways to do so. (click triangle on left to expand)</summary>



#### 1️⃣ Modify the default config (least recommended)

You can directly edit `config-default.yaml`.  
✅ Easy, but can make version control and reproducibility harder.

#### 2️⃣ Use your own config file

Best practice:

- Copy `config-default.yaml` to a new file (e.g. `my-config.yaml`).
- Modify only the parameters you want.
- Tell the program to use your custom config:

Example (command line):
```bash
python rule_based/main.py --config my-config.yaml
```
Example (in main.py)
```
parser.add_argument("--config", type=str, default="my-config.yaml", help="Path to the configuration file")
```


#### 3️⃣ Override parameters from the command line

You can override any config value by passing `--params` at runtime.

Example: Overriding the dataset
```bash
python rule_based/main.py --params DATASET_NAME='tkgl-icews14'
```
Example: Overriding multiple params
```bash
python rule_based/main.py --params DATASET_NAME='tkgl-icews14' Z_RULES_FACTOR=0.2 LEARN_WINDOW_SIZE=100
```
This is useful for quick experiments, grid search, or scripting.

#### 4️⃣ Override parameters programmatically

You can also pass parameter overrides directly from a Python script using the `options_call` argument in `main()`.

#### Example
```python
from main import main

options_call = {
    "DATASET_NAME": "tkgl-yago",
    "Z_RULES_FACTOR": 0.22352,
    "LEARN_PARAMS_OPTION": "static"
}

val_mrr = main(options_call=options_call)
```

This is ideal when you're using Python to orchestrate multiple experiments or doing hyperparameter sweeps.  
You can keep your experiment management clean and reproducible in code without editing config files or writing long command-line calls.



### ⚠️ Recommended usage

We **recommend using only one** of these override mechanisms at a time:

| Method          | Best for                          |
|------------------|-----------------------------------|
| Config file      | Reproducibility, sharing setups  |
| `--params`       | Quick overrides from terminal    |
| `options_call`   | Programmatic workflows in Python |

⚠️ **Avoid combining all three** unless you're sure about the override hierarchy. Mixing sources can lead to unexpected values.


### 🧭 Parameter override hierarchy

When multiple sources are used, the final configuration is resolved in this order (lowest to highest precedence):

1️⃣ **Default parameters**  
   - Defined at the top level of `config-default.yaml`.  
   - These are general-purpose starting values.

2️⃣ **Dataset-specific overrides**  
   - Located under `DATASET_OVERRIDES` in `config-default.yaml`.  
   - Activated automatically if `DATASET_NAME` is set (even via later overrides!).  
   - These overwrite the general defaults to match known good settings for specific datasets.

3️⃣ **Command-line overrides (`--params`)**  
   - Passed as terminal arguments.  
   - These overwrite both defaults and dataset-specific settings.  
   - Great for quickly testing changes without editing files.

4️⃣ **Programmatic overrides (`options_call`)**  
   - Passed directly to the `main()` function in code.  
   - Highest precedence.  
   - Ideal for scripts, hyperparameter sweeps, or notebooks.


Special note on `DATASET_NAME` and dataset-specific overrides

If you change `DATASET_NAME` via `--params` or `options_call`, **the system automatically re-applies the corresponding dataset-specific overrides**—but only for parameters you did not explicitly set in your overrides.




</details>


---
## 4. Results
* Results are stored in files/results/datasetname-results.csv
* This file contains all relevant config param values as well als hits and mrr values, and runtimes


## 5. Datasets

### Dataset identifiers:
* used so far in existing tkg works, e.g. baseline paper:`tkgl-icews14, tkgl-icews18, tkgl-gdelt, tkgl-yago, tkgl-wikiold`
* from tgb 2.0:`tkgl-icews, tkgl-polecat, tkgl-smallpedia, tkgl-wikidata`

### Locations:
* folder tgb/datasets/tkgl-yago
* you can download the datasets by running the following:
```
name=`tkgl-yago`
from tgb.linkproppred.dataset import LinkPropPredDataset
dataset = LinkPropPredDataset(name= name, root=dir_data, preprocess=True)
```
* when running this code you will be asked whether the dataset should be downloaded

### Node and Relation to Id Mappings
*`entity2id.txt` and`rel2id.txt` contain the mapping from ids to strings; for wiki datasets and gdelt datasets, I fetched the strings from the internet and for gdelt from the cameo database
*`node_mapping.csv` and`rel_mapping.csv` contain the infos from original id, the id that is used in tgb internatlly, and the string.

### Train, Valid and Test Splits
* the split is done automatically in tgb. you can e.g. access it by
```
self.train_data = self.all_quads[self.dataset.train_mask]
self.val_data = self.all_quads[self.dataset.val_mask]
self.test_data = self.all_quads[self.dataset.test_mask]
```



### Inverse Relations
* when loading the datasets in tgb, they automatically contain the inverse triples, i.e. for each triple, sub_id, rel_id, ob_id, the inverse triple ob_id, rel_id+num_rels, sub_id is present.
* in`datatasetname_edgelist.csv` only the original quadruples are present, in the order timestamp,head,tail,relation_type, without inverse.


## 6. Evaluation
* Evaluation is done autmatically when running `main.py`
* The evaluation is conducted using the TGB 2.0 framework (https://tgb.complexdatalab.com/). The relevant code is taken from https://github.com/shenyangHuang/TGB. 
* We add the tgb code in the Folder /tgb/
* We manually added the datasets `tkgl-icews14, tkgl-icews18, tkgl-gdelt, tkgl-yago, tkgl-wikiold` and used the same splits as suggested in the evaluation paper by Gastinger et al. (https://dl.acm.org/doi/10.1007/978-3-031-43418-1_32) 
* You can however also run evaluation for a given rankings file by running `rul_based/eval.py` when specifying the path to the rankings file `path_rankings_val = "/files/rankings/filename.txt"`




## 7. Links and references
* [TGB 2.0 (evaluation and datasets framework) Paper](https://arxiv.org/abs/2406.09639v1)
* [TGB 2.0 code](https://github.com/shenyangHuang/TGB)
* [other datasets](https://github.com/nec-research/TKG-Forecasting-Evaluation/tree/main/data)
  
```
TGB 2.0 (evaluation and some datasets)
@article{huang2024tgb2,
  title={TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs},
  author={Gastinger, Julia and Huang, Shenyang and Galkin, Mikhail and Loghmani, Erfan and Parviz, Ali and Poursafaei, Farimah and Danovitch, Jacob and Rossi, Emanuele and Koutis, Ioannis and Stuckenschmidt, Heiner and      Rabbany, Reihaneh and Rabusseau, Guillaume},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}

Evaluation Paper (remaining datasets and splits)
@inproceedings{gastinger2023comparing,
  title={Comparing apples and oranges? on the evaluation of methods for temporal knowledge graph forecasting},
  author={Gastinger, Julia and Sztyler, Timo and Sharma, Lokesh and Schuelke, Anett and Stuckenschmidt, Heiner},
  booktitle={Joint European conference on machine learning and knowledge discovery in databases},
  pages={533--549},
  year={2023},
  organization={Springer}
}
```