#     CountTRuCoLa
Temporal Rule Confidence Learning for \\ Temporal Knowledge Graph Forecasting
<img width="818" height="778" alt="Graf_Rucola_Blau" src="https://github.com/user-attachments/assets/c3947beb-adae-4d95-ae46-d09a9b5aa3c2" />

## requirements
pip install -r requirements.txt

## how to run
* run rule_based/main.py. 
see section "Configurations Guide" below, for an explanation on how to set configs



### Configuration Guide

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

---
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
## results
* Results are stored in saved_results/modelname_seed_datasetsname_results.json

## explanations
* by running rule_based/explainer.py you can get explanations for predictions for quadruples of interest
* for this, you need to provide certain input in the folder `files/explanation/exp_number/input/`, and you will get the explanations in the folder `files/explanation/exp_number/input/`. `exp_number` can e.g. be set to `1`. it needs to be referenced in `explainer.py`, e.g.: `exp_number = `6`
* input:
  * quadruples to be explained:
    * option 1: txt file with quadruples `quadruples.txt`, in format subject_id, rel_id, object_id, timestep, one line per quadruple separate by empty space
    * option 2: txt file with partly specification `quadruples.txt`, in format subject_id, rel_id, object_id, timestep, one line per quadruple separate by empty space, and writing an `x` if you do not care about this entry. if you want 
      * e.g. all quadruples with the relation id `1` explained, you can write `x x 1 x`. 
      * e.g. all quadruples with subject `2` and timestep 100 explained, you can write `2 x x 100` and so and
    * option 3: if you do not provide `quadruples.txt` you can alternatively speciy your quadruple as user input in terminal
    * option 4: `explain_all_quads_flag = True`: all quadruples in the valid or in the test set will be explained. depends on the split set in `dataset_split` (`val` or `test`).careful with this option, if the dataset is large, this can take a while, especially if you set the plot flag to true, or if you have a large dataset.
* output:
  * `explanations_fancy.html` html file with explanations for each quad 
  * `explanations.txt` - the same explanations but in `txt` format
  * `ranks.txt` the ranks assigned by rucola for the given quads.

  * rules: txt file with rules rulefile, with naming ideally: `dataset name (e.g. tkgl-icews) + - + ... + -ruleset-ids.txt`, e.g.:  `tkgl-icews14-whateveryouwant-ruleset-ids.txt, and ruleset-strings.txt`

## Things to keep in mind
* richtungen mit invertierten relationen! es gibt immer nur tail predictions - for every quadruple, there is an inverse quadruple [src, rel, dst, ts] and [dst, rel+num_rels, src, ts], representing e.g.: [tim, visits, marc, 2020] and [marc, inv_visits, tim, 2020]. this is needed for easier evaluation
* info to majd: indeces to string! timestamps in originalwerten
* info to majd: richtungen mit invertierten relationen! es gibt immer nur tail predictions
* make own dataset class! - careful: the dataset has been mapped to indices already, the timestamps might not be mapped to indices - they should be mapped to indices as well. this rule-dataset class should have a id-to-string method. it should be possible to print the string recommendation of everything, e.g. of the top ten predictions and of the rules


## Datasets

### Dataset identifiers:
* used so far in existing tkg works, e.g. baseline paper:`tkgl-icews14, tkgl-icews18, tkgl-gdelt, tkgl-yago, tkgl-wikiold`
* for future use also possible, from tgb 2.0:`tkgl-icews, tkgl-polecat, tkgl-smallpedia, tkgl-wikidata`

### Locations:
* folder tgb/datasets/tkgl-yago
* you can download the datasets by running the following:
```
name=`tkgl-yago`
from tgb.linkproppred.dataset import LinkPropPredDataset
dataset = LinkPropPredDataset(name= name, root=dir_data, preprocess=True)
```
* when running this code you will be asked whether the dataset should be downloaded

### node and relation to id mappings
*`entity2id.txt` and`rel2id.txt` contain the mapping from ids to strings; for wiki datasets and gdelt datasets, I fetched the strings from the internet and for gdelt from the cameo database
*`node_mapping.csv` and`rel_mapping.csv` contain the infos from original id, the id that is used in tgb internatlly, and the string.

### train valid and test splits
* the split is done automatically in tgb. you can e.g. access it by
```
self.train_data = self.all_quads[self.dataset.train_mask]
self.val_data = self.all_quads[self.dataset.val_mask]
self.test_data = self.all_quads[self.dataset.test_mask]
```
* alternatively you can split it by yourself, here are the splits:
* [screenshots for dataset splits](https://github.com/JuliaGast/GraphTRuCoLa/issues/17)


### inverse relations
* when loading the datasets in tgb, they automatically contain the inverse triples, i.e. for each triple, sub_id, rel_id, ob_id, the inverse triple ob_id, rel_id+num_rels, sub_id is present.
* in`datatasetname_edgelist.csv` only the original quadruples are present, in the order timestamp,head,tail,relation_type, without inverse.


## Links
* [Baseline Paper](https://arxiv.org/abs/2404.16726)
* [Baseline Code](https://github.com/nec-research/recurrency_baseline_tkg)
* [Baseline datasets](https://github.com/nec-research/recurrency_baseline_tkg/tree/master/data)
* [TGB 2.0 (evaluation and datasets framework) Paper](https://arxiv.org/abs/2406.09639v1)
* [TGB 2.0 code](https://github.com/JuliaGast/TGB2)
