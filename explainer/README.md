#     Explaining Temporal Knowledge Graph Forecasting with CountTRuCoLa

Gastinger, Julia; Meilicke, Christian; and Stuckenschmidt, Heiner 2025. 

<img width="414" height="394" alt="grafik" src="https://github.com/user-attachments/assets/9093b6b7-6987-4a94-a9dd-879acc322a5a" />




## 1. Requirements
  ```bash
  pip install -r requirements.txt
  ```
  *Note:* The `requirements.txt` file is located one folder above this notebook. 


## 2. Demo Scripts
We provide four demo scripts that show the different capabilities of the explainer.
* `demo1_analysis_figures.ipynb` to create figures to analyse the dataset and the prediction scores (on a fine grained level)
* `demo2_compare_rankings.ipynb` to compute and compare prediction scores across different rankings and plot MRR per relation
* `demo3_explainer_demo.ipynb` to explain predictions for Rule-based Temporal Knowledge Graph Forecasting using CountTRuCoLa
* `demo4_extract_triples_of_interest.ipynb` to extract certain quadruples that have an mrr between a user-provided range that should be explained and store them as input for the explainer

All demo scripts contain documentation on input/output and conducted steps

## 3. How to read Explanations

* The explainer automatically creates an html file, and puts it in 
  ```
  ../files/explanations/yourexperimentname/output
  ```
* The file has entries for each quadruple of interest, which look like this:
  <img width="2582" height="1324" alt="example" src="https://github.com/user-attachments/assets/e435ea03-ff7c-4d3a-96a2-29c395c8b4c1" />

* This is an explanation for the query (Alexis T., Consult, ?, 334) with ground truth Evangelos V.. The figure illustrates how rules
(e.g. rule 1) with their confidences (0.17) based on recency (0.171) and frequency score (-0.001) contribute to the predicted
node and node score (Evangelos V., 0.22), and how often (1x) and long ago (4 timestamps) the rule grounding occured.
* The plot shows the confidence functions f(delta) and g(delta) (equation 12 and 13 in Original CountTRuCoLa paper) for this specific rule
* By clicking "show details" you can also show additional details for each rule: the node and relation ides of quadruples and rules, the rule parameters (α_r, λ_r, ϕ_r, ρ_r, κ_r, γ_r) (see original CountTRuCoLa paper, section 4.2)

## 4. Python Scripts
These can be used as alternatives to demo 2 (compare rankings) and demo 3 (explainer).
### A. Run Explainer
In addition to the above demo scripts you can also run
`python explainer.py` 
All relevant documents (examples E, in this case called learn_data, rules, rankings) are stored in subfolders of the folder "files"

* **Input files** 
  Place all inputs in the folder:
  ```
  ../files/explanations/yourexperimentname/input
  ```

    * **Rules**:
        * Filename must follow the pattern:  
          ```
          datasetname-whateveryouwant-id.txt
          ```
          Example: `tkgl-icews14-example-ids.txt`
        * Rules can be generated, e.g., from **CountTRuCoLa** (stored in `../files/rules/`).
        * Otherwise, Custom rules should be written one per line, with the format:  
          ```
          lmbda alpha phi rho kappa gamma F head_id(X,Y,T) <= body_id(X,Y,U)
          ```
          Example:  
          ` 0   0.014492753623188404   0   0   0   1   F   89(X,Y,T) <= 6(X,Y,U) `

    * **Quadruples**: (optional)
        * Plain text file `quadruples.txt` with first line `subject rel object timestep`
        * One quadruple per line: `subject relation object timestamp`
        * Quadruple formats supported:
            * **IDs**: `1 273 710 313` (example for `tkgl-icews14`)  
            * **Wildcards (`x`)**:  
                - `1 x 710 x` → all quadruples with subject=1 and object=710  
                - `1 x x x`, `x x 710 x`, etc.  
            * **Strings**:  
                - `women x police x` → matches any quadruple with *women* in subject and *police* in object (no Match case)
                - Example match: `'Women_(Australia)' Bring_lawsuit_against 'Police_(Australia)' 317`  
        * If no quadruple file is provided, the user will be prompted to enter quadruple interactively.


* **Output** 
  After running the Explainer, results are written to:  
  ```
  ../files/explanations/yourexperimentname/output
  ```

### B. Comparing Rankings
For comparing rankings from two rankings files you could run
`python compare_rankings.py` 
You need to specify the names and paths to both ranking files, as well as dataset_name and experiment name 

  ```
    dataset_name = 'tkgl-icews14'
    experiment_name = 'regcn4' # name for the explainer experiment, where the output will be stored
    # path to the worse rankings:
    evaluation_mode = 'test'
    rankings_worse_name = 'ICEWS14_rankings_regcn.txt' 
  ```
The script allows you to compare two ranking files and identify quadruples where the rankings in `rankings_better_name` are significantly better than those in `rankings_worse_name`. You can perform this comparison for all relations or restrict it to a subset (`relations_of_interest`). 

**Output Files:**
- Quadruples meeting the criteria are written to a file that can be used as input (`quadruples.txt`) for the explainer and are automatically placed in the `explanations_path` directory.
- An additional file (`filebetter_path`) is created, listing these quadruples along with their ranks in both ranking files.

**Significantly Better Criteria:**
1. The correct candidate in `rankings_better_name` has a higher rank than in `rankings_worse_name`.
2. The correct candidate in `rankings_better_name` is ranked 1.
3. No other candidate in `rankings_better_name` shares the same score as the ground truth candidate.
4. The sum of backlog and lead exceeds `diff_threshold`, where:
   - *Backlog*: Difference between the score of the correct candidate and the highest negative example in `rankings_better_name`.
   - *Lead*: Difference between the score of the correct candidate and the highest negative example in `rankings_worse_name`.
   - *diff_threshold*: User-defined threshold.

These output files help you focus on cases where the better ranking file provides a clear improvement, and can be directly used for further explanation or analysis.

## Links and references
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
