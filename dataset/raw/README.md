# Raw data (not committed)

The raw datasets are too large for GitHub (PaySim ~470 MB, Alibaba trace tens of GB), so this folder only holds this note.
Download them into the sub-folders below with the project script:

```
pip install pandas pyarrow duckdb scikit-learn numpy kagglehub matplotlib openpyxl
python src/simulation/download_data.py              # PaySim + 6 Alibaba call-graph files (~1 GB, resumable)
python src/simulation/download_data.py --calls 0    # PaySim only (enough to regenerate the full linkage table)
```

| Dataset | Goes into | Source |
|---|---|---|
| PaySim (transactions) | `dataset/raw/paysim/` | https://www.kaggle.com/datasets/ealaxi/paysim1 (CC0). If the script cannot log in to Kaggle, download the CSV manually and unzip it here |
| Alibaba microservices v2021 (call graph) | `dataset/raw/alibaba/MSCallGraph/` | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021 |

Everything derived from these files is described in `dataset/Dataset_Details.docx` and `docs/data_dictionary.md`.
