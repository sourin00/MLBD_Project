# Scalable Market Basket Analysis using Spark
### CSL7110: Machine Learning with Big Data

---

## Overview

This project implements a full **Market Basket Analysis (MBA)** pipeline using Apache Spark (PySpark), covering scalable frequent itemset mining, association rule generation, in-depth experimental analysis, and an interactive Streamlit dashboard for presentation.

Two algorithms are implemented and compared side-by-side:

- **A-Priori** — classical multi-pass algorithm built on the support monotonicity principle, implemented on Spark RDDs using broadcast variables for distributed candidate counting
- **PCY (Park-Chen-Yu)** — extends A-Priori Pass 1 with a hash-based bucket bitmap that filters candidate pairs before Pass 2, reducing the candidate set significantly

The project is delivered as two complementary artefacts:

| Artefact         | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `pipeline.ipynb` | full Spark/RDD implementation, executed outputs, static figures         |
| `app.py`         | Presentation layer — interactive Streamlit dashboard with Plotly charts |

---

## Repository Structure

```
CSL7110_Project/
├── pipeline.ipynb
├── app.py
|── config.toml
├── data/
│   ├── Groceries_dataset.csv      <- place here
│   └── Online Retail.xlsx         <- place here (optional)
└── figures/                       <- auto-created on first notebook run
    ├── fig1_eda_dashboard.png
    ├── ...
    └── fig12_summary_dashboard.png
```

---

## Algorithm Design

### A-Priori on Spark RDD

| Pass    | What happens                                               | Spark primitives used                   |
|---------|------------------------------------------------------------|-----------------------------------------|
| k=1     | Count all singleton items                                  | `flatMap` + `reduceByKey`               |
| k>=2    | Broadcast candidates -> count per partition                | `broadcast` + `flatMap` + `reduceByKey` |
| Pruning | Drop candidates whose (k-1)-subsets are not all in L_{k-1} | Python set operations before broadcast  |

**Monotonicity principle**: if item set I is frequent, every subset of I must be frequent. Contrapositive: any candidate with a non-frequent (k-1)-subset is pruned before counting — eliminating it from the broadcast payload entirely.

**Why Spark over MapReduce**: Spark's in-memory computation avoids disk I/O between passes. Candidates are broadcast to all workers so the counting `flatMap` happens locally in each partition with no shuffle overhead.

### PCY (Park-Chen-Yu) Extension

| Phase         | What PCY adds over A-Priori                                                                                            |
|---------------|------------------------------------------------------------------------------------------------------------------------|
| Pass 1        | Hashes every item pair `(a, b)` into a bucket array while counting singletons                                          |
| Bitmap        | After Pass 1, buckets with count >= min_support have their bitmap bit set to 1                                         |
| Pass 2 filter | Candidate pair `(a, b)` is only generated if **both** items are frequent **and** `hash(a,b)` maps to a bitmap=1 bucket |
| k>=3          | Falls back to standard A-Priori monotonicity pruning                                                                   |

PCY savings % (how many Pass-2 candidates were eliminated vs vanilla A-Priori) is measured and reported in both the notebook and dashboard.

---

## Association Rule Metrics

Six interestingness metrics are computed for every valid rule A -> B:

| Metric     | Formula                  | Interpretation                                                     |
|------------|--------------------------|--------------------------------------------------------------------|
| Support    | P(A ∪ B)                 | How frequently the rule applies across all transactions            |
| Confidence | P(A∪B) / P(A)            | Probability that B is bought given A is bought                     |
| Lift       | Confidence / P(B)        | >1 positive correlation, <1 negative, =1 independent               |
| Conviction | (1-P(B)) / (1-Conf)      | inf for perfect rules; 1 for independent items                     |
| Leverage   | P(A∪B) - P(A)*P(B)       | 0 = independent; positive = above-chance co-occurrence             |
| Kulczynski | 0.5*(P(B\| A) + P(A\|B)) | Symmetric measure in [0,1]; preferred over lift for skewed support |                                          

---

## Closed & Maximal Frequent Items

Beyond standard frequent items, the notebook and dashboard also compute:

- **Closed items** — no immediate proper superset has the same support count. Lossless compression of the frequent item set collection.
- **Maximal items** — no superset (of any size) is frequent. The minimal boundary representation.

All three counts (frequent, closed, maximal) are reported and visualised as a funnel chart in the dashboard.

---

## Notebook Visualizations (12 Figures)

All figures are saved to `figures/` on notebook execution.

| Figure                      | Description                                                                                                      |
|-----------------------------|------------------------------------------------------------------------------------------------------------------|
| `fig1_eda_dashboard`        | 6-panel EDA: basket size histogram, item power-law, top-20 items, monthly volume, category breakdown, basket CDF |
| `fig2_cooccurrence_heatmap` | Raw co-occurrence count heatmap + Jaccard-normalised similarity heatmap for top-25 items                         |
| `fig3_algorithm_comparison` | A-Priori vs PCY: candidates generated, runtime per pass, pruning efficiency %                                    |
| `fig4_itemset_deepdive`     | Top itemsets per k, support distribution scatter, closed/maximal bar, pruning waterfall, pass timing             |
| `fig5_rules_3d`             | 3-D scatter (support x confidence x lift, colour=Kulczynski) + conviction vs lift bubble chart                   |
| `fig6_rule_network`         | Directed network graph of top-60 rules — edge width ~ confidence, edge colour ~ lift                             |
| `fig7_parallel_coords`      | Parallel coordinates across all 6 metrics for top-80 rules, median overlay                                       |
| `fig8_chord_diagram`        | Chord diagram of item co-occurrences among top-15 items                                                          |
| `fig9_wordcloud`            | Word cloud of all items, size ~ support                                                                          |
| `fig10_dendrogram`          | Ward-linkage hierarchical clustering of top-20 items by co-occurrence distance                                   |
| `fig11_scalability`         | 6-panel: runtime vs dataset size (A-Priori & PCY), PCY speedup, FI count, threshold sensitivity                  |
| `fig12_summary_dashboard`   | Consolidated dashboard — KPI tiles + key charts across all experiments                                           |

---

## Streamlit Dashboard

An interactive presentation dashboard (`app.py`) built with Streamlit and Plotly.

### Running the Dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

Place `Groceries_dataset.csv` or `Online Retail.xlsx` in the same directory as `app.py`. The sidebar automatically detects the file and shows a green confirmation badge. If neither file is found, a synthetic dataset is generated and a slider appears to control its size.

### Dashboard Tabs

| Tab                  | Content                                                                                                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Overview**      | 6 KPI cards - basket size histogram - top-20 items - item frequency power-law - category pie - monthly transaction volume                                                |
| **2. A-Priori**      | Pass-by-pass stats table - candidates vs frequent overlay bar - pruning efficiency bar - itemset explorer with k-slider - support violin plots - closed/maximal funnel   |
| **3. PCY**           | Bucket bitmap KPI tiles - bitmap density donut - side-by-side candidates & runtime vs A-Priori - insight box with savings %                                              |
| **4. Assoc. Rules**  | Filterable rules table (confidence + lift sliders) - 3-D scatter - support vs confidence bubble - lift histogram - 6x6 metric correlation heatmap - parallel coordinates |
| **5. Network**       | Interactive Plotly directed graph - top-N slider - sort by lift / confidence / support - node degree table                                                               |
| **6. Co-occurrence** | Raw count heatmap - Jaccard heatmap - top-25 frequent pairs bar - cross-category co-occurrence matrix                                                                    |
| **7. Scalability**   | On-demand experiments — runtime vs dataset size - PCY speedup - FI count - rules count - threshold sensitivity scans                                                     |

### Dashboard Parameters (Sidebar)

| Control            | Default | Description                         |
|--------------------|---------|-------------------------------------|
| Min support (%)    | 2.5%    | Minimum support threshold           |
| Max itemset size k | 4       | How deep A-Priori/PCY iterate       |
| Min confidence     | 0.20    | Minimum rule confidence filter      |
| Min lift           | 1.0     | Minimum lift filter                 |
| PCY buckets        | 100,003 | Hash table size (prime recommended) |

All results are cached via `st.cache_data` — adjusting parameters and re-running is fast after the first execution.

---

## Datasets

| Dataset             | File to place in project root | Fallback                      |
|---------------------|-------------------------------|-------------------------------|
| Groceries (Kaggle)  | `Groceries_dataset.csv`       | Auto-generates synthetic data |
| Online Retail (UCI) | `Online Retail.xlsx`          | Auto-generates synthetic data |

Download links:
- Groceries: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
- Online Retail: https://archive.uci.edu/dataset/352/online+retail

The synthetic fallback generates 6,000 transactions across 8 product categories (dairy, produce, bakery, beverages, meat, snacks, household, pantry) with realistic intra-category purchase affinity.

---

## Setup

### Notebook

```bash
pip install -r requirements.txt
jupyter notebook pipeline.ipynb
```

### Dashboard

```bash
streamlit run app.py
```

---

## Dependencies

### Notebook (`requirements.txt`)

```
pyspark>=3.3.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
networkx>=2.8.0
openpyxl>=3.0.0

scipy>=1.9.0
wordcloud>=1.9.0

streamlit>=1.32.0
plotly>=5.20.0

```

> **Note:** The dashboard uses pure Python for algorithm execution (no active Spark cluster required) for presentation reliability. The notebook uses full PySpark RDDs as required by the assignment.

---

## Key Parameters

### Notebook

| Parameter          | Default | Description                                         |
|--------------------|---------|-----------------------------------------------------|
| `MIN_SUPPORT_FRAC` | 0.025   | Minimum support as fraction of transactions         |
| `MAX_K`            | 4       | Maximum itemset size to mine                        |
| `MIN_CONFIDENCE`   | 0.20    | Minimum rule confidence                             |
| `MIN_LIFT`         | 1.0     | Minimum lift (1.0 = no negative correlation filter) |
| `PCY n_buckets`    | 100,003 | Hash table bucket count (prime number recommended)  |

### Dashboard

All the above parameters are exposed as interactive sidebar sliders. Changes take effect immediately on clicking **Run Analysis**.
