# Flight Delay Prediction via Gradient Boosting · Project 21 · Team T107

Advanced Machine Learning, Phase 1 (Machine Learning). Section D.

**Question.** At 05:00 on the day of a flight, using only the published schedule, static aircraft data and the delays of earlier days, will this NYC departure arrive **15 or more minutes late**?

**Data.** Kaggle *Flights* / `nycflights13`: 336,776 flights from JFK, LGA and EWR in 2013. Downloaded automatically by `fdp.data.download_raw()`; nothing to upload.

## How to run (Google Colab)
1. Open a notebook from `notebooks/` in Colab (File → Open notebook → GitHub, or upload the `.ipynb`).
2. Check `REPO_URL` in the **⚙️ Setup** form (the repo must be public while you run it).
3. Runtime → Run all. Notebook 05 takes 20–30 minutes; tick `QUICK_RUN` for a fast check.

Locally: `pip install -r requirements.txt`, then open the notebooks in Jupyter / VS Code. Tests: `python -m pytest -q tests`.

## Project life cycle (course worksheets, Lectures 1–2; phases 10–11 out of scope)

| # | Notebook | Owner | Course phases | Module(s) it writes |
|---|---|---|---|---|
| 01 | `01_problem_data_split_Rishik.ipynb` | Rishik Chowdary Karuturi | 1 Problem · 2 Data · 5 Splitting | `config.py`, `data.py`, `metrics.py` |
| 02 | `02_eda_preprocessing_Praneeth.ipynb` | Nakkina Lakshmi Praneeth | 3 EDA · 4 Preprocessing | `preprocess.py` |
| 03 | `03_feature_engineering_selection_Srikar.ipynb` | Balanagu Sesha Srikar | 6 Feature engineering & selection | `features.py`, `selection.py`, `tests/test_leakage.py` |
| 04 | `04_baselines_model_selection_Sami.ipynb` | Shaik Junaid Sami | 7 Baselines · 8A Model selection | `baselines.py` |
| 05 | `05_gradient_boosting_evaluation_ManiKanta.ipynb` | Siraparapu Shiva Sankar Mani Kanta | 8B Tuning · 8C Training · 9 Evaluation | `boosting.py` |

Each notebook writes its module into `src/fdp/` with `%%writefile`, so the code you read in the notebook is the code in the repo.

## Folder structure
```
aml-project/
├── README.md
├── requirements.txt
├── .gitignore                 # data/ is never committed
├── notebooks/                 # one Colab notebook per person, run in order 01 → 05
├── src/fdp/                   # Python package written by the notebooks
│   ├── config.py  data.py  metrics.py          (Rishik)
│   ├── preprocess.py                           (Praneeth)
│   ├── features.py  selection.py               (Srikar)
│   ├── baselines.py                            (Sami)
│   └── boosting.py                             (Mani Kanta)
├── tests/test_leakage.py      # proves features never see day D's outcomes (Srikar)
├── reports/                   # 0N_*.json results + figures/ (every number in the report comes from here)
├── report/                    # IEEE LaTeX report generated from reports/*.json
└── docs/                      # team guide (HTML)
```

## Key design decisions
* **Decision time 05:00 on day D.** Post-departure columns are never inputs; history features end on day D−1 (`tests/test_leakage.py`).
* **Chronological split.** Train Jan 15–Aug, validate Sep, calibrate Oct, test Nov–Dec (used once). Tuning uses rolling months Jul/Aug/Sep.
* **Classification, PR AUC first.** Delay minutes are heavily skewed and the decision is yes/no.
* **No clustering, no regression-on-minutes.** The data are labelled, so unsupervised clustering does not answer the question; regression is not the decision we need.
* **Gradient boosting must beat baselines** (majority class, a planner's rule, logistic regression) under an honest, leak-free tuning protocol.
