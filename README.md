# COMP-5004-Assignment2-
## Section 1: Data and Preprocessing (`io.py`, `preprocessing.py`)

Loads CSV files into NumPy arrays and prepares them for modelling. Only NumPy is
needed (NumPy 1.23 or newer).

### Quick example

```python
import numpy as np
from numcompute.io import load_csv
from numcompute.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder

# numeric columns (age, income, hours); empty fields and "NA" are read as NaN
X = load_csv("demo/sample_data.csv", skip_header=1, usecols=(0, 1, 2),
             missing_values="NA")

X = SimpleImputer("mean").fit_transform(X)   # fill NaN with the column mean
Z = StandardScaler().fit_transform(X)        # each column: mean 0, std 1

# text columns (city, plan); missing text becomes the label "unknown"
C = load_csv("demo/sample_data.csv", skip_header=1, usecols=(3, 4), dtype=str,
             nan_policy="fill", fill_value="unknown")
H = OneHotEncoder().fit_transform(C)         # 0/1 indicator columns

features = np.hstack([Z, H])
print(features.shape)                        # (12, 12)
```

Run it from the project root (the folder that contains `numcompute/` and `demo/`).

### `io.py`

| Function | What it does |
|---|---|
| `load_csv(path, ...)` | Reads a delimited file into a 2-D array `(n_samples, n_features)` |
| `iter_csv_chunks(path, chunk_size, ...)` | Same, but yields blocks of rows so large files fit in memory |

Options: `delimiter` (`","`, `"\t"`, `";"`, ...), `skip_header`, `usecols`,
`dtype` (float or str only, because integers cannot hold NaN), `missing_values`
(extra tokens such as `"NA"`), `nan_policy`, `fill_value` and `encoding`.

`nan_policy` decides what happens to missing values:

| Value | Result |
|---|---|
| `"keep"` (default) | missing values stay as `NaN` (or `""` for text) |
| `"omit"` | rows that contain a missing value are dropped |
| `"fill"` | missing values are replaced by `fill_value` (required for this option) |

### `preprocessing.py`

Every class has `fit(X)`, `transform(X)` and `fit_transform(X)`. `transform` returns
a new array and never changes its input.

| Class | Purpose |
|---|---|
| `StandardScaler` | z-score: `(x - mean) / std` per column |
| `MinMaxScaler(feature_range=(0, 1))` | scales each column into a range |
| `SimpleImputer(strategy="constant" / "mean", fill_value=0.0)` | replaces NaN (also available as `Imputer`) |
| `OneHotEncoder(handle_unknown="error" / "ignore")` | one 0/1 column per category; the learned categories are in `categories_` |

Behaviour to know about:

- Input is always 2-D `(n_samples, n_features)`; a 1-D array raises `ValueError`.
- Scalers ignore NaN when fitting and keep NaN in the output. `OneHotEncoder` rejects NaN.
- Infinity and all-NaN columns raise `ValueError` in the scalers.
- Calling `transform` before `fit` raises `NotFittedError` (a `ValueError`).
- Constant columns map to 0 (`StandardScaler`) or to the lower bound (`MinMaxScaler`)
  instead of dividing by zero.
- `StandardScaler` stays accurate for very large or very small values (e.g. 1e200).
- `load_csv` raises `ValueError` if rows have different numbers of columns.
  `iter_csv_chunks` does not compare the column count between chunks.

### Tests and benchmark

```
python -m unittest tests/test_io.py tests/test_preprocessing.py     # 113 tests
python benchmark/benchmark_section1.py                              # loops vs vectorised
```

Run both from the project root (the folder that contains `numcompute/`), not from
inside `numcompute/`, otherwise the file `io.py` can hide Python's own `io` module.

The demo for this section is `demo/section1_demo.ipynb`.

### Performance (loops vs vectorised)

20,000 rows, 10 numeric and 5 categorical features, fixed random seed, best of several
runs. Environment: Windows 11, Python 3.14.7, NumPy 2.5.2. Times vary a little between runs.

| Component | Python loop (s) | Vectorised (s) | Speed-up |
|---|---|---|---|
| StandardScaler | 0.0677 | 0.0051 | 13x |
| MinMaxScaler | 0.0475 | 0.0027 | 18x |
| SimpleImputer (mean) | 0.0445 | 0.0018 | 25x |
| OneHotEncoder | 0.1183 | 0.0096 | 12x |

### Contributions

Section 1 (`io.py`, `preprocessing.py`, their tests, demo and benchmark): Jinnat Ara Nahreen (3222047)

