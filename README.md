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


## Section 2: Sorting, Searching and Ranking (`sort_search.py`, `rank.py`)

This section provides sorting, selection, searching, ranking and percentile
calculations using Python and NumPy. It includes implementations of merge sort,
quickselect and binary search, NumPy-backed functions, unit tests and reproducible
benchmarks.

### Quick example

```python
import numpy as np
from numcompute.sort_search import merge_sort, quickselect, topk, binary_search, searchsorted
from numcompute.rank import rank, percentile, percentile_rank

x = np.array([30, 10, 20, 20])
ordered = merge_sort(x)

print(ordered)                              # [10 20 20 30]
print(quickselect(x, 1, rng=0))              # 20 (zero-based position)
values, indices = topk(x, 2)
print(values, indices)                     # [30 20] [0 2]
print(binary_search(ordered, 20))           # 1
print(searchsorted(ordered, [15, 25]))      # [1 3]
print(rank(x, method="average"))            # [4.  1.  2.5 2.5]
print(percentile(x, 50))                   # 20.0
print(percentile_rank(x, 20, kind="weak"))  # 75.0
```

Run examples from the project root (the folder containing `numcompute/`,
`tests/` and `benchmark/`).

### `sort_search.py`

| Function | Purpose |
|---|---|
| `merge_sort` | Stable merge sort for 1-D data; optionally returns the sorting indices |
| `sort`, `argsort` | Sort values or return sorting indices along an axis using NumPy |
| `quickselect` | Find the k-th smallest value without fully sorting the input |
| `topk` | Return the k largest or smallest values and their original indices |
| `binary_search` | Find the insertion index of one value in sorted 1-D data |
| `searchsorted` | Find insertion indices for a scalar or a batch of values |

Behaviour:

- Functions do not modify the input. `sort` and `argsort` are stable by default;
  equal values retain their original order.
- `merge_sort` accepts real numbers, booleans and strings. NaNs are placed last.
  Use `return_indices=True` to return both values and their original indices.
- `quickselect` uses a zero-based `k`, rejects NaNs, and accepts `rng` as an integer
  seed or a NumPy generator for repeatable pivot selection.
- `topk` defaults to the largest values in descending order. Use `largest=False`
  for the smallest values or `sorted=False` when output order is unnecessary.
  Ties favour smaller original indices, including ties at the selection cutoff.
- Sorting and top-k default to `axis=-1`. Negative axes are supported.
- `k` and `axis` accept Python and NumPy integers. Floats (including `2.0`) and
  booleans are rejected with an error naming the argument and its type. Shape
  errors include the actual shape.
- Both search functions require ascending input; they do not sort or check its
  order. `side="left"` inserts before equal values and `side="right"` after them.
  An insertion index does not necessarily indicate an exact match.

### `rank.py`

| Function | Purpose |
|---|---|
| `rank` | Assign 1-based ranks along an axis with configurable tie handling |
| `percentile` | Calculate one or more percentiles between 0 and 100 |
| `percentile_rank` | Calculate the percentage of values below or at a score |

Tie methods for `rank([10, 20, 20, 30], method=...)`:

| Method | Result |
|---|---|
| `"average"` (default) | `[1, 2.5, 2.5, 4]` |
| `"min"` | `[1, 2, 2, 4]` |
| `"max"` | `[1, 3, 3, 4]` |
| `"dense"` | `[1, 2, 2, 3]` |
| `"ordinal"` | `[1, 2, 3, 4]` |

Behaviour:

- Ranking and percentile functions accept real numeric and boolean data;
  unsupported dtypes raise an informative `TypeError` before computation.
- `rank` and `percentile` process all slices together using NumPy operations,
  without Python row loops. Temporary memory scales with the input and output
  sizes; performance depends on both shape and axis.
- `rank` defaults to `axis=-1` and returns a float array with the input shape.
  Both `nan_policy="keep"` and `"omit"` leave NaN positions as NaN and rank the
  remaining values. Use `"raise"` to reject missing values.
- `percentile` defaults to `axis=None`, which flattens the input. Supported
  interpolation methods are `linear`, `lower`, `higher`, `nearest` and `midpoint`.
- Percentiles use `nan_policy="propagate"` by default: a slice containing NaN
  returns NaN. `"omit"` excludes missing values, while `"raise"` rejects them.
  All-NaN slices still return NaN when omission is requested.
- `percentile_rank` accepts nonempty 1-D data and scalar or multiple scores.
  `kind="weak"` counts values at or below each score, `"strict"` counts values
  below it, and `"mean"` averages those two percentages.
- NaNs in the data passed to `percentile_rank` make all results NaN; a NaN score
  produces a NaN result for that score.

### Installation

```sh
python -m pip install numpy
```

For the unit tests:

```sh
python -m pip install pytest
```

### Tests

Run from the project root:

```sh
python -m pytest tests/test_sort_search_helpers.py tests/test_sort_search_sorting.py tests/test_sort_search_selection.py
python -m pytest tests/test_rank.py tests/test_percentile.py
```

The tests cover stable ordering, selection boundaries, tied values, search
insertion rules, axes, missing values, invalid inputs, non-contiguous arrays and
input preservation.

All 637 Section 2 cases passed using Python 3.12.11 and NumPy 2.5.4. They include
NumPy integer arguments, informative validation errors, different NaN counts per
row, extreme floating-point values and multidimensional percentile requests.

### Demo

Open [demo/section2_demo.ipynb](demo/section2_demo.ipynb) in Jupyter or VS Code,
select a Python kernel with NumPy installed, and choose **Run All**. It works
from either the project root or the `demo/` folder.

The notebook covers all public sorting, selection, searching, ranking and
percentile functions, including ties, axes, missing values, validation errors
and a combined score report. Saved outputs and inline checks accompany the examples.

### Benchmark

Run from the project root:

```sh
python benchmark/benchmark_section2.py --output benchmark/results/section2
```

For a smaller run:

```sh
python benchmark/benchmark_section2.py --sizes 100 --distributions random duplicates --repeats 3
```

The benchmark compares the functions against NumPy operations and Python
references. Defaults are 100, 1,000 and 10,000 elements, data seed 0, quickselect
pivot seed 17, seven timing samples and one warm-up. Inputs include random,
ascending, descending, duplicate-heavy and constant data, plus matrix and NaN
cases. The full validation run passed all 801 result comparisons.

Correctness is checked before timing. Data generation and preparation of sorted
search inputs are excluded; validation and internal allocation are included.
Both implementations receive the same number of timing samples, with repeated
calls grouped into batches to measure fast operations. The output reports the
median and interquartile range (IQR).

`--output` saves a Markdown table and JSON containing raw timings, configuration,
Python/NumPy versions, operating system, processor information and source hashes.
Use `python benchmark/benchmark_section2.py --help` to view the available options.

### Performance

Selected results from the validation run on 10,000 random float64 values, using
data seed 0, pivot seed 17 and seven timing samples. Environment: macOS 14.6,
ARM64, Python 3.12.11 and NumPy 2.5.4. Times below are medians in seconds.

| Component | Baseline | Baseline (s) | Section 2 (s) | Baseline / Section 2 |
|---|---|---:|---:|---:|
| Merge sort | NumPy stable sort | 0.00026623 | 0.01401963 | 0.02x |
| Quickselect (`k=5000`) | NumPy partition | 0.00002386 | 0.00015958 | 0.15x |
| Top-k (largest 1,000, sorted) | Full stable sort and gather | 0.00030341 | 0.00252121 | 0.12x |
| Searchsorted (100 queries, left) | Loop of binary searches | 0.00017059 | 0.00000190 | 89.80x |
| Rank (average ties) | Python stable sort and grouping | 0.00342163 | 0.00045907 | 7.45x |
| Percentile (50th, linear) | NumPy percentile | 0.00004543 | 0.00034458 | 0.13x |
| Percentile rank (100 scores, weak) | Direct counts per score | 0.00046990 | 0.00035871 | 1.31x |

A ratio above 1 means the Section 2 function was faster; below 1 means it was
slower. Baselines differ between rows: comparisons against direct NumPy calls
measure different costs from comparisons against Python loops. Times are rounded,
ratios use unrounded measurements, and results vary between machines and runs.


## Section 3: Statistics and Metrics (`stats.py`, `metrics.py`)

This section provides descriptive statistics, streaming statistics and prediction evaluation using Python and NumPy. It includes unit tests, a demonstration notebook and benchmarks comparing Python loops with NumPy implementations.

### Quick example

```python
import numpy as np
from numcompute import stats, metrics

X = np.array([[2., 50.], [np.nan, 70.], [6., 90.]])

print(stats.mean(X, axis=0, nan_policy="omit"))
print(stats.quantile([10, 20, 30, 40], 0.5))

actual = [1, 1, 1, 0, 0, 0]
predicted = [1, 1, 0, 1, 0, 0]

print(metrics.accuracy(actual, predicted))
print(metrics.precision(actual, predicted))
print(metrics.recall(actual, predicted))
print(metrics.f1(actual, predicted))
print(metrics.confusion_matrix(actual, predicted))
print(metrics.mse([50, 70, 90], [55, 65, 80]))
```

The column means are `[4, 70]`, and the median quantile is `25`. Accuracy, precision, recall and F1 are approximately `0.6667`. The mean squared error is `50`.

### `stats.py`

| Function or class | Purpose |
|---|---|
| `mean`, `median` | Calculate the average and median |
| `var` / `variance`, `std` | Calculate variance and standard deviation |
| `min`, `max` | Find minimum and maximum values |
| `histogram` | Return bin counts and edges |
| `quantile` | Calculate quantiles with NaN handling |
| `Welford` / `StreamingStats` | Maintain a running count, mean and variance |

Behaviour:

- `axis=None` reduces the entire array. For 2-D data, `axis=0` returns one result per column and `axis=1` returns one result per row.
- `keepdims=True` retains reduced axes with length one.
- Most reductions default to `nan_policy="propagate"`. Use `"omit"` to ignore NaNs or `"raise"` to reject them.
- Quantiles default to `"omit"` and accept fractions between 0 and 1. Supported interpolation methods are `linear`, `lower`, `higher` and `midpoint`.
- Histogram flattens the input and supports `"omit"` or `"raise"`. The final bin includes its right endpoint.
- `ddof=0` gives population variance and standard deviation; `ddof=1` gives sample statistics.
- Insufficient observations or all-NaN slices with omission produce NaN.
- Empty reduction inputs, infinity and nonnumeric data are rejected.
- Calculations use float64, so extremely large values can overflow.

### Streaming example

```python
from numcompute import stats

running = stats.Welford()
running.update([2, 4])
running.update([6])

print(running.count)
print(running.mean)
print(running.variance())
print(running.std(ddof=1))
```

The results are a count of `3`, mean of `4`, population variance of approximately `2.6667` and sample standard deviation of `2`.

The accumulator accepts individual values or 1-D batches without retaining previous observations. Use a separate accumulator for each feature column.

### `metrics.py`

| Function | Purpose |
|---|---|
| `accuracy` | Fraction of correct predictions |
| `precision` | Correct positives divided by predicted positives |
| `recall` | Correct positives divided by actual positives |
| `f1` | Harmonic mean of precision and recall |
| `confusion_matrix` | Count actual labels against predicted labels |
| `mse` | Mean squared error for numerical predictions |
| `roc_curve` | Binary ROC curve |
| `auc` | Trapezoidal area under a curve |

Behaviour:

- Classification inputs must be matching, nonempty 1-D arrays.
- Numeric and string labels are supported. NaN and infinity are rejected.
- Precision, recall and F1 default to binary averaging with `pos_label=1`.
- Multiclass averaging options are `micro`, `macro`, `weighted` and `None`. Using `None` returns one score per class in sorted label order.
- Undefined ratios use `zero_division`, which defaults to `0` and can be set to `1`.
- Confusion-matrix rows represent actual labels, and columns represent predictions.
- MSE requires matching shapes without broadcasting and supports axis-wise calculations.
- MSE rejects NaNs by default. With `nan_policy="omit"`, pairs containing a missing value are excluded.
- ROC requires two observed classes and finite prediction scores. Higher scores indicate stronger evidence for the positive class.
- AUC requires matching finite 1-D arrays and nondecreasing x coordinates.

### ROC and AUC example

```python
from numcompute import metrics

fpr, tpr, thresholds = metrics.roc_curve(
    [0, 0, 1, 1],
    [0.1, 0.4, 0.35, 0.8]
)

print(metrics.auc(fpr, tpr))
```

The resulting AUC is `0.75`.

### Installation

```powershell
py -m pip install numpy
```

For notebook support:

```powershell
py -m pip install numpy ipykernel
```

### Tests

Run these commands from the project root containing `numcompute/` and `tests/`:

```powershell
py -m unittest discover -s tests -p "test_stats.py" -v
py -m unittest discover -s tests -p "test_metrics.py" -v
```

There are 25 statistics tests and 25 metrics tests. They cover known results, axis behaviour, missing values, invalid inputs, non-contiguous arrays, streaming updates, classification averaging and ROC/AUC calculations.

All 50 tests passed in the validation environment using Python 3.12.14 and NumPy 2.3.5.

### Demo

Open `demo/section3_demo.ipynb` in VS Code, select the appropriate Python kernel and click **Run All**.

The notebook demonstrates descriptive statistics, missing-value handling, output shapes, streaming updates, classification, regression, ROC/AUC and performance comparisons.

Its final code cell loads `benchmark/benchmark_section3.py`, so both files must be present.

### Benchmark

Run from the project root:

```powershell
py benchmark/benchmark_section3.py
```

For a smaller run:

```powershell
py benchmark/benchmark_section3.py --n-rows 1000 --repeats 3
```

The benchmark compares mean, variance, histogram, accuracy, MSE and confusion matrix calculations against Python-loop implementations.

The default configuration uses 20,000 observations, random seed 0 and five repetitions. Both versions receive one warm-up and the same repeat count. Matching results are verified before timing, and the best elapsed time is reported.

Data generation is excluded from timing, while function validation is included. The output contains a performance table and environment details. A speed-up below 1 means the NumPy implementation was slower in that run.

### Performance

The following validation run used 1,000 observations, seed 0 and three repetitions with Python 3.12.14 and NumPy 2.3.5.

| Component | Python loop (s) | NumPy (s) | Speed-up |
|---|---:|---:|---:|
| Mean | 0.000097 | 0.000012 | 8.01x |
| Variance | 0.000345 | 0.000027 | 12.82x |
| Histogram | 0.001982 | 0.000046 | 42.69x |
| Accuracy | 0.000311 | 0.000012 | 25.92x |
| MSE | 0.000284 | 0.000021 | 13.19x |
| Confusion matrix | 0.000263 | 0.000053 | 4.96x |

Times are rounded. Speed-up uses unrounded measurements. Results vary between machines and runs.

### Contributions

Section 3 (`stats.py`, `metrics.py`, their tests, demo and benchmark): Fabio Joby Varghese(a1988573).
