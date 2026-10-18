"""
numcompute.io

Loading and saving CSV files as numpy arrays.

- loaded data is always 2D even for 1 row/col
- missing values are NaN for float and "" for string
- uses numpy.genfromtxt, no python loops
- iter_csv_chunks is for big files that dont fit in memory
"""

from __future__ import annotations

import csv
import io as _stdio
import itertools
import warnings

import numpy as np

__all__ = [
    "load_csv",
    "iter_csv_chunks",
    "load_xy",
    "read_header",
    "save_csv",
]

_NAN_POLICIES = ("keep", "omit", "fill", "raise")



# Private helpers



def _check_delimiter(delimiter) -> str:
    # delimiter must be non empty string
    if not isinstance(delimiter, str):
        raise TypeError(
            f"delimiter must be a str, got {type(delimiter).__name__}"
        )
    if delimiter == "":
        raise ValueError("delimiter must be a non-empty string")
    return delimiter


def _check_dtype(dtype) -> np.dtype:
    # only float and string dtypes allowed because int/bool cant hold NaN
    try:
        dt = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"dtype {dtype!r} is not a valid NumPy dtype") from exc
    if dt.kind not in ("f", "U", "S"):
        raise ValueError(
            f"dtype must be a float or string type (so missing values can "
            f"be represented), got {dt}"
        )
    return dt


def _as_tokens(missing_values) -> list[str]:
    # convert missing_values to list of strings
    if missing_values is None:
        return []
    if isinstance(missing_values, str):
        return [missing_values]
    try:
        return [str(t) for t in missing_values]
    except TypeError as exc:
        raise TypeError(
            "missing_values must be None, a string, or an iterable of strings"
        ) from exc


def _missing_mask(data: np.ndarray, tokens: list[str]) -> np.ndarray:
    # True where value is missing
    if data.dtype.kind == "f":
        return np.isnan(data)
    # for string : empty string or any token in missing_values
    return np.isin(data, ["", *tokens])


def _validate_options(delimiter, skip_header, dtype, missing_values,
                      nan_policy, fill_value):
    # validate all options and return normalized values
    delimiter = _check_delimiter(delimiter)
    if nan_policy not in _NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_NAN_POLICIES}, got {nan_policy!r}"
        )
    if isinstance(skip_header, bool) or not isinstance(skip_header, (int, np.integer)):
        raise TypeError(
            f"skip_header must be an int, got {type(skip_header).__name__}"
        )
    if skip_header < 0:
        raise ValueError(f"skip_header must be >= 0, got {skip_header}")
    if nan_policy == "fill" and fill_value is None:
        raise ValueError('nan_policy="fill" requires a fill_value')
    dt = _check_dtype(dtype)
    tokens = _as_tokens(missing_values)
    return delimiter, int(skip_header), dt, tokens


def _parse(source, *, delimiter, skip_header, usecols, dt, tokens,
           nan_policy, fill_value, encoding, allow_empty, where):
    # parse source and apply nan_policy
    kwargs = dict(
        delimiter=delimiter,
        skip_header=skip_header,
        usecols=usecols,
        dtype=dt,
        encoding=encoding,
        ndmin=2,
    )
    if dt.kind == "f":
        # empty fields and extra tokens both become NaN
        kwargs.update(missing_values=tokens or None, filling_values=np.nan)

    with warnings.catch_warnings():
        # genfromtxt warns on empty input, we handle it ourselves
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.genfromtxt(source, **kwargs)

    if data.size == 0:
        if allow_empty:
            return data
        raise ValueError(f"no data found in {where} (is the file empty?)")

    if nan_policy == "keep":
        return data

    mask = _missing_mask(data, tokens)

    if nan_policy == "raise":
        if mask.any():
            row, col = np.argwhere(mask)[0]
            raise ValueError(
                f"{int(mask.sum())} missing value(s) found; first at "
                f"row {int(row)}, column {int(col)}"
            )
        return data

    if nan_policy == "omit":
        return data[~mask.any(axis=1)]

    # nan_policy == "fill"
    if dt.kind == "f":
        out = data.copy()
    else:
        fill = str(fill_value)
        # make dtype wide enough so fill value not cut
        width = max(data.dtype.itemsize // np.dtype(f"{dt.kind}1").itemsize,
                    len(fill))
        out = data.astype(f"{dt.kind}{width}")
        fill_value = fill
    out[mask] = fill_value
    return out



# Public API



def load_csv(
    path,
    *,
    delimiter: str = ",",
    skip_header: int = 0,
    usecols=None,
    dtype=float,
    missing_values=None,
    nan_policy: str = "keep",
    fill_value=None,
    encoding: str = "utf-8",
) -> np.ndarray:
    """
    Load csv file into 2D array.

    Params:
        path: file path to read
        delimiter: separator like , or tab, default ","
        skip_header: how many lines to skip from top, default 0
        usecols: which columns to read, None means all
        dtype: float or string type, float by default. int not allowed because cant hold NaN
        missing_values: extra strings to treat as missing like "NA" or "?"
        nan_policy: what to do with missing - keep, omit, fill, raise. default keep
        fill_value: value to use if nan_policy is fill, must give it then
        encoding: file encoding, default utf-8

    Returns:
        array: 2D numpy array with shape (n_samples, n_features)
        even if file has 1 row or 1 col it is still 2D. if omit, rows can be 0.

    Raises:
        ValueError: if delimiter empty, skip_header negative, bad nan_policy/dtype,
                    no fill_value for fill, empty file, rows with different lengths,
                    or missing found when policy is raise
        TypeError: if delimiter not str or missing_values wrong type
        FileNotFoundError: if file not found

    Time: O(n_samples * n_features) one pass
    Space: O(n_samples * n_features) whole file in memory
    """
    # load csv file into 2D numpy array
    delimiter, skip_header, dt, tokens = _validate_options(
        delimiter, skip_header, dtype, missing_values, nan_policy, fill_value
    )
    return _parse(
        path, delimiter=delimiter, skip_header=skip_header, usecols=usecols,
        dt=dt, tokens=tokens, nan_policy=nan_policy, fill_value=fill_value,
        encoding=encoding, allow_empty=False, where=repr(path),
    )


def iter_csv_chunks(
    path,
    chunk_size: int,
    *,
    delimiter: str = ",",
    skip_header: int = 0,
    usecols=None,
    dtype=float,
    missing_values=None,
    nan_policy: str = "keep",
    fill_value=None,
    encoding: str = "utf-8",
):
    """
    Read csv file in chunks, for big files.

    Params:
        path: file path
        chunk_size: how many rows per chunk, must be >=1
        delimiter, skip_header, usecols, dtype, missing_values, nan_policy,
            fill_value, encoding: same as load_csv
        skip_header is only skipped once at start

    Yields:
        chunk: 2D array shape (rows_in_chunk, n_features).
        All chunks have same number of cols. If omit, chunk can be smaller than chunk_size.
        Empty chunks are skipped. vstack of all chunks == load_csv

    Raises:
        ValueError: if chunk_size <1 or any option invalid, or missing with raise policy
        TypeError: if chunk_size not int
        FileNotFoundError: if file not found. File check happens right away not on first chunk.

    Time: O(n_samples * n_features) total
    Space: O(chunk_size * n_features) only one chunk in memory
    """
    # read csv in chunks, for big files
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(
            f"chunk_size must be an int, got {type(chunk_size).__name__}"
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    delimiter, skip_header, dt, tokens = _validate_options(
        delimiter, skip_header, dtype, missing_values, nan_policy, fill_value
    )
    # check file exists now, not later
    with open(path, encoding=encoding):
        pass
    chunk_size = int(chunk_size)

    def _blocks():
        with open(path, encoding=encoding) as fh:
            for _ in range(skip_header):
                next(fh, None)
            while True:
                lines = list(itertools.islice(fh, chunk_size))
                if not lines:
                    return
                block = _parse(
                    _stdio.StringIO("".join(lines)),
                    delimiter=delimiter, skip_header=0, usecols=usecols,
                    dt=dt, tokens=tokens, nan_policy=nan_policy,
                    fill_value=fill_value, encoding=encoding,
                    allow_empty=True, where=repr(path),
                )
                if block.size:
                    yield block

    return _blocks()


def load_xy(path, *, target: int = -1, **kwargs):
    """
    Load csv and split into X and y.

    Params:
        path: file path
        target: which column is y, default -1 means last col. negative counts from end
        **kwargs: same as load_csv like delimiter, skip_header, nan_policy etc.
            omit drops row from both X and y

    Returns:
        X: 2D array shape (n_samples, n_features-1)
        y: 1D array shape (n_samples,)
        X and y have same number of rows

    Raises:
        ValueError: if file has <2 cols, target out of range, or load_csv errors
        TypeError: if target not int

    Time: O(n_samples * n_features)
    Space: O(n_samples * n_features)
    """
    # load csv and split into X and y
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)):
        raise TypeError(f"target must be an int, got {type(target).__name__}")
    data = load_csv(path, **kwargs)
    n_cols = data.shape[1]
    if n_cols < 2:
        raise ValueError(
            f"need at least 2 columns to split into X and y, got {n_cols}"
        )
    if not -n_cols <= target < n_cols:
        raise ValueError(
            f"target index {target} is out of range for {n_cols} columns"
        )
    target = int(target) % n_cols
    y = data[:, target].copy()
    X = np.delete(data, target, axis=1)
    return X, y


def read_header(path, *, delimiter: str = ",", skip_rows: int = 0,
                encoding: str = "utf-8") -> list[str]:
    """
    Get header names from csv file.

    Params:
        path: file path
        delimiter: separator, default ","
        skip_rows: how many lines to skip before header, default 0
        encoding: default utf-8

    Returns:
        list of str: header names, stripped of whitespace. length = num cols

    Raises:
        ValueError: if delimiter empty, skip_rows negative, or no header line
        TypeError: if delimiter not str

    Time: O(line length) only reads one line
    Space: O(line length)
    """
    # get header line from csv file
    delimiter = _check_delimiter(delimiter)
    if skip_rows < 0:
        raise ValueError(f"skip_rows must be >= 0, got {skip_rows}")
    with open(path, newline="", encoding=encoding) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for _ in range(skip_rows):
            next(reader, None)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"no header line found in {path!r}")
    return [name.strip() for name in row]


def save_csv(path, X, *, delimiter: str = ",", header=None,
             fmt: str = "%.18e", encoding: str = "utf-8") -> None:
    """
    Save array to csv file.

    Params:
        path: where to save
        X: array to save, shape (n_samples,) or (n_samples, n_features). 1D becomes one column
        delimiter: separator, default ","
        header: optional list of column names, length must match num cols
        fmt: format string for numbers, default "%.18e" keeps float64 exact
        encoding: default utf-8

    Returns:
        None

    Raises:
        ValueError: if X not 1D or 2D, or header length wrong
        TypeError: if delimiter not str

    Time: O(n_samples * n_features)
    Space: O(n_samples * n_features)
    """
    # save array to csv file
    delimiter = _check_delimiter(delimiter)
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"X must be 1-D or 2-D, got an array with shape {arr.shape}"
        )
    head = ""
    if header is not None:
        names = [str(h) for h in header]
        if len(names) != arr.shape[1]:
            raise ValueError(
                f"header has {len(names)} names but X has "
                f"{arr.shape[1]} columns"
            )
        head = delimiter.join(names)
    np.savetxt(path, arr, delimiter=delimiter, header=head, comments="",
               fmt=fmt, encoding=encoding)
