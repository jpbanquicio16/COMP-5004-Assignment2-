import os
import numpy as np

# for handling csv files
# Section 1 - io.py

__all__ = ["load_csv", "save_csv"]

def _validate_path(path):
    # checking if path is valid, raise error if not
    if not isinstance(path, str):
        raise TypeError("path must be string")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path

def load_csv(path, delimiter=',', skip_header=True, missing='skip', fill_value=0.0):
    """
    Load csv using numpy.

    Parameters

    path : str
        file path
    delimiter : str, default ','
        delimiter like ',' or '\t'
    skip_header : bool, default True
        skip first row if True
    missing : {"skip", "fill"}, default "skip"
        what to do with missing values
    fill_value : float, default 0.0
        used when missing='fill'

    Returns

    ndarray, shape (n_samples, n_features)
        data as numpy array

    Raises

    FileNotFoundError
        if file not found
    ValueError
        if missing value param wrong or file empty

    Notes

    Used np.genfromtxt which is built on loadtxt.
    Time O(n*m) where n rows m cols, Space O(n*m)
    NaN handling: skip removes rows with NaN, fill replaces with fill_value
    """
    _validate_path(path)

    if missing not in ("skip", "fill"):
        raise ValueError("missing should be skip or fill")

    skip = 1 if skip_header else 0

    try:
        data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip,
                             dtype=float, filling_values=fill_value)
    except Exception:
        # fallback to loadtxt if genfromtxt fails
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skip, dtype=float)

    if data.size == 0:
        raise ValueError("empty file")

    # handle missing values
    if missing == "skip":
        if data.ndim == 1:
            data = data[~np.isnan(data)]
        else:
            data = data[~np.isnan(data).any(axis=1)]
    else:
        # fill
        data = np.where(np.isnan(data), fill_value, data)

    return data

def save_csv(path, data, delimiter=','):
    """
    Save array to csv.

    Parameters

    path : str
    data : array_like, shape (n_samples, n_features)
    delimiter : str, default ','

    Returns

    None

    Raises

    ValueError if empty

    Notes

    Used np.savetxt, Time O(n*m), Space O(1)
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("empty array")

    # make dir if not exists
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    np.savetxt(path, arr, delimiter=delimiter, comments='')
