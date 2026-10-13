import numpy as np
import csv
import os

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        return data
    except Exception:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        return np.array(rows, dtype=float)

def save_csv(path, data):
    np.savetxt(path, data, delimiter=',', comments='')
