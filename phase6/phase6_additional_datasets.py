# Phase 6 notebook code - will be converted to .ipynb

# ─────────────────────────────────────────────
# CELL 1: Imports & Config
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

DATA_DIR = Path("../data")
SEED = 42
np.random.seed(SEED)

PHASE6_BASELINE_MAG_MAE = 26.6   # Phase 3 Ridge w36 MAE
PHASE6_BASELINE_TIM_MAE = 7.6    # Phase 4 Ridge w36 MAE

print("Phase 6: Additional Datasets")
print("=" * 50)
print(f"Baseline magnitude MAE (Phase 3 Ridge w36): {PHASE6_BASELINE_MAG_MAE}")
print(f"Baseline timing MAE   (Phase 4 Ridge w36): {PHASE6_BASELINE_TIM_MAE}")

