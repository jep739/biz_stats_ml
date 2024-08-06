import datetime
import getpass
import math
import multiprocessing
import os
import sys
import time
import warnings

import gcsfs
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import seaborn as sns
import tensorflow as tf
from IPython.display import clear_output, display
from joblib import Parallel, delayed
from pyjstat import pyjstat
from scikeras.wrappers import KerasRegressor
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from xgboost import XGBRegressor

from dapla import FileClient
from dapla.auth import AuthClient
import sgis as sg  # If sgis is re-imported, ensure it is intended.

sys.path.append("../functions")
import ao
import input_data
import kommune
import kommune_inntekt
import kommune_pop
import kommune_translate
import kpi
import visualisations 
import create_datafiles
import ml_modeller
import dash_application
import oppdateringsfil

fs = FileClient.get_gcs_file_system()

warnings.filterwarnings("ignore")
