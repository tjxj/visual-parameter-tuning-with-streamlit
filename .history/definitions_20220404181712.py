import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score,roc_curve,auc,mean_squared_error,accuracy_score
from sklearn  import metrics
import lightgbm as lgb
import altair as alt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


#from LightGBM import LGBMRegressor,LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')