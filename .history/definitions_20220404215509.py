import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score,classification_report,confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_auc(fpr, tpr, label=None):
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot(fpr, tpr, color='black', lw=1)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.text(0.5, 0.3, 'ROC curve (area=%0.2f)' % roc_auc_score(y_test, probs))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.show()
    st.pyplot()

def plot_roc(fpr, tpr, label=None):
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    st.pyplot()
