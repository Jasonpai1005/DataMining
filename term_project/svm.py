import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

print("Loading dataset")
train_df = pd.read_csv('./newdataset/train_dataset.csv')
test_df = pd.read_csv('./newdataset/test_dataset.csv')
sub = pd.read_csv('./newdataset/sample_submission.csv')
# train_df = train_df.drop(train_df.columns[66:114], axis=1)
# test_df = test_df.drop('id', axis=1)
# test_df = test_df.drop(test_df.columns[66:114], axis=1)
print("Loading Done!")

print(train_df.info())
print(test_df.info())