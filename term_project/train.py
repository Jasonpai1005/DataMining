import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
# from imblearn.combine import SMOTEENN
# from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, NearMiss
# import xgboost as xgb
# import catboost as cb

print("Loading dataset")
train_df = pd.read_csv('./newdataset/train_dataset.csv')
test_df = pd.read_csv('./newdataset/test_dataset.csv')
sub = pd.read_csv('./newdataset/sample_submission.csv')
# train_df = train_df.drop(train_df.columns[66:114], axis=1)
test_df = test_df.drop('id', axis=1)
# test_df = test_df.drop(test_df.columns[66:114], axis=1)
print("Loading Done!")

print("Encoding")
label_encoder = LabelEncoder()

for i in train_df.columns:
    train_df[i] = label_encoder.fit_transform(train_df[i])
for j in test_df.columns:
    test_df[j] = label_encoder.fit_transform(test_df[j])

print("Modeling")
X = train_df.drop('target', axis=1)
Y = train_df['target']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# print("Resampling")
# X_res, Y_res = EditedNearestNeighbours().fit_resample(X_train, Y_train)

print("Training")
ada_model = AdaBoostClassifier(n_estimators=1000, random_state=0)
ada_model.fit(X_train, Y_train)

val_pred = ada_model.predict(X_val)
print(classification_report(Y_val, val_pred))
print()
print("Accuracy : ", accuracy_score(Y_val, val_pred))
print("ROC : ", roc_auc_score(Y_val, val_pred))

prediction = ada_model.predict(test_df)
sub['target'] = prediction
sub.head()

sub.to_csv('new_sub_ada.csv', index=False)
# model_with("Random Forest Classifier", X_train, X_val, Y_train, Y_val)