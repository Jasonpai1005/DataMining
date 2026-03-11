import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

print("Loading dataset")
train_df = pd.read_csv('./dataset/train_dataset.csv')
print("Loading Done!")

print("Encoding")
label_encoder = LabelEncoder()
one_hot = OneHotEncoder()

for i in train_df.columns:
    train_df[i] = label_encoder.fit_transform(train_df[i])

print("Modeling")
X = train_df.drop('target', axis=1)
Y = train_df['target']
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.25, random_state = 0)

def model_with(model_name, X_train, X_val, Y_train, Y_val):
    model = linear_model.LogisticRegression(solver='saga', max_iter=1000, n_jobs=80)
    model.fit(X_train, Y_train)

    val_pred = model.predict(X_val)
    print(classification_report(Y_val, val_pred))
    print()
    print("Accuracy : ", accuracy_score(Y_val, val_pred))
    print("ROC : ", roc_auc_score(Y_val, val_pred))

print("Training")
model_with("Random Forest Classifier", X_train, X_val, Y_train, Y_val)