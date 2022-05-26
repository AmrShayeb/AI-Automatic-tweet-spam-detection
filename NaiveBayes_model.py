import csv
import re
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.combine import SMOTETomek
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.naive_bayes import GaussianNB
import sys
import os
py_file_location = "/content/drive/My Drive/Colab Notebooks"
sys.path.append(os.path.abspath("/content/drive/MyDrive/Colab Notebooks/data.py"))
!cp /content/drive/MyDrive/Colab\ Notebooks/data.py /content
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb

import sys
import time
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
print("Naive Bayes")
df = pd.read_csv("/training.csv")
X = df.drop('Type', axis=1)
Y = df['Type']
# Balancing approcah #Over-sample using SMOTE followed by under-sampling using Edited Nearest Neighbours.
smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X, Y)
X = X_smt
Y = y_smt
# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Testing new
count_class_0, count_class_1 = df.Type.value_counts()
# Divide by class
df_class_0 = df[df['Type'] == 0]
df_class_1 = df[df['Type'] == 1]
#          #          #
model = GaussianNB()
start = time.time()
model.fit(X_train, Y_train)
stop = time.time()
print(f"time needed = {stop-start}s")
GaussianNB(priors=None, var_smoothing=1e-09)
# saving tree model
pkl_filename = "NaiveBayes_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
Y_pred = model.predict(X_test)
#print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(metrics.classification_report(Y_test, Y_pred))


# Load from file
with open("NaiveBayes_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)
# getting the data
if len(sys.argv) > 2 and ( sys.argv[2] == 'p' or sys.argv[2] == 'P' ) : # need to do processing on the data
    readFile("Test")

df = pd.read_csv("/proccessed_test.csv")
X = df.drop('Type', axis=1)
Y = df['Type']
# Calculate the accuracy score and predict target values
score = pickle_model.score(X, Y)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X)
print(metrics.classification_report(Y, Ypredict))