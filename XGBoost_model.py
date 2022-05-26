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
print("XGBoost")
df = pd.read_csv("/training.csv")
X = df.drop('Type', axis=1)
Y = df['Type']
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # # # Feature Scaling   the same result
    # sc = sklearn.preprocessing.StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, use_label_encoder=False)
start = time.time()
model.fit(X_train, y_train)
stop = time.time()
print(f"time needed = {stop - start}s")
# saving tree model
pkl_filename = "XGBoost_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
y_pred = model.predict(X_test)
#print(model.score(X_test, y_test))
print(metrics.classification_report(y_test,y_pred))


# Load from file
with open("XGBoost_model.pkl", 'rb') as file:
    pickle_model = pickle.load(file)
    # getting the data
if len(sys.argv) > 2 and (sys.argv[2] == 'p' or sys.argv[2] == 'P'):  # need to do processing on the data
    readFile("Test")
df = pd.read_csv("/proccessed_test.csv")
X = df.drop('Type', axis=1)
Y = df['Type']
# Calculate the accuracy score and predict target values
score = pickle_model.score(X, Y)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X)
print(metrics.classification_report(Y, Ypredict))
