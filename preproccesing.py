import csv
import re
import pickle
import pandas
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
sys.path.append(os.path.abspath("/content/drive/MyDrive/Colab Notebooks/raw_input.py"))
!cp /content/drive/MyDrive/Colab\ Notebooks/raw_input.py /content
from raw_input import Data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import string
import re
import sys
import time
import pandas as pd
dataset=pd.read_csv("/train.csv")
#dataset.fillna(dataset.mean(), inplace=True)
#dataset['location']=dataset['location'].fillna((dataset['location'].mode()[0]))
dataset1=dataset.dropna()
dataset1.isnull().sum()
dataset1.to_csv("ttt.csv")


data = []
docs = []  # used in TF_IDF

dict = {}
counter = 0
data.clear()
def wordCount(tweetContent):
    token = tweetContent.split(" ")
    return len(token)


def capWordCount(tweetContent):
    token = tweetContent.split(" ")
    count = 0
    for i in token:
        if (len(i) > 1 and i.isupper() == True):
            count += 1
    return count


def handlingSpaces(tweetContent):
    pattern = re.compile(r'\s+')
    return re.sub(pattern, '', tweetContent)
def handlingnonnum(Id):
    return re.sub('[^0-9]','', Id)  
    # Id.translate({"[^a-zA-Z ]":None})

# checks if the content has URL using regex
def hasURL(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, text)
    return [x[0] for x in url]


def removeSpecialCharacter(text):
    text = re.sub("[^a-zA-Z0-9 ]", " ", str(text))
    return text


def convertToSmallLetters(text):
  text = text.lower()
  return text
 


# removing stop words using sckit-learn
def removeStopWords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence

# converti g the review content to stem for each word
def stemSentence(sentence):
    ps = PorterStemmer()
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(ps.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


# calculate TF_IDF for each sentence with respect to the whole dataset 'docs'
def calculateTF_IDF():
    print("Entered TF_IDF")
    tfIdfTransformer = TfidfTransformer(use_idf=True)
    countVectorizer = CountVectorizer(stop_words=None)
    wordCount = countVectorizer.fit_transform(docs)
    newTfIdf = tfIdfTransformer.fit_transform(wordCount)
    for i in range(0, len(docs)):
        df = pd.DataFrame(newTfIdf[i].T.todense(), index=countVectorizer.get_feature_names(), columns=["TF-IDF"])
        dd = df.values
        # dd = [item for elem in dd for item in elem]
        data[i].setDataVector(dd)



def gettwitterID(d):
    return d.gettwitterID()


# cosine similarity is used to check the similarity between each two sentences and the max similarity is stored
def calculateSimilarity():
    data.sort(key=gettwitterID)
    print("Entered Similarity")
    for i in range(0, len(data)):
        for j in range(0, len(data)):
            if (i != j):
                if (data[i].gettwitterID() > data[j].gettwitterID()):
                    continue
                if (data[i].gettwitterID() == data[j].gettwitterID()):
                    # cos similarity
                    cos_sim = cosine_similarity(data[i].getDataVector().reshape(1, -1),
                                                data[j].getDataVector().reshape(1, -1))
                    if (cos_sim[0][0] > data[i].getSimilarity()):
                        data[i].setSimilarity(cos_sim[0][0])
                else:
                    break

# write the final dataset into a csv file
def writeIntoCSVfile(fileName):
    with open(fileName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["twitterID",  "followers", "following", "actions", "isretweet", 
             "Type", "tweetContentLength", "hasURL", "similarity",  "perCapWords"])
        for i in range(1, len(data)):
            writer.writerow(
                [data[i].gettwitterID(), data[i].getfollowers(), data[i].getfollowing(),
                 data[i].getactions(), data[i].getisretweet(), 
                 data[i].gettype(), data[i].gettweetContentLength(), data[i].getHasURL(),
                 data[i].getsimilarity(),  data[i].getPerCapWords()])



with open("ttt.csv") as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  for col in csv_reader:
    print(col[1],col[2], col[3], col[4], col[5], col[6], col[7], col[8])
    temp = Data(col[1], col[2], col[3], col[4], col[5], col[6], col[7], col[8],
                                    0, 0, 0, 0) 
    print(temp.getfollowers) 
    temp.settwitterID(handlingnonnum(temp.gettwitterID()))
    # handling spaces
    temp.setfinaltweetContent(handlingSpaces(temp.gettweetContent()))
    # word count
    temp.settweetContentLength(wordCount(temp.gettweetContent()))
    # number of capitalized words
    count = capWordCount(temp.gettweetContent())
    temp.setPerCapWords((float(count) / temp.gettweetContentLength()) * 100.0)
    # has URL
    if (hasURL(temp.gettweetContent()) != []):
      temp.setHasURL(1)
    else:
      temp.setHasURL(0)
    # remove special characters
    temp.setlocation(handlingSpaces(temp.getlocation()))
    # remove special characters
    temp.setfinaltweetContent(removeSpecialCharacter(temp.getfinaltweetContent()))
    # convert to small
    temp.setfinaltweetContent(convertToSmallLetters(temp.getfinaltweetContent()))
    # remove stop words
    temp.setfinaltweetContent(removeStopWords(temp.getfinaltweetContent()))
    # stemming
    temp.setfinaltweetContent(stemSentence(temp.getfinaltweetContent()))
    docs.append(temp.getfinaltweetContent())
    data.append(temp)
  writeIntoCSVfile('CleanedData_Preprocessed.csv')
  print("Go into calculateTF_IDF : ")
  calculateTF_IDF()
  print("Go into calculateSimilairty : ")
  calculateSimilarity()
  writeIntoCSVfile('ready.csv')
# counting the words in contents








