# Extracting the compressed dataset
# from zipfile import ZipFile
# dataset = r'C:\Users\RentoBees\PycharmProjects\Sentiment-Analysis\sentiment140.zip'
#
# with ZipFile(dataset, 'r') as zp:
#     zp.extractall()
#     print('The data has been extracted.')

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import nltk
nltk.download('stopwords')

#Loading data from csv file to pandas dataframe
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv(r'C:\Users\RentoBees\PycharmProjects\Sentiment-Analysis\data.csv', names=column_names, encoding = 'ISO-8859-1')
print(twitter_data.shape) #checking the no. of rows and columns
print(twitter_data.head())
print(twitter_data.isnull().sum()) #for missing values/text
# '0' stands for 'negative' and '1' stands for 'positive'
twitter_data.replace({'target':{4:1}}, inplace=True) #convert target '4' to '1'
print(twitter_data['target'].value_counts()) #print the distribution

def stemming(content):
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    stop_words = set(stopwords.words('english'))
    port_stem = PorterStemmer()

    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
num_cores = multiprocessing.cpu_count()
twitter_data['stemmed_content'] = Parallel(n_jobs=num_cores)(
    delayed(stemming)(text) for text in twitter_data['text']
)
print(twitter_data.head())
print(twitter_data['stemmed_content'])
X = twitter_data['stemmed_content']
Y = twitter_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#to convert textual data into numerical data
Vectorizer = TfidfVectorizer()
X_train = Vectorizer.fit_transform(X_train)
X_test = Vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score on the training data :', training_data_accuracy)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score on the test data :', test_data_accuracy)

import pickle
file_name = 'trained_model.sav'
pickle.dump(model, open(file_name, 'wb'))
loaded_model = pickle.load(open(r'C:\Users\RentoBees\PycharmProjects\Sentiment-Analysis\trained_model.sav', 'rb'))

index = int(input("Enter tweet index (0 to {}): ".format(X_test.shape[0] - 1)))
if 0 <= index < X_test.shape[0]:
    X_new = X_test.getrow(index)  # use .getrow for sparse matrix!
    print("Actual label:", Y_test.iloc[index])
    prediction = model.predict(X_new)
    print("Prediction:", "Negative Tweet" if prediction[0] == 0 else "Positive Tweet")
else:
    print("Invalid index. Please enter a number between 0 and {}.".format(X_test.shape[0] - 1))
