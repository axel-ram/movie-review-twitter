#!/usr/bin/python3
#automatically compile while importing in r4.py file
#have to train the model first
#checks for 5000 dataset rows memory use about 1gb
#checked for 40000 dataset rows, memory use about 4gb (do not run if you system dont have atleast 6gb ram otherwise program will crash) takes 5 min to analyze data
#checked upto 150000 dataset rows, only ran one time, memory use full 6.5gb (8g on system), took 15 min to analyze data
#working on to import a trained model directly
# importing libs
import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords
s = stopwords.words('English')
temp=[]
for word in s:
	#removed all the stopwords having negative words
    if not (word.endswith(('not', 'no', 'nor','n\'','n\'t','dn','an','sn','weren','tn','\''))) :
        temp.append(word)

#importing the dataset
dataset = pd.read_csv('data.tsv', delimiter = "\t", quoting = 3)


#cleaning dataset texts
corpus = []
for i in range(0,5000):
    review = re.sub('[^a-xA-z]',' ',dataset['Phrase'][i])
    review = review.lower()
    review = review.split()
    #stemming
    review = [ps.stem(word) for word in review if not word in temp]
    review = ' '.join(review)
    corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:5000,3]

# splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
print('model trained')
# dont need to predict here
# predicting the Test set results
#y_pred = classifier.predict(X_test)

# output given below
# making the Confusion Matrix
'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

[[ 125  143   40    3    2]
 [ 126  616  533   25    3]
 [  42  353 3414  424   23]
 [   0   41  633  802  160]
 [   0    3   51  255  183]]


efficiency
0 - 0.75
1 - 0.77
2 - 0.87
3 - 0.77
4 - 0.76
 '''
