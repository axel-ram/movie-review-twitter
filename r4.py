#!/usr/bin/python3
#still a prototype, working on it...
import movie_review as mr
import collections
import sys
import  tweepy
from textblob import  TextBlob

# creating  consumer key and secret key 
#  to authenticate twitter from ur accont 
consumer_key='mPLp5SGMADa7HOKqtubHUTdZ2'

consumer_secret='zZyoSpPgMCG5AznXXtlhivM6hW6cJ8ILVPnL7XcoLgy4IpC9Cz'

#  creating  access key and secret key 

access_key='1007186669244518400-6aFdNPYk6A2byYiUtLcQIsMLrTACAp'
access_secret='T6oPrSC9aaU5s1BOsHpRNuFWRcyc30q4GhNwAMTzC4KB9'


#  connecting for authenitcation 
#  just like a sessional variable
auth=tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_key,access_secret)

#  connecting to API  
connect=tweepy.API(auth)

review_list = []
def recommend_sys(movie_name):
	# serching for tweets to predict movie review and rating
	get_data=connect.search(movie_name, count=500)
	print("breakpoint 1")
	review_list = []
	for i  in  get_data:
	    analysis=TextBlob(i.text)
	    #ignoring nearly neutral polarity tweets
	    #if (analysis.sentiment.polarity > 0.4 or analysis.sentiment.polarity < -0.4):
	    review = mr.re.sub('[^a-xA-z]',' ',str(analysis))
	    review = review.lower()
	    review = review.split()
	    #stemming
	    review = [mr.ps.stem(word) for word in review if not word in mr.temp]
	    review = ' '.join(review)
	    review_list.append(review)
	print("breakpoint 2")
	#creating bag of words for training the model
	from sklearn.feature_extraction.text import CountVectorizer
	cv = CountVectorizer(max_features=100)
	X = cv.fit_transform(review_list).toarray()
	if(len(X[0]) < 100):
		print("sorry this movie was not much reviewed and tweeted so system can not predict rating")
		sys.exit()
	# prediction from tweets
	pred = mr.classifier.predict(X)
	print('predicting rating')
	#rating variables
	zero=0
	one=0
	two=0
	three=0
	four=0
	#determning rating
	dict = collections.Counter(pred)
	zero = dict[0] * 0.75
	one = dict[1] * 0.77
	two = dict[2] * 0.87
	three = dict[3] * 0.77
	four = dict[4] * 0.76
	rating = max(zero,one,two,three,four)
	print(rating/len(X))



print('reciew system starting...')
# taking input for movie name
while(True):
	movie_name = input("enter movie name")
	if(movie_name == 'tixe'):
		break
	recommend_sys(movie_name)
