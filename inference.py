import configparser as cfp
import pickle
import praw
import re
import sklearn
import ast
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


REPLACE_SPACES = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

with open('stopwords.txt', 'r') as f:
    STOPWORDS = ast.literal_eval(f.read())

def get_reddit_credentials():
	config = cfp.ConfigParser()
	config.read('config.ini')
	return config['reddit']['client_id'], config['reddit']['client_secret'], config['reddit']['user']

client_id, client_secret, user = get_reddit_credentials()
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user)

def clean_text(text):
	text = text.lower()
	text = REPLACE_SPACES.sub(' ', text) 
	text = BAD_SYMBOLS.sub(' ', text)
	text = text.replace('x', ' ')
	text = text.replace('\n', ' ')
	
	text = ' '.join(word for word in text.split() if word not in STOPWORDS)
	return text

def get_flair(url, loaded_model):
	data_from_url = reddit.submission(url=url)
	data = {}

	data["Title"] = data_from_url.title
	data["url"] = data_from_url.url
	data["Body"] = data_from_url.selftext
	comments = []
	data_from_url.comments.replace_more(limit=10)
	for comment in data_from_url.comments:
		comments.append(str(comment.body))
	data["Comments"] = str([' '.join(sentence).strip() for sentence in comments])

	data["Combine"] = data["Title"]
	if type(data['Body']) != float:
		data["Combine"] += ' ' + data["Body"]
	if type(data['Comments']) != float:
		data["Combine"] += ' ' + data["Comments"]
	data['Combine'] = clean_text(data['Combine'])
	data['Combine'] = data['Combine'].replace('\d+', ' ')

	return loaded_model.predict([data['Combine']])