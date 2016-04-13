# -*- coding: ISO-8859-1 -*-
# encoding=utf8  

import sys  

reload(sys)  
sys.setdefaultencoding('utf-8')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import time
import math
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn import ensemble
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.datasets import load_digits
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from nltk.corpus import stopwords # Import the stop word list
from nltk.stem.lancaster import LancasterStemmer
import re
import math
import string
from nltk.stem.porter import *
stemmer = PorterStemmer()
from sklearn.metrics import mean_squared_error as MSE
from sklearn.feature_extraction.text import TfidfTransformer
from bs4 import BeautifulSoup
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import inflect

# Chargement des donnees ###############
path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
# dans le terminal : iconv -f utf-8 -t utf-8 -c attributes.csv > attributes_clean.csv
train_fname = path + '/input/train_clean.csv'
test_fname = path + '/input/test_clean.csv'
fname_explore_attr = path + '/data/explore_attr.csv'
attributes_fname = path + '/input/attributes.csv'
product_descriptions_fname = path + '/input/product_descriptions.csv'
final_xtrain = path + '/src/data_res/xtrain.csv'
final_xtest = path + '/src/data_res/xtest.csv'
final_ytrain = path + '/src/data_res/ytrain.csv'
final_id_test = path + '/src/data_res/id_test.csv'
fname_bullet_final = path + '/data/bullet_final.csv'

# lecture des fichiers
df_ini = pd.read_csv(train_fname, sep=',', encoding="ISO-8859-1")
#df = df_ini.drop(['VARIABLE_CIBLE'], axis=1)
df_test = pd.read_csv(test_fname, sep=',', encoding="ISO-8859-1")

df_bullet = pd.read_csv(fname_bullet_final, sep=',', encoding="ISO-8859-1")

# creation des fichiers y_train et id_train
y_train = df_ini['relevance']
id_train = df_ini['id']
id_test = df_test['id']
y_train.columns = ['relevance']
ytrain = pd.DataFrame(y_train).to_csv(final_ytrain, header=['relevance'], index=False)
pd.DataFrame(id_test).to_csv(final_id_test, header=['id'], index=False)


df_attributes = pd.read_csv(attributes_fname, sep=',')
df_attributes = df_attributes.fillna(' ')

# Les autres attribus sont traités dans le script exploration attribues
df_brand = df_attributes[df_attributes['name'] == 'MFG Brand Name']
df_brand = df_brand.drop(['name'], axis=1)
df_brand = df_brand.rename(columns={'value': 'brand'})

df_product_descriptions = pd.read_csv(product_descriptions_fname, sep=',')

train_row = df_ini.shape[0]

id_global = pd.concat([id_train, id_test], axis=0, ignore_index=True)

frames = []
df_global = pd.concat([df_ini, df_test], axis=0, ignore_index=True)

df = df_global.drop(['relevance'], axis=1)
df = df.drop(['id'], axis=1)


result = pd.merge(df, df_product_descriptions, how='left', on='product_uid')
result = pd.merge(result, df_brand, how='left', on='product_uid')
result = pd.merge(result, df_bullet, how='left', on='product_uid')
result['brand'] = result['brand'].replace(np.nan, ' ', regex=True)
result = result.fillna(' ')

result_review = result#[:500]

# result['brand'] = result['brand'].replace(np.nan, ' ', regex=True)
# result_review = result.drop(['product_uid'], axis=1)

nonNumericColumns = []
numericColumns = []

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numericColumns = result.select_dtypes(include=numerics).columns
nonNumericColumns = result.columns.difference(numericColumns)


wnl = WordNetLemmatizer()
st = LancasterStemmer()


def review_to_words(review_text):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters
    global i
    i = review_text
    p = inflect.engine()
    review_text = str(review_text).lower()
    # ACCENT
    review_text = re.sub(r"(\w)\.([A-Z])", r"\1 \2", review_text)
    # Saut de ligne mal geres
    review_text = re.sub(r"(\W)([A-Z])", r"\1 \2", review_text)
    review_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", review_text)

    # review_text = re.sub("(\d)+", lambda m: p.number_to_words(m.group(0)), review_text)
    # Clean ponctuation manuellement
    review_text = review_text.replace("deg.", " degre")
    review_text = review_text.replace(",", " ")
    review_text = review_text.replace("?", " ")
    review_text = review_text.replace(":", " ")
    review_text = review_text.replace("-", " ")
    review_text = review_text.replace("_", " ")
    review_text = review_text.replace("$", " ")
    review_text = review_text.replace("#", " ")
    review_text = review_text.replace("(", " ")
    review_text = review_text.replace(")", " ")
    review_text = review_text.replace("&", " ")
    review_text = review_text.replace("'", " ")
    review_text = review_text.replace("..", ".")
    review_text = review_text.replace("//", "/")
    review_text = review_text.replace(" / ", " ")
    review_text = review_text.replace(" \\ ", " ")
    review_text = review_text.replace(".", " . ")
    review_text = str(review_text).decode("utf-8").lower()
    review_text = review_text.replace(" x ", " xbi ")
    review_text = review_text.replace("*", " xbi ")
    review_text = review_text.replace(" by ", " xbi ")
    review_text = re.sub(r"([0-9])([a-z])", r"\1 \2", review_text)
    review_text = re.sub(r"([a-z])([0-9])", r"\1 \2", review_text)
    review_text = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", review_text)
    review_text = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", review_text)
    # letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = review_text.decode("utf-8").lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # sentence = [wnl.lemmatize(w) for w in words]
    # All stop word
    stops = set(stopwords.words("english"))
    # 5. Remove stop words and steamming
    meaningful_words = [stemmer.stem(w) for w in words if not (w or stemmer.stem(w)) in stops]
    # 6. Join the words back into one string separated by space,
    # and return the result.
    sentence = (" ".join(meaningful_words))
    # On remplace des mots par des synonymes
    sentence = sentence.replace("exterior", "outdoor")
    sentence = sentence.replace("interior", "indoor")
    sentence = sentence.replace("°", " degre")
    sentence = sentence.replace("deg.", " degre")
    return str(sentence)
    # .translate(string.maketrans("", ""), string.punctuation)



print '####### Start cleaning #########'
for col in nonNumericColumns:
    print col
    #result[col] = result[col].apply(lambda x: lower_str(x.encode("utf-8")))
    result_review[col] = result_review[col].apply(lambda x: review_to_words(x.encode("utf-8")))
print 'End cleaning'

result_review['product_info'] = result_review['search_term']+" "+result_review['product_title'] +" "+result_review['product_description']

# df_explore_attribus = pd.read_csv(fname_explore_attr, sep=',')
# df_explore_attribus = df_explore_attribus.fillna(' ')
# result_review = pd.merge(result_review, df_explore_attribus, how='left', on='product_uid')
result_review_final = result_review.drop(['product_uid'], axis=1)
result_review_final = result_review_final.applymap(lambda x: str(x))
fname = path + '/data/clean_data.csv'
result_review_final.to_csv(fname, index=False, encoding="utf-8")

fname_data_concat = path + '/data/concat_data.csv'
# result.to_csv(fname_data_concat, index=False)

print 'END START CHALLENGE'




