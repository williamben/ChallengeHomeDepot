# # -*- coding: ISO-8859-1 -*-
# # encoding=utf8  

# import sys  

# reload(sys)  
# sys.setdefaultencoding('utf-8')

import numpy as np
import pandas as pd
from nltk.stem.porter import *
from collections import Counter
stemmer = PorterStemmer()

path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
data_cleaning = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot/data/clean_data.csv'
fname_explore_attr = path + '/data/explore_attr.csv'
result_review = pd.read_csv(
    data_cleaning, sep=',', na_values='(MISSING)', encoding="utf-8")

#

# df_explore_attribus = pd.read_csv(fname_explore_attr, sep=',', encoding="ISO-8859-1")
# result_review = pd.merge(result_review, df_explore_attribus, how='left', on='product_uid')
# result_review = result_review.fillna(' ')
# result_review = result_review.drop(['product_uid'], axis=1)


# def to_str(word):
#     word = str(word).lower()
#     if word == 'nan':
#         word = ' '
#     return word


# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# numericColumns = result_review.select_dtypes(include=numerics).columns
# nonNumericColumns = result_review.columns.difference(numericColumns)

# for col in nonNumericColumns:
#     print col
#     #result[col] = result[col].apply(lambda x: lower_str(x.encode("utf-8")))
#     result_review[col] = result_review[col].apply(lambda x: to_str(x))
# print 'End cleaning'

result_review = result_review.applymap(lambda x: str(x))

counter_search_term = Counter(result_review['search_term'])
counter_product_description = Counter(result_review['product_description'])
counter_product_title = Counter(result_review['product_title'])
counter_brand = Counter(result_review['brand'])



def search_term_description(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['product_description']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0


def ratio_search_term_description(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['product_description']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords) / len(row['search_term'].split())
    else:
        return 0


def search_term_product_title(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['product_title']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0


def ratio_search_term_product_title(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['product_title']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords) / len(row['search_term'].split())
    else:
        return 0


def search_term_brand(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['brand']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0


def ratio_search_term_brand(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['brand']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords) / len(row['search_term'].split())
    else:
        return 0


def brand_in_search_term(row):
    return int(str(row['brand']) in str(row['search_term'] or str(row['search_term']).replace(' ', '')))

def search_term_color(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['color']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0

def search_term_in_out(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['in_out']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0

def search_term_material(row):
    if (row['search_term'] is not None and (len(row['search_term'].split()) > 0)):
        s1 = row['search_term']
        s2 = row['material']
        sameWords = set.intersection(set(s1.split(" ")), set(s2.split(" ")))
        return len(sameWords)
    else:
        return 0


print '####### Start create feature #########'

num = pd.DataFrame()
# On teste si la marque apparait dans la recherche ( 1 ou 0)
num['brand_in_search'] = result_review.apply(
    brand_in_search_term, axis=1).astype(np.int64)


# On prend les mots en commun et les ratios
num['commun_brand_ratio'] = result_review.apply(
    search_term_brand, axis=1)
num['commun_brand'] = result_review.apply(
    ratio_search_term_brand, axis=1)

num['commun_desctiption_ratio'] = result_review.apply(
    ratio_search_term_description, axis=1)
num['commun_desctiption'] = result_review.apply(
    search_term_description, axis=1)

num['commun_product_title_ratio'] = result_review.apply(
    ratio_search_term_product_title, axis=1)
num['commun_product_title'] = result_review.apply(
    search_term_product_title, axis=1)




# On fait la taille de tous les champs
num['size_search'] = result_review[
    'search_term'].apply(lambda x: len(x.split()))
num['len_search'] = result_review[
    'search_term'].apply(lambda x: len(x))
num['size_product_description'] = result_review[
    'product_description'].apply(lambda x: len(x.split()))
num['size_product_title'] = result_review[
    'product_title'].apply(lambda x: len(x.split()))
num['size_brand'] = result_review[
    'brand'].apply(lambda x: len(x.split()))

# # Exploration fine des attribus
# num['commun_in_out'] = result_review.apply(
#     search_term_in_out, axis=1)
# num['commun_color'] = result_review.apply(
#     search_term_color, axis=1)
# num['commun_material'] = result_review.apply(
#     search_term_material, axis=1)

# Apparition des champs
num['count_brand'] = result_review[
    'brand'].apply(lambda x: counter_brand[x])
num['count_search_term'] = result_review[
    'search_term'].apply(lambda x: counter_search_term[x])
num['count_product_description'] = result_review[
    'product_description'].apply(lambda x: counter_product_description[x])
# num['count_product_title'] = result_review[
#     'product_title'].apply(lambda x: counter_brand[x])


num = num.fillna('0')
print 'End create feature'

fname_new_feature = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot/data/new_feature.csv'
num.to_csv(fname_new_feature, index=False)
