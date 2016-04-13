import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
data_cleaning = path + '/data/clean_data.csv'
result_review = pd.read_csv(data_cleaning, sep=',', na_values=' ', encoding="utf-8")
result_review = result_review.fillna(' ')
result_review = result_review.applymap(lambda x: str(x))
# result_review['product_description'] = result_review['product_description'].apply(lambda x: str(x))
# result_review['product_title'] = result_review['product_title'].apply(lambda x: str(x))
# result_review['search_term'] = result_review['search_term'].apply(lambda x: str(x))
# result_review['product_info'] = result_review['product_info'].apply(lambda x: str(x))
# result_review = result_review.drop(['product_uid'], axis=1)
print '######### Start TFIDF ##########'


def transform_tfidf_svd(df):
    vectorizer = TfidfVectorizer(min_df=0)
    df_tfidf = vectorizer.fit_transform(df)
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_svd = svd.fit_transform(df_tfidf)
    return pd.DataFrame(X_svd)


df_svd = transform_tfidf_svd(result_review['product_description'])
df_svd2 = transform_tfidf_svd(result_review['product_title'])
df_svd3 = transform_tfidf_svd(result_review['search_term'])
df_svd4 = transform_tfidf_svd(result_review['product_info'])
df_svd5 = transform_tfidf_svd(result_review['bullet'])

frame = [df_svd2, df_svd, df_svd3, df_svd4]#,df_svd5]
df_result_tfidf = pd.concat(frame, axis=1)

print 'End TFIDF'

fname = path + '/data/result_tfidf.csv'
df_result_tfidf.to_csv(fname, index=False)
