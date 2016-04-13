import re
import sys
#import cPickle
import numpy as np
import ngram_utils
from nlp_utils import stopwords#, str_cleaning, str_stem, seg_words, segmentit, str_common_word, str_whole_word
from feat_utils import try_divide #, dump_feat_name
sys.path.append("../")
from param_config import config
#import itertools
import pandas as pd
import time
start_time = time.time()
import itertools

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    
    ## tokenize
    tokens = [x.encode('utf8').lower() for x in token_pattern.findall(line)]
    
    ## stem
    tokens_stemmed = tokens

    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed


def extract_feat(df):

    
    ## unigram
    print "generate unigram"
    df["search_term_unigram"] = list(df.apply(lambda x: preprocess_data(x["search_term"]), axis=1))
    df["product_title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    df["product_description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    df["brand_unigram"] = list(df.apply(lambda x: preprocess_data(x["brand"]), axis=1))

    ## bigram
    print "generate bigram"
    join_str = "_"
    df["search_term_bigram"] = list(df.apply(lambda x: ngram_utils.getBigram(x["search_term_unigram"], join_str), axis=1))
    df["product_title_bigram"] = list(df.apply(lambda x: ngram_utils.getBigram(x["product_title_unigram"], join_str), axis=1))
    df["product_description_bigram"] = list(df.apply(lambda x: ngram_utils.getBigram(x["product_description_unigram"], join_str), axis=1))
    df["brand_bigram"] = list(df.apply(lambda x: ngram_utils.getBigram(x["brand_unigram"], join_str), axis=1))

    ## trigram
    print "generate trigram"
    join_str = "_"
    df["search_term_trigram"] = list(df.apply(lambda x: ngram_utils.getTrigram(x["search_term_unigram"], join_str), axis=1))
    df["product_title_trigram"] = list(df.apply(lambda x: ngram_utils.getTrigram(x["product_title_unigram"], join_str), axis=1))
    df["product_description_trigram"] = list(df.apply(lambda x: ngram_utils.getTrigram(x["product_description_unigram"], join_str), axis=1))
    df["brand_trigram"] = list(df.apply(lambda x: ngram_utils.getTrigram(x["brand_unigram"], join_str), axis=1))

    ################################
    ## word count and digit count ##
    ################################
    print "generate word counting features"
    feat_names = ["search_term", "product_title", "product_description", "brand"]
    grams = ["unigram", "bigram", "trigram"]
    count_digit = lambda x: sum([1. for w in x if w.isdigit()])
    for feat_name in feat_names:
        for gram in grams:
            
            ## word count
            df["count_of_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
            df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
            df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        ## digit count
        df["count_of_digit_in_%s"%feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
        df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])


    ##########################
    ## intersect word count ##
    ##########################
    print "generate intersect word counting features"
    #### unigram
    for gram in grams:
        for obs_name in feat_names:
            for target_name in feat_names:
                if target_name != obs_name:
                    ## query
                    df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
                    df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

        ## some other feat
        df["product_title_%s_in_search_term_div_search_term_%s"%(gram,gram)] = map(try_divide, df["count_of_product_title_%s_in_search_term"%gram], df["count_of_search_term_%s"%gram])
        df["product_title_%s_in_search_term_div_search_term_%s_in_product_title"%(gram,gram)] = map(try_divide, df["count_of_product_title_%s_in_search_term"%gram], df["count_of_search_term_%s_in_product_title"%gram])
        df["product_description_%s_in_search_term_div_search_term_%s"%(gram,gram)] = map(try_divide, df["count_of_product_description_%s_in_search_term"%gram], df["count_of_search_term_%s"%gram])
        df["product_description_%s_in_search_term_div_search_term_%s_in_product_description"%(gram,gram)] = map(try_divide, df["count_of_product_description_%s_in_search_term"%gram], df["count_of_search_term_%s_in_product_description"%gram])


    ##################################
    ## intersect word position feat ##
    ##################################
    print "generate intersect word position features"
    for gram in grams:
        for target_name in feat_names:
            for obs_name in feat_names:
                if target_name != obs_name:
                    pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
                    
                    # stats feat on pos
                    df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
                    df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
                    df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
                    df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
                    df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
                    
                    # stats feat on normalized_pos
                    df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
                    df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])


if __name__ == "__main__":

    ###############
    ## Load Data ##
    ###############
    ## load data
    data_cleaning = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot/data/clean_data.csv'
    df_data_counting = pd.read_csv(
        data_cleaning, sep=',', na_values='(MISSING)', encoding="utf-8").fillna("")

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate counting features...")
    start_time = time.time()

    extract_feat(df_data_counting)
    result_data_counting = df_data_counting._get_numeric_data()
    col_to_remove = []
    for pair in itertools.combinations(result_data_counting.columns,2):
        if all(result_data_counting[pair[0]] == result_data_counting[pair[1]]):
            if pair[1] not in col_to_remove:
                col_to_remove.append(pair[1])

    result_data_counting.drop(col_to_remove, inplace=True, axis=1)

    path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
    fname = path + '/data/data_counting.csv'
    result_data_counting.to_csv(fname, index=False, encoding="utf-8")

    print("--- Feat All: %s minutes ---" % round(((time.time() - start_time)/60),2))


    print("All Done.")