import numpy as np
import pandas as pd
from nltk.stem.porter import *
from nltk.corpus import stopwords
path = '/Users/williambenhaim/Desktop/TelecomParisTech/P3/MachineLearningAvances/Challenge/HomeDepot'
# dans le terminal : iconv -f utf-8 -t utf-8 -c attributes.csv > attributes_clean.csv
attributes_fname = path + '/input/attributes.csv'
allcolor_fname = path + '/input/allColor.csv'
allcolor = pd.read_csv(allcolor_fname, sep=',')
df_attributes = pd.read_csv(attributes_fname, sep=',')#, encoding="ISO-8859-1")

int_ext = df_attributes[df_attributes['name'] == 'Interior/Exterior']
int_ext = int_ext.rename(columns={'value': 'int_ext'})
int_ext = int_ext.drop(['name'], axis=1)
out_in = df_attributes[df_attributes['name'] == 'Indoor/Outdoor']
out_in = out_in.rename(columns={'value': 'Indoor_Outdoor'})
out_in = out_in.drop(['name'], axis=1)
df_out_in_tmp1 = pd.merge(int_ext, out_in, how='outer', on='product_uid')

df_out_in_tmp1 = df_out_in_tmp1.fillna(' ')
df_out_in_tmp = df_out_in_tmp1['Indoor_Outdoor'] + ' ' + df_out_in_tmp1['int_ext']
df_out_in_tmp = pd.DataFrame(df_out_in_tmp)

df_CF = df_attributes[df_attributes['name'] == 'Color Family']
df_CF = df_CF.rename(columns={'value': 'Color_Family'})
df_CF = df_CF.drop(['name'], axis=1)
df_C = df_attributes[df_attributes['name'] == 'Color']
df_C = df_C.rename(columns={'value': 'Color'})
df_C = df_C.drop(['name'], axis=1)
df_C_F = df_attributes[df_attributes['name'] == 'Color/Finish']
df_C_F = df_C_F.rename(columns={'value': 'Color/Finish'})
df_C_F = df_C_F.drop(['name'], axis=1)

df_color_tmp1 = pd.merge(df_CF, df_C, how='outer', on='product_uid')
df_color_tmp1 = pd.merge(df_color_tmp1, df_C_F, how='outer', on='product_uid')

stemmer = PorterStemmer()
df_color_tmp1 = df_color_tmp1.fillna(' ')
df_color_tmp = df_color_tmp1['Color/Finish'] + ' ' + df_color_tmp1['Color_Family'] + ' ' + df_color_tmp1['Color']
df_color_tmp = pd.DataFrame(df_color_tmp)

df_material = df_attributes[df_attributes['name']=='Material']
df_material = df_material.drop(['name'], axis=1)
df_material = df_material.rename(columns={'value': 'material'})
df_material = df_material.fillna(' ')

def clean_attr(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"(\w)\.([A-Z])", r"\1 \2", sentence)
    # Saut de ligne mal geres
    sentence = re.sub(r"(\w)([A-Z])", r"\1 \2", sentence)
    sentence = re.sub(r"([0-9])([a-z])", r"\1 \2", sentence)
    sentence = re.sub(r"([a-z])([0-9])", r"\1 \2", sentence)
    sentence = sentence.decode("utf-8").lower()
    # sentence = sentence.replace("/", " ")
    # sentence = sentence.replace(",", " ")
    # sentence = sentence.replace("-", " ")
    # sentence = sentence.replace(",", " ")
    # sentence = sentence.replace("?", " ")
    # sentence = sentence.replace(":", " ")
    # sentence = sentence.replace("$", " ")
    # sentence = sentence.replace("#", " ")
    # sentence = sentence.replace("(", " ")
    # sentence = sentence.replace(")", " ")
    # sentence = sentence.replace("&", " ")

    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = sentence.replace("color", " ")
    sentence = sentence.replace("colored", " ")
    sentence = sentence.replace("exterior", "outdoor")
    sentence = sentence.replace("interior", "indoor")
    sentence = sentence.replace("multi", " ")
    words = str(sentence).decode("utf-8").lower().split()
    stops = set(stopwords.words("english"))
    words = [stemmer.stem(w) for w in words if not (w or stemmer.stem(w)) in stops]
    rd = " ".join(sorted(set(words), key=words.index))
    return rd


def multi_color(rd):
    if len(rd.split()) > 3:
        res = ''
        for color in rd.split():
            if color in allcolor:
                res = res + ' '+color
        return res
    else:
        return rd

print '##### START'
df_color_tmp = df_color_tmp.applymap(lambda x:  clean_attr(x))# .encode("utf-8")))
df_color_tmp = df_color_tmp.applymap(lambda x:  multi_color(x.encode("utf-8")))
df_out_in_tmp = df_out_in_tmp.applymap(lambda x:  clean_attr(x.encode("utf-8")))
df_material_tmp = df_material['material'].apply(lambda x: clean_attr(x)) # .encode("utf-8")))

# df_brand = df_brand.applymap(lambda x:  clean_attr(x.encode("utf-8")))
print'##### END'

df_res_color = pd.concat([df_color_tmp1['product_uid'], df_color_tmp], axis=1)
df_res_color.columns = ['product_uid', 'color']

df_res_out_in = pd.concat([df_out_in_tmp1['product_uid'], df_out_in_tmp], axis=1)
df_res_out_in.columns = ['product_uid', 'in_out']

df_res_material = pd.concat([df_material['product_uid'], df_material_tmp], axis=1)
df_res_material.columns = ['product_uid', 'material']

res_attribus = pd.merge(df_res_color, df_res_material, how='left', on='product_uid')
res_attribus = pd.merge(res_attribus, df_res_out_in, how='left', on='product_uid')
res_attribus = res_attribus.fillna(' ')
res_attribus = res_attribus.applymap(lambda x: str(x))
fname_explore_attr = path + '/data/explore_attr.csv'
res_attribus.to_csv(fname_explore_attr, index=False)
print '##### END Script'
