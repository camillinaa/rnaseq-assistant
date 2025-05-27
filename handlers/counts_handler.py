import pandas as pd
import numpy as np
import glob
import os
import spacy
from fuzzywuzzy import fuzz
import pandasai as pai
from pandasai import SmartDataframe
from pandasai.llm import HuggingFaceTextGen

pai.api_key.set("PAI-95096dde-70c2-46e9-9aa0-2ff24003a3cc")

'''
assumes that the intent_classifier has already identified that the intent is enrichr
and the user has provided comparison_1 and comparison_2
'''

query = "what is the expression of CATG00000000004.1 for _pma24_ compared to _pma96_?"

df = pd.read_csv("data/count_matrix.tsv", sep="\t", index_col=0)
df.columns = [col.strip().lower().replace(".", "_") for col in df.columns]
df1, df2, df3, df4 = np.array_split(df, 4, axis=1)

df1 = pai.DataFrame(df1)
df1.chat(query)
