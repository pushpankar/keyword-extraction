from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

df = pd.read_csv("./data.tsv", header=None, delimiter="\t")

def remove(sent):
	words = word_tokenize(sent)
	sw = stopwords.words()
	clean_words = list(filter(lambda x: x not in sw, words))
	return " ".join(clean_words)

df[2] = df[2].map(remove)
df.to_csv("./clean_data.tsv", sep="\t", index=False)
