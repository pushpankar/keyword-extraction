import pandas as pd
import csv
from nltk.tokenize import sent_tokenize

df = pd.read_csv("./TrainData (1) (1) (1).csv", header=None).dropna()

wf = open("./data.tsv", "w")
csv_writer = csv.writer(wf, delimiter="\t")

df[2] = df[2].map(lambda x: sent_tokenize(x))
import pdb; pdb.set_trace()
for _, row in df.iterrows():
	for sent in row[2]:
		csv_writer.writerow([row[0], row[1], sent, row[3]])

wf.close()
