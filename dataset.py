import torch
import pandas as pd
from torch.utils.data import Dataset
from random import randint

class JobDataset(Dataset):
	def __init__(self, df):
		self.data = df

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		r = randint(1,10)

		sample = {
			"description": self.data[2][idx]
		}
		if r > 5:
			sample["title"] = self.data[1][idx]
			sample["related"] = 1
		else:
			true_title = self.data[1][idx]
			false_label = self.data[self.data[1] != true_title][1].sample(1)
			sample["title"] = list(false_label)[0]
			sample["related"] = -1

		return sample
