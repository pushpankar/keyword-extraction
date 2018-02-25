import torch
import torch.nn as nn
from attention import Attn


class Net(nn.Module):
	def __init__(self, vocab_size, hidden_size, embed_size):
		super(Net, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.embed_size = embed_size

		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size)
		self.attn = Attn(hidden_size)
		self.fc1 = nn.Linear(hidden_size, hidden_size//8)
		self.fc2 = nn.Linear(hidden_size//8, 1)

	def forward(self, description, title):
		description_embed = self.embedding(description)
		title_embed = self.embedding(description)
		description_hiddens, description_hidden = self.gru(description_embed)

		_, title_hidden = self.gru(title_embed)
		attn_weights = self.attn(description_hidden,
								 description_hiddens)

		context = attn_weights.bmm(description_hiddens.transpose(0, 1))

		return context, title_hidden, attn_weights
