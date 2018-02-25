"""Source https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attn(nn.Module):
	def __init__(self, hidden_size):
		super(Attn, self).__init__()
		self.hidden_size = hidden_size
		self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
		self.v = nn.Parameter(torch.rand(hidden_size))
		stdv = 1. / math.sqrt(self.v.size(0))
		self.v.data.normal_(mean=0, std=stdv)

	def forward(self, hidden, encoder_outputs):
		'''
		:param hidden: 
			previous hidden state of the decoder, in shape (layers*directions,B,H)
		:param encoder_outputs:
			encoder outputs from Encoder, in shape (T,B,H)
		:return
			attention energies in shape (B,T)
		'''
		max_len = encoder_outputs.size(0)
		this_batch_size = encoder_outputs.size(1)
		H = hidden.repeat(max_len,1,1).transpose(0,1)
		encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
		attn_energies = self.score(H,encoder_outputs) # compute attention score
		return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

	def score(self, hidden, encoder_outputs):
		energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
		energy = energy.transpose(2,1) # [B*H*T]
		v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
		energy = torch.bmm(v,energy) # [B*1*T]
		return energy.squeeze(1) #[B*T]

