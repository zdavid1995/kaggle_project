import torch
import torch.nn as nn
cuda = torch.cuda.is_available()


input_dimensions = [6, 74, 124, 9623, 2, 9, 2, 2610, 39832,
			9, 7, 2, 222, 130524, 52, 296, 280, 4, 3,
			69, 101, 14, 9, 776, 8, 3, 2, 3,
			3, 351, 28, 3, 14, 14, 3, 4642, 233244,
			50, 11, 3889, 4, 8797, 5, 636118, 2, 5277,
			58, 870, 2688, 1809, 11, 93, 52838, 579, 3,
			40, 216, 325, 40, 32, 9,
			40, 151, 6, 2, 6, 6, 3, 3, 11,
			3, 860, 58958, 2, 3, 3, 2, 2, 3, 3, 16]
#ninp = sum(input_dimensions)
#condense_dim = [6, 74, 124, 500, 2, 9, 2, 500, 500,
#		9, 7, 2, 222, 500, 52, 296, 280, 4,
#		3, 69, 101, 14, 9, 500, 8, 3, 2, 3,
#		3, 351, 28, 3, 14, 14, 3, 500, 500,
#		50, 11, 500, 4, 500, 5, 500, 2, 500,
#		58, 500, 500, 500, 11, 93, 500, 500,
#		3, 40, 216, 325, 40, 32, 9, 40, 151,
#		6, 2, 6, 6, 3, 3, 11, 3, 500, 500,
#		2, 3, 3, 2, 2, 3, 3, 16]
#ninp = len(input_dimensions)*condense_dim
cutoff_dim = 100
condense_dim = []
for i in input_dimensions:
	if i > cutoff_dim:
		condense_dim.append(cutoff_dim)
	else:
		condense_dim.append(i)

ninp = sum(condense_dim)
class MLP(nn.Module):
	def __init__(self,nhid=2048,nlayers=5,nout=1):
		super(MLP,self).__init__()
		self.leak = 0.15
		self.ninp = ninp
		self.nlayers = nlayers
		self.nout = nout
		self.nhid = nhid
		self.input_emb_layers = []
		for i,j in zip(input_dimensions,condense_dim):
			temb = torch.nn.Embedding(i,j)
			if cuda:
				temb = temb.cuda()
			self.input_emb_layers.append(temb)
		self.mlp_layers = []
		self.mlp_layers.append(nn.BatchNorm1d(ninp))
		for i in range(nlayers):
			if i == 0:
				self.mlp_layers.append(nn.Linear(ninp,nhid))
				self.mlp_layers.append(nn.LeakyReLU(self.leak))
				self.mlp_layers.append(nn.BatchNorm1d(nhid))
			elif i!= nlayers - 1:
				self.mlp_layers.append(nn.Linear(nhid,nhid))
				self.mlp_layers.append(nn.LeakyReLU(self.leak))
				self.mlp_layers.append(nn.BatchNorm1d(nhid))
			else:
				self.mlp_layers.append(nn.Linear(nhid,nout))
		self.mlp_layers = nn.Sequential(*self.mlp_layers)
		self.mlp_layers.apply(self.init_weights)



	def init_weights(self,l):
		if type(l) == nn.Linear:
			torch.nn.init.xavier_uniform_(l.weight.data)

	def forward(self,input_list):
		full_input = []
		for i in range(len(input_list)):
			full_input.append(self.input_emb_layers[i](input_list[i]))
		full_input = torch.cat(full_input,dim=-1)
		output = self.mlp_layers(full_input)
		return output.squeeze(1)
