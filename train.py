import torch
import numpy as np
from model import *
from tqdm import tqdm
cuda = torch.cuda.is_available()
import os
import sys, traceback
from tensorboardX import SummaryWriter

partition_size = 10000
batch_size = 32
epoch = 100
log_step = 100
input_dimensions = [6, 74, 124, 9623, 2, 9, 2, 2610, 39832,
					9, 7, 2, 222, 130524, 52, 296, 280, 4, 3,
					69, 101, 14, 9, 776, 8, 3, 2, 3,
					3, 351, 28, 3, 14, 14, 3, 4642, 233244,
					50, 11, 3889, 4, 8797, 5, 636118, 2, 5277,
					58, 870, 2688, 1809, 11, 93, 52838, 579, 3,
					40, 216, 325, 40, 32, 9,
					40, 151, 6, 2, 6, 6, 3, 3, 11,
					3, 860, 58958, 2, 3, 3, 2, 2, 3, 3, 16]
model_file_name = "model.pt"
output_csv_filename = "output.csv"
writer = SummaryWriter("./logs")
# test_data = np.load("test_data.npy")



def get_batch(idxes,predict=False):
	X = []
	if not predict:
		for i in range(len(input_dimensions)):
			tx = torch.LongTensor(train_data[idxes,i])
			if cuda:
				tx = tx.cuda()
			X.append(tx)
		Y = torch.FloatTensor(train_labels[idxes])
		if cuda:
			Y = Y.cuda()
		return X,Y
	else:
		for i in range(len(input_dimensions)):
			tx = torch.LongTensor(test_data[idxes,i])
			if cuda:
				tx = tx.cuda()
			X.append(tx)
		if cuda:
			X = X.cuda()
		return X,None


def train(model,optimizer,criterion,r_idx):
	model.train()
	train_idx = r_idx[:len(train_data) - partition_size]
	total_loss = 0
	l_step = 0
	for i in tqdm(range(0,len(train_data) - partition_size,batch_size)):
		optimizer.zero_grad()
		bsz = min(len(train_data) - partition_size - i,batch_size)
		x,y = get_batch(train_idx[i:i+bsz])
		output = model(x)
		loss = criterion(output,y)
		loss.backward()
		optimizer.step()
		total_loss += loss.data.item()
		if l_step % log_step == 0 and l_step != 0:
			writer.add_scalar('train_loss',total_loss/log_step)
			# print(total_loss.item()/log_step)
			total_loss = 0
		l_step += 1

def validate(model,criterion,r_idx):
	model.eval()
	val_idx = r_idx[-partition_size:]
	total_loss = 0
	l_step = 0
	for i in tqdm(range(0,partition_size,batch_size)):
		bsz = min(partition_size - i,batch_size)
		x,y = get_batch(val_idx)
		output = model(x)
		loss = criterion(output,y)
		total_loss += loss.data.item()
		if l_step % log_step == 0 and l_step != 0:
			writer.add_scalar('val_loss',total_loss/log_step)
			total_loss = 0
		l_step += 1
	# total_loss = total_loss/partition_size * batch_size
	# print("Validation loss: " + str(total_loss))
	return total_loss

def prediction():
	test_data = np.load("test_data.npy")
	test_machine_id = np.load("test_machine_id.npy")
	model_state = torch.load(model_file_name,map_location= lambda storage,loc:storage)
	model = MLP()
	model.load_state_dict(model_state)
	model.eval()
	idx = np.arange(len(test_data))
	outlist = []
	l_step = 0
	sig = torch.nn.Sigmoid()
	for i in tqdm(range(0,len(test_data),batch_size)):
		bsz = min(len(test_data) - i,batch_size)
		x,_ = get_batch(idx,predict=True)
		output = sig(model(x))
		output = output.squeeze(1)
		out_list.extend(output.data)
	with open(output_csv_filename,'w') as f:
		for mid,p in zip(test_machine_id,outlist):
			f.write(str(test_machine_id) + "," + str(p.item()) + "\n")





def main():
	pass
if __name__== "__main__":
	print("Starting")
	train_data = np.load("train_data.npy")
	train_labels = np.load("train_labels.npy")
	model = MLP()
	if cuda:
		model = model.cuda()
	print(model)
	optimizer = torch.optim.Adam(model.parameters())
	criterion = torch.nn.BCEWithLogitsLoss()
	try:
		best_loss = validate(model,criterion,np.arange(partition_size))
		for e in range(epoch):
			print("Starting epoch " + str(e))
			r_idx = np.random.permutation(len(train_data))
			train(model,optimizer,criterion,r_idx)
			tloss = validate(model,criterion,r_idx)
			if tloss < best_loss:
				best_loss = tloss
				with open(model_file_name,'wb') as f:
					torch.save(model.state_dict(),f)
	except KeyboardInterrupt:
		print("Stop")
	except Exception:
		traceback.print_exc(file=sys.stdout)

