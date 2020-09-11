import torch.nn as nn
import torch.nn.functional as F
import helper
import torch
from torch.autograd import Variable
import numpy
import sys
import copy
from torch.utils.data import Dataset,DataLoader,random_split
import matplotlib.pyplot as plt
import cfg

Cfg = cfg.Config()

class human3DptsDataset(Dataset):
	def __init__(self):
		
		train,label = helper.extractDataLabel(Cfg.dataPath)	
		self.train = torch.from_numpy(numpy.array(train)).type('torch.DoubleTensor')
		self.label = torch.from_numpy(numpy.array(label))
		
	def __len__(self):
		return len(self.train)

	def __getitem__(self,idx):
		return self.train[idx], self.label[idx]


class Net(nn.Module):
	def __init__(self,inp,out,neuronNum):
		super(Net,self).__init__()
		self.inp = inp
		self.out = out
		self.neuronNum = neuronNum
		self.fc1=nn.Linear(1*self.inp,self.neuronNum)
		self.fc2=nn.Linear(self.neuronNum,self.neuronNum)
		self.fc3=nn.Linear(self.neuronNum,self.out)
	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x)


def valid(net,weight_path,valid_dataset,criterion):
	correct = 0
	total = 0
	valid_loss = 0
	with torch.no_grad():
		for valid in valid_dataset:
			inputs,label = valid
			inputs = inputs.cuda()
			label = label.cuda()
			net_out = net(inputs.float())
			loss = criterion(net_out.unsqueeze(0), label)
			
			_,predicted = torch.max(net_out.data,0)
			total += label.size(0)
			correct+= (predicted==label).sum().item()
			valid_loss+=loss
	return valid_loss/total


def test(net,weight_path,test_dataset,criterion):
	correct = 0
	total = 0
	test_loss = 0
	with torch.no_grad():
		for test in test_dataset:
			
			inputs,label = test
			inputs = inputs.cuda()
			label = label.cuda()
			net_out = net(inputs.float())
			_,predicted = torch.max(net_out.data,0)
			loss = criterion(net_out.unsqueeze(0), label)
			total += label.size(0)
			correct+= (predicted==label).sum().item()
	    		test_loss += loss
			accuracy = 100*correct/total
	return (test_loss/total,accuracy)

def train():
	
	weight_path = "../weights/wts" + str(Cfg.neuronNum) + "_"
	model_path = "../net/model" + str(Cfg.neuronNum) + "_"	
	inp = 56
	out = 13
	bsize = Cfg.batchSize
	neuronNum = Cfg.neuronNum
	max_acc = 0

	all_test_loss = []
	all_valid_loss = []
	all_train_loss = []
	all_epoch = []
	acc = []
	dataset = human3DptsDataset()
	# seperate dataset
	train_size = int((1-Cfg.testPortion)*len(dataset))
	test_size = len(dataset)-train_size
	train_dataset, test_dataset = random_split(dataset,[train_size,test_size])
	
	train_size = int(Cfg.trainPortion*len(train_dataset))
	valid_size = len(train_dataset)-train_size
	train_dataset, valid_dataset = random_split(train_dataset,[train_size,valid_size])
	
	train_loader = 	DataLoader(train_dataset,
				batch_size = bsize,
				shuffle = True)
	
	valid_loader = DataLoader(valid_dataset,
				batch_size = bsize)

	test_loader = DataLoader(test_dataset,
				batch_size = bsize)

    # construct model
	net = Net(inp,out,neuronNum)
	net.cuda()
	
	learning_rate = 0.0001
	optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
	criterion = nn.NLLLoss()

	epochs = Cfg.epochs
	log_interval = Cfg.logInterval

	# save the model params
	best_model_wts = copy.deepcopy(net.state_dict())
	
	
	# run the main training loop
	for epoch in range(1,epochs+1):
	    for batch_idx, data in enumerate(train_loader):
		weight_path = '../weights/wts' + str(Cfg.neuronNum) + "_"
		model_path = "../net/model" + str(Cfg.neuronNum) + "_"
		inputs,target = data
	        inputs, target = Variable(inputs.cuda()), Variable(target.cuda())
        	optimizer.zero_grad()
        	net_out = net(inputs.float())
		#print(net_out.shape)
		#sys.exit()	
        	loss = criterion(net_out, target.squeeze())
        	loss.backward()
        	optimizer.step()
	#if batch_idx % log_interval == 0 :
	    #print('Train Epoch:{}, Loss={:.6f}'.format(epoch,loss.data.item()))
            if epoch % log_interval ==0:
		net.eval()	
		weight_path  += str(epoch)+'.pth'
		model_path   += str(epoch)+'.pth'
		torch.save(model_path,weight_path)
		
		test_loss,accuracy  = test(  net, weight_path,  test_dataset, criterion)
		valid_loss = valid( net, weight_path, valid_dataset, criterion)
		print('Train epoch = {}, valid loss = {}'.format(epoch, valid_loss))
		if accuracy>max_acc:
		  max_acc = accuracy
		all_test_loss.append(test_loss)
		all_valid_loss.append(valid_loss)
		all_train_loss.append(loss.data.item())
		
		#all_loss.append([loss.data.item(), valid_loss, test_loss])
		all_epoch.append(epoch)
		acc.append(accuracy)
		
		#net.train()		
	torch.save(net,model_path)

	
	plt.plot(all_epoch ,all_test_loss,'b')
	plt.title('testing loss',fontsize=16)
	plt.savefig('test_{}_{}.jpg'.format(neuronNum,epochs))
	plt.xlabel('epochs',fontsize=16)
	plt.ylabel('loss',fontsize=16)
	plt.show()
	
	plt.close()
	plt.plot(all_epoch ,all_valid_loss,'r')
	plt.title('validation loss',fontsize=16)
	plt.xlabel('epochs',fontsize=16)
	plt.ylabel('loss',fontsize = 16)
	plt.savefig('valid_{}_{}.jpg'.format(neuronNum,epochs))
	plt.show()
	plt.close()
	plt.plot(all_epoch ,all_train_loss,'k')
	plt.title('training loss',fontsize=16)
	plt.xlabel('epochs',fontsize=16)
	plt.ylabel('loss',fontsize=16)
	plt.savefig('train_{}_{}.jpg'.format(neuronNum,epochs))
	plt.show()
	plt.close()
	
	plt.plot(all_epoch ,all_test_loss,'b' ,label = 'testing loss')
	plt.plot(all_epoch ,all_valid_loss,'r',label = 'validation loss')
	plt.plot(all_epoch ,all_train_loss,'k--',label = 'training loss')
	plt.xlabel('epochs',fontsize=16)
	plt.ylabel('loss',fontsize=16)
	plt.legend(loc='upper right')
	plt.savefig('all_loss_{}_{}.jpg'.format(neuronNum,epochs))
	plt.show()
	plt.close()
	
	
	plt.plot(all_epoch,acc)
	plt.title('Accuracy',fontsize=16)
	plt.xlabel('epochs',fontsize=16)
	plt.ylabel('%',fontsize=16)
	plt.savefig('acc_{}_{}.jpg'.format(neuronNum,epochs))
	plt.show()
	plt.close()
	
	print(max_acc)
	
	# 4000 128 88
	# 10000 128 89
	# 20000 128 89




if __name__ == "__main__":
		
	train()
