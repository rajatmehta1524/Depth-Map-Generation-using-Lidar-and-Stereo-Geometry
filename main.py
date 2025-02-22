import os
import time
import copy
import torch
import random
import os.path
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from stackhourglass import stackhourglass
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Densedepthmap')
parser.add_argument('--maxdisp', type=int ,default=192,
					help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
					help='select model')
parser.add_argument('--datapath', default='./training/',
					help='datapath')
parser.add_argument('--epochs', type=int, default=300,
					help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained_model.tar',
					help='load model')
parser.add_argument('--savemodel', default='./',
					help='save model')

args = parser.parse_args()

#################################################################################
# 								   LOAD DATASET  							    #
#################################################################################
def default_loader(path):
	return Image.open(path).convert('RGB')

def disparity_loader(path):
	return Image.open(path)

def get_transform(name='imagenet', input_size=None,
				  scale_size=None, normalize=None, augment=True):
	normalize = __imagenet_stats
	input_size = 256
	if augment:
			return inception_color_preproccess(input_size, normalize=normalize)
	else:
			return scale_crop(input_size=input_size,
							  scale_size=scale_size, normalize=normalize)

class myImageFloder(torch.utils.data.Dataset):
	def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):
 
		self.left = left
		self.right = right
		self.disp_L = left_disparity
		self.loader = loader
		self.dploader = dploader
		self.training = training

	def __getitem__(self, index):
		left  = self.left[index]
		right = self.right[index]
		disp_L= self.disp_L[index]

		left_img = self.loader(left)
		right_img = self.loader(right)
		dataL = self.dploader(disp_L)


		if self.training:  
			w, h = left_img.size
			th, tw = 256, 512

			x1 = random.randint(0, w - tw)
			y1 = random.randint(0, h - th)

			left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
			right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

			dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256
			dataL = dataL[y1:y1 + th, x1:x1 + tw]

			processed = get_transform(augment=False)  
			left_img   = processed(left_img)
			right_img  = processed(right_img)

			return left_img, right_img, dataL
		else:
			w, h = left_img.size

			left_img = left_img.crop((w-1232, h-368, w, h))
			right_img = right_img.crop((w-1232, h-368, w, h))
			w1, h1 = left_img.size

			dataL = dataL.crop((w-1232, h-368, w, h))
			dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

			processed = get_transform(augment=False)  
			left_img       = processed(left_img)
			right_img      = processed(right_img)

			return left_img, right_img, dataL

	def __len__(self):
		return len(self.left)

def dataloader(filepath):

	left_fold  = 'left/'
	right_fold = 'right/'
	disp_noc   = 'disp_occ/'

	image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

	train = image[:]
	val   = image[160:]

	left_train  = [filepath+left_fold+img for img in train]
	right_train = [filepath+right_fold+img for img in train]
	disp_train = [filepath+disp_noc+img for img in train]


	left_val  = [filepath+left_fold+img for img in val]
	right_val = [filepath+right_fold+img for img in val]
	disp_val = [filepath+disp_noc+img for img in val]

	return left_train, right_train, disp_train, left_val, right_val, disp_val

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
		 myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
		 batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
		 myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
		 batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


#################################################################################
# 	            Define the model and make it cuda compatible  					#
#################################################################################
model = stackhourglass(args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

if args.loadmodel is not None:
	state_dict = torch.load(args.loadmodel)
	model.load_state_dict(state_dict['state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


#################################################################################
# 	            					TRAINING  									#
#################################################################################
def train(imgL,imgR,disp_L):
		model.train()
		imgL   = Variable(torch.FloatTensor(imgL))
		imgR   = Variable(torch.FloatTensor(imgR))   
		disp_L = Variable(torch.FloatTensor(disp_L))

		imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

		#---------
		mask = (disp_true > 0)
		mask.detach_()
		#----

		optimizer.zero_grad()
		
		if args.model == 'stackhourglass':
			output1, output2, output3 = model(imgL,imgR)
			output1 = torch.squeeze(output1,1)
			output2 = torch.squeeze(output2,1)
			output3 = torch.squeeze(output3,1)
			loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
		elif args.model == 'basic':
			output = model(imgL,imgR)
			output = torch.squeeze(output3,1)
			loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

		loss.backward()
		optimizer.step()

		return loss.data[0]

def test(imgL,imgR,disp_true):
		model.eval()
		imgL   = Variable(torch.FloatTensor(imgL))
		imgR   = Variable(torch.FloatTensor(imgR))   
		if args.cuda:
			imgL, imgR = imgL.cuda(), imgR.cuda()

		with torch.no_grad():
			output3 = model(imgL,imgR)

		pred_disp = output3.data.cpu()

		true_disp = copy.deepcopy(disp_true)
		index = np.argwhere(true_disp>0)
		disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
		correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
		torch.cuda.empty_cache()

		return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
	if epoch <= 200:
	   lr = 0.001
	else:
	   lr = 0.0001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def main():
	max_acc=0
	max_epo=0
	start_full_time = time.time()

	for epoch in range(1, args.epochs+1):
		total_train_loss = 0
		total_test_loss = 0
		adjust_learning_rate(optimizer,epoch)
		
		# Let's train the damn model
		for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
			start_time = time.time() 

			loss = train(imgL_crop,imgR_crop, disp_crop_L)
		print(f'Iter {batch_idx} training loss = {loss} , time = {time.time()-start_time}')
		total_train_loss += loss
		print(f'epoch {epoch} total training loss = {total_train_loss/len(TrainImgLoader)}')

		# Model validation and testing
		for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
			test_loss = test(imgL,imgR, disp_L)
			print(f'Iter {batch_idx} 3-px error in val = {test_loss*100}')
			total_test_loss += test_loss


		print(f'epoch {epoch} total 3-px error in val = {total_test_loss/len(TestImgLoader)*100}')
		if total_test_loss/len(TestImgLoader)*100 > max_acc:
			max_acc = total_test_loss/len(TestImgLoader)*100
			max_epo = epoch
		print(f'MAX epoch {max_epo} total test error = {max_acc}')

		# Save the precious model file
		savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
		torch.save({'epoch': epoch,'state_dict': model.state_dict(),
			'train_loss': total_train_loss/len(TrainImgLoader),
			'test_loss': total_test_loss/len(TestImgLoader)*100,
		}, savefilename)

		print(f'full finetune time = {(time.time() - start_full_time)/3600} HR')
	print(f'Maximum epochs: {max_epo}')
	print(f'Maximum accuracy achieved: {max_acc}')


if __name__ == '__main__':
   main()
