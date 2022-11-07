import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader
import time
import os
import sys
import torchvision.transforms as transforms
import torchvision.datasets
from model import *
import json

#configuration
with open("./config_client.json",'rb') as f:
  conf = json.load(f)

num_data = conf['num_data'] # *32 per client, data chunk
batch_dize=conf['batch_size']
clientID = sys.argv[1]

# round내에 여러 epoch존재
# epoch마다 통신 필요
# client에서 가능한 1epoch데이터 모아서 통신

# ######################################################
# 데이터 로더
# ######################################################
def load_data():
  #data loader
  '''Load data from source'''
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,))])
  trainset = torchvision.datasets.MNIST(root='./dataset',train=True,download=True,transform=transform)
  trainloader = DataLoader(trainset, batch_size=batch_dize, shuffle=True)
  return trainloader

# ######################################################
# 데이터 학습
# ######################################################
def train(trainloader, epochs):
  net.eval()
  for i,data in enumerate(trainloader):
      if i== num_data:
        break
      images, labels = data
      output= net(images)
      output = output.data
      output_arr = output.numpy()
      target_arr = labels.numpy()
      if i==0:
        input_arr = output_arr
        label_arr = target_arr
      else:
        input_arr = np.concatenate((input_arr,output_arr),axis=0)
        label_arr = np.concatenate((label_arr,target_arr),axis=0)
  return input_arr, label_arr


net = Client_Net()
    
    # main
def main():
    # Flower client
    class FLClient(fl.client.NumPyClient):
        def get_parameters(self):
            # 최초 client weight init에 필요
            print('@get_parameters :',time.time())
            return [val.cpu().numpy() for _, val in net.state_dict().items()]
            
        def set_parameters(self, parameters):
            print('@set_parameters :',time.time())
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict,strict=True)

        def fit(self, parameters, config):
            print('@round:',config['rnd'])
            self.set_parameters(parameters)
            print('@fit :',time.time())
            # 데이터 불러오기
            train_loader = load_data()
            input,label = train(train_loader, config['epoch'])
            
            return [input,label], 0, {}
            '''
            np.save('localData/input_arr_'+clientID+'.npy',input)
            np.save('localData/label_arr_'+clientID+'.npy',label)
            return [[0]], 0, {} #학습 후 parameter 넘기지 않음
            '''
            #return self.get_parameters(), num_examples, {}

        def evaluate(self, parameters, config):
            pass

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=FLClient())


if __name__ == "__main__":
    main()