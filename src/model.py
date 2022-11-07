import torch.nn as nn
import torch.nn.functional as F

# 서버 클라이언트 분리
'''
전체 모델은 A,B로 분리되어 있음
클라이언트는 A를 가짐, 데이터 학습 후 서버로 전송
서버는 A,B를 가짐, 클라이언트로부터 데이터 받으면 B를 학습, 역전파 시에는 A,B모두 참여
서버는 업데이트된 A를 클라이언트로 배포
'''

# 서버 모델
'''
forward함수는 클라이언트로부터 받은 데이터를 중간(B)부터 학습
forward함수에서만 순전파를 틀어주고, 역전파는 동일하게 로스함수로 적용
'''
class Server_Net(nn.Module):
    def __init__(self): # layer 정의
        super(Server_Net, self).__init__()

        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        
    def forward(self, x):
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def test(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# 클라이언트 모델
'''
클라이언트 함수는 서버로 학습 결과를 전송하는 역할
'''
class Client_Net(nn.Module):
    def __init__(self): # layer 정의
        super(Client_Net, self).__init__()

        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x
