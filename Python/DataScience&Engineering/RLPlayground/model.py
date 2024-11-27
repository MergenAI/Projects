import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import os


class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Linear_QNet, self).__init__()
        self.linear_layer1=nn.Linear(input_size,hidden_size)
        self.linear_layer2=nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x=f.relu(self.linear_layer1(x))
        x=self.linear_layer2(x)
        return x
    def save(self,file_path="model.pth"):
        folder_path="./model"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path=os.path.join(folder_path,file_path)
        torch.save(self.state_dict(),file_path)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.model=model
        self.lr=lr
        self.gamma=gamma
        self.opt=optim.Adam(model.parameters(),lr=self.lr)
        self.criterion=nn.MSELoss()
    def train_step(self,state,action,reward,next_state,done):
        state=torch.tensor(state,dtype=torch.float)
        next_state=torch.tensor(next_state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)
        if len(state.shape)==1:
            state=torch.unsqueeze(state,0)
            next_state=torch.unsqueeze(next_state,0)
            reward=torch.unsqueeze(reward,0)
            action=torch.unsqueeze(action,0)
            done=(done,)
        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.opt.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.opt.step()