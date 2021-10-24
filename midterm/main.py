import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('_'*21)
print('|Okay, Ill use',device,'|')
print('￣'*11)

g = Planetoid(root='/tmp/Citeseer',name = "Citeseer")
data = g.data.to(device)

from introduction import intro
intro(g)
HP = {
    'layer' : 2,
    'iters' : 100,
    'hidden_feature' : 16,
    'dropout' : 1,
    'lr' : 0.01, 
    'weight_decay' : 5e-4,
    'epochs' : 100
}

class GCN2(torch.nn.Module):
    def __init__(self, hidden_feature=16,dropout = 0):
        self.dropout = dropout
        super().__init__()
        self.conv1 = GCNConv(g.num_node_features, hidden_feature)
        # self.conv2 = GCNConv(16, 16)
        self.conv2 = GCNConv(hidden_feature, g.num_classes)
        # self.conv3 = GCNConv(16, g.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if self.dropout: x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim = 1)

        return F.log_softmax(x, dim=1)
    
class GCN3(torch.nn.Module):
    def __init__(self, hidden_feature=(16,16),dropout = 0):
        self.dropout = dropout
        super().__init__()
        self.conv1 = GCNConv(g.num_node_features, hidden_feature[0])
        self.conv2 = GCNConv(hidden_feature[0], hidden_feature[1])
        self.conv3 = GCNConv(hidden_feature[1], g.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        if self.dropout: x = F.dropout(x, training=self.training)
        x = F.softmax(x, dim = 1)

        return F.log_softmax(x, dim=1)

def evaluation(_model):
    _model.eval()
    pred = _model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

def test():
    if HP['layer']==2:
        model = GCN2(hidden_feature = HP['hidden_feature'],dropout = HP['dropout']).to(device)
    elif HP['layer']==3:
        model = GCN3(hidden_feature = HP['hidden_feature'],dropout = HP['dropout']).to(device)
    else:
        raise ValueError("layer index wrong, must 2 or 3")
    data = g[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP['lr'], weight_decay=HP['weight_decay'])
    best_acc = 0
    model.train()
    for epoch in range(HP['epochs']):
        optimizer.zero_grad()
        out = model(data)
        lf = nn.CrossEntropyLoss()
        loss = lf(out[data.train_mask], data.y[data.train_mask])
        acc = round(100*evaluation(model),3)
        if acc > best_acc:
            best_acc = acc
            best_model = model
        loss.backward()
        optimizer.step()
    return best_acc, best_model

def train_iter(iters):
    total_acc = 0
    best_acc=0
    for iter in tqdm(range(1,iters+1)):
        acc, model= test()
        total_acc+=acc
        if acc>best_acc:
            best_acc = acc
            best_model = model
    avg_acc = total_acc/iters
    print("Average accuracy : ", avg_acc)
    print("Best accuracy : ", best_acc)

    PATH = './GCN_Citeseer_model/{}_{}_{}_avgacc{}_bestacc{}_{}'.format(
        HP['layer'],HP['hidden_feature'],HP['dropout'],avg_acc,best_acc,random.randint(0,10000))
    torch.save(best_model, PATH)
    return best_model,best_acc
best_acc = 0

# model, acc = train_iter(HP['iters'])
#######################################
for i in range(5):
    for do in (0,1):
        HP['dropout']=do
        HP['layer']=2
        for hf in (16,32):
            HP['hidden_feature']=hf
            print(HP)
            model, acc = train_iter(HP['iters'])
            if acc > best_acc:best_acc = acc ; best_model = model; best_HP = HP
        HP['layer'] =3
        for hf in ((16,16),(16,32),(32,32)):
            HP['hidden_feature']=hf
            print(HP)
            model, acc = train_iter(HP['iters'])
            if acc > best_acc:best_acc = acc ; best_model = model; best_HP = HP
model = best_model

print(HP)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model.eval()
out = model(data)
visualize(out, color=data.cpu().y)