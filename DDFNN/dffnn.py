import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import interpol
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class Net(nn.Module):
    def __init__(self, B, size, n_layers):
        super().__init__()
        self.B = B
        n_input = 2*B.shape[1]
        n_output = 2

        self.proc = lambda x: torch.concat([torch.cos(x@self.B),torch.sin(x@self.B)], dim=1)
        
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(size, size))
            layers.append(nn.Tanh())
        
        self.fc = nn.Sequential(
            nn.Linear(n_input,size),
            nn.Tanh(),
            *layers,
            nn.Linear(size,n_output)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.00)

        self.fc.apply(init_weights)
    
    def save(self, dir):
        params = [self.B.cpu().numpy()]
        for name, param in self.named_parameters():
            params.append(param.data.cpu().numpy())
        np.savez_compressed(dir+".ddf", *params)

    def forward(self, x):
        x = self.proc(x)
        x = self.fc(x)
        return x
    
    def efforward(self, x, bs=1000):
        batches = (x.shape[0]//bs)+1
        y = []
        for b in range(batches):
            y.append(self(x[b*bs:(1+b)*bs]).detach())
        return torch.concat(y)

    def trainloop(self, optimizer, X, Y1, Y1_img, Y2, Y2_img, batch_size, its, tr, 
                  fix_mask = None,
                  loss_fun=nn.MSELoss(), 
                  lamb = [1e-3, 1e-3]):
        self.train()
        hist = {"loss":[], "loss_val":[], "regu":[]}
        mask = (Y1_img>=tr)
        if batch_size==0:
            batch_size = mask.sum()
        
        Xm = X[mask.flatten()]
        Ym = Y1_img.flatten()[mask.flatten()]
        
        for epoch in tqdm(range(its), leave=False):
            permutation = torch.randperm(mask.sum())
            n_iter = mask.sum()//batch_size
            for b in range(n_iter):
                batch = permutation[b*batch_size:(b+1)*batch_size]
                batch_random = torch.randint(0, mask.sum(), (batch_size,))

                Xb = Xm[batch]
                Yb = Ym[batch,None]

                dis = self(Xb)

                newx = Xb+dis
                Y_p = interpol.grid_pull(Y2.T, newx, interpolation=1, extrapolate=True).T
                loss = loss_fun(Yb, Y_p)


                if torch.isnan(torch.tensor(loss.item())):
                    print("Boomb!!", loss.item(), epoch)
                    break
                hist["loss"].append(loss.item())
                self.last_loss = loss.item()

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return hist
    
def loader(dir):
    data = np.load(dir, allow_pickle=False)
    B = torch.Tensor(data[data.files[0]]).to(device)
    params = [data[k] for k in data.files[1:]]
    model = Net(B, params[0].shape[0], params.__len__()//2 - 2).to(device)
    for i, (name, param) in enumerate(model.named_parameters()):
        param.data = torch.Tensor(params[i]).to(device)
    return model

def froze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
    with torch.no_grad():
        torch.cuda.empty_cache()
        
def fastVel(
        img1, img2,
        net=None, optimizer=None,
        random_seed = 784, nB = 200, 
        scaler = 100, nlayers = 1, sizelayer = 100 ,
        lamb = [0, 0], its = 50,
        batch_size = 100000, lr = 1e-3,
        tr=0.0,
        verbose=False,
        ):
    

    x = np.arange(img1.shape[0])
    y = np.arange(img1.shape[1])
    X, Y = np.meshgrid(y, x[::-1])
    if batch_size > np.prod(img1.shape):
        batch_size = np.prod(img1.shape)

    Y2_img = torch.tensor(img2[::-1].copy()).float().to(device)
    Y1_img = torch.tensor(img1[::-1].copy()).float().to(device)

    Y2 = torch.tensor(img2).float().to(device)
    Y1 = torch.tensor(img1).float().to(device)

    
    mesh = (X, Y)
    X0 = torch.movedim(torch.tensor(np.array(mesh)), 0, 2).float().to(device)
    X0 = X0.reshape([-1,2])
    X0.requires_grad = lamb[1]>0

    torch.manual_seed(random_seed)
    rng_agent2 = np.random.default_rng(random_seed)
    B = (1/scaler)*rng_agent2.normal(size=(2,nB))
    B = torch.tensor(B).float().to(device)
    if net is None:
        net = Net(B, sizelayer, nlayers).to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    hist = net.trainloop(optimizer, X0, 
                            Y1, Y1_img, Y2, Y2_img, 
                        batch_size, its, tr, 
                        fix_mask=None,
                        lamb = lamb)
    if verbose:
            base = np.mean((img1-img2)**2)
            
            plt.semilogy()
            plt.plot(hist["loss"], label="loss")
            plt.plot([0,len(hist["loss"])], [base]*2, label="baseline")
            plt.legend()
            plt.grid()
            plt.show()
    dis = net.efforward(X0)
    vel = dis.cpu().numpy()
    return net, optimizer, vel