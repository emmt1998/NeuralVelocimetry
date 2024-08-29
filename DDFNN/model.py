
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import interpol
from tqdm import tqdm
import matplotlib.pyplot as plt

rng_agent = np.random.default_rng(6516)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class Net(nn.Module):
    def __init__(self, n_input, size, n_layers ,n_output, B, fourier=True):
        super().__init__()
        self.B = B
        self.fourier = fourier
        n_input = 2*B.shape[1]*self.fourier + 2*(not self.fourier)

        self.proc = lambda x: torch.concat([torch.cos(x@self.B),torch.sin(x@self.B)], dim=1)
        if not self.fourier:
            self.proc = lambda x: x
        
        layers = n_layers*[
            nn.Linear(size, size),
            nn.Tanh(),
            # nn.Tanh()
            ]
        
        self.fc = nn.Sequential(
            nn.Linear(n_input,size),
            nn.Tanh(),
            *layers,
            nn.Linear(size,2)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.00)

        self.fc.apply(init_weights)

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
    
    def splitforward(self, x, y):
        x = torch.concat([x,y], dim=1)
        return self(x)
    
    def D(self, f, coord):
        f_coord =  torch.autograd.grad(f, coord,
                            grad_outputs=torch.ones_like(f), 
                            allow_unused=True,
                            create_graph=True)[0]
        return f_coord
    
    def I(self, f, coord):
        return torch.ones_like(f)


    def continuity(self, x, calculate):
        modes = {"D":self.D, "I":self.I}
        mode = calculate*"D" + (not calculate)*"I"
        
        y = x[:,1][:,None]
        x = x[:,0][:,None]
        u = self.splitforward(x, y)
        ux = u[:,0][:,None]
        uy = u[:,1][:,None]
        ux_x =  modes[mode](ux, x)
        uy_y =  modes[mode](uy, y)
        return ux_x+uy_y
    
    def vorticity(self, x):
        y = x[:,1][:,None]
        x = x[:,0][:,None]
        u = self.splitforward(x, y)
        ux = u[:,0][:,None]
        uy = u[:,1][:,None]
        uy_x =  self.D(uy, x).cpu().detach().numpy()
        ux_y =  self.D(ux, y).cpu().detach().numpy()
        return -ux_y+uy_x

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
        
        for epoch in tqdm(range(its)):
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
                
                # newx = Xb-dis
                # Ym = Y2_img.flatten()[mask.flatten()]
                # Y_p = interpol.grid_pull(Y1.T, newx, interpolation=1, extrapolate=True).T
                # loss += loss_fun(Ym[batch,None], Y_p)*1e-6
                if fix_mask is not None:
                    Xf = X[fix_mask]
                    Yf = Y1_img.flatten()[fix_mask]

                    dis = self(Xf)
                    newx = Xf+dis
                    Y_p = interpol.grid_pull(Y2.T, newx, interpolation=1, extrapolate=True).T
                    loss += loss_fun(Yf[:,None], Y_p)

                # loss += lamb[0]*loss_fun(dis, 0*dis)
                # cont = self.continuity(X[batch], calculate=lamb[1]>1e-6)
                # loss += lamb[1]*loss_fun(cont, 0*cont)


                if torch.isnan(torch.tensor(loss.item())):
                    print("Boomb!!", loss.item(), epoch)
                    break
                hist["loss"].append(loss.item())
                self.last_loss = loss.item()
                # hist["regu"].append(loss_fun(dis, 0*dis).item())

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return hist

def fastVel(
        img1, img2,
        net=None, optimizer=None,
        random_seed = 784, fourier = True, nB = 200, 
        scaler = 100, nlayers = 1,sizelayer = 100 ,
        lamb = [0, 0], its = 10,
        batch_size = 10000,lr = 1e-3,
        tr=0.0,
        verbose=False,
        step=10
        ):
    

    x = np.arange(img1.shape[0])
    y = np.arange(img1.shape[1])
    X, Y = np.meshgrid(y, x[::-1])
   

    # Y1 = torch.tensor(img1).float().to(device)
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
        net = Net(2, sizelayer, nlayers, 2, B, fourier=fourier).to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if step==0:
        hist = net.trainloop(optimizer, X0, 
                             Y1, Y1_img, Y2, Y2_img, 
                            batch_size, its, tr, 
                            fix_mask=None,
                            lamb = lamb)
    if step>0: 
        mask = np.zeros_like(X)
        mask[1::step,1::step] = 1
        mask = np.argwhere(mask.flatten())[:,0]
        mask = torch.tensor(mask)
        hist = net.trainloop(optimizer, X0, 
                             Y1, Y1_img, Y2, Y2_img, 
                            batch_size, its, tr, 
                            fix_mask=mask,
                            lamb = lamb)
    if verbose:
            try:
                print(mask.shape[0])
            except Exception: pass
            base = np.mean((img1-img2)**2)
            
            plt.semilogy()
            plt.plot(hist["loss"], label="loss")
            plt.plot([0,len(hist["loss"])], [base]*2, label="baseline")
            plt.legend()
            plt.grid()
            plt.show()
    dis = net.efforward(X0)
    newx = X0+dis

    vel = dis.cpu().numpy()
    return net, optimizer, vel