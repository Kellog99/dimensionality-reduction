import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap



###################################################
####
####            AUTOENCODER 
####    
##################################################  

class Autoencoder(nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 hidden_dim = 2, 
                 moments:int = 3, 
                 device = torch.device('cpu')):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.moments = moments
        self.gamma = 0.7
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, hidden_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128), 
            nn.Sigmoid(),
            nn.Linear(128, 256), 
            nn.Sigmoid(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    
    def momentum(self, rec, x):
        out = 0
        for i in range(self.moments):
            out += (self.gamma**i )*torch.sum(torch.abs(torch.mean(x**(i+1), 0)-torch.mean(rec**(i+1), 0)))
        return out
    
    def jacobian(self, x):
        return vmap(jacrev(self.forward))(x)
    

###################################################
####
####            VAE 
####    
################################################## 
class dist_fun(nn.Module):
    def __init__(self,
                 inverse: bool, 
                 hidden_dim:int = 64):
        super(dist_fun, self).__init__()
        self.hidden_dim = hidden_dim
        self.inverse = inverse
        # Encoder layers        
        fc1 = [nn.Linear(1, hidden_dim), 
               nn.Sigmoid(), 
               nn.Linear(hidden_dim, 1)]

        self.fc1 = nn.Sequential(*fc1)
    
    def forward(self, x):
        return self.fc1(x)
        
    def derivative(self, y, x):
        derivative = torch.autograd.grad(y, x, 
                                       grad_outputs = torch.ones_like(x),
                                       create_graph = True, 
                                       retain_graph = True)[0]
        return derivative

# Define VAE model
class VAE(nn.Module):
    def __init__(self,
                 input_feat: int,
                 criterium,
                 device, 
                 hidden_dim: int):
        super(VAE, self).__init__()

        # Encoder layers
        self.criterium = criterium
        self.hidden_dim = hidden_dim
        self.device = device
        self.input_feat = input_feat
        self.upper_bound_var = torch.tensor([5.]*hidden_dim, device = device, requires_grad = True).float()
        self.fc1 = nn.Sequential(nn.Flatten(1),
                                 nn.Linear(input_feat, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256))
        
        self.fc_mu = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, hidden_dim))
        
        self.fc_logvar = nn.Sequential(nn.Linear(256, 128),
                                       nn.Tanh(),
                                       nn.Linear(128, hidden_dim))

        # Decoder layers
        
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, input_feat))

        #### F^{-1}(u) ####
        self.F_inv = dist_fun(inverse = True)

        #### F(F^{-1}(u)) ####
        self.F = dist_fun(inverse = False)
        
    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        log_var = torch.max(torch.min(log_var,torch.ones_like(log_var)*4),torch.ones_like(log_var)*(-4)) 
        var = torch.exp(log_var)
        return mu, var#.view(-1,self.hidden_dim, self.hidden_dim)

    def decode(self, z):
        return self.fc2(z)

        
    def reparameterize(self, mu, var):

        #### Generating the random distribution #####
        u = torch.rand_like(mu, requires_grad = True).float().view(-1,1)
        x = self.F_inv(u)
        u_hat = self.F(x)
        
        ### Perturbing the embedding 
        z = mu + var*x.view(-1,self.hidden_dim)#torch.bmm(var,x.view(-1,self.hidden_dim, 1)).view(-1, self.hidden_dim)
        return z, u, u_hat, x
    
    def forward(self, x):
        mu, var = self.encode(x.view(-1, self.input_feat))
        z, u, u_hat, x, = self.reparameterize(mu, var)
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, u, x, u_hat, var
    
    
    def loss_density(self):
        u = torch.rand(500, requires_grad = True).view(-1,1).float().to(self.device)
        X = self.F_inv(u)

        ### Voglio che mu = 0 e std = 1
        mean = torch.abs(torch.mean(X))
        std = torch.mean(X**2)

        #### proprietà densità
        x = torch.tensor([-30.], requires_grad = True).float().to(self.device)
        lower = self.F(x)[0]
        upper = self.F(-x)[0]

        domain = torch.linspace(-30, 30, 
                               steps = 500, 
                               requires_grad = True).view(-1,1).float().to(self.device)     ## positività
        
        y = self.F(domain.requires_grad_())
        
        density = self.F.derivative(y,domain)
        p = torch.sum(density)
        positivity = torch.sum(F.relu(-density))      # f(x)>= 0
        ####### Constraints della distribuzione 
        one = torch.tensor(1.).to(self.device)
        # media 0
        l = mean
        # varianza 1
        std_loss = self.criterium(std, one)
        # upper = 1 ==> F(infty)=1
        upper_loss = self.criterium(upper, one)
        #lower = 0 ==> F(-infty)= 0
        lower_loss = torch.sum(lower)
        # int f(x)dx = 1
        normality = self.criterium(p, one)

        l = mean + 10*std_loss + upper_loss + lower_loss + positivity + normality
        
        return l, (mean.item(), std_loss.item(), upper_loss.item(), lower_loss.item(), positivity.item(), normality.item())

    def loss_functional(self, img, img_rec, u, x, u_hat, var):
        density1 = self.F_inv.derivative(x, u)
        density2 = self.F.derivative(u_hat, x)

        l = 0
    
        #### chain rule
        identity = self.criterium(density1, 1/density2)
        ### reconstruction loss for distribution
        reconstruction1 =  self.criterium(u, u_hat)
        ### reconstruction loss for image
        reconstruction2 = self.criterium(img, img_rec)
        ### Kullenback Leiberg divergence

        l = identity + reconstruction1 + 2000*reconstruction2
        
        kl = torch.mean(torch.log(density2[density2>0]))
        #logA = torch.mean(torch.log(torch.linalg.det(var)))
        det_var = torch.prod(var,1)
        if torch.any(det_var<0):
            print("det negativo")
        logA = torch.mean(torch.log(det_var))
        if torch.any(torch.isnan(kl)) or torch.any(torch.isnan(logA)):
            kl = torch.tensor(0, device = self.device).float()
        else:
            kl = logA-kl
        l += kl
        l += 0.001/torch.mean(det_var)
        return l, (identity.item(), reconstruction1.item(), reconstruction2.item(), kl.item())


###################################################
####
####            VAE Recurrent 
####    
##################################################  
class dist_fun_rec(nn.Module):
    def __init__(self,
                 inverse: bool, 
                 input_feat:int, 
                 hidden_dim:int = 16):
        super(dist_fun_rec, self).__init__()
        self.hidden_dim = hidden_dim
        self.inverse = inverse
        self.input_feat = input_feat

        # Encoder layers        
        fc1 = [nn.Linear(input_feat, hidden_dim), 
               nn.Sigmoid(), 
               nn.Linear(hidden_dim, 1)]

        self.fc1 = nn.Sequential(*fc1)
    
    def forward(self, x):
        return self.fc1(x)
        
    def derivative(self, x):
        derivative = vmap(jacrev(self.forward))(x)
        return derivative

    def functional_loss(self, conditional, var = None):
        batch = conditional.shape[0]
        device = conditional.device.type
        
        if self.inverse:
            # La funzione inversa deve produrre campioni di media zero e varianza 1
            # Genero 500 rv in [0,1]
            u = torch.rand(batch, 500, 1, requires_grad = True).float().to(device)
            u = torch.cat((u, conditional.unsqueeze(1).repeat(1,500,1)),-1)          # B x 500 x (c+1)
            X = self.forward(u)
    
            ### Voglio che mu = 0 e std = 1
            mean = torch.mean(X)
            std = torch.mean(X**2)-mean**2

            zero = torch.mean(torch.abs(mean))
            one = torch.mean(torch.abs(std - torch.ones_like(std)))
            l = zero + one
            return l
        else:    
            #### proprietà densità
            # 1) lim_{x --> -infty} F(x)=0
            # 2) lim_{x --> infty} F(x)=1
            x = -40 * torch.ones(batch, 1 , requires_grad = True).float().to(device)
            lw = torch.cat((conditional, x), -1 )
            up = torch.cat((conditional, -x), -1)
            lower = self.forward(lw)
            upper = self.forward(up)
            
            zero = torch.mean(torch.abs(lower))
            one = torch.mean(torch.abs(upper-torch.ones_like(upper)))
            
            # 3) F è crescente ==> controllo in un dominio [a,b]
            a = -30
            b = 30
            domain = torch.rand(batch, 500, 1, device = device)*(b-a) + a
            input_pos = torch.cat((domain, conditional.unsqueeze(1).repeat(1,500,1)),-1).requires_grad_()
            density = torch.cat([self.derivative(input_pos[i])[:,:,0].view(1,-1) for i in range(x.shape[0])])
            positivity = torch.sum(F.relu(-density))

            # Poichè la derivata è una densità allora il suo integrale deve essere 1
            prob = torch.sum(density, -1)
            normality = torch.mean(torch.abs(prob-torch.ones_like(prob)))   
            l = zero + one + positivity + normality
        return l
        


class VAE_recurrent(nn.Module):
    def __init__(self,
                 input_feat: int,
                 criterium,
                 device, 
                 hidden_dim: int):
        super(VAE_recurrent, self).__init__()

        # Encoder layers
        self.criterium = criterium
        self.hidden_dim = hidden_dim
        self.device = device
        self.input_feat = input_feat
        self.upper_bound_var = torch.tensor([5.]*hidden_dim, device = device, requires_grad = True).float()
        self.fc1 = nn.Sequential(nn.Flatten(1),
                                 nn.Linear(input_feat, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, 256))
        
        self.fc_mu = nn.Sequential(nn.Linear(256, 128),
                                   nn.Tanh(),
                                   nn.Linear(128, hidden_dim))
        
        self.fc_logvar = nn.Sequential(nn.Linear(256, 128),
                                       nn.Tanh(),
                                       nn.Linear(128, hidden_dim))

        # Decoder layers
        
        self.fc2 = nn.Sequential(nn.Tanh(), 
                                 nn.Linear(hidden_dim, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 512),
                                 nn.Tanh(),
                                 nn.Linear(512, input_feat))

        #### F^{-1}(u) ####
        self.F_inv = nn.ModuleList([dist_fun_rec(inverse = True, input_feat = i+1) for i in range(hidden_dim)])

        #### F(F^{-1}(u)) ####
        self.F = nn.ModuleList([dist_fun_rec(inverse = False, input_feat = i+1) for i in range(hidden_dim)])
        
    def encode(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        log_var = torch.max(torch.min(log_var,torch.ones_like(log_var)*4),torch.ones_like(log_var)*(-4)) 
        var = torch.exp(log_var)
        return mu, var#.view(-1,self.hidden_dim, self.hidden_dim)

    def decode(self, z):
        return self.fc2(z)

        
    def reparameterize(self, mu, var):

        #### Generating the random distribution #####
        b,_ = mu.shape
        eps = torch.tensor([]).to(self.device)
        input = []
        decode = []
        for i in range(self.hidden_dim):
            u = torch.rand(b,1, requires_grad = True).float().to(self.device)
            input.append(u)
            x = self.F_inv[i](torch.cat((u, eps),-1))
            eps = torch.cat((x, eps), -1)
            decode.append(self.F[i](eps))
            
        # tutti gli input che sono stati generati dalla distribuzione uniforme
        u = torch.cat(input, -1)
        
        # tutte le ricostruzioni generate dalla rete
        u_hat = torch.cat(decode, -1)
        ### Perturbing the embedding 
        z = mu + var*eps
        return z, u, u_hat, eps
    
    def forward(self, x):
        mu, var = self.encode(x.view(-1, self.input_feat))
        z, u, u_hat, eps, = self.reparameterize(mu, var)
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, u, eps, u_hat, var
    
     
    def loss_density(self, u, eps, var):
        loss_density_F = 0
        loss_density_F_inv = 0
        loss_derivative = 0

        ### Kullenback Leiberg divergence        
        kl = torch.tensor([0]).float().to(self.device)
                
        for i in range(self.hidden_dim):
            dFinv_du = self.F[i].derivative(torch.cat((u[:,i:i+1],eps[:,:i]), -1))[:, 0, 0]
            dF_dy = self.F[i].derivative(eps[:,:i+1])[:, 0, 0]
            
            loss_derivative +=  self.criterium(dFinv_du, dF_dy)
            loss_density_F += self.F[i].functional_loss(eps[:,:i])
            loss_density_F_inv += self.F_inv[i].functional_loss(eps[:,:i])
        
            if len(dF_dy[dF_dy>0])>0:
                kl += torch.mean(torch.log(var[:,i][dF_dy>0]) - torch.log(var[:,i][dF_dy>0]))

        l = loss_density_F + loss_density_F_inv + loss_derivative + kl
        return l, (loss_derivative.item(), loss_density_F.item(), loss_density_F_inv.item(), kl.item())
    
    def loss_functional(self, img, img_rec, u, eps, u_hat, var):
        
        
        ### reconstruction loss for distribution
        reconstruction1 =  self.criterium(u, u_hat)
        ### reconstruction loss for image
        reconstruction2 = self.criterium(img, img_rec)
        ### Anti annullamento varianza
        var_emb = torch.mean(torch.prod(var, 1))
    
        l = reconstruction1 + 500*reconstruction2 + 1/var_emb
        
        return l, (reconstruction1.item(), reconstruction2.item())


