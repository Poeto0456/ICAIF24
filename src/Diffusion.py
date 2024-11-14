import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import deterministic_NeuralSort
from src.evaluation.strategies import *
equal_weight = EqualWeightPortfolioStrategy()
strategy = MeanReversionStrategy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#scale parameter for W
W = 10.0
#quantiles
alphas = [0.05]
#score function
score = 'quant'

def G1(v):
    return v

def G2(e, scale=1):
    return scale * torch.exp(e / scale)

def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)

def G1_quant(v, W=W):
    return - W * v ** 2 / 2

def G2_quant(e, alpha):
    return alpha * e

def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2

def S_stats(v, e, X, alpha):
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1(v) - G1(X)) + 1. / alpha * G2(e) * (X<=v).float() * (v - X) + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1(X) - G1(v)) + 1. / alpha_inverse * G2(-e) * (X>=v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)

def S_quant(v, e, X, alpha, W=W):
    X = X.to(v.device)
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha * G2_quant(e,alpha) * (X<=v).float() * (v - X) + G2_quant(e,alpha) * (e - v) - G2in_quant(e,alpha)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha_inverse * G2_quant(-e,alpha_inverse) * (X>=v).float() * (X - v) + G2_quant(-e,alpha_inverse) * (v - e) - G2in_quant(-e,alpha_inverse)
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.alphas = alphas
        self.score_name = score
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            self.score_alpha = None

    def forward(self, PNL_validity, PNL):
        loss = 0
        for i, alpha in enumerate(self.alphas):
            PNL_var = PNL_validity[:, [2 * i]]
            PNL_es = PNL_validity[:, [2 * i + 1]]
            loss += self.score_alpha(PNL_var, PNL_es, PNL, alpha)

        return loss

def linear_beta_schedule(timesteps, start=1e-5, end=0.02):
    return torch.linspace(start, end, timesteps)

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=3):
        super(ScoreNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)  
        self.relu = nn.ReLU()

    def forward(self, x, t):
        t = t.float().unsqueeze(-1).unsqueeze(-1) / 1000.
        t = t.expand(-1, 24, 1) 
        x = torch.cat([x, t], dim=-1)
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(batch_size * seq_len, feat_dim)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x) 
        x = x.view(batch_size, seq_len, -1)
        return x

class DDPM:
    def __init__(self, model, betas, train_dl, timesteps=100):
        self.model = model.to(device)
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.timesteps = timesteps
        self.train_dl = train_dl 
        self.criterion = Score().to(device)
        
        self.model_pnl = nn.Sequential(
            nn.Linear(13, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(betas)),
        ).to(device)


    def forward_diffusion(self, x0, t):
        sqrt_alpha_cumprod_t = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - self.alpha_cumprod[t]).view(-1, 1, 1)
        noise = torch.randn_like(x0).to(x0.device)  
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        return xt, noise

    def reverse_process(self, xt, t):
        return self.model(xt, t)
    
    def PnL(self, x):
        PNL = strategy.get_pnl_trajectory(x)
        PNL = PNL.to(device) 
        PNL_s = PNL.reshape(*PNL.shape, 1).to(device)
        perm_matrix = deterministic_NeuralSort(PNL_s, tau=1).to(device)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        batch_size, seq_len, _ = PNL_s.shape
        PNL_validity = self.model_pnl(PNL_sort.view(batch_size, -1))
        
        return PNL.to(device), PNL_validity.to(device)

    def p_losses(self, log_return, x0, x0_hat, t):
        log_return = log_return.to(device)  
        x0 = x0.to(device)  
        t = t.to(device)
        x0_hat = x0_hat.to(device)
        
        xt, noise = self.forward_diffusion(log_return, t)
        predicted_noise = self.reverse_process(xt, t)

        PNL, PNL_validity = self.PnL(x0)
        gen_PNL, gen_PNL_validity = self.PnL(x0_hat)

        real_score = self.criterion(PNL_validity, PNL)
        fake_score = self.criterion(gen_PNL_validity, PNL)
        loss_D = torch.abs(real_score - fake_score)     
        
        mse_loss = nn.functional.mse_loss(x0_hat, x0)
        
        total_loss = mse_loss + 0.5 * loss_D
        
        return total_loss


    def train(self, optimizer, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_dl:
                init_prices_real = batch[0].float().to(device)  
                real_log_return = batch[1].float().to(device)  

                t = torch.randint(0, self.timesteps, (init_prices_real.size(0),)).to(device)

                with torch.no_grad():
                    x_fake_log_return = self.sample(eval_size=real_log_return.size(0))

                price_real = log_return_to_price(real_log_return, init_prices_real)
                price_gen = log_return_to_price(x_fake_log_return, init_prices_real)
                
                optimizer.zero_grad() 

                loss = self.p_losses(real_log_return, price_real, price_gen, t)
                loss.backward() 
                optimizer.step()  
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_dl)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


    @torch.no_grad()
    def sample(self, eval_size=1600):
        xt = torch.randn((eval_size, 24, 3)).to(device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((eval_size,), t, dtype=torch.long).to(device)
            predicted_noise = self.reverse_process(xt, t_tensor)

            # Reverse update of xt based on DDPM equations
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_t = self.alphas[t]
            xt = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
                
        return xt.view(eval_size, 24, 3)



print('Done')