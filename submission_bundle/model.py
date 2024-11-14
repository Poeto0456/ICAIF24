import os
import pickle
import torch
import torch.nn as nn

print(os.path.abspath(__file__))
PATH_TO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_dict.pkl')
PATH_TO_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_log_return.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def linear_beta_schedule(timesteps, start=1e-5, end=0.02):
    return torch.linspace(start, end, timesteps)

class DDPM:
    def __init__(self, model, betas, timesteps=100):
        self.model = model
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = timesteps

    def reverse_process(self, xt, t):
        return self.model(xt, t)

    @torch.no_grad()
    def sample(self, eval_size=1800, steps=100, device='cuda'):
        self.model = self.model.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        
        xt = torch.randn((eval_size, 24, 3)).to(device)

        for t in reversed(range(steps)):
            t_tensor = torch.full((eval_size,), t, dtype=torch.long).to(device)
            predicted_noise = self.reverse_process(xt, t_tensor)
            alpha_cumprod_t = self.alpha_cumprod[t]
            alpha_t = self.alphas[t]
            xt = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_t)
                
        return xt

class GeneratorWrapper(nn.Module):
    def __init__(self, ddpm_model):
        super(GeneratorWrapper, self).__init__()
        self.ddpm_model = ddpm_model

    def forward(self, batch_size, device='cuda'):
        return self.ddpm_model.sample(eval_size=batch_size, device=device)

def init_generator():
    print("Initializing the model...")
    input_dim = 3
    hidden_dim = 512
    timesteps = 100

    model = ScoreNetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    betas = linear_beta_schedule(timesteps)
    ddpm = DDPM(model=model, betas=betas, timesteps=timesteps)

    print("Loading model parameters...")
    with open(PATH_TO_MODEL, "rb") as f:
        model_params = pickle.load(f)
    model.load_state_dict(model_params)
    model.eval()  

    generator = GeneratorWrapper(ddpm)
    return generator

if __name__ == '__main__':
    generator = init_generator()
    print("Generator loaded. Generating synthetic data...")
    with torch.no_grad():
        generated_sample = generator(batch_size=1800, device=device)
    print(f"Generated sample shape: {generated_sample.shape}")
    print(f"Generated sample: {generated_sample[0, 0:10, :]}")