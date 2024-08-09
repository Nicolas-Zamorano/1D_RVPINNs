import sys
import torch

sys.path.insert(0, "../src/")
sys.path.insert(1, "../utils/")
torch.cuda.empty_cache
torch.set_default_dtype(torch.float64)

from RK4 import RK4

domain = (0, 50)

h = 2**(-5)

batch_size = round((domain[1]-domain[0])/h)

initial_points = torch.tensor([0.0, 0.0], requires_grad = False)
initial_values = torch.tensor([0.5, 1.0], requires_grad = False)

collocation_points = torch.linspace(domain[0], 
                                    domain[1], 
                                    batch_size, 
                                    requires_grad = False)

mu_max = 2.7
K = 0.274
D = 0.33
s_in = 10.0

parameters = [mu_max, K, D, s_in]

def governing_equations(times, values, parameters):
    
    mu_max, K, D, s_in = parameters
    
    x, s = torch.split(values, 1 , dim = 1)
    
    f_1 = (mu_max * s * x) / (s + K) - D * x
    f_2 = (s_in - s) * D - (mu_max * s * x) / (s + K)

    return torch.concat([f_1,f_2], dim = 1)

solution = RK4(governing_equations, 
               initial_values.unsqueeze(0), 
               collocation_points, 
               parameters)

x, s = torch.split(solution, 1, dim = 1)

import matplotlib.pyplot as plt

x_np =  x.detach().numpy()
s_np = s.detach().numpy()

plt.plot(collocation_points, x_np, label='x')
plt.plot(collocation_points, s_np, label='s')
plt.legend()
plt.show()