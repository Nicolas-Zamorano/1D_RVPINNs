import sys
import torch
import matplotlib.pyplot as plt


sys.path.insert(1, "../utils/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

from RK4 import RK4

domain = (0, 100)

batch_size = 3000

initial_points = torch.tensor([0.0, 0.0], requires_grad = False)
initial_values = torch.tensor([1.0, 0.5], requires_grad = False)

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

solution_np =  solution.cpu().detach().numpy()
collocation_points_np = collocation_points.cpu().detach().numpy()

plt.plot(collocation_points_np, solution_np[:,0], label='x')
plt.plot(collocation_points_np, solution_np[:,1], label='s')
plt.legend()
plt.show()


# def lotka_volterra(times, values, parameters):
    
#     alpha, beta, gamma, delta = parameters
    
#     x, y = torch.split(values, 1 , dim = 1)

#     f_1 = alpha * x - beta * x * y
#     f_2 = delta * x * y - gamma * y
    
#     return torch.concat([f_1,f_2], dim = 1)

# alpha = 1.1
# beta = 0.4
# gamma = 0.4
# delta = 0.1

# parameters = [alpha, beta, gamma, delta]

# solution = RK4(lotka_volterra, 
#                 initial_values.unsqueeze(0), 
#                 collocation_points, 
#                 parameters)

# solution_np =  solution.cpu().detach().numpy()
# collocation_points_np = collocation_points.cpu().detach().numpy()

# plt.plot(collocation_points_np, solution_np[:,0], label='x')
# plt.plot(collocation_points_np, solution_np[:,1], label='y')
# plt.legend()
# plt.show()
