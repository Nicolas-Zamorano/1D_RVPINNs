import sys
import torch
from datetime import datetime
from numpy import pi

sys.path.insert(0, "../src/")
sys.path.insert(1, "../utils/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache
torch.set_default_dtype(torch.float64)

from Neural_Network import Neural_Network
from Quadrature_Rule import Quadrature_Rule
from Residual import Residual
from Plotting import Plotting

##----------------------Neural Network Parameters------------------##

input_dimension = 1
output_dimension = 2
deep_layers = 5
hidden_layers_dimension = 50

##----------------------Training Parameters------------------##

batch_size = 1000
epochs = 3000
learning_rate = 0.00005
optimizer = "Adam" # Adam or SGD

##----------------------ODE Parameters------------------##

domain = (-pi, pi)

initial_points = torch.tensor([0.0, 0.0], requires_grad = False)
initial_values = torch.tensor([1.0, 0.0], requires_grad = False)

collocation_points = torch.linspace(domain[0], 
                                    domain[1], 
                                    batch_size, 
                                    requires_grad = False)

def governing_equations(NN_evaluation, NN_initial_values, jac_evaluation, initial_values):
    
    x, y = torch.split(NN_evaluation, 1, dim=1)
    dx, dy = torch.split(jac_evaluation, 1, dim=1)
    
    f_1 = -y
    f_2 = x
    
    
    constrain_vector = NN_initial_values - initial_values
    
    return dx, dy, f_1, f_2, constrain_vector
##-------------------Residual Parameters---------------------##

constrain_parameter = 0.01

gram_matrix_inv = torch.tensor([[4.0, -2.0], 
                                [-2.0, 4.0]], 
                               requires_grad = False)
gram_boundary_matrix = torch.eye(output_dimension)

##-------------------Initialization---------------------##

print("Initializating Neural Network...")

NN = Neural_Network(input_dimension = input_dimension, 
                    output_dimension = output_dimension, 
                    deep_layers = deep_layers, 
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = "Adam",
                    learning_rate = learning_rate)

print("Initializating Quadrature Rule...")

quad = Quadrature_Rule(collocation_points = collocation_points)

print("Initializating Residual...")

res = Residual(model_evaluation = NN.model_evaluation,
               quadrature_rule = quad,
               gram_elemental_inv_matrix = gram_matrix_inv,
               gram_boundary_inv_matrix = gram_boundary_matrix,
               governing_equations = governing_equations,
               initial_points = initial_points,
               initial_values = initial_values,
               constrain_parameter= constrain_parameter)

##----------------------Training------------------##

loss_evolution = []

print(f"{'='*30} Training {'='*30}")

for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    res_value = res.residual_value()
    print(f"Loss: {res_value.item():.8f}")
    NN.optimizer_step(res_value)
    loss_evolution.append(res_value.item())
        
##----------------------Plotting------------------##

def exact_solution(x):
    return torch.concat([torch.sin(x),torch.cos(x)], dim = 1)


plt = Plotting(NN.evaluate, domain, exact_solution, loss_evolution=loss_evolution)

plt.plot_IVP()

torch.cuda.empty_cache()
