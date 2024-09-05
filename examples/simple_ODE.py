import sys
import torch
from datetime import datetime
from numpy import pi 
from matplotlib.pyplot import subplots

sys.path.insert(0, "../src/")
sys.path.insert(1, "../utils/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

from Neural_Network import Neural_Network
from Quadrature_Rule import Quadrature_Rule
from Residual import Residual

##----------------------Neural Network Parameters------------------##

input_dimension = 1
output_dimension = 2
deep_layers = 5
hidden_layers_dimension = 25

##----------------------Training Parameters------------------##

batch_size = 100
epochs = 15000
learning_rate = 0.00002
optimizer = "Adam" # Adam or SGD

NN = Neural_Network(input_dimension = input_dimension, 
                    output_dimension = output_dimension, 
                    deep_layers = deep_layers, 
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = "Adam",
                    learning_rate = learning_rate)

##----------------------ODE Parameters------------------##

domain = (0.0, 2*pi)

initial_points = torch.tensor([0.0, 0.0], requires_grad = False).unsqueeze(1)
initial_values = torch.tensor([1.0, 0.0], requires_grad = False).unsqueeze(1)

collocation_points = torch.linspace(domain[0], 
                                    domain[1], 
                                    batch_size, 
                                    requires_grad = False).unsqueeze(1)

parameters = None

def governing_equations(times, values, parameters):
    
    x, y = torch.split(values, 1 , dim = 1)
    
    f_1 = -y
    f_2 = x

    return torch.concat([f_1,f_2], dim = 1)

quad = Quadrature_Rule(collocation_points = collocation_points,
                       boundary_points = initial_points)

##----------------------Posteriori Error------------------##

def exact_solution(x):
    return torch.concat([torch.cos(x),
                         torch.sin(x)], 
                        dim = 1)

def exact_jacobian_solution(x):
    return governing_equations(x, 
                               exact_evaluation, 
                               parameters)

exact_evaluation = quad.interpolate(exact_solution)

x_exact, y_exact = torch.split(exact_evaluation, 1, dim = 1)

exact_jacobian_evaluation = quad.interpolate(lambda x: governing_equations(x, 
                                                                           exact_evaluation, 
                                                                           parameters = parameters))

dx_exact, dy_exact = torch.split(exact_evaluation, 1, dim = 1)

# error = quad.integrate(dx_exact + y_exact) + quad.integrate 

exact_H_1_norm = quad.H_1_norm(function_evaluation = torch.zeros_like(exact_evaluation),
                               jacobian_evalution = torch.zeros_like(exact_jacobian_evaluation),
                               boundary_evaluation = initial_values)

exact_norm = torch.sqrt(torch.sum(initial_values**2))

##-------------------Residual Parameters---------------------##

constrain_parameter = 1

gram_matrix_inv = torch.tensor([[4.0, -2.0], 
                                [-2.0, 4.0]], 
                               requires_grad = False)
gram_boundary_matrix = torch.eye(output_dimension)


res = Residual(neural_network = NN,
               quadrature_rule = quad,
               gram_elemental_inv_matrix = gram_matrix_inv,
               gram_boundary_inv_matrix = gram_boundary_matrix,
               governing_equations = governing_equations,
               governing_equations_parameters = parameters,
               initial_points = initial_points,
               initial_values = initial_values,
               constrain_parameter= constrain_parameter)

##----------------------Training------------------##

loss_relative_error = []
H_1_relative_error = []

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    res_value = res.residual_value_IVP()
    
    eval_error = quad.interpolate(NN.evaluate) - exact_evaluation
    
    jac_error = quad.interpolate(NN.jacobian).squeeze(-1) - exact_jacobian_evaluation
    
    initial_error = quad.interpolate_boundary(NN.evaluate) - initial_values
    
    H_1_error = quad.H_1_norm(function_evaluation = eval_error,
                              jacobian_evalution = jac_error,
                              boundary_evaluation = initial_error)/exact_H_1_norm
    
    res_error = torch.sqrt(res_value)/exact_H_1_norm
    
    print(f"Loss: {res_value.item():.8f} Relative Loss: {res_error.item():.8f} H^1 norm:{H_1_error.item():.8f}")
    
    NN.optimizer_step(res_value)
    
    loss_relative_error.append(res_error.item())
    H_1_relative_error.append(H_1_error.item())

solution = NN.evaluate

##----------------------Plotting------------------##

NN_evaluation = quad.interpolate(solution)

solution_labels = [r"$u_1$", r"$u_2$"]
solution_colors = ["blue", "red"]
NN_labels = [r"$u^{\theta}_1$", r"$u^{\theta}_2$"]
NN_colors = ["orange", "purple"]
NN_linestyle = [":", "-."]

NN_evaluation = NN_evaluation.cpu().detach().numpy()
exact_evaluation = exact_evaluation.cpu().detach().numpy()
plot_points = quad.mapped_integration_nodes_single_dimension.cpu().detach().numpy()

figure_solution, axis_solution = subplots(dpi=500,
                                          figsize=(12, 8))

for i in range(len(exact_evaluation[1,:])):
    
    axis_solution.plot(plot_points,
                       exact_evaluation[:,i],
                       label = solution_labels[i],
                       color = solution_colors[i],
                       alpha= 0.6)

for i in range(len(NN_evaluation[1,:])):
    
    axis_solution.plot(plot_points,
                       NN_evaluation[:,i],
                       label = NN_labels[i],
                       color = NN_colors[i],
                       linestyle = NN_linestyle[i])
    
axis_solution.set(title="VPINNs final solution", xlabel="t", ylabel="u (t)")
axis_solution.legend()
    
figure_loss, axis_loss = subplots(dpi=500,
                                  figsize=(12,8))

figure_loglog, axis_loglog = subplots(dpi=500,
                                  figsize=(12,8))

axis_loss.semilogy(loss_relative_error, label = r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{H^1(\Omega)}}$")
axis_loss.semilogy(H_1_relative_error, label = r"$\frac{\|u-u_\theta\|_{H^1(\Omega)}}{\|u\|_{H^1(\Omega)}}$")
axis_loss.set(title="Loss evolution",
              xlabel="# epochs", 
              ylabel="Loss")
axis_loss.legend()

axis_loglog.loglog(loss_relative_error,
                   H_1_relative_error)

torch.cuda.empty_cache()
