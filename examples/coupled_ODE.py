import sys
import torch
from datetime import datetime
from matplotlib.pyplot import subplots

sys.path.insert(0, "../src/")
sys.path.insert(1, "../utils/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

from Neural_Network import Neural_Network
from Quadrature_Rule import Quadrature_Rule
from Residual import Residual
from RK4 import RK4

##----------------------Neural Network Parameters------------------##

input_dimension = 1
output_dimension = 2
deep_layers = 5
hidden_layers_dimension = 25

##----------------------Training Parameters------------------##

batch_size = 601
epochs = 12000
learning_rate = 0.0005
optimizer = "Adam" # Adam or SGD

NN = Neural_Network(input_dimension = input_dimension, 
                    output_dimension = output_dimension, 
                    deep_layers = deep_layers, 
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = "Adam",
                    learning_rate = learning_rate)

##----------------------ODE Parameters------------------##

domain = (0, 100)

initial_points = torch.tensor([0.0, 0.0], requires_grad = False).unsqueeze(1)
initial_values = torch.tensor([1.0, 0.5], requires_grad = False).unsqueeze(1)

collocation_points = torch.linspace(domain[0], 
                                    domain[1], 
                                    batch_size, 
                                    requires_grad = False).unsqueeze(1)

mu_max = 2.7
K = 0.274
D = 0.33
s_in = 10

parameters = [mu_max, K, D, s_in]

def governing_equations(times, values, parameters):
    
    mu_max, K, D, s_in = parameters
    
    x, s = torch.split(values, 1 , dim = 1)
    
    f_1 = (mu_max * s * x) / (s + K) - D * x
    f_2 = (s_in - s) * D - (mu_max * s * x) / (s + K)

    return torch.concat([f_1,f_2], dim = 1)


quad = Quadrature_Rule(collocation_points = collocation_points,
                       initial_points = initial_points)

##----------------------Posteriori Error------------------##

def exact_solution(x):
    return RK4(governing_equations, 
               initial_values.T,
               x, 
               parameters)

def exact_jacobian_solution(x):
    return governing_equations(x, 
                               exact_evaluation, 
                               parameters)

exact_evaluation = quad.interpolate(exact_solution)


exact_jacobian_evaluation = quad.interpolate(lambda x: governing_equations(x, 
                                                                           exact_evaluation, 
                                                                           parameters = parameters))
x_exact, y_exact = torch.split(exact_evaluation,
                               1,
                               dim = 1)

dx_exact, dy_exact = torch.split(exact_jacobian_evaluation,
                                 1,
                                 dim = 1)

L_2_norm = torch.sum((quad.integrate(x_exact**2+y_exact**2)))

L_2_jacobian_norm = torch.sum((quad.integrate(dx_exact**2+dy_exact**2)))

L_2_boundary_norm = torch.sum((initial_values**2))

exact_H_1_norm = torch.sqrt(L_2_norm + L_2_jacobian_norm + L_2_boundary_norm)


##-------------------Residual Parameters---------------------##

constrain_parameter = 20

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
               constrain_parameter= constrain_parameter,
               compute_relative_error = True,
               error_quad = quad,
               exact_evaluation = exact_evaluation,
               exact_jacobian_evaluation = exact_jacobian_evaluation,
               H_1_exact_norm = exact_H_1_norm)

##----------------------Training------------------##

loss_relative_error = []
H_1_relative_error = []

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    res_value, H_1_error = res.residual_value_IVP(compute_error = True)
    
    res_error = torch.sqrt(res_value)/exact_H_1_norm
    
    print(f"Loss: {res_error.item():.8f} H^1 norm:{H_1_error.item():.8f}")
    
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
