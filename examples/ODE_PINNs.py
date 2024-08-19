import sys
import torch
from datetime import datetime
from matplotlib.pyplot import subplots, show

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
deep_layers = 10
hidden_layers_dimension = 100

##----------------------Training Parameters------------------##

batch_size = 5000
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

domain = (0.0, 25.0)

alpha = 0.1
beta = 0.05
gamma = 1.1
delta = 0.1

parameters = [alpha, beta, gamma, delta]

initial_points = torch.tensor([0.0, 0.0], requires_grad = False).unsqueeze(1)
initial_values = torch.tensor([5.0, 1.0], requires_grad = False).unsqueeze(1)

collocation_points = domain[0] + (domain[1] - domain[0]) * torch.rand(batch_size, 1)

def governing_equations(times, values, parameters):
    
    alpha, beta, gamma, delta = parameters
    
    x, y = torch.split(values, 1 , dim = 1)

    f_1 = alpha * x - beta * x * y
    f_2 = delta * x * y - gamma * y
    
    return torch.concat([f_1,f_2], dim = 1)

quad = Quadrature_Rule(collocation_points = collocation_points,
                       boundary_points = initial_points)

##----------------------Posteriori Error------------------##

plot_points = torch.linspace(domain[0],domain[1], 2000).unsqueeze(1)

def exact_solution(x):
    return RK4(governing_equations, 
               initial_values.T,
               x, 
               parameters)

def exact_jacobian_solution(x):
    return governing_equations(x, 
                               exact_evaluation, 
                               parameters)



exact_evaluation = exact_solution(plot_points)

exact_evaluation_np = exact_evaluation.cpu().detach().numpy()
plot_points_np = plot_points.cpu().detach().numpy()

figure_exact, axis_exact = subplots(dpi=500,
                                          figsize=(12, 8))

axis_exact.plot(plot_points_np, exact_evaluation_np)

show()

# exact_jacobian_evaluation = quad.interpolate(lambda x: governing_equations(x, 
#                                                                            exact_evaluation, 
#                                                                            parameters = parameters))


# exact_H_1_norm = quad.H_1_norm(function_evaluation = exact_evaluation,
#                                jacobian_evalution = exact_jacobian_evaluation,
#                                boundary_evaluation = initial_values)

##-------------------Residual Parameters---------------------##

constrain_parameter = 5

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

# loss_relative_error = []
# H_1_relative_error = []

loss_error = []
res_opt = 10^16

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    res_value = res.PINNs_residual_IVP()
    
    if res_value < res_opt:
        res_opt = res_value
        params_opt = NN.state_dict()
    
    # eval_error = quad.interpolate(NN.evaluate) - exact_evaluation
    
    # jac_error = quad.interpolate(NN.jacobian).squeeze(-1) - exact_jacobian_evaluation
    
    # initial_error = quad.interpolate_boundary(NN.evaluate) - initial_values
    
    # H_1_error = quad.H_1_norm(function_evaluation = eval_error,
    #                           jacobian_evalution = jac_error,
    #                           boundary_evaluation = initial_error)/exact_H_1_norm
    
    # res_error = torch.sqrt(res_value)/exact_H_1_norm
    
    # print(f"Loss: {res_value.item():.8f} Relative Loss: {res_error.item():.8f} H^1 norm:{H_1_error.item():.8f}")
    
    print(f"Loss: {res_value.item():.8f}")
    
    NN.optimizer_step(res_value)
    
    quad.update_collocation_points(domain[0] + (domain[1] - domain[0]) * torch.rand(batch_size, 1))
    
    # loss_relative_error.append(res_error.item())
    # H_1_relative_error.append(H_1_error.item())
    
    loss_error.append(res_value.item())
    

NN.load_state_dict(params_opt)

solution = NN.evaluate

##----------------------Plotting------------------##

NN_evaluation = solution(plot_points)

solution_labels = [r"$u_1$", r"$u_2$"]
solution_colors = ["blue", "red"]
NN_labels = [r"$u^{\theta}_1$", r"$u^{\theta}_2$"]
NN_colors = ["orange", "purple"]
NN_linestyle = [":", "-."]

NN_evaluation = NN_evaluation.cpu().detach().numpy()

figure_solution, axis_solution = subplots(dpi=500,
                                          figsize=(12, 8))

for i in range(len(exact_evaluation[1,:])):
    
    axis_solution.plot(plot_points_np,
                       exact_evaluation_np[:,i],
                       label = solution_labels[i],
                       color = solution_colors[i],
                       alpha= 0.6)

for i in range(len(NN_evaluation[1,:])):
    
    axis_solution.plot(plot_points_np,
                       NN_evaluation[:,i],
                       label = NN_labels[i],
                       color = NN_colors[i],
                       linestyle = NN_linestyle[i])
    
axis_solution.set(title="VPINNs final solution", xlabel="t", ylabel="u (t)")
axis_solution.legend()
    
figure_loss, axis_loss = subplots(dpi=500,
                                  figsize=(12,8))

# figure_loglog, axis_loglog = subplots(dpi=500,
#                                   figsize=(12,8))

# axis_loss.semilogy(loss_relative_error, label = r"$\frac{\sqrt{\mathcal{L}(u_\theta)}}{\|u\|_{H^1(\Omega)}}$")
# axis_loss.semilogy(H_1_relative_error, label = r"$\frac{\|u-u_\theta\|_{H^1(\Omega)}}{\|u\|_{H^1(\Omega)}}$")
# axis_loss.set(title="Loss evolution",
#               xlabel="# epochs", 
#               ylabel="Loss")
# axis_loss.legend()

axis_loss.semilogy(loss_error, label = r"\mathcal{L}(u_\theta)$")
axis_loss.set(title="Loss evolution",
              xlabel="# epochs", 
              ylabel="Loss")
axis_loss.legend()

# axis_loglog.loglog(loss_relative_error,
#                    H_1_relative_error)

torch.cuda.empty_cache()

