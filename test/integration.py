import sys
import torch

sys.path.insert(0, "../src/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

from Quadrature_Rule import Quadrature_Rule
from Neural_Network import Neural_Network

"""1° Test: Legendre-Gauss with 5 nodes need to be exact in each subinterval for a 5th order polynomial"""

def polynomial(x:torch.Tensor):
    return 6*torch.pow(x, 5)

def poly_integral(x:torch.Tensor):
    return torch.pow(x[1:],6)-torch.pow(x[:-1],6)

collocation_points = torch.linspace(-100,100, 100)

quad = Quadrature_Rule(collocation_points)

integral = poly_integral(collocation_points)

integration_nodes = polynomial(quad.mapped_integration_nodes_single_dimension)

integral_app = quad.integrate(integration_nodes, 
                              multiply_by_test = False)

max_error = max(abs(integral-integral_app)/integral)

"""2° Test: By means of Fundamental Theorem of Calculus, integral of jacobian is NN evalued in boundary of the integration interval."""

def NN_exact(NN: Neural_Network,
             x: torch.Tensor):
    return NN.evaluate(x[1:])-NN.evaluate(x[:-1])

NN = Neural_Network(input_dimension = 1, 
                    output_dimension = 1)

NN_collocation_points = torch.linspace(0, 10, 100)

NN_quad = Quadrature_Rule(NN_collocation_points)

NN_integral = NN_exact(NN, NN_collocation_points.unsqueeze(1))

NN_integration_nodes = NN.jacobian(NN_quad.mapped_integration_nodes_single_dimension.unsqueeze(1)).squeeze(2)

NN_integral_app = NN_quad.integrate(NN_integration_nodes,
                                 multiply_by_test = False).unsqueeze(1)

NN_max_error = torch.max((abs(NN_integral-NN_integral_app)/abs(NN_integral)))

print("Relative error of 5th order polynomial:", max_error.item())
print("Relative error of NN integration:      ", NN_max_error.item())

