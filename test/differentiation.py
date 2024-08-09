import sys
import torch

sys.path.insert(0, "../src/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

from Neural_Network import Neural_Network

def jacobian(NN_eval: torch.nn.Module,
             x: torch.Tensor,
             h: float = 0.03125):
    """
    Computes jacobian by finite difference method
    
    parameters:
    - NN_eval (torch.nn.Module): NN model to evalute.
    - x (torch.Tensor): Input.
    - h (torch.Tensor): step size for finite difference method (default is  0.03125)
    
    Returns:
    - jacobian(torch.Tensor): jacobian of NN_eval w.r.t. input.
    """
    
    jacobian = (1/h) * (NN_eval(x+h) - NN_eval(x))    

    return jacobian

NN_ODE = Neural_Network(input_dimension = 1, 
                    output_dimension = 1)

ODE_example_point = torch.tensor([0.0]).unsqueeze(1)

NN_ODE_jac = NN_ODE.jacobian(ODE_example_point).squeeze(2)
NN_ODE_jac_app = jacobian(NN_ODE.evaluate, 
                          ODE_example_point,
                          h = 2**(-12))
error_ODE = abs(NN_ODE_jac - NN_ODE_jac_app)/abs(NN_ODE_jac)

print("Relative error of differenciation of Neural Network:", error_ODE.item())



