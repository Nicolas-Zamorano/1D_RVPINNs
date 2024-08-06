import torch
from matplotlib.pyplot import subplots

class Plotting():
    """
    Initialize the Plotting class.
    
    Parameters:
    - NN_evaluate (torch.nn.Module): Function to evaluate the neural network model.
    - domain (): Domain of 
    - exact_solution: 
    - loss_evolution (torch.Tensor):
    - norm_evolution (torch.Tensor): 
    """
    def __init__(self,
                 NN_evaluate: torch.nn.Module,
                 domain,
                 exact_solution,
                 loss_evolution = None,
                 norm_evolution = None
                 ):
        self.NN_evaluate = NN_evaluate
        self.domain = domain
        self.exact_solution = exact_solution
        self.loss_evolution = loss_evolution
        self.norm_evolution = norm_evolution
        
    def plot_IVP(self,
                 solution_labels = ["$x^1$", "$x^2$"],
                 solution_colors = ["blue", "red"],
                 NN_labels = ["$u_{\theta}^1$", "$u_{\theta}^2$"],
                 NN_colors = ["orange", "purple"],
                 NN_linestyle = [":", "-."],
                 plot_norm: bool = False):
        """
        Plotting Neural Network solution of a IVP
        
        Parameters:
        -solution_labels:
        -solution_colors:
        -NN_labels:
        -NN_colors:
        -NN_linestyle:
        -plot_norm:
        """
        plot_points = torch.linspace(self.domain[0], 
                                     self.domain[1], 
                                     steps=1000).unsqueeze(1)
        
        NN_evaluation = self.NN_evaluate(plot_points).cpu().detach().numpy()
        exact_evaluation = self.exact_solution(plot_points).cpu().detach().numpy()
        plot_points_np = plot_points.cpu().detach().numpy()

        figure_solution, axis_solution = subplots(dpi=500,
                                                  figsize=(12, 8))
        
        for i in range(len(exact_evaluation[1,:])):
            
            axis_solution.plot(plot_points_np,
                               exact_evaluation[:,i],
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

        if plot_norm == True:
            
            axis_loss.semilogy(self.loss_evolution)
            axis_loss.set(title="Loss evolution") 
                   xlabel="# epochs", 
                   ylabel="Loss")

        else:
            
            axis_loss.semilogy(self.loss_evolution)
            axis_loss.set(title="Loss evolution with learning rate =%.3f" %learning_rate, 
                   xlabel="# epochs", 
                   ylabel="Loss")
            
            

