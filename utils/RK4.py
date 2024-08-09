import torch

def RK4(function, initial_value, collocation_points, parameters):
    """
    Solve initial value Problem (IVP) using 4th order Runge-Kutta.
    
    Parameters:
    -function: function of IVP. it has to have form: f(t,x).
    -initial_value: initial value of IVP.
    -collocation_points: values it witch aproximate the solution of IVP.
    -parameters: parameters of IVP.
    
    return:
    -solution_values: values of the solution in collocation_points.
    """
    
    solution_values = torch.zeros([collocation_points.size(0), initial_value.size(1)])
    
    solution_values[0] = initial_value
    
    N = len(collocation_points)
    
    h = collocation_points[1:] - collocation_points[:-1]
    
    for i in range(1, N):
        print(f"\rComputing Exact Solution: {i} of {N}", end='',flush = True)
        
        t_prev = collocation_points[i-1]
        t_next = collocation_points[i]
        
        x_prev = solution_values[-1].unsqueeze_(0)
        
        k1 = function(t_prev, x_prev, parameters)
        k2 = function(t_prev + h[i-1]/2, x_prev + h[i-1]*k1/2, parameters)
        k3 = function(t_prev + h[i-1]/2, x_prev + h[i-1]*k2/2, parameters)
        k4 = function(t_next, x_prev + h[i-1]*k3, parameters)
        
        solution_values[i] = x_prev + h[i-1]/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    print("")
    del function, initial_value, collocation_points
    del N, t_prev, t_next, h, x_prev, k1, k2, k3, k4 
    
    return solution_values