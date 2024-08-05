import torch

def RK4(function, initial_value, collocation_points):
    """
    Solve initial value Problem (IVP) using 4th order Runge-Kutta.
    
    Parameters:
    -function: function of IVP. it has to have form: f(t,x).
    -initial_value: initial value of IVP.
    -collocation_points: values it witch aproximate the solution of IVP.
    
    return:
    -solution_values: values of the solution in collocation_points.
    """
    
    solution_values = [initial_value]
    
    N = len(collocation_points)
    
    for i in range(1, N):
        print(f"Computing Exact Solution: {i} of {N}", end='',flush = True)
        
        t_prev = collocation_points[i-1]
        t_next = collocation_points[i]
        h = t_next - t_prev
        
        x_prev = solution_values[-1]
        
        k1 = function(t_prev, x_prev)
        k2 = function(t_prev + h/2, x_prev + h*k1/2)
        k3 = function(t_prev + h/2, x_prev + h*k2/2)
        k4 = function(t_next, x_prev + h*k3)
        
        x_next = x_prev + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        solution_values.append(x_next)
        
    print("")
    del function, initial_value, collocation_points
    del N, t_prev, t_next, h, x_prev, k1, k2, k3, k4, x_next 
    
    return torch.stack(solution_values)