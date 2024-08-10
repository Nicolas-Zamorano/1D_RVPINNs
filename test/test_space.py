import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "../src/")
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.cuda.empty_cache()

from Quadrature_Rule import Quadrature_Rule

collocation_points = torch.linspace(0,10, 11).unsqueeze(1)

quad = Quadrature_Rule(collocation_points)

poly_eval = quad.polynomial_evaluation.cpu().numpy()
poly_points = quad.mapped_integration_nodes.cpu().numpy()

for i in range(2):
    for j in range(10):
        plt.plot(poly_points[j,:],poly_eval[i,j,:])
    
plt.ylim([0,1])
plt.show()