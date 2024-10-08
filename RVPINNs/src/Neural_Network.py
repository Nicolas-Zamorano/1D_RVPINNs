import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Neural_Network(torch.nn.Module):
    """
    Neural Network class for constructing a deep neural network with configurable layers, optimizer, and learning rate schedule.

    Parameters:
    - input_dimension (int): Input dimension of the network.
    - output_dimension (int): Output dimension of the network.
    - deep_layers (int): Number of hidden layers (default is 5).
    - hidden_layers_dimension (int): Dimension of each hidden layer (default is 25).
    - activation_function (torch.nn.Module): Activation function (default is torch.nn.Tanh()).
    - optimizer (str): Optimizer name (default is "Adam").
    - learning_rate (float): Learning rate (default is 0.0005).
    - scheduler (str): Type of learning rate scheduler (default is "None").
    - decay_rate (float): Decay rate for learning rate scheduler (default is 0.9).
    - decay_steps (int): Number of steps for learning rate decay (default is 200).
    """
    def __init__(self, 
                 input_dimension: int,          
                 output_dimension: int,         
                 deep_layers: int = 5,      
                 hidden_layers_dimension: int = 25,      
                 activation_function: torch.nn.Module = torch.nn.Tanh(),  
                 optimizer: str = "Adam",   
                 learning_rate: float = 0.0005,
                 scheduler: str = "None",
                 decay_rate: float = 0.9,
                 decay_steps: int = 200):
        
        super().__init__()  
        
        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        self.layer_in = torch.nn.Linear(input_dimension, 
                                        hidden_layers_dimension)
        self.layer_out = torch.nn.Linear(hidden_layers_dimension, 
                                         output_dimension)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_layers_dimension, 
                                                                  hidden_layers_dimension) for _ in range(deep_layers)])
        self.activation_function = activation_function
        
        self.evaluate = torch.func.vmap(self.forward)  
        self.jacobian = torch.func.vmap(torch.func.jacrev(self.forward))
        
        self.optimizer_name = optimizer  
        self.learning_rate = learning_rate  
        
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              lr=self.learning_rate)  
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.learning_rate)  


        if scheduler == "Exponential":
            self.gamma = decay_rate **(1/decay_steps)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                    self.gamma)
        else:
            self.scheduler = None

    def forward(self, 
                Input: torch.Tensor):
        """
        Define how layers are connected.
        
        Parameters:
        - Input (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        Input = Input
        output = self.activation_function(self.layer_in(Input))
        
        for layer in self.middle_layers:
            output = self.activation_function(layer(output))
        
        return self.layer_out(output)

    def optimizer_step(self, 
                       loss_value: torch.Tensor):
        """
        Perform an optimization step.
        
        Parameters:
        - loss_value (torch.Tensor): The loss value to minimize.
        
        Updates:
        - Model parameters using the optimizer.
        """
        if self.scheduler != None:
            print(f"Updating Neural Network - Learning Rate:{self.scheduler.get_last_lr()}")
            self.optimizer.zero_grad() 
            loss_value.backward(retain_graph=True)  
            self.optimizer.step()  
            self.scheduler.step()  
        else:
            print("Updating Neural Network...")
            self.optimizer.zero_grad() 
            loss_value.backward(retain_graph=True)  
            self.optimizer.step()  
