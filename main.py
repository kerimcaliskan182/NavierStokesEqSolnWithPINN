import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from time import sleep

# Viscosity coefficient for the fluid
nu = 0.01

# Define a class to solve Navier-Stokes equations using PINN
class NavierStokes():
    def __init__(self, X, Y, T, u, v):
        # Initialize tensors for spatial coordinates (X, Y), time (T), and velocity components (u, v)
        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

        # Null vector for comparing against the Navier-Stokes residuals
        self.null = torch.zeros((self.x.shape[0], 1))

        # Initialize the neural network model
        self.network()

        # Set up the optimizer with the L-BFGS algorithm, suitable for solving complex physical simulations
        self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=10000, max_eval=50000,
                                           history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")
        self.mse = nn.MSELoss()
        self.ls = 0  # Initialize loss
        self.iter = 0  # Initialize iteration counter

    def network(self):
        # Define the architecture of the neural network with 9 hidden layers and Tanh activation function
        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2))  # Output layer for predicting velocity components u and v

    def function(self, x, y, t):
        # Pass inputs through the network and compute derivatives required for Navier-Stokes equations
        res = self.net(torch.hstack((x, y, t)))
        psi, p = res[:, 0:1], res[:, 1:2]

        # Compute derivatives of velocity and pressure
        u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u_x, u_xx, u_y, u_yy, u_t = [torch.autograd.grad(u, var, grad_outputs=torch.ones_like(u), create_graph=True)[0] for var in [x, x, y, y, t]]
        v_x, v_xx, v_y, v_yy, v_t = [torch.autograd.grad(v, var, grad_outputs=torch.ones_like(v), create_graph=True)[0] for var in [x, x, y, y, t]]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        # Compute residuals of the Navier-Stokes equations
        f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        return u, v, p, f, g

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, f_prediction, g_prediction = self.function(self.x, self.y, self.t)

        # calculate losses
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        self.ls = u_loss + v_loss + f_loss +g_loss
 
        # derivative with respect to net's weights:
        self.ls.backward()

        self.iter += 1
        if not self.iter % 100:
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))

        return self.ls

    def train(self):

        # training loop
        self.net.train()
        self.optimizer.step(self.closure)

# Number of training samples
N_train = 50

# Load data from a MATLAB file
data = scipy.io.loadmat(r'C:\Users\kerim\Desktop\phys400 code\NavierStokes\navier stokes eq\cylinder_wake.mat')

# Extract velocity, pressure, time, and spatial coordinates from the dataset
U_star = data['U_star']  # Velocity data (N x 2 x T)
P_star = data['p_star']  # Pressure data (N x T)
t_star = data['t']       # Time data (T x 1)
X_star = data['X_star']  # Spatial coordinates data (N x 2)

# Determine the number of data points (N) and time instances (T)
N = X_star.shape[0]
T = t_star.shape[0]

# Prepare test data for validation
x_test = X_star[:, 0:1]  # x-coordinates for testing
y_test = X_star[:, 1:2]  # y-coordinates for testing
p_test = P_star[:, 0:1]  # Pressure for testing
u_test = U_star[:, 0:1, 0]  # u-component of velocity for testing
t_test = np.ones((x_test.shape[0], x_test.shape[1]))  # Dummy time vector for testing

# Rearrange Data for training
XX = np.tile(X_star[:, 0:1], (1, T))  # Repeat x-coordinate for each time instance
YY = np.tile(X_star[:, 1:2], (1, T))  # Repeat y-coordinate for each time instance
TT = np.tile(t_star, (1, N)).T  # Repeat time vector for each spatial coordinate

# Flatten data to create a training dataset
UU = U_star[:, 0, :].flatten()[:, None]  # Flatten u-component of velocity
VV = U_star[:, 1, :].flatten()[:, None]  # Flatten v-component of velocity
PP = P_star.flatten()[:, None]  # Flatten pressure data

# Training data
x = XX.flatten()[:, None]  # Flatten x-coordinates
y = YY.flatten()[:, None]  # Flatten y-coordinates
t = TT.flatten()[:, None]  # Flatten time data

# Select a random subset of data for training
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = UU[idx, :]
v_train = VV[idx, :]

# Initialize the PINN model with the selected training data
pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)

# Train the model
pinn.train()

# Path to save the trained model
save_path = r'C:\Users\kerim\Desktop\phys400 code\NavierStokes\navier stokes eq\model.pt'

# Optionally save and reload the model
# torch.save(pinn.net.state_dict(), save_path)
# pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
pinn.net.load_state_dict(torch.load(save_path))
pinn.net.eval()

# Convert test data to tensors with gradient tracking
x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)

# Predict the pressure field using the trained model
u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, t_test)

# Reshape the predicted pressure field for visualization
u_plot = p_out.data.cpu().numpy()
u_plot = np.reshape(u_plot, (50, 100))

# Initialize a figure for plotting
fig, ax = plt.subplots()

# Create a contour plot of the initial pressure field
plt.contourf(u_plot, levels=30, cmap='jet')
plt.colorbar()

# Function to update the plot for each frame in the animation
def animate(i):
    ax.clear()
    # Update the plot with new predictions at each time step
    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i*t_test)
    u_plot = p_out.data.cpu().numpy()
    u_plot = np.reshape(u_plot, (50, 100))
    
    # Contour plot for the pressure field at the current time step
    cax = ax.contourf(u_plot, levels=20, cmap='jet')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$p(x,\; y, \; t)$')

ani = animation.FuncAnimation(fig, animate, frames=np.linspace(0, 1, 20), interval=100)

#HTML(ani.to_html5_video())

ani.save('p_field_animation.gif', writer='imagemagick', fps=10)

#plt.show()

# Display the plot for a given interval and then close it
plt.pause(60)  # Display the plot for 60 seconds
plt.close()  # Closes the matplotlib figure window
