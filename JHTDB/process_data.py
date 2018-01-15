import numpy as np
from glob import glob
from util import take_ave, plot_figure

"""
Process channel flow data.

Input:

U_all -- All velocity data          {ndarray of size Nt X Nx X Ny X Nz X 3}
J_all -- All velocity gradient data {ndarray of size Nt X Nx X Ny X Nz X 9}
y     -- y coordinates              {ndarray of size Ny X 1}

Calculate:

U_ave     -- Mean velocity              {ndarray of size Ny X 3}
J_ave     -- Mean velocity gradient     {ndarray of size Ny X 3 x 3}
Res_ave   -- Reynolds stresses          {ndarray of size Ny X 3 X 3}
tke       -- Turbulence kinetic energy  {ndarray of size Ny X 1}
epsilon   -- Dissipation rate           {ndarray of size Ny X 1}
tau_wall  -- Wall shear stress          {scalar}
u_tau     -- Friction velocity          {scalar}
delta_nu  -- Viscous length scale       {scalar}

Output:

1. profiles of mean velocities, Reynolds stresses and normalized anisotropy stress tensor in original or viscous units
2. txt file consisting of y, tke, epsilon, J_ave and Res_ave
"""


# Simulation parameters
rho = 1  # density ?
nu = 5 * 10 ** (-5)  # viscosity

# # Flow statistics averaged over t = [0,26] (frame = [0,4000])
# u_tau = 4.9968 * 10 ** (-2)     # friction velocity
# delta_nu = 1.0006 * 10 ** (-3)  # viscous length scale = nu/u_tau


# Load in all velocity and velocity gradient data, and also y coordinates
U_files = glob('../Data/raw_82X512X31/u_t*.npy')
J_files = glob('../Data/raw_82X512X31/gu_t*.npy')
U_all = np.array([np.load(file) for file in U_files])
J_all = np.array([np.load(file) for file in J_files])
y = np.load('../Data/raw_82X512X31/y_coords.npy')

# Print the list of time frames
Nt = len(U_files)
print('Imported data from %s time frames: %s' % (Nt, [int(file[-8:-4]) for file in U_files]))


# Reshape velocity gradient array from ... X 9 to ... X 3 x 3
J_all = J_all.reshape(J_all.shape[0], J_all.shape[1], J_all.shape[2], J_all.shape[3], 3, 3)

# Calculate mean velocity and mean velocity gradient
U_ave = take_ave(U_all)
J_ave = take_ave(J_all)

# Calculate Reynolds stresses
u = U_all - U_ave[None, None, :, None, :]
Res = np.matmul(u[:, :, :, :, :, None], u[:, :, :, :, None, :])
Res_ave = take_ave(Res)

# Calculate mean strain rate and mean rotation rate
S = 0.5 * (J_all + J_all.transpose((0, 1, 2, 3, 5, 4)))
R = 0.5 * (J_all - J_all.transpose((0, 1, 2, 3, 5, 4)))
S_ave = take_ave(S)
R_ave = take_ave(R)

# Calculate turbulence kinetic energy
tke = 0.5 * np.sum(take_ave(u ** 2), axis=1)

# Calculate normalized Reynolds anisotropy stress tensor
b = 0.5 * Res_ave / tke[:, None, None] - 1./3. * np.eye(3)

# Calculate dissipation rate
s = S - S_ave[None, None, :, None, :, :]
epsilon = 2 * nu * np.sum(np.sum(take_ave(s ** 2), axis=1), axis=1)

# Calculate wall shear stress: tao_wall = rho*nu*(d<U>/dy) evaluated at y = -1
tau_wall = rho*nu*J_ave[0, 0, 1]

# Calculate friction velocity and viscous length scale
u_tau_calc = np.sqrt(tau_wall/rho)
delta_nu_calc = nu/u_tau_calc

print('Wall shear stress: %s' % tau_wall)
print('Friction velocity: %s' % u_tau_calc)
print('Viscous length scale: %s' % delta_nu_calc)

# Make plots
filename = '../Data/raw_82X512X31/profiles_'+str(Nt)+'frames.png'
plot_figure(y, U_ave, Res_ave, b, delta_nu_calc, u_tau_calc, tke, scale='viscous', fname=filename)

# Output txt file
header = 'y, tke, epsilon, grad_u_11, grad_u_12, grad_u_13, grad_u_21, grad_u_22, grad_u_23, \
grad_u_31, grad_u_32, grad_u_33, uu_11, uu_12, uu_13, uu_21, uu_22, uu_23, uu_31, uu_32, uu_33'
output_array = np.zeros((y.shape[0], 21))
output_array[:, 0] = y
output_array[:, 1] = tke
output_array[:, 2] = epsilon
output_array[:, 3:12] = J_ave.reshape(J_ave.shape[0], 9)
output_array[:, 12:] = Res_ave.reshape(Res_ave.shape[0], 9)
np.savetxt('../Data/JHTDB_channel_82X512X31_'+str(Nt)+'frames.txt', output_array, header=header)
