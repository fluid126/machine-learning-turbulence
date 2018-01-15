import pyJHTDB.dbinfo
import numpy as np
import time
from util import get_data

"""
Get velocity and velocity gradient data in the JHTDB channel flow database at the raw data points (DNS nodes)
of specified simulation times.

DNS parameters:
t - 4000 frames, domain [0, 25.9935], uniform spacing
x - 2048 nodes, domain [0, 8*pi], uniform spacing, periodic
y - 512 nodes, domain [-1, 1], non-uniform spacing, not periodic
z - 1536 nodes, domain [0, 3*pi], uniform spacing, periodic

To obtain data,
1) specify the desired number of points in each of x, y, z dimensions: Nx, Ny, Nz = (integers).

Three stride sizes will be calculated accordingly in order to obtain one node per stride of nodes in each dimension.
The actual number of points in each dimension could slightly differ from the input one. For example, if the desired
Nx = 100, the resulting stride size is sx = 2048//100 = 20, the actual Nx = ceil(2048/20) = 103.

2) specify the desired time frames: frames = (a list of integers between 0 and 3999).
"""


# Get the entire DNS nodes coordinates and times
xnodes = pyJHTDB.dbinfo.channel['xnodes']
ynodes = pyJHTDB.dbinfo.channel['ynodes']
znodes = pyJHTDB.dbinfo.channel['znodes']
total_times = np.array(list(range(4000)), dtype = np.float32) * 0.0065

# Specify desired number of points in each of x, y, z dimensions
Nx, Ny, Nz = 80, 512, 30

# Specify desired time frames (list of integers between 0 and 3999)
frames = [499, 699, 899]

# Calculate corresponding stride size
sx = xnodes.shape[0] // Nx
sy = ynodes.shape[0] // Ny
sz = znodes.shape[0] // Nz

# Obtain one node per stride
x = xnodes[::sx]
y = ynodes[::sy]
z = znodes[::sz]

# Print actual number of points and stride size in each dimension
Nx = x.shape[0]
Ny = y.shape[0]
Nz = z.shape[0]
print('Nx = %s  sx = %s\nNy = %s  sy = %s\nNz = %s  sz = %s\n'
      % (Nx, sx, Ny, sy, Nz, sz))

# Generate point coordinates
point_coords = np.zeros((Nx, Ny, Nz, 3), np.float32)
point_coords[:, :, :, 0] = x[:, None, None]
point_coords[:, :, :, 1] = y[None, :, None]
point_coords[:, :, :, 2] = z[None, None, :]


start = time.time()  # start timer

# Get data
for frame in frames:
    t = total_times[frame]
    print('t = %s' % t)
    u, grad_u = get_data(point_coords, t)
    np.save('../Data/raw_82X512X31/u_t'+str(frame).zfill(4), u)
    np.save('../Data/raw_82X512X31/gu_t'+str(frame).zfill(4), grad_u)

elapsed = time.time() - start  # end timer
print('Elapsed time = %.2f seconds' % elapsed)


# Save y coordinates
np.save('../Data/raw_82X512X31/y_coords', y)
