import pyJHTDB
import pyJHTDB.dbinfo
import numpy as np
import matplotlib.pyplot as plt

"""
Utility functions
"""


def get_data(point_coords, time):
    """
    Get velocity and velocity gradient at specified spatial points and a specified time in channel flow database.
    :param point_coords: Spatial coordinates of the data points of interest. Must be in single precision.
    :param time: Time of interest.
    :return: Velocity and velocity gradient arrays.
    """

    # Create library object
    lJHTDB = pyJHTDB.libJHTDB()

    # Initialize library object
    lJHTDB.initialize()

    # Get velocity
    u = lJHTDB.getData(time, point_coords,
                       sinterp='NoSInt',
                       data_set='channel',
                       getFunction='getVelocity')

    # Get velocity gradient
    grad_u = lJHTDB.getData(time, point_coords,
                            sinterp='FD4NoInt',
                            data_set='channel',
                            getFunction='getVelocityGradient')

    # Finalize library object
    lJHTDB.finalize()

    return u, grad_u


def take_ave(arr):
    """
    Average data over time and xz-plane.
    :param arr: Data stored in an ndarray. The beginning four dimensions must correspond to time, x, y, z.
    :return: Averaged data.
    """

    arr_tave = np.mean(arr, axis=0)  # average over time
    arr_txave = np.mean(arr_tave, axis=0)  # average over x axis
    arr_txzave = np.mean(arr_txave, axis=1)  # average over z axis

    return arr_txzave


def plot_figure(y, U_ave, Res_ave, b, delta_nu, u_tau, tke, scale, fname):
    """
    Plot mean velocity profile and Reynolds stresses profile in original or viscous units.
    :param y: y-coordinates. {Ny X 1 ndarray}
    :param U_ave: Mean velocities (3 components). {Ny X 3 ndarray}
    :param Res_ave: Reynolds stresses (9 components). {Ny X 3 X 3 ndarray}
    :param b: Reynolds anisotropy tensor (9 components). {Ny X 3 X 3 ndarray}
    :param delta_nu: Viscous length scale. {scalar}
    :param u_tau: Friction velocity. {scalar}
    :param tke: Turbulence kinetic energy. {scalar}
    :param scale: In which units to make the plots, 'original' or 'viscous'. {string}
    :param fname: A path to a filename for saving the plot. {string}
    """

    Re_11 = Res_ave[:, 0, 0]
    Re_12 = Res_ave[:, 0, 1]
    Re_13 = Res_ave[:, 0, 2]
    Re_22 = Res_ave[:, 1, 1]
    Re_23 = Res_ave[:, 1, 2]
    Re_33 = Res_ave[:, 2, 2]

    b_11 = b[:, 0, 0]
    b_12 = b[:, 0, 1]
    b_13 = b[:, 0, 2]
    b_22 = b[:, 1, 1]
    b_23 = b[:, 1, 2]
    b_33 = b[:, 2, 2]

    fig = plt.figure(figsize=(15, 5))

    if scale == 'original':
        fig.suptitle('Channel flow profiles of mean velocities, Reynolds stresses and normalized anisotropy stress \
tensor in original units')

        ax = fig.add_subplot(131)
        ax.plot(y, U_ave)
        ax.set_xlabel('y')
        ax.set_ylabel(r'$<U_i>$')
        # ax.set_title('Mean velocity profile')
        ax.legend(['U', 'V', 'W'], loc='best')

        ax = fig.add_subplot(132)
        ax.plot(y, Re_11, y, Re_22, y, Re_33, y, Re_12)
        ax.set_xlabel('y')
        ax.set_ylabel(r'$<u_i u_j>$')
        # ax.set_title('Reynolds stress profile')
        ax.legend(['uu', 'vv', 'ww', 'uv'], loc='best')

        ax = fig.add_subplot(133)
        ax.plot(y, b_11, y, b_22, y, b_33, y, b_12)
        ax.set_xlabel('y')
        ax.set_ylabel(r'$\frac{<u_i u_j>}{2k} - \frac{1}{3}\delta_{ij}$')
        # ax.set_title('Normalized anisotropy tensor profile')
        ax.legend(['$b_{uu}$', '$b_{vv}$', '$b_{ww}$', '$b_{uv}$'], loc='best')

    if scale == 'viscous':
        fig.suptitle('Channel flow profiles of mean velocities, Reynolds stresses and normalized anisotropy stress \
tensor in viscous units')

        ax = fig.add_subplot(131)
        ax.plot((y + 1) / delta_nu, U_ave / u_tau)
        ax.set_xlabel('y+')
        ax.set_ylabel(r'$<U_i>/u_{\tau}$')
        # ax.set_title('Mean velocity profile')
        ax.legend(['U+', 'V+', 'W+'], loc='best')

        ax = fig.add_subplot(132)
        ax.plot((y + 1) / delta_nu, Re_11 / (u_tau ** 2), label='uu+')
        ax.plot((y + 1) / delta_nu, Re_22 / (u_tau ** 2), label='vv+')
        ax.plot((y + 1) / delta_nu, Re_33 / (u_tau ** 2), label='ww+')
        ax.plot((y + 1) / delta_nu, Re_12 / (u_tau ** 2), label='uv+')
        ax.plot((y + 1) / delta_nu, tke / (u_tau ** 2), label='k+')
        ax.set_xlabel('y+')
        ax.set_ylabel(r'$<u_i u_j>/u_{\tau}^2$')
        # ax.set_title('Reynolds stresses profile')
        ax.legend(loc='best')

        ax = fig.add_subplot(133)
        ax.plot((y + 1) / delta_nu, b_11, label='$b_{uu}$')
        ax.plot((y + 1) / delta_nu, b_22, label='$b_{vv}$')
        ax.plot((y + 1) / delta_nu, b_33, label='$b_{ww}$')
        ax.plot((y + 1) / delta_nu, b_12, label='$b_{uv}$')
        ax.set_xlabel('y+')
        ax.set_ylabel(r'$\frac{<u_i u_j>}{2k} - \frac{1}{3}\delta_{ij}$')
        # ax.set_title('Anisotropy tensors profile')
        ax.legend(loc='best')

    plt.tight_layout(w_pad=3.0, rect=[0.02, 0, 0.98, 0.93])

    plt.savefig(fname, dpi=200)
    plt.show()
