import numpy as np
import matplotlib.pyplot as plt


def load_channel_data(filename):
    """
    Loads in channel flow data
    """

    # Load in data from txt file
    data = np.loadtxt(filename, skiprows=1)
    pos = data[:, 0]
    k = data[:, 1]
    eps = data[:, 2]
    grad_u_flat = data[:, 3:12]
    stresses_flat = data[:, 12:]

    # Reshape grad_u and stresses to num_points X 3 X 3 arrays
    num_points = data.shape[0]
    grad_u = grad_u_flat.reshape(num_points, 3, 3)
    stresses = stresses_flat.reshape(num_points, 3, 3)

    return k, eps, grad_u, stresses, pos


def load_channel_data_old():
    """
    Loads in channel flow data
    :return:
    """

    # Load in data from Moser_channel.txt
    data = np.loadtxt('Data/Moser_channel.txt', skiprows=4)
    k = data[:, 0]
    eps = data[:, 1]
    grad_u_flat = data[:, 2:11]
    stresses_flat = data[:, 11:]

    # Reshape grad_u and stresses to num_points X 3 X 3 arrays
    num_points = data.shape[0]
    grad_u = np.zeros((num_points, 3, 3))
    stresses = np.zeros((num_points, 3, 3))
    for i in xrange(3):
        for j in xrange(3):
            grad_u[:, i, j] = grad_u_flat[:, i*3+j]
            stresses[:, i, j] = stresses_flat[:, i*3+j]
    return k, eps, grad_u, stresses


def plot_convergence_results(convergence_results):
    convergence_results = np.array(convergence_results)
    epochs = convergence_results[:, 0]
    train_error = convergence_results[:, 1]
    val_error = convergence_results[:, 2]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_error, label='train error')
    plt.plot(epochs, val_error, label='val error')
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.savefig('convergence.png', dpi=200)
    plt.show()


def plot_results_vs_position(y_true, y_rans, y_tbnn, position):
    y_true_train, y_true_val, y_true_test = y_true[0], y_true[1], y_true[2]
    y_rans_train, y_rans_val, y_rans_test = y_rans[0], y_rans[1], y_rans[2]
    y_tbnn_train, y_tbnn_val, y_tbnn_test = y_tbnn[0], y_tbnn[1], y_tbnn[2]
    pos_train, pos_val, pos_test = position[0], position[1], position[2]

    labels = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    for count, i in enumerate([0, 4, 8, 1]):
        ax0 = axes[count, 0]
        ax0.plot(pos_train, y_true_train[:, i], 'k.', ms=2, label='DNS')
        ax0.plot(pos_train, y_tbnn_train[:, i], '.', ms=2, label='TBNN')
        ax0.plot(pos_train, y_rans_train[:, i], '.', ms=2, label='LEVM')
        ax0.set_xlim(-1, 1)
        ax0.set_xlabel('y')
        ax0.set_ylabel(r'$b_{%s}$' % labels[i])
        ax0.set_title('training set')
        ax0.legend(loc='best', prop={'size': 6})

        ax1 = axes[count, 1]
        ax1.plot(pos_val, y_true_val[:, i], 'k.', ms=2, label='DNS')
        ax1.plot(pos_val, y_tbnn_val[:, i], '.', ms=2, label='TBNN')
        ax1.plot(pos_val, y_rans_val[:, i], '.', ms=2, label='LEVM')
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel('y')
        ax1.set_ylabel(r'$b_{%s}$' % labels[i])
        ax1.set_title('validation set')
        ax1.legend(loc='best', prop={'size': 6})

        ax2 = axes[count, 2]
        ax2.plot(pos_test, y_true_test[:, i], 'k.', ms=2, label='DNS')
        ax2.plot(pos_test, y_tbnn_test[:, i], '.', ms=2, label='TBNN')
        ax2.plot(pos_test, y_rans_test[:, i], '.', ms=2, label='LEVM')
        ax2.set_xlim(-1, 1)
        ax2.set_xlabel('y')
        ax2.set_ylabel(r'$b_{%s}$' % labels[i])
        ax2.set_title('test set')
        ax2.legend(loc='best', prop={'size': 6})

    plt.tight_layout()
    plt.savefig('b_vs_y.png', dpi=200)
    plt.show()


def plot_results(y_true, y_rans, y_tbnn):
    """
    Create a plot with 9 subplots.  Each subplot shows the predicted vs the true value of that
    stress anisotropy component.  Correct predictions should lie on the y=x line (shown with
    red dash).
    :param predicted_stresses: Predicted Reynolds stress anisotropy (from TBNN predictions)
    :param true_stresses: True Reynolds stress anisotropy (from DNS)
    """
    y_true_train, y_true_val, y_true_test = y_true[0], y_true[1], y_true[2]
    y_rans_train, y_rans_val, y_rans_test = y_rans[0], y_rans[1], y_rans[2]
    y_tbnn_train, y_tbnn_val, y_tbnn_test = y_tbnn[0], y_tbnn[1], y_tbnn[2]

    labels = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']

    fig = plt.figure(figsize=(9, 9))
    fig.patch.set_facecolor('white')
    on_diag = [0, 4, 8]
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        ax = fig.gca()
        ax.scatter(y_true_test[:, i], y_rans_test[:, i], c='C1', marker='o', s=10, label='LEVM test')
        ax.scatter(y_true_test[:, i], y_tbnn_test[:, i], c='C0', marker='o', s=10, label='TBNN test')
        #         ax.scatter(y_true_train[:, i], y_rans_train[:, i], c='C1', marker='x', s=10, label='LEVM train')
        #         ax.scatter(y_true_train[:, i], y_tbnn_train[:, i], c='C0', marker='x', s=10, label='TBNN train')
        ax.set_aspect('equal')
        ax.autoscale(False)
        ax.plot([-1., 1.], [-1., 1.], 'r--')
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        ax.set_title(r'$b_{%s}$' % labels[i])
        if i in on_diag:
            ax.set_xlim([-1. / 3., 2. / 3.])
            ax.set_ylim([-1. / 3., 2. / 3.])
        else:
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
        ax.legend(loc='upper left', prop={'size': 6})
    plt.tight_layout()
    # plt.savefig('tbnn_vs_levm.png', dpi=200)
    plt.show()


def plot_results_separate(y_true, y_rans, y_tbnn):
    y_true_train, y_true_val, y_true_test = y_true[0], y_true[1], y_true[2]
    y_rans_train, y_rans_val, y_rans_test = y_rans[0], y_rans[1], y_rans[2]
    y_tbnn_train, y_tbnn_val, y_tbnn_test = y_tbnn[0], y_tbnn[1], y_tbnn[2]

    labels = ['uu', 'uv', 'uw', 'vu', 'vv', 'vw', 'wu', 'wv', 'ww']
    on_diag = [0, 4, 8]
    for i in [0, 1, 2, 4, 5, 8]:
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true_test[:, i], y_rans_test[:, i], c='C1', marker='o', s=45, label='LEVM test')
        plt.scatter(y_true_test[:, i], y_tbnn_test[:, i], c='C0', marker='o', s=45, label='TBNN test')
        #         plt.scatter(y_true_train[:, i], y_rans_train[:, i], c='C1', marker='x', s=15, label='LEVM train')
        #         plt.scatter(y_true_train[:, i], y_tbnn_train[:, i], c='C0', marker='x', s=15, label='TBNN train')
        plt.axes().set_aspect('equal')
        plt.axes().autoscale(False)
        plt.plot([-1., 1.], [-1., 1.], 'r--')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.title(r'$b_{%s}$' % labels[i])
        if i in on_diag:
            plt.xlim([-0.1, 0.3])
            plt.ylim([-0.1, 0.3])
        else:
            plt.xlim([-0.25, 0.25])
            plt.ylim([-0.25, 0.25])
        plt.tight_layout()
        plt.legend(loc='upper left')
        plt.savefig('tbnn_vs_levm_%s.png' % labels[i], dpi=200)
    plt.show()


