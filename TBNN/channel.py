###############################################################
#
# Copyright 2017 Sandia Corporation. Under the terms of
# Contract DE-AC04-94AL85000 with Sandia Corporation, the
# U.S. Government retains certain rights in this software.
# This software is distributed under the BSD-3-Clause license.
#
##############################################################

import numpy as np
from turbulencekepspreprocessor import TurbulenceKEpsDataProcessor
from tbnn import NetworkStructure, TBNN
from util import *


# Define parameters
num_layers = 10  # Number of hidden layers in the TBNN
num_nodes = 15  # Number of nodes per hidden layer
batch_size = 20  # Batch size
init_learning_rate = 0.001  # Initial learning rate to use
learning_rate_decay = 1. # Decay rate for the learning rate effecting convergence, must be between 0 and 1
min_learning_rate = 1.e-6  # Minimum learning rate floor the optimizer will not go below, must be greater than 0
max_epochs = 2000  # Max number of epochs during training
min_epochs = 50  # Min number of training epochs required
interval = 2  # Frequency at which convergence is checked
average_interval = 4  # Number of intervals averaged over for early stopping criteria
split_fraction = 0.8  # Fraction of data to use for training and validation
seed = 12345  # Use for reproducibility, set equal to None for no seeding
train_fraction = 0.8  # Fraction of data used for training the NN vs. validation
print_freq = 50  # Frequency with which diagnostic data will be printed to the screen, in epochs
enforce_realizability = False  # Whether or not we want to enforce realizability constraint on Reynolds stresses
num_realizability_its = 5  # Number of iterations to enforce realizability

# Load in data
k, eps, grad_u, stresses, pos = load_channel_data('../Data/JHTDB_channel_82X512X31_40frames_cg.txt')
# k, eps, grad_u, stresses = load_channel_data_old()

# Calculate inputs and outputs
data_processor = TurbulenceKEpsDataProcessor()
# Sij, Rij = data_processor.calc_Sij_Rij(grad_u, k, eps)
k_eps = k / eps
Sij = k_eps[:, None, None] * 0.5 * (grad_u + grad_u.transpose((0, 2, 1)))
Rij = k_eps[:, None, None] * 0.5 * (grad_u - grad_u.transpose((0, 2, 1)))
x = data_processor.calc_scalar_basis(Sij, Rij, is_train=True)  # Scalar basis
tb = data_processor.calc_tensor_basis(Sij, Rij, quadratic_only=False)  # Tensor basis
y = data_processor.calc_output(stresses)  # Anisotropy tensor

# Enforce realizability
if enforce_realizability:
    for i in range(num_realizability_its):
        y = TurbulenceKEpsDataProcessor.make_realizable(y)

# Split into pretrain (train+val) and test data sets
if seed:
    np.random.seed(seed)  # sets the random seed for Theano
x_pretrain, tb_pretrain, y_pretrain, x_test, tb_test, y_test, pretrain_idx, test_idx = \
    TurbulenceKEpsDataProcessor.train_test_split(x, tb, y, fraction=split_fraction, randomize=True, seed=seed)

# Split into training and validation for early stopping
num_train = int(train_fraction * len(pretrain_idx))
train_idx = pretrain_idx[:num_train]
val_idx = pretrain_idx[num_train:]
x_train = x[train_idx, :]
x_val = x[val_idx, :]
y_train = y[train_idx, :]
y_val = y[val_idx, :]
tb_train = tb[train_idx, :, :]
tb_val = tb[val_idx, :, :]

# Define network structure
structure = NetworkStructure()
structure.set_num_layers(num_layers)
structure.set_num_nodes(num_nodes)

# Initialize and fit TBNN
tbnn = TBNN(structure,
            train_fraction=train_fraction,
            print_freq=print_freq,
            learning_rate_decay=learning_rate_decay,
            min_learning_rate=min_learning_rate)
convergence_results = tbnn.fit(x_pretrain, tb_pretrain, y_pretrain,
                               max_epochs=max_epochs,
                               min_epochs=min_epochs,
                               init_learning_rate=init_learning_rate,
                               interval=interval,
                               average_interval=average_interval,
                               batch_size=batch_size)

# Plot convergence results
plot_convergence_results(convergence_results)

# Make predictions on train, val and test data to get train error, val error and test error
y_tbnn_train = tbnn.predict(x_train, tb_train)
y_tbnn_val = tbnn.predict(x_val, tb_val)
y_tbnn_test = tbnn.predict(x_test, tb_test)

# Enforce realizability
if enforce_realizability:
    for i in range(num_realizability_its):
        y_tbnn_train = TurbulenceKEpsDataProcessor.make_realizable(labels_train)
        y_tbnn_test = TurbulenceKEpsDataProcessor.make_realizable(labels_test)

# Determine RMSE
rmse_tbnn_train = tbnn.rmse_score(y_train, y_tbnn_train)
rmse_tbnn_val = tbnn.rmse_score(y_val, y_tbnn_val)
rmse_tbnn_test = tbnn.rmse_score(y_test, y_tbnn_test)

print "===============TBNN==============="
print "RMSE of training data:", rmse_tbnn_train
print "RMSE of validation data:", rmse_tbnn_val
print "RMSE of test data:", rmse_tbnn_test

# Calculate the Reynolds stress anisotropy tensor (num_points X 9) that RANS would have predicted
# given a linear eddy viscosity hypothesis: a_ij = -2*nu_t*Sij/(2*k) = - C_mu * k / eps * Sij
y_rans = data_processor.calc_rans_anisotropy(grad_u, k, eps)

# Enforce realizability
if enforce_realizability:
    for i in range(num_realizability_its):
        y_rans = TurbulenceKEpsDataProcessor.make_realizable(y_rans)

# Align sample points with TBNN train, val and test data
y_rans_train = y_rans[train_idx]
y_rans_val = y_rans[val_idx]
y_rans_test = y_rans[test_idx]

# Determine RMSE of RANS predictions
rmse_rans_train = tbnn.rmse_score(y_train, y_rans_train)
rmse_rans_val = tbnn.rmse_score(y_val, y_rans_val)
rmse_rans_test = tbnn.rmse_score(y_test, y_rans_test)

print "===============LEVM==============="
print "RMSE of training data:", rmse_rans_train
print "RMSE of validation data:", rmse_rans_val
print "RMSE of test data:", rmse_rans_test


# Prepare results for making plots
true_result = [y_train, y_val, y_test]
rans_result = [y_rans_train, y_rans_val, y_rans_test]
tbnn_result = [y_tbnn_train, y_tbnn_val, y_tbnn_test]
position = [pos[train_idx], pos[val_idx], pos[test_idx]]

# Plot true values, RANS and TBNN predictions vs position
plot_results_vs_position(true_result, rans_result, tbnn_result, position)

# # Plot RANS and TBNN predictions versus true values in single figure
# plot_results(true_result, rans_result, tbnn_result)

# # Plot RANS and TBNN predictions versus true values in separate figures
# plot_results_separate(true_result, rans_result, tbnn_result)


