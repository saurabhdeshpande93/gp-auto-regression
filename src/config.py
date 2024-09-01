import os

# Paths
base_path = os.getcwd()
training_data_path = os.getcwd() + '/data/training_data/'   # Path to the training data
trained_wts_dir =  os.getcwd() + '/pretrained_models/'      # Directory for pre-trained autoencoder and GP models
tensorboard_log_dir = os.getcwd() +"/logs/AutoTrain/"       # Directory for TensorBoard logs
result_path = os.getcwd() + '/results/'                     # Directory for saving results of test example predictions


# Autoencoder configs
dof = 9171           # Degrees of freedom: 3057 nodes * 3 DOFs per node
latent_dim = 16      # Latent dimension for the autoencoder, also no. of output distributions for GPs
activation = 'relu'  # Activation function to be used in the autoencoder.
epochs = 3000          # Number of training epochs.
batch_size = 16      # Batch size for training
lr0 = 0.0001         # Initial learning rate for training autoencoder.

# GP configs
length_scale = 1                                # Length scale for the GP kernel
nu = 2.5                                        # Smoothness parameter for the Matern kernel
noise_level = 1e-7                              # Noise level in the GP model
noise_level_bounds = (1e-10, 1)                 # Bounds for the noise level
alpha = 1e-7                                    # Additive noise for stability in GP
gp_optimizer = 'fmin_l_bfgs_b'                  # Optimizer for training GP hyperparameters
n_restarts_optimizer = 5                        # Number of restarts for the optimizer
