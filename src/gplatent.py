import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import joblib
from joblib import Parallel, delayed
import multiprocessing
import time
from config import trained_wts_dir,latent_dim, length_scale, nu, noise_level, noise_level_bounds, alpha, gp_optimizer, n_restarts_optimizer
import os

class GPLatent:
    """
    A class to implement indepenedet Gaussian Processes (GPs) for training and predicting in the latent space.

    Training phase: It is important to note that retraining the GP is necessary if a different autoencoder model is used.
                    This is because two autoencoder models trained on similar data can produce entirely different encoded data.
                    Consequently, a trained GP is specific to the autoencoder model that was used to generate the encoded data.

    Attributes:
        latent_dim (int): The dimension of the latent space.
        Training (bool): Indicates if the model is in training mode.
        use_cores (int): Number of CPU cores available for parallel processing.
        timestamp (str): Timestamp for saving trained models.
        alpha (float): The alpha parameter for the Gaussian Process.
        gp_optimizer (str): The optimization method for the Gaussian Process.
        n_restarts_optimizer (int): Number of restarts for the optimizer.
        saved_models_path (str): Directory path for saving trained models.
        kernel: The kernel used for the Gaussian Process.
    """
    def __init__(self, Training = False, timestamp = None):
        """
        Initialises the GPLatent class.

        Parameters:
            Training (bool): Indicates whether the independent GP models to be trained. Default is False.
            timestamp (str or None): A timestamp for saving trained models. Used only when Training is True.
        """
        self.latent_dim = latent_dim
        self.Training = Training
        self.use_cores = multiprocessing.cpu_count() - 1
        self.timestamp = timestamp
        self.alpha = alpha
        self.gp_optimizer = gp_optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        if not self.Training:
            print("\n=== Predicting latent distributions using GPs ...")
            self.saved_models_path = trained_wts_dir + "best_GPs/"
        else:
            print("\n=== Training of independent GPs ...")
            self.kernel = 1.0 * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
            self.saved_models_path = trained_wts_dir + self.timestamp + "_GPs/"
            self.create_dir_trained_GPs()

    def create_dir_trained_GPs(self):
        """
        Creates a directory for storing trained Gaussian Process models.
        """
        os.makedirs(self.saved_models_path, exist_ok = True)
        print(f"    Directory created for storing trained GPs : '{self.saved_models_path}'")

    def train(self, latent_inputs_train, latent_outputs_train):
        """
       Trains independent Gaussian Processes in parallel for each latent output. Trained GPs corresponding
       to respective latent units are saved independently. This makes it easy for them to load on CPUs
       during the prediction phase.

       Args:
           latent_inputs_train (numpy.ndarray): Input data for training.
           latent_outputs_train (numpy.ndarray): Output data for training.
       """
        print("\n=== Training independent GPs parallely ...")

        def train_parallel(i):
            print(f"    Training GP for {i + 1}-th latent output")
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha,
                                          optimizer=self.gp_optimizer, n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=True)
            gp.fit(latent_inputs_train, latent_outputs_train[:, i].reshape(-1, 1))

            # Save models
            model_filename = self.saved_models_path + f'gp_{i}.pkl'
            joblib.dump(gp, model_filename)

        # Run the training in parallel using all available cores
        tic = time.time()
        Parallel(n_jobs=-1)(delayed(train_parallel)(i) for i in range(latent_outputs_train.shape[1]))
        toc = time.time()

        print(f"\n=== Trained GPs saved here : {self.saved_models_path}")

        # Print the training time of GPs in mins
        time_seconds = toc - tic
        minutes = time_seconds // 60
        print(f"    GP training time = {minutes} mins")


    def predict(self,inputs):
        """
        This function predicts latent distributions using trained Gaussian Processes (GPs). Each GP is loaded independently,
        which is especially advantageous for CPU systems with limited memory. This approach prevents memory overload and reduces
        the overall prediction time.  Alternatively, all trained GPs can be loaded at once during the initialization of the class
        instance by defining them in the constructor.

        Args:
            inputs (numpy.ndarray): Input data, which is force in latent representation.

        Returns:
            Tuple (numpy.ndarray, numpy.ndarray): Predicted means and standard deviations for the latent distributions (corresponding to displacements).
        """
        # Add extra dimension for 1D inputs to be compatible with gp.predict method.
        if inputs.ndim == 1:
            inputs = np.expand_dims(inputs, axis=0)

        def load_and_predict(i):
            # Load the optimised models
            model_filename = self.saved_models_path + f'gp_{i}.pkl'
            gp = joblib.load(model_filename)

            # Predict gaussian distributions latent units
            y_pred, std = gp.predict(inputs, return_std=True)

            return (y_pred, std)

        # Predict in a parallel way using one less than all available cores
        tic = time.time()
        results = Parallel(n_jobs=self.use_cores)(delayed(load_and_predict)(i) for i in range(self.latent_dim))
        toc = time.time()

        print(f"    Time taken for GP prediction in the latent space = {toc - tic:0.2f} secs")

        # Extract results and stack them
        y_pred_all = np.vstack([res[0] for res in results])
        std_all = np.vstack([res[1] for res in results])

        # Reshape to a compatible format
        y_pred_all = y_pred_all.T
        std_all = std_all.T

        return y_pred_all, std_all