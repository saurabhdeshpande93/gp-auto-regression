import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from utils import get_data
from config import dof, latent_dim, activation, epochs, batch_size, lr0, trained_wts_dir, tensorboard_log_dir, training_data_path

np.random.seed(123)
tf.random.set_seed(750)

class Autoencoder:
    """
    A class implemented for training autoencoders, encoding full field displacements to latent representations using encoder of
    the trained autoencoder and reconstructing latent distributions predicted by GPs to full field space by using decoder.

    The proposed framework has two stages
    1. Training  : Train autoencoder and then train Gaussian Processes (GPS) on the latent representations
                   of full field displacements obtained using the encoder part of the autoencoder network

    2. Prediction: Given a force in the latent representation, as a first step predict distributions for latent displacements (using GPs)
                   and then project them to full field using the decoder part of the autoencoder network.

   Attributes:
       Training (bool): Indicates whether the model should be trained from scratch or not
       timestamp (str): Timestamp for saving model weights/compressed representations of inputs.
       dof (int): Degrees of freedom of the full field data (9171 for the liver case).
       latent_dim (int): Dimensionality of the latent space.
       activation (str): Activation function used in the autoencoder model.
       epochs (int): Number of training epochs.
       batch_size (int): Size of batches while training.
       lr0 (float): Initial learning rate.
       encoder (keras.models.Model):  Model for the encoder part.
       decoder (keras.models.Model):  Model for the decoder part.
       model (keras.models.Model): Complete autoencoder model.
       Y_test (numpy.ndarray): Test dataset of the full field displacements.
       trained_wts_path (str): Path to save or load model weights.
       tensorboard_log_dir (str): Directory for TensorBoard logs.
       training_data_path (str): Path to load given data or to save compressed data.
   """

    def __init__(self, wts_path = None, Training = False, timestamp = None):
        """
        Initialises the Autoencoder with specified parameters.

        Args:
            wts_path (str): Path to pre-trained weights (if not training).
            Training (bool): Flag indicating training from scratch or not.
            timestamp (str): Timestamp used in naming while saving model weights and latent displacements..
        """
        self.Training = Training
        self.timestamp = timestamp
        self.dof = dof
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr0 = lr0
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.model = self.create_autoencoder()
        self.Y_test = get_data(data_type='full_field', dataset='test')
        if not self.Training:
            self.trained_wts_path = wts_path   # Path of trained model weights
            print("\n=== Projecting latent distributions to full space by reconstructing latent samples using decoder.")
            self.load_wts()  # Load pre-saved weights of the autoencoder model
        else:
            self.trained_wts_path = trained_wts_dir + self.timestamp + "_auto.h5"  # Path to save trained weights
            self.tensorboard_log_dir = tensorboard_log_dir + self.timestamp    # Directory for TensorBoard logs
            self.training_data_path = training_data_path
            print("\n=== Training autoencoder from scratch ... \n")

    def create_encoder(self):
        """
        Creates the encoder model of the autoencoder.

        Returns:
            keras.models.Model: Encoder model.
        """
        inputs = Input(shape=(self.dof,))
        dense1 = Dense(4096, activation=self.activation)(inputs)
        dense2 = Dense(4096, activation=self.activation)(dense1)
        dense2 = Add()([dense1, dense2])

        dense3 = Dense(2048, activation=self.activation)(dense2)
        dense4 = Dense(2048, activation=self.activation)(dense3)
        dense4 = Add()([dense3, dense4])

        dense5 = Dense(1024, activation=self.activation)(dense4)
        dense6 = Dense(1024, activation=self.activation)(dense5)
        dense6 = Add()([dense5, dense6])

        dense7 = Dense(512, activation=self.activation)(dense6)
        dense8 = Dense(512, activation=self.activation)(dense7)
        dense8 = Add()([dense7, dense8])

        encoded = Dense(units=self.latent_dim)(dense8)
        return Model(inputs=inputs, outputs=encoded, name='encoder')

    def create_decoder(self):
        """
        Creates the decoder model of the autoencoder.

        Returns:
            keras.models.Model: Decoder model.
        """
        decoder_inputs = Input(shape=(self.latent_dim,))
        dense9 = Dense(512, activation=self.activation)(decoder_inputs)
        dense10 = Dense(512, activation=self.activation)(dense9)
        dense10 = Add()([dense9, dense10])

        dense11 = Dense(1024, activation=self.activation)(dense10)
        dense12 = Dense(1024, activation=self.activation)(dense11)
        dense12 = Add()([dense11, dense12])

        dense13 = Dense(2048, activation=self.activation)(dense12)
        dense14 = Dense(2048, activation=self.activation)(dense13)
        dense14 = Add()([dense13, dense14])

        dense15 = Dense(4096, activation=self.activation)(dense14)
        dense16 = Dense(4096, activation=self.activation)(dense15)
        dense17 = Add()([dense15, dense16])

        dense18 = Dense(self.dof, activation=None)(dense17)
        return Model(decoder_inputs, dense18, name='decoder')

    def create_autoencoder(self):
        """
        Creates the full autoencoder model by combining encoder and decoder.

        Returns:
            keras.models.Model: The complete autoencoder model.
        """
        inputs = Input(shape=(self.dof,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        autoencoder_model = Model(inputs=inputs, outputs=decoded)
        return autoencoder_model

    def load_wts(self):
        """
        Loads the weights of the autoencoder using the specified path.

        Raises:
            ValueError: If the weight file does not exist at the specified path.
        """
        if os.path.exists(self.trained_wts_path):
            self.model.load_weights(self.trained_wts_path)
            print(f"    Weights assigned to the autoencoder : {self.trained_wts_path}")
        else:
            raise ValueError(f"Error: Weight file '{self.trained_wts_path}' not found.")

    def compress(self, full_field_input):
        """
        Encodes the input data into the latent space.

        Args:
            full_field_input (numpy.ndarray): Input data to be compressed.

        Returns:
            numpy.ndarray: Encoded representation in latent space.
        """
        # keras.models.Model.predict() doesn't take 1D inputs hence add a dimension if required
        if full_field_input.ndim == 1:
            full_field_input = np.expand_dims(full_field_input, axis=0)
        print("\n=== Given input is encoded to latent space")
        return self.encoder.predict(full_field_input)

    def reconstruct(self, latent_input):
        """
        Reconstructs the input from the latent representation using the decoder.

        Args:
           latent_input (ndarray): Latent representation to be decoded

        Returns:
            numpy.ndarray: Reconstructed data in full space.
        """
        if latent_input.ndim == 1:
            latent_input = np.expand_dims(latent_input, axis=0)
        print("\n=== Given latent is reconstructed to full space (deterministic reconstruction.)")
        return self.decoder.predict(latent_input)

    def reconstruct_distribution(self, latent_disps, latent_sigmas, samples=300):
        """
        This method reconstructs distributions in the full space by generating multiple samples in
        latent distributions. Each sample is projected through a decoder to obtain corresponding samples in the full field space.
        Finally, the mean and standard deviation of these samples are calculated. Since autoencoders provide non-linear compression,
        we can't simply reconstruct means and standard deviations to get full field distributions.

        Non vectorised way for generating sample would be
        for i in range(samples):
            single_sample = np.random.normal(latent_disps,latent_sigma)
            predicts_temp = decoder.predict(single_sample)

        Note: latent_disps.shape = (n_examples, latent_dim), n_examples = latent_disps.ndim

        Args:
            latent_disps (numpy.ndarray): Mean values of latent distributions (obtained as an output of GPs).
                                          Shape: (n_examples, latent_dim), where 'n_examples' is the number of examples.
            latent_sigmas (numpy.ndarray): Standard deviations of latent distributions (obtained as an output of GPs).
                                           Shape: (n_examples, latent_dim), where 'n_examples' is the number of examples.
            samples (int): Number of samples to generate.

        Returns:
            tuple of numpy.ndarray: Mean and standard deviation of the reconstructed samples.
        """
        print("    Reconstructing distributions ...")

        if latent_disps.ndim == 1:
            latent_disps = np.expand_dims(latent_disps, axis=0)
            n_examples = 1
        else:
            n_examples = latent_disps.shape[0]

        # Flatten both arrays to enable generating samples in vectorised way.
        latent_disps_flat = latent_disps.flatten()   # latent_disps_flat.shape = (n_examples*latent_dim,)
        latent_sigmas_flat = latent_sigmas.flatten()

        tic = time.time()
        samples_flat = np.random.normal(latent_disps_flat, latent_sigmas_flat, (samples, latent_disps_flat.size)) # Generate samples in one go
        samples_flat = samples_flat.reshape((samples * n_examples, self.latent_dim)) # Reshape for decoder input

        # Reconstruct all samples using decoder
        samples_flat_full_field = self.decoder.predict(samples_flat)

        # Reshape to compute mean and stds in vectorised way
        samples_full_field = samples_flat_full_field.reshape((samples,n_examples*self.dof))

        # Get the mean and std of all full field samples corresponding to every example
        mean_predicts = np.mean(samples_full_field, axis=0).flatten()
        mean_predicts = mean_predicts.reshape((n_examples, self.dof ))

        std_predicts = np.std(samples_full_field, axis=0).flatten()
        std_predicts = std_predicts.reshape((n_examples, self.dof ))
        toc = time.time()

        print(f"    Time taken for reconstructing with {samples} samples is = {toc - tic:0.2f} secs")

        return mean_predicts, std_predicts

    def train_model(self, Y_train, save_compressed = True):
        """
        Trains the autoencoder model on the provided training data.

        Args:
            Y_train (numpy.ndarray): Training data representing displacements in the full field.
            save_compressed (bool): Flag to save compressed representations.
        """
        opt = Adam(learning_rate = self.lr0)
        self.model.compile(optimizer=opt, loss='mean_squared_error')

        if not os.path.exists(self.tensorboard_log_dir ):
            os.makedirs(self.tensorboard_log_dir )

        tensorboard_callback = TensorBoard(log_dir=self.tensorboard_log_dir, histogram_freq=1)

        # Define ModelCheckpoint callback
        model_checkpoint_callback = ModelCheckpoint(self.trained_wts_path, monitor='val_loss', verbose=1,
            save_best_only=True, save_weights_only=True, mode='min')

        callbacks = [LearningRateScheduler(Autoencoder.lr_scheduler, verbose=1), model_checkpoint_callback, tensorboard_callback]

        self.model.fit(Y_train, Y_train, epochs= self.epochs, batch_size= self.batch_size, shuffle=True, validation_split=0.05, callbacks=callbacks)

        # Now assigned the best weights saved during the training
        self.load_wts()

        # Check performance on the test set
        self.evaluate_auto_model()

        if save_compressed:
            # Get the encoded/compressed representations for both test and train dataset
            print("\n=== Compressing full field train and test data using encoder ... ")
            compressed_train = self.compress(Y_train)
            compressed_test = self.compress(self.Y_test)

            self.save_np_arrays(compressed_train, compressed_test)

    @staticmethod
    def lr_scheduler(epoch, lr):
        """
        Learning rate scheduler for adjusting the learning rate during training.

        Args:
            epoch (int): Current epoch number.
            lr (float): Current learning rate.

        Returns:
            float: Updated learning rate.
        """
        if epoch < 10:
            return lr
        elif 10 <= epoch < 300:
            return lr - (0.0001 - 0.000001) / 290
        else:
            return lr

    def evaluate_auto_model(self):
        """
        Evaluates the autoencoder model on the test dataset and prints performance metrics.
        MAE (float) : Mean absolute metric for the entire test set
        Max displacement (float) : Maximum displacement of a dof in the entire test set
        std (float) : Gives spread of mean errors of test examples, not to be confused with std of each dof prediction
        Max error (float) : Maximum error in predicting the displacement of a particular dof (the worst prediction dof).
         """

        reconstructions = self.model.predict(self.Y_test)
        # Calculate differences
        diff = np.abs(reconstructions - self.Y_test)

        # Calculate mean and max error for each sample
        e = np.mean(diff, axis=1)
        e_max = np.max(diff, axis=1)

        # Calculate MAE and standard deviation of the mean errors
        MAE = np.mean(e)
        std_mean = np.std(e, ddof=1)

        print("\n\n=== Performance over the test set is :")
        print(f"    Max displacement in the test set is: {np.max(np.abs(self.Y_test)):.6f}")
        print(f"    Mean Absolute Error (mean of mean errors of test examples) for the test set is: {MAE:.6f}")
        print(f"    Std of mean errors of test examples is: {std_mean:.6f}")
        print(f"    Max error in the entire test set is: {np.max(e_max):.6f}")

    def save_np_arrays(self, compressed_train, compressed_test):
        """
        Saves encoded (compressed) training and test data to NumPy files.

        Args:
            compressed_train (numpy.ndarray): Encoded training data.
            compressed_test (numpy.ndarray):  Encoded test data.
        """
        np.save(self.training_data_path + self.timestamp + "_train_latent_outputs.npy", compressed_train)
        np.save(self.training_data_path + self.timestamp + "_test_latent_outputs.npy", compressed_test)

        print(f"\n=== Compressed arrays saved at : {self.training_data_path + self.timestamp}_train(test)_latent_outputs.npy")
