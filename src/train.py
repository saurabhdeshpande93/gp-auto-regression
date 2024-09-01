import argparse
from autoencoder import Autoencoder
from gplatent import GPLatent
from utils import get_data
import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib

def train_autoencoder():
    """
    Train an autoencoder model and save compressed representations of training and test data to further use them to
    train GPs.
    """
    # Save the timestamp to a file to use it later in the GP part
    timestamp = datetime.datetime.now().strftime("%d%m_%H%M")
    with open('timestamp.txt', 'w') as f:
        f.write(timestamp)

    # STEP 1 = Get training data
    full_outputs_train = get_data(data_type='full_field', dataset='train', print_art = True)

    # STEP 2: Define the autoencoder, train it, and then save encoded representations for both the training and test sets
    auto = Autoencoder(Training=True, timestamp = timestamp)
    auto.train_model(Y_train=full_outputs_train, save_compressed=True)

    print(f"\n=== Autoencoder training completed (Training timestamp: {timestamp})")

def train_gp():
    """
    Train independent GP models using latent representations from the trained autoencoder.
    The timestamp generated while autoencoder training is used to load the correct latent representations and
    also for saving the trained GPs with the same timestamp.
    """
    # STEP 3 = Get latent inputs and latent outputs (obtained using the autoencoder trained in the STEP 2)
    latent_inputs_train, latent_outputs_train = get_data(data_type='latent', dataset='train', latest_saved=True)

    # Retrieve the time-stamp generated while autoencoder training
    with open('timestamp.txt', 'r') as f:
        timestamp_auto = f.read().strip()

    # STEP 4 = Define GP and train it
    gp = GPLatent(Training=True, timestamp = timestamp_auto)
    gp.train(latent_inputs_train, latent_outputs_train, with_mask = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Autoencoder or GP')
    parser.add_argument('--model', choices=['autoencoder', 'gp'], required=True, help='Specify the model to train: autoencoder or gp')
    args = parser.parse_args()

    if args.model == 'autoencoder':
        train_autoencoder()
    elif args.model == 'gp':
        train_gp()
