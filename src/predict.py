import argparse
from autoencoder import Autoencoder
from gplatent import GPLatent
from utils import get_data, process_and_save_prediction
from config import trained_wts_dir

def main(best_wts_path, test_no):
    # STEP 1: Get the input in the latent representation for a new example
    latent_inputs_test, latent_outputs_test = get_data(data_type='latent', dataset='test', auto_wts_path=best_wts_path, print_art=True)

    # STEP 2: Define GP and predict latent distributions
    gp = GPLatent()
    latent_disps, latent_sigmas = gp.predict(inputs=latent_inputs_test[test_no])

    # STEP 3: Define autoencoder and project latent distributions to full field
    auto = Autoencoder(wts_path=best_wts_path)
    full_disps, full_sigmas = auto.reconstruct_distribution(latent_disps=latent_disps, latent_sigmas=latent_sigmas, samples=300)
    true_latent_reconstruction = auto.reconstruct(latent_outputs_test[test_no])

    # STEP 4: Save results to visualize
    process_and_save_prediction(test_no, full_disps, full_sigmas, true_latent_reconstruction)

if __name__ == '__main__':

    default_best_wts_path = trained_wts_dir + "best_auto.h5"  # Default path for best autoencoder weights
    test_no = 276  # Default test example number for predicting and visualising results

    parser = argparse.ArgumentParser(description='Run the autoencoder with specified weights and choose test number for visualising framework prediction.')
    parser.add_argument('--wts_path', type=str, default=default_best_wts_path, help='Path to the best weights file')
    parser.add_argument('--test_no', type=int, default=test_no, help='Number of the example from the test set to predict solutions for')

    args = parser.parse_args()
    best_wts_path = args.wts_path
    test_no = args.test_no

    # Call the main function
    main(best_wts_path, test_no)
