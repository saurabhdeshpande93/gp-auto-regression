import numpy as np
from config import training_data_path, result_path, dof
import os


def get_data(data_type='full_field', dataset='both', auto_wts_path = None, latest_saved=False, print_art = False):
    """
    Load training or testing data based on the specified type and dataset.

    Note: -
    Framework Input: The input consists of forces represented as body force densities applied to the liver geometry.
                     Each input dimension is a vector of three components  ( b x , b y , b z )  corresponding to the body force
                     densities in the x, y, z directions. These inputs are always provided to GPs within the latent space.

    Framework Ouput: Output is full field displacements of the liver geometry. Dimension of each output is 'dof'(9171).
                     While the latent space has a dimension of 'latent_dim' (16).

    Args:
        data_type (str): Type of data to load ('full_field' or 'latent').
        dataset (str): Which dataset to load ('train', 'test', or 'both').
        auto_wts_path (str): Path to pretrained autoencoder weights (if using pretrained model).
        latest_saved (bool): Boolean indicating whether to load the latest saved data from the recently trained autoencoder model).
        print_art (bool): Boolean indicating whether to print ASCII art.

    Returns:
        Tuple (numpy.ndarray) : Arrays of loaded data based on the case.

            Y_train.shape = (n_train, dof)                          -->   Full field displacements of training set (originally provided)
            Y_test.shape  = (n_test, dof)                           -->   Full field displacements of test set (originally provided)
            latent_inputs_train.shape  = (n_train, 3)               -->   Latent input forces of training set (originally provided)
            latent_inputs_test.shape   = (n_test, 3)                -->   Latent input forces of test set    (originally provided)
            latent_outputs_train.shape = (n_train, latent_dim)      -->   Latent displacements of train set (obtained after encoding Y_train)
            latent_outputs_test.shape  = (n_test, latent_dim)       -->   Latent displacements of test set  (obtained after encoding Y_test)
    """
    # Print ASCII art on terminal if requested
    if print_art:
        framework_art()

    # If using pretrained models, check if latent representations corresponding to the trained autoencoder are available
    if auto_wts_path:
        check_latent_exists(auto_wts_path)

    data_path = training_data_path

    # In case of training from scratch, use the unique timestamp to load the data
    if latest_saved:
        with open('timestamp.txt', 'r') as f:
            timestamp = f.read().strip()
        print(f"\n=== Loading recently compressed test/train displacements (corresponding to the timestamp '{timestamp}') ...")

    # Load the full field data
    if data_type == 'full_field':
        if dataset == 'train':
            Y_train = np.load(data_path+'train_full_outputs.npy')
            return Y_train
        elif dataset == 'test':
            Y_test = np.load(data_path+'test_full_outputs.npy')
            return Y_test
        elif dataset == 'both':
            Y_train = np.load(data_path+'train_full_outputs.npy')
            Y_test = np.load(data_path+'test_full_outputs.npy')
            return Y_train, Y_test
        else:
            raise ValueError("Invalid dataset specified. Choose either 'train', 'test', or 'both'.")

    # Load the latent data
    elif data_type == 'latent':
        if dataset == 'train':
            latent_inputs_train = np.load(data_path+'train_latent_inputs.npy')

            if not latest_saved:
                # If not training autoencoder from scratch, use the best compressed representations
                latent_outputs_train = np.load(data_path+'best_auto_train_latent_outputs.npy')
            else:
                # If training autoencoder from scratch, use the latest compressed representations
                latent_outputs_train = np.load(data_path + timestamp +'_train_latent_outputs.npy')
            return latent_inputs_train, latent_outputs_train

        elif dataset == 'test':
            latent_inputs_test = np.load(data_path+'test_latent_inputs.npy')
            if not latest_saved:
                latent_outputs_test = np.load(data_path+'best_auto_test_latent_outputs.npy')
            else:
                latent_outputs_test = np.load(data_path + timestamp + '_test_latent_outputs.npy')
            return latent_inputs_test, latent_outputs_test

        elif dataset == 'both':
            latent_inputs_train = np.load(data_path+'train_latent_inputs.npy')
            latent_inputs_test = np.load(data_path+'test_latent_inputs.npy')
            if not latest_saved:
                latent_outputs_train = np.load(data_path + 'best_auto_train_latent_outputs.npy')
                latent_outputs_test = np.load(data_path+ 'best_auto_test_latent_outputs.npy')
            else:
                latent_outputs_train = np.load(data_path + timestamp + '_train_latent_outputs.npy')
                latent_outputs_test = np.load(data_path + timestamp + '_test_latent_outputs.npy')
            return latent_inputs_train, latent_outputs_train, latent_inputs_test, latent_outputs_test
        else:
            raise ValueError("Invalid dataset specified. Choose either 'train', 'test', or 'both'.")

    else:
        raise ValueError("Invalid data type specified. Choose either 'full_field' or 'latent'.")



def check_latent_exists(best_wts_path):
    """
    Check if the latent representations exist corresponding to the model with the given weights path.

    Args:
         best_wts_path (str): Path to the weights file corresponding to the trained autoencoder.

    Raises:
        ValueError: If the latent representations file does not exist.
    """
    wts_file = os.path.basename(best_wts_path)
    wts_file_name = os.path.splitext(wts_file)[0]  # Extract file name without extension.

    # Construct the full path to the file
    compressed_path = training_data_path + wts_file_name + '_test_latent_outputs.npy'

    # check if the dir exists
    if os.path.isfile(compressed_path):
        print(f"\n=== Latent representations found corresponding to '{best_wts_path}'.")
    else:
        raise ValueError("\n=== Error: Latent representations don't exit. Generate and train GP on them first.")



def original_order(input):

    """
    Reorder the output array into the original Acegen ordering. Quantities of interest obtained by GP+Autoencoder framework
    represent displacements, stds, errors corresponding to 3 degrees of freedom of each node of the mesh and usual
    finite element tools store in   (1x, 1y, 1z, 2x, 2y, 2z, ....) ordering format. While the proposed framework uses
    (1x, 2x, ... , nx, 1y, 2y ..., ny, 1z, ... , nz) ordering format.

    Args:
        input (numpy.ndarray): Array in the form (1x, 2x, ... , nx, 1y, 2y ..., ny, 1z, ... , nz)

    Returns:
        numpy.ndarray: Array with original Acegen ordering (1x, 1y, 1z, 2x, 2y, 2z ..., nx, ny, nz).
    """

    dim = 3   # 3 dimensional problem
    org_order = input.reshape(dim, int(dof / dim)).transpose().flatten()
    return org_order



def process_and_save_prediction(test_no, full_disps, full_sigmas, true_latent_reconstruction):
    """
    Process the predictions and save the results to a CSV file to visualise it in Acegen framework.

    Args:
        test_no (int): The test example number.
        full_disps (numpy.ndarray): Mean full field displacements predicted by the Gp+Autoencoder framework.
        full_sigmas (numpy.ndarray): Standard deviations predicted by the Gp+Autoencoder framework.
        true_latent_reconstruction (numpy.ndarray): True latent reconstruction from the model with shape (dof,).
    """

    # Remove the extra dimension
    full_disps = np.squeeze(full_disps)
    true_latent_reconstruction = np.squeeze(true_latent_reconstruction)
    full_sigmas = np.squeeze(full_sigmas)

    # True FEM solution
    full_outputs_test = get_data(data_type='full_field', dataset='test')
    fem_solution = full_outputs_test[test_no]

    # Errors
    e_f = np.abs(full_disps - fem_solution)  # Error of the framework
    e_r = np.abs(true_latent_reconstruction - fem_solution)  # Reconstruction error
    e_gp = np.abs(e_f - e_r)  # GP componet error corresponding to the full space

    print(f"\n=== Results for the test example {test_no}:")
    print(f"    Max displacement for a dof  =  {np.max(np.abs(full_disps))}")
    print(f"    Mean error of GP+Autoencoder prediction  =  {np.mean(e_f)}")
    print(f"    Mean reconstruction error =  {np.mean(e_r)}")
    print(f"    Mean GP component error (corresponding to full field) =  {np.mean(e_gp)}")

    data_list = [full_disps, fem_solution, e_f, full_sigmas, e_gp, e_r]

    # Store data to be visualised in an array
    data_example = np.zeros((dof, 6))
    for i, array in enumerate(data_list):
        data_example[:, i] = original_order(array)

    # Create the result directory
    folder_path = create_result_dir(test_no)

    # Save the data
    viz_path = os.path.join(folder_path, f"t{test_no}.csv")
    np.savetxt(viz_path, data_example, delimiter=",")
    print(f"    Data for the test example {test_no} is saved.")
    print(f"    Visualise it in Acegen using the 'visualization.nb' notebook present in the '{result_path}'.\n")



def create_result_dir(test_no):
    """
    Create a directory to save results for a specific test example.

    Args:
        test_no (int): The test example number.

    Returns:
        str: The path to the created or existing directory for storing result of the test example.
    """
    folder_name = f"test_{test_no}"
    folder_path = result_path + folder_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"\n=== Result directory created successfully : '{folder_path}'.")
    else:
        print(f"\n=== Result directory already exists {folder_path}.")
    return folder_path


def framework_art():
    """
    Print ASCII art representation of the framework.
    """
    ascii_art = """
                _                   _____ _____  
     /\        | |           _     / ____|  __ \ 
    /  \  _   _| |_ ___    _| |_  | |  __| |__) |
   / /\ \| | | | __/ _ \  |_   _| | | |_ |  ___/ 
  / ____ \ |_| | || (_) |   |_|   | |__| | |     
 /_/    \_\__,_|\__\___/           \_____|_|  
    """
    print(ascii_art)

