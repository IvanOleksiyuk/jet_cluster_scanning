import h5py
import numpy as np


def create_subsample_h5(
    original_file_path, subsample_file_path, n=1000, seed=42, sort_indices=None
):
    # Open the original H5 file
    file = h5py.File(original_file_path, "r")

    # Access the dataset containing the data instances
    dataset = file["data"]

    # Get the total number of data instances in the dataset
    total_instances = dataset.shape[0]

    # Generate random indices for the subsample
    np.random.seed(seed)
    subsample_indices = np.random.choice(total_instances, size=n, replace=False)
    subsample_indices.sort()
    # Retrieve the subsample from the dataset
    subsample = dataset[subsample_indices]
    if sort_indices is not None:
        subsample = subsample[sort_indices]
    # Close the original H5 file
    file.close()

    # Create a new H5 file to store the subsample
    new_file = h5py.File(subsample_file_path, "w")

    # Create a new dataset in the new H5 file and write the subsample to it
    new_dataset = new_file.create_dataset("data", data=subsample)

    # Close the new H5 file
    new_file.close()

    print(f"Subsample {len(subsample)} saved to {subsample_file_path} successfully.")


def create_subsample_npy_1d(
    original_array_path, subsample_array_path, n=1000, seed=42, sort=True
):
    # Load the original NumPy array
    original_data = np.load(original_array_path)

    # Get the total number of data instances in the array
    total_instances = original_data.shape[0]

    # Generate random indices for the subsample
    np.random.seed(seed)
    subsample_indices = np.random.choice(total_instances, size=n, replace=False)

    # Retrieve the subsample from the array
    subsample = original_data[subsample_indices]
    if sort:
        sort_indices = np.argsort(subsample)
        subsample = subsample[sort_indices]
    # Save the subsample to a new .npy file
    np.save(subsample_array_path, subsample)

    print(
        f"Subsample with {len(subsample)} saved to {subsample_array_path} successfully."
    )
    if sort:
        return sort_indices


original_array_path = "../../DATA/LHCO/mjj_sig_sort.npy"
subsample_array_path = "mjj_sig_sort.npy"

sort_indices = create_subsample_npy_1d(original_array_path, subsample_array_path)

original_file_path = "../../DATA/LHCO/v2JetImSort_sig.h5"
subsample_file_path = "v2JetImSort_sig.h5"

create_subsample_h5(original_file_path, subsample_file_path, sort_indices=sort_indices)

original_array_path = "../../DATA/LHCO/mjj_bkg_sort.npy"
subsample_array_path = "mjj_bkg_sort.npy"

sort_indices = create_subsample_npy_1d(original_array_path, subsample_array_path)

original_file_path = "../../DATA/LHCO/v2JetImSort_bkg.h5"
subsample_file_path = "v2JetImSort_bkg.h5"

create_subsample_h5(original_file_path, subsample_file_path, sort_indices=sort_indices)
