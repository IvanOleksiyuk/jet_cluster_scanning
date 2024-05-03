import os
import shutil
import sys

def create_multirun_configs(start_id, end_id, increment, directory, delete_dir=False):
    """
    Generates YAML files with specified content.
    
    Args:
    start_id (int): The starting ID number.
    end_id (int): The ending ID number.
    increment (int): The increment between ID ranges in each file.
    directory (str): The directory where the files will be saved.
    """
    
    if os.path.exists(directory):
        # Remove the directory and all its contents
        shutil.rmtree(directory)
        print(f"Directory '{directory}' has been deleted.")
    else:
        print(f"Directory '{directory}' does not exist.")
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Generate files within the ID range
    for i in range(start_id, end_id, increment):
        file_name = f"b{i}_{i+increment}i0.yml"
        file_path = os.path.join(directory, file_name)
        content = (
            "restart: False\n"
            "bootstrap: True\n"
            f"IDb_start: {i}\n"
            f"IDb_finish: {i + increment}\n"
        )
        
        # Write the content to the YAML file
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"Generated file: {file_path}")

if __name__ == '__main__':
    #create_multirun_configs(0, 4200, 100, '/home/users/o/oleksiyu/WORK/jet_cluster_scanning/config/multirun/background_only')
    create_multirun_configs(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], True if sys.argv[5] == 'True' else False)
