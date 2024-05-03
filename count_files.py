import os

def count_files(directory):
    if not os.path.isdir(directory):
        print("Error: Not a valid directory")
        return

    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)

    print(f"Number of files in '{directory}': {file_count}")

# Example usage
directory_path = "/home/users/o/oleksiyu/scratch/CS/responce/k50MB2048_20iret0con0.1W3450_3550_w0.5s1Nboot"  #k50MB2048_20iret0con0W3450_3550_w0.5s1Nboot
count_files(directory_path)