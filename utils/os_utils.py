import os


def list_files(folder_path, just_name=True):
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                if just_name:
                    files.append(filename)
                else:
                    files.append(file_path)
    return files
