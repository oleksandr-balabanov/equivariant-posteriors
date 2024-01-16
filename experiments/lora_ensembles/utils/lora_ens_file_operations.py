import dill

def load_from_dill(file_path):
    """
    Load data from a dill file.

    :param filename: The name of the file to load the data from.
    :return: The data loaded from the file.
    """
    with open(file_path, 'rb') as file:
        return dill.load(file)
    
def save_to_dill(data, file_path):
    """
    Save the given data to a file using dill.

    :param data: The data to be saved.
    :param filename: The name of the file where the data will be saved.
    """
    with open(file_path, 'wb') as file:
        dill.dump(data, file)

