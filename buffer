import torch
import numpy as np

class Buffer(torch.nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args

        if input_size is None:
            input_size = args.input_size

        # Load permuted MNIST dataset
        self.x_train_permuted = []
        self.y_train_permuted = []
        self.x_test_permuted = []
        self.y_test_permuted = []

        for i in range(args.num_permutations):
            permuted_data = np.load(f'mnist_permuted_{i}.npz')
            self.x_train_permuted.append(torch.tensor(permuted_data['x_train'], dtype=torch.float32))
            self.y_train_permuted.append(torch.tensor(permuted_data['y_train'], dtype=torch.long))
            self.x_test_permuted.append(torch.tensor(permuted_data['x_test'], dtype=torch.float32))
            self.y_test_permuted.append(torch.tensor(permuted_data['y_test'], dtype=torch.long))

        # Initialize buffer parameters
        self.current_index = 0
        self.buffer_size = len(self.x_train_permuted[0])

    @property
    def x(self):
        return self.x_train_permuted[0][:self.current_index]

    @property
    def y(self):
        return self.y_train_permuted[0][:self.current_index]

    @property
    def t(self):
        return torch.zeros(self.current_index, dtype=torch.long)  # Assuming all samples belong to the same task

    @property
    def valid(self):
        return torch.ones(self.current_index, dtype=torch.bool)  # Assuming all samples are valid

    def add_reservoir(self, x, y, logits=None, t=None):
        n_elem = len(x)
        place_left = max(0, self.buffer_size - self.current_index)
        
        if place_left:
            offset = min(place_left, n_elem)
            self.x_train_permuted[0][self.current_index: self.current_index + offset] = x[:offset]
            self.y_train_permuted[0][self.current_index: self.current_index + offset] = y[:offset]
            self.current_index += offset

    def display(self):
        # You can implement visualization of data if needed
        pass

    def measure_valid(self, generator, classifier):
        # You can implement this method based on your needs
        pass

    def shuffle_(self):
        indices = torch.randperm(self.current_index)
        self.x_train_permuted[0] = self.x_train_permuted[0][indices]
        self.y_train_permuted[0] = self.y_train_permuted[0][indices]

    def delete_up_to(self, remove_after_this_idx):
        self.current_index = min(self.current_index, remove_after_this_idx)

    def sample(self, amt, exclude_task=None, ret_ind=False):
        if exclude_task is not None:
            raise NotImplementedError("Excluding samples from a specific task is not implemented.")
        else:
            indices = torch.randperm(self.current_index)[:amt]
            return self.x_train_permuted[0][indices], self.y_train_permuted[0][indices], self.t[indices]

    def split(self, amt):
        raise NotImplementedError("Splitting the buffer into two parts is not implemented.")


def get_mnist_buffer(args):
    args.input_size = (28, 28)  # Assuming MNIST image size
    args.num_permutations = 10  # Define the number of permutations
    return Buffer(args)
