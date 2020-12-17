import numpy as np
import pandas as pd
import torch as torch


class Preprocess:

    def __init__(self, max_sequence_length, truncation_side):
        self.__max_sequence_length = max_sequence_length
        self.__truncation_side = truncation_side

    def truncate(self, sequence: np.array) -> np.array:
        truncated = {
            'head': sequence[(sequence.shape[0] - self.__max_sequence_length):, :],
            'tail': sequence[:self.__max_sequence_length, :]
        }
        return truncated[self.__truncation_side]

    def pad_sequence(self, sequence: np.array) -> np.array:
        padding = np.zeros((self.__max_sequence_length - len(sequence), sequence.shape[1]))
        return np.append(padding, sequence, axis=0)

    def transform(self, sequence: pd.DataFrame) -> np.array:
        sequence = sequence.values
        sequence = self.truncate(sequence)

        if len(sequence) < self.__max_sequence_length:
            sequence = self.pad_sequence(sequence)

        return torch.from_numpy(sequence)
