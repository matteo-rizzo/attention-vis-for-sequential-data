import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from classes.data.ETDataset import ETDataset
from classes.data.Preprocess import Preprocess
from params import *


class Splitter:

    def __init__(self, path_to_sequences: str, n_splits: int, max_sequence_length: int, truncation_side: str):

        self.__path_to_sequences = path_to_sequences
        self.__n_splits = n_splits
        self.__classes = {}
        self.__max_sequence_length = max_sequence_length
        self.__truncation_side = truncation_side
        self.__split_data = []

    @staticmethod
    def get_group(filename: str) -> str:
        """ Grouping policy """
        return filename.split('-')[0]

    def read_paths(self, path_to_sequences: str):

        for class_folder in os.listdir(path_to_sequences):
            label = class_folder.split('_')[0]
            self.__classes[label] = {}

            fs = os.listdir(os.path.join(path_to_sequences, class_folder))
            self.__classes[label]['x_paths'] = np.array([os.path.join(path_to_sequences, class_folder, f) for f in fs])
            self.__classes[label]['gs'] = np.array([self.get_group(f) for f in fs])
            self.__classes[label]['y'] = np.array([int(label)] * len(self.__classes[label]['x_paths']))

    def split(self, seed: int):

        self.read_paths(self.__path_to_sequences)
        gss = GroupShuffleSplit(n_splits=self.__n_splits, test_size=1 / self.__n_splits, random_state=seed)

        for fold in range(self.__n_splits):

            x_train_paths, x_val_paths, x_test_paths = [], [], []
            y_train, y_val, y_test = [], [], []

            for label in self.__classes.keys():
                x_paths, y = self.__classes[label]['x_paths'], self.__classes[label]['y']
                gs = self.__classes[label]['gs']

                train_val_idx, test_idx = list(gss.split(x_paths, y, gs))[fold]
                train_idx, val_idx = list(gss.split(x_paths[train_val_idx], y[train_val_idx], gs[train_val_idx]))[fold]

                x_train_paths.extend(list(x_paths[train_val_idx][train_idx]))
                y_train.extend(list(y[train_val_idx][train_idx]))

                x_val_paths.extend(list(x_paths[train_val_idx][val_idx]))
                y_val.extend(list(y[train_val_idx][val_idx]))

                x_test_paths.extend(list(x_paths[test_idx]))
                y_test.extend(list(y[test_idx]))

            fold_paths = {
                'train': (x_train_paths, y_train),
                'val': (x_val_paths, y_val),
                'test': (x_test_paths, y_test)
            }

            self.__split_data.append(fold_paths)

    def get_split_info(self) -> list:
        """ Shows how the data has been split in each fold """

        split_info = []
        for fold_paths in self.__split_data:
            fold_info = {
                'train': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['train'][0]])),
                'val': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['val'][0]])),
                'test': list(set([self.get_group(item.split(os.sep)[-1]) for item in fold_paths['test'][0]]))
            }
            split_info.append(fold_info)

        return split_info

    def load_split_datasets(self, fold: int) -> dict:
        """ Loads the data based on the fold paths """

        fold_paths = self.__split_data[fold]
        x_train_paths, y_train = fold_paths['train']
        x_val_paths, y_val = fold_paths['val']
        x_test_paths, y_test = fold_paths['test']

        pp = Preprocess(self.__max_sequence_length, self.__truncation_side)

        return {
            'train': ETDataset(x_train_paths, y_train, transform=pp.transform),
            'val': ETDataset(x_val_paths, y_val, transform=pp.transform),
            'test': ETDataset(x_test_paths, y_test, transform=pp.transform)
        }

    def load_full_dataset(self, fold: int) -> ETDataset:

        fold_paths = self.__split_data[fold]
        x_train_paths, y_train = fold_paths['train']
        x_val_paths, y_val = fold_paths['val']
        x_test_paths, y_test = fold_paths['test']

        x_paths = x_train_paths + x_val_paths + x_test_paths
        y = y_train + y_val + y_test

        pp = Preprocess(self.__max_sequence_length, self.__truncation_side)

        return ETDataset(x_paths, y, transform=pp.transform)

    def save_split_to_file(self, path_to_saved_folds: str):
        pd.DataFrame(self.__split_data).to_csv(path_to_saved_folds, index=False)

    def load_split_from_file(self, path_to_saved_split: str):
        saved_split = pd.read_csv(path_to_saved_split, converters={'train': eval, 'val': eval, 'test': eval})
        self.__split_data = saved_split.to_dict("record")
