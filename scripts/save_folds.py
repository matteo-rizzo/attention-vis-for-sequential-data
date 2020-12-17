import pprint

from termcolor import colored

from classes.data.Splitter import Splitter
from params import *


def main():
    seed = 0
    path_to_saved_folds = os.path.join(PATH_TO_DATASET, "folds_seed_{}.csv".format(seed))

    splitter = Splitter(PATH_TO_SEQS, K, MAX_SEQ_LEN, TRUNCATION_SIDE)
    splitter.split(seed)

    pp = pprint.PrettyPrinter(compact=True)
    split_info = splitter.get_split_info()
    for fold in range(K):
        print(colored(f"fold {fold}: ", "blue"))
        pp.pprint(split_info[fold])
        print('\n')

    splitter.save_split_to_file(path_to_saved_folds)
    print("\n Saved folds at {} \n".format(path_to_saved_folds))


if __name__ == '__main__':
    main()
