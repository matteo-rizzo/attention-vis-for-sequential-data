from torch import nn
from torch.utils.data import DataLoader

from classes.data.Splitter import Splitter
from classes.networks.AttentionRNN import AttentionRNN
from functions.evaluate import evaluate
from functions.metrics import pprint_metrics, average_metrics
from functions.train import train
from params import *


def main():
    print("\n =========================================================== \n")
    print("\t\t Repeated Cross Validation")
    print("\n =========================================================== \n")

    os.makedirs(PATH_TO_LOG)
    num_reps = NUM_REPETITIONS
    rcv_val_metrics, rcv_test_metrics = [], []
    splitter = Splitter(PATH_TO_SEQS, K, MAX_SEQ_LEN, TRUNCATION_SIDE)

    for rep in range(num_reps):

        print("\n ----------------------------------------------------------- \n")
        print("\t\t CV repetition {}/{} \n".format(rep + 1, num_reps))
        print("\n ----------------------------------------------------------- \n")

        # Split based on the seed of the repetition before performing CV
        splitter.split(seed=rep)
        splitter.save_split_to_file("folds_seed_{}.csv".format(rep))

        num_folds = K
        cv_val_metrics, cv_test_metrics = [], []

        for fold in range(num_folds):
            print("\n *********************************************************** \n")
            print("\n \t\t CV fold {}/{} \n".format(fold + 1, num_folds))
            print("\n *********************************************************** \n")

            fold_datasets = splitter.load_split_datasets(fold)

            train_dataset = fold_datasets['train']
            val_dataset = fold_datasets['val']
            test_dataset = fold_datasets['test']

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            train_size, val_size, test_size = len(train_dataset), len(val_dataset), len(test_dataset)
            print("\n Dataset size - Train: {} | Val: {} | Test: {} \n".format(train_size, val_size, test_size))

            network = AttentionRNN().float().to(DEVICE)
            criterion = nn.CrossEntropyLoss()

            print(network)

            # --- TRAINING ---

            val_metrics = train(network, train_loader, val_loader, criterion, fold)
            cv_val_metrics.append(val_metrics)

            print("\n\n Finished training! \n")

            print("\n Saving fully trained model... \n")
            torch.save(network.state_dict(), os.path.join(PATH_TO_LOG, "trained_fold_{}.pth".format(fold)))

            print("\n *********************************************************** \n")

            # --- TEST ---

            print("\n Testing the model... \n")

            print("\n Loading checkpoint... \n")
            network.load_state_dict(torch.load(os.path.join(PATH_TO_LOG, "best_model_fold_{}.pth".format(fold))))

            test_metrics = evaluate(network, test_loader, criterion)

            # Pretty print the test metrics
            print("\n Test metrics for fold {}/{}: \n".format(fold + 1, num_folds))
            pprint_metrics(test_metrics)

            cv_test_metrics.append(test_metrics)

            print("\n *********************************************************** \n")

        print("\n ----------------------------------------------------------- \n")
        print("\t\t Finished Cross Validation")
        print("\n ----------------------------------------------------------- \n")

        # Average CV metrics across folds for validation and test
        cv_val_avg, cv_test_avg = average_metrics(cv_val_metrics), average_metrics(cv_test_metrics)

        # Pretty print the average CV metrics

        print("\n Average VALIDATION metrics across folds: \n")
        pprint_metrics(cv_val_avg)

        print("\n ........................................................... \n")

        print("\n Average TEST metrics across folds: \n")
        pprint_metrics(cv_test_avg)

        rcv_val_metrics.append(cv_val_avg)
        rcv_test_metrics.append(cv_test_avg)

    print("\n =========================================================== \n")
    print("\t\t Finished Repeated Cross Validation")
    print("\n =========================================================== \n")

    # Average RCV metrics across repetitions for validation and test
    rcv_val_avg, rcv_test_avg = average_metrics(rcv_val_metrics), average_metrics(rcv_test_metrics)

    # Pretty print the average RCV metrics

    print("\n Average VALIDATION metrics across repetitions: \n")
    pprint_metrics(rcv_val_avg)

    print("\n ........................................................... \n")

    print("\n Average TEST metrics across repetitions: \n")
    pprint_metrics(rcv_test_avg)


if __name__ == '__main__':
    main()
