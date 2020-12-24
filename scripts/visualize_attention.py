import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.data.Splitter import Splitter
from classes.networks.AttentionRNN import AttentionRNN
from functions.metrics import compute_batch_accuracy, compute_optimal_roc_threshold, compute_metrics, pprint_metrics
from params import *

PATH_TO_PLOTS = os.path.join("plots", "confusion_{}".format(time.time()))
os.makedirs(PATH_TO_PLOTS)


def fetch_attention_weights(network: nn.Module, dataloader: DataLoader, path_to_plots: str = ""):
    criterion = nn.CrossEntropyLoss()
    network = network.eval()

    attentions, y_scores, y_true, loss, accuracy = [], [], [], [], []

    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader)):
            # Move validation inputs and labels to device
            x = x.float().to(DEVICE)
            y = torch.from_numpy(np.asarray(y)).long().to(DEVICE)

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x.shape[0]).to(DEVICE)

            # Predict
            a, o = network(x, h)
            o = o.to(DEVICE)
            a = a.squeeze()

            if path_to_plots:
                plot_attention_overlay(x.numpy(), a.numpy(), path_to_plots, batch_idx=i)

            # Store attentions
            attentions += a.tolist()

            # Store all validation loss and accuracy values for computing avg
            loss += [criterion(o, y).item()]
            accuracy += [compute_batch_accuracy(o, y)]

            # Store predicted scores and ground truth labels
            y_scores += torch.exp(o).cpu().numpy().tolist()
            y_true += y.cpu().numpy().tolist()

    y_scores, y_true = np.array(y_scores).reshape((len(y_scores), 2)), np.array(y_true)

    # Compute predicted labels based on the optimal ROC threshold
    threshold = compute_optimal_roc_threshold(y_true, y_scores[:, 1])
    y_pred = np.array(y_scores[:, 1] >= threshold, dtype=np.int)

    # Compute the validation metrics
    avg_loss, avg_accuracy = np.mean(loss), np.mean(accuracy)
    metrics = compute_metrics(y_true, y_pred, y_scores[:, 1])
    metrics["loss"] = avg_loss
    metrics["accuracy"] = avg_accuracy

    return attentions, metrics


def plot_attention_overlay(x: np.array, a: list, path_to_plots: str, batch_idx: int):
    for i in range(x.shape[0]):
        item = x[i, :, :]
        plt.scatter(item[:, 0], item[:, 1], c=a[i, :], vmin=0, vmax=1050)
        plt.plot(item[:, 0], item[:, 1], color="silver", alpha=.5)
        plt.axis('off')
        plt.savefig(os.path.join(path_to_plots, str((i + 1) * (batch_idx + 1))), bbox_inches='tight')
        plt.clf()


def plot_set_attention(network: nn.Module, dataloader: DataLoader, set_type: str, fold: int, path_to_plots: str):
    path_to_sp = os.path.join(path_to_plots, "sp", set_type.lower())
    os.makedirs(path_to_sp)

    print("\n Computing attention weights on {} set... \n".format(set_type))
    attention, metrics = fetch_attention_weights(network, dataloader, path_to_sp)

    print("\n {} metrics: \n".format(set_type))
    pprint_metrics(metrics)

    plt.imshow(np.array(attention).transpose(), cmap="magma", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()

    plt.title("Attention for {} set (fold {}) | accuracy: {:.4}".format(set_type, fold, metrics["accuracy"]))
    plt.xlabel('Data Item')
    plt.ylabel('Time Step')

    plt.savefig(os.path.join(path_to_plots, set_type.lower()), bbox_inches="tight", dpi=300)
    plt.clf()


def plot_dataset_attention(network: nn.Module, dataloader: DataLoader, train_size: int, fold: int, path_to_plot: str):
    print("\n Computing attention weights for full dataset... \n")
    attention, metrics = fetch_attention_weights(network, dataloader)

    print("\n Dataset metrics: \n")
    pprint_metrics(metrics)

    plt.imshow(np.array(attention).transpose(), cmap="magma", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(orientation='horizontal')
    plt.xticks([train_size])
    plt.tight_layout()

    plt.title("Attention for full dataset (fold {}) | accuracy: {:.4}".format(fold, metrics["accuracy"]))
    plt.xlabel('Data Item')
    plt.ylabel('Time Step')

    plt.savefig(os.path.join(path_to_plot, "dataset"), bbox_inches="tight", dpi=300)
    plt.clf()


def main():
    splitter = Splitter(PATH_TO_SEQS, K, MAX_SEQ_LEN, TRUNCATION_SIDE)
    splitter.load_split_from_file(os.path.join(PATH_TO_DATASET, "folds_seed_0.csv"))

    for fold in range(K):
        print("\n ---------------------------------------------- \n")
        print("\t Plotting attention for fold {}/{}".format(fold + 1, K))
        print("\n ---------------------------------------------- \n")

        fold_datasets = splitter.load_split_datasets(fold)

        train_dataset, val_dataset, test_dataset = fold_datasets['train'], fold_datasets['val'], fold_datasets['test']
        train_size, val_size, test_size = len(train_dataset), len(val_dataset), len(test_dataset)
        print("\n Dataset size - Train: {} | Val: {} | Test: {} \n".format(train_size, val_size, test_size))

        network = AttentionRNN(return_attention=True).to(DEVICE)
        print("\n Loading pretrained model... \n")
        path_to_pretrained = os.path.join(PATH_TO_PRETRAINED, "fold_{}_best_model.pth".format(fold))
        network.load_state_dict(torch.load(path_to_pretrained, map_location=DEVICE))
        print(network)

        path_to_plots = os.path.join(PATH_TO_PLOTS, "fold_{}".format(fold))
        os.makedirs(path_to_plots)

        plot_set_attention(network, DataLoader(train_dataset, batch_size=BATCH_SIZE), "TRAINING", fold, path_to_plots)
        plot_set_attention(network, DataLoader(val_dataset, batch_size=BATCH_SIZE), "VALIDATION", fold, path_to_plots)
        plot_set_attention(network, DataLoader(test_dataset, batch_size=BATCH_SIZE), "TEST", fold, path_to_plots)

        dataset = DataLoader(splitter.load_full_dataset(fold), batch_size=BATCH_SIZE)
        plot_dataset_attention(network, dataset, train_size, fold, path_to_plots)


if __name__ == '__main__':
    main()
