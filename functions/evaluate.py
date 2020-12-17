import numpy as np

from functions.metrics import compute_batch_accuracy, compute_optimal_roc_threshold, compute_metrics
from params import *


def evaluate(network: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.optim) -> dict:
    network = network.eval()

    y_scores, y_true = [], []
    loss, accuracy = [], []
    running_loss, running_accuracy = 0.0, 0.0

    with torch.no_grad():

        for i, (x, y) in enumerate(dataloader):

            # Move validation inputs and labels to device
            x = x.float().to(DEVICE)
            y = torch.from_numpy(np.asarray(y)).long().to(DEVICE)

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x.shape[0]).to(DEVICE)

            # Predict
            o = network(x, h).to(DEVICE)

            loss_value, batch_accuracy = criterion(o, y).item(), compute_batch_accuracy(o, y)

            # Accumulate validation loss and accuracy for the log
            running_loss += loss_value
            running_accuracy += batch_accuracy

            # Store all validation loss and accuracy values for computing avg
            loss += [loss_value]
            accuracy += [batch_accuracy]

            # Store predicted scores and ground truth labels
            y_scores += torch.exp(o).cpu().numpy().tolist()
            y_true += y.cpu().numpy().tolist()

            if not (i + 1) % 10:
                print("[ batch: {}/{} ] [ loss: {:.5f} | accuracy: {:.5f} ]"
                      .format(i + 1, len(dataloader), running_loss / 10, running_accuracy / 10))
                running_loss, running_accuracy = 0.0, 0.0

    y_scores, y_true = np.array(y_scores).reshape((len(y_scores), 2)), np.array(y_true)

    # Compute predicted labels based on the optimal ROC threshold
    threshold = compute_optimal_roc_threshold(y_true, y_scores[:, 1])
    y_pred = np.array(y_scores[:, 1] >= threshold, dtype=np.int)

    # Compute the validation metrics
    avg_loss, avg_accuracy = np.mean(loss), np.mean(accuracy)
    metrics = compute_metrics(y_true, y_pred, y_scores[:, 1])
    metrics["loss"] = avg_loss
    metrics["accuracy"] = avg_accuracy

    return metrics
