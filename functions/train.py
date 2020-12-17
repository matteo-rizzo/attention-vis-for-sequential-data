from typing import Callable

import numpy as np

from functions.evaluate import evaluate
from functions.metrics import compute_batch_accuracy, pprint_metrics
from params import *


def train(network: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          criterion: Callable,
          fold: int):
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    best_val_loss = 1000.0
    cv_val_metrics = None

    for epoch in range(EPOCHS):

        print("\n *** Epoch {}/{} *** \n".format(epoch + 1, EPOCHS))

        print("\n Training the model... \n")

        network = network.train()

        train_loss, train_accuracy = [], []
        running_loss, running_accuracy = 0.0, 0.0

        for i, (x, y) in enumerate(train_loader):

            # Zero out all the gradients
            optimizer.zero_grad()

            # Move training inputs and labels to device
            x = x.float().to(DEVICE)
            y = torch.from_numpy(np.asarray(y)).long().to(DEVICE)

            # Initialize the hidden state of the RNN and move it to device
            h = network.init_state(x.shape[0]).to(DEVICE)

            # Predict
            o = network(x, h).to(DEVICE)

            # Compute the error
            loss = criterion(o, y)

            # Backpropagate
            loss.backward()

            # Update model parameters
            optimizer.step()

            loss_value, batch_accuracy = loss.item(), compute_batch_accuracy(o, y)

            # Accumulate training loss and accuracy for the log
            running_loss += loss_value
            running_accuracy += batch_accuracy

            # Store all training loss and accuracy for computing avg
            train_loss += [loss_value]
            train_accuracy += [batch_accuracy]

            if not (i + 1) % 10:
                print("[ Epoch: {}/{} - batch: {}/{} ] [ loss: {:.5f} | accuracy: {:.5f} ]"
                      .format(epoch + 1, EPOCHS, i + 1, len(train_loader), running_loss / 10, running_accuracy / 10))
                running_loss, running_accuracy = 0.0, 0.0

        avg_train_loss, avg_train_accuracy = np.mean(train_loss), np.mean(train_accuracy)

        print("\n ........................................................... \n")
        print("[ Avg train loss: {:.4f} - Avg train accuracy: {:.4f} ]".format(avg_train_loss, avg_train_accuracy))
        print("\n ........................................................... \n")

        # --- VALIDATION ---

        print("\n Validating the model... \n")

        val_metrics = evaluate(network, val_loader, criterion)

        # Pretty print the validation metrics
        print("\n Validation metrics for epoch {}/{}: \n".format(epoch + 1, EPOCHS))
        pprint_metrics(val_metrics)

        # Update best model

        avg_val_loss = val_metrics["loss"]

        if avg_val_loss < best_val_loss:
            print("\n Avg val loss ({:.4f}) better that current best val loss ({:.4f}) \n"
                  .format(avg_val_loss, best_val_loss))
            print("\n --> Saving new best model... \n")
            torch.save(network.state_dict(), os.path.join(PATH_TO_LOG, "best_model_fold_{}.pth".format(fold)))
            best_val_loss = val_metrics["loss"]
            cv_val_metrics = val_metrics

    return cv_val_metrics
