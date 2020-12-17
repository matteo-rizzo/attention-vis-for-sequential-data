import os
import time

import torch

# --- Data ---

TARGET = ['Meara', 'BarChartLit'][0]
SEC = 1
DATASET_NAME = "{} ({} sec)".format(TARGET, SEC)
PATH_TO_DATASET = os.path.join("dataset", DATASET_NAME)
PATH_TO_SEQS = os.path.join(PATH_TO_DATASET, "sequences")
MAX_SEQ_LEN = 30
TRUNCATION_SIDE = ['head', 'tail'][0]

# -- Network --

INPUT_SIZE = 10
OUTPUT_SIZE = 2
HIDDEN_SIZE = 256
BIDIRECTIONAL = True
NUM_LAYERS = 1
DROPOUT = 0.0
RNN_TYPE = "GRU"

# --- Training ---

K = 10
NUM_REPETITIONS = 1
EPOCHS = 100
LEARNING_RATE = 0.00003
BATCH_SIZE = 128
PATH_TO_LOG = os.path.join("logs", "attention_rnn_" + str(time.time()))
PATH_TO_PRETRAINED = os.path.join("trained_models")

# --- Device ---

if not torch.cuda.is_available():
    print("WARNING: running on CPU since GPU is not available")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(0)
