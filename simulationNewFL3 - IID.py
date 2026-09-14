# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:14:52 2026

@author: vikas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:00:29 2026
@author: vikas
Updated: added timing and resource monitoring
"""

import os
import time
import threading          # NEW: for resource monitor
from multiprocessing import Process
from typing import Tuple
import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import shutil
import itertools
import cv2
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from keras.callbacks import CSVLogger

# NEW: resource monitoring packages
import psutil
try:
    import GPUtil
    GPUs = GPUtil.getGPUs()
except ImportError:
    GPUs = []

import warnings
warnings.filterwarnings("ignore")
print('modules loaded')

from flwr.server.strategy import FedAvg as FA
import dataset as dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"

DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


# -------------------------------------------------------------------
# NEW: Custom callback to measure per‑epoch time
# -------------------------------------------------------------------
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start)


# -------------------------------------------------------------------
# NEW: Resource monitor (runs in a background thread)
# -------------------------------------------------------------------
class ResourceMonitor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.peak_ram = 0.0          # GB
        self.peak_gpu_mem = 0.0      # MB
        self.peak_gpu_util = 0.0     # %

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _monitor(self):
        while self.running:
            # RAM usage (GB)
            mem = psutil.virtual_memory()
            used_ram = mem.used / (1024**3)
            if used_ram > self.peak_ram:
                self.peak_ram = used_ram

            # GPU usage (if available)
            if GPUs:
                for gpu in GPUs:
                    if gpu.memoryUsed > self.peak_gpu_mem:
                        self.peak_gpu_mem = gpu.memoryUsed
                    util = gpu.load * 100
                    if util > self.peak_gpu_util:
                        self.peak_gpu_util = util
            time.sleep(0.5)   # sample every half second


# -------------------------------------------------------------------
# Model creation (unchanged)
# -------------------------------------------------------------------
def create_model():
    """Create a simple feedforward neural network."""
    model = keras.Sequential()
    model.add(Dense(128, input_shape=(4, 1), activation="relu", name="Hidden_Layer_1"))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', 'Precision', 'Recall'])
    model.summary()
    return model


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    print("number clients ", num_clients)
    model = create_model()
    weights = model.get_weights()
    parameters = fl.common.ndarrays_to_parameters(weights)
    strategy = FA(min_available_clients=num_clients,
                  min_fit_clients=num_clients,
                  fraction_fit=fraction_fit,
                  initial_parameters=parameters)
    fl.server.start_server(server_address="127.0.0.1:8080",
                           config=fl.server.ServerConfig(num_rounds=num_rounds),
                           strategy=strategy)


def start_client(dataset: DATASET, fcntr) -> None:
    """Start a single client with the provided dataset."""
    model = create_model()
    csv_logger = CSVLogger(f"{fcntr}_IR.csv", append=True)
    csv_logger1 = CSVLogger(f"{fcntr}_Eval_IR.csv", append=True)

    (x_train, y_train), (x_test, y_test) = dataset

    # NEW: prepare timing CSV for this client
    timing_csv = f"client_{fcntr}_timing.csv"
    if not os.path.exists(timing_csv):
        with open(timing_csv, "w") as f:
            f.write("client_id,round,epoch,epoch_time,total_fit_time,"
                    "peak_ram_gb,peak_gpu_mem_mb,peak_gpu_util_pct\n")

    class CifarClient(fl.client.NumPyClient):
        def __init__(self, client_id):
            self.client_id = client_id
            self.model = model

        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)

            # NEW: get round number
            round_num = config.get("server_round", 0)

            # NEW: timing and resource monitoring
            fit_start = time.time()
            time_callback = TimeHistory()
            monitor = ResourceMonitor()
            monitor.start()

            # Train the model
            model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=10,
                      callbacks=[csv_logger, time_callback])

            # NEW: stop monitoring and collect peaks
            monitor.stop()
            fit_end = time.time()
            total_fit_time = fit_end - fit_start
            peak_ram = monitor.peak_ram
            peak_gpu_mem = monitor.peak_gpu_mem
            peak_gpu_util = monitor.peak_gpu_util

            # NEW: write per-epoch times and resource peaks to CSV
            with open(timing_csv, "a") as f:
                for epoch_idx, epoch_time in enumerate(time_callback.epoch_times):
                    f.write(f"{self.client_id},{round_num},{epoch_idx+1},"
                            f"{epoch_time:.4f},{total_fit_time:.4f},"
                            f"{peak_ram:.2f},{peak_gpu_mem:.2f},{peak_gpu_util:.2f}\n")

            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            """Evaluate the model, generate plots, and compute class-wise metrics."""
            model.set_weights(parameters)

            round_num = config.get("server_round", 0)

            # Overall metrics
            loss, accuracy, recall, precision = model.evaluate(x_test, y_test,
                                                               callbacks=[csv_logger1])

            # Predictions and true labels
            y_pred_proba = model.predict(x_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Save overall results
            with open(f"EvalRes{self.client_id}.csv", "a+") as fl:
                fl.write(f"{self.client_id},{loss},{accuracy},{recall},{precision}\n")

            # Confusion matrix & heatmap
            cm = confusion_matrix(y_true, y_pred)
            plot_dir = f"client_{self.client_id}"
            os.makedirs(plot_dir, exist_ok=True)
            lab = os.listdir('FData\\Lung Disease Dataset')  # assumes directory exists
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=lab, yticklabels=lab, cbar=False)
            plt.title(f"Confusion Matrix - Client {self.client_id}")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "confusion_heatmap.png"))
            plt.close()

            # ROC curves (one per class)
            num_classes = y_test.shape[1]   # 5 classes
            fpr = {}
            tpr = {}
            roc_auc = {}
            plt.figure(figsize=(3, 3))
            for i in range(num_classes):
                true_bin = (y_true == i).astype(int)
                prob_i = y_pred_proba[:, i]
                fpr[i], tpr[i], _ = roc_curve(true_bin, prob_i)
                roc_auc[i] = auc(fpr[i], tpr[i])
                plt.plot(fpr[i], tpr[i], lw=2,
                         label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - Client {self.client_id}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "roc_curves.png"))
            plt.close()

            # Per-class metrics
            class_metrics = {
                'Class': [],
                'Precision': [],
                'Recall (Sensitivity)': [],
                'Specificity': [],
                'F1-score': []
            }

            for i in range(num_classes):
                TP = cm[i, i]
                FP = cm[:, i].sum() - TP
                FN = cm[i, :].sum() - TP
                TN = cm.sum() - (TP + FP + FN)

                precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                specificity_i = TN / (TN + FP) if (TN + FP) > 0 else 0.0
                f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0

                class_metrics['Class'].append(i)
                class_metrics['Precision'].append(precision_i)
                class_metrics['Recall (Sensitivity)'].append(recall_i)
                class_metrics['Specificity'].append(specificity_i)
                class_metrics['F1-score'].append(f1_i)

                print(f"Class {i}: Precision={precision_i:.4f}, Recall={recall_i:.4f}, "
                      f"Specificity={specificity_i:.4f}, F1={f1_i:.4f}")

            # Save per-class metrics to CSV
            df_metrics = pd.DataFrame(class_metrics)
            metrics_csv = f"ClassWiseMetrics_client{self.client_id}.csv"
            if not os.path.isfile(metrics_csv):
                df_metrics.to_csv(metrics_csv, mode='a', index=False, header=True)
            else:
                df_metrics.to_csv(metrics_csv, mode='a', index=False, header=False)

            # Save confusion matrix to CSV (original behaviour)
            df_cm = pd.DataFrame(cm)
            df_cm.to_csv(f"Confusion_IID{self.client_id}.csv", mode='a', header=False)

            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=CifarClient(client_id=fcntr))


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Run the entire simulation and record overall time."""
    overall_start = time.time()   # NEW

    processes = []
    server_process = Process(target=start_server, args=(num_rounds, num_clients, fraction_fit))
    server_process.start()
    processes.append(server_process)

    time.sleep(2)

    partitions = dataset.load(num_partitions=num_clients)
    fcntr = 0

    for partition in partitions:
        fcntr += 1
        client_process = Process(target=start_client, args=(partition, fcntr))
        client_process.start()
        processes.append(client_process)

    for p in processes:
        print("Process:", p)
        p.join()

    overall_end = time.time()   # NEW
    total_simulation_time = overall_end - overall_start
    print(f"\nTotal simulation time: {total_simulation_time:.2f} seconds")

    # NEW: write overall time to file
    with open("overall_time.csv", "w") as f:
        f.write("total_time_seconds\n")
        f.write(f"{total_simulation_time:.2f}\n")


if __name__ == "__main__":
    run_simulation(num_rounds=10, num_clients=4, fraction_fit=1)