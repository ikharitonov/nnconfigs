#!/usr/bin/env python3
import time
import os
import pandas as pd
from datetime import datetime

class Metrics:
    def __init__(self):
        self.per_iteration_training_accuracies = []
        self.per_iteration_training_losses = []
        self.per_iteration_testing_accuracies = []
    def init_iteration(self):
        self.per_epoch_training_accuracies = []
        self.per_epoch_training_losses = []
        self.per_epoch_testing_accuracies = []
    def init_epoch(self):
        self.epoch_loss = 0
        self.sum_of_per_batch_training_accuracies = 0
        self.batch_count = 0
        self.epoch_start_time = time.time()
    def batch_update(self, num_correct, num_total, loss):
        self.sum_of_per_batch_training_accuracies += (100*(num_correct/num_total))
        self.batch_count += 1
        self.epoch_loss += loss
    def epoch_update(self, epoch_testing_accuracy):
        self.per_epoch_training_accuracies.append(self.sum_of_per_batch_training_accuracies/self.batch_count)
        self.per_epoch_training_losses.append(self.epoch_loss/self.batch_count)
        self.per_epoch_testing_accuracies.append(epoch_testing_accuracy)
        self.epoch_end_time = time.time()
    def iteration_update(self):
        self.per_iteration_training_accuracies.append(self.per_epoch_training_accuracies)
        self.per_iteration_training_losses.append(self.per_epoch_training_losses)
        self.per_iteration_testing_accuracies.append(self.per_epoch_testing_accuracies)
    def get_epoch_runtime(self):
        return str((self.epoch_end_time - self.epoch_start_time) / 60) # in minutes
    def epoch_end_print(self):
        # print("Iteration:", len(self.per_iteration_training_accuracies), "| Epoch:", len(self.per_epoch_training_accuracies), "| Training Accuracy:", int(self.per_epoch_training_accuracies[-1]), "| Training Loss:", int(self.per_epoch_training_losses[-1]), "| Testing Accuracy:", int(self.per_epoch_testing_accuracies[-1]), "| Runtime (mins):", self.get_epoch_runtime())
        print("Iteration:", len(self.per_iteration_training_accuracies), "| Epoch:", len(self.per_epoch_training_accuracies), "| Training Accuracy:", self.per_epoch_training_accuracies[-1], "| Training Loss:", self.per_epoch_training_losses[-1], "| Testing Accuracy:", self.per_epoch_testing_accuracies[-1], "| Runtime (mins):", self.get_epoch_runtime())
        print("=== === ===")
    def save_metrics(self, c):
        file_path = c.weights_save_dir + c.parameter_config + c.slash + "metrics.csv"
        if os.path.isfile(file_path):
            f = open(file_path, "a")
            f.write(str(len(self.per_iteration_training_accuracies)) + ',' + str(len(self.per_epoch_training_accuracies)) + ',' + str(float(self.per_epoch_training_accuracies[-1])) + ',' + str(float(self.per_epoch_training_losses[-1])) + ',' + str(float(self.per_epoch_testing_accuracies[-1])) + ',' + self.get_epoch_runtime() + ',' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ',' + c.running_machine + "\n")
            f.close()
        else:
            f = open(file_path, "x")
            f.write(str(c.training_parameters) + "\n")
            f.write("Iteration,Epoch,Training Accuracy,Training Loss,Testing Accuracy,Runtime,Date and Time,Machine" + "\n")
            f.write(str(len(self.per_iteration_training_accuracies)) + ',' + str(len(self.per_epoch_training_accuracies)) + ',' + str(float(self.per_epoch_training_accuracies[-1])) + ',' + str(float(self.per_epoch_training_losses[-1])) + ',' + str(float(self.per_epoch_testing_accuracies[-1])) + ',' + self.get_epoch_runtime() + ',' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ',' + c.running_machine + "\n")
            f.close()
    def load_interrupted_iteration(self,c):
        file_path = c.weights_save_dir + c.parameter_config + c.slash + "metrics.csv"
        df = pd.read_csv(file_path, skiprows=1)
        self.per_epoch_training_accuracies = df["Training Accuracy"].iloc[0:].tolist()
        self.per_epoch_training_losses = df["Training Loss"].iloc[0:].tolist()
        self.per_epoch_testing_accuracies = df["Testing Accuracy"].iloc[0:].tolist()
    def save_lr_metrics(self, c, epoch, batch, lr, loss):
        file_path = c.weights_save_dir + c.parameter_config + c.slash + "lr_metrics.csv"
        if os.path.isfile(file_path):
            f = open(file_path, "a")
        else:
            f = open(file_path, "x")
            f.write("<<<logged per " + c.training_parameters["step_scheduler_per"] + ">>>" + "\n")
            f.write("epoch,batch,lr,loss,date and time,machine" + "\n")
        f.write(str(epoch) + ',' + str(batch) + ',' + str(float(lr)) + ',' + str(float(loss)) + ',' + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ',' + c.running_machine + "\n")
        f.close()