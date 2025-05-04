import os
import csv
import json
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

##### For Training #####
def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def write_config_log(logfile_path, purpose, model_type, epochs, batch_size, lr):
    ''' write experiment log file for config to ./experiment/{exp_name}/log/config_log.txt '''
    with open(logfile_path, 'w') as f:
        f.write(f'Model Type      = {purpose}\n')
        f.write(f'Model Type      = {model_type}\n')
        f.write(f'Num epochs      = {epochs}\n')
        f.write(f'Batch size      = {batch_size}\n')
        f.write(f'Learning rate   = {lr}\n')
       

def write_result_log(logfile_path, epoch, epoch_num, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better):
    ''' write experiment log file for result of each epoch to ./experiment/{exp_name}/log/result_log.txt '''
    with open(logfile_path, 'a') as f:
        f.write(f'[{epoch + 1}/{epoch_num}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
        if is_better:
            f.write(' -> val best (acc)')
        f.write('\n')

##### For Inference #####
def write_csv(output_path, predictions, test_loader):
    ''' write csv file of filenames and predicted labels '''
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        for i, label in enumerate(predictions):
            filename = test_loader.dataset.image_names[i]
            writer.writerow([filename, str(label)])

##### For Evaluation #####
def read_csv(filepath):
    ''' read csv file return filenames list and labels list '''
    with open(filepath, 'r', newline='') as f:
        data = csv.reader(f)
        header = next(data)
        data = list(data)
    filenames = [x[0] for x in data]
    labels = [int(x[1]) for x in data]
    return filenames, labels

