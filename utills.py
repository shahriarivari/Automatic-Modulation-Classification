import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm

def plot_constellation(dataset, mod, snr, num128pts=1, offset=0, scale=False):
  modtype = dataset[(mod, snr)]
  points_i = []
  points_q = []
  points = [points_i, points_q]
  for i in range(num128pts):
    for j in range(128):
      points[0].append(modtype[i + offset][0][j])
      points[1].append(modtype[i + offset][1][j])

  marksz = max(0.01, 3/num128pts)
  plt.plot(points[0], points[1], '.', markersize=marksz)
  plt.title(str(mod)[1:] + str(snr))
  plt.tight_layout()
  if (scale):
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.02, 0.02)
    plt.gca().set_aspect('equal', adjustable='box')
  return points


def plot_avg_all_snr_constellation(dataset, mod, snrs, num128pts=1, offset=0, scale=False):
  # Create a colormap to assign different colors to different SNRs
  colormap = cm.get_cmap('viridis', len(snrs))
  
  plt.figure(figsize=(8, 8))
  
  for idx, snr in enumerate(snrs):
    modtype = dataset[(mod, snr)]
    points_i = []
    points_q = []
    
    for i in range(num128pts):
      for j in range(128):
        points_i.append(modtype[i + offset][0][j])
        points_q.append(modtype[i + offset][1][j])
    
    avg_i = np.mean(points_i)
    avg_q = np.mean(points_q)
    
    marksz = max(0.01, 3/num128pts)
    plt.plot(avg_i, avg_q, '.', color=colormap(idx), markersize=marksz, label=f'SNR {snr} dB')
  
  plt.title(f'{mod} Constellation Diagram')
  plt.xlabel('In-phase (I)')
  plt.ylabel('Quadrature (Q)')
  plt.tight_layout()
  
  if scale:
    plt.xlim(-0.02, 0.02)
    plt.ylim(-0.02, 0.02)
    plt.gca().set_aspect('equal', adjustable='box')
  
  plt.legend()
  plt.grid(True)

def set_seed(seed=42):
  # Set the seed for the random number generator in Python
  random.seed(seed)
  
  # Set the seed for the random number generator in NumPy
  np.random.seed(seed)
  
  # Set the seed for the random number generator in PyTorch (both CPU and CUDA)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_label_encoder(label_encoder, dir_path, file_name):
  """
  Save a LabelEncoder to a file, creating the directory if it does not exist.

  Parameters:
  label_encoder (LabelEncoder): The LabelEncoder object to be saved.
  dir_path (str): The directory where the LabelEncoder should be saved.
  file_name (str): The name of the file to save the LabelEncoder.
  """
  # Check and create the directory if it does not exist
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  # Full file path
  file_path = os.path.join(dir_path, file_name)
  
  # Save the LabelEncoder to the file
  with open(file_path, 'wb') as f:
    pickle.dump(label_encoder, f)
  
  print(f"LabelEncoder saved to {file_path}")

  
def load_label_encoder(file_path):
  """
  Load a LabelEncoder from a file.

  Parameters:
  file_path (str): The path to the file containing the saved LabelEncoder.

  Returns:
  LabelEncoder: The loaded LabelEncoder object.
  """
  with open(file_path, 'rb') as f:
      label_encoder = pickle.load(f)
  print(f"LabelEncoder loaded from {file_path}")
  return label_encoder


def save_model(model, dir_path, file_name):
  """
  Save a PyTorch model to a file, creating the directory if it does not exist.

  Parameters:
  model (torch.nn.Module): The PyTorch model to be saved.
  dir_path (str): The directory where the model should be saved.
  file_name (str): The name of the file to save the model.
  """
  # Check and create the directory if it does not exist
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  # Full file path
  file_path = os.path.join(dir_path, file_name)
  
  # Save the model state dictionary to the file
  torch.save(model.state_dict(), file_path)
  
  print(f"Model saved to {file_path}")


def load_model(model, file_path):
  """
  Load a PyTorch model from a file.

  Parameters:
  model (torch.nn.Module): The PyTorch model instance to load the state dictionary into.
  file_path (str): The path to the file containing the saved model state dictionary.

  Returns:
  torch.nn.Module: The model loaded with the state dictionary.
  """
  # Load the model state dictionary from the file
  model.load_state_dict(torch.load(file_path))
  print(f"Model loaded from {file_path}")

  return model

def save_metrics(metrics, dir_path, file_name):
  """
  Save performance metrics to a file, creating the directory if it does not exist.

  Parameters:
  metrics (dict): The dictionary containing performance metrics.
  dir_path (str): The directory where the metrics should be saved.
  file_name (str): The name of the file to save the metrics.
  """
  # Check and create the directory if it does not exist
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  
  # Full file path
  file_path = os.path.join(dir_path, file_name)
  
  # Save the metrics to the file
  with open(file_path, 'wb') as f:
    pickle.dump(metrics, f)
  
  print(f"Metrics saved to {file_path}")


def load_metrics(file_path):
  """
  Load performance metrics from a file.

  Parameters:
  file_path (str): The path to the file containing the saved metrics.

  Returns:
  dict: The loaded metrics dictionary.
  """
  with open(file_path, 'rb') as f:
    metrics = pickle.load(f)

  print(f"Metrics loaded from {file_path}")

  return metrics


def count_parameters(model):
  """
  Print the number of parameters in each layer and the total number of parameters in the model.

  Parameters:
  model (torch.nn.Module): The PyTorch model.
  """
  total_params = 0
  print(f"{'Layer':<30} {'Parameters':<20}")
  print("="*50)
  
  for name, param in model.named_parameters():
      if param.requires_grad:
          num_params = param.numel()
          total_params += num_params
          print(f"{name:<30} {num_params:<20}")
  
  print("="*50)
  print(f"Total Parameters: {total_params}")