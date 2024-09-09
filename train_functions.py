import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def accuracy(output_logits, true_labels) -> float:
  """
  This function gets the raw logits from model's output and
  comapres them with the true labels.
  Args:
    output_logits: which are torch.Tensors outputed from the model
    true_labels: the ground truth labels
  Retruns:
    The accuracy
  """

  if output_logits.shape[1] == 1: # incase we have a bin-array classification

    # Convert logits to predicted labels (0 or 1)
    predicted_indices = (output_logits > 0).float()
    # Ensure true_labels are also floats for comparison
    true_indices = true_labels.float()

  else: # multi-class classification
    # Convert logits to predicted class indices
    predicted_indices = torch.argmax(output_logits, dim=1)

  # Compute the number of correct predictions
  correct_predictions = (predicted_indices == true_labels).sum().item()
  
  # Compute accuracy
  accuracy = correct_predictions / len(output_logits)
  
  return accuracy

# Function to train and evaluate the model
def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device):
  """
  This function performs a training step for a single epoch.
  Turns a pytoch model into training mode and then runs it 
  through all of the required training steps.

  Args:
    model: a Pytorch model
    dataloader: A DataLoader instance to train the model on.
    criterion: A pytorch loss function to minimize.
    optimizer: A pytorch optimizer to help minimize the loss function.
    device: a target device to compute on. e.g.("cpu" or "cuda")

  Returns:
    The Training loss and accuracy
  """
  # put the model into traing mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # loop throught the batches of the DataLoader and train
  for batch, (X,y) in enumerate(dataloader):

    # send the data to target device
    X, y = X.to(device) , y.to(device)
    
    # forward pass through the model
    y_pred = model(X)
    
    # calculate and accumulate the loss
    loss = criterion(y_pred,y)
    train_loss += loss.item()

    # set optimizer zero grad
    optimizer.zero_grad()

    # loss backwards
    loss.backward()

    # optimizer step
    optimizer.step()

    # Calculate and accumulate accuracy metric across all batches
    train_acc += accuracy(y_pred , y)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def batch_train(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               accumulation_steps: int = 1):
  """
  This function performs a training step for a single epoch with gradient accumulation.
  Turns a pytorch model into training mode and then runs it 
  through all of the required training steps.

  Args:
      model: a Pytorch model
      dataloader: A DataLoader instance to train the model on.
      criterion: A pytorch loss function to minimize.
      optimizer: A pytorch optimizer to help minimize the loss function.
      device: a target device to compute on. e.g.("cpu" or "cuda")
      accumulation_steps: Number of steps to accumulate gradients before performing an optimizer step.

  Returns:
      The Training loss and accuracy
  """
  # put the model into training mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # loop through the batches of the DataLoader and train
  for batch, (X, y) in enumerate(dataloader):
    # send the data to target device
    X, y = X.to(device), y.to(device)
    
    # forward pass through the model
    y_pred = model(X)
    
    # calculate and accumulate the loss
    loss = criterion(y_pred, y)
    train_loss += loss.item()

    # loss backwards
    loss.backward()

    # Perform optimizer step and zero grad every `accumulation_steps`
    if (batch + 1) % accumulation_steps == 0:
        # optimizer step
        optimizer.step()
        # set optimizer zero grad
        optimizer.zero_grad()

    # Calculate and accumulate accuracy metric across all batches
    train_acc += accuracy(y_pred, y)

  # If we have leftover gradients, perform a step
  if (batch + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              criterion: torch.nn.Module,
              device: torch.device):
  """
  Tests a PyTorch model for a single epoch.
  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    criterion: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    The test loss and test accuracy
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
    
  # Turn on inference context manager
  with torch.inference_mode():

    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):

      # Send data to target device
      X, y = X.to(device), y.to(device)

      # Forward pass
      test_pred_logits = model(X)

      # Calculate and accumulate loss
      loss = criterion(test_pred_logits, y)
      test_loss += loss.item()
      # Calculate and accumulate accuracy
      test_acc += accuracy(test_pred_logits , y)

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

  return test_loss,  test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: torch.device,
          batched_acc: bool = False,
          accumulation_steps: int = 1) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through batch_train() or train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_datalaoder: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    criterion: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    batched_acc: A boolean to indicate if gradient accumulation should be used.
    accumulation_steps: Number of steps to accumulate gradients before performing an optimizer step.

  Returns:
    A dictionary of training and testing loss.
    Each metric has a value in a list for each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]}
  """
  # Create empty results dictionary
  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      if batched_acc:
          train_loss, train_acc = batch_train(model=model,
                                              dataloader=train_dataloader,
                                              criterion=criterion,
                                              optimizer=optimizer,
                                              device=device,
                                              accumulation_steps=accumulation_steps)
      else:
          train_loss, train_acc = train_step(model=model,
                                             dataloader=train_dataloader,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             device=device)
      
      test_loss, test_acc = test_step(model=model,
                                      dataloader=test_dataloader,
                                      criterion=criterion,
                                      device=device)

      print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results


def plot_loss(loss_results:dict, save_path:str= None):
  """Takes a Dict of results produced by the train function
  and plots the loss
  Args:
    loss_results: its a dict with two key values 
    train_loss and test_loss
  Returns:
    plots the models train and test loss with respect to the epochs
  """
    # Get the number of epochs or data points
  epochs = len(loss_results['train_loss'])

  # Plotting the train loss
  plt.plot(range(1, epochs + 1), loss_results['train_loss'], label='Train Loss')

  # Plotting the test loss
  plt.plot(range(1, epochs + 1), loss_results['test_loss'], label='Test Loss')

  # Adding labels and title
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Train and Test Loss Over Epochs')

  # Adding a legend
  plt.legend()

  # Show the plot
  plt.show()
  if  save_path:
    plt.savefig(save_path, format='jpeg')


def plot_accuracy(loss_results:dict, save_path:str= None):
  """Takes a Dict of results produced by the train function
  and plots the accuracy
  Args:
    loss_results: its a dict with two key values 
    train_loss and test_loss
  Returns:
    plots the models train and test accuracy with respect to the epochs
  """
    # Get the number of epochs or data points
  epochs = len(loss_results['train_loss'])

  # Plotting the train loss
  plt.plot(range(1, epochs + 1), loss_results['train_acc'], label='Train Accuracy')

  # Plotting the test loss
  plt.plot(range(1, epochs + 1), loss_results['test_acc'], label='Test Accuracy')

  # Adding labels and title
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Train and Test Loss Over Epochs')

  # Adding a legend
  plt.legend()

  # Show the plot
  plt.show()
  if  save_path:
    plt.savefig(save_path, format='jpeg')
