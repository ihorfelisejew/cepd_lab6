"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int = NUM_WORKERS,
    subset_fraction: float = 1.0,
    random_seed: int = 42
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.
  Supports reproducible and stratified subset selection.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.
    subset_fraction: Fraction of the dataset to load (0.0 < subset_fraction <= 1.0).
    random_seed: Seed for reproducible deterministic sample selection.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
  """
  full_train_data = datasets.ImageFolder(train_dir, transform=transform)
  full_test_data = datasets.ImageFolder(test_dir, transform=transform)

  class_names = full_train_data.classes

  if not (0.0 < subset_fraction <= 1.0):
      raise ValueError(f"subset_fraction must be greater than 0 and less than or equal to 1, got {subset_fraction}")

  if subset_fraction < 1.0:
      train_indices, _ = train_test_split(
          range(len(full_train_data)),
          train_size=subset_fraction,
          stratify=full_train_data.targets,
          random_state=random_seed
      )
      train_data = Subset(full_train_data, train_indices)

      test_indices, _ = train_test_split(
          range(len(full_test_data)),
          train_size=subset_fraction,
          stratify=full_test_data.targets,
          random_state=random_seed
      )
      test_data = Subset(full_test_data, test_indices)
  else:
      train_data = full_train_data
      test_data = full_test_data

  print(f"Original train size: {len(full_train_data)} -> Subset train size: {len(train_data)}")
  print(f"Original test size: {len(full_test_data)} -> Subset test size: {len(test_data)}")

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names