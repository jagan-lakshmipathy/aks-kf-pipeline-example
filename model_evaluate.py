from __future__ import print_function

import os
import sys
import json
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from azure.storage.blob import BlobServiceClient


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def evaluate_model(model_artifact_path, device, test_loader, test_batch_size=1000):
    """
    Evaluate the model given the path to the artifact.
    Args:
        model_artifact_path (str): Path to the saved model artifact.
        test_batch_size (int): Batch size for testing.
    """

    # Attach model to the device.
    model = Net()

    # Load the trained model
    model.load_state_dict(torch.load(model_artifact_path)) 
    model = model.to(device)

    # Generating New Logs
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Attach tensors to the device.
            if device != 'cpu':
                data, target = data.to(device), target.to(device)

            output = model(data)
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = float(correct) / len(test_loader.dataset)

    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))


    return {
        "accuracy": accuracy,
    }



def parse_container_and_blob(url):
    """
    Parses the CONTAINER_NAME and BLOB_NAME from the given URL.

    Args:
        url (str): The URL in the format 
                   "https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{BLOB_NAME}"

    Returns:
        tuple: A tuple containing CONTAINER_NAME and BLOB_NAME.
    """
    pattern = r"https://[^/]+\.blob\.core\.windows\.net/([^/]+)/(.+)"
    match = re.match(pattern, url)
    if match:
        container_name = match.group(1)
        blob_name = match.group(2)
        return container_name, blob_name
    else:
        raise ValueError("Invalid URL format")
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        required=False,
        help="Name of the azure container to write the model",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        required=False,
        help="Name of the model (blob) to write",
    )
    parser.add_argument(
        "--model-info-path",
        type=str,
        required=False,
        help="Name of the model (blob) to write",
    )
    args = parser.parse_args()


    # Add code here to load the model from Azure Container Registry when model-path is empty

    # Replace with your Azure Storage account connection string and container details
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('ACR_CONNECTION_STRING', '')

    # we choose not to use these env variable as this info comes through the model-info file
    #CONTAINER_NAME = os.getenv('ACR_CONTAINER_NAME', '')
    #BLOB_NAME = os.getenv('ACR_BLOB_NAME', '') 


    if "--model-info-path" in sys.argv:
        # Read JSON from file
        with open(args.model_info_path, "r") as f:
            artifact_json = json.load(f)

        # Safely access values with default fallbacks
        BLOB_NAME = artifact_json.get("model_filename", "")
        CONTAINER_NAME = artifact_json.get("container_name", "")
        LOCAL_MODEL_FILEPATH = os.path.join(os.path.dirname(args.model_info_path), BLOB_NAME)


    #Container and blob names when provided in command line overrides what comes through the model-info file
    if "--container-name" in sys.argv:
        CONTAINER_NAME = args.container_name

    if "--model-filename" in sys.argv:
        BLOB_NAME = args.model_filename
     
    if "--model-info-path" not in sys.argv:
        LOCAL_MODEL_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), BLOB_NAME)


    # Check for missing or empty values
    if not CONTAINER_NAME or not BLOB_NAME or not LOCAL_MODEL_FILEPATH:
        print("Error: One or more required arguments are missing or empty.", file=sys.stderr)
        sys.exit(1)
        
    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # Download the model file from Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)
    
    with open(LOCAL_MODEL_FILEPATH, "wb") as model_file:
        model_file.write(blob_client.download_blob().readall())

    args.model_path = LOCAL_MODEL_FILEPATH

    print('args =>', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    test_ds = datasets.FashionMNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
    )
 
    # Pass the model artifact path to the evaluation function
    evaluation_results = evaluate_model(args.model_path, device, test_loader, args.batch_size)
    print("Evaluation Results:", evaluation_results)


if __name__ == "__main__":
    main()
