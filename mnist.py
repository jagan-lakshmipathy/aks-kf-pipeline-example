from __future__ import print_function

import argparse
import os
import sys
import json

import torch
#import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DistributedSampler
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


def train(args, model, device, train_loader, epoch):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Attach tensors to the device.
        if device != 'cpu':
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f} \n".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item()
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            #writer.add_scalar("loss", loss.item(), niter)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
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
        "--output-path",
        type=str,
        required=True,
        help="Path to save the trained model",
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
    args = parser.parse_args()

    print('args =>', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Attach model to the device.
    if use_cuda:
        model = Net().to(device)
    else:
        model = Net()

 
    # Get FashionMNIST train and test dataset.
    train_ds = datasets.FashionMNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_ds = datasets.FashionMNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Add train and test loaders.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )


    # Replace with your Azure Storage account connection string and container details
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('ACR_CONNECTION_STRING', '')
    CONTAINER_NAME = os.getenv('ACR_CONTAINER_NAME', '')
    BLOB_NAME = os.getenv('ACR_BLOB_NAME', '')

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch)

    if args.save_model:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        local_model_path = os.path.join(os.path.dirname(args.output_path), BLOB_NAME)

        torch.save(model.state_dict(), local_model_path)
        print(f"Model saved to: {local_model_path}")

    if "--container-name" in sys.argv:
        CONATINER_NAME = args.container_name

    if "--model-filename" in sys.argv:
        BLOB_NAME = args.model_filename

    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    # Get a client for the container
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Ensure the container exists
    if not container_client.exists():
        container_client.create_container()

    # Upload the model file to the container
    with open(local_model_path, "rb") as data:
        container_client.upload_blob(name=BLOB_NAME, data=data, overwrite=True)

    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{BLOB_NAME}"
    print(f"Blob URL: {blob_url}")  

    # with open(args.output_path, "w") as f:
    #     f.write(blob_url)  # <--- THIS is the "artifact" content Kubeflow will track

    # Write JSON content to output path for Kubeflow tracking
    artifact_json = {
    "model_filename": BLOB_NAME,
    "container_name": CONTAINER_NAME
    }

    with open(args.output_path, "w") as f:
        json.dump(artifact_json, f)

    print(f"Model info written to: {args.output_path}")

    # params = model.named_parameters()
    # d = dict(params)
    #print('d: ', d)


if __name__ == "__main__":
    main()
