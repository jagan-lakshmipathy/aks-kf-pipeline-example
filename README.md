# A simple Kubeflow Pipeline example
###### Jagan Lakshmipathy 
###### 05/08/2025

### 1. Introduction

In our earlier [work](https://github.com/jagan-lakshmipathy/aks-kf-top-distrib-training), we demonstrated a step-by-step process for distributing machine learning model training using the [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/).
In this repository, we present a simple example of a Kubeflow pipeline. The pipeline will be deployed and tested on Azure Kubernetes Service (AKS), using a combination of Azure CLI and kubectl commands to manage the cluster from the console.

Please note that the steps outlined here are not entirely cloud-agnostic, as they include Azure-specific commands. However, the general workflow can be adapted to other cloud providers with minimal changes—primarily replacing the Azure CLI steps.

We will also create a GPU-enabled node pool to execute the pipeline on GPUs. While a GPU is not required to run this example, using one allows us to easily scale the pipeline later for more compute-intensive model training or execution.

### 2. Prerequesites

You don’t need to review our earlier work to follow this example, but we do assume that you have a solid understanding of Microsoft Azure. If you're new to Azure, you can get started [here](https://azure.microsoft.com/en-us/get-started).

Ensure that you have the Azure CLI installed. If not, follow the instructions in this [guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

We also recommend reviewing the basics of [Azure Kubernetes Service (AKS)](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli) If you're planning to request vCPU quotas (necessary for certain VM types), refer to the same AKS guide for quota request instructions.

To better understand compute options in Azure, including GPU-based instances, see the [Azure VM sizes documentation](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist).

In this example, we use two Azure VM types:

* Standard_D4ds_v5: for Kubernetes system workloads.

* Standard_NC40ads_H100_v5: for GPU-based workloads.

The steps required to request quotas and configure other GPU-enabled VM types are similar.

We will run a simple machine learning example on the GPU node pool. A general familiarity with Azure and Kubernetes is assumed. If you're new to Kubernetes, start with the official [Kubernetes documentation](https://kubernetes.io/docs/home/).

You should also have a working knowledge of Git and GitHub. Clone this repository to your local machine:

```
    bash>   git clone https://github.com/jagan-lakshmipathy/aks-kf-pipeline-example.git
```
Make sure you have kubectl (Kubernetes CLI) installed. Follow the instructions here[here](https://kubernetes.io/docs/tasks/tools/) to install it.

This guide uses macOS and Bash to run Azure CLI and kubectl commands, but you can follow along on any operating system and shell of your choice.

### 3. What's in this Repository?

This repository provides an end-to-end example of building and deploying a simple Kubeflow pipeline using a Docker image based on the[NVIDIA PyTorch 24.07 base iamge](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html).

It includes all the necessary components and templates required to build container images, define the pipeline, and deploy it to an AKS cluster. The main deployment logic is encapsulated in the kf-pipeline-deploy.sh script, which ties everything together.
Key components include:

Key Files:

1. kf-pipeline-deploy.sh – A Zsh script that automates the end-to-end deployment of the pipeline.

2. template_mnist_pipeline2.py – A pipeline template script where the Azure Container Registry (ACR) URL is injected to generate the actual mnist_pipeline2.py.

3. Dockerfile.mnist-trainer.template – Template Dockerfile used to build the trainer component.

4. Dockerfile.mnist-evaluator.template – Template Dockerfile used to build the evaluator component.

5. Dockerfile.pipeline – Dockerfile used to build the main pipeline image.

6. mnist.py – Python script for the training component.

7. model_evaluate.py – Python script for the model evaluation component.

8. nvidia-device-plugin-ds.yaml – Kubernetes manifest to enable NVIDIA GPU support on cluster nodes.

9. template-pipeline-deployment.yaml – A deployment template used to generate pipeline-deployment.yaml, which is applied to deploy the pipeline.

Together, these files form a functional baseline for running PyTorch-based machine learning workloads on Kubeflow, and can be extended for more complex workflows.

In the next steps, we’ll walk through the kf-pipeline-deploy.sh script in detail to understand how it orchestrates the deployment process.

### 4. Authenticate Your Console
We assume that your Kubernetes cluster on AKS is already up and running. To interact with it from your local console, you'll need to complete the following two steps:

#### 1. Verify the AKS Cluster
Log in to your [Azure Portal](portal.azure.com) and ensure that your AKS cluster is active. Alternatively, you can verify this from your terminal using kubectl—but only after completing Step 2 to configure your local access.

#### 2. Merge AKS Credentials with Local Kubeconfig
To issue kubectl commands against your AKS cluster from your local machine, you need to merge the cluster credentials into your local Kubernetes configuration file (typically located at ~/.kube/config on macOS).

Run the following Azure CLI command to do so (see line 36 of kf-pipeline-deploy.sh):
```
    bash> az aks get-credentials --resource-group <resource-group-name> --name <aks-cluster-name>
```
You can now verify connectivity by listing the running pods:
```
    bash> kubectl get pods --watch
```
This will stream the status of pods in the default namespace.

### 5. Install Azure CLI Extensions
Before proceeding with the deployment, you’ll need to install a specific version of the Azure CLI aks-preview extension. We observed issues with the latest version, so we recommend reverting to a stable version (14.0.0b3) that has been verified to work with this setup. You can refer to lines 24–31 in the kf-pipeline-deploy.sh script for context.

Run the following commands in your terminal:
```
    bash>   az extension remove --name aks-preview
    bash>   az extension add --name aks-preview --version 14.0.0b3
    bash>   az extension show --name aks-preview
```
These commands will ensure you're using the expected version of the extension for compatibility with AKS features used in this project.
### 6. Add GPU nodepool to AKS Cluster

Although this example doesn't require a GPU, we provision a GPU-enabled node pool to support future extensions involving more complex or compute-intensive workloads.

In this step, we add a new GPU node pool with 3 nodes. You can choose any GPU-enabled VM size from the [Azure offerings](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist#gpu-accelerated) based on your available quota. For example, we tested with the following SKUs:

Standard_NC24s_v3 from the NCv3-series

Standard_NC40ads_H100_v5 from the NCads H100 v5-series

Below is the command used to create a node pool of up to 3 nodes, each with 40 vCPUs and an H100 GPU. You can modify the --node-vm-size, --min-count, and --max-count values as needed for your workload (see lines 54 through 56 in the kf-pipeline-deploy.sh):
```
    bash> az aks nodepool add \
            --resource-group $RG_NAME \
            --cluster-name $CLSTR_NAME \
            --name gpunp \
            --node-count 1 \
            --node-vm-size Standard_NC6s_v3 \
            --node-taints sku=gpu:NoSchedule \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 3

```
This command also applies a taint to the nodes (sku=gpu:NoSchedule) to restrict scheduling only to pods that tolerate this taint—ensuring GPU nodes are used only when explicitly required.

### 7. Install Kubeflow Pipelines on AKS
We will now install Kubeflow Pipelines (version 2.3.0) on our AKS cluster (see lines 88 through 92 for context). This setup uses the platform-agnostic configuration recommended by the Kubeflow community to ensure compatibility with AKS (as opposed to GKE-specific configurations).

Run the following commands to install:
```
    bash> export PIPELINE_VERSION=2.3.0

    bash> kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"

    bash> kubectl wait crd/applications.app.k8s.io --for=condition=established --timeout=60s

    bash> kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

```

Note: The third command replaces the standard GKE-specific installation step with a platform-agnostic alternative, as discussed here.

Please be patient—initialization may take several minutes. Some pods may briefly fail during startup but should self-heal and stabilize on their own.

### 8. Create an Azure Container Registry (ACR)
To store the container image for the simple pipeline defined in component_with_optional_inputs.py, you’ll need to create an Azure Container Registry (ACR). Use the following command (see lines 93 through 112 in the kf-pipeline-deploy.sh):

```
    bash> az acr create --name <name-of-acr> --resource-group <resource-group-associated> --sku basic
```
Replace \<acr-name\> and \<resource-group-name\> with your desired ACR name and the resource group it's associated with.

### 8. Login to ACR
Before pushing any images to ACR, you need to log in (see lines 168 through 198):

```
    bash> az acr login --name <name-of-acr>
```
This step authenticates your local Docker client with the Azure Container Registry, allowing you to push and pull container images.

### 9.  Install NVIDIA Device Plugin
To enable GPU support within your AKS cluster, apply the NVIDIA device plugin. This plugin ensures that Kubernetes can detect and schedule workloads on GPU-enabled nodes.

(Refer to lines 200–202 in the deployment script.)

```
    bash> kubectl create namespace gpu-operator
    bash> kubectl apply -f nvidia-device-plugin-ds.yaml
```
### 10. Build, Tag, and Push Component Images

We now create and push Docker images for each component defined in the COMP_LIST array. This process is automated through a script that performs the following steps for each component (e.g., mnist-trainer, mnist-evaluator):

#### 10.1 Create Workload Image Locally
For each component:

* A Dockerfile is dynamically generated from a corresponding template file (e.g., Dockerfile.mnist-trainer.template), replacing placeholders like \<my-acr\> and \<model-name\> with actual environment variable values.

* The resulting Dockerfile (e.g., Dockerfile.mnist-trainer) is used to build a Docker image locally.

Additionally, this repository includes a special Dockerfile named Dockerfile.ce, used to build a GPU workload image. It pulls the NVIDIA PyTorch base image tagged 24.07-py3, which includes:
* Python 3.10
* CUDA Toolkit
* NCCL backend
* JupyterLab
* A prebuilt and installed version of PyTorch (located at /usr/local/lib/python3.10/dist-packages/torch)
* The full PyTorch source located in /opt/pytorch


The image also includes component_with_optional_inputs.py, a simple Kubeflow pipeline component. This script is copied into the image and executed at build time (line #15 in the Dockerfile). The pipeline itself is launched at line #45 using the Kubeflow Pipelines client.
```
    Refer to lines 255–258 of the deployment script for this context.
```
Build the Docker image using:
```
    docker build --platform="linux/amd64" -t "kubeflow/${comp_value}:1.0" .
```
Ensure you’re in the correct directory containing the Dockerfile and the Python script.

#### 10.2. Tag and push the image to ACR
After building the image, tag and push it to your Azure Container Registry so it can be pulled by your AKS cluster.
##### 1. Tag the image:
```
    docker tag pipline-example:1.0 <acr-name>.azurecr.io/pipeline-example:latest
```
##### 2. Push the image:
```
    docker push <acr-name>.azurecr.io/pipeline-example:latest
```
Replace \<acr-name\> with your actual ACR name.

This repeatable and automated approach ensures all pipeline components are built, tagged, and pushed consistently—simplifying deployment across environments.

### 11. Attach the ACR to the AKS Cluster
After pushing the Docker image to your Azure Container Registry (ACR), you need to grant your AKS cluster permission to pull images from it. This is done by attaching the ACR to your AKS cluster using the following Azure CLI command:

```
    bash> az aks update \
            --name <aks-cluster-name> \
            --resource-group <aks-rg-name> \
            --attach-acr <acr-name>

```
Replace \<aks-cluster-name\>, \<aks-rg-name\>, and \<acr-name\> with the appropriate values for your setup. Once attached, your Kubernetes workloads running in AKS can securely pull images from your ACR without additional authentication steps.

### 12. Build and Deploy MNIST Pipeline Image

This section automates the final steps required to build, tag, push, and prepare the deployment manifest for the MNIST pipeline in AKS.

#### 12.1 Generate the Python Pipeline Script
We begin by dynamically generating the mnist_pipeline2.py file from the template_mnist_pipeline2.py template. The script replaces the placeholder \<my-acr\> with the actual Azure Container Registry (ACR) name, ensuring the pipeline references the correct container registry.

#### 12.2 Build, Tag, and Push the Pipeline Image
A Docker image is built using Dockerfile.pipeline, tagged with the pipeline name, and then pushed to the specified ACR. This makes the image available for use by the Kubeflow Pipeline running in AKS. The process includes:

* Building the Docker image for the pipeline logic
* Tagging it with the latest version
* Pushing it to your ACR

Each step includes checks to ensure required values and files are set, improving reliability and preventing misconfigurations (See the lines between 284 in kf-pipeline-deploy.sh for details)

#### 12.3 Generate Deployment YAML from Template
Finally, a deployment manifest YAML file is generated from its corresponding template. Placeholders such as \<my-acr\> and \<pipeline-name\> are replaced using sed to match your environment. The resulting YAML file can be used to deploy the Kubeflow pipeline to the AKS cluster.

This end-to-end automation ensures consistent and repeatable builds and deployments for your Kubeflow pipeline components.


### 13. Deploy Job
We deploy the job as follows. Feel free to review the provided template-pipeline-deployment.yaml to understand manifest.

```
    kubectl apply -f pipeline-deployment.yaml
```
Issue the following command to the port forwarding of UI. Use your browser to check if the UI is running. 

```
    bash> kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

```
### 15. Job Monitoring Commands:
```
    kubectl get pods --watch
    kubectl get jobs --watch
    kubectl logs <pod-id>
    kkubectl describe pod <pod-id>
```
### 16. Tear-down
Once you are successfully done testing this code, make sure to cleanup the jobs in AKS. Finally, don't forget to tear-down the AKS cluster to avoid incurring unnecessary billing costs.

