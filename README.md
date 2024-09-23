# A simple Kubeflow Pipeline example
###### Jagan Lakshmipathy 
###### 09/22/2024

### 1. Introduction
In one of our earlier [work](https://github.com/jagan-lakshmipathy/aks-kf-top-distrib-training), we demonstrated a step-by-step process on how to distribute training of machine learning models using  [Kubeflow Traing Operator](https://www.kubeflow.org/docs/components/training/). Here will demonstrate a simple Kubeflow pipeline example. We will deploy and test it in AKS. So, we will use Azure CLI commands with kubectl commands to control the Azure Kubernetes Service (AKS) cluster from our console. So, the steps listed here is not completely cloud provider agnostic. We are going to assume that you are going to follow along using AKS. However, you can follow along with any of your preferred cloud provider for the most part with the exception of Azure CLI commands. We will create a GPU nodepool to run our pipeline in GPUs. While it is not necessary to leverage a GPU to run this example we use the GPU so that we can easily extend to run complicated compute intestive model training or executtion later.

### 2. Prerequesites
While we don't expect you to have reviewed our earlier work for you to follow along this example, We assume that you have a good understanding of Azure. If you would like to read about Azure please go [here](https://azure.microsoft.com/en-us/get-started). If you haven't done already installed Azure CLI, do install it as instructed in this [link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

We refer you to learn about Azure Kubernetes Service (AKS) from [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli). Also we refer to [here](https://learn.microsoft.com/en-us/azure/aks/learn/quick-kubernetes-deploy-portal?tabs=azure-cli) on how to request vCPU quotas from azure portal. If you would like to learn about different compute options in Azure please review this [link](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview?tabs=breakdownseries%2Cgeneralsizelist%2Ccomputesizelist%2Cmemorysizelist%2Cstoragesizelist%2Cgpusizelist%2Cfpgasizelist%2Chpcsizelist). In this example we will use two types of vCPUs Standard\_D4ds\_v5 and Standard\_NC40ads\_H100\_v5. We will use the D4ds\_v5 CPUs to run the kubernetes system workloads and NC40ads\_H100\_v5 CPUs to run the GPU workloads. Steps involved in requesting any other vCPUs with GPU will be very similar. In our example we run a simple Machine Learning example on the GPU.  We assume you have a reasonable understanding of Azure Cloud Platform. We are also assume you have a fairly good understanding of Kubernetes. Please do find the Kubernetes reading material from [here](https://kubernetes.io/docs/setup/). We assume that you also have a fairly good working knowledge of github. Please clone this [repo](www.github.com) to your local. Install kubectl, kubernetes cli tool, from [here](https://kubernetes.io/docs/tasks/tools/).

We will be using MacOS to run the kubernetes commands and Azure CLI commands using bash shell. You can follow along with your prefered host, operating system and shell.

### 3. What's in this Repo?
This repo has a docker file that builds from a PyTorch 24.07 [base iamge](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html) from NVidia and runs a simple Kubeflow pipeline with a simple component. This simple pipeline is provided in a python file component\_with\_optional\_inputs.py. This python file we grabbed it from Kubeflow [samples](https://github.com/kubeflow/pipelines/blob/master/samples/v2/component_with_optional_inputs.py). We have also checked in a manifest pipeline-example.yaml that we used to deploy the pipeline.  We discuss these files in detail in our coming sections here.


### 4. Authenticate Your Console
We assume that the kubernetes cluster is up and running. We will do the following two steps to prepare our console to be authenticated to interact with the AKS cluster remotely.

1. Login to your Azure Portal and make sure the kubernetes cluster is up and running.You can also check the cluster from your bash console. For that to work we need to have the *kubectl* working. So go to Step 2, before you try out any *kubectl* commands.

2. In order to issue kubectl commands to control AKS cluster from your local console we need to merge the credentials with the local kube config. Kubernetes config file is typically located under /Users/\<username\>/.kube/config in MacOS. The following azure cli command would merge the config. The second command lets you see the running pods in the cluster:

```
    bash> az aks get-credentials --resource-group <resource-group-name> --name <aks-cluster-name>
    bash> kubectl get pods --watch

```



### 5. Register Microsoft Container Service
We will issue the following Azure CLI commands to register the container service.
```
    bash>   az extension add --name aks-preview
    bash>   az extension update --name aks-preview

    bash>   az feature register --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
    bash>   az feature show --namespace "Microsoft.ContainerService" --name "GPUDedicatedVHDPreview"
    bash>   az provider register --namespace Microsoft.ContainerService

```
### 6. Add nodepool to AKS Cluster

As mentioned before, we don't need a GPU for this example. However, for complex workloads we may need GPUs. So, we create a nodepool with 3 nodes(check Azure documentation to see the Azure's latest offering). You can choose any GPU loaded vCPU from Azure offering that you are eligible to request as per your quota requirements. I tried these GPU loaded nodes Standard\_NC24s\_v3, and Standard\_NC40ads\_H100\_v5 from the NCv3-series and NCads H100 v5-series familes respectively. But the following command adds 3 40 core vCPU with 1 H100 GPU each. We can adjust the min and max counts depending on your workload. We picked a min of 1 and max of 3. This command also taints the nodes with key and value with 'sku' and 'gpu' respectively.

```
    bash> az aks nodepool add --resource-group <name-of-resource-group> --cluster-name <cluster-name> --name <nodepool-name> --node-count 2 --node-vm-size Standard_NC40ads_H100_v5 --node-taints sku=gpu:NoSchedule --aks-custom-headers UseGPUDedicatedVHD=true --enable-cluster-autoscaler --min-count 1 --max-count 3

```

### 7. Create a Azure Container Registry
We need to store the container image of the Simple Pipeline in component\_with\_optional\_inputs.py. We create an Azure Container Registry (ACR) to push your image as follows.

```
    bash> az acr create --name <name-of-acr> --resource-group <resource-group-associated> --sku basic
```

### 8. Login to ACR
Now lets login to the ACR before you can upload any images to ACR.

```
    bash> az acr login --name <name-of-acr>
```

### 9. Create Workload Image Locally
Let's create a docker images that we would like to run as a GPU workload. This repo contains a dockerfile named Dockerfile.ce. At line # 4, this docker file pulls a PyTorch container base image from NVIDIA. Tag 24.07-py3 is the latest available at the time of this writing. This container image contains the complete source of the version of PyTorch in /opt/pytorch. It is a prebuild and installed in the default environment (/usr/local/lib/python3.10/dist-packages/torch). This container also includes the following pacakges: (a) Pyton 3.10, (b) CUDA, (c) NCCL backend, (d) JupyterLab and beyond. Please look at this link for more details [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-07.html). This docker file also copies component\_with\_optional\_inputs.py from current directory to working directory. The component\_with\_optional\_inputs.py is a python file with a simple pipeline with a simple component. At line # 15 in the docker file we execute this python file. Please free to review the python file. In the main, we execute the pipeline at line # 45 using the Kubeflow Pipeline client.

```
    bash> docker build  --platform="linux/amd64"  -t pipeline-example:1.0 .
```

### 10. Tag and push the image to ACR
Now that we have created an image in the local registry, we need to push this image to the ACR before it can be run in the AKS. First, we need to tag the image and then we push the tagged image to an already created ACR. The following are the commands in that order.
```
    bash> docker tag pipline-example:1.0 <acr-name>.azurecr.io/pipeline-example:latest
    bash> docker push <acr-name>.azurecr.io/pipeline-example:latest
```

### 11. Attach the ACR to the AKS Cluster
Now that we have pushed the image to the ACR, we have to now attach that ACR to the cluster so that our job can access the ACR to pull the image from. We use the following command.

```
    bash> az aks update --name <aks-cluster-name> --resource-group <aks-rg-name>  --attach-acr <name-of-acr-to-attach>
```

### 12. Install Kubeflow Training Operator
We will install the Pipeline in AKS using the following commands. We are installing the Pipline version 2.3.0. The third step is different from the Kubeflow Pipline installation step outlined in the documentation. This change is necessary as we are runnng in AKS (not GKS). We use the platform agnostic changes as recommended [here](https://github.com/kubeflow/pipelines/issues/9546).
```
bash> export PIPELINE_VERSION=2.3.0
bash> kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
bash> kubectl wait crd/applications.app.k8s.io --for=condition=established --timeout=60s
bash> kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"
```

Kubeflow Pipeline installation may take a few minutes depending on how long it takes to initialize the pods. Sometimes, some pods may fail in the process but they will self heal and eventually will stabilize.

### 13. Job Manifest
In this repo, we have provided a manifest named pipeline-example.yaml. The image is specfied under Job.spec.template.spec.image path. Two things are worth mentioning in this manifest. Kubeflow pipeline requires root authentication. See [here](https://www.kubeflow.org/docs/components/pipelines/concepts/pipeline-root/) to understand the concepts of pipeline root. We can adjust the Pod efinition to mount ServiceAccount token volume that can be used to authenticate with the Kubeflow Pipelines API. We have defined Job.spec.template.spec.volumes and  Job.spec.template.spec.containers.volumeMounts to project the serviceAccountToken as described [(here)](https://www.deploykf.org/user-guides/access-kubeflow-pipelines-api/).

### 14. Deploy Job
We deploy the job as follows.  

```
    kubectl apply -f pipeline-example.yaml
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

