#!/bin/zsh

RG_NAME="jagan-aks-05112025-rg"
CLSTR_NAME="jagan-aks-05112025-cluster"
ACR_NAME="jaganacr05112025"

MODEL_NAME="mnist"
TRAINING_COMP=$MODEL_NAME"-trainer"
EVAL_COMP=$MODEL_NAME"-evaluator"

PIPELINE_NAME=$MODEL_NAME"-pipeline"

# Define an array with the names of the environment variables
COMP_LIST=("TRAINING_COMP" "EVAL_COMP")

PIPELINE_DEPLOYMENT_YAML_FILE="pipeline-deployment.yaml"


set -e  # Exit script on any command failure

# Trap to capture the last error before exiting
trap 'echo "Error encountered at line $LINENO with exit code $?"; exit 1' ERR

# Remove the problematic extension
az extension remove --name aks-preview

# Install the stable version (14.0.0b3)
az extension add --name aks-preview --version 14.0.0b3

# Verify the installed extension version
az extension show --name aks-preview

echo "AZ extension commands executed successfully"

# Get AKS credentials
az aks get-credentials --resource-group $RG_NAME --name $CLSTR_NAME

# Verify kubeconfig
kubectl config current-context || { echo "Cluster credentials not set correctly"; exit 1; }

echo "Command executed successfully"

# Prompt for confirmation with timeout
echo -n "All looks good? Continue? (y/n): "
read -t 10 choice || choice="y"

if [[ ! "$choice" =~ ^[yY]$ ]]; then
  echo "Operation canceled."
  exit 0
fi

echo "Okay, continuing....."

# Add nodepool
az aks nodepool add --resource-group $RG_NAME --cluster-name $CLSTR_NAME --name gpunp --node-count 1 --node-vm-size Standard_NC6s_v3 --node-taints sku=gpu:NoSchedule --enable-cluster-autoscaler --min-count 1 --max-count 3


echo "Waiting for node pool 'gpunp' to be ready..."
while true; do
  STATUS=$(az aks nodepool show --resource-group "$RG_NAME" --cluster-name "$CLSTR_NAME" --name gpunp --query "provisioningState" -o tsv)
  
  if [[ "$STATUS" == "Succeeded" ]]; then
    echo "Node pool 'gpunp' is ready!"
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    echo "Node pool creation failed!"
    exit 1
  else
    echo "Node pool status: $STATUS. Waiting..."
    sleep 30  # Check every 30 seconds
  fi
done


# Prompt user for confirmation
echo -n "All looks good? Continue? (y/n): "
read choice
case "$choice" in 
  y|Y ) 
    echo "Okay, continuing....."
    ;;
  * ) 
    echo "Operation canceled."
    exit 0
    ;;
esac

export PIPELINE_VERSION=2.3.0
kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait crd/applications.app.k8s.io --for=condition=established --timeout=60s
kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"
echo "Successfully installed Kubeflow Pipeline."

az acr create --name $ACR_NAME --resource-group $RG_NAME --sku basic

# Wait for the ACR to be fully ready
echo "Waiting for Azure Container Registry '$ACR_NAME' to be ready..."
while true; do
  STATUS=$(az acr show --name "$ACR_NAME" --resource-group "$RG_NAME" --query "provisioningState" -o tsv)

  if [[ "$STATUS" == "Succeeded" ]]; then
    echo "Azure Container Registry '$ACR_NAME' is ready!"
    break
  elif [[ "$STATUS" == "Failed" ]]; then
    echo "ACR creation failed!"
    exit 1
  else
    echo "ACR status: $STATUS. Waiting..."
    sleep 30  # Check every 30 seconds
  fi
done

# Prompt user for confirmation
echo -n "All looks good? Continue? (y/n): "
read choice
case "$choice" in 
  y|Y ) 
    echo "Okay, continuing....."
    ;;
  * ) 
    echo "Operation canceled."
    exit 0
    ;;
esac

az aks update --name $CLSTR_NAME --resource-group $RG_NAME  --attach-acr $ACR_NAME 

# Fetch AKS principal ID
AKS_PRINCIPAL_ID=$(az aks show -g "$RG_NAME" -n "$CLSTR_NAME" --query "identity.principalId" -o tsv)

# Fetch ACR ID
ACR_ID=$(az acr show -n "$ACR_NAME" --query "id" -o tsv)

az role assignment create --assignee "$AKS_PRINCIPAL_ID" --role "AcrPull" --scope "$ACR_ID"



# Check for existing AcrPull role assignment
ROLE_EXISTS=$(az role assignment list \
  --assignee "$AKS_PRINCIPAL_ID" \
  --scope "$ACR_ID" \
  --query "[?roleDefinitionName=='AcrPull']" \
  -o tsv)

if [[ -n "$ROLE_EXISTS" ]]; then
  echo "✅ AcrPull role assignment exists. Proceeding with ACR login and push..."
fi


# Prompt user for confirmation
echo -n "All looks good? Continue? (y/n): "
read choice
case "$choice" in 
  y|Y ) 
    echo "Okay, continuing....."
    ;;
  * ) 
    echo "Operation canceled."
    exit 0
    ;;
esac

# Attempt ACR login

# Function to log errors
trap 'echo "❌ Error encountered at line $LINENO with exit code $?"; exit 1' ERR

# Retrieve ACR token
TOKEN=$(az acr login --name "$ACR_NAME" --expose-token --output json | jq -r '.accessToken')

# Ensure the token is valid
if [[ -z "$TOKEN" ]]; then
    echo "❌ Failed to retrieve ACR token! Exiting..."
    exit 1
fi

# Authenticate Docker using the token
if ! docker login "$ACR_NAME.azurecr.io" -u 00000000-0000-0000-0000-000000000000 -p "$TOKEN"; then
    echo "❌ Docker login to ACR failed! Exiting..."
    exit 1
fi

echo "✅ Docker login successful!"

# Ask for confirmation before proceeding (Fixes Zsh prompt issue)
echo -n "All looks good? Continue? (y/n): "
read choice

case "$choice" in
  y|Y ) 
    echo "Okay, continuing....."
    ;;
  * ) 
    echo "Operation canceled."
    exit 0
    ;;
esac


kubectl create namespace gpu-operator
kubectl apply -f nvidia-device-plugin-ds.yaml
echo "Installed NVidia Plugin DaemonSet"

# Loop through each item in the array
for var in "${COMP_LIST[@]}"; do

    # 💥 Expand the variable reference first
    comp_value="${(P)var}"

    echo "Processing ${comp_value}"

    # Now you can use $comp_value safely
    docker_template_file="Dockerfile.${comp_value}.template"
    docker_file="Dockerfile.${comp_value}"

    if [[ -z "$ACR_NAME" || -z "$docker_template_file" || -z "$docker_file" ]]; then
        echo "Error: One or more required files are missing."
        echo "Ensure ACR_NAME, $docker_template_file, and $docker_file are set."
        exit 1
    fi

    sed -e "s|<my-acr>|${ACR_NAME}|g" \
        -e "s|<model-name>|${MODEL_NAME}|g" \
        "$docker_template_file" > "$docker_file"

    docker build --platform="linux/amd64" -f "$docker_file" -t "kubeflow/${comp_value}:1.0" ./
    echo "Completed docker build for component: ${comp_value}"

    docker tag "kubeflow/${comp_value}:1.0" "$ACR_NAME.azurecr.io/kubeflow/${comp_value}:latest"
    echo "Tagged: ${comp_value}"

    docker push "$ACR_NAME.azurecr.io/kubeflow/${comp_value}:latest"
    echo "Pushed: ${comp_value}"

done

# Resolve the actual file names first
PY_MNIST_PIPELINE_TEMPLATE_FILE="template_mnist_pipeline2.py"
PY_MNIST_PIPELINE_FILE="mnist_pipeline2.py"


# Use sed to replace "<my-acr>" with the replacement value
sed -e "s|<my-acr>|${ACR_NAME}|g" \
    "${PY_MNIST_PIPELINE_TEMPLATE_FILE}" > "${PY_MNIST_PIPELINE_FILE}"
 
# Resolve the actual file names first
docker_file="Dockerfile.pipeline"

# Ensure values are set and files exist (if needed)
if [[ -z "$ACR_NAME" || -z "$docker_file" ]]; then
    echo "Error: One or more required files are missing."
    echo "Ensure ACR_NAME, $docker_template_file, and $docker_file are set."
    exit 1
fi

docker build --platform="linux/amd64" -f $docker_file -t kubeflow/${PIPELINE_NAME}:1.0 ./
echo "Completed ${PIPELINE_NAME} Image Build."

docker tag  kubeflow/${PIPELINE_NAME}:1.0 $ACR_NAME.azurecr.io/kubeflow/${PIPELINE_NAME}:latest
echo "Tagged: ${PIPELINE_NAME} Image."

docker push $ACR_NAME.azurecr.io/kubeflow/${PIPELINE_NAME}:latest
echo "Pushed: ${PIPELINE_NAME} Image."


# Resolve the actual file names first
YAML_TEMPLATE_FILE="template-$PIPELINE_DEPLOYMENT_YAML_FILE"
YAML_FILE="$PIPELINE_DEPLOYMENT_YAML_FILE"


# Ensure replacement value and file paths are set
if [[ -z "$YAML_TEMPLATE_FILE" || -z "$YAML_FILE" ]]; then
    echo "Error: One or more required environment variables are missing."
    echo "Ensure replacement, YAML_TEMPLATE_FILE, and YAML_FILE are set."
    exit 1
fi

# Use sed to replace "<my-acr>" with the replacement value
sed -e "s|<my-acr>|${ACR_NAME}|g" \
    -e "s|<pipeline-name>|${PIPELINE_NAME}|g" \
    "${YAML_TEMPLATE_FILE}" > "${YAML_FILE}"

echo "Replacement done. New YAML file created: $YAML_FILE"

kubectl apply -f $YAML_FILE