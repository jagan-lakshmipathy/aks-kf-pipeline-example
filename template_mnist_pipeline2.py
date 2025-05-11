import os
from typing import Optional

import kfp
from kfp import compiler
from kfp import dsl
from kfp.dsl import component, Output, Input, Artifact

# In tests, we install a KFP package from the PR under test. Users should not
# normally need to specify `kfp_package_path` in their component definitions.
_KFP_PACKAGE_PATH = os.getenv('KFP_PACKAGE_PATH')


@dsl.container_component
def model_train(model_info_path: Output[Artifact]):
    #return dsl.ContainerSpec(image='registry.digitalocean.com/do-dev-jagan-06022024/kubeflow/pipeline-example:latest', 
    return dsl.ContainerSpec(image='<my-acr>.azurecr.io/kubeflow/mnist-trainer:latest', 
                             command=['/bin/sh'], args=['-c' , f'python /opt/kfp_pipeline/src/mnist.py --save-model --output-path {model_info_path.path}'])

@dsl.container_component
def model_evaluate(model_info_path: Input[Artifact]):
    return dsl.ContainerSpec(
        image='<my-acr>.azurecr.io/kubeflow/mnist-evaluator:latest',
        command=['/bin/sh'],
        args=[
            '-c',
            f'python /opt/kfp_pipeline/src/model_evaluate.py --model-info-path {model_info_path.path}'
        ]
    )

# @dsl.pipeline
# def model_pipeline():
#     # greeting argument is provided automatically at runtime!
#     mt = model_train()
#     print('Printing mt: ', mt)



@dsl.pipeline(name="train-eval-pipeline")
def model_pipeline():
    # Step 1: Train the model
    train_step = model_train()

    # Step 2: Evaluate the model using the output of train_step
    eval_step = model_evaluate(model_info_path=train_step.outputs['model_info_path'])


if __name__ == "__main__":
    # execute only if run as a script
    compiler.Compiler().compile(
        pipeline_func=model_pipeline, package_path=__file__.replace('.py', '.yaml'))


    #http://ml-pipeline.kubeflow.svc.cluster.local:8888
    #ml_pipeline_url='host.docker.internal'
    ml_pipeline_url = 'ml-pipeline.kubeflow.svc.cluster.local'

    _kfp_host_and_port = os.getenv('KFP_API_HOST_AND_PORT', f'http://{ml_pipeline_url}:8888')
    _kfp_ui_and_port = os.getenv('KFP_UI_HOST_AND_PORT', f'http://{ml_pipeline_url}:8080')
    kfp_client = kfp.Client(host=_kfp_host_and_port, ui_host=_kfp_ui_and_port)


    kfp_client.create_run_from_pipeline_package("./mnist_pipeline2.yaml")
