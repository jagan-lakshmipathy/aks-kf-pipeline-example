# We need to use the nvcr.io/nvidia/pytorch image as a base image to support both linux/amd64 and linux_arm64 platforms.
# PyTorch=2.2.0, cuda=12.3.2
# Ref: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html#rel-24-01
FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install kfp
RUN mkdir -p /opt/pipeline_test/

WORKDIR /opt/pipeline_test/src
ADD component_with_optional_inputs.py /opt/pipeline_test/src/component_with_optional_inputs.py

RUN chgrp -R 0 /opt/pipeline_test \
    && chmod -R g+rwX /opt/pipeline_test

ENTRYPOINT ["python", "/opt/pipeline_test/src/component_with_optional_inputs.py"]
