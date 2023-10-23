FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 as build-image
LABEL maintainer="Mike Schwendeman <michael.schwendeman@eagleview.com>" \
      description="ev-nerf run dependencies"

# Get build tools
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git cmake python3-dev python3-tk python3-pip python3-venv

# Use a virtualenv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Pip install requirements pytorch and modules with pytorch extensions TODO: Derive architecture value somehow
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"
ARG TCNN_CUDA_ARCHITECTURES="75"
ARG CUDA_HOME="/usr/local/cuda-11.7"
ARG CUDA_PATH="/usr/local/cuda-11.7"
RUN pip install --no-cache-dir --force-reinstall torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/michael-schwendeman-ev/torch_efficient_distloss
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
RUN pip install --no-cache-dir --force-reinstall git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.3

# Pip install remaining requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt 

# Get ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

######################################  Runtime image #######################################
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y python3 libgl1 libglib2.0-0 

# Copy compiled python
COPY --from=build-image /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# Copy ffmpeg binaries
COPY --from=build-image /usr/bin/ffmpeg /usr/bin/ffmpeg
COPY --from=build-image /usr/lib/*-linux-gnu/* /usr/lib/

# Copy source code to container
COPY ./baangp /baangp

# Run the training script
WORKDIR /baangp
ENTRYPOINT ["python3", "-u", "train_baangp.py"]