FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# build-time variables to be set by the Makefile
ARG group
ARG gid
ARG user
ARG uid

RUN groupadd --gid $gid $group && \
    useradd --create-home --shell /bin/bash --gid $gid --uid $uid $user && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        less \
        libfreetype6-dev \
        libjpeg-dev \
        libzmq3-dev \
        nano \
        pkg-config \
        python \
        python3 \
        python3-dev \
        python-pip \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        wget \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        cython \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        pyyaml \
        && \
    python3 -m ipykernel.kernelspec

# fix missing drivers and links for CUDA-libraries
COPY files/lib/libcuda.so.418.43 /usr/lib/
COPY files/lib/libnvidia-fatbinaryloader.so.418.43 /usr/lib/
RUN ln -s /usr/lib/libcuda.so.418.43 /usr/lib/libcuda.so.1 && \
    ln -s /usr/lib/libcuda.so.418.43 /usr/lib/libcuda.so && \
    chmod a+r /usr/lib/libcuda.so* && \
    chmod a+r /usr/lib/libnvidia-fatbinaryloader.so.418.43 && \
    cp -P /usr/include/cudnn.h /usr/local/cuda/include && \
    cp -P /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/lib64/libcudnn* && \
    echo "/usr/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/cuda/include" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig
ENV LD_LIBRARY_PATH /usr/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/include:/usr/local/cuda
ENV CUDA_HOME /usr/local/cuda

RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && \
    HOME=/opt /bin/bash Anaconda3-2020.11-Linux-x86_64.sh -b && \
    rm Anaconda3-2020.11-Linux-x86_64.sh
ENV PATH="/opt/anaconda3/bin:$PATH" \
    CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
RUN conda update conda && \
    conda update anaconda && \
    conda update --all && \
    conda install \
        numpy \
        ninja \
        pyyaml \
        mkl \
        mkl-include \
        setuptools \
        cmake \
        cffi \
        typing_extensions \
        future \
        six \
        requests \
        dataclasses \
        && \
    conda install -c pytorch magma-cuda101

# tini is a process subreaper for Jupyter which prevents kernel crashes
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# wrapper script to start Jupyter (https://github.com/ipython/ipython/issues/7062)
COPY files/run_jupyter.sh /home/$user/
COPY files/jupyter_notebook_config.py /home/$user/.jupyter/
RUN chmod +x /home/$user/run_jupyter.sh

USER $user
EXPOSE 5555

CMD ["/bin/bash"]
