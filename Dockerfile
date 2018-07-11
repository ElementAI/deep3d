# mxnet builder
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04 as mxnet-builder

RUN export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get -y upgrade \
    && apt install -y python-pip python3-pip graphviz build-essential gcc-6 g++-6 libopenblas-dev git pkg-config python-opencv libopencv-dev wget unzip libcurl4-openssl-dev liblapack-pic libgoogle-perftools-dev google-perftools libjemalloc-dev revolution-mkl libatlas-base-dev libturbojpeg0-dev cython cython3 python-dev python3-dev \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone -b 1.2.0 --recursive https://github.com/dmlc/mxnet

RUN cd mxnet \
    && make -j 6 CC=gcc-6 CXX=g++-6 NVCC=/usr/local/cuda/bin/nvcc USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-9.2/targets/x86_64-linux/ USE_CUDNN=1 USE_NCCL=1 EXTRA_OPERATORS=../operators USE_BLAS=openblas USE_LAPACK_PATH=/usr/lib/x86_64-linux-gnu/openblas/ USE_LIBJPEG_TURBO=1

RUN cd mxnet/python \
    && python2 setup.py bdist && mv dist/mxnet-1.2.0.linux-x86_64.tar.gz ../mxnet-1.2.0.linux-x86_64.py2.tar.gz \
    && python3 setup.py bdist && mv dist/mxnet-1.2.0.linux-x86_64.tar.gz ../mxnet-1.2.0.linux-x86_64.py3.tar.gz

# deep3d builder
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get -y upgrade \
    && apt install -y python-pip graphviz gcc-6 g++-6 libopenblas-dev git pkg-config python-opencv libopencv-dev wget unzip \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /work

COPY --from=mxnet-builder /work/mxnet-1.2.0.linux-x86_64.py2.tar.gz /work
COPY --from=mxnet-builder /work/mxnet-1.2.0.linux-x86_64.py3.tar.gz /work
RUN pip install -f /work/mxnet-1.2.0.linux-x86_64.py2.tar.gz

COPY requirements.txt /work
RUN pip install -r requirements.txt

COPY . /work
