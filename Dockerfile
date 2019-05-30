FROM nvidia/cuda

MAINTAINER Tim O'Donnell <timodonnell@gmail.com>

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get clean && \
    apt-get update && \
    apt-get install --yes \
        locales \
        gfortran \
        git \
        libatlas-base-dev \
        libatlas3-base \
        libblas-dev \
        libfreetype6-dev \
        libhdf5-dev \
        liblapack-dev \
        libpng-dev \
        libxml2-dev \
        libxslt1-dev \
        libyaml-dev \
        libzmq3-dev \
        pkg-config \
        python-virtualenv \
        python3-dev \
        python-dev && \
    apt-get clean && \
    useradd --create-home --home-dir /home/user --shell /bin/bash -G sudo user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
        
# Set the locale (otherwise dask-distributed complains).
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER user
ENV HOME=/home/user
ENV SHELL=/bin/bash
ENV USER=user
WORKDIR /home/user

# Setup virtual envs and install convenience packages.  Note: installing
# cherrypy as part of the mhcflurry installation weirdly fails on a unicode
# issue in python2, but installing it separately seems to work.
RUN virtualenv venv-py3 --python=python3 && \
    venv-py3/bin/pip install --upgrade pip && \
    venv-py3/bin/pip install --upgrade \
        numpy \
        cherrypy \
        jupyter \
        lxml \
        scipy \
        scikit-learn \
        seaborn

ENV KERAS_BACKEND tensorflow
# RUN venv-py3/bin/pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl

# Install mhcflurry and latest kubeface and download data and models.
COPY . ./mhcflurry
RUN venv-py3/bin/pip install --upgrade ./mhcflurry git+https://github.com/hammerlab/kubeface.git \
    && venv-py3/bin/mhcflurry-downloads fetch
 
EXPOSE 8888
ENTRYPOINT ["venv-py3/bin/jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
