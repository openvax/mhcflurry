FROM nvidia/cuda:cudnn-runtime

MAINTAINER Tim O'Donnell <timodonnell@gmail.com>

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install --yes \
        python3-dev python-dev \
        python-virtualenv \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libhdf5-serial-dev \
        libyaml-dev \
        libzmq3-dev \
        libfreetype6-dev \
        libpng12-dev \
        pkg-config && \
    apt-get clean && \
    useradd --create-home --home-dir /home/user --shell /bin/bash user

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

# Setup virtual envs and install convenience packages.
RUN virtualenv venv-py3 --python=python3 && \
    venv-py3/bin/pip install distributed jupyter seaborn && \
    virtualenv venv-py2 --python=python && \
    venv-py2/bin/pip install distributed jupyter seaborn

# Install mhcflurry.
COPY . ./mhcflurry
RUN venv-py3/bin/pip install ./mhcflurry && venv-py2/bin/pip install ./mhcflurry
 
EXPOSE 8888
CMD venv-py3/bin/jupyter notebook --no-browser
