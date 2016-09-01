FROM nvidia/cuda:cudnn-runtime

MAINTAINER Tim O'Donnell <timodonnell@gmail.com>

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update && \
    apt-get install --yes \
        gfortran \
        libatlas-base-dev \
        libatlas3gf-base \
        libblas-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        liblapack-dev \
        libpng12-dev \
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
# We also install bokeh so that dask distributed will have an admin web interface.
RUN virtualenv venv-py3 --python=python3 && \
    venv-py3/bin/pip install cherrypy bokeh distributed jupyter seaborn

# Install mhcflurry.
COPY . ./mhcflurry
RUN venv-py3/bin/pip install ./mhcflurry
 
EXPOSE 8888
CMD venv-py3/bin/jupyter notebook --no-browser
