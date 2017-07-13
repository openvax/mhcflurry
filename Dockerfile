FROM debian:8.5

MAINTAINER Rohan Pai <rohanpai@hammerlab.org>

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean


RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install mhcflurry
RUN /opt/conda/bin/pip install pandas==0.19.2
RUN /opt/conda/bin/pip install numpy>=1.11
RUN /opt/conda/bin/pip install Keras==2.0.4
RUN /opt/conda/bin/pip install appdirs
RUN /opt/conda/bin/pip install tensorflow
RUN /opt/conda/bin/pip install scikit-learn
RUN /opt/conda/bin/pip install typechecks
RUN /opt/conda/bin/pip install dill>=0.2.5
RUN /opt/conda/bin/pip install parse
RUN /opt/conda/bin/pip install oauth2client==4.0.0
RUN /opt/conda/bin/pip install google-api-python-client==1.5.5
RUN /opt/conda/bin/pip install mock
RUN /opt/conda/bin/pip install nose>=1.3.1

ENV PATH /opt/conda/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

