FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/openvax/mhcflurry"
LABEL org.opencontainers.image.description="MHCflurry prediction CLI and notebooks with pretrained presentation weights"

ENV MHCFLURRY_DATA_DIR=/opt/mhcflurry-data \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The public notebook/prediction image uses CPU PyTorch on Intel and ARM.
# CUDA training has its own image in docker/Dockerfile.train.
RUN pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.0.0'
COPY requirements.txt /tmp/mhcflurry-requirements.txt
RUN pip install -r /tmp/mhcflurry-requirements.txt jupyter seaborn

COPY . /opt/mhcflurry
RUN pip install /opt/mhcflurry && \
    mhcflurry downloads fetch models_class1_presentation && \
    pip check && \
    useradd --create-home --uid 1000 mhcflurry && \
    mkdir /work && cp /opt/mhcflurry/notebooks/*.ipynb /work/ && \
    chown -R mhcflurry:mhcflurry /work /opt/mhcflurry-data

USER mhcflurry
WORKDIR /work
EXPOSE 9999
CMD ["jupyter", "notebook", "--port=9999", "--no-browser", "--ip=0.0.0.0"]
