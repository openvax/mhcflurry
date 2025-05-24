FROM continuumio/miniconda3:latest

LABEL maintainer="Tim O'Donnell timodonnell@gmail.com"

WORKDIR /root

# Install system dependencies
RUN apt-get update -y && apt-get install -y gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a lightweight conda env with Python 3.10
RUN conda create -n mhcflurry python=3.10 -y && \
    conda clean -afy

# Activate the env by modifying PATH
ENV PATH /opt/conda/envs/mhcflurry/bin:$PATH

# Install pip packages in the env
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jupyter seaborn

# Install dependencies (doing this first to have them cached)
COPY requirements.txt /tmp/mhcflurry-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mhcflurry-requirements.txt

# Pre-download resources for mhcflurry
RUN mkdir /tmp/mhcflurry-downloads
COPY mhcflurry/downloads.yml /tmp/mhcflurry-downloads
RUN python -c '\
import yaml, subprocess; \
d = yaml.safe_load(open("/tmp/mhcflurry-downloads/downloads.yml")); \
downloads = d["releases"][d["current-release"]]["downloads"]; \
urls = [item["url"] for item in downloads if item["default"]]; \
[subprocess.run(["wget", "-P", "/tmp/mhcflurry-downloads", url]) for url in urls]'

# Copy example notebooks
COPY notebooks/* ./

# Copy source code and install mhcflurry in editable mode
COPY . mhcflurry
RUN pip install -e mhcflurry/

# Fetch resources from pre-downloaded data
RUN mhcflurry-downloads fetch --already-downloaded-dir /tmp/mhcflurry-downloads

EXPOSE 9999
CMD ["jupyter", "notebook", "--port=9999", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

