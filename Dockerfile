FROM continuumio/miniconda3:latest

LABEL maintainer="Tim O'Donnell timodonnell@gmail.com"

WORKDIR /root

# Install system dependencies
RUN apt-get update -y && apt-get install -y gcc

# Install / upgrade packages
RUN pip install --upgrade pip && pip install jupyter seaborn

# Install dependencies (doing this first to have them cached).
COPY requirements.txt /tmp/mhcflurry-requirements.txt
RUN pip install -r /tmp/mhcflurry-requirements.txt

# We pre-download the downloads here to avoid having to redownload them every
# time the source code changes (i.e. we do this before the COPY . so these
# downloads are part of the docker cache).
RUN mkdir /tmp/mhcflurry-downloads
COPY mhcflurry/downloads.yml /tmp/mhcflurry-downloads
RUN wget -P /tmp/mhcflurry-downloads \
    $(python -c 'import yaml ; d = yaml.safe_load(open("/tmp/mhcflurry-downloads/downloads.yml")) ; downloads = d["releases"][d["current-release"]]["downloads"] ; defaults = [item["url"] for item in downloads if item["default"]] ; print("\n".join(defaults))')

# Copy example notebook to current directory so it's easily found.
COPY notebooks/* ./

# Copy over source code and install mhcflurry.
COPY . mhcflurry
RUN pip install -e mhcflurry/
RUN mhcflurry-downloads fetch --already-downloaded-dir /tmp/mhcflurry-downloads

EXPOSE 9999
CMD ["jupyter", "notebook", "--port=9999", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
