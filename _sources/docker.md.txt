# Running MHCflurry in Docker

This page covers the published Docker images: what they contain, how to run
predictions and notebooks with them, and how to build your own. Use it if you
want MHCflurry without installing Python packages on the host. For a normal
`pip` installation, see {doc}`intro` instead.

## What the images contain

Images built from version 2.3.6 onward include the full presentation weights,
the command-line tools and the Jupyter notebooks. They run predictions on the
CPU and support Intel/AMD and ARM Linux.

Check the published image's version with:

```shell
docker pull openvax/mhcflurry:latest
docker run --rm openvax/mhcflurry:latest mhcflurry --version
```

## Running predictions

Run predictions against files in the current directory:

```shell
docker run --rm -v "$PWD:/work" openvax/mhcflurry:latest \
    mhcflurry predict input.csv --out predictions.csv
```

Mounted directories must be writable by the container's user (UID 1000).

## Running the notebooks

To start Jupyter, run the image without a command:

```shell
docker run --rm -p 127.0.0.1:9999:9999 -v "$PWD:/work" openvax/mhcflurry:latest
```

Open the localhost URL printed in the logs, including its access token.
Without the volume mount, the image starts in a directory containing the
example notebooks.

## Image tags

Each stable release publishes a matching version tag. The `latest` tag is
updated only after both architecture builds pass an offline prediction check,
so an incomplete publication can leave `latest` on the previous version; see
the [Docker workflow](https://github.com/openvax/mhcflurry/actions/workflows/docker.yml)
for publication status.

## Building from a checkout

```shell
docker build -t mhcflurry:local .
# Separate image for CUDA training
docker build -f docker/Dockerfile.train -t mhcflurry:train .
```
