# Running MHCflurry in Docker

This page covers the published Docker images: what they contain, how to run
predictions and notebooks with them, and how to build your own. Use it if you
want MHCflurry without installing Python packages on the host. For a normal
`pip` installation, see {doc}`intro` instead.

## What the images contain

Images built from version 2.3.6 onward include the full `2.3.0` presentation
weights, the command-line tools and the Jupyter notebooks. They run predictions
on the CPU and support Intel/AMD (`linux/amd64`) and ARM (`linux/arm64`) Linux.

Multi-platform images are published automatically to the **GitHub Container
Registry** (`ghcr.io/openvax/mhcflurry`) on every stable release (and mirrored
to Docker Hub `openvax/mhcflurry` when Docker Hub credentials are configured).

Check the published image's version with:

```shell
docker pull ghcr.io/openvax/mhcflurry:latest
docker run --rm ghcr.io/openvax/mhcflurry:latest mhcflurry --version
```

## Running predictions

Run predictions against files in the current directory:

```shell
docker run --rm -v "$PWD:/work" ghcr.io/openvax/mhcflurry:latest \
    mhcflurry predict input.csv --out predictions.csv
```

Mounted directories must be writable by the container's user (UID 1000).

## Running the notebooks

To start Jupyter, run the image without a command:

```shell
docker run --rm -p 127.0.0.1:9999:9999 -v "$PWD:/work" ghcr.io/openvax/mhcflurry:latest
```

Open the localhost URL printed in the logs, including its access token.
Without the volume mount, the image starts in a directory containing the
example notebooks.

## Image tags

Each stable release publishes a matching version tag (e.g.
`ghcr.io/openvax/mhcflurry:2.5.17`). The `latest` tag is updated only after
both `amd64` and `arm64` builds pass an offline prediction check, so an
incomplete publication can leave `latest` on the previous version; see the
[Docker workflow](https://github.com/openvax/mhcflurry/actions/workflows/docker.yml)
for publication status.

## Building from a checkout

```shell
docker build -t mhcflurry:local .
# Separate image for CUDA training
docker build -f docker/Dockerfile.train -t mhcflurry:train .
```
