FROM hammerlab/mhcflurry:latest

# Extend the main MHCflurry docker image to additionally have a Python2 virtual
# environment.

MAINTAINER Tim O'Donnell <timodonnell@gmail.com>

RUN virtualenv venv-py2 --python=python && \
    venv-py2/bin/pip install cherrypy bokeh distributed jupyter seaborn ./mhcflurry

EXPOSE 8888
CMD venv-py2/bin/jupyter notebook --no-browser
