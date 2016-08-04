FROM floydhub/dl-docker:cpu
# TODO: seems like it updates; freeze a version of this

MAINTAINER Jeff Hammerbacher <jeff.hammerbacher@gmail.com>

#RUN apt-get update
#RUN apt-get -y upgrade

RUN pip install --upgrade numpy
RUN git clone https://github.com/hammerlab/mhcflurry.git # replace this with a COPY command if you want to make a dev image
RUN pip install -e mhcflurry/

# set up mhcflurry
RUN mhcflurry/script/download-iedb.sh
RUN mhcflurry/script/download-kim-2013-dataset.sh
RUN mhcflurry/script/create-iedb-class1-dataset.py
RUN mhcflurry/script/create-combined-class1-dataset.py
# do not run nosetests, they seem to fail right now

CMD ["/bin/bash"]