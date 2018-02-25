FROM neurodebian:xenial-non-free
MAINTAINER Teng-Yi Huang

# System packages
RUN apt-get update && apt-get install -y wget bzip2 build-essential
# install miniconda
RUN wget -q http://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh
RUN bash Miniconda3-4.3.31-Linux-x86_64.sh -b -p /usr/local/miniconda
RUN rm Miniconda3-4.3.31-Linux-x86_64.sh

# update path to include conda
ENV PATH=$PATH:/usr/local/miniconda/bin

# install conda dependencies
RUN conda install -y numpy==1.11 pip python=3.5 scipy matplotlib pillow

# install python dependencies
RUN pip install nibabel psutil sklearn

# install FSL
RUN apt-get update
RUN apt-get install -y --no-install-recommends fsl-core fsl-atlases fsl-mni152-templates
RUN pip install nilearn
# setup fsl environment
ENV FSLDIR=/usr/share/fsl/5.0
ENV FSLOUTPUTTYPE=NIFTI_GZ
#FSLMULTIFILEQUIT=TRUE \
#    POSSUMDIR=/usr/share/fsl/5.0 \
#    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH \
#    FSLTCLSH=/usr/bin/tclsh \
#    FSLWISH=/usr/bin/wish \
ENV    PATH=/usr/lib/fsl/5.0:$PATH

# clean up
RUN apt-get clean
RUN apt-get autoremove -y
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup application
ADD ./pymars /pymars
ENTRYPOINT ["bash" ,"-c", "/usr/local/miniconda/bin/python /pymars/mars_run.py"]
