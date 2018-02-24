FROM neurodebian:xenial-non-free
MAINTAINER Teng-Yi Huang

# System packages
RUN apt-get update && apt-get install -y wget
# install miniconda
RUN wget -q http://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh
RUN bash Miniconda3-4.3.31-Linux-x86_64.sh -b -p /usr/local/miniconda
RUN rm Miniconda3-4.3.31-Linux-x86_64.sh

# update path to include conda
ENV PATH=/usr/local/miniconda/bin:$PATH

# install conda dependencies
RUN conda install -y numpy==1.11 pip python=3.5 scipy matplotlib

# install python dependencies
RUN pip install nibabel psutil

# install FSL
RUN apt-get update
RUN apt-get install -y --no-install-recommends fsl-core fsl-atlases fsl-mni152-templates

# setup fsl environment
ENV FSLDIR=/usr/share/fsl/5.0 \
    FSLOUTPUTTYPE=NIFTI_GZ \
    FSLMULTIFILEQUIT=TRUE \
    POSSUMDIR=/usr/share/fsl/5.0 \
    LD_LIBRARY_PATH=/usr/lib/fsl/5.0:$LD_LIBRARY_PATH \
    FSLTCLSH=/usr/bin/tclsh \
    FSLWISH=/usr/bin/wish \
    PATH=/usr/lib/fsl/5.0:$PATH

# clean up
RUN apt-get clean
RUN apt-get autoremove -y
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT [""]
