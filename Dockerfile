ARG TF_TAG=latest-py3
FROM tensorflow/tensorflow:$TF_TAG
ADD . /home/DCSeg/
WORKDIR /home/DCSeg/src/
RUN apt-get update
RUN apt-get install -y vim wget
# Installing nodejs required for jupyterlab plugins
RUN wget https://deb.nodesource.com/setup_11.x
RUN chmod +x setup_11.x
RUN bash setup_11.x
RUN apt-get install -y nodejs
# Installing required packages (some are already present in tensorflow distribution)
RUN pip install --upgrade request scikit-image scikit-learn seaborn tensorflow-probability SimpleITK jupyterlab ipywidgets segmentation-models 
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets
# Running jupyterlab by default
CMD jupyter lab --allow-root
# CMD bash -C 'dockerrc.sh'; '/bin/bash'

