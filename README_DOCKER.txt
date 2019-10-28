# Building
run the script build_*pu.sh according to your machine configuration.
The corresponding environment folders are for avoid loading all the sources in the docker images, to keep it lightweight. 

# Running
Docker image should be "RUN" ONLY ONCE in the repository root, using the following synthax

nvidia-docker run -p 8889:8888 -v $PWD/../datasets/:/home/datasets/:ro -v $PWD:/tf $CONTAINER_NAME -it $IMAGENAME

8889 is the port where to expose jupyter lab
../datasets [it is outside of the repository!] is a folder that is mounted for packing the original datasets into .tfrecords datasets, which are usually stored into ./src/datasets/. You can remove this binding if you don't need it
$CONTAINER_NAME is a user friendly name for your running container
$IMAGENAME is the name of the image built using build.sh. You can find it using the "docker images" command

1) The images run a bash shell by default. For running jupiter use the following procedure:

tmux
jupyter lab --allow-root --ip=0.0.0.0

then open your browser and go to <host-ip>:8889 (or whatever port you assigned in the run command) and use the token that is printed in the docker shell.
This will allow to run an instance of jupyter lab without losing your bash shell.

You can now detach from the shell using ctrl+p,q.
From now on you can start/stop/attach to your docker that will keep the state. 

If you need to rebuild the image, first delete the container with docker rm $CONTAINER_NAME (the current state of the environment will be lost)
