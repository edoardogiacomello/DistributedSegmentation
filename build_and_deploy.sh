#! /bin/bash
DEFAULTNAME="dcseg"
DEVICE="gpu"
IMAGENAME="edoardogiacomello/dcseg:latest"
if [ $# -gt 0 ]
then    CONTAINER_NAME="--name $1";  echo "Building a docker image ";
else    CONTAINER_NAME="--name $DEFAULTNAME";
fi
docker build -t $IMAGENAME .
echo "Running container $CONTAINER_NAME based on $IMAGENAME"
echo "Mapping container port 8888 to host port 8887"
echo "Mapping container port 6006 to host port 6005"
nvidia-docker run -p 8889:8888 -v $PWD/../datasets/:/home/datasets/:ro -v $PWD:/home/DCSeg $CONTAINER_NAME -it $IMAGENAME
