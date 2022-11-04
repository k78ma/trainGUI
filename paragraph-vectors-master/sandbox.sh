#!/bin/bash
####################################
#
# Source the tensorflow docker container if it doesn't exist, otherwise restart
# the existing sandbox.
#
####################################

#Name for the sandbox's container

xhost +local:
XSOCK=/tmp/.X11-unix

docker run -it \
--runtime=nvidia \
--shm-size 8G \
-v /home/javad/Implementation:/workspace \
-w=/workspace \
-v $XSOCK:$XSOCK \
-e "DISPLAY=$DISPLAY" \
pytorch/pytorch:1.0-cuda10.0-cudnn7-devel /bin/bash
#        darwinai/tensorflow:tf1.8.0-D0.4-gpu /bin/bash
#        tensorflow/tensorflow:1.8.0-devel-gpu-py3


#-u $(id -u):$(id -g) \



# CONTAINER_NAME="tensorflow_py3_sandbox"

# xhost +local:
# XSOCK=/tmp/.X11-unix
# DOCKER_NAMES=$(docker container ls --all --filter="status=exited" | awk '{if (NR!=1) {print $NF}}')
# EXISTS=false
# for name in $DOCKER_NAMES;
# do
#     if [ "$name" == "$CONTAINER_NAME" ]; then
#         EXISTS=true
#         break
#     fi
# done
# if $EXISTS; then
#   docker container start $CONTAINER_NAME -i
# else
#     #!/bin/sh
#     IMAGE=darwinai/tensorflow:tf1.8.0-D0.2-gpu\
#     PYTHONPATH=/workspace:/workspace/licenseclient/licenseClient/python_extension/build/lib.linux-x86_64-3.5/
#     # Mimic /var/lib/gensynth/ (for holding license save_context) on a real system.
#     mkdir -p $SCRIPT_DIR/var/lib/gensynth

#     nvidia-docker run -it  -u $(id -u):$(id -g) \
#         -h "$(hostname)" \
#         -e PYTHONPATH=$PYTHONPATH \
#         -e "TZ=UTC+4" \
#         -e PS1='# ' \
#         -e "DISPLAY=$DISPLAY" \
#         -v ${PWD}/etc:/etc/gensynth \
#         -v ${PWD}/var/lib/gensynth:/var/lib/gensynth \
#         -v "$PWD:/workspace" \
#         -v $XSOCK:$XSOCK \
#         --name=$CONTAINER_NAME \
#         -w /workspace --rm \
#         $IMAGE /bin/bash

# fi
