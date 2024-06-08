# Instructions

## 1. Prerequisites

To run this you will require `docker engine` and `docker compose plugin`

### Check for existing installations

```bash 
docker run hello-world
```

```bash
docker compose version
```

### Install required software

If there is no existing installation, follow [these instructions](https://docs.docker.com/engine/install/ubuntu/#installation-methods) or follow the command below to install the required software

```bash
curl -sSL https://get.docker.com | sh
```

## 2. Creating the container

### Aquiring the docker image

The docker image can be pulled from my personal repo (subject to change) with the command below

```bash
docker pull gsavle/reconcycle-screw-detection:latest
```

Now that we the image has been pulled, it can either be ran directly by running:

```bash
docker run -it \
  --name gsavle/reconcycle-screw-detection:latest \
  --gpus '"device=0"' \
  --device /dev:/dev \
  --env "ROS_MASTER_URI=http://<ROS master IP>" \
  --env "ROS_IP=<host IP>" \
  --env "NVIDIA_VISIBLE_DEVICES=0" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all" \
  --env "DEBUG_COLORS=true" \
  --env "TERM=xterm-256color" \
  --env "COLORTERM=truecolor" \
  --privileged \
  --network host \
  yoloscrews:devel \
  tail -f /dev/null
```

*make sure to fill out the <ROS master IP> and <host IP> fields with the proper information of your particular workcell*

Or the container can be configured in a more streamlined fashion by using a `docker compose` file:

Create a `docker-compose.yml` file

```bash
vim docker-compose.yml
```

Paste the following text into the newly created file

```yaml
version: '3'
services:
  reconcycle-screw-detection:
    container_name: reconcycle-screws
    image: gsavle/reconcycle-screw-detection:latest 
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
    command: tail -f /dev/null
    devices:
      - /dev:/dev
    environment:
      - "ROS_MASTER_URI=http://<ROS master IP>"
      - "ROS_IP=<host IP>"
      - NVIDIA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=all
      - DEBUG_COLORS=true
      - TERM=xterm-256color
      - COLORTERM=truecolor
    tty: true
    privileged: true
    network_mode: "host"
```

*Once again, make sure to fill out the <ROS master IP> and <host IP> fields with the proper information of your particular workcell*

## 3. Running the container

To run the container, navigate to the directory where the `docker-compose.yml` is located.

Run the command below to start the container

```bash
docker compose up
```

### Attaching to the running container

To interact with the software running inside the container, a shell must be attached to it. We can attach a shell by running:

```bash
docker exec -it reconcycle-screws bash
```

A shell will be attached to the running container and you will be located in the `/catkin_ws` workspace

**You can also attach to the container by using VS Code's `dev containers` plugin**

## 4. Using the software

Once you're located in the `/catkin_ws` directory, the workspace needs to be sourced by running

```bash
source devel/setup.bash
```

Now that the workspace has been sourced, the screw detection inference can be ran with the command below:

```bash
rosrun reconcycle-screw-detection inference.py
```

*Prior to running the software, some variables need to be adjusted to suit each users's particular setup*
```python
#TODO
```
