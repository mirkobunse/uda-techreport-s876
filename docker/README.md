# Jupyter notebooks with PyTorch and CUDA 10.1

If someone else already set up the Docker container for you, please jump to the [Usage](#Usage) section.


## Building and running the Docker image

You first need to build the image and start a container from it.

```
make BUILD_ARGS="--build-arg constraint:node==s876cn01" push

./run.sh
```

where the `BUILD_ARGS` can be omitted or replaced by other arguments for `docker build`. Use `make -n push` to see what happens during the build without starting it. Run `make` without specifying the `push` target to omit pushing the image to the Docker registry that is configured via a `DOCKER_REGISTRY` environment variable. The script `run.sh` allows you to change the default name and resources allocated with the Docker container.

Inside the container, you need to build PyTorch. With our setup at SFB 876, this step would not have been possible during `docker build` because the runtime of each build step is limited. Building PyTorch requires root privileges, so we log in as the root user from our host.

```
docker exec -tiu root <user name>-uda bash
```

And inside the container, we execude the following lines:
```
cd /root
git clone --recursive --branch v1.7.1 https://github.com/pytorch/pytorch
cd pytorch
python setup.py install

pip install git+https://bitbucket.org/mbunse/sigma-test
mkdir /data
chmod go+w /data
sed -i 's|: $(RESOURCE_DIR)|: /data|g' /opt/anaconda3/lib/python3.8/site-packages/sigma/resources/config.yml
```

### Optional: commit a new Docker image

At this point, you can store the container as a new image. This image will have PyTorch readily built, so that additional containers will not need to repeat the steps above!

```
docker commit ${USER}-uda ${USER}/uda-ready
docker tag ${USER}/uda-ready ${DOCKER_REPOSITORY}/${USER}/uda-ready
docker push ${DOCKER_REPOSITORY}/${USER}/uda-ready
docker pull ${DOCKER_REPOSITORY}/${USER}/uda-ready
```

### Optional: starting a Jupyter server

For running the experiments (see `../README.md`) you do not need a Jupyter server, just a terminal to type `make`. However, we have configured a Jupyter server for rapid prototyping in your local web browser.

Logging in as the Jupyter user (`docker attach <user name>-uda`), you can start the Jupyter server:

```
cd
./run_jupyter.sh
```

Store the token that is printed; you will need it to access the web frontend.

Press STRG+P and STRG+Q immediately after another to leave the container without exiting the currently running terminal. The terminal will continue to run the Jupyter server in the background.


## Usage

The Jupyter notebook exposes the port 5555. If your container lives behind a gateway, you will need to forward this port via ssh:

```
ssh -L 5555:<host machine>:5555 <gateway url>
```

at the SFB 876, the `<gateway url>` is `gwkilab` and the `<host machine>` might be `s876gn01`, `s876gn02`, or  `s876gn03`. You can check the host with `docker ps`.

Having forwarded the port, you can access the Jupyter GUI through your browser, as if it were on your local machine. However, all code is executed in the container, with all computational power it provides. To log in, you need the token of the server:

```
http://localhost:5555/?token=<token>
```
