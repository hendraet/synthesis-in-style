FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	build-essential \
	git \
	software-properties-common \
	pkg-config \
	unzip \
	wget \
	libgl1-mesa-glx \
	libgl1 \
	zsh \
	python3-pip \
	ninja-build \
	cython

ARG UNAME=user
ARG UID=10001
ARG GID=100

RUN groupadd -g $GID -o $UNAME && \
    useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

RUN mkdir /data
ARG BASE=/app
RUN mkdir -p ${BASE}

COPY stylegan_code_finder/requirements.txt ${BASE}/requirements.txt

WORKDIR ${BASE}
RUN pip3 install -r requirements.txt  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 uninstall -y pycocotools && pip3 install pycocotools==2.0.0

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
