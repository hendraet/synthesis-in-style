FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        pkg-config \
        netcat \
        zsh

ARG UNAME=user
ARG UID=1000
ARG GID=100

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

ARG BASE=/app
RUN mkdir -p ${BASE}
WORKDIR ${BASE}

COPY requirements_docker.txt requirements.txt
RUN pip install -r requirements.txt

ARG WAIT_DIR=/opt/wait-for
RUN git clone https://github.com/eficode/wait-for.git ${WAIT_DIR}
RUN chmod +x ${WAIT_DIR}/wait-for

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
