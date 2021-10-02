FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt update && apt -y upgrade && apt install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

EXPOSE 8501
EXPOSE 8888

CMD bash
