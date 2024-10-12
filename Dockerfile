# nvidia docker: include cuda11.4.2, pytorch1.10.0
FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
 && apt-get install -y apt-utils git vim curl ca-certificates bzip2 cmake tmux wget tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
 && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /spatial-relation-benchmark
