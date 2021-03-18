ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.02-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/tacotron2
WORKDIR /workspace/tacotron2
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo @TZ > /etc/timezone
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y ffmpeg
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
