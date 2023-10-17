# Use Ubuntu 20.04 as the base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Git (if not already installed)
RUN apt-get update && apt-get install -y git

# Install Opencv Python
RUN pip install opencv-python

# Install numpy using pip
RUN pip install pandas

# Install PyTorch and torchvision
RUN pip install torch torchvision torchaudio

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /Projects

RUN apt-get install -y htop 

RUN git clone -b main https://github.com/rikhsitlladeveloper/Ad_tracking_with_color_filter.git

WORKDIR /Projects/Ad_tracking_with_color_filter
 
RUN apt-get update && apt-get install -y tzdata
ENV TZ=Asia/Tashkent
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN git pull

RUN pip install pyyaml
