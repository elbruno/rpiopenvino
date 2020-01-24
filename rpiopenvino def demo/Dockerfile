FROM balenalib/rpi-raspbian:latest

ARG DOWNLOAD_LINK=https://download.01.org/opencv/2019/openvinotoolkit/R2/l_openvino_toolkit_runtime_raspbian_p_2019.2.242.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG BIN_FILE=https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
ARG WEIGHTS_FILE=https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
# ARG IMAGE_FILE_0=https://pixabay.com/images/download/girl-237871_640.jpg
# ARG IMAGE_FILE_1=https://pixabay.com/images/download/human-740259_640.jpg
ARG IMAGE_FILE_0=https://raw.githubusercontent.com/elbruno/rpiopenvino/master/images/girl-237871_640.jpg
ARG IMAGE_FILE_1=https://raw.githubusercontent.com/elbruno/rpiopenvino/master/images/human-740259_640.jpg

RUN apt-get update && apt-get install -y --no-install-recommends \
	apt-utils \
	automake \
	cmake \
	cpio \
	gcc \
	g++ \
	libatlas-base-dev \
	libgtk-3-dev \
	libstdc++6 \
	libtool \
	libusb-1.0.0-dev \
	lsb-release \
	make \
	python3-pip \
	python3-numpy \
	python3-scipy \
	sudo \
	udev \
	unzip \
	vim \
	wget && \
	rm -rf /var/lib/apt/lists/*
RUN mkdir -p $INSTALL_DIR && cd $INSTALL_DIR && \
	wget -c $DOWNLOAD_LINK && \
	tar xf l_openvino_toolkit_runtime_raspbian_p*.tgz --strip 1 -C $INSTALL_DIR 
# add USB rules
RUN sudo usermod -a -G users "$(whoami)" && \
	/bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && \
	sh $INSTALL_DIR/install_dependencies/install_NCS_udev_rules.sh"
# build Object Detection sample
RUN echo "source /opt/intel/openvino/bin/setupvars.sh" >> ~/.bashrc && \
    mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && \
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=armv7-a" \
	$INSTALL_DIR/deployment_tools/inference_engine/samples && \
	make -j2 object_detection_sample_ssd" && \ 
    wget --no-check-certificate $BIN_FILE && \
    wget --no-check-certificate $WEIGHTS_FILE && \
    wget --no-check-certificate $IMAGE_FILE_0 && \
    wget --no-check-certificate $IMAGE_FILE_1
# Create script to run Inference of Face Detection Model Using OpenCV* API
RUN echo "import cv2 as cv \n\
# Load the model. \n\
net = cv.dnn.readNet('face-detection-adas-0001.xml',\
'face-detection-adas-0001.bin') \n\
# Specify target device. \n\
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD) \n\
# Read an image. \n\
frame = cv.imread('human-740259_640.jpg') \n\
if frame is None: \n\
    raise Exception('Image not found!') \n\
# Prepare input blob and perform an inference. \n\
blob = cv.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv.CV_8U) \n\
net.setInput(blob) \n\
out = net.forward() \n\
# Draw detected faces on the frame. \n\
for detection in out.reshape(-1, 7): \n\
    confidence = float(detection[2]) \n\
    xmin = int(detection[3] * frame.shape[1]) \n\
    ymin = int(detection[4] * frame.shape[0]) \n\
    xmax = int(detection[5] * frame.shape[1]) \n\
    ymax = int(detection[6] * frame.shape[0]) \n\
    if confidence > 0.5: \n\
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0)) \n\
# Save the frame to an image file. \n\
cv.imwrite('out.png', frame) \n\
print('Detection results in out.png')"  >> $INSTALL_DIR/deployment_tools/inference_engine/samples/build/openvino_fd_myriad.py
