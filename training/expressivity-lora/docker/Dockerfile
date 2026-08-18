FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
# matched stock torch+torchaudio (wheel bundles CUDA 13 runtime; GPU via nvidia runtime)
RUN pip install --break-system-packages --no-cache-dir torch torchaudio
COPY requirements.txt /tmp/qwen-training-requirements.txt
COPY rocm_validation.py /tmp/training_validation.py
RUN pip install --break-system-packages --no-cache-dir -r /tmp/qwen-training-requirements.txt
RUN PYTHONPATH=/tmp python3 -c "from training_validation import check_training_packages; print('Training packages:', check_training_packages())"
RUN python3 -c "import torch, torchaudio; assert torch.version.cuda, 'pip replaced the CUDA torch wheel'; print('CUDA torch preserved:', torch.__version__, torchaudio.__version__, torch.version.cuda)"
WORKDIR /root/qwen-ft
