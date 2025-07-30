FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.5.1
ENV PYTHONIOENCODING="utf-8"
ENV AMDGPU_TARGETS="gfx1030"
SHELL ["/bin/bash", "-c"]
RUN apt update && apt -y upgrade && apt -y install python3-venv
RUN python3 -m venv /venv
WORKDIR /venv
RUN source bin/activate
RUN pip install --upgrade pip
RUN pip install jupyter pandas scikit-learn transformers datasets transformers[torch]
RUN pip install tensorflow-rocm==2.18.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/ --upgrade

WORKDIR /tmp
RUN git clone --recurse https://github.com/ROCm/bitsandbytes.git
WORKDIR /tmp/bitsandbytes
RUN git checkout rocm_enabled_multi_backend
RUN pip install -r requirements-dev.txt
RUN cmake -DCOMPUTE_BACKEND=hip -S .
RUN make
RUN pip install .

RUN pip install --root-user-action=ignore "huggingface_hub[cli]"
ADD requirements.txt /tmp/requirements.txt
RUN pip install --root-user-action=ignore -r /tmp/requirements.txt
RUN pip install --upgrade boto3 botocore
# https://github.com/Unbabel/COMET/issues/250
RUN pip install --upgrade jsonargparse
#RUN python3 -m spacy download de_core_news_sm
VOLUME ["/data"]
WORKDIR /data
CMD ["bash"]