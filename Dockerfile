FROM rocm/pytorch:latest
ENV PYTHONIOENCODING="utf-8"
SHELL ["/bin/bash", "-c"]
RUN apt update && apt -y upgrade && apt -y install python3-venv
RUN python3 -m venv /venv
WORKDIR /venv
RUN source bin/activate
RUN pip install --upgrade pip
RUN pip install jupyter pandas scikit-learn transformers datasets transformers[torch]
RUN pip install tensorflow-rocm==2.17.0 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/ --upgrade
WORKDIR /tmp
RUN git clone --recurse https://github.com/ROCm/bitsandbytes.git
WORKDIR /tmp/bitsandbytes
RUN git checkout rocm_enabled_multi_backend
RUN pip install -r requirements-dev.txt
RUN cmake -DCOMPUTE_BACKEND=hip -S .
RUN make
RUN python setup.py install

RUN pip install --root-user-action=ignore "huggingface_hub[cli]"
ADD requirements.txt /tmp/requirements.txt
RUN pip install --root-user-action=ignore -r /tmp/requirements.txt
VOLUME ["/data"]
WORKDIR /data
CMD ["bash"]