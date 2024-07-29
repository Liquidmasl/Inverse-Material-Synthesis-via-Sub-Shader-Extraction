FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel


RUN pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11==24.6.* dask-cudf-cu11==24.6.* cuml-cu11==24.6.* \
    cugraph-cu11==24.6.* cuspatial-cu11==24.6.* cuproj-cu11==24.6.* \
    cuxfilter-cu11==24.6.* cucim-cu11==24.6.* pylibraft-cu11==24.6.* \
    raft-dask-cu11==24.6.* cuvs-cu11==24.6.*


ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "loadvgg.py"]