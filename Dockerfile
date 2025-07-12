# Dockerfile per YOLO Pipeline
FROM python:3.9-slim

# Installa dipendenze di sistema necessarie per OpenCV e PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /app

# Copia requirements.txt e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt update && apt install -y libgl1
# Copia tutto il codice sorgente
COPY . .

# Crea le directory necessarie
RUN mkdir -p runs custom_dataset data

CMD ["bash"]

#docker build -t yolo_pipeline .
#with gpu and more size
# docker run --gpus all -it --rm --shm-size=2g yolo_pipeline