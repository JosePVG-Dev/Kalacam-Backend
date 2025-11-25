FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements para aprovechar cache de Docker
COPY requirements-prod.txt .

# Instalar dependencias en directorio temporal
RUN pip install --no-cache-dir --user -r requirements-prod.txt

# Imagen final
FROM python:3.11-slim

# Instalar solo dependencias de runtime del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /root/.local

# Copiar código de la aplicación
COPY . /app

# Crear directorio para modelos en el volumen y crear symlink
# DeepFace busca modelos en ~/.deepface (hardcodeado en código fuente)
# Creamos un symlink para que apunte a nuestro volumen persistente
RUN mkdir -p /data/models/deepface/.deepface/weights && \
    ln -sf /data/models/deepface/.deepface /root/.deepface

# PATCH: Modificar DeepFace para usar nuestra URL personalizada
# El link original de DeepFace está caído, usamos nuestro backup
RUN if [ -f /root/.local/lib/python3.11/site-packages/deepface/basemodels/ArcFace.py ]; then \
    sed -i 's|https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY|https://drive.google.com/uc?id=1mjLC2mBJz71SDWnTcYTbrqE27RtOmMTk|g' \
    /root/.local/lib/python3.11/site-packages/deepface/basemodels/ArcFace.py; \
    echo "✅ DeepFace ArcFace URL patched"; \
    fi

# Asegurar que el PATH incluya las dependencias instaladas
ENV PATH=/root/.local/bin:$PATH
ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]