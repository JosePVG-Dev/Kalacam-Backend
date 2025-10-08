from deepface import DeepFace
from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import HTTPException

def validarRostro(contenido: bytes) -> str:
    if not contenido:
        raise HTTPException(status_code=400, detail="No se envió contenido de imagen")

    try:
        img = Image.open(BytesIO(contenido))
        resultado = DeepFace.represent(img_path=np.array(img), model_name="Facenet")

        if not resultado or "embedding" not in resultado[0]:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro")

        embedding = resultado[0]["embedding"]

        if len(embedding) == 0:
            raise HTTPException(status_code=400, detail="Rostro inválido o embedding vacío")

        return ",".join(map(str, embedding))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"No se detectó ningún rostro: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
