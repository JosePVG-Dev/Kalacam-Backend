
from fastapi import FastAPI, UploadFile, File, HTTPException
from services import face_service
app = FastAPI()


@app.post("/subirCara")
async def subir_cara(imagen: UploadFile = File(...)):
    if imagen.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido")


    contenido = await imagen.read()

    resultado = face_service.validarRostro(contenido)

    return {"mensaje": "Imagen recibida correctamente", "filename": imagen.filename}


