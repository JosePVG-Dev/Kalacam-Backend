# Modelos de DeepFace

Esta carpeta contiene los modelos pre-entrenados de DeepFace para reconocimiento facial.

## Estructura

```
models/
└── weights/
    ├── arcface_weights.h5      # Modelo ArcFace (99.41% precisión)
    ├── facenet_weights.h5      # Modelo Facenet (99.20% precisión)
    ├── facenet512_weights.h5   # Modelo Facenet512 (99.65% precisión)
    └── retinaface/             # Detector RetinaFace
        ├── Resnet50_Final.pth
        └── mobilenet0.25_Final.pth
```

## Descargar modelos

### ArcFace
```
https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5
```

### Facenet
```
https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5
```

### Facenet512
```
https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5
```

### VGG-Face
```
https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
```

## Instrucciones

1. Descarga el archivo `.h5` del modelo que necesites
2. Colócalo en la carpeta `weights/`
3. El código copiará automáticamente estos modelos al volumen de Railway al iniciar

## Nota

Los modelos son archivos grandes (100-200 MB cada uno). Si usas Git, considera usar Git LFS para estos archivos.

