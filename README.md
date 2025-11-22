# 🔐 API de Reconocimiento Facial IoT

API REST desarrollada con FastAPI para gestión de usuarios mediante reconocimiento facial utilizando DeepFace.

## 🌟 Características

- ✅ **Reconocimiento facial** con DeepFace (modelo Facenet)
- ✅ **Autenticación por token** con middleware personalizado
- ✅ **CRUD completo de usuarios** con validación de rostros duplicados
- ✅ **Almacenamiento de imágenes** en volumen persistente (Railway)
- ✅ **Historial de acciones** con registro automático vía middleware
- ✅ **Validación de datos** con reglas de negocio
- ✅ **CORS configurado** para aplicaciones frontend
- ✅ **Base de datos MySQL** con SQLAlchemy ORM
- ✅ **Documentación automática** con Swagger UI

## 🛠️ Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **DeepFace** - Librería de reconocimiento facial
- **SQLAlchemy** - ORM para base de datos
- **MySQL** - Base de datos relacional
- **Pydantic** - Validación de datos
- **Pillow** - Procesamiento de imágenes
- **NumPy & SciPy** - Cálculos numéricos y comparación de embeddings

## 📋 Requisitos Previos

- Python 3.10 o superior
- MySQL 8.0 o superior
- pip (gestor de paquetes de Python)

## 🚀 Instalación Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/JoseVG-Dev/Kalacam-Backend
cd Iot-Backend
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

### 3. Activar entorno virtual

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Base de datos
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña
DB_HOST=localhost
DB_PORT=3306
DB_NAME=iot_backend

# Almacenamiento local (opcional, por defecto usa "uploads")
VOLUMEN_PATH=uploads
```

### 6. Crear la base de datos

Conecta a MySQL y crea la base de datos:

```sql
CREATE DATABASE iot_backend CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Las tablas se crearán automáticamente al iniciar la aplicación gracias a SQLAlchemy.

### 7. Ejecutar la aplicación

```bash
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

## 📚 Documentación de la API

Una vez iniciada la aplicación, accede a la documentación interactiva:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 🔑 Endpoints Principales

### Autenticación

#### `POST /compararCara`
Compara un rostro con los usuarios registrados y retorna un token si es reconocido.

**Body:** `multipart/form-data`
- `imagen` (file): Imagen del rostro (JPEG/PNG)

**Response:**
```json
{
  "token": "Hola Juan, token: 123456"
}
```

#### `GET /generarToken`
Genera un token de prueba (solo para desarrollo).

**Response:**
```json
{
  "token": "123456",
  "mensaje": "Token generado exitosamente para pruebas"
}
```

### Usuarios

#### `POST /subirUsuario`
Crea un nuevo usuario con reconocimiento facial.

**Body:** `multipart/form-data`
- `nombre` (string): Nombre del usuario
- `apellido` (string): Apellido del usuario
- `email` (string): Email único
- `imagen` (file): Imagen del rostro (JPEG/PNG)

**Response:**
```json
{
  "mensaje": "El usuario Juan Pérez, ha sido creado exitosamente"
}
```

#### `GET /usuarios` 🔒
Lista todos los usuarios (requiere autenticación).

**Headers:**
```
Authorization: Bearer <token>
```

#### `GET /usuarios/{usuario_id}` 🔒
Obtiene un usuario específico (requiere autenticación).

#### `PUT /usuarios/{usuario_id}` 🔒
Actualiza los datos de un usuario (requiere autenticación).

**Body:** `multipart/form-data`
- `nombre` (string, opcional)
- `apellido` (string, opcional)
- `email` (string, opcional)
- `imagen` (file, opcional)

#### `DELETE /usuarios/{usuario_id}` 🔒
Elimina un usuario y su imagen asociada (requiere autenticación).

### Imágenes

#### `GET /imagenes/{ruta:path}` 🔒
Sirve una imagen del volumen (requiere autenticación).

### Historial

#### `GET /historial` 🔒
Obtiene el historial de acciones (requiere autenticación).

🔒 = Requiere token de autenticación en el header `Authorization: Bearer <token>`

## 🏗️ Estructura del Proyecto

```
Iot-Backend/
├── database/
│   └── database.py          # Configuración de SQLAlchemy
├── middleware/
│   ├── auth_middleware.py   # Middleware de autenticación
│   └── historial_middleware.py  # Middleware de historial
├── model/
│   └── models.py            # Modelos de datos (Usuario, Historial)
├── repository/
│   ├── usuario_repository.py    # CRUD de usuarios
│   └── historial_repository.py  # CRUD de historial
├── service/
│   ├── usuario_service.py   # Lógica de negocio de usuarios
│   ├── token_service.py     # Gestión de tokens
│   └── storage_service.py   # Gestión de almacenamiento de imágenes
├── uploads/                 # Imágenes locales (gitignored)
├── .env                     # Variables de entorno (gitignored)
├── .gitignore
├── main.py                  # Punto de entrada de la aplicación
├── requirements.txt         # Dependencias del proyecto
├── Dockerfile               # Configuración de Docker
└── README.md
```

## 🌐 Despliegue en Railway

### 1. Crear Volumen
En el Dashboard de Railway:
- Agrega un **Volume** (almacenamiento persistente)
- Monta el volumen en la ruta: `/data`

### 2. Configurar Variables de Entorno

Agrega las siguientes variables en Railway:

```env
DB_USER=<usuario_mysql>
DB_PASSWORD=<contraseña_mysql>
DB_HOST=<host_mysql>
DB_PORT=3306
DB_NAME=<nombre_base_datos>
VOLUMEN_PATH=/data
```

### 3. Conectar Repositorio

Railway detectará automáticamente el `Dockerfile` y construirá la aplicación.

## 🔐 Seguridad

- **Tokens en memoria:** Actualmente los tokens se almacenan en memoria. Para producción, considera usar:
  - Redis para gestión de sesiones
  - JWT (JSON Web Tokens) con firma criptográfica
  - Base de datos con tabla de sesiones

- **CORS:** Configurado para permitir solo orígenes específicos en `main.py`:
  ```python
  origins = [
      "http://localhost:3000",
      "http://127.0.0.1:3000",
  ]
  ```

## 🧪 Validaciones

### Rostros Duplicados
El sistema valida que no se registren rostros duplicados utilizando:
- **Umbral de similitud:** 0.37 (distancia de coseno)
- **Comparación automática** con todos los usuarios existentes
- **Exclusión en actualizaciones** para permitir actualizar la imagen del mismo usuario

### Datos de Usuario
- Nombre y apellido: máximo 100 caracteres
- Email: formato válido y único
- Imagen: solo JPEG y PNG

## 📝 Notas de Desarrollo

### Activar entorno virtual
```bash
# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### Agregar nuevas dependencias
```bash
pip install nombre_paquete
pip freeze > requirements.txt
```

### Estructura de Imágenes
Las imágenes se almacenan con nombres UUID únicos:
```
uploads/images/usuarios/
├── a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg
├── b2c3d4e5-f6g7-8901-bcde-fg2345678901.png
└── ...
```

La base de datos solo guarda la ruta relativa: `usuarios/uuid.jpg`

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 🐛 Reportar Issues


---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!

