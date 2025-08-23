# 🚀 TalentAI Backend

API REST desarrollada con FastAPI para el sistema de recomendación de programas académicos basado en inteligencia artificial.

## 📋 Descripción

TalentAI Backend es una API robusta que proporciona:

- **Generación de datasets** sintéticos para entrenamiento
- **Entrenamiento de modelos** de Machine Learning (KNN y Redes Neuronales)
- **Predicciones** de áreas de conocimiento basadas en competencias
- **Recomendaciones** de programas académicos personalizadas
- **Filtrado avanzado** de programas por múltiples criterios
- **Monitoreo y salud** del sistema

## 🏗️ Arquitectura

### Estructura del Proyecto Completo
```
talent_ai/                 # Directorio raíz del proyecto
├── DOCKER_README.md       # Documentación Docker
└── backend/               # Aplicación backend
    ├── docker-compose.yml # ⚠️ Configuración Docker Compose
    ├── deploy.sh          # ⚠️ Script de despliegue
    ├── .env.example       # Variables de entorno ejemplo
    ├── Dockerfile         # Configuración Docker del backend
    ├── .dockerignore      # Archivos excluidos de Docker
    ├── requirements.txt   # Dependencias Python
    ├── main.py           # Punto de entrada de la aplicación
    ├── app/
    │   ├── core/         # Configuración y logging
    │   ├── database/     # Modelos y conexión DB
    │   ├── routers/      # Endpoints de la API
    │   └── schemas/      # Esquemas Pydantic
    ├── data/             # Archivos CSV de datos
    ├── ml_models/        # Modelos de Machine Learning
    ├── models/           # Modelos entrenados guardados
    ├── logs/             # Archivos de log
    └── scripts/          # Scripts de inicialización
```

### Arquitectura del Backend
```
backend/
├── app/
│   ├── core/              # Configuración y logging
│   ├── database/          # Modelos y conexión DB
│   ├── routers/           # Endpoints de la API
│   └── schemas/           # Esquemas Pydantic
├── data/                  # Archivos CSV de datos
├── ml_models/             # Modelos de Machine Learning
├── models/                # Modelos entrenados guardados
├── logs/                  # Archivos de log
└── scripts/               # Scripts de inicialización
```

## 🛠️ Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **SQLAlchemy** - ORM para base de datos
- **PostgreSQL** - Base de datos principal
- **Scikit-learn** - Machine Learning
- **Pandas** - Procesamiento de datos
- **Uvicorn** - Servidor ASGI
- **Docker** - Containerización

## 📦 Instalación y Configuración

### Prerrequisitos

- Python 3.12+
- PostgreSQL 13+
- Conda (recomendado)

### 🔧 Desarrollo Local

#### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd talent_ai/backend
```

#### 2. Configurar ambiente Conda
```bash
# Crear ambiente
conda create -n talent_ai python=3.12
conda activate talent_ai
```

#### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

#### 4. Configurar variables de entorno
```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus configuraciones
vim .env
```

**Variables principales:**
```env
# Base de datos
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_DB=talentai_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Aplicación
APP_NAME=TalentAI
APP_VERSION=1.0.0
DEBUG=true
ENVIRONMENT=development

# Servidor
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

#### 5. Configurar PostgreSQL
```bash
# Crear base de datos
psql -U postgres
CREATE DATABASE talentai_db;
\q
```

#### 6. Inicializar base de datos
```bash
python scripts/init_database.py
```

#### 7. Ejecutar servidor de desarrollo
```bash
# Activar ambiente
conda activate talent_ai

# Ejecutar servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
```

#### 8. Verificar instalación
```bash
# Health check
curl http://localhost:8000/health

# Documentación interactiva
open http://localhost:8000/docs
```

### 🐳 Despliegue con Docker

> **⚠️ NOTA IMPORTANTE**: Los archivos de Docker (`docker-compose.yml` y `deploy.sh`) están ubicados en el **directorio raíz del proyecto** (`/Users/michaelpage/Documents/Desarrollo/proyectos/talent_ai/`), **NO** en el directorio `/backend`.

#### Opción 1: Docker Compose (Recomendado)

```bash
# Navegar al directorio backend
cd /Users/michaelpage/Documents/Desarrollo/proyectos/talent_ai/backend

# Configurar variables de entorno
cp .env.example .env
# Editar .env según necesidades

# Construir y ejecutar servicios
docker-compose up -d

# Verificar servicios
docker-compose ps

# Ver logs
docker-compose logs -f backend
```

#### Opción 2: Script de Despliegue

```bash
# Navegar al directorio backend
cd /Users/michaelpage/Documents/Desarrollo/proyectos/talent_ai/backend

# Hacer ejecutable el script
chmod +x deploy.sh

# Desplegar aplicación
./deploy.sh start

# Verificar estado
./deploy.sh status

# Ver logs
./deploy.sh logs

# Detener servicios
./deploy.sh stop
```

#### Opción 3: Docker Manual

```bash
# Construir imagen
docker build -t talentai-backend .

# Ejecutar contenedor
docker run -d \
  --name talentai-backend \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  talentai-backend
```

## 🔗 Endpoints Principales

### Salud del Sistema
- `GET /health` - Estado básico
- `GET /health/detailed` - Estado detallado con DB
- `GET /health/readiness` - Preparación del servicio
- `GET /health/liveness` - Verificación de vida

### Dataset
- `POST /dataset/generate` - Generar dataset sintético
- `GET /dataset/validate` - Validar dataset existente

### Entrenamiento
- `POST /training/knn` - Entrenar modelo KNN
- `POST /training/neural-network` - Entrenar red neuronal
- `GET /training/models` - Listar modelos disponibles

### Predicciones
- `POST /predictions/predict` - Realizar predicción
- `GET /predictions/history/{user_email}` - Historial de predicciones

### Programas
- `GET /programs/search` - Buscar programas con filtros
- `GET /programs/recommendations/{user_email}` - Recomendaciones personalizadas
- `GET /programs/areas` - Listar áreas de conocimiento
- `GET /programs/filters` - Obtener filtros disponibles

## 📊 Monitoreo

### Logs
```bash
# Ver logs en tiempo real
tail -f logs/talentai.log

# Ver errores
tail -f logs/errors.log

# Con Docker Compose
docker-compose logs -f backend
```

### Health Checks
```bash
# Estado básico
curl http://localhost:8000/health

# Estado detallado
curl http://localhost:8000/health/detailed

# Métricas de rendimiento
curl http://localhost:8000/health/readiness
```

## 🧪 Testing

```bash
# Activar ambiente
conda activate talent_ai

# Ejecutar tests (cuando estén implementados)
pytest tests/

# Test de endpoints
curl -X POST http://localhost:8000/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 100}'
```

## 🔧 Comandos Útiles

### Desarrollo
```bash
# Reinstalar dependencias
pip install -r requirements.txt --upgrade

# Generar requirements.txt
pip freeze > requirements.txt

# Limpiar cache Python
find . -type d -name "__pycache__" -delete
```

### Docker
```bash
# Reconstruir imagen
docker-compose build --no-cache backend

# Reiniciar servicios
docker-compose restart

# Limpiar volúmenes
docker-compose down -v

# Ver uso de recursos
docker stats
```

### Base de Datos
```bash
# Conectar a PostgreSQL en Docker
docker-compose exec postgres psql -U postgres -d talentai_db

# Backup de base de datos
docker-compose exec postgres pg_dump -U postgres talentai_db > backup.sql

# Restaurar backup
docker-compose exec -T postgres psql -U postgres talentai_db < backup.sql
```

## 🚨 Solución de Problemas

### Error de Conexión a Base de Datos
```bash
# Verificar que PostgreSQL esté ejecutándose
docker-compose ps postgres

# Verificar logs de PostgreSQL
docker-compose logs postgres

# Reiniciar servicio de base de datos
docker-compose restart postgres
```

### Error de Puertos
```bash
# Verificar puertos en uso
lsof -i :8000

# Cambiar puerto en .env
SERVER_PORT=8001
```

### Problemas de Permisos
```bash
# Ajustar permisos de directorios
chmod -R 755 data/ models/ logs/

# Verificar propietario
ls -la data/ models/ logs/
```

## 📚 Documentación Adicional

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Docker Guide**: [DOCKER_README.md](../DOCKER_README.md)

