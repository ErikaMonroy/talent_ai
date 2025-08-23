# TalentAI Frontend

Aplicación web de evaluación de competencias que permite a estudiantes realizar un assessment de 100 preguntas distribuidas en 8 dimensiones cognitivas. Utiliza inteligencia artificial para predecir las mejores opciones de programas académicos basándose en las respuestas del usuario.

## Tecnologías

- **Framework**:  Next.js 14 + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui
- **Estado**: Zustand
- **Formularios**: React Hook Form
- **HTTP Client**: Axios
- **Iconos**: Lucide React
- **Gráficos**: Recharts
- **Animaciones**: Framer Motion

## Características Principales

- Formulario de evaluación de 100 preguntas en 8 dimensiones
- Sistema de persistencia local con auto-guardado
- Navegación intuitiva entre dimensiones
- Visualización de resultados con gráficos interactivos
- Recomendaciones de programas académicos
- Diseño responsive y accesible
- Modo oscuro

## Instalación y Configuración

### Prerrequisitos

- Node.js 18+ 
- npm o yarn

### Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd frontend

# Instalar dependencias
npm install

# Configurar variables de entorno
cp .env.example .env
```

### Variables de Entorno

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=TalentAI
VITE_VERSION=1.0.0
```

### Desarrollo

```bash
# Iniciar servidor de desarrollo
npm run dev

# Abrir en http://localhost:5173
```

### Construcción

```bash
# Construir para producción
npm run build

# Previsualizar build
npm run preview
```

## Estructura del Proyecto

```
src/
├── components/          # Componentes reutilizables
│   ├── ui/             # Componentes base (shadcn/ui)
│   ├── assessment/     # Componentes del formulario
│   └── results/        # Componentes de resultados
├── pages/              # Páginas de la aplicación
├── hooks/              # Custom hooks
├── store/              # Estado global (Zustand)
├── services/           # Servicios API
├── types/              # Definiciones TypeScript
└── utils/              # Utilidades
```

## API Backend

Esta aplicación se conecta con el backend de TalentAI que debe estar ejecutándose en `http://localhost:8000`. Consulta la documentación del backend para instrucciones de instalación.
