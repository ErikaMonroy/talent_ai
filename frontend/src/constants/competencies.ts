import { Competency } from '@/types/assessment';

// Competencias completas según FORMULARIO_COMPLETO.md
export const COMPETENCIES: Competency[] = [
  // DIMENSIÓN 1: RAZONAMIENTO LÓGICO-MATEMÁTICO (8 competencias)
  {
    id: "C001",
    title: "Realizar operaciones matemáticas mentalmente o con papel",
    description: "Cálculo y Operaciones Numéricas",
    dimensionId: 1
  },
  {
    id: "C002",
    title: "Resolver ecuaciones con variables (x, y, etc.)",
    description: "Álgebra y Resolución de Ecuaciones",
    dimensionId: 1
  },
  {
    id: "C003",
    title: "Entender formas, volúmenes y espacios en 3D",
    description: "Geometría y Visualización Espacial",
    dimensionId: 1
  },
  {
    id: "C004",
    title: "Interpretar gráficos, tablas y calcular promedios",
    description: "Estadística y Análisis de Datos",
    dimensionId: 1
  },
  {
    id: "C005",
    title: "Seguir secuencias lógicas y sacar conclusiones",
    description: "Lógica Formal y Deductiva",
    dimensionId: 1
  },
  {
    id: "C006",
    title: "Abordar problemas complejos paso a paso",
    description: "Resolución Sistemática de Problemas",
    dimensionId: 1
  },
  {
    id: "C007",
    title: "Descomponer situaciones complejas en partes simples",
    description: "Pensamiento Analítico",
    dimensionId: 1
  },
  {
    id: "C008",
    title: "Representar situaciones reales con fórmulas",
    description: "Modelado Matemático",
    dimensionId: 1
  },

  // DIMENSIÓN 2: COMUNICACIÓN Y LENGUAJE (10 competencias)
  {
    id: "C009",
    title: "Hablar en público y expresar ideas claramente",
    description: "Expresión Oral y Oratoria",
    dimensionId: 2
  },
  {
    id: "C010",
    title: "Escribir textos claros, coherentes y bien estructurados",
    description: "Redacción y Escritura",
    dimensionId: 2
  },
  {
    id: "C011",
    title: "Entender y analizar textos complejos",
    description: "Comprensión Lectora",
    dimensionId: 2
  },
  {
    id: "C012",
    title: "Defender puntos de vista con razones sólidas",
    description: "Argumentación y Debate",
    dimensionId: 2
  },
  {
    id: "C013",
    title: "Prestar atención y entender a otros cuando hablan",
    description: "Escucha Activa",
    dimensionId: 2
  },
  {
    id: "C014",
    title: "Preparar y realizar presentaciones atractivas",
    description: "Presentaciones Efectivas",
    dimensionId: 2
  },
  {
    id: "C015",
    title: "Mantener conversaciones básicas en inglés",
    description: "Inglés Conversacional",
    dimensionId: 2
  },
  {
    id: "C016",
    title: "Leer textos académicos y técnicos en inglés",
    description: "Inglés Técnico y Académico",
    dimensionId: 2
  },
  {
    id: "C017",
    title: "Comunicarse en idiomas además del inglés",
    description: "Otros Idiomas Extranjeros",
    dimensionId: 2
  },
  {
    id: "C018",
    title: "Adaptarse a diferentes culturas al comunicarse",
    description: "Comunicación Intercultural",
    dimensionId: 2
  },

  // DIMENSIÓN 3: CIENCIAS Y TECNOLOGÍA (12 competencias)
  {
    id: "C019",
    title: "Entender movimiento, energía y fuerzas",
    description: "Física y Fenómenos Naturales",
    dimensionId: 3
  },
  {
    id: "C020",
    title: "Comprender reacciones químicas y composiciones",
    description: "Química y Procesos",
    dimensionId: 3
  },
  {
    id: "C021",
    title: "Conocer sistemas biológicos y vida natural",
    description: "Biología y Ciencias de la Vida",
    dimensionId: 3
  },
  {
    id: "C022",
    title: "Diseñar experimentos y probar hipótesis",
    description: "Método Científico",
    dimensionId: 3
  },
  {
    id: "C023",
    title: "Notar detalles y realizar experimentos",
    description: "Observación y Experimentación",
    dimensionId: 3
  },
  {
    id: "C024",
    title: "Usar equipos científicos y seguir protocolos",
    description: "Análisis de Laboratorio",
    dimensionId: 3
  },
  {
    id: "C025",
    title: "Crear código y aplicaciones de software",
    description: "Programación y Desarrollo",
    dimensionId: 3
  },
  {
    id: "C026",
    title: "Entender computadoras, hardware y software",
    description: "Sistemas Digitales",
    dimensionId: 3
  },
  {
    id: "C027",
    title: "Configurar conexiones de internet y redes",
    description: "Redes y Conectividad",
    dimensionId: 3
  },
  {
    id: "C028",
    title: "Trabajar con bases de datos y análisis",
    description: "Análisis de Datos Digitales",
    dimensionId: 3
  },
  {
    id: "C029",
    title: "Adaptarse rápidamente a nuevas tecnologías",
    description: "Innovación Tecnológica",
    dimensionId: 3
  },
  {
    id: "C030",
    title: "Proteger información y sistemas digitales",
    description: "Ciberseguridad",
    dimensionId: 3
  },

  // DIMENSIÓN 4: HUMANIDADES Y CIENCIAS SOCIALES (12 competencias)
  {
    id: "C031",
    title: "Entender problemas sociales y culturales",
    description: "Análisis Social y Cultural",
    dimensionId: 4
  },
  {
    id: "C032",
    title: "Conocer eventos históricos y su importancia",
    description: "Historia y Contexto",
    dimensionId: 4
  },
  {
    id: "C033",
    title: "Ubicación espacial y conocimiento geográfico",
    description: "Geografía y Territorio",
    dimensionId: 4
  },
  {
    id: "C034",
    title: "Reflexionar sobre valores y dilemas morales",
    description: "Filosofía y Ética",
    dimensionId: 4
  },
  {
    id: "C035",
    title: "Entender por qué las personas actúan así",
    description: "Psicología y Comportamiento",
    dimensionId: 4
  },
  {
    id: "C036",
    title: "Apreciar diferentes culturas y tradiciones",
    description: "Antropología Cultural",
    dimensionId: 4
  },
  {
    id: "C037",
    title: "Entender y manejar emociones propias y ajenas",
    description: "Empatía e Inteligencia Emocional",
    dimensionId: 4
  },
  {
    id: "C038",
    title: "Colaborar efectivamente con otros",
    description: "Trabajo en Equipo",
    dimensionId: 4
  },
  {
    id: "C039",
    title: "Guiar y motivar a otros espontáneamente",
    description: "Liderazgo Natural",
    dimensionId: 4
  },
  {
    id: "C040",
    title: "Ayudar a llegar a acuerdos entre partes",
    description: "Negociación y Mediación",
    dimensionId: 4
  },
  {
    id: "C041",
    title: "Manejar disputas y tensiones interpersonales",
    description: "Resolución de Conflictos",
    dimensionId: 4
  },
  {
    id: "C042",
    title: "Hacer conexiones y mantener relaciones",
    description: "Construcción de Redes Sociales",
    dimensionId: 4
  },

  // DIMENSIÓN 5: CREATIVIDAD Y ARTE (11 competencias)
  {
    id: "C043",
    title: "Crear imágenes y representaciones gráficas",
    description: "Dibujo y Representación Visual",
    dimensionId: 5
  },
  {
    id: "C044",
    title: "Organizar elementos visuales de forma atractiva",
    description: "Diseño y Composición",
    dimensionId: 5
  },
  {
    id: "C045",
    title: "Tocar instrumentos, cantar o crear música",
    description: "Música y Expresión Sonora",
    dimensionId: 5
  },
  {
    id: "C046",
    title: "Crear historias, poemas o textos originales",
    description: "Escritura Creativa y Narrativa",
    dimensionId: 5
  },
  {
    id: "C047",
    title: "Actuar, expresarse con el cuerpo y gestos",
    description: "Teatro y Expresión Corporal",
    dimensionId: 5
  },
  {
    id: "C048",
    title: "Crear contenido visual y multimedia",
    description: "Fotografía y Medios Audiovisuales",
    dimensionId: 5
  },
  {
    id: "C049",
    title: "Generar múltiples ideas originales",
    description: "Pensamiento Divergente",
    dimensionId: 5
  },
  {
    id: "C050",
    title: "Crear soluciones en el momento",
    description: "Improvisación y Espontaneidad",
    dimensionId: 5
  },
  {
    id: "C051",
    title: "Encontrar formas nuevas de hacer las cosas",
    description: "Innovación en Soluciones",
    dimensionId: 5
  },
  {
    id: "C052",
    title: "Reconocer y valorar la belleza y el arte",
    description: "Apreciación Estética",
    dimensionId: 5
  },
  {
    id: "C053",
    title: "Probar nuevas técnicas y estilos creativos",
    description: "Experimentación Artística",
    dimensionId: 5
  },

  // DIMENSIÓN 6: GESTIÓN Y EMPRENDIMIENTO (15 competencias)
  {
    id: "C054",
    title: "Crear planes a largo plazo y objetivos",
    description: "Planificación Estratégica",
    dimensionId: 6
  },
  {
    id: "C055",
    title: "Administrar tiempo, dinero y materiales",
    description: "Gestión de Recursos",
    dimensionId: 6
  },
  {
    id: "C056",
    title: "Entender números, costos y presupuestos",
    description: "Análisis Financiero",
    dimensionId: 6
  },
  {
    id: "C057",
    title: "Asegurar que las cosas se hagan bien",
    description: "Control de Calidad",
    dimensionId: 6
  },
  {
    id: "C058",
    title: "Coordinar actividades y equipos de trabajo",
    description: "Gestión de Proyectos",
    dimensionId: 6
  },
  {
    id: "C059",
    title: "Decidir rápida y efectivamente bajo presión",
    description: "Toma de Decisiones",
    dimensionId: 6
  },
  {
    id: "C060",
    title: "Convencer a otros de adoptar tus ideas",
    description: "Persuasión e Influencia",
    dimensionId: 6
  },
  {
    id: "C061",
    title: "Satisfacer necesidades y expectativas",
    description: "Atención al Cliente",
    dimensionId: 6
  },
  {
    id: "C062",
    title: "Entender qué quiere y necesita la gente",
    description: "Investigación de Mercados",
    dimensionId: 6
  },
  {
    id: "C063",
    title: "Promocionar ideas, productos o servicios",
    description: "Marketing y Promoción",
    dimensionId: 6
  },
  {
    id: "C064",
    title: "Mantener buenas relaciones de negocios",
    description: "Relaciones Comerciales",
    dimensionId: 6
  },
  {
    id: "C065",
    title: "Identificar oportunidades comerciales",
    description: "Visión de Negocios",
    dimensionId: 6
  },
  {
    id: "C066",
    title: "Manejar incertidumbre y tomar riesgos calculados",
    description: "Tolerancia al Riesgo",
    dimensionId: 6
  },
  {
    id: "C067",
    title: "Crear nuevos modelos y formas de hacer negocios",
    description: "Innovación Comercial",
    dimensionId: 6
  },
  {
    id: "C068",
    title: "No rendirse ante obstáculos y dificultades",
    description: "Persistencia y Determinación",
    dimensionId: 6
  },

  // DIMENSIÓN 7: HABILIDADES TÉCNICAS Y OPERATIVAS (12 competencias)
  {
    id: "C069",
    title: "Trabajos que requieren precisión con las manos",
    description: "Destreza Manual Fina",
    dimensionId: 7
  },
  {
    id: "C070",
    title: "Arreglar y mantener máquinas y equipos",
    description: "Mecánica y Reparación",
    dimensionId: 7
  },
  {
    id: "C071",
    title: "Construir, armar y ensamblar estructuras",
    description: "Construcción y Edificación",
    dimensionId: 7
  },
  {
    id: "C072",
    title: "Trabajar con cables, circuitos y electricidad",
    description: "Sistemas Eléctricos",
    dimensionId: 7
  },
  {
    id: "C073",
    title: "Usar herramientas manuales y eléctricas",
    description: "Manejo de Herramientas",
    dimensionId: 7
  },
  {
    id: "C074",
    title: "Realizar trabajos que requieren exactitud",
    description: "Precisión Técnica",
    dimensionId: 7
  },
  {
    id: "C075",
    title: "Organizar espacios y distribuir eficientemente",
    description: "Organización Espacial",
    dimensionId: 7
  },
  {
    id: "C076",
    title: "Controlar stock, entradas y salidas",
    description: "Gestión de Inventarios",
    dimensionId: 7
  },
  {
    id: "C077",
    title: "Coordinar movimiento de personas y cosas",
    description: "Logística y Transporte",
    dimensionId: 7
  },
  {
    id: "C078",
    title: "Seguir instrucciones y protocolos exactamente",
    description: "Seguimiento de Procedimientos",
    dimensionId: 7
  },
  {
    id: "C079",
    title: "Supervisar que las operaciones funcionen bien",
    description: "Control de Procesos Operativos",
    dimensionId: 7
  },
  {
    id: "C080",
    title: "Anticipar y prevenir problemas técnicos",
    description: "Mantenimiento Preventivo",
    dimensionId: 7
  },

  // DIMENSIÓN 8: CUIDADO Y SERVICIO (20 competencias)
  {
    id: "C081",
    title: "Entender y conectar con los sentimientos de otros",
    description: "Empatía y Comprensión",
    dimensionId: 8
  },
  {
    id: "C082",
    title: "Mantener la calma en situaciones difíciles",
    description: "Paciencia y Tolerancia",
    dimensionId: 8
  },
  {
    id: "C083",
    title: "Expresarse de manera clara y respetuosa",
    description: "Comunicación Asertiva",
    dimensionId: 8
  },
  {
    id: "C084",
    title: "Prestar atención completa a lo que otros dicen",
    description: "Escucha Activa",
    dimensionId: 8
  },
  {
    id: "C085",
    title: "Mediar y solucionar problemas entre personas",
    description: "Resolución de Conflictos",
    dimensionId: 8
  },
  {
    id: "C086",
    title: "Ayudar con higiene, alimentación y bienestar",
    description: "Cuidado Personal",
    dimensionId: 8
  },
  {
    id: "C087",
    title: "Responder a emergencias médicas básicas",
    description: "Primeros Auxilios",
    dimensionId: 8
  },
  {
    id: "C088",
    title: "Cuidados de salud elementales",
    description: "Atención Médica Básica",
    dimensionId: 8
  },
  {
    id: "C089",
    title: "Brindar consuelo y apoyo psicológico",
    description: "Apoyo Emocional",
    dimensionId: 8
  },
  {
    id: "C090",
    title: "Guiar y aconsejar a personas en dificultades",
    description: "Orientación y Consejería",
    dimensionId: 8
  },
  {
    id: "C091",
    title: "Colaborar efectivamente con otros",
    description: "Trabajo en Equipo",
    dimensionId: 8
  },
  {
    id: "C092",
    title: "Compromiso con el bienestar comunitario",
    description: "Responsabilidad Social",
    dimensionId: 8
  },
  {
    id: "C093",
    title: "Ética y Valores",
    description: "Actuar con integridad y principios morales",
    dimensionId: 8
  },
  {
    id: "C094",
    title: "Servicio al Cliente",
    description: "Atender necesidades con excelencia",
    dimensionId: 8
  },
  {
    id: "C095",
    title: "Hospitalidad",
    description: "Recibir y atender visitantes cálidamente",
    dimensionId: 8
  },
  {
    id: "C096",
    title: "Organización de Eventos",
    description: "Planificar y coordinar actividades grupales",
    dimensionId: 8
  },
  {
    id: "C097",
    title: "Animación y Entretenimiento",
    description: "Crear ambiente alegre y divertido",
    dimensionId: 8
  },
  {
    id: "C098",
    title: "Deportes y Actividad Física",
    description: "Promover ejercicio y vida saludable",
    dimensionId: 8
  },
  {
    id: "C099",
    title: "Educación y Enseñanza",
    description: "Transmitir conocimientos y habilidades",
    dimensionId: 8
  },
  {
    id: "C100",
    title: "Mentoría y Coaching",
    description: "Guiar el desarrollo personal y profesional",
    dimensionId: 8
  }
];

// Función para obtener competencias por dimensión
export const getCompetenciesByDimension = (dimensionId: number): Competency[] => {
  return COMPETENCIES.filter(competency => competency.dimensionId === dimensionId);
};

// Función para obtener una competencia por ID
export const getCompetencyById = (competencyId: string): Competency | undefined => {
  return COMPETENCIES.find(competency => competency.id === competencyId);
};