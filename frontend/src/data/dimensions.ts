import { Dimension, Competency } from '../types/assessment';

export const COMPETENCIES: Competency[] = [
  // Liderazgo (Dimensión 1)
  {
    id: 'C001',
    title: 'Tengo facilidad para dirigir equipos de trabajo',
    description: 'Evalúa la capacidad natural para liderar grupos',
    dimensionId: 1
  },
  {
    id: 'C002',
    title: 'Soy capaz de motivar a otros para alcanzar objetivos',
    description: 'Mide la habilidad para inspirar y motivar',
    dimensionId: 1
  },
  {
    id: 'C003',
    title: 'Tomo decisiones difíciles cuando es necesario',
    description: 'Evalúa la capacidad de toma de decisiones bajo presión',
    dimensionId: 1
  },
  {
    id: 'C004',
    title: 'Delego responsabilidades de manera efectiva',
    description: 'Mide la habilidad para delegar apropiadamente',
    dimensionId: 1
  },
  {
    id: 'C005',
    title: 'Inspiro confianza en mi equipo',
    description: 'Evalúa la capacidad para generar confianza',
    dimensionId: 1
  },
  // Comunicación (Dimensión 2)
  {
    id: 'C006',
    title: 'Me expreso de manera clara y concisa',
    description: 'Evalúa la claridad en la comunicación',
    dimensionId: 2
  },
  {
    id: 'C007',
    title: 'Escucho activamente a los demás',
    description: 'Mide la capacidad de escucha activa',
    dimensionId: 2
  },
  {
    id: 'C008',
    title: 'Adapto mi comunicación según la audiencia',
    description: 'Evalúa la flexibilidad comunicativa',
    dimensionId: 2
  },
  {
    id: 'C009',
    title: 'Manejo bien las presentaciones públicas',
    description: 'Mide la habilidad para hablar en público',
    dimensionId: 2
  },
  {
    id: 'C010',
    title: 'Escribo de manera efectiva y profesional',
    description: 'Evalúa las habilidades de comunicación escrita',
    dimensionId: 2
  },
  // Trabajo en Equipo (Dimensión 3)
  {
    id: 'C011',
    title: 'Colaboro efectivamente con personas diversas',
    description: 'Evalúa la capacidad de trabajar con diversidad',
    dimensionId: 3
  },
  {
    id: 'C012',
    title: 'Contribuyo activamente a los objetivos del equipo',
    description: 'Mide el compromiso con objetivos grupales',
    dimensionId: 3
  },
  {
    id: 'C013',
    title: 'Resuelvo conflictos de manera constructiva',
    description: 'Evalúa la gestión de conflictos',
    dimensionId: 3
  },
  {
    id: 'C014',
    title: 'Apoyo a mis compañeros cuando lo necesitan',
    description: 'Mide la solidaridad y apoyo mutuo',
    dimensionId: 3
  },
  {
    id: 'C015',
    title: 'Acepto feedback y críticas constructivas',
    description: 'Evalúa la receptividad al feedback',
    dimensionId: 3
  },
  // Resolución de Problemas (Dimensión 4)
  {
    id: 'C016',
    title: 'Identifico rápidamente la raíz de los problemas',
    description: 'Evalúa la capacidad de análisis',
    dimensionId: 4
  },
  {
    id: 'C017',
    title: 'Genero múltiples soluciones alternativas',
    description: 'Mide la creatividad en la resolución',
    dimensionId: 4
  },
  {
    id: 'C018',
    title: 'Evalúo pros y contras antes de decidir',
    description: 'Evalúa el pensamiento crítico',
    dimensionId: 4
  },
  {
    id: 'C019',
    title: 'Implemento soluciones de manera efectiva',
    description: 'Mide la capacidad de ejecución',
    dimensionId: 4
  },
  {
    id: 'C020',
    title: 'Aprendo de los errores para mejorar',
    description: 'Evalúa la capacidad de aprendizaje',
    dimensionId: 4
  },
  // Creatividad y Arte (Dimensión 5)
  {
    id: 'C021',
    title: 'Tengo habilidades para el dibujo y diseño',
    description: 'Evalúa capacidades artísticas visuales',
    dimensionId: 5
  },
  {
    id: 'C022',
    title: 'Disfruto creando música o contenido sonoro',
    description: 'Mide habilidades musicales y auditivas',
    dimensionId: 5
  },
  {
    id: 'C023',
    title: 'Escribo de manera creativa y original',
    description: 'Evalúa la escritura creativa',
    dimensionId: 5
  },
  {
    id: 'C024',
    title: 'Genero ideas innovadoras fácilmente',
    description: 'Mide el pensamiento divergente',
    dimensionId: 5
  },
  {
    id: 'C025',
    title: 'Aprecio y entiendo diferentes formas de arte',
    description: 'Evalúa la sensibilidad estética',
    dimensionId: 5
  },
  // Gestión y Emprendimiento (Dimensión 6)
  {
    id: 'C026',
    title: 'Planifico proyectos a largo plazo',
    description: 'Evalúa capacidades de planificación estratégica',
    dimensionId: 6
  },
  {
    id: 'C027',
    title: 'Manejo bien los recursos y presupuestos',
    description: 'Mide habilidades de gestión financiera',
    dimensionId: 6
  },
  {
    id: 'C028',
    title: 'Identifico oportunidades de negocio',
    description: 'Evalúa visión empresarial',
    dimensionId: 6
  },
  {
    id: 'C029',
    title: 'Tomo riesgos calculados',
    description: 'Mide tolerancia al riesgo',
    dimensionId: 6
  },
  {
    id: 'C030',
    title: 'Persisto ante las dificultades',
    description: 'Evalúa determinación y persistencia',
    dimensionId: 6
  },
  // Habilidades Técnicas (Dimensión 7)
  {
    id: 'C031',
    title: 'Tengo destreza manual para trabajos precisos',
    description: 'Evalúa habilidades manuales finas',
    dimensionId: 7
  },
  {
    id: 'C032',
    title: 'Reparo y mantengo equipos mecánicos',
    description: 'Mide capacidades mecánicas',
    dimensionId: 7
  },
  {
    id: 'C033',
    title: 'Trabajo bien con herramientas y maquinaria',
    description: 'Evalúa manejo de herramientas',
    dimensionId: 7
  },
  {
    id: 'C034',
    title: 'Organizo espacios y procesos eficientemente',
    description: 'Mide organización operativa',
    dimensionId: 7
  },
  {
    id: 'C035',
    title: 'Sigo procedimientos técnicos exactamente',
    description: 'Evalúa precisión técnica',
    dimensionId: 7
  },
  // Cuidado y Servicio (Dimensión 8)
  {
    id: 'C036',
    title: 'Entiendo y conecto con los sentimientos de otros',
    description: 'Evalúa empatía y comprensión',
    dimensionId: 8
  },
  {
    id: 'C037',
    title: 'Mantengo la paciencia en situaciones difíciles',
    description: 'Mide paciencia y tolerancia',
    dimensionId: 8
  },
  {
    id: 'C038',
    title: 'Brindo apoyo emocional a quienes lo necesitan',
    description: 'Evalúa capacidad de apoyo',
    dimensionId: 8
  },
  {
    id: 'C039',
    title: 'Enseño y guío a otros efectivamente',
    description: 'Mide habilidades educativas',
    dimensionId: 8
  },
  {
    id: 'C040',
    title: 'Actúo con ética y valores sólidos',
    description: 'Evalúa integridad moral',
    dimensionId: 8
  }
];

export const DIMENSIONS: Dimension[] = [
  {
    id: 1,
    name: 'Liderazgo',
    shortName: 'Liderazgo',
    description: 'Capacidad para dirigir, motivar e influir en otros hacia el logro de objetivos comunes.',
    icon: 'Crown',
    color: '#3B82F6',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 1)
  },
  {
    id: 2,
    name: 'Comunicación',
    shortName: 'Comunicación',
    description: 'Habilidad para transmitir ideas de manera clara y efectiva, tanto verbal como escrita.',
    icon: 'MessageCircle',
    color: '#10B981',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 2)
  },
  {
    id: 3,
    name: 'Trabajo en Equipo',
    shortName: 'Trabajo en Equipo',
    description: 'Capacidad para colaborar efectivamente con otros y contribuir al éxito colectivo.',
    icon: 'Users',
    color: '#8B5CF6',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 3)
  },
  {
    id: 4,
    name: 'Resolución de Problemas',
    shortName: 'Resolución de Problemas',
    description: 'Habilidad para identificar, analizar y resolver problemas de manera efectiva.',
    icon: 'Lightbulb',
    color: '#F59E0B',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 4)
  },
  {
    id: 5,
    name: 'Creatividad y Arte',
    shortName: 'Creatividad',
    description: 'Capacidad para crear, innovar y apreciar expresiones artísticas y estéticas.',
    icon: 'Palette',
    color: '#EC4899',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 5)
  },
  {
    id: 6,
    name: 'Gestión y Emprendimiento',
    shortName: 'Gestión',
    description: 'Habilidad para planificar, organizar recursos y desarrollar iniciativas empresariales.',
    icon: 'TrendingUp',
    color: '#EF4444',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 6)
  },
  {
    id: 7,
    name: 'Habilidades Técnicas',
    shortName: 'Técnicas',
    description: 'Destreza manual y capacidad para trabajar con herramientas y procesos técnicos.',
    icon: 'Settings',
    color: '#6B7280',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 7)
  },
  {
    id: 8,
    name: 'Cuidado y Servicio',
    shortName: 'Servicio',
    description: 'Capacidad para cuidar, apoyar y servir a otros con empatía y dedicación.',
    icon: 'Heart',
    color: '#06B6D4',
    competencies: COMPETENCIES.filter(c => c.dimensionId === 8)
  }
];

// Utilidades
export const getDimensionById = (id: number): Dimension | undefined => {
  return DIMENSIONS.find(d => d.id === id);
};

export const getCompetencyById = (id: string): { competency: Competency; dimension: Dimension } | undefined => {
  const competency = COMPETENCIES.find(c => c.id === id);
  if (!competency) return undefined;
  
  const dimension = DIMENSIONS.find(d => d.id === competency.dimensionId);
  if (!dimension) return undefined;
  
  return { competency, dimension };
};

export const getDimensionsByCompetencyArea = (area: string): Dimension[] => {
  // Esta función puede ser implementada si se necesita agrupar por área
  return DIMENSIONS;
};

export const getAllCompetencyAreas = (): string[] => {
  // Retorna las áreas únicas de competencias
  return ['Habilidades Directivas', 'Habilidades Interpersonales', 'Habilidades Cognitivas', 'Habilidades Personales'];
};

export const getTotalQuestions = (): number => {
  return COMPETENCIES.length;
};