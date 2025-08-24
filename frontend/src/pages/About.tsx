import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Brain, Target, Users, Lightbulb, Award, BookOpen, TrendingUp, Heart } from 'lucide-react'
import { Link } from 'react-router-dom'

const features = [
  {
    icon: Brain,
    title: 'Evaluación Científica',
    description: 'Metodología basada en investigación psicométrica y neurociencia cognitiva'
  },
  {
    icon: Target,
    title: 'Personalización',
    description: 'Evaluaciones adaptadas a tu perfil y objetivos específicos'
  },
  {
    icon: TrendingUp,
    title: 'Análisis Profundo',
    description: 'Insights detallados sobre fortalezas y áreas de oportunidad'
  },
  {
    icon: Award,
    title: 'Certificación',
    description: 'Reportes profesionales validados para uso académico y laboral'
  }
]

const dimensions = [
  {
    id: 1,
    name: 'Lógico-Matemático',
    description: 'Capacidad para el razonamiento lógico, resolución de problemas matemáticos y pensamiento abstracto.',
    applications: ['Ingeniería', 'Ciencias', 'Finanzas', 'Programación']
  },
  {
    id: 2,
    name: 'Comunicación',
    description: 'Habilidades para expresar ideas de forma clara, persuasiva y efectiva en diferentes contextos.',
    applications: ['Marketing', 'Ventas', 'Educación', 'Periodismo']
  },
  {
    id: 3,
    name: 'Ciencias',
    description: 'Comprensión y aplicación del método científico, análisis de datos y pensamiento empírico.',
    applications: ['Investigación', 'Medicina', 'Biotecnología', 'Análisis']
  },
  {
    id: 4,
    name: 'Humanidades',
    description: 'Conocimiento cultural, histórico y filosófico para comprender el comportamiento humano.',
    applications: ['Psicología', 'Historia', 'Filosofía', 'Antropología']
  },
  {
    id: 5,
    name: 'Creatividad',
    description: 'Capacidad para generar ideas originales, innovar y encontrar soluciones no convencionales.',
    applications: ['Diseño', 'Arte', 'Innovación', 'Emprendimiento']
  },
  {
    id: 6,
    name: 'Liderazgo',
    description: 'Habilidades para inspirar, motivar y dirigir equipos hacia objetivos comunes.',
    applications: ['Gestión', 'Dirección', 'Consultoría', 'Política']
  },
  {
    id: 7,
    name: 'Pensamiento Crítico',
    description: 'Capacidad para analizar información objetivamente y tomar decisiones fundamentadas.',
    applications: ['Consultoría', 'Auditoría', 'Investigación', 'Estrategia']
  },
  {
    id: 8,
    name: 'Adaptabilidad',
    description: 'Flexibilidad para ajustarse a cambios, aprender continuamente y prosperar en entornos dinámicos.',
    applications: ['Tecnología', 'Startups', 'Cambio organizacional', 'Globalización']
  }
]

const team = [
  {
    name: 'Erika Monroy',
    role: 'Directora de Investigación',
    description: 'Ingeniera en Telecomunicaciones, especialista en Ciencias de Datos y Analítica en formación',
    icon: BookOpen
  }
]

export default function About() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-foreground mb-6">
          Sobre TalentAI
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
          Revolucionamos la evaluación de talento mediante inteligencia artificial y metodologías científicas, 
          ayudando a personas y organizaciones a descubrir y desarrollar su máximo potencial.
        </p>
        
        {/* MVP Notice */}
        <Card className="mt-8 bg-amber-50 border-amber-200 dark:bg-amber-950/20 dark:border-amber-800">
          <CardContent className="py-4">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <Lightbulb className="w-5 h-5 text-amber-600" />
              <h3 className="font-semibold text-amber-800 dark:text-amber-200">Versión MVP - Proyecto de Grado</h3>
            </div>
            <p className="text-sm text-amber-700 dark:text-amber-300 max-w-2xl mx-auto">
              TalentAI es actualmente un <strong>Producto Mínimo Viable (MVP)</strong> desarrollado como proyecto de grado 
              para la especialización en "Ciencias de Datos y Analítica". Esta es una versión de prueba que está 
              en constante evolución y algunas funcionalidades se encuentran en desarrollo.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Mission Section */}
      <Card className="mb-12">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl flex items-center justify-center">
            <Target className="w-6 h-6 mr-2 text-primary" />
            Nuestra Misión
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-lg text-center text-muted-foreground leading-relaxed">
            Democratizar el acceso a evaluaciones de talento de alta calidad, proporcionando herramientas 
            científicamente validadas que permitan a cada persona conocer sus fortalezas únicas y desarrollar 
            todo su potencial profesional y personal.
          </p>
        </CardContent>
      </Card>

      {/* Features Section */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">¿Por qué elegir TalentAI?</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card key={index} className="text-center hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Dimensions Section */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">Las 8 Dimensiones del Talento</h2>
        <p className="text-center text-muted-foreground mb-8 max-w-3xl mx-auto">
          Nuestro modelo evalúa ocho dimensiones fundamentales del talento humano, 
          cada una respaldada por investigación científica y validada en contextos reales.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {dimensions.map((dimension) => (
            <Card key={dimension.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-center space-x-3">
                  <div className={`w-4 h-4 rounded-full bg-dimension-${dimension.id}`} />
                  <CardTitle className="text-lg">{dimension.name}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription className="mb-4 leading-relaxed">
                  {dimension.description}
                </CardDescription>
                <div>
                  <h4 className="font-medium text-sm mb-2">Aplicaciones profesionales:</h4>
                  <div className="flex flex-wrap gap-2">
                    {dimension.applications.map((app, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {app}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Team Section */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-center mb-8">Nuestro Equipo</h2>
        <div className="flex justify-center">
          {team.map((member, index) => {
            const Icon = member.icon
            return (
              <Card key={index} className="text-center">
                <CardHeader>
                  <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Icon className="w-8 h-8 text-primary" />
                  </div>
                  <CardTitle className="text-lg">{member.name}</CardTitle>
                  <Badge variant="outline" className="mx-auto">
                    {member.role}
                  </Badge>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {member.description}
                  </p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* CTA Section */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="text-center py-8">
          <h2 className="text-2xl font-bold mb-4">¿Listo para probar TalentAI?</h2>
          <p className="text-muted-foreground mb-6 max-w-2xl mx-auto">
            Experimenta con nuestra versión MVP y ayúdanos a mejorar con tu feedback. 
            Obtén insights sobre tus fortalezas mientras contribuyes al desarrollo de la plataforma.
          </p>
          <div className="flex justify-center space-x-4">
            <Button asChild size="lg">
              <Link to="/evaluation">
                <Users className="w-5 h-5 mr-2" />
                Probar Evaluación
              </Link>
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-4">
            * Versión de prueba - Los resultados pueden variar y están sujetos a mejoras continuas
          </p>
        </CardContent>
      </Card>
    </div>
  )
}