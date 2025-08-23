import { Link } from 'react-router-dom'
import { Brain, Target, BarChart3, Users, Zap, Shield } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

const features = [
  {
    icon: Brain,
    title: 'Inteligencia Artificial',
    description: 'Algoritmos avanzados para evaluar múltiples dimensiones del talento humano.',
    color: 'dimension-1'
  },
  {
    icon: Target,
    title: 'Evaluación Precisa',
    description: 'Medición objetiva de 8 dimensiones clave del potencial profesional.',
    color: 'dimension-2'
  },
  {
    icon: BarChart3,
    title: 'Análisis Detallado',
    description: 'Reportes completos con insights accionables para el desarrollo.',
    color: 'dimension-3'
  },
  {
    icon: Users,
    title: 'Para Equipos',
    description: 'Herramientas diseñadas para RRHH y líderes de equipos.',
    color: 'dimension-4'
  },
  {
    icon: Zap,
    title: 'Resultados Rápidos',
    description: 'Evaluaciones eficientes con resultados en tiempo real.',
    color: 'dimension-5'
  },
  {
    icon: Shield,
    title: 'Datos Seguros',
    description: 'Máxima privacidad y seguridad en el manejo de información.',
    color: 'dimension-6'
  }
]

const dimensions = [
  { name: 'Lógico-Matemático', color: 'dimension-1' },
  { name: 'Comunicación', color: 'dimension-2' },
  { name: 'Ciencias', color: 'dimension-3' },
  { name: 'Humanidades', color: 'dimension-4' },
  { name: 'Creatividad', color: 'dimension-5' },
  { name: 'Liderazgo', color: 'dimension-6' },
  { name: 'Pensamiento Crítico', color: 'dimension-7' },
  { name: 'Adaptabilidad', color: 'dimension-8' }
]

export default function Home() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-primary/5 via-background to-secondary/5">
        <div className="container mx-auto text-center">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6">
              Descubre el{' '}
              <span className="text-primary">Talento</span>{' '}
              con IA
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 leading-relaxed">
              Plataforma avanzada de evaluación de talento que utiliza inteligencia artificial 
              para identificar y medir las capacidades profesionales en 8 dimensiones clave.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" asChild className="text-lg px-8 py-6">
                <Link to="/evaluation">
                  Comenzar Evaluación
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild className="text-lg px-8 py-6">
                <Link to="/about">
                  Conocer Más
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Dimensions Section */}
      <section className="py-16 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              8 Dimensiones del Talento
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Evaluamos de manera integral las competencias que definen el potencial profesional
            </p>
          </div>
          <div className="flex flex-wrap justify-center gap-3 mb-16">
            {dimensions.map((dimension, index) => (
              <Badge
                key={index}
                variant="secondary"
                className={`px-4 py-2 text-sm font-medium bg-${dimension.color}-light text-${dimension.color} border-${dimension.color}/20`}
              >
                {dimension.name}
              </Badge>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              ¿Por qué TalentAI?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Tecnología de vanguardia al servicio del desarrollo humano
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <Card key={index} className="hover:shadow-lg transition-shadow duration-300">
                  <CardHeader>
                    <div className={`w-12 h-12 rounded-lg bg-${feature.color}-light flex items-center justify-center mb-4`}>
                      <Icon className={`h-6 w-6 text-${feature.color}`} />
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-base leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-primary/5">
        <div className="container mx-auto text-center">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
              ¿Listo para descubrir tu potencial?
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Inicia tu evaluación personalizada y obtén insights valiosos sobre tus fortalezas y áreas de desarrollo.
            </p>
            <Button size="lg" asChild className="text-lg px-8 py-6">
              <Link to="/evaluation">
                Comenzar Ahora
              </Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}