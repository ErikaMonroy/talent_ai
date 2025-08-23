import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { EvaluationCard } from '@/components/evaluation-card'
import { Progress } from '@/components/ui/progress'
import { BarChart3, Download, Share2, TrendingUp, User, Calendar } from 'lucide-react'

// Mock data for demonstration
const mockResults = {
  user: {
    name: 'Usuario Demo',
    evaluationDate: '2024-01-15',
    totalScore: 78
  },
  dimensions: [
    {
      id: 1,
      name: 'Lógico-Matemático',
      score: 85,
      description: 'Excelente capacidad para resolver problemas lógicos y matemáticos',
      strengths: ['Análisis numérico', 'Resolución de problemas', 'Pensamiento abstracto'],
      improvements: ['Aplicación práctica', 'Comunicación de resultados']
    },
    {
      id: 2,
      name: 'Comunicación',
      score: 72,
      description: 'Buenas habilidades de comunicación con oportunidades de mejora',
      strengths: ['Expresión escrita', 'Escucha activa'],
      improvements: ['Presentaciones públicas', 'Comunicación no verbal', 'Persuasión']
    },
    {
      id: 5,
      name: 'Creatividad',
      score: 90,
      description: 'Capacidad excepcional para generar ideas innovadoras',
      strengths: ['Pensamiento divergente', 'Originalidad', 'Flexibilidad mental'],
      improvements: ['Implementación de ideas', 'Trabajo en equipo creativo']
    },
    {
      id: 7,
      name: 'Pensamiento Crítico',
      score: 68,
      description: 'Capacidad sólida de análisis con potencial de desarrollo',
      strengths: ['Evaluación de información', 'Identificación de sesgos'],
      improvements: ['Síntesis de conclusiones', 'Argumentación', 'Toma de decisiones']
    }
  ]
}

const getScoreLevel = (score: number) => {
  if (score >= 90) return { level: 'Excepcional', color: 'success', description: 'Fortaleza destacada' }
  if (score >= 80) return { level: 'Excelente', color: 'primary', description: 'Muy competente' }
  if (score >= 70) return { level: 'Bueno', color: 'secondary', description: 'Competente' }
  if (score >= 60) return { level: 'Regular', color: 'warning', description: 'En desarrollo' }
  return { level: 'Necesita Mejora', color: 'destructive', description: 'Requiere atención' }
}

export default function Results() {
  const [selectedView, setSelectedView] = useState<'overview' | 'detailed' | 'recommendations'>('overview')

  const averageScore = Math.round(
    mockResults.dimensions.reduce((sum, dim) => sum + dim.score, 0) / mockResults.dimensions.length
  )

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-4">
          Resultados de Evaluación
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Tu perfil de talento personalizado basado en la evaluación completada
        </p>
      </div>

      {/* User Info Card */}
      <Card className="mb-8">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center">
                <User className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <CardTitle className="text-xl">{mockResults.user.name}</CardTitle>
                <CardDescription className="flex items-center space-x-4">
                  <span className="flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    Evaluado el {new Date(mockResults.user.evaluationDate).toLocaleDateString('es-ES')}
                  </span>
                </CardDescription>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-primary">{averageScore}</div>
              <div className="text-sm text-muted-foreground">Puntuación General</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4 mb-4">
            <div className="flex-1">
              <div className="flex justify-between text-sm mb-2">
                <span>Progreso General</span>
                <span className="font-medium">{averageScore}%</span>
              </div>
              <Progress value={averageScore} className="h-3" />
            </div>
            <Badge variant="secondary" className="text-sm">
              {getScoreLevel(averageScore).level}
            </Badge>
          </div>
          
          <div className="flex space-x-2">
            <Button size="sm" variant="outline">
              <Download className="w-4 h-4 mr-2" />
              Descargar PDF
            </Button>
            <Button size="sm" variant="outline">
              <Share2 className="w-4 h-4 mr-2" />
              Compartir
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* View Selector */}
      <div className="flex space-x-2 mb-6">
        <Button 
          variant={selectedView === 'overview' ? 'default' : 'outline'}
          onClick={() => setSelectedView('overview')}
        >
          <BarChart3 className="w-4 h-4 mr-2" />
          Resumen
        </Button>
        <Button 
          variant={selectedView === 'detailed' ? 'default' : 'outline'}
          onClick={() => setSelectedView('detailed')}
        >
          <TrendingUp className="w-4 h-4 mr-2" />
          Detallado
        </Button>
        <Button 
          variant={selectedView === 'recommendations' ? 'default' : 'outline'}
          onClick={() => setSelectedView('recommendations')}
        >
          Recomendaciones
        </Button>
      </div>

      {/* Content based on selected view */}
      {selectedView === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {mockResults.dimensions.map((dimension) => {
            const scoreInfo = getScoreLevel(dimension.score)
            return (
              <EvaluationCard
                key={dimension.id}
                title={dimension.name}
                description={dimension.description}
                dimension={dimension.id}
                score={dimension.score}
                status="completed"
              >
                <div className="mt-4">
                  <Badge 
                    variant="secondary" 
                    className={`bg-${scoreInfo.color}/10 text-${scoreInfo.color} border-${scoreInfo.color}/20`}
                  >
                    {scoreInfo.level} - {scoreInfo.description}
                  </Badge>
                </div>
              </EvaluationCard>
            )
          })}
        </div>
      )}

      {selectedView === 'detailed' && (
        <div className="space-y-6">
          {mockResults.dimensions.map((dimension) => {
            const scoreInfo = getScoreLevel(dimension.score)
            return (
              <Card key={dimension.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`w-4 h-4 rounded-full bg-dimension-${dimension.id}`} />
                      <CardTitle className="text-lg">{dimension.name}</CardTitle>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-primary">{dimension.score}</div>
                      <Badge variant="outline" className="text-xs">
                        {scoreInfo.level}
                      </Badge>
                    </div>
                  </div>
                  <CardDescription>{dimension.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium text-success mb-3 flex items-center">
                        <TrendingUp className="w-4 h-4 mr-2" />
                        Fortalezas
                      </h4>
                      <ul className="space-y-2">
                        {dimension.strengths.map((strength, index) => (
                          <li key={index} className="flex items-center text-sm">
                            <div className="w-2 h-2 bg-success rounded-full mr-2" />
                            {strength}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium text-warning mb-3 flex items-center">
                        <TrendingUp className="w-4 h-4 mr-2" />
                        Áreas de Mejora
                      </h4>
                      <ul className="space-y-2">
                        {dimension.improvements.map((improvement, index) => (
                          <li key={index} className="flex items-center text-sm">
                            <div className="w-2 h-2 bg-warning rounded-full mr-2" />
                            {improvement}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {selectedView === 'recommendations' && (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Plan de Desarrollo Personalizado</CardTitle>
              <CardDescription>
                Recomendaciones específicas basadas en tu perfil de talento
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {mockResults.dimensions
                  .sort((a, b) => a.score - b.score) // Sort by score to prioritize improvements
                  .map((dimension, index) => (
                    <div key={dimension.id} className="border-l-4 border-primary pl-4">
                      <h3 className="font-semibold text-lg mb-2">
                        {index + 1}. {dimension.name}
                      </h3>
                      <p className="text-muted-foreground mb-3">
                        Puntuación actual: <span className="font-medium">{dimension.score}/100</span>
                      </p>
                      <div className="bg-muted p-4 rounded-lg">
                        <h4 className="font-medium mb-2">Acciones Recomendadas:</h4>
                        <ul className="space-y-1 text-sm">
                          {dimension.improvements.map((improvement, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-primary mr-2">•</span>
                              Enfócate en desarrollar: {improvement}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))
                }
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}