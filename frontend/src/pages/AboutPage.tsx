import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Brain, Target, Users, Zap, CheckCircle, Star } from 'lucide-react';

export default function AboutPage() {
  const features = [
    {
      icon: Brain,
      title: 'Inteligencia Artificial Avanzada',
      description: 'Utilizamos algoritmos de machine learning para analizar tu perfil de talento y predecir las mejores opciones académicas para ti.'
    },
    {
      icon: Target,
      title: 'Evaluación Personalizada',
      description: 'Nuestro sistema evalúa múltiples dimensiones de talento para crear un perfil completo y personalizado de tus habilidades.'
    },
    {
      icon: Users,
      title: 'Recomendaciones Precisas',
      description: 'Basado en tu perfil, te recomendamos programas académicos que se alinean perfectamente con tus fortalezas y objetivos.'
    },
    {
      icon: Zap,
      title: 'Resultados Instantáneos',
      description: 'Obtén tu análisis completo y recomendaciones en tiempo real, sin esperas ni complicaciones.'
    }
  ];

  const benefits = [
    'Identificación precisa de fortalezas y áreas de mejora',
    'Recomendaciones personalizadas de programas académicos',
    'Análisis detallado de competencias profesionales',
    'Visualización clara de tu perfil de talento',
    'Orientación para la toma de decisiones académicas',
    'Acceso a una amplia base de datos de programas'
  ];

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-primary/10 rounded-full">
            <Brain className="h-16 w-16 text-primary" />
          </div>
        </div>
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Acerca de TalentAI
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
          Una plataforma innovadora que utiliza inteligencia artificial para evaluar tu talento 
          y recomendarte los mejores programas académicos según tu perfil único.
        </p>
      </div>

      {/* Mission Section */}
      <Card className="mb-12">
        <CardHeader>
          <CardTitle className="text-2xl text-center">
            Nuestra Misión
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-lg text-gray-700 dark:text-gray-300 text-center leading-relaxed">
            Democratizar el acceso a la orientación vocacional de calidad mediante el uso de 
            inteligencia artificial, ayudando a estudiantes y profesionales a tomar decisiones 
            informadas sobre su futuro académico y profesional.
          </p>
        </CardContent>
      </Card>

      {/* Features Section */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-8">
          ¿Cómo Funciona?
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <Icon className="h-6 w-6 text-primary" />
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600 dark:text-gray-400">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Benefits Section */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-8">
          Beneficios de TalentAI
        </h2>
        <Card>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {benefits.map((benefit, index) => (
                <div key={index} className="flex items-start gap-3">
                  <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700 dark:text-gray-300">{benefit}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Process Section */}
      <div className="mb-12">
        <h2 className="text-3xl font-bold text-center text-gray-900 dark:text-white mb-8">
          Proceso de Evaluación
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto p-3 bg-blue-100 dark:bg-blue-900 rounded-full w-fit">
                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">1</span>
              </div>
              <CardTitle>Completa la Evaluación</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400">
                Responde nuestro cuestionario integral que evalúa múltiples dimensiones de tu talento y personalidad.
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto p-3 bg-green-100 dark:bg-green-900 rounded-full w-fit">
                <span className="text-2xl font-bold text-green-600 dark:text-green-400">2</span>
              </div>
              <CardTitle>Análisis con IA</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400">
                Nuestros algoritmos analizan tus respuestas y crean un perfil detallado de tus competencias y preferencias.
              </p>
            </CardContent>
          </Card>

          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto p-3 bg-purple-100 dark:bg-purple-900 rounded-full w-fit">
                <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">3</span>
              </div>
              <CardTitle>Recibe Recomendaciones</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600 dark:text-gray-400">
                Obtén recomendaciones personalizadas de programas académicos y un análisis completo de tu perfil.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Technology Section */}
      <Card className="mb-12">
        <CardHeader>
          <CardTitle className="text-2xl text-center">
            Tecnología de Vanguardia
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center">
            <p className="text-lg text-gray-700 dark:text-gray-300 mb-6">
              TalentAI utiliza las últimas tecnologías en inteligencia artificial y análisis de datos:
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              <Badge variant="secondary" className="text-sm py-2 px-4">
                Machine Learning
              </Badge>
              <Badge variant="secondary" className="text-sm py-2 px-4">
                Análisis Predictivo
              </Badge>
              <Badge variant="secondary" className="text-sm py-2 px-4">
                Procesamiento de Lenguaje Natural
              </Badge>
              <Badge variant="secondary" className="text-sm py-2 px-4">
                Big Data Analytics
              </Badge>
              <Badge variant="secondary" className="text-sm py-2 px-4">
                Algoritmos de Recomendación
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* CTA Section */}
      <div className="text-center">
        <Card className="bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
          <CardContent className="pt-8 pb-8">
            <Star className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              ¿Listo para Descubrir tu Potencial?
            </h3>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-6 max-w-2xl mx-auto">
              Comienza tu evaluación ahora y descubre qué programas académicos se alinean mejor con tu perfil de talento.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="/evaluation" 
                className="inline-flex items-center justify-center px-6 py-3 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors font-medium"
              >
                Comenzar Evaluación
              </a>
              <a 
                href="/programs" 
                className="inline-flex items-center justify-center px-6 py-3 border border-primary text-primary rounded-md hover:bg-primary/10 transition-colors font-medium"
              >
                Explorar Programas
              </a>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}