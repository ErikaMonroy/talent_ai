'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Clock, 
  CheckCircle, 
  Users, 
  Target, 
  ArrowRight,
  Info,
  Shield
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { DIMENSIONS } from '@/data/dimensions';

export function WelcomeStep() {
  const { goToStep } = useAssessmentStore();

  const features = [
    {
      icon: <Brain className="h-6 w-6 text-blue-600" />,
      title: "Evaluación Integral",
      description: "100 preguntas que evalúan 8 dimensiones clave del talento humano"
    },
    {
      icon: <Clock className="h-6 w-6 text-green-600" />,
      title: "25 Minutos",
      description: "Tiempo estimado para completar toda la evaluación"
    },
    {
      icon: <Target className="h-6 w-6 text-purple-600" />,
      title: "Resultados Personalizados",
      description: "Análisis detallado de tus fortalezas y áreas de desarrollo"
    },
    {
      icon: <Shield className="h-6 w-6 text-orange-600" />,
      title: "Datos Seguros",
      description: "Tu información se guarda localmente y de forma segura"
    }
  ];

  const dimensionsWithIcons = DIMENSIONS.map((dimension, index) => {
    const icons = ["🧮", "💬", "🔬", "📚", "🎨", "💼", "🔧", "❤️"];
    return {
      ...dimension,
      icon: icons[index] || "🎯"
    };
  });

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Encabezado principal */}
      <Card className="text-center bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border-2 border-blue-200 dark:border-blue-800">
        <CardHeader className="pb-4">
          <div className="flex justify-center mb-4">
            <div className="p-4 bg-blue-600 rounded-full">
              <Brain className="h-12 w-12 text-white" />
            </div>
          </div>
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Evaluación de Talento TalentAI
          </CardTitle>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Descubre tus fortalezas y potencial a través de una evaluación integral 
            que analiza múltiples dimensiones del talento humano.
          </p>
        </CardHeader>
        
        <CardContent className="pt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {features.map((feature, index) => (
              <div key={index} className="flex flex-col items-center text-center p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                <div className="mb-2">{feature.icon}</div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-1">
                  {feature.title}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
          
          <Button 
            onClick={() => goToStep('personal-data')}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 text-lg font-semibold"
          >
            Comenzar Evaluación
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </CardContent>
      </Card>

      {/* Dimensiones que se evaluarán */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-blue-600" />
            Dimensiones de Talento a Evaluar
          </CardTitle>
          <p className="text-gray-600 dark:text-gray-400">
            La evaluación cubre 8 dimensiones principales con un total de 100 competencias específicas.
          </p>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {dimensionsWithIcons.map((dimension, index) => (
              <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <span className="text-2xl">{dimension.icon}</span>
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {dimension.name}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {dimension.competencies.length} competencias
                  </p>
                </div>
                <Badge variant="secondary" className="text-xs">
                  ~{Math.ceil(dimension.competencies.length * 0.25)} min
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Instrucciones importantes */}
      <Card className="border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-800 dark:text-orange-200">
            <Info className="h-5 w-5" />
            Instrucciones Importantes
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  Responde con honestidad
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  No hay respuestas correctas o incorrectas. La evaluación es más precisa cuando respondes según tu experiencia real.
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  Tómate tu tiempo
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Aunque el tiempo estimado es de 25 minutos, puedes pausar y continuar cuando quieras. Tu progreso se guarda automáticamente.
                </p>
              </div>
            </div>
            
            <div className="flex items-start gap-3">
              <CheckCircle className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  Usa la escala completa
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Utiliza toda la escala de 1 a 5. No tengas miedo de usar los extremos cuando realmente apliquen a tu situación.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}