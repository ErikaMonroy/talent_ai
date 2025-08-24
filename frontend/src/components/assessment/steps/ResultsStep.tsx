'use client';

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Trophy,
  Target,
  TrendingUp,
  Download,
  Share2,
  RotateCcw,
  Star,
  Award,
  Brain,
  BarChart3,
  ArrowRight,
  Sparkles
} from 'lucide-react';
import { toast } from 'sonner';
import { useAssessmentStore } from '@/store/assessmentStore';
import { DIMENSIONS_CONFIG } from '@/types/assessment';
import { ModelSelectorStep } from './ModelSelectorStep';
import { PredictionLoaderStep } from './PredictionLoaderStep';
import { PredictionResultsStep } from './PredictionResultsStep';
import { ProgramsTableStep } from './ProgramsTableStep';

export function ResultsStep() {
  const navigate = useNavigate();
  const { 
    personalData,
    responses,
    getDimensionProgress,
    getCompletionPercentage,
    calculateDimensionAverage,
    resetAssessment,
    // Estados de predicción
    selectedModelType,
    isLoadingPrediction,
    predictionResult,
    predictionError
  } = useAssessmentStore();

  // Calcular resultados por dimensión
  const dimensionResults = DIMENSIONS_CONFIG.map(dimension => {
    const progress = getDimensionProgress(dimension.id);
    const average = calculateDimensionAverage(dimension.id);
    
    return {
      ...dimension,
      progress: progress.percentage,
      average: average,
      level: getPerformanceLevel(average),
      color: getPerformanceColor(average)
    };
  });

  // Calcular estadísticas generales
  const completedDimensions = dimensionResults.filter(d => d.progress === 100);
  const overallAverage = dimensionResults.reduce((sum, d) => sum + d.average, 0) / dimensionResults.length;
  const topStrengths = dimensionResults
    .filter(d => d.average >= 4.0)
    .sort((a, b) => b.average - a.average)
    .slice(0, 3);
  const developmentAreas = dimensionResults
    .filter(d => d.average < 3.5)
    .sort((a, b) => a.average - b.average)
    .slice(0, 3);

  function getPerformanceLevel(average: number): string {
    if (average >= 4.5) return 'Excelente';
    if (average >= 4.0) return 'Muy bueno';
    if (average >= 3.5) return 'Bueno';
    if (average >= 3.0) return 'Regular';
    return 'Necesita desarrollo';
  }

  function getPerformanceColor(average: number): string {
    if (average >= 4.5) return 'text-green-600';
    if (average >= 4.0) return 'text-blue-600';
    if (average >= 3.5) return 'text-yellow-600';
    if (average >= 3.0) return 'text-orange-600';
    return 'text-red-600';
  }

  function getProgressColor(average: number): string {
    if (average >= 4.5) return 'bg-green-500';
    if (average >= 4.0) return 'bg-blue-500';
    if (average >= 3.5) return 'bg-yellow-500';
    if (average >= 3.0) return 'bg-orange-500';
    return 'bg-red-500';
  }

  const handleViewDetailedResults = () => {
    navigate('/results');
  };

  const handleNewAssessment = () => {
    toast('¿Iniciar nueva evaluación?', {
      description: 'Se perderán todos los datos de la evaluación actual.',
      duration: Infinity,
      action: {
        label: 'Confirmar',
        onClick: () => {
          resetAssessment();
          navigate('/');
          toast.success('Nueva evaluación iniciada', { duration: 3000 });
        },
      },
      cancel: {
        label: 'Cancelar',
        onClick: () => {},
      },
    });
  };

  // Determinar qué mostrar según el estado de la predicción
  const showInitialResults = !selectedModelType;
  const showModelSelector = selectedModelType === null || (!isLoadingPrediction && !predictionResult && !predictionError);
  const showPredictionLoader = isLoadingPrediction;
  const showPredictionResults = predictionResult && !isLoadingPrediction;
  const showProgramsTable = predictionResult && !isLoadingPrediction;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Encabezado de resultados - siempre visible */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border-2 border-green-200 dark:border-green-800">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="p-4 bg-green-600 rounded-full">
              <Trophy className="h-12 w-12 text-white" />
            </div>
          </div>
          
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            ¡Evaluación Completada!
          </CardTitle>
          
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-4">
            Hola {personalData.name}, aquí están tus resultados de la evaluación de talento TalentAI.
          </p>
          
          <div className="flex justify-center gap-6 text-center">
            <div>
              <div className="text-2xl font-bold text-green-600">
                {completedDimensions.length}/8
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Dimensiones completadas
              </div>
            </div>
            

            
            <div>
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(getCompletionPercentage())}%
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Evaluación completada
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Resultados iniciales - solo si no se ha iniciado predicción */}
      {showInitialResults && (
        <>
          {/* Resumen ejecutivo */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Fortalezas principales */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Star className="h-5 w-5 text-yellow-500" />
                  Principales Fortalezas
                </CardTitle>
              </CardHeader>
              
              <CardContent>
                {topStrengths.length > 0 ? (
                  <div className="space-y-3">
                    {topStrengths.map((dimension, index) => (
                      <div key={dimension.id} className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {dimension.name}
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Puntuación: {dimension.average.toFixed(1)}/5.0
                          </p>
                        </div>
                        <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                          #{index + 1}
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-600 dark:text-gray-400 text-center py-4">
                    Completa más dimensiones para ver tus fortalezas.
                  </p>
                )}
              </CardContent>
            </Card>

            {/* Áreas de desarrollo */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-blue-500" />
                  Áreas de Desarrollo
                </CardTitle>
              </CardHeader>
              
              <CardContent>
                {developmentAreas.length > 0 ? (
                  <div className="space-y-3">
                    {developmentAreas.map((dimension) => (
                      <div key={dimension.id} className="flex items-center gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {dimension.name}
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Puntuación: {dimension.average.toFixed(1)}/5.0
                          </p>
                        </div>
                        <Badge variant="outline" className="text-blue-600 border-blue-600">
                          Mejorar
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-600 dark:text-gray-400 text-center py-4">
                    ¡Excelente! No hay áreas críticas de desarrollo identificadas.
                  </p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Resultados detallados por dimensión */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-500" />
                Resultados Detallados por Dimensión
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              <div className="space-y-4">
                {dimensionResults.map((dimension) => (
                  <div key={dimension.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div>
                          <h4 className="font-medium text-gray-900 dark:text-white">
                            {dimension.name}
                          </h4>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            {dimension.description}
                          </p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className={`text-lg font-bold ${dimension.color}`}>
                          {dimension.average.toFixed(1)}/5.0
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {dimension.level}
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Progreso: {Math.round(dimension.progress)}%</span>
                        <span>Puntuación: {dimension.average.toFixed(1)}/5.0</span>
                      </div>
                      
                      <div className="relative">
                        <Progress value={dimension.progress} className="h-2" />
                        <div 
                          className={`absolute top-0 left-0 h-2 rounded-full ${getProgressColor(dimension.average)}`}
                          style={{ width: `${(dimension.average / 5) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Llamada a la acción para predicción */}
          <Card className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border-2 border-purple-200 dark:border-purple-800">
            <CardHeader className="text-center">
              <div className="flex justify-center mb-4">
                <div className="p-4 bg-purple-600 rounded-full">
                  <Sparkles className="h-12 w-12 text-white" />
                </div>
              </div>
              
              <CardTitle className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                ¡Descubre tu Futuro Académico!
              </CardTitle>
              
              <p className="text-lg text-gray-600 dark:text-gray-300 mb-4">
                Basándose en tu perfil de competencias, nuestro sistema de IA puede predecir las áreas académicas más afines a ti.
              </p>
              
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                Utiliza algoritmos avanzados de Machine Learning para encontrar tu camino académico ideal.
              </p>
            </CardHeader>
          </Card>
        </>
      )}

      {/* Selector de modelo de predicción */}
      {!showInitialResults && !showPredictionLoader && !showPredictionResults && (
        <ModelSelectorStep />
      )}

      {/* Loader de predicción */}
      {showPredictionLoader && (
        <PredictionLoaderStep />
      )}

      {/* Resultados de predicción */}
      {showPredictionResults && (
        <PredictionResultsStep />
      )}

      {/* Tabla de programas */}
      {showProgramsTable && (
        <ProgramsTableStep />
      )}

      {/* Acciones finales - siempre visibles */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap justify-center gap-4">
            {!predictionResult && (
              <Button
                onClick={handleViewDetailedResults}
                className="flex items-center gap-2"
              >
                <ArrowRight className="h-4 w-4" />
                Ver Resultados Detallados
              </Button>
            )}
            
            <Button
              onClick={handleNewAssessment}
              variant="destructive"
              className="flex items-center gap-2"
            >
              <RotateCcw className="h-4 w-4" />
              Nueva Evaluación
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Información adicional */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <Brain className="h-6 w-6 text-blue-600 mt-1" />
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                Sobre tus resultados
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                Esta evaluación proporciona una visión integral de tus competencias actuales. 
                Los resultados son una fotografía de tu perfil de talento en este momento y pueden 
                cambiar con el tiempo, la experiencia y el desarrollo profesional. 
                {predictionResult 
                  ? 'Las predicciones de áreas académicas se basan en análisis de patrones de estudiantes exitosos con perfiles similares al tuyo.'
                  : 'Te recomendamos usar estos insights para identificar oportunidades de crecimiento y planificar tu desarrollo profesional.'
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}