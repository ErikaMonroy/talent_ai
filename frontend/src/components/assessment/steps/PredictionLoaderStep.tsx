'use client';

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { 
  Brain,
  Target,
  Loader2,
  Sparkles,
  Database,
  TrendingUp,
  CheckCircle
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';

export function PredictionLoaderStep() {
  const { 
    selectedModelType,
    isLoadingPrediction,
    predictionResult,
    predictionError
  } = useAssessmentStore();

  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

  const modelInfo = {
    knn: {
      name: 'K-Nearest Neighbors',
      icon: Target,
      color: 'blue',
      estimatedTime: 3000 // 3 segundos
    },
    neural_network: {
      name: 'Red Neuronal Artificial',
      icon: Brain,
      color: 'purple',
      estimatedTime: 6000 // 6 segundos
    }
  };

  const processingSteps = [
    {
      id: 1,
      title: 'Preparando datos',
      description: 'Procesando tu perfil académico y competencias',
      icon: Database,
      duration: 1000
    },
    {
      id: 2,
      title: 'Analizando patrones',
      description: selectedModelType === 'knn' 
        ? 'Comparando con perfiles similares en la base de datos'
        : 'Procesando datos a través de la red neuronal',
      icon: selectedModelType === 'knn' ? Target : Brain,
      duration: selectedModelType === 'knn' ? 1500 : 3000
    },
    {
      id: 3,
      title: 'Calculando afinidades',
      description: 'Determinando las áreas académicas más compatibles',
      icon: TrendingUp,
      duration: 1000
    },
    {
      id: 4,
      title: 'Generando resultados',
      description: 'Preparando tu reporte personalizado de recomendaciones',
      icon: Sparkles,
      duration: 500
    }
  ];

  useEffect(() => {
    if (!isLoadingPrediction) {
      setProgress(0);
      setCurrentStep(0);
      return;
    }

    let totalDuration = 0;
    let currentDuration = 0;
    
    const intervals: NodeJS.Timeout[] = [];
    
    processingSteps.forEach((step, index) => {
      totalDuration += step.duration;
      
      const timeout = setTimeout(() => {
        setCurrentStep(index + 1);
        
        // Animar progreso para este paso
        const stepInterval = setInterval(() => {
          currentDuration += 50;
          const newProgress = Math.min((currentDuration / totalDuration) * 100, 100);
          setProgress(newProgress);
          
          if (currentDuration >= totalDuration) {
            clearInterval(stepInterval);
          }
        }, 50);
        
        intervals.push(stepInterval);
      }, currentDuration);
      
      currentDuration += step.duration;
    });

    return () => {
      intervals.forEach(clearInterval);
    };
  }, [isLoadingPrediction, selectedModelType]);

  const currentModel = modelInfo[selectedModelType];
  const CurrentIcon = currentModel.icon;
  const currentProcessingStep = processingSteps[currentStep - 1];
  const StepIcon = currentProcessingStep?.icon || Loader2;

  if (predictionError) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card className="bg-red-50 dark:bg-red-900/20 border-2 border-red-200 dark:border-red-800">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="p-4 bg-red-600 rounded-full">
                <Target className="h-12 w-12 text-white" />
              </div>
            </div>
            
            <CardTitle className="text-2xl font-bold text-red-900 dark:text-red-100 mb-2">
              Error en la Predicción
            </CardTitle>
            
            <p className="text-red-700 dark:text-red-300">
              {predictionError}
            </p>
          </CardHeader>
        </Card>
      </div>
    );
  }

  if (predictionResult) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card className="bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="p-4 bg-green-600 rounded-full">
                <CheckCircle className="h-12 w-12 text-white" />
              </div>
            </div>
            
            <CardTitle className="text-2xl font-bold text-green-900 dark:text-green-100 mb-2">
              ¡Predicción Completada!
            </CardTitle>
            
            <p className="text-green-700 dark:text-green-300">
              Tu análisis de afinidad académica está listo. Los resultados se mostrarán a continuación.
            </p>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Encabezado */}
      <Card className={`bg-gradient-to-r from-${currentModel.color}-50 to-${currentModel.color}-100 dark:from-${currentModel.color}-900/20 dark:to-${currentModel.color}-800/20 border-2 border-${currentModel.color}-200 dark:border-${currentModel.color}-800`}>
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className={`p-4 bg-${currentModel.color}-600 rounded-full animate-pulse`}>
              <CurrentIcon className="h-12 w-12 text-white" />
            </div>
          </div>
          
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Analizando tu Perfil
          </CardTitle>
          
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Usando {currentModel.name} para predecir tus áreas académicas más afines
          </p>
        </CardHeader>
      </Card>

      {/* Progreso principal */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-6">
            {/* Barra de progreso */}
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="font-medium text-gray-700 dark:text-gray-300">
                  Progreso del análisis
                </span>
                <span className="text-gray-500 dark:text-gray-400">
                  {Math.round(progress)}%
                </span>
              </div>
              
              <Progress 
                value={progress} 
                className="h-3"
              />
            </div>

            {/* Paso actual */}
            {currentProcessingStep && (
              <div className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                <div className={`p-3 bg-${currentModel.color}-100 dark:bg-${currentModel.color}-900/30 rounded-full`}>
                  <StepIcon className={`h-6 w-6 text-${currentModel.color}-600 animate-spin`} />
                </div>
                
                <div className="flex-1">
                  <h3 className="font-medium text-gray-900 dark:text-white">
                    {currentProcessingStep.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {currentProcessingStep.description}
                  </p>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Pasos del proceso */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5" />
            Proceso de Análisis
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {processingSteps.map((step, index) => {
              const StepIcon = step.icon;
              const isCompleted = currentStep > step.id;
              const isCurrent = currentStep === step.id;
              const isPending = currentStep < step.id;
              
              return (
                <div 
                  key={step.id}
                  className={`flex items-center gap-4 p-3 rounded-lg transition-all duration-300 ${
                    isCompleted 
                      ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                      : isCurrent 
                        ? `bg-${currentModel.color}-50 dark:bg-${currentModel.color}-900/20 border border-${currentModel.color}-200 dark:border-${currentModel.color}-800`
                        : 'bg-gray-50 dark:bg-gray-800/50'
                  }`}
                >
                  <div className={`p-2 rounded-full ${
                    isCompleted 
                      ? 'bg-green-100 dark:bg-green-900/30'
                      : isCurrent 
                        ? `bg-${currentModel.color}-100 dark:bg-${currentModel.color}-900/30`
                        : 'bg-gray-100 dark:bg-gray-700'
                  }`}>
                    {isCompleted ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <StepIcon className={`h-5 w-5 ${
                        isCurrent 
                          ? `text-${currentModel.color}-600 animate-pulse`
                          : 'text-gray-400'
                      }`} />
                    )}
                  </div>
                  
                  <div className="flex-1">
                    <h4 className={`font-medium ${
                      isCompleted 
                        ? 'text-green-900 dark:text-green-100'
                        : isCurrent 
                          ? 'text-gray-900 dark:text-white'
                          : 'text-gray-500 dark:text-gray-400'
                    }`}>
                      {step.title}
                    </h4>
                    <p className={`text-sm ${
                      isCompleted 
                        ? 'text-green-700 dark:text-green-300'
                        : isCurrent 
                          ? 'text-gray-600 dark:text-gray-300'
                          : 'text-gray-400 dark:text-gray-500'
                    }`}>
                      {step.description}
                    </p>
                  </div>
                  
                  {isCompleted && (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Información del modelo */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <CurrentIcon className="h-6 w-6 text-blue-600 mt-1" />
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                Sobre el modelo {currentModel.name}
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                {selectedModelType === 'knn' 
                  ? 'Este modelo analiza tu perfil comparándolo con miles de casos similares en nuestra base de datos, identificando patrones de éxito académico para recomendarte las áreas más afines.'
                  : 'Esta red neuronal artificial utiliza algoritmos de aprendizaje profundo para analizar relaciones complejas entre tus competencias y las áreas académicas, proporcionando predicciones altamente precisas y personalizadas.'
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}