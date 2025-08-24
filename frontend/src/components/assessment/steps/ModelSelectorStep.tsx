'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Brain,
  Network,
  Zap,
  Target,
  Clock,
  TrendingUp,
  ArrowRight,
  Info
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { apiService, prepareFormDataForPrediction } from '@/services/api';
import { toast } from 'sonner';

export function ModelSelectorStep() {
  const { 
    personalData,
    responses,
    selectedModelType,
    setSelectedModelType,
    setLoadingPrediction,
    setPredictionResult,
    setPredictionError,
    calculateDimensionAverage
  } = useAssessmentStore();

  const handleModelSelect = (modelType: 'knn' | 'neural_network') => {
    setSelectedModelType(modelType);
  };

  const handleStartPrediction = async () => {
    if (!personalData.icfesScores.matematicas || !personalData.name) {
      toast.error('Datos incompletos', {
        description: 'Por favor completa todos los datos personales y puntajes ICFES.'
      });
      return;
    }

    setLoadingPrediction(true);
    setPredictionError(null);

    try {
      // Preparar datos para la predicción
      const dimensionAverages: Record<number, number> = {};
      for (let i = 1; i <= 8; i++) {
        dimensionAverages[i] = calculateDimensionAverage(i);
      }

      const predictionInput = prepareFormDataForPrediction(
        personalData.icfesScores.matematicas ? `${personalData.name.toLowerCase().replace(/\s+/g, '.')}@talentai.com` : '',
        personalData.icfesScores,
        dimensionAverages,
        selectedModelType
      );

      console.log('Datos de predicción enviados:', predictionInput);

      const response = await apiService.predict(predictionInput);
      
      if (response.success && response.data) {
        setPredictionResult(response.data);
        toast.success('¡Predicción completada!', {
          description: 'Los resultados están listos para revisar.'
        });
      } else {
        throw new Error(response.error?.message || 'Error en la predicción');
      }
    } catch (error) {
      console.error('Error en la predicción:', error);
      let errorMessage = 'Error desconocido';
      
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'object' && error !== null) {
        // Manejar errores de validación 422
        if ('code' in error && error.code === '422') {
          errorMessage = 'Error de validación: Algunos datos no cumplen con los requisitos del servidor.';
          if ('details' in error && error.details) {
            console.error('Detalles del error 422:', error.details);
          }
        } else {
          errorMessage = JSON.stringify(error);
        }
      }
      
      setPredictionError(errorMessage);
      toast.error('Error en la predicción', {
        description: errorMessage
      });
    } finally {
      setLoadingPrediction(false);
    }
  };

  const modelOptions = [
    {
      id: 'knn' as const,
      name: 'K-Nearest Neighbors (KNN)',
      shortName: 'KNN',
      description: 'Modelo basado en similitud que encuentra patrones comparando tu perfil con casos similares.',
      icon: Target,
      features: [
        'Rápido y eficiente',
        'Basado en casos similares',
        'Fácil interpretación',
        'Ideal para perfiles estándar'
      ],
      pros: [
        'Resultados rápidos',
        'Método probado y confiable',
        'Buena precisión general'
      ],
      processingTime: '~2-3 segundos',
      accuracy: '85-90%',
      color: 'blue'
    },
    {
      id: 'neural_network' as const,
      name: 'Red Neuronal Artificial',
      shortName: 'Red Neuronal',
      description: 'Modelo avanzado de inteligencia artificial que aprende patrones complejos y relaciones no lineales.',
      icon: Brain,
      features: [
        'Inteligencia artificial avanzada',
        'Detecta patrones complejos',
        'Adaptación continua',
        'Ideal para perfiles únicos'
      ],
      pros: [
        'Mayor precisión',
        'Análisis más profundo',
        'Mejor para casos complejos'
      ],
      processingTime: '~5-8 segundos',
      accuracy: '90-95%',
      color: 'purple'
    }
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Encabezado */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border-2 border-blue-200 dark:border-blue-800">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className="p-4 bg-blue-600 rounded-full">
              <Network className="h-12 w-12 text-white" />
            </div>
          </div>
          
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Selecciona el Modelo de Predicción
          </CardTitle>
          
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Elige el tipo de inteligencia artificial que analizará tu perfil para predecir las áreas académicas más afines a ti.
          </p>
        </CardHeader>
      </Card>

      {/* Comparación de modelos */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {modelOptions.map((model) => {
          const Icon = model.icon;
          const isSelected = selectedModelType === model.id;
          
          return (
            <Card 
              key={model.id}
              className={`cursor-pointer transition-all duration-300 hover:shadow-lg ${
                isSelected 
                  ? `ring-2 ring-${model.color}-500 border-${model.color}-300 bg-${model.color}-50 dark:bg-${model.color}-900/20` 
                  : 'hover:border-gray-300 dark:hover:border-gray-600'
              }`}
              onClick={() => handleModelSelect(model.id)}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-full ${
                      model.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/30' : 'bg-purple-100 dark:bg-purple-900/30'
                    }`}>
                      <Icon className={`h-6 w-6 ${
                        model.color === 'blue' ? 'text-blue-600' : 'text-purple-600'
                      }`} />
                    </div>
                    <div>
                      <CardTitle className="text-xl">{model.shortName}</CardTitle>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {model.name}
                      </p>
                    </div>
                  </div>
                  
                  {isSelected && (
                    <Badge className={`${
                      model.color === 'blue' ? 'bg-blue-600' : 'bg-purple-600'
                    } text-white`}>
                      Seleccionado
                    </Badge>
                  )}
                </div>
                
                <p className="text-gray-700 dark:text-gray-300 mt-3">
                  {model.description}
                </p>
              </CardHeader>
              
              <CardContent>
                {/* Características */}
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                      <Zap className="h-4 w-4" />
                      Características
                    </h4>
                    <ul className="space-y-1">
                      {model.features.map((feature, index) => (
                        <li key={index} className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                          <div className="w-1.5 h-1.5 bg-gray-400 rounded-full" />
                          {feature}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Ventajas */}
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2 flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Ventajas
                    </h4>
                    <ul className="space-y-1">
                      {model.pros.map((pro, index) => (
                        <li key={index} className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2">
                          <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                          {pro}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  {/* Métricas */}
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <Clock className="h-3 w-3 text-gray-500" />
                        <span className="text-xs text-gray-500">Tiempo</span>
                      </div>
                      <div className="text-sm font-medium">{model.processingTime}</div>
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <Target className="h-3 w-3 text-gray-500" />
                        <span className="text-xs text-gray-500">Precisión</span>
                      </div>
                      <div className="text-sm font-medium">{model.accuracy}</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Información adicional */}
      <Card className="bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <Info className="h-5 w-5 text-amber-600 mt-0.5" />
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                ¿Cuál modelo elegir?
              </h4>
              <div className="text-sm text-gray-700 dark:text-gray-300 space-y-2">
                <p>
                  <strong>Elige KNN si:</strong> Prefieres resultados rápidos y confiables, o si tu perfil se ajusta a patrones académicos tradicionales.
                </p>
                <p>
                  <strong>Elige Red Neuronal si:</strong> Buscas el análisis más preciso posible, tienes un perfil único o multidisciplinario, o no te importa esperar un poco más por mejores resultados.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Botón de acción */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex justify-center">
            <Button
              onClick={handleStartPrediction}
              size="lg"
              className="flex items-center gap-2 px-8 py-3 text-lg"
              disabled={!selectedModelType}
            >
              <ArrowRight className="h-5 w-5" />
              Iniciar Predicción con {modelOptions.find(m => m.id === selectedModelType)?.shortName}
            </Button>
          </div>
          
          {!selectedModelType && (
            <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-3">
              Selecciona un modelo para continuar
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}