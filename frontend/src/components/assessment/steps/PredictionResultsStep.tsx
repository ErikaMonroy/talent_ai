'use client';

import React, { useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import { 
  Trophy,
  Target,
  Brain,
  TrendingUp,
  CheckCircle2,
  Star,
  Award,
  Sparkles,
  ArrowRight
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { AreaPrediction } from '@/types/api';

export function PredictionResultsStep() {
  const { 
    predictionResult,
    selectedModelType,
    selectedAreaIds,
    toggleAreaSelection,
    getTopPredictedAreas
  } = useAssessmentStore();

  const topAreas = getTopPredictedAreas(5);

  // Seleccionar automáticamente el área #1 recomendada al cargar
  useEffect(() => {
    if (topAreas.length > 0 && selectedAreaIds.length === 0) {
      toggleAreaSelection(topAreas[0].area_id);
    }
  }, [topAreas, selectedAreaIds.length, toggleAreaSelection]);

  const modelInfo = {
    knn: {
      name: 'K-Nearest Neighbors',
      icon: Target,
      color: 'blue',
      description: 'Basado en perfiles similares'
    },
    neural_network: {
      name: 'Red Neuronal Artificial',
      icon: Brain,
      color: 'purple',
      description: 'Análisis de patrones complejos'
    }
  };

  const currentModel = modelInfo[selectedModelType];
  const ModelIcon = currentModel.icon;

  const getPercentageColor = (percentage: number) => {
    if (percentage >= 80) return 'text-green-600';
    if (percentage >= 60) return 'text-blue-600';
    if (percentage >= 40) return 'text-yellow-600';
    return 'text-gray-600';
  };

  const getProgressColor = (percentage: number) => {
    if (percentage >= 80) return 'bg-green-500';
    if (percentage >= 60) return 'bg-blue-500';
    if (percentage >= 40) return 'bg-yellow-500';
    return 'bg-gray-500';
  };

  const getRankIcon = (index: number) => {
    switch (index) {
      case 0: return <Trophy className="h-5 w-5 text-yellow-500" />;
      case 1: return <Award className="h-5 w-5 text-gray-400" />;
      case 2: return <Star className="h-5 w-5 text-amber-600" />;
      default: return <TrendingUp className="h-4 w-4 text-gray-400" />;
    }
  };

  const getRankBadge = (index: number) => {
    const ranks = ['1°', '2°', '3°', '4°', '5°'];
    const colors = ['bg-yellow-500', 'bg-gray-400', 'bg-amber-600', 'bg-blue-500', 'bg-green-500'];
    
    return (
      <Badge className={`${colors[index]} text-white font-bold text-xs px-2 py-1`}>
        {ranks[index]}
      </Badge>
    );
  };

  if (!predictionResult || topAreas.length === 0) {
    return (
      <div className="max-w-4xl mx-auto">
        <Card>
          <CardContent className="pt-6 text-center">
            <p className="text-gray-500 dark:text-gray-400">
              No hay resultados de predicción disponibles.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Encabezado de resultados */}
      <Card className={`bg-gradient-to-r from-${currentModel.color}-50 to-${currentModel.color}-100 dark:from-${currentModel.color}-900/20 dark:to-${currentModel.color}-800/20 border-2 border-${currentModel.color}-200 dark:border-${currentModel.color}-800`}>
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            <div className={`p-4 bg-${currentModel.color}-600 rounded-full`}>
              <ModelIcon className="h-12 w-12 text-white" />
            </div>
          </div>
          
          <CardTitle className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            ¡Tu Análisis está Completo!
          </CardTitle>
          
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-2">
            Modelo utilizado: <span className="font-semibold">{currentModel.name}</span>
          </p>
          
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {currentModel.description}
          </p>
        </CardHeader>
      </Card>

      {/* Estadísticas generales */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="flex justify-center mb-2">
              <Target className="h-8 w-8 text-blue-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {topAreas.length}
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Áreas Recomendadas
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="flex justify-center mb-2">
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {Math.round(topAreas[0]?.percentage || 0)}%
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Mejor Afinidad
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6 text-center">
            <div className="flex justify-center mb-2">
              <CheckCircle2 className="h-8 w-8 text-purple-600" />
            </div>
            <div className="text-2xl font-bold text-gray-900 dark:text-white">
              {selectedAreaIds.length}
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Áreas Seleccionadas
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Top 5 Áreas Recomendadas */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-yellow-500" />
            Top 5 Áreas Académicas Recomendadas
          </CardTitle>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Selecciona las áreas que más te interesen para ver programas relacionados
          </p>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {topAreas.map((area, index) => {
              const isSelected = selectedAreaIds.includes(area.area_id);
              const isTopRecommendation = index === 0;
              
              return (
                <div 
                  key={area.area_id}
                  className={`relative p-4 rounded-lg border-2 transition-all duration-200 cursor-pointer hover:shadow-md ${
                    isSelected 
                      ? 'border-blue-300 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-700'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  } ${
                    isTopRecommendation 
                      ? 'ring-2 ring-yellow-400 ring-opacity-50'
                      : ''
                  }`}
                  onClick={() => toggleAreaSelection(area.area_id)}
                >
                  {/* Badge de ranking */}
                  <div className="absolute -top-2 -left-2">
                    {getRankBadge(index)}
                  </div>
                  
                  {/* Indicador de mejor recomendación */}
                  {isTopRecommendation && (
                    <div className="absolute -top-2 -right-2">
                      <Badge className="bg-yellow-500 text-white font-bold text-xs px-2 py-1">
                        <Star className="h-3 w-3 mr-1" />
                        Mejor Match
                      </Badge>
                    </div>
                  )}
                  
                  <div className="flex items-start gap-4">
                    {/* Checkbox */}
                    <div className="mt-1">
                      <Checkbox 
                        checked={isSelected}
                        onChange={() => toggleAreaSelection(area.area_id)}
                        className="h-5 w-5"
                      />
                    </div>
                    
                    {/* Icono de ranking */}
                    <div className="mt-1">
                      {getRankIcon(index)}
                    </div>
                    
                    {/* Contenido principal */}
                    <div className="flex-1 space-y-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <h3 className={`font-semibold text-lg ${
                            isSelected 
                              ? 'text-blue-900 dark:text-blue-100'
                              : 'text-gray-900 dark:text-white'
                          }`}>
                            {area.area_name}
                          </h3>
                          
                          {isTopRecommendation && (
                            <p className="text-sm text-yellow-600 dark:text-yellow-400 font-medium mt-1">
                              🎯 Recomendación principal basada en tu perfil
                            </p>
                          )}
                        </div>
                        
                        <div className="text-right">
                          <div className={`text-2xl font-bold ${getPercentageColor(area.percentage)}`}>
                            {Math.round(area.percentage)}%
                          </div>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            Afinidad
                          </p>
                        </div>
                      </div>
                      
                      {/* Barra de progreso */}
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">
                            Nivel de compatibilidad
                          </span>
                          <span className={`font-medium ${getPercentageColor(area.percentage)}`}>
                            {area.percentage >= 80 ? 'Excelente' :
                             area.percentage >= 60 ? 'Muy Bueno' :
                             area.percentage >= 40 ? 'Bueno' : 'Regular'}
                          </span>
                        </div>
                        
                        <div className="relative">
                          <Progress 
                            value={area.percentage} 
                            className="h-2"
                          />
                          <div 
                            className={`absolute top-0 left-0 h-2 rounded-full transition-all duration-500 ${getProgressColor(area.percentage)}`}
                            style={{ width: `${area.percentage}%` }}
                          />
                        </div>
                      </div>
                      
                      {/* Información adicional para el top 1 */}
                      {isTopRecommendation && (
                        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded-lg border border-yellow-200 dark:border-yellow-800">
                          <p className="text-sm text-yellow-800 dark:text-yellow-200">
                            <strong>¿Por qué es tu mejor opción?</strong> Esta área muestra la mayor compatibilidad con tu perfil académico y competencias evaluadas.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Información sobre la selección */}
      {selectedAreaIds.length > 0 && (
        <Card className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="h-6 w-6 text-green-600" />
              <div>
                <h4 className="font-medium text-green-900 dark:text-green-100">
                  {selectedAreaIds.length} área{selectedAreaIds.length > 1 ? 's' : ''} seleccionada{selectedAreaIds.length > 1 ? 's' : ''}
                </h4>
                <p className="text-sm text-green-700 dark:text-green-300">
                  A continuación podrás explorar los programas académicos relacionados con tu{selectedAreaIds.length > 1 ? 's' : ''} selección{selectedAreaIds.length > 1 ? 'es' : ''}.
                </p>
              </div>
              <ArrowRight className="h-5 w-5 text-green-600 ml-auto" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Nota metodológica */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <ModelIcon className="h-6 w-6 text-blue-600 mt-1" />
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                Sobre estos resultados
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                {selectedModelType === 'knn' 
                  ? 'Estos porcentajes representan la similitud de tu perfil con estudiantes exitosos en cada área académica. El modelo KNN analiza patrones históricos para identificar las mejores coincidencias.'
                  : 'Estos porcentajes indican la probabilidad de éxito y satisfacción en cada área académica según el análisis de la red neuronal. El modelo considera múltiples factores y sus interrelaciones complejas.'
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}