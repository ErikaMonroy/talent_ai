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
  ArrowRight,
  Loader2
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { useMultipleAreas } from '@/hooks/useAreas';
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
  const areaIds = topAreas.map(area => area.area_id);
  const { areasInfo, loading: areasLoading } = useMultipleAreas(areaIds);

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

      {/* Estadísticas compactas */}
      <Card>
        <CardContent className="py-4">
          <div className="flex justify-around items-center">
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-blue-600" />
              <div className="text-center">
                <div className="text-lg font-bold text-gray-900 dark:text-white">{topAreas.length}</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Recomendadas</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 text-purple-600" />
              <div className="text-center">
                <div className="text-lg font-bold text-gray-900 dark:text-white">{selectedAreaIds.length}</div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Seleccionadas</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Top 5 Áreas Recomendadas */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-yellow-500" />
            Top 4 Áreas Académicas Recomendadas
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Áreas con mayor compatibilidad según tu perfil académico
          </p>
        </CardHeader>
        
        <CardContent>
          <div className="space-y-4">
            {areasLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin mr-2" />
                <span className="text-gray-500 dark:text-gray-400">Cargando información de áreas...</span>
              </div>
            ) : (
              topAreas.slice(0, 4).map((area, index) => {
                const isTopRecommendation = index === 0;
                const areaInfo = areasInfo[area.area_id];
                
                return (
                  <div 
                    key={area.area_id}
                    className={`relative p-4 rounded-lg border bg-card text-card-foreground transition-all duration-200 ${
                      isTopRecommendation 
                        ? 'border-primary/20 bg-primary/5'
                        : 'border-border'
                    }`}
                  >
                    {/* Badge de ranking */}
                    <div className="absolute -top-2 -left-2">
                      {getRankBadge(index)}
                    </div>
                    
                    {/* Indicador de mejor recomendación */}
                    {isTopRecommendation && (
                      <div className="absolute -top-2 -right-2">
                        <Badge className="bg-primary text-primary-foreground font-bold text-xs px-2 py-1">
                          <Star className="h-3 w-3 mr-1" />
                          Mejor Match
                        </Badge>
                      </div>
                    )}
                    
                    <div className="flex items-start gap-4">
                      {/* Icono de ranking */}
                      <div className="mt-1">
                        {getRankIcon(index)}
                      </div>
                      
                      {/* Contenido principal */}
                      <div className="flex-1 space-y-3">
                        <div className="flex items-start">
                          <div className="flex-1">
                            <h3 className="font-semibold text-lg text-foreground">
                              {areaInfo?.name || `Área ${area.area_id}`}
                            </h3>
                            
                            {areaInfo?.description && (
                              <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                                {areaInfo.description}
                              </p>
                            )}
                            
                            {areaInfo?.category && (
                              <Badge variant="outline" className="mt-2 text-xs">
                                {areaInfo.category}
                              </Badge>
                            )}
                          </div>
                          

                        </div>
                        

                      </div>
                    </div>
                  </div>
                );
              })
            )}
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