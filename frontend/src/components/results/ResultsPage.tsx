import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import {
  Trophy,
  Target,
  TrendingUp,
  Download,
  Share2,
  ArrowLeft,
  Star,
  Award,
  Brain,
  BarChart3,
  BookOpen,
  GraduationCap,
  Users,
  Lightbulb,
  AlertCircle,
  Loader2,
  ExternalLink
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { usePrediction, useProgramSearch } from '@/hooks/useApi';
import { DIMENSIONS_CONFIG } from '@/types/assessment';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

export function ResultsPage() {
  const navigate = useNavigate();
  const { predictionId } = useParams();
  const [showPredictions, setShowPredictions] = useState(false);
  
  const {
    personalData,
    responses,
    getDimensionProgress,
    getCompletionPercentage,
    calculateDimensionAverage,
    canProceedToResults
  } = useAssessmentStore();

  const {
    predict,
    prediction,
    loading: predictionLoading,
    error: predictionError
  } = usePrediction();

  const {
    searchPrograms,
    programs,
    loading: programsLoading,
    error: programsError
  } = useProgramSearch();

  // Efecto para hacer la predicción al cargar el componente
  useEffect(() => {
    const makePrediction = async () => {
      try {
        await predict('knn');
      } catch (error) {
        console.error('Error making prediction:', error);
      }
    };
    
    if (personalData.name && Object.keys(responses).length > 0) {
      makePrediction();
    }
  }, [personalData, responses, predict]);

  // Comentado: Permitir acceso a resultados sin navegación automática
  // useEffect(() => {
  //   if (!canProceedToResults()) {
  //     navigate('/assessment');
  //     return;
  //   }
  // }, [canProceedToResults, navigate]);

  // Generar predicción automáticamente
  useEffect(() => {
    if (canProceedToResults() && !prediction && !predictionLoading) {
      predict('knn');
    }
  }, [canProceedToResults, prediction, predictionLoading, predict, personalData, responses]);

  // Buscar programas cuando hay predicción
  useEffect(() => {
    if (prediction && prediction.predictions.length > 0) {
      const topArea = prediction.predictions[0];
      searchPrograms({ area_id: topArea.area_id, limit: 10 });
      setShowPredictions(true);
    }
  }, [prediction, searchPrograms]);

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

  // Preparar datos para gráficos
  const chartData = dimensionResults.map(d => ({
    name: d.name.split(' ').slice(0, 2).join(' '), // Acortar nombres
    value: d.average,
    fullName: d.name,
    progress: d.progress
  }));

  const radarData = dimensionResults.map(d => ({
    dimension: d.name.split(' ')[0], // Primera palabra
    value: d.average,
    fullName: d.name
  }));

  // Estadísticas generales
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

  const handleDownloadResults = () => {
    // TODO: Implementar descarga de resultados
    console.log('Descargando resultados...');
  };

  const handleShareResults = () => {
    // Implementar funcionalidad de compartir resultados
    if (navigator.share) {
      navigator.share({
        title: 'Mis Resultados TalentAI',
        text: 'He completado mi evaluación de competencias en TalentAI',
        url: window.location.href
      });
    } else {
      // Fallback: copiar URL al portapapeles
      navigator.clipboard.writeText(window.location.href);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'bg-green-500';
    if (confidence >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getMatchColor = (match: number) => {
    if (match >= 85) return 'text-green-600';
    if (match >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Mostrar estado de carga
  if (predictionLoading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <Card>
          <CardContent className="flex items-center justify-center py-12">
            <div className="text-center space-y-4">
              <Loader2 className="h-8 w-8 animate-spin mx-auto text-blue-600" />
              <h3 className="text-lg font-medium">Analizando tus respuestas...</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Estamos procesando tu evaluación para generar recomendaciones personalizadas.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Mostrar error si ocurre
  if (predictionError) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <Card>
          <CardContent className="py-12">
            <div className="text-center space-y-4">
              <AlertCircle className="h-8 w-8 mx-auto text-red-600" />
              <h3 className="text-lg font-medium text-red-600">Error al generar resultados</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Hubo un problema al procesar tu evaluación. Por favor, intenta nuevamente.
              </p>
              <Button onClick={() => navigate('/evaluation')} variant="outline">
                Volver a la evaluación
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }



  const handleBackToAssessment = () => {
    navigate('/assessment');
  };

  if (!canProceedToResults()) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        {/* Encabezado */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              onClick={handleBackToAssessment}
              variant="outline"
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Volver a Evaluación
            </Button>
            
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Resultados Detallados - TalentAI
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Análisis completo de tu perfil de talento y recomendaciones personalizadas
              </p>
            </div>
          </div>
          
          <div className="flex gap-3">
            <Button
              onClick={handleDownloadResults}
              className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
            >
              <Download className="h-4 w-4" />
              Descargar PDF
            </Button>
            
            <Button
              onClick={handleShareResults}
              variant="outline"
              className="flex items-center gap-2"
            >
              <Share2 className="h-4 w-4" />
              Compartir
            </Button>
          </div>
        </div>

        {/* Resumen ejecutivo */}
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border-2 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl">
              <Trophy className="h-8 w-8 text-yellow-500" />
              Resumen Ejecutivo - {personalData.name}
            </CardTitle>
          </CardHeader>
          
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-1">
                  {overallAverage.toFixed(1)}/5.0
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Puntuación General
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-1">
                  {completedDimensions.length}/8
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Dimensiones Completadas
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-1">
                  {topStrengths.length}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Fortalezas Principales
                </div>
              </div>
              
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-600 mb-1">
                  {Math.round(getCompletionPercentage())}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  Evaluación Completada
                </div>
              </div>
            </div>
            
            {showPredictions && prediction && (
              <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                <h3 className="font-semibold text-lg mb-3 flex items-center gap-2">
                  <Lightbulb className="h-5 w-5 text-yellow-500" />
                  Recomendación Principal
                </h3>
                <p className="text-gray-700 dark:text-gray-300 mb-2">
                  Basado en tu perfil de talento, tu área más recomendada es:
                </p>
                <div className="flex items-center gap-3">
                  <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 text-lg px-3 py-1">
                    {prediction.predictions[0]?.area_name}
                  </Badge>
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Compatibilidad: {prediction.predictions[0]?.percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Visualizaciones */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Gráfico de barras - Puntuaciones por dimensión */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                Puntuaciones por Dimensión
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="name" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                  />
                  <YAxis domain={[0, 5]} />
                  <Tooltip 
                    formatter={(value, name) => [typeof value === 'number' ? value.toFixed(2) : value, 'Puntuación']}
                    labelFormatter={(label) => {
                      const item = chartData.find(d => d.name === label);
                      return item?.fullName || label;
                    }}
                  />
                  <Bar dataKey="value" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Gráfico radar - Perfil de competencias */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-purple-500" />
                Perfil de Competencias
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" fontSize={12} />
                  <PolarRadiusAxis domain={[0, 5]} tick={false} />
                  <Radar
                    name="Puntuación"
                    dataKey="value"
                    stroke="#8B5CF6"
                    fill="#8B5CF6"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                  <Tooltip 
                    formatter={(value) => [typeof value === 'number' ? value.toFixed(2) : value, 'Puntuación']}
                    labelFormatter={(label) => {
                      const item = radarData.find(d => d.dimension === label);
                      return item?.fullName || label;
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Predicciones de IA */}
        {showPredictions && prediction && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-green-500" />
                Predicciones de IA - Áreas Recomendadas
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              {predictionLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600 dark:text-gray-400">Generando predicciones...</p>
                </div>
              ) : predictionError ? (
                <div className="text-center py-8">
                  <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-4" />
                  <p className="text-red-600 dark:text-red-400">Error al generar predicciones</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {prediction.predictions.map((area, index) => (
                    <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold text-gray-900 dark:text-white">
                          {area.area_name}
                        </h4>
                        <Badge 
                          variant={index === 0 ? "default" : "outline"}
                          className={index === 0 ? "bg-green-100 text-green-800" : ""}
                        >
                          #{index + 1}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600 dark:text-gray-400">Compatibilidad:</span>
                          <span className="font-medium">
                            {area.percentage.toFixed(1)}%
                          </span>
                        </div>
                        
                        <Progress value={area.percentage} className="h-2" />
                        
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                          Área recomendada basada en tu perfil de competencias
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Programas académicos recomendados */}
        {programs && programs.programs && programs.programs.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GraduationCap className="h-5 w-5 text-indigo-500" />
                Programas Académicos Recomendados
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              {programsLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-4"></div>
                  <p className="text-gray-600 dark:text-gray-400">Buscando programas...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {programs.programs.slice(0, 6).map((program, index) => (
                    <div key={program.id} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-900 dark:text-white text-sm">
                          {program.name}
                        </h4>
                        <Badge variant="outline" className="text-xs">
                          {program.academic_level}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        {program.institution}
                      </p>
                      
                      <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                        <span className="flex items-center gap-1">
                          <Users className="h-3 w-3" />
                          Área {program.knowledge_area.name}
                        </span>
                        <span>Nivel: {program.academic_level}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Análisis detallado por dimensión */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Award className="h-5 w-5 text-yellow-500" />
              Análisis Detallado por Dimensión
            </CardTitle>
          </CardHeader>
          
          <CardContent>
            <div className="space-y-6">
              {dimensionResults.map((dimension, index) => (
                <div key={dimension.id}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{dimension.icon}</span>
                      <div>
                        <h4 className="font-semibold text-gray-900 dark:text-white">
                          {dimension.name}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {dimension.description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <div className={`text-2xl font-bold ${dimension.color}`}>
                        {dimension.average.toFixed(1)}/5.0
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {dimension.level}
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Progreso de evaluación:</span>
                        <span>{Math.round(dimension.progress)}%</span>
                      </div>
                      <Progress value={dimension.progress} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Nivel de competencia:</span>
                        <span>{(dimension.average / 5 * 100).toFixed(0)}%</span>
                      </div>
                      <Progress value={dimension.average / 5 * 100} className="h-2" />
                    </div>
                  </div>
                  
                  {index < dimensionResults.length - 1 && (
                    <Separator className="mt-6" />
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recomendaciones de desarrollo */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Fortalezas */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Star className="h-5 w-5 text-yellow-500" />
                Principales Fortalezas
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              {topStrengths.length > 0 ? (
                <div className="space-y-4">
                  {topStrengths.map((dimension, index) => (
                    <div key={dimension.id} className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <div className="text-2xl">{dimension.icon}</div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {dimension.name}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Puntuación: {dimension.average.toFixed(1)}/5.0 - {dimension.level}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          Esta es una de tus competencias más desarrolladas. Considera roles que aprovechen esta fortaleza.
                        </p>
                      </div>
                      <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                        #{index + 1}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-600 dark:text-gray-400 text-center py-8">
                  Completa más dimensiones para identificar tus fortalezas principales.
                </p>
              )}
            </CardContent>
          </Card>

          {/* Áreas de desarrollo */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-blue-500" />
                Oportunidades de Desarrollo
              </CardTitle>
            </CardHeader>
            
            <CardContent>
              {developmentAreas.length > 0 ? (
                <div className="space-y-4">
                  {developmentAreas.map((dimension) => (
                    <div key={dimension.id} className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="text-2xl">{dimension.icon}</div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 dark:text-white">
                          {dimension.name}
                        </h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Puntuación: {dimension.average.toFixed(1)}/5.0 - {dimension.level}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          Área con potencial de crecimiento. Considera capacitación o experiencias que fortalezcan esta competencia.
                        </p>
                      </div>
                      <Badge variant="outline" className="text-blue-600 border-blue-600">
                        Desarrollar
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-600 dark:text-gray-400 text-center py-8">
                  ¡Excelente! No se identificaron áreas críticas de desarrollo.
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Información adicional */}
        <Card className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <BookOpen className="h-8 w-8 text-blue-600 mt-1" />
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-white mb-3 text-lg">
                  Sobre tu Evaluación TalentAI
                </h4>
                <div className="prose prose-sm text-gray-600 dark:text-gray-400 max-w-none">
                  <p className="mb-3">
                    Esta evaluación integral analiza 8 dimensiones clave de tu perfil profesional utilizando 
                    inteligencia artificial avanzada para generar recomendaciones personalizadas.
                  </p>
                  <p className="mb-3">
                    <strong>Metodología:</strong> Basada en modelos psicométricos validados y algoritmos de 
                    machine learning que correlacionan tus respuestas con perfiles de éxito en diferentes áreas profesionales.
                  </p>
                  <p>
                    <strong>Recomendación:</strong> Utiliza estos resultados como guía para tu desarrollo profesional. 
                    Los resultados pueden evolucionar con el tiempo, la experiencia y el aprendizaje continuo.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default ResultsPage;