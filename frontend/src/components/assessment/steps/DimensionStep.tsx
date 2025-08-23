'use client';

import React from 'react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { EnhancedQuestionCard } from '../EnhancedQuestionCard';
import { DIMENSIONS_CONFIG } from '@/types/assessment';
import { DIMENSIONS } from '@/data/dimensions';

export function DimensionStep() {
  const {
    currentDimension,
    currentQuestionIndex,
    responses,
    saveResponse,
    nextQuestion,
    previousQuestion,
    nextDimension,
    previousDimension,
    goToStep,
    dimensionProgress,
    resetAssessment
  } = useAssessmentStore();

  const dimension = DIMENSIONS_CONFIG.find(d => d.id === currentDimension);
  const dimensionData = DIMENSIONS.find(d => d.id === currentDimension);
  const competencies = dimensionData?.competencies || [];
  const currentCompetency = competencies[currentQuestionIndex];
  const progress = dimensionProgress[currentDimension] || { completed: 0, total: competencies.length };
  
  if (!dimension || !currentCompetency) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">Cargando pregunta...</p>
      </div>
    );
  }

  const handleResponse = (value: number) => {
    saveResponse(currentCompetency.id, value);
  };

  const handleNext = () => {
    if (currentQuestionIndex < competencies.length - 1) {
      nextQuestion();
    } else {
      // Last question of dimension
      if (currentDimension < 8) {
        nextDimension();
      } else {
        // Last dimension, go to results
        goToStep('results');
      }
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      previousQuestion();
    } else {
      // First question of dimension
      if (currentDimension > 1) {
        previousDimension();
      } else {
        // First dimension, go to personal data
        goToStep('personal-data');
      }
    }
  };

  const currentResponse = responses[currentCompetency.id];
  const totalQuestions = Object.values(dimensionProgress).reduce((acc, curr) => acc + curr.total, 0);
  const completedQuestions = Object.values(dimensionProgress).reduce((acc, curr) => acc + curr.completed, 0);

  const getNextLabel = () => {
    if (currentQuestionIndex === competencies.length - 1) {
      return currentDimension === 8 ? 'Ver Resultados' : 'Siguiente Dimensión';
    }
    return 'Siguiente';
  };

  return (
    <div className="space-y-6">
      {/* Dimension Header */}
      <div className="text-center space-y-2">

      </div>



      {/* Enhanced Question Card */}
      <EnhancedQuestionCard
        title={dimension.description}
        question={currentCompetency.title}
        description={currentCompetency.description}
        questionNumber={currentQuestionIndex + 1}
        totalQuestions={competencies.length}
        currentValue={currentResponse?.value}
        onChange={handleResponse}
        onNext={handleNext}
        onPrevious={handlePrevious}
        dimensionName={dimension.name}
        competencyArea={dimension.name}
        showProgress={true}
      />


    </div>
  );
}