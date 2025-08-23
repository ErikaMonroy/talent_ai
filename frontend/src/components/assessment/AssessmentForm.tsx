'use client';

import React, { useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Home, 
  RotateCcw, 
  ArrowLeft, 
  ArrowRight,
  User,
  Brain
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { AssessmentHeader } from './AssessmentHeader';
import { DimensionNavigation } from './DimensionNavigation';
import { AssessmentNavigation } from './AssessmentNavigation';
import { DimensionPills } from './DimensionPills';
import { ValidationFeedback } from './ValidationFeedback';
import { WelcomeStep } from './steps/WelcomeStep';
import { PersonalDataStep } from './steps/PersonalDataStep';
import { IcfesStep } from './steps/IcfesStep';
import { DimensionStep } from './steps/DimensionStep';
import { ResultsStep } from './steps/ResultsStep';

const STEP_NAMES = [
  'Bienvenida',
  'Datos Personales',
  'Puntajes ICFES',
  'Razonamiento Lógico-Matemático',
  'Comunicación y Lenguaje',
  'Ciencias y Tecnología',
  'Humanidades y Ciencias Sociales',
  'Creatividad y Arte',
  'Gestión y Emprendimiento',
  'Habilidades Técnicas y Operativas',
  'Cuidado y Servicio',
  'Resultados'
];

const getStepNumber = (step: string): number => {
  switch (step) {
    case 'welcome': return 0;
    case 'personal-data': return 1;
    case 'icfes': return 2;
    case 'dimension': return 3;
    case 'results': return 11;
    default: return 0;
  }
};

export function AssessmentForm() {
  const {
    currentStep,
    currentDimension,
    session,
    initializeSession,
    getCompletionPercentage,
    goToStep,
    goToPreviousStep,
    resetAssessment,
    responses
  } = useAssessmentStore();
  
  const currentStepNumber = getStepNumber(currentStep);

  // Inicializar sesión al montar el componente
  useEffect(() => {
    if (!session) {
      initializeSession();
    }
  }, [session, initializeSession]);

  const completionPercentage = getCompletionPercentage();

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'welcome':
        return <WelcomeStep />;
      
      case 'personal-data':
        return <PersonalDataStep />;
      
      case 'icfes':
        return (
          <IcfesStep 
            onNext={() => goToStep('dimension')}
            onPrevious={() => goToStep('personal-data')}
          />
        );
      
      case 'dimension':
        return <DimensionStep />;
      
      case 'results':
        return <ResultsStep />;
      
      default:
        return (
          <Card>
            <CardContent>
              <p>Paso no encontrado</p>
            </CardContent>
          </Card>
        );
    }
  };

  const canGoBack = currentStepNumber > 0 && currentStepNumber < 11;
  const isInDimensionSteps = currentStep === 'dimension';

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      {currentStepNumber > 0 && currentStepNumber < 11 && (
        <AssessmentHeader 
          currentStep={currentStepNumber}
          totalSteps={11}
          stepName={STEP_NAMES[currentStepNumber]}
          completionPercentage={completionPercentage}
        />
      )}



      {/* Contenido principal */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-4xl">

          
          {renderCurrentStep()}
        </div>
      </div>

      {/* Navegación como footer */}
      {currentStepNumber > 0 && currentStepNumber < 11 && (
        <footer className="bg-card border-t border">
          <div className="max-w-4xl mx-auto p-4">
            <AssessmentNavigation />
          </div>
        </footer>
      )}
    </div>
  );
}

// Componente de navegación rápida entre pasos (para desarrollo/debug)
export function StepNavigation() {
  const { currentStep, goToStep } = useAssessmentStore();
  const currentStepNumber = getStepNumber(currentStep);
  
  return (
    <Card className="p-4 mb-4">
      <h3 className="text-sm font-medium mb-2">Navegación rápida (Debug)</h3>
      <div className="flex flex-wrap gap-1">
        {STEP_NAMES.map((name, index) => (
          <Button
            key={index}
            variant={currentStepNumber === index ? "default" : "outline"}
            size="sm"
            onClick={() => goToStep(index === 0 ? 'welcome' : index === 1 ? 'personal-data' : index === 2 ? 'icfes' : index === 11 ? 'results' : 'dimension')}
            className="text-xs"
          >
            {index}: {name}
          </Button>
        ))}
      </div>
    </Card>
  );
}