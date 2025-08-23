'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { CheckCircle, HelpCircle, Lightbulb } from 'lucide-react';
import { LikertScale } from './LikertScale';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface EnhancedQuestionCardProps {
  title: string;
  question: string;
  description?: string;
  questionNumber: number;
  totalQuestions: number;
  currentValue?: number;
  onChange: (value: number) => void;
  onNext?: () => void;
  onPrevious?: () => void;
  isRequired?: boolean;
  showProgress?: boolean;
  dimensionName?: string;
  competencyArea?: string;
  className?: string;
}

export function EnhancedQuestionCard({
  title,
  question,
  description,
  questionNumber,
  totalQuestions,
  currentValue,
  onChange,
  onNext,
  onPrevious,
  isRequired = true,
  showProgress = true,
  dimensionName,
  competencyArea,
  className
}: EnhancedQuestionCardProps) {
  const [showFeedback, setShowFeedback] = useState(false);
  const [hasAnswered, setHasAnswered] = useState(false);

  useEffect(() => {
    if (currentValue && !hasAnswered) {
      setHasAnswered(true);
      setShowFeedback(true);
      // Ocultar feedback después de 2 segundos
      const timer = setTimeout(() => setShowFeedback(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [currentValue, hasAnswered]);

  const handleValueChange = (value: number) => {
    onChange(value);
    if (!hasAnswered) {
      setHasAnswered(true);
      setShowFeedback(true);
      // Keep feedback visible until user manually advances
    }
  };

  const progressPercentage = (questionNumber / totalQuestions) * 100;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className={cn("w-full max-w-4xl mx-auto", className)}
    >
      <Card className="relative overflow-hidden border-2 transition-all duration-300 hover:shadow-lg">
        {/* Header con información contextual */}
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="text-sm">
                Pregunta {questionNumber} de {totalQuestions}
              </Badge>
              {dimensionName && (
                <Badge variant="secondary" className="text-sm">
                  {dimensionName}
                </Badge>
              )}
            </div>
            
            {/* Indicador de respuesta */}
            <AnimatePresence>
              {currentValue && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  exit={{ scale: 0 }}
                  className="flex items-center gap-2 text-green-600 dark:text-green-400"
                >
                  <CheckCircle className="h-5 w-5" />
                  <span className="text-sm font-medium">Respondida</span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Barra de progreso */}
          {showProgress && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Progreso de la dimensión</span>
                <span>{Math.round(progressPercentage)}%</span>
              </div>
              <Progress value={progressPercentage} className="h-2" />
            </div>
          )}
        </CardHeader>

        <CardContent className="space-y-6">


          {/* Título de dimensión */}
          <h2 className="text-2xl font-bold leading-relaxed text-foreground mb-4">
            {title}
          </h2>

          {/* Pregunta principal */}
          <div className="space-y-3">
            <h3 className="text-xl font-semibold leading-relaxed text-foreground">
              {question}
              {isRequired && <span className="text-destructive ml-1">*</span>}
            </h3>
            
            {/*description && (
              <p className="text-sm text-muted-foreground leading-relaxed">
                {description}
              </p>
            )*/}
          </div>

          {/* Escala Likert */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <HelpCircle className="h-4 w-4" />
              <span>Selecciona tu nivel de acuerdo con la afirmación:</span>
            </div>
            
            <LikertScale
              value={currentValue}
              onChange={handleValueChange}
            />
          </div>

          {/* Feedback visual */}
          <AnimatePresence>
            {showFeedback && currentValue && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
                  <div>
                    <p className="text-sm font-medium text-green-800 dark:text-green-200">
                      ¡Respuesta guardada!
                    </p>
                    <p className="text-xs text-green-600 dark:text-green-400">
                      Tu respuesta: {getLikertLabel(currentValue)}
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Botones de navegación */}
          <div className="flex items-center justify-between pt-4 border-t">
            <Button
              onClick={onPrevious}
              variant="outline"
              disabled={!onPrevious}
              className="flex items-center gap-2"
            >
              ← Anterior
            </Button>
            
            <Button
              onClick={onNext}
              disabled={!currentValue || !onNext}
              className="flex items-center gap-2"
            >
              Siguiente →
            </Button>
          </div>
        </CardContent>

        {/* Indicador visual de estado */}
        <div className={cn(
          "absolute top-0 left-0 w-1 h-full transition-all duration-300",
          currentValue ? "bg-success" : "bg-muted"
        )} />
      </Card>
    </motion.div>
  );
}

// Función auxiliar para obtener la etiqueta de la escala Likert
function getLikertLabel(value: number): string {
  const labels = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Neutral",
    4: "De acuerdo",
    5: "Totalmente de acuerdo"
  };
  return labels[value as keyof typeof labels] || "";
}

// Versión compacta para móviles
export function CompactQuestionCard({
  question,
  questionNumber,
  totalQuestions,
  currentValue,
  onChange,
  className
}: Pick<EnhancedQuestionCardProps, 'question' | 'questionNumber' | 'totalQuestions' | 'currentValue' | 'onChange' | 'className'>) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.2 }}
      className={cn("w-full", className)}
    >
      <Card className="border-2">
        <CardContent className="p-4 space-y-4">
          {/* Header compacto */}
          <div className="flex items-center justify-between">
            <Badge variant="outline" className="text-xs">
              {questionNumber}/{totalQuestions}
            </Badge>
            {currentValue && (
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
            )}
          </div>

          {/* Pregunta */}
          <h3 className="text-base font-medium leading-relaxed">
            {question}
          </h3>

          {/* Escala Likert compacta */}
          <LikertScale
            value={currentValue}
            onChange={onChange}
          />
        </CardContent>
      </Card>
    </motion.div>
  );
}