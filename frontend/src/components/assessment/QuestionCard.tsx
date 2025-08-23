'use client';

import React from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LikertScale } from './LikertScale';
import { QuestionCardProps } from '@/types/assessment';
import { DIMENSIONS_CONFIG } from '@/types/assessment';

export function QuestionCard({
  competency,
  questionNumber,
  totalQuestions,
  value,
  onChange,
  disabled = false
}: QuestionCardProps) {
  const dimension = DIMENSIONS_CONFIG.find(d => d.id === competency.dimensionId);
  
  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader className="space-y-4">
        <div className="flex items-center justify-between">
          <Badge 
            variant="secondary" 
            className="text-sm"
            style={{ backgroundColor: dimension?.color + '20', color: dimension?.color }}
          >
            {dimension?.name}
          </Badge>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Pregunta {questionNumber} de {totalQuestions}
          </span>
        </div>
        
        <div className="space-y-3">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            {competency.title}
          </h2>
          
          {competency.description && (
            <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
              {competency.description}
            </p>
          )}
        </div>
        
        <div className="border-t pt-4">
          <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
            ¿En qué medida esta afirmación te describe?
          </p>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        <LikertScale
          value={value?.value}
          onChange={onChange}
          disabled={disabled}
          size="lg"
        />
        
        {value && (
          <div className="text-center">
            <p className="text-sm text-green-600 dark:text-green-400">
              ✓ Respuesta guardada
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}