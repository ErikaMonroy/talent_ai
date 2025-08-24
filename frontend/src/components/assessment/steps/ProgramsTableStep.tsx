'use client';

import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { BookOpen } from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { useMultipleAreas } from '@/hooks/useAreas';
import { ProgramsTable } from '@/components/programs/ProgramsTable';

interface ProgramsTableStepProps {
  className?: string;
}

export function ProgramsTableStep({ className }: ProgramsTableStepProps) {
  const { selectedAreaIds, getTopPredictedAreas } = useAssessmentStore();
  
  // Obtener información de áreas recomendadas
  const recommendedAreas = getTopPredictedAreas();
  const recommendedAreaIds = recommendedAreas.map(area => area.area_id);
  const { areasInfo } = useMultipleAreas(recommendedAreaIds);



  if (selectedAreaIds.length === 0) {
    return (
      <div className={`max-w-6xl mx-auto ${className}`}>
        <Card>
          <CardContent className="pt-6 text-center py-12">
            <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              Selecciona áreas de interés
            </h3>
            <p className="text-gray-500 dark:text-gray-400">
              Para ver programas académicos, primero selecciona al menos un área de las recomendaciones anteriores.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      <ProgramsTable 
         initialAreaId={selectedAreaIds[0]}
         showAreaRecommendations={true}
         recommendedAreas={recommendedAreas}
         areasInfo={areasInfo}
         title="Programas Encontrados"
       />
    </div>
  );
}