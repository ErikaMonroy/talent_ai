'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Circle, Lock, Clock } from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { DIMENSIONS_CONFIG } from '@/types/assessment';
import { cn } from '@/lib/utils';

interface DimensionNavigationProps {
  currentDimension: number;
}

export function DimensionNavigation({ currentDimension }: DimensionNavigationProps) {
  const { 
    goToDimension, 
    getDimensionProgress 
  } = useAssessmentStore();

  const getDimensionStatus = (dimensionId: number) => {
    const progress = getDimensionProgress(dimensionId);
    const dimensionConfig = DIMENSIONS_CONFIG.find(d => d.id === dimensionId);
    
    if (!dimensionConfig) {
      return {
        status: 'not-started' as const,
        completion: 0,
        canAccess: dimensionId === 1 // Solo la primera dimensión es accesible inicialmente
      };
    }
    
    const completion = progress.percentage;
    
    let status: 'not-started' | 'in-progress' | 'completed';
    if (progress.completed === 0) {
      status = 'not-started';
    } else if (progress.completed < progress.total) {
      status = 'in-progress';
    } else {
      status = 'completed';
    }
    
    // Permitir acceso libre a todas las dimensiones una vez iniciada la evaluación
    const canAccess = true;
    
    return {
      status,
      completion,
      canAccess
    };
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'in-progress':
        return <Circle className="h-4 w-4 text-blue-600 fill-current" />;
      case 'available':
        return <Circle className="h-4 w-4 text-muted-foreground" />;
      case 'locked':
        return <Lock className="h-4 w-4 text-gray-300" />;
      default:
        return <Circle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusColor = (status: string, isActive: boolean) => {
    if (isActive) {
      return 'bg-blue-600 text-white border-blue-600';
    }
    
    switch (status) {
      case 'completed':
        return 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100';
      case 'in-progress':
        return 'bg-primary/10 text-primary border-primary/20 hover:bg-primary/20';
      case 'available':
        return 'bg-muted text-foreground border hover:bg-muted/80';
      case 'locked':
        return 'bg-muted text-muted-foreground border cursor-not-allowed';
      default:
        return 'bg-muted text-foreground border hover:bg-muted/80';
    }
  };

  return (
    <div className="">
      <div className="flex items-center gap-2 mb-3">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Dimensiones de Talento
        </h3>
        <Badge variant="secondary" className="text-xs">
          {DIMENSIONS_CONFIG.filter(d => getDimensionProgress(d.id).completed > 0).length} de 8 iniciadas
        </Badge>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-2">
        {DIMENSIONS_CONFIG.map((dimension) => {
          const { status, completion, canAccess } = getDimensionStatus(dimension.id);
          const isActive = dimension.id === currentDimension;
          const progress = getDimensionProgress(dimension.id);
          
          return (
            <Button
              key={dimension.id}
              variant="outline"
              size="sm"
              onClick={() => canAccess ? goToDimension(dimension.id) : null}
              disabled={!canAccess}
              className={cn(
                'flex flex-col items-center gap-1 h-auto py-2 px-2 text-xs transition-all',
                {
                  'bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800': status === 'completed',
                  'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800': status === 'in-progress' && isActive,
                  'bg-muted border': status === 'not-started' && canAccess && !isActive,
                  'bg-gray-100 border-gray-300 dark:bg-gray-900 dark:border-gray-600 opacity-50 cursor-not-allowed': !canAccess,
                  'ring-2 ring-primary ring-offset-2': isActive
                }
              )}
            >
              <div className="flex items-center gap-1 mb-1">
                <span className="text-base">{dimension.icon}</span>
                {status === 'completed' && (
                  <CheckCircle className="h-5 w-5 text-green-600" />
                )}
                {status === 'in-progress' && (
                  <Clock className="h-5 w-5 text-blue-600" />
                )}
                {status === 'not-started' && canAccess && (
                  <Circle className="h-5 w-5 text-muted-foreground" />
                )}
                {!canAccess && (
                  <Lock className="h-5 w-5 text-muted-foreground" />
                )}
              </div>
              
              <div className="text-center leading-tight">
                <div className="font-medium truncate max-w-full">
                  {dimension.name}
                </div>
                
                {progress && progress.completed > 0 && (
                  <div className="text-xs opacity-75 mt-1">
                    {progress.completed}/{dimension.competencyCount}
                  </div>
                )}
                
                {completion > 0 && (
                  <div className="w-full bg-gray-200 rounded-full h-1 mt-1">
                    <div 
                      className={cn(
                        'h-1 rounded-full transition-all',
                        status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                      )}
                      style={{ width: `${completion}%` }}
                    />
                  </div>
                )}
              </div>
            </Button>
          );
        })}
      </div>
      
      {/* Información adicional */}
      <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <CheckCircle className="h-3 w-3 text-green-600" />
            <span>Completada</span>
          </div>
          <div className="flex items-center gap-1">
            <Circle className="h-3 w-3 text-blue-600 fill-current" />
            <span>En progreso</span>
          </div>
          <div className="flex items-center gap-1">
            <Circle className="h-3 w-3 text-muted-foreground" />
            <span>Disponible</span>
          </div>
          <div className="flex items-center gap-1">
            <Lock className="h-3 w-3 text-gray-300" />
            <span>Bloqueada</span>
          </div>
        </div>
        
        <div className="text-right">
          <span>Haz clic para navegar entre dimensiones</span>
        </div>
      </div>
    </div>
  );
}