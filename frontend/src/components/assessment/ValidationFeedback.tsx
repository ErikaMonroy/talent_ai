'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  Target,
  TrendingUp,
  X,
  Info,
  AlertCircle
} from 'lucide-react';
import { useFormValidation, useValidationAlerts, useValidationStats } from '../../hooks/useFormValidation';
import { getErrorMessage } from '../../utils/validation';
import { cn } from '@/lib/utils';

interface ValidationFeedbackProps {
  className?: string;
  showProgress?: boolean;
  showEstimatedTime?: boolean;
  compact?: boolean;
  showDismissible?: boolean;
}

export function ValidationFeedback({ 
  className,
  showProgress = true,
  showEstimatedTime = true,
  compact = false,
  showDismissible = true
}: ValidationFeedbackProps) {
  const { validation, hasErrors, hasWarnings, canProceedToResults } = useFormValidation();
  const { 
    errors, 
    warnings, 
    hasActiveWarnings, 
    dismissWarning, 
    dismissAllWarnings 
  } = useValidationAlerts();
  const stats = useValidationStats();

  // Calcular tiempo estimado restante (asumiendo 30 segundos por pregunta)
  const estimatedMinutesRemaining = Math.ceil((stats.missingQuestions * 0.5));
  
  // Determinar estado de validación
  const getValidationStatus = () => {
    if (hasErrors) return 'error';
    if (stats.completionPercentage === 100) return 'complete';
    if (canProceedToResults) return 'good';
    if (stats.completionPercentage >= 50) return 'warning';
    return 'incomplete';
  };

  const status = getValidationStatus();
  
  const statusConfig = {
    error: {
      icon: XCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      message: 'Hay errores que deben corregirse antes de continuar.'
    },
    complete: {
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      message: '¡Evaluación completa! Puede proceder a ver los resultados.'
    },
    good: {
      icon: Target,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      message: 'Buen progreso. Puede proceder a los resultados o continuar completando.'
    },
    warning: {
      icon: AlertTriangle,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      message: 'Complete más preguntas para obtener resultados más precisos.'
    },
    incomplete: {
      icon: XCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      message: 'Necesita completar más preguntas antes de proceder.'
    }
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  if (compact) {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        <Icon className={cn('h-4 w-4', config.color)} />
        <span className="text-sm font-medium">
          {stats.completionPercentage}% completado
        </span>
        {hasErrors && (
          <Badge variant="destructive" className="text-xs">
            {errors.length} error{errors.length !== 1 ? 'es' : ''}
          </Badge>
        )}
        {hasActiveWarnings && (
          <Badge variant="secondary" className="text-xs">
            {warnings.length} aviso{warnings.length !== 1 ? 's' : ''}
          </Badge>
        )}
        {showEstimatedTime && stats.missingQuestions > 0 && (
          <Badge variant="outline" className="text-xs">
            <Clock className="h-3 w-3 mr-1" />
            ~{estimatedMinutesRemaining} min
          </Badge>
        )}
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-4", className)}
    >
      {/* Estado principal */}
      <Alert className={cn(config.bgColor, config.borderColor)}>
        <Icon className={cn('h-4 w-4', config.color)} />
        <AlertDescription className={config.color}>
          {config.message}
        </AlertDescription>
      </Alert>

      {/* Progreso general */}
      {showProgress && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Progreso general</span>
            <span className="font-medium">{stats.completionPercentage}%</span>
          </div>
          <Progress value={stats.completionPercentage} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{stats.completedQuestions} de {stats.totalQuestions} preguntas</span>
            {showEstimatedTime && stats.missingQuestions > 0 && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                ~{estimatedMinutesRemaining} min restantes
              </span>
            )}
          </div>
          
          {/* Mostrar estadísticas adicionales */}
          <div className="flex gap-2 mt-2">
            <Badge variant="outline" className="text-xs">
              <TrendingUp className="h-3 w-3 mr-1" />
              Calidad: {stats.qualityScore}%
            </Badge>
            {canProceedToResults && (
              <Badge variant="default" className="text-xs">
                <CheckCircle className="h-3 w-3 mr-1" />
                Listo para resultados
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Mostrar errores */}
      <AnimatePresence>
        {errors.map((error, index) => (
          <motion.div
            key={`error-${index}`}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2"
          >
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>
                {getErrorMessage(error)}
              </AlertDescription>
            </Alert>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Mostrar advertencias */}
      <AnimatePresence>
        {warnings.map((warning, index) => (
          <motion.div
            key={`warning-${index}`}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-2"
          >
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription className="flex items-center justify-between">
                <span>{warning}</span>
                {showDismissible && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => dismissWarning(warning)}
                    className="h-6 w-6 p-0 ml-2"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </AlertDescription>
            </Alert>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Botón para descartar todas las advertencias */}
      {hasActiveWarnings && showDismissible && (
        <div className="flex justify-end mt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={dismissAllWarnings}
            className="text-xs"
          >
            Descartar todas las advertencias
          </Button>
        </div>
      )}
    </motion.div>
  );
}

// Componente compacto para validación inline
export function InlineValidation({ 
  isValid, 
  message, 
  type = 'error' 
}: { 
  isValid: boolean; 
  message?: string; 
  type?: 'error' | 'warning' | 'success'; 
}) {
  if (isValid && type !== 'success') return null;

  const getIcon = () => {
    switch (type) {
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTextColor = () => {
    switch (type) {
      case 'error':
        return 'text-red-600 dark:text-red-400';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'success':
        return 'text-green-600 dark:text-green-400';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-2 mt-1"
    >
      {getIcon()}
      <span className={cn("text-sm", getTextColor())}>
        {message}
      </span>
    </motion.div>
  );
}