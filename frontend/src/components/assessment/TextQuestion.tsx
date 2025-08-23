'use client';

import React, { useState, useEffect } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { 
  CheckCircle, 
  AlertCircle,
  Type,
  FileText,
  Clock
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface TextQuestionProps {
  question: {
    id: string;
    text: string;
    description?: string;
    type: 'short' | 'long';
    placeholder?: string;
    required?: boolean;
    minLength?: number;
    maxLength?: number;
    pattern?: string;
    patternMessage?: string;
  };
  value?: string;
  onChange: (value: string) => void;
  showValidation?: boolean;
  autoSave?: boolean;
  autoSaveDelay?: number;
  className?: string;
}

export function TextQuestion({
  question,
  value = '',
  onChange,
  showValidation = false,
  autoSave = true,
  autoSaveDelay = 1000,
  className
}: TextQuestionProps) {
  const [localValue, setLocalValue] = useState(value);
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [saveTimeout, setSaveTimeout] = useState<NodeJS.Timeout | null>(null);

  const isLongText = question.type === 'long';
  const currentLength = localValue.length;
  const maxLength = question.maxLength || (isLongText ? 1000 : 200);
  const minLength = question.minLength || 0;

  // Sincronizar valor local con prop value
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  // Auto-guardado
  useEffect(() => {
    if (!autoSave) return;

    if (saveTimeout) {
      clearTimeout(saveTimeout);
    }

    if (localValue !== value) {
      setIsSaving(true);
      const timeout = setTimeout(() => {
        onChange(localValue);
        setIsSaving(false);
        setLastSaved(new Date());
      }, autoSaveDelay);
      setSaveTimeout(timeout);
    }

    return () => {
      if (saveTimeout) {
        clearTimeout(saveTimeout);
      }
    };
  }, [localValue, value, onChange, autoSave, autoSaveDelay, saveTimeout]);

  // Validación
  const isValid = () => {
    if (!question.required && !localValue.trim()) return true;
    
    if (question.required && !localValue.trim()) return false;
    
    if (minLength > 0 && localValue.length < minLength) return false;
    
    if (maxLength > 0 && localValue.length > maxLength) return false;
    
    if (question.pattern) {
      const regex = new RegExp(question.pattern);
      if (!regex.test(localValue)) return false;
    }
    
    return true;
  };

  const getValidationMessage = () => {
    if (isValid()) return null;
    
    if (question.required && !localValue.trim()) {
      return 'Este campo es requerido';
    }
    
    if (minLength > 0 && localValue.length < minLength) {
      return `Mínimo ${minLength} caracteres requeridos`;
    }
    
    if (maxLength > 0 && localValue.length > maxLength) {
      return `Máximo ${maxLength} caracteres permitidos`;
    }
    
    if (question.pattern && question.patternMessage) {
      return question.patternMessage;
    }
    
    return 'Formato inválido';
  };

  const handleChange = (newValue: string) => {
    // Limitar longitud si es necesario
    if (maxLength > 0 && newValue.length > maxLength) {
      return;
    }
    
    setLocalValue(newValue);
  };

  const validationMessage = getValidationMessage();
  const hasError = showValidation && !isValid();
  const isNearLimit = currentLength > maxLength * 0.8;
  const isOverLimit = currentLength > maxLength;

  const getCharacterCountColor = () => {
    if (isOverLimit) return 'text-red-600 dark:text-red-400';
    if (isNearLimit) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-muted-foreground';
  };

  const formatLastSaved = () => {
    if (!lastSaved) return null;
    const now = new Date();
    const diff = Math.floor((now.getTime() - lastSaved.getTime()) / 1000);
    
    if (diff < 60) return 'Guardado hace unos segundos';
    if (diff < 3600) return `Guardado hace ${Math.floor(diff / 60)} minutos`;
    return `Guardado hace ${Math.floor(diff / 3600)} horas`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-4", className)}
    >
      {/* Encabezado de la pregunta */}
      <div className="space-y-2">
        <div className="flex items-start gap-2">
          <div className="flex items-center gap-2">
            {isLongText ? (
              <FileText className="h-5 w-5 text-muted-foreground" />
            ) : (
              <Type className="h-5 w-5 text-muted-foreground" />
            )}
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 leading-tight">
              {question.text}
            </h3>
          </div>
          {question.required && (
            <Badge variant="secondary" className="text-xs shrink-0">
              Requerido
            </Badge>
          )}
        </div>
        
        {question.description && (
          <p className="text-sm text-muted-foreground">
            {question.description}
          </p>
        )}
      </div>

      {/* Campo de entrada */}
      <Card className={cn(
        "transition-all duration-200",
        hasError && "border-red-300 dark:border-red-700",
        isValid() && localValue && "border-green-300 dark:border-green-700"
      )}>
        <CardContent className="p-4">
          {isLongText ? (
            <Textarea
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              placeholder={question.placeholder || 'Escribe tu respuesta aquí...'}
              className={cn(
                "min-h-[120px] resize-none border-0 p-0 focus-visible:ring-0 focus-visible:ring-offset-0",
                hasError && "text-red-900 dark:text-red-100"
              )}
              disabled={isSaving}
            />
          ) : (
            <Input
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              placeholder={question.placeholder || 'Escribe tu respuesta...'}
              className={cn(
                "border-0 p-0 focus-visible:ring-0 focus-visible:ring-offset-0",
                hasError && "text-red-900 dark:text-red-100"
              )}
              disabled={isSaving}
            />
          )}
        </CardContent>
      </Card>

      {/* Información adicional */}
      <div className="flex items-center justify-between text-xs">
        {/* Contador de caracteres */}
        <div className="flex items-center gap-4">
          <span className={cn("font-medium", getCharacterCountColor())}>
            {currentLength}{maxLength > 0 && `/${maxLength}`} caracteres
          </span>
          
          {minLength > 0 && currentLength < minLength && (
            <span className="text-yellow-600 dark:text-yellow-400">
              Faltan {minLength - currentLength} caracteres
            </span>
          )}
        </div>

        {/* Estado de guardado */}
        <div className="flex items-center gap-2">
          {isSaving && (
            <div className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
              <Clock className="h-3 w-3 animate-spin" />
              <span>Guardando...</span>
            </div>
          )}
          
          {!isSaving && lastSaved && autoSave && (
            <div className="flex items-center gap-1 text-green-600 dark:text-green-400">
              <CheckCircle className="h-3 w-3" />
              <span>{formatLastSaved()}</span>
            </div>
          )}
        </div>
      </div>

      {/* Mensaje de validación */}
      <AnimatePresence>
        {showValidation && validationMessage && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400"
          >
            <AlertCircle className="h-4 w-4" />
            <span>{validationMessage}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Indicador de respuesta válida */}
      {isValid() && localValue.trim() && !isSaving && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400"
        >
          <CheckCircle className="h-4 w-4" />
          <span>Respuesta guardada</span>
        </motion.div>
      )}

      {/* Consejos de escritura para texto largo */}
      {isLongText && !localValue && (
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 border border-blue-200 dark:border-blue-800">
          <div className="flex items-start gap-2">
            <FileText className="h-4 w-4 text-blue-600 dark:text-blue-400 mt-0.5" />
            <div className="text-sm text-blue-800 dark:text-blue-200">
              <p className="font-medium mb-1">Consejos para tu respuesta:</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>Sé específico y proporciona ejemplos concretos</li>
                <li>Reflexiona sobre tus experiencias personales</li>
                <li>No hay respuestas correctas o incorrectas</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}