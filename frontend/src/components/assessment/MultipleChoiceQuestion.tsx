'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  CheckCircle, 
  Circle, 
  Square, 
  CheckSquare,
  AlertCircle,
  Info
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

interface Option {
  id: string;
  text: string;
  description?: string;
  value: number | string;
  disabled?: boolean;
}

interface MultipleChoiceQuestionProps {
  question: {
    id: string;
    text: string;
    description?: string;
    type: 'single' | 'multiple';
    options: Option[];
    required?: boolean;
    maxSelections?: number;
    minSelections?: number;
  };
  value?: string | string[];
  onChange: (value: string | string[]) => void;
  showValidation?: boolean;
  className?: string;
}

export function MultipleChoiceQuestion({
  question,
  value,
  onChange,
  showValidation = false,
  className
}: MultipleChoiceQuestionProps) {
  const isSingleChoice = question.type === 'single';
  const currentValue = value || (isSingleChoice ? '' : []);
  const isArray = Array.isArray(currentValue);
  const selectedValues = isArray ? currentValue : [currentValue].filter(Boolean);

  // Validación
  const isValid = () => {
    if (!question.required) return true;
    
    if (isSingleChoice) {
      return Boolean(currentValue && currentValue !== '');
    } else {
      const selections = selectedValues.length;
      const minValid = !question.minSelections || selections >= question.minSelections;
      const maxValid = !question.maxSelections || selections <= question.maxSelections;
      return selections > 0 && minValid && maxValid;
    }
  };

  const getValidationMessage = () => {
    if (isValid()) return null;
    
    if (isSingleChoice) {
      return 'Debes seleccionar una opción';
    } else {
      const selections = selectedValues.length;
      if (selections === 0) {
        return 'Debes seleccionar al menos una opción';
      }
      if (question.minSelections && selections < question.minSelections) {
        return `Debes seleccionar al menos ${question.minSelections} opciones`;
      }
      if (question.maxSelections && selections > question.maxSelections) {
        return `Puedes seleccionar máximo ${question.maxSelections} opciones`;
      }
    }
    return null;
  };

  // Manejadores de eventos
  const handleSingleChange = (optionValue: string) => {
    onChange(optionValue);
  };

  const handleMultipleChange = (optionValue: string, checked: boolean) => {
    const currentSelections = selectedValues;
    let newSelections: string[];

    if (checked) {
      // Verificar límite máximo
      if (question.maxSelections && currentSelections.length >= question.maxSelections) {
        return; // No permitir más selecciones
      }
      newSelections = [...currentSelections, optionValue];
    } else {
      newSelections = currentSelections.filter(v => v !== optionValue);
    }

    onChange(newSelections);
  };

  const validationMessage = getValidationMessage();
  const hasError = showValidation && !isValid();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("space-y-4", className)}
    >
      {/* Encabezado de la pregunta */}
      <div className="space-y-2">
        <div className="flex items-start gap-2">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 leading-tight">
            {question.text}
          </h3>
          {question.required && (
            <Badge variant="secondary" className="text-xs shrink-0">
              Requerido
            </Badge>
          )}
        </div>
        
        {question.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {question.description}
          </p>
        )}

        {/* Información adicional para selección múltiple */}
        {!isSingleChoice && (
          <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <Info className="h-3 w-3" />
            <span>
              {question.minSelections && question.maxSelections
                ? `Selecciona entre ${question.minSelections} y ${question.maxSelections} opciones`
                : question.minSelections
                ? `Selecciona al menos ${question.minSelections} opciones`
                : question.maxSelections
                ? `Selecciona máximo ${question.maxSelections} opciones`
                : 'Puedes seleccionar múltiples opciones'
              }
            </span>
          </div>
        )}
      </div>

      {/* Opciones */}
      <div className="space-y-3">
        {isSingleChoice ? (
          <RadioGroup
            value={currentValue as string}
            onValueChange={handleSingleChange}
            className="space-y-3"
          >
            {question.options.map((option, index) => (
              <motion.div
                key={option.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className={cn(
                  "transition-all duration-200 cursor-pointer hover:shadow-md",
                  currentValue === option.value.toString() 
                    ? "ring-2 ring-primary bg-primary/10" 
                    : "hover:bg-gray-50 dark:hover:bg-gray-800",
                  option.disabled && "opacity-50 cursor-not-allowed",
                  hasError && "border-red-300 dark:border-red-700"
                )}>
                  <CardContent className="p-4">
                    <div className="flex items-start gap-3">
                      <RadioGroupItem
                        value={option.value.toString()}
                        id={option.id}
                        disabled={option.disabled}
                        className="mt-0.5"
                      />
                      <Label
                        htmlFor={option.id}
                        className="flex-1 cursor-pointer"
                      >
                        <div className="space-y-1">
                          <p className="font-medium text-gray-900 dark:text-gray-100">
                            {option.text}
                          </p>
                          {option.description && (
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              {option.description}
                            </p>
                          )}
                        </div>
                      </Label>
                      {currentValue === option.value.toString() && (
                        <CheckCircle className="h-5 w-5 text-blue-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </RadioGroup>
        ) : (
          <div className="space-y-3">
            {question.options.map((option, index) => {
              const isSelected = selectedValues.includes(option.value.toString());
              const canSelect = !question.maxSelections || 
                selectedValues.length < question.maxSelections || 
                isSelected;

              return (
                <motion.div
                  key={option.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className={cn(
                    "transition-all duration-200 cursor-pointer hover:shadow-md",
                    isSelected 
                      ? "ring-2 ring-primary bg-primary/10" 
                      : "hover:bg-gray-50 dark:hover:bg-gray-800",
                    (!canSelect || option.disabled) && "opacity-50 cursor-not-allowed",
                    hasError && "border-red-300 dark:border-red-700"
                  )}>
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <Checkbox
                          id={option.id}
                          checked={isSelected}
                          disabled={option.disabled || !canSelect}
                          onCheckedChange={(checked) => 
                            handleMultipleChange(option.value.toString(), checked as boolean)
                          }
                          className="mt-0.5"
                        />
                        <Label
                          htmlFor={option.id}
                          className="flex-1 cursor-pointer"
                        >
                          <div className="space-y-1">
                            <p className="font-medium text-gray-900 dark:text-gray-100">
                              {option.text}
                            </p>
                            {option.description && (
                              <p className="text-sm text-gray-600 dark:text-gray-400">
                                {option.description}
                              </p>
                            )}
                          </div>
                        </Label>
                        {isSelected && (
                          <CheckCircle className="h-5 w-5 text-blue-500 shrink-0" />
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* Contador de selecciones para múltiple opción */}
      {!isSingleChoice && question.maxSelections && (
        <div className="flex justify-end">
          <Badge variant="outline" className="text-xs">
            {selectedValues.length} de {question.maxSelections} seleccionadas
          </Badge>
        </div>
      )}

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

      {/* Indicador de respuesta guardada */}
      {isValid() && value && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400"
        >
          <CheckCircle className="h-4 w-4" />
          <span>Respuesta guardada</span>
        </motion.div>
      )}
    </motion.div>
  );
}