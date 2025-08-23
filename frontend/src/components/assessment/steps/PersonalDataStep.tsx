'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { 
  User, 
  Calendar, 
  MapPin, 
  GraduationCap, 
  Briefcase,
  ArrowRight,
  ArrowLeft
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';

export function PersonalDataStep() {
  const { 
    personalData, 
    updatePersonalData, 
    goToStep, 
    goToPreviousStep 
  } = useAssessmentStore();

  const handleInputChange = (field: string, value: string | number) => {
    updatePersonalData({ [field]: value });
  };



  const isFormValid = () => {
    return (
      personalData.name?.trim() &&
      personalData.age &&
      personalData.gender &&
      personalData.location?.trim()
    );
  };

  const handleNext = () => {
    if (isFormValid()) {
      goToStep('icfes'); // Ir al paso ICFES
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-6 w-6 text-primary" />
            Información Personal
          </CardTitle>
          <p className="text-muted-foreground">
            Esta información nos ayuda a personalizar tu experiencia y generar 
            resultados más precisos. Todos los datos se mantienen privados y seguros.
          </p>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Nombre completo */}
          <div className="space-y-2">
            <Label htmlFor="name" className="flex items-center gap-2">
              <User className="h-4 w-4" />
              Nombre completo *
            </Label>
            <Input
              id="name"
              type="text"
              placeholder="Ingresa tu nombre completo"
              value={personalData.name || ''}
              onChange={(e) => handleInputChange('name', e.target.value)}
              className="w-full"
            />
          </div>

          {/* Edad */}
          <div className="space-y-2">
            <Label htmlFor="age" className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              Edad *
            </Label>
            <Input
              id="age"
              type="number"
              min="16"
              max="100"
              placeholder="Ingresa tu edad"
              value={personalData.age || ''}
              onChange={(e) => handleInputChange('age', e.target.value)}
              className="w-full"
            />
          </div>

          {/* Género */}
          <div className="space-y-3">
            <Label className="text-base font-medium">
              Género *
            </Label>
            <RadioGroup
              value={personalData.gender || ''}
              onValueChange={(value) => handleInputChange('gender', value)}
              className="grid grid-cols-1 sm:grid-cols-3 gap-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="male" id="male" />
                <Label htmlFor="male" className="cursor-pointer">
                  Masculino
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="female" id="female" />
                <Label htmlFor="female" className="cursor-pointer">
                  Femenino
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="other" id="other" />
                <Label htmlFor="other" className="cursor-pointer">
                  Otro
                </Label>
              </div>
            </RadioGroup>
          </div>


          {/* Ubicación */}
          <div className="space-y-2">
            <Label htmlFor="location" className="flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Ciudad/País *
            </Label>
            <Input
              id="location"
              type="text"
              placeholder="Ej: Bogotá, Colombia"
              value={personalData.location || ''}
              onChange={(e) => handleInputChange('location', e.target.value)}
              className="w-full"
            />
          </div>



          {/* Botones de navegación */}
          <div className="flex justify-between pt-6">
            <Button
              variant="outline"
              onClick={goToPreviousStep}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Anterior
            </Button>
            
            <Button
              onClick={handleNext}
              disabled={!isFormValid()}
              className="flex items-center gap-2"
            >
              Continuar
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>

          {/* Mensaje de validación */}
          {!isFormValid() && (
            <p className="text-sm text-orange-600 dark:text-orange-400 text-center">
              Por favor completa todos los campos marcados con * para continuar.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}