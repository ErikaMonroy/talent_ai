import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { BookOpen, ArrowLeft, ArrowRight } from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import type { PersonalData } from '@/types/assessment';

interface IcfesStepProps {
  onNext: () => void;
  onPrevious: () => void;
}

export function IcfesStep({ onNext, onPrevious }: IcfesStepProps) {
  const { personalData, updatePersonalData } = useAssessmentStore();

  const handleIcfesChange = (subject: keyof PersonalData['icfesScores'], value: string) => {
    const numValue = value === '' ? undefined : parseInt(value, 10);
    updatePersonalData({
      icfesScores: {
        ...personalData.icfesScores,
        [subject]: numValue
      }
    });
  };

  const isFormValid = () => {
    const scores = personalData.icfesScores;
    return scores && 
           scores.matematicas !== undefined && scores.matematicas > 0 &&
           scores.lectura_critica !== undefined && scores.lectura_critica > 0 &&
           scores.ciencias_naturales !== undefined && scores.ciencias_naturales > 0 &&
           scores.sociales_ciudadanas !== undefined && scores.sociales_ciudadanas > 0 &&
           scores.ingles !== undefined && scores.ingles > 0;
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Puntajes ICFES
          </CardTitle>
          <CardDescription>
            Ingresa tus puntajes ICFES para obtener recomendaciones más precisas sobre tu perfil académico.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="matematicas" className="text-sm font-medium">
                Matemáticas *
              </Label>
              <Input
                id="matematicas"
                type="number"
                min="0"
                max="100"
                placeholder="0-100"
                value={personalData.icfesScores?.matematicas || ''}
                onChange={(e) => handleIcfesChange('matematicas', e.target.value)}
                className="w-full"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="lectura_critica" className="text-sm font-medium">
                Lectura Crítica *
              </Label>
              <Input
                id="lectura_critica"
                type="number"
                min="0"
                max="100"
                placeholder="0-100"
                value={personalData.icfesScores?.lectura_critica || ''}
                onChange={(e) => handleIcfesChange('lectura_critica', e.target.value)}
                className="w-full"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="ciencias_naturales" className="text-sm font-medium">
                Ciencias Naturales *
              </Label>
              <Input
                id="ciencias_naturales"
                type="number"
                min="0"
                max="100"
                placeholder="0-100"
                value={personalData.icfesScores?.ciencias_naturales || ''}
                onChange={(e) => handleIcfesChange('ciencias_naturales', e.target.value)}
                className="w-full"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="sociales_ciudadanas" className="text-sm font-medium">
                Sociales y Ciudadanas *
              </Label>
              <Input
                id="sociales_ciudadanas"
                type="number"
                min="0"
                max="100"
                placeholder="0-100"
                value={personalData.icfesScores?.sociales_ciudadanas || ''}
                onChange={(e) => handleIcfesChange('sociales_ciudadanas', e.target.value)}
                className="w-full"
              />
            </div>
            
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="ingles" className="text-sm font-medium">
                Inglés *
              </Label>
              <Input
                id="ingles"
                type="number"
                min="0"
                max="100"
                placeholder="0-100"
                value={personalData.icfesScores?.ingles || ''}
                onChange={(e) => handleIcfesChange('ingles', e.target.value)}
                className="md:w-1/2"
              />
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Nota:</strong> Los puntajes ICFES son obligatorios para continuar. 
              Ingresa valores entre 0 y 100 para cada área.
            </p>
          </div>

          {/* Botones de navegación */}
          <div className="flex justify-between pt-6">
            <Button
              type="button"
              variant="outline"
              onClick={onPrevious}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Anterior
            </Button>
            
            <Button
              type="button"
              onClick={onNext}
              disabled={!isFormValid()}
              className="flex items-center gap-2"
            >
              Continuar
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}