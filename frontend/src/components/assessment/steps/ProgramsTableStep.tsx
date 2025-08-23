'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { 
  Search,
  Filter,
  BookOpen,
  MapPin,
  Building2,
  GraduationCap,
  Users,
  Clock,
  Star,
  ExternalLink,
  RefreshCw,
  AlertCircle,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { useAssessmentStore } from '@/store/assessmentStore';
import { apiService } from '@/services/api';
import { Program, ProgramSearchParams, ProgramFilters } from '@/types/api';

interface ProgramsTableStepProps {
  className?: string;
}

export function ProgramsTableStep({ className }: ProgramsTableStepProps) {
  const { selectedAreaIds, getTopPredictedAreas } = useAssessmentStore();
  
  // Estados locales
  const [programs, setPrograms] = useState<Program[]>([]);
  const [availableFilters, setAvailableFilters] = useState<{
    cities: string[];
    departments: string[];
    academic_levels: string[];
    knowledge_areas: { id: number; name: string; code: string; }[];
  }>({ cities: [], departments: [], academic_levels: [], knowledge_areas: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingFilters, setIsLoadingFilters] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  
  // Filtros de búsqueda
  const [searchParams, setSearchParams] = useState<ProgramSearchParams>({
    area_id: selectedAreaIds[0], // Solo un área por vez según la API
    page: 1,
    size: 20
  });
  
  const [localFilters, setLocalFilters] = useState({
    name: '',
    city: 'all',
    department: 'all',
    academic_level: 'all',
    selectedAreas: selectedAreaIds
  });

  const topAreas = getTopPredictedAreas(5);

  // Cargar filtros disponibles
  useEffect(() => {
    const loadAvailableFilters = async () => {
      setIsLoadingFilters(true);
      try {
        const filtersResponse = await apiService.getAvailableFilters();
        setAvailableFilters({
          cities: filtersResponse.data.cities || [],
          departments: filtersResponse.data.departments || [],
          academic_levels: filtersResponse.data.academic_levels || [],
          knowledge_areas: filtersResponse.data.knowledge_areas || []
        });
      } catch (err) {
        console.error('Error loading filters:', err);
        // No mostrar error para filtros, usar valores por defecto
      } finally {
        setIsLoadingFilters(false);
      }
    };

    loadAvailableFilters();
  }, []);

  // Cargar programas cuando cambien los parámetros
  useEffect(() => {
    const loadPrograms = async () => {
      if (!searchParams.area_id) {
        setPrograms([]);
        return;
      }

      setIsLoading(true);
      setError(null);
      
      try {
        const response = await apiService.searchPrograms(searchParams);
        setPrograms(response.data?.programs || []);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar programas');
        setPrograms([]);
      } finally {
        setIsLoading(false);
      }
    };

    loadPrograms();
  }, [searchParams]);

  // Actualizar parámetros de búsqueda cuando cambien los filtros locales
  useEffect(() => {
    const params: ProgramSearchParams = {
      area_id: localFilters.selectedAreas[0], // Solo un área por vez
      page: 1,
      size: 20
    };

    if (localFilters.name.trim()) {
      params.name = localFilters.name.trim();
    }
    if (localFilters.city && localFilters.city !== 'all') {
      params.city = localFilters.city;
    }
    if (localFilters.department && localFilters.department !== 'all') {
      params.department = localFilters.department;
    }
    if (localFilters.academic_level && localFilters.academic_level !== 'all') {
      params.academic_level = localFilters.academic_level;
    }

    setSearchParams(params);
  }, [localFilters]);

  // Actualizar áreas seleccionadas cuando cambien en el store
  useEffect(() => {
    setLocalFilters(prev => ({
      ...prev,
      selectedAreas: selectedAreaIds
    }));
  }, [selectedAreaIds]);

  const handleAreaToggle = (areaId: number) => {
    setLocalFilters(prev => {
      const newSelectedAreas = prev.selectedAreas.includes(areaId)
        ? prev.selectedAreas.filter(id => id !== areaId)
        : [...prev.selectedAreas, areaId];
      
      return {
        ...prev,
        selectedAreas: newSelectedAreas
      };
    });
  };

  const handleResetFilters = () => {
    setLocalFilters({
      name: '',
      city: 'all',
      department: 'all',
      academic_level: 'all',
      selectedAreas: selectedAreaIds
    });
  };

  const getAcademicLevelBadge = (level: string) => {
    const levelMap: Record<string, { color: string; label: string }> = {
      'pregrado': { color: 'bg-blue-100 text-blue-800', label: 'Pregrado' },
      'posgrado': { color: 'bg-purple-100 text-purple-800', label: 'Posgrado' },
      'maestria': { color: 'bg-green-100 text-green-800', label: 'Maestría' },
      'doctorado': { color: 'bg-red-100 text-red-800', label: 'Doctorado' },
      'especializacion': { color: 'bg-yellow-100 text-yellow-800', label: 'Especialización' }
    };
    
    const config = levelMap[level.toLowerCase()] || { color: 'bg-gray-100 text-gray-800', label: level };
    
    return (
      <Badge className={`${config.color} text-xs`}>
        {config.label}
      </Badge>
    );
  };

  const selectedAreasFromTop = useMemo(() => {
    return topAreas.filter(area => localFilters.selectedAreas.includes(area.area_id));
  }, [topAreas, localFilters.selectedAreas]);

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
      {/* Encabezado */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-6 w-6 text-blue-600" />
            Programas Académicos Recomendados
          </CardTitle>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Explora programas relacionados con las áreas seleccionadas
          </p>
        </CardHeader>
      </Card>

      {/* Áreas seleccionadas */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">Áreas Seleccionadas</h3>
            <Badge variant="outline">
              {localFilters.selectedAreas.length} área{localFilters.selectedAreas.length !== 1 ? 's' : ''}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {topAreas.map((area) => (
              <div 
                key={area.area_id}
                className="flex items-center gap-3 p-3 rounded-lg border"
              >
                <Checkbox 
                  checked={localFilters.selectedAreas.includes(area.area_id)}
                  onCheckedChange={() => handleAreaToggle(area.area_id)}
                />
                <div className="flex-1">
                  <span className="font-medium">{area.area_name}</span>
                  <span className="ml-2 text-sm text-gray-500">
                    ({Math.round(area.percentage)}% afinidad)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Filtros */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Filtros de Búsqueda
            </h3>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
            >
              {showFilters ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              {showFilters ? 'Ocultar' : 'Mostrar'} Filtros
            </Button>
          </div>
        </CardHeader>
        
        {showFilters && (
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Búsqueda por texto */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Buscar programa</label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <Input
                    placeholder="Nombre del programa..."
                    value={localFilters.name}
                    onChange={(e) => setLocalFilters(prev => ({ ...prev, name: e.target.value }))}
                    className="pl-10"
                  />
                </div>
              </div>

              {/* Ciudad */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Ciudad</label>
                <Select 
                  value={localFilters.city} 
                  onValueChange={(value) => setLocalFilters(prev => ({ ...prev, city: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Seleccionar ciudad" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todas las ciudades</SelectItem>
                    {availableFilters.cities.map((city) => (
                      <SelectItem key={city} value={city}>{city}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Departamento */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Departamento</label>
                <Select 
                  value={localFilters.department} 
                  onValueChange={(value) => setLocalFilters(prev => ({ ...prev, department: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Seleccionar departamento" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todos los departamentos</SelectItem>
                    {availableFilters.departments.map((dept) => (
                      <SelectItem key={dept} value={dept}>{dept}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Nivel académico */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Nivel Académico</label>
                <Select 
                  value={localFilters.academic_level} 
                  onValueChange={(value) => setLocalFilters(prev => ({ ...prev, academic_level: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Seleccionar nivel" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todos los niveles</SelectItem>
                    {availableFilters.academic_levels.map((level) => (
                      <SelectItem key={level} value={level}>{level}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Espacio para mantener el grid */}
              <div></div>

              {/* Botón de reset */}
              <div className="flex items-end">
                <Button 
                  variant="outline" 
                  onClick={handleResetFilters}
                  className="w-full"
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Limpiar Filtros
                </Button>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Resultados */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium">
              Programas Encontrados
            </h3>
            {programs.length > 0 && (
              <Badge variant="outline">
                {programs.length} programa{programs.length !== 1 ? 's' : ''}
              </Badge>
            )}
          </div>
        </CardHeader>
        
        <CardContent>
          {isLoading ? (
            <div className="text-center py-12">
              <RefreshCw className="h-8 w-8 text-gray-400 mx-auto mb-4 animate-spin" />
              <p className="text-gray-500 dark:text-gray-400">Cargando programas...</p>
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <AlertCircle className="h-8 w-8 text-red-400 mx-auto mb-4" />
              <p className="text-red-600 dark:text-red-400 mb-2">Error al cargar programas</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">{error}</p>
            </div>
          ) : programs.length === 0 ? (
            <div className="text-center py-12">
              <BookOpen className="h-8 w-8 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 dark:text-gray-400">No se encontraron programas con los filtros seleccionados</p>
            </div>
          ) : (
            <div className="space-y-4">
              {programs.map((program) => (
                <div 
                  key={program.id}
                  className="p-4 border rounded-lg hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg text-gray-900 dark:text-white mb-1">
                        {program.name}
                      </h4>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
                        <div className="flex items-center gap-1">
                          <Building2 className="h-4 w-4" />
                          {program.institution}
                        </div>
                        
                        <div className="flex items-center gap-1">
                          <MapPin className="h-4 w-4" />
                          {program.city}
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2 mb-3">
                        {getAcademicLevelBadge(program.academic_level)}
                        
                        <Badge variant="outline" className="text-xs">
                          Área ID: {program.area_id}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                        Programa de {program.academic_level} en {program.institution}
                      </p>
                    </div>
                    
                    <Button variant="outline" size="sm">
                      <ExternalLink className="h-4 w-4 mr-2" />
                      Ver Más
                    </Button>
                  </div>
                  
                  {/* Información adicional */}
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400 pt-3 border-t">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {program.academic_level}
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <Users className="h-3 w-3" />
                      {program.department}
                    </div>
                    
                    {selectedAreasFromTop.find(area => area.area_id === program.area_id) && (
                      <div className="flex items-center gap-1 text-blue-600">
                        <Star className="h-3 w-3" />
                        Área recomendada ({Math.round(selectedAreasFromTop.find(area => area.area_id === program.area_id)?.percentage || 0)}% afinidad)
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}