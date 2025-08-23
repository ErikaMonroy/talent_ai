import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import {
  Search,
  Filter,
  MapPin,
  GraduationCap,
  Users,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  X
} from 'lucide-react';
import { useProgramSearch } from '@/hooks/useApi';
import { ProgramSearchParams } from '@/types/api';

interface FilterState {
  query: string;
  area_id: number | null;
  city: string;
  department: string;
  academic_level: string;
  page: number;
  size: number;
}

const ACADEMIC_LEVELS = [
  { value: 'tecnico', label: 'Técnico' },
  { value: 'tecnologo', label: 'Tecnólogo' },
  { value: 'universitario', label: 'Universitario' },
  { value: 'especializacion', label: 'Especialización' },
  { value: 'maestria', label: 'Maestría' },
  { value: 'doctorado', label: 'Doctorado' }
];

const AREAS = [
  { id: 1, name: 'Ingeniería y Tecnología' },
  { id: 2, name: 'Ciencias de la Salud' },
  { id: 3, name: 'Ciencias Sociales y Humanas' },
  { id: 4, name: 'Ciencias Naturales y Exactas' },
  { id: 5, name: 'Artes y Humanidades' },
  { id: 6, name: 'Ciencias Económicas y Administrativas' },
  { id: 7, name: 'Ciencias de la Educación' },
  { id: 8, name: 'Ciencias Agrarias' }
];

const DEPARTMENTS = [
  'Antioquia', 'Atlántico', 'Bogotá D.C.', 'Bolívar', 'Boyacá', 'Caldas',
  'Caquetá', 'Cauca', 'César', 'Córdoba', 'Cundinamarca', 'Chocó',
  'Huila', 'La Guajira', 'Magdalena', 'Meta', 'Nariño', 'Norte de Santander',
  'Quindío', 'Risaralda', 'Santander', 'Sucre', 'Tolima', 'Valle del Cauca'
];

export default function ProgramSearch() {
  const { programs, loading, error, searchPrograms } = useProgramSearch();
  
  const [filters, setFilters] = useState<FilterState>({
    query: '',
    area_id: null,
    city: '',
    department: '',
    academic_level: '',
    page: 1,
    size: 12
  });

  const [activeFilters, setActiveFilters] = useState<string[]>([]);

  // Realizar búsqueda inicial
  useEffect(() => {
    handleSearch();
  }, []);

  const handleSearch = () => {
    const searchParams: ProgramSearchParams = {
      ...(filters.query && { q: filters.query }),
      ...(filters.area_id && { area_id: filters.area_id }),
      ...(filters.city && { city: filters.city }),
      ...(filters.department && { department: filters.department }),
      ...(filters.academic_level && { academic_level: filters.academic_level }),
      page: filters.page,
      size: filters.size
    };

    searchPrograms(searchParams);
    updateActiveFilters();
  };

  const updateActiveFilters = () => {
    const active: string[] = [];
    if (filters.query) active.push(`Búsqueda: "${filters.query}"`);
    if (filters.area_id) {
      const area = AREAS.find(a => a.id === filters.area_id);
      if (area) active.push(`Área: ${area.name}`);
    }
    if (filters.city) active.push(`Ciudad: ${filters.city}`);
    if (filters.department) active.push(`Departamento: ${filters.department}`);
    if (filters.academic_level) {
      const level = ACADEMIC_LEVELS.find(l => l.value === filters.academic_level);
      if (level) active.push(`Nivel: ${level.label}`);
    }
    setActiveFilters(active);
  };

  const clearFilter = (filterText: string) => {
    if (filterText.startsWith('Búsqueda:')) {
      setFilters(prev => ({ ...prev, query: '' }));
    } else if (filterText.startsWith('Área:')) {
      setFilters(prev => ({ ...prev, area_id: null }));
    } else if (filterText.startsWith('Ciudad:')) {
      setFilters(prev => ({ ...prev, city: '' }));
    } else if (filterText.startsWith('Departamento:')) {
      setFilters(prev => ({ ...prev, department: '' }));
    } else if (filterText.startsWith('Nivel:')) {
      setFilters(prev => ({ ...prev, academic_level: '' }));
    }
  };

  const clearAllFilters = () => {
    setFilters({
      query: '',
      area_id: null,
      city: '',
      department: '',
      academic_level: '',
      page: 1,
      size: 12
    });
  };

  const handlePageChange = (newPage: number) => {
    setFilters(prev => ({ ...prev, page: newPage }));
  };

  useEffect(() => {
    if (filters.page !== 1) {
      handleSearch();
    }
  }, [filters.page]);

  const totalPages = programs ? Math.ceil(programs.total / filters.size) : 0;

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Búsqueda de Programas Académicos
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Encuentra el programa académico ideal para tu perfil profesional
        </p>
      </div>

      {/* Filtros de búsqueda */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filtros de Búsqueda
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            {/* Búsqueda por texto */}
            <div className="space-y-2">
              <Label htmlFor="search-query">Buscar programa</Label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  id="search-query"
                  placeholder="Nombre del programa..."
                  value={filters.query}
                  onChange={(e) => setFilters(prev => ({ ...prev, query: e.target.value }))}
                  className="pl-10"
                />
              </div>
            </div>

            {/* Área de conocimiento */}
            <div className="space-y-2">
              <Label>Área de conocimiento</Label>
              <Select
                value={filters.area_id?.toString() || 'all'}
                onValueChange={(value) => setFilters(prev => ({ 
                  ...prev, 
                  area_id: value === 'all' ? null : parseInt(value) 
                }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Seleccionar área" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todas las áreas</SelectItem>
                  {AREAS.map((area) => (
                    <SelectItem key={area.id} value={area.id.toString()}>
                      {area.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Nivel académico */}
            <div className="space-y-2">
              <Label>Nivel académico</Label>
              <Select
                value={filters.academic_level || 'all'}
                onValueChange={(value) => setFilters(prev => ({ ...prev, academic_level: value === 'all' ? '' : value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Seleccionar nivel" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todos los niveles</SelectItem>
                  {ACADEMIC_LEVELS.map((level) => (
                    <SelectItem key={level.value} value={level.value}>
                      {level.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Departamento */}
            <div className="space-y-2">
              <Label>Departamento</Label>
              <Select
                value={filters.department || 'all'}
                onValueChange={(value) => setFilters(prev => ({ ...prev, department: value === 'all' ? '' : value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Seleccionar departamento" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todos los departamentos</SelectItem>
                  {DEPARTMENTS.map((dept) => (
                    <SelectItem key={dept} value={dept}>
                      {dept}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Ciudad */}
            <div className="space-y-2">
              <Label htmlFor="city">Ciudad</Label>
              <Input
                id="city"
                placeholder="Nombre de la ciudad..."
                value={filters.city}
                onChange={(e) => setFilters(prev => ({ ...prev, city: e.target.value }))}
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-2 justify-between items-center">
            <Button onClick={handleSearch} className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Buscar Programas
            </Button>
            
            {activeFilters.length > 0 && (
              <Button variant="outline" onClick={clearAllFilters}>
                Limpiar filtros
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Filtros activos */}
      {activeFilters.length > 0 && (
        <div className="mb-6">
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400 self-center">
              Filtros activos:
            </span>
            {activeFilters.map((filter, index) => (
              <Badge key={index} variant="secondary" className="flex items-center gap-1">
                {filter}
                <X 
                  className="h-3 w-3 cursor-pointer hover:text-red-500" 
                  onClick={() => clearFilter(filter)}
                />
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Resultados */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Buscando programas...</p>
        </div>
      ) : error ? (
        <Card>
          <CardContent className="text-center py-12">
            <p className="text-red-600 dark:text-red-400 mb-4">Error al buscar programas</p>
            <Button onClick={handleSearch} variant="outline">
              Intentar nuevamente
            </Button>
          </CardContent>
        </Card>
      ) : programs && programs.programs.length > 0 ? (
        <>
          {/* Información de resultados */}
          <div className="mb-6">
            <p className="text-gray-600 dark:text-gray-400">
              Mostrando {programs.programs.length} de {programs.total} programas encontrados
            </p>
          </div>

          {/* Grid de programas */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {programs.programs.map((program) => (
              <Card key={program.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <CardTitle className="text-lg leading-tight">
                    {program.name}
                  </CardTitle>
                </CardHeader>
                
                <CardContent>
                  <div className="space-y-3">
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      {program.institution}
                    </p>
                    
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="flex items-center gap-1">
                        <GraduationCap className="h-3 w-3" />
                        {program.academic_level}
                      </Badge>
                      
                      <Badge variant="outline" className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        Área {program.area_id}
                      </Badge>
                    </div>
                    
                    {(program.city || program.department) && (
                      <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
                        <MapPin className="h-3 w-3" />
                        {[program.city, program.department].filter(Boolean).join(', ')}
                      </div>
                    )}
                    
                    <Button className="w-full mt-4" variant="outline">
                      <BookOpen className="h-4 w-4 mr-2" />
                      Ver detalles
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Paginación */}
          {totalPages > 1 && (
            <div className="flex justify-center items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(filters.page - 1)}
                disabled={filters.page <= 1}
              >
                <ChevronLeft className="h-4 w-4" />
                Anterior
              </Button>
              
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const pageNum = Math.max(1, Math.min(totalPages - 4, filters.page - 2)) + i;
                  return (
                    <Button
                      key={pageNum}
                      variant={pageNum === filters.page ? "default" : "outline"}
                      size="sm"
                      onClick={() => handlePageChange(pageNum)}
                    >
                      {pageNum}
                    </Button>
                  );
                })}
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(filters.page + 1)}
                disabled={filters.page >= totalPages}
              >
                Siguiente
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="text-center py-12">
            <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              No se encontraron programas
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Intenta ajustar los filtros de búsqueda para encontrar más resultados.
            </p>
            <Button onClick={clearAllFilters} variant="outline">
              Limpiar filtros
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}