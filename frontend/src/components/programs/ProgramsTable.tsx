'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
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
import { apiService } from '@/services/api';
import { Program, ProgramSearchParams } from '@/types/api';

interface ProgramsTableProps {
  className?: string;
  initialAreaId?: number;
  showAreaRecommendations?: boolean;
  recommendedAreas?: Array<{ area_id: number; percentage: number }>;
  areasInfo?: Record<number, { name: string; code?: string }>;
  title?: string;
}

interface LocalFilters {
  name: string;
  area: string;
  city: string;
  department: string;
  academic_level: string;
}

interface PaginationState {
  currentPage: number;
  itemsPerPage: number;
  totalItems: number;
  totalPages: number;
}

export function ProgramsTable({ 
  className,
  initialAreaId,
  showAreaRecommendations = false,
  recommendedAreas = [],
  areasInfo = {},
  title = "Programas Encontrados"
}: ProgramsTableProps) {
  // Estados locales
  const [programs, setPrograms] = useState<Program[]>([]);
  const [availableFilters, setAvailableFilters] = useState<{
    cities: string[];
    departments: string[];
    academic_levels: string[];
    knowledge_areas: { id: number; name: string; code: string; }[];
  }>({ cities: [], departments: [], academic_levels: [], knowledge_areas: [] });
  const [allAreas, setAllAreas] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingFilters, setIsLoadingFilters] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  
  // Filtros de búsqueda
  const [searchParams, setSearchParams] = useState<ProgramSearchParams>({
    area_id: initialAreaId,
    limit: 10,
    offset: 0
  });
  
  const [localFilters, setLocalFilters] = useState<LocalFilters>({
    name: '',
    area: initialAreaId ? initialAreaId.toString() : 'all',
    city: 'all',
    department: 'all',
    academic_level: 'all'
  });

  const [pagination, setPagination] = useState<PaginationState>({
    currentPage: 1,
    itemsPerPage: 10,
    totalItems: 0,
    totalPages: 0
  });

  // Cargar filtros disponibles y áreas
  useEffect(() => {
    const loadFiltersAndAreas = async () => {
      setIsLoadingFilters(true);
      try {
        const [filtersResponse, areasResponse] = await Promise.all([
          apiService.getAvailableFilters(),
          apiService.getKnowledgeAreas({ limit: 100 })
        ]);
        
        setAvailableFilters({
          cities: filtersResponse.data.cities || [],
          departments: filtersResponse.data.departments || [],
          academic_levels: filtersResponse.data.academic_levels || [],
          knowledge_areas: filtersResponse.data.knowledge_areas || []
        });
        
        if (areasResponse.success && areasResponse.data?.areas) {
          setAllAreas(areasResponse.data.areas);
        }
      } catch (err) {
        console.error('Error loading filters and areas:', err);
        // No mostrar error para filtros, usar valores por defecto
      } finally {
        setIsLoadingFilters(false);
      }
    };

    loadFiltersAndAreas();
  }, []);

  // Cargar programas con paginación
  useEffect(() => {
    const loadPrograms = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const params = {
          ...searchParams,
          area_id: localFilters.area !== 'all' ? parseInt(localFilters.area) : searchParams.area_id,
          limit: pagination.itemsPerPage,
          offset: (pagination.currentPage - 1) * pagination.itemsPerPage
        };
        
        console.log('🔍 Loading programs with params:', params);
        
        const response = await apiService.searchPrograms(params);
        console.log('📡 API Response:', response);
        
        const programsData = response.data?.programs || [];
        const paginationData = response.data?.pagination;
        
        console.log('📊 Programs data:', programsData);
        console.log('📄 Pagination data:', paginationData);
        
        setPrograms(programsData);
        setPagination(prev => ({
          ...prev,
          totalItems: paginationData?.total || programsData.length,
          totalPages: paginationData?.total_pages || Math.ceil((paginationData?.total || programsData.length) / prev.itemsPerPage)
        }));
      } catch (err) {
        console.error('❌ Error loading programs:', err);
        setError(err instanceof Error ? err.message : 'Error al cargar programas');
        setPrograms([]);
        setPagination(prev => ({ ...prev, totalItems: 0, totalPages: 0 }));
      } finally {
        setIsLoading(false);
      }
    };

    loadPrograms();
  }, [searchParams, localFilters.area, pagination.currentPage, pagination.itemsPerPage]);

  // Actualizar parámetros de búsqueda cuando cambien los filtros locales
  useEffect(() => {
    const params: ProgramSearchParams = {
      area_id: localFilters.area !== 'all' ? parseInt(localFilters.area) : initialAreaId,
      limit: pagination.itemsPerPage,
      offset: 0
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
    setPagination(prev => ({ ...prev, currentPage: 1 }));
  }, [localFilters, pagination.itemsPerPage, initialAreaId]);

  const handleResetFilters = () => {
    setLocalFilters({
      name: '',
      area: initialAreaId ? initialAreaId.toString() : 'all',
      city: 'all',
      department: 'all',
      academic_level: 'all'
    });
    setPagination(prev => ({ ...prev, currentPage: 1 }));
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

  const recommendedAreaIds = recommendedAreas.map(area => area.area_id);

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
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

              {/* Área de Conocimiento */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Área de Conocimiento</label>
                <Select 
                  value={localFilters.area} 
                  onValueChange={(value) => {
                    setLocalFilters(prev => ({ ...prev, area: value }));
                    setPagination(prev => ({ ...prev, currentPage: 1 }));
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Seleccionar área" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todas las áreas</SelectItem>
                    
                    {/* Áreas recomendadas */}
                    {showAreaRecommendations && recommendedAreas.length > 0 && (
                      <>
                        <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/50">
                          🎯 Áreas Recomendadas
                        </div>
                        {recommendedAreas.map((area) => (
                          <SelectItem key={`rec-${area.area_id}`} value={area.area_id.toString()}>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs bg-yellow-100 text-yellow-800">
                                Top
                              </Badge>
                              {areasInfo[area.area_id]?.name || `Área ${area.area_id}`}
                            </div>
                          </SelectItem>
                        ))}
                        <div className="h-px bg-border my-1" />
                      </>
                    )}
                    
                    {/* Todas las áreas disponibles */}
                    <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground bg-muted/50">
                      📚 Todas las Áreas
                    </div>
                    {allAreas.map((area) => (
                      <SelectItem key={`all-${area.id}`} value={area.id.toString()}>
                        {area.name || `Área ${area.id}`}
                      </SelectItem>
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
              {pagination.totalItems} Programa{pagination.totalItems !== 1 ? 's' : ''} encontrado{pagination.totalItems !== 1 ? 's' : ''}
            </h3>
            {programs.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Mostrar:</span>
                <Select 
                  value={pagination.itemsPerPage.toString()} 
                  onValueChange={(value) => {
                    setPagination(prev => ({
                      ...prev,
                      itemsPerPage: parseInt(value),
                      currentPage: 1
                    }));
                  }}
                >
                  <SelectTrigger className="w-20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="5">5</SelectItem>
                    <SelectItem value="10">10</SelectItem>
                    <SelectItem value="20">20</SelectItem>
                    <SelectItem value="50">50</SelectItem>
                  </SelectContent>
                </Select>
                <span className="text-sm text-muted-foreground">por página</span>
              </div>
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
              {programs.filter(program => program && program.id).map((program) => (
                <div 
                  key={program.id}
                  className="p-4 border rounded-lg hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg text-gray-900 dark:text-white mb-1">
                        {program.name || 'Programa sin nombre'}
                      </h4>
                      
                      <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400 mb-2">
                        <div className="flex items-center gap-1">
                          <Building2 className="h-4 w-4" />
                          {program.institution || 'Institución no disponible'}
                        </div>
                        
                        <div className="flex items-center gap-1">
                          <MapPin className="h-4 w-4" />
                          {program.city || 'Ciudad no disponible'}
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2 mb-3">
                        {getAcademicLevelBadge(program.academic_level || 'No especificado')}
                        
                        <Badge variant="outline" className="text-xs">
                          {program.knowledge_area?.name || 'Área no disponible'}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                        {program.description || `Programa de ${program.academic_level || 'nivel no especificado'} en ${program.institution || 'institución no disponible'}`}
                      </p>
                    </div>
                    
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => {
                        if (program?.institution && program?.name) {
                          const searchQuery = `${program.institution} - ${program.name}`;
                          const googleUrl = `https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`;
                          window.open(googleUrl, '_blank');
                        }
                      }}
                    >
                      <ExternalLink className="h-4 w-4 mr-2" />
                      Ver Más
                    </Button>
                  </div>
                  
                  {/* Información adicional */}
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400 pt-3 border-t">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {program.duration || 'Duración no disponible'}
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <Users className="h-3 w-3" />
                      {program.modality || 'Modalidad no disponible'}
                    </div>
                    
                    {showAreaRecommendations && program.knowledge_area?.id && recommendedAreaIds.includes(program.knowledge_area.id) && (
                      <div className="flex items-center gap-1 text-blue-600">
                        <Star className="h-3 w-3" />
                        Área recomendada
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
        
        {/* Paginación */}
        {pagination.totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 border-t">
            <div className="text-sm text-muted-foreground">
              Mostrando {((pagination.currentPage - 1) * pagination.itemsPerPage) + 1} a{' '}
              {Math.min(pagination.currentPage * pagination.itemsPerPage, pagination.totalItems)} de{' '}
              {pagination.totalItems} programas
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPagination(prev => ({ ...prev, currentPage: prev.currentPage - 1 }))}
                disabled={pagination.currentPage === 1}
              >
                Anterior
              </Button>
              
              <div className="flex items-center gap-1">
                {Array.from({ length: Math.min(5, pagination.totalPages) }, (_, i) => {
                  let pageNum;
                  if (pagination.totalPages <= 5) {
                    pageNum = i + 1;
                  } else if (pagination.currentPage <= 3) {
                    pageNum = i + 1;
                  } else if (pagination.currentPage >= pagination.totalPages - 2) {
                    pageNum = pagination.totalPages - 4 + i;
                  } else {
                    pageNum = pagination.currentPage - 2 + i;
                  }
                  
                  return (
                    <Button
                      key={pageNum}
                      variant={pagination.currentPage === pageNum ? "default" : "outline"}
                      size="sm"
                      className="w-8 h-8 p-0"
                      onClick={() => setPagination(prev => ({ ...prev, currentPage: pageNum }))}
                    >
                      {pageNum}
                    </Button>
                  );
                })}
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPagination(prev => ({ ...prev, currentPage: prev.currentPage + 1 }))}
                disabled={pagination.currentPage === pagination.totalPages}
              >
                Siguiente
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}