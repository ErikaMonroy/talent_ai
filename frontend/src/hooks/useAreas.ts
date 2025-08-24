import { useState, useEffect } from 'react';
import { apiService } from '../services/api';

export interface AreaInfo {
  id: number;
  name: string;
  description?: string;
  category?: string;
}

export const useAreaInfo = (areaId: number | null) => {
  const [areaInfo, setAreaInfo] = useState<AreaInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!areaId) {
      setAreaInfo(null);
      return;
    }

    const fetchAreaInfo = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await apiService.getAreaById(areaId);
        
        if (response.success && response.data) {
          setAreaInfo({
            id: response.data.id,
            name: response.data.name || `Área ${areaId}`,
            description: response.data.description,
            category: response.data.category
          });
        } else {
          setError(response.error?.message || 'Error al obtener información del área');
          // Fallback: usar ID como nombre si no se puede obtener la información
          setAreaInfo({
            id: areaId,
            name: `Área ${areaId}`
          });
        }
      } catch (err) {
        setError('Error de conexión');
        // Fallback: usar ID como nombre
        setAreaInfo({
          id: areaId,
          name: `Área ${areaId}`
        });
      } finally {
        setLoading(false);
      }
    };

    fetchAreaInfo();
  }, [areaId]);

  return { areaInfo, loading, error };
};

// Hook para obtener múltiples áreas
export const useMultipleAreas = (areaIds: number[]) => {
  const [areasInfo, setAreasInfo] = useState<Record<number, AreaInfo>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (areaIds.length === 0) {
      setAreasInfo({});
      return;
    }

    const fetchAreasInfo = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const results: Record<number, AreaInfo> = {};
        
        // Obtener información de cada área
        await Promise.all(
          areaIds.map(async (areaId) => {
            try {
              const response = await apiService.getAreaById(areaId);
              
              if (response.success && response.data) {
                results[areaId] = {
                  id: response.data.id,
                  name: response.data.name || `Área ${areaId}`,
                  description: response.data.description,
                  category: response.data.category
                };
              } else {
                // Fallback
                results[areaId] = {
                  id: areaId,
                  name: `Área ${areaId}`
                };
              }
            } catch {
              // Fallback en caso de error
              results[areaId] = {
                id: areaId,
                name: `Área ${areaId}`
              };
            }
          })
        );
        
        setAreasInfo(results);
      } catch (err) {
        setError('Error al obtener información de las áreas');
        // Crear fallbacks para todas las áreas
        const fallbacks: Record<number, AreaInfo> = {};
        areaIds.forEach(id => {
          fallbacks[id] = { id, name: `Área ${id}` };
        });
        setAreasInfo(fallbacks);
      } finally {
        setLoading(false);
      }
    };

    fetchAreasInfo();
  }, [areaIds.join(',')]);

  return { areasInfo, loading, error };
};