'use client';

import React, { useEffect, useState } from 'react';
import { apiService } from '@/services/api';
import { Program } from '@/types/api';

export default function Debug() {
  const [programs, setPrograms] = useState<Program[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const testAPI = async () => {
      setLoading(true);
      setError(null);
      
      try {
        console.log('🧪 Testing API connection...');
        const response = await apiService.searchPrograms({ limit: 5 });
        console.log('🧪 API Response:', response);
        
        if (response.success && response.data) {
          setPrograms(response.data.programs || []);
          console.log('🧪 Programs loaded:', response.data.programs?.length);
        } else {
          setError('API returned unsuccessful response');
          console.error('🧪 API Error:', response.error);
        }
      } catch (err) {
        console.error('🧪 Exception:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    testAPI();
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-4">API Debug Page</h1>
      
      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      
      <div className="mt-4">
        <h2 className="text-lg font-semibold mb-2">Programs ({programs.length}):</h2>
        {programs.map((program) => (
          <div key={program.id} className="border p-4 mb-2 rounded">
            <h3 className="font-medium">{program.name}</h3>
            <p className="text-sm text-gray-600">{program.institution} - {program.city}</p>
            <p className="text-xs text-gray-500">{program.academic_level} | {program.modality}</p>
          </div>
        ))}
      </div>
    </div>
  );
}