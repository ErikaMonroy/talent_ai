'use client';

import React from 'react';
import { ProgramsTable } from '@/components/programs/ProgramsTable';

export default function ProgramSearch() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Búsqueda de Programas Académicos
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Explora y encuentra programas académicos que se ajusten a tus intereses y objetivos profesionales.
          </p>
        </div>

        <ProgramsTable 
          showAreaRecommendations={false}
          title="Todos los Programas Académicos"
        />
      </div>
    </div>
  );
}