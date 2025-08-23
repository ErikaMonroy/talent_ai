'use client';

import React from 'react';
import { cn } from '@/lib/utils';
import { LikertScaleProps } from '@/types/assessment';

const LIKERT_OPTIONS = [
  { value: 1, label: 'Nunca', description: 'No me describe en absoluto' },
  { value: 2, label: 'Raramente', description: 'Me describe muy poco' },
  { value: 3, label: 'A veces', description: 'Me describe parcialmente' },
  { value: 4, label: 'Frecuentemente', description: 'Me describe bastante bien' },
  { value: 5, label: 'Siempre', description: 'Me describe completamente' }
];

export function LikertScale({ value, onChange, disabled = false }: LikertScaleProps) {
  return (
    <div className="space-y-4">
      {/* Desktop version */}
      <div className="hidden md:block">
        <div className="grid grid-cols-5 gap-4">
          {LIKERT_OPTIONS.map((option) => (
            <div key={option.value} className="text-center">
              <button
                type="button"
                onClick={() => onChange(option.value)}
                disabled={disabled}
                className={cn(
                  'w-full p-4 rounded-lg border-2 transition-all duration-200',
                  'hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  value === option.value
                    ? 'border-primary bg-primary/10 text-primary-foreground'
                    : 'border-border hover:border-muted-foreground'
                )}
              >
                <div className="space-y-2">
                  <div className="text-sm font-medium text-foreground">
                    {option.label}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {option.description}
                  </div>
                </div>
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Mobile version */}
      <div className="md:hidden space-y-3">
        {LIKERT_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            disabled={disabled}
            className={cn(
              'w-full p-4 rounded-lg border-2 transition-all duration-200',
              'hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'flex items-center space-x-4',
              value === option.value
                ? 'border-primary bg-primary/10 text-primary-foreground'
                : 'border-border hover:border-muted-foreground'
            )}
          >
            <div className="flex-1 text-left">
              <div className="text-sm font-medium text-foreground">
                {option.label}
              </div>
              <div className="text-xs text-muted-foreground">
                {option.description}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}