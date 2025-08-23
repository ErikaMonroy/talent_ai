'use client';

import React from 'react';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Home, RotateCcw } from 'lucide-react';
import { NavigationControlsProps } from '@/types/assessment';
import { cn } from '@/lib/utils';

export function NavigationControls({
  onPrevious,
  onNext,
  onHome,
  onReset,
  canGoPrevious = true,
  canGoNext = true,
  nextLabel = 'Siguiente',
  previousLabel = 'Anterior',
  showHome = true,
  showReset = true,
  className
}: NavigationControlsProps) {
  return (
    <div className={cn('flex items-center justify-between w-full', className)}>
      {/* Left side - Previous button */}
      <div className="flex items-center space-x-2">
        {canGoPrevious && onPrevious && (
          <Button
            variant="outline"
            onClick={onPrevious}
            className="flex items-center space-x-2"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>{previousLabel}</span>
          </Button>
        )}
      </div>

      {/* Center - Home and Reset buttons */}
      <div className="flex items-center space-x-2">
        {showHome && onHome && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onHome}
            className="flex items-center space-x-2 text-muted-foreground hover:text-foreground"
          >
            <Home className="w-4 h-4" />
            <span className="hidden sm:inline">Inicio</span>
          </Button>
        )}
        
        {showReset && onReset && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onReset}
            className="flex items-center space-x-2 text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="hidden sm:inline">Reiniciar</span>
          </Button>
        )}
      </div>

      {/* Right side - Next button */}
      <div className="flex items-center space-x-2">
        {canGoNext && onNext && (
          <Button
            onClick={onNext}
            className="flex items-center space-x-2"
          >
            <span>{nextLabel}</span>
            <ChevronRight className="w-4 h-4" />
          </Button>
        )}
      </div>
    </div>
  );
}

// Variant for mobile with stacked layout
export function MobileNavigationControls({
  onPrevious,
  onNext,
  onHome,
  onReset,
  canGoPrevious = true,
  canGoNext = true,
  nextLabel = 'Siguiente',
  previousLabel = 'Anterior',
  showHome = true,
  showReset = true,
  className
}: NavigationControlsProps) {
  return (
    <div className={cn('space-y-4 w-full', className)}>
      {/* Main navigation buttons */}
      <div className="flex items-center justify-between space-x-4">
        {canGoPrevious && onPrevious ? (
          <Button
            variant="outline"
            onClick={onPrevious}
            className="flex-1 flex items-center justify-center space-x-2"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>{previousLabel}</span>
          </Button>
        ) : (
          <div className="flex-1" />
        )}
        
        {canGoNext && onNext && (
          <Button
            onClick={onNext}
            className="flex-1 flex items-center justify-center space-x-2"
          >
            <span>{nextLabel}</span>
            <ChevronRight className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Secondary actions */}
      {(showHome || showReset) && (
        <div className="flex items-center justify-center space-x-4">
          {showHome && onHome && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onHome}
              className="flex items-center space-x-2 text-muted-foreground hover:text-foreground"
            >
              <Home className="w-4 h-4" />
              <span>Inicio</span>
            </Button>
          )}
          
          {showReset && onReset && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onReset}
              className="flex items-center space-x-2 text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300"
            >
              <RotateCcw className="w-4 h-4" />
              <span>Reiniciar</span>
            </Button>
          )}
        </div>
      )}
    </div>
  );
}