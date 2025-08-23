import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface Dimension {
  id: number
  name: string
  description: string
  color: string
}

interface DimensionSelectorProps {
  dimensions: Dimension[]
  selectedDimensions: number[]
  onSelectionChange: (dimensions: number[]) => void
  maxSelections?: number
  className?: string
  disabled?: boolean
}

const defaultDimensions: Dimension[] = [
  {
    id: 1,
    name: 'Lógico-Matemático',
    description: 'Capacidad para resolver problemas lógicos y matemáticos',
    color: 'dimension-1'
  },
  {
    id: 2,
    name: 'Comunicación',
    description: 'Habilidades de comunicación verbal y escrita',
    color: 'dimension-2'
  },
  {
    id: 3,
    name: 'Ciencias',
    description: 'Conocimiento y aplicación de conceptos científicos',
    color: 'dimension-3'
  },
  {
    id: 4,
    name: 'Humanidades',
    description: 'Comprensión de historia, literatura y cultura',
    color: 'dimension-4'
  },
  {
    id: 5,
    name: 'Creatividad',
    description: 'Capacidad de generar ideas innovadoras y originales',
    color: 'dimension-5'
  },
  {
    id: 6,
    name: 'Liderazgo',
    description: 'Habilidades para dirigir y motivar equipos',
    color: 'dimension-6'
  },
  {
    id: 7,
    name: 'Pensamiento Crítico',
    description: 'Análisis objetivo y evaluación de información',
    color: 'dimension-7'
  },
  {
    id: 8,
    name: 'Adaptabilidad',
    description: 'Flexibilidad ante cambios y nuevas situaciones',
    color: 'dimension-8'
  }
]

export function DimensionSelector({
  dimensions = defaultDimensions,
  selectedDimensions,
  onSelectionChange,
  maxSelections,
  className,
  disabled = false
}: DimensionSelectorProps) {
  const handleDimensionToggle = (dimensionId: number) => {
    if (disabled) return

    const isSelected = selectedDimensions.includes(dimensionId)
    
    if (isSelected) {
      // Remove dimension
      onSelectionChange(selectedDimensions.filter(id => id !== dimensionId))
    } else {
      // Add dimension if under max limit
      if (!maxSelections || selectedDimensions.length < maxSelections) {
        onSelectionChange([...selectedDimensions, dimensionId])
      }
    }
  }

  return (
    <div className={cn('space-y-4', className)}>
      {maxSelections && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Selecciona hasta {maxSelections} dimensiones
          </p>
          <Badge variant="outline">
            {selectedDimensions.length} / {maxSelections}
          </Badge>
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {dimensions.map((dimension) => {
          const isSelected = selectedDimensions.includes(dimension.id)
          const canSelect = !maxSelections || selectedDimensions.length < maxSelections || isSelected
          
          return (
            <Card
              key={dimension.id}
              className={cn(
                'cursor-pointer transition-all duration-200 hover:shadow-md',
                isSelected && 'ring-2 ring-primary shadow-lg',
                !canSelect && 'opacity-50 cursor-not-allowed',
                disabled && 'opacity-50 cursor-not-allowed'
              )}
              onClick={() => canSelect && handleDimensionToggle(dimension.id)}
            >
              <CardContent className="p-4">
                <div className="flex items-start space-x-3">
                  <div 
                    className={cn(
                      'w-4 h-4 rounded-full flex-shrink-0 mt-0.5',
                      `bg-${dimension.color}`,
                      isSelected && 'ring-2 ring-white shadow-md'
                    )}
                  />
                  <div className="flex-1 min-w-0">
                    <h3 className={cn(
                      'font-medium text-sm leading-tight',
                      isSelected && 'text-primary font-semibold'
                    )}>
                      {dimension.name}
                    </h3>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                      {dimension.description}
                    </p>
                  </div>
                  {isSelected && (
                    <div className="flex-shrink-0">
                      <Badge className="bg-primary text-primary-foreground text-sm">
                        ✓
                      </Badge>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
      
      {selectedDimensions.length > 0 && (
        <div className="mt-4 p-4 bg-muted rounded-lg">
          <h4 className="font-medium text-sm mb-2">Dimensiones seleccionadas:</h4>
          <div className="flex flex-wrap gap-2">
            {selectedDimensions.map((id) => {
              const dimension = dimensions.find(d => d.id === id)
              if (!dimension) return null
              
              return (
                <Badge
                  key={id}
                  variant="secondary"
                  className={cn(
                    'text-xs',
                    `border-${dimension.color} bg-${dimension.color}/10`
                  )}
                >
                  <div className={`w-2 h-2 rounded-full bg-${dimension.color} mr-1`} />
                  {dimension.name}
                </Badge>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}