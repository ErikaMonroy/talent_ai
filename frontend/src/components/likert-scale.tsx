import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Label } from '@/components/ui/label'

interface LikertScaleProps {
  question: string
  name: string
  value?: number
  onChange: (value: number) => void
  scale?: number
  labels?: string[]
  required?: boolean
  className?: string
  disabled?: boolean
}

const defaultLabels = [
  'Totalmente en desacuerdo',
  'En desacuerdo',
  'Neutral',
  'De acuerdo',
  'Totalmente de acuerdo'
]

export function LikertScale({
  question,
  name,
  value,
  onChange,
  scale = 5,
  labels = defaultLabels,
  required = false,
  className,
  disabled = false,
}: LikertScaleProps) {
  const [hoveredValue, setHoveredValue] = useState<number | null>(null)

  const handleClick = (selectedValue: number) => {
    if (!disabled) {
      onChange(selectedValue)
    }
  }

  const getScaleColor = (index: number) => {
    const intensity = ((index + 1) / scale) * 100
    if (intensity <= 20) return 'bg-destructive'
    if (intensity <= 40) return 'bg-warning'
    if (intensity <= 60) return 'bg-secondary'
    if (intensity <= 80) return 'bg-primary'
    return 'bg-success'
  }

  const getScaleColorHover = (index: number) => {
    const intensity = ((index + 1) / scale) * 100
    if (intensity <= 20) return 'hover:bg-destructive/80'
    if (intensity <= 40) return 'hover:bg-warning/80'
    if (intensity <= 60) return 'hover:bg-secondary/80'
    if (intensity <= 80) return 'hover:bg-primary/80'
    return 'hover:bg-success/80'
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div className="space-y-2">
        <Label className="text-base font-medium leading-relaxed">
          {question}
          {required && <span className="text-destructive ml-1">*</span>}
        </Label>
      </div>

      <div className="space-y-3">
        {/* Scale buttons */}
        <div className="flex justify-between items-center gap-2">
          {Array.from({ length: scale }, (_, index) => {
            const scaleValue = index + 1
            const isSelected = value === scaleValue
            const isHovered = hoveredValue === scaleValue
            
            return (
              <button
                key={index}
                type="button"
                onClick={() => handleClick(scaleValue)}
                onMouseEnter={() => setHoveredValue(scaleValue)}
                onMouseLeave={() => setHoveredValue(null)}
                disabled={disabled}
                className={cn(
                  'flex-1 h-12 rounded-lg border-2 transition-all duration-200 font-medium text-sm',
                  'focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
                  disabled && 'opacity-50 cursor-not-allowed',
                  !disabled && 'cursor-pointer',
                  isSelected || isHovered
                    ? cn(
                        getScaleColor(index),
                        'border-transparent text-white shadow-md transform scale-105'
                      )
                    : cn(
                        'border-border bg-background text-foreground',
                        !disabled && getScaleColorHover(index)
                      )
                )}
                aria-label={`Escala ${scaleValue} de ${scale}`}
              >
                {scaleValue}
              </button>
            )
          })}
        </div>

        {/* Scale labels */}
        <div className="flex justify-between text-xs text-muted-foreground px-1">
          <span className="text-left max-w-[80px]">
            {labels[0] || `1 - Muy bajo`}
          </span>
          <span className="text-center">
            {labels[Math.floor(scale / 2)] || 'Neutral'}
          </span>
          <span className="text-right max-w-[80px]">
            {labels[scale - 1] || `${scale} - Muy alto`}
          </span>
        </div>

        {/* Current selection indicator */}
        {value && (
          <div className="text-sm text-center p-2 bg-muted rounded-md">
            <span className="font-medium">Seleccionado: </span>
            <span className="text-primary font-semibold">
              {value} de {scale}
            </span>
            {labels[value - 1] && (
              <span className="text-muted-foreground ml-2">
                ({labels[value - 1]})
              </span>
            )}
          </div>
        )}
      </div>

      {/* Hidden input for form submission */}
      <input
        type="hidden"
        name={name}
        value={value || ''}
      />
    </div>
  )
}