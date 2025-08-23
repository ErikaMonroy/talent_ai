import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface EvaluationCardProps {
  title: string
  description: string
  dimension: number
  score?: number
  maxScore?: number
  status?: 'pending' | 'in-progress' | 'completed'
  className?: string
  children?: React.ReactNode
}

const dimensionColors = {
  1: 'dimension-1', // Lógico-Matemático
  2: 'dimension-2', // Comunicación
  3: 'dimension-3', // Ciencias
  4: 'dimension-4', // Humanidades
  5: 'dimension-5', // Creatividad
  6: 'dimension-6', // Liderazgo
  7: 'dimension-7', // Pensamiento Crítico
  8: 'dimension-8', // Adaptabilidad
}

const statusConfig = {
  pending: {
    label: 'Pendiente',
    variant: 'secondary' as const,
  },
  'in-progress': {
    label: 'En Progreso',
    variant: 'default' as const,
  },
  completed: {
    label: 'Completado',
    variant: 'default' as const,
  },
}

export function EvaluationCard({
  title,
  description,
  dimension,
  score,
  maxScore = 100,
  status = 'pending',
  className,
  children,
}: EvaluationCardProps) {
  const colorClass = dimensionColors[dimension as keyof typeof dimensionColors] || 'dimension-1'
  const statusInfo = statusConfig[status]
  const progressPercentage = score ? (score / maxScore) * 100 : 0

  return (
    <Card className={cn('hover:shadow-lg transition-all duration-300', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <div className={`w-3 h-3 rounded-full bg-${colorClass}`} />
              <Badge variant={statusInfo.variant} className="text-xs">
                {statusInfo.label}
              </Badge>
            </div>
            <CardTitle className="text-lg font-semibold">{title}</CardTitle>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-foreground">
              {score !== undefined ? score : '--'}
            </div>
            <div className="text-xs text-muted-foreground">/ {maxScore}</div>
          </div>
        </div>
        <CardDescription className="text-sm leading-relaxed">
          {description}
        </CardDescription>
      </CardHeader>
      
      {(score !== undefined || children) && (
        <CardContent className="pt-0">
          {score !== undefined && (
            <div className="mb-4">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-muted-foreground">Progreso</span>
                <span className="font-medium">{Math.round(progressPercentage)}%</span>
              </div>
              <Progress 
                value={progressPercentage} 
                className={`h-2 bg-${colorClass}-light`}
              />
            </div>
          )}
          {children}
        </CardContent>
      )}
    </Card>
  )
}