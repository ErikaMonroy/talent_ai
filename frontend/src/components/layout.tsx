import { Link, useLocation } from 'react-router-dom'
import { Brain, Home, FileText, BarChart3, Search, Info } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ThemeToggle } from '@/components/theme-toggle'
import { cn } from '@/lib/utils'

interface LayoutProps {
  children: React.ReactNode
}

const navigation = [
  {
    name: 'Inicio',
    href: '/',
    icon: Home,
  },
  {
    name: 'Evaluación',
    href: '/evaluation',
    icon: FileText,
  },
  {
    name: 'Resultados',
    href: '/results',
    icon: BarChart3,
  },
  {
    name: 'Programas',
    href: '/programs',
    icon: Search,
  },
  {
    name: 'Acerca de',
    href: '/about',
    icon: Info,
  },
]

export function Layout({ children }: LayoutProps) {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <Brain className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold text-primary">TalentAI</span>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.href
              
              return (
                <Button
                  key={item.name}
                  variant={isActive ? 'default' : 'ghost'}
                  size="sm"
                  asChild
                  className={cn(
                    'flex items-center space-x-2',
                    isActive && 'bg-primary text-primary-foreground'
                  )}
                >
                  <Link to={item.href}>
                    <Icon className="h-4 w-4" />
                    <span>{item.name}</span>
                  </Link>
                </Button>
              )
            })}
          </nav>

          {/* Theme Toggle */}
          <div className="flex items-center space-x-2">
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Mobile Navigation */}
      <nav className="md:hidden border-b bg-background">
        <div className="container">
          <div className="flex items-center justify-around py-2">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.href
              
              return (
                <Button
                  key={item.name}
                  variant={isActive ? 'default' : 'ghost'}
                  size="sm"
                  asChild
                  className={cn(
                    'flex flex-col items-center space-y-1 h-auto py-2 px-3',
                    isActive && 'bg-primary text-primary-foreground'
                  )}
                >
                  <Link to={item.href}>
                    <Icon className="h-4 w-4" />
                    <span className="text-xs">{item.name}</span>
                  </Link>
                </Button>
              )
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="border-t bg-muted/50">
        <div className="container py-8">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6 text-primary" />
              <span className="text-lg font-semibold text-primary">TalentAI</span>
            </div>
            <p className="text-sm text-muted-foreground text-center md:text-right">
              © 2024 TalentAI. Plataforma de evaluación de talento con inteligencia artificial.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}