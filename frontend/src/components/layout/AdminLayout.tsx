import { useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import {
  LayoutDashboard,
  Target,
  History,
  FlaskConical,
  Brain,
  BarChart3,
  Beaker,
  Globe2,
  Users,
  Settings,
  ChevronLeft,
  ChevronRight,
  LogOut,
  Sun,
  Moon,
  Monitor,
  Zap,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useAuthStore } from '@/stores/authStore'
import { useThemeStore } from '@/stores/themeStore'

const navItems = [
  { path: '/', icon: LayoutDashboard, labelKey: 'nav.dashboard' },
  { path: '/predictions', icon: Target, labelKey: 'nav.predictions' },
  { path: '/models', icon: History, labelKey: 'nav.models' },
  { path: '/backtests', icon: FlaskConical, labelKey: 'nav.backtests' },
  { path: '/rdagent', icon: Brain, labelKey: 'nav.rdagent' },
  { path: '/performance', icon: BarChart3, labelKey: 'nav.performance' },
  { path: '/qlib', icon: Beaker, labelKey: 'nav.qlib' },
  { path: '/universes', icon: Globe2, labelKey: 'nav.universes' },
  { path: '/consumers', icon: Users, labelKey: 'nav.consumers' },
  { path: '/settings', icon: Settings, labelKey: 'nav.settings' },
]

export function AdminLayout() {
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const { t } = useTranslation()
  const { user, logout } = useAuthStore()
  const { theme, setTheme } = useThemeStore()

  const handleLogout = async () => {
    await logout()
  }

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dark')
    else if (theme === 'dark') setTheme('system')
    else setTheme('light')
  }

  const themeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <aside
          className={cn(
            'flex flex-col border-r bg-card transition-all duration-300',
            collapsed ? 'w-16' : 'w-64'
          )}
        >
          {/* Logo */}
          <div className="flex h-14 items-center border-b px-4">
            {!collapsed && (
              <Link to="/" className="flex items-center gap-2">
                <Zap className="h-6 w-6 text-primary" />
                <span className="text-lg font-bold">AlphaForge</span>
              </Link>
            )}
            {collapsed && (
              <Link to="/" className="mx-auto">
                <Zap className="h-6 w-6 text-primary" />
              </Link>
            )}
          </div>

          {/* Nav */}
          <ScrollArea className="flex-1 py-2">
            <nav className="flex flex-col gap-1 px-2">
              {navItems.map((item) => {
                const isActive =
                  item.path === '/'
                    ? location.pathname === '/'
                    : location.pathname.startsWith(item.path)
                const Icon = item.icon

                if (collapsed) {
                  return (
                    <Tooltip key={item.path}>
                      <TooltipTrigger asChild>
                        <Link to={item.path}>
                          <Button
                            variant={isActive ? 'secondary' : 'ghost'}
                            size="icon"
                            className="w-full"
                          >
                            <Icon className="h-5 w-5" />
                          </Button>
                        </Link>
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        {t(item.labelKey)}
                      </TooltipContent>
                    </Tooltip>
                  )
                }

                return (
                  <Link key={item.path} to={item.path}>
                    <Button
                      variant={isActive ? 'secondary' : 'ghost'}
                      className="w-full justify-start gap-3"
                    >
                      <Icon className="h-5 w-5" />
                      {t(item.labelKey)}
                    </Button>
                  </Link>
                )
              })}
            </nav>
          </ScrollArea>

          {/* Collapse toggle */}
          <Separator />
          <div className="p-2">
            <Button
              variant="ghost"
              size="icon"
              className="w-full"
              onClick={() => setCollapsed(!collapsed)}
            >
              {collapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Header */}
          <header className="flex h-14 items-center justify-between border-b px-6">
            <h1 className="text-lg font-semibold">{t('app.title')}</h1>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={cycleTheme}>
                {(() => {
                  const ThemeIcon = themeIcon
                  return <ThemeIcon className="h-5 w-5" />
                })()}
              </Button>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="gap-2">
                    <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-medium">
                      {user?.email?.charAt(0).toUpperCase()}
                    </div>
                    {!collapsed && (
                      <span className="text-sm">{user?.email}</span>
                    )}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>{user?.email}</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={handleLogout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    {t('common.logout')}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </header>

          {/* Page content */}
          <main className="flex-1 overflow-auto">
            <div className="container py-6">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </TooltipProvider>
  )
}
