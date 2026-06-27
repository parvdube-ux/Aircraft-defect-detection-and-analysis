import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Home, History, Settings, Activity } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import InspectionPage from './pages/InspectionPage';
import HistoryPage from './pages/HistoryPage';
import SettingsPage from './pages/SettingsPage';

function Sidebar() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/history', label: 'History', icon: History },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside style={{
      width: '260px',
      borderRight: '1px solid var(--border)',
      background: 'rgba(0,0,0,0.3)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      flexDirection: 'column',
      padding: '24px 0'
    }}>
      <div style={{ padding: '0 24px', marginBottom: '40px' }}>
        <div style={{
          fontFamily: "'Outfit', sans-serif",
          fontSize: '1.5rem',
          fontWeight: 700,
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <div style={{
            width: '36px', height: '36px',
            background: 'linear-gradient(135deg, var(--primary), var(--accent))',
            borderRadius: '10px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px var(--primary-glow)'
          }}>
            <Activity color="white" size={20} />
          </div>
          EdgeVision AI
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '8px', padding: '0 16px' }}>
        {navItems.map(item => {
          const isActive = location.pathname === item.path;
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                padding: '12px 16px',
                borderRadius: '12px',
                color: isActive ? 'white' : 'var(--text-muted)',
                background: isActive ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
                textDecoration: 'none',
                fontWeight: 600,
                transition: 'all 0.2s',
                border: isActive ? '1px solid rgba(59, 130, 246, 0.3)' : '1px solid transparent'
              }}
            >
              <Icon size={20} color={isActive ? 'var(--primary)' : 'currentColor'} />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', width: '100%', minHeight: '100vh' }}>
        <Sidebar />
        <main style={{ flex: 1, padding: '32px 48px', display: 'flex', flexDirection: 'column' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/inspection" element={<InspectionPage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
