import React, { useState } from 'react';
import { Settings, Save, CheckCircle } from 'lucide-react';

const API_BASE = '';

export default function SettingsPage() {
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('gpt-4o-mini');
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSave = async (e) => {
    e.preventDefault();
    if (!apiKey) {
      setStatus({ type: 'error', message: 'API Key is required.' });
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      const res = await fetch(`${API_BASE}/api/configure/openai`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ api_key: apiKey, model: model })
      });

      if (!res.ok) throw new Error('Failed to update settings');
      
      setStatus({ type: 'success', message: 'Settings saved successfully. The API is now configured.' });
    } catch (err) {
      setStatus({ type: 'error', message: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card animate-slide-in" style={{ maxWidth: '600px', margin: '0 auto' }}>
      <div className="card-header">
        <Settings size={24} />
        System Configuration
      </div>

      <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {status && (
          <div style={{
            padding: '16px', borderRadius: '12px',
            background: status.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${status.type === 'success' ? 'var(--success)' : 'var(--danger)'}`,
            color: status.type === 'success' ? 'var(--success)' : '#fca5a5',
            display: 'flex', alignItems: 'center', gap: '10px'
          }}>
            {status.type === 'success' && <CheckCircle size={18} />}
            {status.message}
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.9rem' }}>OpenAI API Key</label>
          <input 
            type="password" 
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="sk-..."
            style={{
              padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border)',
              background: 'rgba(255,255,255,0.05)', color: 'white', outline: 'none',
              fontFamily: 'monospace'
            }}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label style={{ fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.9rem' }}>Model Preference</label>
          <select 
            value={model}
            onChange={e => setModel(e.target.value)}
            style={{
              padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--border)',
              background: 'rgba(255,255,255,0.05)', color: 'white', outline: 'none',
              appearance: 'none'
            }}
          >
            <option value="gpt-4o-mini">gpt-4o-mini (Fastest)</option>
            <option value="gpt-4o">gpt-4o (Most Capable)</option>
            <option value="gpt-4-turbo">gpt-4-turbo</option>
          </select>
        </div>

        <button type="submit" className="btn btn-primary" disabled={loading} style={{ marginTop: '10px' }}>
          {loading ? <div className="spinner" /> : <Save size={18} />}
          Save Configuration
        </button>
      </form>
    </div>
  );
}
