import React, { useEffect, useState } from 'react';
import { History, FileText } from 'lucide-react';

const API_BASE = '';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history?limit=20`);
      if (!res.ok) throw new Error('Failed to fetch history');
      const data = await res.json();
      setHistory(data.records);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    const r = risk?.toLowerCase() || 'none';
    if (r === 'critical') return '#fca5a5';
    if (r === 'high') return 'var(--danger)';
    if (r === 'medium') return 'var(--warning)';
    if (r === 'low') return 'var(--success)';
    return 'var(--text-muted)';
  };

  return (
    <div className="card animate-slide-in">
      <div className="card-header">
        <History size={24} />
        Inspection History
      </div>
      
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" style={{ margin: 'auto' }} /></div>
      ) : error ? (
        <div style={{ color: 'var(--danger)' }}>{error}</div>
      ) : history.length === 0 ? (
        <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px' }}>No inspection history found.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {history.map((record, i) => (
            <div key={i} className="animate-fade-in" style={{
              background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)',
              padding: '20px', borderRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              animationDelay: `${i * 0.05}s`
            }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: '1.1rem', marginBottom: '4px' }}>{record.image}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  {new Date(record.timestamp).toLocaleString()} &bull; {record.num_defects} defects found
                </div>
              </div>
              
              <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Highest Risk</div>
                  <div style={{ fontWeight: 700, color: getRiskColor(record.highest_risk) }}>{record.highest_risk}</div>
                </div>
                {record.pdf_report && (
                  <a href={`${API_BASE}/reports/${record.pdf_report.split('/').pop().split('\\').pop()}`} target="_blank" rel="noreferrer" className="btn" style={{ padding: '10px' }}>
                    <FileText size={18} />
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
