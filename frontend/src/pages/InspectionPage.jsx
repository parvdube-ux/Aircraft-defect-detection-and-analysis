import React, { useState, useEffect } from 'react';
import { marked } from 'marked';
import { UploadCloud, CheckCircle, AlertTriangle, FileText, Download, Activity, ArrowLeft } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';

const API_BASE = '';

export default function InspectionPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const file = location.state?.file;

  const [originalB64, setOriginalB64] = useState(null);
  const [detectResult, setDetectResult] = useState(null);
  const [heatmapB64, setHeatmapB64] = useState(null);
  const [reportMarkdown, setReportMarkdown] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);
  const [autoRunInitiated, setAutoRunInitiated] = useState(false);
  
  const [loading, setLoading] = useState({
    detect: false,
    heatmap: false,
    report: false,
    pdf: false
  });

  const [error, setError] = useState(null);

  useEffect(() => {
    if (!file) {
      navigate('/');
      return;
    }
    
    // Read the file for the original preview
    const reader = new FileReader();
    reader.onload = (ev) => {
      setOriginalB64(ev.target.result);
    };
    reader.readAsDataURL(file);

  }, [file, navigate]);

  useEffect(() => {
    if (file && !autoRunInitiated) {
      setAutoRunInitiated(true);
      runDetection();
      runHeatmap();
      runReport();
    }
  }, [file, autoRunInitiated]);

  const getRiskBadge = (risk) => {
    const r = risk?.toLowerCase() || 'low';
    const classes = {
      low: 'rgba(16, 185, 129, 0.15)',
      medium: 'rgba(245, 158, 11, 0.15)',
      high: 'rgba(239, 68, 68, 0.15)',
      critical: 'rgba(220, 38, 38, 0.2)'
    };
    const colors = { low: 'var(--success)', medium: 'var(--warning)', high: 'var(--danger)', critical: '#fca5a5' };
    
    return (
      <span style={{
        padding: '6px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase',
        background: classes[r], color: colors[r], border: `1px solid ${colors[r]}50`
      }}>
        {risk}
      </span>
    );
  };

  const runDetection = async () => {
    if (!file) return;
    setLoading(l => ({ ...l, detect: true }));
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setDetectResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(l => ({ ...l, detect: false }));
    }
  };

  const runHeatmap = async () => {
    if (!file) return;
    setLoading(l => ({ ...l, heatmap: true }));
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/eigencam?layer_index=-2`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setHeatmapB64(data.heatmap_b64);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(l => ({ ...l, heatmap: false }));
    }
  };

  const runReport = async () => {
    if (!file) return;
    setLoading(l => ({ ...l, report: true }));
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/report/generate`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setReportMarkdown(data.report);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(l => ({ ...l, report: false }));
    }
  };

  const runPdf = async () => {
    if (!file) return;
    setLoading(l => ({ ...l, pdf: true }));
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`${API_BASE}/api/pdf`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setPdfUrl(`${API_BASE}${data.pdf_url}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(l => ({ ...l, pdf: false }));
    }
  };

  if (!file) return null; // Prevent rendering while redirecting

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px', flex: 1 }}>
      {/* Left Panel */}
      <section className="card animate-slide-in">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Activity size={24} />
            Component Inspection
          </div>
          <button className="btn" onClick={() => navigate('/')} style={{ padding: '8px 12px', fontSize: '0.85rem' }}>
            <ArrowLeft size={16} /> New Upload
          </button>
        </div>

        {originalB64 && (
          <div className="animate-fade-in" style={{
            position: 'relative', borderRadius: '16px', overflow: 'hidden', background: '#000',
            border: '1px solid var(--border)'
          }}>
            <div style={{
              position: 'absolute', top: '16px', left: '16px', background: 'rgba(0,0,0,0.8)',
              padding: '6px 14px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--primary)',
              backdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.1)'
            }}>
              Original Image
            </div>
            <img src={originalB64} style={{ width: '100%', display: 'block', maxHeight: '400px', objectFit: 'contain' }} alt="Upload" />
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginTop: 'auto' }}>
          <button className="btn btn-primary" onClick={runDetection} disabled={loading.detect}>
            {loading.detect ? <div className="spinner" /> : <CheckCircle size={18} />}
            Detect Defects
          </button>
          <button className="btn" onClick={runHeatmap} disabled={loading.heatmap}>
            {loading.heatmap ? <div className="spinner" /> : <AlertTriangle size={18} />}
            Heatmap
          </button>
          <button className="btn" onClick={runReport} disabled={loading.report}>
            {loading.report ? <div className="spinner" /> : <FileText size={18} />}
            AI Report
          </button>
          <button className="btn" onClick={runPdf} disabled={loading.pdf}>
            {loading.pdf ? <div className="spinner" /> : <Download size={18} />}
            Export PDF
          </button>
        </div>
      </section>

      {/* Right Panel */}
      <section className="card animate-slide-in" style={{ animationDelay: '0.1s' }}>
        <div className="card-header">
          <Activity size={24} />
          Intelligence Hub
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {error && (
            <div style={{ padding: '20px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', borderRadius: '12px', color: '#fca5a5' }}>
              {error}
            </div>
          )}

          {!detectResult && !heatmapB64 && !reportMarkdown && !error && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', margin: 'auto', opacity: 0.6, padding: '40px' }}>
              <div className="spinner" style={{ margin: '0 auto 16px auto', width: '30px', height: '30px' }} />
              <p>Analyzing component...</p>
            </div>
          )}

          {detectResult && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ position: 'relative', borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border)' }}>
                <div style={{ position: 'absolute', top: '16px', left: '16px', background: 'rgba(0,0,0,0.8)', padding: '6px 14px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--primary)' }}>
                  Detection Map
                </div>
                <img src={`data:image/png;base64,${detectResult.annotated_b64 || detectResult.original_b64}`} style={{ width: '100%', display: 'block' }} alt="Detected" />
              </div>
              
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)', padding: '20px', borderRadius: '16px', textAlign: 'center' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Total Defects</div>
                  <div style={{ fontSize: '2rem', fontFamily: "'Outfit', sans-serif", fontWeight: 700 }}>{detectResult.total_defects}</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)', padding: '20px', borderRadius: '16px', textAlign: 'center' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '8px' }}>Highest Risk</div>
                  <div>{getRiskBadge(detectResult.highest_risk)}</div>
                </div>
              </div>

              {detectResult.detections.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {detectResult.detections.map((d, i) => (
                    <div key={i} style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border)', padding: '16px', borderRadius: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontWeight: 600, textTransform: 'capitalize', fontSize: '1.05rem' }}>{d.class}</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Conf: {(d.confidence * 100).toFixed(1)}% &bull; Area: {(d.area_ratio * 100).toFixed(1)}%</div>
                      </div>
                      {getRiskBadge(d.risk)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {heatmapB64 && (
            <div className="animate-fade-in" style={{ position: 'relative', borderRadius: '16px', overflow: 'hidden', border: '1px solid var(--border)' }}>
              <div style={{ position: 'absolute', top: '16px', left: '16px', background: 'rgba(0,0,0,0.8)', padding: '6px 14px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 600, color: 'var(--warning)' }}>
                EigenCAM Heatmap
              </div>
              <img src={`data:image/png;base64,${heatmapB64}`} style={{ width: '100%', display: 'block' }} alt="Heatmap" />
            </div>
          )}

          {reportMarkdown && (
            <div className="animate-fade-in" style={{ background: 'rgba(14, 165, 233, 0.03)', border: '1px solid rgba(14, 165, 233, 0.2)', borderRadius: '16px', padding: '24px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '4px', background: 'linear-gradient(90deg, var(--primary), var(--accent))' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px', color: 'var(--accent)', fontFamily: "'Outfit', sans-serif", fontWeight: 600, fontSize: '1.1rem' }}>
                <FileText size={20} /> Standard Technical Report
              </div>
              <div className="md-content" dangerouslySetInnerHTML={{ __html: marked(reportMarkdown) }} />
            </div>
          )}

          {pdfUrl && (
            <div className="animate-fade-in" style={{ padding: '20px', background: 'rgba(16, 185, 129, 0.1)', border: '1px dashed var(--success)', borderRadius: '12px', textAlign: 'center' }}>
              <div style={{ color: 'var(--success)', fontWeight: 600, marginBottom: '10px' }}>PDF Report Generated!</div>
              <a href={pdfUrl} download target="_blank" rel="noreferrer" className="btn btn-primary" style={{ display: 'inline-flex', width: 'auto' }}>
                <Download size={18} style={{ marginRight: '8px' }} />
                Download PDF
              </a>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
