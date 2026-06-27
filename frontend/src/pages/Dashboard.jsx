import React from 'react';
import { UploadCloud } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Dashboard() {
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      // Navigate to the inspection view and pass the file via router state
      navigate('/inspection', { state: { file: selectedFile } });
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, minHeight: '60vh' }}>
      <section className="card animate-slide-in" style={{ width: '100%', maxWidth: '700px', textAlign: 'center', padding: '60px 40px' }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px', color: 'var(--primary)' }}>
          <UploadCloud size={64} strokeWidth={1.5} />
        </div>
        
        <h2 style={{ fontFamily: "'Outfit', sans-serif", fontSize: '2rem', marginBottom: '12px', fontWeight: 600 }}>
          New Component Inspection
        </h2>
        <p style={{ color: 'var(--text-muted)', marginBottom: '40px', fontSize: '1.1rem' }}>
          Upload an aerospace component image to automatically run YOLO defect detection, EigenCAM analysis, and generate an AI maintenance report.
        </p>

        <div style={{ position: 'relative', width: '100%', maxWidth: '500px', margin: '0 auto' }}>
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            style={{
              width: '100%', padding: '40px 20px', background: 'rgba(255,255,255,0.02)',
              border: '2px dashed rgba(255,255,255,0.2)', borderRadius: '16px',
              color: 'white', cursor: 'pointer', fontSize: '1.1rem',
              transition: 'all 0.3s ease'
            }} 
            onMouseOver={(e) => {
              e.currentTarget.style.borderColor = 'var(--primary)';
              e.currentTarget.style.background = 'rgba(59, 130, 246, 0.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)';
              e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
            }}
          />
        </div>
      </section>
    </div>
  );
}
