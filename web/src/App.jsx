import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import FloodMap from './components/FloodMap';
import ExplainableAI from './components/ExplainableAI';
import Hydrograph from './components/Hydrograph';
import LandUse from './components/LandUse';
import UploadPipeline from './components/UploadPipeline';

const API_BASE = 'http://localhost:5000/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [healthInfo, setHealthInfo] = useState(null);
  const [stats, setStats] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [hRes, sRes, sampleRes] = await Promise.all([
        axios.get(`${API_BASE}/health`),
        axios.get(`${API_BASE}/stats`),
        axios.get(`${API_BASE}/samples`),
      ]);
      setHealthInfo(hRes.data);
      setStats(sRes.data);
      setSamples(sampleRes.data);
      if (sampleRes.data.length > 0) {
        setSelectedSample(sampleRes.data[0]);
      }
    } catch (err) {
      console.warn('Backend API connection failed, falling back to mock state:', err);
      setHealthInfo({ status: 'offline', device: 'CPU' });
      setSamples([
        {
          id: 'mekong_delta_2020',
          name: 'Mekong Delta, Vietnam',
          coordinates: [10.2833, 105.9667],
          date_t1: '2020-09-12 (Pre-Flood)',
          date_t2: '2020-10-18 (Peak Monsoon)',
          date_t3: '2020-11-04 (Receding)',
          region: 'Southeast Asia',
          description: 'Monsoon overflow along major agricultural delta system.',
          seed: 101
        },
        {
          id: 'rhine_basin_2021',
          name: 'Rhine River Basin, Germany',
          coordinates: [50.3569, 7.5890],
          date_t1: '2021-07-02 (Pre-Event)',
          date_t2: '2021-07-15 (Flash Flood)',
          date_t3: '2021-07-28 (Post-Event)',
          region: 'Central Europe',
          description: 'Extreme rainfall event triggering riverbank bursts.',
          seed: 202
        }
      ]);
    }
  };

  useEffect(() => {
    if (selectedSample) {
      runInference();
    }
  }, [selectedSample]);

  const runInference = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        seed: selectedSample?.seed || 101,
      });
      setPredictionData(res.data);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} healthInfo={healthInfo} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {activeTab === 'dashboard' && (
          <Dashboard
            stats={stats}
            samples={samples}
            selectedSample={selectedSample}
            setSelectedSample={setSelectedSample}
            predictionData={predictionData}
            runInference={runInference}
            loading={loading}
          />
        )}

        {activeTab === 'map' && (
          <FloodMap
            selectedSample={selectedSample}
            predictionData={predictionData}
            samples={samples}
            setSelectedSample={setSelectedSample}
          />
        )}

        {activeTab === 'xai' && (
          <ExplainableAI selectedSample={selectedSample} predictionData={predictionData} />
        )}

        {activeTab === 'hydrograph' && (
          <Hydrograph selectedSample={selectedSample} />
        )}

        {activeTab === 'landuse' && (
          <LandUse selectedSample={selectedSample} />
        )}

        {activeTab === 'upload' && (
          <UploadPipeline runInference={runInference} loading={loading} predictionData={predictionData} />
        )}
      </main>

      <footer className="glass-header border-t border-slate-800/80 py-4 text-center text-xs text-slate-400 font-mono">
        <p>GEO-SAR System Architecture — Spatio-Temporal ConvAutoencoder + Transformer Flood Monitoring</p>
      </footer>
    </div>
  );
}
