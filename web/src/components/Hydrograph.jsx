import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, AlertTriangle, TrendingUp, ShieldCheck } from 'lucide-react';

export default function Hydrograph({ selectedSample }) {
  const [hydroData, setHydroData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchHydrograph();
  }, [selectedSample]);

  const fetchHydrograph = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/hydrograph', {
        seed: selectedSample?.seed || 101,
      });
      setHydroData(res.data);
    } catch (err) {
      console.error('Error fetching hydrograph data:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/60 flex items-center justify-between">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-2">
            <Activity className="w-3.5 h-3.5" /> Spatio-Temporal Hydrological Trajectory
          </div>
          <h2 className="text-2xl font-bold text-white">Temporal Inundation Hydrograph</h2>
          <p className="text-slate-300 text-sm mt-1">
            Tracking temporal flood progression, crest timeline, and surface water volume changes for {selectedSample?.name || 'Mekong Delta'}.
          </p>
        </div>

        {hydroData && (
          <div className="hidden md:flex items-center gap-4 font-mono text-xs text-right">
            <div className="bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-slate-400 block">PEAK WATER COVERAGE</span>
              <span className="text-lg font-bold text-cyan-400">{hydroData.peak_coverage_pct}%</span>
            </div>
            <div className="bg-slate-900/80 px-4 py-2 rounded-xl border border-slate-800">
              <span className="text-slate-400 block">PEAK CREST TIMELINE</span>
              <span className="text-lg font-bold text-rose-400">{hydroData.peak_day}</span>
            </div>
          </div>
        )}
      </div>

      {/* Main Hydrograph Recharts Visualization */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" /> Surface Water Area Progression (%) vs Time
          </h3>
          <span className="text-xs font-mono text-slate-400">Sentinel-1 Radar Sequence</span>
        </div>

        <div className="h-80 w-full pt-4">
          {hydroData?.time_series ? (
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hydroData.time_series} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="waterColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.6} />
                    <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#38bdf8', borderRadius: '8px', color: '#f8fafc' }}
                />
                <ReferenceLine y={15} label={{ value: 'Flood Risk Warning Threshold (15%)', fill: '#eab308', fontSize: 11 }} stroke="#eab308" strokeDasharray="4 4" />
                <ReferenceLine y={30} label={{ value: 'Critical Emergency Threshold (30%)', fill: '#f43f5e', fontSize: 11 }} stroke="#f43f5e" strokeDasharray="4 4" />
                <Area type="monotone" dataKey="water_coverage_pct" stroke="#38bdf8" strokeWidth={3} fillOpacity={1} fill="url(#waterColor)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-slate-400 font-mono text-sm">
              Loading Hydrological Time Series Chart...
            </div>
          )}
        </div>
      </div>

      {/* Flood Stage Breakdown Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel p-5 space-y-2 border-l-4 border-l-blue-500">
          <span className="text-xs font-mono text-slate-400">STAGE 1: PRE-FLOOD BASELINE</span>
          <h4 className="font-bold text-slate-100">Normal Channel Flow</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            SAR specular backscatter exhibits stable baseline returns (-22dB to -18dB) over natural river paths.
          </p>
        </div>

        <div className="glass-panel p-5 space-y-2 border-l-4 border-l-rose-500">
          <span className="text-xs font-mono text-rose-400 font-bold">STAGE 2: PEAK INUNDATION</span>
          <h4 className="font-bold text-slate-100">Bank Overtopping & Overflow</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Rapid decrease in radar backscatter as floodwaters submerge agricultural farmlands and surrounding terrain.
          </p>
        </div>

        <div className="glass-panel p-5 space-y-2 border-l-4 border-l-emerald-500">
          <span className="text-xs font-mono text-slate-400">STAGE 3: RECESSION & RECOVERY</span>
          <h4 className="font-bold text-slate-100">Water Subsidence</h4>
          <p className="text-xs text-slate-400 leading-relaxed">
            Gradual drying and water recession, tracked precisely across sequence time frame T3.
          </p>
        </div>
      </div>
    </div>
  );
}
