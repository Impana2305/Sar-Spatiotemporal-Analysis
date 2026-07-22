import React from 'react';
import { Radar, Layers, Eye, Activity, PieChart, Upload, Cpu } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthInfo }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Radar },
    { id: 'upload', label: 'Upload SAR Raster', icon: Upload },
    { id: 'map', label: 'GIS Flood Prediction Map', icon: Layers },
    { id: 'xai', label: 'Explainable AI (XAI)', icon: Eye },
    { id: 'hydrograph', label: 'Temporal Hydrograph', icon: Activity },
    { id: 'landuse', label: 'Land Use Classification', icon: PieChart },
  ];

  return (
    <header className="glass-header sticky top-0 z-50 px-8 py-4 flex flex-wrap items-center justify-between shadow-lg">
      <div className="flex items-center gap-4">
        <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center shadow-inner">
          <Radar className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
              GEO-SAR
            </h1>
            
          </div>
          <p className="text-xs text-slate-400">Spatio-Temporal Transformer & Autoencoder Flood Analytics</p>
        </div>
      </div>

      {/* Tabs */}
      <nav className="flex items-center gap-1.5 my-2 sm:my-0">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Status Badge */}
      <div className="flex items-center gap-3 text-xs font-mono">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/80 border border-slate-700/60">
          <span className={`w-2 h-2 rounded-full ${healthInfo?.status === 'online' ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
          <span className="text-slate-300">
            {healthInfo?.status === 'online' ? 'API CONNECTED' : 'OFFLINE MODE'}
          </span>
        </div>
        {healthInfo?.device && (
          <div className="hidden lg:flex items-center gap-1.5 text-slate-400 bg-slate-900/60 px-2.5 py-1 rounded-md border border-slate-800">
            <Cpu className="w-3.5 h-3.5" />
            <span>{healthInfo.device.toUpperCase()}</span>
          </div>
        )}
      </div>
    </header>
  );
}
