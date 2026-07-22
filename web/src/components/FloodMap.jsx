import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, LayersControl } from 'react-leaflet';
import L from 'leaflet';
import { Layers, Sliders, MapPin, AlertTriangle, Activity } from 'lucide-react';

// Fix Leaflet default icon paths in React
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

export default function FloodMap({ selectedSample, predictionData, samples, setSelectedSample }) {
  const [activeStep, setActiveStep] = useState(1); // 0: T1, 1: T2, 2: T3
  const [opacity, setOpacity] = useState(0.75);
  const [mapMode, setMapMode] = useState('probability'); // 'probability' | 'binary' | 'sar'

  const center = selectedSample ? selectedSample.coordinates : [10.2833, 105.9667];

  return (
    <div className="space-y-6">
      {/* Header controls bar */}
      <div className="glass-panel p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">GIS Interactive Sentinel-1 Flood Overlay</h2>
            <p className="text-xs text-slate-400">
              Coordinates: <span className="font-mono text-cyan-300">{center[0].toFixed(4)}°N, {center[1].toFixed(4)}°E</span>
            </p>
          </div>
        </div>

        {/* Map View Mode Toggles */}
        <div className="flex items-center gap-2 bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setMapMode('probability')}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              mapMode === 'probability' ? 'bg-cyan-500 text-slate-950 font-bold shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Probability Heatmap
          </button>
          <button
            onClick={() => setMapMode('binary')}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              mapMode === 'binary' ? 'bg-blue-500 text-white font-bold shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Binary Water Mask
          </button>
          <button
            onClick={() => setMapMode('sar')}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              mapMode === 'sar' ? 'bg-purple-500 text-white font-bold shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            SAR VV/VH Composite
          </button>
        </div>

        {/* Opacity slider */}
        <div className="flex items-center gap-3 text-xs font-mono text-slate-300">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span>Overlay Opacity: {(opacity * 100).toFixed(0)}%</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-24 accent-cyan-400 cursor-pointer"
          />
        </div>
      </div>

      {/* Main Map Container & Right Overlay Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Leaflet Map (Left 3 cols) */}
        <div className="lg:col-span-3 glass-panel p-2 relative h-[550px] overflow-hidden rounded-2xl">
          <MapContainer
            center={center}
            zoom={8}
            scrollWheelZoom={true}
            style={{ height: '100%', width: '100%', borderRadius: '12px' }}
          >
            {/* Esri World Imagery / OpenStreetMap base */}
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {/* Region Markers */}
            {samples.map((s) => (
              <Marker
                key={s.id}
                position={s.coordinates}
                eventHandlers={{
                  click: () => setSelectedSample(s),
                }}
              >
                <Popup className="bg-slate-900 text-slate-100">
                  <div className="p-1">
                    <strong className="text-sm font-bold text-cyan-400">{s.name}</strong>
                    <p className="text-xs text-slate-300 mt-1">{s.description}</p>
                  </div>
                </Popup>
              </Marker>
            ))}

            {/* Simulated Flood Overlay Circle */}
            <Circle
              center={center}
              radius={25000}
              pathOptions={{
                color: mapMode === 'probability' ? '#06b6d4' : mapMode === 'binary' ? '#2563eb' : '#a855f7',
                fillColor: mapMode === 'probability' ? '#0891b2' : mapMode === 'binary' ? '#3b82f6' : '#c084fc',
                fillOpacity: opacity,
                weight: 2,
              }}
            />
          </MapContainer>

          {/* Floating Time Slider Overlay on Map */}
          <div className="absolute bottom-6 left-6 right-6 z-[1000] glass-panel p-3 bg-slate-950/85 backdrop-blur-md border border-cyan-500/30 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
              <span className="text-xs font-mono text-cyan-300 font-semibold">TEMPORAL TIMELINE:</span>
            </div>
            
            <div className="flex items-center gap-4 text-xs font-mono">
              {['T1: Pre-Event', 'T2: Peak Flood', 'T3: Post-Event'].map((label, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveStep(idx)}
                  className={`px-3 py-1.5 rounded-lg border transition-all ${
                    activeStep === idx
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/60 font-bold'
                      : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Telemetry Controls */}
        <div className="glass-panel p-5 space-y-5 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <MapPin className="w-4 h-4 text-cyan-400" /> Geographic Info
              </h3>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block mb-1">TARGET LOCATION</span>
                <span className="text-sm font-bold text-slate-100">{selectedSample?.name || 'Mekong Delta'}</span>
              </div>

              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block mb-1">ACQUISITION DATES</span>
                <span className="text-slate-200 block">T1: {selectedSample?.date_t1}</span>
                <span className="text-cyan-300 font-bold block">T2: {selectedSample?.date_t2}</span>
                <span className="text-slate-200 block">T3: {selectedSample?.date_t3}</span>
              </div>

              {predictionData && (
                <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-2">
                  <span className="text-slate-400 block">PREDICTED METRICS</span>
                  <div className="flex justify-between">
                    <span>Water Coverage:</span>
                    <span className="text-cyan-400 font-bold">{predictionData.water_area_pct}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Flood Risk:</span>
                    <span className={`font-bold ${predictionData.risk_level === 'CRITICAL' ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {predictionData.risk_level}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="text-emerald-400 font-bold">{predictionData.metrics.confidence}%</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-cyan-950/30 border border-cyan-500/20 text-xs text-slate-300 space-y-1">
            <div className="flex items-center gap-1.5 font-bold text-cyan-400">
              <AlertTriangle className="w-3.5 h-3.5" /> Sentinel-1 SAR Note
            </div>
            <p className="text-[11px] leading-relaxed text-slate-400">
              Synthetic Aperture Radar penetrates cloud cover and darkness. Dark backscatter regions represent smooth open water surfaces.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
