import React from 'react';
import { Activity, ShieldAlert, Award, Database, Layers, ArrowRight, Zap, CheckCircle2, Eye, TrendingUp } from 'lucide-react';

export default function Dashboard({ stats, samples, selectedSample, setSelectedSample, predictionData, runInference, loading, setActiveTab }) {
  const metrics = stats?.model_metrics || {
    accuracy: 0.942,
    dice: 0.915,
    iou: 0.843,
    precision: 0.921,
    recall: 0.908
  };

  return (
    <div className="space-y-6">
      {/* Upload Success Alert Banner */}
      {predictionData && predictionData.uploaded_filename && (
        <div className="glass-panel p-4 bg-cyan-950/60 border border-cyan-500/50 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            <div>
              <h4 className="text-sm font-bold text-white font-mono">
                Live Assessment for Uploaded Image: <span className="text-cyan-400">{predictionData.uploaded_filename}</span>
              </h4>
              <p className="text-xs text-slate-300">
                PyTorch ConvAutoencoder + Transformer inference completed ({predictionData.water_area_pct}% surface water detected).
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono">
            <button
              onClick={() => setActiveTab && setActiveTab('map')}
              className="px-3 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 flex items-center gap-1.5"
            >
              <Layers className="w-3.5 h-3.5" /> GIS Map
            </button>
            <button
              onClick={() => setActiveTab && setActiveTab('xai')}
              className="px-3 py-1.5 rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/30 flex items-center gap-1.5"
            >
              <Eye className="w-3.5 h-3.5" /> Grad-CAM XAI
            </button>
            <button
              onClick={() => setActiveTab && setActiveTab('hydrograph')}
              className="px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 border border-purple-500/40 hover:bg-purple-500/30 flex items-center gap-1.5"
            >
              <TrendingUp className="w-3.5 h-3.5" /> Hydrograph
            </button>
          </div>
        </div>
      )}

      {/* Hero Banner */}
      <div className="glass-panel p-6 radar-scan-container bg-gradient-to-r from-slate-900/90 via-slate-900/60 to-cyan-950/40 relative overflow-hidden">
        <div className="radar-scan-line" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-3">
              <Zap className="w-3.5 h-3.5" /> Synthetic Aperture Radar (SAR) Deep Learning Pipeline
            </div>
            <h2 className="text-3xl font-extrabold tracking-tight text-white mb-2">
              Spatio-Temporal Sentinel-1 Flood Prediction System
            </h2>
            <p className="text-slate-300 text-sm max-w-2xl leading-relaxed">
              Integrating <strong>Convolutional Autoencoders</strong> for spatial feature extraction with a <strong>Temporal Transformer Encoder</strong> to perform end-to-end multi-temporal SAR flood detection and explainable risk mapping.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={runInference}
              disabled={loading}
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-sm shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Activity className="w-4 h-4" />
              )}
              <span>{loading ? 'Running PyTorch Model...' : 'Execute Model Inference'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>DICE COEFFICIENT</span>
            <Award className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-extrabold text-cyan-400 font-mono">
            {(metrics.dice * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">Overlapped segmentation accuracy</p>
        </div>

        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>INTERSECTION (IoU)</span>
            <Layers className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-extrabold text-blue-400 font-mono">
            {(metrics.iou * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">Jaccard index score</p>
        </div>

        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>ACCURACY RATE</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-extrabold text-emerald-400 font-mono">
            {(metrics.accuracy * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">Overall classification score</p>
        </div>

        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>PRECISION</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-extrabold text-indigo-400 font-mono">
            {(metrics.precision * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400 mt-1">Low false alarm rate</p>
        </div>

        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>TEMPORAL STEPS</span>
            <Database className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-extrabold text-purple-400 font-mono">
            T = 3 (T1,T2,T3)
          </div>
          <p className="text-xs text-slate-400 mt-1">Sequential SAR backscatter</p>
        </div>

        <div className="glass-panel p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono mb-2">
            <span>LATENT DIMENSION</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-extrabold text-amber-400 font-mono">
            16,384
          </div>
          <p className="text-xs text-slate-400 mt-1">Autoencoder feature map</p>
        </div>
      </div>

      {/* Main Grid: Sample Selector & Prediction Output */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Sample Locations */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-cyan-400" /> Select Target SAR Region
            </h3>
            <span className="text-xs text-slate-400 font-mono">{samples.length} Benchmark Scenes</span>
          </div>

          <div className="space-y-3">
            {samples.map((s) => {
              const isSelected = selectedSample?.id === s.id;
              return (
                <div
                  key={s.id}
                  onClick={() => setSelectedSample(s)}
                  className={`p-4 rounded-xl cursor-pointer transition-all border ${
                    isSelected
                      ? 'bg-cyan-950/40 border-cyan-500/50 shadow-md shadow-cyan-500/10'
                      : 'bg-slate-900/50 border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <h4 className="font-semibold text-sm text-slate-100">{s.name}</h4>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
                      {s.region}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1 line-clamp-2">{s.description}</p>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono mt-3 pt-2 border-t border-slate-800/60">
                    <span>{s.date_t2}</span>
                    <span className="text-slate-400">{s.coordinates[0]}°, {s.coordinates[1]}°</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right 2 Columns: Live Model Output Summary */}
        <div className="lg:col-span-2 glass-panel p-5 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-bold text-white">Current Region Assessment</h3>
              <p className="text-xs text-slate-400">{selectedSample?.name || 'Mekong Delta'}</p>
            </div>
            {predictionData && (
              <div className={`px-3 py-1 rounded-full font-mono text-xs font-bold border ${
                predictionData.risk_level === 'CRITICAL'
                  ? 'bg-rose-500/20 text-rose-400 border-rose-500/40 animate-pulse'
                  : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
              }`}>
                {predictionData.risk_level} FLOOD RISK
              </div>
            )}
          </div>

          {predictionData ? (
            <div className="space-y-5">
              {/* Image previews */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <span className="text-xs font-mono text-slate-400">SAR T1 (Pre-Flood)</span>
                  <div className="rounded-lg overflow-hidden border border-slate-800 aspect-square bg-slate-950">
                    <img src={predictionData.visualizations.sar_t1} alt="T1 SAR" className="w-full h-full object-cover" />
                  </div>
                </div>
                <div className="space-y-2">
                  <span className="text-xs font-mono text-cyan-400 font-medium">SAR T2 (Peak Flood)</span>
                  <div className="rounded-lg overflow-hidden border border-cyan-500/40 aspect-square bg-slate-950 shadow-lg shadow-cyan-500/10">
                    <img src={predictionData.visualizations.sar_t2} alt="T2 SAR" className="w-full h-full object-cover" />
                  </div>
                </div>
                <div className="space-y-2">
                  <span className="text-xs font-mono text-slate-400">Predicted Water Mask</span>
                  <div className="rounded-lg overflow-hidden border border-blue-500/40 aspect-square bg-slate-950">
                    <img src={predictionData.visualizations.prediction_mask} alt="Prediction Mask" className="w-full h-full object-cover" />
                  </div>
                </div>
              </div>

              {/* Assessment Stats */}
              <div className="grid grid-cols-3 gap-4 p-4 rounded-xl bg-slate-900/80 border border-slate-800 font-mono text-xs">
                <div>
                  <span className="text-slate-400 block mb-1">INUNDATED WATER AREA</span>
                  <span className="text-lg font-bold text-cyan-400">{predictionData.water_area_pct}%</span>
                </div>
                <div>
                  <span className="text-slate-400 block mb-1">PREDICTION CONFIDENCE</span>
                  <div className="flex items-center space-x-2">
  <div className="w-32 bg-slate-800 rounded-full h-4 overflow-hidden">
    <div className={`${predictionData.metrics.confidence >= 80 ? 'bg-emerald-500' : 'bg-amber-500'} transition-width duration-500`} style={{ width: `${predictionData.metrics.confidence}%` }}></div>
  </div>
  <span className="text-sm font-bold text-emerald-400">{predictionData.metrics.confidence}%</span>
</div>
                </div>
                <div>
                  <span className="text-slate-400 block mb-1">BATCH SEGMENTATION IoU</span>
                  <span className="text-lg font-bold text-blue-400">{predictionData.metrics.iou}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-slate-400 space-y-3">
              <Activity className="w-8 h-8 text-cyan-400/40 animate-bounce" />
              <p className="text-sm">Click "Execute Model Inference" to generate real-time flood predictions.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
