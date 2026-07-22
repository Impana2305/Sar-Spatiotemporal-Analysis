import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Eye, Zap, Layers, Activity, HelpCircle, BarChart2 } from 'lucide-react';

export default function ExplainableAI({ selectedSample, predictionData }) {
  const [xaiData, setXaiData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchExplainability();
  }, [selectedSample]);

  const fetchExplainability = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/explainability', {
        seed: selectedSample?.seed || 101,
      });
      setXaiData(res.data);
    } catch (err) {
      console.error('Error fetching XAI data:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/60 flex items-center justify-between">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-xs font-mono mb-2">
            <Eye className="w-3.5 h-3.5" /> Model Transparency & Trust Audit
          </div>
          <h2 className="text-2xl font-bold text-white">Explainable AI (XAI) Diagnostics</h2>
          <p className="text-slate-300 text-sm mt-1">
            Inspecting <strong>Grad-CAM Spatial Heatmaps</strong>, <strong>Temporal Self-Attention Weights</strong>, and <strong>SAR Change Detection Maps</strong>.
          </p>
        </div>

        <button
          onClick={fetchExplainability}
          disabled={loading}
          className="px-4 py-2 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 text-xs font-mono flex items-center gap-2 transition-all"
        >
          {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
          <span>Refresh XAI Gradient Maps</span>
        </button>
      </div>

      {/* Main Grid: 3 Visual XAI Columns */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Column 1: Grad-CAM Activation Map */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" /> Spatial Grad-CAM
            </h3>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              ConvAutoencoder Layer 5
            </span>
          </div>

          <div className="aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-950 relative">
            {xaiData ? (
              <img src={xaiData.gradcam_heatmap} alt="Grad-CAM" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-400 font-mono text-xs">
                Loading Grad-CAM...
              </div>
            )}
            <div className="absolute bottom-2 left-2 right-2 p-2 rounded bg-slate-950/80 backdrop-blur-sm text-[11px] text-slate-300 font-mono border border-slate-800">
              High intensity (Red/Yellow) indicates strong CNN neural activation for inundated surface water.
            </div>
          </div>
        </div>

        {/* Column 2: SAR Temporal Change Detection */}
        <div className="glass-panel p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" /> Temporal Change Δ (T2 - T1)
            </h3>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Backscatter Difference
            </span>
          </div>

          <div className="aspect-square rounded-xl overflow-hidden border border-slate-800 bg-slate-950 relative">
            {xaiData ? (
              <img src={xaiData.change_detection_heatmap} alt="Change Detection" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-slate-400 font-mono text-xs">
                Loading Change Detection...
              </div>
            )}
            <div className="absolute bottom-2 left-2 right-2 p-2 rounded bg-slate-950/80 backdrop-blur-sm text-[11px] text-slate-300 font-mono border border-slate-800">
              Highlights exact pixels experiencing flood inundation expansion between pre-event and peak flood.
            </div>
          </div>
        </div>

        {/* Column 3: Temporal Transformer Attention Weights */}
        <div className="glass-panel p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-purple-400" /> Self-Attention Weights
              </h3>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
                Multi-Head Attention
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {xaiData?.attention_weights?.map((item, idx) => (
                <div key={idx} className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 space-y-2">
                  <div className="flex justify-between font-bold text-slate-200">
                    <span>{item.time_step}</span>
                    <span className="text-purple-400">{(item.weight * 100).toFixed(0)}% Attention</span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full transition-all duration-500"
                      style={{ width: `${item.weight * 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-400 font-sans">{item.contribution}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs space-y-2">
            <div className="flex items-center gap-2 text-cyan-400 font-bold">
              <HelpCircle className="w-4 h-4" /> Why is this important?
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Standard black-box AI cannot explain its reasoning. Our XAI system proves the model focuses on physical SAR specular backscatter changes rather than spurious land artifacts.
            </p>
          </div>
        </div>
      </div>

      {/* Feature Salience Breakdown Table */}
      <div className="glass-panel p-5 space-y-4">
        <h3 className="text-base font-bold text-white">Feature Salience & Physical Predictors Importance</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {xaiData?.feature_salience?.map((feat, i) => (
            <div key={i} className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 font-mono text-xs">
              <span className="text-slate-400 block">{feat.feature}</span>
              <div className="flex items-end justify-between">
                <span className="text-xl font-extrabold text-cyan-400">{feat.importance}%</span>
                <span className="text-[10px] text-slate-400">Salience Weight</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-cyan-400 rounded-full"
                  style={{ width: `${feat.importance}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
