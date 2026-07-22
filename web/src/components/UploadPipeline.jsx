import React, { useState } from 'react';
import { Upload, CheckCircle2, FileUp, Cpu, Activity, AlertCircle } from 'lucide-react';

export default function UploadPipeline({ runInference, loading, predictionData }) {
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [polarization, setPolarization] = useState('dual'); // 'vv' | 'vh' | 'dual'

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      setUploadedFiles(files);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);
      setUploadedFiles(files);
    }
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/60 flex items-center justify-between">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono mb-2">
            <Upload className="w-3.5 h-3.5" /> Sentinel-1 GeoTIFF / PNG Ingestion
          </div>
          <h2 className="text-2xl font-bold text-white">Custom SAR Image Sequence Pipeline</h2>
          <p className="text-slate-300 text-sm mt-1">
            Upload custom 3-step temporal SAR sequences ($T_1, T_2, T_3$) for real-time model inference and XAI generation.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Drop Zone (Left 2 cols) */}
        <div className="lg:col-span-2 glass-panel p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white">Upload SAR Raster Data</h3>
            <span className="text-xs font-mono text-slate-400">Accepted: GeoTIFF, PNG, NPY, NPK</span>
          </div>

          {/* Drag and Drop Zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleFileDrop}
            className={`p-10 rounded-2xl border-2 border-dashed transition-all flex flex-col items-center justify-center text-center cursor-pointer ${
              dragOver
                ? 'border-cyan-400 bg-cyan-500/10'
                : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/80'
            }`}
          >
            <input
              type="file"
              multiple
              onChange={handleFileSelect}
              className="hidden"
              id="sar-file-input"
            />
            <label htmlFor="sar-file-input" className="cursor-pointer space-y-3">
              <div className="p-4 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 mx-auto w-fit">
                <FileUp className="w-8 h-8" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-200">
                  Drag and drop temporal SAR imagery here, or <span className="text-cyan-400 underline">browse files</span>
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  Upload 3 files representing $T_1$ (pre-flood), $T_2$ (peak flood), $T_3$ (post-flood)
                </p>
              </div>
            </label>
          </div>

          {/* Polarization selection */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono">
            <span className="text-slate-300">POLARIZATION BANDS:</span>
            <div className="flex gap-2">
              {[
                { id: 'dual', label: 'Dual Pol (VV + VH)' },
                { id: 'vv', label: 'VV Only' },
                { id: 'vh', label: 'VH Only' }
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => setPolarization(p.id)}
                  className={`px-3 py-1.5 rounded-lg border transition-all ${
                    polarization === p.id
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-bold'
                      : 'bg-slate-900 text-slate-400 border-slate-800'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* File list */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-2 font-mono text-xs">
              <span className="text-slate-400 block">SELECTED FILES ({uploadedFiles.length}):</span>
              {uploadedFiles.map((f, idx) => (
                <div key={idx} className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between">
                  <span className="text-slate-200">{f.name}</span>
                  <span className="text-slate-400">{(f.size / 1024).toFixed(1)} KB</span>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={runInference}
            disabled={loading}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-sm shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {loading ? <Activity className="w-5 h-5 animate-spin" /> : <Cpu className="w-5 h-5" />}
            <span>{loading ? 'Processing Neural Model Pipeline...' : 'Run Full Model & XAI Pipeline'}</span>
          </button>
        </div>

        {/* Right Info Box */}
        <div className="glass-panel p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white">Pipeline Execution Steps</h3>
            
            <div className="space-y-3 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">1. PREPROCESSING</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Median filter speckle reduction, min-max log backscatter normalization, resampling to 256x256 tensor.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">2. SPATIAL ENCODER</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  5-layer ConvAutoencoder generates 256x8x8 bottleneck spatial feature vectors for each time step.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">3. TEMPORAL TRANSFORMER</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  2-layer Transformer Encoder with positional encoding captures temporal inundation kinetics across T1-T3.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">4. XAI EXPLANATION</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Grad-CAM backpropagation computes spatial activation heatmaps and temporal self-attention weights.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
