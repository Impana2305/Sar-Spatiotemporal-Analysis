import React, { useState } from 'react';
import { Upload, CheckCircle2, FileUp, Cpu, Activity, AlertCircle, Eye, Layers } from 'lucide-react';

export default function UploadPipeline({ runInference, handleFileUpload, loading, predictionData }) {
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [polarization, setPolarization] = useState('dual');

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

  const triggerPipeline = () => {
    if (uploadedFiles.length > 0) {
      handleFileUpload(uploadedFiles);
    } else {
      runInference();
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
            Upload custom temporal SAR images (GeoTIFF `.tif` or `.png`) for <strong>live PyTorch model inference</strong> and XAI generation.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Drop Zone (Left 2 cols) */}
        <div className="lg:col-span-2 glass-panel p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white">Upload SAR Raster Data</h3>
            <span className="text-xs font-mono text-slate-400">Accepted: GeoTIFF (.tif), PNG, JPG</span>
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
              accept=".tif,.tiff,.png,.jpg,.jpeg"
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
                  Upload 1 to 3 files representing $T_1$ (pre-flood), $T_2$ (peak flood), $T_3$ (post-flood)
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
                  <span className="text-cyan-300 font-bold">{f.name}</span>
                  <span className="text-slate-400">{(f.size / 1024).toFixed(1)} KB</span>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={triggerPipeline}
            disabled={loading}
            className="w-full py-3.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-sm shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
          >
            {loading ? <Activity className="w-5 h-5 animate-spin" /> : <Cpu className="w-5 h-5" />}
            <span>
              {loading
                ? 'Parsing File & Executing PyTorch Model...'
                : uploadedFiles.length > 0
                ? `Analyze ${uploadedFiles.length} Uploaded Image File(s) with PyTorch`
                : 'Run Benchmark Model & XAI Pipeline'}
            </span>
          </button>

          {/* Uploaded Results Live Display */}
          {predictionData && predictionData.uploaded_filename && (
            <div className="p-4 rounded-xl bg-cyan-950/40 border border-cyan-500/40 space-y-4">
              <div className="flex items-center justify-between border-b border-cyan-500/30 pb-2">
                <span className="text-xs font-mono text-cyan-300 font-bold flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" /> PyTorch Inference Complete for {predictionData.uploaded_filename}
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
                  {predictionData.risk_level} RISK ({predictionData.water_area_pct}%)
                </span>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <span className="text-[11px] font-mono text-slate-400">Parsed SAR Input</span>
                  <img src={predictionData.visualizations.sar_t1} alt="Uploaded SAR" className="w-full rounded border border-slate-800" />
                </div>
                <div className="space-y-1">
                  <span className="text-[11px] font-mono text-cyan-400">Predicted Flood Mask</span>
                  <img src={predictionData.visualizations.prediction_mask} alt="Predicted Mask" className="w-full rounded border border-cyan-500/40" />
                </div>
                <div className="space-y-1">
                  <span className="text-[11px] font-mono text-amber-400">Grad-CAM XAI Map</span>
                  <img src={predictionData.xai?.gradcam_heatmap || predictionData.visualizations.binary_mask} alt="GradCAM" className="w-full rounded border border-amber-500/40" />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Info Box */}
        <div className="glass-panel p-6 space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <h3 className="text-base font-bold text-white">Live Execution Pipeline</h3>
            
            <div className="space-y-3 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">1. REAL FILE PARSING</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Extracts pixel backscatter intensities from uploaded GeoTIFF/PNG files, resizes to 256x256, and normalizes VV/VH bands.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">2. CONVAUTOENCODER LATENT</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Pre-trained ConvAutoencoder extracts 16,384-dimensional spatial features for each temporal frame.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">3. TEMPORAL TRANSFORMER</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Temporal Transformer Encoder processes sequence dependencies and outputs the binary flood prediction mask.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                <span className="text-cyan-400 font-bold">4. DYNAMIC GRAD-CAM XAI</span>
                <p className="text-slate-400 text-[11px] font-sans">
                  Computes backpropagation gradients w.r.t uploaded image pixels to build real-time spatial activation heatmaps.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
