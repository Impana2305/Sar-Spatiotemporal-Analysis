import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart as RePie, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { PieChart, Trees, Building2, Droplets, Mountain } from 'lucide-react';

export default function LandUse({ selectedSample }) {
  const [landData, setLandData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchLandUse();
  }, [selectedSample]);

  const fetchLandUse = async () => {
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/api/landuse', {
        seed: selectedSample?.seed || 101,
      });
      setLandData(res.data);
    } catch (err) {
      console.error('Error fetching land use data:', err);
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#3b82f6', '#22c55e', '#ef4444', '#eab308'];

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-slate-900 via-slate-900 to-blue-950/60 flex items-center justify-between">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-mono mb-2">
            <PieChart className="w-3.5 h-3.5" /> Multispectral & SAR Terrain Categorization
          </div>
          <h2 className="text-2xl font-bold text-white">Land Use / Land Cover (LULC) Classification</h2>
          <p className="text-slate-300 text-sm mt-1">
            Analyzing terrain cover distribution and flood exposure for <strong>{selectedSample?.name || 'Mekong Delta'}</strong>.
          </p>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart Card */}
        <div className="glass-panel p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white">Terrain Cover Distribution (%)</h3>
            <span className="text-xs font-mono text-slate-400">Random Forest Classifier</span>
          </div>

          <div className="h-72 w-full">
            {landData?.land_use_distribution ? (
              <ResponsiveContainer width="100%" height="100%">
                <RePie>
                  <Pie
                    data={landData.land_use_distribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="percentage"
                    nameKey="name"
                  >
                    {landData.land_use_distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color || COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#38bdf8', borderRadius: '8px', color: '#f8fafc' }}
                  />
                  <Legend />
                </RePie>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 font-mono text-sm">
                Loading Land Cover Distribution...
              </div>
            )}
          </div>
        </div>

        {/* Category Breakdown Cards */}
        <div className="space-y-4">
          {landData?.land_use_distribution?.map((item, idx) => {
            const icons = [Droplets, Trees, Building2, Mountain];
            const Icon = icons[idx % icons.length];
            return (
              <div key={idx} className="glass-panel p-4 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className="p-3 rounded-xl flex items-center justify-center text-white"
                    style={{ backgroundColor: `${item.color}20`, border: `1px solid ${item.color}50` }}
                  >
                    <Icon className="w-5 h-5" style={{ color: item.color }} />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-slate-100">{item.name}</h4>
                    <p className="text-xs text-slate-400">Surface Coverage Share</p>
                  </div>
                </div>

                <div className="text-right font-mono">
                  <span className="text-2xl font-extrabold" style={{ color: item.color }}>
                    {item.percentage}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
