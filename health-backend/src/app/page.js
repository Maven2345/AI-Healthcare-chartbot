import React, { useState } from 'react';

export default function PredictWidget() {
  const [symptoms, setSymptoms] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDiagnosticSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Convert comma-separated string inputs into an array
    const symptomArray = symptoms.split(',').map(s => s.trim()).filter(s => s.length > 0);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: symptomArray }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze metrics.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto my-10 p-6 bg-white dark:bg-slate-900 rounded-2xl shadow-xl border border-slate-100 dark:border-slate-800">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-white mb-2">AI Diagnostic Portal</h2>
      <p className="text-slate-500 dark:text-slate-400 mb-6 text-sm">Enter your current physical manifestations separated by commas (e.g., itching, skin rash, joint pain).</p>

      <form onSubmit={handleDiagnosticSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">Symptoms Spectrum</label>
          <input
            type="text"
            className="w-full px-4 py-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-transparent text-slate-800 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all"
            placeholder="fever, chills, muscle_wasting"
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            required
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-700 hover:to-violet-700 text-white font-medium rounded-xl shadow-lg shadow-indigo-200 dark:shadow-none transition-all disabled:opacity-50"
        >
          {loading ? 'Analyzing Clinical Indicators...' : 'Run Diagnostics'}
        </button>
      </form>

      {error && (
        <div className="mt-6 p-4 bg-rose-50 dark:bg-rose-950/30 border border-rose-100 dark:border-rose-900/50 rounded-xl text-rose-600 dark:text-rose-400 text-sm">
          ⚠️ <strong>Analysis Stopped:</strong> {error}
        </div>
      )}

      {result && (
        <div className="mt-8 p-6 bg-slate-50 dark:bg-slate-800/50 rounded-xl border border-slate-100 dark:border-slate-800 animate-fadeIn">
          <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">Target Condition Match</span>
          <h3 className="text-2xl font-bold text-slate-950 dark:text-white mt-1 mb-3">{result.prognosis}</h3>
          
          <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed mb-4">{result.description}</p>
          
          <div className="border-t border-slate-200 dark:border-slate-700 pt-4">
            <h4 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">Recommended Precautions</h4>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {result.precautions.map((precaution, idx) => (
                <li key={idx} className="text-xs text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 p-2 rounded-lg border border-slate-100 dark:border-slate-700 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 flex-shrink-0" />
                  {precaution}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
