import React from 'react';
// 1. Import the new widget from its sibling path
import PredictWidget from './PredictWidget'; 

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 p-6">
      {/* Existing voice elements can stay here... */}
      
      {/* 2. Place the PredictWidget wherever you want it to appear */}
      <PredictWidget />
    </main>
  );
}
