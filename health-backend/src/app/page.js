'use client'; // Required for LiveKit frontend state management

import React from 'react';
import { LiveKitRoom, useTranscript, useConnectionState } from '@livekit/components-react';
import { PhoneOff, Activity } from 'lucide-react';

// =====================================================================
// 🎨 LAYER 1: THE SCROLL-FREE MINIMALIST VOICE APP LAYOUT
// =====================================================================
function VoiceAppLayout({ state = 'listening', transcript = 'Listening for spoken regional text...', latency = 'Stable' }) {
  return (
    <div className="h-screen w-screen bg-slate-950 text-white flex flex-col justify-between p-6 overflow-hidden select-none">
      
      {/* 🟢 Top Bar: Connection Status Metrics */}
      <div className="flex justify-between items-center h-12">
        <div className="flex items-center gap-2 bg-slate-900/50 px-4 py-2 rounded-full border border-slate-900">
          <span className={`h-2.5 w-2.5 rounded-full animate-pulse ${
            state === 'speaking' ? 'bg-indigo-400' : state === 'listening' ? 'bg-emerald-500' : 'bg-amber-500'
          }`} />
          <span className="text-xs font-medium text-slate-400 tracking-wide uppercase">
            {state === 'speaking' ? 'Agent Speaking' : state === 'listening' ? 'AI Listening' : 'Connecting'} ({latency})
          </span>
        </div>
        
        <button className="flex items-center gap-2 px-4 py-2 bg-rose-950/30 hover:bg-rose-900/40 text-rose-400 hover:text-rose-300 rounded-full text-xs font-semibold border border-rose-900/50 transition-all active:scale-95">
          <PhoneOff size={14} />
          End Call
        </button>
      </div>

      {/* 🌀 Center Layer: Ambient Audio Canvas Core */}
      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="relative flex items-center justify-center h-72 w-72">
          
          {/* Animated Ripples */}
          <div className={`absolute inset-0 rounded-full animate-ping opacity-20 duration-1000 ${
            state === 'speaking' ? 'bg-violet-500' : 'bg-indigo-500'
          }`} />
          <div className={`absolute inset-6 rounded-full animate-pulse duration-700 opacity-40 ${
            state === 'speaking' ? 'bg-purple-500/30' : 'bg-violet-500/20'
          }`} />
          
          {/* Core Glowing Orb */}
          <div className={`h-36 w-36 bg-gradient-to-tr rounded-full flex items-center justify-center shadow-2xl transition-all duration-500 ${
            state === 'speaking' 
              ? 'from-purple-600 to-pink-600 shadow-purple-500/40 scale-105' 
              : 'from-indigo-600 to-violet-600 shadow-indigo-500/40'
          }`}>
            
            {/* Visualizer audio animation bounce bars */}
            <div className="flex gap-1.5 items-center justify-center h-12">
              <span className="w-1 bg-white/80 rounded-full animate-bounce h-6" style={{ animationDelay: '0.1s' }} />
              <span className="w-1 bg-white rounded-full animate-bounce h-10" style={{ animationDelay: '0.2s' }} />
              <span className="w-1 bg-white/80 rounded-full animate-bounce h-5" style={{ animationDelay: '0.3s' }} />
            </div>
          </div>
        </div>
        
        <p className="mt-6 text-sm font-medium text-indigo-200/70 tracking-widest uppercase animate-pulse">
          {state === 'speaking' ? 'Sharing Insights...' : state === 'listening' ? "Speak now, I'm listening" : 'Initializing Channels...'}
        </p>
      </div>

      {/* 📝 Bottom Layer: Live Partial Transcripts (Enforced height container) */}
      <div className="h-24 max-w-xl mx-auto w-full bg-slate-900/30 border border-slate-900/80 p-4 rounded-2xl backdrop-blur-xl flex flex-col justify-center shadow-inner">
        <div className="flex items-center gap-1.5 mb-1 opacity-40">
          <Activity size={10} className="text-slate-400" />
          <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Live Transcript</p>
        </div>
        <p className="text-sm text-slate-300 font-normal leading-relaxed line-clamp-2 pl-4 border-l border-indigo-500/30 italic">
          {transcript}
        </p>
      </div>
      
    </div>
  );
}

// =====================================================================
// 📡 LAYER 2: INTERACTIVE LIVEKIT SESSION STREAM CONTROLLER
// =====================================================================
function VoiceAgentSession() {
  const connectionState = useConnectionState();
  const { segments } = useTranscript();
  
  const currentTranscript = segments.length > 0 
    ? `"${segments[segments.length - 1].text}"` 
    : "Listening for spoken text analysis...";
    
  const derivedState = connectionState === 'connected' ? 'listening' : 'connecting';

  return (
    <VoiceAppLayout 
      state={derivedState} 
      transcript={currentTranscript}
      latency="Connected"
    />
  );
}

// =====================================================================
// 🚀 LAYER 3: MAIN PAGE ENTRY POINT RENDERER
// =====================================================================
export default function Home() {
  return (
    <LiveKitRoom 
      serverUrl="wss://health-bot-dquftlpx.livekit.cloud" 
      // 🎫 MAKE SURE TO REPLACE THIS PLAIN STRING WITH THE GENERATED JWT FROM YOUR TERMINAL:
      token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiTmFtaWJpYW4gUGF0aWVudCIsInZpZGVvIjp7InJvb21Kb2luIjp0cnVlLCJyb29tIjoiaGVhbHRoLXJvb20iLCJjYW5QdWJsaXNoIjp0cnVlLCJjYW5TdWJzY3JpYmUiOnRydWUsImNhblB1Ymxpc2hEYXRhIjp0cnVlfSwic3ViIjoid2ViLWNsaWVudC11c2VyIiwiaXNzIjoiQVBJNWk2aEd0eFd1akd3IiwibmJmIjoxNzgwNjAwMDIxLCJleHAiOjE3ODA2MjE2MjF9.LoYHv50Y2dpJyf7pn-AyCpNSuZfO4BambGa9ZFvlRcY" 
      audio={true}
      video={false} 
    >
      <VoiceAgentSession />
    </LiveKitRoom>
  );
}
