import React, { useState, useEffect, useRef } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
  Cpu, Activity, Zap, Layers, Terminal, Info, RefreshCw, Box, MessageSquare,
  LineChart as ChartIcon, Send, Clock, Save, HardDrive, Monitor, Database
} from 'lucide-react';

interface DataPoint {
  x: number;
  y: number;
}

interface Stats {
  batch: number;
  totalBatches: number;
  loss: number;
  ppl: number;
  lr: number;
  speed: number;
  tokensPerSec: number;
  paramCount: number;
  usedMemory: number;
  maxMemory: number;
  os: string;
  cpu: string;
  javaVersion: string;
  startTime: number;
  lastSave: string;
  dModel: number;
  layers: number;
  heads: number;
  seqLen: number;
  lossHistory: DataPoint[];
  lrHistory: DataPoint[];
  status: string;
}

interface Message {
  role: 'user' | 'bot';
  text: string;
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'training' | 'chat'>('training');
  const [stats, setStats] = useState<Stats | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [loadingDots,setLoadingDots]=useState<String>('...')


  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:7070/api/stats');
        const data: Stats = await response.json();
        setStats(data);
      } catch (error) {
        console.error("Dashboard failed to connect to Transformer Backend:", error);
      }
    };

    const interval = setInterval(fetchStats, 100); // 10Hz updates for memory gauge
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  

  const handleSend = async () => {
    if (!input.trim() || isTyping) return;
    const userMsg: Message = { role: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:7070/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'bot', text: data.response || "No response." }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: "Error connecting to model server." }]);
    } finally {
      setIsTyping(false);
    }
  };

  if (!stats) {
    return (
      <div className="dashboard-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div style={{ textAlign: 'center' }}>
          <RefreshCw className="animate-spin" size={48} color="#00d2ff" />
          <p style={{ marginTop: '1rem', color: '#00d2ff', fontWeight: 500 }}>Connecting to Dream  Engine{loadingDots}</p>
        </div>
      </div>
    );
  }

  const progress = (stats.batch / stats.totalBatches) * 100;
  const memProgress = (stats.usedMemory / stats.maxMemory) * 100;

  return (
    <div className="dashboard-container">
      <header>
        <div className="logo">
          <Box size={28} />
          DREAM ENGINE 
        </div>
        <div className="status-badge">
          <div className="pulse" />
          {stats.status}
        </div>
      </header>

      <div className="tab-nav">
        <button className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`} onClick={() => setActiveTab('training')}>
          <ChartIcon size={18} /> Training Logs
        </button>
        <button className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>
          <MessageSquare size={18} /> Chat Interface
        </button>
      </div>

      {activeTab === 'training' ? (
        <div className="training-view">
          {/* Main Stats Grid */}
          <div className="dashboard-grid">
            <div className="info-card">
              <div className="info-label"><Activity size={14} /> Batch Progress</div>
              <div className="info-value">{stats.batch.toLocaleString()} / {stats.totalBatches.toLocaleString()}</div>
              <div className="progress-container">
                <div className="progress-fill batch-bar" style={{ width: `${progress}%` }} />
              </div>
            </div>

            <div className="info-card">
              <div className="info-label"><Zap size={14} /> Loss (Smoothed)</div>
              <div className="info-value" style={{ color: '#ff4d4d' }}>{stats.loss.toFixed(4)}</div>
            </div>

            <div className="info-card">
              <div className="info-label"><Database size={14} /> Tokens/Sec</div>
              <div className="info-value" style={{ color: '#00ffcc' }}>{(stats.tokensPerSec / 1000).toFixed(1)}K</div>
            </div>

            <div className="info-card">
              <div className="info-label"><HardDrive size={14} /> JVM Memory</div>
              <div className="info-value">{stats.usedMemory}MB / {stats.maxMemory}MB</div>
              <div className="progress-container">
                <div className="progress-fill memory-bar" style={{ width: `${memProgress}%`, background: memProgress > 80 ? '#ef4444' : '' }} />
              </div>
            </div>
          </div>

          <div className="main-content">
            {/* Left: Charts Column */}
            <div className="charts-container">
              <div className="chart-card">
                <div className="info-label"><Activity size={14} /> Loss Curve</div>
                <div style={{ width: '100%', height: 260, marginTop: '1rem' }}>
                  <ResponsiveContainer>
                    <AreaChart data={stats.lossHistory}>
                      <defs>
                        <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00d2ff" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#00d2ff" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="x" stroke="#444" fontSize={10} type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v) => `B${v}`} />
                      <YAxis stroke="#444" fontSize={10} domain={['auto', 'auto']} />
                      <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                      <Tooltip contentStyle={{ background: '#000', border: '1px solid #333', fontSize: 12 }} />
                      <Area type="monotone" dataKey="y" stroke="#00d2ff" fill="url(#colorLoss)" strokeWidth={2} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="chart-card">
                <div className="info-label"><Send size={14} /> Learning Rate Decay</div>
                <div style={{ width: '100%', height: 180, marginTop: '1rem' }}>
                  <ResponsiveContainer>
                    <AreaChart data={stats.lrHistory}>
                      <defs>
                        <linearGradient id="colorLR" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="x" hide />
                      <YAxis stroke="#444" fontSize={10} />
                      <Tooltip contentStyle={{ background: '#000', border: '1px solid #333', fontSize: 12 }} labelFormatter={(v) => `Batch ${v}`} />
                      <Area type="monotone" dataKey="y" stroke="#8b5cf6" fill="url(#colorLR)" strokeWidth={2} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Right: Info Sidebar */}
            <div className="stats-panel">
              <div className="info-card">
                <div className="info-label"><Monitor size={14} /> System Info</div>
                <div className="info-item" style={{ marginTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>OS: {stats.os}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>CPU: {stats.cpu}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Java: {stats.javaVersion}</div>
                </div>
              </div>

              <div className="info-card">
                <div className="info-label"><Cpu size={14} /> Model Architecture</div>
                <div className="info-item" style={{ marginTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Layers: {stats.layers} Blocks</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Attention: {stats.heads} heads x {stats.dModel / stats.heads}d</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Context: {stats.seqLen} tokens</div>
                  <div style={{ fontSize: '1rem', color: '#ffcc00', marginTop: '0.5rem', fontWeight: 600 }}>
                    {(stats.paramCount / 1e6).toFixed(2)}M Params
                  </div>
                </div>
              </div>

              <div className="info-card">
                <div className="info-label"><Clock size={14} /> Training Status</div>
                <div className="info-item" style={{ marginTop: '0.5rem' }}>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Started: {new Date(stats.startTime).toLocaleTimeString()}</div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>Speed: {stats.speed.toFixed(2)} batches/s</div>
                  <div style={{ fontSize: '0.8rem', color: '#00ffcc', marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <Save size={12} /> Last Saved: {stats.lastSave}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div style={{ textAlign: 'center', opacity: 0.5, marginTop: '20%' }}>
                <MessageSquare size={48} style={{ marginBottom: '1rem' }} />
                <p>Hello! I am the Dream Engine. How can I assist you today?</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.role}`}>
                {msg.text}
              </div>
            ))}
            {isTyping && (
              <div className="message bot">
                <RefreshCw size={14} className="animate-spin" /> Thinking...
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input-area">
            <input
              type="text" className="chat-input" placeholder="Ask anything..."
              value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            />
            <button className="send-btn" onClick={handleSend} disabled={isTyping || !input.trim()}>
              <Send size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
