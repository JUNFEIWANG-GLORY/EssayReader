"use client";
import { useState } from 'react';
import { Upload, Send, FileText, Loader2 } from 'lucide-react';

export default function ScholarPage() {
  const [file, setFile] = useState<File | null>(null);
  const [messages, setMessages] = useState<{role: 'user' | 'ai', content: string}[]>([]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleUpload = async () => {
    if (!file) return;
    setIsProcessing(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("api_key", "your_openai_key"); // In real app, get from a secure input

    await fetch(`${BACKEND_URL}/upload`, { method: "POST", body: formData });
    setIsProcessing(false);
    alert("Paper Analyzed Successfully!");
  };

  const sendMessage = async () => {
    if (!input) return;
    const userMsg = input;
    setInput("");
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);

    const formData = new FormData();
    formData.append("message", userMsg);

    const res = await fetch(`${BACKEND_URL}/chat`, { method: "POST", body: formData });
    const data = await res.json();
    setMessages(prev => [...prev, { role: 'ai', content: data.answer }]);
  };

  return (
    <div className="flex h-screen bg-neutral-900 text-white font-sans">
      {/* Sidebar */}
      <aside className="w-72 border-r border-neutral-800 p-6 flex flex-col gap-6 bg-neutral-950">
        <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
          <FileText className="text-blue-500" /> ScholarAgent
        </div>
        
        <div className="group relative border border-dashed border-neutral-700 rounded-xl p-8 hover:border-blue-500 transition-colors">
          <input type="file" className="absolute inset-0 opacity-0 cursor-pointer" onChange={e => setFile(e.target.files?.[0] || null)} />
          <div className="text-center text-sm text-neutral-400">
            <Upload className="mx-auto mb-2 opacity-50" />
            {file ? file.name : "Drop Paper Here"}
          </div>
        </div>

        <button 
          onClick={handleUpload}
          disabled={!file || isProcessing}
          className="w-full py-3 bg-blue-600 rounded-lg font-semibold hover:bg-blue-500 disabled:opacity-50 flex justify-center"
        >
          {isProcessing ? <Loader2 className="animate-spin" /> : "Process Paper"}
        </button>
      </aside>

      {/* Main Chat */}
      <main className="flex-1 flex flex-col bg-neutral-900">
        <header className="h-16 border-b border-neutral-800 flex items-center px-8 text-neutral-400 text-sm">
          Academic Research Mode | GPT-4o Powered
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-6">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-2xl p-4 rounded-2xl leading-relaxed ${m.role === 'user' ? 'bg-blue-600' : 'bg-neutral-800 border border-neutral-700'}`}>
                {m.content}
              </div>
            </div>
          ))}
        </div>

        <footer className="p-6">
          <div className="max-w-4xl mx-auto relative">
            <input 
              className="w-full bg-neutral-800 border border-neutral-700 rounded-2xl px-6 py-4 pr-16 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Ask about specific methodology or results..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && sendMessage()}
            />
            <button onClick={sendMessage} className="absolute right-3 top-3 p-2 bg-blue-600 rounded-xl hover:bg-blue-500 transition">
              <Send size={20} />
            </button>
          </div>
        </footer>
      </main>
    </div>
  );
}