import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Lock,
  User,
  LogOut,
  Send,
  Sparkles,
  BookOpen,
  X,
  Check,
  AlertCircle,
  FileText,
  ChevronRight,
  MessageSquare,
  RefreshCw,
  ShieldCheck,
  AlertTriangle,
  Ban,
  Database,
  Search,
  Plus,
  Settings,
  ChevronsLeft,
  ChevronsRight,
  Paperclip,
  BarChart2,
  Users,
  Lightbulb,
  HelpCircle,
  FileCheck,
  Info,
  ChevronDown,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export default function App() {
  // Authentication State
  const [token, setToken] = useState(() => localStorage.getItem('insidehq_token') || '');
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem('insidehq_user');
    return saved ? JSON.parse(saved) : null;
  });

  const [loginEmail, setLoginEmail] = useState('usera@company.com');
  const [loginPassword, setLoginPassword] = useState('secret123');
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);

  // Chat State
  const [messages, setMessages] = useState([]);
  const [inputQuery, setInputQuery] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  // Side Panel State for Citation Preview
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [sidePanelOpen, setSidePanelOpen] = useState(false);

  // Layout & Navigation State
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeNav, setActiveNav] = useState('chat'); // 'chat' | 'knowledge' | 'settings'
  const [showKnowledgeModal, setShowKnowledgeModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  // Auto-scroll ref
  const chatBottomRef = useRef(null);


  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Handle Login
  const handleLogin = async (e) => {
    if (e) e.preventDefault();
    setLoginError('');
    setLoginLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: loginEmail, password: loginPassword }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Login failed. Please check credentials.');
      }

      const data = await res.json();
      setToken(data.access_token);
      const userProfile = {
        id: data.user_id,
        email: loginEmail,
        role: data.role,
        department: data.department,
      };
      setCurrentUser(userProfile);
      localStorage.setItem('insidehq_token', data.access_token);
      localStorage.setItem('insidehq_user', JSON.stringify(userProfile));
    } catch (err) {
      setLoginError(err.message || 'Login failed.');
    } finally {
      setLoginLoading(false);
    }
  };

  // Handle Logout
  const handleLogout = () => {
    setToken('');
    setCurrentUser(null);
    localStorage.removeItem('insidehq_token');
    localStorage.removeItem('insidehq_user');
    setMessages([]);
    setConversationId(null);
  };

  // Pre-fill demo accounts
  const fillDemoAccount = (email, role) => {
    setLoginEmail(email);
    setLoginPassword('secret123');
  };

  // Send Chat Message via Streaming Fetch (SSE format from /chat)
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    const query = inputQuery.trim();
    if (!query || isGenerating) return;

    setInputQuery('');
    const userMsgIndex = messages.length;
    const assistantMsgIndex = messages.length + 1;

    const newMessages = [
      ...messages,
      { role: 'user', content: query },
      {
        role: 'assistant',
        content: '',
        citations: [],
        turnNumber: Math.floor(messages.length / 2) + 1,
        feedback: null, // 'up' | 'down'
      },
    ];

    setMessages(newMessages);
    setIsGenerating(true);

    try {
      const activeConvId = conversationId || crypto.randomUUID();
      if (!conversationId) setConversationId(activeConvId);

      const requestBody = {
        query: query,
        conversation_id: activeConvId,
        stream: true,
      };

      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestBody),
      });

      if (response.status === 401) {
        handleLogout();
        throw new Error('Session expired. Please log in again.');
      }

      if (!response.ok) {
        const errorText = await response.text();
        let detail = 'Failed to generate answer.';
        try {
          const errObj = JSON.parse(errorText);
          detail = errObj.detail || detail;
        } catch {
          detail = errorText || detail;
        }
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            try {
              const event = JSON.parse(dataStr);

              if (event.type === 'token') {
                setMessages((prev) => {
                  const updated = [...prev];
                  if (updated[assistantMsgIndex]) {
                    updated[assistantMsgIndex] = {
                      ...updated[assistantMsgIndex],
                      content: updated[assistantMsgIndex].content + event.content,
                    };
                  }
                  return updated;
                });
              } else if (event.type === 'citations') {
                setMessages((prev) => {
                  const updated = [...prev];
                  if (updated[assistantMsgIndex]) {
                    updated[assistantMsgIndex] = {
                      ...updated[assistantMsgIndex],
                      citations: event.citations || [],
                    };
                  }
                  return updated;
                });
              } else if (event.type === 'done') {
                if (event.conversation_id) setConversationId(event.conversation_id);
                setMessages((prev) => {
                  const updated = [...prev];
                  if (updated[assistantMsgIndex]) {
                    updated[assistantMsgIndex] = {
                      ...updated[assistantMsgIndex],
                      turnNumber: event.turn_number || updated[assistantMsgIndex].turnNumber,
                      conversationId: event.conversation_id || activeConvId,
                      // capture answer_type from SSE done event
                      answerType: event.answer_type || 'normal',
                    };
                  }
                  return updated;
                });
              }
            } catch (err) {
              console.warn('Could not parse SSE JSON line:', dataStr);
            }
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[assistantMsgIndex]) {
          updated[assistantMsgIndex] = {
            ...updated[assistantMsgIndex],
            content: `⚠️ Error: ${err.message || 'Network error'}`,
            answerType: 'error',
          };
        }
        return updated;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  // Reset conversation
  const handleNewConversation = useCallback(() => {
    setMessages([]);
    setConversationId(null);
  }, []);


  // Open Citation Side Panel
  const openCitationSidePanel = (citation, index) => {
    setSelectedCitation({ ...citation, displayIndex: index + 1 });
    setSidePanelOpen(true);
  };

  // Render Inline Numbered Citations in Assistant Answer Text
  const renderMessageContent = (msg) => {
    const { content, citations } = msg;
    if (!content) return null;

    if (!citations || citations.length === 0) {
      return <span className="whitespace-pre-wrap">{content}</span>;
    }

    // Map chunk_ids or bracketed markers [1], [uuid] to numbered citation badges
    const parts = content.split(/(\[[a-fA-F0-9\-]{32,36}\]|\[\d+\])/g);

    return parts.map((part, i) => {
      // Check if part matches a bracketed citation marker
      const match = part.match(/\[([a-fA-F0-9\-]{32,36}|\d+)\]/);
      if (match) {
        const markerVal = match[1];
        // Match against citations array
        const foundIndex = citations.findIndex(
          (c) =>
            String(c.chunk_id) === markerVal ||
            String(c.chunk_index) === markerVal ||
            markerVal === String(i)
        );

        const citationObj = foundIndex !== -1 ? citations[foundIndex] : citations[0];
        const displayNum = foundIndex !== -1 ? foundIndex + 1 : 1;

        return (
          <button
            key={i}
            onClick={() => openCitationSidePanel(citationObj, foundIndex !== -1 ? foundIndex : 0)}
            className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/40 hover:border-indigo-400 transition-all cursor-pointer shadow-sm"
            title={`View citation details: ${citationObj?.document_title || 'Document'}`}
          >
            <BookOpen className="w-3 h-3 text-indigo-400" />
            <span>[{displayNum}]</span>
          </button>
        );
      }

      return <span key={i} className="whitespace-pre-wrap">{part}</span>;
    });
  };

  // ---------------------------------------------------------------------------
  // Unauthenticated Login View
  // ---------------------------------------------------------------------------
  if (!token || !currentUser) {
    return (
      <div className="min-h-screen w-full flex relative overflow-hidden bg-[linear-gradient(120deg,#F8FAFC_0%,#EFF6FF_55%,#EEF2FF_100%)] text-[#0F172A] font-['Inter'] selection:bg-blue-600 selection:text-white">
        {/* Faint background watermark text: KNOWLEDGE → CONTEXT → ANSWERS */}
        <div className="absolute inset-0 pointer-events-none select-none flex items-center justify-center overflow-hidden z-0">
          <span className="text-[#0F172A]/[0.035] text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-black tracking-[0.25em] uppercase whitespace-nowrap">
            KNOWLEDGE &rarr; CONTEXT &rarr; ANSWERS
          </span>
        </div>

        {/* Ambient soft glow overlays */}
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-400/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-[-10%] right-[30%] w-[600px] h-[600px] bg-indigo-400/10 rounded-full blur-3xl pointer-events-none" />

        <div className="w-full flex min-h-screen z-10 relative">
          {/* ========================================================================= */}
          {/* LEFT COLUMN (~45%): Login UI / Card                                      */}
          {/* ========================================================================= */}
          <div className="w-full lg:w-[45%] flex flex-col justify-between p-6 sm:p-10 lg:p-12 xl:p-16 min-h-screen">
            {/* Top Brand Logo & Header */}
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[linear-gradient(135deg,#2563EB_0%,#7C3AED_100%)] flex items-center justify-center shadow-md shadow-blue-500/15">
                <ShieldCheck className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-['Outfit'] font-bold text-xl tracking-tight text-[#0F172A]">
                    InsideHQ
                  </span>

                </div>
                <p className="text-xs text-[#64748B]">Company Knowledge Assistant</p>
              </div>
            </div>

            {/* Main Login Card */}
            <div className="my-auto py-8 max-w-md w-full mx-auto">
              <div className="bg-white border border-[#E2E8F0] rounded-2xl p-6 sm:p-8 shadow-xl shadow-slate-200/60 space-y-6">
                <div>
                  <h1 className="text-2xl font-bold tracking-tight text-[#0F172A] font-['Outfit']">
                    Welcome back
                  </h1>
                  <p className="text-sm text-[#64748B] mt-1">
                    Sign in with your employee credentials to access permissioned knowledge.
                  </p>
                </div>

                {loginError && (
                  <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2.5">
                    <AlertCircle className="w-4 h-4 flex-shrink-0 text-red-500" />
                    <span className="font-medium">{loginError}</span>
                  </div>
                )}

                <form onSubmit={handleLogin} className="space-y-4">
                  <div>
                    <label className="block text-xs font-semibold text-[#0F172A] mb-1.5 uppercase tracking-wider">
                      Work Email
                    </label>
                    <div className="relative">
                      <input
                        type="email"
                        required
                        value={loginEmail}
                        onChange={(e) => setLoginEmail(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white border border-[#E2E8F0] text-sm text-[#0F172A] placeholder:text-slate-400 focus:outline-none focus:border-[#2563EB] focus:ring-2 focus:ring-[#2563EB]/20 transition-all shadow-xs"
                        placeholder="employee@company.com"
                      />
                      <User className="w-4 h-4 text-[#64748B] absolute left-3.5 top-3 pointer-events-none" />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-[#0F172A] mb-1.5 uppercase tracking-wider">
                      Password
                    </label>
                    <div className="relative">
                      <input
                        type="password"
                        required
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-white border border-[#E2E8F0] text-sm text-[#0F172A] placeholder:text-slate-400 focus:outline-none focus:border-[#2563EB] focus:ring-2 focus:ring-[#2563EB]/20 transition-all shadow-xs"
                        placeholder="••••••••"
                      />
                      <Lock className="w-4 h-4 text-[#64748B] absolute left-3.5 top-3 pointer-events-none" />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loginLoading}
                    className="w-full py-3 px-4 rounded-xl bg-[#2563EB] hover:bg-blue-700 text-white font-semibold text-sm transition-all shadow-md shadow-blue-600/20 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 active:scale-[0.99]"
                  >
                    {loginLoading ? (
                      <RefreshCw className="w-4 h-4 animate-spin text-white" />
                    ) : (
                      <>
                        <span>Sign In to Knowledge Base</span>
                        <ChevronRight className="w-4 h-4" />
                      </>
                    )}
                  </button>
                </form>

                {/* Quick Demo Pre-fill Accounts */}
                <div className="pt-5 border-t border-[#E2E8F0] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-[#64748B] font-semibold uppercase tracking-wider">
                      Quick Demo Roles
                    </span>
                    <span className="text-[10px] text-slate-400 font-normal">Click to auto-fill</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <button
                      type="button"
                      onClick={() => fillDemoAccount('usera@company.com', 'employee')}
                      className={`px-2.5 py-2 rounded-xl text-xs font-medium border transition-all text-center cursor-pointer ${loginEmail === 'usera@company.com'
                        ? 'bg-blue-50 border-[#2563EB] text-[#2563EB] font-semibold shadow-xs'
                        : 'bg-slate-50 hover:bg-slate-100 border-[#E2E8F0] text-[#0F172A]'
                        }`}
                    >
                      Employee A
                    </button>
                    <button
                      type="button"
                      onClick={() => fillDemoAccount('curator@company.com', 'curator')}
                      className={`px-2.5 py-2 rounded-xl text-xs font-medium border transition-all text-center cursor-pointer ${loginEmail === 'curator@company.com'
                        ? 'bg-blue-50 border-[#2563EB] text-[#2563EB] font-semibold shadow-xs'
                        : 'bg-slate-50 hover:bg-slate-100 border-[#E2E8F0] text-[#0F172A]'
                        }`}
                    >
                      Curator
                    </button>
                    <button
                      type="button"
                      onClick={() => fillDemoAccount('admin@company.com', 'admin')}
                      className={`px-2.5 py-2 rounded-xl text-xs font-medium border transition-all text-center cursor-pointer ${loginEmail === 'admin@company.com'
                        ? 'bg-blue-50 border-[#2563EB] text-[#2563EB] font-semibold shadow-xs'
                        : 'bg-slate-50 hover:bg-slate-100 border-[#E2E8F0] text-[#0F172A]'
                        }`}
                    >
                      Admin
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Left Footer */}
            <div className="flex items-center justify-between text-xs text-[#64748B] pt-4">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-[#2563EB]" />
                Role-Based Security Active
              </span>
              <span>InsideHQ &copy; {new Date().getFullYear()}</span>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* RIGHT COLUMN (~55%): RAG Product Concept Visual                           */}
          {/* ========================================================================= */}
          <div className="hidden lg:flex lg:w-[55%] relative flex-col justify-between p-12 xl:p-16 border-l border-[#E2E8F0]/80 bg-white/40 backdrop-blur-sm overflow-hidden">
            {/* Subtle Background Vector/Knowledge Graph Grid (Opacity ~5-12%) */}
            <div className="absolute inset-0 pointer-events-none z-0 overflow-hidden">
              <svg className="w-full h-full opacity-[0.08]" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  <pattern id="grid-pattern" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#0F172A" strokeWidth="1" />
                    <circle cx="40" cy="40" r="1.5" fill="#2563EB" />
                  </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#grid-pattern)" />
              </svg>

              {/* Subtle connecting lines & graph nodes (Opacity 8-12%) */}
              <svg className="absolute inset-0 w-full h-full opacity-10" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20%" cy="30%" r="4" fill="#2563EB" />
                <circle cx="35%" cy="20%" r="3" fill="#7C3AED" />
                <circle cx="50%" cy="40%" r="5" fill="#2563EB" />
                <circle cx="75%" cy="25%" r="4" fill="#2563EB" />
                <circle cx="80%" cy="55%" r="3" fill="#7C3AED" />
                <circle cx="65%" cy="75%" r="4" fill="#2563EB" />
                <circle cx="30%" cy="70%" r="5" fill="#7C3AED" />

                <line x1="20%" y1="30%" x2="35%" y2="20%" stroke="#0F172A" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="35%" y1="20%" x2="50%" y2="40%" stroke="#2563EB" strokeWidth="1.5" />
                <line x1="50%" y1="40%" x2="75%" y2="25%" stroke="#0F172A" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="75%" y1="25%" x2="80%" y2="55%" stroke="#7C3AED" strokeWidth="1.5" />
                <line x1="80%" y1="55%" x2="65%" y2="75%" stroke="#0F172A" strokeWidth="1" />
                <line x1="65%" y1="75%" x2="30%" y2="70%" stroke="#2563EB" strokeWidth="1" strokeDasharray="3,3" />
                <line x1="30%" y1="70%" x2="20%" y2="30%" stroke="#7C3AED" strokeWidth="1" />
                <line x1="50%" y1="40%" x2="65%" y2="75%" stroke="#2563EB" strokeWidth="1" />
              </svg>
            </div>

            {/* Header Description */}
            <div className="relative z-10 max-w-lg">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#2563EB]/10 border border-[#2563EB]/20 text-[#2563EB] text-xs font-semibold uppercase tracking-wider mb-4">
                <Sparkles className="w-3.5 h-3.5 text-[#7C3AED]" />
                <span>How InsideHQ RAG Works</span>
              </div>
              <h2 className="text-2xl xl:text-3xl font-bold tracking-tight text-[#0F172A] font-['Outfit']">
                Context-aware answers from your enterprise knowledge base
              </h2>
              <p className="text-sm text-[#64748B] mt-2 leading-relaxed">
                InsideHQ indexes your company's documents, retrieves verified facts with permission control, and delivers precise answers with line-level citations.
              </p>
            </div>

            {/* RAG Pipeline Flow Cards: Documents → Knowledge Base → Retrieval → Answers */}
            <div className="relative z-10 my-auto py-8 w-full max-w-xl mx-auto space-y-4">

              {/* 1. DOCUMENTS */}
              <div className="group bg-white/80 hover:bg-white border border-[#E2E8F0] rounded-2xl p-4 shadow-sm hover:shadow-md transition-all flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center flex-shrink-0">
                    <FileText className="w-5 h-5 text-[#2563EB]" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-[#64748B]">Step 1</span>
                      <span className="text-sm font-bold text-[#0F172A]">Company Documents</span>
                    </div>
                    <p className="text-xs text-[#64748B] mt-0.5">
                      HR Policies, IT Guides, Engineering Specs, Financial Reports
                    </p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-100 text-[11px] text-[#64748B] font-medium border border-slate-200">
                  PDF &bull; DOCX &bull; MD
                </div>
              </div>

              {/* Connecting Line */}
              <div className="flex justify-center -my-2 relative z-20">
                <div className="w-0.5 h-5 bg-gradient-to-b from-[#2563EB]/40 to-[#7C3AED]/40" />
              </div>

              {/* 2. KNOWLEDGE BASE */}
              <div className="group bg-white/80 hover:bg-white border border-[#E2E8F0] rounded-2xl p-4 shadow-sm hover:shadow-md transition-all flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center flex-shrink-0">
                    <Database className="w-5 h-5 text-[#7C3AED]" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-[#64748B]">Step 2</span>
                      <span className="text-sm font-bold text-[#0F172A]">Knowledge Base &amp; Vector Store</span>
                    </div>
                    <p className="text-xs text-[#64748B] mt-0.5">
                      Chunks documents into vector embeddings stored in PostgreSQL pgvector
                    </p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-purple-50 text-[11px] text-[#7C3AED] font-semibold border border-purple-100">
                  Vector Indexing
                </div>
              </div>

              {/* Connecting Line */}
              <div className="flex justify-center -my-2 relative z-20">
                <div className="w-0.5 h-5 bg-gradient-to-b from-[#7C3AED]/40 to-[#2563EB]/40" />
              </div>

              {/* 3. RETRIEVAL */}
              <div className="group bg-white/80 hover:bg-white border border-[#E2E8F0] rounded-2xl p-4 shadow-sm hover:shadow-md transition-all flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center flex-shrink-0">
                    <Search className="w-5 h-5 text-[#2563EB]" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-[#64748B]">Step 3</span>
                      <span className="text-sm font-bold text-[#0F172A]">Permission-Aware Retrieval</span>
                    </div>
                    <p className="text-xs text-[#64748B] mt-0.5">
                      Hybrid BM25 + Vector search with strict RBAC department filtering
                    </p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-50 text-[11px] text-[#2563EB] font-semibold border border-blue-100">
                  Hybrid Rerank
                </div>
              </div>

              {/* Connecting Line */}
              <div className="flex justify-center -my-2 relative z-20">
                <div className="w-0.5 h-5 bg-gradient-to-b from-[#2563EB]/40 to-[#7C3AED]/40" />
              </div>

              {/* 4. ANSWERS */}
              <div className="group bg-white border-2 border-[#2563EB]/30 rounded-2xl p-4 shadow-md shadow-blue-500/5 transition-all flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-[linear-gradient(135deg,#2563EB_0%,#7C3AED_100%)] flex items-center justify-center flex-shrink-0 shadow-sm">
                    <MessageSquare className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold uppercase tracking-wider text-[#7C3AED]">Step 4</span>
                      <span className="text-sm font-bold text-[#0F172A]">Synthesized Answers</span>
                    </div>
                    <p className="text-xs text-[#64748B] mt-0.5">
                      Accurate LLM response streaming with exact document citation badges
                    </p>
                  </div>
                </div>
                <div className="hidden sm:flex items-center gap-1 px-2 py-0.5 rounded-md bg-indigo-50 text-[10px] text-[#7C3AED] font-bold border border-indigo-200">
                  <BookOpen className="w-3 h-3 text-[#7C3AED]" />
                  [1] Citation
                </div>
              </div>

            </div>

          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Authenticated Main Chat Application Layout (Redesigned Enterprise UI)
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-[#F1F5F9] text-[#0F172A] flex p-3 sm:p-4 font-['Inter'] relative gap-3 sm:gap-4 overflow-hidden selection:bg-blue-600 selection:text-white">

      {/* ========================================================================= */}
      {/* LEFT SIDEBAR: Navigation & User Profile                                   */}
      {/* ========================================================================= */}
      <aside
        className={`${sidebarCollapsed ? 'w-20' : 'w-64 xl:w-72'
          } transition-all duration-300 bg-white border border-[#E2E8F0] rounded-3xl p-4 shadow-xs flex flex-col justify-between z-20 flex-shrink-0`}
      >
        <div className="space-y-6">
          {/* Brand Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="w-10 h-10 rounded-xl bg-[linear-gradient(135deg,#2563EB_0%,#7C3AED_100%)] flex items-center justify-center flex-shrink-0 shadow-md shadow-blue-500/15">
                <ShieldCheck className="w-5 h-5 text-white" />
              </div>
              {!sidebarCollapsed && (
                <div className="truncate">
                  <h1 className="font-['Outfit'] font-bold text-lg tracking-tight text-[#0F172A]">
                    InsideHQ
                  </h1>
                  <p className="text-[11px] text-[#64748B] truncate">Company Knowledge Assistant</p>
                </div>
              )}
            </div>

            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="p-1.5 rounded-xl hover:bg-slate-100 text-[#64748B] hover:text-[#0F172A] transition-all cursor-pointer flex-shrink-0"
              title={sidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
            >
              {sidebarCollapsed ? (
                <ChevronsRight className="w-4 h-4" />
              ) : (
                <ChevronsLeft className="w-4 h-4" />
              )}
            </button>
          </div>

          {/* New Question Button */}
          <button
            onClick={handleNewConversation}
            className={`w-full py-3 px-4 rounded-2xl bg-[#2563EB] hover:bg-blue-700 text-white font-semibold text-sm transition-all shadow-md shadow-blue-600/20 flex items-center justify-center gap-2 cursor-pointer ${sidebarCollapsed ? 'px-2' : ''
              }`}
            title="Start a new conversation"
          >
            <Plus className="w-5 h-5 flex-shrink-0" />
            {!sidebarCollapsed && <span>New Question</span>}
          </button>

          {/* Navigation Links */}
          <nav className="space-y-1.5">
            <button
              onClick={() => {
                setActiveNav('chat');
              }}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all cursor-pointer ${activeNav === 'chat'
                ? 'bg-blue-50 text-[#2563EB] font-semibold border border-blue-100/80 shadow-2xs'
                : 'text-[#64748B] hover:text-[#0F172A] hover:bg-slate-50'
                }`}
            >
              <MessageSquare className={`w-4 h-4 flex-shrink-0 ${activeNav === 'chat' ? 'text-[#2563EB]' : ''}`} />
              {!sidebarCollapsed && <span>Ask Assistant</span>}
            </button>

            <button
              onClick={() => setShowKnowledgeModal(true)}
              className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium text-[#64748B] hover:text-[#0F172A] hover:bg-slate-50 transition-all cursor-pointer"
            >
              <FileText className="w-4 h-4 flex-shrink-0" />
              {!sidebarCollapsed && <span>Knowledge Base</span>}
            </button>

            <button
              onClick={() => setShowSettingsModal(true)}
              className="w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium text-[#64748B] hover:text-[#0F172A] hover:bg-slate-50 transition-all cursor-pointer"
            >
              <Settings className="w-4 h-4 flex-shrink-0" />
              {!sidebarCollapsed && <span>Settings</span>}
            </button>
          </nav>
        </div>

        {/* User Profile Card */}
        <div className="pt-4 border-t border-[#E2E8F0]">
          <div className={`flex items-center gap-3 p-2 rounded-2xl bg-slate-50 border border-slate-100 ${sidebarCollapsed ? 'justify-center' : ''}`}>
            <div className="w-9 h-9 rounded-full bg-indigo-100 text-[#7C3AED] font-bold text-xs flex items-center justify-center flex-shrink-0 shadow-2xs">
              {currentUser.email ? currentUser.email.slice(0, 2).toUpperCase() : 'UA'}
            </div>

            {!sidebarCollapsed && (
              <div className="flex-1 truncate min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#0F172A] truncate">
                    {currentUser.email.split('@')[0]}
                  </span>
                  <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-blue-100/60 text-[#2563EB]">
                    {currentUser.role}
                  </span>
                </div>
                <span className="text-[11px] text-[#64748B] truncate block">
                  {currentUser.email}
                </span>
              </div>
            )}

            {!sidebarCollapsed && (
              <button
                onClick={handleLogout}
                className="p-1.5 rounded-lg hover:bg-slate-200/60 text-[#64748B] hover:text-red-600 transition-all cursor-pointer"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* ========================================================================= */}
      {/* MAIN CHAT AREA (Right Panel)                                             */}
      {/* ========================================================================= */}
      <main className="flex-1 bg-white border border-[#E2E8F0] rounded-3xl p-4 sm:p-6 shadow-xs flex flex-col justify-between relative overflow-hidden h-[calc(100vh-2rem)]">

        {/* Top Header Bar inside Main Workspace */}
        <header className="flex items-center justify-between pb-4 border-b border-[#E2E8F0]/80">
          <div>
            <h2 className="font-['Outfit'] font-bold text-xl sm:text-2xl tracking-tight text-[#0F172A]">
              InsideHQ
            </h2>
            <p className="text-xs text-[#64748B] mt-0.5">
              Your company's knowledge, one conversation away.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-xs font-semibold text-[#2563EB]">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span className="capitalize">{currentUser.department || 'All'} Scoped</span>
            </div>

            <button
              onClick={() => setShowKnowledgeModal(true)}
              className="w-10 h-10 rounded-full bg-slate-50 hover:bg-slate-100 border border-[#E2E8F0] text-[#64748B] flex items-center justify-center transition-all cursor-pointer"
              title="Search Knowledge Base"
            >
              <Search className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Middle Content Area */}
        <div className="flex-1 overflow-y-auto py-6 space-y-6 pr-1">
          {messages.length === 0 ? (
            /* ========================================================================= */
            /* EMPTY STATE HERO VIEW                                                     */
            /* ========================================================================= */
            <div className="h-full flex flex-col items-center justify-center text-center max-w-3xl mx-auto space-y-8 py-4">

              {/* Glowing Hero Icon Badge */}
              <div className="space-y-3">
                <div className="w-16 h-16 rounded-full bg-gradient-to-tr from-purple-100 via-indigo-50 to-blue-100 border border-purple-200/80 flex items-center justify-center text-[#7C3AED] shadow-sm mx-auto">
                  <Sparkles className="w-8 h-8" />
                </div>
                <h3 className="font-['Outfit'] text-2xl sm:text-3xl font-bold tracking-tight text-[#0F172A]">
                  How can I help you today?
                </h3>
                <p className="text-xs sm:text-sm text-[#64748B] max-w-lg mx-auto leading-relaxed">
                  Ask anything about company documents, policies, processes, projects, or internal knowledge. I can search across all your company resources.
                </p>
              </div>

              {/* 6 Suggested Prompt Cards Grid (Matching reference image) */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 w-full text-left">

                {/* 1. Summarize a document */}
                <button
                  onClick={() => setInputQuery('Summarize the company parental leave policy')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-blue-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center text-[#2563EB] flex-shrink-0 group-hover:scale-105 transition-transform">
                    <FileText className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">Summarize a document</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Get key insights quickly</p>
                  </div>
                </button>

                {/* 2. Find company policy */}
                <button
                  onClick={() => setInputQuery('What are the remote work guidelines?')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-purple-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-purple-50 border border-purple-100 flex items-center justify-center text-[#7C3AED] flex-shrink-0 group-hover:scale-105 transition-transform">
                    <Search className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">Find company policy</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Search internal policies</p>
                  </div>
                </button>

                {/* 3. Explain a process */}
                <button
                  onClick={() => setInputQuery('How do I request hardware equipment?')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-emerald-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-100 flex items-center justify-center text-emerald-600 flex-shrink-0 group-hover:scale-105 transition-transform">
                    <Users className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">Explain a process</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Step-by-step guidance</p>
                  </div>
                </button>

                {/* 4. Get project information */}
                <button
                  onClick={() => setInputQuery('What is the engineering architecture setup?')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-cyan-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-cyan-50 border border-cyan-100 flex items-center justify-center text-cyan-600 flex-shrink-0 group-hover:scale-105 transition-transform">
                    <BarChart2 className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">Get project information</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Find updates and details</p>
                  </div>
                </button>

                {/* 5. Ask about HR */}
                <button
                  onClick={() => setInputQuery('What leave benefits are available?')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-orange-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-orange-50 border border-orange-100 flex items-center justify-center text-orange-600 flex-shrink-0 group-hover:scale-105 transition-transform">
                    <MessageSquare className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">Ask about HR</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Policies, leaves, benefits</p>
                  </div>
                </button>

                {/* 6. General questions */}
                <button
                  onClick={() => setInputQuery('What internal employee training resources exist?')}
                  className="p-4 rounded-2xl bg-white hover:bg-slate-50 border border-[#E2E8F0] hover:border-amber-300 shadow-xs hover:shadow-md transition-all cursor-pointer flex items-center gap-3.5 group"
                >
                  <div className="w-10 h-10 rounded-xl bg-amber-50 border border-amber-100 flex items-center justify-center text-amber-600 flex-shrink-0 group-hover:scale-105 transition-transform">
                    <Lightbulb className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#0F172A]">General questions</h4>
                    <p className="text-[11px] text-[#64748B] mt-0.5">Ask anything</p>
                  </div>
                </button>

              </div>
            </div>
          ) : (
            /* ========================================================================= */
            /* ACTIVE MESSAGES STREAMING LOG                                            */
            /* ========================================================================= */
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex gap-3.5 ${msg.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
              >
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-xl bg-[linear-gradient(135deg,#2563EB_0%,#7C3AED_100%)] flex items-center justify-center text-white flex-shrink-0 shadow-sm mt-1">
                    <Sparkles className="w-4 h-4" />
                  </div>
                )}

                <div
                  className={`max-w-2xl rounded-2xl p-4 sm:p-5 text-sm leading-relaxed space-y-3 ${msg.role === 'user'
                    ? 'bg-[#2563EB] text-white shadow-md shadow-blue-600/15 rounded-tr-none'
                    : 'bg-white border border-[#E2E8F0] text-[#0F172A] shadow-xs rounded-tl-none'
                    }`}
                >
                  <div>{renderMessageContent(msg)}</div>

                  {/* Answer-type badge */}
                  {msg.role === 'assistant' && msg.answerType && msg.answerType !== 'normal' && msg.content && (
                    <div
                      role="status"
                      aria-live="polite"
                      className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold border ${msg.answerType === 'restricted'
                        ? 'bg-red-50 border-red-200 text-red-700'
                        : 'bg-amber-50 border-amber-200 text-amber-800'
                        }`}
                    >
                      {msg.answerType === 'restricted' ? (
                        <>
                          <Ban className="w-3.5 h-3.5 flex-shrink-0 text-red-600" aria-hidden="true" />
                          <span>Restricted — this query was blocked by safety guardrails.</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 text-amber-600" aria-hidden="true" />
                          <span>Abstained — insufficient information in permitted knowledge base.</span>
                        </>
                      )}
                    </div>
                  )}

                  {/* Citations List */}
                  {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                    <div className="pt-3 border-t border-[#E2E8F0] space-y-2">
                      <span className="text-[11px] font-semibold text-[#64748B] flex items-center gap-1.5">
                        <BookOpen className="w-3.5 h-3.5 text-[#2563EB]" />
                        <span>Sources &amp; Citations ({msg.citations.length})</span>
                      </span>
                      <div className="flex flex-wrap gap-2">
                        {msg.citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            onClick={() => openCitationSidePanel(c, cIdx)}
                            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-[#E2E8F0] hover:border-blue-300 text-xs text-[#0F172A] hover:text-[#2563EB] transition-all cursor-pointer shadow-2xs"
                          >
                            <span className="w-4 h-4 rounded-full bg-blue-100 text-[#2563EB] font-bold text-[10px] flex items-center justify-center">
                              {cIdx + 1}
                            </span>
                            <span className="truncate max-w-[160px]">
                              {c.document_title || 'Document'}
                            </span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-xl bg-slate-200 text-[#0F172A] flex items-center justify-center flex-shrink-0 font-bold text-xs mt-1">
                    {currentUser.email[0].toUpperCase()}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={chatBottomRef} />
        </div>

        {/* ========================================================================= */}
        {/* BOTTOM FLOATING INPUT BAR (Matching reference image)                       */}
        {/* ========================================================================= */}
        <form onSubmit={handleSendMessage} className="relative max-w-3xl w-full mx-auto pt-2">
          <div className="bg-white border border-[#E2E8F0] rounded-full p-2 pl-4 shadow-lg shadow-slate-200/80 flex items-center gap-3 transition-all focus-within:border-[#2563EB] focus-within:ring-2 focus-within:ring-blue-500/10">
            <button
              type="button"
              onClick={() => setShowKnowledgeModal(true)}
              className="p-1.5 rounded-full hover:bg-slate-100 text-[#64748B] hover:text-[#0F172A] transition-all cursor-pointer flex-shrink-0"
              title="Attach context or view Knowledge Base"
            >
              <Paperclip className="w-4 h-4" />
            </button>

            <input
              id="chat-input"
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Type your question here..."
              disabled={isGenerating}
              className="flex-1 bg-transparent border-none outline-none text-sm text-[#0F172A] placeholder-slate-400 focus:outline-none"
            />

            <button
              type="submit"
              disabled={!inputQuery.trim() || isGenerating}
              className="w-10 h-10 rounded-full bg-[#2563EB] hover:bg-blue-700 disabled:opacity-40 text-white flex items-center justify-center transition-all shadow-md shadow-blue-600/20 cursor-pointer flex-shrink-0"
              title="Send question"
            >
              {isGenerating ? (
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
              ) : (
                <Send className="w-4 h-4 text-white" />
              )}
            </button>
          </div>
        </form>

        {/* Slide-Over Side Panel for Citations */}
        {sidePanelOpen && selectedCitation && (
          <aside className="w-96 bg-white border-l border-[#E2E8F0] p-6 flex flex-col justify-between shadow-2xl absolute right-0 top-0 bottom-0 z-30 animate-in slide-in-from-right duration-200">
            <div className="space-y-6 overflow-y-auto pr-1">
              <div className="flex items-center justify-between pb-4 border-b border-[#E2E8F0]">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center text-[#2563EB] text-xs font-bold">
                    [{selectedCitation.displayIndex || 1}]
                  </div>
                  <h3 className="font-['Outfit'] font-bold text-base text-[#0F172A]">
                    Citation Details
                  </h3>
                </div>
                <button
                  onClick={() => setSidePanelOpen(false)}
                  className="p-1 rounded-lg hover:bg-slate-100 text-[#64748B] hover:text-[#0F172A] transition-all cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Document Meta Info */}
              <div className="space-y-3">
                <div>
                  <span className="text-[10px] uppercase font-semibold text-[#64748B] tracking-wider">
                    Document Title
                  </span>
                  <p className="text-sm font-semibold text-[#2563EB] mt-0.5 flex items-center gap-1.5">
                    <FileText className="w-4 h-4 text-[#2563EB]" />
                    <span>{selectedCitation.document_title || 'Document'}</span>
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-[#E2E8F0]">
                    <span className="text-[10px] text-[#64748B] uppercase font-semibold block">
                      Source Type
                    </span>
                    <span className="text-[#0F172A] font-mono mt-0.5 block">
                      {selectedCitation.source_type || 'PDF'}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-[#E2E8F0]">
                    <span className="text-[10px] text-[#64748B] uppercase font-semibold block">
                      Section / Page
                    </span>
                    <span className="text-[#0F172A] font-mono mt-0.5 block truncate">
                      {selectedCitation.section || selectedCitation.page || 'General'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Exact Chunk Excerpt */}
              <div className="space-y-2">
                <span className="text-[10px] uppercase font-semibold text-[#64748B] tracking-wider block">
                  Retrieved Chunk Excerpt
                </span>
                <div className="p-3.5 rounded-xl bg-slate-50 border border-[#E2E8F0] text-xs text-[#0F172A] leading-relaxed font-mono whitespace-pre-wrap max-h-60 overflow-y-auto">
                  {selectedCitation.chunk_text || selectedCitation.excerpt || 'Chunk content details.'}
                </div>
              </div>
            </div>
          </aside>
        )}
      </main>

      {/* ========================================================================= */}
      {/* KNOWLEDGE BASE OVERLAY MODAL                                              */}
      {/* ========================================================================= */}
      {showKnowledgeModal && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-in fade-in duration-150">
          <div className="bg-white border border-[#E2E8F0] rounded-3xl p-6 sm:p-8 max-w-xl w-full shadow-2xl space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E2E8F0]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-50 border border-blue-100 flex items-center justify-center text-[#2563EB]">
                  <Database className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-['Outfit'] font-bold text-lg text-[#0F172A]">
                    Company Knowledge Base
                  </h3>
                  <p className="text-xs text-[#64748B]">Permission-Aware Document Store</p>
                </div>
              </div>
              <button
                onClick={() => setShowKnowledgeModal(false)}
                className="p-1.5 rounded-xl hover:bg-slate-100 text-[#64748B] hover:text-[#0F172A] transition-all cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3">
              <div className="p-4 rounded-2xl bg-slate-50 border border-[#E2E8F0] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#0F172A]">HR &amp; Employee Handbook</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">Indexed</span>
                </div>
                <p className="text-xs text-[#64748B]">Contains parental leave, remote work guidelines, and health benefits.</p>
              </div>

              <div className="p-4 rounded-2xl bg-slate-50 border border-[#E2E8F0] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#0F172A]">IT &amp; Hardware Procurement</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">Indexed</span>
                </div>
                <p className="text-xs text-[#64748B]">Laptop request workflows, VPN access rules, and security guidelines.</p>
              </div>

              <div className="p-4 rounded-2xl bg-slate-50 border border-[#E2E8F0] space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-[#0F172A]">Engineering Architecture Specs</span>
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 font-semibold">Indexed</span>
                </div>
                <p className="text-xs text-[#64748B]">Backend microservices, PostgreSQL pgvector setup, and streaming API design.</p>
              </div>
            </div>

            <button
              onClick={() => setShowKnowledgeModal(false)}
              className="w-full py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-[#0F172A] font-semibold text-xs transition-all cursor-pointer"
            >
              Close Window
            </button>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SETTINGS OVERLAY MODAL                                                    */}
      {/* ========================================================================= */}
      {showSettingsModal && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-in fade-in duration-150">
          <div className="bg-white border border-[#E2E8F0] rounded-3xl p-6 sm:p-8 max-w-md w-full shadow-2xl space-y-6">
            <div className="flex items-center justify-between pb-4 border-b border-[#E2E8F0]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-purple-50 border border-purple-100 flex items-center justify-center text-[#7C3AED]">
                  <Settings className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-['Outfit'] font-bold text-lg text-[#0F172A]">
                    Account &amp; RBAC Settings
                  </h3>
                  <p className="text-xs text-[#64748B]">Current session configuration</p>
                </div>
              </div>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="p-1.5 rounded-xl hover:bg-slate-100 text-[#64748B] hover:text-[#0F172A] transition-all cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-50 border border-[#E2E8F0] flex justify-between items-center">
                <span className="text-[#64748B]">Authenticated Email</span>
                <span className="font-semibold text-[#0F172A]">{currentUser.email}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-[#E2E8F0] flex justify-between items-center">
                <span className="text-[#64748B]">Role Permission</span>
                <span className="font-bold text-[#2563EB] uppercase">{currentUser.role}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-[#E2E8F0] flex justify-between items-center">
                <span className="text-[#64748B]">Department Scope</span>
                <span className="font-semibold text-[#0F172A]">{currentUser.department || 'Global'}</span>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-[#E2E8F0] flex justify-between items-center">
                <span className="text-[#64748B]">Retrieval Engine</span>
                <span className="font-mono text-[#7C3AED]">Hybrid BM25 + Vector</span>
              </div>
            </div>

            <button
              onClick={() => setShowSettingsModal(false)}
              className="w-full py-2.5 rounded-xl bg-[#2563EB] hover:bg-blue-700 text-white font-semibold text-xs transition-all cursor-pointer shadow-xs"
            >
              Done
            </button>
          </div>
        </div>
      )}

    </div>
  );
}

