import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { TrajectorySimulator } from './components/TrajectorySimulator';
import { SearchStudio } from './components/SearchStudio';
import { DatasetsViewer } from './components/DatasetsViewer';
import { ZeroCostProxies } from './components/ZeroCostProxies';
import { EvaluationBenchmarking } from './components/EvaluationBenchmarking';
import { ProposalDossier } from './components/ProposalDossier';
import { CodeExplorer } from './components/CodeExplorer';
import { Cpu, Zap, Activity, CheckCircle2, BookOpen } from 'lucide-react';

import { JuniorDevDocumentation } from './components/JuniorDevDocumentation';

import { ExperimentProvider } from './context/ExperimentContext';

export default function App() {
  const [activeTab, setActiveTab] = useState<'simulator' | 'search' | 'datasets' | 'proxies' | 'evaluate' | 'dossier' | 'code' | 'docs'>('simulator');

  const handleQuickRunSearch = () => {
    setActiveTab('search');
  };

  return (
    <ExperimentProvider>
      <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
        {/* Universal Top Bar */}
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onQuickRunSearch={handleQuickRunSearch}
        />

        {/* Main Content Area */}
        <main className="flex-1">
          {activeTab === 'simulator' && <TrajectorySimulator />}
          {activeTab === 'search' && <SearchStudio />}
          {activeTab === 'datasets' && <DatasetsViewer />}
          {activeTab === 'proxies' && <ZeroCostProxies />}
          {activeTab === 'evaluate' && <EvaluationBenchmarking />}
          {activeTab === 'dossier' && <ProposalDossier />}
          {activeTab === 'code' && <CodeExplorer />}
          {activeTab === 'docs' && <JuniorDevDocumentation />}
        </main>

      {/* Clean Scientific Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 px-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-mono text-slate-300 font-semibold">TrajectoryLNAS</span>
            <span>·</span>
            <span>پایان‌نامه کارشناسی ارشد مهندسی کامپیوتر هوش مصنوعی</span>
            <span>·</span>
            <span>دانشجو: علیرضا غفوری</span>
          </div>

          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span>شماره پژوهش: ۲۸۲۴۵۲۵۹</span>
            <span>ایرانداک: ۱۹۸۷۵۶۸</span>
            <span>سال تحصیلی: ۱۴۰۳-۱۴۰۴</span>
          </div>
        </div>
      </footer>
    </div>
    </ExperimentProvider>
  );
}
