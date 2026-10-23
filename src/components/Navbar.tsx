import React, { useState } from 'react';
import { Download, Cpu, Play, Upload, CheckCircle2, Menu, X, Layers, Activity, Database, Server, BarChart3, BookOpen, Code2, GraduationCap } from 'lucide-react';
import JSZip from 'jszip';
import { PYTHON_CODE_FILES } from '../data/pythonCodeFiles';
import { useExperiment } from '../context/ExperimentContext';

interface NavbarProps {
  activeTab: 'simulator' | 'search' | 'datasets' | 'proxies' | 'evaluate' | 'dossier' | 'code' | 'docs';
  setActiveTab: (tab: 'simulator' | 'search' | 'datasets' | 'proxies' | 'evaluate' | 'dossier' | 'code' | 'docs') => void;
  onQuickRunSearch: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, onQuickRunSearch }) => {
  const [downloading, setDownloading] = React.useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { loadJsonData, isRealDataLoaded } = useExperiment();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        const parsed = JSON.parse(text);
        const success = loadJsonData(parsed);
        if (success) {
          alert('اطلاعات واقعی آزمایش با موفقیت بارگذاری شد و در تمامی تب‌ها اعمال گردید!');
        } else {
          alert('خطا در خواندن فایل JSON. لطفا فایل real_experiment_results.json را انتخاب کنید.');
        }
      } catch (err) {
        alert('فایل انتخاب‌شده یک فرمت معتبر JSON نیست.');
      }
    };
    reader.readAsText(file);
  };

  const handleDownloadZip = async () => {
    try {
      setDownloading(true);
      const zip = new JSZip();

      // Add models folder
      const modelsFolder = zip.folder('models');
      const opFile = PYTHON_CODE_FILES.find((f) => f.id === 'supernet_operators');
      const netFile = PYTHON_CODE_FILES.find((f) => f.id === 'search_supernet');
      const finalFile = PYTHON_CODE_FILES.find((f) => f.id === 'final_model');

      if (opFile && modelsFolder) modelsFolder.file('supernet_operators.py', opFile.code);
      if (netFile && modelsFolder) modelsFolder.file('search_supernet.py', netFile.code);
      if (finalFile && modelsFolder) modelsFolder.file('final_model.py', finalFile.code);

      // Add data folder with Argoverse & nuScenes loaders
      const dataFolder = zip.folder('data');
      const argoFile = PYTHON_CODE_FILES.find((f) => f.id === 'argoverse_dataset');
      const nuFile = PYTHON_CODE_FILES.find((f) => f.id === 'nuscenes_dataset');
      const factoryFile = PYTHON_CODE_FILES.find((f) => f.id === 'data_factory');

      if (argoFile && dataFolder) dataFolder.file('argoverse_dataset.py', argoFile.code);
      if (nuFile && dataFolder) dataFolder.file('nuscenes_dataset.py', nuFile.code);
      if (factoryFile && dataFolder) dataFolder.file('data_factory.py', factoryFile.code);
      const prepareDataFile = PYTHON_CODE_FILES.find((f) => f.id === 'prepare_data');
      if (prepareDataFile && dataFolder) dataFolder.file('prepare_data.py', prepareDataFile.code);

      // Add config and core folders (OOP Architecture)
      const configFolder = zip.folder('config');
      const searchConfigDoc = PYTHON_CODE_FILES.find((f) => f.id === 'search_config');
      if (configFolder && searchConfigDoc) configFolder.file('search_config.py', searchConfigDoc.code);

      const coreFolder = zip.folder('core');
      const baseTrainerDoc = PYTHON_CODE_FILES.find((f) => f.id === 'base_trainer');
      const dartsSearcherDoc = PYTHON_CODE_FILES.find((f) => f.id === 'darts_searcher');
      const finalTrainerDoc = PYTHON_CODE_FILES.find((f) => f.id === 'final_trainer');
      if (coreFolder && baseTrainerDoc) coreFolder.file('base_trainer.py', baseTrainerDoc.code);
      if (coreFolder && dartsSearcherDoc) coreFolder.file('darts_searcher.py', dartsSearcherDoc.code);
      if (coreFolder && finalTrainerDoc) coreFolder.file('final_trainer.py', finalTrainerDoc.code);

      // Add train folder (modular search and retraining)
      const trainFolder = zip.folder('train');
      const searchFile = PYTHON_CODE_FILES.find((f) => f.id === 'search_script');
      const trainFinalFile = PYTHON_CODE_FILES.find((f) => f.id === 'train_final_script');
      if (trainFolder && searchFile) trainFolder.file('search.py', searchFile.code);
      if (trainFolder && trainFinalFile) trainFolder.file('train_final.py', trainFinalFile.code);
      if (trainFolder) trainFolder.file('__init__.py', 'from train.search import run_architecture_search\nfrom train.train_final import run_final_training\n');

      // Add evaluate folder (modular metrics, profiling, and plotting)
      const evaluateFolder = zip.folder('evaluate');
      const metricsDoc = PYTHON_CODE_FILES.find((f) => f.id === 'eval_metrics');
      const plotterDoc = PYTHON_CODE_FILES.find((f) => f.id === 'eval_plotter');
      const evalFile = PYTHON_CODE_FILES.find((f) => f.id === 'evaluate_script');
      if (evaluateFolder && metricsDoc) evaluateFolder.file('metrics.py', metricsDoc.code);
      if (evaluateFolder && plotterDoc) evaluateFolder.file('plotter.py', plotterDoc.code);
      if (evaluateFolder && evalFile) evaluateFolder.file('evaluator.py', evalFile.code);
      if (evaluateFolder) evaluateFolder.file('__init__.py', 'from evaluate.metrics import ade, fde, miss_rate\nfrom evaluate.plotter import plot_horizon_error_curve, plot_benchmark_comparison\nfrom evaluate.evaluator import TrajectoryEvaluator\n');

      // Root files (Clean Architecture)
      const mainFile = PYTHON_CODE_FILES.find((f) => f.id === 'main_script');
      const utilsFile = PYTHON_CODE_FILES.find((f) => f.id === 'utils_script');

      if (mainFile) zip.file('main.py', mainFile.code);
      if (utilsFile) zip.file('utils.py', utilsFile.code);

      // Report folder (dedicated reports, logs, and plots)
      const reportFolder = zip.folder('report');
      const logFile = PYTHON_CODE_FILES.find((f) => f.id === 'search_log');
      if (reportFolder && logFile) {
        reportFolder.file('search_execution.log', logFile.code);
      }

      zip.file(
        'requirements.txt',
        'torch>=2.0.0\nnumpy>=1.24.0\npandas>=2.0.0\nscipy>=1.10.0\nmatplotlib>=3.7.0\ntqdm>=4.65.0\n'
      );
      zip.file(
        'README.md',
        '# TrajectoryLNAS: Lightweight Neural Architecture Search for Trajectory Prediction\n\n' +
          'Master Thesis Project: Alireza Ghafouri (Student ID: 4031631209)\n' +
          'Supervisors: Dr. Zeinab Ghasemi Naraaghi & Dr. Zeinab Abedini\n\n' +
          '## Dataset Directories:\n' +
          '- Argoverse 1: data/argoverse/train/data/ (Download: https://www.argoverse.org/av1.html)\n' +
          '- nuScenes:    data/nuscenes/v1.0-trainval/ (Download: https://www.nuscenes.org/download)\n\n' +
          '## How to run:\n' +
          '1. pip install -r requirements.txt\n' +
          '2. python search.py --dataset argoverse --epochs 30 --lambda_val 0.01\n' +
          '3. python train_final.py --dataset argoverse\n' +
          '4. python evaluate.py\n'
      );

      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'TrajectoryLNAS_AlirezaGhafouri_PyTorch.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to generate ZIP:', err);
    } finally {
      setDownloading(false);
    }
  };

  const navLinks = [
    { id: 'simulator', label: 'شبیه‌ساز و پیش‌بینی مسیر', icon: Layers },
    { id: 'search', label: 'استودیو جستجوی DARTS', icon: Activity },
    { id: 'datasets', label: 'دیتاست‌ها و DataLoader', icon: Database },
    { id: 'proxies', label: 'پروکسی‌ها و سخت‌افزار لبه', icon: Server },
    { id: 'evaluate', label: 'ارزیابی و بنچمارک‌ها', icon: BarChart3 },
    { id: 'dossier', label: 'شناسنامه کاربرگ ۲', icon: BookOpen },
    { id: 'code', label: 'مخزن کد پایتون', icon: Code2 },
    { id: 'docs', label: 'داکیومنت و راهنمای سورس', icon: GraduationCap },
  ] as const;

  return (
    <header className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-md border-b border-slate-800/80">
      <div className="flex items-center justify-between px-4 sm:px-6 py-3">
        {/* Zone 1: Wordmark Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
            <Cpu className="w-4 h-4" />
          </div>
          <span className="text-base font-bold tracking-tight text-white font-mono">
            TrajectoryLNAS
          </span>
        </div>

        {/* Zone 2: Desktop Navigation Bar (Full screen >= 1280px) */}
        <nav className="hidden xl:flex items-center gap-1 bg-slate-900/60 p-1 rounded-lg border border-slate-800">
          {navLinks.map((link) => (
            <button
              key={link.id}
              onClick={() => setActiveTab(link.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 whitespace-nowrap ${
                activeTab === link.id
                  ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {link.label}
            </button>
          ))}
        </nav>

        {/* Zone 3: Actions + Mobile Hamburger Toggle */}
        <div className="flex items-center gap-2">
          {/* Upload JSON Button */}
          <label className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs font-medium text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 rounded-md hover:bg-emerald-900/40 hover:text-emerald-300 transition-colors whitespace-nowrap cursor-pointer shadow-sm">
            <Upload className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            <span className="hidden sm:inline">بارگذاری نتایج واقعی (.JSON)</span>
            <span className="sm:hidden">آپلود JSON</span>
            <input
              type="file"
              accept=".json,.log"
              className="hidden"
              onChange={handleFileUpload}
            />
          </label>

          {/* Download Zip */}
          <button
            onClick={handleDownloadZip}
            disabled={downloading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-950 bg-cyan-400 rounded-md hover:bg-cyan-300 transition-colors whitespace-nowrap shadow-sm disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5 shrink-0" />
            <span className="hidden md:inline">{downloading ? 'در حال ایجاد...' : 'دانلود پکیج کامل (.ZIP)'}</span>
            <span className="md:hidden">ZIP</span>
          </button>

          {/* Mobile Menu Hamburger (Visible on screens < xl) */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="xl:hidden p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800 transition-colors"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Responsive Horizontal Scrollable Tab Bar for Medium Screens (Tablets / Small Laptops 768px - 1280px) */}
      <div className="hidden md:flex xl:hidden px-4 pb-2.5 overflow-x-auto gap-1 border-t border-slate-900/60 pt-2 no-scrollbar">
        {navLinks.map((link) => (
          <button
            key={link.id}
            onClick={() => setActiveTab(link.id)}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 whitespace-nowrap shrink-0 ${
              activeTab === link.id
                ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900/40'
            }`}
          >
            {link.label}
          </button>
        ))}
      </div>

      {/* Mobile Drawer Menu (Phones < 768px) */}
      {mobileMenuOpen && (
        <div className="xl:hidden border-t border-slate-800 bg-slate-950/98 px-4 py-3 space-y-1.5 animate-in slide-in-from-top-2 duration-200 shadow-2xl">
          <div className="text-[11px] font-mono text-cyan-400 px-2 py-1">فهرست صفحات و ماژول‌های اپلیکیشن:</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {navLinks.map((link) => {
              const IconComp = link.icon;
              const isActive = activeTab === link.id;
              return (
                <button
                  key={link.id}
                  onClick={() => {
                    setActiveTab(link.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`flex items-center gap-2.5 px-3 py-2 text-xs font-medium rounded-lg transition-all text-right ${
                    isActive
                      ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 font-semibold'
                      : 'text-slate-300 hover:bg-slate-900 hover:text-white border border-transparent'
                  }`}
                >
                  <IconComp className={`w-4 h-4 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span>{link.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
};
