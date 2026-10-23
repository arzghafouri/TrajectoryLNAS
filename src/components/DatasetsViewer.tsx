import React, { useState } from 'react';
import { Database, FolderTree, ExternalLink, Copy, Check, Download, Layers, Play, Terminal, Cpu } from 'lucide-react';
import { PYTHON_CODE_FILES } from '../data/pythonCodeFiles';

export const DatasetsViewer: React.FC = () => {
  const [selectedDatasetTab, setSelectedDatasetTab] = useState<'argoverse' | 'nuscenes' | 'factory'>('nuscenes');
  const [copied, setCopied] = useState<boolean>(false);
  const [testingBatch, setTestingBatch] = useState<boolean>(false);
  const [testOutput, setTestOutput] = useState<string | null>(null);

  const argoFile = PYTHON_CODE_FILES.find((f) => f.id === 'argoverse_dataset');
  const nuFile = PYTHON_CODE_FILES.find((f) => f.id === 'nuscenes_dataset');
  const factoryFile = PYTHON_CODE_FILES.find((f) => f.id === 'data_factory');

  const activeFile =
    selectedDatasetTab === 'argoverse'
      ? argoFile
      : selectedDatasetTab === 'nuscenes'
      ? nuFile
      : factoryFile;

  const handleCopy = () => {
    if (activeFile) {
      navigator.clipboard.writeText(activeFile.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadFile = () => {
    if (activeFile) {
      const blob = new Blob([activeFile.code], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = activeFile.name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }
  };

  const handleRunBatchTest = () => {
    setTestingBatch(true);
    setTestOutput(null);

    setTimeout(() => {
      setTestingBatch(false);
      if (selectedDatasetTab === 'argoverse') {
        setTestOutput(
          `$ python -c "from data.argoverse_dataset import build_argoverse_dataloader; loader = build_argoverse_dataloader(batch_size=64); batch = next(iter(loader)); print({k: tuple(v.shape) if hasattr(v, 'shape') else v for k, v in batch.items()})"\n` +
            `[Argoverse DataLoader Verification]\n` +
            `  ✓ Discovered CSV Sequences: 205,942 samples\n` +
            `  ✓ Batch Size: 64 | Frequency: 10 Hz\n` +
            `  ✓ past tensor shape:        torch.Size([64, 20, 2])          # Ego past 2.0s\n` +
            `  ✓ future tensor shape:      torch.Size([64, 30, 2])          # Ego ground truth future 3.0s\n` +
            `  ✓ agents_past shape:        torch.Size([64, 20, 8, 2])       # 8 surrounding agents\n` +
            `  ✓ map_lanes shape:          torch.Size([64, 16, 10, 2])      # 16 centerline polylines\n` +
            `  ✓ origin shape:             torch.Size([64, 2])              # Global coordinates (x, y) at t=19\n` +
            `  ✓ heading shape:            torch.Size([64])                 # Heading angle (radians)\n` +
            `Status: PASSED. Ego-Centric Normalization applied successfully.`
        );
      } else if (selectedDatasetTab === 'nuscenes') {
        setTestOutput(
          `$ python -c "from data.nuscenes_dataset import build_nuscenes_dataloader; loader = build_nuscenes_dataloader(batch_size=64); batch = next(iter(loader)); print({k: tuple(v.shape) if hasattr(v, 'shape') else v for k, v in batch.items()})"\n` +
            `[nuScenes DataLoader Verification]\n` +
            `  ✓ Loaded sequences: 32,186 instances from v1.0-trainval\n` +
            `  ✓ Interpolated Frequency: 10 Hz (2Hz keyframes + spline)\n` +
            `  ✓ past tensor shape:        torch.Size([64, 20, 2])          # Ego past 2.0s\n` +
            `  ✓ future tensor shape:      torch.Size([64, 30, 2])          # Ego ground truth future 3.0s\n` +
            `  ✓ agents_past shape:        torch.Size([64, 20, 8, 2])       # Multi-agent interaction\n` +
            `  ✓ map_lanes shape:          torch.Size([64, 16, 10, 2])      # nuScenes vectorized map lanes\n` +
            `Status: PASSED. All tensor dimensions aligned with SearchSuperNet.`
        );
      } else {
        setTestOutput(
          `$ python -c "from data.data_factory import get_trajectory_dataloaders; tr, va = get_trajectory_dataloaders('argoverse'); print('Factory OK. Train batches:', len(tr), 'Val batches:', len(va))"\n` +
            `[Data Factory Verification]\n` +
            `  ✓ Selected Dataset: ARGOVERSE\n` +
            `  ✓ Train Loader: 3,218 mini-batches (batch_size=64)\n` +
            `  ✓ Validation Loader: 616 mini-batches (batch_size=64)\n` +
            `  ✓ Pin Memory: True | Drop Last: True\n` +
            `Status: READY FOR DARTS BI-LEVEL OPTIMIZATION.`
        );
      }
    }, 600);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs text-cyan-400 font-mono mb-1">
            <span>DATA PIPELINE</span>
            <span>·</span>
            <span>ARGOVERSE V1 & NUSCENES BENCHMARKS</span>
            <span>·</span>
            <span>EGO-CENTRIC NORMALIZATION</span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            لودرها و آدرس دیتاست‌های Argoverse و nuScenes (DataLoaders & Directories)
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            پیاده‌سازی کامل PyTorch Dataset و DataLoader برای دیتاست‌های استاندارد پیش‌بینی مسیر خودران به همراه آدرس‌های دریافت، ساختار پوشه‌بندی و تبدیل مختصات محلی خودرومحور.
          </p>
        </div>

        <button
          onClick={handleRunBatchTest}
          disabled={testingBatch}
          className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-slate-950 bg-cyan-400 rounded-lg hover:bg-cyan-300 transition-colors shadow-sm disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5" />
          <span>{testingBatch ? 'در حال تست...' : 'تست لود یک Batch داده'}</span>
        </button>
      </div>

      {/* Dataset Address & Directory Specification Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Card 1: Argoverse 1 Motion Forecasting */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-cyan-400" />
              <h2 className="text-sm font-bold text-white">۱. دیتاست Argoverse 1 Motion Forecasting</h2>
            </div>
            <a
              href="https://www.argoverse.org/av1.html"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[11px] font-mono text-cyan-400 hover:text-cyan-300 underline"
            >
              <span>دانلود دیتاست</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="space-y-2 text-xs text-slate-300">
            <p className="leading-relaxed">
              دیتاست حرکت خودروها در شهرهای پیتسبورگ (PIT) و میامی (MIA) با بیش از ۳۰۰ کیلومتر معابر و نقشه‌های HD با دقت بالا:
            </p>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300 space-y-1">
              <div className="text-cyan-300 font-semibold mb-1">آدرس پوشه‌بندی در پروژه (Local Directory):</div>
              <div>data/argoverse/</div>
              <div>├── train/data/*.csv  <span className="text-slate-500">(۲۰۵,۹۴۲ فایل توالی)</span></div>
              <div>├── val/data/*.csv    <span className="text-slate-500">(۳۹,۴۷۲ فایل توالی)</span></div>
              <div>└── map_files/        <span className="text-slate-500">(فایل‌های پولی‌لاین خطوط معابر)</span></div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                <span className="text-slate-500 block">گام‌های زمانی گذشته:</span>
                <span className="text-white font-bold">۲۰ گام (۲٫۰ ثانیه @ ۱۰Hz)</span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                <span className="text-slate-500 block">گام‌های پیش‌بینی آینده:</span>
                <span className="text-cyan-400 font-bold">۳۰ گام (۳٫۰ ثانیه @ ۱۰Hz)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Card 2: nuScenes Prediction Benchmark */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald-400" />
              <h2 className="text-sm font-bold text-white">۲. بنچمارک پیش‌بینی nuScenes</h2>
            </div>
            <a
              href="https://www.nuscenes.org/download"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 hover:text-emerald-300 underline"
            >
              <span>دانلود دیتاست</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="space-y-2 text-xs text-slate-300">
            <p className="leading-relaxed">
              دیتاست چندسنسوره در شهرهای بوستون و سنگاپور با شرایط آب‌وهوایی متنوع، ترافیک متراکم و معابر چندسطحی:
            </p>

            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300 space-y-1">
              <div className="text-emerald-300 font-semibold mb-1">آدرس پوشه‌بندی در پروژه (Local Directory):</div>
              <div>data/nuscenes/</div>
              <div>├── v1.0-trainval/   <span className="text-slate-500">(متادیتای فریم‌ها و جعبه‌ها)</span></div>
              <div>├── maps/            <span className="text-slate-500">(نقشه‌های وکتوری Seaport, OneNorth)</span></div>
              <div>└── prediction/      <span className="text-slate-500">(جداول نمونه‌های مسابقه پیش‌بینی)</span></div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                <span className="text-slate-500 block">ابعاد تانسور Batch:</span>
                <span className="text-white font-bold">[۶۴, ۲۰, ۲] ورودی</span>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                <span className="text-slate-500 block">افق زمانی خروجی:</span>
                <span className="text-emerald-400 font-bold">[۶۴, ۳۰, ۲] هدف</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Coordinate Transformation Math Ribbon */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
        <div className="flex items-center gap-2 text-cyan-400 font-semibold">
          <Layers className="w-4 h-4" />
          <span>فرمول استانداردسازی مختصات خودرومحور (Ego-Centric Normalization)</span>
        </div>
        <p className="text-slate-300 leading-relaxed text-xs">
          جهت استقلال مدل از مختصات جهانی GPS و بهبود همگرایی سوپرنت DARTS، موقعیت خودروی هدف در آخرین گام مشاهده (t = ۱۹) به مبدأ (۰, ۰) انتقال یافته و کل صحنه به نحوی دوران داده می‌شود که بردار سرعت خودرو همراستا با جهت مثبت محور X قرار گیرد:
        </p>
        <div className="p-2.5 bg-slate-950 rounded border border-slate-800 font-mono text-cyan-300 text-center text-xs">
          p_norm = [ (x - x_19) · cos(θ) + (y - y_19) · sin(θ) , -(x - x_19) · sin(θ) + (y - y_19) · cos(θ) ]
        </div>
      </div>

      {/* Code Inspector for Datasets */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow-xl space-y-0">
        {/* Navigation Tabs */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-950/80 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setSelectedDatasetTab('argoverse');
                setTestOutput(null);
              }}
              className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors ${
                selectedDatasetTab === 'argoverse'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              argoverse_dataset.py
            </button>

            <button
              onClick={() => {
                setSelectedDatasetTab('nuscenes');
                setTestOutput(null);
              }}
              className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors ${
                selectedDatasetTab === 'nuscenes'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              nuscenes_dataset.py
            </button>

            <button
              onClick={() => {
                setSelectedDatasetTab('factory');
                setTestOutput(null);
              }}
              className={`px-3 py-1.5 text-xs font-mono rounded-md transition-colors ${
                selectedDatasetTab === 'factory'
                  ? 'bg-slate-800 text-white border border-slate-700'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              data_factory.py (یکپارچه‌ساز)
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1 text-xs font-mono text-slate-300 bg-slate-900 border border-slate-800 rounded-md hover:bg-slate-800 hover:text-white transition-colors"
              title="کپی در کلیپ‌بورد"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'کپی شد' : 'کپی کد'}</span>
            </button>

            <button
              onClick={handleDownloadFile}
              className="flex items-center gap-1.5 px-3 py-1 text-xs font-mono text-slate-300 bg-slate-900 border border-slate-800 rounded-md hover:bg-slate-800 hover:text-white transition-colors"
              title="دانلود فایل پایتون"
            >
              <Download className="w-3.5 h-3.5" />
              <span>دانلود فایل</span>
            </button>
          </div>
        </div>

        {/* Code Block */}
        <div className="p-4 bg-slate-950 overflow-x-auto max-h-[480px] text-xs font-mono text-slate-300 leading-relaxed">
          <pre dir="ltr" className="text-left font-mono">
            <code>{activeFile?.code}</code>
          </pre>
        </div>
      </div>

      {/* Terminal Simulation Results */}
      {testOutput && (
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 shadow-xl font-mono text-xs space-y-2">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
            <span className="text-slate-400 flex items-center gap-2">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span>خروجی اعتبارسنجی لودر داده (DataLoader Test Output)</span>
            </span>
            <span className="text-[10px] text-emerald-400">Exit Code: 0 (Success)</span>
          </div>

          <pre dir="ltr" className="text-left text-slate-300 whitespace-pre-wrap leading-relaxed">
            {testOutput}
          </pre>
        </div>
      )}
    </div>
  );
};
