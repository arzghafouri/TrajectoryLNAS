import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Eye, Navigation, AlertTriangle, Layers, Info } from 'lucide-react';
import { MOCK_SCENARIOS } from '../data/mockScenarios';
import { TrajectoryPoint } from '../types/trajectory';

export const TrajectorySimulator: React.FC = () => {
  const [selectedScenarioId, setSelectedScenarioId] = useState<string>('s1');
  const [currentStep, setCurrentStep] = useState<number>(20); // 0 to 20 = past, 21 to 50 = future
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [visibleModels, setVisibleModels] = useState({
    groundTruth: true,
    lnas: true,
    nas: true,
    vanilla: true,
  });
  const [showMultiModal, setShowMultiModal] = useState<boolean>(true);

  const scenario = MOCK_SCENARIOS.find((s) => s.id === selectedScenarioId) || MOCK_SCENARIOS[0];
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= 49) {
            setIsPlaying(false);
            return 49;
          }
          return prev + 1;
        });
      }, 80);
      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentStep(20);
  };

  // Convert array of points to SVG polyline string
  const pointsToPolyline = (pts: TrajectoryPoint[]): string => {
    return pts.map((p) => `${p.x},${p.y}`).join(' ');
  };

  // Calculate live metric distances
  const getADE = (pred: TrajectoryPoint[], gt: TrajectoryPoint[]): number => {
    let sum = 0;
    const count = Math.min(pred.length, gt.length);
    for (let i = 0; i < count; i++) {
      const dx = (pred[i].x - gt[i].x) * 0.15; // scale to meters
      const dy = (pred[i].y - gt[i].y) * 0.15;
      sum += Math.sqrt(dx * dx + dy * dy);
    }
    return Math.round((sum / count) * 100) / 100;
  };

  const getFDE = (pred: TrajectoryPoint[], gt: TrajectoryPoint[]): number => {
    const lastP = pred[pred.length - 1];
    const lastG = gt[gt.length - 1];
    const dx = (lastP.x - lastG.x) * 0.15;
    const dy = (lastP.y - lastG.y) * 0.15;
    return Math.round(Math.sqrt(dx * dx + dy * dy) * 100) / 100;
  };

  const lnasADE = getADE(scenario.egoLnasPred, scenario.egoGroundTruthFuture);
  const lnasFDE = getFDE(scenario.egoLnasPred, scenario.egoGroundTruthFuture);

  const nasADE = getADE(scenario.egoNasPred, scenario.egoGroundTruthFuture);
  const nasFDE = getFDE(scenario.egoNasPred, scenario.egoGroundTruthFuture);

  const vanillaADE = getADE(scenario.egoVanillaPred, scenario.egoGroundTruthFuture);
  const vanillaFDE = getFDE(scenario.egoVanillaPred, scenario.egoGroundTruthFuture);

  // Current active positions at currentStep
  const getEgoPositionAtStep = (): { x: number; y: number } => {
    if (currentStep < 20) {
      return scenario.egoPast[currentStep] || scenario.egoPast[0];
    } else {
      const futureIndex = currentStep - 20;
      return (
        scenario.egoGroundTruthFuture[futureIndex] ||
        scenario.egoGroundTruthFuture[scenario.egoGroundTruthFuture.length - 1]
      );
    }
  };

  const currentEgoPos = getEgoPositionAtStep();

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Top Header & Scenario Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs text-cyan-400 font-mono mb-1">
            <span>BENCHMARK DATASET</span>
            <span>·</span>
            <span>ARGOVERSE / NUSCENES MULTI-AGENT</span>
            <span>·</span>
            <span>HORIZON: 2.0s PAST (20 STEPS) → 3.0s FUTURE (30 STEPS)</span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight">
            شبیه‌ساز تعاملی پیش‌بینی مسیر خودران (Autonomous Motion Simulator)
          </h1>
          <p className="text-sm text-slate-400 mt-1 max-w-3xl">
            مقایسه بلادرنگ خروجی مدل کشف‌شده TrajectoryLNAS در برابر نسخه سنگین TrajectoryNAS (روش MOSA) و مدل پایه LSTM در سناریوهای ترافیکی با چگالی چندعاملی و محدودیت نقشه معابر.
          </p>
        </div>

        {/* Scenario Select Buttons */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-900 border border-slate-800 rounded-lg">
          {MOCK_SCENARIOS.map((s) => (
            <button
              key={s.id}
              onClick={() => {
                setSelectedScenarioId(s.id);
                handleReset();
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                selectedScenarioId === s.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {s.id === 's1' ? 'تقاطع چهارراه شهری (Argoverse)' : 'سبقت سریع در بزرگراه (nuScenes)'}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Visualizer on Left (Flex-1) + Control Telemetry Panel on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Canvas Stage (8 cols) */}
        <div className="lg:col-span-8 bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col justify-between overflow-hidden shadow-xl">
          {/* Top Canvas Status Ribbon */}
          <div className="flex items-center justify-between text-xs font-mono text-slate-400 pb-3 border-b border-slate-800/80 mb-3">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>
                {currentStep < 20
                  ? `مشاهده گذشته (Past Step: ${currentStep + 1}/20)`
                  : `پیش‌بینی آینده (Prediction Step: ${currentStep - 19}/30)`}
              </span>
              <span>·</span>
              <span className="text-slate-300 font-bold tabular-nums">
                T = {((currentStep - 20) * 0.1).toFixed(1)}s
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-slate-500">مختصات خودرو هدف:</span>
              <span className="text-cyan-400 tabular-nums">
                X: {currentEgoPos.x.toFixed(1)}m, Y: {currentEgoPos.y.toFixed(1)}m
              </span>
            </div>
          </div>

          {/* SVG Map & Trajectory Stage */}
          <div className="relative w-full aspect-[4/3] max-h-[520px] bg-slate-950 rounded-lg border border-slate-800/60 overflow-hidden flex items-center justify-center">
            <svg
              viewBox="0 0 600 600"
              className="w-full h-full select-none"
              style={{ background: '#070b13' }}
            >
              {/* Grid Lines */}
              <defs>
                <pattern id="roadGrid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#131c2e" strokeWidth="0.8" />
                </pattern>
                {/* Glow Filter for TrajectoryLNAS */}
                <filter id="cyanGlow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              <rect width="600" height="600" fill="url(#roadGrid)" />

              {/* HD Map Lanes & Road Boundaries */}
              {scenario.mapLanes.map((lane) => {
                const ptsStr = lane.points.map((p) => `${p[0]},${p[1]}`).join(' ');
                if (lane.type === 'boundary') {
                  return (
                    <polyline
                      key={lane.id}
                      points={ptsStr}
                      fill="none"
                      stroke="#334155"
                      strokeWidth="3.5"
                    />
                  );
                }
                if (lane.type === 'crosswalk') {
                  return (
                    <line
                      key={lane.id}
                      x1={lane.points[0][0]}
                      y1={lane.points[0][1]}
                      x2={lane.points[1][0]}
                      y2={lane.points[1][1]}
                      stroke="#475569"
                      strokeWidth="12"
                      strokeDasharray="4 4"
                    />
                  );
                }
                return (
                  <polyline
                    key={lane.id}
                    points={ptsStr}
                    fill="none"
                    stroke="#1e293b"
                    strokeWidth="2"
                    strokeDasharray="6 6"
                  />
                );
              })}

              {/* Neighbors Past & Future */}
              {scenario.neighbors.map((neighbor) => {
                const pastPts = neighbor.past;
                const futurePts = neighbor.future;
                const lastPast = pastPts[pastPts.length - 1];

                return (
                  <g key={neighbor.id}>
                    {/* Past path */}
                    <polyline
                      points={pointsToPolyline(pastPts)}
                      fill="none"
                      stroke={neighbor.color}
                      strokeWidth="1.5"
                      strokeOpacity="0.4"
                    />
                    {/* Future ground truth */}
                    <polyline
                      points={pointsToPolyline(futurePts)}
                      fill="none"
                      stroke={neighbor.color}
                      strokeWidth="2"
                      strokeDasharray="3 3"
                      strokeOpacity="0.7"
                    />
                    {/* Current Position Marker */}
                    <circle
                      cx={lastPast.x}
                      cy={lastPast.y}
                      r={neighbor.type === 'pedestrian' ? 4 : 7}
                      fill={neighbor.color}
                      stroke="#ffffff"
                      strokeWidth="1.5"
                    />
                  </g>
                );
              })}

              {/* Ego Past Trajectory (History 20 steps) */}
              <polyline
                points={pointsToPolyline(scenario.egoPast)}
                fill="none"
                stroke="#06b6d4"
                strokeWidth="2.5"
                strokeOpacity="0.5"
              />
              {scenario.egoPast.map((p, idx) => (
                <circle
                  key={`past-${idx}`}
                  cx={p.x}
                  cy={p.y}
                  r="2"
                  fill="#06b6d4"
                  fillOpacity="0.6"
                />
              ))}

              {/* Predicted Trajectory: Vanilla LSTM (Purple) */}
              {visibleModels.vanilla && (
                <polyline
                  points={pointsToPolyline(scenario.egoVanillaPred)}
                  fill="none"
                  stroke="#a855f7"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                  strokeOpacity="0.75"
                />
              )}

              {/* Predicted Trajectory: TrajectoryNAS MOSA (Orange) */}
              {visibleModels.nas && (
                <polyline
                  points={pointsToPolyline(scenario.egoNasPred)}
                  fill="none"
                  stroke="#f97316"
                  strokeWidth="2.2"
                  strokeDasharray="5 3"
                  strokeOpacity="0.85"
                />
              )}

              {/* Predicted Trajectory: Ground Truth (White dashed) */}
              {visibleModels.groundTruth && (
                <polyline
                  points={pointsToPolyline(scenario.egoGroundTruthFuture)}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth="2"
                  strokeDasharray="4 4"
                  strokeOpacity="0.8"
                />
              )}

              {/* Multi-Modal K=6 Candidate Modes */}
              {showMultiModal && scenario.candidateModes && scenario.candidateModes.map((mode) => (
                <g key={`mode-${mode.id}`}>
                  <polyline
                    points={pointsToPolyline(mode.points)}
                    fill="none"
                    stroke={mode.color}
                    strokeWidth={mode.id === 1 ? "3" : "1.8"}
                    strokeDasharray={mode.id === 1 ? undefined : "5 3"}
                    strokeOpacity={mode.id === 1 ? 0.95 : 0.6}
                  />
                  <circle
                    cx={mode.points[mode.points.length - 1].x}
                    cy={mode.points[mode.points.length - 1].y}
                    r={mode.id === 1 ? 4.5 : 3}
                    fill={mode.color}
                  />
                </g>
              ))}

              {/* Predicted Trajectory: TrajectoryLNAS (Ours - Cyan glowing) when not showing multi-modal modes */}
              {visibleModels.lnas && !showMultiModal && (
                <g filter="url(#cyanGlow)">
                  <polyline
                    points={pointsToPolyline(scenario.egoLnasPred)}
                    fill="none"
                    stroke="#22d3ee"
                    strokeWidth="3"
                  />
                  {scenario.egoLnasPred.map((p, idx) => (
                    <circle key={`lnas-${idx}`} cx={p.x} cy={p.y} r="2.5" fill="#22d3ee" />
                  ))}
                </g>
              )}

              {/* Animated Current Ego Vehicle Marker */}
              <g transform={`translate(${currentEgoPos.x}, ${currentEgoPos.y})`}>
                <circle r="14" fill="#06b6d4" fillOpacity="0.2" className="animate-ping" />
                <rect
                  x="-7"
                  y="-12"
                  width="14"
                  height="24"
                  rx="3"
                  fill="#22d3ee"
                  stroke="#ffffff"
                  strokeWidth="2"
                  className="shadow-lg"
                />
                <circle cx="0" cy="-6" r="2" fill="#083344" />
              </g>
            </svg>

            {/* In-canvas Legend */}
            <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur-md border border-slate-800 p-2.5 rounded-lg text-[11px] font-mono space-y-1.5 shadow-lg">
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-cyan-400" />
                <span className="text-cyan-300 font-bold">TrajectoryLNAS (مدل سبک ما)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-white border-t border-dashed" />
                <span className="text-slate-300">مسیر واقعی (Ground Truth)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-orange-500 border-t border-dashed" />
                <span className="text-orange-300">TrajectoryNAS (سنگین MOSA)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-purple-500 border-t border-dashed" />
                <span className="text-purple-300">پایه Vanilla LSTM</span>
              </div>
            </div>
          </div>

          {/* Scrubber & Player Controls Bar */}
          <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-col md:flex-row items-center gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 rounded-lg bg-cyan-500 text-slate-950 flex items-center justify-center hover:bg-cyan-400 transition-colors shadow-sm"
                title={isPlaying ? 'توقف' : 'پخش'}
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
              </button>

              <button
                onClick={handleReset}
                className="w-8 h-8 rounded-lg bg-slate-800 text-slate-300 flex items-center justify-center hover:bg-slate-700 hover:text-white transition-colors"
                title="بازنشانی به گام ۲۰"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Slider Scrubber */}
            <div className="flex-1 flex items-center gap-3 w-full">
              <span className="text-xs font-mono text-slate-400 tabular-nums">t=0</span>
              <input
                type="range"
                min="0"
                max="49"
                value={currentStep}
                onChange={(e) => {
                  setCurrentStep(Number(e.target.value));
                  setIsPlaying(false);
                }}
                className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <span className="text-xs font-mono text-slate-400 tabular-nums">t=5.0s</span>
            </div>

            {/* Layer Visibility Toggles */}
            <div className="flex items-center gap-2 text-xs">
              <label className="flex items-center gap-1.5 text-cyan-300 font-semibold cursor-pointer border-r border-slate-700 pr-3 mr-1">
                <input
                  type="checkbox"
                  checked={showMultiModal}
                  onChange={(e) => setShowMultiModal(e.target.checked)}
                  className="rounded border-cyan-700 bg-slate-800 text-cyan-400 focus:ring-0"
                />
                <span>چندمسیره K=6</span>
              </label>

              <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={visibleModels.lnas}
                  onChange={(e) =>
                    setVisibleModels({ ...visibleModels, lnas: e.target.checked })
                  }
                  className="rounded border-slate-700 bg-slate-800 text-cyan-500 focus:ring-0"
                />
                <span>LNAS</span>
              </label>

              <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={visibleModels.nas}
                  onChange={(e) =>
                    setVisibleModels({ ...visibleModels, nas: e.target.checked })
                  }
                  className="rounded border-slate-700 bg-slate-800 text-orange-500 focus:ring-0"
                />
                <span>NAS (MOSA)</span>
              </label>

              <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={visibleModels.vanilla}
                  onChange={(e) =>
                    setVisibleModels({ ...visibleModels, vanilla: e.target.checked })
                  }
                  className="rounded border-slate-700 bg-slate-800 text-purple-500 focus:ring-0"
                />
                <span>LSTM</span>
              </label>
            </div>
          </div>
        </div>

        {/* Telemetry & Metrics HUD (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Card 1: Live Benchmark Scoreboard */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3.5 shadow-xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                <Navigation className="w-4 h-4 text-cyan-400" />
                <span>سنجش خطای جابه‌جایی (Displacement Errors)</span>
              </h2>
              <span className="text-[11px] font-mono text-slate-500">افق ۳٫۰ ثانیه</span>
            </div>

            {/* Comparative Metric Rows */}
            <div className="space-y-2.5">
              {/* TrajectoryLNAS */}
              <div className="p-2.5 rounded-lg bg-cyan-950/20 border border-cyan-500/30">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-cyan-300">
                    TrajectoryLNAS (طرح ما)
                  </span>
                  <span className="text-[10px] font-mono text-cyan-400">DARTS + ZCP</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 pt-1.5 border-t border-cyan-500/20 font-mono">
                  <div>
                    <span className="text-[10px] text-slate-400 block">خطای میانگین (ADE)</span>
                    <span className="text-base font-bold text-white tabular-nums">
                      {lnasADE}{' '}
                      <span className="text-[10px] font-normal text-slate-400">متر</span>
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">خطای نهایی (FDE)</span>
                    <span className="text-base font-bold text-cyan-400 tabular-nums">
                      {lnasFDE}{' '}
                      <span className="text-[10px] font-normal text-slate-400">متر</span>
                    </span>
                  </div>
                </div>
              </div>

              {/* TrajectoryNAS */}
              <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-300">
                    TrajectoryNAS (نسخه اصلی)
                  </span>
                  <span className="text-[10px] font-mono text-orange-400">MOSA Search</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 pt-1.5 border-t border-slate-800 font-mono">
                  <div>
                    <span className="text-[10px] text-slate-500 block">ADE</span>
                    <span className="text-sm font-semibold text-slate-300 tabular-nums">
                      {nasADE} m
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">FDE</span>
                    <span className="text-sm font-semibold text-slate-300 tabular-nums">
                      {nasFDE} m
                    </span>
                  </div>
                </div>
              </div>

              {/* Vanilla LSTM */}
              <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400">Vanilla LSTM (پایه)</span>
                  <span className="text-[10px] font-mono text-purple-400">Manual</span>
                </div>
                <div className="grid grid-cols-2 gap-2 mt-2 pt-1.5 border-t border-slate-800 font-mono">
                  <div>
                    <span className="text-[10px] text-slate-500 block">ADE</span>
                    <span className="text-sm font-semibold text-slate-400 tabular-nums">
                      {vanillaADE} m
                    </span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">FDE</span>
                    <span className="text-sm font-semibold text-slate-400 tabular-nums">
                      {vanillaFDE} m
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Card: Multi-Modal K=6 Hypotheses & Probability Distribution */}
          {showMultiModal && scenario.candidateModes && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3 shadow-xl">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <span>توزیع احتمالاتی کاندیدها (K=6 Multi-Modal)</span>
                </h3>
                <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 px-2 py-0.5 rounded border border-cyan-800">
                  WTA Loss
                </span>
              </div>
              <div className="space-y-1.5 text-xs font-mono">
                {scenario.candidateModes.map((mode) => (
                  <div key={mode.id} className="flex items-center justify-between p-2 rounded bg-slate-950/70 border border-slate-800/80">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: mode.color }} />
                      <span className="text-slate-300 text-[11px] font-sans truncate max-w-[160px]">{mode.labelFa}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <div className="w-14 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${mode.probability * 100}%`,
                            backgroundColor: mode.color
                          }}
                        />
                      </div>
                      <span className="text-[10px] text-cyan-300 font-bold tabular-nums w-8 text-left">
                        {(mode.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Card 2: Computational Resource Gains (From Karbarg 2) */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3 shadow-xl">
            <h2 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>کاهش هزینه‌های محاسباتی (کاربرگ ۲)</span>
            </h2>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="text-slate-400">زمان جستجوی معماری:</span>
                <span className="font-mono text-emerald-400 font-semibold tabular-nums">
                  ۴٫۲ ساعت (در برابر ۱۲۰ ساعت MOSA)
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="text-slate-400">حجم عملیات شناور (FLOPs):</span>
                <span className="font-mono text-cyan-400 font-semibold tabular-nums">
                  ۰٫۹۴ گیگافلاپس (۵۵٪ سبک‌تر)
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="text-slate-400">تعداد پارامترهای فعال:</span>
                <span className="font-mono text-slate-200 font-semibold tabular-nums">
                  ۸۸۰ هزار پارامتر (۰٫۸۸M)
                </span>
              </div>

              <div className="flex items-center justify-between p-2 rounded bg-slate-950/60 border border-slate-800/80">
                <span className="text-slate-400">تأخیر روی NVIDIA Orin:</span>
                <span className="font-mono text-emerald-400 font-semibold tabular-nums">
                  ۳٫۲ میلی‌ثانیه (۳۱۲ FPS)
                </span>
              </div>
            </div>

            <div className="p-2.5 rounded-lg bg-slate-800/40 border border-slate-700/50 text-[11px] text-slate-300 leading-relaxed flex items-start gap-2">
              <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
              <span>
                طبق بند ۴-۷ کاربرگ پیشنهاد، در تابع هزینه به جای Cross Entropy از توابع خطای جابه‌جایی
                FDE و ADE بهره گرفته شده و جریمه سخت‌افزاری λ·FLOPs سبب انتخاب سبک‌ترین عملگرهای ممکن شده است.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
