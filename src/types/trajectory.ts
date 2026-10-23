export interface TrajectoryPoint {
  x: number;
  y: number;
  heading?: number;
  speed?: number;
  timestamp?: number;
}

export interface AgentScenario {
  id: string;
  name: string;
  nameFa: string;
  description: string;
  descriptionFa: string;
  roadType: 'intersection' | 'highway' | 'roundabout' | 'crosswalk';
  egoPast: TrajectoryPoint[];
  egoGroundTruthFuture: TrajectoryPoint[];
  egoLnasPred: TrajectoryPoint[];
  candidateModes?: {
    id: number;
    probability: number;
    color: string;
    labelFa: string;
    points: TrajectoryPoint[];
  }[];
  egoNasPred: TrajectoryPoint[];
  egoVanillaPred: TrajectoryPoint[];
  neighbors: {
    id: string;
    type: 'car' | 'pedestrian' | 'cyclist';
    color: string;
    past: TrajectoryPoint[];
    future: TrajectoryPoint[];
  }[];
  mapLanes: {
    id: string;
    points: [number, number][];
    type: 'centerline' | 'boundary' | 'crosswalk';
  }[];
}

export interface OperatorSpec {
  id: string;
  name: string;
  nameFa: string;
  category: 'temporal' | 'interaction' | 'map';
  params: number;
  flops: number; // in MFLOPs
  snipScore: number;
  synflowScore: number;
  descriptionFa: string;
}

export interface SearchEpochLog {
  epoch: number;
  trainAde: number;
  valAde: number;
  valFde: number;
  flops: number;
  totalLoss: number;
  alphaTemp: number[];
  alphaInter: number[];
  alphaMap: number[];
}

export interface HardwarePlatform {
  name: string;
  type: string;
  lnasLatencyMs: number;
  nasLatencyMs: number;
  lnasFps: number;
  nasFps: number;
  powerWatts: number;
  memoryMb: number;
}
