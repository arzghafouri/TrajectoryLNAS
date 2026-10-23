import { AgentScenario, TrajectoryPoint } from '../types/trajectory';

// Helper to generate smooth points
function generateArc(
  startX: number,
  startY: number,
  length: number,
  startAngleDeg: number,
  curveRateDeg: number,
  stepDist: number
): TrajectoryPoint[] {
  const points: TrajectoryPoint[] = [];
  let x = startX;
  let y = startY;
  let angle = (startAngleDeg * Math.PI) / 180;
  const radCurve = (curveRateDeg * Math.PI) / 180;

  for (let i = 0; i < length; i++) {
    points.push({ x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10 });
    x += Math.cos(angle) * stepDist;
    y += Math.sin(angle) * stepDist;
    angle += radCurve;
  }
  return points;
}

// 1. Urban 4-Way Left Turn (Argoverse Sample #2841)
const s1Past = generateArc(250, 480, 20, -90, -0.2, 4.5);
const s1EgoLast = s1Past[s1Past.length - 1];
const s1TrueFuture = generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -2.8, 5.2);
const s1LnasFuture = generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -2.75, 5.15).map((p, idx) => ({
  x: p.x + (Math.sin(idx * 0.4) * 0.8),
  y: p.y + (Math.cos(idx * 0.3) * 0.6),
}));
const s1NasFuture = generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -2.4, 5.0).map((p, idx) => ({
  x: p.x + (idx * 0.25),
  y: p.y + (idx * 0.15),
}));
const s1VanillaFuture = generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -1.5, 4.8).map((p, idx) => ({
  x: p.x + (idx * 0.8),
  y: p.y - (idx * 0.4),
}));

// Neighbors for S1
const s1Neighbor1Past = generateArc(450, 220, 20, 180, 0.1, 4.0);
const s1N1Last = s1Neighbor1Past[s1Neighbor1Past.length - 1];
const s1Neighbor1Future = generateArc(s1N1Last.x, s1N1Last.y, 30, 180, 0, 4.2);

const s1PedPast = generateArc(360, 310, 20, 90, 0, 1.2);
const s1PedLast = s1PedPast[s1PedPast.length - 1];
const s1PedFuture = generateArc(s1PedLast.x, s1PedLast.y, 30, 90, 0, 1.2);

// 2. Highway Overtake & Merge (nuScenes Sample #1092)
const s2Past = generateArc(80, 250, 20, 0, 0.1, 6.0);
const s2Last = s2Past[s2Past.length - 1];
const s2TrueFuture = generateArc(s2Last.x, s2Last.y, 30, 0, 0.8, 6.8);
const s2LnasFuture = generateArc(s2Last.x, s2Last.y, 30, 0, 0.78, 6.75);
const s2NasFuture = generateArc(s2Last.x, s2Last.y, 30, 0, 0.65, 6.6);
const s2VanillaFuture = generateArc(s2Last.x, s2Last.y, 30, 0, 0.3, 6.3);

const s2SlowCarPast = generateArc(160, 250, 20, 0, 0, 4.2);
const s2SlowLast = s2SlowCarPast[s2SlowCarPast.length - 1];
const s2SlowCarFuture = generateArc(s2SlowLast.x, s2SlowLast.y, 30, 0, 0, 4.2);

// S1 Multi-Modal Modes (K=6)
const s1Modes = [
  { id: 1, probability: 0.48, color: '#22d3ee', labelFa: 'گردش به چپ استاندارد (برنده)', points: s1LnasFuture },
  { id: 2, probability: 0.22, color: '#06b6d4', labelFa: 'گردش سریع و زاویه تند', points: generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -3.4, 5.3) },
  { id: 3, probability: 0.14, color: '#38bdf8', labelFa: 'توقف در تقاطع برای عبور عابر', points: generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -1.8, 2.5) },
  { id: 4, probability: 0.08, color: '#60a5fa', labelFa: 'انحراف به مسیر مستقیم', points: generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -0.4, 4.9) },
  { id: 5, probability: 0.05, color: '#818cf8', labelFa: 'گردش با شعاع باز', points: generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -2.1, 5.6) },
  { id: 6, probability: 0.03, color: '#a78bfa', labelFa: 'کاهش شدید سرعت / توقف اضطراری', points: generateArc(s1EgoLast.x, s1EgoLast.y, 30, -90, -0.9, 1.4) },
];

// S2 Multi-Modal Modes (K=6)
const s2Modes = [
  { id: 1, probability: 0.54, color: '#22d3ee', labelFa: 'تغییر لاین به چپ سبقت (برنده)', points: s2LnasFuture },
  { id: 2, probability: 0.20, color: '#06b6d4', labelFa: 'حرکت در لاین مستقیم پشت تریلی', points: generateArc(s2Last.x, s2Last.y, 30, 0, 0.0, 4.8) },
  { id: 3, probability: 0.12, color: '#38bdf8', labelFa: 'تعویض لاین دیرتر با فاصله بیشتر', points: generateArc(s2Last.x, s2Last.y, 30, 0, 0.45, 6.0) },
  { id: 4, probability: 0.06, color: '#60a5fa', labelFa: 'سبقت پرشتاب با زاویه تند', points: generateArc(s2Last.x, s2Last.y, 30, 0, 1.15, 7.3) },
  { id: 5, probability: 0.05, color: '#818cf8', labelFa: 'کاهش تدریجی سرعت', points: generateArc(s2Last.x, s2Last.y, 30, 0, -0.1, 3.8) },
  { id: 6, probability: 0.03, color: '#a78bfa', labelFa: 'انحراف به شانه سمت راست', points: generateArc(s2Last.x, s2Last.y, 30, 0, -0.7, 4.5) },
];

export const MOCK_SCENARIOS: AgentScenario[] = [
  {
    id: 's1',
    name: 'Argoverse Urban 4-Way Left Turn',
    nameFa: 'تقاطع چهارراه شهری و گردش به چپ (Argoverse #2841)',
    description: 'Ego vehicle executes an unprotected left turn across oncoming traffic with pedestrian in crosswalk.',
    descriptionFa: 'خودروی آزمون (Ego) در تقاطع چهارراه شهری با چراغ چشمک‌زن اقدام به گردش به چپ می‌کند، در حالی که ترافیک روبرو و عابر پیاده حضور دارند.',
    roadType: 'intersection',
    egoPast: s1Past,
    egoGroundTruthFuture: s1TrueFuture,
    egoLnasPred: s1LnasFuture,
    candidateModes: s1Modes,
    egoNasPred: s1NasFuture,
    egoVanillaPred: s1VanillaFuture,
    neighbors: [
      {
        id: 'n1',
        type: 'car',
        color: '#38bdf8', // sky-400
        past: s1Neighbor1Past,
        future: s1Neighbor1Future,
      },
      {
        id: 'p1',
        type: 'pedestrian',
        color: '#f43f5e', // rose-500
        past: s1PedPast,
        future: s1PedFuture,
      },
    ],
    mapLanes: [
      {
        id: 'l1',
        points: [[250, 50], [250, 550]],
        type: 'centerline',
      },
      {
        id: 'l2',
        points: [[50, 250], [550, 250]],
        type: 'centerline',
      },
      {
        id: 'b1',
        points: [[200, 50], [200, 200], [50, 200]],
        type: 'boundary',
      },
      {
        id: 'b2',
        points: [[300, 50], [300, 200], [550, 200]],
        type: 'boundary',
      },
      {
        id: 'b3',
        points: [[200, 550], [200, 300], [50, 300]],
        type: 'boundary',
      },
      {
        id: 'b4',
        points: [[300, 550], [300, 300], [550, 300]],
        type: 'boundary',
      },
      {
        id: 'cw1',
        points: [[210, 320], [290, 320]],
        type: 'crosswalk',
      },
    ],
  },
  {
    id: 's2',
    name: 'nuScenes Highway High-Speed Overtake',
    nameFa: 'سبقت سریع و تغییر لاین در بزرگراه (nuScenes #1092)',
    description: 'Ego vehicle closes distance with a slower lead truck and performs lane change to the left express lane.',
    descriptionFa: 'خودروی خودران با سرعت ۸۵ کیلومتر بر ساعت به یک وسیله نقلیه سنگین کندرو نزدیک شده و مانور تعویض خط به لاین سبقت را اجرا می‌کند.',
    roadType: 'highway',
    egoPast: s2Past,
    egoGroundTruthFuture: s2TrueFuture,
    egoLnasPred: s2LnasFuture,
    candidateModes: s2Modes,
    egoNasPred: s2NasFuture,
    egoVanillaPred: s2VanillaFuture,
    neighbors: [
      {
        id: 'truck1',
        type: 'car',
        color: '#f59e0b', // amber-500
        past: s2SlowCarPast,
        future: s2SlowCarFuture,
      },
    ],
    mapLanes: [
      {
        id: 'hl1',
        points: [[40, 180], [560, 180]],
        type: 'boundary',
      },
      {
        id: 'hl2',
        points: [[40, 250], [560, 250]],
        type: 'centerline',
      },
      {
        id: 'hl3',
        points: [[40, 320], [560, 320]],
        type: 'centerline',
      },
      {
        id: 'hl4',
        points: [[40, 390], [560, 390]],
        type: 'boundary',
      },
    ],
  },
];
