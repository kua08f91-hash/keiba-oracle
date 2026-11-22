export type RacecourseCode =
  | "01" | "02" | "03" | "04" | "05"
  | "06" | "07" | "08" | "09" | "10";

export type RacecourseName =
  | "札幌" | "函館" | "福島" | "新潟" | "東京"
  | "中山" | "中京" | "京都" | "阪神" | "小倉";

export type Surface = "芝" | "ダート" | "障害";
export type PredictionMark = "◎" | "◯" | "△" | "▲" | "";

export interface Racecourse {
  code: RacecourseCode;
  name: RacecourseName;
}

export interface RaceInfo {
  raceId: string;
  raceName: string;
  raceNumber: number;
  grade: string | null;
  distance: number;
  surface: Surface;
  courseDetail: string;
  startTime: string;
  racecourseCode: RacecourseCode;
  date: string;
  headCount: number;
  trackCondition: string;
}

export interface HorseEntry {
  frameNumber: number;
  horseNumber: number;
  horseName: string;
  horseId: string;
  sireName: string;
  damName: string;
  coatColor: string;
  weightCarried: number;
  age: string;
  jockeyName: string;
  jockeyId: string;
  trainerName: string;
  trainerId: string;
  horseWeight: string;
  odds: number | null;
  popularity: number | null;
  isScratched: boolean;
}

export interface FactorBreakdown {
  marketScore: number;
  pastPerformance: number;
  courseAffinity: number;
  distanceAptitude: number;
  ageAndSex: number;
  weightCarried: number;
  jockeyAbility: number;
  trainerAbility: number;
  trackCondition: number;
  trackDirection: number;
  trackSpecific: number;
  horseWeightChange: number;
  formTrend: number;
  sameDistance: number;
  sameSurface: number;
  sameCondition: number;
  speedFigure: number;
  runningStyle: number;
  daysSinceLast: number;
  weightCarriedTrend: number;
  agari3f: number;
  marginScore: number;
  drawBias: number;
  jockeyCourseDistance: number;
  pacePositionAdvantage: number;
  rotationFitness: number;
  bloodlineTrackCondition: number;
  valueDrop: number;
  paceScenario: number;
  cornerMove: number;
  bmsAptitude: number;
  classChange: number;
  fieldLevel: number;
  jockeyChange: number;
  trackBias: number;
  workoutEval: number;
}

export interface PredictionResult {
  horseNumber: number;
  score: number;
  mark: PredictionMark;
  factors: FactorBreakdown;
}

export interface RaceSummary {
  raceId: string;
  raceNumber: number;
  raceName: string;
  startTime: string;
}

export interface RacecourseSchedule {
  code: RacecourseCode;
  name: RacecourseName;
  races: RaceSummary[];
}

export interface RaceCardData {
  raceInfo: RaceInfo;
  entries: HorseEntry[];
  predictions: PredictionResult[];
}
