import { Racecourse } from "./types";

export const RACECOURSES: Racecourse[] = [
  { code: "05", name: "東京" },
  { code: "06", name: "中山" },
  { code: "09", name: "阪神" },
  { code: "08", name: "京都" },
  { code: "10", name: "小倉" },
  { code: "01", name: "札幌" },
  { code: "02", name: "函館" },
  { code: "03", name: "福島" },
  { code: "04", name: "新潟" },
  { code: "07", name: "中京" },
];

export const FRAME_COLORS: Record<number, { bg: string; text: string; border: string }> = {
  1: { bg: "#FFFFFF", text: "#000000", border: "#999999" },
  2: { bg: "#000000", text: "#FFFFFF", border: "#000000" },
  3: { bg: "#DC2626", text: "#FFFFFF", border: "#DC2626" },
  4: { bg: "#2563EB", text: "#FFFFFF", border: "#2563EB" },
  5: { bg: "#EAB308", text: "#000000", border: "#EAB308" },
  6: { bg: "#16A34A", text: "#FFFFFF", border: "#16A34A" },
  7: { bg: "#F97316", text: "#FFFFFF", border: "#F97316" },
  8: { bg: "#EC4899", text: "#FFFFFF", border: "#EC4899" },
};

export const RACE_NUMBERS = Array.from({ length: 12 }, (_, i) => i + 1);

export const RACECOURSE_CODE_TO_NAME: Record<string, string> = {
  "01": "札幌",
  "02": "函館",
  "03": "福島",
  "04": "新潟",
  "05": "東京",
  "06": "中山",
  "07": "中京",
  "08": "京都",
  "09": "阪神",
  "10": "小倉",
};
