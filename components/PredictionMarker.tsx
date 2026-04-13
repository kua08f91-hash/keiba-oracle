"use client";

import { PredictionMark } from "@/lib/types";

interface Props {
  mark: PredictionMark;
}

const MARK_CONFIG: Record<PredictionMark, { style: string; label: string }> = {
  "◎": { style: "text-red-600 font-black text-lg", label: "本命" },
  "◯": { style: "text-blue-600 font-black text-lg", label: "対抗" },
  "▲": { style: "text-green-700 font-bold text-base", label: "穴馬" },
  "△": { style: "text-orange-500 font-bold text-base", label: "次点" },
  "": { style: "text-gray-300 text-sm", label: "" },
};

export default function PredictionMarker({ mark }: Props) {
  const config = MARK_CONFIG[mark];
  return (
    <div className="flex flex-col items-center leading-tight">
      <span className={config.style}>{mark || "-"}</span>
      {mark && (
        <span className="text-[9px] text-gray-500 leading-none">{config.label}</span>
      )}
    </div>
  );
}
