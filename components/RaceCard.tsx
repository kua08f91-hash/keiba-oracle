"use client";

import { useState, Fragment } from "react";
import { HorseEntry, PredictionResult, PredictionMark, FactorBreakdown } from "@/lib/types";
import FrameColorBox from "./FrameColorBox";

interface Props {
  entries: HorseEntry[];
  predictions: PredictionResult[];
}

function getMarkForHorse(
  horseNumber: number,
  predictions: PredictionResult[]
): PredictionMark {
  const pred = predictions.find((p) => p.horseNumber === horseNumber);
  return pred?.mark ?? "";
}

function getScoreForHorse(
  horseNumber: number,
  predictions: PredictionResult[]
): number {
  const pred = predictions.find((p) => p.horseNumber === horseNumber);
  return pred?.score ?? 0;
}

function getFactorsForHorse(
  horseNumber: number,
  predictions: PredictionResult[]
): FactorBreakdown | null {
  const pred = predictions.find((p) => p.horseNumber === horseNumber);
  return pred?.factors ?? null;
}

const MARK_STYLE: Record<string, { text: string; bg: string; label: string }> = {
  "◎": { text: "text-red-700", bg: "bg-red-50 border-l-4 border-l-red-500", label: "本命" },
  "◯": { text: "text-blue-700", bg: "bg-blue-50 border-l-4 border-l-blue-500", label: "対抗" },
  "▲": { text: "text-green-800", bg: "bg-green-50 border-l-4 border-l-green-500", label: "穴馬" },
  "△": { text: "text-orange-600", bg: "bg-orange-50 border-l-4 border-l-orange-400", label: "次点" },
  "": { text: "text-gray-300", bg: "border-l-4 border-l-transparent", label: "" },
};

const FACTOR_LABELS: Record<string, string> = {
  trackDirection: "回り適性",
  trackCondition: "馬場適性",
  jockeyAbility: "騎手能力",
  trackSpecific: "コース実績",
  pastPerformance: "近走成績",
  formTrend: "調子傾向",
  ageAndSex: "年齢・性別",
  weightCarried: "斤量",
  courseAffinity: "コース血統",
  horseWeightChange: "馬体重変動",
  trainerAbility: "調教師",
  distanceAptitude: "距離血統",
  marketScore: "市場評価",
};

// Analytical weights + market weight for contribution calculation
const FACTOR_WEIGHTS: Record<string, number> = {
  trackDirection: 0.2822 * 0.85,
  trackCondition: 0.2786 * 0.85,
  jockeyAbility: 0.1541 * 0.85,
  trackSpecific: 0.0815 * 0.85,
  pastPerformance: 0.0687 * 0.85,
  formTrend: 0.0300 * 0.85,
  ageAndSex: 0.0314 * 0.85,
  weightCarried: 0.0307 * 0.85,
  courseAffinity: 0.0113 * 0.85,
  horseWeightChange: 0.0110 * 0.85,
  trainerAbility: 0.0102 * 0.85,
  distanceAptitude: 0.0102 * 0.85,
  marketScore: 0.15,
};

function getScoreColor(score: number): string {
  if (score >= 75) return "bg-gradient-to-r from-red-400 to-red-600";
  if (score >= 65) return "bg-gradient-to-r from-blue-400 to-blue-600";
  if (score >= 55) return "bg-gradient-to-r from-green-400 to-green-600";
  if (score >= 45) return "bg-gradient-to-r from-yellow-400 to-orange-500";
  return "bg-gray-300";
}

function getFactorColor(score: number): string {
  if (score > 60) return "text-green-700 bg-green-50";
  if (score >= 40) return "text-yellow-700 bg-yellow-50";
  return "text-red-700 bg-red-50";
}

function FactorPanel({ factors }: { factors: FactorBreakdown }) {
  const entries = Object.entries(FACTOR_LABELS)
    .map(([key, label]) => {
      const score = (factors as Record<string, number>)[key] ?? 0;
      const weight = FACTOR_WEIGHTS[key] ?? 0;
      const contribution = score * weight;
      return { key, label, score, weight, contribution };
    })
    .sort((a, b) => b.contribution - a.contribution);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-1.5 p-3">
      {entries.map(({ key, label, score, weight, contribution }) => (
        <div
          key={key}
          className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs ${getFactorColor(score)}`}
        >
          <div className="flex-1 min-w-0">
            <div className="font-bold truncate">{label}</div>
            <div className="text-[10px] opacity-70">
              {score.toFixed(0)}pt &times; {(weight * 100).toFixed(1)}%
            </div>
          </div>
          <div className="w-12 h-1.5 bg-black/10 rounded-full overflow-hidden flex-shrink-0">
            <div
              className="h-full rounded-full bg-current opacity-50"
              style={{ width: `${Math.min(contribution * 1.5, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function RaceCard({ entries, predictions }: Props) {
  const [sortByScore, setSortByScore] = useState(false);
  const [expandedHorse, setExpandedHorse] = useState<number | null>(null);

  const sortedEntries = [...entries].sort((a, b) => {
    if (sortByScore) {
      const sa = a.isScratched ? -1 : getScoreForHorse(a.horseNumber, predictions);
      const sb = b.isScratched ? -1 : getScoreForHorse(b.horseNumber, predictions);
      return sb - sa;
    }
    return a.horseNumber - b.horseNumber;
  });

  return (
    <div>
      {/* Sort toggle */}
      <div className="flex justify-end px-2 py-1.5">
        <button
          onClick={() => setSortByScore((v) => !v)}
          className="flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold border transition-colors
            bg-white border-[#3a6b4a] text-[#3a6b4a] hover:bg-[#f0f7f0]"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d={sortByScore ? "M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" : "M3 4h13M3 8h9m-9 4h9m5-4v12m0 0l-4-4m4 4l4-4"} />
          </svg>
          {sortByScore ? "Score順" : "馬番順"}
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-gradient-to-b from-[#3a6b4a] to-[#2a5438] text-white text-[11px] tracking-wide">
              <th className="w-[52px] py-2.5 text-center font-bold border-r border-[#4a7b5a]">予想</th>
              <th className="w-[36px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden md:table-cell">枠</th>
              <th className="w-[36px] py-2.5 text-center font-bold border-r border-[#4a7b5a]">馬番</th>
              <th className="py-2.5 px-3 text-left font-bold border-r border-[#4a7b5a] min-w-[100px] sm:min-w-[140px]">馬名</th>
              <th className="w-[44px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden md:table-cell">性齢</th>
              <th className="w-[44px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden md:table-cell">斤量</th>
              <th className="py-2.5 px-2 text-left font-bold border-r border-[#4a7b5a] min-w-[76px] hidden sm:table-cell">騎手</th>
              <th className="py-2.5 px-2 text-left font-bold border-r border-[#4a7b5a] min-w-[68px] hidden lg:table-cell">調教師</th>
              <th className="w-[72px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden lg:table-cell">馬体重</th>
              <th className="py-2.5 px-2 text-left font-bold border-r border-[#4a7b5a] min-w-[90px] hidden lg:table-cell">父</th>
              <th className="py-2.5 px-2 text-left font-bold border-r border-[#4a7b5a] min-w-[90px] hidden lg:table-cell">母</th>
              <th className="w-[56px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden sm:table-cell">オッズ</th>
              <th className="w-[40px] py-2.5 text-center font-bold border-r border-[#4a7b5a] hidden sm:table-cell">人気</th>
              <th className="w-[56px] py-2.5 text-center font-bold">Score</th>
            </tr>
          </thead>
          <tbody>
            {sortedEntries.map((entry, idx) => {
              const mark = getMarkForHorse(entry.horseNumber, predictions);
              const score = getScoreForHorse(entry.horseNumber, predictions);
              const factors = getFactorsForHorse(entry.horseNumber, predictions);
              const isScratched = entry.isScratched;
              const ms = MARK_STYLE[mark] || MARK_STYLE[""];
              const stripe = idx % 2 === 0 ? "bg-white" : "bg-[#f8faf8]";
              const isExpanded = expandedHorse === entry.horseNumber;

              return (
                <Fragment key={entry.horseNumber}>
                  <tr
                    className={`${isScratched ? "opacity-40" : ""} ${mark ? ms.bg : stripe} border-b border-gray-200 hover:bg-[#fffde7] transition-colors group`}
                  >
                    {/* 予想 */}
                    <td className="py-2 text-center">
                      {mark ? (
                        <div className="flex flex-col items-center leading-none">
                          <span className={`${ms.text} font-black text-xl leading-none`}>{mark}</span>
                          <span className="text-[9px] text-gray-500 mt-0.5">{ms.label}</span>
                        </div>
                      ) : (
                        <span className="text-gray-200">-</span>
                      )}
                    </td>

                    {/* 枠 */}
                    <td className="py-2 text-center border-r border-gray-100 hidden md:table-cell">
                      <FrameColorBox frameNumber={entry.frameNumber} />
                    </td>

                    {/* 馬番 */}
                    <td className="py-2 text-center font-black text-base text-gray-800 border-r border-gray-100">
                      {entry.horseNumber}
                    </td>

                    {/* 馬名 */}
                    <td className={`py-2 px-3 border-r border-gray-100 ${isScratched ? "line-through text-gray-400" : ""}`}>
                      <span className="font-bold text-[13px] text-gray-900 group-hover:text-[#1a472a]">
                        {entry.horseName}
                      </span>
                    </td>

                    {/* 性齢 */}
                    <td className="py-2 text-center text-xs text-gray-600 border-r border-gray-100 hidden md:table-cell">
                      {isScratched ? "—" : entry.age}
                    </td>

                    {/* 斤量 */}
                    <td className="py-2 text-center text-sm font-medium text-gray-700 border-r border-gray-100 hidden md:table-cell">
                      {isScratched ? "—" : entry.weightCarried.toFixed(1)}
                    </td>

                    {/* 騎手 */}
                    <td className="py-2 px-2 text-[13px] font-bold text-gray-800 border-r border-gray-100 hidden sm:table-cell">
                      {isScratched ? "—" : entry.jockeyName}
                    </td>

                    {/* 調教師 */}
                    <td className="py-2 px-2 text-xs text-gray-500 border-r border-gray-100 hidden lg:table-cell">
                      {isScratched ? "—" : entry.trainerName}
                    </td>

                    {/* 馬体重 */}
                    <td className="py-2 text-center text-xs border-r border-gray-100 hidden lg:table-cell">
                      {isScratched ? "—" : entry.horseWeight ? (
                        <span className="text-gray-700 font-medium">{entry.horseWeight}</span>
                      ) : (
                        <span className="text-gray-300">-</span>
                      )}
                    </td>

                    {/* 父 */}
                    <td className="py-2 px-2 text-xs text-gray-500 border-r border-gray-100 max-w-[110px] truncate hidden lg:table-cell" title={entry.sireName}>
                      {isScratched ? "—" : entry.sireName || "-"}
                    </td>

                    {/* 母 */}
                    <td className="py-2 px-2 text-xs text-gray-500 border-r border-gray-100 max-w-[110px] truncate hidden lg:table-cell" title={entry.damName}>
                      {isScratched ? "—" : entry.damName || "-"}
                    </td>

                    {/* オッズ */}
                    <td className="py-2 text-center border-r border-gray-100 hidden sm:table-cell">
                      {isScratched || entry.odds == null ? (
                        <span className="text-gray-300">-</span>
                      ) : (
                        <span className={`font-black text-sm ${
                          entry.popularity === 1 ? "text-red-600" :
                          entry.popularity === 2 ? "text-blue-700" :
                          entry.popularity != null && entry.popularity <= 3 ? "text-green-700" :
                          "text-gray-800"
                        }`}>
                          {entry.odds.toFixed(1)}
                        </span>
                      )}
                    </td>

                    {/* 人気 */}
                    <td className="py-2 text-center border-r border-gray-100 hidden sm:table-cell">
                      {isScratched || entry.popularity == null ? (
                        <span className="text-gray-300">-</span>
                      ) : (
                        <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-[10px] font-black ${
                          entry.popularity === 1 ? "bg-red-600 text-white" :
                          entry.popularity === 2 ? "bg-blue-600 text-white" :
                          entry.popularity === 3 ? "bg-green-600 text-white" :
                          entry.popularity <= 5 ? "bg-orange-100 text-orange-800 border border-orange-300" :
                          "bg-gray-100 text-gray-500 border border-gray-200"
                        }`}>
                          {entry.popularity}
                        </span>
                      )}
                    </td>

                    {/* Score */}
                    <td
                      className="py-2 text-center cursor-pointer hover:bg-[#e8f5e9] rounded transition-colors"
                      onClick={() => !isScratched && setExpandedHorse(isExpanded ? null : entry.horseNumber)}
                      title={isScratched ? "" : "クリックで詳細を表示"}
                    >
                      {isScratched ? (
                        <span className="text-gray-300">-</span>
                      ) : (
                        <div className="flex flex-col items-center gap-0.5">
                          <span className={`font-mono font-bold ${
                            mark ? "text-sm text-gray-900" : "text-[11px] text-gray-600"
                          }`}>
                            {score.toFixed(1)}
                          </span>
                          <div className="w-12 h-2.5 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${getScoreColor(score)}`}
                              style={{ width: `${Math.min(score * 1.2, 100)}%` }}
                            />
                          </div>
                          <svg className={`w-3 h-3 text-gray-400 transition-transform ${isExpanded ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      )}
                    </td>
                  </tr>
                  {/* Factor breakdown panel */}
                  {isExpanded && factors && (
                    <tr className="bg-[#f5f8f5] border-b border-gray-200">
                      <td colSpan={14} className="p-0">
                        <div className="border-l-4 border-l-[#3a6b4a]">
                          <div className="px-3 py-1.5 bg-[#e8f0e8] text-[11px] font-bold text-[#2a5438]">
                            {entry.horseName} のスコア内訳
                          </div>
                          <FactorPanel factors={factors} />
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
