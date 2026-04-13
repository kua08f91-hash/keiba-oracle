"use client";

import { useEffect, useState } from "react";
import { PredictionResult, HorseEntry } from "@/lib/types";
import FrameColorBox from "./FrameColorBox";

interface Props {
  predictions: PredictionResult[];
  entries: HorseEntry[];
  raceId?: string;
}

interface OptimizedBet {
  type: string;
  typeLabel: string;
  horses: number[];
  ordered: boolean;
  odds?: number;
  payout?: number;
  ev?: number;
  hitProb?: number;
  hasRealOdds?: boolean;
  rank?: number;
}

interface OptimizedBetsResponse {
  bets: OptimizedBet[];
  pattern: string;
  raceId: string;
}

function getEntry(horseNumber: number, entries: HorseEntry[]): HorseEntry | undefined {
  return entries.find((e) => e.horseNumber === horseNumber);
}

const TYPE_BADGE: Record<string, { bg: string; text: string }> = {
  tansho: { bg: "bg-red-600", text: "text-white" },
  fukusho: { bg: "bg-blue-600", text: "text-white" },
  umaren: { bg: "bg-purple-600", text: "text-white" },
  wide: { bg: "bg-teal-600", text: "text-white" },
  sanrenpuku: { bg: "bg-green-600", text: "text-white" },
  sanrentan: { bg: "bg-orange-600", text: "text-white" },
};

const PATTERN_STYLE: Record<string, { label: string; color: string }> = {
  "本命堅軸": { label: "本命堅軸", color: "text-red-500" },
  "混戦模様": { label: "混戦模様", color: "text-purple-500" },
  "2強対決": { label: "2強対決", color: "text-blue-500" },
  "標準配置": { label: "標準配置", color: "text-gray-500" },
  "少頭数": { label: "少頭数", color: "text-gray-400" },
};

function HorseChip({
  horseNumber,
  entries,
}: {
  horseNumber: number;
  entries: HorseEntry[];
}) {
  const e = getEntry(horseNumber, entries);
  const name = e?.horseName || `${horseNumber}番`;
  const frame = e?.frameNumber || 1;
  return (
    <span className="inline-flex items-center gap-1.5 bg-white border border-gray-200 rounded-lg px-2 py-1 shadow-sm">
      <FrameColorBox frameNumber={frame} size="sm" />
      <span className="font-black text-sm text-gray-800">{horseNumber}</span>
      <span className="text-xs text-gray-600 font-medium">{name}</span>
    </span>
  );
}

function EvBadge({ ev }: { ev: number }) {
  const isPositive = ev > 0;
  const isNearZero = Math.abs(ev) < 0.1;
  const color = isPositive
    ? "text-green-600 bg-green-50 border-green-200"
    : isNearZero
    ? "text-yellow-600 bg-yellow-50 border-yellow-200"
    : "text-gray-400 bg-gray-50 border-gray-200";

  return (
    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${color}`}>
      EV {ev >= 0 ? "+" : ""}{(ev * 100).toFixed(0)}%
    </span>
  );
}

export default function BettingPredictions({ predictions, entries, raceId }: Props) {
  const [betsData, setBetsData] = useState<OptimizedBetsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!raceId) return;
    setLoading(true);
    fetch(`/backend/optimized-bets/${raceId}`)
      .then((res) => res.json())
      .then((data: OptimizedBetsResponse) => {
        setBetsData(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, [raceId]);

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
        <div className="bg-gradient-to-b from-[#1e5c30] to-[#164422] px-5 py-3">
          <h3 className="font-black text-white text-base tracking-wide">AI買い目予想</h3>
        </div>
        <div className="text-center py-10">
          <div className="animate-spin inline-block w-8 h-8 border-3 border-[#1a472a] border-t-transparent rounded-full mb-2"></div>
          <p className="text-sm text-gray-500">最適買い目を計算中...</p>
        </div>
      </div>
    );
  }

  if (!betsData || betsData.bets.length === 0) return null;

  const { bets, pattern } = betsData;
  const patternStyle = PATTERN_STYLE[pattern] || PATTERN_STYLE["標準配置"];

  // Summarize bet types for subtitle
  const typeSummary = bets.map((b) => b.typeLabel).join(" + ");

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      {/* Header */}
      <div className="bg-gradient-to-b from-[#1e5c30] to-[#164422] px-5 py-3 flex items-center gap-3">
        <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-black text-white text-base tracking-wide">AI買い目予想</h3>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full bg-white/15 ${patternStyle.color === "text-red-500" ? "text-red-300" : patternStyle.color === "text-purple-500" ? "text-purple-300" : patternStyle.color === "text-blue-500" ? "text-blue-300" : "text-green-300"}`}>
              {patternStyle.label}
            </span>
          </div>
          <p className="text-green-300 text-[10px]">EV最適化 — {typeSummary}</p>
        </div>
      </div>

      {/* Top 5 bets */}
      <div className="divide-y divide-gray-100">
        {bets.map((bet, idx) => {
          const badge = TYPE_BADGE[bet.type] || TYPE_BADGE.tansho;

          return (
            <div
              key={idx}
              className={`px-5 py-3.5 flex items-center gap-4 ${
                idx === 0 ? "bg-gradient-to-r from-yellow-50 to-amber-50 border-l-4 border-l-yellow-400" : ""
              }`}
            >
              {/* Rank */}
              <div className={`w-8 h-8 flex items-center justify-center rounded-full font-black text-sm shrink-0 ${
                idx === 0 ? "bg-gradient-to-br from-yellow-400 to-amber-500 text-white shadow-md" :
                idx === 1 ? "bg-gradient-to-br from-gray-300 to-gray-400 text-white shadow" :
                idx === 2 ? "bg-gradient-to-br from-orange-300 to-orange-400 text-white shadow" :
                "bg-gray-100 text-gray-500 border border-gray-200"
              }`}>
                {idx + 1}
              </div>

              {/* Bet type badge */}
              <span className={`${badge.bg} ${badge.text} text-[11px] font-black px-2.5 py-1 rounded-md min-w-[52px] text-center shadow-sm shrink-0`}>
                {bet.typeLabel}
              </span>

              {/* Horse chips */}
              <div className="flex items-center gap-1.5 flex-wrap flex-1">
                {bet.horses.map((hn, hi) => (
                  <div key={hn} className="flex items-center gap-1">
                    {bet.ordered && hi > 0 && (
                      <svg className="w-4 h-4 text-gray-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
                      </svg>
                    )}
                    {!bet.ordered && hi > 0 && (
                      <span className="text-gray-300 text-xs font-bold">-</span>
                    )}
                    <HorseChip horseNumber={hn} entries={entries} />
                  </div>
                ))}
              </div>

              {/* EV + Odds display */}
              <div className="shrink-0 text-right space-y-0.5">
                {bet.ev !== undefined && <EvBadge ev={bet.ev} />}
                {bet.odds ? (
                  <div>
                    <span className="text-sm font-black text-amber-600">
                      {bet.odds >= 10 ? bet.odds.toFixed(0) : bet.odds.toFixed(1)}
                    </span>
                    <span className="text-[10px] text-gray-400 ml-0.5">倍</span>
                    {bet.hitProb !== undefined && (
                      <div className="text-[10px] text-gray-400">
                        的中率 {(bet.hitProb * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                ) : (
                  <span className="text-xs text-gray-300">—</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="bg-[#f8faf8] border-t border-gray-200 px-5 py-2.5 text-[10px] text-gray-400">
        ※ 各レースのスコア分布・オッズから期待値(EV)を最大化する買い目をAIが動的に選択しています。投票は自己責任でお願いします。
      </div>
    </div>
  );
}
