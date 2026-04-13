"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { RaceCardData } from "@/lib/types";
import RaceHeader from "@/components/RaceHeader";
import RaceCard from "@/components/RaceCard";
import BettingPredictions from "@/components/BettingPredictions";

export default function RaceDetailPage() {
  const params = useParams();
  const raceId = params.raceId as string;
  const [raceCard, setRaceCard] = useState<RaceCardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!raceId) return;
    setLoading(true);
    setError(null);
    fetch(`/backend/racecard/${raceId}`)
      .then((res) => {
        if (!res.ok) throw new Error("出馬表の取得に失敗しました");
        return res.json();
      })
      .then((data: RaceCardData) => setRaceCard(data))
      .catch((err) => setError(err instanceof Error ? err.message : "エラー"))
      .finally(() => setLoading(false));
  }, [raceId]);

  return (
    <div className="min-h-screen bg-[#eef2ee]">
      {/* Header */}
      <header className="bg-gradient-to-r from-[#1a472a] via-[#1e5c30] to-[#1a472a] text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-3.5 flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-1.5 text-green-200 hover:text-white transition-colors shrink-0"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="text-sm font-bold">一覧</span>
          </Link>
          <div className="h-5 w-px bg-white/20" />
          <div>
            <h1 className="text-lg font-black tracking-wider">KEIBA ORACLE</h1>
            <p className="text-[10px] text-green-300 -mt-0.5">AI-Powered Racing Intelligence</p>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-5 space-y-4">
        {loading && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 text-center py-20">
            <div className="animate-spin inline-block w-10 h-10 border-4 border-[#1a472a] border-t-transparent rounded-full mb-3" />
            <p className="text-sm text-gray-500">出馬表を読み込み中...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl shadow-sm">
            <p className="font-medium">{error}</p>
            <Link href="/" className="mt-2 inline-block text-sm text-red-600 underline hover:no-underline">
              一覧に戻る
            </Link>
          </div>
        )}

        {!loading && raceCard && (
          <div className="space-y-4">
            {/* Race card */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
              <RaceHeader raceInfo={raceCard.raceInfo} />
              <RaceCard entries={raceCard.entries} predictions={raceCard.predictions} />
            </div>

            {/* Legend */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 px-5 py-3 flex items-center gap-5 text-sm flex-wrap">
              <span className="text-xs font-bold text-gray-400">予想記号</span>
              <div className="flex items-center gap-1">
                <span className="text-red-700 font-black text-lg">◎</span>
                <span className="text-xs text-gray-600">本命</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-blue-700 font-black text-lg">◯</span>
                <span className="text-xs text-gray-600">対抗</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-green-800 font-black text-lg">▲</span>
                <span className="text-xs text-gray-600">穴馬</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-orange-600 font-black text-lg">△</span>
                <span className="text-xs text-gray-600">次点</span>
              </div>
              <span className="text-gray-400 text-[10px] ml-auto">※ Score列クリックで因子分解を表示</span>
            </div>

            {/* Betting predictions */}
            <BettingPredictions
              predictions={raceCard.predictions}
              entries={raceCard.entries}
              raceId={raceCard.raceInfo?.raceId}
            />
          </div>
        )}
      </div>

      <footer className="mt-8 bg-[#1a472a] text-green-200 text-center py-4 text-[10px]">
        KEIBA ORACLE - AI Racing Intelligence
      </footer>
    </div>
  );
}
