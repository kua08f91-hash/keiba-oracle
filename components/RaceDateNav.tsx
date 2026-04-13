"use client";

import { useState, useEffect } from "react";

interface RaceDate {
  date: string;
  display: string;
  weekLabel: string;
  courses: string[];
}

interface Props {
  selectedDate: string;
  onSelectDate: (date: string) => void;
}

export default function RaceDateNav({ selectedDate, onSelectDate }: Props) {
  const [raceDates, setRaceDates] = useState<RaceDate[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetch("/backend/race-dates?weeks=3")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: RaceDate[]) => setRaceDates(data))
      .catch(() => setRaceDates([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-4 shadow-sm">
        <div className="text-xs text-gray-400 animate-pulse">開催日を検索中...</div>
      </div>
    );
  }

  if (raceDates.length === 0) return null;

  const grouped: Record<string, RaceDate[]> = {};
  for (const rd of raceDates) {
    if (!grouped[rd.weekLabel]) grouped[rd.weekLabel] = [];
    grouped[rd.weekLabel].push(rd);
  }

  const weekOrder = ["先週", "今週", "来週", "再来週"];

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="bg-[#f8faf8] border-b border-gray-200 px-4 py-2">
        <span className="text-xs font-bold text-gray-500 tracking-wide">開催スケジュール</span>
      </div>
      <div className="p-3 space-y-2">
        {weekOrder.map((label) => {
          const dates = grouped[label];
          if (!dates || dates.length === 0) return null;
          return (
            <div key={label} className="flex items-start gap-2">
              <span className="text-[11px] font-black text-[#1a472a] bg-[#e8f5e9] rounded-md px-2.5 py-1.5 min-w-[52px] text-center shrink-0 border border-[#c8e6c9]">
                {label}
              </span>
              <div className="flex gap-1.5 flex-wrap">
                {dates.map((rd) => {
                  const isSelected = rd.date === selectedDate;
                  return (
                    <button
                      key={rd.date}
                      onClick={() => onSelectDate(rd.date)}
                      className={`
                        px-3 py-1.5 rounded-lg text-sm transition-all border font-medium
                        ${isSelected
                          ? "bg-[#1a472a] text-white border-[#1a472a] shadow-md"
                          : "bg-white text-gray-700 border-gray-200 hover:bg-green-50 hover:border-green-300 hover:shadow-sm"
                        }
                      `}
                    >
                      <span className="font-bold">{rd.display}</span>
                      <span className="text-[10px] ml-1 opacity-70">
                        {rd.courses.join("・")}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
