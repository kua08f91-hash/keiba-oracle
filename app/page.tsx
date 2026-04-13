"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { RacecourseSchedule } from "@/lib/types";

interface RaceDate {
  date: string;
  display: string;
  weekLabel: string;
  courses: string[];
}

interface DaySchedule {
  date: string;
  display: string;
  weekLabel: string;
  schedules: RacecourseSchedule[];
}

const WEEK_TABS = ["今週", "来週"] as const;

export default function Home() {
  const [raceDates, setRaceDates] = useState<RaceDate[]>([]);
  const [daySchedules, setDaySchedules] = useState<DaySchedule[]>([]);
  const [activeWeek, setActiveWeek] = useState<string>("今週");
  const [loadingDates, setLoadingDates] = useState(true);
  const [loadingSchedules, setLoadingSchedules] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch race dates on mount
  useEffect(() => {
    setLoadingDates(true);
    fetch("/backend/race-dates?weeks=3")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: RaceDate[]) => {
        setRaceDates(data);
        // Default to 今週, fallback to first available week
        const hasThisWeek = data.some((d) => d.weekLabel === "今週");
        if (!hasThisWeek && data.length > 0) {
          setActiveWeek(data[0].weekLabel);
        }
      })
      .catch(() => setRaceDates([]))
      .finally(() => setLoadingDates(false));
  }, []);

  // Fetch schedules for active week's dates
  const fetchWeekSchedules = useCallback(async (dates: RaceDate[]) => {
    setLoadingSchedules(true);
    setError(null);
    const results: DaySchedule[] = [];

    for (const rd of dates) {
      try {
        const res = await fetch(`/backend/race-list?date=${rd.date}`);
        if (!res.ok) continue;
        const rawData = await res.json();
        const schedules: RacecourseSchedule[] = rawData.map((s: any) => ({
          code: s.code,
          name: s.name,
          races: (s.races || []).map((r: any) => ({
            raceId: r.race_id || r.raceId,
            raceNumber: r.race_number ?? r.raceNumber,
            raceName: r.race_name || r.raceName || "",
            startTime: r.start_time || r.startTime || "",
          })),
        }));
        if (schedules.length > 0) {
          results.push({
            date: rd.date,
            display: rd.display,
            weekLabel: rd.weekLabel,
            schedules,
          });
        }
      } catch {
        // skip failed dates
      }
    }
    setDaySchedules(results);
    setLoadingSchedules(false);
  }, []);

  useEffect(() => {
    const weekDates = raceDates.filter((d) => d.weekLabel === activeWeek);
    if (weekDates.length > 0) {
      fetchWeekSchedules(weekDates);
    } else {
      setDaySchedules([]);
    }
  }, [activeWeek, raceDates, fetchWeekSchedules]);

  // Available week labels from data
  const availableWeeks = [...new Set(raceDates.map((d) => d.weekLabel))];
  const weekOrder = ["先週", "今週", "来週", "再来週"];
  const sortedWeeks = weekOrder.filter((w) => availableWeeks.includes(w));

  return (
    <div className="min-h-screen bg-[#eef2ee]">
      {/* Header */}
      <header className="bg-gradient-to-r from-[#1a472a] via-[#1e5c30] to-[#1a472a] text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-3.5 flex items-center gap-3">
          <div className="w-9 h-9 bg-white/15 rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6 text-green-200" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-black tracking-wider">KEIBA ORACLE</h1>
            <p className="text-[10px] text-green-300 -mt-0.5">AI-Powered Racing Intelligence</p>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-5 space-y-4">
        {/* Week tabs */}
        {sortedWeeks.length > 0 && (
          <div className="flex gap-1 bg-white rounded-xl shadow-sm border border-gray-200 p-1.5">
            {sortedWeeks.map((week) => (
              <button
                key={week}
                onClick={() => setActiveWeek(week)}
                className={`flex-1 py-2 px-4 rounded-lg text-sm font-bold transition-all ${
                  activeWeek === week
                    ? "bg-gradient-to-b from-[#1e5c30] to-[#164422] text-white shadow"
                    : "text-gray-500 hover:bg-[#f0f7f0] hover:text-[#1a472a]"
                }`}
              >
                {week}
                <span className="ml-1 text-[10px] opacity-70">
                  ({raceDates.filter((d) => d.weekLabel === week).length}日)
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Loading */}
        {(loadingDates || loadingSchedules) && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 text-center py-16">
            <div className="animate-spin inline-block w-10 h-10 border-4 border-[#1a472a] border-t-transparent rounded-full mb-3" />
            <p className="text-sm text-gray-500">
              {loadingDates ? "開催日を検索中..." : "レース情報を取得中..."}
            </p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-5 py-4 rounded-xl">
            <p className="font-medium">{error}</p>
          </div>
        )}

        {/* No races */}
        {!loadingDates && !loadingSchedules && daySchedules.length === 0 && !error && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 text-center py-16">
            <p className="text-lg font-bold text-gray-400">レースが見つかりません</p>
            <p className="text-sm text-gray-400 mt-1">別の週を選択してください</p>
          </div>
        )}

        {/* Race list by day */}
        {!loadingSchedules && daySchedules.map((day) => (
          <div key={day.date} className="space-y-3">
            {/* Day header */}
            <div className="flex items-center gap-2">
              <span className="text-sm font-black text-[#1a472a] bg-[#e8f5e9] rounded-lg px-3 py-1.5 border border-[#c8e6c9]">
                {day.display}
              </span>
              <div className="flex-1 h-px bg-gray-200" />
            </div>

            {/* Course cards */}
            {day.schedules.map((schedule) => (
              <div
                key={`${day.date}-${schedule.code}`}
                className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden"
              >
                {/* Course header */}
                <div className="bg-gradient-to-r from-[#2d5a3a] to-[#1e4a2e] px-4 py-2.5 flex items-center gap-2">
                  <span className="text-white font-black text-sm">{schedule.name}</span>
                  <span className="text-green-200 text-[10px]">
                    {schedule.races.length}レース
                  </span>
                </div>

                {/* Race list */}
                <div className="divide-y divide-gray-100">
                  {schedule.races
                    .sort((a, b) => a.raceNumber - b.raceNumber)
                    .map((race) => (
                    <Link
                      key={race.raceId}
                      href={`/race/${race.raceId}`}
                      className="flex items-center px-4 py-3 hover:bg-[#f5faf5] transition-colors group"
                    >
                      {/* Race number */}
                      <span className="w-10 text-center font-black text-sm text-[#1a472a] shrink-0">
                        {race.raceNumber}R
                      </span>

                      {/* Race info */}
                      <div className="flex-1 min-w-0 ml-2">
                        <div className="font-bold text-sm text-gray-900 truncate group-hover:text-[#1a472a]">
                          {race.raceName || `${race.raceNumber}R`}
                        </div>
                        {race.startTime && (
                          <span className="text-[11px] text-gray-400">{race.startTime}発走</span>
                        )}
                      </div>

                      {/* Arrow */}
                      <svg
                        className="w-4 h-4 text-gray-300 group-hover:text-[#1a472a] transition-colors shrink-0 ml-2"
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>

      <footer className="mt-8 bg-[#1a472a] text-green-200 text-center py-4 text-[10px]">
        KEIBA ORACLE - AI Racing Intelligence
      </footer>
    </div>
  );
}
