"use client";

import { RaceInfo } from "@/lib/types";

interface Props {
  raceInfo: RaceInfo;
}

const GRADE_STYLES: Record<string, string> = {
  GI: "bg-[#c5a030] text-white border-[#9a7a1a] shadow-md",
  GII: "bg-[#8a8a8a] text-white border-[#666] shadow-md",
  GIII: "bg-[#c06820] text-white border-[#8a4a10] shadow-md",
};

export default function RaceHeader({ raceInfo }: Props) {
  const gradeStyle = raceInfo.grade ? GRADE_STYLES[raceInfo.grade] : null;

  return (
    <div className="border-b-2 border-[#2d6b3f]">
      {/* Race name bar */}
      <div className="bg-gradient-to-b from-[#1e5c30] to-[#164422] px-5 py-3 flex items-center gap-3">
        <span className="bg-white text-[#1a472a] w-12 h-12 rounded-lg flex items-center justify-center font-black text-lg shadow-md border border-gray-200">
          {raceInfo.raceNumber}R
        </span>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-2xl font-black text-white tracking-wide drop-shadow-sm">
              {raceInfo.raceName}
            </h2>
            {gradeStyle && (
              <span className={`px-3 py-0.5 rounded-full text-xs font-black border-2 ${gradeStyle}`}>
                {raceInfo.grade}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Race condition bar */}
      <div className="bg-[#f0f4f0] border-t border-[#d0ddd0] px-5 py-2.5 flex items-center gap-3 text-sm flex-wrap">
        {/* Surface badge */}
        <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-bold border ${
          raceInfo.surface === "芝"
            ? "bg-[#e8f5e9] text-[#2e7d32] border-[#a5d6a7]"
            : raceInfo.surface === "ダート"
            ? "bg-[#efebe9] text-[#5d4037] border-[#bcaaa4]"
            : "bg-gray-100 text-gray-600 border-gray-300"
        }`}>
          <span className={`w-2 h-2 rounded-full ${
            raceInfo.surface === "芝" ? "bg-[#4caf50]" :
            raceInfo.surface === "ダート" ? "bg-[#8d6e63]" : "bg-gray-500"
          }`} />
          {raceInfo.surface}
        </span>

        <span className="font-black text-gray-800 text-base">{raceInfo.distance}m</span>
        <span className="text-gray-500 text-xs">({raceInfo.courseDetail})</span>

        <span className="w-px h-4 bg-gray-300" />

        <span className="text-gray-500 text-xs">発走</span>
        <span className="font-bold text-gray-800">{raceInfo.startTime}</span>

        {raceInfo.headCount > 0 && (
          <>
            <span className="w-px h-4 bg-gray-300" />
            <span className="text-gray-500 text-xs">出走</span>
            <span className="font-bold text-gray-800">{raceInfo.headCount}頭</span>
          </>
        )}
      </div>
    </div>
  );
}
