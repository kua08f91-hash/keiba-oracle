"use client";

import { RACE_NUMBERS } from "@/lib/constants";

interface Props {
  selectedRace: number;
  availableRaces: Set<number>;
  onSelect: (n: number) => void;
}

export default function RaceNumberTab({ selectedRace, availableRaces, onSelect }: Props) {
  return (
    <div className="flex gap-0 rounded-xl overflow-hidden shadow-sm border border-gray-200">
      {RACE_NUMBERS.map((n, idx) => {
        const isAvailable = availableRaces.has(n);
        const isSelected = selectedRace === n;
        return (
          <button
            key={n}
            onClick={() => isAvailable && onSelect(n)}
            disabled={!isAvailable}
            className={`
              flex-1 px-1 py-2 text-sm font-black transition-all min-w-[40px]
              ${idx < RACE_NUMBERS.length - 1 ? "border-r border-gray-200" : ""}
              ${isSelected
                ? "bg-gradient-to-b from-[#1e5c30] to-[#164422] text-white shadow-inner"
                : isAvailable
                  ? "bg-white text-gray-700 hover:bg-[#f0f7f0]"
                  : "bg-gray-50 text-gray-300 cursor-not-allowed"
              }
            `}
          >
            {n}R
          </button>
        );
      })}
    </div>
  );
}
