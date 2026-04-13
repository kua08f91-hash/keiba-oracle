"use client";

import { RacecourseCode } from "@/lib/types";
import { RACECOURSES } from "@/lib/constants";

interface Props {
  selectedCourse: RacecourseCode | null;
  availableCourses: Set<string>;
  onSelect: (code: RacecourseCode) => void;
}

export default function RacecourseTab({ selectedCourse, availableCourses, onSelect }: Props) {
  return (
    <div className="flex gap-0 rounded-xl overflow-hidden shadow-sm border border-gray-200">
      {RACECOURSES.map((course, idx) => {
        const isAvailable = availableCourses.has(course.code);
        const isSelected = selectedCourse === course.code;
        return (
          <button
            key={course.code}
            onClick={() => isAvailable && onSelect(course.code)}
            disabled={!isAvailable}
            className={`
              flex-1 px-2 py-2.5 text-sm font-black transition-all
              ${idx < RACECOURSES.length - 1 ? "border-r border-gray-200" : ""}
              ${isSelected
                ? "bg-gradient-to-b from-[#1e5c30] to-[#164422] text-white shadow-inner"
                : isAvailable
                  ? "bg-white text-gray-700 hover:bg-[#f0f7f0]"
                  : "bg-gray-50 text-gray-300 cursor-not-allowed"
              }
            `}
          >
            {course.name}
          </button>
        );
      })}
    </div>
  );
}
