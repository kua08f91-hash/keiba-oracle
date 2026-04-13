"use client";

import { FRAME_COLORS } from "@/lib/constants";

interface Props {
  frameNumber: number;
  size?: "sm" | "md";
}

export default function FrameColorBox({ frameNumber, size = "md" }: Props) {
  const color = FRAME_COLORS[frameNumber] || FRAME_COLORS[1];
  const sizeClass = size === "sm" ? "w-5 h-5 text-[10px]" : "w-7 h-7 text-xs";
  return (
    <span
      className={`inline-flex items-center justify-center ${sizeClass} rounded-full font-black shadow-sm`}
      style={{
        backgroundColor: color.bg,
        color: color.text,
        border: `2px solid ${color.border}`,
      }}
    >
      {frameNumber}
    </span>
  );
}
