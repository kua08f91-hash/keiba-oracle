"""Claude Fable 5 integration for race analysis, weight suggestions, and text analysis.

All functions gracefully degrade when ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
_client = None


def _get_client():
    """Get or create Anthropic client. Returns None if API key not set."""
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
        return _client
    except Exception as e:
        logger.warning("Failed to initialize Anthropic client: %s", e)
        return None


def is_available() -> bool:
    """Check if LLM integration is available."""
    return _get_client() is not None


# ─── Feature 1: Race Analysis Text Generation ───


def generate_race_analysis(race_info: dict, entries: list, predictions: list) -> str:
    """Generate AI narrative analysis for a race.

    Returns analysis text in Japanese, or empty string if unavailable.
    """
    client = _get_client()
    if not client:
        return ""

    try:
        # Build context
        race_name = race_info.get("raceName", "")
        distance = race_info.get("distance", 0)
        surface = race_info.get("surface", "")
        condition = race_info.get("trackCondition", "")
        head_count = race_info.get("headCount", len(entries))
        grade = race_info.get("grade", "")

        # Top horses by prediction score
        sorted_preds = sorted(predictions, key=lambda p: -p.get("score", 0))
        top5 = sorted_preds[:5]

        horse_details = []
        for p in top5:
            hn = p.get("horseNumber", 0)
            entry = next((e for e in entries if e.get("horseNumber") == hn), {})
            horse_details.append(
                "  %s %d番 %s (odds=%.1f, score=%.1f, 騎手=%s)" % (
                    p.get("mark", ""),
                    hn,
                    entry.get("horseName", "?"),
                    entry.get("odds", 0) or 0,
                    p.get("score", 0),
                    entry.get("jockeyName", "?"),
                )
            )

        prompt = """以下のJRA競馬レースについて、150字以内で簡潔な分析コメントを日本語で生成してください。
展望・注目ポイント・波乱要素を含めてください。

レース: %s %s
距離: %dm %s
馬場: %s
頭数: %d頭
グレード: %s

AI予想上位5頭:
%s

分析コメント:""" % (
            race_name, grade or "",
            distance, surface,
            condition or "未定",
            head_count,
            grade or "一般",
            "\n".join(horse_details),
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        logger.info("Generated analysis for %s (%d chars)", race_name, len(text))
        return text

    except Exception as e:
        logger.warning("generate_race_analysis failed: %s", e)
        return ""


# ─── Feature 2: Factor Weight Optimization Suggestions ───


def suggest_weight_adjustments(performance_data: dict, current_weights: dict) -> dict:
    """Analyze recent performance and suggest weight adjustments.

    Returns dict with suggestions and summary, or empty dict if unavailable.
    """
    client = _get_client()
    if not client:
        return {}

    try:
        prompt = """あなたは競馬AI予想システムの重み最適化アドバイザーです。

現在の22ファクターの重み:
%s

直近1ヶ月の成績:
%s

上記データを分析し、以下のJSON形式で重み調整提案を出してください:
{
  "suggestions": [
    {"factor": "ファクター名", "current": 現在値, "suggested": 提案値, "reasoning": "理由"}
  ],
  "summary": "全体的な所見（100字以内）"
}

JSONのみ出力してください。""" % (
            json.dumps(current_weights, indent=2, ensure_ascii=False),
            json.dumps(performance_data, indent=2, ensure_ascii=False),
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        logger.info("Generated %d weight suggestions", len(result.get("suggestions", [])))
        return result

    except Exception as e:
        logger.warning("suggest_weight_adjustments failed: %s", e)
        return {}


# ─── Feature 3: News/Paddock Text Analysis ───


def analyze_external_text(text: str, race_info: dict) -> dict:
    """Extract structured signals from free-form Japanese text.

    Returns dict with signals list, or empty dict if unavailable.
    """
    client = _get_client()
    if not client:
        return {}

    try:
        race_name = race_info.get("raceName", "")
        prompt = """以下はJRA競馬レース「%s」に関する情報テキストです。
テキストから競馬予想に影響する要素を抽出し、以下のJSON形式で返してください:

{
  "signals": [
    {
      "horse": "馬名",
      "form_change": "up" | "down" | "neutral",
      "risk_flags": ["フラグ1", "フラグ2"],
      "key_changes": ["変更点1"],
      "adjustment": -10〜+10の整数（スコア修正値）
    }
  ],
  "summary": "全体的な分析（50字以内）"
}

テキスト:
%s

JSONのみ出力してください。""" % (race_name, text)

        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        resp_text = response.content[0].text.strip()

        if resp_text.startswith("```"):
            resp_text = resp_text.split("```")[1]
            if resp_text.startswith("json"):
                resp_text = resp_text[4:]
        result = json.loads(resp_text)
        logger.info("Extracted %d signals from text", len(result.get("signals", [])))
        return result

    except Exception as e:
        logger.warning("analyze_external_text failed: %s", e)
        return {}
