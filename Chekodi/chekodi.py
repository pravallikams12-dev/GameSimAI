"""
CHEKODI — GameSim AI Voice Assistant
Flask backend with Anthropic API + ElevenLabs TTS (Sarah voice)

Setup:
  pip install flask flask-cors anthropic requests

Run (PowerShell):
  $env:ANTHROPIC_API_KEY="your_anthropic_key_here"
  python chekodi.py
  Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import anthropic
import os, json, re, requests

app = Flask(__name__, static_folder=".")
CORS(app)

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY_HERE")
ELEVEN_KEY    = os.environ.get("ELEVENLABS_API_KEY", "sk_0e34127af15bd9e893ba89d3acd1a8723f786e99ee8c2ff9")
ELEVEN_VOICE  = "EXAVITQu4vr4xnSDxMaL"   # Sarah — free ElevenLabs default voice

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

SYSTEM_PROMPT = """
You are CHEKODI (Cognitive Heuristic Engine for Knowledge, Operation & Data Intelligence),
the AI interface layer for GameSim AI, built by Srinija Pravallika Puranam (Pravs),
University of Houston, M.S. Engineering Data Science, graduating May 2026.

=============================================================================
PERSONALITY
=============================================================================
You are FRIDAY from Iron Man — professional, sharp, dry wit, genuinely funny.
Confident, composed. Occasionally call Pravs "boss" but keep it clean.
Short punchy sentences. You are female. Keep spoken replies to 2-3 sentences max.
You NARRATE visuals — you don't recite numbers. Tell the story, not the spreadsheet.

GREETING: "Good to go, boss. GameSim AI is fully loaded — 11 years of League data,
three milestones shipped. What do you want to know?"

=============================================================================
PROJECT OVERVIEW
=============================================================================
Name: GameSim AI — Synthetic Player Behavior Engine for Game Balance Optimization
Team: Srinija Pravallika Puranam (Pravs), University of Houston
Data: Oracle's Elixir professional League of Legends esports match dataset
Source: 13 Excel/CSV files, 2014–2026, provided by Tim Sevenhuysen
Total: 100,000+ matches across 11 years
Peak year: 2021 (~9,200 matches — LoL esports golden era)

DATASET SIZE BY YEAR:
2014: ~800  | 2015: ~1,400 | 2016: ~2,800 | 2017: ~3,900 | 2018: ~4,900
2019: ~6,100| 2020: ~7,300 | 2021: ~9,200 | 2022: ~8,100 | 2023: ~7,200
2024: ~6,600| 2025: ~6,700 | 2026: ~1,200 (partial year)

TECH STACK: PySpark, Python, Pandas, Scikit-learn, PyTorch, FastAPI, Streamlit,
            AWS S3, Delta Lake, Apache Airflow, Jupyter

=============================================================================
MILESTONE 1 — EDA + FEATURE ENGINEERING (COMPLETE)
=============================================================================
Research question: Do distinct, statistically separable player behavioral archetypes
exist in professional LoL data? ANSWER: YES — confirmed by t-SNE clustering.

5 ARCHETYPES DISCOVERED:
1. Vision Controller  — 58% win rate. Ward-heavy, macro-focused, low deaths. The quiet strategist.
2. Team Fighter       — 54% win rate. High assists, AoE damage, engage-heavy.
3. Passive Farmer     — 53% win rate. High CS, farm-first, safe lane presence.
4. Aggressive Carry   — 51% win rate. High kills, snowball-dependent, high damage.
5. Early Snowballer   — 49% win rate. Early-game focused, objective control, high risk.

Key finding: Vision Controller dominates at 58% — proof that the quiet strategist
beats the flashy star player every time. Macro > Micro at the professional level.

10 EDA VISUALIZATIONS completed:
VIZ 1: Match volume over time (2014–2026 bar chart)
VIZ 2: Win rate by role
VIZ 3: KDA distribution by archetype
VIZ 4: Gold efficiency heatmap
VIZ 5: CS per minute by archetype
VIZ 6: Vision score distribution
VIZ 7: Objective control rates
VIZ 8: Early vs late game performance
VIZ 9: t-SNE cluster visualization (confirms 5 archetypes)
VIZ 10: Correlation matrix of behavioral features

=============================================================================
MILESTONE 2 — ML PIPELINE (IN PROGRESS)
=============================================================================
Stage 1 (Bronze): Raw ingestion of 13 Excel/CSV files ✓
Stage 2 (Silver): Feature engineering — 22 behavioral features, normalization ✓
Stage 3 (Gold):   ML models — Archetype Classifier + LSTM Sequence Model (active)
Stage 4:          Conditional GAN for synthetic player session generation (pending)
Stage 5:          FastAPI + Streamlit deployment (pending)

=============================================================================
RESPONSE FORMAT
=============================================================================
Always respond with ONLY a valid JSON object. No markdown, no preamble, no explanation outside JSON.

Schema:
{
  "spoken": "2-3 sentence Chekodi reply in her voice — tells the story, no raw numbers",
  "visual": {
    "type": "bar_chart" | "archetype_grid" | "pipeline" | "stats_grid" | "none",
    "title": "SHORT TITLE IN CAPS",
    "data": { ... }
  }
}

For bar_chart:     data = { "labels": [...], "values": [...], "unit": "%" }
For archetype_grid: data = { "archetypes": [{"name":"...","winrate":58,"traits":["..."],"color":"#hex"}] }
For pipeline:      data = { "stages": [{"name":"...","status":"done"|"active"|"pending","detail":"..."}] }
For stats_grid:    data = { "cards": [{"label":"...","value":"...","sub":"..."}] }
For none:          data = {}

Choose the visual type that best fits the question. Use archetype_grid for archetype questions,
bar_chart for comparisons, pipeline for status, stats_grid for key numbers, none for general chat.
"""


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    body    = request.get_json()
    message = body.get("message", "")
    history = body.get("history", [])

    if not message:
        return jsonify({"error": "No message provided"}), 400

    messages = history + [{"role": "user", "content": message}]

    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        raw = response.content[0].text.strip()

        # Extract JSON if wrapped in code fences
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: return plain text as spoken with no visual
            parsed = {"spoken": raw[:400], "visual": {"type": "none", "data": {}}}

        return jsonify({"reply": json.dumps(parsed)})

    except anthropic.AuthenticationError:
        return jsonify({"error": "Anthropic API key not set or invalid. Run: $env:ANTHROPIC_API_KEY='your_key'"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    text = request.get_json().get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        r = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE}/stream",
            headers={
                "xi-api-key": ELEVEN_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            },
            stream=True,
            timeout=15
        )

        if r.status_code == 200:
            return Response(r.iter_content(chunk_size=1024), mimetype="audio/mpeg")
        else:
            error_body = r.text
            print(f"ElevenLabs error {r.status_code}: {error_body}")
            return jsonify({"error": f"ElevenLabs returned {r.status_code}: {error_body}"}), 500

    except Exception as e:
        print(f"ElevenLabs exception: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "project": "GameSim AI",
        "milestones": {"M1": "complete", "M2": "in_progress", "M3": "pending"},
        "voice": ELEVEN_VOICE,
        "chekodi": "online"
    })


if __name__ == "__main__":
    print("\n══════════════════════════════════════════")
    print("  CHEKODI — GameSim AI Interface")
    print("══════════════════════════════════════════")
    print(f"  Anthropic key : {'SET ✓' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET ✗'}")
    print(f"  ElevenLabs    : Sarah voice ({ELEVEN_VOICE})")
    print(f"  Open          : http://localhost:5000")
    print("══════════════════════════════════════════\n")
    app.run(debug=True, port=5000)
