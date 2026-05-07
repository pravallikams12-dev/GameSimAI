# CHEKODI — Setup Guide
GameSim AI Voice Assistant · University of Houston · Srinija Pravallika Puranam

---

## What's in this folder

| File             | What it does                          |
|------------------|---------------------------------------|
| chekodi.py       | Python backend (Flask + Claude API)   |
| index.html       | The voice UI (runs in your browser)   |
| requirements.txt | Python packages needed                |
| README.md        | This file                             |

---

## Step 1 — Place the folder

Copy the entire `chekodi/` folder inside your GameSim AI project directory:

  OE Public Match Data/
  ├── 2014_LoL_esports_match_data...xlsx
  ├── ...all your data files...
  ├── GameSim_AI_Milestone1.ipynb
  └── chekodi/               ← put it here
      ├── chekodi.py
      ├── index.html
      ├── requirements.txt
      └── README.md

---

## Step 2 — Install dependencies (one time only)

Open a terminal / Anaconda Prompt and run:

  pip install anthropic flask flask-cors

You likely already have anthropic installed from your grader project.
If you get errors, try:

  pip install anthropic flask flask-cors --upgrade

---

## Step 3 — Set your API key

In your terminal (same one you'll use to run the server):

  Windows (Command Prompt):
    set ANTHROPIC_API_KEY=your_key_here

  Windows (PowerShell):
    $env:ANTHROPIC_API_KEY="your_key_here"

  Mac/Linux:
    export ANTHROPIC_API_KEY=your_key_here

Use the SAME API key you used for your automated grader project.
You can find it at: https://console.anthropic.com

---

## Step 4 — Run Chekodi

In your terminal, navigate to the chekodi folder:

  cd "C:\Users\prava\OneDrive\Desktop\AI Engg\OE Public Match Data-...\OE Public Match Data\chekodi"

Then run:

  python chekodi.py

You should see:
  CHEKODI powering up... okay she's ready.
  API key: SET — let's go!
  Open: http://localhost:5000

---

## Step 5 — Open in browser

Open Chrome or Edge and go to:

  http://localhost:5000

Chekodi will greet you automatically. No login, no setup, nothing else.

---

## Voice — No Training Required

Chekodi uses TWO built-in voice systems. Zero training, zero setup:

VOICE INPUT (you speaking to her):
  - Uses Web Speech API — built into Chrome and Edge
  - Click the microphone button, speak, it transcribes automatically
  - Works best in Chrome. Does NOT work in Firefox.
  - No account needed, no API key, completely free

VOICE OUTPUT (her speaking to you):
  - Uses your browser's built-in Text-to-Speech
  - She will automatically pick the best available voice on your system
  - Windows has "Microsoft Zira" / "Microsoft Aria" built in
  - No setup needed — it just works

---

## Quick-ask buttons

These are pre-loaded in the UI for demos:
  - Archetypes      → shows all 5 player behavior cards
  - Pipeline        → shows full 7-stage pipeline status
  - Dataset         → shows match count bar chart by year
  - Best archetype  → explains Vision Controller win rate
  - Key stats       → shows project stat cards
  - Vision profile  → radar chart of Vision Controller

---

## Stopping the server

In your terminal, press:  Ctrl + C

---

## Troubleshooting

"API key not configured"
  → You forgot Step 3. Set the key and restart python chekodi.py

"Cannot reach backend"
  → chekodi.py is not running. Run it again in terminal.

Voice button does nothing
  → Use Chrome or Edge. Firefox does not support Web Speech API.

Mic permission denied
  → Click the lock icon in your browser address bar → allow microphone

Port already in use
  → Another program is on port 5000. In chekodi.py, change port=5000 to port=5001
    Then open http://localhost:5001 instead

---

Built for GameSim AI Capstone · M.S. Engineering Data Science · University of Houston · May 2026
