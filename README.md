# 🎮 GameSim AI
### Synthetic Player Behavior Engine for Game Balance Optimization

**University of Houston · M.S. Engineering Data Science · May 2026**  
**Srinija Pravallika Puranam**

---

## What is this?

GameSim AI learns behavioral patterns from 12 years of professional League of Legends esports data and generates synthetic players that behave like real ones — statistically validated, archetype-specific, and scalable.

Game studios spend **$2–5 million per patch cycle** on human playtesting. GameSim AI replaces that.

---

## Repository Structure

```
GameSimAI/
│
├── Chekodi/                        # Voice AI interface layer
│   ├── chekodi.py                  # Flask backend — Claude API + ElevenLabs TTS
│   ├── index.html                  # Frontend — voice in, voice out
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Chekodi setup guide
│
├── GameSimAI_Milestone2.ipynb      # Full pipeline — Bronze → Silver → Gold → ML
├── GameSimAI_GAN_v2.py             # Conditional GAN training script (500 epochs)
├── GameSimAI_Dashboard_v2.html     # Interactive client-facing dashboard
│
├── gamesim_gan_generator.h5        # Trained GAN generator weights
├── gamesim_gan_discriminator.h5    # Trained GAN discriminator weights
├── gamesim_lstm.h5                 # Trained LSTM win predictor weights
│
├── gan_metadata.json               # GAN training config and archetype mappings
├── archetype_insights.png          # Win rate + KDA charts per archetype
├── tsne_archetypes.png             # t-SNE cluster visualization
├── gan_training_loss.png           # GAN generator vs discriminator loss curves
├── gan_distribution_comparison.png # Synthetic vs real player feature distributions
├── gan_validation.png              # KS-test validation results
└── lstm_training.png               # LSTM accuracy and loss curves
```

---

## Pipeline Architecture

```
Raw CSVs (Oracle's Elixir, 2014–2026)
            │
            ▼
    ┌───────────────┐
    │  BRONZE LAYER │  13 files · 1M+ rows · raw ingestion via PySpark
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  SILVER LAYER │  Schema enforcement · deduplication · normalization
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   GOLD LAYER  │  18 engineered features · archetype labels · model-ready
    └───────┬───────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
  LSTM           Conditional GAN
  Win Predictor  Synthetic Player Generator
     │             │
     └──────┬──────┘
            │
            ▼
        CHEKODI
   Voice AI Interface
```

---

## 5 Archetypes Discovered

| Archetype | Win Rate | Signature |
|---|---|---|
| 🔵 Vision Controller | **58%** | High wards · low deaths · macro dominance |
| 🟡 Team Fighter | 54% | High assists · AoE engage |
| 🟣 Passive Farmer | 53% | High CS · late game scaling |
| 🔴 Aggressive Carry | 51% | High kills · snowball dependent |
| 🟢 Early Snowballer | 49% | Early gold diff · high variance |

**Key finding:** Vision Controller wins at 58% with the lowest KDA. Map awareness beats mechanics at the professional level.

---

## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | PySpark, Delta Lake |
| Feature Engineering | Pandas, NumPy, Scikit-learn |
| Clustering | PCA + KMeans · t-SNE (viz only) |
| Sequence Model | PyTorch LSTM |
| Generative Model | PyTorch Conditional GAN |
| Orchestration | Apache Airflow |
| Backend | Flask |
| LLM | Anthropic Claude Sonnet |
| Voice TTS | ElevenLabs (eleven-turbo-v2) |
| Frontend | HTML · CSS · JavaScript · Web Speech API |

---

## Milestones

| Milestone | Status | Description |
|---|---|---|
| M1 | ✅ Complete | EDA · 10 visualizations · archetype discovery via t-SNE |
| M2 | ✅ Complete | Archetype classifier · LSTM · Conditional GAN · CHEKODI |
| M3 | 🔄 Pending | FastAPI deployment · Streamlit dashboard · production scaling |

---

## Quick Start

```bash
# Run the notebook end to end
jupyter notebook GameSimAI_Milestone2.ipynb

# Launch the dashboard
open GameSimAI_Dashboard_v2.html

# Run Chekodi voice interface
cd Chekodi
pip install -r requirements.txt
python chekodi.py
# Open http://127.0.0.1:5000
```

---

## GAN Validation

Synthetic player validity confirmed via **Kolmogorov-Smirnov test** per feature.  
p-value > 0.05 = synthetic distribution statistically indistinguishable from real players.

---

## Data Source

[Oracle's Elixir](https://oracleselixir.com/) · Professional LoL match data  
13 files · 2014–2026 · 100,000+ matches · 165 raw features

---

*M.S. Engineering Data Science Capstone · University of Houston · May 2026*
