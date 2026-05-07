# ============================================================
# GAMESIM AI — CONDITIONAL GAN
# Run this AFTER GameSimAI_Milestone2.ipynb has completed
# It reads gold_features.csv and trains a Conditional GAN
# to generate realistic synthetic LoL player stats
# ============================================================

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import json

print("="*60)
print("🎮 GAMESIM AI — CONDITIONAL GAN")
print("   Training synthetic player behavior engine...")
print("="*60)

# ── LOAD DATA ──────────────────────────────────────────────
print("\n📂 Loading gold_features.csv...")
try:
    df = pd.read_csv("gold_features.csv")
    print(f"   Loaded {len(df):,} rows")
except FileNotFoundError:
    print("❌ gold_features.csv not found!")
    print("   Run GameSimAI_Milestone2.ipynb first.")
    exit()

# ── FEATURE SELECTION ──────────────────────────────────────
GAN_FEATURES = ['kda', 'cspm', 'vision_per_min', 'dpm',
                'early_gold_advantage', 'kill_participation',
                'wardsplaced', 'is_carry', 'is_support', 'is_jungle']

available = [f for f in GAN_FEATURES if f in df.columns]
print(f"   Using {len(available)} features: {available}")

# Load archetype labels if saved, else re-cluster
if 'archetype' in df.columns:
    gan_df = df[available + ['archetype']].dropna()
else:
    print("   No archetype column found — using position-based approximation")
    df['archetype'] = 'Team Fighter'
    if 'position' in df.columns:
        df.loc[df['position']=='bot', 'archetype'] = 'Aggressive Carry'
        df.loc[df['position']=='sup', 'archetype'] = 'Vision Controller'
        df.loc[df['position']=='jng', 'archetype'] = 'Early Snowballer'
        df.loc[df['position']=='top', 'archetype'] = 'Passive Farmer'
    gan_df = df[available + ['archetype']].dropna()

ARCHETYPE_NAMES = sorted(gan_df['archetype'].unique().tolist())
N_CLASSES = len(ARCHETYPE_NAMES)
N_FEATURES = len(available)

print(f"   Archetypes: {ARCHETYPE_NAMES}")
print(f"   Classes: {N_CLASSES} | Features: {N_FEATURES}")

# Sample for training speed
sample_size = min(80000, len(gan_df))
gan_df = gan_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# Encode labels
le = LabelEncoder()
le.fit(ARCHETYPE_NAMES)
y_int = le.transform(gan_df['archetype'])

# Scale features
scaler_gan = StandardScaler()
X_real = scaler_gan.fit_transform(gan_df[available].fillna(0))

print(f"\n✅ Data ready: {len(X_real):,} samples")

# ── HYPERPARAMETERS ────────────────────────────────────────
NOISE_DIM    = 64
BATCH_SIZE   = 256
EPOCHS       = 2000
LR_GEN       = 0.0002
LR_DISC      = 0.0001
LABEL_SMOOTH = 0.1

# ── BUILD GENERATOR ────────────────────────────────────────
def build_generator(noise_dim, n_classes, n_features):
    noise_in = layers.Input(shape=(noise_dim,), name='noise_input')
    label_in = layers.Input(shape=(1,), name='label_input')

    label_emb = layers.Embedding(n_classes, 16)(label_in)
    label_emb = layers.Flatten()(label_emb)

    x = layers.Concatenate()([noise_in, label_emb])
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    out = layers.Dense(n_features, activation='tanh')(x)

    return Model([noise_in, label_in], out, name='Generator')

# ── BUILD DISCRIMINATOR ────────────────────────────────────
def build_discriminator(n_features, n_classes):
    feat_in  = layers.Input(shape=(n_features,), name='feat_input')
    label_in = layers.Input(shape=(1,), name='label_input')

    label_emb = layers.Embedding(n_classes, 16)(label_in)
    label_emb = layers.Flatten()(label_emb)

    x = layers.Concatenate()([feat_in, label_emb])
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return Model([feat_in, label_in], out, name='Discriminator')

generator     = build_generator(NOISE_DIM, N_CLASSES, N_FEATURES)
discriminator = build_discriminator(N_FEATURES, N_CLASSES)

gen_optimizer  = optimizers.Adam(LR_GEN,  beta_1=0.5)
disc_optimizer = optimizers.Adam(LR_DISC, beta_1=0.5)

bce = tf.keras.losses.BinaryCrossentropy()

generator.summary()
discriminator.summary()

# ── TRAINING STEP ─────────────────────────────────────────
@tf.function
def train_step(real_feats, real_labels):
    batch_size = tf.shape(real_feats)[0]
    noise = tf.random.normal([batch_size, NOISE_DIM])
    fake_labels = tf.random.uniform([batch_size, 1], 0, N_CLASSES, dtype=tf.int32)

    # ── Discriminator ──
    with tf.GradientTape() as disc_tape:
        fake_feats = generator([noise, fake_labels], training=True)
        real_out   = discriminator([real_feats, real_labels], training=True)
        fake_out   = discriminator([fake_feats, fake_labels], training=True)

        real_loss = bce(tf.ones_like(real_out)  * (1 - LABEL_SMOOTH), real_out)
        fake_loss = bce(tf.zeros_like(fake_out) + LABEL_SMOOTH,        fake_out)
        disc_loss = (real_loss + fake_loss) / 2

    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # ── Generator (train 2x per disc step) ──
    gen_total_loss = 0
    for _ in range(2):
        noise2      = tf.random.normal([batch_size, NOISE_DIM])
        fake_labs2  = tf.random.uniform([batch_size, 1], 0, N_CLASSES, dtype=tf.int32)
        with tf.GradientTape() as gen_tape:
            fake_feats2 = generator([noise2, fake_labs2], training=True)
            fake_out2   = discriminator([fake_feats2, fake_labs2], training=False)
            gen_loss    = bce(tf.ones_like(fake_out2), fake_out2)
        gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
        gen_total_loss += gen_loss

    return disc_loss, gen_total_loss / 2

# ── DATASET ────────────────────────────────────────────────
X_tf = tf.constant(X_real, dtype=tf.float32)
y_tf = tf.constant(y_int.reshape(-1,1), dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((X_tf, y_tf)) \
    .shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── TRAIN ──────────────────────────────────────────────────
print(f"\n🧠 Training Conditional GAN — {EPOCHS} epochs...")
print(f"   Batch size: {BATCH_SIZE} | Noise dim: {NOISE_DIM}")
print(f"   This will take 40–80 minutes on CPU (2000 epochs)...")
print()

d_losses, g_losses = [], []
FIXED_NOISE  = tf.random.normal([N_CLASSES * 10, NOISE_DIM], seed=42)
FIXED_LABELS = tf.constant(
    np.repeat(np.arange(N_CLASSES), 10).reshape(-1,1), dtype=tf.int32
)

for epoch in range(EPOCHS):
    epoch_d, epoch_g = [], []
    for real_feats, real_labels in dataset:
        d_loss, g_loss = train_step(real_feats, real_labels)
        epoch_d.append(float(d_loss))
        epoch_g.append(float(g_loss))

    avg_d = np.mean(epoch_d)
    avg_g = np.mean(epoch_g)
    d_losses.append(avg_d)
    g_losses.append(avg_g)

    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:>3}/{EPOCHS} | D-loss: {avg_d:.4f} | G-loss: {avg_g:.4f}")

print("\n✅ GAN training complete!")

# ── GENERATE SYNTHETIC PLAYERS ─────────────────────────────
print("\n🎮 Generating synthetic players from trained GAN...")

def gan_generate_player(archetype_name, n=1):
    arch_idx = le.transform([archetype_name])[0]
    noise    = tf.random.normal([n, NOISE_DIM])
    labels   = tf.constant([[arch_idx]] * n, dtype=tf.int32)
    fake     = generator([noise, labels], training=False).numpy()
    players  = scaler_gan.inverse_transform(fake)
    results  = []
    for p in players:
        player = {feat: float(p[i]) for i, feat in enumerate(available)}
        player['archetype'] = archetype_name
        results.append(player)
    return results

# Generate 50 players per archetype
all_synthetic = []
for arch in ARCHETYPE_NAMES:
    players = gan_generate_player(arch, n=5000)
    all_synthetic.extend(players)

synthetic_df = pd.DataFrame(all_synthetic)
synthetic_df.to_csv('gan_synthetic_players.csv', index=False)
print(f"   Generated {len(synthetic_df)} synthetic players")
print("   💾 Saved → gan_synthetic_players.csv")

# ── QUALITY EVALUATION ─────────────────────────────────────
print("\n📊 Evaluating GAN quality (Real vs Synthetic)...")

# Compare distributions per archetype
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0D1117')

COLORS = {'Aggressive Carry':'#E63946', 'Vision Controller':'#457B9D',
          'Passive Farmer':'#2A9D8F', 'Early Snowballer':'#E9C46A',
          'Team Fighter':'#F4A261'}

features_to_plot = available[:6]
n_feat = len(features_to_plot)
n_arch = len(ARCHETYPE_NAMES)

for fi, feat in enumerate(features_to_plot):
    ax = fig.add_subplot(2, 3, fi + 1)
    ax.set_facecolor('#0D1117')

    for arch in ARCHETYPE_NAMES:
        color = COLORS.get(arch, '#888')
        real_vals = gan_df[gan_df['archetype']==arch][feat].dropna().values
        fake_rows = synthetic_df[synthetic_df['archetype']==arch]
        if feat in fake_rows.columns:
            fake_vals = fake_rows[feat].values
        else:
            continue

        if len(real_vals) > 0:
            ax.hist(real_vals, bins=30, alpha=0.4, color=color, density=True, label=f'{arch} (real)')
            ax.hist(fake_vals, bins=30, alpha=0.7, color=color, density=True,
                    histtype='step', linewidth=2, label=f'{arch} (synth)')

    ax.set_title(feat.replace('_',' ').title(), color='white', fontsize=11)
    ax.tick_params(colors='#8B8B8B', labelsize=8)
    for spine in ax.spines.values(): spine.set_color('#2D2D2D')
    ax.set_xlabel('', color='#8B8B8B')

plt.suptitle('🎮 GameSim AI — GAN Quality: Real vs Synthetic Distribution',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('gan_distribution_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("   💾 Saved → gan_distribution_comparison.png")

# ── TRAINING LOSS PLOT ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')
ax.plot(d_losses, color='#E63946', linewidth=2, label='Discriminator Loss')
ax.plot(g_losses, color='#2A9D8F', linewidth=2, label='Generator Loss')
ax.set_title('🎮 GameSim AI — Conditional GAN Training', color='white', fontsize=14)
ax.set_xlabel('Epoch', color='#8B8B8B')
ax.set_ylabel('Loss', color='#8B8B8B')
ax.tick_params(colors='#8B8B8B')
for spine in ax.spines.values(): spine.set_color('#2D2D2D')
ax.legend(facecolor='#1C1C1C', labelcolor='white', fontsize=11)
ax.axhline(y=0.693, color='#C9A84C', linestyle='--', alpha=0.5, label='Nash Equilibrium (~0.693)')
plt.tight_layout()
plt.savefig('gan_training_loss.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("   💾 Saved → gan_training_loss.png")

# ── PER-ARCHETYPE COMPARISON CHART ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0D1117')

compare_feat = 'kda' if 'kda' in available else available[0]
compare_feat2 = 'cspm' if 'cspm' in available else available[1]

for ax, feat in zip(axes, [compare_feat, compare_feat2]):
    ax.set_facecolor('#0D1117')
    real_means  = [gan_df[gan_df['archetype']==a][feat].mean() for a in ARCHETYPE_NAMES]
    synth_means = [synthetic_df[synthetic_df['archetype']==a][feat].mean() for a in ARCHETYPE_NAMES]
    x = np.arange(len(ARCHETYPE_NAMES))
    bars1 = ax.bar(x - 0.2, real_means,  0.35, label='Real Data',   color='#457B9D', alpha=0.85)
    bars2 = ax.bar(x + 0.2, synth_means, 0.35, label='GAN Synthetic', color='#2A9D8F', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(' ','\n') for a in ARCHETYPE_NAMES], color='white', fontsize=9)
    ax.set_ylabel(f'Mean {feat}', color='#8B8B8B')
    ax.set_title(f'Real vs GAN — {feat.upper()}', color='white', fontsize=13)
    ax.tick_params(colors='#8B8B8B')
    for spine in ax.spines.values(): spine.set_color('#2D2D2D')
    ax.legend(facecolor='#1C1C1C', labelcolor='white')

plt.suptitle('🎮 GameSim AI — GAN Validation: Mean Feature Comparison',
             color='white', fontsize=14)
plt.tight_layout()
plt.savefig('gan_validation.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print("   💾 Saved → gan_validation.png")

# ── SAVE MODEL WEIGHTS + METADATA ─────────────────────────
generator.save('gamesim_gan_generator.h5')
discriminator.save('gamesim_gan_discriminator.h5')

metadata = {
    'archetype_names': ARCHETYPE_NAMES,
    'features': available,
    'noise_dim': NOISE_DIM,
    'n_classes': N_CLASSES,
    'n_features': N_FEATURES,
    'epochs_trained': EPOCHS,
    'final_d_loss': float(d_losses[-1]),
    'final_g_loss': float(g_losses[-1]),
    'synthetic_players_generated': len(synthetic_df)
}
with open('gan_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n💾 Saved → gamesim_gan_generator.h5")
print("💾 Saved → gan_metadata.json")

# ── FINAL SUMMARY ──────────────────────────────────────────
print("\n" + "="*60)
print("🎮 GAMESIM AI — CONDITIONAL GAN COMPLETE")
print("="*60)
print(f"   Architecture:    Conditional GAN (CGAN)")
print(f"   Generator:       64-dim noise → {N_FEATURES} player features")
print(f"   Discriminator:   {N_FEATURES} features → real/fake probability")
print(f"   Classes:         {N_CLASSES} player archetypes")
print(f"   Epochs trained:  {EPOCHS}")
print(f"   Final D-loss:    {d_losses[-1]:.4f} (target ~0.693)")
print(f"   Final G-loss:    {g_losses[-1]:.4f}")
print(f"   Synthetic players: {len(synthetic_df):,}")
print(f"\n📁 Output Files:")
print(f"   gan_synthetic_players.csv      → {len(synthetic_df)} AI players")
print(f"   gan_distribution_comparison.png → real vs synthetic")
print(f"   gan_training_loss.png          → training curve")
print(f"   gan_validation.png             → feature comparison")
print(f"   gamesim_gan_generator.h5       → saved generator")
print(f"\n💡 What this means:")
print(f"   The GAN learned real player behavior from {sample_size:,} matches.")
print(f"   It can now generate unlimited synthetic players per archetype.")
print(f"   This replaces human playtesting in game balance testing.")
print("="*60)
