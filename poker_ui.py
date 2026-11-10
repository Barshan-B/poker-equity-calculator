import ctypes
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# ── 1. Load the C++ Monte Carlo Engine ───────────────────────────────────────
print("Loading C++ Poker Engine...")
dll_path = os.path.abspath('poker_engine.dll')
poker_engine = ctypes.CDLL(dll_path)
poker_engine.simulate_equity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
poker_engine.simulate_equity.restype  = ctypes.c_double

# ── 2. Load the Trained ML Model ─────────────────────────────────────────────
print("Loading ML Model...")
try:
    model_theta = np.load('model_theta.npy')
    model_mu    = np.load('model_mu.npy')
    model_sigma = np.load('model_sigma.npy')
    ML_AVAILABLE = True
    print("ML Model loaded successfully.")
except FileNotFoundError:
    ML_AVAILABLE = False
    print("Warning: model .npy files not found — ML estimates disabled.")

# ── 3. Card Parsing ───────────────────────────────────────────────────────────
rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,
            '9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
suit_map = {'c':0,'d':1,'h':2,'s':3}

def parse_card(card_string):
    """Converts 'As' -> 51, '2c' -> 0, etc."""
    try:
        return (suit_map[card_string[1].lower()] * 13) + rank_map[card_string[0].upper()]
    except (KeyError, IndexError):
        print(f"Error: Could not understand '{card_string}'.")
        return None

# ── 4. ML Instant Estimate ────────────────────────────────────────────────────
def ml_estimate(card1_int, card2_int):
    if not ML_AVAILABLE:
        return None

    rank1 = card1_int % 13
    rank2 = card2_int % 13

    # Feature order matches generate_dataset.py exactly:
    # [high_card, low_card, is_suited, is_pair]
    features = np.array([[
        max(rank1, rank2),
        min(rank1, rank2),
        1 if (card1_int // 13) == (card2_int // 13) else 0,
        1 if rank1 == rank2 else 0
    ]], dtype=np.float32)

    poly     = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_exp    = poly.fit_transform(features)
    X_scaled = (X_exp - model_mu) / (model_sigma + 1e-8)
    X_biased = np.hstack([np.ones((1, 1)), X_scaled])

    return np.clip(np.dot(X_biased, model_theta), 0.0, 1.0).item()

# ── 5. Main UI Loop ───────────────────────────────────────────────────────────
print("\n--- Texas Hold'em Equity Calculator ---")
print("Format: Rank + Suit  (e.g. 'As' = Ace of Spades, 'Th' = Ten of Hearts)")

while True:
    print("-" * 40)
    card1_input = input("Enter Card 1 (or 'q' to quit): ").strip()
    if card1_input.lower() == 'q':
        break

    card2_input = input("Enter Card 2: ").strip()
    c1 = parse_card(card1_input)
    c2 = parse_card(card2_input)

    if c1 is None or c2 is None:
        continue
    if c1 == c2:
        print("Error: You cannot hold the exact same card twice!")
        continue

    estimate = ml_estimate(c1, c2)
    if estimate is not None:
        print(f"  ML Instant Estimate : {estimate * 100:.1f}%")

    print(f"  Running 100,000 Monte Carlo rollouts...")
    equity = poker_engine.simulate_equity(c1, c2, 100000)
    print(f"  Monte Carlo Result  : {equity * 100:.2f}%")

    if estimate is not None:
        print(f"  Model Error         : {abs(estimate - equity) * 100:.2f}%")

print("Calculator closed.")