import ctypes
import os
import numpy as np
import time

# 1. Load the C++ Engine
dll_path = os.path.abspath('poker_engine.dll')
poker_engine = ctypes.CDLL(dll_path)
poker_engine.simulate_equity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
poker_engine.simulate_equity.restype = ctypes.c_double

def human_rank_to_engine(human_rank):
    """Converts human rank (2-14) to engine's 0-based rank index (0-12)."""
    return human_rank - 2

def encode_hand(card1_rank, card2_rank, is_suited):
    """
    Encodes a poker hand into a 4-feature ML array.
    Features: [high_card, low_card, is_suited, is_pair]
    All ranks are engine 0-based (0-12).
    """
    high_card = max(card1_rank, card2_rank)
    low_card  = min(card1_rank, card2_rank)
    is_pair   = 1 if card1_rank == card2_rank else 0
    return [high_card, low_card, int(is_suited), is_pair]

print("Generating dataset of all 169 Hold'em hands...")
start_time = time.time()

X_data = []
y_data = []
rollouts = 50000

for i in range(2, 15):
    for j in range(2, i + 1):
        er_i = human_rank_to_engine(i)
        er_j = human_rank_to_engine(j)

        if i == j:
            # Pocket pair — same rank, clubs vs diamonds
            c1 = er_i
            c2 = er_i + 13
            X_data.append(encode_hand(er_i, er_j, False))
            y_data.append(poker_engine.simulate_equity(c1, c2, rollouts))

        else:
            # Suited — both clubs (suit 0)
            c1_suited = er_i
            c2_suited = er_j
            X_data.append(encode_hand(er_i, er_j, True))
            y_data.append(poker_engine.simulate_equity(c1_suited, c2_suited, rollouts))

            # Offsuit — clubs vs diamonds
            c1_offsuit = er_i
            c2_offsuit = er_j + 13
            X_data.append(encode_hand(er_i, er_j, False))
            y_data.append(poker_engine.simulate_equity(c1_offsuit, c2_offsuit, rollouts))

X_raw = np.array(X_data, dtype=np.float32)
y_raw = np.array(y_data, dtype=np.float32)

end_time = time.time()
print(f"Dataset generated in {end_time - start_time:.2f} seconds.")
print(f"X shape: {X_raw.shape}, y shape: {y_raw.shape}")
print(f"Feature columns: [high_card, low_card, is_suited, is_pair]")
print(f"Equity range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")

np.save('X_raw.npy', X_raw)
np.save('y_raw.npy', y_raw)
print("Successfully saved X_raw.npy and y_raw.npy!")