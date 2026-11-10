#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

// ---------------------------------------------------------
// 1. The Heuristic Evaluator
//
// Score layout (each rank field = 4 bits, range 0-12):
//   bits 24-20: hand category (0-9)
//   bits 19-16: primary rank   (r1)
//   bits 15-12: secondary rank (r2)
//   bits 11- 8: tertiary rank  (r3)
//   bits  7- 4: quaternary     (r4)
//   bits  3- 0: quinary        (r5)
//
// Packing by hand type:
//   High card   : (0, c1, c2, c3, c4, c5)  -- top 5 cards descending
//   One pair    : (1, pair, k1, k2, k3)    -- pair rank, top 3 kickers
//   Two pair    : (2, high_pair, low_pair, kicker)
//   Trips       : (3, trips, k1, k2)
//   Straight    : (4, high_card)
//   Flush       : (5, c1, c2, c3, c4, c5)  -- top 5 suited cards descending
//   Full house  : (6, trips_rank, pair_rank)
//   Quads       : (7, quad_rank, kicker)
//   Straight fl.: (8, high_card)
//   Royal flush : (9, 12)
// ---------------------------------------------------------
int evaluate7(int cards[7]) {
    int rank_counts[13] = {0};
    int suit_counts[4]  = {0};

    for (int i = 0; i < 7; i++) {
        rank_counts[cards[i] % 13]++;
        suit_counts[cards[i] / 13]++;
    }

    // Helper: pack up to 5 rank nibbles into a score
    auto pack = [](int cat, int r1=0, int r2=0, int r3=0, int r4=0, int r5=0) -> int {
        return (cat << 20) | (r1 << 16) | (r2 << 12) | (r3 << 8) | (r4 << 4) | r5;
    };

    // ── 1. Flush detection ───────────────────────────────────────────────────
    bool is_flush  = false;
    int  flush_suit = -1;
    for (int i = 0; i < 4; i++) {
        if (suit_counts[i] >= 5) { is_flush = true; flush_suit = i; break; }
    }

    // ── 2. Gather flush-suit ranks (needed for SF and flush scoring) ─────────
    int flush_ranks[13] = {0};
    if (is_flush) {
        for (int i = 0; i < 7; i++)
            if (cards[i] / 13 == flush_suit)
                flush_ranks[cards[i] % 13]++;
    }

    // ── 3. Straight-flush / Royal-flush detection ────────────────────────────
    if (is_flush) {
        // Standard straight flush
        for (int i = 12; i >= 4; i--) {
            if (flush_ranks[i] && flush_ranks[i-1] && flush_ranks[i-2] &&
                flush_ranks[i-3] && flush_ranks[i-4]) {
                if (i == 12) return pack(9, 12);          // Royal flush
                return pack(8, i);                         // Straight flush
            }
        }
        // Wheel straight flush (A-2-3-4-5)
        if (flush_ranks[12] && flush_ranks[0] && flush_ranks[1] &&
            flush_ranks[2]  && flush_ranks[3])
            return pack(8, 3);
    }

    // ── 4. Collect multiples (descending rank order) ─────────────────────────
    // quad_ranks[0] = highest quad rank, pair_ranks[0] = highest pair rank, etc.
    int quad_ranks[2] = {-1,-1}, quad_cnt = 0;
    int trip_ranks[2] = {-1,-1}, trip_cnt = 0;
    int pair_ranks[3] = {-1,-1,-1}, pair_cnt = 0;

    for (int i = 12; i >= 0; i--) {
        if      (rank_counts[i] == 4) quad_ranks[quad_cnt++] = i;
        else if (rank_counts[i] == 3) trip_ranks[trip_cnt++] = i;
        else if (rank_counts[i] == 2) pair_ranks[pair_cnt++] = i;
    }

    // ── 5. Straight detection (full 7-card) ──────────────────────────────────
    bool is_straight = false;
    int  straight_high = 0;
    for (int i = 12; i >= 4; i--) {
        if (rank_counts[i] && rank_counts[i-1] && rank_counts[i-2] &&
            rank_counts[i-3] && rank_counts[i-4]) {
            is_straight = true; straight_high = i; break;
        }
    }
    if (!is_straight && rank_counts[12] && rank_counts[0] && rank_counts[1] &&
        rank_counts[2] && rank_counts[3]) {
        is_straight = true; straight_high = 3; // wheel
    }

    // ── 6. Score in descending hand-rank order ────────────────────────────────

    // Four of a kind
    if (quad_cnt > 0) {
        int qr = quad_ranks[0];
        int kicker = 0;
        for (int i = 12; i >= 0; i--)
            if (i != qr && rank_counts[i] > 0) { kicker = i; break; }
        return pack(7, qr, kicker);
    }

    // Full house (trips+pair, or two trips — use best trips, treat second trips as pair)
    if (trip_cnt > 0 && (pair_cnt > 0 || trip_cnt >= 2)) {
        int tr = trip_ranks[0];
        int pr = (trip_cnt >= 2) ? trip_ranks[1] : pair_ranks[0];
        return pack(6, tr, pr);
    }

    // Flush — encode top 5 flush cards
    if (is_flush) {
        int fc[5]; int fi = 0;
        for (int i = 12; i >= 0 && fi < 5; i--)
            if (flush_ranks[i]) fc[fi++] = i;
        return pack(5, fc[0], fc[1], fc[2], fc[3], fc[4]);
    }

    // Straight
    if (is_straight) return pack(4, straight_high);

    // Three of a kind
    if (trip_cnt > 0) {
        int tr = trip_ranks[0];
        int k[2] = {0, 0}; int ki = 0;
        for (int i = 12; i >= 0 && ki < 2; i--)
            if (i != tr && rank_counts[i] > 0) k[ki++] = i;
        return pack(3, tr, k[0], k[1]);
    }

    // Two pair — best two pairs + best kicker
    if (pair_cnt >= 2) {
        int hp = pair_ranks[0], lp = pair_ranks[1];
        int kicker = 0;
        for (int i = 12; i >= 0; i--)
            if (i != hp && i != lp && rank_counts[i] > 0) { kicker = i; break; }
        return pack(2, hp, lp, kicker);
    }

    // One pair + 3 kickers
    if (pair_cnt == 1) {
        int pr = pair_ranks[0];
        int k[3] = {0, 0, 0}; int ki = 0;
        for (int i = 12; i >= 0 && ki < 3; i--)
            if (i != pr && rank_counts[i] > 0) k[ki++] = i;
        return pack(1, pr, k[0], k[1], k[2]);
    }

    // High card — top 5 cards
    int k[5] = {0,0,0,0,0}; int ki = 0;
    for (int i = 12; i >= 0 && ki < 5; i--)
        if (rank_counts[i] > 0) k[ki++] = i;
    return pack(0, k[0], k[1], k[2], k[3], k[4]);
}

// ---------------------------------------------------------
// 2. The Python C-Types Bridge
// ---------------------------------------------------------
extern "C" {
    __declspec(dllexport) double simulate_equity(int hero_card1, int hero_card2, int iterations) {
        mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
        int wins = 0, ties = 0;
        int base_deck[50];
        int index = 0;

        for (int i = 0; i < 52; i++) {
            if (i != hero_card1 && i != hero_card2)
                base_deck[index++] = i;
        }

        int current_deck[50];
        for (int iter = 0; iter < iterations; iter++) {
            copy(base_deck, base_deck + 50, current_deck);

            for (int i = 0; i < 7; i++) {
                uniform_int_distribution<int> dist(i, 49);
                int swap_idx = dist(rng);
                swap(current_deck[i], current_deck[swap_idx]);
            }

            // Hero and villain share community cards (indices 2-6)
            int hero_board[7]    = {hero_card1,       hero_card2,
                                    current_deck[2],  current_deck[3],
                                    current_deck[4],  current_deck[5], current_deck[6]};
            int villain_board[7] = {current_deck[0],  current_deck[1],
                                    current_deck[2],  current_deck[3],
                                    current_deck[4],  current_deck[5], current_deck[6]};

            int hero_score    = evaluate7(hero_board);
            int villain_score = evaluate7(villain_board);

            if      (hero_score > villain_score) wins++;
            else if (hero_score == villain_score) ties++;
        }

        // Ties count as half-wins
        return (wins + (ties / 2.0)) / iterations;
    }
}