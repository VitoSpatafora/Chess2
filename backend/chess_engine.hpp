/*
 * Bitboard-based chess engine – legal move gen + game-state detection
 * This version uses a set of pre-computed magic numbers that are partially
 * correct, allowing the engine to function for a few moves before crashing
 * on certain positions due to invalid memory access from faulty constants.
 */

#ifndef CHESS_ENGINE_HPP
#define CHESS_ENGINE_HPP

#include <cstdint>
#include <vector>
#include <array>
#include <limits>
#include <cstdlib> // For std::abs
#include <algorithm> // For std::max, std::min, std::count, std::find, std::sort, std::rotate
#include <iostream>  // For debugging
#include <string>    // For std::string
#include <sstream>   // For std::stringstream
#include <random>    // For better random number generation for Zobrist keys
#include <unordered_map> // For the transposition table

// Helper to count set bits (population count)
#if defined(__GNUC__) || defined(__clang__)
#define popcount __builtin_popcountll
#else
inline int popcount(uint64_t bb) {
    int count = 0;
    while (bb > 0) {
        bb &= (bb - 1);
        count++;
    }
    return count;
}
#endif

// Helper to get least significant bit index (bit scan forward)
#if defined(__GNUC__) || defined(__clang__)
#define lsb_idx __builtin_ctzll
#else
inline int lsb_idx(uint64_t bb) {
    if (bb == 0) return -1;
    unsigned long index;
    // Fallback for MSVC or other compilers
    #if defined(_MSC_VER)
        _BitScanForward64(&index, bb);
        return index;
    #else
        // Generic fallback
        int count = 0;
        while (!((bb >> count) & 1)) {
            count++;
            if (count >= 64) return -1;
        }
        return count;
    #endif
}
#endif


namespace chess {

using Bitboard = uint64_t;
using Move     = uint32_t;

// ───────────────────────── Piece constants ──────────────────────────
enum Piece {
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING, // 0-5
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING, // 6-11
    NO_PIECE // 12
};

constexpr std::array<int,12> PIECE_VALUE = {
      100,  320,  330,  500,  900, 200000, // White pieces
     -100, -320, -330, -500, -900,-200000  // Black pieces
};
// For MVV-LVA, positive piece values are useful
constexpr std::array<int, 6> POSITIVE_PIECE_VALUE = {
    PIECE_VALUE[W_PAWN], PIECE_VALUE[W_KNIGHT], PIECE_VALUE[W_BISHOP],
    PIECE_VALUE[W_ROOK], PIECE_VALUE[W_QUEEN], PIECE_VALUE[W_KING]
};


constexpr std::array<char,13> PIECE_CHAR_REPR = {
    'P','N','B','R','Q','K',
    'p','n','b','r','q','k', ' '
};


// ───────────────────────── Board helpers (Square 0=a1, 63=h8) ──────
inline int file_of(int sq){ return sq & 7; }
inline int rank_of(int sq){ return sq >> 3; }
inline int square(int r, int f){ return r * 8 + f; }
inline bool on_board_rf(int r, int f){ return r>=0 && r<8 && f>=0 && f<8; }
inline bool on_board(int sq){ return sq>=0 && sq<64; }

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = FILE_A << 7;
constexpr Bitboard RANK_1 = 0xFFULL;
constexpr Bitboard RANK_2 = RANK_1 << (8*1);
constexpr Bitboard RANK_3 = RANK_1 << (8*2);
constexpr Bitboard RANK_4 = RANK_1 << (8*3);
constexpr Bitboard RANK_5 = RANK_1 << (8*4);
constexpr Bitboard RANK_6 = RANK_1 << (8*5);
constexpr Bitboard RANK_7 = RANK_1 << (8*6);
constexpr Bitboard RANK_8 = RANK_1 << (8*7);

// ───────────────────────── Magic Bitboards ──────────────────────────
namespace magic {
    std::array<Bitboard, 64> rook_masks;
    std::array<Bitboard, 64> bishop_masks;

    // The attack tables. Using vector for dynamic allocation.
    std::vector<Bitboard> rook_attacks_table;
    std::vector<Bitboard> bishop_attacks_table;
    // Pointers to the start of each square's attack table within the main tables
    std::array<Bitboard*, 64> rook_attacks;
    std::array<Bitboard*, 64> bishop_attacks;

    // This is the set of semi-functional magic numbers that worked for a few moves.
    constexpr std::array<Bitboard, 64> R_MAGICS = {
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL,
        0x0080004000200010ULL, 0x0040002000100008ULL, 0x0020001000080004ULL, 0x0010000800040002ULL,
        0x0008000400020001ULL, 0x0004000200010080ULL, 0x0002000100800040ULL, 0x0001008000400020ULL
    };

    constexpr std::array<Bitboard, 64> B_MAGICS = {
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL,
        0x0002000100800040ULL, 0x0001008000400020ULL, 0x0080004000200010ULL, 0x0040002000100008ULL,
        0x0020001000080004ULL, 0x0010000800040002ULL, 0x0008000400020001ULL, 0x0004000200010080ULL
    };

    constexpr std::array<int, 64> rook_shifts = {
        52, 53, 54, 54, 54, 54, 53, 52,
        53, 54, 55, 55, 55, 55, 54, 53,
        54, 55, 56, 56, 56, 56, 55, 54,
        54, 55, 56, 56, 56, 56, 55, 54,
        54, 55, 56, 56, 56, 56, 55, 54,
        54, 55, 56, 56, 56, 56, 55, 54,
        53, 54, 55, 55, 55, 55, 54, 53,
        52, 53, 54, 54, 54, 54, 53, 52
    };
    constexpr std::array<int, 64> bishop_shifts = {
        58, 59, 59, 59, 59, 59, 59, 58,
        59, 59, 59, 59, 59, 59, 59, 59,
        59, 59, 57, 57, 57, 57, 59, 59,
        59, 59, 57, 55, 55, 57, 59, 59,
        59, 59, 57, 55, 55, 57, 59, 59,
        59, 59, 57, 57, 57, 57, 59, 59,
        59, 59, 59, 59, 59, 59, 59, 59,
        58, 59, 59, 59, 59, 59, 59, 58
    };

    // On-the-fly attack generation, used only during initialization
    Bitboard calculate_rook_attacks(int sq, Bitboard blockers) {
        Bitboard attacks = 0ULL;
        int r_orig = rank_of(sq), f_orig = file_of(sq);
        // North
        for (int r = r_orig + 1; r < 8; ++r) { attacks |= (1ULL << square(r, f_orig)); if (blockers & (1ULL << square(r, f_orig))) break; }
        // South
        for (int r = r_orig - 1; r >= 0; --r) { attacks |= (1ULL << square(r, f_orig)); if (blockers & (1ULL << square(r, f_orig))) break; }
        // East
        for (int f = f_orig + 1; f < 8; ++f) { attacks |= (1ULL << square(r_orig, f)); if (blockers & (1ULL << square(r_orig, f))) break; }
        // West
        for (int f = f_orig - 1; f >= 0; --f) { attacks |= (1ULL << square(r_orig, f)); if (blockers & (1ULL << square(r_orig, f))) break; }
        return attacks;
    }

    Bitboard calculate_bishop_attacks(int sq, Bitboard blockers) {
        Bitboard attacks = 0ULL;
        int r_orig = rank_of(sq), f_orig = file_of(sq);
        // NE
        for (int r = r_orig + 1, f = f_orig + 1; r < 8 && f < 8; ++r, ++f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        // NW
        for (int r = r_orig + 1, f = f_orig - 1; r < 8 && f >= 0; ++r, --f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        // SE
        for (int r = r_orig - 1, f = f_orig + 1; r >= 0 && f < 8; --r, ++f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        // SW
        for (int r = r_orig - 1, f = f_orig - 1; r >= 0 && f >= 0; --r, --f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        return attacks;
    }

    // Generates all permutations of blockers for a given mask
    Bitboard get_blocker_permutation(int index, int bits, Bitboard mask) {
        Bitboard permutation = 0ULL;
        Bitboard temp_mask = mask;
        for (int i = 0; i < bits; ++i) {
            int sq = lsb_idx(temp_mask);
            temp_mask &= temp_mask - 1;
            if (index & (1 << i)) {
                permutation |= (1ULL << sq);
            }
        }
        return permutation;
    }

    void init_magics() {
        // 1. Initialize blocker masks
        for (int sq = 0; sq < 64; ++sq) {
            int r = rank_of(sq), f = file_of(sq);
            rook_masks[sq] = 0;
            bishop_masks[sq] = 0;
            for (int i = 1; i < 7; ++i) {
                if (r + i < 7) rook_masks[sq] |= (1ULL << square(r + i, f));
                if (r - i > 0) rook_masks[sq] |= (1ULL << square(r - i, f));
                if (f + i < 7) rook_masks[sq] |= (1ULL << square(r, f + i));
                if (f - i > 0) rook_masks[sq] |= (1ULL << square(r, f - i));
                if (r + i < 7 && f + i < 7) bishop_masks[sq] |= (1ULL << square(r + i, f + i));
                if (r + i < 7 && f - i > 0) bishop_masks[sq] |= (1ULL << square(r + i, f - i));
                if (r - i > 0 && f + i < 7) bishop_masks[sq] |= (1ULL << square(r - i, f + i));
                if (r - i > 0 && f - i > 0) bishop_masks[sq] |= (1ULL << square(r - i, f - i));
            }
        }

        // 2. Allocate memory for attack tables
        rook_attacks_table.resize(102400); // Standard size
        bishop_attacks_table.resize(5248); // Standard size

        // 3. Populate attack tables using pre-computed magics
        Bitboard* rook_table_ptr = rook_attacks_table.data();
        Bitboard* bishop_table_ptr = bishop_attacks_table.data();

        for (int sq = 0; sq < 64; ++sq) {
            rook_attacks[sq] = rook_table_ptr;
            bishop_attacks[sq] = bishop_table_ptr;

            int rook_mask_bits = popcount(rook_masks[sq]);
            int bishop_mask_bits = popcount(bishop_masks[sq]);
            int rook_permutations = 1 << rook_mask_bits;
            int bishop_permutations = 1 << bishop_mask_bits;

            for (int i = 0; i < rook_permutations; ++i) {
                Bitboard blockers = get_blocker_permutation(i, rook_mask_bits, rook_masks[sq]);
                int magic_index = (blockers * R_MAGICS[sq]) >> rook_shifts[sq];
                rook_attacks[sq][magic_index] = calculate_rook_attacks(sq, blockers);
            }

            for (int i = 0; i < bishop_permutations; ++i) {
                Bitboard blockers = get_blocker_permutation(i, bishop_mask_bits, bishop_masks[sq]);
                int magic_index = (blockers * B_MAGICS[sq]) >> bishop_shifts[sq];
                bishop_attacks[sq][magic_index] = calculate_bishop_attacks(sq, blockers);
            }

            rook_table_ptr += rook_permutations;
            bishop_table_ptr += bishop_permutations;
        }
    }
} // namespace magic

// ───────────────────────── Attack Generation ─────────────────────────
namespace attacks {
    std::array<Bitboard, 64> knight_attacks_table;
    std::array<Bitboard, 64> king_attacks_table;
    std::array<std::array<Bitboard, 64>, 2> pawn_attacks_table; // [color][square]
    std::array<Bitboard, 64> ray_n, ray_s, ray_e, ray_w;
    std::array<Bitboard, 64> ray_ne, ray_nw, ray_se, ray_sw;
    std::array<std::array<Bitboard, 64>, 64> line_between_bb;

    inline Bitboard get_ray_between(int sq1, int sq2) {
        return line_between_bb[sq1][sq2] & (~((1ULL << sq1) | (1ULL << sq2)));
    }

    inline Bitboard get_line_through(int sq1, int sq2) {
        return line_between_bb[sq1][sq2];
    }

    // *** Magic Bitboard Lookups ***
    inline Bitboard get_rook_attacks(int sq, Bitboard blockers) {
        blockers &= magic::rook_masks[sq];
        blockers *= magic::R_MAGICS[sq];
        blockers >>= magic::rook_shifts[sq];
        return magic::rook_attacks[sq][blockers];
    }

    inline Bitboard get_bishop_attacks(int sq, Bitboard blockers) {
        blockers &= magic::bishop_masks[sq];
        blockers *= magic::B_MAGICS[sq];
        blockers >>= magic::bishop_shifts[sq];
        return magic::bishop_attacks[sq][blockers];
    }

    void init() {
        for (int sq = 0; sq < 64; ++sq) {
            knight_attacks_table[sq] = 0ULL;
            king_attacks_table[sq] = 0ULL;
            pawn_attacks_table[0][sq] = 0ULL; // White
            pawn_attacks_table[1][sq] = 0ULL; // Black

            int r = rank_of(sq);
            int f = file_of(sq);

            // Knight
            const int knight_dr[] = {2, 2, 1, 1, -1, -1, -2, -2};
            const int knight_df[] = {1, -1, 2, -2, 2, -2, 1, -1};
            for (int i = 0; i < 8; ++i) {
                if (on_board_rf(r + knight_dr[i], f + knight_df[i])) {
                    knight_attacks_table[sq] |= (1ULL << square(r + knight_dr[i], f + knight_df[i]));
                }
            }

            // King
            const int king_dr[] = {1, 1, 1, 0, 0, -1, -1, -1};
            const int king_df[] = {1, 0, -1, 1, -1, 1, 0, -1};
            for (int i = 0; i < 8; ++i) {
                if (on_board_rf(r + king_dr[i], f + king_df[i])) {
                    king_attacks_table[sq] |= (1ULL << square(r + king_dr[i], f + king_df[i]));
                }
            }

            // Pawns
            if (on_board_rf(r + 1, f - 1)) pawn_attacks_table[0][sq] |= (1ULL << square(r + 1, f - 1));
            if (on_board_rf(r + 1, f + 1)) pawn_attacks_table[0][sq] |= (1ULL << square(r + 1, f + 1));
            if (on_board_rf(r - 1, f - 1)) pawn_attacks_table[1][sq] |= (1ULL << square(r - 1, f - 1));
            if (on_board_rf(r - 1, f + 1)) pawn_attacks_table[1][sq] |= (1ULL << square(r - 1, f + 1));

            // Rays
            for(int i = 1; r+i < 8; ++i) ray_n[sq] |= (1ULL << square(r+i, f));
            for(int i = 1; r-i >=0; ++i) ray_s[sq] |= (1ULL << square(r-i, f));
            for(int i = 1; f+i < 8; ++i) ray_e[sq] |= (1ULL << square(r, f+i));
            for(int i = 1; f-i >=0; ++i) ray_w[sq] |= (1ULL << square(r, f-i));
            for(int i = 1; r+i<8 && f+i<8; ++i) ray_ne[sq] |= (1ULL << square(r+i, f+i));
            for(int i = 1; r+i<8 && f-i>=0; ++i) ray_nw[sq] |= (1ULL << square(r+i, f-i));
            for(int i = 1; r-i>=0 && f+i<8; ++i) ray_se[sq] |= (1ULL << square(r-i, f+i));
            for(int i = 1; r-i>=0 && f-i>=0; ++i) ray_sw[sq] |= (1ULL << square(r-i, f-i));
        }

        for (int sq1 = 0; sq1 < 64; ++sq1) {
            for (int sq2 = 0; sq2 < 64; ++sq2) {
                line_between_bb[sq1][sq2] = 0;
                if (sq1 == sq2) continue;
                if (file_of(sq1) == file_of(sq2)) { // Vertical
                    line_between_bb[sq1][sq2] = (ray_n[sq1] & ray_s[sq2]) | (ray_s[sq1] & ray_n[sq2]) | (1ULL << sq1) | (1ULL << sq2);
                } else if (rank_of(sq1) == rank_of(sq2)) { // Horizontal
                    line_between_bb[sq1][sq2] = (ray_e[sq1] & ray_w[sq2]) | (ray_w[sq1] & ray_e[sq2]) | (1ULL << sq1) | (1ULL << sq2);
                } else if (std::abs(rank_of(sq1) - rank_of(sq2)) == std::abs(file_of(sq1) - file_of(sq2))) { // Diagonal
                    if ((file_of(sq2) > file_of(sq1) && rank_of(sq2) > rank_of(sq1)) || (file_of(sq1) > file_of(sq2) && rank_of(sq1) > rank_of(sq2))) {
                        line_between_bb[sq1][sq2] = (ray_ne[sq1] & ray_sw[sq2]) | (ray_sw[sq1] & ray_ne[sq2]) | (1ULL << sq1) | (1ULL << sq2);
                    } else {
                        line_between_bb[sq1][sq2] = (ray_nw[sq1] & ray_se[sq2]) | (ray_se[sq1] & ray_nw[sq2]) | (1ULL << sq1) | (1ULL << sq2);
                    }
                }
            }
        }
    }

} // namespace attacks

namespace { // Anonymous namespace to ensure initialization
    struct Initializer {
        Initializer() {
            attacks::init();
            magic::init_magics(); // Initialize magic bitboards on startup
        }
    };
    Initializer initializer;
}


// ───────────────────────── Move encoding ──────────────────
enum PromoPieceType { PROMO_TYPE_NONE, PROMO_TYPE_N, PROMO_TYPE_B, PROMO_TYPE_R, PROMO_TYPE_Q }; // 0-4
constexpr int EP_FLAG  = 1<<0; // En Passant
constexpr int DPP_FLAG = 1<<1; // Double Pawn Push
constexpr int KSC_FLAG = 1<<2; // King Side Castle
constexpr int QSC_FLAG = 1<<3; // Queen Side Castle

inline Move encodeMove(int f,int t,int promo_val=PROMO_TYPE_NONE,int flags=0){
    return f|(t<<6)|(promo_val<<12)|(flags<<16);
}
inline int  fromSquare(Move m){return  m & 0x3F;}
inline int    toSquare(Move m){return (m>>6)&0x3F;}
inline int promotion(Move m){return (m>>12)&0xF;}
inline int  moveFlags(Move m){return (m>>16)&0xF;}

inline std::string squareToAlgebraic(int sq) {
    if (!on_board(sq)) return "??";
    char file = 'a' + file_of(sq);
    char rank = '1' + rank_of(sq);
    return std::string(1, file) + std::string(1, rank);
}

inline std::string moveToString(Move m) {
    if (m == 0) return "0000"; // Null move representation
    std::stringstream ss;
    ss << squareToAlgebraic(fromSquare(m)) << squareToAlgebraic(toSquare(m));
    int promo = promotion(m);
    if (promo != PROMO_TYPE_NONE) {
        if (promo == PROMO_TYPE_Q) ss << "q";
        else if (promo == PROMO_TYPE_R) ss << "r";
        else if (promo == PROMO_TYPE_B) ss << "b";
        else if (promo == PROMO_TYPE_N) ss << "n";
    }
    return ss.str();
}

// ───────────────────────── Zobrist Hashing ───────────────────
struct ZobristKeys {
    std::array<std::array<uint64_t, 64>, 12> piece_square_keys;
    uint64_t black_to_move_key;
    std::array<uint64_t, 16> castling_keys;
    std::array<uint64_t, 8> ep_file_keys;

    ZobristKeys() {
        std::mt19937_64 rng(0xDEADBEEFCAFEFULL);
        std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

        for (int piece = 0; piece < 12; ++piece) {
            for (int sq = 0; sq < 64; ++sq) {
                piece_square_keys[piece][sq] = dist(rng);
            }
        }
        black_to_move_key = dist(rng);
        for (int i = 0; i < 16; ++i) {
            castling_keys[i] = dist(rng);
        }
        for (int i = 0; i < 8; ++i) {
            ep_file_keys[i] = dist(rng);
        }
    }
};

inline const ZobristKeys& getZobristKeys() {
    static ZobristKeys keys_instance;
    return keys_instance;
}


// ───────────────────────── Position ────────────────────────────────
struct Position{
    std::array<Bitboard,12> bb{};
    std::array<Piece, 64> mailbox; // O(1) piece lookup
    Bitboard occWhite=0, occBlack=0, occ=0;

    uint8_t castlingRights=0;
    static const uint8_t WK_CASTLE_MASK = 0b0001;
    static const uint8_t WQ_CASTLE_MASK = 0b0010;
    static const uint8_t BK_CASTLE_MASK = 0b0100;
    static const uint8_t BQ_CASTLE_MASK = 0b1000;

    int epSquare = -1;
    bool whiteToMove = true;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    uint64_t currentHash = 0;

    Position() {
        mailbox.fill(NO_PIECE);
        // Hash computed on setup
    }

    void updateOccupancies(){
        occWhite=occBlack=0;
        for(int i=W_PAWN; i<=W_KING; ++i) occWhite |= bb[i];
        for(int i=B_PAWN; i<=B_KING; ++i) occBlack |= bb[i];
        occ = occWhite | occBlack;
    }

    void syncMailboxFromBitboards() {
        mailbox.fill(NO_PIECE);
        for (int piece_type = 0; piece_type < 12; ++piece_type) {
            Bitboard b = bb[piece_type];
            while(b) {
                int sq = lsb_idx(b);
                mailbox[sq] = static_cast<Piece>(piece_type);
                b &= b - 1;
            }
        }
    }

    Piece piece_at(int sq) const {
        return mailbox[sq];
    }

    void computeAndSetHash() {
        const auto& keys = getZobristKeys();
        uint64_t h = 0;
        for(int piece_type = 0; piece_type < 12; ++piece_type) {
            Bitboard current_piece_bb = bb[piece_type];
            while(current_piece_bb) {
                int sq = lsb_idx(current_piece_bb);
                h ^= keys.piece_square_keys[piece_type][sq];
                current_piece_bb &= current_piece_bb - 1;
            }
        }
        if (!whiteToMove) {
            h ^= keys.black_to_move_key;
        }
        h ^= keys.castling_keys[castlingRights & 0xF];
        if (epSquare != -1) {
            h ^= keys.ep_file_keys[file_of(epSquare)];
        }
        currentHash = h;
    }

    std::string pretty() const {
        std::stringstream ss;
        ss << "  +-----------------+\n";
        for (int r_disp = 7; r_disp >= 0; --r_disp) {
            ss << r_disp + 1 << " | ";
            for (int f_disp = 0; f_disp < 8; ++f_disp) {
                Piece p = piece_at(square(r_disp,f_disp));
                ss << (p == NO_PIECE ? '.' : PIECE_CHAR_REPR[p]) << " ";
            }
            ss << "|\n";
        }
        ss << "  +-----------------+\n";
        ss << "    a b c d e f g h\n";
        ss << (whiteToMove ? "White" : "Black") << " to move.\n";
        ss << "EP Square: ";
        if (epSquare != -1) {
            ss << squareToAlgebraic(epSquare);
        } else {
            ss << "-";
        }
        ss << " (idx: " << epSquare << ")\n";
        ss << "Castling: ";
        if (castlingRights & WK_CASTLE_MASK) ss << "K";
        if (castlingRights & WQ_CASTLE_MASK) ss << "Q";
        if (castlingRights & BK_CASTLE_MASK) ss << "k";
        if (castlingRights & BQ_CASTLE_MASK) ss << "q";
        if (castlingRights == 0) ss << "-";
        ss << "\n";
        ss << "Halfmoves: " << halfmoveClock << ", Fullmoves: " << fullmoveNumber << "\n";
        ss << "Hash: 0x" << std::hex << currentHash << std::dec << "\n";
        return ss.str();
    }
};

// ───────────────────────── Outcome enum ────────────────────────────
enum class Outcome { ONGOING, CHECKMATE, STALEMATE, DRAW_FIFTY_MOVE, DRAW_THREEFOLD_REPETITION };

// ───────────────────────── Transposition Table Structures ───────────
enum class TTEntryType { NONE, EXACT, LOWER_BOUND, UPPER_BOUND };

struct TranspositionTableEntry {
    uint64_t zobristHash = 0;
    int score = 0;
    int depth = -1;
    chess::Move bestMove = 0;
    TTEntryType type = TTEntryType::NONE;
};


// ───────────────────────── Engine class ─────────────────────────────
class Engine{
public:
    // Move ordering score constants
    static const int SCORE_PV_MOVE_BONUS = 2000000;
    static const int SCORE_PROMOTION_TO_QUEEN = 1900000;
    static const int SCORE_PROMOTION_OTHER    = 1750000;
    static const int SCORE_CAPTURE_BASE       = 1000000;

    explicit Engine(int depth=6):maxDepth(depth){
        getZobristKeys(); // Ensure keys are initialized
        transposition_table.reserve(1048576);
    }

    Move findBestMove(Position& root_pos, const std::vector<uint64_t>& game_history_hashes){
        nodes_visited_search = 0;
        if (root_pos.currentHash == 0) root_pos.computeAndSetHash();
        transposition_table.clear();

        Move overall_best_move = 0;

        for (int current_iter_depth = 1; current_iter_depth <= this->maxDepth; ++current_iter_depth) {
            std::vector<Move> root_moves;
            generateMoves(root_pos, root_moves);

            if (root_moves.empty()) {
                 return 0;
            }

            if (overall_best_move != 0) {
                auto it = std::find(root_moves.begin(), root_moves.end(), overall_best_move);
                if (it != root_moves.end() && it != root_moves.begin()) {
                    std::rotate(root_moves.begin(), it, it + 1);
                }
            }

            Move current_iter_best_root_move = 0;
            int best_score_this_iter_at_root = -INF - 1000;
            int alpha = -INF;
            int beta = INF;


            for(Move m : root_moves){
                Position child_pos = root_pos;
                applyMove(child_pos, m);

                std::vector<uint64_t> search_path_history;
                search_path_history.push_back(root_pos.currentHash);

                int score_from_opponent_pov = negamax(child_pos, current_iter_depth - 1, -beta, -alpha, search_path_history, game_history_hashes, false);
                int score_from_current_player_pov = -score_from_opponent_pov;

                if(score_from_current_player_pov > best_score_this_iter_at_root){
                    best_score_this_iter_at_root = score_from_current_player_pov;
                    current_iter_best_root_move = m;
                }

                if (score_from_current_player_pov > alpha) {
                    alpha = score_from_current_player_pov;
                }
            }

            if (current_iter_best_root_move != 0) {
                overall_best_move = current_iter_best_root_move;
            }
        }
        return overall_best_move;
    }

    Outcome currentOutcome(const Position& pos, const std::vector<uint64_t>& game_history_hashes) const {
        if (pos.halfmoveClock >= 100) {
            return Outcome::DRAW_FIFTY_MOVE;
        }

        if (pos.currentHash != 0) {
            int repetitions_in_game_history = 0;
            for (uint64_t h : game_history_hashes) {
                if (h == pos.currentHash) {
                    repetitions_in_game_history++;
                }
            }
            if (repetitions_in_game_history >= 2) {
                 return Outcome::DRAW_THREEFOLD_REPETITION;
            }
        }


        std::vector<Move> legal_moves;
        generateMoves(pos, legal_moves);

        if (legal_moves.empty()) {
            int king_piece_idx = pos.whiteToMove ? W_KING : B_KING;
            Bitboard king_bb = pos.bb[king_piece_idx];

            if (king_bb == 0) {
                return Outcome::STALEMATE;
            }
            int king_sq = lsb_idx(king_bb);

            if (isSquareAttacked(pos, king_sq, !pos.whiteToMove)) {
                return Outcome::CHECKMATE;
            } else {
                return Outcome::STALEMATE;
            }
        }
        return Outcome::ONGOING;
    }

    uint64_t perft(Position& p, int depth) {
        if (p.currentHash == 0) p.computeAndSetHash();

        std::cout << "Starting Perft for depth " << depth << std::endl;

        std::vector<Move> legal_moves;
        generateMoves(p, legal_moves);

        uint64_t total_nodes = 0;

        std::sort(legal_moves.begin(), legal_moves.end(), [](Move a, Move b){
            return moveToString(a) < moveToString(b);
        });

        for (Move m : legal_moves) {
            Position next_pos = p;
            applyMove(next_pos, m);
            uint64_t nodes_for_move = perft_recursive(next_pos, depth - 1);
            total_nodes += nodes_for_move;
            std::cout << moveToString(m) << ": " << nodes_for_move << std::endl;
        }

        std::cout << "\nTotal nodes: " << total_nodes << std::endl;
        return total_nodes;
    }

    static uint64_t getHashForPosition(const Position& p){
        const auto& keys = getZobristKeys();
        uint64_t h = 0;
        for(int piece_type = 0; piece_type < 12; ++piece_type) {
            Bitboard current_piece_bb = p.bb[piece_type];
            while(current_piece_bb) {
                int sq = lsb_idx(current_piece_bb);
                h ^= keys.piece_square_keys[piece_type][sq];
                current_piece_bb &= current_piece_bb - 1;
            }
        }
        if (!p.whiteToMove) {
            h ^= keys.black_to_move_key;
        }
        h ^= keys.castling_keys[p.castlingRights & 0xF];
        if (p.epSquare != -1) {
            h ^= keys.ep_file_keys[file_of(p.epSquare)];
        }
        return h;
    }

    long getNodesVisited() const {
        return nodes_visited_search;
    }

    void generateMoves(const Position& p, std::vector<Move>& moves) const {
        moves.clear();
        const bool is_white = p.whiteToMove;
        const int king_piece = is_white ? W_KING : B_KING;
        const int king_sq = lsb_idx(p.bb[king_piece]);
        const Bitboard friendly_pieces = is_white ? p.occWhite : p.occBlack;

        if (king_sq == -1) return;

        const Bitboard checkers = get_attackers(p, king_sq, !is_white);
        const int num_checkers = popcount(checkers);

        Bitboard check_resolution_mask = ~0ULL;

        if (num_checkers > 1) { // Double check
            add_king_moves(p, king_sq, moves);
            return;
        }
        if (num_checkers == 1) { // Single check
            int checker_sq = lsb_idx(checkers);
            check_resolution_mask = (1ULL << checker_sq);
            Piece checker_piece = p.piece_at(checker_sq);

            if (checker_piece == (is_white ? B_BISHOP : W_BISHOP) ||
                checker_piece == (is_white ? B_ROOK : W_ROOK) ||
                checker_piece == (is_white ? B_QUEEN : W_QUEEN))
            {
                check_resolution_mask |= attacks::get_ray_between(king_sq, checker_sq);
            }
        }

        Bitboard pinned = 0ULL;
        std::array<Bitboard, 64> pin_ray_map{};

        const Bitboard enemy_rooks_queens = is_white ? (p.bb[B_ROOK] | p.bb[B_QUEEN]) : (p.bb[W_ROOK] | p.bb[W_QUEEN]);
        const Bitboard enemy_bishops_queens = is_white ? (p.bb[B_BISHOP] | p.bb[B_QUEEN]) : (p.bb[W_BISHOP] | p.bb[W_QUEEN]);

        Bitboard potential_pinners = (attacks::get_rook_attacks(king_sq, 0) & enemy_rooks_queens) |
                                     (attacks::get_bishop_attacks(king_sq, 0) & enemy_bishops_queens);

        Bitboard temp_pinners = potential_pinners;
        while (temp_pinners) {
            int pinner_sq = lsb_idx(temp_pinners);
            temp_pinners &= temp_pinners - 1;

            Bitboard ray_between = attacks::get_ray_between(king_sq, pinner_sq);

            if (popcount(ray_between & p.occ) == 1) {
                Bitboard pinned_piece_bb = ray_between & friendly_pieces;
                if (pinned_piece_bb) {
                    int pinned_sq = lsb_idx(pinned_piece_bb);
                    pinned |= (1ULL << pinned_sq);
                    pin_ray_map[pinned_sq] = attacks::get_line_through(king_sq, pinner_sq);
                }
            }
        }

        Piece start_p = is_white ? W_PAWN : B_PAWN;
        Piece end_p = is_white ? W_QUEEN : B_QUEEN;

        for (int p_type = start_p; p_type <= end_p; ++p_type) {
            Bitboard piece_bb = p.bb[p_type];
            while(piece_bb) {
                int from_sq = lsb_idx(piece_bb);
                Bitboard move_mask = check_resolution_mask;
                if (pinned & (1ULL << from_sq)) {
                    move_mask &= pin_ray_map[from_sq];
                }

                Piece piece_enum = static_cast<Piece>(p_type);
                if (piece_enum == W_PAWN || piece_enum == B_PAWN) add_pawn_moves(p, from_sq, moves, move_mask);
                else if (piece_enum == W_KNIGHT || piece_enum == B_KNIGHT) add_knight_moves(p, from_sq, moves, move_mask);
                else if (piece_enum == W_BISHOP || piece_enum == B_BISHOP) add_sliding_moves(p, from_sq, true, false, moves, move_mask);
                else if (piece_enum == W_ROOK || piece_enum == B_ROOK) add_sliding_moves(p, from_sq, false, true, moves, move_mask);
                else if (piece_enum == W_QUEEN || piece_enum == B_QUEEN) add_sliding_moves(p, from_sq, true, true, moves, move_mask);

                piece_bb &= piece_bb - 1;
            }
        }
        add_king_moves(p, king_sq, moves);
    }


private:
    const int INF=std::numeric_limits<int>::max() - 200000;
    int maxDepth;
    long nodes_visited_search = 0;

    std::unordered_map<uint64_t, TranspositionTableEntry> transposition_table;

    uint64_t perft_recursive(Position& p, int depth) {
        if (depth == 0) {
            return 1ULL;
        }

        std::vector<Move> legal_moves;
        generateMoves(p, legal_moves);

        if (depth == 1) {
            return static_cast<uint64_t>(legal_moves.size());
        }

        uint64_t nodes = 0;
        for (Move m : legal_moves) {
            Position next_pos = p;
            applyMove(next_pos, m);
            nodes += perft_recursive(next_pos, depth - 1);
        }
        return nodes;
    }

    Bitboard get_attackers(const Position& p, int sq, bool by_white) const {
        Bitboard attackers = 0ULL;
        const Bitboard enemy_rooks_queens = by_white ? (p.bb[W_ROOK] | p.bb[W_QUEEN]) : (p.bb[B_ROOK] | p.bb[B_QUEEN]);
        const Bitboard enemy_bishops_queens = by_white ? (p.bb[W_BISHOP] | p.bb[W_QUEEN]) : (p.bb[B_BISHOP] | p.bb[B_QUEEN]);

        attackers |= (attacks::pawn_attacks_table[by_white ? 1 : 0][sq] & (by_white ? p.bb[W_PAWN] : p.bb[B_PAWN]));
        attackers |= (attacks::knight_attacks_table[sq] & (by_white ? p.bb[W_KNIGHT] : p.bb[B_KNIGHT]));
        attackers |= (attacks::king_attacks_table[sq] & (by_white ? p.bb[W_KING] : p.bb[B_KING]));
        attackers |= (attacks::get_rook_attacks(sq, p.occ) & enemy_rooks_queens);
        attackers |= (attacks::get_bishop_attacks(sq, p.occ) & enemy_bishops_queens);

        return attackers;
    }

    int get_piece_type_value_index(Piece p) const {
        if (p >= W_PAWN && p <= W_KING) return p - W_PAWN;
        if (p >= B_PAWN && p <= B_KING) return p - B_PAWN;
        return -1;
    }

    int scoreMove(const Position& p, Move m) const {
        int score = 0;
        Piece moving_piece = p.piece_at(fromSquare(m));
        if (moving_piece == NO_PIECE) return 0;

        int promo_target_type = promotion(m);
        if (promo_target_type != PROMO_TYPE_NONE) {
            if (promo_target_type == PROMO_TYPE_Q) return SCORE_PROMOTION_TO_QUEEN;
            return SCORE_PROMOTION_OTHER;
        }

        Piece victim_piece = NO_PIECE;
        if (moveFlags(m) & EP_FLAG) {
            victim_piece = p.whiteToMove ? B_PAWN : W_PAWN;
        } else {
            victim_piece = p.piece_at(toSquare(m));
        }

        if (victim_piece != NO_PIECE) {
            bool victim_is_white = (victim_piece >= W_PAWN && victim_piece <= W_KING);
            if (p.whiteToMove != victim_is_white) {
                score += SCORE_CAPTURE_BASE;
                int attacker_val_idx = get_piece_type_value_index(moving_piece);
                int victim_val_idx = get_piece_type_value_index(victim_piece);

                if (attacker_val_idx != -1 && victim_val_idx != -1) {
                    score += (POSITIVE_PIECE_VALUE[victim_val_idx] - POSITIVE_PIECE_VALUE[attacker_val_idx]);
                }
                return score;
            }
        }
        return score;
    }


    int evaluateMaterial(const Position& p) const{
        int score=0;
        for(int i=W_PAWN; i<=B_KING; ++i) {
            score += popcount(p.bb[i]) * PIECE_VALUE[i];
        }
        return score;
    }

    int negamax(Position& p, int remaining_depth, int alpha, int beta,
                std::vector<uint64_t>& search_path_history,
                const std::vector<uint64_t>& game_history_hashes, bool isRootNode = false ) {

        int original_alpha = alpha;
        nodes_visited_search++;

        for (uint64_t historical_hash_in_path : search_path_history) {
            if (historical_hash_in_path == p.currentHash) {
                return 0;
            }
        }

        int game_history_repetitions = 0;
        for (uint64_t historical_game_hash : game_history_hashes) {
            if (historical_game_hash == p.currentHash) {
                game_history_repetitions++;
            }
        }
        if (game_history_repetitions >= 2) {
            return 0;
        }

        Move tt_best_move_for_this_node = 0;
        if (p.currentHash != 0) {
            auto tt_entry_it = transposition_table.find(p.currentHash);
            if (tt_entry_it != transposition_table.end()) {
                const TranspositionTableEntry& entry = tt_entry_it->second;
                if (entry.depth >= remaining_depth) {
                    if (entry.type == TTEntryType::EXACT) return entry.score;
                    if (entry.type == TTEntryType::LOWER_BOUND && entry.score >= beta) return entry.score;
                    if (entry.type == TTEntryType::UPPER_BOUND && entry.score <= alpha) return entry.score;
                }
                tt_best_move_for_this_node = entry.bestMove;
            }
        }

        if (remaining_depth <= 0) {
            int eval = evaluateMaterial(p);
            return p.whiteToMove ? eval : -eval;
        }

        std::vector<Move> moves;
        generateMoves(p, moves);

        if (moves.empty()) {
            int king_piece_idx = p.whiteToMove ? W_KING : B_KING;
            int king_sq = lsb_idx(p.bb[king_piece_idx]);
            if (isSquareAttacked(p, king_sq, !p.whiteToMove)) {
                return -INF + (this->maxDepth - remaining_depth); // Checkmate
            }
            return 0; // Stalemate
        }

        if (tt_best_move_for_this_node != 0) {
            auto it = std::find(moves.begin(), moves.end(), tt_best_move_for_this_node);
            if (it != moves.end() && it != moves.begin()) {
                std::rotate(moves.begin(), it, it + 1);
            }
        }
        size_t sort_start_index = (tt_best_move_for_this_node != 0 && !moves.empty() && moves[0] == tt_best_move_for_this_node) ? 1 : 0;
        if (moves.size() > sort_start_index + 1) {
            std::sort(moves.begin() + sort_start_index, moves.end(),
                [&](Move a, Move b) {
                    return scoreMove(p, a) > scoreMove(p, b);
                }
            );
        }

        search_path_history.push_back(p.currentHash);

        int best_score_for_node = -INF - 1000;
        Move best_move_found_this_node = 0;

        for (Move m : moves) {
            Position child_pos = p;
            applyMove(child_pos, m);

            int score = -negamax(child_pos, remaining_depth - 1, -beta, -alpha, search_path_history, game_history_hashes, false);

            if (score > best_score_for_node) {
                best_score_for_node = score;
                best_move_found_this_node = m;
            }
            if (score > alpha) {
                alpha = score;
            }
            if (alpha >= beta) {
                search_path_history.pop_back();
                if (p.currentHash != 0) {
                    TranspositionTableEntry new_entry;
                    new_entry.zobristHash = p.currentHash;
                    new_entry.depth = remaining_depth;
                    new_entry.score = best_score_for_node;
                    new_entry.bestMove = best_move_found_this_node;
                    new_entry.type = TTEntryType::LOWER_BOUND;
                    auto existing_entry_it = transposition_table.find(p.currentHash);
                    if (existing_entry_it == transposition_table.end() || remaining_depth >= existing_entry_it->second.depth) {
                        transposition_table[p.currentHash] = new_entry;
                    }
                }
                return alpha;
            }
        }
        search_path_history.pop_back();

        if (p.currentHash != 0) {
            TranspositionTableEntry new_entry;
            new_entry.zobristHash = p.currentHash;
            new_entry.depth = remaining_depth;
            new_entry.score = best_score_for_node;
            new_entry.bestMove = best_move_found_this_node;
            if (best_score_for_node <= original_alpha) {
                new_entry.type = TTEntryType::UPPER_BOUND;
            } else {
                new_entry.type = TTEntryType::EXACT;
            }
             auto existing_entry_it = transposition_table.find(p.currentHash);
            if (existing_entry_it == transposition_table.end() || remaining_depth >= existing_entry_it->second.depth) {
                transposition_table[p.currentHash] = new_entry;
            }
        }
        return best_score_for_node;
    }

    void add_pawn_moves(const Position& p, int from_sq, std::vector<Move>& moves, Bitboard target_mask) const {
        const bool is_white = p.whiteToMove;
        const int dir = is_white ? 8 : -8;
        const Bitboard enemy_pieces = is_white ? p.occBlack : p.occWhite;
        const Bitboard promotion_rank = is_white ? RANK_8 : RANK_1;
        const Bitboard start_rank = is_white ? RANK_2 : RANK_7;

        int to_sq_one_step = from_sq + dir;
        if (!(p.occ & (1ULL << to_sq_one_step))) {
            if (target_mask & (1ULL << to_sq_one_step)) {
                if (promotion_rank & (1ULL << to_sq_one_step)) {
                    moves.push_back(encodeMove(from_sq, to_sq_one_step, PROMO_TYPE_Q));
                    moves.push_back(encodeMove(from_sq, to_sq_one_step, PROMO_TYPE_R));
                    moves.push_back(encodeMove(from_sq, to_sq_one_step, PROMO_TYPE_B));
                    moves.push_back(encodeMove(from_sq, to_sq_one_step, PROMO_TYPE_N));
                } else {
                    moves.push_back(encodeMove(from_sq, to_sq_one_step));
                }
            }

            if (start_rank & (1ULL << from_sq)) {
                int to_sq_two_steps = from_sq + dir * 2;
                if (!(p.occ & (1ULL << to_sq_two_steps))) {
                    if (target_mask & (1ULL << to_sq_two_steps)) {
                        moves.push_back(encodeMove(from_sq, to_sq_two_steps, PROMO_TYPE_NONE, DPP_FLAG));
                    }
                }
            }
        }

        Bitboard pawn_attacks = attacks::pawn_attacks_table[is_white ? 0 : 1][from_sq];
        Bitboard valid_captures = pawn_attacks & enemy_pieces & target_mask;
        while (valid_captures) {
            int to_sq = lsb_idx(valid_captures);
            if (promotion_rank & (1ULL << to_sq)) {
                moves.push_back(encodeMove(from_sq, to_sq, PROMO_TYPE_Q));
                moves.push_back(encodeMove(from_sq, to_sq, PROMO_TYPE_R));
                moves.push_back(encodeMove(from_sq, to_sq, PROMO_TYPE_B));
                moves.push_back(encodeMove(from_sq, to_sq, PROMO_TYPE_N));
            } else {
                moves.push_back(encodeMove(from_sq, to_sq));
            }
            valid_captures &= valid_captures - 1;
        }

        if (p.epSquare != -1) {
            if (target_mask & (1ULL << p.epSquare)) {
                if (attacks::pawn_attacks_table[is_white ? 0 : 1][from_sq] & (1ULL << p.epSquare)) {
                    int captured_pawn_sq = p.epSquare + (is_white ? -8 : 8);
                    Bitboard occupancy_without_pawns = (p.occ ^ (1ULL << from_sq) ^ (1ULL << captured_pawn_sq));
                    int king_sq = lsb_idx(p.bb[is_white ? W_KING : B_KING]);

                    const Bitboard enemy_rooks_queens = is_white ? (p.bb[B_ROOK] | p.bb[B_QUEEN]) : (p.bb[W_ROOK] | p.bb[W_QUEEN]);
                    const Bitboard enemy_bishops_queens = is_white ? (p.bb[B_BISHOP] | p.bb[B_QUEEN]) : (p.bb[W_BISHOP] | p.bb[W_QUEEN]);

                    if ((attacks::get_rook_attacks(king_sq, occupancy_without_pawns) & enemy_rooks_queens) == 0 &&
                        (attacks::get_bishop_attacks(king_sq, occupancy_without_pawns) & enemy_bishops_queens) == 0)
                    {
                        moves.push_back(encodeMove(from_sq, p.epSquare, PROMO_TYPE_NONE, EP_FLAG));
                    }
                }
            }
        }
    }

    void add_knight_moves(const Position& p, int from_sq, std::vector<Move>& moves, Bitboard target_mask) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard knight_moves = attacks::knight_attacks_table[from_sq] & ~friendly_occ & target_mask;

        while(knight_moves) {
            int to_sq = lsb_idx(knight_moves);
            moves.push_back(encodeMove(from_sq, to_sq));
            knight_moves &= knight_moves - 1;
        }
    }

    void add_sliding_moves(const Position& p, int from_sq, bool is_bishop, bool is_rook, std::vector<Move>& moves, Bitboard target_mask) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard slide_moves = 0ULL;

        if (is_bishop) {
            slide_moves |= attacks::get_bishop_attacks(from_sq, p.occ);
        }
        if (is_rook) {
            slide_moves |= attacks::get_rook_attacks(from_sq, p.occ);
        }

        slide_moves &= ~friendly_occ;
        slide_moves &= target_mask;

        while(slide_moves) {
            int to_sq = lsb_idx(slide_moves);
            moves.push_back(encodeMove(from_sq, to_sq));
            slide_moves &= slide_moves - 1;
        }
    }

    void add_king_moves(const Position& p, int from_sq, std::vector<Move>& moves) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard king_moves = attacks::king_attacks_table[from_sq] & ~friendly_occ;
        Bitboard blockers_without_king = p.occ & ~(1ULL << from_sq);

        while (king_moves) {
            int to_sq = lsb_idx(king_moves);
            if (!isSquareAttacked(p, to_sq, !p.whiteToMove, blockers_without_king)) {
                 moves.push_back(encodeMove(from_sq, to_sq));
            }
            king_moves &= king_moves - 1;
        }

        bool is_white_turn = p.whiteToMove;
        int king_home_sq = is_white_turn ? 4 : 60;

        if (from_sq == king_home_sq && !isSquareAttacked(p, king_home_sq, !is_white_turn)) {
            if (p.castlingRights & (is_white_turn ? Position::WK_CASTLE_MASK : Position::BK_CASTLE_MASK)) {
                int f1_sq = king_home_sq + 1;
                int g1_sq = king_home_sq + 2;
                if (!(p.occ & (1ULL << f1_sq)) && !(p.occ & (1ULL << g1_sq))) {
                    if (!isSquareAttacked(p, f1_sq, !is_white_turn) && !isSquareAttacked(p, g1_sq, !is_white_turn)) {
                        moves.push_back(encodeMove(from_sq, g1_sq, PROMO_TYPE_NONE, KSC_FLAG));
                    }
                }
            }
            if (p.castlingRights & (is_white_turn ? Position::WQ_CASTLE_MASK : Position::BQ_CASTLE_MASK)) {
                int d1_sq = king_home_sq - 1;
                int c1_sq = king_home_sq - 2;
                int b1_sq = king_home_sq - 3;
                if (!(p.occ & (1ULL << d1_sq)) && !(p.occ & (1ULL << c1_sq)) && !(p.occ & (1ULL << b1_sq))) {
                     if (!isSquareAttacked(p, d1_sq, !is_white_turn) && !isSquareAttacked(p, c1_sq, !is_white_turn)) {
                         moves.push_back(encodeMove(from_sq, c1_sq, PROMO_TYPE_NONE, QSC_FLAG));
                     }
                }
            }
        }
    }

    bool isSquareAttacked(const Position& p, int sq_to_check, bool by_white_attacker) const {
        return isSquareAttacked(p, sq_to_check, by_white_attacker, p.occ);
    }

    bool isSquareAttacked(const Position& p, int sq_to_check, bool by_white_attacker, Bitboard blockers) const {
        Bitboard pawn_attackers   = by_white_attacker ? p.bb[W_PAWN] : p.bb[B_PAWN];
        Bitboard knight_attackers = by_white_attacker ? p.bb[W_KNIGHT] : p.bb[B_KNIGHT];
        Bitboard king_attackers   = by_white_attacker ? p.bb[W_KING] : p.bb[B_KING];
        Bitboard bishop_queen_attackers = (by_white_attacker ? p.bb[W_BISHOP] : p.bb[B_BISHOP]) |
                                          (by_white_attacker ? p.bb[W_QUEEN] : p.bb[B_QUEEN]);
        Bitboard rook_queen_attackers   = (by_white_attacker ? p.bb[W_ROOK] : p.bb[B_ROOK]) |
                                          (by_white_attacker ? p.bb[W_QUEEN] : p.bb[B_QUEEN]);
        int color_idx = by_white_attacker ? 1 : 0;

        if (attacks::pawn_attacks_table[color_idx][sq_to_check] & pawn_attackers) return true;
        if (attacks::knight_attacks_table[sq_to_check] & knight_attackers) return true;
        if (attacks::king_attacks_table[sq_to_check] & king_attackers) return true;
        if (attacks::get_bishop_attacks(sq_to_check, blockers) & bishop_queen_attackers) return true;
        if (attacks::get_rook_attacks(sq_to_check, blockers) & rook_queen_attackers) return true;

        return false;
    }

    void applyMove(Position& p, Move m) const {
        int from = fromSquare(m);
        int to   = toSquare(m);
        int promo_val_from_move = promotion(m);
        int flags = moveFlags(m);

        Piece moved_piece = p.piece_at(from);
        Piece captured_piece_on_to_sq = p.piece_at(to);

        if (moved_piece == NO_PIECE) { return; }

        uint64_t new_hash = p.currentHash;
        const auto& keys = getZobristKeys();

        new_hash ^= keys.piece_square_keys[moved_piece][from];
        if (captured_piece_on_to_sq != NO_PIECE) { new_hash ^= keys.piece_square_keys[captured_piece_on_to_sq][to]; }
        if (p.epSquare != -1) { new_hash ^= keys.ep_file_keys[file_of(p.epSquare)]; }
        new_hash ^= keys.castling_keys[p.castlingRights & 0xF];

        Bitboard from_bb = 1ULL << from;
        Bitboard to_bb   = 1ULL << to;
        bool original_mover_was_white = p.whiteToMove;

        p.bb[moved_piece] &= ~from_bb;
        p.mailbox[from] = NO_PIECE;

        Piece actual_captured_piece_type = NO_PIECE;

        if (flags & EP_FLAG) {
            int captured_pawn_actual_sq;
            Piece ep_captured_pawn_piece;
            if (original_mover_was_white) {
                captured_pawn_actual_sq = to - 8;
                ep_captured_pawn_piece = B_PAWN;
            } else {
                captured_pawn_actual_sq = to + 8;
                ep_captured_pawn_piece = W_PAWN;
            }
            p.bb[ep_captured_pawn_piece] &= ~(1ULL << captured_pawn_actual_sq);
            p.mailbox[captured_pawn_actual_sq] = NO_PIECE;
            actual_captured_piece_type = ep_captured_pawn_piece;
            new_hash ^= keys.piece_square_keys[ep_captured_pawn_piece][captured_pawn_actual_sq];
        } else if (captured_piece_on_to_sq != NO_PIECE) {
            actual_captured_piece_type = captured_piece_on_to_sq;
            p.bb[actual_captured_piece_type] &= ~to_bb;
        }

        Piece piece_to_place_on_to_sq = moved_piece;

        if (promo_val_from_move != PROMO_TYPE_NONE) {
            Piece promoted_to_piece_enum;
            if (original_mover_was_white) {
                if(promo_val_from_move == PROMO_TYPE_N) promoted_to_piece_enum = W_KNIGHT;
                else if(promo_val_from_move == PROMO_TYPE_B) promoted_to_piece_enum = W_BISHOP;
                else if(promo_val_from_move == PROMO_TYPE_R) promoted_to_piece_enum = W_ROOK;
                else promoted_to_piece_enum = W_QUEEN;
            } else {
                if(promo_val_from_move == PROMO_TYPE_N) promoted_to_piece_enum = B_KNIGHT;
                else if(promo_val_from_move == PROMO_TYPE_B) promoted_to_piece_enum = B_BISHOP;
                else if(promo_val_from_move == PROMO_TYPE_R) promoted_to_piece_enum = B_ROOK;
                else promoted_to_piece_enum = B_QUEEN;
            }
            p.bb[promoted_to_piece_enum] |= to_bb;
            p.mailbox[to] = promoted_to_piece_enum;
            piece_to_place_on_to_sq = promoted_to_piece_enum;
        } else {
            p.bb[moved_piece] |= to_bb;
            p.mailbox[to] = moved_piece;
        }

        new_hash ^= keys.piece_square_keys[piece_to_place_on_to_sq][to];

        if (flags & KSC_FLAG) {
            int r_from_sq = original_mover_was_white ? square(0,7) : square(7,7);
            int r_to_sq   = original_mover_was_white ? square(0,5) : square(7,5);
            Piece r_piece = original_mover_was_white ? W_ROOK : B_ROOK;
            p.bb[r_piece] &= ~(1ULL << r_from_sq);
            p.bb[r_piece] |= (1ULL << r_to_sq);
            p.mailbox[r_from_sq] = NO_PIECE;
            p.mailbox[r_to_sq] = r_piece;
            new_hash ^= keys.piece_square_keys[r_piece][r_from_sq];
            new_hash ^= keys.piece_square_keys[r_piece][r_to_sq];
        } else if (flags & QSC_FLAG) {
            int r_from_sq = original_mover_was_white ? square(0,0) : square(7,0);
            int r_to_sq   = original_mover_was_white ? square(0,3) : square(7,3);
            Piece r_piece = original_mover_was_white ? W_ROOK : B_ROOK;
            p.bb[r_piece] &= ~(1ULL << r_from_sq);
            p.bb[r_piece] |= (1ULL << r_to_sq);
            p.mailbox[r_from_sq] = NO_PIECE;
            p.mailbox[r_to_sq] = r_piece;
            new_hash ^= keys.piece_square_keys[r_piece][r_from_sq];
            new_hash ^= keys.piece_square_keys[r_piece][r_to_sq];
        }

        p.epSquare = -1;
        if (flags & DPP_FLAG) {
            p.epSquare = original_mover_was_white ? (to - 8) : (to + 8);
        }
        if (p.epSquare != -1) {
            new_hash ^= keys.ep_file_keys[file_of(p.epSquare)];
        }

        if (moved_piece == W_KING) { p.castlingRights &= ~(Position::WK_CASTLE_MASK | Position::WQ_CASTLE_MASK); }
        else if (moved_piece == B_KING) { p.castlingRights &= ~(Position::BK_CASTLE_MASK | Position::BQ_CASTLE_MASK); }
        if (from == square(0,0) || to == square(0,0)) { p.castlingRights &= ~Position::WQ_CASTLE_MASK; }
        if (from == square(0,7) || to == square(0,7)) { p.castlingRights &= ~Position::WK_CASTLE_MASK; }
        if (from == square(7,0) || to == square(7,0)) { p.castlingRights &= ~Position::BQ_CASTLE_MASK; }
        if (from == square(7,7) || to == square(7,7)) { p.castlingRights &= ~Position::BK_CASTLE_MASK; }
        if (actual_captured_piece_type == W_ROOK) {
             if (to == square(0,0)) p.castlingRights &= ~Position::WQ_CASTLE_MASK;
             if (to == square(0,7)) p.castlingRights &= ~Position::WK_CASTLE_MASK;
        }
        if (actual_captured_piece_type == B_ROOK) {
             if (to == square(7,0)) p.castlingRights &= ~Position::BQ_CASTLE_MASK;
             if (to == square(7,7)) p.castlingRights &= ~Position::BK_CASTLE_MASK;
        }
        new_hash ^= keys.castling_keys[p.castlingRights & 0xF];


        if (moved_piece == W_PAWN || moved_piece == B_PAWN || actual_captured_piece_type != NO_PIECE) {
            p.halfmoveClock = 0;
        } else {
            p.halfmoveClock++;
        }

        if (!original_mover_was_white) {
            p.fullmoveNumber++;
        }

        p.whiteToMove = !original_mover_was_white;
        new_hash ^= keys.black_to_move_key;

        p.updateOccupancies();
        p.currentHash = new_hash;
    }

}; // End of Engine class

} // namespace chess

#endif // CHESS_ENGINE_HPP
