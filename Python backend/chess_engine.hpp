/*
 * Bitboard-based chess engine – legal move gen + game-state detection
 * Implemented Zobrist hashing for position hashing and repetition detection.
 * Enhanced negamax to consider game history for threefold repetition.
 * Added method to get nodes visited.
 * Implemented Transposition Table with Zobrist Hashing.
 * Implemented Iterative Deepening in findBestMove.
 * Added advanced move ordering (Promotions, MVV-LVA Captures, TT Move).
 *
 * REFACTOR 1:
 * - Added `mailbox` array to Position struct for O(1) piece lookups.
 * - `piece_at()` is now O(1).
 * - `applyMove()` now incrementally updates the mailbox.
 * - Added precomputed attack tables for non-sliding pieces.
 * - `isSquareAttacked()` completely rewritten to use fast bitboard operations,
 * removing the major performance bottleneck.
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

// ───────────────────────── Attack Generation ─────────────────────────
namespace attacks {
    std::array<Bitboard, 64> knight_attacks_table;
    std::array<Bitboard, 64> king_attacks_table;
    std::array<std::array<Bitboard, 64>, 2> pawn_attacks_table; // [color][square]

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
        }
    }

    inline Bitboard get_rook_attacks(int sq, Bitboard blockers) {
        Bitboard attacks = 0ULL;
        int r, f;
        int r_orig = rank_of(sq);
        int f_orig = file_of(sq);

        for (r = r_orig + 1; r < 8; ++r) { attacks |= (1ULL << square(r, f_orig)); if (blockers & (1ULL << square(r, f_orig))) break; }
        for (r = r_orig - 1; r >= 0; --r) { attacks |= (1ULL << square(r, f_orig)); if (blockers & (1ULL << square(r, f_orig))) break; }
        for (f = f_orig + 1; f < 8; ++f) { attacks |= (1ULL << square(r_orig, f)); if (blockers & (1ULL << square(r_orig, f))) break; }
        for (f = f_orig - 1; f >= 0; --f) { attacks |= (1ULL << square(r_orig, f)); if (blockers & (1ULL << square(r_orig, f))) break; }
        return attacks;
    }

    inline Bitboard get_bishop_attacks(int sq, Bitboard blockers) {
        Bitboard attacks = 0ULL;
        int r, f;
        int r_orig = rank_of(sq);
        int f_orig = file_of(sq);

        for (r = r_orig + 1, f = f_orig + 1; r < 8 && f < 8; ++r, ++f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        for (r = r_orig + 1, f = f_orig - 1; r < 8 && f >= 0; ++r, --f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        for (r = r_orig - 1, f = f_orig + 1; r >= 0 && f < 8; --r, ++f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        for (r = r_orig - 1, f = f_orig - 1; r >= 0 && f >= 0; --r, --f) { attacks |= (1ULL << square(r, f)); if (blockers & (1ULL << square(r, f))) break; }
        return attacks;
    }
} // namespace attacks

namespace { // Anonymous namespace to ensure initialization
    struct Initializer { Initializer() { attacks::init(); } };
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

    // **NEW**: Must be called once after setting up bitboards for a new position.
    // `applyMove` will maintain the mailbox incrementally afterwards.
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

    // **REFACTORED**: Now O(1) thanks to the mailbox.
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
    static const int SCORE_PV_MOVE_BONUS = 2000000; // If it's the principal variation move from TT
    static const int SCORE_PROMOTION_TO_QUEEN = 1900000;
    static const int SCORE_PROMOTION_OTHER    = 1750000;
    static const int SCORE_CAPTURE_BASE       = 1000000; // Base for any capture

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
            generateLegalMoves(root_pos, root_moves);

            if (root_moves.empty()) {
                 return 0;
            }

            // Move ordering for root moves: use best move from previous iteration
            if (overall_best_move != 0) {
                auto it = std::find(root_moves.begin(), root_moves.end(), overall_best_move);
                if (it != root_moves.end() && it != root_moves.begin()) {
                    std::rotate(root_moves.begin(), it, it + 1);
                }
            }

            Move current_iter_best_root_move = 0;
            int best_score_this_iter_at_root = -INF - 1000;
            int alpha = -INF; // Initial alpha for root
            int beta = INF;   // Initial beta for root


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
        generateLegalMoves(pos, legal_moves);

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

        if (depth == 0) {
            return 1ULL;
        }

        std::vector<Move> legal_moves;
        generateLegalMoves(p, legal_moves);
        uint64_t nodes = 0;

        if (depth == 1) {
            return static_cast<uint64_t>(legal_moves.size());
        }

        for (Move m : legal_moves) {
            Position next_pos = p;
            applyMove(next_pos, m);
            nodes += perft(next_pos, depth - 1);
        }
        return nodes;
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


private:
    const int INF=std::numeric_limits<int>::max() - 200000;
    int maxDepth;
    long nodes_visited_search = 0;

    std::unordered_map<uint64_t, TranspositionTableEntry> transposition_table;

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
                return 0; // Draw by repetition in current search path
            }
        }

        int game_history_repetitions = 0;
        for (uint64_t historical_game_hash : game_history_hashes) {
            if (historical_game_hash == p.currentHash) {
                game_history_repetitions++;
            }
        }
        if (game_history_repetitions >= 2) {
            return 0; // Draw by threefold repetition including game history
        }

        // TT Lookup
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

        // At leaf node of search or in quiescence search, we would return evaluation
        if (remaining_depth <= 0) {
            int eval = evaluateMaterial(p);
            return p.whiteToMove ? eval : -eval;
        }

        std::vector<Move> moves;
        generateLegalMoves(p, moves);

        if (moves.empty()) {
            int king_piece_idx = p.whiteToMove ? W_KING : B_KING;
            int king_sq = lsb_idx(p.bb[king_piece_idx]);
            if (isSquareAttacked(p, king_sq, !p.whiteToMove)) {
                return -INF + (this->maxDepth - remaining_depth); // Checkmate
            }
            return 0; // Stalemate
        }

        // Move Ordering
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
            if (alpha >= beta) { // Beta-cutoff
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

        // Store result in TT
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

    void add_pawn_moves(const Position& p, int from_sq, std::vector<Move>& moves) const {
        bool is_white = p.whiteToMove;
        int dir = is_white ? 1 : -1;
        int move_one_step_idx_change = dir * 8;
        int move_two_steps_idx_change = dir * 16;

        int start_rank_board_idx = is_white ? 1 : 6;
        int promotion_rank_board_idx = is_white ? 7 : 0;

        int to_sq1 = from_sq + move_one_step_idx_change;
        if (on_board(to_sq1) && !(p.occ & (1ULL << to_sq1))) {
            if (rank_of(to_sq1) == promotion_rank_board_idx) {
                moves.push_back(encodeMove(from_sq, to_sq1, PROMO_TYPE_Q));
                moves.push_back(encodeMove(from_sq, to_sq1, PROMO_TYPE_R));
                moves.push_back(encodeMove(from_sq, to_sq1, PROMO_TYPE_B));
                moves.push_back(encodeMove(from_sq, to_sq1, PROMO_TYPE_N));
            } else {
                moves.push_back(encodeMove(from_sq, to_sq1));
            }

            if (rank_of(from_sq) == start_rank_board_idx) {
                int to_sq2 = from_sq + move_two_steps_idx_change;
                if (on_board(to_sq2) && !(p.occ & (1ULL << to_sq2))) {
                    moves.push_back(encodeMove(from_sq, to_sq2, PROMO_TYPE_NONE, DPP_FLAG));
                }
            }
        }

        for (int df = -1; df <= 1; df += 2) {
            int capture_col_offset = df;
            int to_sq_cap = from_sq + move_one_step_idx_change + capture_col_offset;

            if (on_board(to_sq_cap) && file_of(to_sq_cap) == (file_of(from_sq) + capture_col_offset)) {
                Bitboard enemy_occ = is_white ? p.occBlack : p.occWhite;
                if (enemy_occ & (1ULL << to_sq_cap)) {
                    if (rank_of(to_sq_cap) == promotion_rank_board_idx) {
                        moves.push_back(encodeMove(from_sq, to_sq_cap, PROMO_TYPE_Q));
                        moves.push_back(encodeMove(from_sq, to_sq_cap, PROMO_TYPE_R));
                        moves.push_back(encodeMove(from_sq, to_sq_cap, PROMO_TYPE_B));
                        moves.push_back(encodeMove(from_sq, to_sq_cap, PROMO_TYPE_N));
                    } else {
                        moves.push_back(encodeMove(from_sq, to_sq_cap));
                    }
                } else if (to_sq_cap == p.epSquare) {
                    moves.push_back(encodeMove(from_sq, to_sq_cap, PROMO_TYPE_NONE, EP_FLAG));
                }
            }
        }
    }

    void add_knight_moves(const Position& p, int from_sq, std::vector<Move>& moves) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard knight_moves = attacks::knight_attacks_table[from_sq] & ~friendly_occ;

        while(knight_moves) {
            int to_sq = lsb_idx(knight_moves);
            moves.push_back(encodeMove(from_sq, to_sq));
            knight_moves &= knight_moves - 1;
        }
    }

    void add_sliding_moves(const Position& p, int from_sq, bool is_bishop, bool is_rook, std::vector<Move>& moves) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard slide_moves = 0ULL;

        if (is_bishop) {
            slide_moves |= attacks::get_bishop_attacks(from_sq, p.occ);
        }
        if (is_rook) {
            slide_moves |= attacks::get_rook_attacks(from_sq, p.occ);
        }

        slide_moves &= ~friendly_occ;

        while(slide_moves) {
            int to_sq = lsb_idx(slide_moves);
            moves.push_back(encodeMove(from_sq, to_sq));
            slide_moves &= slide_moves - 1;
        }
    }

    void add_king_moves(const Position& p, int from_sq, std::vector<Move>& moves) const {
        Bitboard friendly_occ = p.whiteToMove ? p.occWhite : p.occBlack;
        Bitboard king_moves = attacks::king_attacks_table[from_sq] & ~friendly_occ;

        while (king_moves) {
            int to_sq = lsb_idx(king_moves);
            moves.push_back(encodeMove(from_sq, to_sq));
            king_moves &= king_moves - 1;
        }

        // Castling
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

    void generatePseudoLegalMoves(const Position& p, std::vector<Move>& moves) const {
        moves.clear();
        Piece start_piece_idx = p.whiteToMove ? W_PAWN : B_PAWN;
        Piece end_piece_idx   = p.whiteToMove ? W_KING : B_KING;

        for (int piece_type_val = start_piece_idx; piece_type_val <= end_piece_idx; ++piece_type_val) {
            Bitboard current_bb = p.bb[piece_type_val];
            while (current_bb) {
                int from_sq = lsb_idx(current_bb);
                Piece pt = static_cast<Piece>(piece_type_val);

                if (pt == W_PAWN || pt == B_PAWN) add_pawn_moves(p, from_sq, moves);
                else if (pt == W_KNIGHT || pt == B_KNIGHT) add_knight_moves(p, from_sq, moves);
                else if (pt == W_BISHOP || pt == B_BISHOP) add_sliding_moves(p, from_sq, true, false, moves);
                else if (pt == W_ROOK || pt == B_ROOK) add_sliding_moves(p, from_sq, false, true, moves);
                else if (pt == W_QUEEN || pt == B_QUEEN) add_sliding_moves(p, from_sq, true, true, moves);
                else if (pt == W_KING || pt == B_KING) add_king_moves(p, from_sq, moves);

                current_bb &= current_bb - 1;
            }
        }
    }

    void generateLegalMoves(const Position& p, std::vector<Move>& legal_moves) const {
        legal_moves.clear();
        std::vector<Move> pseudo_legal;
        generatePseudoLegalMoves(p, pseudo_legal);

        Piece king_piece_for_current_player = p.whiteToMove ? W_KING : B_KING;

        for (Move m : pseudo_legal) {
            Position next_pos = p;
            applyMove(next_pos, m); // This is slow, but correct for now. Future optimization here.

            Bitboard king_bb_after_move = next_pos.bb[king_piece_for_current_player];

            if (king_bb_after_move == 0) {
                 continue;
            }
            int king_sq_after_move = lsb_idx(king_bb_after_move);

            // Check if own king is attacked by the opponent
            if (!isSquareAttacked(next_pos, king_sq_after_move, next_pos.whiteToMove )) {
                legal_moves.push_back(m);
            }
        }
    }

    // **REFACTORED**: Now uses fast bitboard operations.
    bool isSquareAttacked(const Position& p, int sq_to_check, bool by_white_attacker) const {
        Bitboard pawn_attackers   = by_white_attacker ? p.bb[W_PAWN] : p.bb[B_PAWN];
        Bitboard knight_attackers = by_white_attacker ? p.bb[W_KNIGHT] : p.bb[B_KNIGHT];
        Bitboard king_attackers   = by_white_attacker ? p.bb[W_KING] : p.bb[B_KING];
        Bitboard bishop_queen_attackers = (by_white_attacker ? p.bb[W_BISHOP] : p.bb[B_BISHOP]) |
                                          (by_white_attacker ? p.bb[W_QUEEN] : p.bb[B_QUEEN]);
        Bitboard rook_queen_attackers   = (by_white_attacker ? p.bb[W_ROOK] : p.bb[B_ROOK]) |
                                          (by_white_attacker ? p.bb[W_QUEEN] : p.bb[B_QUEEN]);
        int color_idx = by_white_attacker ? 1 : 0; // Pawn attacks are from perspective of color being attacked

        if (attacks::pawn_attacks_table[color_idx][sq_to_check] & pawn_attackers) return true;
        if (attacks::knight_attacks_table[sq_to_check] & knight_attackers) return true;
        if (attacks::king_attacks_table[sq_to_check] & king_attackers) return true;
        if (attacks::get_bishop_attacks(sq_to_check, p.occ) & bishop_queen_attackers) return true;
        if (attacks::get_rook_attacks(sq_to_check, p.occ) & rook_queen_attackers) return true;

        return false;
    }

    // **REFACTORED**: Updates mailbox incrementally.
    void applyMove(Position& p, Move m) const {
        int from = fromSquare(m);
        int to   = toSquare(m);
        int promo_val_from_move = promotion(m);
        int flags = moveFlags(m);

        Piece moved_piece = p.piece_at(from);
        Piece captured_piece_on_to_sq = p.piece_at(to); // From mailbox

        if (moved_piece == NO_PIECE) { return; }

        uint64_t new_hash = p.currentHash;
        const auto& keys = getZobristKeys();

        // Update hash for pieces, castling, and ep before state changes
        new_hash ^= keys.piece_square_keys[moved_piece][from];
        if (captured_piece_on_to_sq != NO_PIECE) { new_hash ^= keys.piece_square_keys[captured_piece_on_to_sq][to]; }
        if (p.epSquare != -1) { new_hash ^= keys.ep_file_keys[file_of(p.epSquare)]; }
        new_hash ^= keys.castling_keys[p.castlingRights & 0xF];

        // --- Make move on bitboards and mailbox ---
        Bitboard from_bb = 1ULL << from;
        Bitboard to_bb   = 1ULL << to;
        bool original_mover_was_white = p.whiteToMove;

        // Move piece
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
            // Mailbox at 'to' will be overwritten by moving piece, no extra action needed.
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

        // Update castling rights
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