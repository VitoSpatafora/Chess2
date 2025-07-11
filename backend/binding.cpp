#include <cstdint>
#include <array>
#include <vector>
#include <limits>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // Needed for std::vector, std::array, etc.
#include <pybind11/operators.h> // For operator overloading if needed
#include "chess_engine.hpp"     // Your main C++ chess engine header

namespace py = pybind11;
// We will use explicit chess:: qualification for enums in py::enum_ to be safe.

PYBIND11_MODULE(chesscore, m) {
    m.doc() = "Python bindings for the C++ Chess Engine";

    // Expose enum Piece
    py::enum_<chess::Piece>(m, "Piece", "Enum for chess pieces")
        .value("W_PAWN",   chess::Piece::W_PAWN)
        .value("W_KNIGHT", chess::Piece::W_KNIGHT)
        .value("W_BISHOP", chess::Piece::W_BISHOP)
        .value("W_ROOK",   chess::Piece::W_ROOK)
        .value("W_QUEEN",  chess::Piece::W_QUEEN)
        .value("W_KING",   chess::Piece::W_KING)
        .value("B_PAWN",   chess::Piece::B_PAWN)
        .value("B_KNIGHT", chess::Piece::B_KNIGHT)
        .value("B_BISHOP", chess::Piece::B_BISHOP)
        .value("B_ROOK",   chess::Piece::B_ROOK)
        .value("B_QUEEN",  chess::Piece::B_QUEEN)
        .value("B_KING",   chess::Piece::B_KING)
        .value("NO_PIECE", chess::Piece::NO_PIECE)
        .export_values();

    // Expose enum PromoPieceType with explicit namespace
    py::enum_<chess::PromoPieceType>(m, "PromoPieceType", "Enum for promotion piece types")
        .value("PROMO_TYPE_NONE", chess::PromoPieceType::PROMO_TYPE_NONE)
        .value("PROMO_TYPE_N",    chess::PromoPieceType::PROMO_TYPE_N)
        .value("PROMO_TYPE_B",    chess::PromoPieceType::PROMO_TYPE_B)
        .value("PROMO_TYPE_R",    chess::PromoPieceType::PROMO_TYPE_R)
        .value("PROMO_TYPE_Q",    chess::PromoPieceType::PROMO_TYPE_Q)
        .export_values();

    // Expose struct Position
    py::class_<chess::Position>(m, "Position", "Represents the chess board position and game state")
        .def(py::init<>())
        .def_readwrite("bb", &chess::Position::bb, "Array of 12 bitboards for each piece type")
        .def_readwrite("occWhite", &chess::Position::occWhite, "Bitboard of all white pieces")
        .def_readwrite("occBlack", &chess::Position::occBlack, "Bitboard of all black pieces")
        .def_readwrite("occ", &chess::Position::occ, "Bitboard of all occupied squares")
        .def_readwrite("castlingRights", &chess::Position::castlingRights, "Castling availability (KQkq masks)")
        .def_readwrite("epSquare", &chess::Position::epSquare, "En passant target square index (-1 if none)")
        .def_readwrite("whiteToMove", &chess::Position::whiteToMove, "True if it's White's turn")
        .def_readwrite("halfmoveClock", &chess::Position::halfmoveClock, "Halfmove clock for 50-move rule")
        .def_readwrite("fullmoveNumber", &chess::Position::fullmoveNumber, "Current fullmove number")
        .def_readwrite("currentHash", &chess::Position::currentHash, "Current Zobrist hash of the position") // Expose currentHash
        .def("updateOccupancies", &chess::Position::updateOccupancies, "Recalculates occWhite, occBlack, and occ bitboards")
        .def("syncMailboxFromBitboards", &chess::Position::syncMailboxFromBitboards, "Synchronizes the mailbox array from the bitboards. MUST be called after setting up bitboards.")
        .def("piece_at", &chess::Position::piece_at, py::arg("sq"), "Get piece at a given square index (0-63)")
        .def("computeAndSetHash", &chess::Position::computeAndSetHash, "Computes and stores the Zobrist hash for the position") // Expose computeAndSetHash
        .def("pretty", &chess::Position::pretty, "Returns a string representation of the board for debugging")
        .def_property_readonly_static("WK_CASTLE_MASK", [](py::object /* self */) { return chess::Position::WK_CASTLE_MASK; })
        .def_property_readonly_static("WQ_CASTLE_MASK", [](py::object /* self */) { return chess::Position::WQ_CASTLE_MASK; })
        .def_property_readonly_static("BK_CASTLE_MASK", [](py::object /* self */) { return chess::Position::BK_CASTLE_MASK; })
        .def_property_readonly_static("BQ_CASTLE_MASK", [](py::object /* self */) { return chess::Position::BQ_CASTLE_MASK; });


    // Expose enum Outcome
     py::enum_<chess::Outcome>(m, "Outcome", "Possible game outcomes")
        .value("ONGOING", chess::Outcome::ONGOING)
        .value("CHECKMATE", chess::Outcome::CHECKMATE)
        .value("STALEMATE", chess::Outcome::STALEMATE)
        .value("DRAW_FIFTY_MOVE", chess::Outcome::DRAW_FIFTY_MOVE)
        .value("DRAW_THREEFOLD_REPETITION", chess::Outcome::DRAW_THREEFOLD_REPETITION)
        .export_values();

    // Expose class Engine
    py::class_<chess::Engine>(m, "Engine", "The chess AI engine")
        .def(py::init<int>(), py::arg("depth") = 4, "Engine constructor, sets search depth")
        .def("findBestMove", &chess::Engine::findBestMove,
             py::arg("root_pos"),
             py::arg("history_hashes"),
             "Finds the best move for the given position, considering game history hashes.")
        .def("currentOutcome", &chess::Engine::currentOutcome,
             py::arg("pos"),
             py::arg("game_history_hashes"),
             "Determines the current game outcome based on position and game history hashes")
        .def("perft", &chess::Engine::perft,
             py::arg("position"),
             py::arg("depth"),
             "Run a performance test (perft) to the given depth from the position.")
        .def("getNodesVisited", &chess::Engine::getNodesVisited, "Get the number of nodes visited in the last search") // Corrected function name
        // Corrected static method binding for hashing:
        .def_static("getHashForPosition", &chess::Engine::getHashForPosition,
             py::arg("p"),
             "Calculates a Zobrist hash for the given position object (does not store it in the object).");

}