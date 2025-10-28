Minerva Chess Engine

Project Overview

Minerva is a high-performance chess engine developed in C++, integrated with a minimal web interface powered by Flask (Python) and HTML.

The primary objective of this project is to serve as a robust platform for research into the efficiency and optimization of chess engines, specifically through self-play and curriculum-based learning methodologies.

Architecture

The system is split into two main components:

Engine Core (C++): Handles all game logic, search, and evaluation. This core is designed for speed and is intended to be called by the web server.

Web Frontend (Python/Flask & HTML): Provides a simple graphical user interface (GUI) via an HTML page, allowing users to interact with the C++ engine logic through a REST or similar interface exposed by the Flask server.

Core Engine Optimizations

The C++ engine is built on standard, high-efficiency techniques to ensure competitive performance and reliable search results:

Bitboards: Utilized for fast and efficient representation of the chess board state and move generation.

Alpha-Beta Pruning: Implemented to drastically reduce the search space.

Iterative Deepening: Provides dynamic control over search depth and enables time management.

Move Ordering: Employs heuristics (like MVV/LVA) to prioritize promising moves early in the search.

Transposition Tables: Used to store and reuse results of previously analyzed positions, preventing redundant calculations.

Research Focus

This engine is specifically configured for experiments involving training algorithms where the complexity and depth of opponent play are systematically increased over time (curriculum learning).
