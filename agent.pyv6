import math
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path

import chess
import chess.polyglot

PIECE_VALUE: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE_SCORE = 100_000
INFINITY = 1_000_000
MAX_DEPTH = 64
MAX_QUIESCENCE_PLY = 64
MATE_THRESHOLD = MATE_SCORE - MAX_DEPTH - MAX_QUIESCENCE_PLY - 1

# Fixed number of slots: the table cannot grow without limit.
TT_SIZE = 1 << 17
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
HISTORY_MASK = (1 << 128) - 1

# Use a local Polyglot opening book through move 25. The book file should sit
# alongside agent.py in the submission directory.
BOOK_MAX_FULLMOVE = 25
_BOOK_PATH = Path(__file__).resolve().with_name("book.bin")

try:
    _BOOK: chess.polyglot.MemoryMappedReader | None = chess.polyglot.open_reader(
        _BOOK_PATH
    )
except (FileNotFoundError, OSError):
    _BOOK = None

type PositionKey = tuple[
    int, int, int, int, int, int, int, int, bool, int, int | None
]


@dataclass(frozen=True, slots=True)
class TTEntry:
    key: PositionKey
    depth: int
    score: int
    bound: int
    move: chess.Move
    history: int
    generation: int


_DEADLINE = math.inf
_NODES = 0
_GAME_BOARD: chess.Board | None = None

_TT: list[TTEntry | None] = [None] * TT_SIZE
_TT_GENERATION = 0
_TT_HITS = 0
_TT_CUTOFFS = 0


class SearchTimeout(Exception):
    """Raised internally to abandon an unfinished search safely."""


def _board_with_history(fen: str) -> chess.Board:
    """Recover the opponent's last move and retain the game's move stack."""
    incoming = chess.Board(fen)
    if _GAME_BOARD is None:
        return incoming

    target = incoming.fen()
    previous = _GAME_BOARD.copy()
    if previous.fen() == target:
        return previous

    # A repeated request for our last position must not count it twice.
    if previous.move_stack:
        previous.pop()
        if previous.fen() == target:
            return previous

    previous = _GAME_BOARD.copy()
    for move in list(previous.legal_moves):
        previous.push(move)
        if previous.fen() == target:
            return previous
        previous.pop()

    # A local caller may skip moves or start another game.
    # Discard stale history instead of inventing moves.
    return incoming


def _remember_move(board: chess.Board, move: chess.Move) -> str:
    """Keep the position after our real move, never a search branch."""
    global _GAME_BOARD
    _GAME_BOARD = board.copy()
    _GAME_BOARD.push(move)
    return move.uci()


def _book_move(board: chess.Board) -> chess.Move | None:
    """Return the highest-weight legal opening-book move, if one exists."""
    if _BOOK is None or board.fullmove_number > BOOK_MAX_FULLMOVE:
        return None

    best_move: chess.Move | None = None
    best_weight = -1

    for entry in _BOOK.find_all(board):
        if board.is_legal(entry.move) and entry.weight > best_weight:
            best_move = entry.move
            best_weight = entry.weight

    return best_move


def _position_key(board: chess.Board) -> PositionKey:
    """Exact board identity, including legally available en passant."""
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def _position_fingerprint(key: PositionKey) -> int:
    """A stable 128-bit contribution to the repetition-history fingerprint."""
    digest = blake2b(repr(key).encode("ascii"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def _history_signature(board: chess.Board) -> int:
    """Fingerprint the position counts relevant to future repetitions.

    Position order does not affect repetition counts. Adding a fingerprint
    for each occurrence preserves multiplicity while allowing different
    move orders with equivalent histories to share cached scores.

    Positions before the last pawn move or capture cannot recur.
    """
    previous = board.copy()
    signature = _position_fingerprint(_position_key(previous))

    steps = min(previous.halfmove_clock, len(previous.move_stack))
    for _ in range(steps):
        previous.pop()
        signature += _position_fingerprint(_position_key(previous))

    return signature & HISTORY_MASK


def _next_history(board: chess.Board, history: int) -> int:
    """Update the fingerprint after a move has been pushed."""
    contribution = _position_fingerprint(_position_key(board))
    if board.halfmove_clock == 0:
        return contribution
    return (history + contribution) & HISTORY_MASK


def _score_to_tt(score: int, ply: int) -> int:
    """Store mate distance relative to the cached position."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    """Restore mate distance relative to the current search root."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def _tt_find(key: PositionKey) -> TTEntry | None:
    """Check the full key so a slot collision cannot impersonate a position."""
    global _TT_HITS
    entry = _TT[hash(key) & (TT_SIZE - 1)]
    if entry is not None and entry.key == key:
        _TT_HITS += 1
        return entry
    return None


def _tt_cutoff(
    entry: TTEntry,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    history: int,
) -> int | None:
    """Reuse a score only with sufficient depth and matching history."""
    global _TT_CUTOFFS

    if entry.depth < depth or entry.history != history:
        return None

    score = _score_from_tt(entry.score, ply)
    if (
        entry.bound == TT_EXACT
        or (entry.bound == TT_LOWER and score >= beta)
        or (entry.bound == TT_UPPER and score <= alpha)
    ):
        _TT_CUTOFFS += 1
        return score

    # Keep the original window when the stored bound cannot cause a cutoff.
    return None


def _tt_store(
    key: PositionKey,
    depth: int,
    score: int,
    bound: int,
    move: chess.Move,
    history: int,
    ply: int,
) -> None:
    """Prefer deeper entries within a move; allow replacement of older ones."""
    index = hash(key) & (TT_SIZE - 1)
    previous = _TT[index]

    if previous is not None and previous.generation == _TT_GENERATION:
        if previous.depth > depth:
            return
        if (
            previous.key == key
            and previous.history == history
            and previous.depth == depth
            and previous.bound == TT_EXACT
            and bound != TT_EXACT
        ):
            return

    _TT[index] = TTEntry(
        key=key,
        depth=depth,
        score=_score_to_tt(score, ply),
        bound=bound,
        move=move,
        history=history,
        generation=_TT_GENERATION,
    )


def _relative_square(square: int, color: chess.Color) -> int:
    """Return a square viewed from that colour's side of the board."""
    return square if color == chess.WHITE else chess.square_mirror(square)


def _piece_square_bonus(
    piece_type: int, square: int, color: chess.Color, endgame: bool
) -> int:
    """A small, deliberately simple positional bonus in centipawns."""
    relative = _relative_square(square, color)
    file_index = chess.square_file(relative)
    rank_index = chess.square_rank(relative)

    centre = 14 - abs(2 * file_index - 7) - abs(2 * rank_index - 7)

    if piece_type == chess.PAWN:
        return 5 * rank_index + centre // 2
    if piece_type == chess.KNIGHT:
        return 3 * centre
    if piece_type == chess.BISHOP:
        return 2 * centre
    if piece_type == chess.ROOK:
        seventh_rank = 12 if rank_index == 6 else 0
        return centre // 2 + seventh_rank
    if piece_type == chess.QUEEN:
        return centre
    if piece_type == chess.KING:
        if endgame:
            return 3 * centre
        castled = 25 if rank_index == 0 and file_index in (2, 6) else 0
        return castled - 2 * centre
    return 0


def _evaluate_white(board: chess.Board) -> int:
    """Evaluate from White's point of view, in centipawns."""
    score = 0
    non_pawn_material = 0

    for piece_type, value in PIECE_VALUE.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        score += value * (white_count - black_count)
        if piece_type != chess.PAWN:
            non_pawn_material += value * (white_count + black_count)

    endgame = non_pawn_material <= 1_600

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for piece_type in (
            chess.PAWN,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
            chess.KING,
        ):
            for square in board.pieces(piece_type, color):
                score += sign * _piece_square_bonus(
                    piece_type, square, color, endgame
                )

        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 30

    return score


def _evaluate(board: chess.Board) -> int:
    """Evaluate from the side-to-move's point of view."""
    white_score = _evaluate_white(board)
    return white_score if board.turn == chess.WHITE else -white_score


def _check_time() -> None:
    """Check the wall clock periodically."""
    global _NODES
    _NODES += 1
    if (_NODES & 63) == 0 and time.perf_counter() >= _DEADLINE:
        raise SearchTimeout


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA-like score: valuable victim, cheap attacker first."""
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        return 0

    victim = (
        chess.PAWN
        if board.is_en_passant(move)
        else board.piece_type_at(move.to_square)
    )
    if victim is None:
        return 0

    return 10 * PIECE_VALUE.get(victim, 0) - PIECE_VALUE.get(attacker, 0)


def _move_order_score(
    board: chess.Board, move: chess.Move, preferred: chess.Move | None
) -> int:
    """Higher scores are searched first."""
    score = 0
    if preferred is not None and move == preferred:
        score += 1_000_000
    if move.promotion is not None:
        score += 100_000 + PIECE_VALUE.get(move.promotion, 0)
    if board.is_capture(move):
        score += 50_000 + _capture_score(board, move)
    if board.gives_check(move):
        score += 5_000
    return score


def _ordered_moves(
    board: chess.Board,
    moves: list[chess.Move],
    preferred: chess.Move | None = None,
) -> list[chess.Move]:
    return sorted(
        moves,
        key=lambda move: _move_order_score(board, move, preferred),
        reverse=True,
    )


def _quiescence(
    board: chess.Board, alpha: int, beta: int, ply: int, qply: int = 0
) -> int:
    """Resolve captures, promotions and check evasions before evaluating."""
    _check_time()

    in_check = board.is_check()
    moves = list(board.legal_moves)
    if not moves:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    if board.is_repetition(3):
        return 0

    if qply >= MAX_QUIESCENCE_PLY:
        raise SearchTimeout

    if in_check:
        best = -INFINITY
    else:
        best = _evaluate(board)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        moves = [
            move
            for move in moves
            if board.is_capture(move) or move.promotion is not None
        ]

    for move in _ordered_moves(board, moves):
        board.push(move)
        try:
            score = -_quiescence(board, -beta, -alpha, ply + 1, qply + 1)
        finally:
            board.pop()

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return best


def _negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    history: int | None = None,
) -> int:
    """Negamax with a history-sensitive transposition table."""
    if depth == 0:
        return _quiescence(board, alpha, beta, ply)

    _check_time()

    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -MATE_SCORE + ply
        return 0

    # A real draw takes precedence over every cached score.
    if board.is_repetition(3):
        return 0

    if history is None:
        history = _history_signature(board)

    key = _position_key(board)
    entry = _tt_find(key)
    preferred: chess.Move | None = None

    if entry is not None and entry.move in moves:
        preferred = entry.move
        cached = _tt_cutoff(entry, depth, alpha, beta, ply, history)
        if cached is not None:
            return cached

    original_alpha = alpha
    ordered = _ordered_moves(board, moves, preferred)
    best = -INFINITY
    best_move = ordered[0]

    for move in ordered:
        board.push(move)
        try:
            # Quiescence is not cached, so its history fingerprint is unused.
            child_history = _next_history(board, history) if depth > 1 else None
            score = -_negamax(
                board, depth - 1, -beta, -alpha, ply + 1, child_history
            )
        finally:
            board.pop()

        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    if best <= original_alpha:
        bound = TT_UPPER
    elif best >= beta:
        bound = TT_LOWER
    else:
        bound = TT_EXACT

    # This is reached only after the node completes or proves a cutoff.
    # An exception leaves this unfinished node out of the table.
    _tt_store(key, depth, best, bound, best_move, history, ply)
    return best


def _search_root(
    board: chess.Board, depth: int, preferred: chess.Move | None
) -> tuple[int, chess.Move]:
    """Search one complete iterative-deepening pass."""
    _check_time()

    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("get_move called with no legal moves")

    key = _position_key(board)
    history = _history_signature(board)
    entry = _tt_find(key)

    if entry is not None and entry.move in moves:
        preferred = entry.move
        cached = _tt_cutoff(
            entry, depth, -INFINITY, INFINITY, 0, history
        )
        if cached is not None and not board.is_repetition(3):
            return cached, entry.move

    ordered = _ordered_moves(board, moves, preferred)
    best_move = ordered[0]
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY

    for move in ordered:
        _check_time()
        board.push(move)
        try:
            child_history = _next_history(board, history) if depth > 1 else None
            score = -_negamax(
                board, depth - 1, -beta, -alpha, 1, child_history
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    # Root results are committed only after every root move is handled.
    _tt_store(key, depth, best_score, TT_EXACT, best_move, history, 0)
    return best_score, best_move


def _move_budget_ms(time_left_ms: int) -> int:
    """Conservative first-pass clock management."""
    if time_left_ms <= 100:
        return max(1, time_left_ms // 4)

    reserve_ms = max(50, min(1_000, time_left_ms // 20))
    usable_ms = max(1, time_left_ms - reserve_ms)

    desired_ms = time_left_ms // 35 + 150
    if time_left_ms < 5_000:
        desired_ms = min(desired_ms, 250)

    return max(1, min(3_000, usable_ms, desired_ms))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the supplied FEN."""
    global _DEADLINE, _NODES, _TT_GENERATION, _TT_HITS, _TT_CUTOFFS

    board = _board_with_history(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("get_move called with no legal moves")

    if len(legal_moves) == 1:
        return _remember_move(board, legal_moves[0])

    if board.fullmove_number <= BOOK_MAX_FULLMOVE:
        book_move = _book_move(board)
        if book_move is not None:
            print(f"book_hit=1 move={book_move.uci()}")
            return _remember_move(board, book_move)
        print("book_hit=0")
    else:
        print("book_skip=move_limit")

    _TT_GENERATION += 1
    _TT_HITS = 0
    _TT_CUTOFFS = 0

    best_move = _ordered_moves(board, legal_moves)[0]
    best_score = 0
    completed_depth = 0

    budget_ms = _move_budget_ms(time_left_ms)
    _DEADLINE = time.perf_counter() + budget_ms / 1_000.0
    _NODES = 0

    for depth in range(1, MAX_DEPTH + 1):
        try:
            score, move = _search_root(board, depth, best_move)
        except SearchTimeout:
            break

        best_score = score
        best_move = move
        completed_depth = depth

        if abs(best_score) >= MATE_SCORE - 100:
            break

    print(
        f"depth={completed_depth} score={best_score} nodes={_NODES} "
        f"budget_ms={budget_ms} tt_hits={_TT_HITS} "
        f"tt_cutoffs={_TT_CUTOFFS} move={best_move.uci()}"
    )
    return _remember_move(board, best_move)
