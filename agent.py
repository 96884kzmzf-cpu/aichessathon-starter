"""V1a AI Chessathon agent: iterative deepening + alpha-beta + simple evaluation."""

import math
import time

import chess

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

_DEADLINE = math.inf
_NODES = 0


class SearchTimeout(Exception):
    """Raised internally when the current move's search budget has expired."""


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

    # 0-ish at the corners, largest near d4/e4/d5/e5.
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
                score += sign * _piece_square_bonus(piece_type, square, color, endgame)

        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 30

    return score


def _evaluate(board: chess.Board) -> int:
    """Evaluate from the side-to-move's point of view."""
    white_score = _evaluate_white(board)
    return white_score if board.turn == chess.WHITE else -white_score


def _check_time() -> None:
    """Check the wall clock periodically without paying the cost at every node."""
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
    board: chess.Board, moves: list[chess.Move], preferred: chess.Move | None = None
) -> list[chess.Move]:
    return sorted(
        moves,
        key=lambda move: _move_order_score(board, move, preferred),
        reverse=True,
    )


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Negamax search with alpha-beta pruning."""
    _check_time()

    moves = list(board.legal_moves)
    if not moves:
        if board.is_check():
            return -MATE_SCORE + ply
        return 0

    if depth == 0:
        return _evaluate(board)

    best = -INFINITY
    for move in _ordered_moves(board, moves):
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        finally:
            board.pop()

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return best


def _search_root(
    board: chess.Board, depth: int, preferred: chess.Move | None
) -> tuple[int, chess.Move]:
    """Search one complete iterative-deepening pass."""
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("get_move called with no legal moves")

    ordered = _ordered_moves(board, moves, preferred)
    best_move = ordered[0]
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY

    for move in ordered:
        _check_time()
        board.push(move)
        try:
            score = -_negamax(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return best_score, best_move


def _move_budget_ms(time_left_ms: int) -> int:
    """Conservative first-pass clock management."""
    if time_left_ms <= 100:
        return max(1, time_left_ms // 4)

    reserve_ms = max(50, min(1_000, time_left_ms // 20))
    usable_ms = max(1, time_left_ms - reserve_ms)

    # Roughly 1/35 of the remaining clock, plus part of the next 500 ms increment.
    desired_ms = time_left_ms // 35 + 150
    if time_left_ms < 5_000:
        desired_ms = min(desired_ms, 250)

    return max(1, min(3_000, usable_ms, desired_ms))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move for the supplied FEN."""
    global _DEADLINE, _NODES

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("get_move called with no legal moves")
    if len(legal_moves) == 1:
        return legal_moves[0].uci()

    # Always have a legal fallback before starting timed search.
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

        # No need to spend more time once a forced mate is already found.
        if abs(best_score) >= MATE_SCORE - 100:
            break

    print(
        f"depth={completed_depth} score={best_score} nodes={_NODES} "
        f"budget_ms={budget_ms} move={best_move.uci()}"
    )
    return best_move.uci()
