import math
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.polyglot
import chess.syzygy

# -----------------------------------------------------------------------------
# Core constants
# -----------------------------------------------------------------------------

MATE_SCORE = 100_000
INFINITY = 1_000_000
MAX_DEPTH = 64
MAX_QUIESCENCE_PLY = 64
MAX_PLY = MAX_DEPTH + MAX_QUIESCENCE_PLY + 8
MATE_THRESHOLD = MATE_SCORE - MAX_PLY - 1

TT_SIZE = 1 << 18
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
MASK64 = (1 << 64) - 1

BOOK_MAX_FULLMOVE = 25
_BOOK_PATH = Path(__file__).resolve().with_name("book.bin")

try:
    _BOOK: chess.polyglot.MemoryMappedReader | None = chess.polyglot.open_reader(
        _BOOK_PATH
    )
except (FileNotFoundError, OSError):
    _BOOK = None

TABLEBASE_MAX_PIECES = 3
_SYZYGY_PATH = Path(__file__).resolve().with_name("syzygy")

try:
    _TABLEBASE: chess.syzygy.Tablebase | None = (
        chess.syzygy.open_tablebase(str(_SYZYGY_PATH))
        if _SYZYGY_PATH.is_dir()
        else None
    )
except (OSError, ValueError):
    _TABLEBASE = None

# Piece values are intentionally close to conventional centipawn values.
MG_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
EG_VALUE = {
    chess.PAWN: 120,
    chess.KNIGHT: 310,
    chess.BISHOP: 335,
    chess.ROOK: 520,
    chess.QUEEN: 900,
}
ORDER_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

# Tapered phase: initial position = 24.
PHASE_WEIGHT = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
MAX_PHASE = 24

# Original 64-square MG/EG piece-square tables. They are generated once at
# import from explicit, piece-specific square features, then evaluation is a
# cheap lookup. This keeps the submitted evaluator fully ours and explainable.
def _build_pst(piece_type: int, endgame: bool) -> tuple[int, ...]:
    values: list[int] = []
    for table_index in range(64):
        file_index = table_index % 8
        rank_index = 7 - table_index // 8  # White-relative rank: 0 == rank 1.
        centre = 14 - abs(2 * file_index - 7) - abs(2 * rank_index - 7)
        edge_file = min(file_index, 7 - file_index)

        if piece_type == chess.PAWN:
            if rank_index in (0, 7):
                value = 0
            else:
                advance_mg = (0, 0, 4, 8, 13, 20, 30, 0)
                advance_eg = (0, 0, 6, 12, 20, 32, 48, 0)
                centre_file = (0, 2, 4, 7, 7, 4, 2, 0)[file_index]
                value = (advance_eg if endgame else advance_mg)[rank_index]
                value += centre_file if not endgame else centre_file // 2

        elif piece_type == chess.KNIGHT:
            rank_term_mg = (-18, -6, 4, 9, 10, 5, -5, -18)[rank_index]
            rank_term_eg = (-10, -3, 3, 7, 8, 5, -2, -10)[rank_index]
            rim_penalty = 14 if edge_file == 0 else 4 if edge_file == 1 else 0
            value = (3 if endgame else 4) * centre
            value += rank_term_eg if endgame else rank_term_mg
            value -= rim_penalty

        elif piece_type == chess.BISHOP:
            rank_term_mg = (-8, 0, 4, 7, 8, 6, 2, -8)[rank_index]
            rank_term_eg = (-4, 0, 3, 5, 6, 5, 1, -4)[rank_index]
            value = (2 if not endgame else 2) * centre
            value += rank_term_eg if endgame else rank_term_mg
            if file_index in (0, 7) and rank_index in (0, 7):
                value -= 8

        elif piece_type == chess.ROOK:
            central_file = 4 if file_index in (3, 4) else 2 if file_index in (2, 5) else 0
            value = central_file
            if rank_index == 6:
                value += 24 if not endgame else 20
            elif rank_index == 5:
                value += 7 if not endgame else 9
            elif rank_index == 0:
                value -= 2 if not endgame else 0

        elif piece_type == chess.QUEEN:
            value = centre if not endgame else 2 * centre
            if not endgame and rank_index >= 4:
                value -= 3

        elif piece_type == chess.KING:
            if endgame:
                value = 4 * centre
                if edge_file == 0:
                    value -= 8
            else:
                # General safety plus explicit castled-home preferences.
                value = -3 * centre
                if rank_index == 0:
                    if file_index == 6:  # g1
                        value += 34
                    elif file_index == 2:  # c1
                        value += 28
                    elif file_index in (1, 5):
                        value += 8
                elif rank_index >= 2:
                    value -= 12
        else:
            value = 0

        values.append(value)
    return tuple(values)


MG_PST: dict[int, tuple[int, ...]] = {
    piece_type: _build_pst(piece_type, False)
    for piece_type in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )
}
EG_PST: dict[int, tuple[int, ...]] = {
    piece_type: _build_pst(piece_type, True)
    for piece_type in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )
}

PASSED_MG = (0, 5, 10, 20, 38, 68, 110, 0)
PASSED_EG = (0, 10, 22, 42, 75, 125, 190, 0)

# Mobility is expressed as a small bonus around a reasonable baseline.
MOBILITY_BASE = {
    chess.KNIGHT: 4,
    chess.BISHOP: 7,
    chess.ROOK: 7,
    chess.QUEEN: 14,
}
MOBILITY_MG = {
    chess.KNIGHT: 4,
    chess.BISHOP: 3,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}
MOBILITY_EG = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 3,
    chess.QUEEN: 1,
}

# -----------------------------------------------------------------------------
# Search state
# -----------------------------------------------------------------------------

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
_QNODES = 0
_SELDEPTH = 0
_GAME_BOARD: chess.Board | None = None

_TT: list[TTEntry | None] = [None] * TT_SIZE
_TT_GENERATION = 0
_TT_HITS = 0
_TT_CUTOFFS = 0
_PVS_RESEARCHES = 0
_LMR_REDUCTIONS = 0
_LMR_RESEARCHES = 0

_KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
_HISTORY: list[list[list[int]]] = [
    [[0 for _ in range(64)] for _ in range(64)] for _ in range(2)
]


class SearchTimeout(Exception):
    """Raised internally to abandon an unfinished search safely."""


# -----------------------------------------------------------------------------
# Persistent game history, book and tablebase
# -----------------------------------------------------------------------------


def _board_with_history(fen: str) -> chess.Board:
    """Recover the opponent's last move and retain the game's move stack."""
    incoming = chess.Board(fen)
    if _GAME_BOARD is None:
        return incoming

    target = incoming.fen()
    previous = _GAME_BOARD.copy()
    if previous.fen() == target:
        return previous

    # A repeated request for our last pre-move position must not duplicate it.
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

    # Local tests may skip moves or begin from an arbitrary FEN.
    return incoming


def _remember_move(board: chess.Board, move: chess.Move) -> str:
    global _GAME_BOARD
    _GAME_BOARD = board.copy()
    _GAME_BOARD.push(move)
    return move.uci()


def _book_move(board: chess.Board) -> chess.Move | None:
    if _BOOK is None or board.fullmove_number > BOOK_MAX_FULLMOVE:
        return None

    best_move: chess.Move | None = None
    best_weight = -1
    for entry in _BOOK.find_all(board):
        if board.is_legal(entry.move) and entry.weight > best_weight:
            best_move = entry.move
            best_weight = entry.weight
    return best_move


def _tablebase_child_value(board: chess.Board) -> tuple[int, int] | None:
    if board.is_checkmate():
        return (-2, 0)
    if board.is_stalemate() or board.is_insufficient_material():
        return (0, 0)
    if _TABLEBASE is None:
        return None
    try:
        return (_TABLEBASE.probe_wdl(board), _TABLEBASE.probe_dtz(board))
    except (chess.syzygy.MissingTableError, KeyError, IndexError):
        return None


def _tablebase_move(board: chess.Board) -> tuple[chess.Move, int, int] | None:
    if _TABLEBASE is None:
        return None
    if chess.popcount(board.occupied) > TABLEBASE_MAX_PIECES:
        return None
    if board.castling_rights:
        return None

    try:
        _TABLEBASE.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError, IndexError):
        return None

    candidates: list[tuple[chess.Move, int, int]] = []
    for move in board.legal_moves:
        board.push(move)
        try:
            value = _tablebase_child_value(board)
        finally:
            board.pop()
        if value is not None:
            child_wdl, child_dtz = value
            candidates.append((move, child_wdl, child_dtz))

    if not candidates:
        return None

    # Child values are from the opponent's perspective: lower WDL is better.
    best_child_wdl = min(item[1] for item in candidates)
    best = [item for item in candidates if item[1] == best_child_wdl]

    # For a winning child (negative DTZ for the opponent), the larger value is
    # closer to zero and normally makes faster progress. In a losing child,
    # larger positive DTZ delays the opponent's zeroing progress.
    return max(best, key=lambda item: item[2])


# -----------------------------------------------------------------------------
# Position keys and repetition bookkeeping
# -----------------------------------------------------------------------------


def _position_key(board: chess.Board) -> PositionKey:
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
    """Cheap process-local 64-bit mixer; no cryptographic hashing in search."""
    x = hash(key) & MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK64
    x ^= x >> 31
    return x & MASK64


def _initial_repetition_state(
    board: chess.Board,
) -> tuple[dict[PositionKey, int], int]:
    """Build exact counts plus a cheap TT-history signature once at the root."""
    previous = board.copy()
    keys = [_position_key(previous)]

    # Before the last zeroing move, an exact repetition cannot be reached again.
    steps = min(previous.halfmove_clock, len(previous.move_stack))
    for _ in range(steps):
        previous.pop()
        keys.append(_position_key(previous))

    counts: dict[PositionKey, int] = {}
    signature = 0
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
        signature = (signature + _position_fingerprint(key)) & MASK64
    return counts, signature


def _push_repetition(
    board: chess.Board,
    counts: dict[PositionKey, int],
    history: int,
) -> tuple[PositionKey, int, int]:
    """Update exact repetition count and return (key, new_count, child_history)."""
    key = _position_key(board)
    new_count = counts.get(key, 0) + 1
    counts[key] = new_count
    contribution = _position_fingerprint(key)
    child_history = contribution if board.halfmove_clock == 0 else (
        history + contribution
    ) & MASK64
    return key, new_count, child_history


def _pop_repetition(counts: dict[PositionKey, int], key: PositionKey) -> None:
    count = counts[key]
    if count == 1:
        del counts[key]
    else:
        counts[key] = count - 1


# -----------------------------------------------------------------------------
# Transposition table
# -----------------------------------------------------------------------------


def _score_to_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def _tt_find(key: PositionKey) -> TTEntry | None:
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


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def _pst_index(square: int, color: chess.Color) -> int:
    # First view the board from the piece's side, then convert a1-first python-
    # chess indexing to the rank8-first table layout above.
    relative = square if color == chess.WHITE else chess.square_mirror(square)
    return chess.square_mirror(relative)


def _relative_rank(square: int, color: chess.Color) -> int:
    return chess.square_rank(square) if color == chess.WHITE else 7 - chess.square_rank(square)


def _pawn_attacks_square(board: chess.Board, color: chess.Color, square: int) -> bool:
    return bool(board.attackers(color, square) & board.pieces(chess.PAWN, color))


def _is_passed_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
    enemy = not color
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    enemy_pawns = board.pieces(chess.PAWN, enemy)

    for enemy_sq in enemy_pawns:
        ef = chess.square_file(enemy_sq)
        if abs(ef - file_index) > 1:
            continue
        er = chess.square_rank(enemy_sq)
        if color == chess.WHITE:
            if er > rank_index:
                return False
        elif er < rank_index:
            return False
    return True


def _pawn_structure(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return 0, 0

    files = [0] * 8
    for square in pawns:
        files[chess.square_file(square)] += 1

    mg = 0
    eg = 0

    for count in files:
        if count > 1:
            extras = count - 1
            mg -= 12 * extras
            eg -= 18 * extras

    for square in pawns:
        file_index = chess.square_file(square)
        rel_rank = _relative_rank(square, color)

        left_empty = file_index == 0 or files[file_index - 1] == 0
        right_empty = file_index == 7 or files[file_index + 1] == 0
        if left_empty and right_empty:
            mg -= 11
            eg -= 9

        if not _is_passed_pawn(board, square, color):
            continue

        mg += PASSED_MG[rel_rank]
        eg += PASSED_EG[rel_rank]

        if _pawn_attacks_square(board, color, square):
            mg += 12 + 2 * rel_rank
            eg += 18 + 3 * rel_rank

        # Connected passers: adjacent-file friendly pawn no more than one rank away.
        for other in pawns:
            if other == square:
                continue
            if abs(chess.square_file(other) - file_index) == 1 and abs(
                _relative_rank(other, color) - rel_rank
            ) <= 1:
                mg += 8
                eg += 12
                break

        # In endings, king support/containment around advanced passers matters.
        own_king = board.king(color)
        enemy_king = board.king(not color)
        if own_king is not None and enemy_king is not None and rel_rank >= 4:
            promotion_square = chess.square(
                file_index, 7 if color == chess.WHITE else 0
            )
            own_dist = chess.square_distance(own_king, promotion_square)
            enemy_dist = chess.square_distance(enemy_king, promotion_square)
            eg += 4 * (enemy_dist - own_dist)

    return mg, eg


def _mobility(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    own = board.occupied_co[color]
    mg = 0
    eg = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        base = MOBILITY_BASE[piece_type]
        mg_unit = MOBILITY_MG[piece_type]
        eg_unit = MOBILITY_EG[piece_type]
        for square in board.pieces(piece_type, color):
            count = chess.popcount(board.attacks_mask(square) & ~own)
            mg += (count - base) * mg_unit
            eg += (count - base) * eg_unit
    return mg, eg


def _rook_features(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    mg = 0
    eg = 0
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for square in board.pieces(chess.ROOK, color):
        file_index = chess.square_file(square)
        file_mask = chess.BB_FILES[file_index]
        own_on_file = bool(own_pawns & file_mask)
        enemy_on_file = bool(enemy_pawns & file_mask)

        if not own_on_file:
            if not enemy_on_file:
                mg += 18
                eg += 12
            else:
                mg += 10
                eg += 7

        if _relative_rank(square, color) == 6:
            mg += 20
            eg += 30

    return mg, eg


def _knight_outposts(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    mg = 0
    eg = 0
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for square in board.pieces(chess.KNIGHT, color):
        rel_rank = _relative_rank(square, color)
        if rel_rank < 3 or rel_rank > 5:
            continue
        if not (board.attackers(color, square) & own_pawns):
            continue
        if board.attackers(not color, square) & enemy_pawns:
            continue
        mg += 18 + 4 * (rel_rank - 3)
        eg += 10
    return mg, eg


def _king_safety(board: chess.Board, color: chess.Color) -> int:
    """Return a middlegame king-safety term for one side (higher is safer)."""
    king = board.king(color)
    if king is None:
        return 0

    king_file = chess.square_file(king)
    king_rank = chess.square_rank(king)
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    score = 0

    # Pawn shelter on king and adjacent files. Reward a pawn one/two ranks in
    # front; penalize a missing shield and open files near the king.
    direction = 1 if color == chess.WHITE else -1
    for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        nearest = 99
        for square in own_pawns & chess.BB_FILES[file_index]:
            delta = (chess.square_rank(square) - king_rank) * direction
            if delta > 0:
                nearest = min(nearest, delta)

        if nearest == 1:
            score += 14
        elif nearest == 2:
            score += 8
        elif nearest == 3:
            score += 3
        else:
            score -= 16

        file_mask = chess.BB_FILES[file_index]
        if not (own_pawns & file_mask):
            score -= 10
            if not (enemy_pawns & file_mask):
                score -= 10

    # Enemy attacks intersecting the king zone. This is intentionally moderate
    # and scales down sharply when the enemy queen/heavy material is absent.
    zone = board.attacks_mask(king) | chess.BB_SQUARES[king]
    enemy = not color
    attack_units = 0
    attack_weight = {
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 5,
    }
    attackers = 0
    for piece_type, weight in attack_weight.items():
        for square in board.pieces(piece_type, enemy):
            hits = chess.popcount(board.attacks_mask(square) & zone)
            if hits:
                attackers += 1
                attack_units += weight * hits

    enemy_nonpawn = (
        320 * len(board.pieces(chess.KNIGHT, enemy))
        + 335 * len(board.pieces(chess.BISHOP, enemy))
        + 500 * len(board.pieces(chess.ROOK, enemy))
        + 900 * len(board.pieces(chess.QUEEN, enemy))
    )
    material_scale = min(100, max(20, enemy_nonpawn * 100 // 3200))
    if not board.pieces(chess.QUEEN, enemy):
        material_scale = material_scale * 45 // 100

    danger = attack_units * (6 + 2 * min(attackers, 4))
    score -= danger * material_scale // 100
    return score


def _mopup_eg(board: chess.Board) -> int:
    """White-minus-Black endgame conversion bonus in clearly winning endings."""
    material = {chess.WHITE: 0, chess.BLACK: 0}
    nonpawn = {chess.WHITE: 0, chess.BLACK: 0}
    for color in (chess.WHITE, chess.BLACK):
        for piece_type, value in EG_VALUE.items():
            count = len(board.pieces(piece_type, color))
            material[color] += count * value
            if piece_type != chess.PAWN:
                nonpawn[color] += count * value

    diff = material[chess.WHITE] - material[chess.BLACK]
    if abs(diff) < 500:
        return 0

    winner = chess.WHITE if diff > 0 else chess.BLACK
    loser = not winner
    # Only use mop-up when the defender is genuinely stripped down.
    if nonpawn[loser] > 335:
        return 0

    wk = board.king(winner)
    lk = board.king(loser)
    if wk is None or lk is None:
        return 0

    file_index = chess.square_file(lk)
    rank_index = chess.square_rank(lk)
    edge_distance = min(file_index, 7 - file_index, rank_index, 7 - rank_index)
    drive_to_edge = (3 - min(3, edge_distance)) * 12
    king_closeness = (7 - chess.square_distance(wk, lk)) * 5
    bonus = drive_to_edge + king_closeness
    return bonus if winner == chess.WHITE else -bonus


def _evaluate_white(board: chess.Board) -> int:
    mg = 0
    eg = 0
    phase = 0

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
                index = _pst_index(square, color)
                mg += sign * MG_PST[piece_type][index]
                eg += sign * EG_PST[piece_type][index]
                if piece_type != chess.KING:
                    mg += sign * MG_VALUE[piece_type]
                    eg += sign * EG_VALUE[piece_type]

            if piece_type in PHASE_WEIGHT:
                phase += PHASE_WEIGHT[piece_type] * len(
                    board.pieces(piece_type, color)
                )

        if len(board.pieces(chess.BISHOP, color)) >= 2:
            mg += sign * 28
            eg += sign * 42

        pawn_mg, pawn_eg = _pawn_structure(board, color)
        mg += sign * pawn_mg
        eg += sign * pawn_eg

        mob_mg, mob_eg = _mobility(board, color)
        mg += sign * mob_mg
        eg += sign * mob_eg

        rook_mg, rook_eg = _rook_features(board, color)
        mg += sign * rook_mg
        eg += sign * rook_eg

        outpost_mg, outpost_eg = _knight_outposts(board, color)
        mg += sign * outpost_mg
        eg += sign * outpost_eg

        mg += sign * _king_safety(board, color)

    eg += _mopup_eg(board)

    phase = min(MAX_PHASE, phase)
    return (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE


def _evaluate(board: chess.Board) -> int:
    score = _evaluate_white(board)
    return score if board.turn == chess.WHITE else -score


# -----------------------------------------------------------------------------
# Move ordering and search heuristics
# -----------------------------------------------------------------------------


def _tick(ply: int, qnode: bool = False) -> None:
    global _NODES, _QNODES, _SELDEPTH
    _NODES += 1
    if qnode:
        _QNODES += 1
    if ply > _SELDEPTH:
        _SELDEPTH = ply
    if (_NODES & 63) == 0 and time.perf_counter() >= _DEADLINE:
        raise SearchTimeout


def _is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and move.promotion is None


def _capture_score(board: chess.Board, move: chess.Move) -> int:
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
    return 10 * ORDER_VALUE[victim] - ORDER_VALUE[attacker]


def _history_score(turn: chess.Color, move: chess.Move) -> int:
    return _HISTORY[int(turn)][move.from_square][move.to_square]


def _move_order_score(
    board: chess.Board,
    move: chess.Move,
    preferred: chess.Move | None,
    ply: int,
) -> int:
    if preferred is not None and move == preferred:
        return 10_000_000

    if move.promotion is not None:
        return 8_000_000 + ORDER_VALUE.get(move.promotion, 0)

    if board.is_capture(move):
        capture = _capture_score(board, move)
        attacker = board.piece_type_at(move.from_square)
        victim = (
            chess.PAWN
            if board.is_en_passant(move)
            else board.piece_type_at(move.to_square)
        )
        # Without SEE yet, obvious equal/favourable captures stay ahead of quiets;
        # suspicious high-attacker/low-victim captures fall behind good quiets.
        if attacker is not None and victim is not None and ORDER_VALUE[victim] >= ORDER_VALUE[attacker]:
            return 7_000_000 + capture
        return 1_000_000 + capture

    if ply < MAX_PLY:
        killer0, killer1 = _KILLERS[ply]
        if move == killer0:
            return 6_000_000
        if move == killer1:
            return 5_900_000

    return 2_000_000 + _history_score(board.turn, move)


def _ordered_moves(
    board: chess.Board,
    moves: list[chess.Move],
    preferred: chess.Move | None = None,
    ply: int = 0,
) -> list[chess.Move]:
    return sorted(
        moves,
        key=lambda move: _move_order_score(board, move, preferred, ply),
        reverse=True,
    )


def _record_quiet_cutoff(
    turn: chess.Color,
    move: chess.Move,
    depth: int,
    ply: int,
) -> None:
    if ply < MAX_PLY:
        if _KILLERS[ply][0] != move:
            _KILLERS[ply][1] = _KILLERS[ply][0]
            _KILLERS[ply][0] = move

    bonus = min(2048, 16 * depth * depth)
    value = _HISTORY[int(turn)][move.from_square][move.to_square]
    _HISTORY[int(turn)][move.from_square][move.to_square] = min(
        1_000_000, value + bonus
    )


def _age_history() -> None:
    for color in range(2):
        for from_square in range(64):
            row = _HISTORY[color][from_square]
            for to_square in range(64):
                row[to_square] = row[to_square] * 7 // 8


# -----------------------------------------------------------------------------
# Quiescence and PVS/LMR negamax
# -----------------------------------------------------------------------------


def _quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    ply: int,
    rep_counts: dict[PositionKey, int],
    qply: int = 0,
) -> int:
    _tick(ply, qnode=True)

    in_check = board.is_check()
    moves = list(board.legal_moves)
    if not moves:
        return -MATE_SCORE + ply if in_check else 0

    key = _position_key(board)
    if rep_counts.get(key, 0) >= 3 or board.halfmove_clock >= 100:
        return 0
    if board.is_insufficient_material():
        return 0

    if qply >= MAX_QUIESCENCE_PLY:
        return _evaluate(board)

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

    for move in _ordered_moves(board, moves, ply=ply):
        board.push(move)
        child_key, child_count, _ = _push_repetition(board, rep_counts, 0)
        try:
            if child_count >= 3:
                score = 0
            else:
                score = -_quiescence(
                    board,
                    -beta,
                    -alpha,
                    ply + 1,
                    rep_counts,
                    qply + 1,
                )
        finally:
            _pop_repetition(rep_counts, child_key)
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
    history: int,
    rep_counts: dict[PositionKey, int],
) -> int:
    global _PVS_RESEARCHES, _LMR_REDUCTIONS, _LMR_RESEARCHES

    if depth <= 0:
        return _quiescence(board, alpha, beta, ply, rep_counts)

    _tick(ply)

    in_check = board.is_check()
    moves = list(board.legal_moves)
    if not moves:
        return -MATE_SCORE + ply if in_check else 0

    key = _position_key(board)
    if rep_counts.get(key, 0) >= 3 or board.halfmove_clock >= 100:
        return 0
    if board.is_insufficient_material():
        return 0

    entry = _tt_find(key)
    preferred: chess.Move | None = None
    if entry is not None and entry.move in moves:
        preferred = entry.move
        cached = _tt_cutoff(entry, depth, alpha, beta, ply, history)
        if cached is not None:
            return cached

    original_alpha = alpha
    original_beta = beta
    ordered = _ordered_moves(board, moves, preferred, ply)
    best = -INFINITY
    best_move = ordered[0]

    for move_index, move in enumerate(ordered):
        turn = board.turn
        quiet = _is_quiet(board, move)
        child_depth = depth - 1  # IMPORTANT: the normal decrement happens exactly once.

        board.push(move)
        child_key, child_count, child_history = _push_repetition(
            board, rep_counts, history
        )
        gives_check = board.is_check()

        try:
            if child_count >= 3:
                score = 0
            elif move_index == 0:
                score = -_negamax(
                    board,
                    child_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    child_history,
                    rep_counts,
                )
            else:
                reduce = (
                    depth >= 4
                    and child_depth >= 1
                    and move_index >= 4
                    and quiet
                    and not in_check
                    and not gives_check
                )

                if reduce:
                    _LMR_REDUCTIONS += 1
                    reduced_depth = max(0, child_depth - 1)
                    score = -_negamax(
                        board,
                        reduced_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        child_history,
                        rep_counts,
                    )
                    if score > alpha:
                        _LMR_RESEARCHES += 1
                        score = -_negamax(
                            board,
                            child_depth,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            child_history,
                            rep_counts,
                        )
                else:
                    score = -_negamax(
                        board,
                        child_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        child_history,
                        rep_counts,
                    )

                if score > alpha and score < beta:
                    _PVS_RESEARCHES += 1
                    score = -_negamax(
                        board,
                        child_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        child_history,
                        rep_counts,
                    )
        finally:
            _pop_repetition(rep_counts, child_key)
            board.pop()

        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if quiet:
                _record_quiet_cutoff(turn, move, depth, ply)
            break

    if best <= original_alpha:
        bound = TT_UPPER
    elif best >= original_beta:
        bound = TT_LOWER
    else:
        bound = TT_EXACT

    _tt_store(key, depth, best, bound, best_move, history, ply)
    return best


def _search_root(
    board: chess.Board,
    depth: int,
    preferred: chess.Move | None,
    alpha: int,
    beta: int,
    history: int,
    rep_counts: dict[PositionKey, int],
) -> tuple[int, chess.Move]:
    global _PVS_RESEARCHES

    _tick(0)
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("get_move called with no legal moves")

    key = _position_key(board)
    entry = _tt_find(key)
    if entry is not None and entry.move in moves:
        preferred = entry.move
        cached = _tt_cutoff(entry, depth, alpha, beta, 0, history)
        if cached is not None and rep_counts.get(key, 0) < 3:
            return cached, entry.move

    original_alpha = alpha
    original_beta = beta
    ordered = _ordered_moves(board, moves, preferred, 0)
    best_move = ordered[0]
    best_score = -INFINITY

    for move_index, move in enumerate(ordered):
        _tick(0)
        board.push(move)
        child_key, child_count, child_history = _push_repetition(
            board, rep_counts, history
        )
        try:
            if child_count >= 3:
                score = 0
            elif move_index == 0:
                score = -_negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    child_history,
                    rep_counts,
                )
            else:
                score = -_negamax(
                    board,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    1,
                    child_history,
                    rep_counts,
                )
                if score > alpha and score < beta:
                    _PVS_RESEARCHES += 1
                    score = -_negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        child_history,
                        rep_counts,
                    )
        finally:
            _pop_repetition(rep_counts, child_key)
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    if best_score <= original_alpha:
        bound = TT_UPPER
    elif best_score >= original_beta:
        bound = TT_LOWER
    else:
        bound = TT_EXACT
    _tt_store(key, depth, best_score, bound, best_move, history, 0)
    return best_score, best_move


# -----------------------------------------------------------------------------
# Time management and public entry point
# -----------------------------------------------------------------------------


def _move_budget_ms(time_left_ms: int) -> int:
    if time_left_ms <= 100:
        return max(1, time_left_ms // 4)

    reserve_ms = max(60, min(1_200, time_left_ms // 18))
    usable_ms = max(1, time_left_ms - reserve_ms)

    # Spend a little more while the clock is healthy, but keep a hard cap.
    desired_ms = time_left_ms // 32 + 180
    if time_left_ms < 5_000:
        desired_ms = min(desired_ms, 250)
    elif time_left_ms < 15_000:
        desired_ms = min(desired_ms, 700)

    return max(1, min(3_500, usable_ms, desired_ms))


def _clear_killers() -> None:
    for ply in range(MAX_PLY):
        _KILLERS[ply][0] = None
        _KILLERS[ply][1] = None


def get_move(fen: str, time_left_ms: int) -> str:
    global _DEADLINE, _NODES, _QNODES, _SELDEPTH
    global _TT_GENERATION, _TT_HITS, _TT_CUTOFFS
    global _PVS_RESEARCHES, _LMR_REDUCTIONS, _LMR_RESEARCHES

    board = _board_with_history(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("get_move called with no legal moves")

    if len(legal_moves) == 1:
        return _remember_move(board, legal_moves[0])

    tablebase_result = _tablebase_move(board)
    if tablebase_result is not None:
        tablebase_move, child_wdl, child_dtz = tablebase_result
        print(
            f"tb_hit=1 child_wdl={child_wdl} child_dtz={child_dtz} "
            f"move={tablebase_move.uci()}"
        )
        return _remember_move(board, tablebase_move)

    if board.fullmove_number <= BOOK_MAX_FULLMOVE:
        book_move = _book_move(board)
        if book_move is not None:
            print(f"book_hit=1 move={book_move.uci()}")
            return _remember_move(board, book_move)
        print("book_hit=0")
    else:
        print("book_skip=move_limit")

    # The platform times the entire get_move call. Start our search budget before
    # heuristic aging/history reconstruction so small Python overhead cannot
    # silently push us past the intended deadline.
    budget_ms = _move_budget_ms(time_left_ms)
    start = time.perf_counter()
    _DEADLINE = start + budget_ms / 1_000.0

    _TT_GENERATION += 1
    _TT_HITS = 0
    _TT_CUTOFFS = 0
    _PVS_RESEARCHES = 0
    _LMR_REDUCTIONS = 0
    _LMR_RESEARCHES = 0
    _NODES = 0
    _QNODES = 0
    _SELDEPTH = 0
    _clear_killers()
    _age_history()

    rep_counts, history = _initial_repetition_state(board)

    best_move = _ordered_moves(board, legal_moves, ply=0)[0]
    best_score = 0
    completed_depth = 0

    for depth in range(1, MAX_DEPTH + 1):
        try:
            if depth == 1 or completed_depth == 0:
                score, move = _search_root(
                    board,
                    depth,
                    best_move,
                    -INFINITY,
                    INFINITY,
                    history,
                    rep_counts,
                )
            else:
                window = 35
                alpha = max(-INFINITY, best_score - window)
                beta = min(INFINITY, best_score + window)

                while True:
                    score, move = _search_root(
                        board,
                        depth,
                        best_move,
                        alpha,
                        beta,
                        history,
                        rep_counts,
                    )
                    if score <= alpha:
                        window *= 2
                        alpha = max(-INFINITY, best_score - window)
                        beta = min(INFINITY, best_score + window)
                        continue
                    if score >= beta:
                        window *= 2
                        alpha = max(-INFINITY, best_score - window)
                        beta = min(INFINITY, best_score + window)
                        continue
                    break
        except SearchTimeout:
            break

        best_score = score
        best_move = move
        completed_depth = depth

        if abs(best_score) >= MATE_SCORE - 100:
            break

    elapsed_ms = max(1, int((time.perf_counter() - start) * 1000))
    nps = _NODES * 1000 // elapsed_ms
    print(
        f"depth={completed_depth} seldepth={_SELDEPTH} score={best_score} "
        f"nodes={_NODES} qnodes={_QNODES} time_ms={elapsed_ms} nps={nps} "
        f"budget_ms={budget_ms} tt_hits={_TT_HITS} tt_cutoffs={_TT_CUTOFFS} "
        f"pvs_researches={_PVS_RESEARCHES} lmr_reductions={_LMR_REDUCTIONS} "
        f"lmr_researches={_LMR_RESEARCHES} move={best_move.uci()}"
    )
    return _remember_move(board, best_move)
