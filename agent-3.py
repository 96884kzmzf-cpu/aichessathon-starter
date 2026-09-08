import math
import time
from dataclasses import dataclass
from functools import lru_cache
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
# Experimental pawn-threat extension is off: it lost too much depth in benchmarks.
USE_THREAT_QUIESCENCE = False
USE_LMR = True
USE_POSITIONAL_UPDATES = True
USE_SEE_ORDERING = True
QUIET_THREAT_PLIES = 2
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
    _BOOK: chess.polyglot.MemoryMappedReader | None = chess.polyglot.open_reader(_BOOK_PATH)
except (FileNotFoundError, OSError):
    _BOOK = None

TABLEBASE_MAX_PIECES = 3
_SYZYGY_PATH = Path(__file__).resolve().with_name("syzygy")

try:
    _TABLEBASE: chess.syzygy.Tablebase | None = (
        chess.syzygy.open_tablebase(str(_SYZYGY_PATH)) if _SYZYGY_PATH.is_dir() else None
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
            value = 2 * centre
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

type PositionKey = tuple[int, int, int, int, int, int, int, int, bool, int, int | None]


@dataclass(frozen=True, slots=True)
class TTEntry:
    key: PositionKey
    depth: int
    score: int
    bound: int
    move: chess.Move
    history: int
    generation: int
    halfmove_clock: int


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
_QTHREAT_NODES = 0
_TT_PROBES = 0
_PV_REUSES = 0
_ASPIRATION_RETRIES = 0
# Ordering hints from the last completed principal variation survive TT collisions
# and predicted opponent replies. Only the TT may supply a cached score.
_PV_MOVES: dict[PositionKey, chess.Move] = {}
_PV_LINE: list[chess.Move] = []

_KILLERS: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
_HISTORY: list[list[list[int]]] = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]


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
    child_history = contribution if board.halfmove_clock == 0 else (history + contribution) & MASK64
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
    global _TT_HITS, _TT_PROBES
    _TT_PROBES += 1
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
    halfmove_clock: int,
) -> int | None:
    global _TT_CUTOFFS
    if entry.depth < depth or entry.history != history or entry.halfmove_clock != halfmove_clock:
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
    halfmove_clock: int,
) -> None:
    index = hash(key) & (TT_SIZE - 1)
    previous = _TT[index]

    if previous is not None and previous.generation == _TT_GENERATION:
        if previous.depth > depth:
            return
        if (
            previous.key == key
            and previous.history == history
            and previous.halfmove_clock == halfmove_clock
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
        halfmove_clock=halfmove_clock,
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def _pst_index(square: int, color: chess.Color) -> int:
    return square ^ 56 if color == chess.WHITE else square


def _relative_rank(square: int, color: chess.Color) -> int:
    return (square >> 3) if color == chess.WHITE else 7 - (square >> 3)


# Precomputed geometry, independent of the changing position.
_ADJACENT_FILES = tuple(
    (chess.BB_FILES[f - 1] if f else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0) for f in range(8)
)
_PASSED_MASK = tuple(
    tuple(
        (_ADJACENT_FILES[sq & 7] | chess.BB_FILES[sq & 7])
        & sum(chess.BB_RANKS[r] for r in range(8) if (r > (sq >> 3) if color else r < (sq >> 3)))
        for sq in range(64)
    )
    for color in (chess.BLACK, chess.WHITE)
)
_CONNECTED_MASK = tuple(
    _ADJACENT_FILES[sq & 7]
    & sum(chess.BB_RANKS[r] for r in range(max(0, (sq >> 3) - 1), min(8, (sq >> 3) + 2)))
    for sq in range(64)
)
_ATTACK_WEIGHT = {chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5}


def _pawn_attacks(pawns: int, color: chess.Color) -> int:
    if color:
        return (
            ((pawns & ~chess.BB_FILE_A) << 7) | ((pawns & ~chess.BB_FILE_H) << 9)
        ) & chess.BB_ALL
    return ((pawns & ~chess.BB_FILE_H) >> 7) | ((pawns & ~chess.BB_FILE_A) >> 9)


@lru_cache(maxsize=32_768)
def _pawn_terms(white: int, black: int) -> tuple[int, int, int, int]:
    """Cache pawn-only scores and advanced passers; kings never enter this cache."""
    mg = eg = 0
    advanced = [0, 0]
    for color, pawns, enemy in ((chess.WHITE, white, black), (chess.BLACK, black, white)):
        sign = 1 if color else -1
        defended = _pawn_attacks(pawns, color)
        for file_mask in chess.BB_FILES:
            extras = max(0, (pawns & file_mask).bit_count() - 1)
            mg -= sign * 12 * extras
            eg -= sign * 18 * extras
        for sq in chess.scan_forward(pawns):
            rank = _relative_rank(sq, color)
            if not pawns & _ADJACENT_FILES[sq & 7]:
                mg -= sign * 11
                eg -= sign * 9
            if enemy & _PASSED_MASK[color][sq]:
                continue
            mg += sign * PASSED_MG[rank]
            eg += sign * PASSED_EG[rank]
            if defended & chess.BB_SQUARES[sq]:
                mg += sign * (12 + 2 * rank)
                eg += sign * (18 + 3 * rank)
            if pawns & _CONNECTED_MASK[sq]:
                mg += sign * 8
                eg += sign * 12
            if rank >= 4:
                advanced[color] |= chess.BB_SQUARES[sq]
    return mg, eg, advanced[chess.WHITE], advanced[chess.BLACK]


@lru_cache(maxsize=32_768)
def _shelter(own_pawns: int, enemy_pawns: int, king: int, color: chess.Color) -> int:
    score = 0
    direction = 1 if color else -1
    for f in range(max(0, (king & 7) - 1), min(8, (king & 7) + 2)):
        on_file = own_pawns & chess.BB_FILES[f]
        nearest = 99
        for sq in chess.scan_forward(on_file):
            delta = ((sq >> 3) - (king >> 3)) * direction
            if delta > 0:
                nearest = min(nearest, delta)
        score += {1: 14, 2: 8, 3: 3}.get(nearest, -16)
        if not on_file:
            score -= 10
            if not enemy_pawns & chess.BB_FILES[f]:
                score -= 10
    return score


def _future_shelter_penalty(board: chess.Board, color: chess.Color) -> int:
    """Small cost for damaging still-available castling shelters before castling."""
    if not board.castling_rights & board.occupied_co[color]:
        return 0
    own = board.pawns & board.occupied_co[color]
    enemy = board.pawns & board.occupied_co[not color]
    rank = 0 if color else 7
    cost = weight = 0
    for allowed, file_index, importance in (
        (board.has_kingside_castling_rights(color), 6, 2),
        (board.has_queenside_castling_rights(color), 2, 1),
    ):
        if allowed:
            # A shelter of three pawns one rank ahead scores 42.
            cost += importance * max(
                0, 42 - _shelter(own, enemy, chess.square(file_index, rank), color)
            )
            weight += importance
    return cost // (2 * weight) if weight else 0


def _absolute_pin_masks(board: chess.Board, color: chess.Color) -> dict[int, int]:
    """Find each side's pinned pieces once, sharing the result across mobility."""
    king = board.king(color)
    if king is None:
        return {}
    orthogonal = chess.BB_FILES[king & 7] | chess.BB_RANKS[king >> 3]
    diagonal = chess.BB_DIAG_ATTACKS[king][0]
    snipers = (
        (orthogonal & (board.rooks | board.queens)) | (diagonal & (board.bishops | board.queens))
    ) & board.occupied_co[not color]
    pins: dict[int, int] = {}
    for sniper in chess.scan_forward(snipers):
        blockers = chess.between(king, sniper) & board.occupied
        if blockers & board.occupied_co[color] and blockers.bit_count() == 1:
            pins[blockers.bit_length() - 1] = chess.ray(king, sniper)
    return pins


def _evaluate_white(board: chess.Board) -> int:
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    pawns = [board.pawns & black, board.pawns & white]
    pawn_attacks = [_pawn_attacks(pawns[c], bool(c)) for c in range(2)]
    kings = [board.king(chess.BLACK), board.king(chess.WHITE)]
    mg, eg, white_passers, black_passers = _pawn_terms(pawns[1], pawns[0])
    phase = 0
    material = [0, 0]
    nonpawn_eg = [0, 0]
    nonpawn_mg = [0, 0]
    attack_units = [0, 0]
    attackers = [0, 0]
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        own = board.occupied_co[color]
        pins = _absolute_pin_masks(board, color) if USE_POSITIONAL_UPDATES else {}
        enemy_king = kings[not color]
        zone = (
            0
            if enemy_king is None
            else chess.BB_KING_ATTACKS[enemy_king] | chess.BB_SQUARES[enemy_king]
        )
        for piece_type in chess.PIECE_TYPES:
            pieces = board.pieces_mask(piece_type, color)
            count = pieces.bit_count()
            if piece_type != chess.KING:
                mg += sign * MG_VALUE[piece_type] * count
                eg += sign * EG_VALUE[piece_type] * count
                material[color] += EG_VALUE[piece_type] * count
            if piece_type in PHASE_WEIGHT:
                phase += PHASE_WEIGHT[piece_type] * count
                nonpawn_mg[color] += MG_VALUE[piece_type] * count
                nonpawn_eg[color] += EG_VALUE[piece_type] * count
            if piece_type == chess.BISHOP and count >= 2:
                mg += sign * 28
                eg += sign * 42
            for sq in chess.scan_forward(pieces):
                index = _pst_index(sq, color)
                mg += sign * MG_PST[piece_type][index]
                eg += sign * EG_PST[piece_type][index]
                if piece_type in MOBILITY_BASE:
                    attacks = board.attacks_mask(sq)
                    mobile = attacks & ~own
                    if USE_POSITIONAL_UPDATES:
                        mobile &= pins.get(sq, chess.BB_ALL)
                    mobility = mobile.bit_count() - MOBILITY_BASE[piece_type]
                    mg += sign * mobility * MOBILITY_MG[piece_type]
                    eg += sign * mobility * MOBILITY_EG[piece_type]
                    # Keep rule-defined attacks for king zones: even pinned pieces
                    # attack squares for king legality. Mobility is treated separately.
                    hits = (attacks & zone).bit_count()
                    if hits:
                        attackers[color] += 1
                        attack_units[color] += _ATTACK_WEIGHT[piece_type] * hits
                if piece_type == chess.ROOK:
                    file_mask = chess.BB_FILES[sq & 7]
                    if not pawns[color] & file_mask:
                        open_file = not pawns[not color] & file_mask
                        mg += sign * (18 if open_file else 10)
                        eg += sign * (12 if open_file else 7)
                    if _relative_rank(sq, color) == 6:
                        mg += sign * 20
                        eg += sign * 30
                elif piece_type == chess.KNIGHT:
                    rank = _relative_rank(sq, color)
                    if (
                        3 <= rank <= 5
                        and pawn_attacks[color] & chess.BB_SQUARES[sq]
                        and not pawn_attacks[not color] & chess.BB_SQUARES[sq]
                    ):
                        mg += sign * (18 + 4 * (rank - 3))
                        eg += sign * 10
        king = kings[color]
        if king is not None and enemy_king is not None:
            passers = white_passers if color else black_passers
            for sq in chess.scan_forward(passers):
                promotion = (sq & 7) + (56 if color else 0)
                eg += (
                    sign
                    * 4
                    * (
                        chess.square_distance(enemy_king, promotion)
                        - chess.square_distance(king, promotion)
                    )
                )

    for color in (chess.WHITE, chess.BLACK):
        king = kings[color]
        if king is None:
            continue
        sign = 1 if color else -1
        enemy = not color
        scale = min(100, max(20, nonpawn_mg[enemy] * 100 // 3200))
        if not board.queens & board.occupied_co[enemy]:
            scale = scale * 45 // 100
        safety = _shelter(pawns[color], pawns[enemy], king, color)
        danger = attack_units[enemy] * (6 + 2 * min(attackers[enemy], 4))
        safety -= danger * scale // 100
        if USE_POSITIONAL_UPDATES:
            safety -= _future_shelter_penalty(board, color) * scale // 100
        mg += sign * safety

    diff = material[1] - material[0]
    if abs(diff) >= 500:
        winner = diff > 0
        loser = not winner
        wk, lk = kings[winner], kings[loser]
        if nonpawn_eg[loser] <= 335 and wk is not None and lk is not None:
            f, r = lk & 7, lk >> 3
            edge = min(f, 7 - f, r, 7 - r)
            bonus = (3 - min(3, edge)) * 12 + (7 - chess.square_distance(wk, lk)) * 5
            eg += bonus if winner else -bonus
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
    if (_NODES & 15) == 0 and time.perf_counter() >= _DEADLINE:
        raise SearchTimeout


def _is_quiet(board: chess.Board, move: chess.Move) -> bool:
    return not board.is_capture(move) and move.promotion is None


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        return 0
    victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    if victim is None:
        return 0
    return 10 * ORDER_VALUE[victim] - ORDER_VALUE[attacker]


def _see(board: chess.Board, move: chess.Move) -> int:
    """Legal least-valuable-attacker exchange estimate, used only for ordering.

    Recaptures include x-rays, absolute pins, king safety, EP and promotions.
    Choosing the cheapest attacker is an approximation, never a pruning proof.
    """
    target = chess.BB_SQUARES[move.to_square]
    gains: list[int] = []
    pushed = 0
    current = move
    try:
        while True:
            victim = (
                chess.PAWN
                if board.is_en_passant(current)
                else board.piece_type_at(current.to_square)
            )
            gain = ORDER_VALUE.get(victim or 0, 0)
            if current.promotion:
                gain += ORDER_VALUE[current.promotion] - ORDER_VALUE[chess.PAWN]
            gains.append(gain)
            board.push(current)
            pushed += 1
            if not board.attackers_mask(board.turn, move.to_square):
                break
            captures = board.generate_legal_captures(to_mask=target)
            reply = min(
                captures,
                key=lambda m: ORDER_VALUE[board.piece_type_at(m.from_square) or chess.KING],
                default=None,
            )
            if reply is None:
                break
            current = reply
    finally:
        for _ in range(pushed):
            board.pop()
    response = 0
    for gain in reversed(gains[1:]):
        response = max(0, gain - response)
    return gains[0] - response


def _history_score(turn: chess.Color, move: chess.Move) -> int:
    return _HISTORY[int(turn)][move.from_square][move.to_square]


def _move_order_score(
    board: chess.Board,
    move: chess.Move,
    preferred: chess.Move | None,
    ply: int,
    use_see: bool = True,
) -> int:
    if preferred is not None and move == preferred:
        return 10_000_000

    if move.promotion is not None:
        return 8_000_000 + ORDER_VALUE.get(move.promotion, 0)

    if board.is_capture(move):
        capture = _capture_score(board, move)
        attacker = board.piece_type_at(move.from_square)
        victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        if (
            attacker is not None
            and victim is not None
            and ORDER_VALUE[victim] >= ORDER_VALUE[attacker]
        ):
            return 7_000_000 + capture
        if USE_SEE_ORDERING and use_see and _see(board, move) >= 0:
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
    use_see: bool = True,
) -> list[chess.Move]:
    return sorted(
        moves,
        key=lambda move: _move_order_score(board, move, preferred, ply, use_see),
        reverse=True,
    )


def _record_quiet_cutoff(
    turn: chess.Color,
    move: chess.Move,
    depth: int,
    ply: int,
) -> None:
    if ply < MAX_PLY and _KILLERS[ply][0] != move:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move

    bonus = min(2048, 16 * depth * depth)
    value = _HISTORY[int(turn)][move.from_square][move.to_square]
    _HISTORY[int(turn)][move.from_square][move.to_square] = min(1_000_000, value + bonus)


def _age_history() -> None:
    for color in range(2):
        for from_square in range(64):
            row = _HISTORY[color][from_square]
            for to_square in range(64):
                row[to_square] = row[to_square] * 7 // 8


# -----------------------------------------------------------------------------
# Quiescence and PVS/LMR negamax
# -----------------------------------------------------------------------------


def _piece_under_pawn_attack(board: chess.Board) -> bool:
    """Recognise legal pawn threats to a minor/major piece, including relative pins."""
    enemy = not board.turn
    targets = (board.knights | board.bishops | board.rooks | board.queens) & board.occupied_co[
        board.turn
    ]
    enemy_pawns = board.pawns & board.occupied_co[enemy]
    for sq in chess.scan_forward(targets & _pawn_attacks(enemy_pawns, enemy)):
        for pawn in chess.scan_forward(chess.BB_PAWN_ATTACKS[not enemy][sq] & enemy_pawns):
            if board.pin_mask(enemy, pawn) & chess.BB_SQUARES[sq]:
                return True
    return False


def _quiet_pawn_threats(board: chess.Board) -> list[chess.Move]:
    enemy_pieces = (board.knights | board.bishops | board.rooks | board.queens) & board.occupied_co[
        not board.turn
    ]
    if not enemy_pieces:
        return []
    return [
        move
        for move in board.generate_legal_moves(from_mask=board.pawns, to_mask=~board.occupied)
        if move.promotion is None
        and not board.is_en_passant(move)
        and chess.BB_PAWN_ATTACKS[board.turn][move.to_square] & enemy_pieces
    ]


def _quiescence(
    board: chess.Board,
    alpha: int,
    beta: int,
    ply: int,
    rep_counts: dict[PositionKey, int],
    qply: int = 0,
    threat_budget: int = QUIET_THREAT_PLIES,
) -> int:
    global _QTHREAT_NODES
    _tick(ply, qnode=True)
    in_check = board.is_check()
    # Early terminal detection without materializing every quiet legal move.
    if not any(board.generate_legal_moves()):
        return -MATE_SCORE + ply if in_check else 0
    key = _position_key(board)
    if rep_counts.get(key, 0) >= 3 or board.halfmove_clock >= 100:
        return 0
    if board.is_insufficient_material():
        return 0
    if qply >= MAX_QUIESCENCE_PLY or ply >= MAX_PLY - 1:
        raise SearchTimeout

    threatened = (
        USE_THREAT_QUIESCENCE
        and threat_budget > 0
        and not in_check
        and _piece_under_pawn_attack(board)
    )
    if in_check or threatened:
        # Search every legal defence: moving the attacked piece alone would omit
        # counterattacks, interpositions, queen moves and exchange combinations.
        best = -INFINITY
        moves = list(board.generate_legal_moves())
        if threatened:
            _QTHREAT_NODES += 1
    else:
        best = _evaluate(board)
        if best >= beta:
            return best
        alpha = max(alpha, best)
        moves = list(board.generate_legal_captures())
        # Non-capturing underpromotions must be included, too.
        promotion_rank = chess.BB_RANK_7 if board.turn else chess.BB_RANK_2
        moves.extend(
            board.generate_legal_moves(
                from_mask=board.pawns & promotion_rank, to_mask=~board.occupied
            )
        )
        if USE_THREAT_QUIESCENCE and qply == 0 and threat_budget >= 2:
            moves.extend(_quiet_pawn_threats(board))

    for move in _ordered_moves(board, moves, ply=ply, use_see=False):
        quiet = _is_quiet(board, move)
        child_budget = max(0, threat_budget - int(threatened or (quiet and not in_check)))
        board.push(move)
        child_key = _position_key(board)
        rep_counts[child_key] = rep_counts.get(child_key, 0) + 1
        try:
            score = -_quiescence(board, -beta, -alpha, ply + 1, rep_counts, qply + 1, child_budget)
        finally:
            _pop_repetition(rep_counts, child_key)
            board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
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
    global _PVS_RESEARCHES, _LMR_REDUCTIONS, _LMR_RESEARCHES, _PV_REUSES

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
    preferred = _PV_MOVES.get(key)
    if preferred in moves:
        _PV_REUSES += 1
    if entry is not None and entry.move in moves:
        preferred = entry.move
        cached = _tt_cutoff(entry, depth, alpha, beta, ply, history, board.halfmove_clock)
        if cached is not None:
            return cached

    original_alpha = alpha
    original_beta = beta
    pv_node = beta - alpha > 1
    ordered = _ordered_moves(board, moves, preferred, ply)
    best = -INFINITY
    best_move = ordered[0]

    for move_index, move in enumerate(ordered):
        turn = board.turn
        quiet = _is_quiet(board, move)
        child_depth = depth - 1  # IMPORTANT: the normal decrement happens exactly once.
        reduce_candidate = (
            USE_LMR
            and depth >= 5
            and move_index >= 6
            and quiet
            and not in_check
            and not pv_node
            and _history_score(turn, move) < 256
            and (ply >= MAX_PLY or move not in _KILLERS[ply])
        )

        board.push(move)
        child_key, child_count, child_history = _push_repetition(board, rep_counts, history)

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
                reduce = reduce_candidate and not board.is_check()

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

    _tt_store(key, depth, best, bound, best_move, history, ply, board.halfmove_clock)
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
        cached = _tt_cutoff(entry, depth, alpha, beta, 0, history, board.halfmove_clock)
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
        child_key, child_count, child_history = _push_repetition(board, rep_counts, history)
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
    _tt_store(key, depth, best_score, bound, best_move, history, 0, board.halfmove_clock)
    return best_score, best_move


def _save_pv(board: chess.Board, best_move: chess.Move, depth: int) -> None:
    """Keep legal ordering hints from a completed iteration, never cached scores."""
    global _PV_MOVES, _PV_LINE
    position = board.copy(stack=False)
    hints: dict[PositionKey, chess.Move] = {}
    line: list[chess.Move] = []
    move = best_move
    for _ in range(depth):
        key = _position_key(position)
        if key in hints or not position.is_legal(move):
            break
        hints[key] = move
        line.append(move)
        position.push(move)
        child_key = _position_key(position)
        entry = _TT[hash(child_key) & (TT_SIZE - 1)]
        if entry is None or entry.key != child_key:
            break
        move = entry.move
    _PV_MOVES = hints
    _PV_LINE = line


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
    global _QTHREAT_NODES, _TT_PROBES, _PV_REUSES, _ASPIRATION_RETRIES
    global _PV_LINE

    start = time.perf_counter()
    budget_ms = _move_budget_ms(time_left_ms)
    _DEADLINE = start + budget_ms / 1_000.0
    board = _board_with_history(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("get_move called with no legal moves")

    if len(legal_moves) == 1:
        return _remember_move(board, legal_moves[0])

    # Skip optional work when only a few milliseconds remain. A legal fallback
    # must be available before book probes, ordering or heuristic maintenance.
    if time_left_ms <= 20 or time.perf_counter() >= _DEADLINE:
        return _remember_move(board, legal_moves[0])

    tablebase_result = _tablebase_move(board)
    if tablebase_result is not None:
        tablebase_move, child_wdl, child_dtz = tablebase_result
        print(f"tb_hit=1 child_wdl={child_wdl} child_dtz={child_dtz} move={tablebase_move.uci()}")
        return _remember_move(board, tablebase_move)

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
    _PVS_RESEARCHES = 0
    _LMR_REDUCTIONS = 0
    _LMR_RESEARCHES = 0
    _QTHREAT_NODES = 0
    _TT_PROBES = 0
    _PV_REUSES = 0
    _ASPIRATION_RETRIES = 0
    _PV_LINE = []
    _NODES = 0
    _QNODES = 0
    _SELDEPTH = 0
    _clear_killers()
    _age_history()

    rep_counts, history = _initial_repetition_state(board)

    root_key = _position_key(board)
    best_move = _ordered_moves(board, legal_moves, _PV_MOVES.get(root_key), ply=0)[0]
    best_score = 0
    completed_depth = 0

    # A predicted continuation may already have been searched on our last turn.
    # Reuse only an exact score with matching repetition history and draw clock.
    entry = _tt_find(root_key)
    if (
        entry is not None
        and entry.bound == TT_EXACT
        and entry.move in legal_moves
        and rep_counts.get(root_key, 0) < 3
        and board.halfmove_clock < 100
    ):
        cached = _tt_cutoff(
            entry, entry.depth, -INFINITY, INFINITY, 0, history, board.halfmove_clock
        )
        if cached is not None:
            best_score, best_move, completed_depth = cached, entry.move, entry.depth
            _save_pv(board, best_move, completed_depth)

    for depth in range(completed_depth + 1, MAX_DEPTH + 1):
        try:
            if time.perf_counter() >= _DEADLINE:
                break
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
                        _ASPIRATION_RETRIES += 1
                        window *= 2
                        alpha = max(-INFINITY, best_score - window)
                        beta = min(INFINITY, best_score + window)
                        continue
                    if score >= beta:
                        _ASPIRATION_RETRIES += 1
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
        _save_pv(board, best_move, depth)

        if abs(best_score) >= MATE_SCORE - 100:
            break

    elapsed_ms = max(1, int((time.perf_counter() - start) * 1000))
    nps = _NODES * 1000 // elapsed_ms
    print(
        f"depth={completed_depth} seldepth={_SELDEPTH} score={best_score} "
        f"nodes={_NODES} qnodes={_QNODES} time_ms={elapsed_ms} nps={nps} "
        f"budget_ms={budget_ms} tt_hits={_TT_HITS} tt_cutoffs={_TT_CUTOFFS} "
        f"pvs_researches={_PVS_RESEARCHES} lmr_reductions={_LMR_REDUCTIONS} "
        f"lmr_researches={_LMR_RESEARCHES} qthreat_nodes={_QTHREAT_NODES} "
        f"tt_probes={_TT_PROBES} pv_reuses={_PV_REUSES} "
        f"aspiration_retries={_ASPIRATION_RETRIES} move={best_move.uci()} "
        f"pv={' '.join(move.uci() for move in _PV_LINE[:10])}"
    )
    return _remember_move(board, best_move)
