import chess
import numpy as np

def board_to_input(board):
    """Convert chess board to neural network input format."""
    planes = np.zeros((8, 8, 119), dtype=np.float32)
    piece_chars = 'pnbrqkPNBRQK'
    
    # Piece planes (12 * 8 = 96 planes)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_idx = piece_chars.index(piece.symbol())
            for j in range(8):
                planes[i // 8][i % 8][piece_idx * 8 + j] = 1

    # Repetition counters (7 planes)
    repetitions = board.is_repetition(2)
    if repetitions:
        planes[:, :, 96 + min(repetitions - 1, 6)] = 1

    # Color (1 plane)
    if board.turn == chess.WHITE:
        planes[:, :, 103] = 1

    # Total move count (8 planes)
    total_moves = bin(board.fullmove_number)[2:].zfill(8)
    for i, bit in enumerate(total_moves):
        if bit == '1':
            planes[:, :, 104 + i] = 1

    # Castling rights (4 planes)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[:, :, 112] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[:, :, 113] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[:, :, 114] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[:, :, 115] = 1

    # No-progress count (1 plane)
    planes[:, :, 116] = board.halfmove_clock / 100.0

    # Check (1 plane)
    if board.is_check():
        planes[:, :, 117] = 1

    # En passant (1 plane)
    if board.ep_square:
        planes[board.ep_square // 8][board.ep_square % 8][118] = 1

    return planes