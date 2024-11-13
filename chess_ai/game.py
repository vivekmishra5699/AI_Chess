import pygame
import chess
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Initialize pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (118, 150, 86)
LIGHT_GREEN = (238, 238, 210)

# Pygame window settings
WIDTH, HEIGHT = 512, 512
SQUARE_SIZE = WIDTH // 8
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess AI')

# Load images for the pieces
def load_images():
    pieces = ['wp', 'wr', 'wn', 'wb', 'wq', 'wk', 'bp', 'br', 'bn', 'bb', 'bq', 'bk']
    images = {}
    for piece in pieces:
        try:
            image = pygame.image.load(f'images/{piece}.png')
            images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))  # Resize images
        except pygame.error as e:
            print(f"Error loading image {piece}: {e}")
    return images

# Set up ChessNet model class
class ChessNet(keras.Model):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv_layers = [
            keras.layers.Conv2D(256, 3, activation='relu', padding='same', input_shape=(8, 8, 119))
        ] + [
            keras.layers.Conv2D(256, 3, activation='relu', padding='same') 
            for _ in range(18)
        ]
        self.flatten = keras.layers.Flatten()
        self.policy_dense = keras.layers.Dense(4672, activation='softmax')  # 8*8*73
        self.value_dense = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.flatten(x)
        policy = self.policy_dense(x)
        value = self.value_dense(x)
        return policy, value

class MCTS:
    def board_to_input(self, board):
        planes = np.zeros((8, 8, 119), dtype=np.float32)
        piece_chars = 'pnbrqkPNBRQK'
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                piece_idx = piece_chars.index(piece.symbol())
                for j in range(8):
                    planes[i // 8][i % 8][piece_idx * 8 + j] = 1
        return planes

    def predict_move(self, model, board):
        board_input = np.expand_dims(self.board_to_input(board), axis=0)
        policy, value = model(board_input)
        policy = policy.numpy().reshape((64, 73))
        best_move_idx = np.argmax(policy)
        from_square = best_move_idx // 73
        to_square = best_move_idx % 73
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move
        else:
            return random.choice(list(board.legal_moves))

class ChessAI:
    def __init__(self, model_path):
        self.model = ChessNet()
        self.model.build((None, 8, 8, 119))
        self.model.load_weights(model_path)
        self.mcts = MCTS()

    def play_move(self, board):
        move = self.mcts.predict_move(self.model, board)
        return move

# Draw the chess board
def draw_board(screen, board_state):
    colors = [LIGHT_GREEN, GREEN]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            piece = board_state.piece_at(chess.square(col, 7 - row))
            if piece:
                piece_image = images[f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().lower()}"]
                screen.blit(piece_image, pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Main game loop
# ... (previous code remains the same)

# Main game loop
def chess_ai_game(model_path):
    board = chess.Board()
    ai = ChessAI(model_path)
    selected_square = None
    running = True
    player_turn = True

    while running:
        draw_board(screen, board)
        
        # Highlight selected square
        if selected_square is not None:
            row, col = 7 - chess.square_rank(selected_square), chess.square_file(selected_square)
            pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and player_turn:
                pos = pygame.mouse.get_pos()
                col = pos[0] // SQUARE_SIZE
                row = 7 - (pos[1] // SQUARE_SIZE)
                clicked_square = chess.square(col, row)

                if selected_square is None:
                    piece = board.piece_at(clicked_square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = clicked_square
                        print(f"Selected square: {chess.SQUARE_NAMES[selected_square]}")
                else:
                    move = chess.Move(selected_square, clicked_square)
                    
                    # Handle pawn promotion
                    if board.piece_at(selected_square).piece_type == chess.PAWN:
                        if (chess.square_rank(clicked_square) == 7 and board.turn == chess.WHITE) or \
                           (chess.square_rank(clicked_square) == 0 and board.turn == chess.BLACK):
                            move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

                    if move in board.legal_moves:
                        board.push(move)
                        player_turn = False
                        print(f"Move made: {move}")
                    else:
                        print(f"Illegal move attempted: {move}")
                    selected_square = None

        if not player_turn and not board.is_game_over():
            ai_move = ai.play_move(board)
            board.push(ai_move)
            player_turn = True
            print(f"AI move: {ai_move}")

        if board.is_game_over():
            result = board.result()
            print(f"Game Over: {result}")
            running = False

    pygame.quit()

# Main execution
model_path = 'chess_ai/advanced_chess_model.weights.h5'
images = load_images()  # Load images before starting the game
chess_ai_game(model_path)