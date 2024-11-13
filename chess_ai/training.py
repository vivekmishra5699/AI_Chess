import chess.pgn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import os
import logging
from datetime import datetime
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_gpu():
    """Configure GPU memory growth and mixed precision."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Limit GPU memory to slightly less than 2GB to leave room for system
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1800)]
                )
            logger.info(f"GPU configuration successful. Found {len(gpus)} GPUs")
        else:
            logger.warning("No GPUs found. Running on CPU")

        # Enable mixed precision for better memory efficiency
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision enabled")
        
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
        raise

class ChessDataPreprocessor:
    """Memory-efficient chess data preprocessor."""
    
    @staticmethod
    def board_to_input(board):
        """Convert a chess board position to neural network input format."""
        # Use float16 for memory efficiency
        planes = np.zeros((8, 8, 119), dtype=np.float16)
        piece_chars = 'pnbrqkPNBRQK'
        
        # Piece planes (12 * 8 = 96 planes)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                piece_idx = piece_chars.index(piece.symbol())
                for j in range(8):
                    planes[i // 8][i % 8][piece_idx * 8 + j] = 1

        # Additional features
        ChessDataPreprocessor._add_repetition_planes(board, planes)
        ChessDataPreprocessor._add_color_plane(board, planes)
        ChessDataPreprocessor._add_move_count_planes(board, planes)
        ChessDataPreprocessor._add_castling_planes(board, planes)
        ChessDataPreprocessor._add_misc_planes(board, planes)
        
        return planes

    @staticmethod
    def _add_repetition_planes(board, planes):
        repetitions = board.is_repetition(2)
        if repetitions:
            planes[:, :, 96 + min(repetitions - 1, 6)] = 1

    @staticmethod
    def _add_color_plane(board, planes):
        if board.turn == chess.WHITE:
            planes[:, :, 103] = 1

    @staticmethod
    def _add_move_count_planes(board, planes):
        total_moves = bin(board.fullmove_number)[2:].zfill(8)
        for i, bit in enumerate(total_moves):
            if bit == '1':
                planes[:, :, 104 + i] = 1

    @staticmethod
    def _add_castling_planes(board, planes):
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[:, :, 112] = 1
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[:, :, 113] = 1
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[:, :, 114] = 1
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[:, :, 115] = 1

    @staticmethod
    def _add_misc_planes(board, planes):
        # No-progress count
        planes[:, :, 116] = board.halfmove_clock / 100.0
        # Check
        if board.is_check():
            planes[:, :, 117] = 1
        # En passant
        if board.ep_square:
            planes[board.ep_square // 8][board.ep_square % 8][118] = 1

class ChessNet(keras.Model):
    """Memory-optimized neural network model for chess position evaluation."""
    
    def __init__(self):
        super(ChessNet, self).__init__()
        # Reduce model size by using fewer filters and layers
        self.conv_layers = [
            keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(8, 8, 119))
        ] + [
            keras.layers.Conv2D(64, 3, activation='relu', padding='same') 
            for _ in range(6)  # Reduced from 8 to 4 layers
        ]
        self.flatten = keras.layers.Flatten()
        self.policy_dense = keras.layers.Dense(4672, activation='softmax')
        self.value_dense = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.flatten(x)
        policy = self.policy_dense(x)
        value = self.value_dense(x)
        return policy, value

class MemoryEfficientDataGenerator:
    """Memory-efficient data generator for chess positions."""
    
    def __init__(self, games, batch_size, preprocessor):
        self.games = games
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.current_game_idx = 0
        self.current_position_idx = 0
        self.current_game = None
        self.current_board = None
        self.current_moves = None

    def __iter__(self):
        return self

    def __next__(self):
        inputs = []
        policy_targets = []
        value_targets = []

        while len(inputs) < self.batch_size:
            if self.current_game is None or self.current_position_idx >= len(self.current_moves):
                if self.current_game_idx >= len(self.games):
                    if len(inputs) == 0:
                        raise StopIteration
                    break
                
                self.current_game = self.games[self.current_game_idx]
                self.current_board = self.current_game.board()
                self.current_moves = list(self.current_game.mainline_moves())
                self.current_position_idx = 0
                self.current_game_idx += 1

            # Process current position
            current_input = self.preprocessor.board_to_input(self.current_board)
            current_move = self.current_moves[self.current_position_idx]
            
            # Create policy target
            target_policy = np.zeros(4672, dtype=np.float16)  # Use float16 for memory efficiency
            move_index = current_move.from_square * 73 + current_move.to_square
            target_policy[move_index] = 1

            # Create value target
            result = self.current_game.headers["Result"]
            target_value = 0  # Draw
            if result == "1-0":
                target_value = 1 if self.current_board.turn == chess.WHITE else -1
            elif result == "0-1":
                target_value = -1 if self.current_board.turn == chess.WHITE else 1

            inputs.append(current_input)
            policy_targets.append(target_policy)
            value_targets.append([target_value])

            self.current_board.push(current_move)
            self.current_position_idx += 1

            # Clear memory periodically
            if len(inputs) % (self.batch_size // 2) == 0:
                gc.collect()

        return (np.array(inputs, dtype=np.float16),
                np.array(policy_targets, dtype=np.float16),
                np.array(value_targets, dtype=np.float16))

@tf.function
def train_step(model, input_tensor, target_policy, target_value, optimizer, cce_loss, mse_loss):
    """Memory-efficient training step with gradient accumulation."""
    with tf.GradientTape() as tape:
        policy, value = model(input_tensor)
        policy_loss = cce_loss(target_policy, policy)
        value_loss = mse_loss(target_value, value)
        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss

def train_model(model, games, batch_size=4, epochs=10):
    """Train the chess model with memory optimization."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mse_loss = tf.keras.losses.MeanSquaredError()
    cce_loss = tf.keras.losses.CategoricalCrossentropy()
    preprocessor = ChessDataPreprocessor()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0
        num_batches = 0

        # Use memory-efficient generator
        data_generator = MemoryEfficientDataGenerator(games, batch_size, preprocessor)
        
        for inputs, policy_targets, value_targets in tqdm(data_generator, desc="Training"):
            loss = train_step(
                model, inputs, policy_targets, value_targets,
                optimizer, cce_loss, mse_loss
            )
            epoch_loss += loss
            num_batches += 1

            # Clear memory periodically
            if num_batches % 10 == 0:
                gc.collect()
                tf.keras.backend.clear_session()

        avg_loss = epoch_loss / num_batches
        logger.info(f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = f"checkpoints/epoch_{epoch + 1}_loss_{avg_loss:.4f}.weights.h5"
        model.save_weights(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Clear memory between epochs
        gc.collect()
        tf.keras.backend.clear_session()

def main():
    """Main training function with memory optimization."""
    try:
        os.makedirs("checkpoints", exist_ok=True)
        configure_gpu()

        model = ChessNet()
        model.build((None, 8, 8, 119))
        logger.info("Model initialized successfully")

        # Load games in chunks
        pgn_file_path = 'data/chess_data.pgn'
        logger.info(f"Loading games from {pgn_file_path}")
        
        chunk_size = 1000  # Process games in chunks
        games = []
        
        with open(pgn_file_path) as pgn_file:
            for _ in range(chunk_size):
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        
        logger.info(f"Loaded {len(games)} games")

        # Train with small batch size and memory optimization
        train_model(
            model,
            games,
            batch_size=4,  # Reduced batch size
            epochs=10
        )

        final_model_path = f'chess_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.weights.h5'
        model.save_weights(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()