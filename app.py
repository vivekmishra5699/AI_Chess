from flask import Flask, request, jsonify, render_template
import chess
import numpy as np
from chess_ai.chess_utils import board_to_input
from chess_ai.model import ChessNet
import tensorflow as tf
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the model
def load_chess_model():
    try:
        model = ChessNet()
        model.build((None, 8, 8, 119))
        model.load_weights('chess_ai/advanced_chess_model.weights.h5')
        logger.info("Chess model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_chess_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint to check if the service is running and model is loaded."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """API endpoint to get the AI's next move."""
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.get_json()
        fen = data.get('fen', chess.STARTING_FEN)
        board = chess.Board(fen)
        
        # Log the current game state
        logger.info(f"Making move for position: {fen}")
        logger.info(f"Legal moves: {[move.uci() for move in board.legal_moves]}")

        # Check if game is already over
        if board.is_game_over():
            return jsonify({
                "status": "game_over",
                "result": board.result(),
                "reason": _get_game_over_reason(board)
            })

        # Get AI move
        ai_move = get_best_move(board)
        if not ai_move:
            return jsonify({"status": "error", "message": "No valid moves found"}), 400

        # Apply the move and get new state
        board.push(ai_move)
        
        response = {
            "status": "success",
            "move": ai_move.uci(),
            "fen": board.fen()
        }

        # Check if the game is over after AI's move
        if board.is_game_over():
            response.update({
                "status": "game_over",
                "result": board.result(),
                "reason": _get_game_over_reason(board)
            })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing move: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def get_best_move(board):
    """Get the best move for the current position using the AI model."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_score = float('-inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        input_tensor = np.expand_dims(board_to_input(board), axis=0)
        policy, value = model(input_tensor)
        score = float(value.numpy()[0][0])
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def _get_game_over_reason(board):
    """Get the reason why the game is over."""
    if board.is_checkmate():
        return "Checkmate"
    elif board.is_stalemate():
        return "Stalemate"
    elif board.is_insufficient_material():
        return "Insufficient material"
    elif board.is_fifty_moves():
        return "Fifty-move rule"
    elif board.is_repetition():
        return "Threefold repetition"
    return "Game over"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)