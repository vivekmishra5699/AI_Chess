# AI Chess

AI Chess is an advanced chess-playing artificial intelligence that utilizes deep learning techniques inspired by AlphaZero. The model is trained on chess games to play at a grandmaster level, making strategic decisions based on self-play reinforcement learning.

## Features

- AI-powered chess engine trained using deep reinforcement learning
- Implements AlphaZero-style training with self-play and neural networks
- Flask-based backend for serving AI moves via API
- Web-based UI for playing against the AI

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8+
- TensorFlow
- Flask

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vivekmishra5699/AI_Chess.git
   cd AI_Chess
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the AI Chess Server

Start the Flask server to interact with the AI:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/`.


## Training the AI

To train the AI from scratch, run:

```bash
cd chess_ai
python training.py
```

This will start the reinforcement learning process.

## Contribution

Feel free to contribute by creating issues and pull requests.


## Author

[Vivek Mishra](https://github.com/vivekmishra5699)

