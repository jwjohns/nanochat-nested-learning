# Nested Learning Lite for nanochat

This feature implements **Nested Learning (Variant A: Additive Fast-State Bias)**, allowing the model to adapt its state during a conversation based on "surprisal" signals, without modifying the base weights.

## How it Works

*   **Inner State**: A persistent state vector (initially zero) that tracks short-term context.
*   **Injection**: This state is added to the final hidden representation before the `lm_head`.
*   **Update**: After generating a token, the module calculates the "surprisal" (unexpectedness) of that token. It updates the inner state to minimize this surprisal for similar future contexts, effectively "learning" from the interaction in real-time.

## Usage

### Chat CLI

You can enable nested learning when chatting with the model:

```bash
python -m scripts.chat_cli --inner-learn \
    --inner-lr 0.01 \
    --inner-decay 0.99 \
    --inner-reset conversation
```

**Arguments:**
*   `--inner-learn`: Enable the inner learner.
*   `--inner-lr`: Learning rate (default: 0.1). How fast it adapts.
*   `--inner-decay`: Decay rate (default: 0.999). How fast the state fades (1.0 = no fade).
*   `--inner-clip`: Gradient clipping value (default: 1.0).
*   `--inner-reset`: When to reset the state. Options: `conversation` (default), `turn`, `none`.

### Running Demos

A demo script compares the baseline model vs. the inner learner on a few tasks:

```bash
python scripts/demo_inner_learning.py --source base --inner-lr 0.05
```

(Note: Requires a trained model checkpoint. Use `--model-tag dummy` if you only want to test the code mechanics).

## Implementation Details

*   **`nanochat/inner_learner.py`**: Contains the `InnerLearner` class.
*   **`nanochat/gpt.py`**: Modified to call `inner_learner.apply()` and return hidden states.
*   **`nanochat/engine.py`**: Modified to call `inner_learner.update()` in the generation loop.
