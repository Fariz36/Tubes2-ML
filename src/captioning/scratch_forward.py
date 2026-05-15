from __future__ import annotations

import numpy as np

from captioning.scratch_decoder import BeamCandidate, RecurrentWeights


def decoder_forward(decoder, feature: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    projected_image = np.tanh(feature @ decoder.image_projection.kernel + decoder.image_projection.bias)
    token_embeddings = decoder.token_embedding[tokens]
    x = np.concatenate([projected_image[None, :], token_embeddings], axis=0)

    for weights in decoder.recurrent_layers:
        if decoder.model_kind == "rnn":
            x = simple_rnn_forward(x, weights)
        elif decoder.model_kind == "lstm":
            x = lstm_forward(x, weights)
        else:
            raise ValueError(f"Unsupported model_kind: {decoder.model_kind}")

    caption_steps = x[1:, :]
    logits = caption_steps @ decoder.token_distribution.kernel + decoder.token_distribution.bias
    return softmax(logits)


def decoder_forward_batch(decoder, feature_batch: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    projected_image = np.tanh(feature_batch @ decoder.image_projection.kernel + decoder.image_projection.bias)
    token_embeddings = decoder.token_embedding[tokens]
    x = np.concatenate([projected_image[:, None, :], token_embeddings], axis=1)

    for weights in decoder.recurrent_layers:
        if decoder.model_kind == "rnn":
            x = simple_rnn_forward_batch(x, weights)
        elif decoder.model_kind == "lstm":
            x = lstm_forward_batch(x, weights)
        else:
            raise ValueError(f"Unsupported model_kind: {decoder.model_kind}")

    caption_steps = x[:, 1:, :]
    logits = caption_steps @ decoder.token_distribution.kernel + decoder.token_distribution.bias
    return softmax(logits)


def greedy_decode(decoder, feature: np.ndarray) -> str:
    pad_idx = decoder.word_to_idx["<pad>"]
    start_idx = decoder.word_to_idx["<start>"]
    end_idx = decoder.word_to_idx["<end>"]
    tokens = np.full((decoder.max_steps,), pad_idx, dtype=np.int32)
    tokens[0] = start_idx
    generated: list[str] = []

    for step in range(decoder.max_steps):
        predictions = decoder_forward(decoder, feature, tokens)
        next_idx = int(np.argmax(predictions[step]))
        if next_idx == end_idx:
            break

        word = decoder.idx_to_word.get(str(next_idx), "<unk>")
        if word not in {"<pad>", "<start>", "<unk>"}:
            generated.append(word)
        if step + 1 < decoder.max_steps:
            tokens[step + 1] = next_idx

    return " ".join(generated) if generated else "<empty>"


def greedy_decode_batch(decoder, feature_batch: np.ndarray) -> list[str]:
    pad_idx = decoder.word_to_idx["<pad>"]
    start_idx = decoder.word_to_idx["<start>"]
    end_idx = decoder.word_to_idx["<end>"]
    batch_size = feature_batch.shape[0]
    tokens = np.full((batch_size, decoder.max_steps), pad_idx, dtype=np.int32)
    tokens[:, 0] = start_idx
    generated: list[list[str]] = [[] for _ in range(batch_size)]
    finished = np.zeros((batch_size,), dtype=bool)

    for step in range(decoder.max_steps):
        predictions = decoder_forward_batch(decoder, feature_batch, tokens)
        next_indices = np.argmax(predictions[:, step, :], axis=1).astype(np.int32)

        for row_index, next_idx in enumerate(next_indices):
            if finished[row_index]:
                continue
            if next_idx == end_idx:
                finished[row_index] = True
                continue

            word = decoder.idx_to_word.get(str(int(next_idx)), "<unk>")
            if word not in {"<pad>", "<start>", "<unk>"}:
                generated[row_index].append(word)
            if step + 1 < decoder.max_steps:
                tokens[row_index, step + 1] = next_idx

        if finished.all():
            break

    return [" ".join(words) if words else "<empty>" for words in generated]


def beam_search_decode(decoder, feature: np.ndarray, beam_width: int = 3) -> str:
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    pad_idx = decoder.word_to_idx["<pad>"]
    start_idx = decoder.word_to_idx["<start>"]
    end_idx = decoder.word_to_idx["<end>"]
    blocked_indices = {pad_idx, start_idx, decoder.word_to_idx.get("<unk>")}
    tokens = np.full((decoder.max_steps,), pad_idx, dtype=np.int32)
    tokens[0] = start_idx
    candidates = [BeamCandidate(tokens=tokens, words=(), log_probability=0.0, ended=False)]

    for step in range(decoder.max_steps):
        expanded: list[BeamCandidate] = []
        for candidate in candidates:
            if candidate.ended:
                expanded.append(candidate)
                continue

            probabilities = decoder_forward(decoder, feature, candidate.tokens)[step]
            probabilities = probabilities.copy()
            for blocked_idx in blocked_indices:
                if blocked_idx is not None:
                    probabilities[int(blocked_idx)] = 0.0

            top_indices = np.argsort(probabilities)[-beam_width:][::-1]
            for next_idx in top_indices:
                next_idx = int(next_idx)
                next_probability = max(float(probabilities[next_idx]), 1e-12)
                next_tokens = candidate.tokens.copy()
                if step + 1 < decoder.max_steps:
                    next_tokens[step + 1] = next_idx

                if next_idx == end_idx:
                    expanded.append(
                        BeamCandidate(
                            tokens=next_tokens,
                            words=candidate.words,
                            log_probability=candidate.log_probability + float(np.log(next_probability)),
                            ended=True,
                        )
                    )
                    continue

                word = decoder.idx_to_word.get(str(next_idx), "<unk>")
                expanded.append(
                    BeamCandidate(
                        tokens=next_tokens,
                        words=(*candidate.words, word),
                        log_probability=candidate.log_probability + float(np.log(next_probability)),
                        ended=False,
                    )
                )

        candidates = sorted(expanded, key=beam_candidate_score, reverse=True)[:beam_width]
        if all(candidate.ended for candidate in candidates):
            break

    best = max(candidates, key=beam_candidate_score)
    return " ".join(best.words) if best.words else "<empty>"


def simple_rnn_forward(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((hidden_units,), dtype=np.float32)
    outputs = []
    for timestep in sequence:
        hidden = np.tanh(timestep @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias)
        outputs.append(hidden)
    return np.stack(outputs, axis=0)


def simple_rnn_forward_batch(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=np.float32)
    outputs = []
    for step in range(sequence.shape[1]):
        hidden = np.tanh(sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias)
        outputs.append(hidden)
    return np.stack(outputs, axis=1)


def lstm_forward(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((hidden_units,), dtype=np.float32)
    cell = np.zeros((hidden_units,), dtype=np.float32)
    outputs = []
    for timestep in sequence:
        gates = timestep @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias
        input_gate, forget_gate, cell_candidate, output_gate = np.split(gates, 4)
        input_gate = sigmoid(input_gate)
        forget_gate = sigmoid(forget_gate)
        cell_candidate = np.tanh(cell_candidate)
        output_gate = sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)
    return np.stack(outputs, axis=0)


def lstm_forward_batch(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=np.float32)
    cell = np.zeros((batch_size, hidden_units), dtype=np.float32)
    outputs = []
    for step in range(sequence.shape[1]):
        gates = sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias
        input_gate, forget_gate, cell_candidate, output_gate = np.split(gates, 4, axis=1)
        input_gate = sigmoid(input_gate)
        forget_gate = sigmoid(forget_gate)
        cell_candidate = np.tanh(cell_candidate)
        output_gate = sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)
    return np.stack(outputs, axis=1)


def beam_candidate_score(candidate: BeamCandidate) -> float:
    length = max(1, len(candidate.words))
    return candidate.log_probability / length


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
