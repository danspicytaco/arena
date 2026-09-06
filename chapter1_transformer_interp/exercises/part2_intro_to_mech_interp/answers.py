# %%
# Setup (mirrors the exercise notebook imports)
import functools
import sys
from pathlib import Path
from typing import Callable, List, cast

import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from huggingface_hub import hf_hub_download
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

# Make the exercise helpers available when this file is run directly.
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)


# Saves computation time because this exercise only runs inference.
t.set_grad_enabled(False)
device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

print(f"Number of layers: {gpt2_small.cfg.n_layers}")
print(f"Number of heads per layer: {gpt2_small.cfg.n_heads}")
print(f"Maximum context window: {gpt2_small.cfg.n_ctx}")


model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)


# %%
# A logit at position i predicts the next token, so compare prediction[:-1] with input_tokens[1:].
logits: Tensor = gpt2_small(
    model_description_text, return_type="logits"
)  # shape [batch, seq_len, vocab]

# Get the most likely token to follow each i token in logits[batch, i]
predicted_next_tokens = logits.argmax(dim=-1)  # shape [batch, seq_len]

# Remove the batch dimension
predicted_next_tokens = predicted_next_tokens.squeeze()  # shape [seq_len]

# Prediction is all of the logits up to the last token
prediction = predicted_next_tokens[:-1]  # shape [seq_len - 1]

# Get the true next tokens, identify correct predictions, and render them as strings.
true_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
is_correct = true_tokens == prediction
correct_tokens = gpt2_small.to_str_tokens(prediction[is_correct])

print(f"Accuracy: {is_correct.sum()} / {len(true_tokens)}")
print(f"Correct tokens: {correct_tokens}")


# %%
# Caching all activations

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))

# %%

attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)

# %%
# Verify activations

layer0_pattern_from_cache = gpt2_cache["pattern", 0]  # shape [nhead, seqQ, seqK] [hqk]

# Calculate the raw attention scores
q = gpt2_cache["q", 0]  # shape [seqQ, nhead, headsize] [qhd]
k = gpt2_cache["k", 0]  # shape[seqK, nhead, headsize] [khd]
raw_scores = t.einsum("qhd,khd->hqk", q, k) / gpt2_small.cfg.d_head**0.5

# Manually create the attention/causal mask
seq_len = q.shape[0]
attn_mask = t.triu(t.ones(seq_len, seq_len, dtype=t.bool, device=q.device), diagonal=1)

# Calculate the attention pattern (probabilities)
masked_scores = raw_scores.masked_fill(attn_mask, -t.inf)
layer0_pattern_from_q_and_k = t.softmax(masked_scores, dim=-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %%
# Visualising Attention Heads

print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,  # pyright: ignore[reportArgumentType]
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)

# %%
# Finding induction heads
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)

# %%
REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

# %%

model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)

# %%
# Visualise & inspect attention patterns

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

_logits, cache = model.run_with_cache(text, remove_batch_dim=True)
tokens = cast(list[str], model.to_str_tokens(text))
for layer in range(cfg.n_layers):
    display(
        cv.attention.attention_patterns(
            tokens=tokens,
            attention=cache["pattern", layer],
            attention_head_names=[f"L{layer}H{i}" for i in range(cfg.n_heads)],
        )
    )


# %%
# Write your own detectors


def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    current_attn_heads = []
    current_attn_threshold = 0.4

    for layer in range(cfg.n_layers):
        layer_attn = cache["pattern", layer]  # shape [head, query, key]
        for head in range(cfg.n_heads):
            head_attn_pattern = layer_attn[head]  # shape [query, key]

            # Calculate the average score of the current token (where query == key; i.e. along the diagonal)
            head_score = t.mean(head_attn_pattern.diagonal())

            if head_score > current_attn_threshold:
                current_attn_heads.append(f"{layer}.{head}")

    return current_attn_heads


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    prev_attn_heads = []
    prev_attn_threshold = 0.4

    for layer in range(cfg.n_layers):
        layer_attn = cache["pattern", layer]  # shape [head, query, key]
        for head in range(cfg.n_heads):
            head_attn_pattern = layer_attn[head]  # shape [query, key]

            # Calculate the average score of the prev token (where query == key - 1; i.e. one step below the diagonal)
            head_score = t.mean(head_attn_pattern.diagonal(offset=-1))
            if head_score > prev_attn_threshold:
                prev_attn_heads.append(f"{layer}.{head}")

    return prev_attn_heads


def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    first_attn_heads = []
    first_attn_threshold = 0.4

    for layer in range(cfg.n_layers):
        layer_attn = cache["pattern", layer]  # shape [head, query, key]
        for head in range(cfg.n_heads):
            head_attn_pattern = layer_attn[head]  # shape [query, key]

            # Calculate the average score of the first token (where key == 0; i.e. the first key)
            head_score = t.mean(head_attn_pattern[1:, 0])
            if head_score > first_attn_threshold:
                first_attn_heads.append(f"{layer}.{head}")

    return first_attn_heads


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%
# plot per-token loss on a repeated sequence
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    rand_seq = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len))
    rep_tokens = t.concat((prefix, rand_seq, rand_seq), 1)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    rep_tokens = generate_repeated_tokens(
        model,
        seq_len,
        batch_size,
    )
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return (rep_tokens, rep_logits, rep_cache)


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


seq_len = 50
batch_size = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
    model, seq_len, batch_size
)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)

# %%


def display_heads(model: HookedTransformer, token_str: list[str], cache: ActivationCache):
    for layer in range(model.cfg.n_layers):
        display(
            cv.attention.attention_patterns(
                tokens=token_str,
                attention=cache["pattern", layer],
                attention_head_names=[f"L{layer}H{i}" for i in range(model.cfg.n_heads)],
            )
        )


display_heads(model, rep_str, rep_cache)


# %%
def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    induction_attn_heads = []
    induction_attn_threshold = 0.4

    for layer in range(cfg.n_layers):
        layer_attn = cache["pattern", layer]  # shape [head, query, key]
        for head in range(cfg.n_heads):
            head_attn_pattern = layer_attn[head]  # shape [query, key]

            # Calculate the average score of the induction token (where query == seq_len - 1)
            head_score = t.mean(head_attn_pattern.diagonal(offset=-seq_len + 1))
            if head_score > induction_attn_threshold:
                induction_attn_heads.append(f"{layer}.{head}")

    return induction_attn_heads


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%
# Calculate induction scores with hooks
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
)  # shape [n_layers, n_heads]


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """
    layer = hook.layer()
    for batch in range(pattern.shape[0]):
        layer_attn = pattern[batch]  # shape [head, query, key]
        for head in range(cfg.n_heads):
            head_attn_pattern = layer_attn[head]  # shape [query, key]
            head_score = t.mean(head_attn_pattern.diagonal(offset=-seq_len + 1))
            induction_score_store[layer, head] = head_score


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%
# Find induction heads in GPT2-small

seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

induction_score_store = t.zeros(
    (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
)  # shape [n_layers, n_heads]


def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
        )
    )

    layer = hook.layer()
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    induction_score = einops.reduce(
        induction_stripe, "batch head_index position -> head_index", "mean"
    )
    induction_score_store[layer, :] = induction_score


gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter, visualize_pattern_hook)],
)

imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%
