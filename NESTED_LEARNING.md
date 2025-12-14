Project Guide: “Nested Learning Lite” Demo on Karpathy nano-chat (nanoGPT)
Goal: implement a fast inner-learning loop (temporary state updated during generation) on top of nano-chat so the model adapts within a conversation without changing base weights. This demonstrates the paper’s “nested learning / fast vs slow” idea in a concrete, inspectable way.
0) Deliverables
Working fork/branch of nanoGPT/nano-chat with:
--inner_learn flag enabling inner learner
controllable --inner_reset policy: conversation|turn|never
controllable --inner_decay and --inner_lr
safe update stabilization (norm clipping)
Demo script to run side-by-side comparisons:
baseline nano-chat
nano-chat + inner learner
Evaluation tasks (small text prompts) + a simple scorer:
pattern induction
formatting adherence (JSON)
style shift across turns
README explaining:
what NL-lite is
how it maps to nested learning
how to run the demos
1) Repo Assumptions
Target codebase: Karpathy nanoGPT plus whatever nano-chat wrapper you’re using (often:
a chat-style prompt template
a CLI loop that collects user turns
calls a generation function)
If your nano-chat is a separate repo, the integration points are the same: you must control the token-by-token generation loop.
2) Architecture Overview
Add a module called InnerLearner that holds a fast state (temporary parameters) updated each token step.
Two variants to implement (do Variant A first):
Variant A (Simplest): Additive Fast-State Bias
state: a vector of size n_embd (or per-layer vectors)
applied as x = x + state at a chosen injection point
update rule uses a cheap local signal (no backprop)
Variant B (Still Simple): Low-Rank Adapter With Fast Weights
A ∈ R^{n_embd×r}, B ∈ R^{r×n_embd} are trainable or frozen
fast state is a vector z ∈ R^r (or small matrix)
inject: x = x + (x @ A) * z @ B or similar
update z per token
Start with A. Only do B if A works and you want nicer behavior.
3) Integration Points
Where to inject
Pick one, in this order of preference:
After final transformer block, before lm_head
easiest, least intrusive
affects logits directly
Inside each block after attention + MLP
more “nested”
more work + more risk of destabilization
Implement (1) first.
Where to update
Update after computing logits for the next token:
you have logits_t
you also have either:
the sampled token (self-supervised signal), or
a user-provided “teaching” token stream (optional)
4) Inner Learner Design (Variant A)
Inner learner state
state: torch.Tensor [n_embd]
stored on same device, dtype as model
Forward
x = x + state (broadcast over batch, time)
Update signal options (choose one)
Option 1 (No labels): “Surprisal-driven” update
Use the model’s own confidence to decide how much to adapt.
Let p = softmax(logits_t)
Let token = sampled_token
Let surprisal = -log(p[token])
Use surprisal as a scalar gate:
when model is uncertain, update more
when confident, update less
Update direction can come from the hidden state h_t:
grad_signal ≈ normalize(h_t.mean(dim=batch))
Update:
state = decay*state + lr * surprisal * grad_signal
This is stable and easy.
Option 2 (Pseudo-gradient): error against a “target style”
If you have a formatting constraint (e.g., JSON), define a small heuristic loss and convert to a signal that nudges state.
For first implementation: Option 1.
Stabilization
Mandatory:
clip norm of update: Δ = clamp_norm(Δ, max_norm)
state norm cap: ||state|| <= state_max_norm
decay each step: state *= inner_decay (e.g., 0.995–0.999)
5) Code Tasks (Step-by-Step)
Step 1 — Add config flags
Add CLI flags (wherever args live):
--inner_learn (bool)
--inner_lr (float, default 0.01)
--inner_decay (float, default 0.999)
--inner_clip (float, default 1.0)
--inner_reset (enum: conversation|turn|never, default conversation)
--inner_inject (enum: pre_lm_head|per_block, default pre_lm_head)
Step 2 — Implement InnerLearner module
Create inner_learner.py:
reset()
apply(x) -> x
update(h_t, logits_t, sampled_token)
Support:
batch size 1 is fine initially
handle dtype/device correctly
Step 3 — Modify model forward to allow injection
In model class:
add optional inner_learner argument or attach it to model instance
in forward, right before lm_head:
x = inner_learner.apply(x) if enabled
Step 4 — Modify generation loop (critical)
In your generation function (often generate()):
you must run token-by-token
at each step:
forward pass to get logits
sample token
call inner_learner.update(h_t, logits_t, token)
h_t should be the last hidden state vector (the last time index)
append token and continue
You need access to h_t. If current code only returns logits:
modify model forward to optionally return last hidden state
e.g., return logits, x where x is final hidden sequence
then h_t = x[:, -1, :]
Step 5 — Hook reset policy for chat
In chat loop:
if inner_reset == conversation: reset() when new chat starts
if inner_reset == turn: reset() each user message before assistant generation
if never: never reset (for demonstration; can drift)
Step 6 — Add logging and debug toggles
Print per token step (optional flag):
surprisal
update norm
state norm
This will help you tune quickly.
6) Demo Prompts + Expected Outcomes
Create demos/prompts.jsonl with 3 tasks.
Demo A: Formatting adherence across turns
Turn 1:
“Answer in strict JSON: {‘answer’: string}”
Turn 2:
Ask 3 questions.
Expect:
baseline sometimes drifts
inner learner sticks to JSON stronger across turns
Demo B: Style lock-in
Turn 1:
“Be extremely terse. 1 sentence max.”
Turn 2:
“Explain X”
Expect:
inner learner maintains terse constraint more reliably
Demo C: Pattern induction
Single prompt:
“Mapping: cat→dax, dog→bix, bird→?”
Expect:
slight improvement in mapping continuation (not guaranteed on tiny models, but measurable with repetitions)
7) Simple Evaluation Harness
Implement eval.py:
runs each demo 20 times with fixed seeds
collects:
JSON validity (% parseable)
length (tokens)
constraint violations (regex checks)
No need for fancy metrics—just show clear deltas.
8) Tuning Defaults (Start Here)
inner_lr = 0.01
inner_decay = 0.999
inner_clip = 0.5
state_max_norm = 5.0
sampling:
temperature 0.8
top_k 50 (or whatever baseline uses)
If it destabilizes:
lower inner_lr to 0.001
increase decay to 0.9995
lower clip to 0.2
9) “Nested Learning” Story for README
In README.md, explain:
Base model training = slow loop (standard SGD/Adam)
Inner learner = fast loop (within-context adaptation)
This demo shows how a model can have two learning timescales
It’s not full HOPE/CMS, but it’s the same core principle:
learning inside inference without weight updates
Include a diagram:
Prompt → tokens → (model + inner state updates) → output
plus note that inner state resets by policy
10) Definition of Done
✅ python chat.py --inner_learn runs and does not crash
✅ fast state clearly changes over generation (logged norms)
✅ Demo prompts show measurable difference on at least 1–2 tasks
✅ README explains what’s happening and how to reproduce
11) Stretch Goals (Optional)
Per-layer inner learners (true nesting)
Learned update rule Δstate = f(h_t, surprisal) (small MLP)
“Memory continuum” by adding multiple states with different decays:
state_fast decay 0.99
state_mid decay 0.999
state_slow decay 0.9999
Inject sum of them.
This is the closest “mini CMS” you can do without rewriting training.