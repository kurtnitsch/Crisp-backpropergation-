Cognitive Packets

A Categorical Foundation for Distributed Backpropagation

Author: Kurt Nitsch • Status: Research Prototype v0.1.0 • Requires: Python ≥ 3.10, PyTorch ≥ 2.0

---

Your Distributed Training Pipeline Has No Type System. This Fixes That.

Cognitive Packets wraps distributed neural network training in a formal mathematical structure—the category CP—so composition errors are caught at definition time, not six hours into a training run.

The framework formalizes the forward/backward pass as a provable mathematical duality (an adjunction): every valid forward layer has a correct backward pass that can be automatically derived, not manually debugged.

Think of it as typed, composable, mathematically-grounded building blocks for distributed deep learning.

---

The Problem It Solves

Right now, composing distributed training stages looks like this:

```python
# Naive PyTorch — no type checking. This crashes at runtime, mid-training.
x = torch.randn(batch, 256)
x = nn.Linear(256, 64)(x)   # Node 1 outputs 64-dim  
x = nn.Linear(128, 10)(x)   # Node 2 expects 128-dim ← silent until it explodes
```

Three classes of bugs that PyTorch cannot catch until your training run dies:

Bug Class What Happens When You Find Out
Shape mismatch Pipeline stages have incompatible dimensions Runtime crash, hours in
Semantic mismatch Gradient packet fed into activation slot — same shape, wrong meaning Silent wrong answers, days in
Device mismatch Tensors on different devices composed without transfer Runtime crash, immediately annoying

Cognitive Packets catches all three at pipeline definition time—before a single tensor is computed.

---

What Makes It Different

Feature PyTorch Distributed Federated Learning Cognitive Packets
Type checking ❌ None ❌ None ✅ At definition time
Semantic types ❌ ❌ ✅ Gradients ≠ Activations
Formal correctness ❌ Engineering intuition ❌ Ad-hoc ✅ Categorical laws proven
Backward pass Manual Manual ✅ Auto-derived via adjunction
Provenance tracking ❌ ❌ ✅ Full audit trail
Compositionality ❌ ❌ ✅ Proven by construction

---

Core Concept

Treat your neural network pipeline as a category:

· Objects: Typed tensor descriptors (shape + semantics)
· Morphisms: Differentiable transformations
· Composition: The chain rule, checked mathematically

The math guarantees: if two morphisms compose, the result is correct—no manual verification required.

```python
from cognitive_packets import ActivationType, CP, CompositionError
import torch.nn as nn

cat = CP()

# Types are first-class citizens — not just shape tuples
A = ActivationType(shape=(-1, 256), tag="encoder_input")
B = ActivationType(shape=(-1, 64),  tag="encoder_output")
C = ActivationType(shape=(-1, 128), tag="decoder_input")  # incompatible with B

f = cat.morphism(A, B, nn.Linear(256, 64), name="encoder")
g = cat.morphism(C, ..., nn.Linear(128, 10), name="decoder")

# Raises CompositionError HERE — before any training starts
pipeline = cat.sequential([f, g])
# CompositionError: Cannot compose morphisms: type mismatch
#   'encoder' has codomain:  encoder_output[-1×64]@cpu
#   'decoder' has domain:    decoder_input[-1×128]@cpu
```

The forward/backward duality isn't a convention—it's a theorem. The adjunction F ⊣ B means every valid forward morphism has a canonical, automatically-derived backward pass.

---

Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/crisp-backprop.git
cd crisp-backprop
pip install -e ".[dev]"
```

```python
import torch
import torch.nn as nn
from cognitive_packets import ActivationType, CP, CognitivePacket, Adjunction

cat = CP()

# 1. Declare types — shape + semantics
T_in  = ActivationType(shape=(-1, 128), tag="input")
T_h   = ActivationType(shape=(-1, 64),  tag="hidden")
T_out = ActivationType(shape=(-1, 10),  tag="logits")

# 2. Declare morphisms — type-annotated layers
f1 = cat.morphism(T_in, T_h,  nn.Linear(128, 64), name="layer1")
f2 = cat.morphism(T_h,  T_out, nn.Linear(64, 10),  name="layer2")

# 3. Compose — type-checked at definition
pipeline = cat.sequential([f1, f2])

# 4. Run on typed packets
x = torch.randn(32, 128, requires_grad=True)
packet = CognitivePacket(data=x, type=T_in)
output = pipeline(packet)

print(output.provenance)  # ['layer1', 'layer2'] — every packet knows its history

# 5. Derive backward pass automatically
adj = Adjunction()
backward_layer1 = adj.transpose(f1)
# Morphism(grad_hidden → grad_input) — derived, not handwritten
```

---

The Math (Without the Pain)

You don't need category theory to use this. But here's the intuition:

A category precisely describes composable things. Functions compose. Lego bricks compose. Neural networks compose. PyTorch's composition has no rules—it lets you snap incompatible bricks together and only complains when you try to play.

Category CP adds the rules:

· Every layer declares its input (domain) and output (codomain)
· Two layers chain only if the output type of the first exactly matches the input type of the second
· This is checked at definition time, not runtime

The adjunction F ⊣ B is the categorical way of saying: "the forward and backward passes are two sides of the same coin."

Formally: Hom(F(A), G) ≅ Hom(A, B(G))

For every forward layer f, there exists a unique, provably correct backward layer B(f) derived automatically via PyTorch's autograd. If your forward pass is correct, the backward pass is guaranteed correct too.

Categorical laws verified computationally:

· Identity: id_B ∘ f = f = f ∘ id_A
· Associativity: (h ∘ g) ∘ f = h ∘ (g ∘ f)
· Type closure: Composition of valid morphisms always yields a valid morphism

---

Demos & Tests

```bash
# Run the type safety demo
python examples/demo_type_safety.py
# Shows side-by-side comparison: naive PyTorch vs Cognitive Packets
# Covers shape mismatch, semantic mismatch, correct pipelines, and parallel composition

# Run the test suite
pytest tests/ -v
# 20+ tests covering categorical laws, adjunction properties, provenance, serialization
```

---

Project Structure

```
crisp-backprop/
├── cognitive_packets/
│   ├── types.py          # PacketType, ActivationType, GradientType, LossType
│   ├── morphism.py       # Morphism: domain, codomain, compose(), identity()
│   ├── packet.py         # CognitivePacket: typed tensor + provenance + serialize
│   ├── category.py       # CP: sequential(), parallel() (f ⊗ g), law verification
│   └── adjunction.py     # F ⊣ B: unit η, counit ε, transpose, triangle identities
├── tests/
│   └── test_categorical_laws.py
├── examples/
│   └── demo_type_safety.py
└── setup.py
```

---

Related Work

Cognitive Packets extends three bodies of work into the distributed setting:

· Fong, Spivak, Tuyéras (2019) — Backprop as Functor: categorical semantics of backpropagation for centralized training. This adds distributed packet communication, semantic types, provenance, and monoidal parallelism.
· McMahan et al. (2017) — Federated Averaging: the engineering baseline this framework formalizes.
· Elliott (2018) — The Simple Essence of Automatic Differentiation: connection between AD and cartesian differential categories.

---

Current Scope

This is a research prototype. What it is not (yet):

· ❌ Not a drop-in replacement for torch.distributed
· ❌ Not benchmarked for performance overhead
· ⚠️ Adjunction proof covers type structure; full coherence for arbitrary compositions is ongoing
· ⚠️ Parallel composition simulated on CPU—production would dispatch to separate devices/nodes

What it is: A working implementation of the categorical foundations, with enforced laws, a real adjunction, and a clear path to production.

---

Future Directions

· Traced monoidal structure for recurrent networks (LSTMs, SSMs)
· Byzantine-fault-tolerant gradient routing with topological guarantees
· Real distributed execution via typed gRPC packet serialization
· Quantum and neuromorphic backend integration

---

Citation

```bibtex
@misc{nitsch2025cognitivepackets,
  title  = {Cognitive Packets: A Categorical Foundation for Distributed Backpropagation},
  author = {Nitsch, Kurt},
  year   = {2025},
  note   = {Research prototype, v0.1.0}
}
```

---

License

MIT
