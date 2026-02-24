"""
morphism.py — Morphisms of the category CP
============================================

A morphism f: A → B in CP is a differentiable function from packets
of type A to packets of type B. Each morphism wraps a PyTorch module
and carries:
  - domain:   the PacketType of its input
  - codomain: the PacketType of its output
  - fn:       the forward computation (a callable)
  - fn_grad:  the backward computation (optional; derived via autograd if absent)

Composition of morphisms (g ∘ f): A → C is defined when
codomain(f) == domain(g), and is itself a morphism.

Categorical laws enforced:
  1. Type safety:    composition fails loudly if types don't match
  2. Associativity:  (h ∘ g) ∘ f == h ∘ (g ∘ f)  (by construction)
  3. Identity:       id_A ∘ f == f == f ∘ id_B
"""

from __future__ import annotations
from typing import Callable, Optional, List
import torch
import torch.nn as nn
from .types import PacketType


class CompositionError(TypeError):
    """
    Raised when morphism composition fails type checking.

    This is the key safety guarantee of the categorical framework:
    composing incompatible morphisms raises CompositionError at
    definition time, not at runtime during training.
    """
    pass


class Morphism:
    """
    A morphism f: domain → codomain in the category CP.

    Parameters
    ----------
    domain : PacketType
        The input type of this morphism.
    codomain : PacketType
        The output type of this morphism.
    fn : Callable[[torch.Tensor], torch.Tensor]
        The forward function. Should be differentiable (use torch ops).
    name : str
        Human-readable label for debugging and graph visualization.

    Example
    -------
    >>> import torch
    >>> from cognitive_packets import ActivationType, GradientType, Morphism
    >>>
    >>> A = ActivationType(shape=(-1, 128))
    >>> B = ActivationType(shape=(-1, 64))
    >>>
    >>> linear = nn.Linear(128, 64)
    >>> f = Morphism(
    ...     domain=A,
    ...     codomain=B,
    ...     fn=linear,
    ...     name="linear_128→64"
    ... )
    """

    def __init__(
        self,
        domain: PacketType,
        codomain: PacketType,
        fn: Callable[[torch.Tensor], torch.Tensor],
        name: str = "morphism",
    ):
        self.domain = domain
        self.codomain = codomain
        self.fn = fn
        self.name = name
        self._composed_from: List[str] = [name]

    def __call__(self, packet: "CognitivePacket") -> "CognitivePacket":
        """
        Apply this morphism to a CognitivePacket.

        Performs runtime type validation before computing, ensuring
        the concrete tensor matches the declared domain type.
        """
        from .packet import CognitivePacket

        # Type check
        if not self.domain.is_compatible(packet.type):
            raise CompositionError(
                f"Runtime type mismatch in morphism '{self.name}':\n"
                f"  Expected input of type: {self.domain}\n"
                f"  Got:                    {packet.type}"
            )

        # Forward compute with gradient tracking
        with torch.enable_grad():
            output_tensor = self.fn(packet.data)

        # Validate output shape
        if not self.codomain.validate_tensor(output_tensor):
            raise CompositionError(
                f"Morphism '{self.name}' produced wrong output type:\n"
                f"  Expected: {self.codomain}\n"
                f"  Got shape: {tuple(output_tensor.shape)}, dtype: {output_tensor.dtype}"
            )

        return CognitivePacket(
            data=output_tensor,
            type=self.codomain,
            provenance=packet.provenance + [self.name],
        )

    def then(self, other: "Morphism") -> "Morphism":
        """
        Compose self with other: returns (other ∘ self): A → C.

        This is the forward-reading composition: self runs first,
        other runs second. Equivalent to other.compose(self).

        Raises CompositionError if self.codomain != other.domain.
        """
        return other.compose(self)

    def compose(self, other: "Morphism") -> "Morphism":
        """
        Compose other with self: returns (self ∘ other): A → C.

        Mathematical composition: self runs second, other runs first.
        Standard notation: (self ∘ other)(x) = self(other(x))

        Raises
        ------
        CompositionError
            If other.codomain is not compatible with self.domain.
        """
        # THE KEY SAFETY CHECK — happens at definition time
        if not other.codomain.is_compatible(self.domain):
            raise CompositionError(
                f"Cannot compose morphisms: type mismatch\n"
                f"  '{other.name}' has codomain: {other.codomain}\n"
                f"  '{self.name}' has domain:    {self.domain}\n"
                f"  These are incompatible — composition is undefined."
            )

        outer_fn = self.fn
        inner_fn = other.fn
        composed_name = f"({self.name} ∘ {other.name})"

        def composed_fn(x: torch.Tensor) -> torch.Tensor:
            return outer_fn(inner_fn(x))

        result = Morphism(
            domain=other.domain,
            codomain=self.codomain,
            fn=composed_fn,
            name=composed_name,
        )
        result._composed_from = other._composed_from + self._composed_from
        return result

    @classmethod
    def identity(cls, packet_type: PacketType) -> "Morphism":
        """
        Construct the identity morphism id_A: A → A.

        Satisfies: id_A ∘ f = f = f ∘ id_B for any f: B → A.
        """
        return cls(
            domain=packet_type,
            codomain=packet_type,
            fn=lambda x: x,
            name=f"id_{packet_type.tag}",
        )

    def __repr__(self) -> str:
        return f"Morphism({self.domain} → {self.codomain}, name='{self.name}')"
