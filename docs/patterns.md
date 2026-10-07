The abstract/final pattern sits at the intersection of several ideas worth knowing deeply:
Type theory over OOP
The pattern is fundamentally about types as contracts, not objects as stateful actors. The key distinction: classical OOP uses inheritance to share and mutate behaviour; abstract/final uses it purely to express "this thing satisfies this interface." The intellectual lineage here runs through Liskov Substitution Principle (LSP) and into modern type theory. Worth reading: Barbara Liskov's original 1987 paper, and then Pierce's Types and Programming Languages if you want to go deep.
Nominal vs structural subtyping
The pattern commits hard to nominal subtyping — relationships are declared explicitly, not inferred from shape. This is the opposite of duck typing or TypeScript's structural typing. Understanding why you'd choose one over the other (invariant enforcement vs flexibility) is a genuinely important design judgement to develop.
Composition over inheritance
The "tweak by wrapping, not subclassing" rule is one of the Gang of Four's central recommendations (Design Patterns, 1994), and the pattern enforces it mechanically. The Decorator and Strategy patterns from that book are both natural fits here.
Algebraic data types and sum types
The abstract-or-final rule essentially makes your class hierarchy a closed type. This is the OOP approximation of algebraic data types (ADTs) — the same idea Rust expresses with enum, Haskell with data, and Swift with enum with associated values. Understanding ADTs will make this feel much more principled than it might first appear.
Immutability and value semantics
Equinox modules are immutable, which is why the pattern works so cleanly — there's no "object identity" to worry about, just values. This connects to the broader functional programming philosophy: prefer values over references, make state explicit and local. Rich Hickey's talk The Value of Values is an excellent entry point.
Interface segregation
The pattern naturally pushes you toward small, focused abstract classes rather than large monolithic ones. This is the I in SOLID (Interface Segregation Principle) — clients shouldn't depend on interfaces they don't use.

If I were to suggest a learning path from here, I'd go in this order:

SOLID principles — you'll recognise several already from this pattern; understanding all five gives you vocabulary
Gang of Four Design Patterns — especially Decorator, Strategy, Template Method, and Composite
A language with ADTs — even a little Rust or Haskell makes the type-theoretic underpinning click in a way reading about it doesn't
"A Philosophy of Software Design" by John Ousterhout — short, opinionated, directly relevant to the readability goals this pattern serves