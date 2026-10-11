This is a concise code style guide for the qvl project. Contributors or agents should follow these guidelines unless there is a compelling reason to deviate.

If not specified here, just follow the code style of the existing code around the code you are writing.

## Formatting

Run `pixi run format <paths...>` to format specific files or directories. Pass `--language <cpp|python|markdown|json|toml>` to override language detection for every input.

## Identifier Naming

See [.clang-tidy](../.clang-tidy).

## Header Inclusion

The headers should be grouped with the following order:

```c++
/// interface header of the current .cpp file
#include "interface.h"

// std headers
#include <cstdlib>
#include <vector>

// other 3rd party headers
#include <fmt/format.h>

// other headers from the qvl project
#include <lighter/http/client.h>
#include <lighter/async/async.h>
```

## Structs, Custom Types

Always use `struct`, do not use `class` for custom types. This way, we don't need to write `public:` for every struct, and when we need to inherit, we don't need to write `public` in the inheritance list.

## Polymorphism

We generally avoid virtual functions in this project. If you need polymorphism, use `ngcpp-proxy`. It provides a dynamic dispatch mechanism based on fat pointers, much closer to Rust's trait objects. It also supports static dispatch with `PRO_DEF_MEM_DISPATCH` and `PRO_DEF_FREE_DISPATCH`. An example:

```c++
struct RenderContext {};

PRO_DEF_MEM_DISPATCH(DrawDispatch, draw); // generate a templated dispatch funtion draw at global scope

inline constexpr DrawDispatch draw{};

struct DrawableFacade
    : pro::facade_builder
        ::add_convention<
            DrawDispatch,
            void(RenderContext&) const>
        ::build
{};

struct Mesh {
    void draw(RenderContext&) const
    {
        // ...
    }
};

template<class T>
concept Drawable = requires(
    const T& value,
    RenderContext& ctx)
{
    { draw(value, ctx) } -> std::same_as<void>;
};

void render_static(
    const Drawable auto& value,
    RenderContext& ctx)
{
    draw(value, ctx);        // statically dispatched
}

void render_dynamic(
    const pro::proxy<DrawableFacade>& value,
    RenderContext& ctx)
{
    draw(*value, ctx);       // dynamically dispatched by Proxy
}
```

This way we can keep the main structs in our codebase as plain non virtual structs, with better memory layout and data locality, and still have polymorphism when needed.

## Callable Wrappers

Use the standard callable wrappers according to ownership:

- Use `std::copyable_function` for owning callbacks that are copied.
- Use `std::move_only_function` for owning callbacks that are transferred or may capture move-only state.
- Use `std::function_ref` only for synchronous, non-storing callback parameters whose target is guaranteed to outlive the call.

## Basic Types

Use the short convenience types defined in `lighter/types.hpp`; for example, the project spelling for an unsigned size is `usize`.

These types are directly available inside the `lighter` namespace. In another namespace, `using namespace lighter::types;` is generally recommended.

## Assertions, Contracts and Panic

Do not use `<cassert>` or `assert()`. Use C++26 contracts for programmer obligations and internal invariants:

- Use `pre` and `post` for interface-level preconditions and postconditions. Place them on the first declaration, normally in the public header, and do not repeat them on an out-of-line definition.
- Use `contract_assert` for internal invariants at a particular point in an implementation.

Production builds use `-fcontract-evaluation-semantic=ignore`, while development and test builds use `enforce`. Consequently, correctness in production must not depend on a contract predicate being evaluated.

Use `lighter::check(condition, message)` for conditional failures that must remain checked in production. Use `lighter::panic(message)` for unconditional fatal paths. Prefer `lighter::check` over an `if` statement whose only purpose is to call `lighter::panic`.

## Error Handling

Generally we will have 3 kinds of failures in our codebase:

- **Expected failures**. These are failures that are expected to happen and should be handled by the caller. For example, a file not found error when trying to open a file. These failures should be returned as `lighter::Outcome` (or `std::expected` if cancellation is not needed).
- **Rare failures whose only sensible handler is a distant recovery boundary**. These are failures that hardly happen and cannot be handled by the immediate caller. These should be thrown as runtime exceptions.
- **Fatal failures**. These are the failures that can't be handled meaningfully anywhere in the codebase. For example, an OOM error. These should `panic` .

Distinguish these 3 kinds of failures by asking who can act on the failure. If the immediate caller can act on it, return `lighter::Outcome`. If only a distant recovery boundary can act on it, throw an exception. If no one can act on it, `panic`.

## Synchronization in Tests

We have a couple of mock tests for concurrent code in our codebase. These tests must be deterministic and not rely on timing assumptions. You should coordinate concurrency through observable state transitions (events, barriers, callbacks, process state waits, or mock handshakes), never through an arbitrary sleep or an assumption that enough wall-clock time has passed.

## Commit Messages

The headline of a commit should follow the format: `<type>(<scope>): <subject>`, type or scope can be omitted if not applicable.

If the commit is big enough and you cannot describe all important changes in the headline, you can add a body to the commit message. The body should be separated from the headline by a blank line, and preferably be a list of bullet points.

An example:

```
refactor(tui): centralize slash command parsing

- Lexes REPL input once into either a user prompt or an exact command name and argument string.
- Resolves commands and aliases centrally and keeps command recognition out of individual handlers.
- Reports unknown slash commands as command errors.
- Supports a leading double slash as an escaped slash-prefixed user prompt.
```
