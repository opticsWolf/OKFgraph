---
type: note
title: Rust Borrow Checker
description: affine types lifetimes and the borrow checker
tags: [programming, rust]
---

The Rust borrow checker enforces affine types through lifetimes: each value
has one owner, borrows are checked at compile time, and data races become
impossible. See [Concurrency](concurrency.md).
