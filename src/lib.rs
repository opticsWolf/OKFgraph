//! OKFgraph Rust core.
//!
//! The Python binding in `bindings/python` is a thin adapter over this crate.
//! All graph, retrieval, ingest, and serialization logic belongs here.
//!
//! `OkfError` is a intentionally value-like public error (code/message/fields/
//! remedy/data); boxing it would hurt every call site for a lint-only win.
#![allow(clippy::result_large_err)]

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const SCHEMA_VERSION: u16 = 10;

pub fn version() -> &'static str {
    VERSION
}

pub fn schema_version() -> u16 {
    SCHEMA_VERSION
}

pub mod embed;
pub mod error;
pub mod markdown;
pub mod model;
pub mod ops;
pub mod pycompat;
pub mod search;
pub mod settings;
pub mod store;
pub mod util;

#[cfg(feature = "cli")]
pub mod cli;
#[cfg(any(feature = "embed", feature = "pdf"))]
pub mod ingest;
#[cfg(feature = "server")]
pub mod server;
