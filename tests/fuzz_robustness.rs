#!/usr/bin/env rust
//! M6 robustness properties for untrusted text, paths and op parameters.

use okfgraph::error::error_kind;
use okfgraph::markdown::{parse_frontmatter, parse_yaml, split_into_chunks, word_count};
use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use okfgraph::store::absolute_path;
use okfgraph::util::safe_bundle_id;
use proptest::prelude::*;
use serde_json::{Map, Value as Json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn text_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            prop::char::range('a', 'z'),
            prop::char::range('A', 'Z'),
            prop::char::range('0', '9'),
            Just(' '),
            Just('\n'),
            Just('\r'),
            Just('\t'),
            Just('-'),
            Just(':'),
            Just('['),
            Just(']'),
            Just('|'),
            Just('#'),
            Just('/'),
            Just('\\'),
            Just('é'),
            Just('中'),
        ],
        0..400,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

fn json_value_strategy() -> impl Strategy<Value = Json> {
    prop_oneof![
        Just(Json::Null),
        any::<bool>().prop_map(Json::Bool),
        (0i64..1000).prop_map(|n| Json::Number(n.into())),
        prop::collection::vec(prop::char::range('a', 'z'), 0..12)
            .prop_map(|chars| Json::String(chars.into_iter().collect())),
        prop::collection::vec(json_scalar(), 0..3)
            .prop_map(|items| Json::Array(items.into_iter().map(Json::from).collect())),
    ]
}

fn json_scalar() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range('a', 'z'), 0..8)
        .prop_map(|chars| chars.into_iter().collect())
}

fn params_strategy() -> impl Strategy<Value = Map<String, Json>> {
    prop::collection::vec(
        (
            prop::collection::vec(prop::char::range('a', 'z'), 1..12),
            json_value_strategy(),
        ),
        0..6,
    )
    .prop_map(|entries| {
        entries
            .into_iter()
            .map(|(key, value)| (key.into_iter().collect(), value))
            .collect()
    })
}

fn fixture_router() -> Router {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("parity/fixtures/oracle-v10.db");
    // Keep the proptest body free of global fixture mutation.
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("oracle.db");
    std::fs::copy(source, &path).expect("copy fixture");
    // The directory must outlive the Router, so leak it for the test process.
    let path = path.to_string_lossy().into_owned();
    std::mem::forget(dir);
    let settings = Settings::default()
        .with("db_path", Json::String(path))
        .expect("settings");
    Router::open(&settings).expect("router")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn frontmatter_never_panics_and_never_returns_crlf(text in text_strategy()) {
        let parsed = parse_frontmatter(&text);
        prop_assert!(!parsed.content.contains("\r\n"));
        prop_assert!(serde_json::to_string(&parsed.metadata).is_ok());
    }

    #[test]
    fn yaml_never_panics_and_stays_json(text in text_strategy()) {
        let value = parse_yaml(&text);
        prop_assert!(serde_json::to_string(&value).is_ok());
    }

    #[test]
    fn chunk_offsets_tile_within_the_body(text in text_strategy()) {
        let chunks = split_into_chunks(&text, "doc", 24, word_count);
        let mut previous_end = 0usize;
        for (index, chunk) in chunks.iter().enumerate() {
            prop_assert_eq!(chunk.chunk_index, index as i64);
            prop_assert!(chunk.start_offset >= 0);
            prop_assert!(chunk.end_offset >= chunk.start_offset);
            prop_assert!((chunk.end_offset as usize) <= text.len());
            prop_assert!((chunk.start_offset as usize) >= previous_end);
            previous_end = chunk.end_offset as usize;
        }
    }

    #[test]
    fn safe_bundle_id_is_total(parts in prop::collection::vec(prop::collection::vec(
        prop_oneof![
            prop::char::range('a', 'z'),
            Just('.'),
            Just('/'),
            Just('\\'),
            Just(':'),
        ], 0..8), 1..4)) {
        let id = parts
            .into_iter()
            .map(|part| part.into_iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("/");
        let safe = safe_bundle_id(&id);
        if safe {
            prop_assert!(Path::new(&id).components().all(|c| c.as_os_str() != ".."));
        }
    }

    #[test]
    fn absolute_path_is_total(path in prop::collection::vec(
        prop_oneof![
            prop::char::range('a', 'z'),
            Just('/'),
            Just('\\'),
            Just('.'),
            Just(':'),
        ], 0..80)) {
        let path: String = path.into_iter().collect();
        let resolved: PathBuf = absolute_path(Path::new(&path));
        prop_assert!(!resolved.as_os_str().is_empty());
    }

    #[test]
    fn read_op_params_are_total(op in prop_oneof![
        Just("get_by_id"),
        Just("read"),
        Just("traverse"),
        Just("list_images"),
        Just("get_image"),
        Just("list_directory"),
        Just("list_broken_links"),
        Just("list_deleted"),
        Just("pin_info"),
        Just("lanes"),
    ], params in params_strategy()) {
        let router = fixture_router();
        match router.call(op, &params) {
            Ok(value) => prop_assert!(serde_json::to_string(&value).is_ok()),
            Err(err) => prop_assert!(error_kind(&err.code).is_some(), "code {}", err.code),
        }
    }

    #[test]
    fn unknown_ops_stay_typed(name in text_strategy()) {
        let router = fixture_router();
        let err = router
            .call(&format!("not_an_op_{name}"), &Map::new())
            .expect_err("unknown op must fail");
        prop_assert_eq!(err.code, "BAD_VALUE");
        prop_assert!(err.fields.contains_key("op"));
    }

    #[test]
    fn closed_error_codes_are_stable(code in prop::collection::vec(
        prop::char::range('A', 'Z'), 1..12)) {
        let requested: String = code.into_iter().collect();
        let err = okfgraph::error::OkfError::new(requested.clone(), "fuzz");
        prop_assert!(error_kind(&err.code).is_some());
        if error_kind(&requested).is_none() {
            prop_assert_eq!(err.code, "INTERNAL");
            prop_assert_eq!(err.fields.get("requested_code"), Some(&Json::String(requested)));
        }
    }
}

#[test]
fn known_read_ops_are_all_fuzzed() {
    let covered = [
        "get_by_id",
        "read",
        "traverse",
        "list_images",
        "get_image",
        "list_directory",
        "list_broken_links",
        "list_deleted",
        "pin_info",
        "lanes",
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    for op in covered {
        assert!(okfgraph::ops::is_op(op), "{op} must be a registered op");
    }
}
