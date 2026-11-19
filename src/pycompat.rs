//! Python-compatible value serialization.
//!
//! Python's `json.dumps(..., default=str)` and `datetime.isoformat()` are part
//! of the v10 on-disk contract. `serde_json`'s defaults are not: they emit UTF-8
//! instead of `\\uXXXX`, compact separators, and `1e-7` instead of `1e-07`.
//! This module is deliberately small and pinned by `parity/golden/pycompat.json`.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

/// A JSON value plus the two Python object types that leak through `default=str`.
#[derive(Debug, Clone)]
pub enum PyValue {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    List(Vec<PyValue>),
    Dict(Vec<(String, PyValue)>),
    Date(String),
    DateTime(String),
    Other(String),
}

impl PyValue {
    /// Parse the tagged encoding produced by `parity/tools/gen_pycompat.py`.
    pub fn from_golden(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(*b),
            Value::Number(n) if n.is_i64() => Self::Int(n.as_i64().expect("i64")),
            Value::Number(n) if n.is_u64() => Self::UInt(n.as_u64().expect("u64")),
            Value::Number(n) => Self::Float(n.as_f64().expect("f64")),
            Value::String(s) => Self::Str(s.clone()),
            Value::Array(items) => Self::List(items.iter().map(Self::from_golden).collect()),
            Value::Object(obj) => {
                if obj.len() == 2 && obj.contains_key("__type__") && obj.contains_key("value") {
                    let kind = obj["__type__"].as_str().unwrap_or_default();
                    let text = obj["value"].as_str().unwrap_or_default().to_string();
                    return match kind {
                        "date" => Self::Date(text),
                        "datetime" => Self::DateTime(text),
                        _ => Self::Other(text),
                    };
                }
                Self::Dict(
                    obj.iter()
                        .map(|(k, v)| (k.clone(), Self::from_golden(v)))
                        .collect(),
                )
            }
        }
    }

    /// `json.dumps(value, sort_keys=..., default=str)`.
    pub fn to_python_json(&self, sort_keys: bool) -> String {
        let mut out = String::new();
        self.write_json(&mut out, sort_keys);
        out
    }

    /// `extra_value(value)`: raw strings, ISO dates, otherwise Python JSON.
    pub fn extra_value(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Date(s) | Self::DateTime(s) => s.clone(),
            other => other.to_python_json(false),
        }
    }

    /// `str(value)` for objects passed to `json.dumps(default=str)`.
    fn python_str(&self) -> String {
        match self {
            Self::DateTime(s) => s.replacen('T', " ", 1),
            Self::Date(s) => s.clone(),
            Self::Str(s) => s.clone(),
            Self::Other(s) => s.clone(),
            other => other.to_python_json(false),
        }
    }

    fn write_json(&self, out: &mut String, sort_keys: bool) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::UInt(u) => out.push_str(&u.to_string()),
            Self::Float(f) => out.push_str(&python_float(*f)),
            Self::Date(_) | Self::DateTime(_) | Self::Other(_) => {
                write_python_string(&self.python_str(), out)
            }
            Self::Str(s) => write_python_string(s, out),
            Self::List(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    item.write_json(out, sort_keys);
                }
                out.push(']');
            }
            Self::Dict(entries) => {
                let mut ordered: Vec<&(String, PyValue)> = entries.iter().collect();
                if sort_keys {
                    ordered.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (idx, (key, value)) in ordered.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(", ");
                    }
                    write_python_string(key, out);
                    out.push_str(": ");
                    value.write_json(out, sort_keys);
                }
                out.push('}');
            }
        }
    }
}

impl PartialEq for PyValue {
    fn eq(&self, other: &Self) -> bool {
        self.to_python_json(false) == other.to_python_json(false)
    }
}
impl Eq for PyValue {}
impl PartialOrd for PyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PyValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_python_json(false).cmp(&other.to_python_json(false))
    }
}

/// Free-function form of `PyValue::extra_value`.
pub fn extra_value(value: &PyValue) -> String {
    value.extra_value()
}

/// `str(datetime)`/`str(date)` from their ISO form.
pub fn python_datetime_str(iso: &str) -> String {
    iso.replacen('T', " ", 1)
}

/// `datetime.isoformat()`; the tagged fixture already stores this form.
pub fn python_datetime_isoformat(iso: &str) -> String {
    iso.to_string()
}

/// SHA-256 of `json.dumps(meta, sort_keys=True, default=str) + "\0" + body`.
pub fn content_sig(data: &PyValue, body: &str) -> String {
    let meta = match data {
        PyValue::Dict(entries) => PyValue::Dict(
            entries
                .iter()
                .filter(|(key, _)| key != "body" && key != "embedding")
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    };
    let blob = format!("{}\0{}", meta.to_python_json(true), body);
    hex_encode(&Sha256::digest(blob.as_bytes()))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Python-style `repr(float)` for finite values.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value == f64::INFINITY {
        return "Infinity".to_string();
    }
    if value == f64::NEG_INFINITY {
        return "-Infinity".to_string();
    }
    let text = format!("{:?}", value);
    if let Some(pos) = text.find('e') {
        let mantissa = &text[..pos];
        let exp: i32 = text[pos + 1..].parse().expect("valid Rust exponent");
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{}e{}{:02}", mantissa, sign, exp.abs());
    }
    text
}

/// JSON string escaping with `ensure_ascii=True`.
fn write_python_string(value: &str, out: &mut String) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            c if (c as u32) <= 0xFFFF => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => {
                let v = c as u32 - 0x10000;
                let hi = 0xD800 + (v >> 10);
                let lo = 0xDC00 + (v & 0x3FF);
                out.push_str(&format!("\\u{:04x}\\u{:04x}", hi, lo));
            }
        }
    }
    out.push('"');
}

/// Convert a plain `serde_json::Value` (no tagged date/datetime objects).
pub fn from_json(value: &Value) -> PyValue {
    match value {
        Value::Null => PyValue::Null,
        Value::Bool(b) => PyValue::Bool(*b),
        Value::Number(n) if n.is_i64() => PyValue::Int(n.as_i64().expect("i64")),
        Value::Number(n) if n.is_u64() => PyValue::UInt(n.as_u64().expect("u64")),
        Value::Number(n) => PyValue::Float(n.as_f64().expect("f64")),
        Value::String(s) => PyValue::Str(s.clone()),
        Value::Array(items) => PyValue::List(items.iter().map(from_json).collect()),
        Value::Object(obj) => {
            PyValue::Dict(obj.iter().map(|(k, v)| (k.clone(), from_json(v))).collect())
        }
    }
}

/// Helper used by tests and future `ops` code to strip volatile fields.
pub fn without_keys(value: &PyValue, keys: &[&str]) -> PyValue {
    match value {
        PyValue::Dict(entries) => PyValue::Dict(
            entries
                .iter()
                .filter(|(key, _)| !keys.contains(&key.as_str()))
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/parity/golden/pycompat.json");

    #[test]
    fn matches_python_golden() {
        let text = std::fs::read_to_string(GOLDEN).expect("golden fixture");
        let golden: Value = serde_json::from_str(&text).expect("golden JSON");
        let mut checked = 0usize;

        for case in golden["json_dumps"].as_array().expect("json cases") {
            let value = PyValue::from_golden(&case["value"]);
            let got = value.to_python_json(case["sort_keys"].as_bool().unwrap_or(false));
            assert_eq!(
                got,
                case["expected"].as_str().expect("expected"),
                "json_dumps: {}",
                case["name"]
            );
            checked += 1;
        }

        for case in golden["extra_value"].as_array().expect("extra cases") {
            let value = PyValue::from_golden(&case["value"]);
            assert_eq!(
                value.extra_value(),
                case["expected"].as_str().expect("expected"),
                "extra_value: {}",
                case["name"]
            );
            checked += 1;
        }

        for case in golden["content_sig"].as_array().expect("sig cases") {
            let data = PyValue::from_golden(&case["data"]);
            assert_eq!(
                content_sig(&data, case["body"].as_str().expect("body")),
                case["expected"].as_str().expect("expected"),
                "content_sig: {}",
                case["name"]
            );
            checked += 1;
        }

        assert_eq!(
            python_datetime_str("2024-01-02T03:04:05"),
            golden["datetime"]["str"].as_str().expect("datetime str")
        );
        assert_eq!(
            python_datetime_isoformat("2024-01-02T03:04:05"),
            golden["datetime"]["isoformat"].as_str().expect("isoformat")
        );
        assert_eq!(checked, 17, "expected golden cases, got {checked}");
    }

    #[test]
    fn escapes_non_bmp_as_surrogate_pair() {
        assert_eq!(write_json_string("😀"), "\"\\ud83d\\ude00\"");
    }

    fn write_json_string(s: &str) -> String {
        let mut out = String::new();
        write_python_string(s, &mut out);
        out
    }
}
