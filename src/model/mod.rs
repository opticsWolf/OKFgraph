//! Concept/Chunk/Image models and Python-compatible value serialization.

pub use crate::pycompat::{
    content_sig, extra_value, from_json, python_datetime_isoformat, python_datetime_str,
    without_keys, PyValue,
};
pub use crate::SCHEMA_VERSION as MODEL_SCHEMA_VERSION;

use crate::error::{OkfError, Result};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use time::{Date, Month, PrimitiveDateTime, Time};

pub const GRAPH_TIMES: &[&str] = &["created_at", "updated_at", "effective_at", "content_sig"];

/// Normalize a frontmatter `tags` value the same way Pydantic does.
pub fn normalize_tags(value: &Value) -> Vec<String> {
    match value {
        Value::Null => Vec::new(),
        Value::String(s) if s.is_empty() => Vec::new(),
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => items.iter().map(python_scalar_str).collect(),
        other => vec![python_scalar_str(other)],
    }
}

fn python_scalar_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Number(n) => n.to_string(),
        other => crate::pycompat::PyValue::from_golden(other).to_python_json(false),
    }
}

/// A naive UTC timestamp, serialized with Python's `str(datetime)` spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDateTime(PrimitiveDateTime);

impl PyDateTime {
    pub fn new(dt: PrimitiveDateTime) -> Self {
        Self(dt)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim().trim_matches('"');
        if text.is_empty() {
            return Err(OkfError::new("BAD_VALUE", "empty timestamp"));
        }
        let normalized = text.replace('T', " ");
        let normalized = normalized.strip_suffix('Z').unwrap_or(&normalized).trim();
        // Strip a simple +HH:MM offset: all stored times are naive UTC.
        let naive = if normalized.len() >= 6 {
            let bytes = normalized.as_bytes();
            if bytes[bytes.len() - 6] == b'+' || bytes[bytes.len() - 6] == b'-' {
                &normalized[..normalized.len() - 6]
            } else {
                normalized
            }
        } else {
            normalized
        };
        let (date_part, time_part) = naive.split_once(' ').unwrap_or((naive, "00:00:00"));
        let mut date_parts = date_part.split('-');
        let year = date_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| bad_ts(text))?;
        let month = date_parts
            .next()
            .and_then(|s| s.parse::<u8>().ok())
            .and_then(|m| Month::try_from(m).ok())
            .ok_or_else(|| bad_ts(text))?;
        let day = date_parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| bad_ts(text))?;
        let date = Date::from_calendar_date(year, month, day).map_err(|_| bad_ts(text))?;
        let mut time_parts = time_part.split(':');
        let hour = time_parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minute = time_parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let seconds = time_parts.next().unwrap_or("0");
        let (sec, nanos) = match seconds.split_once('.') {
            Some((sec, frac)) => {
                let sec: u8 = sec.parse().map_err(|_| bad_ts(text))?;
                let mut frac = frac.to_string();
                frac.truncate(9);
                while frac.len() < 9 {
                    frac.push('0');
                }
                (sec, frac.parse::<u32>().map_err(|_| bad_ts(text))?)
            }
            None => (seconds.parse().map_err(|_| bad_ts(text))?, 0),
        };
        let time = Time::from_hms_nano(hour, minute, sec, nanos).map_err(|_| bad_ts(text))?;
        Ok(Self(PrimitiveDateTime::new(date, time)))
    }

    pub fn isoformat(&self) -> String {
        let date = self.0.date();
        let time = self.0.time();
        let mut text = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            date.year(),
            date.month() as u8,
            date.day(),
            time.hour(),
            time.minute(),
            time.second()
        );
        if time.nanosecond() != 0 {
            text.push('.');
            text.push_str(format!("{:09}", time.nanosecond()).trim_end_matches('0'));
        }
        text
    }

    pub fn as_naive(&self) -> PrimitiveDateTime {
        self.0
    }
}

fn bad_ts(text: &str) -> OkfError {
    OkfError::new("BAD_VALUE", format!("invalid timestamp: {text}"))
}

impl Serialize for PyDateTime {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&python_datetime_str(&self.isoformat()))
    }
}

impl<'de> Deserialize<'de> for PyDateTime {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptModel {
    pub id: String,
    #[serde(rename = "type")]
    pub concept_type: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub resource: Option<String>,
    pub tags: Vec<String>,
    pub timestamp: Option<PyDateTime>,
    pub body: String,
    pub embedding: Option<Vec<f64>>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ConceptModel {
    pub fn new(id: impl Into<String>, concept_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            concept_type: concept_type.into(),
            title: None,
            description: None,
            resource: None,
            tags: Vec::new(),
            timestamp: None,
            body: String::new(),
            embedding: None,
            extra: BTreeMap::new(),
        }
    }

    /// Agent/CLI-safe dump: everything except the embedding vector.
    pub fn public_value(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("concept serializes");
        if let Some(map) = value.as_object_mut() {
            map.remove("embedding");
        }
        value
    }

    pub fn export_frontmatter(&self) -> (Value, String) {
        let mut data = self.public_value();
        if let Some(map) = data.as_object_mut() {
            map.remove("body");
            map.remove("id");
            map.remove("source_path");
            for key in GRAPH_TIMES {
                map.remove(*key);
            }
            if let Some(uid) = map.remove("uid") {
                map.insert("id".into(), uid);
            }
        }
        (data, self.body.clone())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkModel {
    pub id: String,
    pub parent_doc_id: String,
    pub chunk_index: i64,
    pub chunk_text: String,
    pub block_type: String,
    pub start_offset: i64,
    pub end_offset: i64,
    pub rrf_score: Option<f64>,
    pub hub_score: Option<f64>,
    pub final_score: Option<f64>,
    pub parent_title: Option<String>,
    pub parent_type: Option<String>,
    pub parent_tags: Vec<String>,
    pub embedding: Option<Vec<f64>>,
}

impl ChunkModel {
    pub fn public_value(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("chunk serializes");
        if let Some(map) = value.as_object_mut() {
            map.remove("embedding");
        }
        value
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageAssetModel {
    pub id: String,
    pub file_name: String,
    pub mime_type: String,
    pub alt_text: Option<String>,
    pub caption: Option<String>,
    pub embed_route: Option<String>,
    pub content_hash: Option<String>,
    pub embedding: Option<Vec<f64>>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ImageAssetModel {
    pub fn public_value(&self) -> Value {
        let mut value = serde_json::to_value(self).expect("image serializes");
        if let Some(map) = value.as_object_mut() {
            map.remove("embedding");
        }
        value
    }
}

/// Convert a flat DB row (column name -> value) into a public concept JSON object.
pub fn concept_public_from_row(row: &Map<String, Value>, include_body: bool) -> Value {
    let mut public = Map::new();
    let mut extra = Map::new();
    for (key, value) in row {
        match key.as_str() {
            "embedding" | "created_at" | "updated_at" | "effective_at" => {}
            "body" if !include_body => {}
            "tags" => {
                public.insert(
                    key.clone(),
                    Value::Array(
                        normalize_tags(value)
                            .into_iter()
                            .map(Value::String)
                            .collect(),
                    ),
                );
            }
            "extra" => {
                if let Some(items) = value.as_array() {
                    for item in items {
                        if let Some(obj) = item.as_object() {
                            for (k, v) in obj {
                                extra.insert(k.clone(), v.clone());
                            }
                        }
                    }
                } else if let Some(obj) = value.as_object() {
                    for (k, v) in obj {
                        extra.insert(k.clone(), v.clone());
                    }
                }
            }
            "timestamp" => {
                let text = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let rendered = PyDateTime::parse(&text)
                    .map(|dt| Value::String(python_datetime_str(&dt.isoformat())))
                    .unwrap_or(value.clone());
                public.insert(key.clone(), rendered);
            }
            _ => {
                public.insert(key.clone(), value.clone());
            }
        }
    }
    if !extra.is_empty() {
        for (key, value) in extra {
            public.entry(key).or_insert(value);
        }
    }
    Value::Object(public)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_tags_cases() {
        assert_eq!(normalize_tags(&Value::Null), Vec::<String>::new());
        assert_eq!(normalize_tags(&Value::String("x".into())), vec!["x"]);
        assert_eq!(
            normalize_tags(&serde_json::json!([1, true])),
            vec!["1", "True"]
        );
    }

    #[test]
    fn timestamp_spelling() {
        let dt = PyDateTime::parse("2024-01-02T03:04:05").expect("parse");
        assert_eq!(dt.isoformat(), "2024-01-02T03:04:05");
        assert_eq!(
            serde_json::to_string(&dt).expect("json"),
            "\"2024-01-02 03:04:05\""
        );
    }

    #[test]
    fn public_concept_excludes_embedding() {
        let concept = ConceptModel::new("alpha", "note");
        let value = concept.public_value();
        assert!(!value.as_object().unwrap().contains_key("embedding"));
        assert!(value.as_object().unwrap().contains_key("tags"));
    }
}
