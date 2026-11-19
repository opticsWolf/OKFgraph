//! Public error type. Codes mirror the closed set in `okfgraph/errors.py`.
//!
//! The class split from Python (`UsageError`/`StateError`/`OutcomeError`) is
//! represented by [`ErrorKind`]; adapters use it for exit codes and Python
//! exception classes.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorKind {
    Usage,
    State,
    Outcome,
}

pub const USAGE_CODES: &[&str] = &[
    "BAD_VALUE",
    "MISSING_PARAM",
    "FILE_NOT_FOUND",
    "CONFIG_INVALID",
];

pub const OUTCOME_CODES: &[&str] = &["DOCTOR_FINDINGS", "LINT_ERRORS", "DIFF_DIFFERENT"];

pub const STATE_CODES: &[&str] = &[
    "UNKNOWN_CONCEPT",
    "UNKNOWN_ASSET",
    "NOT_RECOVERABLE",
    "DETACHED",
    "DETACH_REFUSED",
    "PURGE_REFUSED_ABSENT_ROOT",
    "SEARCH_UNAVAILABLE",
    "WRITE_LOCK_TIMEOUT",
    "DB_LOCKED",
    "DIM_MISMATCH",
    "MODEL_PIN_MISMATCH",
    "PRECISION_PIN_MISMATCH",
    "SPACE_PIN_MISMATCH",
    "IMAGE_PIN_MISMATCH",
    "VISION_INCOMPATIBLE",
    "EXTERNAL_UNAVAILABLE",
    "CONVERSION_FAILED",
    "NO_ORT_RUNTIME",
    "EMBROIDER_TOO_OLD",
    "REEMBED_IN_PROGRESS",
    "INTERNAL",
];

pub fn error_kind(code: &str) -> Option<ErrorKind> {
    if USAGE_CODES.contains(&code) {
        Some(ErrorKind::Usage)
    } else if OUTCOME_CODES.contains(&code) {
        Some(ErrorKind::Outcome)
    } else if STATE_CODES.contains(&code) {
        Some(ErrorKind::State)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
#[error("{message}")]
pub struct OkfError {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub fields: BTreeMap<String, Value>,
    #[serde(default)]
    pub remedy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl OkfError {
    /// Build an error, refusing unknown codes (the Python set is closed).
    #[track_caller]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        let code = code.into();
        assert!(
            error_kind(&code).is_some(),
            "unknown OKF error code: {code}"
        );
        Self {
            code,
            message: message.into(),
            op: None,
            fields: BTreeMap::new(),
            remedy: None,
            data: None,
        }
    }

    pub fn usage(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(code, message)
    }

    pub fn state(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(code, message)
    }

    pub fn outcome(code: impl Into<String>, message: impl Into<String>, data: Value) -> Self {
        let mut err = Self::new(code, message);
        err.data = Some(data);
        err
    }

    pub fn internal(message: impl Into<String>, type_name: &str) -> Self {
        let mut err = Self::new("INTERNAL", message);
        err.fields
            .insert("type".into(), Value::String(type_name.into()));
        err
    }

    pub fn op(mut self, op: impl Into<String>) -> Self {
        self.op = Some(op.into());
        self
    }

    pub fn remedy(mut self, remedy: impl Into<String>) -> Self {
        self.remedy = Some(remedy.into());
        self
    }

    pub fn field(mut self, key: impl Into<String>, value: Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    pub fn kind(&self) -> ErrorKind {
        error_kind(&self.code).expect("validated error code")
    }

    pub fn exit_code(&self) -> i32 {
        match self.kind() {
            ErrorKind::Usage => 2,
            ErrorKind::State | ErrorKind::Outcome => 1,
        }
    }

    /// Adapter-facing one-liner: `CODE: message (remedy)`.
    pub fn titles(&self) -> String {
        match &self.remedy {
            Some(remedy) => format!("{}: {} ({})", self.code, self.message, remedy),
            None => format!("{}: {}", self.code, self.message),
        }
    }

    /// Machine-facing body for the `error` field of an envelope.
    pub fn envelope_body(&self) -> Value {
        let fields = if self.fields.is_empty() {
            Value::Null
        } else {
            serde_json::to_value(&self.fields).unwrap_or(Value::Null)
        };
        serde_json::json!({
            "code": self.code,
            "message": self.message,
            "fields": fields,
            "remedy": self.remedy,
        })
    }
}

pub type Result<T> = std::result::Result<T, OkfError>;

/// Uniform result transport: `{ok, op, data, warnings, error}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub ok: bool,
    pub op: String,
    pub data: Value,
    pub warnings: Vec<String>,
    pub error: Option<Value>,
}

impl Envelope {
    pub fn success(op: impl Into<String>, data: Value) -> Self {
        Self {
            ok: true,
            op: op.into(),
            data,
            warnings: Vec::new(),
            error: None,
        }
    }

    pub fn failure(op: impl Into<String>, error: &OkfError) -> Self {
        Self {
            ok: false,
            op: op.into(),
            data: error.data.clone().unwrap_or(Value::Null),
            warnings: Vec::new(),
            error: Some(error.envelope_body()),
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| OkfError::internal(e.to_string(), "serde_json"))
    }

    pub fn to_json_pretty(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| OkfError::internal(e.to_string(), "serde_json"))
    }
}

impl From<std::io::Error> for OkfError {
    fn from(err: std::io::Error) -> Self {
        OkfError::internal(err.to_string(), "std::io::Error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_code_set() {
        assert_eq!(error_kind("BAD_VALUE"), Some(ErrorKind::Usage));
        assert_eq!(error_kind("UNKNOWN_CONCEPT"), Some(ErrorKind::State));
        assert_eq!(error_kind("DOCTOR_FINDINGS"), Some(ErrorKind::Outcome));
        assert_eq!(error_kind("NOPE"), None);
    }

    #[test]
    fn envelope_field_order() {
        let json = Envelope::success("read", serde_json::json!({"id": "alpha"}))
            .to_json()
            .expect("json");
        assert_eq!(
            json,
            r#"{"ok":true,"op":"read","data":{"id":"alpha"},"warnings":[],"error":null}"#
        );
    }
}
