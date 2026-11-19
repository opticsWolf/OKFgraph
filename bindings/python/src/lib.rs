//! PyO3 binding. Marshalling, GIL release, and error translation only.
//! No graph, retrieval, ingest, or serialization logic lives here.
#![allow(clippy::result_large_err)]
use okfgraph::error::OkfError;
use okfgraph::ops::Router;
use okfgraph::settings::Settings;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

#[pyfunction]
fn version() -> String {
    okfgraph::version().to_string()
}

#[pyfunction]
fn schema_version() -> u64 {
    okfgraph::schema_version() as u64
}

#[pyfunction]
fn op_names() -> Vec<&'static str> {
    okfgraph::ops::OP_NAMES.to_vec()
}

/// M0 spike: proves a blocking native call releases the GIL.
#[pyfunction]
fn run_cli(argv: Vec<String>) -> u32 {
    okfgraph::cli::run(argv) as u32
}

#[pyfunction]
fn sleep_ms(py: Python<'_>, millis: u64) {
    // PyO3 0.30 renamed allow_threads to detach.
    py.detach(|| std::thread::sleep(std::time::Duration::from_millis(millis)));
}

fn to_py_err(err: OkfError) -> PyErr {
    PyRuntimeError::new_err(serde_json::to_string(&err.envelope_body()).expect("error JSON"))
}

fn parse_settings(params_json: &str) -> PyResult<Settings> {
    let mut params: Map<String, Value> = serde_json::from_str(params_json)
        .map_err(|e| PyRuntimeError::new_err(format!("BAD_VALUE: invalid params JSON: {e}")))?;
    if let Some(backend) = params.remove("embed_backend") {
        params.insert("backend".into(), backend);
    }
    Settings::from_map(&params).map_err(to_py_err)
}

#[pyclass(name = "NativeRouter", module = "okfgraph._native")]
pub struct NativeRouter {
    router: Option<Router>,
}

#[pymethods]
impl NativeRouter {
    #[staticmethod]
    fn open(py: Python<'_>, params_json: &str) -> PyResult<Self> {
        let params = parse_settings(params_json)?;
        let router = py.detach(|| Router::open(&params)).map_err(to_py_err)?;
        Ok(Self {
            router: Some(router),
        })
    }

    fn call(&self, py: Python<'_>, op: &str, params_json: &str) -> PyResult<String> {
        let params: Map<String, Value> = serde_json::from_str(params_json)
            .map_err(|e| PyRuntimeError::new_err(format!("BAD_VALUE: invalid params JSON: {e}")))?;
        let data = py
            .detach(|| {
                self.router
                    .as_ref()
                    .ok_or_else(|| OkfError::new("INTERNAL", "router is closed"))
                    .and_then(|router| router.call(op, &params))
            })
            .map_err(to_py_err)?;
        serde_json::to_string(&data).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn close(&mut self) {
        self.router = None;
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", okfgraph::version())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(op_names, m)?)?;
    m.add_function(wrap_pyfunction!(sleep_ms, m)?)?;
    m.add_function(wrap_pyfunction!(run_cli, m)?)?;
    m.add_class::<NativeRouter>()?;
    Ok(())
}
