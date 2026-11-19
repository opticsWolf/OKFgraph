//! PyO3 binding. Marshalling, GIL release, and error translation only.
//! No graph, retrieval, ingest, or serialization logic lives here.
use pyo3::prelude::*;

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
fn sleep_ms(py: Python<'_>, millis: u64) {
    // PyO3 0.30 renamed allow_threads to detach.
    py.detach(|| std::thread::sleep(std::time::Duration::from_millis(millis)));
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", okfgraph::version())?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(op_names, m)?)?;
    m.add_function(wrap_pyfunction!(sleep_ms, m)?)?;
    Ok(())
}
