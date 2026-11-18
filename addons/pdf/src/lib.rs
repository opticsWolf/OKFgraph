//! Optional native PDF/Office plugin for OKFgraph.
//!
//! The C ABI is intentionally tiny. Implementation wraps `bobine` in M3.
#[no_mangle]
pub extern "C" fn okf_pdf_abi_version() -> u32 {
    1
}
