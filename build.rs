fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Ladybug extensions resolve host symbols; export them from binaries/cdylibs.
    match os.as_str() {
        "windows" => {}
        "macos" => println!("cargo:rustc-link-arg=-Wl,-export_dynamic"),
        _ => println!("cargo:rustc-link-arg=-rdynamic"),
    }
}
