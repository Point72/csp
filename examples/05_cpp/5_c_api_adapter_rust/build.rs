//! Build script for the CSP Rust adapter.
//!
//! This build script configures linking for the CSP C API.
//! The actual linking to CSP happens at runtime when Python loads the module.

fn main() {
    // Tell Cargo to re-run this script if these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");

    // cfg! in a build script describes the host, so read the target from Cargo instead
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_os.as_str() {
        // CSP symbols are resolved at runtime once Python has loaded both modules
        "macos" => {
            println!("cargo:rustc-link-arg=-undefined");
            println!("cargo:rustc-link-arg=dynamic_lookup");
        }
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
        }
        // Windows resolves every symbol at link time, so this example needs an import library
        "windows" => {}
        other => {
            println!("cargo:warning=unrecognized target OS '{other}'; no link arguments applied");
        }
    }
}
