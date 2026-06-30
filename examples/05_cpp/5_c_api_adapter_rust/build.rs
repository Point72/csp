//! Build script for the CSP Rust adapter.
//!
//! This build script configures linking for the CSP C API.
//! The actual linking to CSP happens at runtime when Python loads the module.

fn main() {
    // Tell Cargo to re-run this script if these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");

    // For macOS, we need to allow undefined symbols since CSP functions
    // are resolved at runtime when Python loads both our module and CSP
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }

    // On Linux, allow undefined symbols to be resolved at runtime
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
    }
}
