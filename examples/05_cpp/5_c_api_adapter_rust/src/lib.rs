//! CSP Rust Adapter Example
//!
//! This crate demonstrates how to create CSP adapters in Rust using the C API and PyO3.
//! It provides example adapters that mirror the C example in `../4_c_api_adapter/`:
//! - Push Input Adapter: Generates incrementing integers on a background thread
//! - Output Adapter: Prints values to stdout
//! - Adapter Manager: Coordinates adapter lifecycle
//!
//! # Building
//!
//! ```bash
//! hatch-build --hooks-only -t wheel
//! ```
//!
//! # Architecture
//!
//! The Rust code implements adapters using the CSP C ABI, then exposes them to Python
//! via PyO3. The pattern is:
//!
//! 1. Implement adapter logic in Rust
//! 2. Create C-compatible VTable structures with callbacks
//! 3. Use PyO3 to create Python capsules wrapping the VTables
//! 4. CSP's Python layer extracts the VTables from capsules and registers with the engine

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod bindings;
pub mod output_adapter;
pub mod input_adapter;
pub mod adapter_manager;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::bindings::*;
use crate::output_adapter::*;
use crate::input_adapter::*;
use crate::adapter_manager::*;

/// Create an example adapter manager.
///
/// # Arguments
/// * `engine` - Engine capsule (passed from Python)
/// * `properties` - Dict with configuration (e.g., {"prefix": "..."})
///
/// # Returns
/// A Python capsule wrapping the adapter manager VTable.
#[pyfunction]
#[pyo3(signature = (engine, properties))]
fn _example_adapter_manager(
    py: Python<'_>,
    engine: &Bound<'_, PyAny>,
    properties: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let _ = engine; // Engine handle not needed for creating vtable

    // Extract prefix from properties
    let prefix = properties
        .get_item("prefix")?
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default();

    // Create the adapter manager
    let manager = Box::new(RustAdapterManager::new(prefix));
    let user_data = Box::into_raw(manager) as *mut std::ffi::c_void;

    // Create vtable
    let vtable = CCspAdapterManagerVTable {
        user_data,
        name: Some(rust_adapter_manager_name),
        process_next_sim_time_slice: Some(rust_adapter_manager_process_sim_time),
        destroy: Some(rust_adapter_manager_destroy),
        start: Some(rust_adapter_manager_start),
        stop: Some(rust_adapter_manager_stop),
    };

    // Create capsule
    create_adapter_manager_capsule(py, vtable)
}

/// Create an example input adapter that generates incrementing integers.
///
/// # Arguments
/// * `interval_ms` - Interval between generated values in milliseconds
///
/// # Returns
/// A Python capsule wrapping the input adapter VTable.
#[pyfunction]
#[pyo3(signature = (interval_ms=100))]
fn _example_input_adapter(py: Python<'_>, interval_ms: i32) -> PyResult<PyObject> {
    // Create the input adapter
    let adapter = Box::new(RustInputAdapter::new(interval_ms as u64));
    let user_data = Box::into_raw(adapter) as *mut std::ffi::c_void;

    // Create vtable
    let vtable = CCspPushInputAdapterVTable {
        user_data,
        start: Some(rust_input_adapter_start),
        stop: Some(rust_input_adapter_stop),
        destroy: Some(rust_input_adapter_destroy),
    };

    // Create capsule
    create_input_adapter_capsule(py, vtable)
}

/// Create an example output adapter that prints values to stdout.
///
/// # Arguments
/// * `prefix` - Optional prefix for output messages
///
/// # Returns
/// A Python capsule wrapping the output adapter VTable.
#[pyfunction]
#[pyo3(signature = (prefix=None))]
fn _example_output_adapter(py: Python<'_>, prefix: Option<String>) -> PyResult<PyObject> {
    // Create the output adapter
    let adapter = Box::new(RustOutputAdapter::new(prefix));
    let user_data = Box::into_raw(adapter) as *mut std::ffi::c_void;

    // Create vtable
    let vtable = CCspOutputAdapterVTable {
        user_data,
        start: Some(rust_output_adapter_start),
        stop: Some(rust_output_adapter_stop),
        execute: Some(rust_output_adapter_execute),
        destroy: Some(rust_output_adapter_destroy),
    };

    // Create capsule
    create_output_adapter_capsule(py, vtable)
}

/// Rust CSP Adapter Example module
///
/// This module provides PyO3 functions that create VTable capsules for:
/// - AdapterManager: manages lifecycle of adapters
/// - InputAdapter: pushes values into the CSP graph
/// - OutputAdapter: receives values from the CSP graph
#[pymodule]
fn exampleadapter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_example_adapter_manager, m)?)?;
    m.add_function(wrap_pyfunction!(_example_input_adapter, m)?)?;
    m.add_function(wrap_pyfunction!(_example_output_adapter, m)?)?;
    Ok(())
}
