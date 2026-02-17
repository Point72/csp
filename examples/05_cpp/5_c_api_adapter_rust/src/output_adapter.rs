//! Output Adapter implementation in Rust
//!
//! This module demonstrates how to implement a CSP output adapter in Rust
//! using the C API FFI bindings.
//!
//! Note: The actual value retrieval requires the CSP C API symbols to be available
//! at runtime. When built standalone, the execute callback logs that it was invoked
//! but cannot access the actual values.

use std::ffi::c_void;

use crate::bindings::{
    CCspDateTime, CCspEngineHandle, CCspInputHandle,
};

/// Output adapter that prints values to stdout.
///
/// This adapter is called by CSP whenever the input time series ticks.
/// It retrieves the latest value and prints it with an optional prefix.
pub struct RustOutputAdapter {
    /// Optional prefix for output messages
    prefix: Option<String>,

    /// Counter for number of values received
    count: u64,
}

impl RustOutputAdapter {
    /// Create a new output adapter.
    pub fn new(prefix: Option<String>) -> Self {
        Self { prefix, count: 0 }
    }

    /// Called when the adapter starts.
    pub fn start(&mut self, start_time: CCspDateTime, end_time: CCspDateTime) {
        let prefix = self.prefix.as_deref().unwrap_or("");
        eprintln!(
            "{}[RustOutputAdapter] Started. Time range: {} - {} ns",
            prefix, start_time, end_time
        );
    }

    /// Called when the adapter stops.
    pub fn stop(&mut self) {
        let prefix = self.prefix.as_deref().unwrap_or("");
        eprintln!(
            "{}[RustOutputAdapter] Stopped after {} values.",
            prefix, self.count
        );
    }

    /// Called each time the input has a new value.
    ///
    /// # Safety
    ///
    /// The engine and input handles must be valid.
    ///
    /// Note: In a full implementation with CSP C API symbols linked, this would
    /// call ccsp_engine_now, ccsp_input_get_type, and ccsp_input_get_last_* to
    /// retrieve and print the actual values.
    pub unsafe fn execute(&mut self, engine: CCspEngineHandle, input: CCspInputHandle) {
        let prefix = self.prefix.as_deref().unwrap_or("");

        self.count += 1;

        // Note: Without CSP C API symbols, we can only log that execute was called.
        // In a full implementation, we would call:
        //   let now = ccsp_engine_now(engine);
        //   let input_type = ccsp_input_get_type(input);
        //   ccsp_input_get_last_int64/double/bool/string(input, &mut value)
        eprintln!(
            "{}[RustOutputAdapter] execute called (count={}, engine={:?}, input={:?})",
            prefix, self.count, engine, input
        );
    }
}

impl Default for RustOutputAdapter {
    fn default() -> Self {
        Self::new(None)
    }
}

// ============================================================================
// C ABI callback functions
// ============================================================================

/// Start callback for the output adapter.
///
/// # Safety
///
/// Called from C code with valid pointers.
pub unsafe extern "C" fn rust_output_adapter_start(
    user_data: *mut c_void,
    _engine: CCspEngineHandle,
    start_time: CCspDateTime,
    end_time: CCspDateTime,
) {
    if user_data.is_null() {
        return;
    }
    let adapter = &mut *(user_data as *mut RustOutputAdapter);
    adapter.start(start_time, end_time);
}

/// Stop callback for the output adapter.
///
/// # Safety
///
/// Called from C code with valid pointer.
pub unsafe extern "C" fn rust_output_adapter_stop(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let adapter = &mut *(user_data as *mut RustOutputAdapter);
    adapter.stop();
}

/// Execute callback for the output adapter.
///
/// # Safety
///
/// Called from C code with valid pointers.
pub unsafe extern "C" fn rust_output_adapter_execute(
    user_data: *mut c_void,
    engine: CCspEngineHandle,
    input: CCspInputHandle,
) {
    if user_data.is_null() {
        return;
    }
    let adapter = &mut *(user_data as *mut RustOutputAdapter);
    adapter.execute(engine, input);
}

/// Destroy callback for the output adapter.
///
/// # Safety
///
/// Called from C code. Takes ownership and drops the adapter.
pub unsafe extern "C" fn rust_output_adapter_destroy(user_data: *mut c_void) {
    if !user_data.is_null() {
        let _ = Box::from_raw(user_data as *mut RustOutputAdapter);
        // Box is dropped here, freeing memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_adapter() {
        let adapter = RustOutputAdapter::new(None);
        assert_eq!(adapter.count, 0);
    }

    #[test]
    fn test_create_with_prefix() {
        let adapter = RustOutputAdapter::new(Some("[TEST] ".to_string()));
        assert_eq!(adapter.prefix, Some("[TEST] ".to_string()));
    }
}

