//! Output Adapter implementation in Rust
//!
//! This module demonstrates how to implement a CSP output adapter in Rust
//! using the C API FFI bindings.
//!
//! The adapter receives values from the CSP graph and prints them to stderr,
//! demonstrating how to use the C API to retrieve typed values.

use std::ffi::{c_char, c_void};
use std::slice;
use std::str;

use crate::bindings::{
    csp_engine_now, csp_input_get_last_bool, csp_input_get_last_datetime,
    csp_input_get_last_double, csp_input_get_last_int64, csp_input_get_last_string,
    csp_input_get_type, CCspDateTime, CCspEngineHandle, CCspErrorCode, CCspInputHandle,
    CCspType,
};

/// Output adapter that prints values to stderr.
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
    pub unsafe fn execute(&mut self, engine: CCspEngineHandle, input: CCspInputHandle) {
        let prefix = self.prefix.as_deref().unwrap_or("");

        self.count += 1;

        // Get current engine time
        let now = match csp_engine_now(engine) {
            Some(value) => value,
            None => {
                eprintln!("{}[RustOutputAdapter] CSP symbol missing: ccsp_engine_now", prefix);
                return;
            }
        };

        // Get the type of the input
        let input_type = match csp_input_get_type(input) {
            Some(value) => value,
            None => {
                eprintln!("{}[RustOutputAdapter] CSP symbol missing: ccsp_input_get_type", prefix);
                return;
            }
        };

        // Retrieve and print the value based on type
        match input_type {
            CCspType::Bool => {
                let mut value: i8 = 0;
                if csp_input_get_last_bool(input, &mut value) == Some(CCspErrorCode::Ok) {
                    eprintln!(
                        "{}[{}] bool: {}",
                        prefix,
                        now,
                        if value != 0 { "true" } else { "false" }
                    );
                }
            }
            CCspType::Int64 => {
                let mut value: i64 = 0;
                if csp_input_get_last_int64(input, &mut value) == Some(CCspErrorCode::Ok) {
                    eprintln!("{}[{}] int64: {}", prefix, now, value);
                }
            }
            CCspType::Double => {
                let mut value: f64 = 0.0;
                if csp_input_get_last_double(input, &mut value) == Some(CCspErrorCode::Ok) {
                    eprintln!("{}[{}] double: {}", prefix, now, value);
                }
            }
            CCspType::String => {
                let mut data: *const c_char = std::ptr::null();
                let mut len: usize = 0;
                if csp_input_get_last_string(input, &mut data, &mut len) == Some(CCspErrorCode::Ok)
                    && !data.is_null() && len > 0 {
                        let bytes = slice::from_raw_parts(data as *const u8, len);
                        if let Ok(s) = str::from_utf8(bytes) {
                            eprintln!("{}[{}] string: {}", prefix, now, s);
                        } else {
                            eprintln!("{}[{}] string: <invalid utf8>", prefix, now);
                        }
                    }
            }
            CCspType::DateTime => {
                let mut value: CCspDateTime = 0;
                if csp_input_get_last_datetime(input, &mut value) == Some(CCspErrorCode::Ok) {
                    eprintln!("{}[{}] datetime: {} ns", prefix, now, value);
                }
            }
            _ => {
                eprintln!("{}[{}] <type {:?}>", prefix, now, input_type);
            }
        }
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

