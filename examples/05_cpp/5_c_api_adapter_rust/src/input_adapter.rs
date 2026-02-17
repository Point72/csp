//! Input Adapter implementation in Rust
//!
//! This module demonstrates how to implement a CSP push input adapter in Rust
//! using the C API FFI bindings.
//!
//! Note: The actual data pushing requires the CSP C API symbols to be available
//! at runtime. When built standalone, the callbacks are implemented but the
//! push functionality is stubbed.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use crate::bindings::{
    CCspDateTime, CCspEngineHandle, CCspPushInputAdapterHandle,
};

/// Push input adapter that generates incrementing counter values.
///
/// This adapter would spawn a background thread when started that periodically
/// pushes integer values to the CSP graph using the C API.
pub struct RustInputAdapter {
    /// Interval between pushes in milliseconds
    interval_ms: u64,

    /// Counter value (shared with thread)
    counter: Arc<AtomicI64>,

    /// Flag to signal thread to stop
    running: Arc<AtomicBool>,

    /// Push adapter handle (set by start callback)
    #[allow(dead_code)]
    adapter_handle: Option<CCspPushInputAdapterHandle>,
}

impl RustInputAdapter {
    /// Create a new input adapter.
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms: if interval_ms > 0 { interval_ms } else { 100 },
            counter: Arc::new(AtomicI64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            adapter_handle: None,
        }
    }

    /// Start the adapter.
    ///
    /// In a full implementation with CSP C API symbols linked, this would
    /// spawn a background thread that calls ccsp_push_input_adapter_push_int64.
    pub fn start(&mut self, adapter: CCspPushInputAdapterHandle) {
        self.adapter_handle = Some(adapter);
        self.running.store(true, Ordering::SeqCst);

        eprintln!(
            "[RustInputAdapter] Started with interval {} ms (adapter handle: {:?})",
            self.interval_ms, adapter
        );

        // Note: In a full implementation, we would spawn a thread here that
        // calls ccsp_push_input_adapter_push_int64 to push values.
        // This requires the CSP C API symbols to be available at runtime.
    }

    /// Stop the adapter.
    pub fn stop(&mut self) {
        eprintln!(
            "[RustInputAdapter] Stopped after {} values",
            self.counter.load(Ordering::SeqCst)
        );

        self.running.store(false, Ordering::SeqCst);
    }
}

impl Drop for RustInputAdapter {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// C ABI callback functions
// ============================================================================

/// Start callback for the input adapter.
///
/// # Safety
///
/// Called from C code with valid pointers.
pub unsafe extern "C" fn rust_input_adapter_start(
    user_data: *mut c_void,
    _engine: CCspEngineHandle,
    adapter: CCspPushInputAdapterHandle,
    _start_time: CCspDateTime,
    _end_time: CCspDateTime,
) {
    if user_data.is_null() {
        return;
    }
    let rust_adapter = &mut *(user_data as *mut RustInputAdapter);
    rust_adapter.start(adapter);
}

/// Stop callback for the input adapter.
///
/// # Safety
///
/// Called from C code with valid pointer.
pub unsafe extern "C" fn rust_input_adapter_stop(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let rust_adapter = &mut *(user_data as *mut RustInputAdapter);
    rust_adapter.stop();
}

/// Destroy callback for the input adapter.
///
/// # Safety
///
/// Called from C code. Takes ownership and drops the adapter.
pub unsafe extern "C" fn rust_input_adapter_destroy(user_data: *mut c_void) {
    if !user_data.is_null() {
        let mut adapter = Box::from_raw(user_data as *mut RustInputAdapter);
        adapter.stop();
        // Box is dropped here, freeing memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_adapter() {
        let adapter = RustInputAdapter::new(100);
        assert_eq!(adapter.interval_ms, 100);
        assert!(!adapter.running.load(Ordering::SeqCst));
    }

    #[test]
    fn test_default_interval() {
        let adapter = RustInputAdapter::new(0);
        assert_eq!(adapter.interval_ms, 100); // Should default to 100
    }
}

