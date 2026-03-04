//! Adapter Manager implementation in Rust
//!
//! This module demonstrates how to implement a CSP adapter manager in Rust
//! using the C API FFI bindings. The adapter manager coordinates the lifecycle
//! of related adapters.

use std::ffi::{c_char, c_void, CString};

use crate::bindings::{CCspAdapterManagerHandle, CCspDateTime};

/// Adapter manager that coordinates input and output adapters.
///
/// The manager handles:
/// - Starting and stopping all managed adapters together
/// - Status reporting
/// - Simulation time slicing (for replay mode)
pub struct RustAdapterManager {
    /// Prefix for log messages
    #[allow(dead_code)]
    prefix: String,

    /// C-compatible name string (stored to keep pointer valid)
    name_cstring: CString,

    /// Whether the manager is currently running
    running: bool,
}

impl RustAdapterManager {
    /// Create a new adapter manager.
    pub fn new(prefix: String) -> Self {
        let name = format!("RustAdapterManager({})", prefix);
        let name_cstring = CString::new(name).unwrap_or_else(|_| CString::new("RustAdapterManager").unwrap());

        Self {
            prefix,
            name_cstring,
            running: false,
        }
    }

    /// Get the name of this adapter manager.
    pub fn name(&self) -> *const c_char {
        self.name_cstring.as_ptr()
    }

    /// Called when the graph starts.
    pub fn start(&mut self, start_time: CCspDateTime, end_time: CCspDateTime) {
        self.running = true;
        eprintln!(
            "[{}] Started. Time range: {} - {} ns",
            self.name_cstring.to_string_lossy(),
            start_time,
            end_time
        );
    }

    /// Called when the graph stops.
    pub fn stop(&mut self) {
        self.running = false;
        eprintln!(
            "[{}] Stopped.",
            self.name_cstring.to_string_lossy()
        );
    }

    /// Process simulation time slice.
    ///
    /// For realtime adapters that don't support simulation, return 0.
    /// For sim adapters, process data at the given time and return the next time.
    pub fn process_next_sim_time_slice(&mut self, _time: CCspDateTime) -> CCspDateTime {
        // This example doesn't support simulation mode
        0
    }
}

impl Default for RustAdapterManager {
    fn default() -> Self {
        Self::new(String::new())
    }
}

// ============================================================================
// C ABI callback functions
// ============================================================================

/// Return the name of this adapter manager.
///
/// # Safety
///
/// Called from C code with valid pointer.
pub unsafe extern "C" fn rust_adapter_manager_name(user_data: *mut c_void) -> *const c_char {
    if user_data.is_null() {
        return std::ptr::null();
    }
    let manager = &*(user_data as *const RustAdapterManager);
    manager.name()
}

/// Process simulation time slice.
///
/// # Safety
///
/// Called from C code with valid pointer.
pub unsafe extern "C" fn rust_adapter_manager_process_sim_time(
    user_data: *mut c_void,
    time: CCspDateTime,
) -> CCspDateTime {
    if user_data.is_null() {
        return 0;
    }
    let manager = &mut *(user_data as *mut RustAdapterManager);
    manager.process_next_sim_time_slice(time)
}

/// Start callback for the adapter manager.
///
/// # Safety
///
/// Called from C code with valid pointers.
pub unsafe extern "C" fn rust_adapter_manager_start(
    user_data: *mut c_void,
    _manager: CCspAdapterManagerHandle,
    start_time: CCspDateTime,
    end_time: CCspDateTime,
) {
    if user_data.is_null() {
        return;
    }
    let rust_manager = &mut *(user_data as *mut RustAdapterManager);
    rust_manager.start(start_time, end_time);
}

/// Stop callback for the adapter manager.
///
/// # Safety
///
/// Called from C code with valid pointer.
pub unsafe extern "C" fn rust_adapter_manager_stop(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    let rust_manager = &mut *(user_data as *mut RustAdapterManager);
    rust_manager.stop();
}

/// Destroy callback for the adapter manager.
///
/// # Safety
///
/// Called from C code. Takes ownership and drops the manager.
pub unsafe extern "C" fn rust_adapter_manager_destroy(user_data: *mut c_void) {
    if !user_data.is_null() {
        let _ = Box::from_raw(user_data as *mut RustAdapterManager);
        // Box is dropped here, freeing memory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_manager() {
        let manager = RustAdapterManager::new("test".to_string());
        assert!(!manager.running);
        assert!(manager.prefix == "test");
    }

    #[test]
    fn test_name() {
        let manager = RustAdapterManager::new("prefix".to_string());
        unsafe {
            let name_ptr = manager.name();
            let name = std::ffi::CStr::from_ptr(name_ptr);
            assert!(name.to_string_lossy().contains("RustAdapterManager"));
            assert!(name.to_string_lossy().contains("prefix"));
        }
    }
}

