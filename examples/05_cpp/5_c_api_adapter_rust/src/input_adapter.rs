//! Input Adapter implementation in Rust
//!
//! This module demonstrates how to implement a CSP push input adapter in Rust
//! using the C API FFI bindings.
//!
//! The adapter spawns a background thread that periodically pushes integer
//! values to the CSP graph using the C API.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::bindings::{
    csp_push_input_adapter_push_int64, CCspDateTime, CCspEngineHandle,
    CCspPushInputAdapterHandle,
};

/// Push input adapter that generates incrementing counter values.
///
/// This adapter spawns a background thread when started that periodically
/// pushes integer values to the CSP graph using the C API.
pub struct RustInputAdapter {
    /// Interval between pushes in milliseconds
    interval_ms: u64,

    /// Counter value (shared with thread)
    counter: Arc<AtomicI64>,

    /// Stop signal. Waiting on the condvar rather than sleeping lets `stop` interrupt the
    /// interval immediately instead of blocking shutdown for up to `interval_ms`.
    stop: Arc<(Mutex<bool>, Condvar)>,

    /// Push adapter handle (stored as AtomicPtr for thread-safe access)
    adapter_handle: Arc<AtomicPtr<c_void>>,

    /// Thread handle
    thread_handle: Option<JoinHandle<()>>,
}

impl RustInputAdapter {
    /// Create a new input adapter.
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms: if interval_ms > 0 { interval_ms } else { 100 },
            counter: Arc::new(AtomicI64::new(0)),
            stop: Arc::new((Mutex::new(false), Condvar::new())),
            adapter_handle: Arc::new(AtomicPtr::new(ptr::null_mut())),
            thread_handle: None,
        }
    }

    /// Start the adapter.
    ///
    /// Spawns a background thread that pushes integer values at the configured
    /// interval using the CSP C API.
    pub fn start(&mut self, adapter: CCspPushInputAdapterHandle) {
        // A second start would orphan the first worker, leaving it pushing to a stale handle
        if self.thread_handle.is_some() {
            self.stop();
        }

        // Store the adapter handle (AtomicPtr is Send + Sync)
        self.adapter_handle.store(adapter, Ordering::SeqCst);
        *self.stop.0.lock().unwrap() = false;

        let stop = Arc::clone(&self.stop);
        let counter = Arc::clone(&self.counter);
        let adapter_handle = Arc::clone(&self.adapter_handle);
        let interval_ms = self.interval_ms;

        eprintln!(
            "[RustInputAdapter] Starting with interval {} ms (adapter handle: {:?})",
            self.interval_ms, adapter
        );

        // Spawn background thread that pushes values
        let handle = thread::spawn(move || {
            let interval = Duration::from_millis(interval_ms);
            let (lock, cvar) = &*stop;

            loop {
                let value = counter.fetch_add(1, Ordering::SeqCst);
                let adapter_ptr = adapter_handle.load(Ordering::SeqCst);

                // Push the counter value to CSP using the C API
                if !adapter_ptr.is_null() {
                    unsafe {
                        let result = csp_push_input_adapter_push_int64(
                            adapter_ptr,
                            value,
                            ptr::null_mut(),
                        );
                        if result.is_none() {
                            eprintln!("[RustInputAdapter] ccsp_push_input_adapter_push_int64 unavailable or returned an unrecognized code");
                            break;
                        }
                        if result != Some(crate::bindings::CCspErrorCode::Ok) {
                            eprintln!(
                                "[RustInputAdapter] Push failed with error: {:?}",
                                result
                            );
                        }
                    }
                }

                let stopping = lock.lock().unwrap();
                let (stopping, _) = cvar.wait_timeout_while(stopping, interval, |s| !*s).unwrap();
                if *stopping {
                    break;
                }
            }

            eprintln!(
                "[RustInputAdapter] Thread exiting after {} values",
                counter.load(Ordering::SeqCst)
            );
        });

        self.thread_handle = Some(handle);
    }

    /// Stop the adapter.
    pub fn stop(&mut self) {
        eprintln!(
            "[RustInputAdapter] Stopping after {} values",
            self.counter.load(Ordering::SeqCst)
        );

        {
            let (lock, cvar) = &*self.stop;
            *lock.lock().unwrap() = true;
            cvar.notify_all();
        }

        // Wait for the thread to finish
        if let Some(handle) = self.thread_handle.take() {
            if let Err(e) = handle.join() {
                eprintln!("[RustInputAdapter] Thread join error: {:?}", e);
            }
        }
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

