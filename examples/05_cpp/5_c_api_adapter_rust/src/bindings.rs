//! FFI bindings for the CSP C API
//!
//! These bindings match the C ABI structures defined in the CSP headers.
//! See `cpp/csp/engine/c/` for the C header files.
//!
//! Note: The actual CSP C API functions (ccsp_*) are resolved at runtime when
//! this module is loaded alongside CSP. For standalone testing, these functions
//! are stubbed.

use std::ffi::{c_char, c_void};
use pyo3::prelude::*;
use pyo3::ffi;
use libc::{dlsym, RTLD_DEFAULT};

/// CSP DateTime type (nanoseconds since epoch)
pub type CCspDateTime = i64;

/// Opaque handle to the CSP engine
pub type CCspEngineHandle = *mut c_void;

/// Opaque handle to a push input adapter
pub type CCspPushInputAdapterHandle = *mut c_void;

/// Opaque handle to an adapter manager
pub type CCspAdapterManagerHandle = *mut c_void;

/// Opaque handle to an input (for output adapters)
pub type CCspInputHandle = *mut c_void;

/// CSP type enumeration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCspType {
    Unknown = 0,
    Bool = 1,
    Int8 = 2,
    Uint8 = 3,
    Int16 = 4,
    Uint16 = 5,
    Int32 = 6,
    Uint32 = 7,
    Int64 = 8,
    Uint64 = 9,
    Double = 10,
    DateTime = 11,
    TimeDelta = 12,
    Date = 13,
    Time = 14,
    Enum = 15,
    String = 16,
    Struct = 17,
    Array = 18,
    DialectGeneric = 19,
}

/// CSP Error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCspErrorCode {
    Ok = 0,
    Error = 1,
    InvalidArgument = 2,
    NullPointer = 3,
    TypeMismatch = 4,
    NotFound = 5,
}

// ============================================================================
// VTable structures (matching the C API headers)
// ============================================================================

/// VTable for output adapter callbacks
#[repr(C)]
pub struct CCspOutputAdapterVTable {
    pub user_data: *mut c_void,

    /// Called when the graph starts
    pub start: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            engine: CCspEngineHandle,
            start_time: CCspDateTime,
            end_time: CCspDateTime,
        ),
    >,

    /// Called when the graph stops
    pub stop: Option<unsafe extern "C" fn(user_data: *mut c_void)>,

    /// Called each time the input has a new value
    pub execute: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            engine: CCspEngineHandle,
            input: CCspInputHandle,
        ),
    >,

    /// Called to destroy the adapter
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

/// VTable for push input adapter callbacks
#[repr(C)]
pub struct CCspPushInputAdapterVTable {
    pub user_data: *mut c_void,

    /// Called when the graph starts
    pub start: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            engine: CCspEngineHandle,
            adapter: CCspPushInputAdapterHandle,
            start_time: CCspDateTime,
            end_time: CCspDateTime,
        ),
    >,

    /// Called when the graph stops
    pub stop: Option<unsafe extern "C" fn(user_data: *mut c_void)>,

    /// Called to destroy the adapter
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

/// VTable for adapter manager callbacks
#[repr(C)]
pub struct CCspAdapterManagerVTable {
    pub user_data: *mut c_void,

    /// Return the name of this adapter manager
    pub name: Option<unsafe extern "C" fn(user_data: *mut c_void) -> *const c_char>,

    /// Process simulation time slice
    pub process_next_sim_time_slice:
        Option<unsafe extern "C" fn(user_data: *mut c_void, time: CCspDateTime) -> CCspDateTime>,

    /// Called to destroy the adapter manager
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,

    /// Called when the graph starts
    pub start: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            manager: CCspAdapterManagerHandle,
            start_time: CCspDateTime,
            end_time: CCspDateTime,
        ),
    >,

    /// Called when the graph stops
    pub stop: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

// ============================================================================
// CSP C API function type definitions
//
// These are resolved at runtime when loaded alongside CSP.
// We use weak linking / dynamic_lookup on macOS.
// ============================================================================

type FnCcspEngineNow = unsafe extern "C" fn(engine: CCspEngineHandle) -> CCspDateTime;
type FnCcspInputGetType = unsafe extern "C" fn(input: CCspInputHandle) -> CCspType;
type FnCcspInputGetLastBool = unsafe extern "C" fn(input: CCspInputHandle, out: *mut i8) -> CCspErrorCode;
type FnCcspInputGetLastInt64 = unsafe extern "C" fn(input: CCspInputHandle, out: *mut i64) -> CCspErrorCode;
type FnCcspInputGetLastDouble = unsafe extern "C" fn(input: CCspInputHandle, out: *mut f64) -> CCspErrorCode;
type FnCcspInputGetLastString = unsafe extern "C" fn(
    input: CCspInputHandle,
    out_data: *mut *const c_char,
    out_length: *mut usize,
) -> CCspErrorCode;
type FnCcspInputGetLastDatetime = unsafe extern "C" fn(
    input: CCspInputHandle,
    out: *mut CCspDateTime,
) -> CCspErrorCode;
type FnCcspPushInputAdapterPushInt64 = unsafe extern "C" fn(
    adapter: CCspPushInputAdapterHandle,
    value: i64,
    batch: *mut c_void,
) -> CCspErrorCode;

unsafe fn resolve_symbol<T: Copy>(name: &'static [u8]) -> Option<T> {
    let symbol = dlsym(RTLD_DEFAULT, name.as_ptr() as *const c_char);
    if symbol.is_null() {
        None
    } else {
        Some(std::mem::transmute_copy(&symbol))
    }
}

pub unsafe fn csp_engine_now(engine: CCspEngineHandle) -> Option<CCspDateTime> {
    resolve_symbol::<FnCcspEngineNow>(b"ccsp_engine_now\0").map(|f| f(engine))
}

pub unsafe fn csp_input_get_type(input: CCspInputHandle) -> Option<CCspType> {
    resolve_symbol::<FnCcspInputGetType>(b"ccsp_input_get_type\0").map(|f| f(input))
}

pub unsafe fn csp_input_get_last_bool(input: CCspInputHandle, out: *mut i8) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspInputGetLastBool>(b"ccsp_input_get_last_bool\0").map(|f| f(input, out))
}

pub unsafe fn csp_input_get_last_int64(input: CCspInputHandle, out: *mut i64) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspInputGetLastInt64>(b"ccsp_input_get_last_int64\0").map(|f| f(input, out))
}

pub unsafe fn csp_input_get_last_double(input: CCspInputHandle, out: *mut f64) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspInputGetLastDouble>(b"ccsp_input_get_last_double\0").map(|f| f(input, out))
}

pub unsafe fn csp_input_get_last_string(
    input: CCspInputHandle,
    out_data: *mut *const c_char,
    out_length: *mut usize,
) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspInputGetLastString>(b"ccsp_input_get_last_string\0")
        .map(|f| f(input, out_data, out_length))
}

pub unsafe fn csp_input_get_last_datetime(
    input: CCspInputHandle,
    out: *mut CCspDateTime,
) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspInputGetLastDatetime>(b"ccsp_input_get_last_datetime\0")
        .map(|f| f(input, out))
}

pub unsafe fn csp_push_input_adapter_push_int64(
    adapter: CCspPushInputAdapterHandle,
    value: i64,
    batch: *mut c_void,
) -> Option<CCspErrorCode> {
    resolve_symbol::<FnCcspPushInputAdapterPushInt64>(b"ccsp_push_input_adapter_push_int64\0")
        .map(|f| f(adapter, value, batch))
}

// ============================================================================
// Capsule creation helpers
// ============================================================================

/// Capsule name for output adapters (must match C API)
pub const CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME: &[u8] = b"csp.c.OutputAdapterCapsule\0";

/// Capsule name for input adapters (must match C API)
pub const CSP_C_INPUT_ADAPTER_CAPSULE_NAME: &[u8] = b"csp.c.InputAdapterCapsule\0";

/// Capsule name for adapter managers (must match C API)
pub const CSP_C_ADAPTER_MANAGER_CAPSULE_NAME: &[u8] = b"csp.c.AdapterManagerCapsule\0";

/// Destructor for output adapter capsules
unsafe extern "C" fn output_adapter_capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char;
    let ptr = ffi::PyCapsule_GetPointer(capsule, name);
    if !ptr.is_null() {
        let vtable = ptr as *mut CCspOutputAdapterVTable;
        if let Some(destroy) = (*vtable).destroy {
            destroy((*vtable).user_data);
        }
        drop(Box::from_raw(vtable));
    }
}

/// Destructor for input adapter capsules
unsafe extern "C" fn input_adapter_capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = CSP_C_INPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char;
    let ptr = ffi::PyCapsule_GetPointer(capsule, name);
    if !ptr.is_null() {
        let vtable = ptr as *mut CCspPushInputAdapterVTable;
        if let Some(destroy) = (*vtable).destroy {
            destroy((*vtable).user_data);
        }
        drop(Box::from_raw(vtable));
    }
}

/// Destructor for adapter manager capsules
unsafe extern "C" fn adapter_manager_capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = CSP_C_ADAPTER_MANAGER_CAPSULE_NAME.as_ptr() as *const c_char;
    let ptr = ffi::PyCapsule_GetPointer(capsule, name);
    if !ptr.is_null() {
        let vtable = ptr as *mut CCspAdapterManagerVTable;
        if let Some(destroy) = (*vtable).destroy {
            destroy((*vtable).user_data);
        }
        drop(Box::from_raw(vtable));
    }
}

// Static capsule names (must be null-terminated and live forever)
static OUTPUT_ADAPTER_CAPSULE_NAME: &[u8] = b"csp.c.OutputAdapterCapsule\0";
static INPUT_ADAPTER_CAPSULE_NAME: &[u8] = b"csp.c.InputAdapterCapsule\0";
static ADAPTER_MANAGER_CAPSULE_NAME: &[u8] = b"csp.c.AdapterManagerCapsule\0";

/// Create a Python capsule wrapping an output adapter VTable
pub fn create_output_adapter_capsule(
    py: Python<'_>,
    vtable: CCspOutputAdapterVTable,
) -> PyResult<PyObject> {
    let vtable_box = Box::new(vtable);
    let vtable_ptr = Box::into_raw(vtable_box) as *mut c_void;

    unsafe {
        let capsule = ffi::PyCapsule_New(
            vtable_ptr,
            OUTPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(output_adapter_capsule_destructor),
        );
        if capsule.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

/// Create a Python capsule wrapping an input adapter VTable
pub fn create_input_adapter_capsule(
    py: Python<'_>,
    vtable: CCspPushInputAdapterVTable,
) -> PyResult<PyObject> {
    let vtable_box = Box::new(vtable);
    let vtable_ptr = Box::into_raw(vtable_box) as *mut c_void;

    unsafe {
        let capsule = ffi::PyCapsule_New(
            vtable_ptr,
            INPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(input_adapter_capsule_destructor),
        );
        if capsule.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

/// Create a Python capsule wrapping an adapter manager VTable
pub fn create_adapter_manager_capsule(
    py: Python<'_>,
    vtable: CCspAdapterManagerVTable,
) -> PyResult<PyObject> {
    let vtable_box = Box::new(vtable);
    let vtable_ptr = Box::into_raw(vtable_box) as *mut c_void;

    unsafe {
        let capsule = ffi::PyCapsule_New(
            vtable_ptr,
            ADAPTER_MANAGER_CAPSULE_NAME.as_ptr() as *const c_char,
            Some(adapter_manager_capsule_destructor),
        );
        if capsule.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

