//! FFI bindings for the CSP C API
//!
//! These bindings match the C ABI structures defined in the CSP headers.
//! See `cpp/csp/engine/c/` for the C header files.
//!
//! Note: The actual CSP C API functions (ccsp_*) are resolved at runtime when
//! this module is loaded alongside CSP. For standalone testing, these functions
//! are stubbed.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;
use pyo3::prelude::*;
use pyo3::ffi;
use libc::{dlsym, RTLD_DEFAULT};

/// CSP DateTime type (nanoseconds since epoch)
pub type CCspDateTime = i64;

/// Sentinel for "no datetime", mirroring `CCSP_DATETIME_NONE` in `cpp/csp/engine/c/CspTime.h`.
/// Distinct from 0, which is the valid Unix epoch timestamp.
pub const CCSP_DATETIME_NONE: CCspDateTime = i64::MIN;

/// Callback-table ABI version, mirroring `CCSP_ABI_VERSION` in `cpp/csp/engine/c/CspAbi.h`.
pub const CCSP_ABI_VERSION: u32 = 1;

/// Opaque handle to the CSP engine
pub type CCspEngineHandle = *mut c_void;

/// Opaque handle to a push input adapter
pub type CCspPushInputAdapterHandle = *mut c_void;

/// Opaque handle to an adapter manager
pub type CCspAdapterManagerHandle = *mut c_void;

/// Opaque handle to an input (for output adapters)
pub type CCspInputHandle = *mut c_void;

/// CSP type enumeration
///
/// Discriminants must match `CCspType` in `cpp/csp/engine/c/CspType.h` exactly.
#[repr(i32)]
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
    String = 11,
    DateTime = 12,
    TimeDelta = 13,
    Date = 14,
    Time = 15,
    Enum = 16,
    Struct = 17,
    Array = 18,
    DialectGeneric = 19,
}

impl CCspType {
    /// Materialising an out-of-range discriminant into a `repr` enum is undefined behaviour,
    /// so every value crossing the FFI boundary is converted through here.
    fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Unknown),
            1 => Some(Self::Bool),
            2 => Some(Self::Int8),
            3 => Some(Self::Uint8),
            4 => Some(Self::Int16),
            5 => Some(Self::Uint16),
            6 => Some(Self::Int32),
            7 => Some(Self::Uint32),
            8 => Some(Self::Int64),
            9 => Some(Self::Uint64),
            10 => Some(Self::Double),
            11 => Some(Self::String),
            12 => Some(Self::DateTime),
            13 => Some(Self::TimeDelta),
            14 => Some(Self::Date),
            15 => Some(Self::Time),
            16 => Some(Self::Enum),
            17 => Some(Self::Struct),
            18 => Some(Self::Array),
            19 => Some(Self::DialectGeneric),
            _ => None,
        }
    }
}

/// CSP Error codes
///
/// Discriminants must match `CCspErrorCode` in `cpp/csp/engine/c/CspError.h` exactly.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CCspErrorCode {
    Ok = 0,
    NullPointer = 1,
    TypeMismatch = 2,
    KeyNotFound = 3,
    InvalidArgument = 4,
    OutOfMemory = 5,
    OutOfRange = 6,
    Runtime = 7,
    Value = 8,
    NotImplemented = 9,
    Unknown = 10,
}

impl CCspErrorCode {
    fn from_raw(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Ok),
            1 => Some(Self::NullPointer),
            2 => Some(Self::TypeMismatch),
            3 => Some(Self::KeyNotFound),
            4 => Some(Self::InvalidArgument),
            5 => Some(Self::OutOfMemory),
            6 => Some(Self::OutOfRange),
            7 => Some(Self::Runtime),
            8 => Some(Self::Value),
            9 => Some(Self::NotImplemented),
            10 => Some(Self::Unknown),
            _ => None,
        }
    }
}

// ============================================================================
// VTable structures (matching the C API headers)
// ============================================================================

/// VTable for output adapter callbacks
#[repr(C)]
pub struct CCspOutputAdapterVTable {
    /// ABI header, mirroring `CCSP_VTABLE_INIT` in `cpp/csp/engine/c/CspAbi.h`
    pub abi_version: u32,
    pub struct_size: u32,

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
    /// ABI header, mirroring `CCSP_VTABLE_INIT` in `cpp/csp/engine/c/CspAbi.h`
    pub abi_version: u32,
    pub struct_size: u32,

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
    /// ABI header, mirroring `CCSP_VTABLE_INIT` in `cpp/csp/engine/c/CspAbi.h`
    pub abi_version: u32,
    pub struct_size: u32,

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
type FnCcspInputGetType = unsafe extern "C" fn(input: CCspInputHandle) -> i32;
type FnCcspInputGetLastBool = unsafe extern "C" fn(input: CCspInputHandle, out: *mut i8) -> i32;
type FnCcspInputGetLastInt64 = unsafe extern "C" fn(input: CCspInputHandle, out: *mut i64) -> i32;
type FnCcspInputGetLastDouble = unsafe extern "C" fn(input: CCspInputHandle, out: *mut f64) -> i32;
type FnCcspInputGetLastString = unsafe extern "C" fn(
    input: CCspInputHandle,
    out_data: *mut *const c_char,
    out_length: *mut usize,
) -> i32;
type FnCcspInputGetLastDatetime = unsafe extern "C" fn(
    input: CCspInputHandle,
    out: *mut CCspDateTime,
) -> i32;
type FnCcspPushInputAdapterPushInt64 = unsafe extern "C" fn(
    adapter: CCspPushInputAdapterHandle,
    value: i64,
    batch: *mut c_void,
) -> i32;

/// Resolves a CSP symbol from the already-loaded process image, once per symbol.
///
/// The lookup is Unix-only: the example relies on `RTLD_DEFAULT`, which has no direct Windows
/// equivalent. Each wrapper caches the resolved address so the per-tick push path does not repeat
/// a symbol-table lookup.
macro_rules! csp_symbol {
    ( $vis:vis fn $name:ident -> $ty:ty = $sym:literal ) => {
        $vis unsafe fn $name() -> Option<$ty> {
            static CACHE: OnceLock<Option<usize>> = OnceLock::new();

            // transmute below requires the target type to be pointer sized
            const _: () = assert!(std::mem::size_of::<$ty>() == std::mem::size_of::<usize>());

            let address = *CACHE.get_or_init(|| {
                let symbol = dlsym(RTLD_DEFAULT, concat!($sym, "\0").as_ptr() as *const c_char);
                if symbol.is_null() {
                    None
                } else {
                    Some(symbol as usize)
                }
            });

            address.map(|a| std::mem::transmute::<usize, $ty>(a))
        }
    };
}

csp_symbol!(fn sym_engine_now -> FnCcspEngineNow = "ccsp_engine_now");
csp_symbol!(fn sym_input_get_type -> FnCcspInputGetType = "ccsp_input_get_type");
csp_symbol!(fn sym_input_get_last_bool -> FnCcspInputGetLastBool = "ccsp_input_get_last_bool");
csp_symbol!(fn sym_input_get_last_int64 -> FnCcspInputGetLastInt64 = "ccsp_input_get_last_int64");
csp_symbol!(fn sym_input_get_last_double -> FnCcspInputGetLastDouble = "ccsp_input_get_last_double");
csp_symbol!(fn sym_input_get_last_string -> FnCcspInputGetLastString = "ccsp_input_get_last_string");
csp_symbol!(fn sym_input_get_last_datetime -> FnCcspInputGetLastDatetime = "ccsp_input_get_last_datetime");
csp_symbol!(fn sym_push_int64 -> FnCcspPushInputAdapterPushInt64 = "ccsp_push_input_adapter_push_int64");

pub unsafe fn csp_engine_now(engine: CCspEngineHandle) -> Option<CCspDateTime> {
    sym_engine_now().map(|f| f(engine))
}

/// Returns `None` if the symbol is unavailable or the C side returned a value this build does
/// not know about.
pub unsafe fn csp_input_get_type(input: CCspInputHandle) -> Option<CCspType> {
    sym_input_get_type().and_then(|f| CCspType::from_raw(f(input)))
}

pub unsafe fn csp_input_get_last_bool(input: CCspInputHandle, out: *mut i8) -> Option<CCspErrorCode> {
    sym_input_get_last_bool().and_then(|f| CCspErrorCode::from_raw(f(input, out)))
}

pub unsafe fn csp_input_get_last_int64(input: CCspInputHandle, out: *mut i64) -> Option<CCspErrorCode> {
    sym_input_get_last_int64().and_then(|f| CCspErrorCode::from_raw(f(input, out)))
}

pub unsafe fn csp_input_get_last_double(input: CCspInputHandle, out: *mut f64) -> Option<CCspErrorCode> {
    sym_input_get_last_double().and_then(|f| CCspErrorCode::from_raw(f(input, out)))
}

pub unsafe fn csp_input_get_last_string(
    input: CCspInputHandle,
    out_data: *mut *const c_char,
    out_length: *mut usize,
) -> Option<CCspErrorCode> {
    sym_input_get_last_string().and_then(|f| CCspErrorCode::from_raw(f(input, out_data, out_length)))
}

pub unsafe fn csp_input_get_last_datetime(
    input: CCspInputHandle,
    out: *mut CCspDateTime,
) -> Option<CCspErrorCode> {
    sym_input_get_last_datetime().and_then(|f| CCspErrorCode::from_raw(f(input, out)))
}

pub unsafe fn csp_push_input_adapter_push_int64(
    adapter: CCspPushInputAdapterHandle,
    value: i64,
    batch: *mut c_void,
) -> Option<CCspErrorCode> {
    sym_push_int64().and_then(|f| CCspErrorCode::from_raw(f(adapter, value, batch)))
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

/// Capsule context set by the CSP bridge once `user_data` ownership has moved to the engine.
/// The capsule still owns the VTable allocation and must drop it with the allocator that made it,
/// but must not run the destroy callback a second time.
/// Mirrors `CCSP_PY_CAPSULE_TRANSFERRED` in `cpp/csp/python/c/`.
const CAPSULE_TRANSFERRED: *mut c_void = 1usize as *mut c_void;

unsafe fn capsule_ownership_transferred(capsule: *mut ffi::PyObject) -> bool {
    ffi::PyCapsule_GetContext(capsule) == CAPSULE_TRANSFERRED
}

/// Destructor for output adapter capsules
unsafe extern "C" fn output_adapter_capsule_destructor(capsule: *mut ffi::PyObject) {
    let name = CSP_C_OUTPUT_ADAPTER_CAPSULE_NAME.as_ptr() as *const c_char;
    let ptr = ffi::PyCapsule_GetPointer(capsule, name);
    if !ptr.is_null() {
        let vtable = ptr as *mut CCspOutputAdapterVTable;
        if !capsule_ownership_transferred(capsule) {
            if let Some(destroy) = (*vtable).destroy {
                destroy((*vtable).user_data);
            }
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
        if !capsule_ownership_transferred(capsule) {
            if let Some(destroy) = (*vtable).destroy {
                destroy((*vtable).user_data);
            }
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
        if !capsule_ownership_transferred(capsule) {
            if let Some(destroy) = (*vtable).destroy {
                destroy((*vtable).user_data);
            }
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
            drop(Box::from_raw(vtable_ptr as *mut CCspOutputAdapterVTable));
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
            drop(Box::from_raw(vtable_ptr as *mut CCspPushInputAdapterVTable));
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
            drop(Box::from_raw(vtable_ptr as *mut CCspAdapterManagerVTable));
            return Err(PyErr::fetch(py));
        }
        Ok(PyObject::from_owned_ptr(py, capsule))
    }
}

