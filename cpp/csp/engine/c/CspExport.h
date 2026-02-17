/*
 * ABI-stable C API Export Macros for CSP Engine
 *
 * This header provides platform-independent macros for exporting C API symbols
 * from shared libraries. All C API functions should be declared with
 * CSP_C_API_EXPORT to ensure they are available for external adapters.
 */
#ifndef _IN_CSP_ENGINE_C_CSPEXPORT_H
#define _IN_CSP_ENGINE_C_CSPEXPORT_H

/*
 * CSP_C_API_EXPORT - Marks a function for export from the shared library
 *
 * On Windows: Uses __declspec(dllexport/dllimport)
 * On Unix: Uses __attribute__((visibility("default")))
 *
 * This ensures C API symbols are available for runtime linking by external
 * adapters implemented in C, Rust, or other languages.
 */
#if defined(_WIN32) || defined(_WIN64)
    #ifdef CSPIMPL_EXPORTS
        #define CSP_C_API_EXPORT __declspec(dllexport)
    #else
        #define CSP_C_API_EXPORT __declspec(dllimport)
    #endif
#else
    /* Unix/Linux/macOS - ensure default visibility */
    #define CSP_C_API_EXPORT __attribute__((visibility("default")))
#endif

#endif /* _IN_CSP_ENGINE_C_CSPEXPORT_H */
