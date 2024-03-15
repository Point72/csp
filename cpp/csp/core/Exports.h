#ifndef _IN_CSP_CORE_EXPORTS_H
#define _IN_CSP_CORE_EXPORTS_H

#ifdef WIN32

/*
 * <static libs>
 *        \---- csptypesimpl
 *                    \---- cspimpl
 *                             \---- all other libs
 */   

// csptypesimple is imported from cspimpl
// NOTE: DialectGenericType is in csp_types static lib,
// but will be exported symbol from csptypesimpl
#ifdef CSP_TYPES_EXPORTS
#define CSP_TYPES_EXPORT __declspec(dllexport)
#define CSP_CORE_EXPORT __declspec(dllexport)
#else
#define CSP_TYPES_EXPORT __declspec(dllimport)
#define CSP_CORE_EXPORT
#endif

// cspimpl is imported by all downstream libs
#ifdef CSP_IMPL_EXPORTS
#define CSP_IMPL_EXPORT __declspec(dllexport)
#define CSP_CORE_EXPORT __declspec(dllimport)
#else
#define CSP_IMPL_EXPORT __declspec(dllimport)
#endif

// all other libs are terminal
#ifdef CSP_EXPORTS
#define CSP_EXPORT __declspec(dllexport)
#define CSP_CORE_EXPORT __declspec(dllimport)
#else
#define CSP_EXPORT __declspec(dllimport)
#endif

// this is always blank on win
#define CSP_LOCAL

#else

#define CSP_EXPORT __attribute__((visibility("default")))
#define CSP_LOCAL __attribute__ ((visibility ("hidden")))

#endif // WIN32

#endif
