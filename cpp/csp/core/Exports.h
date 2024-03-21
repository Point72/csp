#ifndef _IN_CSP_CORE_EXPORTS_H
#define _IN_CSP_CORE_EXPORTS_H

#ifdef WIN32

#ifdef CSP_CORE_EXPORTS
#define CSP_CORE_EXPORT __declspec(dllexport)
#else
#define CSP_CORE_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_TYPES_EXPORTS
#define CSP_TYPES_EXPORT __declspec(dllexport)
#else
#define CSP_TYPES_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_ENGINE_EXPORTS
#define CSP_ENGINE_EXPORT __declspec(dllexport)
#else
#define CSP_ENGINE_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_ADAPTER_UTILS_EXPORTS
#define CSP_ADAPTER_UTILS_EXPORT __declspec(dllexport)
#else
#define CSP_ADAPTER_UTILS_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_TYPES_EXPORTS
#define CSP_PYTHON_TYPES_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_TYPES_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_IMPL_EXPORTS
#define CSP_PYTHON_IMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_IMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_BASELIBIMPL_EXPORTS
#define CSP_PYTHON_BASELIBIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_BASELIBIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_BASKETLIBIMPL_EXPORTS
#define CSP_PYTHON_BASKETLIBIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_BASKETLIBIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_MATHIMPL_EXPORTS
#define CSP_PYTHON_MATHIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_MATHLIBIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_STATSIMPL_EXPORTS
#define CSP_PYTHON_STATSIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_STATSIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_NPSTATSIMPL_EXPORTS
#define CSP_NPSTATSIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_NPSTATSIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSP_PYTHON_NPSTATSIMPL_EXPORTS
#define CSP_PYTHON_NPSTATSIMPL_EXPORT __declspec(dllexport)
#else
#define CSP_PYTHON_NPSTATSIMPL_EXPORT __declspec(dllimport)
#endif

// this is always blank on win
#define CSP_LOCAL

#else

#define CSP_CORE_EXPORT __attribute__((visibility("default")))
#define CSP_TYPES_EXPORT __attribute__((visibility("default")))
#define CSP_ENGINE_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_TYPES_EXPORTS __attribute__((visibility("default")))
#define CSP_PYTHON_IMPL_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_BASELIBIMPL_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_BASKETLIBIMPL_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_MATHIMPL_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_STATSIMPL_EXPORT __attribute__((visibility("default")))
#define CSP_PYTHON_NPSTATSIMPL_EXPORT __attribute__((visibility("default")))

#define CSP_LOCAL __attribute__ ((visibility ("hidden")))

#endif // ifdef win32
#endif
