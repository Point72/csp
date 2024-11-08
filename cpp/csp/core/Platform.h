#ifndef _IN_CSP_CORE_PLATFORM_H
#define _IN_CSP_CORE_PLATFORM_H
#include <type_traits>
#include <stdint.h>
#include <time.h>

//TODO move Likely.h defines into Platform.h
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#include <assert.h>
#include <synchapi.h>

#undef ERROR
#undef GetMessage

#ifdef KAFKAADAPTERIMPL_EXPORTS
#define KAFKAADAPTERIMPL_EXPORT __declspec(dllexport)
#define CSPKAFKAADAPTER_EXPORT __declspec(dllexport)
#else
#define KAFKAADAPTERIMPL_EXPORT __declspec(dllimport)
#define CSPKAFKAADAPTER_EXPORT
#endif

#ifdef CSPKAFKAADAPTER_EXPORTS
// static
#endif

#ifdef PARQUETADAPTERIMPL_EXPORTS
#define PARQUETADAPTERIMPL_EXPORT __declspec(dllexport)
#define CSPPARQUETADAPTER_EXPORT __declspec(dllexport)
#else
#define PARQUETADAPTERIMPL_EXPORT __declspec(dllimport)
#define CSPPARQUETADAPTER_EXPORT
#endif

#ifdef CSPPARQUETADAPTER_EXPORTS
// static
#endif

#ifdef WEBSOCKETADAPTERIMPL_EXPORTS
#define WEBSOCKETADAPTERIMPL_EXPORTS __declspec(dllexport)
#define CSPWEBSOCKETCLIENTADAPTER_EXPORT __declspec(dllexport)
#else
#define WEBSOCKETADAPTERIMPL_EXPORTS __declspec(dllimport)
#define CSPWEBSOCKETCLIENTADAPTER_EXPORT
#endif

#ifdef CSPWEBSOCKETCLIENTADAPTER_EXPORTS
// static
#endif

// #ifdef CSPADAPTERUTILS_EXPORTS
#define CSPADAPTERUTILS_EXPORT
// #endif

#ifdef CSPCORE_EXPORTS
// static
#endif

#ifdef BASELIBIMPL_EXPORTS
// static
#endif

#ifdef BASKETLIBIMPL_EXPORTS
// static
#endif

#ifdef MATHIMPL_EXPORTS
// static
#endif

#ifdef STATSIMPL_EXPORTS
// static
#endif

#ifdef CSPTYPES_EXPORTS
// static
#endif

#ifdef CSPENGINE_EXPORTS
// static
#endif

#ifdef CSPTYPESIMPL_EXPORTS
#define CSPTYPESIMPL_EXPORT __declspec(dllexport)
#define CSPTYPES_EXPORT __declspec(dllexport)
#else
#define CSPTYPESIMPL_EXPORT __declspec(dllimport)
#define CSPTYPES_EXPORT
#endif

#ifdef CSPIMPL_EXPORTS
#define CSPIMPL_EXPORT __declspec(dllexport)
#define CSPCORE_EXPORT __declspec(dllexport)
#define CSPENGINE_EXPORT __declspec(dllexport)
#else
// #define CSPIMPL_EXPORT  __declspec(dllimport)
#define CSPIMPL_EXPORT
#define CSPCORE_EXPORT
#define CSPENGINE_EXPORT
#endif

#ifdef CSPBASELIBIMPL_EXPORTS
#define CSPBASELIBIMPL_EXPORT __declspec(dllexport)
#define BASELIBIMPL_EXPORT __declspec(dllexport)
#else
#define CSPBASELIBIMPL_EXPORT __declspec(dllimport)
#define BASELIBIMPL_EXPORT
#endif

#ifdef CSPBASKETLIBIMPL_EXPORTS
#define CSPBASKETLIBIMPL_EXPORT __declspec(dllexport)
#define BASKETLIBIMPL_EXPORT __declspec(dllexport)
#else
#define CSPBASKETLIBIMPL_EXPORT __declspec(dllimport)
#define BASKETLIBIMPL_EXPORT
#endif

#ifdef CSPMATHIMPL_EXPORTS
#define CSPMATHIMPL_EXPORT __declspec(dllexport)
#define MATHIMPL_EXPORT __declspec(dllexport)
#else
#define CSPMATHIMPL_EXPORT __declspec(dllimport)
#define MATHIMPL_EXPORT
#endif

#ifdef CSPSTATSIMPL_EXPORTS
#define CSPSTATSIMPL_EXPORT __declspec(dllexport)
#define STATSIMPL_EXPORT __declspec(dllexport)
#else
#define CSPSTATSIMPL_EXPORT __declspec(dllimport)
#define STATSIMPL_EXPORT
#endif

#ifdef CSPTESTLIBIMPL_EXPORTS
#define CSPTESTLIBIMPL_EXPORT __declspec(dllexport)
#else
#define CSPTESTLIBIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef NPSTATSIMPL_EXPORTS
#define NPSTATSIMPL_EXPORT __declspec(dllexport)
#define CSPNPSTATSIMPL_EXPORT __declspec(dllexport)
#else
#define NPSTATSIMPL_EXPORT __declspec(dllimport)
#define CSPNPSTATSIMPL_EXPORT
#endif

#define CSP_PYTHON_EXPORT
#define CSP_PUBLIC_EXPORT __declspec(dllexport)

// KAFKAADAPTERIMPL_EXPORTS
// CSPKAFKAADAPTER_EXPORTS
// PARQUETADAPTERIMPL_EXPORTS
// CSPPARQUETADAPTER_EXPORTS
// WEBSOCKETADAPTERIMPL_EXPORTS
// CSPWEBSOCKETCLIENTADAPTER_EXPORTS
// CSPADAPTERUTILS_EXPORTS
// CSPCORE_EXPORTS
// BASELIBIMPL_EXPORTS
// BASKETLIBIMPL_EXPORTS
// MATHIMPL_EXPORTS
// STATSIMPL_EXPORTS
// CSPTYPES_EXPORTS
// CSPENGINE_EXPORTS
// CSPTYPESIMPL_EXPORTS
// CSPIMPL_EXPORTS
// BASELIBIMPL_EXPORTS
// CSPBASKETLIBIMPL_EXPORTS
// CSPMATHIMPL_EXPORTS
// CSPSTATSIMPL_EXPORTS
// CSPTESTLIBIMPL_EXPORTS
// NPSTATSIMPL_EXPORTS
// CSPNPSTATSIMPL_EXPORTS

#define START_PACKED __pragma( pack(push, 1) )
#define END_PACKED   __pragma( pack(pop))

#define NO_INLINE    __declspec(noinline)

inline tm * localtime_r( const time_t * timep, tm * result )
{
    tm * rv = localtime(timep);
    if (rv)
        *result = *rv;

    return result;
}

#define timegm _mkgmtime

inline int nanosleep(const timespec* req, timespec* rem)
{
    assert(rem == nullptr);
    int64_t millis = req->tv_sec * 1000 + req->tv_nsec * 1000000;
    Sleep(millis);
    return 0;
}

inline uint8_t clz(uint64_t n)
{
    unsigned long index = 0;
    if (_BitScanReverse64(&index, n))
	    return 64 - index - 1;
    return 0;
}

inline uint8_t clz(uint32_t n)
{
    unsigned long index = 0;
    if (_BitScanReverse(&index, n))
	    return 32 - index - 1;
    return 0;
}

inline uint8_t clz(uint16_t n) { return clz(static_cast<uint32_t>(n)) - 16; }
inline uint8_t clz(uint8_t n)  { return clz(static_cast<uint32_t>(n)) - 24; }

template<typename U, std::enable_if_t<std::is_unsigned<U>::value, bool> = true>
inline uint8_t ffs(U n)
{ 
    unsigned long index = 0;
    if (_BitScanForward(&index, n))
	    return index + 1;
    return 0;
}

inline uint8_t ffs(uint64_t n)
{
    unsigned long index = 0;
    if (_BitScanForward64(&index, n))
	    return index + 1;
    return 0;
}

#else

#define KAFKAADAPTERIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPKAFKAADAPTER_EXPORT __attribute__((visibility ("default")))
#define PARQUETADAPTERIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPPARQUETADAPTER_EXPORT __attribute__((visibility ("default")))
#define WEBSOCKETADAPTERIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPWEBSOCKETCLIENTADAPTER_EXPORT __attribute__((visibility ("default")))
#define CSPADAPTERUTILS_EXPORT __attribute__((visibility ("default")))
#define CSPCORE_EXPORT __attribute__((visibility ("default")))
#define BASELIBIMPL_EXPORT __attribute__((visibility ("default")))
#define BASKETLIBIMPL_EXPORT __attribute__((visibility ("default")))
#define MATHIMPL_EXPORT __attribute__((visibility ("default")))
#define STATSIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPTYPES_EXPORT __attribute__((visibility ("default")))
#define CSPENGINE_EXPORT __attribute__((visibility ("default")))
#define CSPTYPESIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPBASELIBIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPBASKETLIBIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPMATHIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPSTATSIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPTESTLIBIMPL_EXPORT __attribute__((visibility ("default")))
#define NPSTATSIMPL_EXPORT __attribute__((visibility ("default")))
#define CSPNPSTATSIMPL_EXPORT __attribute__((visibility ("default")))

// Others
#define CSP_PYTHON_EXPORT  __attribute__ ((visibility ("hidden")))
#define CSP_PUBLIC_EXPORT __attribute__((visibility("default")))

#define START_PACKED
#define END_PACKED __attribute__((packed))

#define NO_INLINE  __attribute__ ((noinline))

inline constexpr uint8_t clz(uint32_t n) { return __builtin_clz(n); }
inline constexpr uint8_t clz(uint64_t n) { return __builtin_clzl(n); }

// clz (count leading zeros) returns number of leading zeros before MSB (i.e. clz(00110..) = 2 )
// __builtin_clz auto-promotes to 32-bits: need to subtract off extra leading zeros
inline constexpr uint8_t clz(uint16_t n) { return clz(static_cast<uint32_t>(n)) - 16; }
inline constexpr uint8_t clz(uint8_t n)  { return clz(static_cast<uint32_t>(n)) - 24; }

// ffs (find first set) returns offset of first set bit (i.e. ffs(..0110) = 2 ), with ffs(0) = 0
template<typename U, std::enable_if_t<std::is_unsigned<U>::value, bool> = true>
inline constexpr uint8_t ffs( U n )        { return __builtin_ffs(n); }
inline constexpr uint8_t ffs( uint64_t n ) { return __builtin_ffsl(n); }

#endif

#endif