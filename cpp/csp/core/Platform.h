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

#define DLL_LOCAL

#ifdef CSPTYPESIMPL_EXPORTS
#define CSPTYPESIMPL_EXPORT __declspec(dllexport)
#else
#define CSPTYPESIMPL_EXPORT __declspec(dllimport)
#endif

#ifdef CSPIMPL_EXPORTS
#define CSPIMPL_EXPORT __declspec(dllexport)
#else
#define CSPIMPL_EXPORT __declspec(dllimport)
#endif

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

#define CSPIMPL_EXPORT
#define CSPTYPESIMPL_EXPORT

#define DLL_LOCAL __attribute__ ((visibility ("hidden")))

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
