#ifndef _IN_CSP_CORE_PLATFORM_H
#define _IN_CSP_CORE_PLATFORM_H
#include <limits>
#include <stdint.h>
#include <time.h>

//TODO move Likely.h defines into Platform.h

#ifdef __linux__
typedef uint64_t binding_int_t;
#else
typedef int64_t binding_int_t;
#endif

#ifdef WIN32
#include <windows.h>
#include <assert.h>
#include <synchapi.h>

#define DLL_PUBLIC __declspec(dllexport)
#define DLL_LOCAL

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
		return 64 - index;
	return 0;
}

inline uint8_t clz(uint32_t n)
{
	unsigned long index = 0;
	if (_BitScanReverse(&index, n))
		return 32 - index;
	return 0;
}

inline uint8_t clz(uint16_t n) { return clz(static_cast<uint32_t>(n)) - 16; }
inline uint8_t clz(uint8_t n)  { return clz(static_cast<uint32_t>(n)) - 24; }

#else
#define DLL_PUBLIC 
#define DLL_LOCAL __attribute__ ((visibility ("hidden")))

#define START_PACKED
#define END_PACKED __attribute__((packed))

#define NO_INLINE  __attribute__ ((noinline))

inline constexpr uint8_t clz(uint32_t n) { return __builtin_clz(n); }
inline constexpr uint8_t clz(uint64_t n) { return __builtin_clzl(n); }
inline constexpr uint8_t clz(uint16_t n) { return clz(static_cast<uint32_t>(n)) - 16; }
inline constexpr uint8_t clz(uint8_t n)  { return clz(static_cast<uint32_t>(n)) - 24; }

#endif

namespace csp
{
	// This is to work around the insanity that we cant call numeric_limits min/max properly on windows because they are #defined as amcros somewhere
	template<typename T>
	struct numeric_limits : public std::numeric_limits<T>
	{
		static constexpr T min_value() { return (std::numeric_limits<T>::min)(); }
		static constexpr T max_value() { return (std::numeric_limits<T>::max)(); }
	};

	template<typename T> constexpr const T& min_value(const T& a, const T& b) { return (std::min)(a, b); }
	template<typename T> constexpr const T& max_value(const T& a, const T& b) { return (std::max)(a, b); }
}
#endif