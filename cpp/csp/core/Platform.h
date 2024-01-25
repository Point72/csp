#ifndef _IN_CSP_CORE_PLATFORM_H
#define _IN_CSP_CORE_PLATFORM_H
#include <stdint.h>

#ifdef __linux__
    typedef uint64_t binding_int_t;
#else
    typedef int64_t binding_int_t;
#endif

#endif