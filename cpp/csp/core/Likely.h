#ifndef _IN_CSP_CORE_LIKELY_H
#define _IN_CSP_CORE_LIKELY_H

// We should move to [[likely]] [[unlikely]] attributes once we enable c++20
#ifndef WIN32
#define likely(x)   __builtin_expect ( (x), 1 )
#define unlikely(x) __builtin_expect ( (x), 0 )
#else
#define likely(x)   x
#define unlikely(x) x
#endif

#endif
