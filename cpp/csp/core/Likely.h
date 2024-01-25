#ifndef _IN_CSP_CORE_LIKELY_H
#define _IN_CSP_CORE_LIKELY_H

#define likely(x)   __builtin_expect ( (x), 1 )
#define unlikely(x) __builtin_expect ( (x), 0 )

#endif
