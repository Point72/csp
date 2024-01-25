#ifndef _IN_CSP_CORE_SYSTEM_H
#define _IN_CSP_CORE_SYSTEM_H

//Common low level system methods / defines
#include <string>
#include <vector>
#include <cxxabi.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <csp/core/Likely.h>

namespace csp
{

static constexpr size_t CACHELINE_SIZE = 64;

//useful for logging type information
template<typename T>
std::string cpp_type_name()
{
    int status = 0;
    std::string result = typeid(*(T*)nullptr).name();
    char * demangled = abi::__cxa_demangle(result.c_str(), NULL, NULL, &status);
    if( demangled )
    {
        result = demangled;
        free( demangled );
    }
    return result;
}

}

#endif
