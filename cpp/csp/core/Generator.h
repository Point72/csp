#ifndef _IN_CSP_CORE_GENERATOR_H
#define _IN_CSP_CORE_GENERATOR_H

#include <memory>

namespace csp
{
template< typename V, typename ...Args >
class Generator
{
public:
    using Ptr = std::shared_ptr<Generator<V, Args...>>;

    virtual ~Generator() {}

    // Called to initialize the generator with the given args
    virtual void init( Args... ) = 0;

    // Called to get the next value. Return true if value retrieved, false otherwise
    virtual bool next( V &value ) = 0;
};

template< typename V, typename ...Args >
using GeneratorPtr = std::shared_ptr<Generator<V, Args...>>;
}

#endif
