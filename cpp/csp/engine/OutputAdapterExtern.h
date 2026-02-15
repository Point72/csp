#ifndef _IN_CSP_ENGINE_OUTPUT_ADAPTER_EXTERN_H
#define _IN_CSP_ENGINE_OUTPUT_ADAPTER_EXTERN_H

/*
 * C++ wrapper for external Output Adapters using the C ABI interface.
 *
 * This class wraps an adapter implemented in C (or any language with C FFI)
 * and integrates it with the CSP engine's OutputAdapter interface.
 */

#include <csp/engine/OutputAdapter.h>
#include <csp/engine/c/OutputAdapter.h>

namespace csp
{

class OutputAdapterExtern final : public OutputAdapter
{
public:
    OutputAdapterExtern( Engine * engine, const CspTypePtr & type,
                         const CCspOutputAdapterVTable & vtable );
    ~OutputAdapterExtern() override;

    const char* name() const override { return "OutputAdapterExtern"; }

    void start() override;
    void stop() override;
    void executeImpl() override;

private:
    CCspOutputAdapterVTable m_vtable;
    DateTime m_startTime;
    DateTime m_endTime;
};

}

#endif /* _IN_CSP_ENGINE_OUTPUT_ADAPTER_EXTERN_H */

