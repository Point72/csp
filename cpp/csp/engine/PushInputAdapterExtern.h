#ifndef _IN_CSP_ENGINE_PUSH_INPUT_ADAPTER_EXTERN_H
#define _IN_CSP_ENGINE_PUSH_INPUT_ADAPTER_EXTERN_H

/*
 * C++ wrapper for external Push Input Adapters using the C ABI interface.
 *
 * This class wraps an adapter implemented in C (or any language with C FFI)
 * and integrates it with the CSP engine's PushInputAdapter interface.
 */

#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/c/InputAdapter.h>

namespace csp
{

class PushInputAdapterExtern final : public PushInputAdapter
{
public:
    PushInputAdapterExtern( Engine * engine, CspTypePtr & type, PushMode pushMode,
                            PushGroup * group, const CCspPushInputAdapterVTable & vtable );
    ~PushInputAdapterExtern() override;

    const char* name() const override { return "PushInputAdapterExtern"; }

    void start( DateTime startTime, DateTime endTime ) override;
    void stop() override;

    // Get the vtable for access to user_data
    const CCspPushInputAdapterVTable & vtable() const { return m_vtable; }

private:
    CCspPushInputAdapterVTable m_vtable;
    DateTime m_startTime;
    DateTime m_endTime;
};

}

#endif /* _IN_CSP_ENGINE_PUSH_INPUT_ADAPTER_EXTERN_H */
