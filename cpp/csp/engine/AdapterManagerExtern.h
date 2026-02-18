#ifndef _IN_CSP_ENGINE_ADAPTER_MANAGER_EXTERN_H
#define _IN_CSP_ENGINE_ADAPTER_MANAGER_EXTERN_H

/*
 * C++ wrapper for external Adapter Managers using the C ABI interface.
 *
 * This class wraps an adapter manager implemented in C (or any language with C FFI)
 * and integrates it with the CSP engine's AdapterManager interface.
 */

#include <csp/engine/AdapterManager.h>
#include <csp/engine/c/AdapterManager.h>

namespace csp
{

class AdapterManagerExtern final : public AdapterManager
{
public:
    AdapterManagerExtern( Engine * engine, const CCspAdapterManagerVTable & vtable );
    ~AdapterManagerExtern() override;

    const char * name() const override;

    void start( DateTime startTime, DateTime endTime ) override;
    void stop() override;
    DateTime processNextSimTimeSlice( DateTime time ) override;

    // Access for C API
    const CCspAdapterManagerVTable & vtable() const { return m_vtable; }

private:
    CCspAdapterManagerVTable m_vtable;
    mutable std::string m_name;  // cached name
};

} // namespace csp

#endif /* _IN_CSP_ENGINE_ADAPTER_MANAGER_EXTERN_H */
