#ifndef _IN_CSP_ENGINE_OUTPUT_ADAPTER_EXTERN_H
#define _IN_CSP_ENGINE_OUTPUT_ADAPTER_EXTERN_H

/*
 * This file wraps an external OutputAdapter implementation into a C++ OutputAdapter
 * communicating across the ABI-stable C interface.
*/

namespace csp
{
    
    class OutputAdapterExtern : public OutputAdapter
    {
    public:
        OutputAdapterExtern(/* parameters to construct the external adapter */);
        ~OutputAdapterExtern();

        void executeImpl();

    private:
        struct COutputAdapter* c_adapter_; // Opaque pointer to the C Output Adapter
    };

};

#endif
