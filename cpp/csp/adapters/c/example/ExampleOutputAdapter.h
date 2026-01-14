#ifndef _IN_CSP_ADAPTERS_C_EXAMPLE_OUTPUT_ADAPTER_H
#define _IN_CSP_ADAPTERS_C_EXAMPLE_OUTPUT_ADAPTER_H

#include <csp/engine/c/COutputAdapter.h>



OutputAdapter * registerExampleOutputAdapter(void * properties);
void * unregisterExampleOutputAdapter(OutputAdapter * adapter);
void executeImpl();


#endif
