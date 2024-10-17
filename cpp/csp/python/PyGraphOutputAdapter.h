#ifndef _IN_CSP_PYTHON_PYGRAPHOUPUTADAPTER_H
#define _IN_CSP_PYTHON_PYGRAPHOUPUTADAPTER_H

#include <csp/core/Platform.h>
#include <csp/engine/GraphOutputAdapter.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

class CSPIMPL_EXPORT PyGraphOutputAdapter : public GraphOutputAdapter
{
public:
    using GraphOutputAdapter::GraphOutputAdapter;

    PyObjectPtr result();

private:
    void processResults() override;

    PyObjectPtr m_result;
};

}

#endif
