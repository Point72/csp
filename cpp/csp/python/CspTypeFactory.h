#ifndef _IN_CSP_PYTHON_CSPTYPEFACTORY_H
#define _IN_CSP_PYTHON_CSPTYPEFACTORY_H

#include <csp/core/Platform.h>
#include <csp/engine/CspType.h>
#include <unordered_map>
#include <Python.h>

namespace csp::python
{

class DLL_PUBLIC CspTypeFactory
{
public:
    static CspTypeFactory & instance();

    CspTypePtr & typeFromPyType( PyObject * );
    void removeCachedType( PyTypeObject * );

private:
    using Cache = std::unordered_map<PyTypeObject *, CspTypePtr>;
    Cache m_cache;
};

}

#endif
