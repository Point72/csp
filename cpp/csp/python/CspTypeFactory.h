#ifndef _IN_CSP_PYTHON_CSPTYPEFACTORY_H
#define _IN_CSP_PYTHON_CSPTYPEFACTORY_H

#include <csp/core/Platform.h>
#include <csp/engine/CspType.h>
#include <Python.h>
#include <unordered_map>

namespace csp::python
{

class CSPTYPESIMPL_EXPORT CspTypeFactory
{
public:
    static CspTypeFactory & instance();

    CspTypePtr & typeFromPyType( PyObject * );
    void removeCachedType( PyTypeObject * );

private:
    using Cache = std::unordered_map<PyTypeObject *, CspTypePtr>;
    Cache m_cache;
};

} // namespace csp::python

#endif
