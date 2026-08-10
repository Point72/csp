#ifndef _IN_CSP_PYTHON_CSPTYPEFACTORY_H
#define _IN_CSP_PYTHON_CSPTYPEFACTORY_H

#include <csp/core/Platform.h>
#include <csp/engine/CspType.h>
#include <csp/python/PyObjectPtr.h>
#include <unordered_map>
#include <Python.h>

namespace csp::python
{

class CSPTYPESIMPL_EXPORT CspTypeFactory
{
public:
    static CspTypeFactory & instance();

    CspTypePtr & typeFromPyType( PyObject * );
    void removeCachedType( PyTypeObject * );

    PyTypeObject * intEnumPyType() { return m_intEnumPyType.get(); }

private:
    using Cache = std::unordered_map<PyTypeObject *, CspTypePtr>;

    std::shared_ptr<CspEnumMeta> createCspEnumMetaFromIntEnum( PyTypeObjectPtr pyIntEnumType );

    CspTypeFactory();
    Cache m_cache;

    PyTypeObjectPtr m_intEnumPyType;
};

}

#endif
