#ifndef _IN_CSP_PYTHON_CSPENUM_H
#define _IN_CSP_PYTHON_CSPENUM_H

#include <csp/core/Platform.h>
#include <csp/engine/CspEnum.h>
#include <csp/python/PyObjectPtr.h>
#include <memory>
#include <string>

namespace csp::python
{

//TODO Windows - need to figure out why adding DLL_PUBLIC to this class leads to weird compilation errors on CspEnumMeta's unordered_map...

//This is an extension of csp::CspEnumMeta for python dialect, we need it in order to 
//keep a reference to the python enum type from conversion to/from csp::CspEnumMeta <-> PyObject properly
class CSPTYPESIMPL_EXPORT DialectCspEnumMeta : public CspEnumMeta
{
public:
    DialectCspEnumMeta( PyTypeObjectPtr pyType, const std::string & name, 
                        const CspEnumMeta::ValueDef & def );
    ~DialectCspEnumMeta() {}

    const PyTypeObjectPtr & pyType() const { return m_pyType; }

    //returns new ref
    PyObject * toPyEnum( int64_t value ) const
    {
        auto it = m_enumsByCValue.find( value );
        if( it == m_enumsByCValue.end() )
            return nullptr;

        PyObject * rv = it -> second.get();
        Py_INCREF( rv );
        return rv;
    }
    
private:

    PyTypeObjectPtr m_pyType;
    PyObjectPtr     m_enumsByName;

    //for fast toPython calls
    std::unordered_map<int64_t,PyObjectPtr> m_enumsByCValue;
};

}

#endif
