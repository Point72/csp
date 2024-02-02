#ifndef _IN_CSP_PYTHON_CSPENUM_H
#define _IN_CSP_PYTHON_CSPENUM_H

#include <csp/engine/CspEnum.h>
#include <csp/python/PyObjectPtr.h>
#include <memory>
#include <string>

namespace csp::python
{

//This is the base class of csp.Enum
struct PyCspEnumMeta : public PyHeapTypeObject
{
    //convert to PyObject ( new ref )
    PyObject * toPyEnum( CspEnum e ) const;

    std::shared_ptr<CspEnumMeta> enumMeta;

    PyObjectPtr enumsByName;
    PyObjectPtr enumsByValue;

    //for fast toPython calls
    std::unordered_map<int64_t,PyObjectPtr> enumsByCValue;

    static PyTypeObject PyType;
};

//This is an extension of csp::CspEnumMeta for python dialect, we need it in order to 
//keep a reference to the python enum type from conversion to/from csp::CspEnumMeta <-> PyObject properly
class DialectCspEnumMeta : public CspEnumMeta
{
public:
    DialectCspEnumMeta( PyTypeObjectPtr pyType, const std::string & name, 
                        const CspEnumMeta::ValueDef & def );
    ~DialectCspEnumMeta() {}

    const PyTypeObjectPtr & pyType() const { return m_pyType; }

    const PyCspEnumMeta * pyMeta() const   { return ( const PyCspEnumMeta * ) m_pyType.get(); }

private:

    PyTypeObjectPtr m_pyType;
};

struct PyCspEnum : public PyObject
{
    PyCspEnum( const CspEnum & e ) : enum_( e ) {}
    ~PyCspEnum() {}

    CspEnum enum_;
    PyObjectPtr enumName;
    PyObjectPtr enumValue;

    PyCspEnumMeta * pyMeta() { return ( PyCspEnumMeta * ) ob_type; };
    const DialectCspEnumMeta * meta() { return static_cast<const DialectCspEnumMeta*>( pyMeta() -> enumMeta.get() ); }

    static PyTypeObject PyType;
};

}

#endif
