#ifndef _IN_CSP_PYTHON_PYSTRUCT_H
#define _IN_CSP_PYTHON_PYSTRUCT_H

#include <csp/core/Platform.h>
#include <csp/engine/Struct.h>
#include <csp/python/PyObjectPtr.h>
#include <memory>
#include <string>

namespace csp::python
{

//This is the base class of csp.StructMeta
struct CSPTYPESIMPL_EXPORT PyStructMeta : public PyHeapTypeObject
{
    std::shared_ptr<StructMeta> structMeta;
    PyObjectPtr                 attrDict; //mapping of attribute key -> PyCapsule holding the StructField * 

    static PyTypeObject PyType;
};

//This is an extension of csp::StructMeta for python dialect, we need it in order to 
//keep a reference to the python struct type from conversion to/from csp::Struct <-> PyObject properly
class CSPTYPESIMPL_EXPORT DialectStructMeta : public StructMeta
{
public:
    DialectStructMeta( PyTypeObject * pyType, const std::string & name, 
                       const Fields & fields, std::shared_ptr<StructMeta> base = nullptr );
    ~DialectStructMeta() {}

    PyTypeObject * pyType() const { return m_pyType.get(); }

    const StructField * field( PyObject * attr ) const
    {
        PyObject * field = PyDict_GetItem( ( ( PyStructMeta * ) m_pyType.get() ) -> attrDict.get(), attr );
        if( likely( field != nullptr ) )
            return ( StructField * ) PyCapsule_GetPointer( field, nullptr );
        return nullptr;
    }

private:
    PyTypeObjectPtr m_pyType;
};


struct CSPTYPESIMPL_EXPORT PyStruct : public PyObject
{
    PyStruct( const StructPtr & s ) : struct_( s ) {}
    PyStruct( StructPtr && s ) : struct_( std::move( s ) ) {}

    StructPtr struct_;

    PyStructMeta * pyStructMeta() { return ( PyStructMeta * ) ob_type; };

    const DialectStructMeta * structMeta() { return static_cast<const DialectStructMeta*>( struct_ -> meta() ); }

    //Helper methods
    PyObject * getattr( PyObject * attr );
    void       setattr( PyObject * attr, PyObject * value ) { setattr( struct_.get(), attr, value ); }

    static void setattr( Struct * s, PyObject * attr, PyObject * value );

    PyObject * repr( bool show_unset ) const;

    static bool isPyStructType( PyTypeObject * typ )
    {
        if( !( typ -> tp_flags & Py_TPFLAGS_HEAPTYPE ) )
            return false;

        if( !PyType_IsSubtype( typ, &PyType ) )
            return false;
            
        //make sure its not csp.Struct python type itself
        PyHeapTypeObject * htyp = ( PyHeapTypeObject * ) typ;
        return htyp -> ht_type.tp_base != &PyType;
    }

    static PyTypeObject PyType;
};

// Array struct field printing function
template<typename ElemT>
void repr_array( const std::vector<ElemT> & val, const CspType & elemType, std::string & tl_repr, bool show_unset );

}

#endif
