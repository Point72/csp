#ifndef _IN_CSP_PYTHON_PYSTRUCTFASTLIST_H
#define _IN_CSP_PYTHON_PYSTRUCTFASTLIST_H

#include <csp/core/Platform.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyIterator.h>
#include <csp/python/PyStruct.h>
#include <csp/python/VectorWrapper.h>
#include <Python.h>
#include <vector>

namespace csp::python
{

template<typename StorageT>
struct CSPIMPL_EXPORT PyStructFastList : public PyObject
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    PyStructFastList( PyStruct * p, std::vector<StorageT> & v, const CspType & type ) : pystruct( p ), vector( VectorWrapper<StorageT>( v ) ), arrayType( type )
    {
        Py_INCREF( pystruct );
    }

    PyStruct * pystruct;              // Pointer to PyStruct for proper reference counting
    VectorWrapper<StorageT> vector;   // Field value for modifying

    const CspType & arrayType;        // We require the type information of any non-primitive type, i.e. Struct or Enum, since they contain a meta
    static PyTypeObject PyType;
    static bool s_typeRegister;

    inline CspTypePtr elemType() const { return static_cast<const CspArrayType &>( arrayType ).elemType(); }

    inline PyObject * toPythonValue( const StorageT & value ) const;
    inline StorageT fromPythonValue( PyObject * value ) const;
};

template<typename StorageT> bool PyStructFastList<StorageT>::s_typeRegister = InitHelper::instance().registerCallback( 
    InitHelper::typeInitCallback( &PyStructFastList<StorageT>::PyType, "" ) );

}

#endif
