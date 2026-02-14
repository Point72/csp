#ifndef _IN_CSP_PYTHON_PYSTRUCTLIST_H
#define _IN_CSP_PYTHON_PYSTRUCTLIST_H

#include <csp/python/InitHelper.h>
#include <csp/python/PyStruct.h>
#include <csp/python/VectorWrapper.h>
#include <Python.h>
#include <vector>

namespace csp::python
{

template<typename StorageT>
struct PyStructList : public PyObject
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    PyStructList( PyStruct * p, std::vector<StorageT> & v, const CspType & type ) : pystruct( p ), vector( VectorWrapper<StorageT>( v ) ), arrayType( type )
    {
        Py_INCREF( pystruct );
    }

    PyListObject base;                // Inherit from PyListObject
    PyStruct * pystruct;              // Pointer to PyStruct for proper reference counting
    VectorWrapper<StorageT> vector;   // Field value for modifying

    const CspType & arrayType;        // We require the type information of any non-primitive type, i.e. Struct or Enum, since they contain a meta
    static PyTypeObject PyType;
    static bool s_typeRegister;

    inline CspTypePtr elemType() const { return static_cast<const CspArrayType &>( arrayType ).elemType(); }

    inline StorageT fromPythonValue( PyObject * value ) const;
};

template<typename StorageT> bool PyStructList<StorageT>::s_typeRegister = InitHelper::instance().registerCallback( 
    InitHelper::typeInitCallback( &PyStructList<StorageT>::PyType, "", &PyList_Type ) );

}

#endif
