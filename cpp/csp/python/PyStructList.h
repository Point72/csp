#ifndef _IN_CSP_PYTHON_PYSTRUCTLIST_H
#define _IN_CSP_PYTHON_PYSTRUCTLIST_H

#include <csp/python/InitHelper.h>
#include <csp/python/PyStruct.h>
#include <Python.h>
#include <vector>

namespace csp::python
{

template<typename StorageT>
struct PyStructList : public PyObject
{
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    PyStructList( PyStruct * p, std::vector<StorageT> & v, const CspType & type ) : pystruct( p ), vector( v ), field_type( type )
    {
        Py_INCREF( pystruct );
    }

    PyListObject base;                // Inherit from PyListObject
    PyStruct * pystruct;             // Pointer to PyStruct for proper reference counting
    std::vector<StorageT> & vector;  // Reference to field value for modifying

    const CspType & field_type;       // We require the type information of any non-primitive type, i.e. Struct or Enum, since they contain a meta
    static PyTypeObject PyType;
    static bool s_typeRegister;
};

template<typename StorageT> bool PyStructList<StorageT>::s_typeRegister = InitHelper::instance().registerCallback( 
    InitHelper::typeInitCallback( &PyStructList<StorageT>::PyType, "", &PyList_Type ) );

}

#endif