#ifndef _IN_CSP_PYTHON_INITHELPER_H
#define _IN_CSP_PYTHON_INITHELPER_H

#include <csp/core/Platform.h>
#include <Python.h>
#include <functional>
#include <string>
#include <vector>

namespace csp::python
{

class DLL_LOCAL InitHelper
{
public:
    ~InitHelper() {}

    //callbacks receive the module being initialized and should return success/failure
    //on failure pyerr should be set
    using InitCallback = std::function<bool( PyObject * )>;

    bool registerCallback( InitCallback cb );

    static InitCallback typeInitCallback( PyTypeObject * pyType, std::string name, PyTypeObject * baseType = nullptr );
    static InitCallback moduleMethodsCallback( PyMethodDef * methods );
    static InitCallback moduleMethod( const char * name, PyCFunction func, int flags, const char * doc );

    static InitHelper & instance();

    bool execute( PyObject * module );

private:
    
    InitHelper() {}

    using Callbacks = std::vector<InitCallback>;
    Callbacks m_callbacks;
};

#define __REGISTER_INIT_HOOK2( CALLBACK, LINE ) static bool inithook_##LINE = csp::python::InitHelper::instance().registerCallback( CALLBACK );
#define __REGISTER_INIT_HOOK( CALLBACK, LINE )  __REGISTER_INIT_HOOK2( CALLBACK, LINE );
#define REGISTER_INIT_HOOK( CALLBACK )          __REGISTER_INIT_HOOK( CALLBACK, __LINE__ );

#define REGISTER_TYPE_INIT( PY_TYPE, NAME )     REGISTER_INIT_HOOK( csp::python::InitHelper::typeInitCallback( PY_TYPE, NAME ) );
#define REGISTER_MODULE_METHODS( METHODS )      REGISTER_INIT_HOOK( csp::python::InitHelper::moduleMethodsCallback( METHODS ) );
#define REGISTER_MODULE_METHOD( NAME, METHOD, FLAGS, DOC ) REGISTER_INIT_HOOK( csp::python::InitHelper::moduleMethod( NAME, METHOD, FLAGS, DOC ) );

inline InitHelper & InitHelper::instance()
{
    static InitHelper s_instance;
    return s_instance;
}

inline InitHelper::InitCallback InitHelper::typeInitCallback( PyTypeObject * pyType, std::string name, PyTypeObject * baseType )
{
    InitCallback cb = [pyType,name,baseType]( PyObject * module ) {
        if( baseType )
            pyType -> tp_base = baseType;
        if( PyType_Ready( pyType ) < 0 )
            return false;

        Py_INCREF( pyType );
        PyModule_AddObject( module, name.c_str(), ( PyObject  * ) pyType );
        return true;
    };

    return cb;
}

inline InitHelper::InitCallback InitHelper::moduleMethodsCallback( PyMethodDef * methods )
{
    InitCallback cb = [methods]( PyObject * module ) {
        if( PyModule_AddFunctions( module, methods ) < 0 )
            return false;
        return true;
    };

    return cb;
}

inline InitHelper::InitCallback InitHelper::moduleMethod( const char * name, PyCFunction func, int flags, const char * doc )
{
    PyMethodDef defs[2]{ { name, func, flags, doc }, { nullptr } };

    //Note that we rely on the lambda closure to keep the lifetime of defs which is kept by ptr
    //m_callbacks will keep the InitCallback around for the life of the program
    InitCallback cb = [defs]( PyObject * module ) {
        if( PyModule_AddFunctions( module, ( PyMethodDef * ) defs ) < 0 )
            return false;
        return true;
    };

    return cb;
}

inline bool InitHelper::registerCallback( InitCallback cb )
{
    m_callbacks.emplace_back( std::move( cb ) );
    return true;
}

inline bool InitHelper::execute( PyObject * module )
{
    for( auto & cb : m_callbacks )
    {
        if( !cb( module ) )
            return false;
    }

    return true;
}

}
#endif
