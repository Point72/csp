#ifndef _IN_CSP_PYTHON_PYENGINE_H
#define _IN_CSP_PYTHON_PYENGINE_H

#include <csp/core/Time.h>
#include <csp/python/Exception.h>
#include <csp/engine/RootEngine.h>
#include <csp/python/PyObjectPtr.h>
#include <Python.h>
#include <memory>

namespace csp { class GraphOutputAdapter; }

namespace csp::python
{

class PyEngine;

//This is the actual python root engine
class PythonEngine final : public csp::RootEngine
{
public:
    PythonEngine( PyEngine * parent, const Dictionary & );
  
    void dialectUnlockGIL() noexcept override;
    void dialectLockGIL() noexcept override;

    PyEngine * parent() { return m_parent; }
    bool outputNumpy() { return m_outputNumpy; }
private:
    PyEngine      * m_parent;
    PyThreadState * m_pyThreadState;
    bool m_outputNumpy;
};

//This is the root engine wrapper object
class CSPIMPL_EXPORT PyEngine final: public PyObject
{
public:
    PyEngine( const Dictionary & settings );
    //for dynamic engines
    PyEngine( Engine * engine );
    ~PyEngine();

    Engine * engine()           { return m_engine; }
    PythonEngine * rootEngine() { return static_cast<PythonEngine *>( engine() -> rootEngine() ); }

    PyObject * collectOutputs();

    static PyEngine * create( Engine * engine );

    static PyTypeObject PyType;

private:
    bool        m_ownEngine;
    Engine    * m_engine;
};

inline std::exception_ptr PyEngine_shutdown_make_exception( PyObject * pyException )
{
    if( !PyExceptionInstance_Check( pyException ) )
    {
        PyObjectPtr pyExceptionStr = PyObjectPtr::own( PyObject_Str( pyException ) );
        if( !pyExceptionStr.ptr() )
            CSP_THROW( PythonPassthrough, "" );
        std::string pyExceptionString = PyUnicode_AsUTF8( pyExceptionStr.ptr() );
        std::string desc = "Expected Exception object as argument for shutdown_engine: got " + pyExceptionString + " of type " + Py_TYPE( pyException ) -> tp_name;
        return std::make_exception_ptr( csp::Exception( "TypeError", desc ) );
    }
    else
        return std::make_exception_ptr( PythonPassthrough( pyException ) );
}

};

#endif
