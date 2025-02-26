#ifndef _IN_CSP_PYTHON_EXCEPTION_H
#define _IN_CSP_PYTHON_EXCEPTION_H

#include <csp/core/Exception.h>
#include <Python.h>
#include <string>

namespace csp::python
{

class PythonPassthrough : public csp::Exception
{
public:
    PythonPassthrough( const char * exType, const std::string &r, const char * file,
                       const char * func, int line ) :
        csp::Exception( exType, r, file, func, line )
    {
        //Fetch the current error to clear out the error indicator while the stack gets unwound
        //We own the references to all the members assigned in PyErr_Fetch
        //We need to hold the reference since PyErr_Restore takes back a reference to each of its arguments
        PyErr_Fetch( &m_type, &m_value, &m_traceback );
    }

    PythonPassthrough( PyObject * pyException ) :
        csp::Exception( "", "" )
    {
        // Note: all of these methods return strong references, so we own them like in the other constructor
        m_type = PyObject_Type( pyException );
        m_value = PyObject_Str( pyException );
        m_traceback = PyException_GetTraceback( pyException );
    }

    void restore()
    {
        if( !description().empty() )
        {
            std::string p = description() + ": ";
            PyObject * prefix = PyUnicode_FromString( p.c_str() );
            PyObject * newmsg = PyUnicode_Concat( prefix, m_value );
            Py_DECREF( m_value );
            Py_DECREF( prefix );
            m_value = newmsg;
        }

        PyErr_Restore( m_type, m_value, m_traceback );
        m_type = m_value = m_traceback = nullptr;
    }

private:
    PyObject * m_type;
    PyObject * m_value;
    PyObject * m_traceback;
};

CSP_DECLARE_EXCEPTION( AttributeError, ::csp::Exception );

inline bool& capture_cpp_exception_trace_flag()
{
    static bool val = false; return val;
}

#define CSP_CATCH_HELPER( EXC_TYPE, PYEXC_TYPE, RETURN_STMT ) catch( const EXC_TYPE & err ) { PyErr_SetString( PYEXC_TYPE, err.full(csp::python::capture_cpp_exception_trace_flag()).c_str() ); RETURN_STMT }
#define CSP_CATCH_HELPER_STD( EXC_TYPE, PYEXC_TYPE, RETURN_STMT ) catch( const EXC_TYPE & err ) { PyErr_SetString( PYEXC_TYPE, err.what() ); RETURN_STMT }

#define CSP_CATCH_HELPERS( RETURN_STMT )  \
    CSP_CATCH_HELPER( ::csp::python::AttributeError,   PyExc_AttributeError, RETURN_STMT ) \
    CSP_CATCH_HELPER( ::csp::InvalidArgument,   PyExc_TypeError,         RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::NotImplemented,    PyExc_NotImplementedError, RETURN_STMT )   \
    CSP_CATCH_HELPER( ::csp::KeyError,          PyExc_KeyError,          RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::ValueError,        PyExc_ValueError,        RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::TypeError,         PyExc_TypeError,         RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::RangeError,        PyExc_IndexError,        RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::OverflowError,     PyExc_OverflowError,     RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::DivideByZero,      PyExc_ZeroDivisionError, RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::RecursionError,    PyExc_RecursionError,    RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::OSError,           PyExc_OSError,           RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::OutOfMemoryError,  PyExc_MemoryError,       RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::FileNotFoundError, PyExc_FileNotFoundError, RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::RuntimeException,  PyExc_RuntimeError,      RETURN_STMT )     \
    CSP_CATCH_HELPER( ::csp::Exception,         PyExc_Exception,         RETURN_STMT )     \
    CSP_CATCH_HELPER_STD( std::exception,       PyExc_Exception,         RETURN_STMT )

#define CSP_BEGIN_METHOD try {
#define CSP_RETURN } catch( ::csp::python::PythonPassthrough & err ) { err.restore(); return; } CSP_CATCH_HELPERS( return; )

#define CSP_RETURN_INT  } catch( ::csp::python::PythonPassthrough & err ) { err.restore(); return -1;      } CSP_CATCH_HELPERS( return -1; ); return 0;
#define CSP_RETURN_NONE } catch( ::csp::python::PythonPassthrough & err ) { err.restore(); return nullptr; } CSP_CATCH_HELPERS( return nullptr; ); Py_RETURN_NONE;
#define CSP_RETURN_NULL } catch( ::csp::python::PythonPassthrough & err ) { err.restore(); return nullptr; } CSP_CATCH_HELPERS( return nullptr; ); return nullptr;

}

#endif
