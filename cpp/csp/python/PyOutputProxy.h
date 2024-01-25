#ifndef _IN_CSP_PYTHON_PYOUTPUTPROXY_H
#define _IN_CSP_PYTHON_PYOUTPUTPROXY_H

#include <csp/engine/Node.h>
#include <csp/engine/InputId.h>
#include <Python.h>

namespace csp::python
{

class PyOutputProxy final: public PyObject
{
public:
    PyOutputProxy( PyObject *pyType, Node * node, OutputId );

    static PyOutputProxy * create( PyObject *pyType, Node * node, OutputId id );

    TimeSeriesProvider * ts() const { return m_node -> output( m_id ); }

    OutputId outputId() const { return m_id; }

    void outputTick( PyObject * value );
    DateTime now() const;

    //used by dynamic basket output
    void setElemId( int64_t elemId ) { m_id.elemId = elemId; }

    static PyTypeObject PyType;

private:
    PyPtr<PyObject> m_pyType;
    Node      * m_node;
    OutputId    m_id;
};

inline void PyOutputProxy::outputTick( PyObject * value )
{
    auto * ts = m_node -> output( m_id );

    try
    {
        //may want to make this an opt-in configuration
        //we only need this check for DIALECT types, other types will trip up in fromPython
        if( !validatePyType( ts -> type(), m_pyType.ptr(), value ) )
            CSP_THROW( TypeError, "" );

        switchCspType( ts -> type(),
                       [ & ]( auto tag )
                       {
                           ts -> outputTickTyped( m_node -> cycleCount(), m_node -> now(),
                                                  fromPython<typename decltype(tag)::type>( value, *ts -> type() ) );
                       } );
    }
    catch( TypeError & e )
    {
        CSP_THROW( TypeError, "\"" << m_node -> name() << "\" node expected output type on output #" << ( int ) m_id.id 
                   << " to be of type \"" << pyTypeToString( m_pyType.ptr() ) << "\" got type \"" << Py_TYPE( value ) -> tp_name << "\"" );
    }
}

}

#endif
