#include <csp/engine/CspType.h>
#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>

namespace csp::test_ext
{

DECLARE_CPPNODE( accumulate )
{
    INIT_CPPNODE( accumulate ) {}

    // try to hit all node features

    TS_INPUT( int64_t, x );
    SCALAR_INPUT( int64_t, start_value );

    SCALAR_INPUT( TimeDelta, reset_interval );
    ALARM( bool, reset );

    STATE_VAR( int64_t, m_sum );
    TS_OUTPUT( int64_t, x );

    START()
    {
        m_sum = start_value;
        csp.schedule_alarm( reset, reset_interval, true );
    }

    INVOKE()
    {
        if( csp.ticked( reset  ) )
        {
            m_sum = 0;
            csp.schedule_alarm( reset, reset_interval, true );
        }
        if( csp.ticked( x ) )
            m_sum += x;
        
        RETURN( m_sum );
    }
};

EXPORT_CPPNODE( accumulate );

}

REGISTER_CPPNODE( csp::test_ext, accumulate );

static PyModuleDef _test_ext_module = {
    PyModuleDef_HEAD_INIT,
    "_test_ext",
    "_test_ext c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__test_ext( void )
{
    PyObject * m;

    m = PyModule_Create( &_test_ext_module );
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
