#include <csp/engine/CspType.h>
#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>
#include <algorithm>
#include <string>

namespace csp::piglatin
{

DECLARE_CPPNODE( piglatin )
{
    INIT_CPPNODE( piglatin )
    {}

    TS_INPUT( std::string, x );
    SCALAR_INPUT( bool, capitalize );
    // ALARM(Generic, alarm);
    TS_OUTPUT( std::string );

    START()
    {}

    INVOKE()
    {
        if( csp.ticked( x ) && csp.valid( x ) )
        {
            std::string str = x.lastValue();
            if( capitalize )
            {
                std::transform( str.begin(), str.end(), str.begin(), ::toupper );
            }
            RETURN( str.substr( 1, std::string::npos ) + str[0] + ( capitalize ? "AY" : "ay" ) );
        }
    }
};

EXPORT_CPPNODE( piglatin );
}

REGISTER_CPPNODE( csp::piglatin, piglatin );

static PyModuleDef _piglatin_module = {
    PyModuleDef_HEAD_INIT,
    "_piglatin",
    "_piglatin c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__piglatin( void )
{
    PyObject * m;

    m = PyModule_Create( &_piglatin_module );
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
