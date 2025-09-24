#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>

// Data processing nodes
REGISTER_CPPNODE( csp::cppnodes,  _tick_window_updates );
REGISTER_CPPNODE( csp::cppnodes,  _time_window_updates );
REGISTER_CPPNODE( csp::cppnodes,  _cross_sectional_as_list );
REGISTER_CPPNODE( csp::cppnodes,  _min_hit_by_tick );
REGISTER_CPPNODE( csp::cppnodes,  _in_sequence_check );
REGISTER_CPPNODE( csp::cppnodes,  _discard_non_overlapping );
REGISTER_CPPNODE( csp::cppnodes,  _sync_nan_f );

// Base statistics
REGISTER_CPPNODE( csp::cppnodes,  _count );
REGISTER_CPPNODE( csp::cppnodes,  _sum );
REGISTER_CPPNODE( csp::cppnodes,  _kahan_sum );
REGISTER_CPPNODE( csp::cppnodes,  _mean );
REGISTER_CPPNODE( csp::cppnodes,  _var );
REGISTER_CPPNODE( csp::cppnodes,  _first );
REGISTER_CPPNODE( csp::cppnodes,  _unique );
REGISTER_CPPNODE( csp::cppnodes,  _prod );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_mean );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_var );
REGISTER_CPPNODE( csp::cppnodes,  _covar );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_covar );
REGISTER_CPPNODE( csp::cppnodes,  _corr );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_corr );
REGISTER_CPPNODE( csp::cppnodes,  _sem );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_sem );
REGISTER_CPPNODE( csp::cppnodes,  _last );
REGISTER_CPPNODE( csp::cppnodes,  _quantile );
REGISTER_CPPNODE( csp::cppnodes,  _min_max );
REGISTER_CPPNODE( csp::cppnodes,  _rank );
REGISTER_CPPNODE( csp::cppnodes,  _arg_min_max );
REGISTER_CPPNODE( csp::cppnodes,  _skew );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_skew );
REGISTER_CPPNODE( csp::cppnodes,  _kurt );
REGISTER_CPPNODE( csp::cppnodes,  _weighted_kurt );

// EMA nodes
REGISTER_CPPNODE( csp::cppnodes,  _ema_compute );
REGISTER_CPPNODE( csp::cppnodes,  _ema_adjusted );
REGISTER_CPPNODE( csp::cppnodes,  _ema_halflife );
REGISTER_CPPNODE( csp::cppnodes,  _ema_halflife_adjusted );
REGISTER_CPPNODE( csp::cppnodes,  _ema_alpha_debias );
REGISTER_CPPNODE( csp::cppnodes,  _ema_halflife_debias );

static PyModuleDef _cspstatsimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_cspstatsimpl",
    "_cspstatsimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspstatsimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspstatsimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
