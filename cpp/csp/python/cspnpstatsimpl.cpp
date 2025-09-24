#include <numpy/ndarrayobject.h>
#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>

// Register nodes and create module

// Data processing
REGISTER_CPPNODE( csp::python,  _np_tick_window_updates );
REGISTER_CPPNODE( csp::python,  _np_time_window_updates );
REGISTER_CPPNODE( csp::python,  _cross_sectional_as_np );
REGISTER_CPPNODE( csp::python,  _np_cross_sectional_as_list );
REGISTER_CPPNODE( csp::python,  _np_cross_sectional_as_np );
REGISTER_CPPNODE( csp::python,  _list_to_np );
REGISTER_CPPNODE( csp::python,  _np_to_list );
REGISTER_CPPNODE( csp::python,  _sync_nan_np );

// Computation nodes
REGISTER_CPPNODE( csp::python,  _np_count );
REGISTER_CPPNODE( csp::python,  _np_first );
REGISTER_CPPNODE( csp::python,  _np_last );
REGISTER_CPPNODE( csp::python,  _np_sum );
REGISTER_CPPNODE( csp::python,  _np_kahan_sum );
REGISTER_CPPNODE( csp::python,  _np_mean );
REGISTER_CPPNODE( csp::python,  _np_prod );
REGISTER_CPPNODE( csp::python,  _np_unique );
REGISTER_CPPNODE( csp::python,  _np_quantile );
REGISTER_CPPNODE( csp::python,  _np_min_max );
REGISTER_CPPNODE( csp::python,  _np_rank );
REGISTER_CPPNODE( csp::python,  _np_arg_min_max );
REGISTER_CPPNODE( csp::python,  _np_weighted_mean );
REGISTER_CPPNODE( csp::python,  _np_var );
REGISTER_CPPNODE( csp::python,  _np_weighted_var );
REGISTER_CPPNODE( csp::python,  _np_sem );
REGISTER_CPPNODE( csp::python,  _np_weighted_sem );
REGISTER_CPPNODE( csp::python,  _np_covar );
REGISTER_CPPNODE( csp::python,  _np_weighted_covar );
REGISTER_CPPNODE( csp::python,  _np_corr );
REGISTER_CPPNODE( csp::python,  _np_weighted_corr );
REGISTER_CPPNODE( csp::python,  _np_cov_matrix );
REGISTER_CPPNODE( csp::python,  _np_weighted_cov_matrix );
REGISTER_CPPNODE( csp::python,  _np_corr_matrix );
REGISTER_CPPNODE( csp::python,  _np_weighted_corr_matrix );
REGISTER_CPPNODE( csp::python,  _np_skew );
REGISTER_CPPNODE( csp::python,  _np_weighted_skew );
REGISTER_CPPNODE( csp::python,  _np_kurt );
REGISTER_CPPNODE( csp::python,  _np_weighted_kurt );

// EMA nodes
REGISTER_CPPNODE( csp::python,  _np_ema_compute );
REGISTER_CPPNODE( csp::python,  _np_ema_adjusted );
REGISTER_CPPNODE( csp::python,  _np_ema_halflife );
REGISTER_CPPNODE( csp::python,  _np_ema_halflife_adjusted );
REGISTER_CPPNODE( csp::python,  _np_ema_alpha_debias );
REGISTER_CPPNODE( csp::python,  _np_ema_halflife_debias );


static PyModuleDef _cspnpstatsimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_cspnpstatsimpl",
    "_cspnpstatsimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspnpstatsimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspnpstatsimpl_module);

    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
