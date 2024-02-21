#include <csp/python/PyCppNode.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Conversions.h>

// Math ops
REGISTER_CPPNODE( csp::cppnodes, add_f );
REGISTER_CPPNODE( csp::cppnodes, add_i );
REGISTER_CPPNODE( csp::cppnodes, sub_f );
REGISTER_CPPNODE( csp::cppnodes, sub_i );
REGISTER_CPPNODE( csp::cppnodes, mul_f );
REGISTER_CPPNODE( csp::cppnodes, mul_i );
REGISTER_CPPNODE( csp::cppnodes, div_f );
REGISTER_CPPNODE( csp::cppnodes, div_i );
REGISTER_CPPNODE( csp::cppnodes, pow_f );
REGISTER_CPPNODE( csp::cppnodes, pow_i );
REGISTER_CPPNODE( csp::cppnodes, max_f );
REGISTER_CPPNODE( csp::cppnodes, max_i );
REGISTER_CPPNODE( csp::cppnodes, min_f );
REGISTER_CPPNODE( csp::cppnodes, min_i );
REGISTER_CPPNODE( csp::cppnodes, abs_f );
REGISTER_CPPNODE( csp::cppnodes, abs_i );
REGISTER_CPPNODE( csp::cppnodes, ln_f );
REGISTER_CPPNODE( csp::cppnodes, ln_i );
REGISTER_CPPNODE( csp::cppnodes, log2_f );
REGISTER_CPPNODE( csp::cppnodes, log2_i );
REGISTER_CPPNODE( csp::cppnodes, log10_f );
REGISTER_CPPNODE( csp::cppnodes, log10_i );
REGISTER_CPPNODE( csp::cppnodes, exp_f );
REGISTER_CPPNODE( csp::cppnodes, exp_i );
REGISTER_CPPNODE( csp::cppnodes, exp2_f );
REGISTER_CPPNODE( csp::cppnodes, exp2_i );
REGISTER_CPPNODE( csp::cppnodes, sqrt_f );
REGISTER_CPPNODE( csp::cppnodes, sqrt_i );
REGISTER_CPPNODE( csp::cppnodes, erf_f );
REGISTER_CPPNODE( csp::cppnodes, erf_i );
REGISTER_CPPNODE( csp::cppnodes, sin_f );
REGISTER_CPPNODE( csp::cppnodes, sin_i );
REGISTER_CPPNODE( csp::cppnodes, cos_f );
REGISTER_CPPNODE( csp::cppnodes, cos_i );
REGISTER_CPPNODE( csp::cppnodes, tan_f );
REGISTER_CPPNODE( csp::cppnodes, tan_i );
REGISTER_CPPNODE( csp::cppnodes, asin_f );
REGISTER_CPPNODE( csp::cppnodes, asin_i );
REGISTER_CPPNODE( csp::cppnodes, acos_f );
REGISTER_CPPNODE( csp::cppnodes, acos_i );
REGISTER_CPPNODE( csp::cppnodes, atan_f );
REGISTER_CPPNODE( csp::cppnodes, atan_i );
REGISTER_CPPNODE( csp::cppnodes, sinh_f );
REGISTER_CPPNODE( csp::cppnodes, sinh_i );
REGISTER_CPPNODE( csp::cppnodes, cosh_f );
REGISTER_CPPNODE( csp::cppnodes, cosh_i );
REGISTER_CPPNODE( csp::cppnodes, tanh_f );
REGISTER_CPPNODE( csp::cppnodes, tanh_i );
REGISTER_CPPNODE( csp::cppnodes, asinh_f );
REGISTER_CPPNODE( csp::cppnodes, asinh_i );
REGISTER_CPPNODE( csp::cppnodes, acosh_f );
REGISTER_CPPNODE( csp::cppnodes, acosh_i );
REGISTER_CPPNODE( csp::cppnodes, atanh_f );
REGISTER_CPPNODE( csp::cppnodes, atanh_i );

REGISTER_CPPNODE( csp::cppnodes, bitwise_not );

// Comparisons
REGISTER_CPPNODE( csp::cppnodes, not_ );
REGISTER_CPPNODE( csp::cppnodes, eq_f );
REGISTER_CPPNODE( csp::cppnodes, eq_i );
REGISTER_CPPNODE( csp::cppnodes, ne_f );
REGISTER_CPPNODE( csp::cppnodes, ne_i );
REGISTER_CPPNODE( csp::cppnodes, gt_f );
REGISTER_CPPNODE( csp::cppnodes, gt_i );
REGISTER_CPPNODE( csp::cppnodes, lt_f );
REGISTER_CPPNODE( csp::cppnodes, lt_i );
REGISTER_CPPNODE( csp::cppnodes, ge_f );
REGISTER_CPPNODE( csp::cppnodes, ge_i );
REGISTER_CPPNODE( csp::cppnodes, le_f );
REGISTER_CPPNODE( csp::cppnodes, le_i );

static PyModuleDef _cspmathimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_cspmathimpl",
    "_cspmathimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspmathimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspmathimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
