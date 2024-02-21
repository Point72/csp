#include <csp/engine/Dictionary.h>
#include <csp/engine/CppNode.h>


namespace csp::cppnodes
{

/*
Math operations
*/

// Unary operation

template<typename ArgT, typename OutT, OutT (*Func)(ArgT)>
DECLARE_CPPNODE( _unary_op )
{
    TS_INPUT( ArgT, x );
    TS_OUTPUT( OutT );

    //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_unary_op"; }

public:
    _STATIC_CREATE_METHOD( SINGLE_ARG( _unary_op<ArgT, OutT, Func> ) );
    _unary_op( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    INVOKE()
    {
        RETURN( Func( x ) );
    }
};

template<typename T> inline T _abs( T x ){ return std::abs( x ); }
template<typename T> inline double _ln( T x ){ return std::log( x ); }
template<typename T> inline double _log2( T x ){ return std::log2( x ); }
template<typename T> inline double _log10( T x ){ return std::log10( x ); }
template<typename T> inline double _exp( T x ){ return std::exp( x ); }
template<typename T> inline double _exp2( T x ){ return std::exp2( x ); }
template<typename T> inline double _sqrt( T x ){ return std::sqrt( x ); }
template<typename T> inline double _erf( T x ){ return std::erf( x ); }
template<typename T> inline double _sin( T x ){ return std::sin( x ); }
template<typename T> inline double _cos( T x ){ return std::cos( x ); }
template<typename T> inline double _tan( T x ){ return std::tan( x ); }
template<typename T> inline double _asin( T x ){ return std::asin( x ); }
template<typename T> inline double _acos( T x ){ return std::acos( x ); }
template<typename T> inline double _atan( T x ){ return std::atan( x ); }
template<typename T> inline double _sinh( T x ){ return std::sinh( x ); }
template<typename T> inline double _cosh( T x ){ return std::cosh( x ); }
template<typename T> inline double _tanh( T x ){ return std::tanh( x ); }
template<typename T> inline double _asinh( T x ){ return std::asinh( x ); }
template<typename T> inline double _acosh( T x ){ return std::acosh( x ); }
template<typename T> inline double _atanh( T x ){ return std::atanh( x ); }

inline bool _not_( bool x ){ return !x; }
inline int64_t _bitwise_not(int64_t x) { return ~x; }

#define EXPORT_UNARY_OP( Name, ArgType, OutType, Func ) EXPORT_TEMPLATE_CPPNODE( Name, SINGLE_ARG( _unary_op<ArgType, OutType, _##Func> ) )
    EXPORT_UNARY_OP( abs_f, double, double, abs );
    EXPORT_UNARY_OP( abs_i, int64_t, int64_t, abs );
    EXPORT_UNARY_OP( ln_f, double, double, ln );
    EXPORT_UNARY_OP( ln_i, int64_t, double, ln );
    EXPORT_UNARY_OP( log2_f, double, double, log2 );
    EXPORT_UNARY_OP( log2_i, int64_t, double, log2 );
    EXPORT_UNARY_OP( log10_f, double, double, log10 );
    EXPORT_UNARY_OP( log10_i, int64_t, double, log10 );
    EXPORT_UNARY_OP( exp_f, double, double, exp );
    EXPORT_UNARY_OP( exp_i, int64_t, double, exp );
    EXPORT_UNARY_OP( exp2_f, double, double, exp2 );
    EXPORT_UNARY_OP( exp2_i, int64_t, double, exp2 );
    EXPORT_UNARY_OP( sqrt_f, double, double, sqrt );
    EXPORT_UNARY_OP( sqrt_i, int64_t, double, sqrt );
    EXPORT_UNARY_OP( erf_f, double, double, erf );
    EXPORT_UNARY_OP( erf_i, int64_t, double, erf );
    EXPORT_UNARY_OP( sin_f, double, double, sin );
    EXPORT_UNARY_OP( sin_i, int64_t, double, sin );
    EXPORT_UNARY_OP( cos_f, double, double, cos );
    EXPORT_UNARY_OP( cos_i, int64_t, double, cos );
    EXPORT_UNARY_OP( tan_f, double, double, tan );
    EXPORT_UNARY_OP( tan_i, int64_t, double, tan );
    EXPORT_UNARY_OP( asin_f, double, double, asin );
    EXPORT_UNARY_OP( asin_i, int64_t, double, asin );
    EXPORT_UNARY_OP( acos_f, double, double, acos );
    EXPORT_UNARY_OP( acos_i, int64_t, double, acos );
    EXPORT_UNARY_OP( atan_f, double, double, atan );
    EXPORT_UNARY_OP( atan_i, int64_t, double, atan );
    EXPORT_UNARY_OP( sinh_f, double, double, sinh );
    EXPORT_UNARY_OP( sinh_i, int64_t, double, sinh );
    EXPORT_UNARY_OP( cosh_f, double, double, cosh );
    EXPORT_UNARY_OP( cosh_i, int64_t, double, cosh );
    EXPORT_UNARY_OP( tanh_f, double, double, tanh );
    EXPORT_UNARY_OP( tanh_i, int64_t, double, tanh );
    EXPORT_UNARY_OP( asinh_f, double, double, asinh );
    EXPORT_UNARY_OP( asinh_i, int64_t, double, asinh );
    EXPORT_UNARY_OP( acosh_f, double, double, acosh );
    EXPORT_UNARY_OP( acosh_i, int64_t, double, acosh );
    EXPORT_UNARY_OP( atanh_f, double, double, atanh );
    EXPORT_UNARY_OP( atanh_i, int64_t, double, atanh );
    EXPORT_UNARY_OP( not_, bool, bool, not_ );
    EXPORT_UNARY_OP( bitwise_not, int64_t, int64_t, bitwise_not );
#undef EXPORT_UNARY_OP

// Binary operation
template<typename ArgT, typename OutT, OutT (*Func)(ArgT, ArgT)>
DECLARE_CPPNODE( _binary_op )
{
    TS_INPUT( ArgT, x );
    TS_INPUT( ArgT, y );
    TS_OUTPUT( OutT );

    //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_binary_op"; }

public:
    _STATIC_CREATE_METHOD( SINGLE_ARG( _binary_op<ArgT, OutT, Func> ) );
    _binary_op( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    INVOKE()
    {
        if( csp.valid( x, y ) )
            RETURN( Func( x, y ) );
    }
};

// Math ops
template<typename T> inline T _add( T x, T y ){ return x + y; }
template<typename T> inline T _sub( T x, T y ){ return x - y; }
template<typename T> inline T _mul( T x, T y ){ return x * y; }
template<typename T> inline T _max( T x, T y ){ return std::max( x, y ); }
template<typename T> inline T _min( T x, T y ){ return std::min( x, y ); }
template<typename T> inline double _div( T x, T y ){ return x / ( double )y; }
inline int64_t _pow_i( int64_t x, int64_t y ){ return ( int64_t )pow( ( double )x, ( double )y ); }
inline double _pow_f( double x, double y ){ return pow( x, y ); }

// Comparison ops
template<typename T> inline bool _eq( T x, T y ){ return x == y; }
template<typename T> inline bool _ne( T x, T y ){ return x != y; }
template<typename T> inline bool _gt( T x, T y ){ return x > y; }
template<typename T> inline bool _ge( T x, T y ){ return x >= y; }
template<typename T> inline bool _lt( T x, T y ){ return x < y; }
template<typename T> inline bool _le( T x, T y ){ return x <= y; }

#define EXPORT_BINARY_OP( Name, ArgType, OutType, Func ) EXPORT_TEMPLATE_CPPNODE( Name, SINGLE_ARG( _binary_op<ArgType, OutType, Func> ) )
    EXPORT_BINARY_OP( add_i, int64_t, int64_t, _add<int64_t> );
    EXPORT_BINARY_OP( sub_i, int64_t, int64_t, _sub<int64_t> );
    EXPORT_BINARY_OP( mul_i, int64_t, int64_t, _mul<int64_t> );
    EXPORT_BINARY_OP( div_i, int64_t, double, _div<int64_t> );
    EXPORT_BINARY_OP( pow_i, int64_t, int64_t, _pow_i );
    EXPORT_BINARY_OP( max_i, int64_t, int64_t, _max<int64_t> );
    EXPORT_BINARY_OP( min_i, int64_t, int64_t, _min<int64_t> );
    EXPORT_BINARY_OP( add_f, double, double, _add<double> );
    EXPORT_BINARY_OP( sub_f, double, double, _sub<double> );
    EXPORT_BINARY_OP( mul_f, double, double, _mul<double> );
    EXPORT_BINARY_OP( div_f, double, double, _div<double> );
    EXPORT_BINARY_OP( pow_f, double, double, _pow_f );
    EXPORT_BINARY_OP( max_f, double, double, _max<double> );
    EXPORT_BINARY_OP( min_f, double, double, _min<double> );
    EXPORT_BINARY_OP( eq_i, int64_t, bool, _eq<int64_t> );
    EXPORT_BINARY_OP( ne_i, int64_t, bool, _ne<int64_t> );
    EXPORT_BINARY_OP( gt_i, int64_t, bool, _gt<int64_t> );
    EXPORT_BINARY_OP( ge_i, int64_t, bool, _ge<int64_t> );
    EXPORT_BINARY_OP( lt_i, int64_t, bool, _lt<int64_t> );
    EXPORT_BINARY_OP( le_i, int64_t, bool, _le<int64_t> );
    EXPORT_BINARY_OP( eq_f, double, bool, _eq<double> );
    EXPORT_BINARY_OP( ne_f, double, bool, _ne<double> );
    EXPORT_BINARY_OP( gt_f, double, bool, _gt<double> );
    EXPORT_BINARY_OP( ge_f, double, bool, _ge<double> );
    EXPORT_BINARY_OP( lt_f, double, bool, _lt<double> );
    EXPORT_BINARY_OP( le_f, double, bool, _le<double> );
#undef EXPORT_BINARY_OP

}
