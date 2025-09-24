#include <csp/cppnodes/statsimpl.h>

namespace csp::cppnodes
{

class _tick_window_updates : public _generic_tick_window_updates<double, _tick_window_updates>
{
public:
    using _generic_tick_window_updates<double, _tick_window_updates>::_generic_tick_window_updates;
    _STATIC_CREATE_METHOD( _tick_window_updates );

    inline double createNan()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    inline void validateShape() { }
    inline void checkValid() { }
};

EXPORT_CPPNODE( _tick_window_updates );

class _time_window_updates : public _generic_time_window_updates<double, _time_window_updates>
{
public:
    using _generic_time_window_updates<double, _time_window_updates>::_generic_time_window_updates;
    _STATIC_CREATE_METHOD( _time_window_updates );

    inline double createNan()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    inline void validateShape() { }
    inline void checkValid() { }
};

EXPORT_CPPNODE( _time_window_updates );

class _cross_sectional_as_list : public _generic_cross_sectional<double, std::vector<double>, _cross_sectional_as_list>
{
public:
    using _generic_cross_sectional<double, std::vector<double>, _cross_sectional_as_list>::_generic_cross_sectional;
    _STATIC_CREATE_METHOD( _cross_sectional_as_list );

    inline void computeCrossSectional()
    {
        s_window.copy_values( &unnamed_output().reserveSpace() );
    }
};

EXPORT_CPPNODE( _cross_sectional_as_list );

/*
@csp.node
def _min_hit_by_tick(x: ts['T'], min_window: int, trigger: ts[object]):
    __outputs__(ts[bool])
*/

DECLARE_CPPNODE( _min_hit_by_tick )
{
    TS_INPUT( Generic, x );
    SCALAR_INPUT( int64_t, min_window );
    TS_INPUT( Generic, trigger );

    TS_OUTPUT( bool );

    INIT_CPPNODE( _min_hit_by_tick ) { }

    START()
    {
        csp.make_passive( x );
    }

    INVOKE()
    {
        if( csp.ticked( trigger ) )
        {
            if( csp.count( x ) >= min_window )
            {
                csp.make_passive( trigger );
                RETURN( true );
            }
        }
    }
};

EXPORT_CPPNODE( _min_hit_by_tick );

/*
@csp.node
def _in_sequence_check(x: ts['T'], y: ts['T']):
*/

DECLARE_CPPNODE( _in_sequence_check )
{
    TS_INPUT( Generic, x );
    TS_INPUT( Generic, y );

    INIT_CPPNODE( _in_sequence_check ) { }

    INVOKE()
    {
        if( csp.ticked( x ) != csp.ticked( y ) )
            CSP_THROW( ValueError, "For multivariate statistics, x and y must tick in sequence." );
    };
};

EXPORT_CPPNODE( _in_sequence_check );

/*
@csp.node
def _discard_non_overlapping(x: ts[float], y: ts[float]):
*/

DECLARE_CPPNODE( _discard_non_overlapping )
{
    TS_INPUT( double, x );
    TS_INPUT( double, y );
    
    TS_NAMED_OUTPUT( double, x_sync );
    TS_NAMED_OUTPUT( double, y_sync );

    INIT_CPPNODE( _discard_non_overlapping ) { }

    INVOKE()
    {
        if( csp.ticked( x ) && csp.ticked( y ) )
        {
            x_sync.output( x );
            y_sync.output( y );
        }
    };
};

EXPORT_CPPNODE( _discard_non_overlapping );

/*
@csp.node(cppimpl=_cspstatsimpl._sync_nan_f)
def _sync_nan_f(x: ts[float], y: ts[float]) -> csp.Outputs(x_sync=ts[float], y_sync=ts[float]):
*/

DECLARE_CPPNODE( _sync_nan_f )
{
    TS_INPUT( double, x );
    TS_INPUT( double, y );

    TS_NAMED_OUTPUT( double, x_sync );
    TS_NAMED_OUTPUT( double, y_sync );

    INIT_CPPNODE( _sync_nan_f ) { }

    INVOKE()
    {
        // Note that it's guaranteed x and y are ticking in sequence when this node is triggered
        if( likely( !isnan( x ) ) )
            y_sync.output( y );
        else
            y_sync.output( std::numeric_limits<double>::quiet_NaN() );

        if( likely( !isnan( y ) ) )
            x_sync.output( x );
        else
            x_sync.output( std::numeric_limits<double>::quiet_NaN() );
    };
};

EXPORT_CPPNODE( _sync_nan_f );

/*
Computation node for single input statistics
*/

// Univariate
template<typename C>
DECLARE_CPPNODE( _compute )
{
protected:
    TS_INPUT( std::vector<double>, additions );
    TS_INPUT( std::vector<double>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( DataValidator<C>, s_computation );
    TS_OUTPUT( double );

    //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_compute"; }

public:
    _compute( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    START()
    {
        initDataValidator( s_computation );
    }

    virtual void initDataValidator( DataValidator<C> & ) = 0;

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            s_computation.reset();
        }
        if( csp.ticked( additions ) )
        {
            for( double x: additions.lastValue() )
                s_computation.add( x );
        }
        if( csp.ticked( removals ) )
        {
            for( double x: removals.lastValue() )
                s_computation.remove( x );
        }
        if( csp.ticked( trigger ) )
        {
            RETURN( s_computation.compute() );
        }
    }
};

template<typename C>
class _computeCommonArgs : public _compute<C>
{
public:
    using _compute<C>::_compute;
    _STATIC_CREATE_METHOD( _computeCommonArgs<C> );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na );
    }
};

template<typename ArgT, typename C>
class _computeOneArg : public _compute<C>
{
public:
    using _compute<C>::_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _computeOneArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg );
    }
};

template<typename ArgT, typename C>
class _computeTwoArg : public _compute<C>
{
public:
    using _compute<C>::_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _computeTwoArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg1 );
    SCALAR_INPUT( ArgT, arg2 );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg1, this -> arg2 );
    }
};

template<typename C>
class _computeEMA : public _compute<C>
{
public:
    using _compute<C>::_compute;
    _STATIC_CREATE_METHOD( _computeEMA<C> );
    SCALAR_INPUT( double, alpha );
    SCALAR_INPUT( int64_t, horizon );
    SCALAR_INPUT( bool, adjust );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, true, alpha, this -> ignore_na, horizon, adjust );
    }
};

// Export node templates
EXPORT_TEMPLATE_CPPNODE( _count,            _computeCommonArgs<Count> );
EXPORT_TEMPLATE_CPPNODE( _sum,              _computeCommonArgs<Sum> );
EXPORT_TEMPLATE_CPPNODE( _kahan_sum,        _computeCommonArgs<KahanSum> );
EXPORT_TEMPLATE_CPPNODE( _mean,             _computeCommonArgs<Mean> );
EXPORT_TEMPLATE_CPPNODE( _first,            _computeCommonArgs<First> );
EXPORT_TEMPLATE_CPPNODE( _prod,             _computeCommonArgs<Product> );
EXPORT_TEMPLATE_CPPNODE( _last,             _computeCommonArgs<Last> );
EXPORT_TEMPLATE_CPPNODE( _unique,           SINGLE_ARG( _computeOneArg<int64_t, Unique> ) );
EXPORT_TEMPLATE_CPPNODE( _min_max,          SINGLE_ARG( _computeOneArg<bool, AscendingMinima> ) );
EXPORT_TEMPLATE_CPPNODE( _var,              SINGLE_ARG( _computeOneArg<int64_t, Variance> ) );
EXPORT_TEMPLATE_CPPNODE( _sem,              SINGLE_ARG( _computeOneArg<int64_t, StandardError> ) );
EXPORT_TEMPLATE_CPPNODE( _skew,             SINGLE_ARG( _computeOneArg<bool, Skew> ) );
EXPORT_TEMPLATE_CPPNODE( _rank,             SINGLE_ARG( _computeTwoArg<int64_t, Rank> ) );
EXPORT_TEMPLATE_CPPNODE( _kurt,             SINGLE_ARG( _computeTwoArg<bool, Kurtosis> ) );
EXPORT_TEMPLATE_CPPNODE( _ema_compute,      _computeEMA<EMA> );
EXPORT_TEMPLATE_CPPNODE( _ema_adjusted,     _computeEMA<AdjustedEMA>);
EXPORT_TEMPLATE_CPPNODE( _ema_alpha_debias, _computeEMA<AlphaDebiasEMA> );


// The following nodes are written independently from _compute
// They either have an additional input (i.e. ddof for covariance) or are implemented differently (i.e. weighted mean, which needs weight inputs)

/*
Computation node for statistics requiring an int argument
*/

// Bivariate statistics
template<typename C>
DECLARE_CPPNODE( _bivariate_compute )
{

protected:
    TS_INPUT( std::vector<double>, x_add );
    TS_INPUT( std::vector<double>, x_rem );
    TS_INPUT( std::vector<double>, y_add );
    TS_INPUT( std::vector<double>, y_rem );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( DataValidator<C>, s_computation );

    TS_OUTPUT( double );

   //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_bivariate_compute"; }

public:
    _bivariate_compute( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    START()
    {
        initDataValidator( s_computation );
    }

    virtual void initDataValidator( DataValidator<C> & ) = 0;

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            s_computation.reset();
        }
        if( csp.ticked( x_add ) )
        {
            const std::vector<double> & additions_x = x_add.lastValue();
            const std::vector<double> & additions_y = y_add.lastValue();
            for( size_t i = 0; i < additions_x.size(); i++ )
                s_computation.add( additions_x[i], additions_y[i] );
        }
        if( csp.ticked( x_rem ) )
        {
            const std::vector<double> & removals_x = x_rem.lastValue();
            const std::vector<double> & removals_y = y_rem.lastValue();
            for ( size_t i = 0; i < removals_x.size(); i++ )
                s_computation.remove( removals_x[i], removals_y[i] );
        }
        if( csp.ticked( trigger ) )
        {
            RETURN( s_computation.compute() );
        }
    }
};

template<typename C>
class _bivarComputeCommonArgs : public _bivariate_compute<C>
{
public:
    using _bivariate_compute<C>::_bivariate_compute;
    _STATIC_CREATE_METHOD( _bivarComputeCommonArgs<C> );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na );
    }
};

template<typename ArgT, typename C>
class _bivarComputeOneArg : public _bivariate_compute<C>
{
public:
    using _bivariate_compute<C>::_bivariate_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _bivarComputeOneArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg );
    }
};

template<typename ArgT, typename C>
class _bivarComputeTwoArg : public _bivariate_compute<C>
{
public:
    using _bivariate_compute<C>::_bivariate_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _bivarComputeTwoArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg1 );
    SCALAR_INPUT( ArgT, arg2 );

    void initDataValidator( DataValidator<C> & validator ) override
    {
        validator = DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg1, this -> arg2 );
    }
};

EXPORT_TEMPLATE_CPPNODE( _weighted_mean,    _bivarComputeCommonArgs<WeightedMean> );
EXPORT_TEMPLATE_CPPNODE( _corr,             _bivarComputeCommonArgs<Correlation> );
EXPORT_TEMPLATE_CPPNODE( _weighted_var,     SINGLE_ARG( _bivarComputeOneArg<int64_t, WeightedVariance> ) );
EXPORT_TEMPLATE_CPPNODE( _weighted_sem,     SINGLE_ARG( _bivarComputeOneArg<int64_t, WeightedStandardError> ) );
EXPORT_TEMPLATE_CPPNODE( _covar,            SINGLE_ARG( _bivarComputeOneArg<int64_t, Covariance> ) );
EXPORT_TEMPLATE_CPPNODE( _weighted_skew,    SINGLE_ARG( _bivarComputeOneArg<bool, WeightedSkew> ) );
EXPORT_TEMPLATE_CPPNODE( _weighted_kurt,    SINGLE_ARG( _bivarComputeTwoArg<bool, WeightedKurtosis> ) );


// Trivariate and multivariate statistics
template<typename C>
DECLARE_CPPNODE( _trivariate_compute )
{
    TS_INPUT( std::vector<double>, x_add );
    TS_INPUT( std::vector<double>, x_rem );
    TS_INPUT( std::vector<double>, y_add );
    TS_INPUT( std::vector<double>, y_rem );
    TS_INPUT( std::vector<double>, w_add );
    TS_INPUT( std::vector<double>, w_rem );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );
    SCALAR_INPUT( int64_t, arg );

    STATE_VAR( DataValidator<C>, s_computation );

    TS_OUTPUT( double );

    INIT_CPPNODE( _trivariate_compute ) { }

    START()
    {
        s_computation = DataValidator<C>( min_data_points, ignore_na, arg );
    }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            s_computation.reset();
        }

        if( csp.ticked( x_add ) )
        {
            const std::vector<double> & additions_x = x_add.lastValue();
            const std::vector<double> & additions_y = y_add.lastValue();
            const std::vector<double> & additions_w = w_add.lastValue();
            for( size_t i = 0; i < additions_x.size(); i++ )
                s_computation.add( additions_x[i], additions_y[i], additions_w[i] );
        }

        if( csp.ticked( x_rem ) )
        {
            const std::vector<double> & removals_x = x_rem.lastValue();
            const std::vector<double> & removals_y = y_rem.lastValue();
            const std::vector<double> & removals_w = w_rem.lastValue();
            for( size_t i = 0; i < removals_x.size(); i++ )
                s_computation.remove( removals_x[i], removals_y[i], removals_w[i] );
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( s_computation.compute() );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _weighted_covar, _trivariate_compute<WeightedCovariance> );
EXPORT_TEMPLATE_CPPNODE( _weighted_corr,  _trivariate_compute<WeightedCorrelation> );

DECLARE_CPPNODE ( _quantile )
{
    TS_INPUT( std::vector<double>, additions );
    TS_INPUT( std::vector<double>, removals );
    SCALAR_INPUT( Dictionary::Vector, quants );
    SCALAR_INPUT( int64_t, interpolation_type );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( DataValidator<Quantile>, s_qtl );

    TS_LISTBASKET_OUTPUT( double );

    INIT_CPPNODE( _quantile ) { }

    START()
    {
        s_qtl = DataValidator<Quantile>( min_data_points, ignore_na, quants.value(), interpolation_type );
    }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            s_qtl.reset();
        }
        if( csp.ticked( additions ) )
        {
            for( double x: additions.lastValue() )
                s_qtl.add( x );
        }
        if( csp.ticked( removals ) )
        {
            for ( double x: removals.lastValue() )
                s_qtl.remove( x );
        }
        if( csp.ticked( trigger ) )
        {
            for ( size_t i = 0; i < quants.value().size(); i++ )
                unnamed_output()[i].output( s_qtl.compute( i ) );
        }
    }

};

EXPORT_CPPNODE ( _quantile );

template<typename C>
DECLARE_CPPNODE( _exp_halflife )
{
    TS_INPUT( double, x );
    SCALAR_INPUT( TimeDelta, halflife );
    SCALAR_INPUT( bool, adjust );

    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );

    STATE_VAR( DataValidator<C>, s_computation );
    TS_OUTPUT( double );

    INIT_CPPNODE( _exp_halflife ) { }

    START()
    {
        s_computation = DataValidator<C>( min_data_points, true, halflife, now(), adjust );
    }

    INVOKE()
    {
        if ( csp.ticked( reset ) )
        {
            s_computation.reset();
        }

        if( csp.ticked( x ) && csp.ticked( sampler ) )
        {
            s_computation.add( x, now() );
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( s_computation.compute() );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _ema_halflife,             _exp_halflife<HalflifeEMA> );
EXPORT_TEMPLATE_CPPNODE( _ema_halflife_adjusted,    _exp_halflife<AdjustedHalflifeEMA> );
EXPORT_TEMPLATE_CPPNODE( _ema_halflife_debias,      _exp_halflife<HalflifeDebiasEMA> );

DECLARE_CPPNODE( _arg_min_max )
{
    TS_INPUT( double, x );
    TS_INPUT( std::vector<double>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( bool, max );
    SCALAR_INPUT( bool, recent );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( DataValidator<ArgMinMax>, s_computation );
    TS_OUTPUT( DateTime );

    INIT_CPPNODE( _arg_min_max ) { }

    START()
    {
        s_computation = DataValidator<ArgMinMax>( min_data_points, ignore_na, max, recent );
    }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            s_computation.reset();
        }

        if( csp.ticked( x ) && csp.ticked( sampler ) )
        {
            s_computation.add( x.lastValue(), now() );
        }

        if( csp.ticked( removals ) )
        {
            for( double v: removals.lastValue() )
                s_computation.remove( v );
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( s_computation.compute_dt() );
        }
    }
};

EXPORT_CPPNODE( _arg_min_max );

}
