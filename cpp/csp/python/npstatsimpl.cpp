
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h> // needs to be included first
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h> // need to be included before csp

//Include first so that csp dialect types are defined
#include <csp/python/Common.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyObjectPtr.h>

#include <csp/core/Time.h>
#include <csp/cppnodes/statsimpl.h>
#include <csp/engine/CppNode.h>
#include <algorithm>
#include <cstddef>

#define NPY_SHAPE_ERROR "Shape of the NumPy array was unknown at the time the trigger ticked."

// Need to copy these from NumpyConversions.h due to the autogen build
// autogen types rely on _cspimpl.so files to be created first
template<typename T> struct NPY_TYPE { static const int value = NPY_OBJECT; };
template<> struct NPY_TYPE<int64_t>  { static const int value = NPY_LONGLONG; };
template<> struct NPY_TYPE<double>   { static const int value = NPY_DOUBLE; };

using namespace csp::cppnodes;

static void * init_nparray()
{
    csp::python::AcquireGIL gil;
    import_array();
    return nullptr;
}
static void * s_init_array = init_nparray();

// NumPy specific statistic functions

namespace csp::python
{

template<typename T = double>
class NumPyIterator
{
    // Code to iterate over an arbitrary NumPy array
    // Iterates over the array in a non-necessarily contiguous order, so all operations must be defined only element-wise

    public:

        NumPyIterator()
        {
            m_nd = 0;
            m_index = 0;
            m_data = nullptr;
            m_strides = nullptr;
            m_stridedDimensions = {};
            m_current = {};
            m_valid = false;
        }

        NumPyIterator( PyArrayObject * arr )
        {
            setup( arr );
        }

        NumPyIterator( PyObject * arr )
        {
            if( unlikely( !PyArray_Check( arr ) ) )
                CSP_THROW( csp::TypeError, "Expected NumPy array type, got " << Py_TYPE( arr ) -> tp_name );
            setup( ( PyArrayObject * ) arr );
        }

        NumPyIterator( NumPyIterator && rhs ) = default;

        NumPyIterator & operator=( NumPyIterator && rhs ) = default;

        NumPyIterator( const NumPyIterator & rhs ) = delete;

        NumPyIterator & operator=( const NumPyIterator & rhs ) = delete;

        operator bool() const { return m_valid; }

        void operator++()
        {
            for( int i = m_nd - 1; i >=0 ; --i )
            {
                if( ++m_current[i] < m_dims[i] )
                {
                    m_data += m_strides[i];;
                    m_index++;
                    return;
                }
                m_data -= m_stridedDimensions[i];
                m_current[i] = 0;
            }
            m_valid = false;
        }

        int64_t index() const { return m_index; }

        const npy_intp * indices() const { return (npy_intp*)m_current.data(); }

        T & value() const { return *reinterpret_cast<T*>( m_data ); }

    private:

        void verify_arr( PyArrayObject * arr )
        {
            auto expType = PyArray_DescrFromType( NPY_TYPE<T>::value );
            if( unlikely( PyObject_RichCompareBool( ( PyObject * ) PyArray_DESCR( arr ), ( PyObject * ) expType, Py_EQ ) ) != 1 )
            {
                CSP_THROW( csp::TypeError,
                            "Expected array of type " << PyObjectPtr::own( PyObject_Repr( ( PyObject * ) expType ) )
                                                        << " got "
                                                        << PyObjectPtr::own( PyObject_Repr( ( PyObject * ) PyArray_DESCR( arr ) ) ) );
            }
        }

        void setup( PyArrayObject * arr )
        {
            verify_arr( arr );
            m_index = 0;
            m_nd = PyArray_NDIM( arr );
            m_current.resize( m_nd, 0 );
            // set strides and dimension
            m_dims = PyArray_DIMS( arr );
            m_strides = PyArray_STRIDES( arr );
            for( int i = 0; i < m_nd; ++i )
                m_stridedDimensions.emplace_back( m_strides[i] * ( m_dims[i] - 1 ) );
            m_data = reinterpret_cast<char*>( PyArray_DATA( arr ) );
            m_valid = ( PyArray_SIZE( arr ) > 0 );
        }

        int64_t m_nd;
        int64_t m_index;
        char* m_data;
        npy_intp* m_strides;
        npy_intp * m_dims;
        std::vector<int64_t> m_stridedDimensions;
        std::vector<int64_t> m_current;
        bool m_valid;
};

struct PyShape
{
    PyShape()
    {
        m_dims = {};
        m_n = 0;
    }

    PyShape( PyArrayObject* arr )
    {
        int64_t nd = PyArray_NDIM( arr );
        m_dims.reserve( nd );
        npy_intp* dims = PyArray_DIMS( arr );
        for( int64_t j = 0; j < nd; j++ )
            m_dims.emplace_back( dims[j] );
        m_n = PyArray_SIZE( arr );
    }

    PyShape( PyObject* arr ) : PyShape( ( PyArrayObject* ) arr ) { }

    PyShape( std::vector<npy_intp> dims, int64_t n )
    {
        m_dims = dims;
        m_n = n;
    }

    void validateShape( PyArrayObject* arr )
    {
        size_t nd = PyArray_NDIM( arr );
        if( nd != m_dims.size() )
            CSP_THROW( ValueError, "Inconsistent shape of NumPy arrays for computation: different number of dimensions");

        npy_intp* dims = PyArray_DIMS( arr );
        for( size_t i = 0; i < nd; i++ )
        {
            if( dims[i] != m_dims[i] )
                CSP_THROW( ValueError, "Inconsistent shape of NumPy arrays for computation: difference in dimension " << i );
        }
    }

    void validateShape( PyObject* arr )
    {
        validateShape( ( PyArrayObject* ) arr );
    }

    bool operator!=( const PyShape & rhs ) const
    {
        if( m_n != rhs.m_n ) return true;
        if( m_dims.size() != rhs.m_dims.size() ) return true;
        for( size_t i = 0; i < m_dims.size(); i++ )
        {
            if( m_dims[i] != rhs.m_dims[i] )
                return true;
        }
        return false;
    }

    bool operator==( const PyShape & rhs ) const { return !( *this != rhs ); }

    std::vector<npy_intp> m_dims;
    int64_t m_n;

};

PyObject* createNanWithShape( PyObject* t )
{
    PyArrayObject* as_pyarr = ( PyArrayObject* )t;
    int64_t nd = PyArray_NDIM( as_pyarr );
    PyObject* ret = PyArray_EMPTY( nd, PyArray_DIMS( as_pyarr ), NPY_DOUBLE, 0 );
    PyArray_FillWithScalar( ( PyArrayObject* ) ret, ( PyObject* ) PyFloat_FromDouble( std::numeric_limits<double>::quiet_NaN() ) );
    return ret;
}

PyObject* createZerosWithShape( PyObject* t )
{
    PyArrayObject* as_pyarr = ( PyArrayObject* )t;
    int64_t nd = PyArray_NDIM( as_pyarr );
    return PyArray_Zeros( nd, PyArray_DIMS( as_pyarr ), PyArray_DescrFromType( NPY_DOUBLE ), 0 );
}

// C: computation class
template<typename C>
inline PyObject* computeArray( const PyShape & shp, std::vector<C> & elem, bool s_first )
{
    if( unlikely( s_first ) )
        CSP_THROW( ValueError, NPY_SHAPE_ERROR );
    PyObject* out = PyArray_EMPTY( shp.m_dims.size(), &shp.m_dims[0], NPY_DOUBLE, 0 );
    for( NumPyIterator iter( out ); iter; ++iter )
        iter.value() = elem[iter.index()].compute();

    return out;
}

// Window update nodes

class _np_tick_window_updates : public _generic_tick_window_updates<PyObjectPtr, _np_tick_window_updates>
{
public:
    using _generic_tick_window_updates<PyObjectPtr, _np_tick_window_updates>::_generic_tick_window_updates;
    _STATIC_CREATE_METHOD( _np_tick_window_updates );

    STATE_VAR( PyShape, s_shp{} );

    inline PyObjectPtr createNan()
    {
        return PyObjectPtr::own( createNanWithShape( x.lastValue().get() ) );
    }

    inline void validateShape()
    {
        PyObject* arr = x.lastValue().get();
        PyShape shp( arr );
        if( unlikely( s_first ) )
            s_shp = shp;
        else
            s_shp.validateShape( arr );
    }

    inline void checkValid()
    {
        if( !csp.valid( this -> x ) )
            CSP_THROW( ValueError, "Error: sampler called on a NumPy array before any data ticks - shape is unknown.");
    }
};

EXPORT_CPPNODE( _np_tick_window_updates );

class _np_time_window_updates : public _generic_time_window_updates<PyObjectPtr, _np_time_window_updates>
{
public:
    using _generic_time_window_updates<PyObjectPtr,_np_time_window_updates>::_generic_time_window_updates;
    _STATIC_CREATE_METHOD( _np_time_window_updates );

    STATE_VAR( PyShape, s_shp{} );

    inline PyObjectPtr createNan()
    {
        return PyObjectPtr::own( createNanWithShape( x.lastValue().get() ) );
    }

    inline void validateShape()
    {
        PyObject* arr = x.lastValue().get();
        PyShape shp( arr );
        if( unlikely( s_first ) )
            s_shp = shp;
        else
            s_shp.validateShape( arr );
    }

    inline void checkValid()
    {
        if( !csp.valid( this -> x ) )
            CSP_THROW( ValueError, "Error: sampler called on a NumPy array before any data ticks - shape is unknown.");
    }
};

EXPORT_CPPNODE( _np_time_window_updates );

class _cross_sectional_as_np : public _generic_cross_sectional<double, PyObjectPtr, _cross_sectional_as_np>
{
public:
    using _generic_cross_sectional<double, PyObjectPtr, _cross_sectional_as_np>::_generic_cross_sectional;
    _STATIC_CREATE_METHOD( _cross_sectional_as_np );

    inline void computeCrossSectional()
    {
        npy_intp n = s_window.count();
        PyObject* out = PyArray_EMPTY(1, &n, NPY_DOUBLE, 0 ); // 1D array
        double* data = reinterpret_cast<double*>( PyArray_DATA( ( PyArrayObject* )out ) );
        s_window.copy_values( data );
        RETURN( PyObjectPtr::own( out ) );
    }
};

EXPORT_CPPNODE( _cross_sectional_as_np );

class _np_cross_sectional_as_list : public _generic_cross_sectional<PyObjectPtr, std::vector<PyObjectPtr>, _np_cross_sectional_as_list>
{
public:
    using _generic_cross_sectional<PyObjectPtr, std::vector<PyObjectPtr>, _np_cross_sectional_as_list>::_generic_cross_sectional;
    _STATIC_CREATE_METHOD( _np_cross_sectional_as_list );

    inline void computeCrossSectional()
    {
        s_window.copy_values( &unnamed_output().reserveSpace() );
    }
};

EXPORT_CPPNODE( _np_cross_sectional_as_list );

class _np_cross_sectional_as_np : public _generic_cross_sectional<PyObjectPtr, PyObjectPtr, _np_cross_sectional_as_np>
{
public:
    using _generic_cross_sectional<PyObjectPtr, PyObjectPtr, _np_cross_sectional_as_np>::_generic_cross_sectional;
    _STATIC_CREATE_METHOD( _np_cross_sectional_as_np );

    inline void computeCrossSectional()
    {
        // Return as a NumPy array of n+1 dimensions
        PyObject* out;
        if( s_window.count() )
        {
            WindowBuffer<PyObjectPtr>::const_iterator it = s_window.left();
            PyArrayObject* fval = ( PyArrayObject* )it.value().get();
            int64_t nd = PyArray_NDIM( fval );
            npy_intp* inner_dims = PyArray_DIMS( fval );
            npy_intp* dims = new npy_intp[nd+1];
            dims[0] = s_window.count();
            for( int64_t i = 0; i < nd; i++ )
                dims[i+1] = inner_dims[i];

            out = PyArray_EMPTY( nd+1, dims, NPY_DOUBLE, 0 );
            double* data = reinterpret_cast<double*>( PyArray_DATA( ( PyArrayObject* )out ) );
            int64_t np_index = 0;

            // Copy directly from buffer
            for( int64_t i = 0; i < s_window.count(); i++ )
            {
                PyObject* v = it.value().get(); // reference
                for( NumPyIterator iter( v ); iter; ++iter )
                {
                    data[np_index] = iter.value();
                    np_index++;
                }
                ++it;
            }
        }
        else
        {
            // No values: return empty 1D array of zero size
            npy_intp dims[]{ ( npy_intp ) 0 };
            out = PyArray_SimpleNew( 1, dims, NPY_DOUBLE );
        }
        RETURN( PyObjectPtr::own( out ) );
    }
};

EXPORT_CPPNODE( _np_cross_sectional_as_np );

DECLARE_CPPNODE( _list_to_np )
{
    TS_LISTBASKET_INPUT( double, x );
    SCALAR_INPUT( bool, fillna );
    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _list_to_np ) { }

    INVOKE()
    {
        npy_intp n = x.size();
        PyObject* out = PyArray_EMPTY( 1, &n, NPY_DOUBLE, 0 );
        double* data = reinterpret_cast<double*>( PyArray_DATA( ( PyArrayObject* )out ) );
        for( int64_t i = 0; i < n; i++ )
        {
            if( csp.valid( x[i] ) && ( fillna || csp.ticked( x[i] ) ) )
                data[i] = x[i];
            else
                data[i] = std::numeric_limits<double>::quiet_NaN();
        }

        RETURN( PyObjectPtr::own( out ) );
    }
};

EXPORT_CPPNODE( _list_to_np );

DECLARE_CPPNODE( _np_to_list )
{
    TS_INPUT( PyObjectPtr, x );
    SCALAR_INPUT( int64_t, n );

    TS_LISTBASKET_OUTPUT( double );

    INIT_CPPNODE( _np_to_list ) { }

    START()
    {
        if( n == 0)
            CSP_THROW( ValueError, "Must provide at least one output channel for NumPy conversion");
    }

    INVOKE()
    {
        PyArrayObject* arr = ( PyArrayObject* )x.lastValue().get();

        // Ensure shape is valid
        int64_t nd = PyArray_NDIM( arr );
        if( nd != 1 )
            CSP_THROW( ValueError, "Cannot convert NumPy array of more than one dimension into listbasket");

        int64_t n_arr = PyArray_SIZE( arr );
        if( n_arr != n )
            CSP_THROW( ValueError, "Cannot convert NumPy array of size " << n_arr << " into listbasket of " << n << " elements");

        double* data = reinterpret_cast<double*>( PyArray_DATA( arr ) );
        for( int64_t i = 0; i < n; i++ )
            unnamed_output()[i].output( data[i] );
    }
};

EXPORT_CPPNODE( _np_to_list );

DECLARE_CPPNODE( _sync_nan_np )
{
protected:
    TS_INPUT( PyObjectPtr, x );
    TS_INPUT( PyObjectPtr, y );

    TS_NAMED_OUTPUT( PyObjectPtr, x_sync );
    TS_NAMED_OUTPUT( PyObjectPtr, y_sync );

    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{ true } );

    INIT_CPPNODE( _sync_nan_np ) { }

    INVOKE()
    {
        // Note x and y are guaranteed to tick in sequence in contexts where sync_nan is used
        PyArrayObject * xval = ( PyArrayObject* )x.lastValue().get();
        PyArrayObject * yval = ( PyArrayObject* )y.lastValue().get();

        if( unlikely( s_first ) )
        {
            s_shp = PyShape( xval );
            s_first = false;
        }
        else
            s_shp.validateShape( xval );

        s_shp.validateShape( yval );

        // Only allocate a new array if we need to change values
        PyArrayObject * xsync = nullptr;
        PyArrayObject * ysync = nullptr;
        for( NumPyIterator itx = {xval}, ity = {yval}; itx && ity; ++itx, ++ity )
        {
            if( isnan( ity.value() ) && !isnan( itx.value() ) ) // need to set corresponding element in x as nan
            {
                if( !xsync )
                {
                    xsync = ( PyArrayObject* )PyArray_NewLikeArray( xval, NPY_KEEPORDER, NULL, 0 );
                    PyArray_CopyInto( xsync, xval );
                }
                double * element = ( double * )PyArray_GetPtr( xsync, itx.indices() );
                *( reinterpret_cast<double*>( element ) ) = std::numeric_limits<double>::quiet_NaN();
            }
            else if( isnan( itx.value() ) && !isnan( ity.value() ) ) // need to set corresponding element in y as nan
            {
                if( !ysync )
                {
                    ysync = ( PyArrayObject* )PyArray_NewLikeArray( yval, NPY_KEEPORDER, NULL, 0 );
                    PyArray_CopyInto( ysync, yval );
                }
                double * element = ( double * )PyArray_GetPtr( ysync, ity.indices() );
                *element = std::numeric_limits<double>::quiet_NaN();
            }
        }

        if( !xsync ) // only copy into a new array if a NaN was substituted
        {
            xsync = xval;
            Py_XINCREF( xval );
        }
        if( !ysync )
        {
            ysync = yval;
            Py_XINCREF( yval );
        }

        x_sync.output( PyObjectPtr::own( ( PyObject * )xsync ) );
        y_sync.output( PyObjectPtr::own( ( PyObject * )ysync ) );
    }
};

EXPORT_CPPNODE( _sync_nan_np );

// C: computation class
template<typename C>
DECLARE_CPPNODE( _np_compute )
{
protected:
    TS_INPUT( std::vector<PyObjectPtr>, additions );
    TS_INPUT( std::vector<PyObjectPtr>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( std::vector<DataValidator<C>>, s_elem );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );
    TS_OUTPUT( PyObjectPtr );

    //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_np_compute"; }

public:
    _np_compute( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    virtual DataValidator<C> initDataValidator() = 0;

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( additions ) )
        {
            if( unlikely( s_first ) )
            {
                PyObject* arr = additions.lastValue()[0].get();
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int j = 0; j < s_shp.m_n; j++ )
                {
                    s_elem.emplace_back( initDataValidator() );
                }
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = additions.lastValue().size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = additions.lastValue()[i].get();
                for( NumPyIterator iter( v ); iter; ++iter )
                {
                    s_elem[iter.index()].add( iter.value() );
                }
            }
        }
        if( csp.ticked( removals ) )
        {
            size_t m = removals.lastValue().size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = removals.lastValue()[i].get();
                for( NumPyIterator iter( v ); iter; ++iter )
                    s_elem[iter.index()].remove( iter.value() );
            }
        }
        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

template<typename C>
class _npComputeCommonArgs : public _np_compute<C>
{
public:
    using _np_compute<C>::_np_compute;
    _STATIC_CREATE_METHOD( _npComputeCommonArgs<C> );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na );
    }
};

template<typename ArgT, typename C>
class _npComputeOneArg : public _np_compute<C>
{
public:
    using _np_compute<C>::_np_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _npComputeOneArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg );
    }
};

template<typename ArgT, typename C>
class _npComputeTwoArg : public _np_compute<C>
{
public:
    using _np_compute<C>::_np_compute;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _npComputeTwoArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg1 );
    SCALAR_INPUT( ArgT, arg2 );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg1, this -> arg2 );
    }
};

template<typename C>
class _npComputeEMA : public _np_compute<C>
{
public:
    using _np_compute<C>::_np_compute;
    _STATIC_CREATE_METHOD( _npComputeEMA<C> );
     SCALAR_INPUT( double, alpha );
    SCALAR_INPUT( int64_t, horizon );
    SCALAR_INPUT( bool, adjust );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, true, alpha, this -> ignore_na, horizon, adjust );
    }
};


// Export node templates
EXPORT_TEMPLATE_CPPNODE( _np_count,             _npComputeCommonArgs<Count> );
EXPORT_TEMPLATE_CPPNODE( _np_sum,               _npComputeCommonArgs<Sum> );
EXPORT_TEMPLATE_CPPNODE( _np_kahan_sum,         _npComputeCommonArgs<KahanSum> );
EXPORT_TEMPLATE_CPPNODE( _np_mean,              _npComputeCommonArgs<Mean> );
EXPORT_TEMPLATE_CPPNODE( _np_prod,              _npComputeCommonArgs<Product> );
EXPORT_TEMPLATE_CPPNODE( _np_first,             _npComputeCommonArgs<First> );
EXPORT_TEMPLATE_CPPNODE( _np_last,              _npComputeCommonArgs<Last> );
EXPORT_TEMPLATE_CPPNODE( _np_unique,            SINGLE_ARG( _npComputeOneArg<int64_t, Unique> ) );
EXPORT_TEMPLATE_CPPNODE( _np_var,               SINGLE_ARG( _npComputeOneArg<int64_t, Variance> ) );
EXPORT_TEMPLATE_CPPNODE( _np_sem,               SINGLE_ARG( _npComputeOneArg<int64_t, StandardError> ) );
EXPORT_TEMPLATE_CPPNODE( _np_min_max,           SINGLE_ARG( _npComputeOneArg<bool, AscendingMinima> ) );
EXPORT_TEMPLATE_CPPNODE( _np_skew,              SINGLE_ARG( _npComputeOneArg<bool, Skew> ) );
EXPORT_TEMPLATE_CPPNODE( _np_rank,              SINGLE_ARG( _npComputeTwoArg<int64_t, Rank> ) );
EXPORT_TEMPLATE_CPPNODE( _np_kurt,              SINGLE_ARG( _npComputeTwoArg<bool, Kurtosis> ) );
EXPORT_TEMPLATE_CPPNODE( _np_ema_compute,       _npComputeEMA<EMA> );
EXPORT_TEMPLATE_CPPNODE( _np_ema_adjusted,      _npComputeEMA<AdjustedEMA>);
EXPORT_TEMPLATE_CPPNODE( _np_ema_alpha_debias,  _npComputeEMA<AlphaDebiasEMA> );

// Bivariate
template<typename C>
DECLARE_CPPNODE( _np_bivariate )
{
protected:
    TS_INPUT( std::vector<PyObjectPtr>, x_add );
    TS_INPUT( std::vector<PyObjectPtr>, x_rem );
    TS_INPUT( std::vector<PyObjectPtr>, w_add );
    TS_INPUT( std::vector<PyObjectPtr>, w_rem );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( std::vector<DataValidator<C>>, s_elem{} );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_OUTPUT( PyObjectPtr );

    //Expanded out INIT_CPPNODE without create call...
    CSP csp;
    const char * name() const override { return "_np_bivariate"; }

public:
    _np_bivariate( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )
    {}

    virtual DataValidator<C> initDataValidator() = 0;

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( x_add ) )
        {
            const std::vector<PyObjectPtr> & add_x = x_add.lastValue();
            const std::vector<PyObjectPtr> & weights = w_add.lastValue();

            if( unlikely( s_first ) )
            {
                PyObject* arr = add_x[0].get();
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int64_t j = 0; j < s_shp.m_n; j++ )
                {
                    s_elem.emplace_back( initDataValidator() );
                }
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = add_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = add_x[i].get();
                PyObject* w = weights[i].get();
                s_shp.validateShape( w );

                for( NumPyIterator iter = {v}, w_iter = {w}; iter && w_iter; ++iter, ++w_iter )
                {
                    s_elem[iter.index()].add( iter.value(), w_iter.value() );
                }
            }
        }

        if( csp.ticked( x_rem ) )
        {
            const std::vector<PyObjectPtr> & rem_x = x_rem.lastValue();
            const std::vector<PyObjectPtr> & weights = w_rem.lastValue();

            size_t m = rem_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = rem_x[i].get();
                PyObject* w = weights[i].get();

                for( NumPyIterator iter = {v}, w_iter = {w}; iter && w_iter; ++iter, ++w_iter )
                    s_elem[iter.index()].remove( iter.value(), w_iter.value() );
            }
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

template<typename C>
class _npBivarCompute : public _np_bivariate<C>
{
public:
    using _np_bivariate<C>::_np_bivariate;
    _STATIC_CREATE_METHOD( _npBivarCompute<C> );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na );
    }
};

template<typename ArgT, typename C>
class _npBivarComputeOneArg : public _np_bivariate<C>
{
public:
    using _np_bivariate<C>::_np_bivariate;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _npBivarComputeOneArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg );
    }
};


template<typename ArgT, typename C>
class _npBivarComputeTwoArg : public _np_bivariate<C>
{
public:
    using _np_bivariate<C>::_np_bivariate;
    _STATIC_CREATE_METHOD( SINGLE_ARG( _npBivarComputeTwoArg<ArgT, C> ) );
    SCALAR_INPUT( ArgT, arg1 );
    SCALAR_INPUT( ArgT, arg2 );

    DataValidator<C> initDataValidator() override
    {
        return DataValidator<C>( this -> min_data_points, this -> ignore_na, this -> arg1, this -> arg2 );
    }
};

EXPORT_TEMPLATE_CPPNODE( _np_weighted_mean,     _npBivarCompute<WeightedMean> );
EXPORT_TEMPLATE_CPPNODE( _np_corr,              _npBivarCompute<Correlation> );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_var,      SINGLE_ARG( _npBivarComputeOneArg<int64_t, WeightedVariance> ) );
EXPORT_TEMPLATE_CPPNODE( _np_covar,             SINGLE_ARG( _npBivarComputeOneArg<int64_t, Covariance> ) );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_sem,      SINGLE_ARG( _npBivarComputeOneArg<int64_t, WeightedStandardError> ) );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_skew,     SINGLE_ARG( _npBivarComputeOneArg<bool, WeightedSkew> ) );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_kurt,     SINGLE_ARG( _npBivarComputeTwoArg<bool, WeightedKurtosis> ) );

// Other node templates

/*
@csp.node
def _np_quantile(additions: ts[[np.ndarray]], removals: ts[[np.ndarray]], quants: typing.List[float], nq: int, interpolation_type: int, trigger: ts[object]):
    __outputs__(ts[np.ndarray])
*/

DECLARE_CPPNODE ( _np_quantile )
{
    TS_INPUT( std::vector<PyObjectPtr>, additions );
    TS_INPUT( std::vector<PyObjectPtr>, removals );
    SCALAR_INPUT( Dictionary::Vector, quants );
    SCALAR_INPUT( int64_t, interpolation_type );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( std::vector<DataValidator<Quantile>>, s_elem );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_LISTBASKET_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_quantile ) { }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( additions ) )
        {
            if( unlikely( s_first ) )
            {
                PyObject* arr = additions.lastValue()[0].get();
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int64_t j = 0; j < s_shp.m_n; j++ )
                {
                    s_elem.emplace_back( DataValidator<Quantile>( min_data_points, ignore_na, quants, interpolation_type ) );
                }
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = additions.lastValue().size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = additions.lastValue()[i].get();
                for( NumPyIterator iter( v ); iter; ++iter )
                    s_elem[iter.index()].add( iter.value() );
            }
        }

        if( csp.ticked( removals ) )
        {
            size_t m = removals.lastValue().size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = removals.lastValue()[i].get();
                for( NumPyIterator iter( v ); iter; ++iter )
                    s_elem[iter.index()].remove( iter.value() );
            }
        }

        if( csp.ticked( trigger ) )
        {
            int64_t nq = quants.value().size();
            for( int64_t j = 0; j < nq; j++ )
            {
                PyObject* out = PyArray_EMPTY( s_shp.m_dims.size(), &s_shp.m_dims[0], NPY_DOUBLE, 0 );
                for( NumPyIterator iter( out ); iter; ++iter )
                {
                    iter.value() = s_elem[iter.index()].compute( j ); // compute jth quantile
                }
                unnamed_output()[j].output( PyObjectPtr::own( out ) );
            }
        }
    }

};

EXPORT_CPPNODE ( _np_quantile );

// C: computation class
template<typename C>
DECLARE_CPPNODE( _np_exp_halflife )
{
    TS_INPUT( PyObjectPtr, x );
    SCALAR_INPUT( TimeDelta, halflife );
    SCALAR_INPUT( bool, adjust );

    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );

    STATE_VAR( std::vector<DataValidator<C>>, s_elem );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_exp_halflife ) { }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }

        if( csp.ticked( sampler ) && csp.ticked( x ) )
        {
            PyObject* arr = x.lastValue().get();
            if( unlikely( s_first ) )
            {
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int64_t j = 0; j < s_shp.m_n; j++ )
                {
                    s_elem.emplace_back( DataValidator<C>( min_data_points, true, halflife, now() - TimeDelta::fromMicroseconds( 1 ), adjust ) );
                }
                s_first = false;
            }
            // Add upon iteration
            for( NumPyIterator iter( arr ); iter; ++iter )
                s_elem[iter.index()].add( iter.value(), now() );
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _np_ema_halflife,          _np_exp_halflife<HalflifeEMA> );
EXPORT_TEMPLATE_CPPNODE( _np_ema_halflife_adjusted, _np_exp_halflife<AdjustedHalflifeEMA> );
EXPORT_TEMPLATE_CPPNODE( _np_ema_halflife_debias,   _np_exp_halflife<HalflifeDebiasEMA> );

template<typename C>
DECLARE_CPPNODE( _np_matrix_compute )
{
    TS_INPUT( std::vector<PyObjectPtr>, additions );
    TS_INPUT( std::vector<PyObjectPtr>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );
    SCALAR_INPUT( int64_t, ddof )

    STATE_VAR( std::vector<DataValidator<C>>, s_elem );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_matrix_compute ) { }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( additions ) )
        {
            const std::vector<PyObjectPtr> & add_x = additions.lastValue();
            if( unlikely( s_first ) )
            {
                PyObject* arr = add_x[0].get();
                if( PyArray_NDIM( (PyArrayObject * ) arr ) != 1 )
                    CSP_THROW( ValueError, "Covariance or correlation matrix called on an array of more than 1 dimension: undefined" );
                // Special setup
                int64_t n = PyArray_Size( arr );
                s_shp = PyShape( {n,n}, n );
                s_elem.reserve( n*n );
                for( int64_t j = 0; j < n*n; j++ )
                {
                    s_elem.emplace_back( DataValidator<C>( min_data_points, ignore_na, ddof ) );
                }
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = add_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = add_x[i].get();
                for( NumPyIterator iter1( v ); iter1; ++iter1 )
                {
                    double x = iter1.value();
                    for( NumPyIterator iter2( v ); iter2; ++iter2 )
                    {
                        double y = iter2.value();
                        s_elem[iter1.index()*s_shp.m_n + iter2.index()].add( x, y );
                    }
                }
            }
        }

        if( csp.ticked( removals ) )
        {
            const std::vector<PyObjectPtr> & rem_x = removals.lastValue();
            // Iterate on each element of the array
            size_t m = rem_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = rem_x[i].get();
                for( NumPyIterator iter1( v ); iter1; ++iter1 )
                {
                    double x = iter1.value();
                    for( NumPyIterator iter2( v ); iter2; ++iter2 )
                    {
                        double y = iter2.value();
                        s_elem[iter1.index()*s_shp.m_n + iter2.index()].remove( x, y );
                    }
                }
            }
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _np_cov_matrix,  _np_matrix_compute<Covariance> );
EXPORT_TEMPLATE_CPPNODE( _np_corr_matrix, _np_matrix_compute<Correlation> );

template<typename C>
DECLARE_CPPNODE( _np_weighted_matrix_compute )
{
    TS_INPUT( std::vector<PyObjectPtr>, x_add );
    TS_INPUT( std::vector<PyObjectPtr>, x_rem );
    TS_INPUT( std::vector<double>, w_add );
    TS_INPUT( std::vector<double>, w_rem );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, ddof );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( std::vector<DataValidator<C>>, s_elem );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_weighted_matrix_compute ) { }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( x_add ) )
        {
            const std::vector<PyObjectPtr> & add_x = x_add.lastValue();
            const std::vector<double> & add_w = w_add.lastValue();
            if( unlikely( s_first ) )
            {
                PyObject* arr = add_x[0].get();
                if( PyArray_NDIM( (PyArrayObject * ) arr ) != 1 )
                    CSP_THROW( ValueError, "Covariance matrix called on an array of more than 1 dimension: undefined" );
                // Special setup
                int64_t n = PyArray_Size( arr );
                s_shp = PyShape( {n,n}, n );
                s_elem.reserve( n*n );
                for( int64_t j = 0; j < n*n; j++ )
                {
                    s_elem.emplace_back( DataValidator<C>( min_data_points, ignore_na, ddof ) );
                }
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = add_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                double w = add_w[i];
                PyObject* v = add_x[i].get();
                for( NumPyIterator iter1( v ); iter1; ++iter1 )
                {
                    double x = iter1.value();
                    for( NumPyIterator iter2( v ); iter2; ++iter2 )
                    {
                        double y = iter2.value();
                        s_elem[iter1.index()*s_shp.m_n + iter2.index()].add( x, y, w );
                    }
                }
            }
        }

        if( csp.ticked( x_rem ) )
        {
            const std::vector<PyObjectPtr> & rem_x = x_rem.lastValue();
            const std::vector<double> & rem_w = w_rem.lastValue();
            // Iterate on each element of the array
            size_t m = rem_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                double w = rem_w[i];
                PyObject* v = rem_x[i].get();
                for( NumPyIterator iter1( v ); iter1; ++iter1 )
                {
                    double x = iter1.value();
                    for( NumPyIterator iter2( v ); iter2; ++iter2 )
                    {
                        double y = iter2.value();
                        s_elem[iter1.index()*s_shp.m_n + iter2.index()].remove( x, y, w );
                    }
                }
            }
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _np_weighted_cov_matrix,  _np_weighted_matrix_compute<WeightedCovariance> );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_corr_matrix, _np_weighted_matrix_compute<WeightedCorrelation> );

// trivariate
template<typename C>
DECLARE_CPPNODE( _np_trivariate )
{
protected:
    TS_INPUT( std::vector<PyObjectPtr>, x_add );
    TS_INPUT( std::vector<PyObjectPtr>, x_rem );
    TS_INPUT( std::vector<PyObjectPtr>, y_add );
    TS_INPUT( std::vector<PyObjectPtr>, y_rem );
    TS_INPUT( std::vector<PyObjectPtr>, w_add );
    TS_INPUT( std::vector<PyObjectPtr>, w_rem );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );
    SCALAR_INPUT( int64_t, arg );

    STATE_VAR( std::vector<DataValidator<C>>, s_elem{} );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );

    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_trivariate ) { }

public:

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( x_add ) )
        {
            const std::vector<PyObjectPtr> & add_x = x_add.lastValue();
            const std::vector<PyObjectPtr> & add_y = y_add.lastValue();
            const std::vector<PyObjectPtr> & weights = w_add.lastValue();

            if( unlikely( s_first ) )
            {
                PyObject* arr = add_x[0].get();
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int64_t j = 0; j < s_shp.m_n; j++ )
                    s_elem.emplace_back( DataValidator<C>( min_data_points, ignore_na, arg ) );
                s_first = false;
            }
            // Iterate on each element of the array
            size_t m = add_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* x = add_x[i].get();
                PyObject* y = add_y[i].get();
                PyObject* w = weights[i].get();
                s_shp.validateShape( y );
                s_shp.validateShape( w );

                for( NumPyIterator x_iter = {x}, y_iter = {y}, w_iter = {w}; x_iter && y_iter && w_iter; ++x_iter, ++w_iter, ++y_iter )
                    s_elem[x_iter.index()].add( x_iter.value(), y_iter.value(), w_iter.value() );
            }
        }

        if( csp.ticked( x_rem ) )
        {
            const std::vector<PyObjectPtr> & rem_x = x_rem.lastValue();
            const std::vector<PyObjectPtr> & rem_y = y_rem.lastValue();
            const std::vector<PyObjectPtr> & weights = w_rem.lastValue();

            size_t m = rem_x.size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* x = rem_x[i].get();
                PyObject* y = rem_y[i].get();
                PyObject* w = weights[i].get();

                for( NumPyIterator x_iter = {x}, y_iter = {y}, w_iter = {w}; x_iter && y_iter && w_iter; ++x_iter, ++w_iter, ++y_iter )
                    s_elem[x_iter.index()].remove( x_iter.value(), y_iter.value(), w_iter.value() );
            }
        }

        if( csp.ticked( trigger ) )
        {
            RETURN( PyObjectPtr::own( computeArray<DataValidator<C>>( s_shp, s_elem, s_first ) ) );
        }
    }
};

EXPORT_TEMPLATE_CPPNODE( _np_weighted_covar, _np_trivariate<WeightedCovariance> );
EXPORT_TEMPLATE_CPPNODE( _np_weighted_corr,  _np_trivariate<WeightedCorrelation> );

DECLARE_CPPNODE( _np_arg_min_max )
{
    TS_INPUT( PyObjectPtr, x );
    TS_INPUT( std::vector<PyObjectPtr>, removals );
    TS_INPUT( Generic, trigger );
    TS_INPUT( Generic, sampler );
    TS_INPUT( Generic, reset );
    SCALAR_INPUT( bool, max );
    SCALAR_INPUT( bool, recent );
    SCALAR_INPUT( int64_t, min_data_points );
    SCALAR_INPUT( bool, ignore_na );

    STATE_VAR( std::vector<DataValidator<ArgMinMax>>, s_elem{} );
    STATE_VAR( PyShape, s_shp{} );
    STATE_VAR( bool, s_first{true} );
    TS_OUTPUT( PyObjectPtr );

    INIT_CPPNODE( _np_arg_min_max ) { }

    INVOKE()
    {
        if( csp.ticked( reset ) )
        {
            for( size_t j = 0; j < s_elem.size(); j++ )
                s_elem[j].reset();
        }
        if( csp.ticked( x ) && csp.ticked( sampler ) )
        {
            PyObject* arr = x.lastValue().get();
            if( unlikely( s_first ) )
            {
                s_shp = PyShape( arr );
                s_elem.reserve( s_shp.m_n );
                for( int64_t j = 0; j < s_shp.m_n; j++ )
                {
                    s_elem.emplace_back( DataValidator<ArgMinMax>( min_data_points, ignore_na, max, recent ) );
                }
                s_first = false;
            }
            for( NumPyIterator iter = {arr}; iter; ++iter )
                s_elem[iter.index()].add( iter.value(), now() );
        }

        if( csp.ticked( removals ) )
        {
            size_t m = removals.lastValue().size();
            for( size_t i = 0; i < m; i++ )
            {
                PyObject* v = removals.lastValue()[i].get();
                for( NumPyIterator iter = {v}; iter; ++iter )
                    s_elem[iter.index()].remove( iter.value() );
            }
        }

        if( csp.ticked( trigger ) )
        {
            if( unlikely( s_first ) )
                CSP_THROW( ValueError, NPY_SHAPE_ERROR );
            PyObject * date_type = PyUnicode_FromString( "<M8[ns]" );
            PyArray_Descr *descr;
            PyArray_DescrConverter( date_type, &descr );
            Py_XDECREF( date_type );
            
            PyObject * out = PyArray_NewFromDescr( &PyArray_Type, descr, s_shp.m_dims.size(), &s_shp.m_dims[0], NULL, NULL, 0, NULL );
            DateTime * values = static_cast<DateTime *>( PyArray_DATA( ( PyArrayObject * )out ) );
            for( size_t i = 0; i < s_elem.size(); ++i )
                values[i] = s_elem[i].compute_dt();
            
            RETURN( PyObjectPtr::own( out ) );
        }
    }
};

EXPORT_CPPNODE( _np_arg_min_max );

}
