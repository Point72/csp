#ifndef _IN_CSP_PYTHON_NUMPYINPUTADAPTER_H
#define _IN_CSP_PYTHON_NUMPYINPUTADAPTER_H

#include <csp/engine/PullInputAdapter.h>
#include <csp/python/NumpyConversions.h>

#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>
#undef NO_IMPORT_ARRAY

namespace csp::python
{

class NumpyCurveAccessor
{
    // Accesses values by indexing only on the outermost dimension of an n-dimensional Numpy array and then creating a view to the subarray
    // Thus, you can get the nested Numpy array values in csp.curve with zero-copy operations

public:
    NumpyCurveAccessor()
    {
        m_nd = 0;
        m_data = nullptr;
        m_outerStride = 0;
        m_outerDim = 0;
        m_innerStrides = nullptr;
        m_innerDims = nullptr;
        m_arr = nullptr;
        m_descr = nullptr;
    }

    NumpyCurveAccessor( PyArrayObject * arr )
    {
        m_nd = PyArray_NDIM( arr );
        if( m_nd < 2 )
            CSP_THROW( csp::TypeError, "NumpyCurveAccessor is inefficient for a 1-D Numpy array: use PyArray_GETPTR1 to access indexed values" );

        // Preprocess strides and dimensions
        npy_intp* strides = PyArray_STRIDES( arr );
        npy_intp* dims = PyArray_DIMS( arr );
        m_outerStride = strides[0];
        m_outerDim = dims[0];
        m_innerStrides = strides + 1;
        m_innerDims = dims + 1;

        m_arr = arr;
        Py_XINCREF( m_arr );
        m_descr = PyArray_DESCR( m_arr );
        m_data = reinterpret_cast<char*>( PyArray_DATA( arr ) );
    }

    ~NumpyCurveAccessor()
    {
        Py_XDECREF( m_arr );
    }

    PyObject * data( int64_t index ) const
    {
        if( index >= m_outerDim )
            CSP_THROW( csp::TypeError, "Requested data index out of range in NumpyCurveAccessor" );

        // Create a view to the (n-1) dimensional array with (n-1) potentially unnatural strides
        /*
        A note on reference counting for the subarray: NewFromDescr will *steal* a reference to the type descr,
        so we need to independently incref m_descr before passing it to NewFromDescr. Furthermore, SetBaseObject
        steals a reference from m_arr to inner_value, so we need to incref m_arr before passing it to SetBaseObject.
        If the order is not as such, deallocation may occur prematurely. After fromPython is called in NumpyInputAdapter,
        the ref count of inner_value will be decremented and ownership is transferred to the NumpyInputAdapter.
        */

        Py_XINCREF( m_descr );
        PyArrayObject * inner_value = ( PyArrayObject * ) PyArray_NewFromDescr(
            &PyArray_Type,
            m_descr,
            m_nd - 1,
            m_innerDims,
            m_innerStrides,
            m_data + index * m_outerStride,
            PyArray_FLAGS( m_arr ),
            NULL
        );
        Py_XINCREF( m_arr );
        PyArray_SetBaseObject( inner_value, ( PyObject * )m_arr );
        return ( PyObject * )inner_value;
    }

private:
    char* m_data;
    int m_nd;

    npy_intp m_outerStride;
    npy_intp m_outerDim;
    npy_intp* m_innerStrides;
    npy_intp* m_innerDims;

    PyArrayObject * m_arr;
    PyArray_Descr * m_descr;
};

template<typename T>
class NumpyInputAdapter : public PullInputAdapter<T>
{
    using PyArrayObjectPtr = PyPtr<PyArrayObject>;

public:
    NumpyInputAdapter( Engine * engine, CspTypePtr & type, PyArrayObject * datetimes,
                       PyArrayObject * values ) : PullInputAdapter<T>( engine, type, PushMode::LAST_VALUE ),
                                                  m_datetimes( PyArrayObjectPtr::incref( datetimes ) ),
                                                  m_values( PyArrayObjectPtr::incref( values ) ),
                                                  m_val_scale( 0 ), m_index( 0 )
    {
        PyArray_Descr* dts_descr = PyArray_DESCR(m_datetimes.ptr());
        PyArray_Descr* vals_descr = PyArray_DESCR(m_values.ptr());

        m_size = static_cast<int>(PyArray_SIZE( datetimes ));
        m_elem_size = PyDataType_ELSIZE(vals_descr);
        m_val_type = vals_descr -> type;

        char out_type = m_val_type;
        if( PyArray_NDIM( m_values.ptr() ) > 1 )
        {
            out_type = NPY_OBJECTLTR;
            m_valueAccessor = std::make_unique<NumpyCurveAccessor>( m_values.ptr() );
        }
        validateNumpyTypeVsCspType( type, out_type );


        auto dt_type = dts_descr -> type;
        if( dt_type != NPY_DATETIMELTR && dt_type != NPY_OBJECTLTR )
            CSP_THROW( ValueError, "timestamps ndarray must be dtype of datetime64 or object, got type code of " << dt_type );

        if( dt_type == NPY_DATETIMELTR )
        {
            auto base = csp::python::datetimeUnitFromDescr( dts_descr );
            m_dt_scale = csp::python::scalingFromNumpyDtUnit( base );
        }
        else
            m_dt_scale = 0; // take 0 to mean its python datetime object not numpy datetime

        // if the values are datetimes or timedeltas, they have a resolution to account for as well
        if( m_val_type == NPY_DATETIMELTR || m_val_type == NPY_TIMEDELTALTR )
        {
            auto val_base = csp::python::datetimeUnitFromDescr( vals_descr );
            m_val_scale = csp::python::scalingFromNumpyDtUnit( val_base );
        }
    }

    void start( DateTime start, DateTime end ) override
    {
        while( m_index < m_size )
        {
            DateTime t;
            if (m_dt_scale == 0)
            {
                auto data = PyArray_GETPTR1(m_datetimes.ptr(), m_index);
                t = fromPython<csp::DateTime>( *( PyObject ** ) data );
            }
            else
            {
                int64_t numpy_dt = *(static_cast<int64_t*>(PyArray_GETPTR1(m_datetimes.ptr(), m_index)));
                t = DateTime::fromNanoseconds( numpy_dt * m_dt_scale );
            }

            if( t >= start )
                break;

            ++m_index;
        }

        PullInputAdapter<T>::start( start, end );
    }

    bool next( DateTime & t, T & value ) override
    {
        if( m_index >= m_size )
            return false;

        if (m_dt_scale == 0)
        {
            auto data = PyArray_GETPTR1(m_datetimes.ptr(), m_index);
            t = fromPython<csp::DateTime>( *( PyObject ** ) data );
        }
        else
        {
            int64_t numpy_dt = *(static_cast<int64_t*>(PyArray_GETPTR1(m_datetimes.ptr(), m_index)));
            t = DateTime::fromNanoseconds( numpy_dt * m_dt_scale );
        }

        if( m_valueAccessor )
        {
            PyObject * subarray = m_valueAccessor -> data( m_index );
            value = fromPython<T>( subarray, *this -> type() );
            Py_XDECREF( subarray );
        }
        else
        {
            auto data = PyArray_GETPTR1( m_values.ptr(), m_index );
            if( m_val_type == NPY_OBJECTLTR )
                value = fromPython<T>( *( PyObject ** ) data, *this -> type() );
            else
                setValue( value, data );
        }

        ++m_index;
        return true;
    }

    void setValue( T & value, void * data )
    {
        value = *((T*)data);
    }

private:
    PyArrayObjectPtr   m_datetimes;
    PyArrayObjectPtr   m_values;
    int64_t            m_dt_scale;  // nanos per unit (so 1e9 for seconds, etc) scale for timestamps
    int64_t            m_val_scale; // and for values that need scaling (datetime64/timedelta64)
    uint32_t           m_index;
    uint32_t           m_size;
    uint32_t           m_elem_size; // in bytes
    char               m_val_type;

    std::unique_ptr<NumpyCurveAccessor> m_valueAccessor; // required for nested arrays
};

template<>
void NumpyInputAdapter<std::string>::setValue( std::string & value, void * data )
{
    csp::python::stringFromNumpyStr( data, value, m_val_type, m_elem_size );
}

template<>
void NumpyInputAdapter<csp::DateTime>::setValue( csp::DateTime & value, void * data )
{
    int64_t dt_val = *( static_cast<int64_t*>(data) );
    value = DateTime::fromNanoseconds( m_val_scale * dt_val );
}

template<>
void NumpyInputAdapter<csp::TimeDelta>::setValue( csp::TimeDelta & value, void * data )
{
    int64_t dt_val = *( static_cast<int64_t*>(data) );
    value = TimeDelta::fromNanoseconds( m_val_scale * dt_val );
}

template<>
void NumpyInputAdapter<int64_t>::setValue( int64_t & value, void * data )
{
    switch( m_val_type )
    {
        case NPY_BYTELTR:
        {
            const char * const val = (const char *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_UBYTELTR:
        {
            const unsigned char * const val = (const unsigned char *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_SHORTLTR:
        {
            const short * const val = (const short *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_USHORTLTR:
        {
            const unsigned short * const val = (const unsigned short *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_INTLTR:
        {
            const int * const val = (const int *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_UINTLTR:
        {
            const unsigned int * const val = (const unsigned int *) data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_LONGLTR:
        {
            const long * const val = (const long*)data;
            value = static_cast<int64_t>(*val);
            break;
        }
        case NPY_LONGLONGLTR:
        {
            const long long * const val = (const long long*)data;
            value = static_cast<int64_t>(*val);
            break;
        }
        default:
            CSP_THROW( ValueError, "NumpyInputAdapter<int64_t>::setValue sees invalid numpy type " << m_val_type );
    }
}

template<>
void NumpyInputAdapter<double>::setValue( double & value, void * data )
{
    switch( m_val_type )
    {
        case NPY_FLOATLTR:
        {
            const float * const val = (const float *) data;
            value = static_cast<double>(*val);
            break;
        }
        case NPY_DOUBLELTR:
        {
            value = *(double *)(data);
            break;
        }
        default:
            CSP_THROW( ValueError, "NumpyInputAdapter<double>::setValue sees invalid numpy type " << m_val_type );
    }
};



};

#endif
