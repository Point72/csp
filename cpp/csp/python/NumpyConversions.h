#ifndef _IN_CSP_PYTHON_NUMPY_CONVERSIONS_H
#define _IN_CSP_PYTHON_NUMPY_CONVERSIONS_H

#include <csp/engine/csp_autogen/autogen_types.h>
#include <csp/engine/CspType.h>
#include <csp/python/Conversions.h>

#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>
#undef NO_IMPORT_ARRAY

namespace csp::python
{

template<typename T> struct NPY_TYPE { static const int value = NPY_OBJECT; };
template<> struct NPY_TYPE<bool>     { static const int value = NPY_BOOL; };
template<> struct NPY_TYPE<int8_t>   { static const int value = NPY_BYTE; };
template<> struct NPY_TYPE<uint8_t>  { static const int value = NPY_UBYTE; };
template<> struct NPY_TYPE<int16_t>  { static const int value = NPY_SHORT; };
template<> struct NPY_TYPE<uint16_t> { static const int value = NPY_USHORT; };
template<> struct NPY_TYPE<int32_t>  { static const int value = NPY_LONG; };
template<> struct NPY_TYPE<uint32_t> { static const int value = NPY_ULONG; };
template<> struct NPY_TYPE<int64_t>  { static const int value = NPY_LONGLONG; };
template<> struct NPY_TYPE<uint64_t> { static const int value = NPY_ULONGLONG; };
template<> struct NPY_TYPE<double>   { static const int value = NPY_DOUBLE; };
template<> struct NPY_TYPE<std::string>   { static const int value = NPY_UNICODE; };

template<typename T> struct is_native  { static constexpr bool value = false; };
template<> struct is_native<bool>      { static constexpr bool value = true; };
template<> struct is_native<int8_t>    { static constexpr bool value = true; };
template<> struct is_native<uint8_t>   { static constexpr bool value = true; };
template<> struct is_native<int16_t>   { static constexpr bool value = true; };
template<> struct is_native<uint16_t>  { static constexpr bool value = true; };
template<> struct is_native<int32_t>   { static constexpr bool value = true; };
template<> struct is_native<uint32_t>  { static constexpr bool value = true; };
template<> struct is_native<int64_t>   { static constexpr bool value = true; };
template<> struct is_native<uint64_t>  { static constexpr bool value = true; };
template<> struct is_native<double>    { static constexpr bool value = true; };
template<> struct is_native<DateTime>  { static constexpr bool value = true; };
template<> struct is_native<TimeDelta> { static constexpr bool value = true; };

static PyArray_Descr * datetime_descr = nullptr;
static PyArray_Descr * timedelta_descr = nullptr;

inline PyObject * empty_array( const int value )
{
    npy_intp dims[]{ ( npy_intp ) 0 };
    return PyArray_SimpleNew( 1, dims, value );
}

template<typename T>
inline T * getValues( const TickBuffer<T> * tickBuffer, const T & lastValue, int32_t startIndex, int32_t endIndex, int32_t * len, bool tailPadding )
{
    T * values = nullptr;
    if( tickBuffer )
    {
        values = tickBuffer -> flatten( startIndex, endIndex, tailPadding );
        if( tailPadding )
        {
            *len += 1;
            values[ *len - 1 ] = values[ *len - 2 ];
        }
    }
    else
    {
        *len = ( tailPadding ? 2 : 1 );
        values = ( T * ) malloc( sizeof( T ) * ( *len ) );
        values[0] = lastValue;
        if( tailPadding )
            values[1] = lastValue;
    }
    return values;
}

template<typename T, std::enable_if_t<is_native<T>::value, bool> = true>
inline PyObject * as_nparray( const csp::TimeSeriesProvider * ts, const TickBuffer<T> * tickBuffer, const T & lastValue,
                              int32_t startIndex, int32_t endIndex, bool tailPadding )
{
    int32_t len = startIndex - endIndex + 1;
    if( len <= 0 || !ts -> valid() || ( !tickBuffer && endIndex != 0 ) )
        return empty_array( NPY_TYPE<T>::value );

    T * values = getValues( tickBuffer, lastValue, startIndex, endIndex, &len, tailPadding );
    npy_intp dims[]{ ( npy_intp ) len };
    auto array = ( PyArrayObject * ) PyArray_SimpleNewFromData( 1, dims, NPY_TYPE<T>::value, values );
    PyArray_ENABLEFLAGS( array, NPY_ARRAY_OWNDATA);

    return ( PyObject * ) array;
}

inline PyObject * as_nparray( const csp::TimeSeriesProvider * ts, const TickBuffer<DateTime> * tickBuffer, const DateTime & lastValue,
                              int32_t startIndex, int32_t endIndex, bool tailPadding )
{
    int32_t len = startIndex - endIndex + 1;
    if( len <= 0 || !ts -> valid() || ( !tickBuffer && endIndex != 0 ) )
        return empty_array( NPY_TYPE<DateTime>::value );

    DateTime * values = getValues( tickBuffer, lastValue, startIndex, endIndex, &len, tailPadding );
    npy_intp dims[]{ ( npy_intp ) len };

    // init datetime_descr once, since datetimes will be used commonly
    if( !datetime_descr )
    {
        auto date_type = PyPtr<PyObject>::own( PyUnicode_FromString( "<M8[ns]" ) );
        PyArray_DescrConverter( date_type.get(), &datetime_descr );
    }
    // PyArray_NewFromDescr steals a reference
    Py_INCREF( datetime_descr );
    auto * array = PyArray_NewFromDescr( &PyArray_Type, datetime_descr, 1, dims, nullptr, values, 0, nullptr );
    PyArray_ENABLEFLAGS( ( PyArrayObject * ) array, NPY_ARRAY_OWNDATA);
    return array;
}

inline PyObject * as_nparray( const csp::TimeSeriesProvider * ts, const TickBuffer<TimeDelta> * tickBuffer, const TimeDelta & lastValue,
                              int32_t startIndex, int32_t endIndex, bool tailPadding )
{
    int32_t len = startIndex - endIndex + 1;
    if( len <= 0 || !ts -> valid()|| ( !tickBuffer && endIndex != 0 ) )
        return empty_array( NPY_TYPE<TimeDelta>::value );

    TimeDelta * values = getValues( tickBuffer, lastValue, startIndex, endIndex, &len, tailPadding );
    npy_intp dims[]{ ( npy_intp ) len };

    // init timedelta_descr once
    if( !timedelta_descr )
    {
        auto timedelta_type = PyPtr<PyObject>::own( PyUnicode_FromString( "<m8[ns]" ) );
        PyArray_DescrConverter( timedelta_type.get(), &timedelta_descr );
    }
    Py_INCREF( timedelta_descr );
    auto * array = PyArray_NewFromDescr( &PyArray_Type, timedelta_descr, 1, dims, nullptr, values, 0, nullptr );
    PyArray_ENABLEFLAGS( ( PyArrayObject * ) array, NPY_ARRAY_OWNDATA);
    return array;
}

template<typename T, std::enable_if_t<!is_native<T>::value, bool> = true>
inline PyObject * as_nparray( const csp::TimeSeriesProvider * ts, const TickBuffer<T> * tickBuffer, const T & lastValue,
                              int32_t startIndex, int32_t endIndex, bool tailPadding )
{
    int32_t len = startIndex - endIndex + 1;
    if( len <= 0 || !ts -> valid() || ( !tickBuffer && endIndex != 0 ) )
        return empty_array( NPY_OBJECT );

    if( !tickBuffer )
    {
        len = 1;
        startIndex = endIndex;
    }

    if( tailPadding )
        len += 1;

    npy_intp dims[]{ ( npy_intp ) len };
    auto array = PyPtr<PyArrayObject>::own( ( PyArrayObject * ) PyArray_SimpleNew( 1, dims, NPY_OBJECT ) );
    PyObject ** data = ( PyObject ** ) PyArray_DATA( array.get() );
    for( int i = startIndex; i >= endIndex; --i )
        data[ startIndex - i ] = toPython( ts -> valueAtIndex<T>( i ), *ts -> type() );

    if( tailPadding )
    {
        data[ len - 1 ] = data[ len - 2 ];
        Py_INCREF(  data[ len - 1 ] );
    }

    return ( PyObject * ) array.release();
}

inline PyObject * adjustStartAndEndTime( PyObject * arrayObj, autogen::TimeIndexPolicy startPolicy,
                                         autogen::TimeIndexPolicy endPolicy, DateTime startDt, DateTime endDt )
{
    bool extrapolateStart = startPolicy == autogen::TimeIndexPolicy::EXTRAPOLATE;
    bool extrapolateEnd = endPolicy == autogen::TimeIndexPolicy::EXTRAPOLATE;
    if( !extrapolateStart && !extrapolateEnd )
        return arrayObj;

    auto array = ( PyArrayObject * ) arrayObj;
    auto len = PyArray_DIM( array, 0 );
    if( len > 0 )
    {
        auto * data = ( DateTime * ) PyArray_DATA( array );
        if( extrapolateStart && data[ 0 ] < startDt )
            data[ 0 ] = startDt;

        if( extrapolateEnd && data[ len - 1 ] < endDt )
            data[ len - 1 ] = endDt;
    }

    return ( PyObject * ) array;
}

template<typename T>
inline PyObject * createNumpyArray( ValueType valueType, const csp::TimeSeriesProvider * ts, int32_t startIndex, int32_t endIndex,
                                    autogen::TimeIndexPolicy startPolicy, autogen::TimeIndexPolicy endPolicy,
                                    DateTime startDt, DateTime endDt )
{
    bool extrapolateEnd = endPolicy == autogen::TimeIndexPolicy::EXTRAPOLATE &&
                          endIndex < ts -> numTicks() &&
                          ts -> timeAtIndex( endIndex ) < endDt;

    //MSVC bug!! They dont eval ternary operators correctly and this code actually evaluates both sides
    //which leads to a crash since lastValueTyped should not be called at all if ts is not valid...
    //T lastValue = ( ts -> valid() ? ts -> lastValueTyped<T>() : T() );

    T lastValue;
    if( ts -> valid() )
        lastValue = ts -> lastValueTyped<T>();

    DateTime lastTime = ( ts -> valid() ? ts -> lastTime() : DateTime() );
    switch( valueType )
    {
        case ValueType::VALUE:
            return as_nparray( ts, ts -> dataline<T>(), lastValue, startIndex, endIndex, extrapolateEnd );

        case ValueType::TIMESTAMP:
        {
            return adjustStartAndEndTime( as_nparray( ts, ts -> timeline(), lastTime, startIndex, endIndex, extrapolateEnd ),
                                          startPolicy, endPolicy, startDt, endDt );
        }
        case ValueType::TIMESTAMP_VALUE_TUPLE:
        {
            PyObject * tuple = PyTuple_New( 2 );
            PyTuple_SET_ITEM( tuple, 0, adjustStartAndEndTime( as_nparray( ts, ts -> timeline(), lastTime, startIndex,
                                        endIndex, extrapolateEnd ), startPolicy, endPolicy, startDt, endDt ) );
            PyTuple_SET_ITEM( tuple, 1, as_nparray( ts, ts -> dataline<T>(), lastValue, startIndex, endIndex, extrapolateEnd ) );
            return tuple;
        }
    }
    return nullptr;
}

PyObject * valuesAtIndexToNumpy( ValueType valueType, const csp::TimeSeriesProvider * ts, int32_t startIndex, int32_t endIndex,
                                 autogen::TimeIndexPolicy startPolicy, autogen::TimeIndexPolicy endPolicy,
                                 DateTime startDt = DateTime::NONE(), DateTime endDt = DateTime::NONE() );

int64_t scalingFromNumpyDtUnit( NPY_DATETIMEUNIT base );
NPY_DATETIMEUNIT datetimeUnitFromDescr( PyArray_Descr* descr );

// for getting strings from elems of numpy arrays of strings
void stringFromNumpyStr( void* data, std::string& out, char numpy_type, int elem_size_bytes );

void validateNumpyTypeVsCspType( const CspTypePtr & type, char numpy_type_char );

}

#endif
