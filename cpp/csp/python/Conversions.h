#ifndef _IN_CSP_PYTHON_CONVERSIONS_H
#define _IN_CSP_PYTHON_CONVERSIONS_H

#include <csp/core/Platform.h>
#include <csp/core/Time.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <csp/python/Common.h>
#include <csp/python/CspTypeFactory.h>
#include <csp/python/Exception.h>
#include <csp/python/PyCspEnum.h>
#include <csp/python/PyCspType.h>
#include <csp/python/PyObjectPtr.h>
#include <csp/python/PyStruct.h>
#include <csp/python/PyStructFastList.h>
#include <csp/python/PyStructList.h>
#include <Python.h>
#include <datetime.h>
#include <string>
#include <variant>

namespace csp::python
{

enum ValueType : uint8_t {
    VALUE,
    TIMESTAMP,
    TIMESTAMP_VALUE_TUPLE
};

struct DateTimeOrTimeDelta : public std::variant<DateTime,TimeDelta>
{
    using BaseT = std::variant<DateTime,TimeDelta>;
    using BaseT::BaseT;

    bool isDateTime() const  { return std::holds_alternative<DateTime>( *this ); }
    bool isTimeDelta() const { return std::holds_alternative<TimeDelta>( *this ); }

    DateTime  datetime() const  { return std::get<DateTime>( *this ); }
    TimeDelta timedelta() const { return std::get<TimeDelta>( *this ); }
};

inline csp::CspTypePtr & pyTypeAsCspType( PyObject * pyType )
{
    return CspTypeFactory::instance().typeFromPyType( pyType );
}

//for exception formatting
inline std::string pyTypeToString( PyObject * pyType )
{
    if( PyType_Check( pyType ) )
        return ( ( PyTypeObject * ) pyType ) -> tp_name;
    else if( PyList_Check( pyType ) )
    {
        char buf[128];
        snprintf( buf, sizeof( buf ), "[ %s ]", ( ( PyTypeObject * ) PyList_GET_ITEM( pyType, 0 ) ) -> tp_name );
        return buf;
    }
    else
        return "<unknown>";
}

template< typename F>
inline auto switchPyType( PyObject * pyType, F && f )
{
    return switchCspType( pyTypeAsCspType( pyType ), f);
}

inline bool validatePyType( const CspType * type, PyObject *pyType, PyObject * value )
{
    //helper method to ensure that dialect generic types pushed from user circuits are aligned with expected types
    //we only check dialect generic because other well known types will fail in fromPython conversion code
    return type -> type() != CspType::Type::DIALECT_GENERIC ||
        PyType_IsSubtype( Py_TYPE( value ), ( PyTypeObject * ) pyType );
}

template<typename T>
inline T fromPython( PyObject * o )
{
    static_assert( !std::is_same<T,T>::value, "no fromPython method implemented for type" );
    return T{};
}

//work around for inabilty to partially specialize fromPython on vector<T>
template<typename T>
struct FromPython
{
    static T impl( PyObject * o, const CspType & type )
    {
        return fromPython<T>( o );
    }
};

template<typename T>
inline T fromPython( PyObject * o, const CspType & type )
{
    return FromPython<T>::impl( o, type );
}

template<typename T>
PyObject * toPython( const T & value );

// Enforce that all types specialized
template<typename T>
PyObject * toPython( const T & value )
{
    static_assert( !std::is_same<T,T>::value, "no toPython method implemented for type" );
    return nullptr;
}

template<typename T>
inline PyObject * toPython( const T & value, const CspType & type )
{
    return toPython<T>( value );
}

inline PyObject * toPythonCheck( PyObject * res )
{
    if( !res )
        CSP_THROW( PythonPassthrough, "" );
    return res;
}

//PyObjectPtr
template<>
inline PyObjectPtr fromPython( PyObject * o )
{
    return PyObjectPtr::incref( o );
}

template<>
inline PyObject * toPython( const PyObjectPtr & value )
{
    PyObject * rv = value.ptr();
    Py_XINCREF( rv );
    return rv;
}

template<>
inline DialectGenericType fromPython( PyObject * o )
{
    return reinterpret_cast<const DialectGenericType &&>(std::move(fromPython<PyObjectPtr>( o )));
}

inline PyObject * toPythonBorrowed( const DialectGenericType & value)
{
    return reinterpret_cast<const PyObjectPtr &>(value).ptr();
}

template<>
inline PyObject * toPython( const DialectGenericType & value)
{
    return toPython( reinterpret_cast<const PyObjectPtr &>(value));
}

template<>
inline PyObject * toPython( PyObject* const& value )
{
    Py_XINCREF( value );
    return value;
}

//bool
template<>
inline bool fromPython( PyObject * o )
{
    if( !PyBool_Check( o ) )
        CSP_THROW( TypeError, "Invalid bool type, expected bool got " << Py_TYPE( o ) -> tp_name );

    return o == Py_True;
}

template<>
inline PyObject * toPython( const bool & value )
{
    auto * rv = value ? Py_True : Py_False;
    Py_INCREF( rv );
    return rv;
}


//double
template<>
inline double fromPython( PyObject * o )
{
    if( !PyFloat_Check( o ) )
    {
        //allow ints as floats
        if( PyLong_Check( o ) )
        {
            int64_t rv = PyLong_AsLongLong( o );
            if( rv == -1 && PyErr_Occurred() )
                CSP_THROW( PythonPassthrough, "" );
            return rv;
        }

        CSP_THROW( TypeError, "Invalid float type, expected float got " << Py_TYPE( o ) -> tp_name );
    }

    return PyFloat_AS_DOUBLE( o );
}

template<>
inline PyObject * toPython( const double & value )
{
    return toPythonCheck( PyFloat_FromDouble( value ) );
}

//int64_t
template<>
inline PyObject * toPython( const int64_t & value )
{
    return toPythonCheck( PyLong_FromLongLong( value ) );
}

template<>
inline int64_t fromPython( PyObject * o )
{
    if( !PyLong_Check( o ) )
        CSP_THROW( TypeError, "Invalid int type, expected long (int) got " << Py_TYPE( o ) -> tp_name );

    int64_t rv = PyLong_AsLongLong( o );
    if( rv == -1 && PyErr_Occurred() )
        CSP_THROW( PythonPassthrough, "" );
    return rv;
}


//uint64_t
template<>
inline PyObject * toPython( const uint64_t & value )
{
    return toPythonCheck( PyLong_FromUnsignedLongLong( value ) );
}

template<>
inline uint64_t fromPython( PyObject * o )
{
    if( !PyLong_Check( o ) )
        CSP_THROW( TypeError, "Invalid int type, expected long (int) got " << Py_TYPE( o ) -> tp_name );

    uint64_t rv = PyLong_AsUnsignedLongLong( o );
    if( rv == ( uint64_t ) -1 && PyErr_Occurred() )
        CSP_THROW( PythonPassthrough, "" );
    return rv;
}


//int32_t
template<>
inline PyObject * toPython( const int32_t & value )
{
    return toPython<int64_t>( value );
}

template<>
inline int32_t fromPython( PyObject * o )
{
    auto rv = fromPython<int64_t>( o );
    if( rv > std::numeric_limits<int32_t>::max() || rv < std::numeric_limits<int32_t>::min() )
        CSP_THROW( OverflowError, rv << " is too big/small to fit in int32" );
    return ( int32_t ) rv;
}


//uint32_t
template<>
inline PyObject * toPython( const uint32_t & value )
{
    return toPython<uint64_t>( value );
}

template<>
inline uint32_t fromPython( PyObject * o )
{
    auto rv = fromPython<uint64_t>( o );
    if( rv > std::numeric_limits<uint32_t>::max() )
        CSP_THROW( OverflowError, rv << " is too big to fit in uint32" );
    return ( uint32_t ) rv;
}

//int16_t
template<>
inline PyObject * toPython( const int16_t & value )
{
    return toPython<int64_t>( value );
}

template<>
inline int16_t fromPython( PyObject * o )
{
    auto rv = fromPython<int64_t>( o );
    if( rv > std::numeric_limits<int16_t>::max() || rv < std::numeric_limits<int16_t>::min() )
        CSP_THROW( OverflowError, rv << " is too big/small to fit in int16" );
    return ( int16_t ) rv;
}


//uint16_t
template<>
inline PyObject * toPython( const uint16_t & value )
{
    return toPython<uint64_t>( value );
}

template<>
inline uint16_t fromPython( PyObject * o )
{
    auto rv = fromPython<uint64_t>( o );
    if( rv > std::numeric_limits<uint16_t>::max() )
        CSP_THROW( OverflowError, rv << " is too big to fit in uint16" );
    return ( uint16_t ) rv;
}

//int8_t
template<>
inline PyObject * toPython( const int8_t & value )
{
    return toPython<int64_t>( value );
}

template<>
inline int8_t fromPython( PyObject * o )
{
    auto rv = fromPython<int64_t>( o );
    if( rv > std::numeric_limits<int8_t>::max() || rv < std::numeric_limits<int8_t>::min() )
        CSP_THROW( OverflowError, rv << " is too big/small to fit in int8" );
    return ( int8_t ) rv;
}


//uint8_t
template<>
inline PyObject * toPython( const uint8_t & value )
{
    return toPython<uint64_t>( value );
}

template<>
inline uint8_t fromPython( PyObject * o )
{
    auto rv = fromPython<uint64_t>( o );
    if( rv > std::numeric_limits<uint8_t>::max() )
        CSP_THROW( OverflowError, rv << " is too big to fit in uint8" );
    return ( uint8_t ) rv;
}

//String
template<>
inline PyObject * toPython( const csp::CspType::StringCType & value )
{
    return toPythonCheck( PyUnicode_FromStringAndSize( value.c_str(), value.size() ) );
}

template<>
inline PyObject * toPython( const csp::CspType::StringCType & value, const CspType & type )
{
    assert( type.type() == CspType::Type::STRING );
    const CspStringType& strType = static_cast<const CspStringType&>(type);
    if(strType.isBytes())
    {
        return toPythonCheck( PyBytes_FromStringAndSize( value.c_str(), value.size() ) );
    }
    else
    {
        return toPython(value);
    }
}



template<>
inline csp::CspType::StringCType fromPython( PyObject * o )
{
    if( PyUnicode_Check( o ) )
    {
        Py_ssize_t len;
        const char * s = PyUnicode_AsUTF8AndSize( o, &len );
        if( !s )
            CSP_THROW( PythonPassthrough, "" );

        return csp::CspType::StringCType( s, len );
    }
    else if(PyBytes_Check( o ))
    {
        const char * s = PyBytes_AsString( o );
        if( !s )
            CSP_THROW( PythonPassthrough, "" );

        return csp::CspType::StringCType( s, PyBytes_Size(o) );
    }
    else
    {
        CSP_THROW( TypeError, "Invalid string type, expected str got " << Py_TYPE( o ) -> tp_name );
    }
}

//Struct
template<>
inline PyObject * toPython( const StructPtr & s )
{
    if( s -> dialectPtr() )
    {
        PyObject * pystruct = reinterpret_cast<PyObject*>( s -> dialectPtr() );
        Py_INCREF( pystruct );
        return pystruct;
    }

    const auto * dmeta = static_cast<const DialectStructMeta*>( s -> meta() );
    PyObject * pystruct = dmeta -> pyType() -> tp_alloc( dmeta -> pyType(), 0 );;
    new ( pystruct ) PyStruct( s );

    //assign dialectptr, but we DO NOT incref the instance on the struct
    const_cast<StructPtr &>( s ) -> setDialectPtr( pystruct );
    return toPythonCheck( pystruct );
}

template<>
inline StructPtr fromPython( PyObject * o, const CspType & type )
{
    assert( type.type() == CspType::Type::STRUCT );

    if( !PyType_IsSubtype( Py_TYPE( o ), &PyStruct::PyType ) ||
        ( static_cast<const CspStructType &>( type ).meta() && //could be csp.Struct as a type annotation which is allowed
          !StructMeta::isDerivedType( static_cast<PyStruct *>( o ) -> structMeta(),
                                      static_cast<const CspStructType &>( type ).meta().get() ) ) )
    {
        std::string name;
        auto meta = static_cast<const CspStructType &>( type ).meta();
        if( meta )
            name = " " + meta -> name();
        CSP_THROW( TypeError, "Invalid struct type, expected struct" << name << " got " << Py_TYPE( o ) -> tp_name );
    }

    return static_cast<PyStruct *>( o ) -> struct_;
}

//CspEnum
template<>
inline PyObject * toPython( const CspEnum & e, const CspType & type )
{
    assert( type.type() == CspType::Type::ENUM );

    auto & enumType = static_cast<const CspEnumType&>( type );
    const auto * emeta = static_cast<const DialectCspEnumMeta*>( enumType.meta().get() );

    PyObject * obj = emeta -> pyMeta() -> toPyEnum( e );
    if( unlikely( !obj ) )
        CSP_THROW( ValueError, e.value() << " is not a valid value on csp.enum type " << emeta -> name() );
    return obj;
}

template<>
inline CspEnum fromPython( PyObject * o, const CspType & type )
{
    assert( type.type() == CspType::Type::ENUM );

    if( !PyType_IsSubtype( Py_TYPE( o ), &PyCspEnum::PyType ) ||
        static_cast<PyCspEnum *>( o ) -> meta() != static_cast<const CspEnumType &>( type ).meta().get() )
        CSP_THROW( TypeError, "Invalid enum type, expected enum type " << static_cast<const CspEnumType &>( type ).meta() -> name() << " got " << Py_TYPE( o ) -> tp_name );

    return static_cast<PyCspEnum *>( o ) -> enum_;
}

//TimeDelta
template<>
inline PyObject * toPython( const TimeDelta & td )
{
    INIT_PYDATETIME;

    if( td == TimeDelta::NONE() )
        Py_RETURN_NONE;

    return toPythonCheck( PyDelta_FromDSU( 0, td.asSeconds(), td.nanoseconds() / 1000 ) );
}

template<>
inline TimeDelta fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( o == Py_None )
        return TimeDelta::NONE();

    if( !PyDelta_Check( o ) )
        CSP_THROW( TypeError, "Invalid timedelta type, expected timedelta got " << Py_TYPE( o ) -> tp_name );

    static const int32_t MIN_DAYS = TimeDelta::MIN_VALUE().days();
    static const int32_t MAX_DAYS = TimeDelta::MAX_VALUE().days();

    PyDateTime_Delta * d = (PyDateTime_Delta * ) o;

    //approximate check
    if( d -> days > MAX_DAYS || d -> days < MIN_DAYS )
    {
        CSP_THROW( OverflowError, "timedelta " << PyObjectPtr::incref( o ) << " out of range for csp timedelta" );
    }

    int64_t seconds = d -> days * 86400LL + d -> seconds;
    return TimeDelta( seconds, d -> microseconds * 1000 );
}

//Date
template<>
inline PyObject * toPython( const Date & d )
{
    INIT_PYDATETIME;

    if ( d == Date::NONE())
        Py_RETURN_NONE;

    return toPythonCheck( PyDate_FromDate( d.year(), d.month(), d.day()));
}

template<>
inline Date fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( o == Py_None )
        return Date::NONE();

    if( !PyDate_Check( o ) )
        CSP_THROW( TypeError, "Invalid date type, expected date got " << Py_TYPE( o ) -> tp_name );

    PyDateTime_Date * d = (PyDateTime_Date * ) o;
    int year = PyDateTime_GET_YEAR( d );
    int mon  = PyDateTime_GET_MONTH( d );
    int day  = PyDateTime_GET_DAY( d );
    return Date(year, mon, day);
}

//Time
template<>
inline PyObject * toPython( const Time & t )
{
    INIT_PYDATETIME;

    if( t == Time::NONE())
        Py_RETURN_NONE;

    return toPythonCheck( PyTime_FromTime( t.hour(), t.minute(), t.second(), t.nanosecond() / 1000 ) );
}

template<>
inline Time fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( o == Py_None )
        return Time::NONE();

    if( !PyTime_Check( o ) )
        CSP_THROW( TypeError, "Invalid time type, expected time got " << Py_TYPE( o ) -> tp_name );

    PyDateTime_Time * t = (PyDateTime_Time * ) o;
    if( t -> hastzinfo )
        CSP_THROW( TypeError, "csp time type does not support timezones.  Please use ts[object] for timezone time values" );

    int hour = PyDateTime_TIME_GET_HOUR( t );
    int min  = PyDateTime_TIME_GET_MINUTE( t );
    int sec  = PyDateTime_TIME_GET_SECOND( t );
    int usec = PyDateTime_TIME_GET_MICROSECOND( t );
    return Time( hour, min, sec, usec * 1000 );
}

//DateTime
template<>
inline PyObject * toPython( const DateTime & dt )
{
    INIT_PYDATETIME;

    DateTimeEx dtEx( dt );
    return toPythonCheck( PyDateTime_FromDateAndTime( dtEx.year(), dtEx.month(), dtEx.day(), dtEx.hour(), dtEx.minute(), dtEx.second(), dtEx.microseconds() ) );
}

template<>
inline DateTime fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( o == Py_None )
        return DateTime::NONE();

    if( !PyDateTime_Check( o ) )
        CSP_THROW( TypeError, "Invalid datetime type, expected datetime got " << Py_TYPE( o ) -> tp_name );

    PyDateTime_DateTime * pydt = ( PyDateTime_DateTime * ) o;
    int year = PyDateTime_GET_YEAR( pydt );
    int mon  = PyDateTime_GET_MONTH( pydt );
    int day  = PyDateTime_GET_DAY( pydt );
    int hour = PyDateTime_DATE_GET_HOUR( pydt );
    int minute = PyDateTime_DATE_GET_MINUTE( pydt );
    int second = PyDateTime_DATE_GET_SECOND( pydt );
    int nanosecond = PyDateTime_DATE_GET_MICROSECOND( pydt ) * 1000;

    //might be pandas.Timestamp with lovely nanos
    if( !PyDateTime_CheckExact( o ) && PyObject_HasAttrString( o, "nanosecond" ) )
    {
        int64_t ns = fromPython<int64_t>( PyObjectPtr::own( PyObject_GetAttrString( o, "nanosecond" ) ).ptr() );
        nanosecond += ns;
    }

    static const DateTimeEx MIN_DATE = DateTimeEx( DateTime::MIN_VALUE() );
    static const DateTimeEx MAX_DATE = DateTimeEx( DateTime::MAX_VALUE() );

    //approximate check for simplicity
    if( year < MIN_DATE.year() + 1 || year > MAX_DATE.year() - 1 )
    {
        CSP_THROW( OverflowError, "datetime " << PyObjectPtr::incref( o ) << " is out of range for csp datetime" );
    }

    DateTime dt( year, mon, day, hour, minute, second, nanosecond );

    if( pydt -> hastzinfo )
    {
        PyObjectPtr utcoffsetStr = PyObjectPtr::own( PyUnicode_FromString( "utcoffset" ) );
        PyObjectPtr utcoffset = PyObjectPtr::own( PyObject_CallMethodObjArgs( pydt -> tzinfo, utcoffsetStr.ptr(), o, nullptr ) );
        auto tzOffset = fromPython<TimeDelta>( utcoffset.ptr() );
        dt -= tzOffset;
    }

    return dt;
}

template<>
inline DateTimeOrTimeDelta fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( PyDateTime_Check( o ) )
        return fromPython<DateTime>( o );

    if( PyDelta_Check( o ) )
        return fromPython<TimeDelta>( o );

    CSP_THROW( TypeError, "Invalid type, expected datetime or timedelta got " << Py_TYPE( o ) -> tp_name );
}

//Dictionary
template<>
inline Dictionary::Value fromPython( PyObject * o );

template<>
inline Dictionary fromPython( PyObject * o )
{
    if( !PyDict_Check( o ) )
        CSP_THROW( TypeError, "Dictionary conversion expected type dict got " << Py_TYPE( o ) -> tp_name );

    Dictionary out;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    while( PyDict_Next( o, &pos, &key, &value) )
    {
        if(value == Py_None)
            continue;

        std::string keystr = fromPython<std::string>( key );
        auto v = fromPython<csp::Dictionary::Value>( value );
        out.insert( keystr, std::move( v ) );
    }

    return out;
}

template<>
inline std::vector<Dictionary::Data> fromPython( PyObject * o )
{
    if( !PyList_Check( o ) )
        CSP_THROW( TypeError, "Dictionary conversion expected type list got " << Py_TYPE( o ) -> tp_name );

    std::vector<Dictionary::Data> out;

    Py_ssize_t size = PyList_GET_SIZE( o );
    for( Py_ssize_t i = 0; i < size; ++i )
    {
        PyObject * item = PyList_GET_ITEM( o, i );
        out.emplace_back( fromPython<Dictionary::Value>( item ) );
    }

    return out;
}

template<>
inline Dictionary::Value fromPython( PyObject * o )
{
    INIT_PYDATETIME;

    if( PyBool_Check( o ) )
        return fromPython<bool>( o );
    else if( PyLong_Check( o ) )
        return fromPython<int64_t>( o );
    else if( PyFloat_Check( o ) )
        return fromPython<double>( o );
    else if( PyUnicode_Check( o ) )
        return std::string( PyUnicode_AsUTF8( o ) );
    else if( PyBytes_Check( o ) )
        return std::string( PyBytes_AsString( o ) );
    else if( PyDateTime_Check( o ) )
        return fromPython<DateTime>( o );
    else if( PyDelta_Check( o ) )
        return fromPython<TimeDelta>( o );
    else if( PyDict_Check( o ) )
        return std::make_shared<Dictionary>( fromPython<Dictionary>( o ) );
    else if( PyList_Check( o ) )
        return fromPython<std::vector<Dictionary::Data>>( o );
    else if( PyType_Check( o ) && PyStruct::isPyStructType( ( PyTypeObject * ) o ) )
        return ( ( PyStructMeta * ) o ) -> structMeta;
    else
        return fromPython<DialectGenericType>( o );
}


template<>
PyObject * toPython( const DictionaryPtr & value);

template<>
PyObject * toPython( const Dictionary::Value & value );

template<>
inline PyObject * toPython( const Dictionary::Vector & v )
{
    PyObject * pylist = PyList_New( v.size() );
    for( size_t i = 0; i < v.size(); i++ )
        PyList_SET_ITEM( pylist, i, toPython( v[i]._data ) );

    return pylist;
}

template<>
inline PyObject * toPython( const Dictionary::Value & value )
{
    switch( static_cast<int64_t>( value.index() ) )
    {
        case Dictionary::DictDataType::MONOSTATE:          CSP_THROW( ValueError, "Monostate value is not convertible to a Python type");
        case Dictionary::DictDataType::BOOL:               return toPython( std::get<bool>( value ) );
        case Dictionary::DictDataType::INT32:              return toPython( std::get<int32_t>( value ) );
        case Dictionary::DictDataType::UINT32:             return toPython( std::get<uint32_t>( value ) );
        case Dictionary::DictDataType::INT64:              return toPython( std::get<int64_t>( value ) );
        case Dictionary::DictDataType::UINT64:             return toPython( std::get<uint64_t>( value ) );
        case Dictionary::DictDataType::DOUBLE:             return toPython( std::get<double>( value ) );
        case Dictionary::DictDataType::STRING:             return toPython( std::get<std::string>( value ) );
        case Dictionary::DictDataType::DATETIME:           return toPython( std::get<DateTime>( value ) );
        case Dictionary::DictDataType::TIMEDELTA:          return toPython( std::get<TimeDelta>( value ) );
        case Dictionary::DictDataType::STRUCTMETAPTR:      CSP_THROW( NotImplemented, "StructMetaPtr value is not convertible to a Python type");
        case Dictionary::DictDataType::DIALECTGENERICTYPE: return toPython( std::get<DialectGenericType>( value ) );
        case Dictionary::DictDataType::DICTIONARYPTR:      return toPython( std::get<DictionaryPtr>( value ) );
        case Dictionary::DictDataType::VECTOR:             return toPython( std::get<Dictionary::Vector>( value ) );
        case Dictionary::DictDataType::DATA:               return toPython( std::get<std::shared_ptr<Dictionary::Data>>( value ).get()->_data );
    }

    CSP_THROW( ValueError, "Given dictionary value is not a valid value type." );
}

template<>
inline PyObject * toPython( const DictionaryPtr & value)
{
    // Convert csp::Dictionary object to Python dictionary

    PyObject* pydict = PyDict_New();
    for( Dictionary::const_iterator it = value -> begin(); it != value -> end(); ++it )
    {
        PyObject* val = toPython( it.getUntypedValue() ); // will recurse on nested dictionaries
        PyDict_SetItemString( pydict, it.key().c_str(), PyObjectPtr::own( val ).get() );
    }
    return pydict;
}

template<typename StorageT>
inline PyObject * toPython( const std::vector<StorageT> & v, const CspType & arrayType )
{
    assert( arrayType.type() == CspType::Type::ARRAY );

    const CspType & elemType = *static_cast<const CspArrayType &>( arrayType ).elemType();
    size_t size = v.size();
    PyObjectPtr list = PyObjectPtr::check( PyList_New( size ) );

    for( size_t idx = 0; idx < size; ++idx )
    {
        using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
        PyList_SET_ITEM( list.ptr(), idx, toPython<ElemT>( v[idx], elemType ) );
    }
    return list.release();
}

template<typename StorageT>
inline PyObject * toPython( const std::vector<StorageT> & v, const CspType & arrayType, const PyStruct * pystruct )
{
    assert( arrayType.type() == CspType::Type::ARRAY );

    const CspArrayType & cspArrayType = static_cast<const CspArrayType &>( arrayType );
    const CspTypePtr elemType = cspArrayType.elemType();
    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
    size_t sz = v.size();

    // Create PyStructFastList when requested
    if( cspArrayType.isPyStructFastList() )
    {
        PyObject * fl = PyStructFastList<StorageT>::PyType.tp_alloc( &PyStructFastList<StorageT>::PyType, 0 );
        new ( fl ) PyStructFastList<StorageT>( const_cast<PyStruct *>( pystruct ), const_cast<std::vector<StorageT> &>( v ), cspArrayType );
        return fl;
    }
    // Create PyStructList otherwise
    // TODO: Implement more efficient list allocation by pre-allocating the space and filling it using PyList_SET_ITEM.
    // As of now, the problem is that Python is not allowing to resize the list via API, and it cannot allocate the list at the base of PyStructList, it can only allocate it somewhere in memory not under control.
    PyObject * psl = PyStructList<StorageT>::PyType.tp_alloc( &PyStructList<StorageT>::PyType, 0 );
    new ( psl ) PyStructList<StorageT>( const_cast<PyStruct *>( pystruct ), const_cast<std::vector<StorageT> &>( v ), cspArrayType );

    for( size_t index = 0; index < sz; ++index )
    {
        PyObjectPtr element = PyObjectPtr::own( toPython<ElemT>( v[ index ], *elemType ) );
        PyList_Append( ( PyObject * ) psl, element.get() );
    }
    
    return psl;
}

template<typename StorageT>
struct FromPython<std::vector<StorageT>>
{
    static std::vector<StorageT> impl( PyObject * o, const CspType & arrayType )
    {
        assert( arrayType.type() == CspType::Type::ARRAY );

        using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

        const CspType & elemType = *static_cast<const CspArrayType &>( arrayType ).elemType();

        std::vector<StorageT> out;
        //fast path for list and tuple since we can size up front
        if( PyList_Check( o ) )
        {
            size_t size = PyList_GET_SIZE( o );
            out.reserve( size );
            for( size_t i = 0; i < size; ++i )
                out.emplace_back( fromPython<ElemT>( PyList_GET_ITEM( o, i ), elemType ) );
        }
        else if( PyTuple_Check( o ) )
        {
            size_t size = PyTuple_GET_SIZE( o );
            out.reserve( size );
            for( size_t i = 0; i < size; ++i )
                out.emplace_back( fromPython<ElemT>( PyTuple_GET_ITEM( o, i ), elemType ) );
        }
        //allow iteratables
        else if( o -> ob_type -> tp_iter )
        {
            PyObjectPtr iter = PyObjectPtr::own( o -> ob_type -> tp_iter( o ) );
            PyObject * value;
            while( ( value = iter -> ob_type -> tp_iternext( iter.get() ) ) )
            {
                out.emplace_back( fromPython<ElemT>( value, elemType ) );
                Py_DECREF( value );
            }
            if( PyErr_Occurred() )
            {
                if( PyErr_ExceptionMatches(PyExc_StopIteration))
                    PyErr_Clear();
                else
                    CSP_THROW( PythonPassthrough, "" );
            }
        }
        else
            CSP_THROW( TypeError,  "Invalid list / iterator type, expected list or iterator got " << Py_TYPE( o ) -> tp_name );

        return out;
    }
};

//timeserise helper method
PyObject * lastValueToPython( const csp::TimeSeriesProvider * ts );
PyObject * valueAtIndexToPython( const csp::TimeSeriesProvider * ts, int32_t index );

}

#endif
