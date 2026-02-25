// FieldReader subclasses for reading Arrow list columns into numpy arrays.
//
// Provides createNumpyArrayReader (1D arrays) and createNumpyNDArrayReader
// (N-dimensional arrays with a separate dimensions column + reshape callback).
// Element types supported: float64, int64, bool, string.

#ifndef _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListReader_H
#define _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListReader_H

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/python/Conversions.h>
#include <csp/python/NumpyConversions.h>

#include <arrow/array.h>
#include <arrow/type.h>

// codecvt is deprecated in C++17 but still the simplest way to do UTF-8 <-> UTF-32
// No replacement in the standard until C++26; suppress the deprecation warning.
#if defined( __GNUC__ ) || defined( __clang__ )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <codecvt>
#if defined( __GNUC__ ) || defined( __clang__ )
#pragma GCC diagnostic pop
#endif
#include <cstring>
#include <functional>
#include <locale>
#include <memory>
#include <type_traits>

namespace csp::python
{

// Callback for reshaping a flat 1D array using dimensions read from another column.
using ReshapeCallback = std::function<DialectGenericType( DialectGenericType flatData, const std::vector<int64_t> & dims )>;

namespace numpy
{

// NaN for doubles in list elements; throw for other types on null
template<typename T>
struct ListValueProvider
{
    template<typename ArrowArrayT>
    static T getValue( const std::shared_ptr<ArrowArrayT> & arr, int64_t i )
    {
        if( !arr -> IsValid( i ) )
            CSP_THROW( ValueError, "Null value in list element at index " << i );
        return arr -> GetView( i );
    }
};

template<>
struct ListValueProvider<double>
{
    template<typename ArrowArrayT>
    static double getValue( const std::shared_ptr<ArrowArrayT> & arr, int64_t i )
    {
        if( !arr -> IsValid( i ) )
            return std::numeric_limits<double>::quiet_NaN();
        return arr -> GetView( i );
    }
};

// Create a readValue lambda for native-typed list data (INT64, DOUBLE, BOOL)
template<typename CspT, typename ArrowValueArrayT>
std::function<DialectGenericType( const ::arrow::Array &, int64_t )>
makeNativeListReadValue()
{
    auto * dtype = PyArray_DescrFromType( NPY_TYPE<CspT>::value );

    return [dtype]( const ::arrow::Array & arr, int64_t row ) -> DialectGenericType
    {
        auto & listArr = static_cast<const ::arrow::ListArray &>( arr );
        auto values = std::static_pointer_cast<ArrowValueArrayT>( listArr.value_slice( row ) );

        npy_intp size = values -> length();
        Py_INCREF( dtype );
        PyObject * pyArr = PyArray_SimpleNewFromDescr( 1, &size, dtype );
        PyObjectPtr arrOwner{ PyObjectPtr::own( pyArr ) };

        auto * buf = reinterpret_cast<CspT *>( PyArray_DATA( reinterpret_cast<PyArrayObject *>( pyArr ) ) );

        // Fast path: bulk memcpy for numeric types when there are no nulls.
        // BooleanArray stores packed bits, so memcpy is not applicable for bool.
        if constexpr( !std::is_same_v<CspT, bool> )
        {
            if( values -> null_count() == 0 )
            {
                std::memcpy( buf, values -> raw_values(), sizeof( CspT ) * size );
                return fromPython<DialectGenericType>( pyArr );
            }
        }

        for( int64_t i = 0; i < values -> length(); ++i )
            buf[i] = ListValueProvider<CspT>::getValue( values, i );

        return fromPython<DialectGenericType>( pyArr );
    };
}

// Create a readValue lambda for string-typed list data
template<typename ArrowStringArrayT>
std::function<DialectGenericType( const ::arrow::Array &, int64_t )>
makeStringListReadValue()
{
    // Shared converter object: avoids recreating codecvt facet per row.
    // The lambda is called sequentially so sharing a mutable converter is safe.
    auto converter = std::make_shared<std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>>();

    return [converter]( const ::arrow::Array & arr, int64_t row ) -> DialectGenericType
    {
        auto & listArr = static_cast<const ::arrow::ListArray &>( arr );
        auto values = std::static_pointer_cast<ArrowStringArrayT>( listArr.value_slice( row ) );

        // First pass: compute max string length (in bytes)
        size_t maxLen = 0;
        for( int64_t i = 0; i < values -> length(); ++i )
        {
            if( values -> IsValid( i ) )
                maxLen = std::max( maxLen, values -> GetView( i ).size() );
        }

        // Create numpy unicode array with dtype "U<maxLen>"
        npy_intp size = values -> length();
        PyArray_Descr * typ;
        PyObject * typeStringDescr = toPython( std::string( "U" ) + std::to_string( maxLen ) );
        PyArray_DescrConverter( typeStringDescr, &typ );
        Py_DECREF( typeStringDescr );

        PyObject * pyArr = PyArray_SimpleNewFromDescr( 1, &size, typ );
        PyObjectPtr arrOwner{ PyObjectPtr::own( pyArr ) };

        auto * arrayObject = reinterpret_cast<PyArrayObject *>( pyArr );
        auto elementSize = PyDataType_ELSIZE( PyArray_DESCR( arrayObject ) );

        for( int64_t i = 0; i < values -> length(); ++i )
        {
            auto view = values -> GetView( i );
            auto wideValue = converter -> from_bytes( view.data(), view.data() + view.size() );
            auto nElementsToCopy = std::min( int( elementSize / sizeof( char32_t ) ), int( wideValue.size() + 1 ) );
            std::copy_n( wideValue.c_str(), nElementsToCopy,
                         reinterpret_cast<char32_t *>( PyArray_GETPTR1( arrayObject, i ) ) );
        }

        return fromPython<DialectGenericType>( pyArr );
    };
}

// Dispatch on arrow list element type to create the appropriate readValue lambda
inline std::function<DialectGenericType( const ::arrow::Array &, int64_t )>
dispatchListReadValue( const std::shared_ptr<::arrow::ListType> & listType, const std::string & columnName )
{
    auto valueTypeId = listType -> value_type() -> id();

    switch( valueTypeId )
    {
        case ::arrow::Type::INT64:
            return makeNativeListReadValue<int64_t, ::arrow::Int64Array>();
        case ::arrow::Type::DOUBLE:
            return makeNativeListReadValue<double, ::arrow::DoubleArray>();
        case ::arrow::Type::BOOL:
            return makeNativeListReadValue<bool, ::arrow::BooleanArray>();
        case ::arrow::Type::STRING:
            return makeStringListReadValue<::arrow::StringArray>();
        case ::arrow::Type::BINARY:
            return makeStringListReadValue<::arrow::BinaryArray>();
        case ::arrow::Type::LARGE_STRING:
            return makeStringListReadValue<::arrow::LargeStringArray>();
        case ::arrow::Type::LARGE_BINARY:
            return makeStringListReadValue<::arrow::LargeBinaryArray>();
        default:
            CSP_THROW( TypeError, "Unsupported list element type " << listType -> value_type() -> ToString()
                                   << " for list column '" << columnName << "'" );
    }
}

// Read dimension values from a list column cell
inline std::vector<int64_t> readDimsFromListCell( const ::arrow::ListArray & listArr, int64_t row )
{
    auto values = listArr.value_slice( row );
    std::vector<int64_t> dims;
    dims.reserve( values -> length() );

    switch( values -> type_id() )
    {
        case ::arrow::Type::INT32:
        {
            auto typed = std::static_pointer_cast<::arrow::Int32Array>( values );
            for( int64_t i = 0; i < typed -> length(); ++i )
                dims.push_back( typed -> Value( i ) );
            break;
        }
        case ::arrow::Type::INT64:
        {
            auto typed = std::static_pointer_cast<::arrow::Int64Array>( values );
            for( int64_t i = 0; i < typed -> length(); ++i )
                dims.push_back( typed -> Value( i ) );
            break;
        }
        default:
            CSP_THROW( TypeError, "Dimensions column has unsupported element type: " << values -> type() -> ToString() );
    }
    return dims;
}

// FieldReader for 1D numpy array columns (Arrow list -> numpy 1D array)
class NumpyArrayReader final : public csp::adapters::arrow::FieldReader
{
public:
    NumpyArrayReader( int colIdx, const StructFieldPtr & structField,
                      std::function<DialectGenericType( const ::arrow::Array &, int64_t )> readValue,
                      std::vector<std::string> columnNames )
        : FieldReader( std::move( columnNames ), structField ),
          m_colIdx( colIdx ),
          m_readValue( std::move( readValue ) )
    {
    }

    void bindBatch( const ::arrow::RecordBatch & batch ) override
    {
        bindColumn( batch.column( m_colIdx ).get() );
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & listArr = static_cast<const ::arrow::ListArray &>( *m_column );
        if( listArr.IsValid( row ) )
        {
            auto arrayValue = m_readValue( listArr, row );
            m_field -> setValue<DialectGenericType>( s, std::move( arrayValue ) );
        }
    }

private:
    int                      m_colIdx;
    std::function<DialectGenericType( const ::arrow::Array &, int64_t )> m_readValue;
};

// FieldReader for NDArray columns (Arrow list + dims column -> numpy NDArray via reshape)
class NumpyNDArrayReader final : public csp::adapters::arrow::FieldReader
{
public:
    NumpyNDArrayReader( int colIdx, int dimsColIdx, const StructFieldPtr & structField,
                        std::function<DialectGenericType( const ::arrow::Array &, int64_t )> readValue,
                        ReshapeCallback reshapeCallback,
                        std::vector<std::string> columnNames )
        : FieldReader( std::move( columnNames ), structField ),
          m_colIdx( colIdx ), m_dimsColIdx( dimsColIdx ),
          m_readValue( std::move( readValue ) ), m_reshapeCallback( std::move( reshapeCallback ) ),
          m_dimsColumn( nullptr )
    {
    }

    void bindBatch( const ::arrow::RecordBatch & batch ) override
    {
        bindColumn( batch.column( m_colIdx ).get() );
        m_dimsColumn = batch.column( m_dimsColIdx ).get();
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & listArr = static_cast<const ::arrow::ListArray &>( *m_column );
        if( listArr.IsValid( row ) )
        {
            auto arrayValue = m_readValue( listArr, row );

            auto & dimsArr = static_cast<const ::arrow::ListArray &>( *m_dimsColumn );
            if( dimsArr.IsValid( row ) )
            {
                auto dims = readDimsFromListCell( dimsArr, row );
                arrayValue = m_reshapeCallback( std::move( arrayValue ), dims );
            }

            m_field -> setValue<DialectGenericType>( s, std::move( arrayValue ) );
        }
    }

private:
    int                      m_colIdx;
    int                      m_dimsColIdx;
    std::function<DialectGenericType( const ::arrow::Array &, int64_t )> m_readValue;
    ReshapeCallback          m_reshapeCallback;
    const ::arrow::Array *   m_dimsColumn;
};

} // namespace numpy

// Create a FieldReader for a 1D numpy array column
inline std::unique_ptr<csp::adapters::arrow::FieldReader> createNumpyArrayReader(
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::string & columnName,
    const StructFieldPtr & structField )
{
    int colIdx = schema -> GetFieldIndex( columnName );
    CSP_TRUE_OR_THROW_RUNTIME( colIdx >= 0,
                               "List column '" << columnName << "' not found in arrow schema" );

    auto arrowField = schema -> field( colIdx );
    auto listType = std::static_pointer_cast<::arrow::ListType>( arrowField -> type() );
    CSP_TRUE_OR_THROW_RUNTIME( listType != nullptr,
                               "Column '" << columnName << "' is not a list type" );

    auto readValue = numpy::dispatchListReadValue( listType, columnName );

    return std::make_unique<numpy::NumpyArrayReader>(
        colIdx, structField, std::move( readValue ), std::vector<std::string>{ columnName } );
}

// Create a FieldReader for an NDArray column (data + dimensions + reshape)
inline std::unique_ptr<csp::adapters::arrow::FieldReader> createNumpyNDArrayReader(
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::string & columnName,
    const std::string & dimsColumnName,
    const StructFieldPtr & structField,
    ReshapeCallback reshapeCallback )
{
    int colIdx = schema -> GetFieldIndex( columnName );
    CSP_TRUE_OR_THROW_RUNTIME( colIdx >= 0,
                               "List column '" << columnName << "' not found in arrow schema" );

    int dimsColIdx = schema -> GetFieldIndex( dimsColumnName );
    CSP_TRUE_OR_THROW_RUNTIME( dimsColIdx >= 0,
                               "Dimensions column '" << dimsColumnName << "' not found in arrow schema" );

    auto arrowField = schema -> field( colIdx );
    auto listType = std::static_pointer_cast<::arrow::ListType>( arrowField -> type() );
    CSP_TRUE_OR_THROW_RUNTIME( listType != nullptr,
                               "Column '" << columnName << "' is not a list type" );

    CSP_TRUE_OR_THROW_RUNTIME( reshapeCallback,
                               "Dimensions column specified for '" << columnName << "' but no reshape callback provided" );

    auto readValue = numpy::dispatchListReadValue( listType, columnName );

    return std::make_unique<numpy::NumpyNDArrayReader>(
        colIdx, dimsColIdx, structField, std::move( readValue ), std::move( reshapeCallback ),
        std::vector<std::string>{ columnName, dimsColumnName } );
}

} // namespace csp::python

#endif
