// FieldWriter subclasses for writing numpy arrays into Arrow list columns.
//
// Provides createNumpyArrayWriter (1D arrays) and createNumpyNDArrayWriter
// (N-dimensional arrays with a separate dimensions column + shape callback).
// Mirrors ArrowNumpyListReader.h in the write direction.
// Element types supported: float64, int64, bool, string.

#ifndef _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListWriter_H
#define _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListWriter_H

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/core/Time.h>
#include <csp/python/Conversions.h>
#include <csp/python/NumpyConversions.h>

#include <arrow/builder.h>
#include <arrow/type.h>

#include <codecvt>
#include <functional>
#include <locale>
#include <memory>

namespace csp::python
{

// Callback to extract shape from an NDArray: returns vector of dimension sizes.
using ShapeCallback = std::function<std::vector<int64_t>( DialectGenericType ndarray )>;

namespace numpy
{

// Helper macro for arrow status checks
#define ARROW_OK_OR_THROW_WRITER( expr, msg ) \
    do { auto __s = ( expr ); if( !__s.ok() ) CSP_THROW( RuntimeException, msg << ": " << __s.ToString() ); } while(0)

// --- Native element writers (double, int64, bool) ---
// Use bulk AppendValues for a single resize + copy instead of per-element Append.

template<typename CspT, typename ArrowBuilderT>
void writeNativeElements( ArrowBuilderT * valueBuilder, PyArrayObject * pyArr, npy_intp len )
{
    auto * data = reinterpret_cast<const typename ArrowBuilderT::value_type *>( PyArray_DATA( pyArr ) );
    ARROW_OK_OR_THROW_WRITER( valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) ),
                              "Failed to append list elements" );
}

// Bool specialization: numpy stores bools as uint8, BooleanBuilder::AppendValues accepts uint8
template<>
inline void writeNativeElements<bool, ::arrow::BooleanBuilder>(
    ::arrow::BooleanBuilder * valueBuilder, PyArrayObject * pyArr, npy_intp len )
{
    auto * data = reinterpret_cast<const uint8_t *>( PyArray_DATA( pyArr ) );
    ARROW_OK_OR_THROW_WRITER( valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) ),
                              "Failed to append bool list elements" );
}

// --- Native list writer ---

template<typename CspT, typename ArrowBuilderT>
class NativeListWriter final : public csp::adapters::arrow::FieldWriter
{
public:
    NativeListWriter( const std::string & columnName, const StructFieldPtr & structField,
                      std::shared_ptr<::arrow::DataType> elementType )
        : FieldWriter( { columnName }, { ::arrow::list( elementType ) }, structField )
    {
        m_valueBuilder = std::make_shared<ArrowBuilderT>();
        m_listBuilder = std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_valueBuilder );
    }

    void reserve( int64_t numRows ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Reserve( numRows ), "Failed to reserve list builder" );
    }

    void writeNull() override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> AppendNull(), "Failed to append null list" );
    }

    std::vector<std::shared_ptr<::arrow::Array>> finish() override
    {
        std::shared_ptr<::arrow::Array> arr;
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Finish( &arr ), "Failed to finish list array" );
        return { arr };
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Append(), "Failed to start list" );
        auto & dgt = m_field -> value<DialectGenericType>( s );
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );

        // writeNativeElements uses PyArray_DATA for bulk copy, which requires C-contiguous layout.
        // Non-contiguous arrays (slices, transposes) must be copied to contiguous form first.
        if( PyArray_IS_C_CONTIGUOUS( pyArr ) )
        {
            writeNativeElements<CspT, ArrowBuilderT>( m_valueBuilder.get(), pyArr, len );
        }
        else
        {
            PyObjectPtr contiguousOwner{ PyObjectPtr::own(
                reinterpret_cast<PyObject *>( PyArray_GETCONTIGUOUS( pyArr ) ) ) };
            writeNativeElements<CspT, ArrowBuilderT>( m_valueBuilder.get(),
                reinterpret_cast<PyArrayObject *>( contiguousOwner.get() ), len );
        }
    }

private:
    std::shared_ptr<ArrowBuilderT>                      m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>               m_listBuilder;
};

// --- String list writer ---

class StringListWriter final : public csp::adapters::arrow::FieldWriter
{
public:
    StringListWriter( const std::string & columnName, const StructFieldPtr & structField )
        : FieldWriter( { columnName }, { ::arrow::list( ::arrow::utf8() ) }, structField )
    {
        m_valueBuilder = std::make_shared<::arrow::StringBuilder>();
        m_listBuilder = std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_valueBuilder );
    }

    void reserve( int64_t numRows ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Reserve( numRows ), "Failed to reserve string list builder" );
    }

    void writeNull() override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> AppendNull(), "Failed to append null list" );
    }

    std::vector<std::shared_ptr<::arrow::Array>> finish() override
    {
        std::shared_ptr<::arrow::Array> arr;
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Finish( &arr ), "Failed to finish string list array" );
        return { arr };
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Append(), "Failed to start list" );
        auto & dgt = m_field -> value<DialectGenericType>( s );
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );
        auto elementSize = PyDataType_ELSIZE( PyArray_DESCR( pyArr ) );
        auto charCount = elementSize / sizeof( char32_t );

        for( npy_intp i = 0; i < len; ++i )
        {
            auto * ptr = reinterpret_cast<char32_t *>( PyArray_GETPTR1( pyArr, i ) );
            // Find actual string length (exclude trailing nulls)
            size_t actualLen = 0;
            for( size_t c = 0; c < charCount; ++c )
            {
                if( ptr[c] == 0 ) break;
                actualLen = c + 1;
            }
            m_utf8Buf = m_converter.to_bytes( ptr, ptr + actualLen );
            ARROW_OK_OR_THROW_WRITER( m_valueBuilder -> Append( m_utf8Buf ), "Failed to append string list element" );
        }
    }

private:
    std::shared_ptr<::arrow::StringBuilder>             m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>               m_listBuilder;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> m_converter;  // reused across rows
    std::string                                         m_utf8Buf;            // reused buffer
};

// --- Temporal list writers (datetime64[ns] → Timestamp, timedelta64[ns] → Duration) ---

// Writes numpy datetime64/timedelta64 arrays into Arrow List<Timestamp(ns,"UTC")> or List<Duration(ns)> columns.
// Detects the numpy datetime unit at write time and converts to nanoseconds using scalingFromNumpyDtUnit().
template<typename ArrowBuilderT>
class TemporalListWriter final : public csp::adapters::arrow::FieldWriter
{
public:
    TemporalListWriter( const std::string & columnName, const StructFieldPtr & structField,
                        std::shared_ptr<::arrow::DataType> elementType )
        : FieldWriter( { columnName }, { ::arrow::list( elementType ) }, structField )
    {
        m_valueBuilder = std::make_shared<ArrowBuilderT>( elementType, ::arrow::default_memory_pool() );
        m_listBuilder = std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_valueBuilder );
    }

    void reserve( int64_t numRows ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Reserve( numRows ), "Failed to reserve temporal list builder" );
    }

    void writeNull() override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> AppendNull(), "Failed to append null temporal list" );
    }

    std::vector<std::shared_ptr<::arrow::Array>> finish() override
    {
        std::shared_ptr<::arrow::Array> arr;
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Finish( &arr ), "Failed to finish temporal list array" );
        return { arr };
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Append(), "Failed to start temporal list" );
        auto & dgt = m_field -> value<DialectGenericType>( s );
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );

        // PyArray_DATA requires C-contiguous layout for correct bulk reads.
        // Non-contiguous arrays (slices, transposes) must be copied to contiguous form first.
        PyArrayObject * contiguousArr = pyArr;
        PyObjectPtr contiguousOwner;
        if( !PyArray_IS_C_CONTIGUOUS( pyArr ) )
        {
            contiguousOwner = PyObjectPtr::own(
                reinterpret_cast<PyObject *>( PyArray_GETCONTIGUOUS( pyArr ) ) );
            contiguousArr = reinterpret_cast<PyArrayObject *>( contiguousOwner.get() );
        }

        auto * data = reinterpret_cast<const int64_t *>( PyArray_DATA( contiguousArr ) );

        // Detect numpy datetime unit and compute scaling to nanoseconds
        auto unit = datetimeUnitFromDescr( PyArray_DESCR( pyArr ) );
        int64_t scaling = scalingFromNumpyDtUnit( unit );

        if( scaling == 1 )
        {
            // Already nanoseconds — bulk append
            ARROW_OK_OR_THROW_WRITER(
                m_valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) ),
                "Failed to append temporal list elements" );
        }
        else
        {
            ARROW_OK_OR_THROW_WRITER(
                m_valueBuilder -> Reserve( static_cast<int64_t>( len ) ),
                "Failed to reserve temporal value builder" );
            for( npy_intp i = 0; i < len; ++i )
                m_valueBuilder -> UnsafeAppend( data[i] * scaling );
        }
    }

private:
    std::shared_ptr<ArrowBuilderT>          m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>   m_listBuilder;
};

// Date writer: numpy datetime64 → Arrow List<Date32> (nanoseconds → days since epoch)
class DateListWriter final : public csp::adapters::arrow::FieldWriter
{
public:
    DateListWriter( const std::string & columnName, const StructFieldPtr & structField )
        : FieldWriter( { columnName }, { ::arrow::list( ::arrow::date32() ) }, structField )
    {
        m_valueBuilder = std::make_shared<::arrow::Date32Builder>();
        m_listBuilder = std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_valueBuilder );
    }

    void reserve( int64_t numRows ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Reserve( numRows ), "Failed to reserve date list builder" );
    }

    void writeNull() override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> AppendNull(), "Failed to append null date list" );
    }

    std::vector<std::shared_ptr<::arrow::Array>> finish() override
    {
        std::shared_ptr<::arrow::Array> arr;
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Finish( &arr ), "Failed to finish date list array" );
        return { arr };
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW_WRITER( m_listBuilder -> Append(), "Failed to start date list" );
        auto & dgt = m_field -> value<DialectGenericType>( s );
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );

        // PyArray_DATA requires C-contiguous layout for correct bulk reads.
        PyArrayObject * contiguousArr = pyArr;
        PyObjectPtr contiguousOwner;
        if( !PyArray_IS_C_CONTIGUOUS( pyArr ) )
        {
            contiguousOwner = PyObjectPtr::own(
                reinterpret_cast<PyObject *>( PyArray_GETCONTIGUOUS( pyArr ) ) );
            contiguousArr = reinterpret_cast<PyArrayObject *>( contiguousOwner.get() );
        }

        auto * data = reinterpret_cast<const int64_t *>( PyArray_DATA( contiguousArr ) );

        // Detect numpy datetime unit and compute scaling to nanoseconds
        auto unit = datetimeUnitFromDescr( PyArray_DESCR( pyArr ) );
        int64_t scaling = scalingFromNumpyDtUnit( unit );

        ARROW_OK_OR_THROW_WRITER(
            m_valueBuilder -> Reserve( static_cast<int64_t>( len ) ),
            "Failed to reserve date value builder" );
        for( npy_intp i = 0; i < len; ++i )
            m_valueBuilder -> UnsafeAppend( static_cast<int32_t>( data[i] * scaling / csp::NANOS_PER_DAY ) );
    }

private:
    std::shared_ptr<::arrow::Date32Builder>  m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>    m_listBuilder;
};

// --- Dispatch by npy_type ---

inline std::unique_ptr<csp::adapters::arrow::FieldWriter> dispatchListWriter(
    const std::string & columnName,
    const StructFieldPtr & structField,
    int npyType )
{
    switch( npyType )
    {
        case NPY_DOUBLE:
            return std::make_unique<NativeListWriter<double, ::arrow::DoubleBuilder>>( columnName, structField, ::arrow::float64() );
        case NPY_LONGLONG:
        case NPY_LONG:
            return std::make_unique<NativeListWriter<int64_t, ::arrow::Int64Builder>>( columnName, structField, ::arrow::int64() );
        case NPY_INT:
            return std::make_unique<NativeListWriter<int32_t, ::arrow::Int32Builder>>( columnName, structField, ::arrow::int32() );
        case NPY_BOOL:
            return std::make_unique<NativeListWriter<bool, ::arrow::BooleanBuilder>>( columnName, structField, ::arrow::boolean() );
        case NPY_UNICODE:
            return std::make_unique<StringListWriter>( columnName, structField );
        case NPY_DATETIME:
            return std::make_unique<TemporalListWriter<::arrow::TimestampBuilder>>(
                columnName, structField, std::make_shared<::arrow::TimestampType>( ::arrow::TimeUnit::NANO, "UTC" ) );
        case NPY_TIMEDELTA:
            return std::make_unique<TemporalListWriter<::arrow::DurationBuilder>>(
                columnName, structField, std::make_shared<::arrow::DurationType>( ::arrow::TimeUnit::NANO ) );
        default:
            CSP_THROW( TypeError, "Unsupported numpy type " << npyType << " for list column '" << columnName << "'" );
    }
}

// --- NDArray writer (data column + dimensions column) ---

class NumpyNDArrayWriter final : public csp::adapters::arrow::FieldWriter
{
public:
    NumpyNDArrayWriter( const std::string & columnName, const std::string & dimsColumnName,
                        const StructFieldPtr & structField, int npyType,
                        ShapeCallback shapeCallback )
        : FieldWriter( { columnName, dimsColumnName }, {}, structField ),
          m_shapeCallback( std::move( shapeCallback ) )
    {
        m_dataWriter = dispatchListWriter( columnName, structField, npyType );

        m_dimsValueBuilder = std::make_shared<::arrow::Int64Builder>();
        m_dimsListBuilder = std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_dimsValueBuilder );

        m_dataTypes = m_dataWriter -> dataTypes();
        m_dataTypes.push_back( ::arrow::list( ::arrow::int64() ) );
    }

    void reserve( int64_t numRows ) override
    {
        m_dataWriter -> reserve( numRows );
        ARROW_OK_OR_THROW_WRITER( m_dimsListBuilder -> Reserve( numRows ), "Failed to reserve dims list builder" );
    }

    void writeNull() override
    {
        m_dataWriter -> writeNull();
        ARROW_OK_OR_THROW_WRITER( m_dimsListBuilder -> AppendNull(), "Failed to append null dims" );
    }

    std::vector<std::shared_ptr<::arrow::Array>> finish() override
    {
        auto dataArrays = m_dataWriter -> finish();
        std::shared_ptr<::arrow::Array> dimsArr;
        ARROW_OK_OR_THROW_WRITER( m_dimsListBuilder -> Finish( &dimsArr ), "Failed to finish dims array" );
        dataArrays.push_back( dimsArr );
        return dataArrays;
    }

protected:
    void doWrite( const Struct * s ) override
    {
        // Write data: for C-contiguous NDArrays, the native writer handles flat data correctly
        // We call writeNext on the inner data writer which will check isSet and delegate to its doWrite
        m_dataWriter -> writeNext( s );

        // Write shape/dims
        auto & dgt = m_field -> value<DialectGenericType>( s );
        auto shape = m_shapeCallback( dgt );

        ARROW_OK_OR_THROW_WRITER( m_dimsListBuilder -> Append(), "Failed to start dims list" );
        ARROW_OK_OR_THROW_WRITER(
            m_dimsValueBuilder -> AppendValues( shape.data(), static_cast<int64_t>( shape.size() ) ),
            "Failed to append dim values" );
    }

private:
    ShapeCallback                                       m_shapeCallback;
    std::unique_ptr<csp::adapters::arrow::FieldWriter>  m_dataWriter;
    std::shared_ptr<::arrow::Int64Builder>              m_dimsValueBuilder;
    std::shared_ptr<::arrow::ListBuilder>               m_dimsListBuilder;
};

#undef ARROW_OK_OR_THROW_WRITER

// --- Standalone list-items writers for the ListFieldWriterFactory ---
// These write all elements of a numpy array into an arrow value builder
// (list start/end is handled by the caller, e.g. ListColumnArrayBuilder).

template<typename CspT, typename ArrowBuilderT>
inline csp::adapters::arrow::ListItemsWriter makeNativeListItemsWriter(
    std::shared_ptr<ArrowBuilderT> valueBuilder )
{
    return [valueBuilder]( const DialectGenericType & dgt )
    {
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );

        if( PyArray_IS_C_CONTIGUOUS( pyArr ) )
        {
            auto * data = reinterpret_cast<const typename ArrowBuilderT::value_type *>( PyArray_DATA( pyArr ) );
            auto s = valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) );
            CSP_TRUE_OR_THROW_RUNTIME( s.ok(), "Failed to append list elements: " << s.ToString() );
        }
        else
        {
            PyObjectPtr contiguousOwner{ PyObjectPtr::own(
                reinterpret_cast<PyObject *>( PyArray_GETCONTIGUOUS( pyArr ) ) ) };
            auto * data = reinterpret_cast<const typename ArrowBuilderT::value_type *>(
                PyArray_DATA( reinterpret_cast<PyArrayObject *>( contiguousOwner.get() ) ) );
            auto s = valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) );
            CSP_TRUE_OR_THROW_RUNTIME( s.ok(), "Failed to append list elements: " << s.ToString() );
        }
    };
}

inline csp::adapters::arrow::ListItemsWriter makeBoolListItemsWriter(
    std::shared_ptr<::arrow::BooleanBuilder> valueBuilder )
{
    return [valueBuilder]( const DialectGenericType & dgt )
    {
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );

        if( PyArray_IS_C_CONTIGUOUS( pyArr ) )
        {
            auto * data = reinterpret_cast<const uint8_t *>( PyArray_DATA( pyArr ) );
            auto s = valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) );
            CSP_TRUE_OR_THROW_RUNTIME( s.ok(), "Failed to append bool list elements: " << s.ToString() );
        }
        else
        {
            PyObjectPtr contiguousOwner{ PyObjectPtr::own(
                reinterpret_cast<PyObject *>( PyArray_GETCONTIGUOUS( pyArr ) ) ) };
            auto * data = reinterpret_cast<const uint8_t *>(
                PyArray_DATA( reinterpret_cast<PyArrayObject *>( contiguousOwner.get() ) ) );
            auto s = valueBuilder -> AppendValues( data, static_cast<int64_t>( len ) );
            CSP_TRUE_OR_THROW_RUNTIME( s.ok(), "Failed to append bool list elements: " << s.ToString() );
        }
    };
}

inline csp::adapters::arrow::ListItemsWriter makeStringListItemsWriter(
    std::shared_ptr<::arrow::StringBuilder> valueBuilder )
{
    return [valueBuilder]( const DialectGenericType & dgt )
    {
        auto * pyArr = reinterpret_cast<PyArrayObject *>( csp::python::toPythonBorrowed( dgt ) );
        npy_intp len = PyArray_SIZE( pyArr );
        auto elementSize = PyDataType_ELSIZE( PyArray_DESCR( pyArr ) );
        auto charCount = elementSize / sizeof( char32_t );

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        for( npy_intp i = 0; i < len; ++i )
        {
            auto * ptr = reinterpret_cast<char32_t *>( PyArray_GETPTR1( pyArr, i ) );
            size_t actualLen = 0;
            for( size_t c = 0; c < charCount; ++c )
            {
                if( ptr[c] == 0 ) break;
                actualLen = c + 1;
            }
            auto utf8 = converter.to_bytes( ptr, ptr + actualLen );
            auto s = valueBuilder -> Append( utf8 );
            CSP_TRUE_OR_THROW_RUNTIME( s.ok(), "Failed to append string list element: " << s.ToString() );
        }
    };
}

} // namespace numpy

// Create a FieldWriter for a 1D numpy array field
inline std::unique_ptr<csp::adapters::arrow::FieldWriter> createNumpyArrayWriter(
    const std::string & columnName,
    const StructFieldPtr & structField,
    int npyType )
{
    return numpy::dispatchListWriter( columnName, structField, npyType );
}

// Create a FieldWriter for an NDArray field (data column + dimensions column)
inline std::unique_ptr<csp::adapters::arrow::FieldWriter> createNumpyNDArrayWriter(
    const std::string & columnName,
    const std::string & dimsColumnName,
    const StructFieldPtr & structField,
    int npyType,
    ShapeCallback shapeCallback )
{
    return std::make_unique<numpy::NumpyNDArrayWriter>( columnName, dimsColumnName, structField, npyType, std::move( shapeCallback ) );
}

// Register the numpy list field writer factory with the arrow adapter layer.
// Call once during Python module initialization so that createListFieldWriter()
// can produce list-writing functions for parquet output and other paths.
inline void registerNumpyListFieldWriterFactory()
{
    csp::adapters::arrow::registerListFieldWriterFactory(
        []( const CspTypePtr & elemType )
            -> std::pair<std::shared_ptr<::arrow::ArrayBuilder>, csp::adapters::arrow::ListItemsWriter>
        {
            switch( elemType -> type() )
            {
                case CspType::Type::DOUBLE:
                {
                    auto b = std::make_shared<::arrow::DoubleBuilder>();
                    return { b, numpy::makeNativeListItemsWriter<double, ::arrow::DoubleBuilder>( b ) };
                }
                case CspType::Type::INT64:
                {
                    auto b = std::make_shared<::arrow::Int64Builder>();
                    return { b, numpy::makeNativeListItemsWriter<int64_t, ::arrow::Int64Builder>( b ) };
                }
                case CspType::Type::BOOL:
                {
                    auto b = std::make_shared<::arrow::BooleanBuilder>();
                    return { b, numpy::makeBoolListItemsWriter( b ) };
                }
                case CspType::Type::STRING:
                {
                    auto b = std::make_shared<::arrow::StringBuilder>();
                    return { b, numpy::makeStringListItemsWriter( b ) };
                }
                default:
                    CSP_THROW( TypeError, "Unsupported list element type for writer factory: "
                                           << elemType -> type().asString() );
            }
        } );
}

} // namespace csp::python

#endif
