// FieldWriter subclasses for writing numpy arrays into Arrow list columns.
//
// Provides createNumpyArrayWriter (1D arrays) and createNumpyNDArrayWriter
// (N-dimensional arrays with a separate dimensions column + shape callback).
// Mirrors ArrowNumpyListReader.h in the write direction.
// Element types supported: float64, int64, bool, string.

#ifndef _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListWriter_H
#define _IN_CSP_PYTHON_ADAPTERS_ArrowNumpyListWriter_H

#include <csp/adapters/arrow/ArrowFieldWriter.h>
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
        case NPY_INT:
            return std::make_unique<NativeListWriter<int64_t, ::arrow::Int64Builder>>( columnName, structField, ::arrow::int64() );
        case NPY_BOOL:
            return std::make_unique<NativeListWriter<bool, ::arrow::BooleanBuilder>>( columnName, structField, ::arrow::boolean() );
        case NPY_UNICODE:
            return std::make_unique<StringListWriter>( columnName, structField );
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

} // namespace csp::python

#endif
