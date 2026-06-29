#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ArrowBackedArrayBuilder.h>
#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>

using namespace csp;
using namespace csp::adapters::parquet;

namespace csp::adapters::parquet
{

uint32_t ParquetOutputHandler::getChunkSize() const
{
    return m_parquetWriter.getChunkSize();
}


SingleColumnParquetOutputHandler::SingleColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type,
                                                                    std::string columnName )
        : ParquetOutputHandler( parquetWriter )
{
    bool isBytes = ( type -> type() == CspType::TypeTraits::STRING )
                   && static_cast<const CspStringType &>( *type ).isBytes();

    csp::adapters::arrow::visitCspValueType( type -> type(),
        [&]( auto tag )
        {
            using T = typename decltype( tag )::type;
            auto builder = std::make_shared<ScalarColumnArrayBuilder<T>>( columnName, getChunkSize(), isBytes );
            m_columnArrayBuilder = builder;
            // On tick, set this row's value directly on the typed builder -- no scratch struct.
            m_valueHandler = std::make_unique<ValueHandler>(
                [ builder ]( const TimeSeriesProvider * input )
                {
                    builder -> setValue( input -> lastValueTyped<T>() );
                } );
        },
        [&]()
        {
            CSP_THROW( TypeError, "Writing of " << type -> type().asString() << " to parquet is not supported" );
        } );
}

void SingleColumnParquetOutputAdapter::executeImpl()
{
    ( *m_valueHandler )( input() );
    m_parquetWriter.scheduleEndCycleEvent();
}

// writeValue: used by dict basket writers to set values directly
template< typename T, typename Ignored >
void SingleColumnParquetOutputHandler::writeValue( const T & value )
{
    static_cast<ScalarColumnArrayBuilder<T> *>( m_columnArrayBuilder.get() ) -> setValue( value );
}

// Explicit instantiations for types used by ParquetDictBasketOutputWriter
template void SingleColumnParquetOutputHandler::writeValue<std::string, void>( const std::string & );
template void SingleColumnParquetOutputHandler::writeValue<std::uint16_t, void>( const std::uint16_t & );

ListColumnParquetOutputHandler::ListColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &elemType,
                                                                const std::string &columnName )
        : ParquetOutputHandler( parquetWriter )
{
    auto [ valueBuilder, writeItemsFn ] = csp::adapters::arrow::createListFieldWriter( elemType );
    m_columnArrayBuilder = std::make_shared<ListColumnArrayBuilder>(
        columnName, getChunkSize(), std::move( valueBuilder ), std::move( writeItemsFn ) );

    m_valueHandler = std::make_unique<ValueHandler>(
            [ this ]( const TimeSeriesProvider *input )
            {
                static_cast<ListColumnArrayBuilder *>(this -> m_columnArrayBuilder.get())
                        -> setValue( input -> lastValueTyped<DialectGenericType>() );
            } );
}

void ListColumnParquetOutputAdapter::executeImpl()
{
    ( *m_valueHandler )( input() );
    m_parquetWriter.scheduleEndCycleEvent();
}

StructParquetOutputHandler::StructParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type,
                                                        DictionaryPtr fieldMap )
        : ParquetOutputHandler( parquetWriter )
{
    auto structMetaPtr = std::static_pointer_cast<const CspStructType>( type ) -> meta().get();

    for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
    {
        auto structFieldName = it.key();
        auto columnName      = it.value<std::string>();
        auto structField     = structMetaPtr -> field( structFieldName );

        if( !structField )
            CSP_THROW( ValueError, "Struct field '" << structFieldName << "' not found" );

        if( structField -> type() -> type() == CspType::Type::DIALECT_GENERIC )
            CSP_THROW( TypeError, "Writing of column " << structField -> fieldname()
                << " of type DIALECT_GENERIC is not supported" );

        auto builder = createArrowBackedArrayBuilderForField( columnName, getChunkSize(), structField );
        m_arrowBuilders.push_back( builder.get() );
        m_columnArrayBuilders.push_back( std::move( builder ) );
    }
}

void StructParquetOutputHandler::writeValueFromTs( const TimeSeriesProvider *input )
{
    const Struct *structData = input -> lastValueTyped<StructPtr>().get();

    for( auto * builder : m_arrowBuilders )
    {
        builder -> setStruct( structData );
    }
    m_parquetWriter.scheduleEndCycleEvent();
}

void StructParquetOutputAdapter::executeImpl()
{
    writeValueFromTs( input() );
}

}
