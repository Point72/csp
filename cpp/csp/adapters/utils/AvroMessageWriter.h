#ifndef _IN_CSP_ADAPTERS_UTILS_AVROMESSAGEWRITER_H
#define _IN_CSP_ADAPTERS_UTILS_AVROMESSAGEWRITER_H

#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/utils/AvroIncludes.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PartialSwitchCspType.h>

#include <cstring>
#include <sstream>

namespace csp::adapters::utils
{

class AvroMessageWriter : public MessageWriter
{
public:
    using SupportedCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                        csp::CspType::Type::UINT8,
                                                        csp::CspType::Type::INT16,
                                                        csp::CspType::Type::INT32,
                                                        csp::CspType::Type::INT64,
                                                        csp::CspType::Type::DOUBLE,
                                                        csp::CspType::Type::DATE,
                                                        csp::CspType::Type::DATETIME,
                                                        csp::CspType::Type::ENUM,
                                                        csp::CspType::Type::STRING,
                                                        csp::CspType::Type::STRUCT,
                                                        csp::CspType::Type::ARRAY
                                                        >;

    using SupportedArrayCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                              csp::CspType::Type::UINT8,
                                                              csp::CspType::Type::INT16,
                                                              csp::CspType::Type::INT32,
                                                              csp::CspType::Type::INT64,
                                                              csp::CspType::Type::DOUBLE,
                                                              csp::CspType::Type::DATETIME,
                                                              csp::CspType::Type::ENUM,
                                                              csp::CspType::Type::STRING
                                                              >;

    AvroMessageWriter( const Dictionary & properties )
    {
        std::string schemaJson = properties.get<std::string>( "avro_schema" );
        if( schemaJson.empty() )
            CSP_THROW( RuntimeException, "AvroMessageWriter: 'avro_schema' property is missing or empty" );

        try
        {
            std::istringstream is( schemaJson );
            avro::compileJsonSchema( is, m_validSchema );
        }
        catch( const avro::Exception & e )
        {
            CSP_THROW( RuntimeException, "AvroMessageWriter: Failed to parse Avro schema: " << e.what() );
        }

        if( properties.exists( "datetime_type" ) )
            m_datetimeWireType = utils::DateTimeWireType( properties.get<std::string>( "datetime_type" ) );
        else
            m_datetimeWireType = utils::DateTimeWireType::UINT64_NANOS;

        m_datum = std::make_unique<avro::GenericDatum>( m_validSchema );
    }

    template<typename T>
    void setField( const std::string & field, const T & value, const CspType & type, const FieldEntry & entry )
    {
        if( m_datum->type() != avro::AVRO_RECORD )
            return;

        avro::GenericRecord & record = m_datum->value<avro::GenericRecord>();

        if( record.hasField( field ) )
        {
            avro::GenericDatum & fieldDatum = record.field( field );

            // If it's a union with default null branch, select the non-null branch
            if( fieldDatum.isUnion() && fieldDatum.type() == avro::AVRO_NULL )
            {
                const avro::NodePtr & recordSchema = record.schema();
                size_t fieldIndex = record.fieldIndex( field );
                const avro::NodePtr & fieldSchema = recordSchema->leafAt( fieldIndex );

                size_t nonNullBranch = 0;
                for( size_t i = 0; i < fieldSchema->leaves(); ++i )
                {
                    if( fieldSchema->leafAt( i )->type() != avro::AVRO_NULL )
                    {
                        nonNullBranch = i;
                        break;
                    }
                }

                fieldDatum.selectBranch( nonNullBranch );
            }

            setFieldValue( fieldDatum, value, type, entry );
        }
    }

    std::pair<const void *, size_t> finalize() override
    {
        m_outputBuffer.clear();

        std::unique_ptr<avro::OutputStream> out = avro::memoryOutputStream();
        avro::EncoderPtr encoder = avro::binaryEncoder();
        encoder->init( *out );

        try
        {
            avro::GenericWriter::write( *encoder, *m_datum );
            encoder->flush();
        }
        catch( const std::exception & e )
        {
            CSP_THROW( RuntimeException, "AvroMessageWriter: Encode failed: " << e.what() );
        }

        size_t byteCount = out->byteCount();
        if( byteCount > 0 )
        {
            size_t currentHeaderSize = m_outputBuffer.size();
            m_outputBuffer.resize( currentHeaderSize + byteCount );

            std::unique_ptr<avro::InputStream> in = avro::memoryInputStream( *out );
            const uint8_t * chunk = nullptr;
            size_t len = 0;
            size_t offset = currentHeaderSize;

            while( in->next( &chunk, &len ) )
            {
                std::memcpy( m_outputBuffer.data() + offset, chunk, len );
                offset += len;
            }
        }

        // Reset datum for next message
        m_datum = std::make_unique<avro::GenericDatum>( m_validSchema );

        return { m_outputBuffer.data(), m_outputBuffer.size() };
    }

private:
    void processTickImpl( const OutputDataMapper & dataMapper, const TimeSeriesProvider * sourcets ) override
    {
        dataMapper.apply( *this, sourcets );
    }

    template<typename T>
    void setFieldValue( avro::GenericDatum & datum, const T & value, const CspType & type, const FieldEntry & entry )
    {
        setDatumValue( datum, value );
    }

    void setFieldValue( avro::GenericDatum & datum, const StructPtr & value, const CspType & type, const FieldEntry & entry )
    {
        setDatumValueStruct( datum, value, entry );
    }

    template<typename StorageT>
    void setFieldValue( avro::GenericDatum & datum, const std::vector<StorageT> & value, const CspType & type, const FieldEntry & entry )
    {
        setDatumValueArray( datum, value, type );
    }

    void setDatumValue( avro::GenericDatum & datum, bool value )
    {
        if( datum.type() == avro::AVRO_BOOL )
            datum.value<bool>() = value;
    }

    void setDatumValue( avro::GenericDatum & datum, uint8_t value )
    {
        if( datum.type() == avro::AVRO_INT )
            datum.value<int32_t>() = static_cast<int32_t>( value );
        else if( datum.type() == avro::AVRO_LONG )
            datum.value<int64_t>() = static_cast<int64_t>( value );
    }

    void setDatumValue( avro::GenericDatum & datum, int16_t value )
    {
        if( datum.type() == avro::AVRO_INT )
            datum.value<int32_t>() = static_cast<int32_t>( value );
        else if( datum.type() == avro::AVRO_LONG )
            datum.value<int64_t>() = static_cast<int64_t>( value );
    }

    void setDatumValue( avro::GenericDatum & datum, int32_t value )
    {
        if( datum.type() == avro::AVRO_INT )
            datum.value<int32_t>() = value;
        else if( datum.type() == avro::AVRO_LONG )
            datum.value<int64_t>() = static_cast<int64_t>( value );
    }

    void setDatumValue( avro::GenericDatum & datum, int64_t value )
    {
        if( datum.type() == avro::AVRO_LONG )
            datum.value<int64_t>() = value;
        else if( datum.type() == avro::AVRO_INT )
            datum.value<int32_t>() = static_cast<int32_t>( value );
    }

    void setDatumValue( avro::GenericDatum & datum, double value )
    {
        if( datum.type() == avro::AVRO_DOUBLE )
            datum.value<double>() = value;
        else if( datum.type() == avro::AVRO_FLOAT )
            datum.value<float>() = static_cast<float>( value );
    }

    void setDatumValue( avro::GenericDatum & datum, const std::string & value )
    {
        if( datum.type() == avro::AVRO_STRING )
            datum.value<std::string>() = value;
        else if( datum.type() == avro::AVRO_BYTES )
        {
            std::vector<uint8_t> bytes( value.begin(), value.end() );
            datum.value<std::vector<uint8_t>>() = std::move( bytes );
        }
    }

    void setDatumValue( avro::GenericDatum & datum, const csp::Date & value )
    {
        // Avro date is days since Unix epoch (1970-01-01)
        if( datum.type() == avro::AVRO_INT )
        {
            // Use the same logic as reader for symmetry
            Date epoch = Date::fromYYYYMMDD( "1970-01-01" );
            TimeDelta delta = value - epoch;
            int32_t days = delta.days();
            datum.value<int32_t>() = days;
        }
        else if( datum.type() == avro::AVRO_STRING )
            datum.value<std::string>() = value.asYYYYMMDD();
    }

    void setDatumValue( avro::GenericDatum & datum, const csp::DateTime & value )
    {
        int64_t convertedValue;
        switch( m_datetimeWireType )
        {
            case utils::DateTimeWireType::UINT64_NANOS:
                convertedValue = value.asNanoseconds();
                break;
            case utils::DateTimeWireType::UINT64_MICROS:
                convertedValue = value.asMicroseconds();
                break;
            case utils::DateTimeWireType::UINT64_MILLIS:
                convertedValue = value.asMilliseconds();
                break;
            case utils::DateTimeWireType::UINT64_SECONDS:
                convertedValue = value.asSeconds();
                break;
            default:
                CSP_THROW( NotImplemented, "datetime wire type " << m_datetimeWireType << " not supported for avro msg publishing" );
        }

        if( datum.type() == avro::AVRO_LONG )
            datum.value<int64_t>() = convertedValue;
        else if( datum.type() == avro::AVRO_INT )
            datum.value<int32_t>() = static_cast<int32_t>( convertedValue );
    }

    void setDatumValue( avro::GenericDatum & datum, const csp::CspEnum & value )
    {
        if( datum.type() == avro::AVRO_ENUM )
        {
            avro::GenericEnum & e = datum.value<avro::GenericEnum>();
            e.set( value.name() );
        }
        else if( datum.type() == avro::AVRO_STRING )
        {
            datum.value<std::string>() = value.name();
        }
        else if( datum.type() == avro::AVRO_INT )
        {
            datum.value<int32_t>() = static_cast<int32_t>( value.value() );
        }
    }

    void setDatumValueStruct( avro::GenericDatum & datum, const StructPtr & struct_, const FieldEntry & entry )
    {
        if( datum.type() != avro::AVRO_RECORD )
            return;

        avro::GenericRecord & record = datum.value<avro::GenericRecord>();

        if( !entry.nestedFields )
            return;

        for( auto & nestedEntry : *entry.nestedFields )
        {
            if( !nestedEntry.sField->isSet( struct_.get() ) )
                continue;

            if( !record.hasField( nestedEntry.outField ) )
                continue;

            avro::GenericDatum & fieldDatum = record.field( nestedEntry.outField );

            SupportedCspTypeSwitch::template invoke<SupportedArrayCspTypeSwitch>(
                nestedEntry.sField->type().get(),
                [&]( auto tag )
                {
                    using T = typename decltype( tag )::type;
                    setFieldValue( fieldDatum, nestedEntry.sField->value<T>( struct_.get() ), *nestedEntry.sField->type(), nestedEntry );
                } );
        }
    }

    template<typename StorageT>
    void setDatumValueArray( avro::GenericDatum & datum, const std::vector<StorageT> & value, const CspType & type )
    {
        if( datum.type() != avro::AVRO_ARRAY )
            return;

        avro::GenericArray & arr = datum.value<avro::GenericArray>();
        arr.value().clear();

        const avro::NodePtr & elemSchema = arr.schema()->leafAt( 0 );

        for( size_t i = 0; i < value.size(); ++i )
        {
            avro::GenericDatum elemDatum( elemSchema );
            using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
            // If element is a union, select non-null branch
            if( elemDatum.isUnion() && elemDatum.type() == avro::AVRO_NULL )
            {
                size_t nonNullBranch = 0;
                for( size_t j = 0; j < elemSchema->leaves(); ++j )
                {
                    if( elemSchema->leafAt( j )->type() != avro::AVRO_NULL )
                    {
                        nonNullBranch = j;
                        break;
                    }
                }
                elemDatum.selectBranch( nonNullBranch );
            }
            setDatumValue( elemDatum, static_cast<ElemT>( value[i] ) );
            arr.value().push_back( elemDatum );
        }
    }

    avro::ValidSchema                   m_validSchema;
    std::unique_ptr<avro::GenericDatum> m_datum;
    std::vector<uint8_t>                m_outputBuffer;
    utils::DateTimeWireType             m_datetimeWireType;
};

}

#endif
