#ifndef _IN_CSP_ADAPTERS_UTILS_MESSAGEWRITER_H
#define _IN_CSP_ADAPTERS_UTILS_MESSAGEWRITER_H

#include <csp/adapters/utils/MessageEnums.h>
#include <csp/core/Enum.h>
#include <csp/engine/CspType.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

namespace csp::adapters::utils
{

//This is used to map data from an output adapter- > message writer ( can have multiple ouput adapters writing to same message )
class OutputDataMapper
{
public:
    OutputDataMapper( const CspTypePtr & type, const Dictionary & fieldMap );

    template<typename MessageWriterT>
    void apply( MessageWriterT & writer, const TimeSeriesProvider * sourcets ) const;

    bool hasFields() const { return m_hasFields; }

    struct FieldEntry
    {
        std::string    outField;
        StructFieldPtr sField;
        std::shared_ptr<std::vector<FieldEntry>> nestedFields; //for nested structs
    };

    using Fields = std::vector<FieldEntry>;

private:
    Fields populateStructFields( const StructMetaPtr & structMeta, const Dictionary & field_map );

    //struct outputs
    template<typename MessageWriterT>
    void applyStruct( MessageWriterT & writer, const Struct * struct_ ) const;

    //non-struct outputs
    template<typename MessageWriterT,typename T>
    void applyField( MessageWriterT & writer, const T & value ) const;

    CspTypePtr    m_type;
    bool          m_hasFields;

    //Struct specific
    StructMetaPtr m_structMeta;
    Fields        m_messageFields;

    //non-struct specific
    std::string m_messageFieldName;
};

using OutputDataMapperPtr=std::shared_ptr<OutputDataMapper>;

// Derived types will be used to create and write timeseries data -> target message protocol ( ie JSON, proto )
// and convert it to bytes for the output adapter
class MessageWriter
{
public:
    using FieldEntry = OutputDataMapper::FieldEntry;

    MessageWriter() {}
    virtual ~MessageWriter();

    //returns the finalized message as bytes
    virtual std::pair<const void *,size_t> finalize() = 0;

    void processTick( const OutputDataMapper & dataMapper, const TimeSeriesProvider * sourcets )
    {
        if( dataMapper.hasFields() )
            processTickImpl( dataMapper, sourcets );
    }

private:
    virtual void processTickImpl( const OutputDataMapper & dataMapper, const TimeSeriesProvider * sourcets ) = 0;
};

using MessageWriterPtr=std::shared_ptr<MessageWriter>;

template<typename MessageWriterT>
inline void OutputDataMapper::apply( MessageWriterT & writer, const TimeSeriesProvider * ts ) const
{
    if( !m_hasFields )
        return;

    if( ts -> type() -> type() == CspType::Type::STRUCT )
        applyStruct( writer, ts -> lastValueTyped<StructPtr>().get() );
    else
    {
        MessageWriterT::SupportedCspTypeSwitch::template invoke<typename MessageWriterT::SupportedArrayCspTypeSwitch>( ts -> type(),
                                                                 [&]( auto tag )
                                                                 {
                                                                     applyField( writer, ts -> lastValueTyped<typename decltype(tag)::type>() );
                                                                 } );
    }
}

template<typename MessageWriterT,typename T>
inline void OutputDataMapper::applyField( MessageWriterT & writer, const T & value ) const
{
    CSP_ASSERT( m_type -> type() != CspType::Type::STRUCT );

    if( !m_messageFieldName.empty() )
        writer.setField( m_messageFieldName, value, *m_type, {} );
}

template<typename MessageWriterT>
inline void OutputDataMapper::applyStruct( MessageWriterT & writer, const Struct * struct_ ) const
{
    CSP_ASSERT( m_type -> type() == CspType::Type::STRUCT );

    for( auto & entry : m_messageFields )
    {
        if( !entry.sField -> isSet( struct_ ) )
            continue;

        using SwitchT = typename MessageWriterT::SupportedCspTypeSwitch;

        SwitchT::template invoke<typename MessageWriterT::SupportedArrayCspTypeSwitch>(
            entry.sField -> type().get(),
            [ & ]( auto tag )
            {
                using T = typename decltype(tag)::type;
                writer.setField( entry.outField, entry.sField -> value<T>( struct_ ), *entry.sField -> type(), entry );
            } );
    };
}

//This ensures we dont recreate duplicate writers unnecessarily
class OutputDataMapperCache
{
public:
    OutputDataMapperCache();
    
    static OutputDataMapperCache & instance();

    OutputDataMapperPtr create( const CspTypePtr &, const Dictionary & fieldMap );

private:
    using CacheKey = std::pair<const CspType*,Dictionary>;
    using Cache = std::unordered_map<CacheKey,OutputDataMapperPtr,csp::hash::hash_pair>;

    std::mutex m_cacheMutex;
    Cache      m_cache;
};

}

#endif
