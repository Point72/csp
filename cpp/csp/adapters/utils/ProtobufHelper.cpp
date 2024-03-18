#include <csp/adapters/utils/ProtobufHelper.h>
#include <csp/core/Exception.h>
#include <csp/core/System.h>
#include <csp/engine/PartialSwitchCspType.h>

namespace proto = google::protobuf;

namespace csp::adapters::utils
{

//For some reason the default proto::Importer does not import well know types, so we had to write
//our own version
class ProtoImporterAux : public proto::compiler::MultiFileErrorCollector
{
public:
    ProtoImporterAux( proto::compiler::SourceTree *source_tree )
            : m_wellKnownTypesDatabase( *proto::DescriptorPool::generated_pool() ),
              m_database( source_tree, &m_wellKnownTypesDatabase ),
              m_pool( &m_database, m_database.GetValidationErrorCollector())
    {
        m_pool.EnforceWeakDependencies( true );
        m_database.RecordErrorsTo( this );
    }

    const proto::FileDescriptor *Import( const std::string & filename )
    {
        return m_pool.FindFileByName( filename );
    }

    void AddError( const std::string& filename, int line, int column,
                   const std::string& message ) override
    {
        CSP_THROW( RuntimeException, "Failed to load proto schema " << filename << ":" << line << ":" << column << ": " << message );
    }
    
private:
    proto::DescriptorPoolDatabase                 m_wellKnownTypesDatabase;
    proto::compiler::SourceTreeDescriptorDatabase m_database;
    proto::DescriptorPool                         m_pool;
};

ProtobufHelper::ProtobufHelper()
{
    m_protoImporter = std::make_shared<ProtoImporterAux>( &m_protoSourceTree );
    m_protoMsgFactory.SetDelegateToGeneratedFactory( true );
}

const proto::FileDescriptor * ProtobufHelper::import( const std::string & schemaDir, const std::string & filename )
{
    {
        std::lock_guard<std::mutex> guard( m_schemaDirLock );
        if( m_schemaDirs.find( schemaDir ) == m_schemaDirs.end() )
        {
            m_protoSourceTree.MapPath( "", schemaDir );
            m_schemaDirs.emplace( schemaDir );
        }
    }

    return m_protoImporter -> Import( filename );
}

const proto::Message * ProtobufHelper::getMessage( const proto::Descriptor* type )
{
    return m_protoMsgFactory.GetPrototype( type );
}

ProtobufHelper & ProtobufHelper::instance()
{
    static ProtobufHelper s_instance;
    return s_instance;
}

proto::FieldDescriptor::CppType ProtobufHelper::cspToProtoCppType( const CspType & type )
{
    switch( type.type() )
    {
        //NOTE if new types are added make sure parsing code switch statement below is updated
        case CspType::Type::BOOL:   return proto::FieldDescriptor::CPPTYPE_BOOL;
        case CspType::Type::INT32:  return proto::FieldDescriptor::CPPTYPE_INT32;
        case CspType::Type::UINT32: return proto::FieldDescriptor::CPPTYPE_UINT32;
        case CspType::Type::INT64:  return proto::FieldDescriptor::CPPTYPE_INT64;
        case CspType::Type::UINT64: return proto::FieldDescriptor::CPPTYPE_UINT64;
        case CspType::Type::DOUBLE: return proto::FieldDescriptor::CPPTYPE_DOUBLE;
        case CspType::Type::STRING: return proto::FieldDescriptor::CPPTYPE_STRING;
        case CspType::Type::STRUCT: return proto::FieldDescriptor::CPPTYPE_MESSAGE;
        case CspType::Type::ARRAY: 
        {
            auto elemType = static_cast<const CspArrayType &>( type ).elemType();
            return cspToProtoCppType( *elemType );
        }

        default:
            CSP_THROW( TypeError, "Struct field type " << type.type() << " not currently mappable to proto field" );
    }
}

bool ProtobufHelper::isCoercible( proto::FieldDescriptor::CppType src, CspType::Type dest )
{
    switch( src )
    {
        case proto::FieldDescriptor::CPPTYPE_INT32:  return ( dest == CspType::Type::UINT32 || dest == CspType::Type::INT64 || dest == CspType::Type::UINT64 );
        case proto::FieldDescriptor::CPPTYPE_UINT32: return ( dest == CspType::Type::INT32  || dest == CspType::Type::INT64 || dest == CspType::Type::UINT64 );
        case proto::FieldDescriptor::CPPTYPE_INT64:  return dest == CspType::Type::UINT64;
        case proto::FieldDescriptor::CPPTYPE_UINT64: return dest == CspType::Type::INT64;
        case proto::FieldDescriptor::CPPTYPE_FLOAT:  return dest == CspType::Type::DOUBLE;
        case proto::FieldDescriptor::CPPTYPE_ENUM:   return dest == CspType::Type::STRING;
        default:
            return false;
    }
}

template<>
int32_t ProtobufHelper::coercedValue( const proto::Reflection * access, const proto::Message & message, const proto::FieldDescriptor * descr, int index )
{
    switch( descr -> cpp_type() )
    {
        case proto::FieldDescriptor::CPPTYPE_INT32:  return index == -1 ? access -> GetInt32( message, descr ) : access -> GetRepeatedInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_UINT32:
        {
            uint32_t v = index == -1 ? access -> GetUInt32( message, descr ) : access -> GetRepeatedUInt32( message, descr, index );
            if( v > uint32_t(std::numeric_limits<int32_t>::max()) )
                CSP_THROW( RangeError, "coercion out of range for UINT32 value into INT32 value for proto msg type " << message.GetTypeName() << " field " << descr -> name() );
            return ( int32_t ) v;
        }
        default:
            CSP_THROW( TypeError, "Attempting to coerce proto field type " << descr -> cpp_type_name() << " to int32_t" );
    }
}

template<>
uint32_t ProtobufHelper::coercedValue( const proto::Reflection * access, const proto::Message & message, const proto::FieldDescriptor * descr, int index )
{
    switch( descr -> cpp_type() )
    {
        case proto::FieldDescriptor::CPPTYPE_UINT32:  return index == -1 ? access -> GetUInt32( message, descr ) : access -> GetRepeatedUInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_INT32:
        {
            int32_t v = index == -1 ? access -> GetInt32( message, descr ) : access -> GetRepeatedInt32( message, descr, index );
            if( v < 0 )
                CSP_THROW( RangeError, "coercion out of range for INT32 value into uint32_t value for proto msg type " << message.GetTypeName() << " field " << descr -> name() );
            return ( uint32_t ) v;
        }
        default:
            CSP_THROW( TypeError, "Attempting to coerce proto field type " << descr -> cpp_type_name() << " to uint32_t" );
    }
}

template<>
int64_t ProtobufHelper::coercedValue( const proto::Reflection * access, const proto::Message & message, const proto::FieldDescriptor * descr, int index )
{
    switch( descr -> cpp_type() )
    {
        case proto::FieldDescriptor::CPPTYPE_INT64:  return index == -1 ? access -> GetInt64( message, descr ) : access -> GetRepeatedInt64( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_UINT64:
        {
            uint64_t v = index == -1 ? access -> GetUInt64( message, descr ) : access -> GetRepeatedUInt64( message, descr, index );
            if( v > uint64_t(std::numeric_limits<int64_t>::max()) )
                CSP_THROW( RangeError, "coercion out of range for UINT64 value into int64_t value for proto msg type " << message.GetTypeName() << " field " << descr -> name() );
            return ( int64_t ) v;
        }
        case proto::FieldDescriptor::CPPTYPE_UINT32: return index == -1 ? access -> GetUInt32( message, descr ) : access -> GetRepeatedUInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_INT32:  return index == -1 ? access -> GetInt32( message, descr ) : access -> GetRepeatedInt32( message, descr, index );

        default:
            CSP_THROW( TypeError, "Attempting to coerce proto field type " << descr -> cpp_type_name() << " to int64_t" );
    }
}

template<>
uint64_t ProtobufHelper::coercedValue( const proto::Reflection * access, const proto::Message & message, const proto::FieldDescriptor * descr, int index )
{
    switch( descr -> cpp_type() )
    {
        case proto::FieldDescriptor::CPPTYPE_UINT64:  return index == -1 ? access -> GetUInt64( message, descr ) : access -> GetRepeatedUInt64( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_INT64:
        {
            int64_t v = index == -1 ? access -> GetInt64( message, descr ) : access -> GetRepeatedInt64( message, descr, index );
            if( v < 0 )
                CSP_THROW( RangeError, "coercion out of range for INT64 value into uint64_t value for proto msg type " << message.GetTypeName() << " field " << descr -> name() );
            return ( uint64_t ) v;
        }
        case proto::FieldDescriptor::CPPTYPE_UINT32:  return index == -1 ? access -> GetUInt32( message, descr ) : access -> GetRepeatedUInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_INT32:
        {
            int32_t v = index == -1 ? access -> GetInt32( message, descr ) : access -> GetRepeatedInt32( message, descr, index );
            if( v < 0 )
                CSP_THROW( RangeError, "coercion out of range for INT32 value into uint64_t value for proto msg type " << message.GetTypeName() << " field " << descr -> name() );
            return ( uint64_t ) v;
        }

        default:
            CSP_THROW( TypeError, "Attempting to coerce proto field type " << descr -> cpp_type_name() << " to int64_t" );
    }
}

template<>
double ProtobufHelper::coercedValue( const proto::Reflection * access, const proto::Message & message, const proto::FieldDescriptor * descr, int index )
{
    switch( descr -> cpp_type() )
    {
        case proto::FieldDescriptor::CPPTYPE_INT32:  return index == -1 ? access -> GetInt32( message, descr ) : access -> GetRepeatedInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_UINT32: return index == -1 ? access -> GetUInt32( message, descr ) : access -> GetRepeatedUInt32( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_INT64:  return index == -1 ? access -> GetInt64( message, descr ) : access -> GetRepeatedInt64( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_UINT64: return index == -1 ? access -> GetUInt64( message, descr ) : access -> GetRepeatedUInt64( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_FLOAT:  return index == -1 ? access -> GetFloat( message, descr ) : access -> GetRepeatedFloat( message, descr, index );
        case proto::FieldDescriptor::CPPTYPE_DOUBLE: return index == -1 ? access -> GetDouble( message, descr ) : access -> GetRepeatedDouble( message, descr, index );

        default:
            CSP_THROW( TypeError, "Attempting to coerce proto field type " << descr -> cpp_type_name() << " to double" );
    }
}

void ProtobufStructMapper::init( const CspTypePtr & type, const Dictionary & fieldMap, const google::protobuf::Descriptor * protoDesc )
{
    m_fields = buildFields( type, fieldMap, protoDesc );
}

ProtobufStructMapper::Fields ProtobufStructMapper::buildFields( const CspTypePtr & type, const Dictionary & fieldMap, const google::protobuf::Descriptor * protoDesc )
{
    auto structType = std::static_pointer_cast<const CspStructType>( type );
    auto structMeta = structType -> meta();

    Fields fields;
    for( auto it = fieldMap.begin(); it != fieldMap.end(); ++it )
    {
        std::string protoField  = it.key();
        std::string structField;
        DictionaryPtr subFieldMap;
        if( it.hasValue<std::string>() )
            structField = it.value<std::string>();
        else
        {
            //for sub-structs we expect value to by a dict of { "structfield" : fieldMap }
            auto subdict = it.value<DictionaryPtr>();
            if( subdict -> size() != 1 )
                CSP_THROW( ValueError, "Expected sub-fieldmap for incoming protofield " << protoField << " to have a single key : map entry" );

            structField = subdict -> begin().key();
            subFieldMap = subdict -> begin().value<DictionaryPtr>();
        }

        auto sField = structMeta -> field( structField );
        if( !sField )
            CSP_THROW( ValueError, "field " << structField << " is not a valid field on struct type " << structMeta -> name() );

        const proto::FieldDescriptor * protoFieldDesc = protoDesc -> FindFieldByName( protoField );
        if( !protoFieldDesc )
            CSP_THROW( ValueError, "field '" << protoField << "' is not a valid field on proto descriptor for " << protoDesc -> name() );

        auto protoType = ProtobufHelper::cspToProtoCppType( *sField -> type() );
        if( ( protoType != protoFieldDesc -> cpp_type() && !ProtobufHelper::isCoercible( protoFieldDesc -> cpp_type(), sField -> type() -> type() ) ) ||
            ( sField -> type() -> type() == CspType::Type::ARRAY && !protoFieldDesc -> is_repeated() ) )
        {
            CSP_THROW( TypeError, "Type mismatch on struct field " << structField << " with type " << sField -> type() -> type() << " and protofield " <<
                       protoField << " with type " << ( protoFieldDesc -> is_repeated() ? "repeated " : "" ) << protoFieldDesc -> cpp_type_name() );
        }

        std::shared_ptr<Fields> subfields;
        if( sField -> type() -> type() == CspType::Type::STRUCT )
        {
            if( !subFieldMap )
                CSP_THROW( ValueError, "invalid field_map entry for sub-struct field " << sField -> fieldname() << " on struct type " << structMeta -> name() );
            subfields = std::make_shared<Fields>( buildFields( sField -> type(), *subFieldMap, protoFieldDesc -> message_type() ) );
        }
        fields.emplace_back( FieldEntry{ protoFieldDesc, sField.get(), subfields } );
    }

    return fields;
}

using SupportedArrayCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                         csp::CspType::Type::INT32,
                                                         csp::CspType::Type::UINT32,
                                                         csp::CspType::Type::INT64,
                                                         csp::CspType::Type::UINT64,
                                                         csp::CspType::Type::DOUBLE,
                                                         csp::CspType::Type::STRING
                                                         >;

template<typename T>
static T extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index );

template<>
bool extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return protoMsg.GetReflection() -> GetRepeatedBool( protoMsg, field, index );
}

template<>
int32_t extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return ProtobufHelper::coercedValue<int32_t>( protoMsg.GetReflection(), protoMsg, field, index );
}

template<>
uint32_t extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return ProtobufHelper::coercedValue<uint32_t>( protoMsg.GetReflection(), protoMsg, field, index );
}

template<>
int64_t extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return ProtobufHelper::coercedValue<int64_t>( protoMsg.GetReflection(), protoMsg, field, index );
}

template<>
uint64_t extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return ProtobufHelper::coercedValue<uint64_t>( protoMsg.GetReflection(), protoMsg, field, index );
}

template<>
double extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return ProtobufHelper::coercedValue<double>( protoMsg.GetReflection(), protoMsg, field, index );
}

template<>
std::string extractRepeatedValue( const proto::Message & protoMsg, const proto::FieldDescriptor * field, int index )
{
    return protoMsg.GetReflection() -> GetRepeatedString( protoMsg, field, index );
}

void ProtobufStructMapper::mapProtoToStruct( StructPtr & struct_, const proto::Message & protoMsg, const Fields & fields )
{
    const proto::Reflection * protoAccess = protoMsg.GetReflection();

    for( auto & entry : fields )
    {
        auto * pField = entry.pField;
        auto * sField = entry.sField;

        switch( sField -> type() -> type() )
        {
            case CspType::Type::BOOL:   sField -> setValue<bool>( struct_.get(),        protoAccess -> GetBool( protoMsg, pField ) );  break;
            case CspType::Type::INT32:  sField -> setValue<int32_t>( struct_.get(),     ProtobufHelper::coercedValue<int32_t>( protoAccess,  protoMsg, pField ) ); break;
            case CspType::Type::UINT32: sField -> setValue<uint32_t>( struct_.get(),    ProtobufHelper::coercedValue<uint32_t>( protoAccess, protoMsg, pField ) ); break;
            case CspType::Type::INT64:  sField -> setValue<int64_t>( struct_.get(),     ProtobufHelper::coercedValue<int64_t>( protoAccess,  protoMsg, pField ) ); break;
            case CspType::Type::UINT64: sField -> setValue<uint64_t>( struct_.get(),    ProtobufHelper::coercedValue<uint64_t>( protoAccess, protoMsg, pField ) ); break;
            case CspType::Type::DOUBLE: sField -> setValue<double>( struct_.get(),      ProtobufHelper::coercedValue<double>( protoAccess,   protoMsg, pField ) ); break;
            case CspType::Type::STRING:
            {
                if( pField -> cpp_type() == proto::FieldDescriptor::CPPTYPE_ENUM )
                {
                    sField -> setValue<std::string>( struct_.get(), protoAccess -> GetEnum( protoMsg, pField ) -> name() );
                    break;
                }
                else
                {
                    sField -> setValue<std::string>( struct_.get(), protoAccess -> GetString( protoMsg, pField ) );
                    break;
                }
            }
            case CspType::Type::STRUCT:
            {
                auto subStructMeta = static_cast<const StructStructField*>( sField ) -> meta();
                auto subStruct = subStructMeta -> create();
                mapProtoToStruct( subStruct, protoAccess -> GetMessage( protoMsg, pField ), *entry.subFields );
                sField -> setValue<StructPtr>( struct_.get(), subStruct );
                break;
            }

            case CspType::Type::ARRAY:
            {
                int count = protoAccess -> FieldSize( protoMsg, pField );

                auto elemType = static_cast<const CspArrayType &>( *sField -> type() ).elemType();
                SupportedArrayCspTypeSwitch::invoke( elemType.get(), [&count,&protoMsg,&pField,&sField,&struct_]( auto tag )
                                                     {
                                                         using ElemT    = typename decltype(tag)::type;
                                                         using StorageT = typename CspType::Type::toCArrayStorageType<ElemT>::type;
                                                         std::vector<StorageT> data;
                                                         data.reserve( count );
                                                         for( int i = 0; i < count; ++i )
                                                             data.emplace_back( extractRepeatedValue<ElemT>( protoMsg, pField, i ) );
                                                         sField -> setValue<std::vector<StorageT>>( struct_.get(), std::move( data ) );
                                                     } );
                break;
            }

            default:
                CSP_THROW( TypeError, "Struct field type " << sField -> type() -> type() << " not currently mappable to proto field" );
        }
    }
}

}
