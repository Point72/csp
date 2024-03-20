#include <csp/core/System.h>
#include <csp/engine/CspType.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <google/protobuf/compiler/importer.h>
#include <google/protobuf/dynamic_message.h>
#include <memory>
#include <mutex>
#include <string>

namespace csp::adapters::utils
{

class ProtoImporterAux;

class ProtobufHelper
{
public:

    static ProtobufHelper & instance();

    //dynamic schema/proto file import help
    const google::protobuf::Message * getMessage( const google::protobuf::Descriptor* type );
    const google::protobuf::FileDescriptor * import( const std::string & schemaDir, const std::string & filename );

    //type conversions
    static google::protobuf::FieldDescriptor::CppType cspToProtoCppType( const CspType & type );
    static bool isCoercible( google::protobuf::FieldDescriptor::CppType src, CspType::Type dest );

    template<typename T>
    static T coercedValue( const google::protobuf::Reflection * access, const google::protobuf::Message & message, const google::protobuf::FieldDescriptor * descr, int index = -1 );

private:
    ProtobufHelper();

    google::protobuf::compiler::DiskSourceTree m_protoSourceTree;
    std::shared_ptr<ProtoImporterAux>          m_protoImporter;
    google::protobuf::DynamicMessageFactory    m_protoMsgFactory;
    std::set<std::string>                      m_schemaDirs;
    std::mutex                                 m_schemaDirLock;
};

//Struct mapping.  Currently this is one way, proto -> struct
//havent decided if same mapper will be used for struct -> proto ( output adapter )
class ProtobufStructMapper
{
public:
    ProtobufStructMapper() {}

    void init( const CspTypePtr & type, const Dictionary & fieldMap,
               const google::protobuf::Descriptor * protoDesc );

    void mapProtoToStruct( StructPtr & struct_, const google::protobuf::Message & protoMsg )
    {
        mapProtoToStruct( struct_, protoMsg, m_fields );
    }

private:
    struct FieldEntry;

    struct FieldEntry
    {
        const google::protobuf::FieldDescriptor* pField;
        StructField * sField;
        std::shared_ptr<std::vector<FieldEntry>> subFields; //for nested structs
    };

    using Fields = std::vector<FieldEntry>;

    static Fields buildFields( const CspTypePtr & type, const Dictionary & fieldMap, const google::protobuf::Descriptor * protoDesc );
    static void mapProtoToStruct( StructPtr & struct_, const google::protobuf::Message & protoMsg, const Fields & fields );

    Fields m_fields;
};

//to avoid alloc/dealloc on every callback of bytes msg
struct ReusableBuffer
{
    ~ReusableBuffer()
    {
        delete [] buf;
    }

    unsigned char * buf;
    size_t          len;

    void grow( size_t newlen )
    {
        if( unlikely( newlen > len ) )
        {
            delete [] buf;
            buf = new unsigned char[ newlen ];
            len = newlen;
        }
    }
};

}
