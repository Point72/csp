#ifndef _IN_CSP_ADAPTERS_UTILS_PROTOBUFMESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_PROTOBUFMESSAGESTRUCTCONVERTER_H

#include <csp/engine/Struct.h>
#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/adapters/utils/ProtobufHelper.h>
#include <memory>
#include <vector>

namespace google::protobuf
{
class Descriptor;
class FieldDescriptor;
class Message;
};

namespace cms { class Message; };
namespace csp { class Dictionary; }

namespace csp::adapters::utils
{

//Use for processing protobuf bytes as the payload
class ProtobufMessageStructConverter final : public MessageStructConverter
{
public:
    ProtobufMessageStructConverter( const CspTypePtr & type, const Dictionary & properties );

    csp::StructPtr asStruct( void * bytes, size_t size ) override;

    MsgProtocol protocol() const override { return MsgProtocol::PROTOBUF; }

    static MessageStructConverter * create( const CspTypePtr & type, const Dictionary & properties )
    {
        return new ProtobufMessageStructConverter( type, properties );
    }

private:

    const google::protobuf::Descriptor * m_protoDesc;
    utils::ProtobufStructMapper m_protoMapper;
};

}

#endif
