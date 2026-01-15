#ifndef _IN_CSP_ADAPTERS_UTILS_AVROMESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_AVROMESSAGESTRUCTCONVERTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/core/Hash.h>
#include <csp/engine/Dictionary.h>

// Workaround for avro-cpp from conda-forge on Windows
// conda-forge's avro-cpp has outdated fmt::formatter<avro::Name> without const
// We define the correct const version first, then disable the redefinition error
#ifdef _MSC_VER
#include <fmt/format.h>
// Forward declare avro::Name
namespace avro { class Name; }
// Define correct const-correct formatter before avro headers include the broken one
namespace fmt {
template<>
struct formatter<avro::Name, char> : formatter<std::string, char> {
    template<typename FormatContext>
    auto format(const avro::Name &n, FormatContext &ctx) const -> decltype(ctx.out());
};
}
// Disable C2766 error (redefinition) - our version will be used
#pragma warning(push)
#pragma warning(disable: 2766)
#endif

#include <avro/Decoder.hh>
#include <avro/Generic.hh>
#include <avro/Schema.hh>
#include <avro/ValidSchema.hh>

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <list>
#include <string>
#include <unordered_map>

namespace csp::adapters::utils
{

class AvroMessageStructConverter: public MessageStructConverter
{
public:
    AvroMessageStructConverter( const CspTypePtr & type, const Dictionary & properties );

    csp::StructPtr asStruct( void * bytes, size_t size ) final;

    static MessageStructConverter * create( const CspTypePtr & type, const Dictionary & properties )
    {
        return new AvroMessageStructConverter( type, properties );
    }

private:
    struct FieldEntry
    {
        StructFieldPtr sField;
        std::string avroFieldName;
        std::shared_ptr<std::unordered_map<std::string, FieldEntry>> nestedFields;
    };

    using Fields = std::unordered_map<std::string, FieldEntry>;

    Fields buildFields( const CspStructType & type, const Dictionary & fieldMap );

    void convertAvroValue( const avro::GenericDatum & datum, const FieldEntry & entry, StructPtr & struct_ );

    template<typename T>
    T extractValue( const avro::GenericDatum & datum, const char * fieldname );

    template<typename StorageT>
    std::vector<StorageT> extractArray( const avro::GenericDatum & datum, const char * fieldname, const CspType & elemType );

    Fields             m_fields;
    DateTimeWireType   m_datetimeType;
    avro::ValidSchema  m_schema;
};

}

#endif
