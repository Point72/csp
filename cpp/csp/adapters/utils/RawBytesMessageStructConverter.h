#ifndef _IN_CSP_ADAPTERS_UTILS_RAWBYTESMESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_RAWBYTESMESSAGESTRUCTCONVERTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/engine/Dictionary.h>

namespace csp::adapters::utils
{

class RawBytesMessageStructConverter : public MessageStructConverter
{
public:

    RawBytesMessageStructConverter( const CspTypePtr & type, const Dictionary & properties );

    csp::StructPtr asStruct( void * bytes, size_t size ) override;

    static MessageStructConverter * create( const CspTypePtr & type, const Dictionary & properties )
    {
        return new RawBytesMessageStructConverter( type, properties );
    }

private:
    const StringStructField * m_targetField;
};

}

#endif
