#include <csp/core/Platform.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/Struct.h>
#include <csp/python/PyStruct.h>

#include <algorithm>
#include <cstdint>

namespace csp::python
{

extern "C" CSPIMPL_EXPORT int64_t csp_numba_struct_enum_field_value( const void * struct_ptr, int64_t field_offset ) noexcept
{
    const char * bytes = static_cast<const char *>( struct_ptr );
    return reinterpret_cast<const CspEnum *>( bytes + field_offset ) -> value();
}

extern "C" CSPIMPL_EXPORT int8_t csp_numba_struct_enum_field_set(
    void * struct_ptr, int64_t field_offset, int64_t value, const void * struct_type ) noexcept
{
    const auto * pyStructMeta = static_cast<const PyStructMeta *>( struct_type );
    if( !struct_ptr || !pyStructMeta || !pyStructMeta -> structMeta || field_offset < 0 )
        return 0;

    const auto & fields = pyStructMeta -> structMeta -> fields();
    auto fieldIt = std::find_if( fields.begin(), fields.end(), [field_offset]( const auto & field )
    {
        return field -> offset() == static_cast<size_t>( field_offset );
    } );
    if( fieldIt == fields.end() || ( *fieldIt ) -> type() -> type() != CspType::Type::ENUM )
        return 0;

    try
    {
        const auto * enumType = static_cast<const CspEnumType *>( ( *fieldIt ) -> type().get() );
        ( *fieldIt ) -> setValue<CspEnum>( static_cast<Struct *>( struct_ptr ), enumType -> meta() -> create( value ) );
        return 1;
    }
    catch( ... )
    {
        return 0;
    }
}

}  // namespace csp::python
