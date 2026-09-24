#include <csp/python/PyStructFromJson.h>


namespace csp::python {

StructPtr structFromJson( const StructMetaPtr& struct_meta, const rapidjson::Value& jValue )
{
    StructPtr s = struct_meta->create();

    if( !jValue.IsObject() ) {
       CSP_THROW( TypeError, "Expected a json object for type " << struct_meta->name() );
    }

    // Iterate all the fields of a valid JSON object.
    for (auto jit = jValue.MemberBegin(); jit != jValue.MemberEnd(); ++jit) {
       auto& field = struct_meta->field(jit->name.GetString());
       if (!field) {
        CSP_THROW( KeyError, "Unexpected key " << jit -> name.GetString() << " for type " 
                        << struct_meta -> name() );
       }
       switchCspType(field->type(), [&](auto tag) {
         using CType = typename decltype(tag)::type;
         auto * typedField = static_cast<const typename StructField::upcast<CType>::type *>( field.get() );

         //to_json writes null for fields that are set to None
         if( typedField -> isOptional() && jit->value.IsNull() )
         {
             typedField -> setNone( s.get() );
             typedField -> clearValue( s.get() );
             return;
         }

         typedField -> setValue( s.get(), fromJson<CType>( jit->value, *field->type() ) );
        }
       );
    }
    
    if (!s->validate()) {
        CSP_THROW(ValueError, "struct " << struct_meta->name() << " is not valid " << "required fields " <<
             s->formatAllUnsetStrictFields() << " were not set on init");
    }

    return s;
}
}
