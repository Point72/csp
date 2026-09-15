#include <csp/python/PyStructFromDict.h>

namespace csp::python{
    
StructPtr structFromDict (const StructMetaPtr& struct_meta, PyObject* dict) {
    PyObject* py_key;
    PyObject* py_value;
    Py_ssize_t ppos = 0;

    if (!PyDict_Check(dict)) {
        CSP_THROW(TypeError, "Wrong type used in `from_dict`, expected a dict.");
    }

    StructPtr s = struct_meta->create();
    while( PyDict_Next( dict, &ppos, &py_key, &py_value ) )
   {
    if( !PyUnicode_Check( py_key ) )
        CSP_THROW( KeyError, "Unexpected key " << PyObjectPtr::incref( py_key )
                             << " for type " << struct_meta -> name() );

    auto & field = struct_meta -> field( PyUnicode_AsUTF8( py_key ) );
    if( !field )
        CSP_THROW( KeyError, "Unexpected key " << PyObjectPtr::incref( py_key )
                             << " for type " << struct_meta -> name() );

    switchCspType(field->type(), [&] (auto tag)
    {
      using CType = typename decltype( tag )::type;
      auto * typedField = static_cast<const typename StructField::upcast<CType>::type *>( field.get() );

      //optional fields accept None, same as PyStruct_setattrs
      if( typedField -> isOptional() && py_value == Py_None )
      {
          typedField -> setNone( s.get() );
          typedField -> clearValue( s.get() );
          return;
      }

      if constexpr (std::is_same_v<CType, StructPtr>)
      {
        auto& nestedMeta = static_cast<const CspStructType &>(* field->type()).meta();
        typedField -> setValue(s.get(), structFromDict(nestedMeta, py_value));
      }
      else if constexpr( std::is_same_v<CType, CspEnum> )
      {
        //enums arrive as name strings unless to_dict was called with preserve_enums
        auto & enumMeta = static_cast<const CspEnumType &>( *field -> type() ).meta();
        if( PyUnicode_Check( py_value ) )
            typedField -> setValue( s.get(), enumMeta -> fromString( PyUnicode_AsUTF8( py_value ) ) );
        else
            typedField -> setValue( s.get(), fromPython<CType>( py_value, *field -> type() ) );
      }
      else if constexpr( std::is_same_v<CType, std::vector<CspEnum>> )
      {
        //convert element-wise, entries may be names or enum objects
        auto & elemType = static_cast<const CspArrayType &>( *field -> type() ).elemType();
        auto & enumMeta = static_cast<const CspEnumType &>( *elemType ).meta();

        PyObjectPtr seq = PyObjectPtr::own( PySequence_Fast( py_value, "expected a sequence of enum values" ) );
        if( !seq.ptr() )
            CSP_THROW( PythonPassthrough, "" );

        std::vector<CspEnum> out;
        Py_ssize_t n = PySequence_Fast_GET_SIZE( seq.ptr() );
        out.reserve( n );
        for( Py_ssize_t i = 0; i < n; ++i )
        {
            PyObject * elem = PySequence_Fast_GET_ITEM( seq.ptr(), i );
            out.push_back( PyUnicode_Check( elem ) ? enumMeta -> fromString( PyUnicode_AsUTF8( elem ) )
                                                   : fromPython<CspEnum>( elem, *elemType ) );
        }
        typedField -> setValue( s.get(), out );
      }
      else
      {
      typedField -> setValue(s.get(), fromPython<CType>(py_value, *field->type()));
      }
   });
 
    }

    //we bypass __init__ so run the strict-struct check ourselves
    if( !s -> validate() ) [[unlikely]]
        CSP_THROW( ValueError, "Struct " << struct_meta -> name() << " is not valid; required fields "
                               << s -> formatAllUnsetStrictFields() << " were not set on init" );

    return s;
}

}
