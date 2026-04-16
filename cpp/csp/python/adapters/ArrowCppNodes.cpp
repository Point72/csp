// CppNode implementations for record_batches_to_struct and struct_to_record_batches.
//
// These nodes bridge the Python/CSP graph layer with the C++ Arrow converters.
// RecordBatches are transported across the Python/C++ boundary as PyCapsules
// using the Arrow C Data Interface.

#include <csp/adapters/arrow/RecordBatchToStruct.h>
#include <csp/adapters/arrow/StructToRecordBatch.h>
#include <csp/engine/CppNode.h>
#include <csp/engine/Dictionary.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyNodeWrapper.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>

using namespace csp::adapters::arrow;

namespace csp::cppnodes
{

// PyCapsule destructors for ArrowSchema/ArrowArray (local copies to avoid including ArrowInputAdapter.h)
static void releaseArrowSchemaCapsule( PyObject * capsule )
{
    ArrowSchema * schema = reinterpret_cast<ArrowSchema *>( PyCapsule_GetPointer( capsule, "arrow_schema" ) );
    if( schema -> release != NULL )
        schema -> release( schema );
    free( schema );
}

static void releaseArrowArrayCapsule( PyObject * capsule )
{
    ArrowArray * array = reinterpret_cast<ArrowArray *>( PyCapsule_GetPointer( capsule, "arrow_array" ) );
    if( array -> release != NULL )
        array -> release( array );
    free( array );
}

DECLARE_CPPNODE( record_batches_to_struct )
{
    SCALAR_INPUT( DialectGenericType, schema_ptr );   // PyCapsule of ArrowSchema
    SCALAR_INPUT( StructMetaPtr,      cls );          // target struct type
    SCALAR_INPUT( DictionaryPtr,      properties );   // field_map, numpy config
    TS_INPUT( Generic,                data );          // List[Tuple[capsule, capsule]]
    TS_OUTPUT( Generic );                             // List[StructPtr]

    STATE_VAR( std::unique_ptr<RecordBatchToStructConverter>, s_converter );
    STATE_VAR( std::shared_ptr<::arrow::Schema>,              s_schema );

    INIT_CPPNODE( record_batches_to_struct ) {}

    START()
    {
        // Import the Arrow schema from PyCapsule
        PyObject * pySchemaCapsule = csp::python::toPythonBorrowed( schema_ptr.value() );
        if( !PyCapsule_IsValid( pySchemaCapsule, "arrow_schema" ) )
            CSP_THROW( csp::TypeError, "schema_ptr must be a PyCapsule with name 'arrow_schema'" );

        ArrowSchema * c_schema = reinterpret_cast<ArrowSchema *>( PyCapsule_GetPointer( pySchemaCapsule, "arrow_schema" ) );
        auto schemaResult = ::arrow::ImportSchema( c_schema );
        if( !schemaResult.ok() )
            CSP_THROW( csp::ValueError, "Failed to import Arrow schema: " << schemaResult.status().ToString() );
        s_schema = std::move( schemaResult.ValueUnsafe() );

        // Parse properties
        auto & props = properties.value();
        DictionaryPtr fieldMap;
        if( props -> exists( "field_map" ) )
            fieldMap = props -> get<DictionaryPtr>( "field_map" );

        auto structMeta = cls.value();
        s_converter = std::make_unique<RecordBatchToStructConverter>( s_schema, structMeta, fieldMap );
    }

    INVOKE()
    {
        // data is a list of (schema_capsule, array_capsule) tuples
        PyObject * pyList = csp::python::toPythonBorrowed( data.lastValue<DialectGenericType>() );
        if( !PyList_Check( pyList ) )
            CSP_THROW( csp::TypeError, "Expected list of PyCapsule tuples, got " << Py_TYPE( pyList ) -> tp_name );

        Py_ssize_t numBatches = PyList_Size( pyList );

        // First pass: import all record batches and compute total row count for a single reserve()
        std::vector<std::shared_ptr<::arrow::RecordBatch>> importedBatches;
        importedBatches.reserve( numBatches );
        int64_t totalRows = 0;

        for( Py_ssize_t i = 0; i < numBatches; ++i )
        {
            PyObject * pyTuple = PyList_GET_ITEM( pyList, i );
            if( !PyTuple_Check( pyTuple ) || PyTuple_Size( pyTuple ) != 2 )
                CSP_THROW( csp::TypeError, "Expected tuple of 2 PyCapsules for record batch " << i );

            PyObject * pyArray = PyTuple_GetItem( pyTuple, 1 );
            if( !PyCapsule_IsValid( pyArray, "arrow_array" ) )
                CSP_THROW( csp::TypeError, "Invalid PyCapsule for record batch array at index " << i );

            ArrowArray * c_array = reinterpret_cast<ArrowArray *>( PyCapsule_GetPointer( pyArray, "arrow_array" ) );
            auto rbResult = ::arrow::ImportRecordBatch( c_array, s_schema );
            if( !rbResult.ok() )
                CSP_THROW( csp::ValueError, "Failed to import record batch at index " << i << ": " << rbResult.status().ToString() );

            auto rb = std::move( rbResult.ValueUnsafe() );
            totalRows += rb -> num_rows();
            importedBatches.push_back( std::move( rb ) );
        }

        // Second pass: convert all batches into structs with a single pre-allocated vector
        std::vector<StructPtr> allStructs;
        allStructs.reserve( totalRows );

        for( auto & rb : importedBatches )
        {
            auto structs = s_converter -> convert( *rb );
            allStructs.insert( allStructs.end(),
                               std::make_move_iterator( structs.begin() ),
                               std::make_move_iterator( structs.end() ) );
        }

        // Output as std::vector<StructPtr> to match the List["T"] output buffer type
        // where "T" resolves to a csp.Struct -> CspArrayType(STRUCT) -> std::vector<StructPtr>
        using ArrayT = std::vector<StructPtr>;
        ArrayT & out = unnamed_output().reserveSpace<ArrayT>();
        out = std::move( allStructs );
    }
};

EXPORT_CPPNODE( record_batches_to_struct );

DECLARE_CPPNODE( struct_to_record_batches )
{
    SCALAR_INPUT( StructMetaPtr,      cls );         // source struct type
    SCALAR_INPUT( DictionaryPtr,      properties );  // field_map, numpy config
    TS_INPUT( std::vector<StructPtr>, data );
    TS_OUTPUT( Generic );                        // DialectGenericType (Python list of capsule tuples)

    STATE_VAR( std::unique_ptr<StructToRecordBatchConverter>, s_converter );
    STATE_VAR( int64_t, s_maxBatchSize );

    INIT_CPPNODE( struct_to_record_batches ) {}

    START()
    {
        auto & props = properties.value();
        auto structMeta = cls.value();

        s_maxBatchSize = props -> get<int64_t>( "max_batch_size", 0 );

        DictionaryPtr fieldMap;
        if( props -> exists( "field_map" ) )
            fieldMap = props -> get<DictionaryPtr>( "field_map" );

        s_converter = std::make_unique<StructToRecordBatchConverter>( structMeta, fieldMap );
    }

    INVOKE()
    {
        auto & structs = data.lastValue();
        auto batches = s_converter -> convert( structs, s_maxBatchSize );

        auto py_list = csp::python::PyObjectPtr::own( PyList_New( static_cast<Py_ssize_t>( batches.size() ) ) );

        for( size_t idx = 0; idx < batches.size(); ++idx )
        {
            ArrowSchema * rb_schema = reinterpret_cast<ArrowSchema *>( malloc( sizeof( ArrowSchema ) ) );
            ArrowArray * rb_array   = reinterpret_cast<ArrowArray *>( malloc( sizeof( ArrowArray ) ) );

            ::arrow::Status st = ::arrow::ExportRecordBatch( *batches[idx], rb_array, rb_schema );
            if( !st.ok() )
            {
                free( rb_schema );
                free( rb_array );
                CSP_THROW( csp::ValueError, "Failed to export RecordBatch at index " << idx << ": " << st.ToString() );
            }

            auto py_schema = csp::python::PyObjectPtr::own(
                PyCapsule_New( rb_schema, "arrow_schema", releaseArrowSchemaCapsule ) );
            auto py_array  = csp::python::PyObjectPtr::own(
                PyCapsule_New( rb_array, "arrow_array", releaseArrowArrayCapsule ) );

            PyObject * py_tuple = PyTuple_Pack( 2, py_schema.get(), py_array.get() );
            PyList_SET_ITEM( py_list.get(), static_cast<Py_ssize_t>( idx ), py_tuple );
        }

        unnamed_output().output( csp::python::fromPython<DialectGenericType>( py_list.get() ) );
    }
};

EXPORT_CPPNODE( struct_to_record_batches );

}

REGISTER_CPPNODE( csp::cppnodes, record_batches_to_struct );
REGISTER_CPPNODE( csp::cppnodes, struct_to_record_batches );
