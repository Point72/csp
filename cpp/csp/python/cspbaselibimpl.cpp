#include <csp/python/Common.h>
#include <csp/python/PyCppNode.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Conversions.h>
#include <exprtk.hpp>
#include <numpy/ndarrayobject.h>

#include <arrow/type.h>
#include <arrow/table.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>

#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/python/PyObjectPtr.h>

static void * init_nparray()
{
    csp::python::AcquireGIL gil;
    import_array();
    return nullptr;
}
static void * s_init_array = init_nparray();

void ReleaseArrowSchemaPyCapsule( PyObject* capsule ) {
    struct ArrowSchema* schema =
        ( struct ArrowSchema* )PyCapsule_GetPointer( capsule, "arrow_schema" );
    if ( schema->release != NULL ) {
        schema->release( schema );
    }
    free( schema );
}

void ReleaseArrowArrayPyCapsule( PyObject* capsule ) {
    struct ArrowArray* array =
        ( struct ArrowArray* )PyCapsule_GetPointer( capsule, "arrow_array");
    if ( array->release != NULL ) {
        array->release( array );
    }
    free( array );
}

namespace csp::cppnodes
{
DECLARE_CPPNODE( exprtk_impl )
{
    class BaseValueContainer
    {
    public:
        virtual ~BaseValueContainer() = default;
        virtual void setValue( const TimeSeriesProvider * ) = 0;
        virtual bool registerValue( exprtk::symbol_table<double> &expr, const std::string &variableName ) = 0;
    };

    template< typename T >
    class ValueContainer : public BaseValueContainer
    {
    public:
        void setValue( const TimeSeriesProvider *tsProvider ) override
        {
            m_value = tsProvider -> lastValueTyped<T>();
        }

        bool registerValue( exprtk::symbol_table<double> &expr, const std::string &variableName ) override
        {
            registerValueImpl( expr, variableName );
            return true;
        }

    private:
        template <typename V=T, std::enable_if_t<std::is_arithmetic_v<V>, bool> = true>
        void registerValueImpl( exprtk::symbol_table<double> &symbolTable, const std::string &variableName )
        {
            symbolTable.add_variable( variableName, m_value );
        }

        template <typename V=T, std::enable_if_t<std::is_same_v<V, std::string>, bool> = true>
        void registerValueImpl( exprtk::symbol_table<double> &symbolTable, const std::string &variableName )
        {
            symbolTable.add_stringvar( variableName, m_value );
        }

        T m_value;
    };

    class NumpyArrayValueContainer : public BaseValueContainer
    {
    public:
        NumpyArrayValueContainer() : m_arr_size(-1) {}

        void validateArray( PyArrayObject* arr )
        {
            auto dim = PyArray_NDIM( arr );
            if( dim != 1 )
                CSP_THROW( ValueError, "csp.exprtk recieved an array of dim " << dim << " but can only take 1D arrays" );

            if( !PyArray_CHKFLAGS( arr, NPY_ARRAY_OWNDATA ) )
                CSP_THROW( ValueError, "csp.exprtk requires arrays be naturally strided" );

            if( !PyArray_ISFLOAT( arr ) )
                CSP_THROW( ValueError, "csp.exprtk requires arrays to contain floats" );
        }

        void setValue( const TimeSeriesProvider *tsProvider ) override
        {
            PyArrayObject* arr = (PyArrayObject*) csp::python::toPythonBorrowed(tsProvider -> lastValueTyped<DialectGenericType>());

            // register on first tick
            if( m_arr_size == -1 )
            {
                validateArray( arr );
                m_arr_size = PyArray_SIZE( arr );
                double* data = reinterpret_cast<double*>( PyArray_DATA( arr ) );
                m_view = std::make_unique<exprtk::vector_view<double>>( data, m_arr_size );

                m_symbolTable -> add_vector( m_var_name, *m_view );
            }
            else
            {
                if( PyArray_SIZE( arr ) != m_arr_size )
                    CSP_THROW( ValueError, "csp.exprtk NumPy array input must have same size each tick, but first saw " << m_arr_size
                                        << " and now saw " << PyArray_SIZE( arr ) << " for " << m_var_name );

                validateArray( arr );
                double* data = reinterpret_cast<double*>( PyArray_DATA( arr ) );
                m_view -> rebase( data );
            }
        }

        bool registerValue( exprtk::symbol_table<double> &symbolTable, const std::string &variableName ) override
        {
            // store symbol table and var name so we can use them to register in setValue, on first tick
            m_symbolTable = &symbolTable;
            m_var_name = variableName;
            return false;
        }

    private:
        exprtk::symbol_table<double> *m_symbolTable;
        std::string m_var_name;
        int64_t m_arr_size;
        std::unique_ptr<exprtk::vector_view<double>> m_view;
    };

    struct csp_now_fn : public exprtk::ifunction<double>
    {
    public:
        csp_now_fn() : exprtk::ifunction<double>(0) {}
        double operator()() { return ( m_engine -> rootEngine() -> now() ).asNanoseconds() / 1e9; }
        void setEngine( csp::Engine * engine ) { m_engine = engine; }
    private:
        csp::Engine * m_engine;
    };

    SCALAR_INPUT(           std::string,        expression_str );
    TS_DICTBASKET_INPUT(    DialectGenericType, inputs );
    SCALAR_INPUT(           DictionaryPtr,      state_vars );
    SCALAR_INPUT(           DictionaryPtr,      constants );
    SCALAR_INPUT(           DictionaryPtr,      functions );
    TS_INPUT(               Generic,            trigger );
    SCALAR_INPUT(           bool,               use_trigger );
    TS_OUTPUT( Generic );

    STATE_VAR( exprtk::function_compositor<double>, s_compositor );
    STATE_VAR( exprtk::expression<double>, s_expr );
    STATE_VAR( exprtk::parser<double>, s_parser );
    STATE_VAR( csp_now_fn, s_csp_now );
    STATE_VAR( std::vector<std::unique_ptr<BaseValueContainer>>, s_valuesContainer );
    STATE_VAR( bool, s_isCompiled );

    void compile_expression()
    {
        s_expr.register_symbol_table( s_compositor.symbol_table() );

        if( !s_parser.compile( expression_str, s_expr ) )
            CSP_THROW( ValueError, "cannot compile expression: " << std::string( expression_str ) << " ERROR: " << s_parser.error() );

        s_isCompiled = true;
    }

    INIT_CPPNODE( exprtk_impl ) {}

    START()
    {
        s_isCompiled = false;
        bool all_registered = true;
        exprtk::symbol_table<double>& symbolTable = s_compositor.symbol_table();

        for( size_t elem = 0; elem < inputs.size(); ++elem )
        {
            auto &&inputName = inputs.shape()[ elem ];
            auto typ = inputs[ elem ].type();

            if( typ -> type() == CspType::Type::DIALECT_GENERIC )
            {
                s_valuesContainer.push_back( std::make_unique<NumpyArrayValueContainer>() );
            }
            else
            {
                PartialSwitchCspType<CspType::Type::STRING, CspType::Type::DOUBLE>::invoke(
                        typ,
                        [ this ]( auto tag )
                        {
                            s_valuesContainer.push_back( std::make_unique<ValueContainer<typename decltype(tag)::type>>() );
                        } );
            }

            all_registered &= s_valuesContainer.back() -> registerValue( symbolTable, inputName );
        }

        for( auto it = state_vars.value() -> begin(); it != state_vars.value() -> end(); ++it )
        {
            if( it.hasValue<std::string>() )
            {
                symbolTable.create_stringvar( it.key() );
                symbolTable.get_stringvar( it.key() ) -> ref() = it.value<std::string>();
            }
            else if( it.hasValue<double>() || it.hasValue<int64_t>() )
            {
                symbolTable.create_variable(it.key());
                symbolTable.get_variable( it.key() ) -> ref() = it.value<double>();
            }
            else
                CSP_THROW( ValueError, "state_vars dictionary contains " << it.key() << " with unsupported type (need be string or float)" );
        }

        for( auto it = constants.value() -> begin(); it != constants.value() -> end(); ++it )
        {
            if( it.hasValue<double>() || it.hasValue<int64_t>() )
                symbolTable.add_constant( it.key(), it.value<double>() );
            else
                CSP_THROW( ValueError, "constants dictionary contains " << it.key() << " with unsupported type (need be float)" );
        }

        if( functions.value() -> size() > 0 )
        {
            typedef exprtk::function_compositor<double> compositor_t;
            typedef typename compositor_t::function function_t;

            for( auto it = functions.value() -> begin(); it != functions.value() -> end(); ++it )
            {
                csp::python::PyObjectPtr fnInfo = csp::python::PyObjectPtr::own( csp::python::toPython( it.value<DialectGenericType>() ) );
                const char * body;
                PyObject * vars;


                if( !PyArg_ParseTuple( fnInfo.get(), "O!s", &PyTuple_Type, &vars , &body ) )
                {
                    CSP_THROW( csp::python::PythonPassthrough, "could not parse function info in csp.exprtk" );
                }

                const char *arg1, *arg2, *arg3, *arg4;
                auto numVars = PyTuple_Size( vars );
                switch( numVars )
                {
                    case 0:
                        CSP_THROW( ValueError, "csp.exprtk functions must take at least one variable" );
                        break;
                    case 1:
                        if( !PyArg_ParseTuple( vars, "s", &arg1 ) )
                        {
                            CSP_THROW( csp::python::PythonPassthrough, "csp.exprtk could not parse variables list" );
                        }
                        s_compositor.add( function_t( it.key(), body, arg1 ) );
                        break;
                    case 2:
                        if( !PyArg_ParseTuple( vars, "ss", &arg1, &arg2 ) )
                        {
                            CSP_THROW( csp::python::PythonPassthrough, "csp.exprtk could not parse variables list" );
                        }
                        s_compositor.add( function_t( it.key(), body, arg1, arg2 ) );
                        break;
                    case 3:
                        if( !PyArg_ParseTuple( vars, "sss", &arg1, &arg2, &arg3 ) )
                        {
                            CSP_THROW( csp::python::PythonPassthrough, "csp.exprtk could not parse variables list" );
                        }
                        s_compositor.add( function_t( it.key(), body, arg1, arg2, arg3 ) );
                        break;
                    case 4:
                        if( !PyArg_ParseTuple( vars, "ssss", &arg1, &arg2, &arg3, &arg4 ) )
                        {
                            CSP_THROW( csp::python::PythonPassthrough, "csp.exprtk could not parse variables list" );
                        }
                        s_compositor.add( function_t( it.key(), body, arg1, arg2, arg3, arg4 ) );
                        break;
                    default:
                        CSP_THROW( ValueError, "csp.exprtk given too many variables (" << numVars << "), max supported is 4" );
                }
            }
        }

        s_csp_now.setEngine( engine() );
        symbolTable.add_function( "csp.now", s_csp_now );

        if( all_registered )
            compile_expression();

        if( use_trigger )
            csp.make_passive( inputs );
    }

    INVOKE()
    {
        if( use_trigger )
        {
            for( auto &&inputIt = inputs.validinputs(); inputIt; ++inputIt )
            {
                s_valuesContainer[ inputIt.elemId() ] -> setValue( inputIt.get() );
            }
        }
        else
        {
            for( auto &&inputIt = inputs.tickedinputs(); inputIt; ++inputIt )
            {
                s_valuesContainer[ inputIt.elemId() ] -> setValue( inputIt.get() );
            }
        }

        if( likely( csp.valid( inputs ) ) )
        {
            if( unlikely( !s_isCompiled ) )
                compile_expression();

            const CspType* outputType = unnamed_output().type();
            if( outputType->type() == CspType::Type::DOUBLE )
            {
                RETURN( s_expr.value() );
            }
            else
            {
                s_expr.value();  // need this to get the expression to evaluate

                const exprtk::results_context<double>& results = s_expr.results();
                npy_intp numResults = results.count();
                PyObject* out = PyArray_EMPTY(1, &numResults, NPY_DOUBLE, 0 ); // 1D array
                double* data = reinterpret_cast<double*>( PyArray_DATA( ( PyArrayObject* )out ) );

                typedef exprtk::results_context<double>::type_store_t::scalar_view scalar_t;

                for (npy_intp i = 0; i < numResults; ++i)
                {
                    data[i] = scalar_t(results[i])();
                }

                RETURN( csp::python::PyObjectPtr::own( out ) );
            }
        }
    }

};

EXPORT_CPPNODE( exprtk_impl );

DECLARE_CPPNODE( record_batches_to_struct )
{
    using InMemoryTableParquetReader = csp::adapters::parquet::InMemoryTableParquetReader;
    class RecordBatchReader : public InMemoryTableParquetReader
    {
    public:
        RecordBatchReader( std::vector<std::string> columns, std::shared_ptr<arrow::Schema> schema ):
            InMemoryTableParquetReader( nullptr, columns, false, {}, false )
        {
            m_schema = schema;
        }
        std::string getCurFileOrTableName() const override{ return "IN_RECORD_BATCH"; }
        void initialize() { setColumnAdaptersFromCurrentTable(); }
        void parseBatches( std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches )
        {
            auto table_result = arrow::Table::FromRecordBatches( m_schema, record_batches );
            if( !table_result.ok() )
                CSP_THROW( ValueError, "Failed to load all the record batches into a table: " << table_result.status().ToString() );

            setTable( table_result.ValueUnsafe() );

            if( !readNextRowGroup() )
                CSP_THROW( ValueError, "Unable to read the first row group from table" );

            while( readNextRow() )
            {
                for( auto& adapter: getStructAdapters() )
                {
                    adapter -> dispatchValue( nullptr );
                }
            }
        }
        void stop()
        {
            InMemoryTableParquetReader::clear();
        }
    protected:
        bool openNextFile() override { return false; }
        void clear() override { setTable( nullptr ); }
    };

    SCALAR_INPUT( DialectGenericType,  schema_ptr );
    SCALAR_INPUT( StructMetaPtr,  cls );
    SCALAR_INPUT( DictionaryPtr,  properties );
    TS_INPUT( Generic, data );

    TS_OUTPUT( Generic );

    std::shared_ptr<RecordBatchReader> reader;
    std::vector<StructPtr>* m_structsVecPtr;

    INIT_CPPNODE( record_batches_to_struct )
    {
        auto & input_def = tsinputDef( "data" );
        if( input_def.type -> type() != CspType::Type::ARRAY )
            CSP_THROW( TypeError, "record_batches_to_struct expected ts array type, got " << input_def.type -> type() );

        auto * aType = static_cast<const CspArrayType *>( input_def.type.get() );
        CspTypePtr elemType = aType -> elemType();
        if( elemType -> type() != CspType::Type::DIALECT_GENERIC )
            CSP_THROW( TypeError, "record_batches_to_struct expected ts array of DIALECT_GENERIC type, got " << elemType -> type() );

        auto & output_def = tsoutputDef( "" );
        if( output_def.type -> type() != CspType::Type::ARRAY )
            CSP_THROW( TypeError, "record_batches_to_struct expected ts array type, got " << output_def.type -> type() );
    }

    START()
    {
        // Create Adapters for Schema
        PyObject* capsule = csp::python::toPythonBorrowed( schema_ptr );
        struct ArrowSchema* c_schema = reinterpret_cast<struct ArrowSchema*>( PyCapsule_GetPointer( capsule, "arrow_schema") );
        auto result = arrow::ImportSchema( c_schema );
        if( !result.ok() )
            CSP_THROW( ValueError, "Failed to load the arrow schema: " << result.status().ToString() );
        std::shared_ptr<arrow::Schema> schema = result.ValueUnsafe();
        std::vector<std::string> columns;
        auto field_map = properties.value() -> get<DictionaryPtr>( "field_map" );
        for( auto it = field_map -> begin(); it != field_map -> end(); ++it )
        {
            if( schema -> GetFieldByName( it.key() ) )
                columns.push_back( it.key() );
            else
                CSP_THROW( ValueError, "column " << it.key() << " not found in schema" );
        }
        reader = std::make_shared<RecordBatchReader>( columns, schema );
        reader -> initialize();

        CspTypePtr outType = std::make_shared<csp::CspStructType>( cls.value() );
        csp::adapters::utils::StructAdapterInfo key{ std::move( outType ), std::move( field_map ) };
        auto& struct_adapter = reader -> getStructAdapter( key );
        struct_adapter.addSubscriber( [this]( StructPtr * s )
                                      {
                                          if( s ) this -> m_structsVecPtr -> push_back( *s );
                                          else CSP_THROW( ValueError, "Failed to create struct while parsing the record batches" );
                                      }, {} );
    }

    INVOKE()
    {
        if( csp.ticked( data ) )
        {
            auto & py_batches = data.lastValue<std::vector<DialectGenericType>>();
            std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
            for( auto& py_batch: py_batches )
            {
                PyObject* py_tuple = csp::python::toPythonBorrowed( py_batch );
                PyObject* py_schema = PyTuple_GET_ITEM( py_tuple, 0 );
                PyObject* py_array = PyTuple_GET_ITEM( py_tuple, 1 );
                struct ArrowSchema* c_schema = reinterpret_cast<struct ArrowSchema*>( PyCapsule_GetPointer( py_schema, "arrow_schema" ) );
                struct ArrowArray* c_array = reinterpret_cast<struct ArrowArray*>( PyCapsule_GetPointer( py_array, "arrow_array" ) );
                auto result = arrow::ImportRecordBatch( c_array, c_schema );
                if( !result.ok() )
                    CSP_THROW( ValueError, "Failed to load record batches through PyCapsule C Data interface: " << result.status().ToString() );
                batches.emplace_back( result.ValueUnsafe() );
            }
            std::vector<StructPtr> & out = unnamed_output().reserveSpace<std::vector<StructPtr>>();
            out.clear();
            m_structsVecPtr = &out;
            reader -> parseBatches( batches );
            m_structsVecPtr = nullptr;
        }
    }
};

EXPORT_CPPNODE( record_batches_to_struct );

DECLARE_CPPNODE( struct_to_record_batches )
{
    SCALAR_INPUT( DialectGenericType,  schema_ptr );
    SCALAR_INPUT( StructMetaPtr,  cls );
    SCALAR_INPUT( DictionaryPtr,  properties );
    SCALAR_INPUT( int64_t, chunk_size );
    TS_INPUT( Generic, data );

    TS_OUTPUT( Generic );

    using StructParquetOutputHandler = csp::adapters::parquet::StructParquetOutputHandler;
    using ParquetWriter = csp::adapters::parquet::ParquetWriter;
    class MyParquetWriter : public ParquetWriter
    {
    public:
        MyParquetWriter( int64_t chunk_size ): ParquetWriter(), m_chunkSize( chunk_size )
        {
            if( m_chunkSize <= 0 )
            {
                CSP_THROW( ValueError, "Chunk size should be >= 0" );
            }
        }
        std::uint32_t getChunkSize() const override{ return m_chunkSize; }
    private:
        int64_t m_chunkSize = 0;
    };

    std::shared_ptr<StructParquetOutputHandler> m_handler;
    CspTypePtr m_cspType;
    std::shared_ptr<MyParquetWriter> m_writer;
    std::shared_ptr<arrow::Schema> m_schema;

    INIT_CPPNODE( struct_to_record_batches )
    {
        auto & input_def = tsinputDef( "data" );
        if( input_def.type -> type() != CspType::Type::ARRAY )
            CSP_THROW( TypeError, "struct_to_record_batches expected ts array type, got " << input_def.type -> type() );

        auto * aType = static_cast<const CspArrayType *>( input_def.type.get() );
        CspTypePtr elemType = aType -> elemType();
        if( elemType -> type() != CspType::Type::STRUCT )
            CSP_THROW( TypeError, "struct_to_record_batches expected ts array of structs type, got " << elemType -> type() );

        auto & output_def = tsoutputDef( "" );
        if( output_def.type -> type() != CspType::Type::ARRAY )
            CSP_THROW( TypeError, "struct_to_record_batches expected ts array type, got " << output_def.type -> type() );
    }

    START()
    {
        // Create Adapters for Schema
        auto field_map = properties.value() -> get<DictionaryPtr>( "field_map" );
        m_writer = std::make_shared<MyParquetWriter>( chunk_size.value() );
        m_cspType = std::make_shared<csp::CspStructType>( cls.value() );
        m_handler = std::make_shared<StructParquetOutputHandler>( engine(), *m_writer, m_cspType, field_map );
        std::vector<std::shared_ptr<arrow::Field>> arrowFields;
        for( unsigned i = 0; i < m_handler -> getNumColumns(); i++ )
        {
            arrowFields.push_back( arrow::field( m_handler -> getColumnArrayBuilder( i ) -> getColumnName(),
                                                 m_handler -> getColumnArrayBuilder( i ) -> getDataType() ) );
        }
        m_schema = arrow::schema( arrowFields );
    }

    DialectGenericType getData( std::shared_ptr<StructParquetOutputHandler> handler, int num_rows )
    {
        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.reserve( handler -> getNumColumns() );
        for( unsigned i = 0; i < handler -> getNumColumns(); i++ )
        {
            columns.push_back( handler -> getColumnArrayBuilder( i ) -> buildArray() );
        }
        auto rb_ptr = arrow::RecordBatch::Make( m_schema, num_rows, columns );
        const arrow::RecordBatch& rb = *rb_ptr;
        struct ArrowSchema* rb_schema = ( struct ArrowSchema* )malloc( sizeof( struct ArrowSchema ) );
        struct ArrowArray* rb_array = ( struct ArrowArray* )malloc( sizeof( struct ArrowArray ) );
        arrow::Status st = arrow::ExportRecordBatch( rb, rb_array, rb_schema );
        auto py_schema = csp::python::PyObjectPtr::own( PyCapsule_New( rb_schema, "arrow_schema", ReleaseArrowSchemaPyCapsule ) );
        auto py_array = csp::python::PyObjectPtr::own( PyCapsule_New( rb_array, "arrow_array", ReleaseArrowArrayPyCapsule ) );
        auto py_tuple = csp::python::PyObjectPtr::own( PyTuple_Pack( 2, py_schema.get(), py_array.get() ) );
        return csp::python::fromPython<DialectGenericType>( py_tuple.get() );
    }

    INVOKE()
    {
        if( csp.ticked( data ) )
        {
            std::vector<DialectGenericType> & out = unnamed_output().reserveSpace<std::vector<DialectGenericType>>();
            out.clear();
            auto & structs = data.lastValue<std::vector<StructPtr>>();
            uint32_t cur_chunk_size = 0;
            for( auto& st: structs )
            {
                m_handler -> writeValueFromArgs( st );
                for( unsigned i = 0; i < m_handler -> getNumColumns(); i++ )
                {
                    m_handler -> getColumnArrayBuilder( i ) -> handleRowFinished();
                }
                if( ++cur_chunk_size >= m_writer -> getChunkSize() )
                {
                    out.emplace_back( getData( m_handler, cur_chunk_size ) );
                    cur_chunk_size = 0;
                }
            }
            if( cur_chunk_size > 0)
            {
                out.emplace_back( getData( m_handler, cur_chunk_size ) );
            }
        }
    }
};

EXPORT_CPPNODE( struct_to_record_batches );

}

// Base nodes
REGISTER_CPPNODE( csp::cppnodes, sample );
REGISTER_CPPNODE( csp::cppnodes, firstN );
REGISTER_CPPNODE( csp::cppnodes, count );
REGISTER_CPPNODE( csp::cppnodes, _delay_by_timedelta );
REGISTER_CPPNODE( csp::cppnodes, _delay_by_ticks );
REGISTER_CPPNODE( csp::cppnodes, merge );
REGISTER_CPPNODE( csp::cppnodes, split );
REGISTER_CPPNODE( csp::cppnodes, cast_int_to_float );
REGISTER_CPPNODE( csp::cppnodes, filter );
REGISTER_CPPNODE( csp::cppnodes, _drop_dups_float );
REGISTER_CPPNODE( csp::cppnodes, drop_nans );
REGISTER_CPPNODE( csp::cppnodes, unroll );
REGISTER_CPPNODE( csp::cppnodes, collect );
REGISTER_CPPNODE( csp::cppnodes, demultiplex );
REGISTER_CPPNODE( csp::cppnodes, multiplex );
REGISTER_CPPNODE( csp::cppnodes, times );
REGISTER_CPPNODE( csp::cppnodes, times_ns );
REGISTER_CPPNODE( csp::cppnodes, struct_field );
REGISTER_CPPNODE( csp::cppnodes, struct_fromts );
REGISTER_CPPNODE( csp::cppnodes, struct_collectts );

REGISTER_CPPNODE( csp::cppnodes, exprtk_impl );
REGISTER_CPPNODE( csp::cppnodes, record_batches_to_struct );
REGISTER_CPPNODE( csp::cppnodes, struct_to_record_batches );

static PyModuleDef _cspbaselibimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_cspbaselibimpl",
    "_cspbaselibimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspbaselibimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspbaselibimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
