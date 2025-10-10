#include <exprtk.hpp>
#include <numpy/ndarrayobject.h>

#include <csp/python/Common.h>
#include <csp/python/PyCppNode.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Conversions.h>

static void * init_nparray()
{
    csp::python::AcquireGIL gil;
    import_array();
    return nullptr;
}
static void * s_init_array = init_nparray();

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

    virtual ~exprtk_impl() 
    {
        // Need to release the expression before clearing values/symbol table
        // https://github.com/ArashPartow/exprtk/blob/cc1b800c2bd1ac3ac260478c915d2aec6f4eb41c/readme.txt#L909
        s_expr.release();
        s_valuesContainer.clear();
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
