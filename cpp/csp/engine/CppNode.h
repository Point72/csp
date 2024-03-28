#ifndef _IN_CSP_ENGINE_CPPNODE_H
#define _IN_CSP_ENGINE_CPPNODE_H

#include <csp/core/Platform.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Node.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <string>

namespace csp
{

//CppNode is used specifically for C++ defined Nodes, and should only be used
//for definig c++ nodes using the macros defined at the end
class CppNode : public csp::Node
{
public:
    using Shape = std::variant<std::uint64_t,std::vector<std::string>>;
    struct InOutDef
    {
        INOUT_ID_TYPE index;
        CspTypePtr    type;
        bool          isAlarm;
        Shape         shape;
    };

    using InOutDefs = std::unordered_map<std::string,InOutDef>;

    struct NodeDef
    {
        InOutDefs  inputs;
        InOutDefs  outputs;
        Dictionary scalars;
    };

    const InOutDef & tsinputDef( const char * inputName ) const
    {
        validateNodeDef();
        auto it = m_nodedef -> inputs.find( inputName );
        if( it == m_nodedef -> inputs.end() )
            CSP_THROW( ValueError, "CppNode failed to find input " << inputName << " on node " << name() );

        return it -> second;
    }

    const InOutDef & tsoutputDef( const char * outputName ) const
    {
        validateNodeDef();
        auto it = m_nodedef -> outputs.find( outputName );
        if( it == m_nodedef -> outputs.end() )
            CSP_THROW( ValueError, "CppNode failed to find output " << outputName << " on node " << name() );

        return it -> second;
    }

    const InOutDef & alarmDef( const char * alarmName ) const
    {
        auto & def = tsinputDef( alarmName );
        if( !def.isAlarm )
            CSP_THROW( TypeError, "CppNode expected alarm " << alarmName << " but found it as an input on node " << name() );
        return def;
    }

    template<typename T>
    T scalarValue( const char * scalarName ) const
    {
        validateNodeDef();
        if( !m_nodedef -> scalars.exists( scalarName ) )
            CSP_THROW( ValueError, "CppNode failed to find scalar " << scalarName << " on node " << name() );

        return m_nodedef -> scalars.get<T>( scalarName );
    }

    void resetNodeDef() { m_nodedef = nullptr; }

    DateTime now() const { return rootEngine() -> now(); }

    using Creator = std::function<CppNode*( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef )>;

protected:
    CppNode( Engine * engine,
             const NodeDef & nodedef ) : csp::Node( asCspNodeDef( nodedef ), engine ),
                                         m_nodedef( &nodedef )
    {}

    csp::NodeDef asCspNodeDef( const NodeDef & nodedef ) const
    {
        if( nodedef.inputs.size() > InputId::maxInputs() )
            CSP_THROW( ValueError, "number of inputs exceeds limit of " << InputId::maxInputs() << " on node " << name() );

        if( nodedef.outputs.size() > OutputId::maxOutputs() )
            CSP_THROW( ValueError, "number of outputs exceeds limit of " << OutputId::maxOutputs() << " on node " << name() );

        return csp::NodeDef{ INOUT_ID_TYPE( nodedef.inputs.size() ), INOUT_ID_TYPE( nodedef.outputs.size() ) };
    }

    void validateNodeDef() const
    {
        if( !m_nodedef )
            CSP_THROW( RuntimeException, "CppNode cpp nodedef information is only available during INIT" );
    }

    struct Generic {};

    class InputWrapper
    {
    public:
        //for non-basket inputs only
        InputWrapper( const char *name, const CppNode & node ) : m_node( node ),
                                                                 m_id( 0 )
        {
            auto & def = m_node.tsinputDef( name );
            if( def.isAlarm )
                CSP_THROW( TypeError, "CppNode expected input " << name << " but found it as an alarm on node " << m_node.name() );
            m_id = InputId( def.index );
        }

        //for basket inputs
        InputWrapper( const CppNode & node, InputId id ) : m_node( node ),
                                                           m_id( id )
        {
        }

        bool valid() const  { return ts() -> valid(); }
        bool ticked() const { return m_node.inputTicked( m_id ); }

        bool makeActive() const  { return const_cast<CppNode&>( m_node ).makeActive( m_id ); }
        bool makePassive() const { return const_cast<CppNode&>( m_node ).makePassive( m_id ); }

        uint32_t count() const   { return ts() -> count(); }

        const CspType * type() const { return ts() -> type(); }

        void setTickCountPolicy( int32_t tickCount ) { const_cast<TimeSeriesProvider *>(ts()) -> setTickCountPolicy( tickCount ); }
        void setTickTimeWindowPolicy( TimeDelta window ) { const_cast<TimeSeriesProvider *>(ts()) -> setTickTimeWindowPolicy( window ); }

    protected:
        const TimeSeriesProvider * ts() const { return m_node.input( m_id ); }

        const CppNode & m_node;
        InputId         m_id;

    };

    class GenericInputWrapper : public InputWrapper
    {
    public:
        using InputWrapper::InputWrapper;

        template<typename T>
        const T & lastValue() const { return ts() -> lastValueTyped<T>(); }

        template<typename T>
        const T & valueAtIndex( int32_t index ) const { return ts() -> valueAtIndex<T>( index ); }
    };

    template<typename T>
    class TypedInputWrapper : public InputWrapper
    {
    public:
        using InputWrapper::InputWrapper;

        operator const T &() { return lastValue(); }
        const T & lastValue() const { return ts() -> template lastValueTyped<T>(); }

        const T & valueAtIndex( int32_t index ) const { return ts() -> template valueAtIndex<T>( index ); }
    };

    template<typename ElemWrapperT>
    class BasketInputWrapper
    {
    public:
        BasketInputWrapper( const char * name, const CppNode & node ) : m_node( node )
        {
            auto & def = m_node.tsinputDef( name );
            m_id = def.index;
            m_type = def.type;
        }

        ElemWrapperT operator[]( INOUT_ELEMID_TYPE elemId ) const
        {
            return ElemWrapperT( m_node, InputId( m_id, elemId ) );
        }

        bool makeActive() const
        {
             return const_cast<CppNode&>( m_node ).makeBasketActive( m_id );
        }

        bool makePassive() const
        {
             return const_cast<CppNode&>( m_node ).makeBasketPassive( m_id );
        }

        bool valid() const  { return basketInfo() -> allValid(); }
        bool ticked() const { return basketInfo() -> ticked(); }

        InputBasketInfo::ticked_iterator tickedinputs() const { return basketInfo() -> begin_ticked(); }
        InputBasketInfo::valid_iterator  validinputs() const  { return basketInfo() -> begin_valid(); }

        size_t size() const { return basketInfo() -> size(); }

        /**
         * @return The declared input types. NOTE: The declared input type. Note, we allow connecting slightly different types than declared
         * types (derived class ts where baseclass ts is expected, native type to dialect generic ...) so care should be taking when using this type
         */
        const CspTypePtr& type() const {return m_type;}

    protected:

        const InputBasketInfo * basketInfo() const { return m_node.inputBasket( m_id ); }
        void initBasket( size_t size )
        {
            const_cast<CppNode&>( this -> m_node ).initInputBasket( m_id, size, false );
        }

        const CppNode &m_node;
        INOUT_ID_TYPE m_id;
        CspTypePtr    m_type;
    };

    template<typename ElemWrapperT>
    class DictInputBasketWrapper : public BasketInputWrapper<ElemWrapperT>
    {
    public:
        DictInputBasketWrapper( const char * name, const CppNode & node ) : BasketInputWrapper<ElemWrapperT>( name, node )
        {
            auto & def = node.tsinputDef( name );
            m_shape = std::get<std::vector<std::string>>( def.shape );
            INOUT_ELEMID_TYPE elemId = 0;
            for( auto & key : m_shape )
                m_keyMap[ key ] = elemId++;

            //init input basket at this point
            this -> initBasket( m_shape.size() );
        }

        int64_t elemId( const std::string & key )
        {
            auto it = m_keyMap.find( key );
            if( it == m_keyMap.end() )
                return InputId::ELEM_ID_NONE;
            return it -> second;
        }

        const std::vector<std::string> & shape() const { return m_shape; }
    private:
        std::vector<std::string>                          m_shape;
        std::unordered_map<std::string,INOUT_ELEMID_TYPE> m_keyMap;
    };

    template<typename ElemWrapperT>
    class ListInputBasketWrapper : public BasketInputWrapper<ElemWrapperT>
    {
    public:
        ListInputBasketWrapper( const char * name, const CppNode & node ) : BasketInputWrapper<ElemWrapperT>( name, node )
        {
            auto & def = node.tsinputDef( name );
            auto shape = std::get<std::uint64_t>( def.shape );

            //init input basket at this point
            this -> initBasket( shape );
        }
    };

    //type selection helper for when requesting Generic input
    template<typename T>
    struct InputTypeHelper
    {
        using type = TypedInputWrapper<T>;
    };

    template<typename T>
    struct ListInputBasketTypeHelper
    {
        using type = ListInputBasketWrapper<TypedInputWrapper<T>>;
    };

    template<typename T>
    struct DictInputBasketTypeHelper
    {
        using type = DictInputBasketWrapper<TypedInputWrapper<T>>;
    };

    //Generic alarm can only be used to schedule from generic inputs
    struct GenericAlarmWrapper : public GenericInputWrapper
    {
        GenericAlarmWrapper( const char *name, const CppNode & node ) : GenericInputWrapper( node,
                                                                                             InputId( node.alarmDef( name ).index ) )
        {
        }

        Scheduler::Handle scheduleAlarm( TimeDelta delta, const GenericInputWrapper & input )
        {
            return scheduleAlarm( m_node.now() + delta, input );
        }

        Scheduler::Handle scheduleAlarm( DateTime time, const GenericInputWrapper & input )
        {
            assert( input.type() == type() );
            return switchCspType( input.type(), [this,time,&input]( auto tag )
                           {
                               using T = typename decltype(tag)::type;
                               auto * alarm = static_cast<AlarmInputAdapter<T> *>( const_cast<TimeSeriesProvider *>( ts() ) );
                               return alarm -> scheduleAlarm( time, input.lastValue<T>() );
                           } );
        }
    };

    template<typename T>
    struct TypedAlarmWrapper : public TypedInputWrapper<T>
    {
        TypedAlarmWrapper( const char *name, const CppNode & node ) : TypedInputWrapper<T>( node,
                                                                                            InputId( node.alarmDef( name ).index ) )
        {
        }

        Scheduler::Handle scheduleAlarm( TimeDelta delta, const T & value )
        {
            auto * alarm = static_cast<AlarmInputAdapter<T> *>( const_cast<TimeSeriesProvider *>( this -> ts() ) );
            return alarm -> scheduleAlarm( delta, value );
        }

        Scheduler::Handle scheduleAlarm( DateTime time, const T & value )
        {
            auto * alarm = static_cast<AlarmInputAdapter<T> *>( const_cast<TimeSeriesProvider *>( this -> ts() ) );
            return alarm -> scheduleAlarm( time, value );
        }

        void cancelAlarm( Scheduler::Handle handle )
        {
            auto * alarm = static_cast<AlarmInputAdapter<T> *>( const_cast<TimeSeriesProvider *>( this -> ts() ) );
            alarm -> cancelAlarm( handle );
        }
    };

    //type selection helper for when requesting Generic alarms
    template<typename T>
    struct AlarmTypeHelper
    {
        using type = TypedAlarmWrapper<T>;
    };


    class OutputWrapper
    {
    public:
        //non-basket outputs
        OutputWrapper( const char * name, const CppNode & node ) : m_node( node ),
                                                                   m_id( m_node.tsoutputDef( name ).index )
        {
        }

        //basket outputs
        OutputWrapper( const CppNode & node, OutputId id ) : m_node( node ),
                                                             m_id( id )
        {
        }

        const CspType * type() const { return ts() -> type(); }

    protected:
        TimeSeriesProvider * ts() const { return m_node.output( m_id ); }

        const CppNode & m_node;
        OutputId        m_id;
    };

    template<typename T>
    class TypedOutputWrapper : public OutputWrapper
    {
    public:
        using OutputWrapper::OutputWrapper;

        void output( const T & value )
        {
            ts() -> outputTickTyped( m_node.cycleCount(), m_node.now(), value );
        }

        T & reserveSpace()
        {
            return ts() -> template reserveTickTyped<T>( m_node.cycleCount(), m_node.now() );
        }
    };

    //this is for nodes that dont actually inspect the data, they get 'T' in
    //and return 'T' out
    class GenericOutputWrapper : public OutputWrapper
    {
    public:
        using OutputWrapper::OutputWrapper;

        void output( const GenericInputWrapper & input )
        {
            assert( input.type() == type() );
            switchCspType( input.type(), [this,&input]( auto tag )
                           {
                               using T = typename decltype(tag)::type;
                               ts() -> outputTickTyped( m_node.cycleCount(), m_node.now(),
                                                        input.lastValue<T>() );
                           } );
        }

        void output( const GenericAlarmWrapper & alarm )
        {
            //force call non-template version
            output( static_cast<const GenericInputWrapper & >( alarm ) );
        }

        template<typename T>
        void output( const T & value )
        {
            ts() -> outputTickTyped( m_node.cycleCount(), m_node.now(), value );
        }

        template<typename T>
        T & reserveSpace()
        {
            return ts() -> reserveTickTyped<T>( m_node.cycleCount(), m_node.now() );
        }
    };

    template<typename ElemWrapperT>
    class BasketOutputWrapper
    {
    public:
        BasketOutputWrapper( const char * name, const CppNode & node ) : m_node( node )
        {
            auto & def = m_node.tsoutputDef( name );
            m_id = def.index;
        }

        ElemWrapperT operator[]( INOUT_ELEMID_TYPE elemId ) const
        {
            return ElemWrapperT( m_node, OutputId( m_id, elemId ) );
        }

    private:
        const CppNode & m_node;
        INOUT_ID_TYPE   m_id;
    };

    template<typename ElemWrapperT>
    class DictOutputBasketWrapper : public BasketOutputWrapper<ElemWrapperT>
    {
    public:
        DictOutputBasketWrapper( const char * name, const CppNode & node ) : BasketOutputWrapper<ElemWrapperT>( name, node )
        {
            auto & def = node.tsoutputDef( name );
            auto & shape = std::get<std::vector<std::string>>( def.shape );
            INOUT_ELEMID_TYPE elemId = 0;
            for( auto & key : shape )
                m_keyMap[ key ] = elemId++;
        }

        int64_t elemId( const std::string & key )
        {
            auto it = m_keyMap.find( key );
            if( it == m_keyMap.end() )
                return InputId::ELEM_ID_NONE;
            return it -> second;
        }

    private:
        std::unordered_map<std::string,INOUT_ELEMID_TYPE> m_keyMap;

    };

    template<typename ElemWrapperT>
    class ListOutputBasketWrapper : public BasketOutputWrapper<ElemWrapperT>
    {
    public:
        ListOutputBasketWrapper( const char * name, const CppNode & node ) : BasketOutputWrapper<ElemWrapperT>( name, node )
        {
            // nothing needs to be done here
        }
    };

    //type selection helper for when requesting Generic outputs
    template<typename T>
    struct OutputTypeHelper
    {
        using type = TypedOutputWrapper<T>;
    };

    template<typename T>
    struct ListOutputBasketTypeHelper
    {
        using type = ListOutputBasketWrapper<TypedOutputWrapper<T>>;
    };

    template<typename T>
    struct DictOutputBasketTypeHelper
    {
        using type = DictOutputBasketWrapper<TypedOutputWrapper<T>>;
    };

    template<typename T>
    struct Scalar
    {
        Scalar( const char * name, const CppNode & node )
        {
            m_value = node.scalarValue<T>( name );
        }

        operator const T &() const { return m_value; }

        const T & value() const { return m_value; }

    private:
        T m_value;
    };

    //for python-like csp namespace interface
    class CSP
    {
    public:
        //these two are templatized to take basket inputs as well
        template<typename InputWrapperT>
        static bool make_passive( const InputWrapperT & input ) { return input.makePassive(); }
        template<typename InputWrapperT>
        static bool make_active( const InputWrapperT & input )  { return input.makeActive(); }

        static bool ticked() { return false; }

        template<typename T,typename... Args>
        static bool ticked( const T & input, const Args&... args )
        {
            return input.ticked() || ticked( args... );
        }


        static bool valid() { return true; }

        template<typename T,typename... Args>
        static bool valid( const T & input, const Args&... args )
        {
            return input.valid() && valid( args... );
        }

        static uint32_t count( const InputWrapper & input )
        {
            return input.count();
        }

        static Scheduler::Handle schedule_alarm( GenericAlarmWrapper & alarm, DateTime time, const GenericInputWrapper & input )
        {
            return alarm.scheduleAlarm( time, input );
        }

        static Scheduler::Handle schedule_alarm( GenericAlarmWrapper & alarm, TimeDelta delta, const GenericInputWrapper & input )
        {
            return alarm.scheduleAlarm( delta, input );
        }

        template<typename T>
        static Scheduler::Handle schedule_alarm( TypedAlarmWrapper<T> & alarm, DateTime time, const T & value )
        {
            return alarm.scheduleAlarm( time, value );
        }

        template<typename T>
        static Scheduler::Handle schedule_alarm( TypedAlarmWrapper<T> & alarm, TimeDelta delta, const T & value )
        {
            return alarm.scheduleAlarm( delta, value );
        }

        template<typename T>
        static Scheduler::Handle schedule_alarm( GenericAlarmWrapper & alarm, DateTime time, const T & value )
        {
            return schedule_alarm( reinterpret_cast<TypedAlarmWrapper<T> &>( alarm ), time, value );
        }

        template<typename T>
        static Scheduler::Handle schedule_alarm( GenericAlarmWrapper & alarm, TimeDelta delta, const T & value )
        {
            return schedule_alarm( reinterpret_cast<TypedAlarmWrapper<T> &>( alarm ), delta, value );
        }

        template<typename T>
        static void cancel_alarm( TypedAlarmWrapper<T> & alarm, Scheduler::Handle handle )
        {
            alarm.cancelAlarm( handle );
        }
    };

    //this is set temporarily for constructing input and scalar wrappers
    //easiuest way to get member-init to find the information.  ptr is null-ed out after init
    const NodeDef * m_nodedef;
};


template<>
struct csp::CppNode::InputTypeHelper<csp::CppNode::Generic>
{
    using type = GenericInputWrapper;
};

template<>
struct csp::CppNode::ListInputBasketTypeHelper<csp::CppNode::Generic>
{
    using type = ListInputBasketWrapper<GenericInputWrapper>;
};

template<>
struct csp::CppNode::DictInputBasketTypeHelper<csp::CppNode::Generic>
{
    using type = DictInputBasketWrapper<GenericInputWrapper>;
};

template<>
struct csp::CppNode::OutputTypeHelper<csp::CppNode::Generic>
{
    using type = GenericOutputWrapper;
};

template<>
struct csp::CppNode::ListOutputBasketTypeHelper<csp::CppNode::Generic>
{
    using type = ListOutputBasketWrapper<GenericOutputWrapper>;
};

template<>
struct csp::CppNode::DictOutputBasketTypeHelper<csp::CppNode::Generic>
{
    using type = DictOutputBasketWrapper<GenericOutputWrapper>;
};

template<>
struct csp::CppNode::AlarmTypeHelper<csp::CppNode::Generic>
{
    using type = GenericAlarmWrapper;
};


#define DECLARE_CPPNODE( Name ) class Name : public csp::CppNode

#define _STATIC_CREATE_METHOD( Class ) \
    static csp::CppNode * create( Engine * engine, const csp::CppNode::NodeDef & nodedef ) \
    { \
        auto * out = engine -> createOwnedObject<Class>( nodedef );   \
        out -> resetNodeDef(); \
        return out;             \
    }

#define INIT_CPPNODE_WITH_NAME( Class, Name )       \
    [[maybe_unused]] CSP csp; \
public:                                                      \
    const char * name() const override { return #Name; } \
    _STATIC_CREATE_METHOD( Class ) \
    Class( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) : csp::CppNode( engine, nodedef )

#define INIT_CPPNODE( Class )                       INIT_CPPNODE_WITH_NAME( Class, Class )

#define TS_INPUT( Type, Name )                      typename InputTypeHelper<Type>::type Name{#Name,*this}
#define TS_INPUT_RENAMED( Type, DeclName, VarName ) typename InputTypeHelper<Type>::type VarName{#DeclName,*this}
#define TS_LISTBASKET_INPUT( Type, Name )           typename ListInputBasketTypeHelper<Type>::type Name{#Name,*this}
#define TS_DICTBASKET_INPUT( Type, Name )           typename DictInputBasketTypeHelper<Type>::type Name{#Name,*this}

#define TS_OUTPUT( Type )                                  typename OutputTypeHelper<Type>::type __unnamed_output{"",*this}; auto & unnamed_output() { return __unnamed_output; }
#define TS_LISTBASKET_OUTPUT( Type )                       typename ListOutputBasketTypeHelper<Type>::type __unnamed_output{"",*this}; auto & unnamed_output() { return __unnamed_output; }
#define TS_NAMED_LISTBASKET_OUTPUT( Type, Name )           typename ListOutputBasketTypeHelper<Type>::type Name{#Name,*this}
#define TS_DICTBASKET_OUTPUT( Type )                       typename DictOutputBasketTypeHelper<Type>::type __unnamed_output{"",*this}; auto & unnamed_output() { return __unnamed_output; }

#define TS_NAMED_OUTPUT( Type, Name )                      typename OutputTypeHelper<Type>::type Name{#Name,*this}
#define TS_NAMED_OUTPUT_RENAMED( Type, DeclName, VarName ) typename OutputTypeHelper<Type>::type VarName{#DeclName,*this}

#define ALARM( Type, Name )                                typename AlarmTypeHelper<Type>::type Name{#Name,*this}

#define SCALAR_INPUT( Type, Name )                      csp::CppNode::Scalar<Type> Name{#Name,*this};
#define SCALAR_INPUT_RENAMED( Type, DeclName, VarName ) csp::CppNode::Scalar<Type> VarName{#DeclName,*this};

#define STATE_VAR( Type, Name ) Type Name;

#define START() NO_INLINE void start() override
#define STOP() void stop() override

//wrap _executeImpl so we clear basket flags at the end of execute call
#define INVOKE() void executeImpl() override

//only for unnamed outputs
#define CSP_OUTPUT( Value ) __unnamed_output.output( Value )
#define RETURN( Value ) do{ CSP_OUTPUT( Value ); return; } while( 0 );

#define SINGLE_ARG(...) __VA_ARGS__
#define CPPNODE_CREATE_METHOD( Name ) Name##_create_method
#define CPPNODE_CREATE_FWD_DECL( Namespace, Name ) namespace Namespace { csp::CppNode * CPPNODE_CREATE_METHOD( Name )( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ); }
#define EXPORT_CPPNODE( Name ) csp::CppNode * CPPNODE_CREATE_METHOD( Name )( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) { return Name::create( engine, nodedef ); }
#define EXPORT_TEMPLATE_CPPNODE( Name, Typed ) csp::CppNode * CPPNODE_CREATE_METHOD( Name )( csp::Engine * engine, const csp::CppNode::NodeDef & nodedef ) { return Typed::create( engine, nodedef ); }

}

#endif
