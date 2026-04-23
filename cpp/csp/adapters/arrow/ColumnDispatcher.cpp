// Typed ColumnDispatcher implementations and factory.
//
// TypedColumnDispatcher<T> stores std::optional<T> and dispatches via
// ValueDispatcher<const T &>.  The string specialization adds enum coercion.

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/CspType.h>
#include <csp/engine/PartialSwitchCspType.h>

namespace csp::adapters::arrow
{

namespace
{

template<typename ValueType>
class TypedColumnDispatcher final : public ColumnDispatcher
{
    using Dispatcher = utils::ValueDispatcher<const ValueType &>;

public:
    TypedColumnDispatcher( std::string name, std::unique_ptr<FieldReader> reader,
                           ::arrow::Type::type arrowTypeId )
        : ColumnDispatcher( std::move( name ), std::move( reader ), arrowTypeId )
    {
    }

    void readNextValue() override
    {
        m_fieldReader -> readNextValue( &m_value );
    }

    void dispatchValue( const utils::Symbol * symbol ) override
    {
        if( m_value.has_value() )
            m_dispatcher.dispatch( &m_value.value(), symbol );
        else
            m_dispatcher.dispatch( nullptr, symbol );
    }

    void addSubscriber( ManagedSimInputAdapter * adapter,
                        std::optional<utils::Symbol> symbol ) override
    {
        addSubscriberImpl( adapter, symbol );
    }

    void * getCurValueUntyped() override { return &m_value; }

protected:
    void addSubscriberImpl( ManagedSimInputAdapter * adapter,
                            std::optional<utils::Symbol> symbol )
    {
        using Switch = ConstructibleTypeSwitch<ValueType>;
        try
        {
            auto callback = Switch::invoke( adapter -> type(), [adapter]( auto tag )
            {
                return typename Dispatcher::SubscriberType(
                    [adapter]( const ValueType * val )
                    {
                        if( val )
                            adapter -> pushTick<typename decltype( tag )::type>( *val );
                        else
                            adapter -> pushNullTick<typename decltype( tag )::type>();
                    } );
            } );
            m_dispatcher.addSubscriber( callback, symbol );
        }
        catch( UnsupportedSwitchType & )
        {
            CSP_THROW( TypeError, "Unsupported subscriber type "
                                   << adapter -> type() -> type().asCString()
                                   << " for column '" << m_columnName << "'" );
        }
    }

    std::optional<ValueType> m_value;
    Dispatcher               m_dispatcher;
};

// String specialization: adds string→enum coercion support.
template<>
void TypedColumnDispatcher<std::string>::addSubscriber(
    ManagedSimInputAdapter * adapter,
    std::optional<utils::Symbol> symbol )
{
    if( adapter -> type() -> type() == CspType::Type::ENUM )
    {
        auto enumMetaPtr = static_cast<const CspEnumType *>( adapter -> type() ) -> meta();
        auto callback = typename Dispatcher::SubscriberType(
            [adapter, enumMetaPtr]( const std::string * val )
            {
                if( val )
                    adapter -> pushTick<CspEnum>( enumMetaPtr -> fromString( val -> c_str() ) );
                else
                    adapter -> pushNullTick<CspEnum>();
            } );
        m_dispatcher.addSubscriber( callback, symbol );
    }
    else
    {
        addSubscriberImpl( adapter, symbol );
    }
}

template<typename T>
std::unique_ptr<ColumnDispatcher> makeDispatcher( const std::string & name,
                                                   std::unique_ptr<FieldReader> reader,
                                                   ::arrow::Type::type arrowTypeId )
{
    return std::make_unique<TypedColumnDispatcher<T>>( name, std::move( reader ), arrowTypeId );
}

} // anonymous namespace

std::unique_ptr<ColumnDispatcher> createColumnDispatcher(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const std::shared_ptr<const StructMeta> & structMeta )
{
    auto typeId = arrowField -> type() -> id();
    auto & name = arrowField -> name();

    // Unsupported types: return nullptr immediately (no FieldReader creation)
    if( typeId == ::arrow::Type::STRUCT && !structMeta )
        return nullptr;

    // Create FieldReader (structField=nullptr: only readNextValue path is used)
    auto fieldReader = createFieldReader( arrowField, nullptr, structMeta );

    return visitArrowValueType( typeId,
        [&]( auto tag ) -> std::unique_ptr<ColumnDispatcher>
        {
            using T = typename decltype( tag )::type;
            return makeDispatcher<T>( name, std::move( fieldReader ), typeId );
        },
        []() -> std::unique_ptr<ColumnDispatcher> { return nullptr; } );
}

} // namespace csp::adapters::arrow
