#include <csp/engine/Consumer.h>
#include <csp/engine/InputAdapter.h>

namespace csp
{

InputAdapter::InputAdapter( Engine *engine, const CspTypePtr &type, PushMode pushMode ) : m_rootEngine( engine -> rootEngine() ),
                                                                                          m_pushMode( pushMode )
{
    if( pushMode == PushMode::BURST )
        init( CspArrayType::create( type ) );
    else
        init( type );
}


template<>
bool InputAdapter::consumeTick( const bool & value )
{
    switch( pushMode() )
    {
        case PushMode::LAST_VALUE:
        {
            if( unlikely( rootEngine() -> cycleCount() == lastCycleCount() ) )
                m_timeseries -> lastValueTyped<bool>() = value;
            else
                this -> outputTickTyped<bool>( rootEngine() -> now(), value );
            return true;
        }

        case PushMode::BURST:
        {
            CSP_ASSERT( type() -> type() == CspType::Type::ARRAY );
            CSP_ASSERT( static_cast<const CspArrayType * >( type() ) -> elemType() -> type() == CspType::Type::fromCType<bool>::type );

            using ArrayT = typename CspType::Type::toCType<CspType::Type::ARRAY, bool>::type;
            if( likely( rootEngine() -> cycleCount() != lastCycleCount() ) )
            {
                //ensure we reuse vector memory in our buffer by using reserve api and
                //clearing existing value if any
                reserveTickTyped<ArrayT>( rootEngine() -> cycleCount(), rootEngine() -> now() ).clear();
            }

            m_timeseries -> lastValueTyped<ArrayT>().push_back( value );
            return true;
        }

        case PushMode::NON_COLLAPSING:
        {
            if( unlikely( rootEngine() -> cycleCount() == lastCycleCount() ) )
                return false;

            this -> outputTickTyped<bool>( rootEngine() -> now(), value );
            return true;
        }

        default:
            CSP_THROW( NotImplemented, pushMode() << " mode is not yet supported" );
            break;
    }

    return true;
}

}
