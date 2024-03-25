#ifndef _IN_CSP_ENGINE_INPUTADAPTER_H
#define _IN_CSP_ENGINE_INPUTADAPTER_H

#include <csp/core/Time.h>
#include <csp/engine/Enums.h>
#include <csp/engine/RootEngine.h>
#include <csp/engine/TimeSeriesProvider.h>

namespace csp
{

class Consumer;

class InputAdapter : public TimeSeriesProvider, public EngineOwned
{
public:
    InputAdapter( Engine * engine, const CspTypePtr & type, PushMode pushMode );
    virtual ~InputAdapter() {}

    using TimeSeriesProvider::outputTickTyped;

    virtual void start( DateTime start, DateTime end ) {}
    virtual void stop() {}

    template< typename T > void outputTickTyped( DateTime timestamp, const T & value )
    {
        outputTickTyped( rootEngine() -> cycleCount(), timestamp, value );
    }

    //used by sim and realtime input adapters
    template<typename T>
    bool consumeTick( const T & value );

    RootEngine * rootEngine() { return m_rootEngine; }

    PushMode pushMode() const { return m_pushMode; }

    void setStarted()         { m_started = true; }
    bool started() const      { return m_started; }

    //if adapter is BURST this will return the type of the data, rather than the BURST vector<Data>
    const CspType * dataType() const
    {
        if( m_pushMode == PushMode::BURST )
            return static_cast<const CspArrayType *>( type() ) -> elemType().get();
        return type();
    }

protected:
    RootEngine * m_rootEngine;
    PushMode     m_pushMode;
    bool         m_started;
};

template<typename T>
bool InputAdapter::consumeTick( const T & value )
{
    switch( pushMode() )
    {
        case PushMode::LAST_VALUE:
        {
            if( unlikely( rootEngine() -> cycleCount() == lastCycleCount() ) )
                m_timeseries -> lastValueTyped<T>() = value;
            else
                this -> outputTickTyped<T>( rootEngine() -> now(), value );
            return true;
        }

        case PushMode::BURST:
        {
            CSP_ASSERT( type() -> type() == CspType::Type::ARRAY );
            CSP_ASSERT( static_cast<const CspArrayType * >( type() ) -> elemType() -> type() == CspType::Type::fromCType<T>::type );

            using ArrayT = typename CspType::Type::toCArrayType<T>::type;
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

            this -> outputTickTyped<T>( rootEngine() -> now(), value );
            return true;
        }

        default:
            CSP_THROW( NotImplemented, pushMode() << " mode is not yet supported" );
            break;
    }

    return true;
}

};

#endif
