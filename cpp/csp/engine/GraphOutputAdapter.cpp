#include <csp/engine/GraphOutputAdapter.h>
#include <csp/engine/TimeSeriesProvider.h>

namespace csp
{

GraphOutputAdapter::GraphOutputAdapter( csp::Engine * engine, int32_t tickCount, TimeDelta tickHistory ) :
    OutputAdapter( engine ),
    m_tickHistory( tickHistory ),
    m_tickCount( tickCount )
{
}

GraphOutputAdapter::~GraphOutputAdapter()
{
}

const char * GraphOutputAdapter::name() const
{
    return "GraphOutputAdapter";
}

void GraphOutputAdapter::start()
{
    //The GraphOutputAdapter is just a sync to keep the timeseries alive
    //as well as set proper buffering on it.  its on different language impls ( ie PyEngine )
    //to keep track of which outputs to collect when done

    //passive input
    input() -> removeConsumer( this, InputId( 0 ) );

    bool hasTimeHistory = !m_tickHistory.isNone() && m_tickHistory > TimeDelta::ZERO();

    if( m_tickCount > 0 )
        input() -> setTickCountPolicy( m_tickCount );
    else if( m_tickCount < 0 && !hasTimeHistory )
        input() -> setTickTimeWindowPolicy( TimeDelta::MAX_VALUE() );

    if( hasTimeHistory )
        input() -> setTickTimeWindowPolicy( m_tickHistory );
}

void GraphOutputAdapter::stop()
{
    //for dynamic engines the timeseries that this is referring to will most likely not exist after the GraphOutputAdapter is stopped.  For this reason, 
    //we pre-convert the requested data up front so its cached until the end of the root engine. 
    if( !engine() -> isRootEngine() )
        processResults();
}

}
