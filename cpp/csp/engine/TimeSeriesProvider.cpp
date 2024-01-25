#include <csp/engine/Node.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <csp/engine/TimeSeries.h>
#include <csp/engine/TimeSeriesProvider.h>

namespace csp
{

void TimeSeriesProvider::init( const CspTypePtr & type, Node * node )
{
    m_node = node;
    m_type = type.get();

    switchCspType( type, [ this ]( auto tag )
                   {
                       this -> initBuffer<typename decltype(tag)::type>();
                   } );
}

void TimeSeriesProvider::reset()
{
    switchCspType( m_type, [ this ]( auto tag )
    {
        using T = typename decltype(tag)::type;
        static_cast<TimeSeriesTyped<T> *>( m_timeseries.get() ) -> reset();
    } );

    m_lastCycleCount = 0;
    m_propagator.clear();
}

bool TimeSeriesProvider::addConsumer( Consumer * consumer, InputId id, bool checkExists )
{
    return m_propagator.addConsumer( consumer, id, checkExists );
}

bool TimeSeriesProvider::removeConsumer( Consumer * consumer, InputId id )
{
    return m_propagator.removeConsumer( consumer, id );
}

const char * TimeSeriesProvider::name() const
{
    return m_node ? m_node -> name() : "";
}

}
