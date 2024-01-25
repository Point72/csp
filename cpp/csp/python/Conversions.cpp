#include <csp/python/Conversions.h>
#include <datetime.h>

namespace csp::python
{

PyObject * lastValueToPython( const csp::TimeSeriesProvider * ts )
{
    return switchCspType( ts -> type(),
                          [ ts ]( auto tag )
                          {
                              return toPython( ts -> lastValueTyped<typename decltype(tag)::type>(), *ts -> type() );
                          } );
}

PyObject * valueAtIndexToPython( const csp::TimeSeriesProvider * ts, int32_t index)
{
    return switchCspType( ts -> type(),
                          [ ts, index ]( auto tag )
                          {
                              return toPython( ts -> valueAtIndex<typename decltype(tag)::type>( index ), *ts -> type() );
                          } );
}

}
