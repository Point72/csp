#include <csp/engine/OutputAdapter.h>
#include <csp/engine/TimeSeriesProvider.h>

namespace csp
{

OutputAdapter::OutputAdapter( csp::Engine * engine ) : Consumer( engine ),
                                                       m_input( nullptr )
{
}

OutputAdapter::~OutputAdapter()
{
}

void OutputAdapter::link( TimeSeriesProvider * input )
{
    if( m_input )
        CSP_THROW( ValueError, "Attempted to link input to output adapter " << name() << " multiple times" );
    m_input = input;
    input -> addConsumer( this, InputId( 0 ) );
}

}
