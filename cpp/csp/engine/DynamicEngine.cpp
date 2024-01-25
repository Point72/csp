#include <csp/engine/DynamicEngine.h>

namespace csp
{

DynamicEngine::GraphOutput::GraphOutput( Engine * engine ) : OutputAdapter( engine )
{
}

DynamicEngine::GraphOutput::~GraphOutput()
{
}

void DynamicEngine::GraphOutput::start()
{
    input() -> removeConsumer( this, InputId( 0 ) );
}


DynamicEngine::DynamicEngine( CycleStepTable & stepTable, RootEngine * root,
                              ShutdownFn && shutdownFn ) : Engine( stepTable, root ),
                                                           m_shutdownFn( std::move( shutdownFn ) )
{
}

void DynamicEngine::registerGraphOutput( const std::string & key, GraphOutput * output )
{
    //no need to check for clash, output names are gauranteed unique
    m_graphOutputs[key] = output;
}

TimeSeriesProvider * DynamicEngine::outputTs( const std::string & key )
{
    return m_graphOutputs[key] -> input();
}

void DynamicEngine::shutdown()
{
    m_shutdownFn();
}

}
