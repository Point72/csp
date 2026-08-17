#include "CounterAdapterManager.h"
#include "CounterInputAdapter.h"
#include "CounterOutputAdapter.h"
#include <csp/engine/Engine.h>
#include <chrono>

namespace csp::adapters::counter
{

CounterAdapterManager::CounterAdapterManager( csp::Engine * engine, const Dictionary & properties )
    : AdapterManager( engine ),
      m_intervalMs( 1000 ),
      m_maxCount( 0 ),
      m_running( false ),
      m_inputAdapter( nullptr ),
      m_outputAdapter( nullptr )
{
    // Extract configuration from properties
    if( properties.exists( "interval_ms" ) )
        m_intervalMs = properties.get<int64_t>( "interval_ms" );

    if( properties.exists( "max_count" ) )
        m_maxCount = properties.get<int64_t>( "max_count" );
}

CounterAdapterManager::~CounterAdapterManager()
{
    stop();
}

void CounterAdapterManager::start( DateTime starttime, DateTime endtime )
{
    // Call base class start
    AdapterManager::start( starttime, endtime );

    // Start the push thread
    m_running = true;
    m_pushThread = std::make_unique<std::thread>( &CounterAdapterManager::runPushThread, this );
}

void CounterAdapterManager::stop()
{
    // Signal the thread to stop
    m_running = false;

    // Wait for thread to complete
    if( m_pushThread && m_pushThread -> joinable() )
    {
        m_pushThread -> join();
        m_pushThread.reset();
    }

    // Call base class stop
    AdapterManager::stop();
}

PushInputAdapter * CounterAdapterManager::getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties )
{
    // For this simple example, we only support one input adapter
    if( m_inputAdapter != nullptr )
        CSP_THROW( RuntimeException, "CounterAdapterManager only supports one input adapter" );

    m_inputAdapter = engine() -> createOwnedObject<CounterInputAdapter>( type, pushMode, &m_pushGroup );
    return m_inputAdapter;
}

OutputAdapter * CounterAdapterManager::getOutputAdapter( CspTypePtr & type, const Dictionary & properties )
{
    // For this simple example, we only support one output adapter
    if( m_outputAdapter != nullptr )
        CSP_THROW( RuntimeException, "CounterAdapterManager only supports one output adapter" );

    m_outputAdapter = engine() -> createOwnedObject<CounterOutputAdapter>( type );
    return m_outputAdapter;
}

void CounterAdapterManager::pushValue( int64_t value )
{
    if( m_inputAdapter )
    {
        PushBatch batch( rootEngine() );
        m_inputAdapter -> pushTick( value, &batch );
    }
}

void CounterAdapterManager::runPushThread()
{
    int64_t counter = 0;

    while( m_running )
    {
        // Sleep for the configured interval
        std::this_thread::sleep_for( std::chrono::milliseconds( m_intervalMs ) );

        if( !m_running )
            break;

        // Push the counter value
        counter++;
        pushValue( counter );

        // Check if we've reached the max count
        if( m_maxCount > 0 && counter >= m_maxCount )
            break;
    }
}

}
