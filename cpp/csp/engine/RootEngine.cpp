#include <csp/engine/Dictionary.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/RootEngine.h>
#include <csp/engine/InputAdapter.h>
#include <csp/engine/Node.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/engine/PushInputAdapter.h>
#include <signal.h>
#include <mutex>

namespace csp
{

static volatile bool    g_SIGNALED = false;
static struct sigaction g_prevSIGTERMaction;

static void handle_SIGTERM( int signum )
{
    g_SIGNALED = true;
    if( g_prevSIGTERMaction.sa_handler )
        (*g_prevSIGTERMaction.sa_handler)( signum );
}

static bool install_signal_handlers()
{
    static bool s_installed = false;
    static std::mutex s_lock;
    if( !s_installed )
    {
        std::lock_guard<std::mutex> guard( s_lock );
        if( !s_installed )
        {
            struct sigaction newhandler;
            sigemptyset( &newhandler.sa_mask );
            newhandler.sa_handler = handle_SIGTERM;
            newhandler.sa_flags = 0;

            if( sigaction(SIGINT,&newhandler, &g_prevSIGTERMaction ) != 0 )
                printf( "Failed to set SIGTERM handler: %s", strerror( errno ) );
            
            s_installed = true;
        }
    }
    return true;
}

static bool s_install_handlers = install_signal_handlers();

RootEngine::Settings::Settings( const Dictionary & settings )
{
    queueWaitTime = settings.get<TimeDelta>( "queue_wait_time", TimeDelta::fromMilliseconds( 100 ) );
    realtime      = settings.get<bool>( "realtime",        false );
}

RootEngine::RootEngine( const Dictionary & settings ) : Engine( m_cycleStepTable ),
                                                        m_now( DateTime::NONE() ),
                                                        m_state( State::NONE ),
                                                        m_cycleCount( 0 ),
                                                        m_settings( settings ),
                                                        m_inRealtime( false ),
                                                        m_pushEventQueue( m_settings.queueWaitTime > TimeDelta::ZERO() )
{
    if( settings.get<bool>( "profile",  false ) )
    {
        m_profiler = std::make_unique<csp::Profiler>();
        std::string cycle_fname = settings.get<std::string>( "cycle_profile_file", "" );
        std::string node_fname = settings.get<std::string>( "node_profile_file", "" );
        if( !cycle_fname.empty() )
            m_profiler.get() -> use_prof_file( cycle_fname, false );
        if( !node_fname.empty() )
            m_profiler.get() -> use_prof_file( node_fname, true );
    }
}

RootEngine::~RootEngine()
{
}

bool RootEngine::interrupted() const
{
    return g_SIGNALED;
}

void RootEngine::preRun( DateTime start, DateTime end )
{
    m_state = State::STARTING;

    m_now = start;
    m_startTime = start;
    m_endTime = end;

    Engine::start();
}


void RootEngine::postRun()
{
    m_state = State::SHUTDOWN;
    stop();
}

void RootEngine::processPendingPushEvents( std::vector<PushGroup *> & dirtyGroups )
{
    m_pendingPushEvents.processPendingEvents( dirtyGroups );
}

void RootEngine::processPushEventQueue( PushEvent * event, std::vector<PushGroup *> & dirtyGroups )
{
    while( event )
    {
        PushEvent * next = event -> next;

        if( !event -> adapter() -> consumeEvent( event, dirtyGroups ) )
            m_pendingPushEvents.addPendingEvent( event );

        event = next;
    }
}

void RootEngine::processEndCycle()
{
    for( auto * listener : m_endCycleListeners )
    {
        listener -> onEndCycle();
        listener -> clearDirtyFlag();
    }

    m_endCycleListeners.clear();
}

void RootEngine::runSim( DateTime end )
{
    m_inRealtime = false;
    while( m_scheduler.hasEvents() && m_state == State::RUNNING && !g_SIGNALED )
    {
        m_now = m_scheduler.nextTime();
        if( m_now > end )
            break;

        ++m_cycleCount;

        m_scheduler.executeNextEvents( m_now );
        m_cycleStepTable.executeCycle( m_profiler.get() );

        processEndCycle();
    }

    m_now = std::min( m_now, end );
}

void RootEngine::runRealtime( DateTime end )
{
    struct DialectReleaseGIL
    {
        DialectReleaseGIL( RootEngine * e ) : engine( e ) { engine -> dialectUnlockGIL(); }
        ~DialectReleaseGIL()                              { engine -> dialectLockGIL(); }
        RootEngine * engine;
    };

    std::vector<PushGroup *> dirtyGroups;

    m_inRealtime = true;
    while( m_state == State::RUNNING && !g_SIGNALED )
    {
        TimeDelta waitTime;
        if( !m_pendingPushEvents.hasEvents() )
        {
            DateTime now = DateTime::now();
            waitTime = std::min( m_endTime - now, m_settings.queueWaitTime );
            if( m_scheduler.hasEvents() )
                waitTime = std::min( m_scheduler.nextTime() - DateTime::now(), waitTime );
        }

        {
            DialectReleaseGIL release( this );
            m_pushEventQueue.wait( waitTime );
        }

        //grab time after waitForEvents so that we dont grab events with time > now
        m_now = DateTime::now();
        if( m_now > end )
            break;

        ++m_cycleCount;

        //We made a conscious decision to execute timers exactly on their requested time in realtime mode
        //therefore timers that are ready are executed on their own time cycle before realtime events all processed
        if( m_scheduler.hasEvents() && m_scheduler.nextTime() < m_now )
        {
            m_now = m_scheduler.nextTime();
            m_scheduler.executeNextEvents( m_now );
        }
        else
        {
            PushEvent * events = m_pushEventQueue.popAll();

            processPendingPushEvents( dirtyGroups );
            processPushEventQueue( events, dirtyGroups );

            for( auto * group : dirtyGroups )
                group -> state = PushGroup::NONE;
            
            dirtyGroups.clear();
        }

        m_cycleStepTable.executeCycle( m_profiler.get() );

        processEndCycle();
    }

    m_now = std::min( m_now, end );
}

void RootEngine::run( DateTime start, DateTime end )
{
    try
    {
        preRun( start, end );
        m_exception_mutex.lock();
        if( m_state != State::SHUTDOWN )
            m_state = State::RUNNING;
        m_exception_mutex.unlock();

        if( m_settings.realtime )
        {
            DateTime rtNow = DateTime::now();
            runSim( std::min( rtNow, end ) );
            if( end > rtNow )
                runRealtime( end );
        }
        else
            runSim( end );
    }
    catch( ... )
    {
        m_exception_ptr = std::current_exception();
    }

    try
    {
        postRun();
    }
    catch( ... )
    {
        if( !m_exception_ptr )
            m_exception_ptr = std::current_exception();
    }

    m_state = State::DONE;

    if( m_exception_ptr )
        std::rethrow_exception( m_exception_ptr );
}

void RootEngine::shutdown()
{
    m_state = State::SHUTDOWN;
}

void RootEngine::shutdown( std::exception_ptr except_ptr )
{
    std::lock_guard<std::mutex> guard( m_exception_mutex );
    m_state = State::SHUTDOWN;
    if( !m_exception_ptr )
        m_exception_ptr = except_ptr;
}

DictionaryPtr RootEngine::engine_stats() const
{
    if( !m_profiler )
        CSP_THROW( ValueError, "Cannot profile a graph unless the graph is run in a profiler context." );

    return m_profiler -> getAllStats( m_nodes.size() );
}


}
