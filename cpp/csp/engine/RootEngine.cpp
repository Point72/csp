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

static volatile int     g_SIGNAL_COUNT = 0;
/*
The signal count variable is maintained to ensure that multiple engine threads shutdown properly.

An interrupt should cause all running engines to stop, but should not affect future runs in the same process.
Thus, each root engine keeps track of the signal count when its created. When an interrupt occurs, one engine thread
handles the interrupt by incrementing the count. Then, all other root engines detect the signal by comparing their
initial count to the current count.

Future runs after the interrupt remain unaffected since they are initialized with the updated signal count, and will
only consider themselves "interupted" if another signal is received during their execution.
*/

static void (*g_prevSIGTERMhandler)(int) = nullptr;

static void handle_SIGTERM( int signum )
{
    g_SIGNAL_COUNT++;

    if(g_prevSIGTERMhandler)
        (*g_prevSIGTERMhandler)( signum );
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
#ifndef WIN32
            struct sigaction newhandler, prev_handler;
            sigemptyset( &newhandler.sa_mask );
            newhandler.sa_handler = handle_SIGTERM;
            newhandler.sa_flags = 0;

            if( sigaction(SIGINT,&newhandler, &prev_handler) != 0 )
                printf( "Failed to set SIGTERM handler: %s", strerror( errno ) );
            g_prevSIGTERMhandler = prev_handler.sa_handler;
#else
            g_prevSIGTERMhandler = signal(SIGINT, handle_SIGTERM);
#endif
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
                                                        m_haveEvents( false ),
                                                        m_initSignalCount( g_SIGNAL_COUNT ),
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
    return g_SIGNAL_COUNT != m_initSignalCount;
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

        auto * pending_event = event -> adapter() -> consumeEvent( event, dirtyGroups );
        if( pending_event != nullptr )
            m_pendingPushEvents.addPendingEvent( pending_event );

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

bool RootEngine::processOneCycle( TimeDelta maxWait )
{
    if( m_state != State::RUNNING || interrupted() )
        return false;

    // Check if we've passed the end time
    if( m_now > m_endTime )
        return false;

    bool hasWork = false;

    if( m_inRealtime )
    {
        // Realtime mode: check for push events and timers
        struct DialectReleaseGIL
        {
            DialectReleaseGIL( RootEngine * e ) : engine( e ) { engine -> dialectUnlockGIL(); }
            ~DialectReleaseGIL()                              { engine -> dialectLockGIL(); }
            RootEngine * engine;
        };

        TimeDelta waitTime = maxWait;
        if( maxWait > TimeDelta::ZERO() && !m_pendingPushEvents.hasEvents() )
        {
            // Only compute wait bounds when maxWait > 0 (i.e. caller wants to block).
            // maxWait == ZERO means non-blocking poll – skip the wait entirely.
            DateTime now = DateTime::now();
            TimeDelta timeToEnd = m_endTime - now;
            if( timeToEnd < waitTime )
                waitTime = timeToEnd;
            if( m_scheduler.hasEvents() )
            {
                TimeDelta timeToNext = m_scheduler.nextTime() - now;
                if( timeToNext < waitTime )
                    waitTime = timeToNext;
            }
        }

        if( !m_haveEvents && waitTime > TimeDelta::ZERO() )
        {
            // We keep the m_haveEvents flag in case there were events, but we only decided to execute
            // timers in the previous cycle, then we shouldn't wait again (which can lead to missed triggers)
            DialectReleaseGIL release( this );
            m_haveEvents = m_pushEventQueue.wait( waitTime );
        }

        // Also check the queue directly – events may have been pushed
        // between cycles (e.g. from asyncio coroutines) without going
        // through the blocking wait path.
        if( !m_haveEvents )
            m_haveEvents = !m_pushEventQueue.empty();

        // Grab time after wait so we don't grab events with time > now
        m_now = DateTime::now();
        if( m_now > m_endTime )
        {
            m_now = m_endTime;
            return false;
        }

        ++m_cycleCount;

        // Execute timers exactly on their requested time - timers that are ready
        // are executed on their own time cycle before realtime push events
        if( m_scheduler.hasEvents() && m_scheduler.nextTime() < m_now )
        {
            m_now = m_scheduler.nextTime();
            m_scheduler.executeNextEvents( m_now );
            hasWork = true;
        }
        else if( m_haveEvents || m_pendingPushEvents.hasEvents() )
        {
            // Process push events
            PushEvent * events = m_pushEventQueue.popAll();

            processPendingPushEvents( m_dirtyGroups );
            processPushEventQueue( events, m_dirtyGroups );

            for( auto * group : m_dirtyGroups )
                group -> state = PushGroup::NONE;

            m_dirtyGroups.clear();
            m_haveEvents = false;
            hasWork = true;
        }
    }
    else
    {
        // Sim mode: process next scheduled event
        if( m_scheduler.hasEvents() )
        {
            m_now = m_scheduler.nextTime();
            if( m_now <= m_endTime )
            {
                ++m_cycleCount;
                m_scheduler.executeNextEvents( m_now );
                hasWork = true;
            }
            else
            {
                m_now = m_endTime;
                return false;
            }
        }
    }

    if( hasWork )
    {
        m_cycleStepTable.executeCycle( m_profiler.get() );
        processEndCycle();
    }

    // In realtime mode, keep running until endtime (push events can arrive
    // at any time from external threads or asyncio coroutines).
    // In sim mode, stop when there are no more scheduled events.
    if( m_inRealtime )
        return m_state == State::RUNNING && !interrupted();
    else
        return m_state == State::RUNNING && !interrupted() &&
               ( m_scheduler.hasEvents() || m_pendingPushEvents.hasEvents() );
}

void RootEngine::start( DateTime startTime, DateTime end )
{
    preRun( startTime, end );

    m_exception_mutex.lock();
    if( m_state != State::SHUTDOWN )
        m_state = State::RUNNING;
    m_exception_mutex.unlock();

    m_inRealtime = m_settings.realtime;
    m_haveEvents = false;
    m_dirtyGroups.clear();

    // In realtime mode, if start is in the past, first run through historical events
    if( m_settings.realtime )
    {
        DateTime rtNow = DateTime::now();
        if( startTime < rtNow )
        {
            // Temporarily disable realtime to process historical sim events
            m_inRealtime = false;
            DateTime simEnd = std::min( rtNow, end );
            while( m_scheduler.hasEvents() && m_state == State::RUNNING && !interrupted() )
            {
                DateTime nextTime = m_scheduler.nextTime();
                if( nextTime > simEnd )
                    break;
                m_now = nextTime;
                ++m_cycleCount;
                m_scheduler.executeNextEvents( m_now );
                m_cycleStepTable.executeCycle( m_profiler.get() );
                processEndCycle();
            }
            m_now = std::min( m_now, simEnd );

            // Only switch to realtime if end is still in the future
            if( rtNow < end )
                m_inRealtime = true;
        }
    }
}

void RootEngine::finish()
{
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
    m_dirtyGroups.clear();

    if( m_exception_ptr )
        std::rethrow_exception( m_exception_ptr );
}

void RootEngine::run( DateTime startTime, DateTime end )
{
    try
    {
        start( startTime, end );

        // Main loop - same code path whether called via run() or step()
        while( processOneCycle( m_settings.queueWaitTime ) ) {}
    }
    catch( ... )
    {
        m_exception_ptr = std::current_exception();
    }

    finish();
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

DateTime RootEngine::nextScheduledTime()
{
    if( m_scheduler.hasEvents() )
        return m_scheduler.nextTime();
    return DateTime::NONE();
}

DictionaryPtr RootEngine::engine_stats() const
{
    if( !m_profiler )
        CSP_THROW( ValueError, "Cannot profile a graph unless the graph is run in a profiler context." );

    return m_profiler -> getAllStats( m_nodes.size() );
}


}
