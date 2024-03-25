#include <csp/engine/AdapterManager.h>
#include <csp/engine/PartialSwitchCspType.h>

namespace csp
{

ManagedSimInputAdapter::ManagedSimInputAdapter( csp::Engine * engine,
                                                const CspTypePtr & type,
                                                AdapterManager *manager,
                                                PushMode pushMode ) : InputAdapter( engine, type, pushMode ),
                                                                      m_manager( manager ),
                                                                      m_lastCycleCount( 0 )
{
}

AdapterManager::AdapterManager( csp::Engine * engine ) : m_engine( engine ), m_statusAdapter( nullptr ), m_started( false )
{
    if( !m_engine -> isRootEngine() )
        CSP_THROW( NotImplemented, "AdapterManager support is not currently available in dynamic graphs" );
}

AdapterManager::~AdapterManager()
{
}

void AdapterManager::start( DateTime starttime, DateTime endtime )
{
    m_starttime = starttime;
    m_endtime   = endtime;

    scheduleTimerCB( starttime );
}

void AdapterManager::stop()
{
}

void AdapterManager::processSimTimerCB()
{
    DateTime next = processNextSimTimeSlice( rootEngine() -> now() );
    if( !next.isNone() )
        scheduleTimerCB( next );
}

StatusAdapter * AdapterManager::createStatusAdapter( CspTypePtr & type, PushMode pushMode )
{
    if( !m_statusAdapter )
        m_statusAdapter = m_engine -> createOwnedObject<StatusAdapter>( type, pushMode, statusPushGroup() );

    return m_statusAdapter;
}

void AdapterManager::pushStatus( int64_t level, int64_t errCode, const std::string & errMsg, PushBatch *batch ) const
{
    if( m_statusAdapter )
        m_statusAdapter -> pushStatus( level, errCode, errMsg, batch );
}

}
