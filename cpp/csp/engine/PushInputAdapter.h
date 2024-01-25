#ifndef _IN_CSP_ENGINE_PUSHINPUTADAPTER_H
#define _IN_CSP_ENGINE_PUSHINPUTADAPTER_H

#include <csp/core/Time.h>
#include <csp/engine/CspType.h>
#include <csp/engine/InputAdapter.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/PushEvent.h>
#include <csp/engine/RootEngine.h>
#include <csp/engine/Struct.h>

namespace csp
{

/*
Input adapters can be combined into "groups" where they wont tick ahead or get out of sync
if they are in the same group.  Classic example is bid/ask/trade push adapters.  They should
be joined with one group so that collapsed bid/asks cant get ahead of non-collapsing trades

Deeper explanation of the impl... 

PushInputAdapter can optionally be registered with a PushGroup object.  This will generally be created and
maintained on the AdapterMAnager servicing the PushInputAdapter ( it may have one or many, ie a MktDataHub
would likely have one group for the hub ).  When a push input adapter is part a group, there are implications
specifically when dealing with non-collapsing inputs that are part of the group.  To maintain "synchronicity" 
of data across inputs in the group, an engine cycle processing a non-collapsing event must ensure that other
LAST_VALUE ( aka collapsing ) inputs do no advance past that non-collapsing event.  Think of bid/ask ticking 
and collapsing, but trade coming in non-collapsed.  Whenever trade ticks, you want the cycle to have the bid/ask 
as of the time of the trade, not to collapse further into the future.

Inputs that tick in a group can / should be submitted to the engine as part of a PushBatch.  Events will be added to the batch
and atomically released to the engine thread.  The last event of the push batch is flagged.

PushGroups can be in one of 3 states.  At the start of a given cycle they are all NONE.  
If a non-collapsing tick comes in, state goes to LOCKING until the end of that group's events in the given batch.  At that point it goes to LOCKED.
while a group is in LOCKING, other inputs within the same batch can still get processed ( note there is a TODO for a non-collapsing input that has
two events in the same batch ).  Once a group is in a LOCKED state events get put in ( or remain ) in the PendingEvents queue for the next engine cycle

For PushInputAdapters that are non-collapsing but have NO group, they get put into the "ungrouped" events in PendingEvents to apply on subsequent
cycles as well
*/

struct PushGroup
{
    enum State { NONE, LOCKING, LOCKED };
    State state = NONE;
};

class PushBatch : public SRMWLockFreeQueue<PushEvent>::Batch
{
    using BaseT = SRMWLockFreeQueue<PushEvent>::Batch;

public:
    PushBatch( RootEngine * engine ) : m_rootEngine( engine ) , m_group( nullptr )
    {
        m_group = nullptr;
    }

    ~PushBatch() { flush(); }

    void append( PushEvent * event );

    void flush()
    {
        if( !empty() )
        {
            last() -> flagGroupEnd();
            m_rootEngine -> schedulePushBatch( *this );
        }
        m_group = nullptr;
    }

private:
    RootEngine* m_rootEngine;
    PushGroup * m_group; //used for assertion check
};

class PushInputAdapter : public InputAdapter
{
public:
    PushInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode,
                      PushGroup * group = nullptr ) : InputAdapter( engine, type, pushMode ),
                                                      m_group( group )
    {
    }

    PushGroup * group()       { return m_group; }

    //called from adapter thread
    template<typename T>
    void pushTick( T &&value, PushBatch *batch = nullptr );

    //called from engine processing thread
    //will delete event if processed, returns true if processed
    bool consumeEvent( PushEvent *, std::vector<PushGroup *> & dirtyGroups );

private:

    PushGroup * m_group;
};

template<typename T>
inline void PushInputAdapter::pushTick( T &&value, PushBatch *batch )
{
    static_assert( std::is_trivially_move_constructible<typename std::remove_reference<T>::type>::value ||
                   std::is_rvalue_reference<decltype( value )>::value, "Push tick must be rvalue or native type" );
    //TBD allocators
    PushEvent * event = new TypedPushEvent<T>( this, std::forward<T>(value) );
    if( batch )
    {
        batch -> append( event );
        return;
    }

    //flag it as a group ending event if its part of a group
    if( m_group )
        event -> flagGroupEnd();

    rootEngine() -> schedulePushEvent( event );
}

inline bool PushInputAdapter::consumeEvent( PushEvent * event, std::vector<PushGroup *> & dirtyGroups )
{
    if( m_group && m_group -> state == PushGroup::LOCKED )
        return false;

    bool isGroupEnd = event -> isGroupEnd();

    bool consumed = switchCspType( dataType(),
                                   [ event ]( auto tag )
                                   {
                                       using T = typename decltype(tag)::type;
                                       TypedPushEvent<T> *tevent  = static_cast<TypedPushEvent<T> *>( event );
                                       bool              consumed = event -> adapter() -> consumeTick( tevent -> data );
                                       if( consumed )
                                           delete tevent;
                                       return consumed;
                                   } );
    if( m_group )
    {
        if( pushMode() == PushMode::NON_COLLAPSING )
        {
            //could be LOCKING from another NC input in the group but shouldnt be LOCKED
            assert( m_group -> state != PushGroup::LOCKED );
            m_group -> state = PushGroup::LOCKING;
        }

        if( m_group -> state == PushGroup::LOCKING && isGroupEnd )
        {
            m_group -> state = PushGroup::LOCKED;
            dirtyGroups.emplace_back( m_group );
        }
    }

    return consumed;
}

inline void PushBatch::append( PushEvent * event )
{
    CSP_ASSERT( m_group == nullptr || m_group == event -> adapter() -> group() );
    m_group = event -> adapter() -> group();

    BaseT::append( event );
}

}

#endif
