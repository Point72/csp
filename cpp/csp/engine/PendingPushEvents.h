#ifndef _IN_CSP_ENGINE_PENDINGPUSHEVENTS_H
#define _IN_CSP_ENGINE_PENDINGPUSHEVENTS_H

#include <csp/engine/PushEvent.h>
#include <unordered_map>

namespace csp
{

struct PushGroup;
class PushInputAdapters;

class PendingPushEvents
{
public:
    PendingPushEvents();
    ~PendingPushEvents();

    void addPendingEvent( PushEvent * event );

    void processPendingEvents( std::vector<PushGroup*> & dirtyGroups );

    bool hasEvents() const { return !m_ungroupedEvents.empty() || !m_groupedEvents.empty(); }

private:
    struct EventList
    {
        PushEvent * head;
        PushEvent * tail;
    };

    //GroupEvents are for inputs that are bound to a group.  They will never progress past a group event
    //that has a NON_COLLAPSING input in it
    using GroupEvents     = std::unordered_map<PushGroup*,EventList>;

    //Ungrouped events are for NON_COLLAPSING PushInputAdapters that can only process one tick per cycle
    //we store them by adapter here so that we dont have to rescan all events every cycle to pop a single 
    //backed up input ( if we merged adaptrers into one list we would have to rescan every cycle )
    using UngroupedEvents = std::unordered_map<PushInputAdapter*,EventList>;

    void deleteEvent( PushEvent * event );
    
    UngroupedEvents m_ungroupedEvents;
    GroupEvents     m_groupedEvents;
};

}

#endif
