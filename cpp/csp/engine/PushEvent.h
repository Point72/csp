#ifndef _IN_CSP_ENGINE_PUSHEVENT_H
#define _IN_CSP_ENGINE_PUSHEVENT_H

#include <csp/engine/CspType.h>

namespace csp
{

class PushInputAdapter;

struct PushEvent
{
    PushEvent( PushInputAdapter *adapter ) : m_adapter( adapter ), next( nullptr )
    {}

    PushInputAdapter * adapter() { return ( PushInputAdapter * ) ( ( ( uint64_t ) m_adapter ) & ~0x1 ); }
    bool isGroupEnd() const      { return ( ( uint64_t ) m_adapter ) & 0x1;  }
    void flagGroupEnd()          { m_adapter = ( PushInputAdapter * )( ( ( uint64_t ) m_adapter ) | 0x1 ); }

private:
    PushInputAdapter * m_adapter;

public:
    PushEvent        * next;
};

template<typename T>
struct TypedPushEvent : public PushEvent
{
    TypedPushEvent( PushInputAdapter *adapter,
                    T &&d ) : PushEvent( adapter ),
                              data( std::forward<T>( d ) )
    {}

    typename std::remove_reference<T>::type data;
};

}

#endif
