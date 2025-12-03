#ifndef _IN_CSP_ENGINE_PUSHPULLEVENT_H
#define _IN_CSP_ENGINE_PUSHPULLEVENT_H

namespace csp
{

class PushPullInputAdapter;

struct PushPullEvent
{
    PushPullEvent( PushPullInputAdapter *adapter_, DateTime time_ ) : time( time_ ),
                                                                      adapter( adapter_ ),
                                                                      next( nullptr )
    {}

    DateTime               time;
    PushPullInputAdapter * adapter;
    PushPullEvent        * next;
};

template<typename T>
struct TypedPushPullEvent : public PushPullEvent
{
    TypedPushPullEvent( PushPullInputAdapter *adapter, DateTime time,
                        T &&d ) : PushPullEvent( adapter, time ),
                                  data( std::forward<T>( d ) )
    {}

    typename std::remove_reference<T>::type data;
};

}

#endif
