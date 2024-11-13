#ifndef _IN_CSP_ENGINE_FEEDBACK_H
#define _IN_CSP_ENGINE_FEEDBACK_H

#include <csp/engine/InputAdapter.h>
#include <csp/engine/OutputAdapter.h>

namespace csp
{

template<typename T>
class FeedbackInputAdapter;

template<typename T>
class FeedbackOutputAdapter final : public OutputAdapter
{
public:
    FeedbackOutputAdapter( csp::Engine * engine, 
                           InputAdapter * boundInput );
    ~FeedbackOutputAdapter() {}

    void executeImpl() override;

    const char * name() const override { return "feedback"; }

private:

    FeedbackInputAdapter<T> * m_boundInput;
};

template<typename T>
class FeedbackInputAdapter final : public InputAdapter
{
public:
    using InputAdapter::InputAdapter;

    void stop() override
    {
        rootEngine() -> cancelCallback( m_timerHandle );
    }

    void pushTick( const T & value )
    {
        m_timerHandle = rootEngine() -> scheduleCallback( TimeDelta::ZERO(), [this,value]() 
                                                          {
                                                              return this -> consumeTick( value ) ? nullptr : this;
                                                          } );
    }
private:
    Scheduler::Handle m_timerHandle;
};

template<typename T>
inline FeedbackOutputAdapter<T>::FeedbackOutputAdapter( csp::Engine * engine, 
                                                        InputAdapter * boundInput ) : OutputAdapter( engine )
{
    m_boundInput = dynamic_cast<FeedbackInputAdapter<T> *>( boundInput );
    if( !m_boundInput )
        CSP_THROW( TypeError, "FeedbackOutputAdapter expected boundInput of type FeedbackOutputAdapter<T> ( " <<
                   typeid(T).name() << " ) got " << typeid( *boundInput ).name() );
}

template<typename T>
inline void FeedbackOutputAdapter<T>::executeImpl()
{
    m_boundInput -> pushTick( input() -> template lastValueTyped<T>() );
}

}

#endif
