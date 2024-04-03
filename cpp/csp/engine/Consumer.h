#ifndef _IN_CSP_ENGINE_CONSUMER_H
#define _IN_CSP_ENGINE_CONSUMER_H

#include <csp/core/TaggedPointerUnion.h>
#include <csp/engine/BasketInfo.h>
#include <csp/engine/InputId.h>
#include <csp/engine/RootEngine.h>
#include <stdint.h>

namespace csp
{

class InputAdapter;

//Base of either regular Nodes or output adapters
class Consumer
{
public:
    Consumer( Engine * );
    virtual ~Consumer();
    Consumer(const Consumer&) = delete;
    Consumer(Consumer&&) = delete;
    Consumer& operator=(const Consumer&) = delete;
    Consumer& operator=(const Consumer&&) = delete;

    virtual void start();
    virtual void stop();

    virtual const char * name() const = 0;

    Engine * engine() const         { return m_engine; }
    RootEngine * rootEngine() const { return m_engine -> rootEngine(); }

    DateTime now() const            { return rootEngine() -> now(); }
    uint64_t cycleCount() const     { return rootEngine() -> cycleCount(); }

    void setStarted()               { m_started = true; }
    bool started() const            { return m_started; }

    //called when input timeseries has an event, schedules in
    //step propagation.  See if we can do better than virtual per tick...
    virtual void handleEvent( InputId id )
    {
        m_engine -> scheduleConsumer( this );
    }

    void execute()
    {
        executeImpl();
    }

    //actual logic
    virtual void executeImpl() = 0;

    //internals

    //graph / links / creation
    int32_t rank() const            { return m_rank; }
    void    setRank( int32_t rank ) { m_rank = rank; }
    
    Consumer * next()               { return m_next; }
    void setNext( Consumer * next ) { m_next = next; }

    //flattens out all inputs and basket inputs into one iteration of all input timeseries
    struct input_iterator
    {
        using InputT = TaggedPointerUnion<TimeSeriesProvider,InputBasketInfo>;

        //single input ( OutputAdapter ) case
        input_iterator( const TimeSeriesProvider * const * input )
        {
            m_id = 0;
            //safe cast since TaggedPointerUnion is a single ptr, and isSet<basket> will be false
            m_inputiter = ( InputT * ) input;
            m_inputend  = m_inputiter + 1;
        }

        input_iterator( const InputT * inputs, size_t num_inputs )
        {
            m_id = 0;
            m_inputiter = inputs;
            m_inputend  = inputs + num_inputs;

            if( m_inputiter != m_inputend && m_inputiter -> isSet<InputBasketInfo>() )
                m_basketIter = m_inputiter -> get<InputBasketInfo>() -> begin_inputs( true );
        }
        
        const TimeSeriesProvider * get() const { return m_basketIter ? m_basketIter.get() : m_inputiter -> get<TimeSeriesProvider>(); }
        const TimeSeriesProvider * operator ->() const { return get(); }
        const TimeSeriesProvider * ts() const          { return get(); }

        InputId inputId() const       { return InputId( m_id, ( m_basketIter ? m_basketIter.elemId() : InputId::ELEM_ID_NONE ) ); }

        operator bool() const         { return m_inputiter != m_inputend; }
        input_iterator & operator++() 
        {
            //advance basket iteration if active
            if( m_basketIter )
                ++m_basketIter;

            //if basket iterator is not active / no longer active go to next input
            if( !m_basketIter )
            {
                ++m_inputiter;
                ++m_id;

                if( m_inputiter != m_inputend && m_inputiter -> isSet<InputBasketInfo>() )
                    m_basketIter = m_inputiter -> get<InputBasketInfo>() -> begin_inputs( true );
            }
            return *this; 
        }

    private:
        const InputT * m_inputiter;
        const InputT * m_inputend;
        INOUT_ID_TYPE  m_id;

        InputBasketInfo::input_iterator m_basketIter;
    };

    //this is only currently used on startup for ranking... shouldnt be called during runtime
    virtual input_iterator inputs() const = 0;

private:

    Engine * m_engine;

    //for intrusive linked list
    Consumer * m_next;
    
    int32_t  m_rank;
    bool     m_started;
};

};

#endif
