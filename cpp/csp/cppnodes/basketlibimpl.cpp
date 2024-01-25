#include <csp/core/Time.h>
#include <csp/engine/CppNode.h>

namespace csp::cppnodes
{

DECLARE_CPPNODE( _sync_list )
{
    TS_LISTBASKET_INPUT( Generic, x );
    SCALAR_INPUT( TimeDelta, threshold );
    SCALAR_INPUT( bool, output_incomplete );

    ALARM( bool, a_end );

    STATE_VAR( size_t, s_count{0} );
    STATE_VAR( Scheduler::Handle, s_alarm_handle );
    STATE_VAR( std::vector<bool>, s_current_ticked{} );

    TS_LISTBASKET_OUTPUT( Generic );

    INIT_CPPNODE( _sync_list ) { }

    START()
    {
        s_current_ticked.resize( x.size(), false );
    }

    INVOKE()
    {
        if( x.tickedinputs() )
        {
            if( s_alarm_handle.expired() )
            {
                s_alarm_handle = csp.schedule_alarm( a_end, threshold, true );
            }

            for( auto it = x.tickedinputs(); it; ++it )
            {
                if( s_current_ticked[ it.elemId() ] == false )
                {
                    s_count++;
                }
                s_current_ticked[ it.elemId() ] = true;
            }
        }

        bool complete = s_count == x.size();

        if( csp.ticked( a_end ) || complete )
        {
            if( complete || output_incomplete )
            {
                for( size_t elemId = 0; elemId < x.size(); elemId++ )
                {
                    if( s_current_ticked[ elemId ] )
                    {
                        unnamed_output()[ elemId ].output( x[ elemId ] );
                    }
                }
            }

            if( s_alarm_handle.active() )
            {
                csp.cancel_alarm( a_end, s_alarm_handle );
            }

            std::fill( s_current_ticked.begin(), s_current_ticked.end(), false );
            s_count = 0;
        }
    }
};

EXPORT_CPPNODE( _sync_list );

/*
@csp.node(cppimpl=_cspbasketlibimpl._sample_list)
def sample_list(trigger: ts['Y'], x: [ts['T']]):
    '''will return valid items in x on trigger'''
    __outputs__([ts['T']].with_shape_of(x))
*/
DECLARE_CPPNODE( _sample_list )
{
    TS_INPUT( Generic, trigger );
    TS_LISTBASKET_INPUT( Generic, x );

    TS_LISTBASKET_OUTPUT( Generic );

    INIT_CPPNODE( _sample_list ) { }

    START()
    {
        x.makePassive();
    }

    INVOKE()
    {
        if( csp.ticked(trigger) )
        {
            for( auto it = x.validinputs(); it; ++it)
            {
                auto idx = it.elemId();
                unnamed_output()[ idx ].output( x[idx] );
            }
        }
    }
};

EXPORT_CPPNODE( _sample_list );

}