#include <csp/engine/Consumer.h>
#include <csp/engine/InputAdapter.h>

namespace csp
{

InputAdapter::InputAdapter( Engine *engine, const CspTypePtr &type, PushMode pushMode ) : m_rootEngine( engine -> rootEngine() ),
                                                                                          m_pushMode( pushMode ),
                                                                                          m_started( false )
{
    if( pushMode == PushMode::BURST )
        init( CspArrayType::create( type ) );
    else
        init( type );
}

}
