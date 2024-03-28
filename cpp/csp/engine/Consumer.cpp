#include <csp/engine/Consumer.h>
#include <csp/engine/Engine.h>
#include <csp/engine/InputAdapter.h>

namespace csp
{

Consumer::Consumer( Engine * engine ) : m_engine( engine ),
                                        m_next( nullptr ),
                                        m_rank( -1 ),
                                        m_started( false )
{
}

Consumer::~Consumer()
{
}

void Consumer::start()
{
}

void Consumer::stop()
{
}

}
