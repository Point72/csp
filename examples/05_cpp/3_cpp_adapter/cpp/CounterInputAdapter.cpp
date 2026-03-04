#include "CounterInputAdapter.h"

namespace csp::adapters::counter
{

CounterInputAdapter::CounterInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode, PushGroup * group )
    : PushInputAdapter( engine, type, pushMode, group )
{
}

}
